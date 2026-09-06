"""Integration tests for ``evaluate_imputation`` — the happy path end to end.

Asserts the entry point's user-facing behaviour: the census covers every active
column, an excluded column is absent from it entirely, the feature matrix holds
what it should, sentinel rows land in the filled pile, the ``m`` axis pools by
the mean ``z``, ``metrics=`` selects, and the event stream is progress only.

The samples are built synthetically wherever the property under test allows it
— C2ST is one classifier fit and is cheap, whereas producing imputed frames is
what melts the machine — with one test driving the real
``decide`` -> ``fit_unit`` -> ``compose`` path so the surface is proved against
a genuinely imputed table.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from dataforge_ml import evaluation
from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.evaluation import (
    C2STAnnotation,
    C2STConfig,
    C2STOutcome,
    C2STVerdict,
    EvaluationConfig,
    EvaluationMetric,
    EvaluationReport,
    evaluate_imputation,
)
from dataforge_ml.evaluation.imputation import _adapter
from dataforge_ml.observability import EventType
from dataforge_ml.profiling._config import ProfileConfig
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from tests.conftest import fit_imputer

N_ROWS = 400
SENTINEL_ROWS = 80
TINY_NULLS = 5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_df(rng):
    """A pre-imputation frame exercising every column shape the adapter gates on."""
    r = rng(seed=11)

    def with_nulls(values, fraction, seed):
        mask = np.random.default_rng(seed).random(N_ROWS) < fraction
        return [None if mask[i] else float(values[i]) for i in range(N_ROWS)]

    ages = np.random.default_rng(4).integers(20, 70, N_ROWS).astype(float)
    age = [-999.0] * SENTINEL_ROWS + [
        float(v) for v in ages[SENTINEL_ROWS:]
    ]
    tiny = [
        None if i < TINY_NULLS else float(v)
        for i, v in enumerate(r.normal(3.0, 1.0, N_ROWS))
    ]

    return pl.DataFrame(
        {
            "score": pl.Series(
                with_nulls(r.normal(50.0, 10.0, N_ROWS), 0.25, 1),
                dtype=pl.Float64,
            ),
            "revenue": pl.Series(
                with_nulls(r.normal(200.0, 30.0, N_ROWS), 0.60, 2),
                dtype=pl.Float64,
            ),
            "age": pl.Series(age, dtype=pl.Float64),
            "complete": pl.Series(r.normal(0.0, 1.0, N_ROWS).tolist(), dtype=pl.Float64),
            "empty": pl.Series([None] * N_ROWS, dtype=pl.Float64),
            "tiny": pl.Series(tiny, dtype=pl.Float64),
            "note": pl.Series(
                [f"a free text note number {i} about things" for i in range(N_ROWS)],
                dtype=pl.Utf8,
            ),
            "user_id": pl.Series(
                [f"u{i:05d}" for i in range(N_ROWS)], dtype=pl.Utf8
            ),
            "grade": pl.Series(
                [
                    None if np.random.default_rng(6).random(N_ROWS)[i] < 0.2
                    else ["A", "B", "C", "D"][i % 4]
                    for i in range(N_ROWS)
                ],
                dtype=pl.Utf8,
            ),
            "excluded_hard": pl.Series(
                r.normal(0.0, 1.0, N_ROWS).tolist(), dtype=pl.Float64
            ),
            "excluded_soft": pl.Series(
                r.normal(0.0, 1.0, N_ROWS).tolist(), dtype=pl.Float64
            ),
        }
    )


@pytest.fixture(scope="module")
def eval_pipeline_config():
    config = PipelineConfig(
        profiling=ProfileConfig(numeric_sentinels={"age": [-999.0]})
    )
    config.add_exclusion("excluded_hard")
    config.add_phase_exclusion(PipelinePhase.Imputation, "excluded_soft")
    return config


@pytest.fixture(scope="module")
def eval_profile(eval_df, eval_pipeline_config):
    return StructuralProfiler(eval_pipeline_config).profile(eval_df)


@pytest.fixture(scope="module")
def imputed_frame(eval_df):
    """A hand-built median fill — the baseline C2ST is designed to catch."""
    filled = {}
    for name, dtype in eval_df.schema.items():
        series = eval_df.get_column(name)
        if dtype != pl.Float64:
            filled[name] = series.fill_null("A") if name == "grade" else series
            continue
        observed = series.filter(series.is_not_null() & (series != -999.0))
        fill = observed.median() if observed.len() else 0.0
        filled[name] = (
            series.map_elements(
                lambda v, fill=fill: fill if v is None or v == -999.0 else v,
                return_dtype=pl.Float64,
            )
        )
    return pl.DataFrame(filled)


@pytest.fixture
def eval_config():
    return EvaluationConfig(c2st=C2STConfig(repeats=3, random_state=0))


@pytest.fixture(scope="module")
def report(eval_df, imputed_frame, eval_profile, eval_pipeline_config):
    return evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(c2st=C2STConfig(repeats=3, random_state=0)),
        pipeline_config=eval_pipeline_config,
    )


@pytest.fixture
def feature_spy(monkeypatch):
    """Capture the frames the adapter hands the generic seam, per column."""
    seen: dict[str, list[str]] = {}
    real = _adapter.c2st
    columns: list[str] = []

    def spy(reference, candidate, **kwargs):
        seen[columns[-1]] = list(reference.columns)
        return real(reference, candidate, **kwargs)

    monkeypatch.setattr(_adapter, "c2st", spy)

    class _Emitter(_adapter.Emitter):
        def item(self, column, message=None):
            columns.append(column)
            super().item(column, message)

    monkeypatch.setattr(_adapter, "Emitter", _Emitter)
    return seen


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def test_the_fixture_carries_the_semantic_types_the_gates_read(eval_profile):
    types = {k: v.semantic_type for k, v in eval_profile.columns.items()}
    assert types["note"] is SemanticType.Text
    assert types["user_id"] is SemanticType.Identifier
    assert types["grade"] is SemanticType.Categorical
    assert types["score"] is SemanticType.Numeric


def test_evaluation_returns_a_report_with_a_z_per_tested_column(report):
    assert isinstance(report, EvaluationReport)
    tested = {
        name
        for name, record in report.c2st.columns.items()
        if record.outcome is C2STOutcome.Tested
    }
    assert tested == {"score", "revenue", "age", "grade"}
    for name in tested:
        assert isinstance(report.c2st[name].score.mean_frame_z, float)


def test_every_active_column_gets_a_record(report, eval_df, eval_pipeline_config):
    active = eval_pipeline_config.resolve_active_columns(
        PipelinePhase.Imputation, list(eval_df.columns)
    )
    assert list(report.c2st.columns) == active


def test_lookup_and_membership_work_on_the_report(report):
    assert "score" in report.c2st
    assert report.c2st["score"].column == "score"


@pytest.mark.parametrize("column", ["excluded_hard", "excluded_soft"])
def test_an_excluded_column_is_absent_from_the_census(report, column):
    assert column not in report.c2st
    with pytest.raises(KeyError):
        report.c2st[column]


def test_a_column_with_no_filled_cells_is_not_tested(report):
    assert report.c2st["complete"].outcome is C2STOutcome.NoFilledCells
    assert report.c2st["complete"].score is None


def test_a_fully_null_column_is_not_tested(report):
    assert report.c2st["empty"].outcome is C2STOutcome.NoObservedCells


def test_a_column_below_the_sample_floor_is_refused(report):
    assert report.c2st["tiny"].outcome is C2STOutcome.BelowSampleFloor
    assert report.c2st["tiny"].n_filled == TINY_NULLS


@pytest.mark.parametrize("column", ["note", "user_id"])
def test_text_and_identifier_columns_are_not_testable(report, column):
    assert report.c2st[column].outcome is C2STOutcome.TypeNotTestable


# ---------------------------------------------------------------------------
# The feature matrix
# ---------------------------------------------------------------------------


def test_text_and_identifier_columns_appear_in_no_feature_matrix(
    feature_spy, eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
    )

    assert feature_spy
    for features in feature_spy.values():
        assert "note" not in features
        assert "user_id" not in features


def test_a_fully_null_column_is_dropped_as_a_feature_and_a_partial_one_is_kept(
    feature_spy, eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
    )

    features = feature_spy["score"]
    assert "empty" not in features
    # 60% missing still carries 40% real data, so the rule stops at the endpoint.
    assert "revenue" in features
    # The column under test is itself a feature, so a median fill's one-value
    # spike is visible rather than hidden.
    assert "score" in features


def test_an_excluded_column_is_never_a_feature(
    feature_spy, eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
    )

    for features in feature_spy.values():
        assert "excluded_hard" not in features
        assert "excluded_soft" not in features


# ---------------------------------------------------------------------------
# The mask
# ---------------------------------------------------------------------------


def test_sentinel_rows_land_in_the_filled_pile(report):
    record = report.c2st["age"]
    assert record.n_filled == SENTINEL_ROWS
    assert record.n_observed == N_ROWS - SENTINEL_ROWS


def test_the_two_pile_sizes_account_for_every_row(report):
    for record in report.c2st.columns.values():
        assert record.n_observed + record.n_filled == N_ROWS


# ---------------------------------------------------------------------------
# The m axis
# ---------------------------------------------------------------------------


def test_more_than_one_frame_pools_by_the_mean_z_and_does_not_shrink_the_null(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    shifted = imputed_frame.with_columns(
        (pl.col("score") + 0.5).alias("score")
    )

    pooled = evaluate_imputation(
        eval_df,
        [imputed_frame, shifted],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
    )

    score = pooled.c2st["score"].score
    assert len(score.frames) == 2
    per_frame = [frame.mean_repeat_z for frame in score.frames]
    assert score.mean_frame_z == pytest.approx(float(np.mean(per_frame)))
    # Dividing the single-run null's SD by sqrt(m) is forbidden: it would
    # multiply the pooled z by sqrt(2) here.
    assert score.mean_frame_z != pytest.approx(
        float(np.mean(per_frame)) * np.sqrt(2)
    )
    assert score.frame_z_spread == pytest.approx(
        float(np.std(per_frame, ddof=1))
    )


def test_the_mask_derived_facts_are_stated_once_for_every_frame(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    pooled = evaluate_imputation(
        eval_df,
        [imputed_frame, imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
    )

    record = pooled.c2st["score"]
    # Pile sizes live on the column record, not per frame, so they cannot
    # disagree with themselves.
    assert not hasattr(record.score.frames[0], "n_observed")
    assert record.n_filled + record.n_observed == N_ROWS


def test_a_frame_of_the_wrong_height_is_refused(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    with pytest.raises(ValueError, match="row-aligned"):
        evaluate_imputation(
            eval_df,
            [imputed_frame.head(10)],
            eval_profile,
            config=eval_config,
            pipeline_config=eval_pipeline_config,
        )


def test_no_frames_at_all_is_refused(
    eval_df, eval_profile, eval_pipeline_config, eval_config
):
    with pytest.raises(ValueError, match="at least one imputed frame"):
        evaluate_imputation(
            eval_df,
            [],
            eval_profile,
            config=eval_config,
            pipeline_config=eval_pipeline_config,
        )


# ---------------------------------------------------------------------------
# metrics=
# ---------------------------------------------------------------------------


def test_a_metric_that_was_not_requested_comes_back_as_none(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    empty = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
        metrics=[],
    )

    assert empty.c2st is None


def test_selecting_c2st_explicitly_runs_it(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    selected = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
        metrics=[EvaluationMetric.C2ST],
    )

    assert selected.c2st is not None


def test_the_default_runs_every_metric_the_library_has(report):
    assert report.c2st is not None


# ---------------------------------------------------------------------------
# The event stream
# ---------------------------------------------------------------------------


@pytest.fixture
def events(eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config):
    captured = []
    evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=eval_config,
        pipeline_config=eval_pipeline_config,
        observer=captured.append,
    )
    return captured


def test_the_stream_is_stamped_evaluation_and_c2st(events):
    assert {e.phase for e in events} == {"evaluation"}
    assert {e.stage for e in events} == {"c2st"}


def test_the_stage_boundaries_are_present(events):
    assert events[0].event_type is EventType.stage_start
    assert events[-1].event_type is EventType.stage_end


def test_every_active_column_ticks_the_bar_including_refused_ones(
    events, eval_df, eval_pipeline_config
):
    active = eval_pipeline_config.resolve_active_columns(
        PipelinePhase.Imputation, list(eval_df.columns)
    )
    items = [e for e in events if e.event_type is EventType.item]

    assert [e.column for e in items] == active
    assert items[-1].index == items[-1].total == len(active)


def test_one_substep_is_emitted_per_repeat(events, eval_config):
    substeps = [e for e in events if e.event_type is EventType.substep]
    tested = 4  # score, revenue, age, grade

    assert len(substeps) == tested * eval_config.c2st.repeats
    assert {e.total for e in substeps} == {eval_config.c2st.repeats}


def test_the_event_stream_carries_no_decision_events(events):
    assert not [e for e in events if e.event_type is EventType.decision]


# ---------------------------------------------------------------------------
# A genuinely imputed table
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_df(rng):
    r = rng(seed=46)
    n = 300
    driver = r.normal(0.0, 1.0, n)
    mask = np.random.default_rng(9).random(n) < 0.3
    return pl.DataFrame(
        {
            "target": pl.Series(
                [None if mask[i] else float(driver[i]) for i in range(n)],
                dtype=pl.Float64,
            ),
            "driver": pl.Series(
                (driver * 2.0 + r.normal(0.0, 0.5, n)).tolist(), dtype=pl.Float64
            ),
        }
    )


def test_evaluation_runs_end_to_end_on_a_real_imputed_table(real_df):
    pipeline_config = PipelineConfig()
    profile = StructuralProfiler(pipeline_config).profile(real_df)
    imputer = fit_imputer(real_df, profile)
    imputed = imputer.transform(real_df).dataframe

    report = evaluate_imputation(
        real_df,
        [imputed],
        profile,
        config=EvaluationConfig(c2st=C2STConfig(repeats=3, random_state=0)),
        pipeline_config=pipeline_config,
    )

    record = report.c2st["target"]
    assert record.outcome is C2STOutcome.Tested
    assert record.score is not None
    assert record.score.frames[0].n_test > 0
    assert 0.0 <= record.score.p_value <= 1.0
    assert report.c2st.provenance.sklearn_version
    assert report.c2st.provenance.classifier_injected is False


# ---------------------------------------------------------------------------
# The floor dial
# ---------------------------------------------------------------------------


def test_lowering_the_floor_dial_tests_a_column_the_default_refuses(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    lowered = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(repeats=2, random_state=0, min_filled_cells=TINY_NULLS)
        ),
        pipeline_config=eval_pipeline_config,
    )

    # The dial reaches the generic seam's own assertion, so a deliberately
    # lowered floor is honoured rather than raising behind the user's back.
    # Five filled cells is not, however, enough for the classifier to split on,
    # so the runtime degeneracy net catches what the lowered floor let through
    # — which is the point of having both: the net is not the floor.
    assert lowered.c2st["tiny"].outcome is not C2STOutcome.BelowSampleFloor
    assert lowered.c2st["tiny"].outcome is C2STOutcome.Uninformative


def test_raising_the_floor_dial_refuses_a_column_the_default_tests(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    raised = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(repeats=2, random_state=0, min_filled_cells=200)
        ),
        pipeline_config=eval_pipeline_config,
    )

    assert raised.c2st["score"].outcome is C2STOutcome.BelowSampleFloor
    assert raised.c2st["score"].score is None


# ---------------------------------------------------------------------------
# Annotations — they qualify a z, they never suppress one (#512)
# ---------------------------------------------------------------------------


def test_a_thin_pile_is_annotated_low_power_and_still_tested(report):
    # 100 filled against 300 observed: honest, and far under the 250 rows per
    # pile at which the degenerate imputer is caught every time.
    record = report.c2st["score"]

    assert record.outcome is C2STOutcome.Tested
    assert C2STAnnotation.LowPower in record.annotations
    assert record.score.mean_frame_z is not None


def test_low_power_annotates_and_never_refuses(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    # A band wide enough to swallow every column: not one of them is refused
    # for it, which is the whole difference between this tier and the floor.
    annotated = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(repeats=2, random_state=0, low_power_below=10_000)
        ),
        pipeline_config=eval_pipeline_config,
    )

    tested = [
        r for r in annotated.c2st.columns.values()
        if r.outcome is C2STOutcome.Tested
    ]
    assert tested
    for record in tested:
        assert C2STAnnotation.LowPower in record.annotations
        assert record.score is not None


def test_imbalanced_piles_are_annotated_without_suppressing_the_z(report):
    # 100 against 300 is a ratio of 0.33, under the 0.5 dial.
    record = report.c2st["score"]

    assert C2STAnnotation.ImbalancedPiles in record.annotations
    assert isinstance(record.score.mean_frame_z, float)


def test_a_balanced_column_carries_no_imbalance_annotation(report):
    # 240 filled against 160 observed is a ratio of 0.67.
    assert (
        C2STAnnotation.ImbalancedPiles
        not in report.c2st["revenue"].annotations
    )


def test_the_imbalance_dial_moves_the_annotation(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    strict = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(
                repeats=2, random_state=0, imbalance_warn_ratio=0.9
            )
        ),
        pipeline_config=eval_pipeline_config,
    )

    assert (
        C2STAnnotation.ImbalancedPiles
        in strict.c2st["revenue"].annotations
    )


# ---------------------------------------------------------------------------
# The unseen-category flag
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def unseen_category_frames(eval_df):
    """A categorical fill inventing a level that never appears in observed rows."""
    grade = eval_df.get_column("grade")
    return eval_df.with_columns(
        pl.Series("grade", [
            "ZZZ" if v is None else v for v in grade.to_list()
        ], dtype=pl.Utf8)
    )


def test_an_unseen_category_is_annotated_and_the_z_is_still_reported(
    eval_df, unseen_category_frames, eval_profile, eval_pipeline_config
):
    invented = evaluate_imputation(
        eval_df,
        [unseen_category_frames],
        eval_profile,
        config=EvaluationConfig(c2st=C2STConfig(repeats=2, random_state=0)),
        pipeline_config=eval_pipeline_config,
    )

    record = invented.c2st["grade"]
    assert C2STAnnotation.UnseenCategory in record.annotations
    # Inventing a category is always wrong, but the flag names the cause
    # rather than replacing the number.
    assert record.outcome is C2STOutcome.Tested
    assert isinstance(record.score.mean_frame_z, float)


def test_a_fill_reusing_observed_levels_carries_no_unseen_flag(report):
    assert (
        C2STAnnotation.UnseenCategory not in report.c2st["grade"].annotations
    )


# ---------------------------------------------------------------------------
# The degeneracy net
# ---------------------------------------------------------------------------


def test_a_classifier_predicting_one_class_every_repeat_is_uninformative(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    # The safety net for an injected classifier: a constant predictor scores
    # exactly 0.5 against a balanced test set and would otherwise report
    # z = 0.000 — a median fill returned as a perfect pass.
    degenerate = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(
                repeats=3,
                random_state=0,
                classifier=DummyClassifier(strategy="constant", constant=0),
            )
        ),
        pipeline_config=eval_pipeline_config,
    )

    record = degenerate.c2st["score"]
    assert record.outcome is C2STOutcome.Uninformative
    assert record.score is None
    assert record.verdict is C2STVerdict.NoVerdict
    with pytest.raises(AttributeError):
        _ = record.score.mean_frame_z


def test_the_degeneracy_net_is_not_the_floor(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    # The same column, the same pile sizes, the library's own classifier: it is
    # the classifier that made the difference, not the sample size.
    honest = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(c2st=C2STConfig(repeats=3, random_state=0)),
        pipeline_config=eval_pipeline_config,
    )

    assert honest.c2st["score"].outcome is C2STOutcome.Tested


def test_all_six_outcomes_are_reachable(report, eval_df, imputed_frame,
                                        eval_profile, eval_pipeline_config):
    seen = {record.outcome for record in report.c2st.columns.values()}
    degenerate = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(
                repeats=2,
                random_state=0,
                classifier=DummyClassifier(strategy="constant", constant=0),
            )
        ),
        pipeline_config=eval_pipeline_config,
    )
    seen |= {r.outcome for r in degenerate.c2st.columns.values()}

    assert seen == set(C2STOutcome)


# ---------------------------------------------------------------------------
# Verdicts and Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_every_tested_column_carries_a_two_valued_verdict(report):
    for record in report.c2st.columns.values():
        if record.outcome is C2STOutcome.Tested:
            assert record.verdict in (
                C2STVerdict.Flagged,
                C2STVerdict.NotFlagged,
            )
            assert record.score.p_adjusted is not None
        else:
            assert record.verdict is C2STVerdict.NoVerdict
            assert record.score is None


def test_bh_runs_across_the_tested_columns_of_one_report(report):
    tested = [
        r for r in report.c2st.columns.values()
        if r.outcome is C2STOutcome.Tested
    ]
    k = len(tested)

    for record in tested:
        # Every adjusted value is a p * k / rank for some rank in 1..k, so it
        # can never sit below the raw p.
        assert record.score.p_adjusted >= record.score.p_value
        assert record.score.p_adjusted <= record.score.p_value * k + 1e-12


def test_the_raw_z_is_left_uncorrected(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config, eval_config
):
    # The same column evaluated alone and evaluated among its neighbours: BH
    # moved the shortlist, and the score did not move at all.
    alone_config = PipelineConfig(
        profiling=ProfileConfig(numeric_sentinels={"age": [-999.0]})
    )
    for name in eval_df.columns:
        if name != "score":
            alone_config.add_phase_exclusion(PipelinePhase.Imputation, name)

    together = evaluate_imputation(
        eval_df, [imputed_frame], eval_profile,
        config=eval_config, pipeline_config=eval_pipeline_config,
    )
    alone = evaluate_imputation(
        eval_df, [imputed_frame], eval_profile,
        config=eval_config, pipeline_config=alone_config,
    )

    assert len(alone.c2st) == 1
    # k = 1 leaves the adjusted p equal to the raw one; k = 4 raises it.
    assert alone.c2st["score"].score.p_adjusted == pytest.approx(
        alone.c2st["score"].score.p_value
    )
    assert (
        together.c2st["score"].score.p_adjusted
        > together.c2st["score"].score.p_value
    )
    # And the score itself did not move by a hair between the two sets.
    assert together.c2st["score"].score.mean_frame_z == pytest.approx(
        alone.c2st["score"].score.mean_frame_z
    )


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def baseline_frames(rng):
    """A table, its median fill, and an oracle fill that withheld nothing."""
    r = rng(seed=31)
    n = 800
    x = r.normal(0.0, 1.0, n)
    c = x * 2.0 + r.normal(0.0, 0.5, n)
    mask = np.random.default_rng(12).random(n) < 0.4

    original = pl.DataFrame(
        {
            "x": pl.Series(x.tolist(), dtype=pl.Float64),
            "c": pl.Series(
                [None if mask[i] else float(c[i]) for i in range(n)],
                dtype=pl.Float64,
            ),
        }
    )
    observed_median = float(np.median(c[~mask]))
    median_filled = original.with_columns(
        pl.col("c").fill_null(observed_median)
    )
    # The control: the values were never actually withheld, so the two piles
    # are two random samples of one table and nothing can tell them apart.
    oracle_filled = original.with_columns(
        pl.Series("c", c.tolist(), dtype=pl.Float64)
    )
    return original, median_filled, oracle_filled


def test_a_median_filled_column_is_flagged_and_an_unimputed_control_is_not(
    baseline_frames,
):
    original, median_filled, oracle_filled = baseline_frames
    pipeline_config = PipelineConfig()
    profile = StructuralProfiler(pipeline_config).profile(original)
    config = EvaluationConfig(c2st=C2STConfig(repeats=5, random_state=0))

    caught = evaluate_imputation(
        original, [median_filled], profile,
        config=config, pipeline_config=pipeline_config,
    ).c2st["c"]
    control = evaluate_imputation(
        original, [oracle_filled], profile,
        config=config, pipeline_config=pipeline_config,
    ).c2st["c"]

    # Scalar strategies failing at ~100% is the design: it is the baseline that
    # makes a model-based number interpretable.
    assert caught.outcome is C2STOutcome.Tested
    assert caught.verdict is C2STVerdict.Flagged
    assert caught.score.mean_frame_z > 5.0
    # The control sits at chance, so nothing is flagged and the z is small.
    assert control.outcome is C2STOutcome.Tested
    assert control.verdict is C2STVerdict.NotFlagged
    assert abs(control.score.mean_frame_z) < 3.0


def test_train_accuracy_is_reported_beside_test_accuracy(baseline_frames):
    original, median_filled, _ = baseline_frames
    pipeline_config = PipelineConfig()
    profile = StructuralProfiler(pipeline_config).profile(original)

    score = evaluate_imputation(
        original, [median_filled], profile,
        config=EvaluationConfig(c2st=C2STConfig(repeats=3, random_state=0)),
        pipeline_config=pipeline_config,
    ).c2st["c"].score

    frame = score.frames[0]
    assert 0.0 <= frame.train_accuracy <= 1.0
    assert 0.0 <= frame.test_accuracy <= 1.0
    # The repeat spread sits beside the mean, so a mean that is the artefact of
    # one lucky draw is visible.
    assert len(frame.repeat_z) == 3
    assert frame.repeat_z_spread >= 0.0
    assert "train_accuracy" in score.to_markdown()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_an_injected_classifier_flips_the_provenance_flag(
    eval_df, imputed_frame, eval_profile, eval_pipeline_config
):
    injected = evaluate_imputation(
        eval_df,
        [imputed_frame],
        eval_profile,
        config=EvaluationConfig(
            c2st=C2STConfig(
                repeats=2,
                random_state=0,
                classifier=HistGradientBoostingClassifier(min_samples_leaf=5),
            )
        ),
        pipeline_config=eval_pipeline_config,
    )

    provenance = injected.c2st.provenance
    assert provenance.classifier_injected is True
    assert provenance.classifier_class.endswith(
        "HistGradientBoostingClassifier"
    )


def test_provenance_params_come_from_get_params_not_repr(report):
    provenance = report.c2st.provenance

    classifier = HistGradientBoostingClassifier(
        min_samples_leaf=5, early_stopping=False
    )
    # ``repr`` elides every parameter left at its default — exactly the fact
    # most needed when arguing with a report six months later.
    assert "learning_rate" not in repr(classifier)
    assert "learning_rate" in provenance.classifier_params
    assert provenance.classifier_params["learning_rate"] == "0.1"


def test_provenance_stamps_the_sklearn_version_and_the_resolved_dials(report):
    provenance = report.c2st.provenance

    assert provenance.sklearn_version == sklearn.__version__
    assert provenance.config.fdr_alpha == 0.05
    assert provenance.config.min_filled_cells == 30


# ---------------------------------------------------------------------------
# What C2ST must not contain
# ---------------------------------------------------------------------------


def test_there_is_no_binomtest_and_no_exact_test_dial_anywhere_in_c2st():
    # The normal approximation's true size never exceeds 5.9% for n_test in
    # 10-400, so an exact test buys at most one lattice step.
    root = Path(evaluation.__file__).parent
    sources = "\n".join(
        path.read_text() for path in root.rglob("*.py")
    )

    assert "binomtest" not in sources
    assert "binom_test" not in sources
    assert "exact_test_below" not in sources
