"""
The Rendering Contract harness (ADR-0086).

Every Result Type the library returns is registered here exactly once,
tagged ``document`` or ``fragment``, and every rule of the contract is
asserted against the whole registry at once:

1. one renderer named ``to_markdown()``
2. ``__str__`` returns ``self.to_markdown()``
3. ``to_dict()`` is untouched (not asserted here — it is not a rendering rule)
4. ``__repr__`` is left alone, so it is never renderer output
5. documents own ``#``/``##``; fragments start at ``###``

The registry is hand-maintained on purpose. Auto-discovery by walking
``__all__`` was rejected: a type whose renderer is missing would blow up
while building the fixture instead of failing the assertion that names
the contract it broke. The registry, not the exported type list, is also
the authoritative record of which types are fragments — that set follows
the parent renderers' delegation, not the declarations.

A **Payload Frame** entry additionally asserts that no cell of the wrapped
``pl.DataFrame`` reaches the output and that rendering is bounded in row
count. Types that carry a Payload Frame register themselves there as they
are brought onto the contract.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, SemanticType
from dataforge_ml.evaluation import (
    C2STConfig,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STResult,
    C2STScore,
    ColumnC2STResult,
    EvaluationReport,
)
from dataforge_ml.imputation import (
    FitSignals,
    FittedImputer,
    UnitFitResult,
    author,
    decide,
    fit_unit,
)
from dataforge_ml.imputation._config import (
    ColumnImputationDecision,
    ColumnImputationRecord,
    ImputationDecision,
    ImputationResult,
    ImputationStrategy,
    ImputationUnit,
)
from dataforge_ml.profiling._boolean_config import BooleanProfileResult
from dataforge_ml.profiling._boolean_profiler import BooleanProfiler
from dataforge_ml.profiling._categorical import CategoricalProfiler
from dataforge_ml.profiling._categorical_config import CategoricalProfileResult
from dataforge_ml.profiling._config import ProfileConfig, StructuralProfileResult
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
from dataforge_ml.profiling._correlation_profiler import CorrelationProfiler
from dataforge_ml.profiling._datetime_config import DatetimeProfileResult
from dataforge_ml.profiling._datetime_profiler import DatetimeProfiler
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessProfileResult,
)
from dataforge_ml.profiling._missingness_profiler import MissingnessProfiler
from dataforge_ml.profiling._nonlinearity_profiler import NonlinearityProfiler
from dataforge_ml.profiling._numeric_config import (
    NonlinearityProfileResult,
    NumericProfileResult,
)
from dataforge_ml.profiling._numeric_profiler import NumericProfiler
from dataforge_ml.profiling._target_config import (
    TargetProblemType,
    TargetProfileResult,
)
from dataforge_ml.profiling._target_profiler import TargetProfiler
from dataforge_ml.profiling._text_config import TextProfileResult
from dataforge_ml.profiling._text_profiler import TextProfiler
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.splitting._config import FoldResult, HoldoutCVResult, SplitResult
from dataforge_ml.splitting._splitter import DataSplitter

DOCUMENT_HEADING = re.compile(r"^# ", re.MULTILINE)
FRAGMENT_VIOLATION = re.compile(r"^#{1,2} ", re.MULTILINE)


@dataclass(frozen=True)
class ContractEntry:
    """One registered Result Type instance and the kind of output it owes."""

    name: str
    instance: Any
    kind: str  # "document" or "fragment"


@dataclass(frozen=True)
class PayloadFrameEntry:
    """A Result Type that wraps a Payload Frame, plus how to build it."""

    name: str
    build: Callable[[int], Any]
    planted_values: tuple[str, ...]


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": pl.Series([1.0, 2.0, None, 4.0, 5.0], dtype=pl.Float64),
            "label": pl.Series(["a", "b", "a", None, "b"], dtype=pl.Utf8),
        }
    )


def _profile_result() -> StructuralProfileResult:
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=True))
    return StructuralProfiler(config).profile(_sample_frame())


def _missingness_result() -> MissingnessProfileResult:
    df = _sample_frame()
    return MissingnessProfiler().profile(df, list(df.columns))


def _column_missingness_profile() -> ColumnMissingnessProfile:
    return _missingness_result().columns["score"]


def _numeric_result() -> NumericProfileResult:
    return NumericProfiler().profile(_sample_frame(), ["score"])


def _categorical_result() -> CategoricalProfileResult:
    return CategoricalProfiler().profile(_sample_frame(), ["label"])


def _datetime_result() -> DatetimeProfileResult:
    frame = pl.DataFrame(
        {
            "event_at": pl.Series(
                [
                    datetime(2024, 1, 1),
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 5),
                ],
                dtype=pl.Datetime,
            )
        }
    )
    return DatetimeProfiler().profile(frame, ["event_at"])


def _target_result() -> TargetProfileResult:
    return TargetProfiler(target_column="label").profile(_sample_frame())


def _wide_frame() -> pl.DataFrame:
    rows = 60
    return pl.DataFrame(
        {
            "x": pl.Series([float(i) for i in range(rows)], dtype=pl.Float64),
            "y": pl.Series([float(i * i % 17) for i in range(rows)], dtype=pl.Float64),
            "z": pl.Series([float(i % 7) for i in range(rows)], dtype=pl.Float64),
            "grp": pl.Series(["a", "b", "c"] * (rows // 3), dtype=pl.Utf8),
            "flag": pl.Series([i % 2 == 0 for i in range(rows)], dtype=pl.Boolean),
            "note": pl.Series(
                ["some free text here", "  ", "another note"] * (rows // 3),
                dtype=pl.Utf8,
            ),
        }
    )


def _boolean_result() -> BooleanProfileResult:
    return BooleanProfiler().profile(_wide_frame(), ["flag"])


def _text_result() -> TextProfileResult:
    return TextProfiler().profile(_wide_frame(), ["note"])


def _nonlinearity_result() -> NonlinearityProfileResult:
    return NonlinearityProfiler(["x", "y", "z"]).profile(_wide_frame())


def _correlation_result() -> CorrelationProfileResult:
    frame = _wide_frame()
    profiler = CorrelationProfiler(["x", "y", "z"], ["grp"])
    features = profiler.profile_features(frame, ["x", "y", "z"], ["grp"])
    return profiler.profile_target(frame, features, ["x", "y"], ["grp"], "z")


def _split_result() -> SplitResult:
    return DataSplitter(_wide_frame(), random_seed=0).random_split(test_size=0.25)


def _fold_result() -> FoldResult:
    return DataSplitter(_wide_frame(), random_seed=0).kfold(k=3)[0]


def _holdout_cv_result() -> HoldoutCVResult:
    return DataSplitter(_wide_frame(), random_seed=0).holdout_cv(
        test_size=0.2, k=3
    )


def _imputation_plan() -> ImputationDecision:
    frame = _sample_frame()
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=True))
    profile = StructuralProfiler(config).profile(frame)
    return decide(profile, frame.height, config)


def _authored_plan() -> ImputationDecision:
    return author(
        {"score": ImputationStrategy.Median, "label": ImputationStrategy.Mode},
        columns=["score", "label"],
    )


def _column_imputation_decision() -> ColumnImputationDecision:
    return next(iter(_imputation_plan().column_decisions.values()))


def _imputation_unit() -> ImputationUnit:
    return _authored_plan().units[0]


def _unit_fit_results() -> list[UnitFitResult]:
    plan = _imputation_plan()
    frame = _sample_frame()
    return [fit_unit(plan, unit, frame) for unit in plan.units]


def _unit_fit_result() -> UnitFitResult:
    return _unit_fit_results()[0]


def _fit_signals() -> FitSignals:
    return _unit_fit_result().signals


def _imputation_result() -> ImputationResult:
    plan = _imputation_plan()
    fitted = FittedImputer.compose(plan, [r.fitted for r in _unit_fit_results()])
    return fitted.transform(_sample_frame())


def _c2st_result() -> C2STResult:
    # Built directly rather than by running the generic test: the contract
    # under assertion is the rendering, and a real ``c2st`` call would pay for
    # several classifier fits to reach the same fields.
    return C2STResult(
        mean_repeat_z=4.2137,
        repeat_z=(3.9, 4.4, 4.3),
        repeat_z_spread=0.2646,
        train_accuracy=1.0,
        test_accuracy=0.6321,
        n_test=180,
        degenerate=False,
        unseen_category=True,
    )


def _c2st_score() -> C2STScore:
    return C2STScore(
        mean_frame_z=4.2137,
        frame_z_spread=0.1041,
        p_value=0.000013,
        p_adjusted=0.000052,
        frames=(_c2st_result(),),
    )


def _column_c2st_result() -> ColumnC2STResult:
    return ColumnC2STResult(
        column="score",
        outcome=C2STOutcome.Tested,
        n_observed=310,
        n_filled=90,
        score=_c2st_score(),
    )


def _c2st_provenance() -> C2STProvenance:
    return C2STProvenance(
        classifier_class="sklearn.ensemble.HistGradientBoostingClassifier",
        classifier_params={"min_samples_leaf": "5", "early_stopping": "False"},
        classifier_injected=False,
        sklearn_version="1.9.0",
        config=C2STConfig(),
    )


def _c2st_report() -> C2STReport:
    return C2STReport(
        columns={
            "score": _column_c2st_result(),
            "note": ColumnC2STResult(
                column="note",
                outcome=C2STOutcome.TypeNotTestable,
                n_observed=400,
                n_filled=0,
            ),
        },
        provenance=_c2st_provenance(),
    )


def _evaluation_report() -> EvaluationReport:
    return EvaluationReport(c2st=_c2st_report())


def _column_imputation_record() -> ColumnImputationRecord:
    return next(iter(_imputation_result().records.values()))


# ---------------------------------------------------------------------------
# The registry — every in-scope Result Type, tagged
# ---------------------------------------------------------------------------

REGISTRY: list[ContractEntry] = [
    ContractEntry("StructuralProfileResult", _profile_result(), "document"),
    ContractEntry("MissingnessProfileResult", _missingness_result(), "document"),
    ContractEntry(
        "ColumnMissingnessProfile", _column_missingness_profile(), "fragment"
    ),
    ContractEntry("NumericProfileResult", _numeric_result(), "document"),
    ContractEntry("CategoricalProfileResult", _categorical_result(), "document"),
    ContractEntry("DatetimeProfileResult", _datetime_result(), "document"),
    ContractEntry("TargetProfileResult", _target_result(), "document"),
    ContractEntry("BooleanProfileResult", _boolean_result(), "document"),
    ContractEntry("TextProfileResult", _text_result(), "document"),
    ContractEntry(
        "NonlinearityProfileResult", _nonlinearity_result(), "document"
    ),
    ContractEntry("CorrelationProfileResult", _correlation_result(), "document"),
    ContractEntry("SplitResult", _split_result(), "document"),
    ContractEntry("FoldResult", _fold_result(), "document"),
    ContractEntry("HoldoutCVResult", _holdout_cv_result(), "document"),
    ContractEntry("ImputationDecision", _imputation_plan(), "document"),
    ContractEntry("ImputationDecision(authored)", _authored_plan(), "document"),
    ContractEntry(
        "ColumnImputationDecision", _column_imputation_decision(), "fragment"
    ),
    ContractEntry("ImputationUnit", _imputation_unit(), "fragment"),
    ContractEntry("ImputationResult", _imputation_result(), "document"),
    ContractEntry(
        "ColumnImputationRecord", _column_imputation_record(), "fragment"
    ),
    ContractEntry("UnitFitResult", _unit_fit_result(), "document"),
    ContractEntry("FitSignals", _fit_signals(), "fragment"),
    ContractEntry("C2STResult", _c2st_result(), "fragment"),
    ContractEntry("C2STScore", _c2st_score(), "fragment"),
    ContractEntry("ColumnC2STResult", _column_c2st_result(), "fragment"),
    ContractEntry("C2STProvenance", _c2st_provenance(), "fragment"),
    ContractEntry("C2STReport", _c2st_report(), "fragment"),
    ContractEntry("EvaluationReport", _evaluation_report(), "document"),
]

# Empty / default-constructed instances of the same types. A renderer must
# survive a result that carries nothing.
EMPTY_REGISTRY: list[ContractEntry] = [
    ContractEntry("StructuralProfileResult", StructuralProfileResult(), "document"),
    ContractEntry(
        "MissingnessProfileResult", MissingnessProfileResult(), "document"
    ),
    ContractEntry(
        "ColumnMissingnessProfile",
        ColumnMissingnessProfile(column="", total_rows=0),
        "fragment",
    ),
    ContractEntry("NumericProfileResult", NumericProfileResult(), "document"),
    ContractEntry("CategoricalProfileResult", CategoricalProfileResult(), "document"),
    ContractEntry("DatetimeProfileResult", DatetimeProfileResult(), "document"),
    ContractEntry(
        "TargetProfileResult",
        TargetProfileResult(column="", problem_type=TargetProblemType.Unknown),
        "document",
    ),
    ContractEntry("BooleanProfileResult", BooleanProfileResult(), "document"),
    ContractEntry("TextProfileResult", TextProfileResult(), "document"),
    ContractEntry(
        "NonlinearityProfileResult", NonlinearityProfileResult(), "document"
    ),
    ContractEntry(
        "CorrelationProfileResult", CorrelationProfileResult(), "document"
    ),
    ContractEntry(
        "SplitResult",
        SplitResult(
            train=pl.DataFrame(),
            test=pl.DataFrame(),
            train_size=0,
            test_size=0,
            train_ratio=0.0,
            test_ratio=0.0,
        ),
        "document",
    ),
    ContractEntry(
        "FoldResult",
        FoldResult(
            train=pl.DataFrame(),
            val=pl.DataFrame(),
            fold_index=0,
            train_size=0,
            val_size=0,
        ),
        "document",
    ),
    ContractEntry(
        "HoldoutCVResult",
        HoldoutCVResult(test=pl.DataFrame(), folds=[]),
        "document",
    ),
    ContractEntry(
        "ImputationDecision",
        ImputationDecision(column_decisions={}, config_snapshot={}),
        "document",
    ),
    ContractEntry(
        "ColumnImputationDecision",
        ColumnImputationDecision(
            column="",
            semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Median,
        ),
        "fragment",
    ),
    ContractEntry(
        "ImputationUnit",
        ImputationUnit(unit_id="", strategy=ImputationStrategy.Median, columns=()),
        "fragment",
    ),
    ContractEntry(
        "ImputationResult", ImputationResult(dataframe=pl.DataFrame()), "document"
    ),
    ContractEntry(
        "ColumnImputationRecord",
        ColumnImputationRecord(
            decision=ColumnImputationDecision(
                column="",
                semantic_type=SemanticType.Numeric,
                strategy=ImputationStrategy.Median,
            )
        ),
        "fragment",
    ),
    # ``fitted`` is ``None`` here on purpose: it is the absent-sub-object case,
    # and the renderer owes a stated absence rather than a bare ``None``.
    ContractEntry(
        "UnitFitResult",
        UnitFitResult(
            unit_id="",
            strategy=ImputationStrategy.Median,
            columns=(),
            fitted=None,  # type: ignore[arg-type]
            signals=FitSignals(unit_id="", strategy=ImputationStrategy.Median),
        ),
        "document",
    ),
    ContractEntry(
        "FitSignals",
        FitSignals(unit_id="", strategy=ImputationStrategy.Median),
        "fragment",
    ),
    ContractEntry("C2STResult", C2STResult(mean_repeat_z=0.0), "fragment"),
    ContractEntry("C2STScore", C2STScore(mean_frame_z=0.0), "fragment"),
    ContractEntry(
        "ColumnC2STResult",
        ColumnC2STResult(
            column="",
            outcome=C2STOutcome.NoFilledCells,
            n_observed=0,
            n_filled=0,
        ),
        "fragment",
    ),
    ContractEntry(
        "C2STProvenance", C2STProvenance(classifier_class=""), "fragment"
    ),
    ContractEntry("C2STReport", C2STReport(), "fragment"),
    ContractEntry("EvaluationReport", EvaluationReport(), "document"),
]

# Payload Frame carriers. ``build(n_rows)`` grows only the wrapped frames; the
# split each fixture *reports* is held fixed, which is exactly the property the
# bounded-rendering assertion tests — the report must not widen because the data
# did. ``HoldoutCVResult`` reads its test partition's row count off the frame
# itself (it has no size field), so that frame stays fixed and the fold frames
# carry the growth.
PLANTED_TRAIN = "PLANTED-TRAIN-CELL-9f3a"
PLANTED_TEST = "PLANTED-TEST-CELL-2b7c"


def _payload_frame(n_rows: int, planted: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": pl.Series([float(i) for i in range(n_rows)], dtype=pl.Float64),
            "note": pl.Series([planted] * n_rows, dtype=pl.Utf8),
        }
    )


def _split_payload(n_rows: int) -> SplitResult:
    return SplitResult(
        train=_payload_frame(n_rows, PLANTED_TRAIN),
        test=_payload_frame(n_rows, PLANTED_TEST),
        train_size=8,
        test_size=2,
        train_ratio=0.8,
        test_ratio=0.2,
    )


def _fold_payload(n_rows: int) -> FoldResult:
    return FoldResult(
        train=_payload_frame(n_rows, PLANTED_TRAIN),
        val=_payload_frame(n_rows, PLANTED_TEST),
        fold_index=0,
        train_size=8,
        val_size=2,
    )


def _holdout_cv_payload(n_rows: int) -> HoldoutCVResult:
    return HoldoutCVResult(
        test=_payload_frame(4, PLANTED_TEST),
        folds=[_fold_payload(n_rows)],
    )


def _imputation_payload(n_rows: int) -> ImputationResult:
    # ``ImputationResult`` reports its row count off the imputed frame itself
    # (it has no size field), so — as with ``HoldoutCVResult``'s test partition
    # — the frame's height is held fixed and the growth lands in the cells,
    # which must not reach the output at all.
    frame = pl.DataFrame(
        {
            "score": pl.Series([1.0] * 5, dtype=pl.Float64),
            "note": pl.Series([PLANTED_TRAIN * n_rows] * 5, dtype=pl.Utf8),
        }
    )
    return ImputationResult(dataframe=frame, dropped_columns=["gone"])


PAYLOAD_FRAME_REGISTRY: list[PayloadFrameEntry] = [
    PayloadFrameEntry(
        "SplitResult", _split_payload, (PLANTED_TRAIN, PLANTED_TEST)
    ),
    PayloadFrameEntry("FoldResult", _fold_payload, (PLANTED_TRAIN, PLANTED_TEST)),
    PayloadFrameEntry(
        "HoldoutCVResult", _holdout_cv_payload, (PLANTED_TRAIN, PLANTED_TEST)
    ),
    PayloadFrameEntry("ImputationResult", _imputation_payload, (PLANTED_TRAIN,)),
]


def _ids(entries: list[ContractEntry]) -> list[str]:
    return [f"{e.name}[{e.kind}]" for e in entries]


@pytest.mark.parametrize("entry", REGISTRY, ids=_ids(REGISTRY))
def test_str_delegates_to_to_markdown(entry: ContractEntry):
    assert str(entry.instance) == entry.instance.to_markdown()


@pytest.mark.parametrize("entry", REGISTRY, ids=_ids(REGISTRY))
def test_to_markdown_returns_non_empty_string(entry: ContractEntry):
    markdown = entry.instance.to_markdown()
    assert isinstance(markdown, str)
    assert markdown.strip() != ""


@pytest.mark.parametrize("entry", REGISTRY, ids=_ids(REGISTRY))
def test_document_emits_a_top_level_heading(entry: ContractEntry):
    if entry.kind != "document":
        pytest.skip("not a document")
    markdown = entry.instance.to_markdown()
    assert DOCUMENT_HEADING.search(markdown), f"{entry.name} emits no '# ' heading"


@pytest.mark.parametrize("entry", REGISTRY, ids=_ids(REGISTRY))
def test_fragment_emits_no_document_heading(entry: ContractEntry):
    if entry.kind != "fragment":
        pytest.skip("not a fragment")
    markdown = entry.instance.to_markdown()
    offending = FRAGMENT_VIOLATION.findall(markdown)
    assert not offending, f"{entry.name} is a fragment but emits {offending}"


@pytest.mark.parametrize("entry", REGISTRY, ids=_ids(REGISTRY))
def test_repr_is_not_renderer_output(entry: ContractEntry):
    representation = repr(entry.instance)
    assert representation != entry.instance.to_markdown()
    assert not DOCUMENT_HEADING.search(representation)


@pytest.mark.parametrize("entry", EMPTY_REGISTRY, ids=_ids(EMPTY_REGISTRY))
def test_empty_instance_renders_without_raising(entry: ContractEntry):
    markdown = entry.instance.to_markdown()
    assert isinstance(markdown, str)
    assert str(entry.instance) == markdown


# ---------------------------------------------------------------------------
# Payload Frame — the rows never reach the output, and rendering is bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    PAYLOAD_FRAME_REGISTRY,
    ids=[e.name for e in PAYLOAD_FRAME_REGISTRY],
)
def test_payload_frame_cells_do_not_reach_the_output(entry: PayloadFrameEntry):
    markdown = entry.build(10).to_markdown()
    for planted in entry.planted_values:
        assert planted not in markdown, (
            f"{entry.name} leaked the planted Payload Frame cell {planted!r}"
        )


@pytest.mark.parametrize(
    "entry",
    PAYLOAD_FRAME_REGISTRY,
    ids=[e.name for e in PAYLOAD_FRAME_REGISTRY],
)
def test_payload_frame_rendering_is_bounded_in_rows(entry: PayloadFrameEntry):
    assert entry.build(10).to_markdown() == entry.build(1_000).to_markdown()
