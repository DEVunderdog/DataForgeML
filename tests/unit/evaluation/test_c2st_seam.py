"""The generic C2ST seam in isolation (#510 / ADR-0087).

Two frames in, one result out, with no imputer anywhere: every property below
is asserted against synthetic samples whose right answer is known in advance.
Calibration and detection are statistical, so each fixes ``random_state`` and
states a tolerance in the assertion rather than pinning an exact number. Pile
contents, subsample indices and per-fit accuracies are deliberately never
asserted — they are how the arithmetic got there, not what the seam promises.

Reference numbers come from the #487 prototype: null ``z`` SD ≈ 1.04 and a
one-sided FPR ≈ 6.2% at the nominal 5%; median fill caught 100%; oracle fill at
chance.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from dataforge_ml.evaluation._c2st import (
    MIN_SAMPLE_FLOOR,
    C2STDtypeError,
    C2STScheme,
    _default_c2st_classifier,
    _meets_sample_floor,
    c2st,
)
from dataforge_ml.observability import (
    Emitter,
    EventType,
    PipelineEvent,
    _ObservabilityMixin,
)

Z_CRIT = 1.645  # one-sided alpha = 0.05

# ---------------------------------------------------------------------------
# A synthetic world whose true conditional is known in closed form
# ---------------------------------------------------------------------------

_GROUP_EFFECT = np.array([-2.0, 0.0, 2.0])


def _table(n: int, rng: np.random.Generator) -> tuple[pl.DataFrame, np.ndarray]:
    """Return a table plus the true conditional mean of its ``c`` column."""
    x = rng.normal(size=(n, 3))
    group = rng.integers(0, 3, n)
    mu = x[:, 0] - 0.5 * x[:, 1] + 0.8 * x[:, 2] + _GROUP_EFFECT[group]
    frame = pl.DataFrame(
        {
            "x1": x[:, 0],
            "x2": x[:, 1],
            "x3": x[:, 2],
            "g": pl.Series([f"g{v}" for v in group], dtype=pl.Categorical),
            "flag": pl.Series(group == 0, dtype=pl.Boolean),
            "c": mu + rng.normal(size=n),
        }
    )
    return frame, mu


def _one_world(n: int, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two samples of ``n`` rows each, drawn from one distribution."""
    frame, _ = _table(2 * n, np.random.default_rng(seed))
    return frame[:n], frame[n:]


class _CountingClassifier(HistGradientBoostingClassifier):
    """The default classifier, counting how many times it is fitted."""

    def fit(self, X, y=None, **kwargs):
        self.fit_calls = getattr(self, "fit_calls", 0) + 1
        return super().fit(X, y, **kwargs)


def _counting() -> _CountingClassifier:
    return _CountingClassifier(
        min_samples_leaf=5, early_stopping=False, random_state=0
    )


# ---------------------------------------------------------------------------
# Calibration and detection
# ---------------------------------------------------------------------------


def test_null_calibration_is_standard_normal_at_the_nominal_level() -> None:
    """Two samples from one distribution score at chance, with FPR at nominal.

    Tolerances rather than equalities: the prototype measured a null ``z`` SD of
    0.92-1.04 and a one-sided FPR of 1.5-6.9% across pile sizes, so the mean is
    allowed half a standard error of drift and the FPR is capped generously
    above the nominal 5%.
    """
    scores = np.array(
        [
            c2st(
                *_one_world(300, seed),
                repeats=1,
                scheme=C2STScheme.Split,
                random_state=seed,
            ).mean_repeat_z
            for seed in range(40)
        ]
    )

    assert abs(float(scores.mean())) < 0.5
    assert 0.6 < float(scores.std(ddof=1)) < 1.4
    assert float((scores > Z_CRIT).mean()) <= 0.15


def test_a_degenerate_candidate_sample_is_caught() -> None:
    """A constant fill — the loudest discrepancy there is — must be detected."""
    rng = np.random.default_rng(11)
    frame, _ = _table(1200, rng)
    observed, filled = frame[:600], frame[600:]
    filled = filled.with_columns(
        pl.lit(float(np.median(observed["c"].to_numpy()))).alias("c")
    )

    result = c2st(
        observed, filled, repeats=3, scheme=C2STScheme.Split, random_state=7
    )

    assert result.mean_repeat_z > Z_CRIT
    assert result.test_accuracy > 0.6


def test_an_oracle_candidate_sample_scores_at_chance() -> None:
    """A draw from the true conditional is indistinguishable, and must be."""
    rng = np.random.default_rng(12)
    n = 1200
    frame, mu = _table(n, rng)
    oracle = frame.with_columns(
        pl.Series("c", mu + rng.normal(size=n))
    )

    result = c2st(
        frame[: n // 2],
        oracle[n // 2 :],
        repeats=3,
        scheme=C2STScheme.Split,
        random_state=7,
    )

    assert abs(result.mean_repeat_z) < 3.0
    assert result.mean_repeat_z <= Z_CRIT


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------


def test_balancing_sizes_the_test_off_the_smaller_sample() -> None:
    """``n_test`` follows the smaller sample and ignores the larger one's size.

    Unequal piles break the ``Binomial(n_test, 1/2)`` null outright, so the
    larger sample is subsampled to the smaller one's size. Growing the larger
    sample must therefore move neither ``n_test`` nor the score.
    """
    frame, _ = _table(1400, np.random.default_rng(13))
    small = frame[:60]
    narrow = c2st(
        frame[60:260], small, repeats=3, scheme=C2STScheme.Split, random_state=5
    )
    wide = c2st(
        frame[60:1400], small, repeats=3, scheme=C2STScheme.Split, random_state=5
    )

    # 60 rows per pile, 30% held out on each side.
    assert narrow.n_test == wide.n_test == 2 * round(60 * 0.3)
    assert abs(narrow.mean_repeat_z - wide.mean_repeat_z) < 2.0
    assert narrow.mean_repeat_z <= Z_CRIT and wide.mean_repeat_z <= Z_CRIT


# ---------------------------------------------------------------------------
# Pooling — the rule people get wrong
# ---------------------------------------------------------------------------


def test_pooling_is_the_mean_against_the_single_run_null() -> None:
    """``mean_repeat_z`` is the plain mean of the repeats, never scaled by sqrt(R).

    Dividing the null's SD by the square root of the repeat count manufactures
    significance, so this is an explicit regression test rather than a property
    left to the docstring.
    """
    repeats = 5
    result = c2st(
        *_one_world(200, 3),
        repeats=repeats,
        scheme=C2STScheme.Split,
        random_state=3,
    )

    assert len(result.repeat_z) == repeats
    assert result.mean_repeat_z == pytest.approx(float(np.mean(result.repeat_z)))
    scaled = float(np.mean(result.repeat_z)) * np.sqrt(repeats)
    assert result.mean_repeat_z != pytest.approx(scaled)
    assert result.repeat_z_spread == pytest.approx(
        float(np.std(result.repeat_z, ddof=1))
    )


def test_a_single_repeat_has_no_spread() -> None:
    result = c2st(
        *_one_world(200, 4), repeats=1, scheme=C2STScheme.Split, random_state=4
    )

    assert len(result.repeat_z) == 1
    assert result.repeat_z_spread == 0.0
    assert result.mean_repeat_z == pytest.approx(result.repeat_z[0])


# ---------------------------------------------------------------------------
# The size floor: a pure predicate, and an internal assertion
# ---------------------------------------------------------------------------


def test_the_floor_predicate_is_pure_and_needs_no_test_run() -> None:
    """The adapter asks first, so 190 refusals need not be 190 exceptions."""
    frame, _ = _table(400, np.random.default_rng(14))
    below = frame[: MIN_SAMPLE_FLOOR - 1]
    above = frame[: MIN_SAMPLE_FLOOR]

    assert _meets_sample_floor(frame[200:], above) is True
    assert _meets_sample_floor(frame[200:], below) is False
    # The floor is the *smaller* sample, whichever side it is on.
    assert _meets_sample_floor(below, frame[200:]) is False
    # And it is a dial the caller may raise.
    assert _meets_sample_floor(above, above, min_rows=1000) is False


def test_the_floor_is_asserted_internally_so_it_cannot_be_bypassed() -> None:
    frame, _ = _table(400, np.random.default_rng(15))

    with pytest.raises(ValueError, match="size floor"):
        c2st(
            frame[200:],
            frame[: MIN_SAMPLE_FLOOR - 1],
            repeats=1,
            scheme=C2STScheme.Split,
            random_state=0,
        )


# ---------------------------------------------------------------------------
# Dtype is the whole type contract
# ---------------------------------------------------------------------------


def test_a_string_column_raises_a_named_error_identifying_the_column() -> None:
    frame, _ = _table(200, np.random.default_rng(16))
    with_text = frame.with_columns(
        pl.Series("note", ["a note"] * frame.height, dtype=pl.String)
    )

    with pytest.raises(C2STDtypeError, match="'note'"):
        c2st(
            with_text[:100],
            with_text[100:],
            repeats=1,
            scheme=C2STScheme.Split,
            random_state=0,
        )


def test_mismatched_columns_are_refused() -> None:
    frame, _ = _table(200, np.random.default_rng(17))

    with pytest.raises(ValueError, match="same columns"):
        c2st(
            frame[:100],
            frame[100:].drop("x3"),
            repeats=1,
            scheme=C2STScheme.Split,
            random_state=0,
        )


def test_repeats_below_one_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 1 repeat"):
        c2st(
            *_one_world(60, 18),
            repeats=0,
            scheme=C2STScheme.Split,
            random_state=0,
        )


# ---------------------------------------------------------------------------
# The two schemes
# ---------------------------------------------------------------------------


def test_split_runs_one_fit_per_repeat_with_n_test_exactly_known() -> None:
    reference, candidate = _one_world(100, 19)
    model = _counting()

    result = c2st(
        reference,
        candidate,
        repeats=3,
        scheme=C2STScheme.Split,
        classifier=model,
        random_state=1,
    )

    assert model.fit_calls == 3
    assert result.n_test == 2 * round(100 * 0.3)


def test_cross_validation_runs_five_fits_per_repeat_over_every_row() -> None:
    reference, candidate = _one_world(100, 20)
    model = _counting()

    result = c2st(
        reference,
        candidate,
        repeats=2,
        scheme=C2STScheme.CrossValidation,
        classifier=model,
        random_state=1,
    )

    assert model.fit_calls == 10
    # Every balanced row is held out exactly once.
    assert result.n_test == 2 * 100


# ---------------------------------------------------------------------------
# Determinism and injection
# ---------------------------------------------------------------------------


def test_the_same_random_state_gives_the_same_result() -> None:
    reference, candidate = _one_world(120, 21)
    kwargs = {"repeats": 3, "scheme": C2STScheme.Split, "random_state": 99}

    first = c2st(reference, candidate, **kwargs)
    second = c2st(reference, candidate, **kwargs)

    assert first == second


def test_an_injected_classifier_is_used_verbatim() -> None:
    """Injection is total: no ``set_params`` reach-in, no clone."""
    reference, candidate = _one_world(120, 22)
    injected = HistGradientBoostingClassifier(
        max_iter=17, min_samples_leaf=9, early_stopping=False, random_state=3
    )
    before = injected.get_params()

    c2st(
        reference,
        candidate,
        repeats=2,
        scheme=C2STScheme.Split,
        classifier=injected,
        random_state=0,
    )

    assert injected.get_params() == before


def test_the_default_factory_carries_exactly_the_two_pins() -> None:
    """``c2st(classifier=None)`` never builds a bare sklearn-default HGB."""
    params = _default_c2st_classifier(random_state=5).get_params()
    inherited = HistGradientBoostingClassifier().get_params()

    assert params["min_samples_leaf"] == 5
    assert params["early_stopping"] is False
    differing = {
        key
        for key, value in params.items()
        if key != "random_state" and inherited[key] != value
    }
    assert differing == {"min_samples_leaf", "early_stopping"}


# ---------------------------------------------------------------------------
# The two runtime flags
# ---------------------------------------------------------------------------


def test_a_single_class_predictor_every_repeat_is_reported_degenerate() -> None:
    reference, candidate = _one_world(80, 23)

    result = c2st(
        reference,
        candidate,
        repeats=2,
        scheme=C2STScheme.Split,
        classifier=DummyClassifier(strategy="constant", constant=0),
        random_state=0,
    )

    assert result.degenerate is True
    assert result.test_accuracy == pytest.approx(0.5)


def test_a_candidate_only_level_sets_the_unseen_category_flag() -> None:
    reference, candidate = _one_world(120, 24)
    candidate = candidate.with_columns(
        pl.Series("g", ["g9"] * candidate.height, dtype=pl.Categorical)
    )

    result = c2st(
        reference, candidate, repeats=1, scheme=C2STScheme.Split, random_state=0
    )

    assert result.unseen_category is True
    # The flag names a cause; the score is still reported.
    assert result.repeat_z


def test_matching_levels_leave_the_unseen_category_flag_clear() -> None:
    result = c2st(
        *_one_world(120, 25), repeats=1, scheme=C2STScheme.Split, random_state=0
    )

    assert result.unseen_category is False
    assert result.degenerate is False


# ---------------------------------------------------------------------------
# Consumer shapes — no imputation anywhere
# ---------------------------------------------------------------------------


def test_a_train_test_drift_check_is_served_with_no_adapter() -> None:
    """reference = training rows, candidate = holdout. Nothing imputation-shaped."""
    frame, _ = _table(600, np.random.default_rng(26))
    train, holdout = frame[:450], frame[450:]

    result = c2st(
        train, holdout, repeats=3, scheme=C2STScheme.Split, random_state=2
    )

    assert result.mean_repeat_z <= Z_CRIT
    assert result.n_test == 2 * round(150 * 0.3)


def test_an_outlier_clipping_check_is_served_with_no_adapter() -> None:
    """reference = rows left alone, candidate = rows clipped."""
    frame, _ = _table(600, np.random.default_rng(27))
    untouched, clipped = frame[:300], frame[300:]
    clipped = clipped.with_columns(pl.col("c").clip(-0.5, 0.5))

    result = c2st(
        untouched, clipped, repeats=3, scheme=C2STScheme.Split, random_state=2
    )

    assert result.mean_repeat_z > Z_CRIT


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_one_substep_is_emitted_per_repeat() -> None:
    seen: list[PipelineEvent] = []
    emitter = Emitter("evaluation", "c2st", seen.append, total=1)

    c2st(
        *_one_world(80, 28),
        repeats=4,
        scheme=C2STScheme.Split,
        random_state=0,
        emitter=emitter,
    )

    substeps = [e for e in seen if e.event_type is EventType.substep]
    assert len(substeps) == 4
    assert [e.index for e in substeps] == [1, 2, 3, 4]
    assert {e.total for e in substeps} == {4}
    assert {(e.phase, e.stage) for e in substeps} == {("evaluation", "c2st")}
    assert not [e for e in seen if e.event_type is EventType.decision]


def test_the_emitter_gains_stage_boundaries_additively() -> None:
    """A free function inherits nothing, so the Emitter carries the boundaries."""
    seen: list[PipelineEvent] = []
    emitter = Emitter("evaluation", "c2st", seen.append)

    emitter.stage_start()
    emitter.stage_end()

    assert [e.event_type for e in seen] == [
        EventType.stage_start,
        EventType.stage_end,
    ]
    assert all(e.phase == "evaluation" and e.stage == "c2st" for e in seen)
    # ``_ObservabilityMixin`` is left exactly as it is.
    assert hasattr(_ObservabilityMixin, "_emit_stage_start")
    assert hasattr(_ObservabilityMixin, "_emit_stage_end")
    assert not hasattr(_ObservabilityMixin, "stage_start")
    assert not hasattr(_ObservabilityMixin, "stage_end")
