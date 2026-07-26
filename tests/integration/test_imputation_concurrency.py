"""Thread-based concurrent fitting under the independence rule (#326 / ADR-0056).

Batch scheduling is user-owned (ADR-0075), so these tests drive the loop the
user actually holds: a hand-rolled ``ThreadPoolExecutor`` of
``fit_unit(..., n_jobs_inner=1)`` calls against a plain sequential
``fit_unit(..., n_jobs_inner=-1)`` drive.  The assertions stay external — the
two schedules must produce bit-identical results (ADR-0069), and the inner
``n_jobs`` knob must never reach the numbers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dataforge_ml import (
    FittedImputer,
    ImputationStrategy,
    PipelineConfig,
    StructuralProfiler,
    fit_unit,
)
from dataforge_ml.imputation import (
    ImputationConfig,
    ModelChoice,
    NumericImputationConfig,
)
from dataforge_ml.imputation._decision_assembler import decide
from dataforge_ml.profiling._config import ProfileConfig

# ---------------------------------------------------------------------------
# A wide, mixed-strategy dataset: two mutually-independent joint blocks (MICE
# and KNN) so the outer thread parallelism actually engages.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_df(rng):
    rng = rng(seed=45)
    n = 300
    base = rng.normal(0.0, 1.0, n)
    cols = {}
    for name in ("a", "b", "c", "d", "e"):
        vals = (base + rng.normal(0.0, 0.5, n)).tolist()
        for i in range(n):
            if rng.random() < 0.10:
                vals[i] = None
        cols[name] = pl.Series(vals, dtype=pl.Float64)
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def wide_profile(wide_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(wide_df)


def _mixed_config() -> PipelineConfig:
    """Force a joint MICE block and a joint KNN block.

    Yields two mutually-independent units of work so the outer thread
    parallelism actually engages (ADR-0079 collapsed the per-column
    Regression fits this fixture used to force into the joint MICE block, so
    a single plan can no longer carry more than one MICE unit).
    """
    numeric = NumericImputationConfig(
        mice_min_rows=10,
        _per_column_strategy={
            "a": ImputationStrategy.MICE,
            "b": ImputationStrategy.MICE,
            "c": ImputationStrategy.MICE,
            "d": ImputationStrategy.KNN,
            "e": ImputationStrategy.KNN,
        },
    )
    return PipelineConfig(
        random_seed=7,
        imputation=ImputationConfig(numeric=numeric),
    )


# ---------------------------------------------------------------------------
# The two user-owned schedules under comparison (ADR-0075)
# ---------------------------------------------------------------------------


def _fit_sequential(plan, df, random_seed):
    """Fit units back to back, each opening its inner parallelism to every core."""
    return {
        unit.unit_id: fit_unit(
            plan, unit.unit_id, df, random_seed=random_seed, n_jobs_inner=-1
        )
        for unit in plan.units
    }


def _fit_pooled(plan, df, random_seed, max_workers=None):
    """Fit units side by side, each inner estimator pinned to a single core."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            unit.unit_id: pool.submit(
                fit_unit,
                plan,
                unit.unit_id,
                df,
                random_seed=random_seed,
                n_jobs_inner=1,
            )
            for unit in plan.units
        }
        return {uid: future.result() for uid, future in futures.items()}


def _mixed_plan(profile):
    config = _mixed_config()
    return decide(profile, profile.dataset.row_count, config), config


# ---------------------------------------------------------------------------
# Result identity: pooled vs sequential drives on the same seed
# ---------------------------------------------------------------------------


def test_pooled_fit_is_result_identical_to_sequential(wide_df, wide_profile):
    plan, config = _mixed_plan(wide_profile)
    fitted_seq = FittedImputer.compose(
        plan, _fit_sequential(plan, wide_df, config.random_seed)
    )
    fitted_par = FittedImputer.compose(
        plan, _fit_pooled(plan, wide_df, config.random_seed, max_workers=4)
    )

    # Same strategy and scalar fill for every column.
    assert set(fitted_seq.records) == set(fitted_par.records)
    for col, rec_seq in fitted_seq.records.items():
        rec_par = fitted_par.records[col]
        assert rec_seq.decision.strategy == rec_par.decision.strategy
        assert rec_seq.fill_value == rec_par.fill_value

    # Same fitted models: transforming the same frame yields identical output
    # (byte-identical imputed values ⇒ identical model parameters).
    out_seq = fitted_seq.transform(wide_df).dataframe
    out_par = fitted_par.transform(wide_df).dataframe
    assert_frame_equal(out_seq, out_par)


def test_pooled_fit_matches_default_sized_pool(wide_df, wide_profile):
    # A pool left to size itself (max_workers=None) must still match a
    # forced-sequential drive on the same seed.
    plan, config = _mixed_plan(wide_profile)
    fitted_seq = FittedImputer.compose(
        plan, _fit_sequential(plan, wide_df, config.random_seed)
    )
    fitted_auto = FittedImputer.compose(
        plan, _fit_pooled(plan, wide_df, config.random_seed, max_workers=None)
    )
    assert_frame_equal(
        fitted_seq.transform(wide_df).dataframe,
        fitted_auto.transform(wide_df).dataframe,
    )


# ---------------------------------------------------------------------------
# The RandomForest path: the schedule must not reach the result (ADR-0069)
#
# ADR-0067 recorded that the inner ``n_jobs`` derivation contradicts ADR-0056's
# determinism promise wherever a column routes to RandomForest, and that no test
# exercised that path -- which is why it survived.  These tests are that path.
# The route is forced through the plan rather than coaxed out of the profiler
# with monotone-nonlinear data, so the estimator under test is pinned by the
# assertion rather than by a tag inference that a later scope could move.
# ---------------------------------------------------------------------------


def _forest_plan(profile):
    """A plan with the MICE block routed to RandomForest.

    KNN carries no ``model_choice`` (it never reads one), so only the MICE
    columns are forced; the KNN block still runs concurrently alongside it,
    exercising the outer schedule the RandomForest determinism claim is about.
    """
    plan, config = _mixed_plan(profile)
    for col in ("a", "b", "c"):
        plan = plan.with_model_choice(col, ModelChoice.RandomForestRegressor)
    return plan, config


def test_forest_fit_is_bit_identical_across_schedules(wide_df, wide_profile):
    """The schedule picks which layer spends the cores, not what the numbers are.

    The sequential drive fits units back-to-back with wide inner parallelism;
    the pooled drive fits them side by side with each estimator pinned to one
    core.  A RandomForest's trees are invariant across that switch and its
    predictions are made core-invariant (ADR-0069), so the two runs must agree
    exactly -- not merely to a tolerance, which is the assertion that would have
    passed against the ~1e-15 drift this guards.
    """
    plan, config = _forest_plan(wide_profile)
    serial = FittedImputer.compose(
        plan, _fit_sequential(plan, wide_df, config.random_seed)
    )
    parallel = FittedImputer.compose(
        plan, _fit_pooled(plan, wide_df, config.random_seed, max_workers=4)
    )

    assert_frame_equal(
        serial.transform(wide_df).dataframe,
        parallel.transform(wide_df).dataframe,
        check_exact=True,
    )


def test_forest_transform_is_stable_against_itself(wide_df, wide_profile):
    """A fitted forest must settle: the same model on the same frame twice.

    Distinct from the test above, and the sharper half of the defect.  A bare
    ``RandomForestRegressor`` carrying ``n_jobs=-1`` accumulates its trees in
    thread-completion order on *every* predict, so repeated transforms of one
    fitted artifact disagree with each other -- reachable from a plain
    sequential drive, without the user pooling anything at all.
    """
    plan, config = _forest_plan(wide_profile)
    fitted = FittedImputer.compose(
        plan, _fit_sequential(plan, wide_df, config.random_seed)
    )

    first = fitted.transform(wide_df).dataframe
    for _ in range(3):
        assert_frame_equal(first, fitted.transform(wide_df).dataframe, check_exact=True)


# ---------------------------------------------------------------------------
# The concurrency knob is a real, validated, user-facing configuration surface
# ---------------------------------------------------------------------------


def test_max_workers_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        NumericImputationConfig(max_workers=0)
    with pytest.raises(ValueError):
        NumericImputationConfig(max_workers=-2)


def test_max_workers_survives_config_round_trip():
    cfg = NumericImputationConfig(max_workers=3)
    assert NumericImputationConfig.from_dict(cfg.to_dict()).max_workers == 3
    # ``None`` (auto) round-trips as well.
    cfg_auto = NumericImputationConfig()
    assert NumericImputationConfig.from_dict(cfg_auto.to_dict()).max_workers is None
