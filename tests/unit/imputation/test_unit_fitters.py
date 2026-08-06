"""
Tests for #371: the fitter bodies live in unit-shaped functions the surface calls.

The defect these guard against: a model-based unit quietly "succeeding" as a
column of zeros.  So these assert on what the ``fit_unit`` loop actually
*learns* for each unit and what its fitted units then *produce*, not on which
function it called.
"""

import numpy as np
import polars as pl
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from dataforge_ml import PipelineConfig, StructuralProfiler, fit_unit
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import UnitNotTrainableError
from dataforge_ml.imputation._config import (
    ColumnImputationDecision,
    ImputationStrategy,
    ImputationUnit,
    ModelChoice,
    NumericImputationConfig,
)
from dataforge_ml.imputation._decision_assembler import decide
from dataforge_ml.imputation._fitted_imputer import FittedMICE, _FittedKNN
from dataforge_ml.imputation._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)
from dataforge_ml.imputation._fitters import (
    UnitFitContext,
    _fill_scalar_predictors,
    fit_cluster_unit,
    fit_gmm_unit,
    fit_mice_unit,
    fit_scalar_unit,
)
from dataforge_ml.imputation._regression_estimator_factory import (
    RegressionEstimatorFactory,
)
from dataforge_ml.imputation._unit_fit import _fit_context
from dataforge_ml.imputation._utils import _df_to_numpy
from dataforge_ml.utils._null_normalization import _resolve_effective_nulls


def _frame(n=600, seed=0, bimodal=False, grouped=False):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x2 = x1 * 1.4 + rng.normal(0, 2, n)
    x3 = rng.normal(0, 1, n)

    def hole(a, frac):
        a = a.copy()
        a[rng.choice(n, int(n * frac), replace=False)] = np.nan
        return a

    data = {
        "a": hole(x1, 0.15),
        "b": hole(x2, 0.12),
        "c": hole(x3, 0.10),
        "d": x1 * 0.3 + x3,
    }
    if bimodal:
        lo = rng.random(n) < 0.5
        bi = np.where(lo, rng.normal(5, 1, n), rng.normal(40, 1, n))
        data["bi"] = hole(bi, 0.10)
        if grouped:
            data["grp"] = pl.Series(np.where(lo, "lo", "hi"))
    return pl.DataFrame(data)


def _drive(df, cfg):
    """Profile, plan, and train every unit; return the plan, frame, and results.

    ``results`` maps unit id to the fitted unit (unwrapped from the
    ``UnitFitResult`` bundle) so the assertions read the learned state directly.
    """
    profile = StructuralProfiler(config=cfg).profile(df)
    plan = decide(profile, profile.dataset.row_count, cfg)
    train = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    fitted = {
        unit.unit_id: fit_unit(
            plan, unit, train, random_seed=cfg.random_seed
        ).fitted
        for unit in plan.units
    }
    return profile, plan, train, fitted


def _config(**per_column_strategy):
    cfg = PipelineConfig()
    cfg.random_seed = 7
    for col, strategy in per_column_strategy.items():
        cfg.imputation.numeric.set_per_column_strategy(col, strategy)
    return cfg


def test_executor_trains_a_real_mice_model_not_a_column_of_zeros():
    cfg = _config(a="mice", b="mice")
    _, _, train, results = _drive(_frame(), cfg)

    fitted = results["mice"]
    assert isinstance(fitted, FittedMICE)

    out = fitted.transform(train)
    filled = out["a"].to_numpy()[train["a"].is_null().to_numpy()]
    assert len(filled) > 0
    assert not np.isnan(filled).any()
    # The regression this guards: every imputed cell was 0.0.
    assert not np.allclose(filled, 0.0)


def test_mice_widens_predictors_past_the_block_and_writes_back_only_its_own_column():
    """Parity check for #417 / ADR-0079: the joint block's fitter must widen its
    predictor set to every active numeric column (mirroring what the former
    per-column regression fitter already read via ``ctx.feature_columns``), and
    its fitted unit must write back only the columns it owns.

    ``y`` is a single-column MICE block with no other block member to regress
    against; ``outside`` is a fully-observed numeric column that never joins
    the block. ``y`` is a noiseless linear function of ``outside``, so a
    block-only fit (no predictors at all) could only fall back to ``y``'s own
    mean, while a fit widened to include ``outside`` recovers it almost
    exactly — a stark, measurable signal that the outside column actually
    participated as a predictor.
    """
    rng = np.random.default_rng(11)
    n = 400
    outside = rng.normal(0.0, 10.0, n)
    y_true = outside * 2.0 + 5.0 + rng.normal(0.0, 0.1, n)

    y = y_true.copy()
    miss_idx = rng.choice(n, int(n * 0.25), replace=False)
    y[miss_idx] = np.nan

    df = pl.DataFrame({"y": y, "outside": outside})
    cfg = _config(y="mice")
    _, plan, train, results = _drive(df, cfg)

    fitted = results["mice"]
    assert isinstance(fitted, FittedMICE)
    assert fitted.columns == ["y"]
    # Widened past the block's own membership (ADR-0079): "outside" is a
    # candidate predictor even though it belongs to no MICE unit.
    assert fitted.all_cols == ["y", "outside"]

    out = fitted.transform(train)
    filled = out["y"].to_numpy()[miss_idx]
    truth = y_true[miss_idx]
    assert np.corrcoef(filled, truth)[0, 1] > 0.9

    # Write-back restriction: the block read "outside" as a predictor but
    # never owns it — it belongs to a different (Passthrough) unit.
    assert out["outside"].equals(train["outside"])


def test_mice_scalar_predictor_fit_frame_matches_serve_frame_fill():
    """#418, acceptance criterion 1: the block's train-frame scalar fill must
    match its serve-frame scalar fill exactly.

    ``outside`` is Median-routed and carries nulls; at serve time
    ``FittedImputer.transform`` fills it with the sibling ``FittedScalar``
    unit's learned ``fill_value`` before any model unit reads it (the shared
    pre-model snapshot). ``fit_mice_unit`` must fill the same column with the
    identical value before it trains, computed independently from
    ``train_df`` under the column's own decided strategy rather than by
    reading the sibling unit's fitted state (ADR-0074: ``fit_unit`` calls stay
    independent of one another).
    """
    rng = np.random.default_rng(5)
    n = 300
    outside = rng.normal(50, 20, n)
    outside[rng.choice(n, 60, replace=False)] = np.nan
    y = outside * 2.0 + rng.normal(0, 1, n)
    y[rng.choice(n, 75, replace=False)] = np.nan

    df = pl.DataFrame({"y": y, "outside": outside})
    cfg = _config(y="mice", outside="median")
    _, plan, train, results = _drive(df, cfg)

    fitted_mice = results["mice"]
    fitted_scalar = results["median:outside"]
    assert isinstance(fitted_mice, FittedMICE)
    assert isinstance(fitted_scalar, FittedScalar)
    assert "outside" in fitted_mice.all_cols

    ctx = _fit_context(plan, cfg.random_seed)
    fit_frame, filled_cols = _fill_scalar_predictors(train, ctx, ["outside"])

    assert filled_cols == ["outside"]
    # Exactly the value the serve-time pre-model snapshot fills "outside"
    # with — the same FittedScalar._fill_value FittedImputer.transform applies.
    expected = train["outside"].fill_null(fitted_scalar.fill_value)
    assert fit_frame["outside"].equals(expected)


def _right_skewed_scalar(rng: np.random.Generator, signal: np.ndarray) -> np.ndarray:
    """Right-skewed scalar predictor: median != mean != a model's own estimate.

    Mirrors the #410 prototype's ``_skewed_scalar`` generator — the shape
    that exposed the skew in the first place.
    """
    return np.exp(0.6 * signal + rng.normal(scale=0.5, size=signal.shape[0]))


def test_mice_closes_scalar_half_train_serve_skew_for_linear_estimator():
    """#418 regression test: the previously-measured linear-estimator RMSE
    skew (#410 prototype: +71-107% RMSE) is closed.

    Three right-skewed scalar predictors (``s1``/``s2``/``s3``, mirroring the
    #410 prototype's skewed-scalar generator) feed a target ``y`` through a
    mildly nonlinear combination, fit with the ``BayesianRidge`` linear
    estimator — the same estimator/target-shape mismatch the prototype scored.
    The pre-#418 behaviour is reconstructed directly for comparison: fit on
    the raw frame, where the scalar predictors' missing cells are filled by
    ``IterativeImputer``'s own internal round-robin during ``.fit()``, then
    transform against the median-filled serve snapshot — train and serve
    disagree on what the scalar predictors look like. The fixed
    ``fit_mice_unit`` trains on the same median-filled predictors it will
    serve against, so it must predict ``y``'s missing cells with lower RMSE.
    """
    rng = np.random.default_rng(5)
    n = 800
    z1, z2, z3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    s1 = _right_skewed_scalar(rng, z1)
    s2 = _right_skewed_scalar(rng, z2)
    s3 = _right_skewed_scalar(rng, z3)
    complete = rng.normal(size=n)
    y_true = (
        0.3 * s1 * s2
        + 0.5 * np.log1p(s3)
        + 0.3 * complete
        + rng.normal(scale=0.4, size=n)
    )

    s1m, s2m, s3m, ym = s1.copy(), s2.copy(), s3.copy(), y_true.copy()
    for arr in (s1m, s2m, s3m):
        arr[rng.choice(n, int(n * 0.18), replace=False)] = np.nan
    y_missing_idx = rng.choice(n, int(n * 0.2), replace=False)
    ym[y_missing_idx] = np.nan

    raw = pl.DataFrame({"y": ym, "s1": s1m, "s2": s2m, "s3": s3m, "c": complete})
    train = _resolve_effective_nulls(raw)

    def _scalar_decision(col: str) -> ColumnImputationDecision:
        return ColumnImputationDecision(
            column=col,
            semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Median,
        )

    decisions = {
        "s1": _scalar_decision("s1"),
        "s2": _scalar_decision("s2"),
        "s3": _scalar_decision("s3"),
        "c": ColumnImputationDecision(
            column="c",
            semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.Passthrough,
        ),
        "y": ColumnImputationDecision(
            column="y",
            semantic_type=SemanticType.Numeric,
            strategy=ImputationStrategy.MICE,
            model_choice=ModelChoice.BayesianRidge,
        ),
    }
    feature_cols = ("y", "s1", "s2", "s3", "c")
    ctx = UnitFitContext(
        column_decisions=decisions,
        config=NumericImputationConfig(),
        feature_columns=feature_cols,
    )
    unit = ImputationUnit(
        unit_id="mice",
        strategy=ImputationStrategy.MICE,
        columns=("y",),
        hyperparameters=(
            ("max_iter", 15),
            ("tol", 1e-4),
            ("initial_strategy", "median"),
            ("n_nearest_features", None),
        ),
    )

    outcome = fit_mice_unit(unit, train, ctx)
    fitted = outcome.fitted
    assert fitted is not None

    medians = {c: float(train[c].median()) for c in ("s1", "s2", "s3")}
    serve_frame = train.with_columns(
        [pl.col(c).fill_null(medians[c]) for c in medians]
    )

    filled_y = fitted.transform(serve_frame)["y"].to_numpy()[y_missing_idx]
    fixed_rmse = float(np.sqrt(np.mean((filled_y - y_true[y_missing_idx]) ** 2)))

    # Pre-#418 behaviour, reconstructed directly: fit on the raw frame, where
    # IterativeImputer's own round-robin — not the median — fills the scalar
    # predictors during .fit(). Only .transform() (mirroring the serve-time
    # shared pre-model snapshot) sees the median-filled predictors.
    skewed_model = IterativeImputer(
        estimator=RegressionEstimatorFactory.build_from_choice(
            ModelChoice.BayesianRidge, n_jobs=1
        ),
        random_state=0,
        max_iter=15,
        tol=1e-4,
        initial_strategy="median",
        n_nearest_features=None,
    )
    skewed_model.fit(_df_to_numpy(train, list(feature_cols)))
    skewed_arr = skewed_model.transform(_df_to_numpy(serve_frame, list(feature_cols)))
    skewed_filled_y = skewed_arr[:, 0][y_missing_idx]
    skewed_rmse = float(np.sqrt(np.mean((skewed_filled_y - y_true[y_missing_idx]) ** 2)))

    # The fix must close a real, measurable share of the gap the #410
    # prototype flagged, not merely match it within noise.
    assert fixed_rmse < skewed_rmse * 0.97


def test_executor_trains_a_real_knn_model():
    cfg = _config()
    _, plan, train, results = _drive(_frame(), cfg)
    assert any(u.unit_id == "knn" for u in plan.units), "expected KNN routing"

    fitted = results["knn"]
    assert isinstance(fitted, _FittedKNN)

    filled = fitted.transform(train)["a"].to_numpy()[train["a"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_executor_trains_a_real_single_column_mice_model():
    cfg = _config(c="mice")
    _, _, train, results = _drive(_frame(), cfg)

    fitted = results["mice"]
    assert isinstance(fitted, FittedMICE)

    filled = fitted.transform(train)["c"].to_numpy()[train["c"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_executor_trains_a_real_gmm_sampling_model():
    cfg = _config()
    _, plan, train, results = _drive(_frame(bimodal=True), cfg)
    unit_id = "gmm_sampling:bi"
    assert any(u.unit_id == unit_id for u in plan.units), "expected GMM routing"

    fitted = results[unit_id]
    assert isinstance(fitted, FittedGMMSampling)

    # The two components separate, and every sample lands near one of them.
    assert abs(fitted.center1 - fitted.center2) > 10
    filled = fitted.transform(train)["bi"].to_numpy()[train["bi"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    near = np.minimum(
        np.abs(filled - fitted.center1), np.abs(filled - fitted.center2)
    )
    assert (near < 5).all()


def test_executor_trains_a_real_cluster_conditional_model():
    cfg = _config()
    cfg.imputation.numeric.set_bimodal_grouping_variable("bi", "grp")
    _, plan, train, results = _drive(_frame(bimodal=True, grouped=True), cfg)
    unit_id = "cluster_conditional:bi"
    assert any(u.unit_id == unit_id for u in plan.units), "expected cluster routing"

    fitted = results[unit_id]
    assert isinstance(fitted, FittedClusterConditional)
    assert fitted.grouping_variable == "grp"
    assert set(fitted.group_fills) == {"lo", "hi"}

    filled = fitted.transform(train)["bi"].to_numpy()[train["bi"].is_null().to_numpy()]
    assert not np.isnan(filled).any()
    assert not np.allclose(filled, 0.0)


def test_no_model_based_unit_degrades_to_a_scalar_on_a_healthy_frame():
    """A plan that routes model-based work must not silently produce scalars."""
    cfg = _config(a="mice", b="mice", c="mice")
    _, plan, _, results = _drive(_frame(bimodal=True), cfg)

    model_based = {
        u.unit_id
        for u in plan.units
        if u.unit_id in ("mice", "knn") or ":" in u.unit_id and not u.unit_id.startswith(
            ("mean:", "median:", "mode:", "constant:", "mnar:")
        )
    }
    assert model_based, "expected the plan to route some model-based work"
    for unit_id in model_based:
        assert not isinstance(results[unit_id], FittedScalar), (
            f"unit '{unit_id}' degraded to a scalar on a frame it should train on"
        )


def test_forced_strategy_that_cannot_train_raises_rather_than_degrading():
    """A user's deliberate choice is never silently swapped out (ADR-0029).

    A MICE block whose only column carries no ``model_choice`` (every column
    ``Unpredictable``) cannot train — the fitter reports a reason instead of a
    fitted unit (single-track failure, ADR-0071), matching the "no feature
    columns" failure the former per-column regression fitter raised before the
    collapse.
    """
    decision = ColumnImputationDecision(
        column="a", semantic_type=SemanticType.Numeric, strategy=ImputationStrategy.MICE
    )
    ctx = UnitFitContext(
        column_decisions={"a": decision},
        config=NumericImputationConfig(),
        feature_columns=("a",),
    )
    unit = ImputationUnit(
        unit_id="mice",
        strategy=ImputationStrategy.MICE,
        columns=("a",),
        hyperparameters=(),
    )
    train = pl.DataFrame({"a": [1.0, None, 3.0, None, 5.0, None]})

    outcome = fit_mice_unit(unit, train, ctx)
    assert outcome.fitted is None
    assert outcome.fallback_reason


def test_hyperparameter_override_reaches_the_fitter():
    """``with_hyperparameters`` is exactly the plan that fits (ADR-0073)."""
    cfg = _config(a="mice", b="mice")
    df = _frame()
    profile = StructuralProfiler(config=cfg).profile(df)
    plan = decide(profile, profile.dataset.row_count, cfg)

    edited = plan.with_hyperparameters("mice", {"max_iter": 3})
    (unit,) = plan.units_for(ImputationStrategy.MICE)
    fitted = fit_unit(plan, unit, df).fitted
    fitted_edited = fit_unit(edited, unit, df).fitted

    # The decided base is untouched; only the edited plan carries the override.
    assert fitted_edited.model.max_iter == 3
    assert fitted.model.max_iter != 3


# ---------------------------------------------------------------------------
# The fitters read the facts off the column decision, never config (#466)
# ---------------------------------------------------------------------------


def _bimodal_frame(grouped: bool = False) -> pl.DataFrame:
    """A bimodal frame whose NaN holes are real nulls, as a fitter always sees."""
    df = _frame(bimodal=True, grouped=grouped)
    return df.with_columns(
        [
            pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
            for c, dtype in df.schema.items()
            if dtype.is_float()
        ]
    )


def _fact_ctx(decision: ColumnImputationDecision, config=None) -> UnitFitContext:
    return UnitFitContext(
        column_decisions={decision.column: decision},
        config=config or NumericImputationConfig(),
        random_seed=7,
    )


def test_constant_fill_is_read_off_the_column_decision():
    decision = ColumnImputationDecision(
        column="a",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Constant,
        constant_fill=3.5,
    )
    unit = ImputationUnit(
        unit_id="constant:a", strategy=ImputationStrategy.Constant, columns=("a",)
    )
    outcome = fit_scalar_unit(
        unit, pl.DataFrame({"a": [1.0, None, 2.0]}), _fact_ctx(decision)
    )
    assert isinstance(outcome.fitted, FittedScalar)
    assert outcome.fitted.fill_value == 3.5


def test_constant_fill_in_config_alone_no_longer_reaches_the_fitter():
    """The behaviour ADR-0083 gives up knowingly: config is read at decide-time only."""
    decision = ColumnImputationDecision(
        column="a",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.Constant,
    )
    config = NumericImputationConfig()
    config.set_per_column_constant_fill("a", 3.5)
    unit = ImputationUnit(
        unit_id="constant:a", strategy=ImputationStrategy.Constant, columns=("a",)
    )
    outcome = fit_scalar_unit(
        unit, pl.DataFrame({"a": [1.0, None, 2.0]}), _fact_ctx(decision, config)
    )
    assert outcome.fitted is None
    assert "constant_fill" in outcome.fallback_reason


def test_gmm_centres_are_read_off_the_column_decision():
    decision = ColumnImputationDecision(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.GMMSampling,
        center1=5.0,
        center2=40.0,
    )
    unit = ImputationUnit(
        unit_id="gmm_sampling:bi",
        strategy=ImputationStrategy.GMMSampling,
        columns=("bi",),
        hyperparameters=(("central_tendency", "median"),),
    )
    df = _bimodal_frame()
    outcome = fit_gmm_unit(unit, df, _fact_ctx(decision))
    assert isinstance(outcome.fitted, FittedGMMSampling)
    assert abs(outcome.fitted.center1 - outcome.fitted.center2) > 10


def test_gmm_without_centres_on_the_decision_cannot_train():
    decision = ColumnImputationDecision(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.GMMSampling,
    )
    unit = ImputationUnit(
        unit_id="gmm_sampling:bi",
        strategy=ImputationStrategy.GMMSampling,
        columns=("bi",),
        # Stale hyperparameters must not resurrect the retired keys.
        hyperparameters=(("center1", 5.0), ("center2", 40.0)),
    )
    outcome = fit_gmm_unit(unit, _bimodal_frame(), _fact_ctx(decision))
    assert outcome.fitted is None
    assert "bimodal centres" in outcome.fallback_reason


def test_cluster_grouping_variable_is_read_off_the_column_decision():
    decision = ColumnImputationDecision(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
        center1=5.0,
        center2=40.0,
        grouping_variable="grp",
    )
    unit = ImputationUnit(
        unit_id="cluster_conditional:bi",
        strategy=ImputationStrategy.ClusterConditional,
        columns=("bi",),
        hyperparameters=(("central_tendency", "median"),),
    )
    outcome = fit_cluster_unit(
        unit, _bimodal_frame(grouped=True), _fact_ctx(decision)
    )
    assert isinstance(outcome.fitted, FittedClusterConditional)
    assert outcome.fitted.grouping_variable == "grp"
    assert set(outcome.fitted.group_fills) == {"lo", "hi"}


def test_cluster_feature_cols_are_read_off_the_column_decision():
    decision = ColumnImputationDecision(
        column="bi",
        semantic_type=SemanticType.Numeric,
        strategy=ImputationStrategy.ClusterConditional,
        center1=5.0,
        center2=40.0,
        feature_cols=("a", "d"),
    )
    unit = ImputationUnit(
        unit_id="cluster_conditional:bi",
        strategy=ImputationStrategy.ClusterConditional,
        columns=("bi",),
        hyperparameters=(("central_tendency", "median"),),
    )
    outcome = fit_cluster_unit(unit, _bimodal_frame(), _fact_ctx(decision))
    assert isinstance(outcome.fitted, FittedClusterConditional)
    assert outcome.fitted.grouping_variable is None
    assert outcome.fitted.feature_cols == ["a", "d"]
    assert outcome.fitted.feature_centroid_1 is not None
