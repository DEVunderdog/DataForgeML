"""
Direct unit tests for the MICE ``n_nearest_features`` scale governor
(``_compute_mice_n_nearest_features``, ADR-0079, issue #416).

This is the one confirmed pre-existing gap the ADR-0079 collapse spec calls
out: the informative-predictor counting logic had no direct test before this
change. Every other private router/fitter/assembler function in this module
is deliberately exercised only through the public ``decide()`` door — this
file is the named exception, plus one ``decide()``-level test confirming the
widened count reaches the assembled plan and leaves the block's
``max_iter``/``tol``/``initial_strategy`` aggregation untouched.
"""

from dataforge_ml import PipelineConfig
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import ImputationStrategy, decide
from dataforge_ml.imputation._config import NumericImputationConfig
from dataforge_ml.imputation._decision_assembler import (
    _compute_mice_n_nearest_features,
)
from dataforge_ml.profiling._config import ColumnProfile, StructuralProfileResult
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import NumericStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numeric_cp(name: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        semantic_type=SemanticType.Numeric,
        missingness=ColumnMissingnessProfile(
            column=name,
            total_rows=100,
            effective_null_count=20,
            effective_null_ratio=0.2,
            severity=MissingSeverity.Moderate,
            flags=[],
            correlated_with=[],
        ),
        stats=NumericStats(),
    )


def _profile(columns: dict[str, ColumnProfile], **corr: dict[str, float]):
    result = StructuralProfileResult()
    result.columns.update(columns)
    result.dataset.row_count = 100
    if corr:
        result.dataset.feature_correlation = CorrelationProfileResult(
            pearson_matrix=corr
        )
    return result


# ---------------------------------------------------------------------------
# Direct tests of _compute_mice_n_nearest_features
# ---------------------------------------------------------------------------


def test_widened_breadth_counts_predictors_outside_the_block() -> None:
    """The block is smaller than the full active-numeric set (#416 AC2).

    ``a``/``b`` form the MICE block; ``c``/``d`` are active numeric columns
    the block does not own. Neither block member correlates with the other
    above threshold, but both correlate with the outside columns — so a
    block-only count would floor to 1, while the widened count must pick up
    the outside correlations.
    """
    feature_correlation = CorrelationProfileResult(
        pearson_matrix={
            "a": {"b": 0.05, "c": 0.5, "d": 0.5},
            "b": {"a": 0.05, "c": 0.5, "d": 0.5},
            "c": {"a": 0.5, "b": 0.5, "d": 0.0},
            "d": {"a": 0.5, "b": 0.5, "c": 0.0},
        }
    )
    config = NumericImputationConfig(mice_n_nearest_features_min_cols=1)

    widened, signal = _compute_mice_n_nearest_features(
        feature_correlation, ["a", "b"], ["a", "b", "c", "d"], config
    )
    block_only, _ = _compute_mice_n_nearest_features(
        feature_correlation, ["a", "b"], ["a", "b"], config
    )

    assert block_only == 1  # median of [0, 0] floored to max(1, 0)
    assert widened == 2  # median of [2, 2] — the outside correlations count
    assert "2 active numeric columns" not in signal
    assert "4 active numeric columns" in signal


def test_min_cols_gate_uses_block_size_not_full_breadth() -> None:
    """The gate stays block-sized: widening the predictor pool must not
    also widen what counts as "small enough to use every predictor".
    """
    config = NumericImputationConfig(mice_n_nearest_features_min_cols=5)

    # Block of 2 is <= the gate of 5, even though the active set has 10 cols.
    n_nearest, signal = _compute_mice_n_nearest_features(
        None, ["a", "b"], [f"c{i}" for i in range(10)], config
    )
    assert n_nearest is None
    assert "block (2 cols) at or below min_cols threshold (5)" in signal


def test_capped_at_mice_max_nearest_features() -> None:
    cols = [f"c{i}" for i in range(6)]
    pearson = {
        ci: {cj: 0.9 for cj in cols if cj != ci} for ci in cols
    }
    feature_correlation = CorrelationProfileResult(pearson_matrix=pearson)
    config = NumericImputationConfig(
        mice_n_nearest_features_min_cols=1, mice_max_nearest_features=2
    )
    n_nearest, signal = _compute_mice_n_nearest_features(
        feature_correlation, cols, cols, config
    )
    assert n_nearest == 2
    assert "capped at mice_max_nearest_features=2" in signal


def test_missing_correlation_contributes_nothing() -> None:
    """No ``feature_correlation`` at all floors to 1, same as an absent pair."""
    config = NumericImputationConfig(mice_n_nearest_features_min_cols=1)
    n_nearest, _ = _compute_mice_n_nearest_features(
        None, ["a", "b", "c"], ["a", "b", "c", "d", "e"], config
    )
    assert n_nearest == 1


# ---------------------------------------------------------------------------
# decide()-level: widened count reaches the plan; max_iter/tol/initial_strategy
# aggregation stays block-scoped
# ---------------------------------------------------------------------------


def test_decide_widens_n_nearest_features_but_not_max_iter_tol_initial_strategy() -> (
    None
):
    profile = _profile(
        {
            "a": _numeric_cp("a"),
            "b": _numeric_cp("b"),
            "c": _numeric_cp("c"),
            "d": _numeric_cp("d"),
        },
        a={"b": 0.05, "c": 0.5, "d": 0.5},
        b={"a": 0.05, "c": 0.5, "d": 0.5},
        c={"a": 0.5, "b": 0.5, "d": 0.0},
        d={"a": 0.5, "b": 0.5, "c": 0.0},
    )
    config = PipelineConfig()
    config.imputation.numeric = NumericImputationConfig(
        mice_n_nearest_features_min_cols=1
    )
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    config.imputation.numeric.set_per_column_strategy(
        ["c", "d"], ImputationStrategy.Median
    )

    plan = decide(profile, profile.dataset.row_count, config)
    mice_hyp = dict(plan.decided_hyperparameters["mice"])
    assert mice_hyp["n_nearest_features"] == 2

    # Re-decide with c/d hard-excluded — same block, narrower active set.
    # n_nearest_features must drop back to the block-only floor of 1; the
    # aggregation of max_iter/tol/initial_strategy (computed over the
    # block's own columns only, both times) must stay identical.
    narrow_config = PipelineConfig()
    narrow_config.imputation.numeric = NumericImputationConfig(
        mice_n_nearest_features_min_cols=1
    )
    narrow_config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    narrow_config.add_exclusion("c")
    narrow_config.add_exclusion("d")
    narrow_plan = decide(profile, profile.dataset.row_count, narrow_config)
    narrow_hyp = dict(narrow_plan.decided_hyperparameters["mice"])

    assert narrow_hyp["n_nearest_features"] == 1
    assert narrow_hyp["max_iter"] == mice_hyp["max_iter"]
    assert narrow_hyp["tol"] == mice_hyp["tol"]
    assert narrow_hyp["initial_strategy"] == mice_hyp["initial_strategy"]
