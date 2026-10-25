"""
Tests for the model-based escalation point (#528): Feasibility Floor
(ADR-0091), Signal Score (ADR-0092), and the Capability Ladder (ADR-0094).

Constructs :class:`ColumnProfile`/:class:`NumericStats` directly at known
Usable Rows / Rows per Predictor / correlation values so each floor boundary
and each Signal Tier is hit exactly, rather than reverse-engineering a real
dataset that happens to land on the boundary.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dataforge_ml.imputation import ImputationStrategy, route
from dataforge_ml.imputation._config import (
    ImputationConfig,
    ModelChoice,
    NumericImputationConfig,
)
from dataforge_ml.imputation._escalation import (
    _bottom_tier_fill,
    _capability_pick,
    _escalation_seam,
    _feasibility_knn,
    _feasibility_mice,
    _forced_feasibility,
    _signal_score,
    _signal_tier,
)
from dataforge_ml.imputation._evidence import (
    EntryCause,
    EntryFact,
    FeasibilityFact,
    RoutingReason,
    SignalScoreFact,
    SignalTier,
    TierFact,
)
from dataforge_ml.profiling._config import ColumnProfile, NumericKind, ProfileConfig
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
from dataforge_ml.profiling._missingness_config import ColumnMissingnessProfile
from dataforge_ml.profiling._numeric_config import (
    NumericFlag,
    NumericStats,
    SkewSeverity,
)
from dataforge_ml.profiling.orchestrator import StructuralProfiler

# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _missingness(effective_null_ratio: float, total_rows: int) -> ColumnMissingnessProfile:
    eff = round(effective_null_ratio * total_rows)
    return ColumnMissingnessProfile(
        column="x",
        total_rows=total_rows,
        standard_null_count=eff,
        effective_null_count=eff,
        standard_null_ratio=effective_null_ratio,
        effective_null_ratio=effective_null_ratio,
    )


def _cp(
    *,
    null_ratio: float = 0.0,
    total_rows: int = 100,
    numeric_kind: NumericKind = NumericKind.Continuous,
    r2_linear: float | None = None,
    r2_rf: float | None = None,
    max_mi: float | None = None,
    skewness_severity: SkewSeverity | None = SkewSeverity.Normal,
) -> ColumnProfile:
    stats = NumericStats(
        r2_linear=r2_linear,
        r2_rf=r2_rf,
        max_mutual_information=max_mi,
        skewness_severity=skewness_severity,
    )
    return ColumnProfile(
        name="x",
        numeric_kind=numeric_kind,
        missingness=_missingness(null_ratio, total_rows),
        stats=stats,
    )


def _corr(col: str, others: dict[str, float]) -> CorrelationProfileResult:
    matrix = {col: dict(others)}
    for other, r in others.items():
        matrix.setdefault(other, {})[col] = r
    return CorrelationProfileResult(pearson_matrix=matrix)


def _config(**overrides) -> NumericImputationConfig:
    return NumericImputationConfig(**overrides)


# ---------------------------------------------------------------------------
# Feasibility Floor — MICE (ADR-0091, ADR-0097: mice_min_rows_per_predictor=2)
# ---------------------------------------------------------------------------


def test_mice_zero_predictors_is_infeasible():
    cp = _cp(null_ratio=0.0, total_rows=1000)
    check = _feasibility_mice(cp, n_rows=1000, n_features=1, config=_config())
    assert check.feasible is False
    (term,) = check.failed_terms
    assert (term.term, term.value, term.limit, term.dial) == ("predictors", 0, 1, None)


def test_mice_exactly_at_rows_per_predictor_bound_is_feasible():
    # n_features=3 -> 2 predictors. Usable rows = 100 * (1-0.0) = 100.
    # rows_per_predictor = 100/2 = 50 >> 2, trivially feasible; construct a
    # tighter case that lands exactly on the bound instead.
    config = _config(mice_min_rows_per_predictor=2)
    # usable_rows / predictors == 2 exactly: usable_rows=4, predictors=2
    cp = _cp(null_ratio=0.96, total_rows=100)  # usable = 100*0.04 = 4
    check = _feasibility_mice(cp, n_rows=100, n_features=3, config=config)
    assert check.feasible is True


def test_mice_just_below_rows_per_predictor_bound_is_infeasible():
    config = _config(mice_min_rows_per_predictor=2)
    # usable_rows=3, predictors=2 -> rows_per_predictor=1.5 < 2
    cp = _cp(null_ratio=0.97, total_rows=100)  # usable = 100*0.03 = 3
    check = _feasibility_mice(cp, n_rows=100, n_features=3, config=config)
    assert check.feasible is False
    (term,) = check.failed_terms
    assert term.term == "rows_per_predictor"
    assert term.value == pytest.approx(1.5)
    assert term.limit == 2
    assert term.dial.operation == "route"
    assert term.dial.dial == "thresholds.numeric.mice_min_rows_per_predictor"
    assert term.dial.value == 2
    assert term.dial.source == "default"


def test_mice_predictor_count_capped_by_mice_max_nearest_features():
    config = _config(mice_min_rows_per_predictor=2, mice_max_nearest_features=5)
    # n_features=50 -> 49 predictors uncapped, capped at 5.
    # usable_rows = 8 -> rows_per_predictor = 8/5 = 1.6 < 2 => infeasible
    cp = _cp(null_ratio=0.92, total_rows=100)  # usable = 100*0.08 = 8
    check = _feasibility_mice(cp, n_rows=100, n_features=50, config=config)
    assert check.feasible is False


def test_mice_has_no_resource_ceiling():
    """MICE is never refused for raw row count — only Rows per Predictor."""
    config = _config(mice_min_rows_per_predictor=2, knn_max_rows=10)
    cp = _cp(null_ratio=0.0, total_rows=1_000_000)
    check = _feasibility_mice(cp, n_rows=1_000_000, n_features=3, config=config)
    assert check.feasible is True


# ---------------------------------------------------------------------------
# Feasibility Floor — KNN: no floor bound, only the Resource Ceiling
# ---------------------------------------------------------------------------


def test_knn_zero_predictors_is_infeasible():
    cp = _cp(null_ratio=0.0, total_rows=1000)
    check = _feasibility_knn(cp, n_rows=1000, n_features=1, config=_config())
    assert check.feasible is False


def test_knn_thin_rows_per_predictor_is_still_feasible_no_floor():
    """KNN has no floor bound — only zero predictors and knn_max_rows fail it."""
    config = _config(knn_max_rows=50_000)
    cp = _cp(null_ratio=0.99, total_rows=100)  # usable = 1 row, 2 predictors
    check = _feasibility_knn(cp, n_rows=100, n_features=3, config=config)
    assert check.feasible is True


def test_knn_resource_ceiling_exact_boundary():
    config = _config(knn_max_rows=1000)
    cp = _cp(null_ratio=0.0, total_rows=1000)
    at_bound = _feasibility_knn(cp, n_rows=1000, n_features=3, config=config)
    over_bound = _feasibility_knn(cp, n_rows=1001, n_features=3, config=config)
    assert at_bound.feasible is True
    assert over_bound.feasible is False
    (term,) = over_bound.failed_terms
    assert (term.term, term.value, term.limit) == ("row_count", 1001, 1000)
    assert term.dial.dial == "thresholds.numeric.knn_max_rows"
    assert term.dial.value == 1000
    assert term.dial.source == "argument"


# ---------------------------------------------------------------------------
# Signal Score (ADR-0092)
# ---------------------------------------------------------------------------


def test_signal_score_none_when_no_predictor_correlation():
    cp = _cp()
    score, fact = _signal_score("x", cp, feature_correlation=None, config=_config())
    assert score is None
    assert fact is None


def test_signal_score_degrades_to_squared_max_r_when_r2_pair_absent():
    cp = _cp(r2_linear=None, r2_rf=None)
    feature_correlation = _corr("x", {"a": 0.5, "b": 0.1})
    score, fact = _signal_score("x", cp, feature_correlation, _config())
    assert score is not None
    assert fact.degraded is True
    assert fact.base == pytest.approx(0.25)
    assert fact.score == pytest.approx(score)
    # base = max|r|^2 = 0.25 at minimum (breadth/latent can only add or floor it)
    assert score >= 0.25 - 1e-6


def test_signal_score_uses_r2_pair_when_available():
    cp = _cp(r2_linear=0.4, r2_rf=0.6)
    feature_correlation = _corr("x", {"a": 0.3})
    score, fact = _signal_score("x", cp, feature_correlation, _config())
    assert fact.degraded is False
    assert fact.base == pytest.approx(0.6)
    # base = max(max(0.4, 0.6), 0.3**2) = 0.6
    assert score >= 0.6 - 1e-6


def test_signal_score_latent_structure_can_only_raise_the_floor():
    config = _config(signal_score_latent_weight=0.5)
    # Weak R2/correlation, but strong mutual information -> latent floor lifts the score.
    cp = _cp(r2_linear=0.01, r2_rf=0.01, max_mi=5.0)
    feature_correlation = _corr("x", {"a": 0.05})
    score, _ = _signal_score("x", cp, feature_correlation, config)
    latent = 1.0 - np.exp(-2.0 * 5.0)
    assert score == pytest.approx(min(1.0, 0.5 * latent), abs=1e-6)


def test_signal_score_clamped_to_one():
    config = _config(signal_score_breadth_weight=10.0)
    cp = _cp(r2_linear=0.9, r2_rf=0.9)
    feature_correlation = _corr("x", {"a": 0.9, "b": 0.9, "c": 0.9})
    score, _ = _signal_score("x", cp, feature_correlation, config)
    assert score == 1.0


def test_signal_tier_boundaries():
    config = _config(signal_score_middle_tier_min=0.3, signal_score_top_tier_min=0.6)
    assert _signal_tier(None, config) == "bottom"
    assert _signal_tier(0.29, config) == "bottom"
    assert _signal_tier(0.30, config) == "middle"
    assert _signal_tier(0.59, config) == "middle"
    assert _signal_tier(0.60, config) == "top"
    assert _signal_tier(1.0, config) == "top"


# ---------------------------------------------------------------------------
# Capability Ladder — KNN below MICE (ADR-0094)
# ---------------------------------------------------------------------------


def test_capability_ladder_middle_tier_prefers_knn():
    assert (
        _capability_pick("middle", mice_feasible=True, knn_feasible=True)
        == ImputationStrategy.KNN
    )


def test_capability_ladder_middle_tier_falls_to_mice_when_knn_infeasible():
    assert (
        _capability_pick("middle", mice_feasible=True, knn_feasible=False)
        == ImputationStrategy.MICE
    )


def test_capability_ladder_top_tier_prefers_mice():
    assert (
        _capability_pick("top", mice_feasible=True, knn_feasible=True)
        == ImputationStrategy.MICE
    )


def test_capability_ladder_top_tier_falls_to_knn_when_mice_infeasible():
    assert (
        _capability_pick("top", mice_feasible=False, knn_feasible=True)
        == ImputationStrategy.KNN
    )


def test_capability_ladder_none_feasible_returns_none():
    assert _capability_pick("middle", mice_feasible=False, knn_feasible=False) is None
    assert _capability_pick("top", mice_feasible=False, knn_feasible=False) is None


# ---------------------------------------------------------------------------
# Bottom-tier fill (ADR-0092)
# ---------------------------------------------------------------------------


def test_bottom_tier_fill_bounded_discrete_is_mode():
    cp = _cp(numeric_kind=NumericKind.BoundedDiscrete)
    assert _bottom_tier_fill(cp, bimodal=False) == ImputationStrategy.Mode


def test_bottom_tier_fill_bimodal_branch_2_is_gmm_sampling():
    cp = _cp(numeric_kind=NumericKind.Continuous)
    assert _bottom_tier_fill(cp, bimodal=True) == ImputationStrategy.GMMSampling


def test_bottom_tier_fill_bounded_discrete_wins_over_bimodal():
    cp = _cp(numeric_kind=NumericKind.BoundedDiscrete)
    assert _bottom_tier_fill(cp, bimodal=True) == ImputationStrategy.Mode


def test_bottom_tier_fill_skew_picked_central_tendency():
    cp_normal = _cp(numeric_kind=NumericKind.Continuous, skewness_severity=SkewSeverity.Normal)
    cp_severe = _cp(numeric_kind=NumericKind.Continuous, skewness_severity=SkewSeverity.Severe)
    assert _bottom_tier_fill(cp_normal, bimodal=False) == ImputationStrategy.Mean
    assert _bottom_tier_fill(cp_severe, bimodal=False) == ImputationStrategy.Median


# ---------------------------------------------------------------------------
# The full seam: floor -> score -> ladder, and the bottom tier is never empty
# ---------------------------------------------------------------------------


def test_escalation_seam_picks_knn_at_middle_tier_when_both_feasible():
    config = _config(signal_score_middle_tier_min=0.1, signal_score_top_tier_min=0.9)
    cp = _cp(null_ratio=0.0, total_rows=1000, r2_linear=0.3, r2_rf=0.3)
    feature_correlation = _corr("x", {"a": 0.5, "b": 0.4})
    strategy, reason, evidence = _escalation_seam(
        "x", cp, config, n_rows=1000, n_features=5, feature_correlation=feature_correlation,
        entries=(EntryCause.mcar_high_severity,),
    )
    assert strategy == ImputationStrategy.KNN
    assert reason == RoutingReason.signal_tier_pick
    assert [fact.kind for fact in evidence] == [
        "entry", "signal_score", "tier", "feasibility", "feasibility",
    ]
    assert evidence[0] == EntryFact(cause=EntryCause.mcar_high_severity)
    assert isinstance(evidence[1], SignalScoreFact)
    assert evidence[2] == TierFact(tier=SignalTier.middle)
    assert evidence[3] == FeasibilityFact(strategy="mice", passed=True, failed_terms=[])
    assert evidence[4] == FeasibilityFact(strategy="knn", passed=True, failed_terms=[])


def test_escalation_seam_picks_mice_at_top_tier_when_both_feasible():
    config = _config(signal_score_middle_tier_min=0.05, signal_score_top_tier_min=0.1)
    cp = _cp(null_ratio=0.0, total_rows=1000, r2_linear=0.9, r2_rf=0.9)
    feature_correlation = _corr("x", {"a": 0.9, "b": 0.9})
    strategy, reason, evidence = _escalation_seam(
        "x", cp, config, n_rows=1000, n_features=5, feature_correlation=feature_correlation,
        entries=(EntryCause.mar_severe,),
    )
    assert strategy == ImputationStrategy.MICE
    assert reason == RoutingReason.signal_tier_pick
    assert TierFact(tier=SignalTier.top) in evidence


def test_escalation_seam_bottom_tier_regardless_of_feasibility():
    config = _config(signal_score_middle_tier_min=0.9, signal_score_top_tier_min=0.95)
    cp = _cp(null_ratio=0.0, total_rows=1000, r2_linear=0.3, r2_rf=0.3)
    feature_correlation = _corr("x", {"a": 0.3})
    strategy, reason, evidence = _escalation_seam(
        "x", cp, config, n_rows=1000, n_features=5, feature_correlation=feature_correlation,
        entries=(EntryCause.mcar_high_severity,),
    )
    # _cp() defaults to SkewSeverity.Normal, so the bottom tier's central
    # tendency is Mean.
    assert strategy == ImputationStrategy.Mean
    assert reason == RoutingReason.bottom_tier_fill
    assert TierFact(tier=SignalTier.bottom) in evidence


def test_escalation_seam_no_feasible_model_when_tier_wants_one():
    """A middle-tier column with neither candidate feasible falls to the scalar fill."""
    config = _config(
        signal_score_middle_tier_min=0.1, signal_score_top_tier_min=0.9, knn_max_rows=10
    )
    cp = _cp(null_ratio=0.99, total_rows=100, r2_linear=0.3, r2_rf=0.3)
    feature_correlation = _corr("x", {"a": 0.5, "b": 0.4})
    strategy, reason, evidence = _escalation_seam(
        "x", cp, config, n_rows=100, n_features=5, feature_correlation=feature_correlation,
        entries=(EntryCause.mar_multi,),
    )
    assert strategy == ImputationStrategy.Mean
    assert reason == RoutingReason.no_feasible_model
    mice, knn = (f for f in evidence if f.kind == "feasibility")
    assert (mice.strategy, mice.passed) == ("mice", False)
    assert [t.term for t in mice.failed_terms] == ["rows_per_predictor"]
    assert (knn.strategy, knn.passed) == ("knn", False)
    assert [t.term for t in knn.failed_terms] == ["row_count"]


def test_escalation_seam_never_returns_a_strategy_with_no_candidate():
    """The skew-picked central tendency is always feasible — the set is never empty."""
    config = _config()
    cp = _cp(null_ratio=0.0, total_rows=2)  # tiny frame, no correlation given
    strategy, reason, evidence = _escalation_seam(
        "x", cp, config, n_rows=2, n_features=1, feature_correlation=None, entries=(),
    )
    assert strategy in (ImputationStrategy.Mean, ImputationStrategy.Median)
    assert reason == RoutingReason.no_predictor_correlation
    assert not any(f.kind == "signal_score" for f in evidence)


# ---------------------------------------------------------------------------
# Forced-column feasibility fact (ADR-0091, informed consent — never blocks)
# ---------------------------------------------------------------------------


def _terms(fact: FeasibilityFact) -> list[str]:
    return [t.term for t in fact.failed_terms]


def test_forced_feasibility_passes_when_feasible():
    config = _config(mice_min_rows_per_predictor=2)
    cp = _cp(null_ratio=0.0, total_rows=1000)
    fact = _forced_feasibility(
        "x", cp, config, n_rows=1000, n_features=5, strategy=ImputationStrategy.MICE
    )
    assert fact == FeasibilityFact(strategy="mice", passed=True, failed_terms=[])


def test_forced_feasibility_names_failed_terms_when_infeasible():
    config = _config(mice_min_rows_per_predictor=2)
    cp = _cp(null_ratio=0.99, total_rows=100)
    fact = _forced_feasibility(
        "x", cp, config, n_rows=100, n_features=5, strategy=ImputationStrategy.MICE
    )
    assert fact.passed is False
    (term,) = fact.failed_terms
    assert term.term == "rows_per_predictor"
    assert term.dial.dial == "thresholds.numeric.mice_min_rows_per_predictor"


def test_forced_feasibility_none_for_non_model_strategy():
    config = _config()
    cp = _cp()
    assert _forced_feasibility(
        "x", cp, config, n_rows=100, n_features=5, strategy=ImputationStrategy.Median
    ) is None


def test_forced_feasibility_gmm_sampling_not_flagged_bimodal():
    config = _config()
    cp = _cp()
    fact = _forced_feasibility(
        "x", cp, config, n_rows=100, n_features=5, strategy=ImputationStrategy.GMMSampling
    )
    assert fact.passed is False
    assert _terms(fact) == ["bimodal_flag"]


def test_forced_feasibility_gmm_sampling_bimodal_passes():
    config = _config()
    cp = _cp()
    cp.stats.flags.append(NumericFlag.Bimodal)
    fact = _forced_feasibility(
        "x", cp, config, n_rows=100, n_features=5, strategy=ImputationStrategy.GMMSampling
    )
    assert fact.passed is True


def test_forced_feasibility_cluster_conditional_unmet_conditions():
    config = _config()
    cp = _cp()
    # Unimodal, no grouping var, no correlated features -> both unmet
    fact = _forced_feasibility(
        "x", cp, config, n_rows=100, n_features=5, strategy=ImputationStrategy.ClusterConditional
    )
    assert _terms(fact) == ["bimodal_flag", "correlated_features"]
    assert fact.failed_terms[1].dial.dial == "thresholds.numeric.bimodal_correlation_threshold"

    # Bimodal, but no grouping var and no correlated features -> only grouping/corr unmet
    cp_bi = _cp()
    cp_bi.stats.flags.append(NumericFlag.Bimodal)
    fact_bi = _forced_feasibility(
        "x", cp_bi, config, n_rows=100, n_features=5, strategy=ImputationStrategy.ClusterConditional
    )
    assert _terms(fact_bi) == ["correlated_features"]

    # Unimodal, but has grouping variable -> only bimodal unmet
    config_grouped = _config()
    config_grouped.set_bimodal_grouping_variable("x", "grp")
    fact_grouped = _forced_feasibility(
        "x", cp, config_grouped, n_rows=100, n_features=5, strategy=ImputationStrategy.ClusterConditional
    )
    assert _terms(fact_grouped) == ["bimodal_flag"]

    # Bimodal AND has grouping variable -> passes
    assert _forced_feasibility(
        "x", cp_bi, config_grouped, n_rows=100, n_features=5, strategy=ImputationStrategy.ClusterConditional
    ).passed is True

    # Bimodal AND has correlated features -> passes
    corr = _corr("x", {"y": 0.8})
    assert _forced_feasibility(
        "x",
        cp_bi,
        config,
        n_rows=100,
        n_features=5,
        strategy=ImputationStrategy.ClusterConditional,
        feature_correlation=corr,
    ).passed is True


def test_route_forcing_trainable_strategy_on_unsupported_column_records_evidence_without_raising():
    """Forcing ClusterConditional or GMMSampling on a column that can't support it
    records a failed feasibility fact rather than raising (ADR-0095).
    """
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]})
    config = ImputationConfig()
    config.numeric.set_per_column_strategy("x", ImputationStrategy.GMMSampling)
    config.numeric.set_per_column_strategy("y", ImputationStrategy.ClusterConditional)

    profile = StructuralProfiler().profile(df)
    routing = route(profile, config)

    # Route succeeds without raising
    r_x = routing.column_routings["x"]
    assert r_x.strategy == ImputationStrategy.GMMSampling
    assert r_x.reason == RoutingReason.strategy_forced
    (fact_x,) = r_x.evidence
    assert (fact_x.strategy, fact_x.passed) == ("gmm_sampling", False)
    assert _terms(fact_x) == ["bimodal_flag"]

    r_y = routing.column_routings["y"]
    assert r_y.strategy == ImputationStrategy.ClusterConditional
    assert r_y.reason == RoutingReason.strategy_forced
    (fact_y,) = r_y.evidence
    assert _terms(fact_y) == ["bimodal_flag", "correlated_features"]


# ---------------------------------------------------------------------------
# resolve_choice is total over NonlinearityTag (ADR-0094) — end-to-end via route()
# ---------------------------------------------------------------------------


def test_route_resolves_mice_model_choice_when_a_column_escalates_to_mice():
    """A column that lands on MICE via the escalation point gets a resolved,
    non-None mice_model_choice off the Estimator Ladder.

    Profiles a real frame for realistic column shape, then overwrites the
    column's stats and pairwise correlations with known ground-truth values
    so this test exercises ``route()``'s own wiring rather than depending on
    ``CorrelationProfiler``/``NonlinearityProfiler``'s numerical behaviour on
    a column with missing values (unrelated to this escalation point).
    """
    import dataclasses

    rng = np.random.default_rng(3)
    n = 800
    a = rng.normal(0, 1, n)
    b = a * 2.0 + rng.normal(0, 0.1, n)
    target = a * 3.0 + b * 2.0 + rng.normal(0, 0.05, n)
    target_missing = target.copy()
    missing_idx = rng.choice(n, int(n * 0.3), replace=False)
    target_missing[missing_idx] = np.nan

    df = pl.DataFrame({"a": a, "b": b, "target": target_missing})
    profile = StructuralProfiler(
        ProfileConfig(compute_correlation=True, compute_nonlinearity=True)
    ).profile(df)

    target_cp = profile.columns["target"]
    target_cp.stats.r2_linear = 0.95
    target_cp.stats.r2_rf = 0.95

    profile.dataset.feature_correlation = dataclasses.replace(
        profile.dataset.feature_correlation,
        pearson_matrix={
            "target": {"a": 0.9, "b": 0.9},
            "a": {"target": 0.9, "b": 0.9},
            "b": {"target": 0.9, "a": 0.9},
        },
    )

    cfg = ImputationConfig()
    cfg.numeric.mice_correlation_threshold = 0.1
    cfg.numeric.signal_score_top_tier_min = 0.5
    routing = route(profile, cfg)

    r = routing.column_routings["target"]
    assert r.strategy in (ImputationStrategy.MICE, ImputationStrategy.KNN)
    if r.strategy == ImputationStrategy.MICE:
        assert routing.mice_model_choice is not None
        assert isinstance(routing.mice_model_choice, ModelChoice)


def test_route_escalates_a_nan_encoded_column_like_a_null_encoded_one():
    """A Severe MCAR column tightly correlated with a predictor reaches MICE
    whether its missing cells arrive as NaN or null. NaN used to zero the
    Pearson matrix and score every such column into the bottom tier."""
    rng = np.random.default_rng(0)
    n = 400
    p = rng.normal(0, 10, n)
    q = rng.normal(0, 1, n)
    t = p * 2.0 + rng.normal(0, 0.5, n)
    t[rng.choice(n, 150, replace=False)] = np.nan

    strategies = []
    for t_series in (pl.Series(t), pl.Series(t).fill_nan(None)):
        frame = pl.DataFrame({"p": p, "q": q, "t": t_series})
        profile = StructuralProfiler(
            ProfileConfig(compute_correlation=True, compute_nonlinearity=True)
        ).profile(frame)
        strategies.append(route(profile).column_routings["t"].strategy)

    assert strategies == [ImputationStrategy.MICE, ImputationStrategy.MICE]


# ---------------------------------------------------------------------------
# Every routed column carries a Routing Reason (ADR-0105)
# ---------------------------------------------------------------------------


def test_route_gives_every_column_a_reason_including_every_passthrough():
    df = pl.DataFrame(
        {
            "full": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "soft": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "mnar": [1.0, None, 3.0, 4.0, 5.0, 6.0],
            "dropme": [1.0, None, None, None, None, 6.0],
            "label": ["a", None, "b", "a", "b", "a"],
        }
    )
    config = ImputationConfig()
    config.add_mnar_column("mnar")
    profile = StructuralProfiler().profile(df)
    routing = route(profile, config, skip_columns=["soft"])

    reasons = {c: r.reason for c, r in routing.column_routings.items()}
    assert all(isinstance(r, RoutingReason) for r in reasons.values())
    assert reasons["full"] == RoutingReason.no_missing_values
    assert reasons["soft"] == RoutingReason.soft_excluded
    assert reasons["label"] == RoutingReason.not_imputed_type
    assert reasons["mnar"] == RoutingReason.mnar_declared
    assert reasons["mnar_missing"] == RoutingReason.mnar_declared
    assert reasons["dropme"] == RoutingReason.drop_candidate
    for col in ("full", "soft", "label", "mnar", "dropme"):
        assert routing.column_routings[col].evidence == ()
