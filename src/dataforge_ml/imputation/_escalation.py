"""The single model-based escalation point: Feasibility Floor → Signal Score → Capability Ladder.

Every routing path that would otherwise run its own "is a model worth it
here" check calls :func:`_escalation_seam` instead of picking ``KNN`` or
``MICE`` directly. The seam answers two questions in order, for one target
column:

1. **Feasibility Floor** (ADR-0091) — *can MICE / KNN even be trained here?*
   A non-compensatory gate on two Feasibility Terms, Usable Rows and Rows per
   Predictor. Only MICE's Rows per Predictor carries a bound
   (``mice_min_rows_per_predictor``, ADR-0097); KNN carries no floor bound,
   only its Resource Ceiling (``knn_max_rows``, reading raw row count).
2. **Signal Score** (ADR-0092) — *is a model worth it over a scalar, for the
   feasible candidates?* A graded [0, 1] reading (or ``None`` when no
   predictor correlation exists at all) that falls into one of three Signal
   Tiers. The bottom tier always takes a scalar fill regardless of
   feasibility; the middle and top tiers pick from the feasible candidates
   via the Capability Ladder (ADR-0094): KNN below MICE.

The skew-picked central tendency (Mode for BoundedDiscrete, GMM Sampling for
a bimodal column reaching this point through branch 2, else the ordinary
Mean/Median split) is always feasible, so the returned strategy is never
absent.

Forced columns (``per_column_strategy`` declaring KNN or MICE) never call the
seam — they bypass routing priorities 2-7 entirely (ADR-0082). They still get
the Feasibility Floor and Resource Ceiling evaluated, via
:func:`_forced_feasibility`, with any failed term recorded as a
``feasibility`` fact only — informed consent, never a block (ADR-0071,
ADR-0088, ADR-0091).

Every builder here returns typed Routing Evidence (ADR-0105), never prose.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ..profiling._config import NumericKind
from ..profiling._numeric_config import (
    NonlinearityTag,
    NumericFlag,
    NumericStats,
    SkewSeverity,
)
from ..records import DialPointer
from ._config import ImputationStrategy, NumericImputationConfig
from ._evidence import (
    EntryCause,
    EntryFact,
    FailedTerm,
    FeasibilityFact,
    RoutingEvidence,
    RoutingReason,
    SignalScoreFact,
    SignalTier,
    TierFact,
)

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile
    from ..profiling._correlation_config import CorrelationProfileResult

__all__ = [
    "_Routed",
    "_escalation_seam",
    "_forced_feasibility",
    "_mice_winning_tag",
    "_resolve_nonlinearity_tag",
]

# The Estimator Ladder's tag precedence (ADR-0094): the MICE block's winning
# tag is the most complex one across its columns, so its estimator ceiling is
# set by the richest column, never diluted by the plainest.
_MICE_TAG_PRECEDENCE: dict[NonlinearityTag, int] = {
    NonlinearityTag.Unpredictable: 0,
    NonlinearityTag.Linear: 1,
    NonlinearityTag.MonotonicNonlinear: 2,
    NonlinearityTag.ComplexNonlinear: 3,
}


def _resolve_nonlinearity_tag(stats: NumericStats | None) -> NonlinearityTag:
    """Resolve the effective ``NonlinearityTag`` for the Estimator Ladder.

    An absent tag defaults to ``Linear``, and a bimodal ``Linear`` column is
    bumped to ``MonotonicNonlinear`` — a linear model would straddle the two
    modes.
    """
    tag = (
        stats.nonlinearity_tag
        if stats is not None and stats.nonlinearity_tag is not None
        else NonlinearityTag.Linear
    )
    if (
        stats is not None
        and stats.bimodal_stats is not None
        and tag == NonlinearityTag.Linear
    ):
        tag = NonlinearityTag.MonotonicNonlinear
    return tag


def _mice_winning_tag(tags: list[NonlinearityTag]) -> NonlinearityTag:
    """The MICE block's winning tag: the most complex one across its columns."""
    return max(tags, key=lambda t: _MICE_TAG_PRECEDENCE.get(t, 0))


def _column_stats(cp: ColumnProfile) -> NumericStats | None:
    """Return the column's NumericStats, or None when unavailable."""
    return cp.stats if isinstance(cp.stats, NumericStats) else None


def _usable_rows(n_rows: int, cp: ColumnProfile) -> float:
    """Usable Rows: rows the target is not an Effective Null on (ADR-0091)."""
    ratio = cp.missingness.effective_null_ratio if cp.missingness is not None else 0.0
    return n_rows * (1.0 - ratio)


def _predictor_count(n_features: int, *, capped: int | None = None) -> int:
    """Active numeric columns other than the target, optionally capped."""
    count = max(0, n_features - 1)
    if capped is not None:
        count = min(count, capped)
    return count


@dataclass(frozen=True)
class _FeasibilityCheck:
    """One candidate's Feasibility Floor / Resource Ceiling verdict.

    Parameters
    ----------
    feasible : bool
        Whether the candidate passed every term it is evaluated on.
    failed_terms : tuple[FailedTerm, ...]
        One typed record per failed term. Empty when ``feasible`` is ``True``.
    """

    feasible: bool
    failed_terms: tuple[FailedTerm, ...]

    def to_fact(self, strategy: ImputationStrategy) -> FeasibilityFact:
        """The verdict as a ``feasibility`` Routing Evidence fact."""
        return FeasibilityFact(
            strategy=strategy.value,
            passed=self.feasible,
            failed_terms=list(self.failed_terms),
        )


class _Routed(NamedTuple):
    """A routing outcome: the strategy, its Routing Reason, and the evidence walk."""

    strategy: ImputationStrategy
    reason: RoutingReason
    evidence: tuple[RoutingEvidence, ...] = ()


_NUMERIC_DIAL_DEFAULTS: dict[str, Any] = {
    f.name: f.default
    for f in fields(NumericImputationConfig)
    if f.default is not MISSING
}


def _dial(config: NumericImputationConfig, name: str) -> DialPointer:
    """The Dial Pointer for one ``NumericImputationConfig`` dial ``route`` read (ADR-0106)."""
    value = getattr(config, name)
    return DialPointer(
        operation="route",
        dial=f"thresholds.numeric.{name}",
        value=value,
        source="default" if value == _NUMERIC_DIAL_DEFAULTS.get(name) else "argument",
    )


def _no_predictors() -> FailedTerm:
    """The structural failure of a target with no predictor: no dial changes it."""
    return FailedTerm(term="predictors", value=0, limit=1, dial=None)


def _feasibility_mice(
    cp: ColumnProfile, n_rows: int, n_features: int, config: NumericImputationConfig
) -> _FeasibilityCheck:
    """Evaluate MICE's Feasibility Floor for one target column (ADR-0091, ADR-0097).

    Zero predictors fails Rows per Predictor outright — no division, no
    infinite pass. Otherwise the sole surviving bound,
    ``mice_min_rows_per_predictor``, is checked. MICE carries no Resource
    Ceiling: a column is never moved to a cheaper strategy to fit the machine.
    """
    predictors = _predictor_count(n_features, capped=config.mice_max_nearest_features)
    if predictors == 0:
        return _FeasibilityCheck(False, (_no_predictors(),))
    usable = _usable_rows(n_rows, cp)
    rows_per_predictor = usable / predictors
    if rows_per_predictor < config.mice_min_rows_per_predictor:
        return _FeasibilityCheck(
            False,
            (
                FailedTerm(
                    term="rows_per_predictor",
                    value=float(rows_per_predictor),
                    limit=config.mice_min_rows_per_predictor,
                    dial=_dial(config, "mice_min_rows_per_predictor"),
                ),
            ),
        )
    return _FeasibilityCheck(True, ())


def _feasibility_knn(
    cp: ColumnProfile, n_rows: int, n_features: int, config: NumericImputationConfig
) -> _FeasibilityCheck:
    """Evaluate KNN's Feasibility Floor and Resource Ceiling (ADR-0091, ADR-0093).

    KNN carries no floor bound on Rows per Predictor — only the structural
    "zero predictors" failure the term itself is undefined on — plus the
    Resource Ceiling ``knn_max_rows``, which reads the raw row count because
    memory scales with the whole matrix, not with observed values.
    """
    predictors = _predictor_count(n_features)
    failed: list[FailedTerm] = []
    if predictors == 0:
        failed.append(_no_predictors())
    if n_rows > config.knn_max_rows:
        failed.append(
            FailedTerm(
                term="row_count",
                value=n_rows,
                limit=config.knn_max_rows,
                dial=_dial(config, "knn_max_rows"),
            )
        )
    return _FeasibilityCheck(not failed, tuple(failed))


def _signal_score(
    col: str,
    cp: ColumnProfile,
    feature_correlation: CorrelationProfileResult | None,
    config: NumericImputationConfig,
) -> tuple[float | None, SignalScoreFact | None]:
    """Compute the Signal Score for one target column (ADR-0092).

    ``score = clamp(base + w_breadth * breadth, min=w_latent * latent, max=1)``
    where ``base`` is Explainable Variance, ``latent`` is Latent Structure and
    ``breadth`` is Signal Breadth. ``(None, None)`` only when no predictor
    correlation exists at all — the column then routes to the bottom Signal
    Tier.
    """
    corrs = (
        feature_correlation.pearson_matrix.get(col, {})
        if feature_correlation is not None
        else {}
    )
    abs_rs = [abs(r) for c, r in corrs.items() if c != col and r is not None]
    if not abs_rs:
        return None, None
    max_abs_r = max(abs_rs)

    stats = _column_stats(cp)
    r2_linear = stats.r2_linear if stats is not None else None
    r2_rf = stats.r2_rf if stats is not None else None
    degraded = r2_linear is None or r2_rf is None
    if degraded:
        base = max_abs_r ** 2
    else:
        r2_best = max(max(r2_linear, 0.0), max(r2_rf, 0.0))
        base = max(r2_best, max_abs_r ** 2)

    max_mi = stats.max_mutual_information if stats is not None else None
    latent = 1.0 - float(np.exp(-2.0 * max_mi)) if max_mi is not None else 0.0

    breadth_count = sum(1 for r in abs_rs if r > config.mice_correlation_threshold)
    breadth = breadth_count / (breadth_count + 1)

    score = base + config.signal_score_breadth_weight * breadth
    score = max(score, config.signal_score_latent_weight * latent)
    score = min(score, 1.0)

    fact = SignalScoreFact(
        score=float(score),
        base=float(base),
        latent=float(latent),
        breadth=float(breadth),
        predictors=breadth_count,
        degraded=degraded,
    )
    return score, fact


def _signal_tier(score: float | None, config: NumericImputationConfig) -> SignalTier:
    """The Signal Tier a score falls in."""
    if score is None or score < config.signal_score_middle_tier_min:
        return SignalTier.bottom
    if score < config.signal_score_top_tier_min:
        return SignalTier.middle
    return SignalTier.top


def _capability_pick(
    tier: SignalTier, mice_feasible: bool, knn_feasible: bool
) -> ImputationStrategy | None:
    """The Capability Ladder's pick for a tier: KNN below MICE (ADR-0094).

    Middle tier takes KNN if feasible, else MICE. Top tier takes MICE if
    feasible, else KNN. ``None`` when neither candidate is feasible — the
    caller falls back to the bottom tier's scalar fill.
    """
    if tier == SignalTier.middle:
        if knn_feasible:
            return ImputationStrategy.KNN
        if mice_feasible:
            return ImputationStrategy.MICE
        return None
    if tier == SignalTier.top:
        if mice_feasible:
            return ImputationStrategy.MICE
        if knn_feasible:
            return ImputationStrategy.KNN
        return None
    return None


def _bottom_tier_fill(cp: ColumnProfile, bimodal: bool) -> ImputationStrategy:
    """The bottom Signal Tier's scalar fill (ADR-0092).

    Mode for BoundedDiscrete, GMM Sampling for a bimodal column reaching the
    escalation point through branch 2 (a median would sit in the valley
    between the peaks), else the skew-picked central tendency.
    """
    if cp.numeric_kind == NumericKind.BoundedDiscrete:
        return ImputationStrategy.Mode
    if bimodal:
        return ImputationStrategy.GMMSampling
    stats = _column_stats(cp)
    skew_sev = stats.skewness_severity if stats is not None else None
    if skew_sev in (None, SkewSeverity.Normal):
        return ImputationStrategy.Mean
    return ImputationStrategy.Median


def _escalation_seam(
    col: str,
    cp: ColumnProfile,
    config: NumericImputationConfig,
    n_rows: int,
    n_features: int,
    feature_correlation: CorrelationProfileResult | None,
    entries: tuple[EntryCause, ...],
    bimodal: bool = False,
) -> _Routed:
    """The one seam every model-worth-it routing branch calls (ADR-0091, ADR-0092, ADR-0094).

    Evaluates the Feasibility Floor for both candidates, the Signal Score,
    and — for the middle and top Signal Tiers — the Capability Ladder pick
    among the feasible candidates. The skew-picked central tendency (or Mode
    / GMM Sampling) is always feasible, so a strategy is always returned.

    The evidence is, in order: one ``entry`` fact per cause in ``entries``,
    the ``signal_score`` fact (absent when no predictor correlation exists),
    the ``tier`` fact, then the ``feasibility`` facts for MICE and KNN.

    Parameters
    ----------
    col : str
        Column name.
    cp : ColumnProfile
        Phase 1 profile for the column.
    config : NumericImputationConfig
        Imputation configuration supplying the floor bound, the Signal Score
        dials, and the Resource Ceiling.
    n_rows : int
        Number of rows routing is performed against — the routing profile's
        own row count (ADR-0091).
    n_features : int
        Total number of active numeric columns.
    feature_correlation : CorrelationProfileResult, optional
        Pre-computed pairwise Pearson correlation matrix from Phase 1.
    entries : tuple[EntryCause, ...]
        The routing rules that sent the column here, outermost first.
    bimodal : bool, default False
        Whether this column reached the seam through the Bimodal Imputation
        Framework's branch 2 — selects GMM Sampling as the bottom tier's fill
        for a non-BoundedDiscrete column.

    Returns
    -------
    _Routed
        The resolved strategy, its Routing Reason, and the evidence walk.
    """
    mice_check = _feasibility_mice(cp, n_rows, n_features, config)
    knn_check = _feasibility_knn(cp, n_rows, n_features, config)
    score, score_fact = _signal_score(col, cp, feature_correlation, config)
    tier = _signal_tier(score, config)

    evidence: list[RoutingEvidence] = [EntryFact(cause=c) for c in entries]
    if score_fact is not None:
        evidence.append(score_fact)
    evidence.append(TierFact(tier=tier))
    evidence.append(mice_check.to_fact(ImputationStrategy.MICE))
    evidence.append(knn_check.to_fact(ImputationStrategy.KNN))

    if tier != SignalTier.bottom:
        picked = _capability_pick(tier, mice_check.feasible, knn_check.feasible)
        if picked is not None:
            return _Routed(picked, RoutingReason.signal_tier_pick, tuple(evidence))
        reason = RoutingReason.no_feasible_model
    elif score is None:
        reason = RoutingReason.no_predictor_correlation
    else:
        reason = RoutingReason.bottom_tier_fill

    return _Routed(_bottom_tier_fill(cp, bimodal), reason, tuple(evidence))


def _forced_feasibility(
    col: str,
    cp: ColumnProfile,
    config: NumericImputationConfig,
    n_rows: int,
    n_features: int,
    strategy: ImputationStrategy,
    feature_correlation: CorrelationProfileResult | None = None,
) -> FeasibilityFact | None:
    """Evaluate the Feasibility Floor and unmet conditions for a ``per_column_strategy``-forced column.

    Forcing a column past a routing gate is informed consent (ADR-0071,
    ADR-0088, ADR-0095): the unit trains regardless, and every failed term
    or unmet condition is recorded as a ``feasibility`` fact rather than
    blocking. ``None`` when ``strategy`` is not ``KNN``, ``MICE``,
    ``ClusterConditional``, or ``GMMSampling``.

    Parameters
    ----------
    col : str
        Column name.
    cp : ColumnProfile
        Phase 1 profile for the column.
    config : NumericImputationConfig
        Imputation configuration.
    n_rows : int
        Number of rows routing is performed against.
    n_features : int
        Total number of active numeric columns.
    strategy : ImputationStrategy
        The user-forced strategy.
    feature_correlation : CorrelationProfileResult, optional
        Pre-computed Pearson correlation matrix from Phase 1.

    Returns
    -------
    FeasibilityFact or None
        The forced strategy's verdict, ``passed`` false when any term or
        condition failed, or ``None`` for a strategy with no floor.
    """
    if strategy == ImputationStrategy.MICE:
        return _feasibility_mice(cp, n_rows, n_features, config).to_fact(strategy)
    if strategy == ImputationStrategy.KNN:
        return _feasibility_knn(cp, n_rows, n_features, config).to_fact(strategy)
    if strategy not in (
        ImputationStrategy.ClusterConditional,
        ImputationStrategy.GMMSampling,
    ):
        return None
    unmet: list[FailedTerm] = []
    stats = _column_stats(cp)
    if stats is None or not stats.has_flag(NumericFlag.Bimodal):
        unmet.append(
            FailedTerm(term="bimodal_flag", value=False, limit=True, dial=None)
        )
    if (
        strategy == ImputationStrategy.ClusterConditional
        and col not in config.bimodal_grouping_variables
    ):
        col_corrs = (
            feature_correlation.pearson_matrix.get(col, {})
            if feature_correlation is not None
            else {}
        )
        correlated = sum(
            1
            for c, r in col_corrs.items()
            if c != col
            and r is not None
            and abs(r) > config.bimodal_correlation_threshold
        )
        if correlated == 0:
            unmet.append(
                FailedTerm(
                    term="correlated_features",
                    value=0,
                    limit=1,
                    dial=_dial(config, "bimodal_correlation_threshold"),
                )
            )
    return _FeasibilityCheck(not unmet, tuple(unmet)).to_fact(strategy)
