"""
Decision assembler for Phase 2 imputation — the pure decide-time layer.

``decide(profile, n_rows, config)`` builds an :class:`ImputationDecision`: a
complete, value-free imputation plan derived purely from
``(profile, shape, config)``, where ``n_rows`` *is* the shape input. The
assembler reuses the routing kernel ``_StrategyRouter.route()`` (which keeps its
one job — return ``(strategy, signals)``), resolves each column's concrete
``ModelChoice`` at decide-time via ``RegressionEstimatorFactory``, surfaces
``domain_snap_bounds``, folds in the indicator/mnar/drop flags, and materialises
the execution units. No training data is touched — routing is a pure function of
the Phase 1 profile (ADR-0060).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..config import PipelineConfig, PipelinePhase, SemanticType
from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag
from ..profiling._numeric_config import NonlinearityTag, NumericStats, SkewSeverity
from ._config import (
    _EXCLUSION_SIGNAL,
    ColumnImputationDecision,
    ImputationDecision,
    ImputationStrategy,
    ModelChoice,
    NumericImputationConfig,
)
from ._regression_estimator_factory import RegressionEstimatorFactory
from ._strategy_router import _StrategyRouter

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile, StructuralProfileResult


_MICE_TAG_PRECEDENCE: dict[NonlinearityTag, int] = {
    NonlinearityTag.Unpredictable: 0,
    NonlinearityTag.Linear: 1,
    NonlinearityTag.MonotonicNonlinear: 2,
    NonlinearityTag.ComplexNonlinear: 3,
}


def _mice_winning_tag(tags: list[NonlinearityTag]) -> NonlinearityTag:
    return max(tags, key=lambda t: _MICE_TAG_PRECEDENCE.get(t, 0))


def _resolve_nonlinearity_tag(
    stats: "Optional[NumericStats]",
) -> NonlinearityTag:
    """Resolve the effective NonlinearityTag for model-choice selection.

    Mirrors the fit-time resolution: an absent tag defaults to ``Linear``, and a
    bimodal ``Linear`` column is bumped to ``MonotonicNonlinear`` (a linear model
    would straddle the two modes).
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


def _column_stats(cp: "ColumnProfile") -> "Optional[NumericStats]":
    """Return the column's NumericStats, or None when unavailable."""
    return cp.stats if isinstance(cp.stats, NumericStats) else None


def _raise_on_excluded_column_overrides(
    excluded: set[str],
    *,
    mnar_columns: set[str],
    per_column_strategy: dict[str, Any],
    per_column_constant_fill: dict[str, float],
    add_indicator_columns: tuple[str, ...],
) -> None:
    """Raise when an excluded column is also named in imputation config.

    The user typed "impute this column like so" and "don't impute this column"
    in the same config; neither side may silently win, so the contradiction
    surfaces as one ``ValueError`` at decide-time — before any routing — listing
    every offending column across the four override surfaces. Phase 1 keeps its
    silent-ignore of overrides for excluded columns; the asymmetry is
    deliberate.
    """
    if not excluded:
        return
    surfaces: dict[str, set[str]] = {
        "mnar_columns": mnar_columns,
        "per_column_strategy": set(per_column_strategy),
        "per_column_constant_fill": set(per_column_constant_fill),
        "add_indicator_columns": set(add_indicator_columns),
    }
    offending = {
        surface: sorted(excluded & cols)
        for surface, cols in surfaces.items()
        if excluded & cols
    }
    if not offending:
        return
    details = "; ".join(
        f"{surface}: {', '.join(repr(c) for c in cols)}"
        for surface, cols in offending.items()
    )
    raise ValueError(
        f"Columns excluded for the Imputation phase are also named in "
        f"imputation config, which is contradictory: {details}. "
        f"Remove the exclusion or the conflicting config entry."
    )


def decide(
    profile: "StructuralProfileResult",
    n_rows: int,
    config: Optional[PipelineConfig] = None,
) -> ImputationDecision:
    """Build the pure, immutable imputation plan for a profiled dataset.

    The single decide-time entry point of the Decision/Execution split
    (ADR-0060). Routes every profiled column to its imputation strategy, resolves
    the concrete estimator family for model-based columns, surfaces domain-snap
    bounds and the indicator/mnar/drop flags, and materialises the execution
    units — all as a pure function of ``(profile, shape, config)`` that trains
    nothing and never touches a DataFrame.

    ``decide`` resolves its own active column set for the Imputation phase from
    ``config`` (Hard Exclusions plus Imputation-phase Soft Exclusions) — no
    upstream filtering is relied on. A hard-excluded column is omitted from the
    plan entirely (recoverable via the config snapshot); a soft-excluded column
    stays visible as a ``Passthrough`` decision carrying a signal that records
    the exclusion. Excluded columns leave the numeric block wholesale: they are
    neither targets nor predictors, and every block-derived value — feature
    counts, MICE/KNN block membership, multi-MAR counting, regression and
    correlated-feature lists, block hyperparameter signals, and the plan's
    decided-for shape — is computed over active columns only.

    The plan is decided for the shape execution will face, so ``n_rows`` is the
    **train** row count, not the profile's full-dataset count (ADR-0066). It
    feeds both the size guards and the row-dependent hyperparameters (ADR-0062),
    so a plan decided against the wrong count can request a model the train split
    cannot train. A caller genuinely planning against the whole dataset passes
    ``profile.dataset.row_count`` explicitly.

    Parameters
    ----------
    profile : StructuralProfileResult
        Full-dataset Phase 1 profile. Supplies the routing signals, the
        imputable-column population, and the declared sentinel maps carried onto
        the plan (ADR-0068).
    n_rows : int
        Row count the plan is decided for — the train split's. Only the row
        dimension is accepted: splitting is row-wise, so ``n_features`` and the
        column set stay derived from ``profile``.
    config : PipelineConfig, optional
        Pipeline configuration supplying the imputation thresholds, MNAR
        declarations, per-column overrides, and the exclusion lists resolved
        into the Imputation-phase active column set. Defaults to
        ``PipelineConfig()`` when omitted.

    Returns
    -------
    ImputationDecision
        The immutable plan: per-column decisions, materialised units, the shape
        it was decided for, the config snapshot, the declared sentinel maps, and
        the profile provenance.

    Raises
    ------
    ValueError
        If ``n_rows`` is negative, or if a hard- or Imputation-soft-excluded
        column is also named in ``mnar_columns``, ``per_column_strategy``,
        ``per_column_constant_fill``, or ``add_indicator_columns`` —
        contradictory configuration surfaces at decide-time, before any
        routing, with every offending column reported in one raise.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative; got {n_rows}.")
    config = config or PipelineConfig()
    config.imputation.validate()

    imp_cfg = config.imputation
    numeric_cfg: NumericImputationConfig = imp_cfg.numeric
    mnar_columns = set(imp_cfg.mnar_columns)
    per_column_strategy = dict(numeric_cfg.per_column_strategy or {})
    per_column_constant_fill = dict(numeric_cfg.per_column_constant_fill or {})

    # Resolve the Imputation-phase active column set from the config it already
    # receives — the phase's own exclusion enforcement, never inherited from an
    # upstream phase. Hard-excluded columns are omitted from the plan entirely;
    # Imputation-soft-excluded columns stay visible as Passthrough.
    hard_excluded = set(config.exclude_columns)
    soft_excluded = set(
        config.phase_exclusions.get(PipelinePhase.Imputation, ())
    ) - hard_excluded
    active_cols = set(
        config.resolve_active_columns(PipelinePhase.Imputation, list(profile.columns))
    )
    _raise_on_excluded_column_overrides(
        hard_excluded | soft_excluded,
        mnar_columns=mnar_columns,
        per_column_strategy=per_column_strategy,
        per_column_constant_fill=per_column_constant_fill,
        add_indicator_columns=imp_cfg.add_indicator_columns,
    )

    numeric_cols = [
        col
        for col, cp in profile.columns.items()
        if cp.semantic_type == SemanticType.Numeric and col in active_cols
    ]
    n_features = len(numeric_cols)

    # Detect multi-MAR: >=2 non-dropped, non-MNAR columns carrying MARSuspect.
    mar_candidates: set[str] = set()
    for col in numeric_cols:
        cp = profile.columns[col]
        if cp.missingness is None or col in mnar_columns:
            continue
        if cp.missingness.has_flag(MissingnessFlag.DropCandidate):
            continue
        if cp.missingness.has_flag(MissingnessFlag.MARSuspect):
            mar_candidates.add(col)
    multi_mar = len(mar_candidates) >= 2

    # Excluded columns are banned as predictors too: restrict the pairwise
    # correlation view to the active numeric block so no routing branch or
    # hyperparameter signal ever counts an excluded column as a feature.
    feature_correlation = profile.dataset.feature_correlation
    if feature_correlation is not None and (hard_excluded or soft_excluded):
        feature_correlation = _filter_feature_correlation(
            feature_correlation, set(numeric_cols)
        )
    router = _StrategyRouter()

    # First pass: route every numeric column, recording strategy, signals, and
    # profile-derived descriptors. Non-numeric semantic columns pass through.
    routed: dict[str, tuple[ImputationStrategy, tuple[str, ...]]] = {}
    forced_columns: dict[str, bool] = {}
    for col in numeric_cols:
        cp = profile.columns[col]
        strategy, signals, forced = router.route(
            col=col,
            cp=cp,
            config=numeric_cfg,
            n_rows=n_rows,
            n_features=n_features,
            multi_mar=multi_mar,
            mnar_columns=mnar_columns,
            feature_correlation=feature_correlation,
            per_column_strategy=per_column_strategy,
            per_column_constant_fill=per_column_constant_fill,
        )
        routed[col] = (strategy, tuple(signals))
        forced_columns[col] = forced

    # The MICE block trains one estimator chosen from the whole block's winning
    # tag, so every MICE column carries that same block-level model choice.
    mice_cols = [c for c, (s, _) in routed.items() if s == ImputationStrategy.MICE]
    mice_model_choice: Optional[ModelChoice] = None
    mice_hyperparameters: Optional[tuple[tuple[str, Any], ...]] = None
    if mice_cols:
        mice_tags = [
            _resolve_nonlinearity_tag(_column_stats(profile.columns[c]))
            for c in mice_cols
        ]
        winning_tag = _mice_winning_tag(mice_tags)
        mice_model_choice = RegressionEstimatorFactory.resolve_choice(
            winning_tag, n_rows, numeric_cfg
        )
        mice_stats = [_column_stats(profile.columns[c]) for c in mice_cols]
        mice_max_iter = _compute_mice_max_iter(
            winning_tag,
            mice_stats,
            _block_miss_fraction(profile, mice_cols),
            _max_pairwise_pearson(feature_correlation, mice_cols),
            _complete_row_fraction(profile),
            numeric_cfg,
        )
        if numeric_cfg.mice_max_iter is not None:
            mice_max_iter = numeric_cfg.mice_max_iter
        mice_tol = _compute_mice_tol(winning_tag, mice_stats)
        mice_initial_strategy = _mice_initial_strategy(mice_stats)
        mice_n_nearest, _ = _compute_mice_n_nearest_features(
            feature_correlation, mice_cols, numeric_cols, numeric_cfg
        )
        mice_hyperparameters = tuple(
            {
                "max_iter": mice_max_iter,
                "tol": mice_tol,
                "initial_strategy": mice_initial_strategy,
                "n_nearest_features": mice_n_nearest,
                "miss_frac": _block_miss_fraction(profile, mice_cols),
                "complete_frac": _complete_row_fraction(profile),
                "nonlinearity_tag": str(winning_tag),
            }.items()
        )

    knn_cols = [c for c, (s, _) in routed.items() if s == ImputationStrategy.KNN]
    knn_hyperparameters: Optional[tuple[tuple[str, Any], ...]] = None
    if knn_cols:
        knn_n_neighbors, knn_weights = _compute_knn_params(
            n_rows,
            len(knn_cols),
            _block_miss_fraction(profile, knn_cols),
            _complete_row_fraction(profile),
            numeric_cfg,
        )
        if numeric_cfg.knn_n_neighbors is not None:
            knn_n_neighbors = numeric_cfg.knn_n_neighbors
        knn_hyperparameters = tuple(
            {
                "n_neighbors": knn_n_neighbors,
                "weights": knn_weights,
                "miss_frac": _block_miss_fraction(profile, knn_cols),
                "complete_frac": _complete_row_fraction(profile),
            }.items()
        )

    decided_hyperparameters = {}
    if mice_hyperparameters:
        decided_hyperparameters["mice"] = mice_hyperparameters
    if knn_hyperparameters:
        decided_hyperparameters["knn"] = knn_hyperparameters

    # Second pass: build the per-column decisions, resolving model_choice.
    decisions: dict[str, ColumnImputationDecision] = {}
    for col, cp in profile.columns.items():
        if col in hard_excluded:
            continue  # omitted from the plan; recoverable via config_snapshot
        if cp.semantic_type is None:
            continue
        if col in soft_excluded:
            decisions[col] = ColumnImputationDecision(
                column=col,
                semantic_type=cp.semantic_type,
                strategy=ImputationStrategy.Passthrough,
                signals=(_EXCLUSION_SIGNAL,),
            )
            continue
        if cp.semantic_type != SemanticType.Numeric:
            decisions[col] = ColumnImputationDecision(
                column=col,
                semantic_type=cp.semantic_type,
                strategy=ImputationStrategy.Passthrough,
            )
            continue

        strategy, signals = routed[col]
        model_choice = _resolve_model_choice(
            strategy=strategy,
            mice_model_choice=mice_model_choice,
        )

        if strategy in (ImputationStrategy.MICE, ImputationStrategy.KNN):
            pass  # Resolved block-wide above.
        elif strategy == ImputationStrategy.MNAR:
            decided_hyperparameters[f"{strategy}:{col}"] = tuple(
                {"central_tendency": _mnar_central_tendency(cp)}.items()
            )
        elif strategy in (
            ImputationStrategy.GMMSampling,
            ImputationStrategy.ClusterConditional,
        ):
            decided_hyperparameters[f"{strategy}:{col}"] = _bimodal_hyperparameters(
                col, cp, strategy, feature_correlation, numeric_cfg
            )

        decisions[col] = ColumnImputationDecision(
            column=col,
            semantic_type=SemanticType.Numeric,
            strategy=strategy,
            signals=signals,
            model_choice=model_choice,
            domain_snap_bounds=_resolve_domain_snap_bounds(cp, strategy),
            indicator_flag=strategy == ImputationStrategy.MNAR,
            mnar=strategy == ImputationStrategy.MNAR,
            drop=strategy == ImputationStrategy.Dropped,
            forced=forced_columns[col],
        )

    # Indicator pass: pre-register the {col}_missing columns the MNAR mechanism
    # appends at transform time, mirroring the fitted record set.
    for col, decision in list(decisions.items()):
        if not decision.indicator_flag:
            continue
        indicator_col = f"{col}_missing"
        decisions[indicator_col] = ColumnImputationDecision(
            column=indicator_col,
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
        )

    return ImputationDecision(
        column_decisions=decisions,
        decided_for_shape=(n_rows, n_features, tuple(numeric_cols)),
        config_snapshot=config.to_dict(),
        profile_provenance={
            "row_count": profile.dataset.row_count,
        },
        decided_hyperparameters=decided_hyperparameters,
        numeric_sentinels={k: list(v) for k, v in profile.numeric_sentinels.items()},
        string_sentinels={k: list(v) for k, v in profile.string_sentinels.items()},
    )


def _mnar_central_tendency(cp: "ColumnProfile") -> str:
    """Resolve which central tendency an MNAR column's fill value will use.

    Profile-fed (ADR-0062), so the execution layer learns the *number* from the
    training frame while the plan already says — inspectably — which statistic it
    will be: the mode for a BoundedDiscrete column, else the mean when the column
    is normally skewed and the median otherwise.
    """
    if cp.numeric_kind == NumericKind.BoundedDiscrete:
        return "mode"
    stats = _column_stats(cp)
    if stats is not None and stats.skewness_severity == SkewSeverity.Normal:
        return "mean"
    return "median"


def _bimodal_hyperparameters(
    col: str,
    cp: "ColumnProfile",
    strategy: ImputationStrategy,
    feature_correlation,
    config: NumericImputationConfig,
) -> tuple[tuple[str, Any], ...]:
    """Resolve the decide-time dials for a GMM-Sampling or Cluster-Conditional unit.

    Both bimodal strategies are driven by profile facts the execution layer has
    no way to see — the two mode centres, the column's skewness, and (for
    Cluster-Conditional's centroid branch) which features correlate with the
    target. Carrying them on the unit is what lets a fitter be handed a plan and
    a frame and nothing else (ADR-0062).
    """
    stats = _column_stats(cp)
    bimodal = stats.bimodal_stats if stats is not None else None
    if bimodal is None:
        return ()

    hyperparameters: dict[str, Any] = {
        "center1": bimodal.center1,
        "center2": bimodal.center2,
        "central_tendency": (
            "mean"
            if stats is not None and stats.skewness_severity == SkewSeverity.Normal
            else "median"
        ),
    }

    if strategy == ImputationStrategy.ClusterConditional:
        hyperparameters["feature_cols"] = _correlated_feature_cols(
            col, feature_correlation, config
        )

    return tuple(hyperparameters.items())


def _correlated_feature_cols(
    col: str,
    feature_correlation,
    config: NumericImputationConfig,
) -> tuple[str, ...]:
    """Features whose ``|r|`` with ``col`` clears ``bimodal_correlation_threshold``.

    The Cluster-Conditional centroid branch assigns an unseen row to a mode by
    its distance to each cluster's feature centroid; these are the features that
    distance is measured over. ``feature_correlation`` is the caller's
    active-column-filtered view, so an excluded column is banned as a predictor
    here like everywhere else. Empty when the profiler recorded no
    correlations — the neutral default, which degrades the branch to plain
    centre-assignment rather than triggering a data peek.
    """
    if feature_correlation is None:
        return ()
    col_corrs = feature_correlation.pearson_matrix.get(col, {})
    return tuple(
        c
        for c, r in col_corrs.items()
        if c != col and abs(r) > config.bimodal_correlation_threshold
    )


def _resolve_model_choice(
    strategy: ImputationStrategy,
    mice_model_choice: Optional[ModelChoice],
) -> Optional[ModelChoice]:
    """Resolve the estimator family for one column at decide-time.

    MICE columns inherit the block-level choice; every other strategy (KNN,
    the bimodal strategies, and every scalar/structural strategy) carries no
    estimator family.
    """
    if strategy == ImputationStrategy.MICE:
        return mice_model_choice
    return None


# ---------------------------------------------------------------------------


def _filter_feature_correlation(feature_correlation, active_cols: set[str]):
    """Restrict the pairwise Pearson view to the active numeric block.

    Returns a copy of the profiler's ``CorrelationProfileResult`` whose
    ``pearson_matrix`` keeps only entries where both columns are active, so
    every consumer downstream of :func:`decide` — routing branches, correlated
    feature lists, block hyperparameter signals — is structurally unable to
    count an excluded column as a predictor.
    """
    return dataclasses.replace(
        feature_correlation,
        pearson_matrix={
            col: {c: r for c, r in corrs.items() if c in active_cols}
            for col, corrs in feature_correlation.pearson_matrix.items()
            if col in active_cols
        },
    )


def _block_miss_fraction(profile: "StructuralProfileResult", cols: list[str]) -> float:
    """Mean per-column ``effective_null_ratio`` across a block of columns.

    Additive by construction, so this equals the block's overall cell
    missingness fraction (columns with no missingness contribute ``0.0``).
    Returns ``0.0`` for an empty block.
    """
    if not cols:
        return 0.0
    total = 0.0
    for c in cols:
        cp = profile.columns.get(c)
        if cp is not None and cp.missingness is not None:
            total += cp.missingness.effective_null_ratio
    return total / len(cols)


def _n_missing_feature_cols(profile: "StructuralProfileResult", cols: list[str]) -> int:
    """Count columns in ``cols`` whose ``effective_null_ratio`` is above zero."""
    n = 0
    for c in cols:
        cp = profile.columns.get(c)
        if (
            cp is not None
            and cp.missingness is not None
            and cp.missingness.effective_null_ratio > 0
        ):
            n += 1
    return n


def _complete_row_fraction(profile: "StructuralProfileResult") -> float:
    """Dataset-level fraction of rows with no missing across analysed columns.

    Read from the profiler's ``RowMissingnessDistribution.complete_row_fraction``
    (ADR-0062). The dataset-wide fraction underestimates any single block's
    completeness, so the convergence dials nudge slightly up — the safe
    direction. Absent (``0.0`` default) when the profiler did not record it.
    """
    rd = profile.dataset.row_distribution
    return float(rd.complete_row_fraction) if rd is not None else 0.0


def _max_pairwise_pearson(feature_correlation, cols: list[str]) -> Optional[float]:
    """Largest absolute pairwise Pearson ``|r|`` among ``cols``.

    Sourced purely from the profiler's ``feature_correlation`` (no array
    fallback). Returns ``None`` when correlations are unavailable or fewer than
    two columns are given, so the caller's correlation signal contributes
    nothing (the documented neutral default).
    """
    if feature_correlation is None or len(cols) < 2:
        return None
    values: list[float] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = feature_correlation.get_pearson(cols[i], cols[j])
            if r is not None:
                values.append(abs(r))
    return max(values) if values else None


def _compute_mice_max_iter(
    winning_tag: NonlinearityTag,
    mice_stats: list[Optional[NumericStats]],
    block_miss_fraction: float,
    max_pairwise_corr: Optional[float],
    complete_row_fraction: float,
    config: NumericImputationConfig,
) -> int:
    """Compute ``max_iter`` for the MICE ``IterativeImputer`` from five signals.

    Profile-fed (ADR-0062), aggregated block-wide: minimum R² gap across
    the block (worst-case convergence speed), maximum pairwise inter-column
    Pearson ``|r|`` (strongest coupling driver), and the block missingness
    fraction.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column.  Entries may be ``None`` when
        stats were not computed.
    block_miss_fraction : float
        Mean per-column ``effective_null_ratio`` across the MICE block (equal to
        the block's overall cell missingness fraction).
    max_pairwise_corr : float or None
        Maximum absolute pairwise Pearson ``|r|`` among the MICE columns; ``None``
        degrades the coupling signal to a no-op.
    complete_row_fraction : float
        Dataset-level fraction of fully-observed rows.
    config : NumericImputationConfig
        Imputation configuration supplying ``base_max_iter``.

    Returns
    -------
    int
        Computed ``max_iter`` value, always at least ``1``.
    """
    base = config.base_max_iter

    # Signal 1: ComplexNonlinear → more iterations required
    if winning_tag == NonlinearityTag.ComplexNonlinear:
        base += 5

    # Signal 2: block missingness fraction
    if block_miss_fraction >= 0.4:
        base += 5
    elif block_miss_fraction >= 0.2:
        base += 3
    elif block_miss_fraction >= 0.1:
        base += 2

    # Signal 3: minimum r2_gap across MICE columns; small gap means near-linear → fewer iterations
    r2_gaps = [s.r2_gap for s in mice_stats if s is not None and s.r2_gap is not None]
    if r2_gaps and min(r2_gaps) < 0.05:
        base = max(1, base - 3)

    # Signal 4: maximum pairwise Pearson |r| among MICE columns; high correlation → more iterations
    if max_pairwise_corr is not None and max_pairwise_corr >= 0.7:
        base += 3

    # Signal 5: low complete-row fraction → more iterations needed
    if complete_row_fraction < 0.2:
        base += 5
    elif complete_row_fraction < 0.5:
        base += 3

    return max(1, base)


def _compute_mice_tol(
    winning_tag: NonlinearityTag,
    mice_stats: list[Optional[NumericStats]],
) -> float:
    """Compute the convergence tolerance for the MICE ``IterativeImputer``.

    Uses the minimum IQR across all MICE columns so that tolerance is
    calibrated to the narrowest-range column in the block.  Applies tighter
    scaling when the block contains complex non-linear structure.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column.  Entries may be ``None``
        when stats were not computed.

    Returns
    -------
    float
        Convergence tolerance, always at least ``1e-7``.  Falls back to
        ``1e-3`` when no IQR is available.
    """
    iqrs = [
        s.iqr for s in mice_stats if s is not None and s.iqr is not None and s.iqr > 0
    ]
    if iqrs:
        min_iqr = min(iqrs)
        scaling_factor = (
            5e-5 if winning_tag == NonlinearityTag.ComplexNonlinear else 1e-4
        )
        return max(1e-7, min_iqr * scaling_factor)
    return 1e-3


_MICE_SKEW_TRIGGERS_MEDIAN: frozenset[SkewSeverity] = frozenset(
    {
        SkewSeverity.Moderate,
        SkewSeverity.High,
        SkewSeverity.Severe,
    }
)


def _mice_initial_strategy(mice_stats: list[Optional[NumericStats]]) -> str:
    """Determine the ``initial_strategy`` for the MICE ``IterativeImputer``.

    Returns ``"median"`` when any MICE column has ``SkewSeverity >= Moderate``;
    otherwise returns ``"mean"``.

    Parameters
    ----------
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column.  Entries may be ``None``
        when stats were not computed.

    Returns
    -------
    str
        Either ``"median"`` or ``"mean"``.
    """
    for stats in mice_stats:
        if stats is not None and stats.skewness_severity in _MICE_SKEW_TRIGGERS_MEDIAN:
            return "median"
    return "mean"


def _compute_mice_n_nearest_features(
    feature_correlation,
    mice_cols: list[str],
    numeric_cols: list[str],
    config: NumericImputationConfig,
) -> tuple[Optional[int], str]:
    """Compute ``n_nearest_features`` for the MICE ``IterativeImputer``.

    Profile-fed (ADR-0062). For blocks at or below
    ``mice_n_nearest_features_min_cols`` columns all predictors are used
    (``n_nearest_features=None``). For larger blocks, the number of informative
    predictors per column is counted against the full active-numeric breadth
    (ADR-0079) — not just the block's own membership, matching the block's true,
    widened predictor set once its fitter also reads every active numeric
    column as a candidate predictor — and the median count across the block's
    own columns is returned, capped at ``mice_max_nearest_features``.

    Correlations are read solely from ``feature_correlation`` (the
    ``CorrelationProfiler`` output). An absent correlation contributes nothing —
    the neutral default — rather than triggering an array peek.

    Parameters
    ----------
    feature_correlation : CorrelationProfileResult or None
        Pre-computed pairwise correlations from Phase 1, or ``None`` when
        unavailable.
    mice_cols : list[str]
        MICE block column names — the block's own owned columns. Gates the
        ``mice_n_nearest_features_min_cols`` threshold and is the set the
        per-column informative-predictor count is computed for.
    numeric_cols : list[str]
        Every active ``SemanticType.Numeric`` column in the plan — the
        candidate predictor pool each block column's count is drawn from,
        widened past the block's own membership.
    config : NumericImputationConfig
        Imputation configuration supplying ``mice_n_nearest_features_min_cols``,
        ``mice_max_nearest_features``, and ``mice_correlation_threshold``.

    Returns
    -------
    tuple[int or None, str]
        Computed ``n_nearest_features`` value (``None`` for small blocks) and
        a human-readable signal string recording the decision.
    """
    n_cols = len(mice_cols)
    if n_cols <= config.mice_n_nearest_features_min_cols:
        return None, (
            f"mice_n_nearest_features: all predictors used "
            f"— block ({n_cols} cols) at or below min_cols threshold "
            f"({config.mice_n_nearest_features_min_cols})"
        )

    threshold = config.mice_correlation_threshold
    counts: list[int] = []

    for col_i in mice_cols:
        count = 0
        for col_j in numeric_cols:
            if col_i == col_j:
                continue
            r: Optional[float] = (
                feature_correlation.get_pearson(col_i, col_j)
                if feature_correlation is not None
                else None
            )
            if r is not None and abs(r) > threshold:
                count += 1
        counts.append(count)

    median_count = int(np.median(counts)) if counts else 0
    n_nearest = min(max(1, median_count), config.mice_max_nearest_features)

    return n_nearest, (
        f"mice_n_nearest_features: {n_nearest} "
        f"(median informative predictors={median_count} across "
        f"{len(numeric_cols)} active numeric columns, "
        f"capped at mice_max_nearest_features={config.mice_max_nearest_features}, "
        f"threshold={threshold})"
    )


def _compute_knn_params(
    n_rows: int,
    n_features: int,
    block_miss_fraction: float,
    complete_row_fraction: float,
    config: NumericImputationConfig,
) -> tuple[int, str]:
    """Resolve the adaptive ``n_neighbors`` and ``weights`` for the KNN block.

    Profile-fed (ADR-0062). ``n_neighbors`` grows with block dimensionality and
    missingness and shrinks with completeness; ``weights`` is ``"distance"`` only
    when the block is reliable (low missingness and few features), else
    ``"uniform"``. The KNN scaling params (``col_means``/``col_stds``) are
    learned fitted-state and are resolved by execution, never here.

    Parameters
    ----------
    n_rows : int
        Number of rows the plan is decided for.
    n_features : int
        Number of columns in the KNN block.
    block_miss_fraction : float
        Mean per-column ``effective_null_ratio`` across the KNN block.
    complete_row_fraction : float
        Dataset-level fraction of fully-observed rows.
    config : NumericImputationConfig
        Imputation configuration supplying the neighbour bounds and the
        distance-weight reliability thresholds.

    Returns
    -------
    tuple[int, str]
        The resolved ``n_neighbors`` and ``weights`` (``"distance"`` /
        ``"uniform"``).
    """
    base_k = max(config.knn_min_neighbors, int(np.sqrt(n_features)))
    k_raw = (
        base_k
        * (1.0 + block_miss_fraction)
        * (1.0 / max(complete_row_fraction, 0.1)) ** 0.5
    )
    adaptive_raw = max(config.knn_min_neighbors, int(k_raw))
    n_neighbors = min(adaptive_raw, n_rows - 1, config.knn_max_neighbors)
    n_neighbors = max(1, n_neighbors)

    reliability_high = (
        block_miss_fraction < config.knn_distance_weight_max_null_ratio
        and n_features <= config.knn_distance_weight_max_features
    )
    weights = "distance" if reliability_high else "uniform"
    return n_neighbors, weights

def _resolve_domain_snap_bounds(
    cp: "ColumnProfile",
    strategy: ImputationStrategy,
) -> Optional[tuple[float, float]]:
    """Return domain-snap bounds for BoundedDiscrete model-based strategies.

    Parameters
    ----------
    cp : ColumnProfile
        Phase 1 profile; ``numeric_kind`` and ``stats.min`` / ``stats.max``
        are read to determine whether snapping applies.
    strategy : ImputationStrategy
        Strategy returned by ``_StrategyRouter.route()``.

    Returns
    -------
    tuple[float, float] or None
        ``(min, max)`` when the column is BoundedDiscrete and the strategy is
        model-based; ``None`` otherwise.
    """
    if cp.numeric_kind != NumericKind.BoundedDiscrete:
        return None
    if strategy not in (
        ImputationStrategy.KNN,
        ImputationStrategy.MICE,
        ImputationStrategy.GMMSampling,
    ):
        return None
    stats = cp.stats if isinstance(cp.stats, NumericStats) else None
    if stats is not None and stats.min is not None and stats.max is not None:
        return (stats.min, stats.max)
    return None
