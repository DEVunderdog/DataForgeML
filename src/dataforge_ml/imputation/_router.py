"""
``route(profile, config) -> ImputationRouting`` — the pure route-time layer.

Routes every profiled column to its imputation strategy for every
non-model-based path — Dropped, Constant, MNAR, Passthrough, the Bimodal
Imputation Framework's branch selection, and the terminal Mean/Median/Mode
fills — and lowers the two config declarations that complete a strategy
(``constant_fill``, ``grouping_variable``) onto each column's
:class:`~dataforge_ml.imputation.ColumnRouting`. No training data is touched:
routing is a pure function of the Phase 1 profile (ADR-0088).

Resolving the concrete MICE estimator, dials, profile-derived estimates
(bimodal centres, ``feature_cols``, ``domain_snap_bounds``) and execution
units belongs to later layers —
:func:`~dataforge_ml.imputation.resolve_recipe` and
:func:`~dataforge_ml.imputation.derive_units` — never here.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from ..config import PipelineConfig, PipelinePhase, SemanticType
from ..profiling._numeric_config import NumericStats
from ._config import (
    _EXCLUSION_SIGNAL,
    ColumnRouting,
    ImputationRouting,
    ImputationStrategy,
    ModelChoice,
    NumericImputationConfig,
)
from ._escalation import _mice_winning_tag, _resolve_nonlinearity_tag
from ._regression_estimator_factory import RegressionEstimatorFactory
from ._strategy_router import _StrategyRouter

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile, StructuralProfileResult


def _column_stats(cp: ColumnProfile) -> NumericStats | None:
    """Return the column's NumericStats, or None when unavailable."""
    return cp.stats if isinstance(cp.stats, NumericStats) else None


def _raise_on_excluded_column_overrides(
    excluded: set[str],
    *,
    mnar_columns: set[str],
    per_column_strategy: dict,
    per_column_constant_fill: dict[str, float],
) -> None:
    """Raise when an excluded column is also named in imputation config.

    The user typed "impute this column like so" and "don't impute this column"
    in the same config; neither side may silently win, so the contradiction
    surfaces as one ``ValueError`` at route-time — before any routing —
    listing every offending column across the three override surfaces.
    """
    if not excluded:
        return
    surfaces: dict[str, set[str]] = {
        "mnar_columns": mnar_columns,
        "per_column_strategy": set(per_column_strategy),
        "per_column_constant_fill": set(per_column_constant_fill),
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


def _filter_feature_correlation(feature_correlation, active_cols: set[str]):
    """Restrict the pairwise Pearson view to the active numeric block.

    Returns a copy of the profiler's ``CorrelationProfileResult`` whose
    ``pearson_matrix`` keeps only entries where both columns are active, so
    every routing branch is structurally unable to count an excluded column
    as a predictor.
    """
    return dataclasses.replace(
        feature_correlation,
        pearson_matrix={
            col: {c: r for c, r in corrs.items() if c in active_cols}
            for col, corrs in feature_correlation.pearson_matrix.items()
            if col in active_cols
        },
    )


def route(
    profile: StructuralProfileResult,
    config: PipelineConfig | None = None,
) -> ImputationRouting:
    """Route every profiled column to its imputation strategy.

    The route-time entry point of the layered imputation door (ADR-0088).
    Routes every non-model-based path — Dropped, Constant, MNAR, Passthrough,
    the Bimodal Imputation Framework's branch selection, and the terminal
    Mean/Median/Mode fills — and lowers ``constant_fill`` and
    ``grouping_variable`` from ``config`` onto each column's
    :class:`~dataforge_ml.imputation.ColumnRouting`. Every routing path that
    would otherwise duplicate a "is a model worth it here" check instead runs
    through the one model-based escalation point in
    :mod:`~dataforge_ml.imputation._escalation`: the Feasibility Floor, the
    Signal Score, and the Capability Ladder (ADR-0091, ADR-0092, ADR-0094)
    determine, per column, whether it lands on ``KNN``, ``MICE``, or a scalar
    fill. The MICE block's estimator is then resolved off the Estimator
    Ladder from the block's winning ``NonlinearityTag`` and stamped onto
    ``mice_model_choice``.

    The row count routing is performed against is read from
    ``profile.dataset.row_count`` — every other routing input already comes
    from the profile, so a mixed-shape decision is structurally impossible.

    ``route`` resolves its own active column set for the Imputation phase
    from ``config`` (Hard Exclusions plus Imputation-phase Soft Exclusions) —
    no upstream filtering is relied on. A hard-excluded column is omitted from
    the routing entirely; a soft-excluded column stays visible as a
    ``Passthrough`` routing carrying a signal that records the exclusion.

    Parameters
    ----------
    profile : StructuralProfileResult
        Full-dataset Phase 1 profile. Supplies the routing signals and the
        routable-column population.
    config : PipelineConfig, optional
        Pipeline configuration supplying the imputation thresholds, MNAR
        declarations, per-column overrides, and the exclusion lists resolved
        into the Imputation-phase active column set. Defaults to
        ``PipelineConfig()`` when omitted.

    Returns
    -------
    ImputationRouting
        The immutable routing: one :class:`~dataforge_ml.imputation.ColumnRouting`
        per active column, plus the derived ``{col}_missing`` indicator entries.

    Raises
    ------
    ValueError
        If a hard- or Imputation-soft-excluded column is also named in
        ``mnar_columns``, ``per_column_strategy``, or
        ``per_column_constant_fill`` — contradictory configuration surfaces at
        route-time, before any routing, with every offending column reported
        in one raise.
    """
    config = config or PipelineConfig()
    config.imputation.validate()

    imp_cfg = config.imputation
    numeric_cfg: NumericImputationConfig = imp_cfg.numeric
    mnar_columns = set(imp_cfg.mnar_columns)
    per_column_strategy = dict(numeric_cfg.per_column_strategy or {})
    per_column_constant_fill = dict(numeric_cfg.per_column_constant_fill or {})

    # Resolve the Imputation-phase active column set from the config it
    # already receives — the phase's own exclusion enforcement, never
    # inherited from an upstream phase. Hard-excluded columns are omitted from
    # the routing entirely; Imputation-soft-excluded columns stay visible as
    # Passthrough.
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
    )

    n_rows = profile.dataset.row_count
    numeric_cols = [
        col
        for col, cp in profile.columns.items()
        if cp.semantic_type == SemanticType.Numeric and col in active_cols
    ]
    n_features = len(numeric_cols)

    # Detect multi-MAR: >=2 non-dropped, non-MNAR columns carrying MARSuspect.
    from ..profiling._missingness_config import MissingnessFlag

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
    # correlation view to the active numeric block so no routing branch ever
    # counts an excluded column as a feature.
    feature_correlation = profile.dataset.feature_correlation
    if feature_correlation is not None and (hard_excluded or soft_excluded):
        feature_correlation = _filter_feature_correlation(
            feature_correlation, set(numeric_cols)
        )
    router = _StrategyRouter()

    routings: dict[str, ColumnRouting] = {}
    for col, cp in profile.columns.items():
        if col in hard_excluded:
            continue  # omitted from the routing entirely
        if cp.semantic_type is None:
            continue
        if col in soft_excluded:
            routings[col] = ColumnRouting(
                column=col,
                semantic_type=cp.semantic_type,
                strategy=ImputationStrategy.Passthrough,
                signals=(_EXCLUSION_SIGNAL,),
                excluded=True,
            )
            continue
        if cp.semantic_type != SemanticType.Numeric:
            routings[col] = ColumnRouting(
                column=col,
                semantic_type=cp.semantic_type,
                strategy=ImputationStrategy.Passthrough,
            )
            continue

        strategy, signals = router.route(
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
        routings[col] = ColumnRouting(
            column=col,
            semantic_type=SemanticType.Numeric,
            strategy=strategy,
            signals=tuple(signals),
            grouping_variable=numeric_cfg.bimodal_grouping_variables.get(col),
            constant_fill=per_column_constant_fill.get(col),
            indicator_flag=strategy == ImputationStrategy.MNAR,
            mnar=strategy == ImputationStrategy.MNAR,
            drop=strategy == ImputationStrategy.Dropped,
        )

    # Indicator pass: pre-register the {col}_missing columns the MNAR
    # mechanism appends at transform time, mirroring the fitted record set.
    for col, routing in list(routings.items()):
        if not routing.indicator_flag:
            continue
        indicator_col = f"{col}_missing"
        routings[indicator_col] = ColumnRouting(
            column=indicator_col,
            semantic_type=SemanticType.Boolean,
            strategy=ImputationStrategy.Indicator,
        )

    # The MICE block trains one estimator, picked from the Estimator Ladder
    # off the block's winning tag — the most complex NonlinearityTag across
    # every MICE column, forced ones included (ADR-0094).
    mice_cols = [
        col
        for col, routing in routings.items()
        if routing.strategy == ImputationStrategy.MICE
    ]
    mice_model_choice: ModelChoice | None = None
    if mice_cols:
        winning_tag = _mice_winning_tag(
            [_resolve_nonlinearity_tag(_column_stats(profile.columns[c])) for c in mice_cols]
        )
        mice_model_choice = RegressionEstimatorFactory.resolve_choice(winning_tag)

    return ImputationRouting(
        column_routings=routings, mice_model_choice=mice_model_choice
    )
