"""``resolve_recipe(routing, profile, config) -> ImputationRecipe`` (ADR-0089).

The recipe is everything a unit needs to train except what training learns:
every dial, both hyperparameter maps, the profile-derived
:class:`ColumnEstimates` (bimodal centres, ``feature_cols``,
``domain_snap_bounds``), and the sentinel maps — resolved **once, off the
profile it is given**. The joint MICE and KNN blocks' dials are resolved here
too, profile-fed (ADR-0062): ``max_iter`` / ``tol`` / ``initial_strategy`` /
``n_nearest_features`` for MICE, ``n_neighbors`` / ``weights`` for KNN — both
widened past each block's own membership to every active numeric column
(ADR-0079, ADR-0093), since that is the predictor space each fitter actually
reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import PipelineConfig, SemanticType
from ..profiling._config import NumericKind
from ..profiling._numeric_config import NonlinearityTag, NumericStats, SkewSeverity
from ._config import (
    _STRATEGY_DIALS,
    ImputationRouting,
    ImputationStrategy,
    NumericImputationConfig,
    _dial_defaults,
    _md_cell,
)
from ._escalation import _mice_winning_tag, _resolve_nonlinearity_tag

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile, StructuralProfileResult

__all__ = ["ColumnEstimates", "ImputationRecipe", "resolve_recipe"]

_BIMODAL_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {ImputationStrategy.GMMSampling, ImputationStrategy.ClusterConditional}
)

# Strategies whose predictions get rounded and clipped into a BoundedDiscrete
# column's observed domain.
_DOMAIN_SNAP_STRATEGIES: frozenset[ImputationStrategy] = frozenset(
    {
        ImputationStrategy.KNN,
        ImputationStrategy.MICE,
        ImputationStrategy.GMMSampling,
    }
)

_ESTIMATE_FIELDS: frozenset[str] = frozenset(
    {"center1", "center2", "feature_cols", "domain_snap_bounds"}
)


@dataclass(frozen=True)
class ColumnEstimates:
    """Profile-derived facts one column's fitter reads, resolved off a profile.

    Populated on :class:`ImputationRecipe` only for a column whose strategy
    actually uses them; every field is ``None`` for a column that has no
    estimates (ADR-0089). Never raised over — a missing estimate is a
    :class:`~dataforge_ml.imputation.UnitNotTrainableError` at
    :func:`~dataforge_ml.imputation.fit_unit` time, naming
    :meth:`ImputationRecipe.with_estimates` as the fix.

    Parameters
    ----------
    center1 : float, optional
        First of the two bimodal mode centres the GMM-Sampling and
        Cluster-Conditional strategies split on.
    center2 : float, optional
        Second bimodal mode centre. See ``center1``.
    feature_cols : tuple[str, ...], optional
        Columns the Cluster-Conditional centroid branch measures its
        per-cluster centroids over.
    domain_snap_bounds : tuple[float, float], optional
        ``(min, max)`` bounds a model-based strategy's predictions are rounded
        and clipped into, for a BoundedDiscrete column.
    """

    center1: float | None = None
    center2: float | None = None
    feature_cols: tuple[str, ...] | None = None
    domain_snap_bounds: tuple[float, float] | None = None

    def to_markdown(self) -> str:
        """Render the estimates as a ``####``-rooted Markdown fragment.

        Returns
        -------
        str
            Markdown subsection with a field table.
        """
        lines = [
            "| Field | Value |",
            "|---|---|",
            f"| center1 | {_md_cell(self.center1)} |",
            f"| center2 | {_md_cell(self.center2)} |",
            f"| feature_cols | {_md_cell(self.feature_cols)} |",
            f"| domain_snap_bounds | {_md_cell(self.domain_snap_bounds)} |",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the column estimates to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            The four estimate fields, with tuples converted to lists.
        """
        return {
            "center1": self.center1,
            "center2": self.center2,
            "feature_cols": (
                list(self.feature_cols) if self.feature_cols is not None else None
            ),
            "domain_snap_bounds": (
                list(self.domain_snap_bounds)
                if self.domain_snap_bounds is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnEstimates:
        """Reconstruct a ``ColumnEstimates`` instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Mapping produced by :meth:`to_dict`. Missing keys default to
            ``None``.

        Returns
        -------
        ColumnEstimates
            The rebuilt estimates, with list fields converted back to tuples.
        """
        fc = data.get("feature_cols")
        dsb = data.get("domain_snap_bounds")
        return cls(
            center1=data.get("center1"),
            center2=data.get("center2"),
            feature_cols=tuple(fc) if fc is not None else None,
            domain_snap_bounds=tuple(dsb) if dsb is not None else None,
        )


def _column_stats(cp: ColumnProfile) -> NumericStats | None:
    """Return the column's NumericStats, or None when unavailable."""
    return cp.stats if isinstance(cp.stats, NumericStats) else None


def _bimodal_centers(
    cp: ColumnProfile, strategy: ImputationStrategy
) -> tuple[float | None, float | None]:
    """The two mode centres a bimodal column's fitter splits on, else ``(None, None)``."""
    if strategy not in _BIMODAL_STRATEGIES:
        return None, None
    stats = _column_stats(cp)
    bimodal = stats.bimodal_stats if stats is not None else None
    if bimodal is None:
        return None, None
    return bimodal.center1, bimodal.center2


def _correlated_feature_cols(
    col: str,
    feature_correlation,
    config: NumericImputationConfig,
) -> tuple[str, ...]:
    """Features whose ``|r|`` with ``col`` clears ``bimodal_correlation_threshold``."""
    if feature_correlation is None:
        return ()
    col_corrs = feature_correlation.pearson_matrix.get(col, {})
    return tuple(
        c
        for c, r in col_corrs.items()
        if c != col and abs(r) > config.bimodal_correlation_threshold
    )


def _resolve_domain_snap_bounds(
    cp: ColumnProfile,
    strategy: ImputationStrategy,
) -> tuple[float, float] | None:
    """Return domain-snap bounds for a BoundedDiscrete model-based strategy."""
    if cp.numeric_kind != NumericKind.BoundedDiscrete:
        return None
    if strategy not in _DOMAIN_SNAP_STRATEGIES:
        return None
    stats = _column_stats(cp)
    if stats is not None and stats.min is not None and stats.max is not None:
        return (stats.min, stats.max)
    return None


def _mnar_central_tendency(cp: ColumnProfile) -> str:
    """Resolve which central tendency an MNAR column's fill value will use."""
    if cp.numeric_kind == NumericKind.BoundedDiscrete:
        return "mode"
    stats = _column_stats(cp)
    if stats is not None and stats.skewness_severity == SkewSeverity.Normal:
        return "mean"
    return "median"


def _cluster_conditional_central_tendency(cp: ColumnProfile) -> str:
    """Resolve the Cluster-Conditional central tendency from the column's skew."""
    stats = _column_stats(cp)
    if stats is not None and stats.skewness_severity == SkewSeverity.Normal:
        return "mean"
    return "median"


def _active_numeric_columns(routing: ImputationRouting) -> list[str]:
    """Active numeric columns — the predictor pool MICE and KNN both widen into.

    Every column with ``SemanticType.Numeric`` on the routing except a
    soft-excluded one (``ColumnRouting.excluded``); a hard-excluded column is
    never present on the routing at all. Mirrors the active set
    :func:`~dataforge_ml.imputation.route` computed its own ``n_features``
    from, so the recipe's widened predictor count agrees with the Feasibility
    Floor's.
    """
    return [
        col
        for col, r in routing.column_routings.items()
        if r.semantic_type == SemanticType.Numeric and not r.excluded
    ]


def _block_miss_fraction(profile: StructuralProfileResult, cols: list[str]) -> float:
    """Mean per-column ``effective_null_ratio`` across a set of columns.

    Additive by construction, so this equals the set's overall cell
    missingness fraction (columns with no missingness contribute ``0.0``).
    Returns ``0.0`` for an empty set.
    """
    if not cols:
        return 0.0
    total = 0.0
    for c in cols:
        cp = profile.columns.get(c)
        if cp is not None and cp.missingness is not None:
            total += cp.missingness.effective_null_ratio
    return total / len(cols)


def _complete_row_fraction(profile: StructuralProfileResult) -> float:
    """Dataset-level fraction of rows with no missing across analysed columns.

    Read from the profiler's ``RowMissingnessDistribution.complete_row_fraction``
    (ADR-0062). The dataset-wide fraction underestimates any single block's
    completeness, so the convergence dials nudge slightly up — the safe
    direction. Absent (``0.0`` default) when the profiler did not record it.
    """
    rd = profile.dataset.row_distribution
    return float(rd.complete_row_fraction) if rd is not None else 0.0


def _max_pairwise_pearson(feature_correlation, cols: list[str]) -> float | None:
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


_MICE_SKEW_TRIGGERS_MEDIAN: frozenset[SkewSeverity] = frozenset(
    {SkewSeverity.Moderate, SkewSeverity.High, SkewSeverity.Severe}
)


def _compute_mice_max_iter(
    winning_tag: NonlinearityTag,
    mice_stats: list[NumericStats | None],
    block_miss_fraction: float,
    max_pairwise_corr: float | None,
    complete_row_fraction: float,
    config: NumericImputationConfig,
) -> int:
    """Compute ``max_iter`` for the MICE ``IterativeImputer`` from five signals.

    Profile-fed (ADR-0062), aggregated block-wide: minimum R² gap across the
    block (worst-case convergence speed), maximum pairwise inter-column
    Pearson ``|r|`` (strongest coupling driver), and the block missingness
    fraction.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column. Entries may be ``None`` when
        stats were not computed.
    block_miss_fraction : float
        Mean per-column ``effective_null_ratio`` across the MICE block.
    max_pairwise_corr : float or None
        Maximum absolute pairwise Pearson ``|r|`` among the MICE columns;
        ``None`` degrades the coupling signal to a no-op.
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

    if winning_tag == NonlinearityTag.ComplexNonlinear:
        base += 5

    if block_miss_fraction >= 0.4:
        base += 5
    elif block_miss_fraction >= 0.2:
        base += 3
    elif block_miss_fraction >= 0.1:
        base += 2

    r2_gaps = [s.r2_gap for s in mice_stats if s is not None and s.r2_gap is not None]
    if r2_gaps and min(r2_gaps) < 0.05:
        base = max(1, base - 3)

    if max_pairwise_corr is not None and max_pairwise_corr >= 0.7:
        base += 3

    if complete_row_fraction < 0.2:
        base += 5
    elif complete_row_fraction < 0.5:
        base += 3

    return max(1, base)


def _compute_mice_tol(
    winning_tag: NonlinearityTag,
    mice_stats: list[NumericStats | None],
) -> float:
    """Compute the convergence tolerance for the MICE ``IterativeImputer``.

    Uses the minimum IQR across all MICE columns so that tolerance is
    calibrated to the narrowest-range column in the block. Applies tighter
    scaling when the block contains complex non-linear structure.

    Parameters
    ----------
    winning_tag : NonlinearityTag
        Most-complex nonlinearity tag across all MICE columns.
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column. Entries may be ``None`` when
        stats were not computed.

    Returns
    -------
    float
        Convergence tolerance, always at least ``1e-7``. Falls back to
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


def _mice_initial_strategy(mice_stats: list[NumericStats | None]) -> str:
    """Determine the ``initial_strategy`` for the MICE ``IterativeImputer``.

    Returns ``"median"`` when any MICE column has ``SkewSeverity >= Moderate``;
    otherwise returns ``"mean"``.

    Parameters
    ----------
    mice_stats : list[NumericStats or None]
        Phase 1 statistics for each MICE column. Entries may be ``None`` when
        stats were not computed.

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
    active_numeric_cols: list[str],
    config: NumericImputationConfig,
) -> tuple[int | None, str]:
    """Compute ``n_nearest_features`` for the MICE ``IterativeImputer``.

    Profile-fed (ADR-0062). Gated on the **full active-numeric breadth**, not
    the MICE block's own size — routing's Feasibility Floor already decides
    block membership, so the recipe reads the same width it does (ADR-0091).
    At or below ``mice_n_nearest_features_min_cols`` active numeric columns,
    every predictor is used (``n_nearest_features=None``). Above it, the
    number of informative predictors per column is counted against the full
    active-numeric breadth (ADR-0079) — the block's true, widened predictor
    set — and the median count across the block's own columns is returned,
    capped at ``mice_max_nearest_features``.

    Correlations are read solely from ``feature_correlation`` (the
    ``CorrelationProfiler`` output). An absent correlation contributes
    nothing — the neutral default — rather than triggering an array peek.

    Parameters
    ----------
    feature_correlation : CorrelationProfileResult or None
        Pre-computed pairwise correlations from Phase 1, or ``None`` when
        unavailable.
    mice_cols : list[str]
        MICE block column names — the block's own owned columns. The set the
        per-column informative-predictor count is computed for.
    active_numeric_cols : list[str]
        Every active ``SemanticType.Numeric`` column — the candidate
        predictor pool each block column's count is drawn from, and the gate
        this function's threshold reads.
    config : NumericImputationConfig
        Imputation configuration supplying ``mice_n_nearest_features_min_cols``,
        ``mice_max_nearest_features``, and ``mice_correlation_threshold``.

    Returns
    -------
    tuple[int or None, str]
        Computed ``n_nearest_features`` value (``None`` for a narrow active
        set) and a human-readable signal string recording the decision.
    """
    n_active = len(active_numeric_cols)
    if n_active <= config.mice_n_nearest_features_min_cols:
        return None, (
            f"mice_n_nearest_features: all predictors used "
            f"— active numeric width ({n_active} cols) at or below "
            f"mice_n_nearest_features_min_cols "
            f"({config.mice_n_nearest_features_min_cols})"
        )

    threshold = config.mice_correlation_threshold
    counts: list[int] = []

    for col_i in mice_cols:
        count = 0
        for col_j in active_numeric_cols:
            if col_i == col_j:
                continue
            r: float | None = (
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
        f"{n_active} active numeric columns, "
        f"capped at mice_max_nearest_features={config.mice_max_nearest_features}, "
        f"threshold={threshold})"
    )


def _compute_knn_params(
    n_features: int,
    block_miss_fraction: float,
    complete_row_fraction: float,
    config: NumericImputationConfig,
) -> tuple[int, str]:
    """Resolve the adaptive ``n_neighbors`` and ``weights`` for the KNN block.

    Profile-fed (ADR-0062), read over the **full active-numeric breadth**
    (ADR-0093): ``n_features`` is the active numeric width and
    ``block_miss_fraction`` the mean effective-null ratio across it, neither
    scoped to the KNN block's own membership, since that is the distance
    space the widened fitter actually reads. ``n_neighbors`` grows with
    dimensionality and missingness and shrinks with completeness;
    ``weights`` is ``"distance"`` only when the distance space is reliable
    (low missingness and narrow width), else ``"uniform"``. There is no
    ``n_rows - 1`` cap: ``KNNImputer`` already caps each column at its own
    donor count, so the recipe reads no row count at all (ADR-0094). The KNN
    scaling params (``col_means``/``col_stds``) are learned fitted-state and
    are resolved by execution, never here.

    Parameters
    ----------
    n_features : int
        Active numeric width — the KNN block's widened distance-space
        dimensionality.
    block_miss_fraction : float
        Mean per-column ``effective_null_ratio`` across the active numeric set.
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
    base_k = max(config.knn_min_neighbors, int(np.sqrt(max(n_features, 1))))
    k_raw = (
        base_k
        * (1.0 + block_miss_fraction)
        * (1.0 / max(complete_row_fraction, 0.1)) ** 0.5
    )
    adaptive_raw = max(config.knn_min_neighbors, int(k_raw))
    n_neighbors = min(adaptive_raw, config.knn_max_neighbors)
    n_neighbors = max(1, n_neighbors)

    reliability_high = (
        block_miss_fraction < config.knn_distance_weight_max_null_ratio
        and n_features <= config.knn_distance_weight_max_features
    )
    weights = "distance" if reliability_high else "uniform"
    return n_neighbors, weights


@dataclass(frozen=True)
class ImputationRecipe:
    """Everything a unit needs to train except what training learns (ADR-0089).

    Resolved once by :func:`resolve_recipe`, off the profile it is given:
    every dial, both hyperparameter maps, the profile-derived
    :class:`ColumnEstimates`, and the sentinel maps. The recipe holds its
    routing by value, so a recipe can never be fitted against a routing it was
    not resolved from.

    Parameters
    ----------
    routing : ImputationRouting
        The routing this recipe was resolved from. Held by value.
    decided_hyperparameters : dict[str, tuple[tuple[str, Any], ...]]
        The resolve-time hyperparameter base per unit id, complete for each
        unit's strategy.
    hyperparameter_overrides : dict[str, tuple[tuple[str, Any], ...]]
        The sparse per-unit override delta, written only by
        :meth:`with_hyperparameters`.
    column_estimates : dict[str, ColumnEstimates]
        Per-column profile-derived estimates, present for every column whose
        strategy uses them.
    numeric_sentinels : dict[str, list[float]]
        Declared numeric sentinel values per column, carried from the profile.
    string_sentinels : dict[str, list[str]]
        Declared string sentinel values per column, carried from the profile.
    """

    routing: ImputationRouting
    decided_hyperparameters: dict[str, tuple[tuple[str, Any], ...]] = field(
        default_factory=dict
    )
    hyperparameter_overrides: dict[str, tuple[tuple[str, Any], ...]] = field(
        default_factory=dict
    )
    column_estimates: dict[str, ColumnEstimates] = field(default_factory=dict)
    numeric_sentinels: dict[str, list[float]] = field(default_factory=dict)
    string_sentinels: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decided_hyperparameters", dict(self.decided_hyperparameters)
        )
        object.__setattr__(
            self, "hyperparameter_overrides", dict(self.hyperparameter_overrides)
        )
        object.__setattr__(self, "column_estimates", dict(self.column_estimates))
        object.__setattr__(
            self,
            "numeric_sentinels",
            {k: list(v) for k, v in self.numeric_sentinels.items()},
        )
        object.__setattr__(
            self,
            "string_sentinels",
            {k: list(v) for k, v in self.string_sentinels.items()},
        )

    def hyperparameters(self, unit_id: str) -> dict[str, Any]:
        """Return the merged, per-key ``decided ⊕ override`` dial row for a unit.

        Parameters
        ----------
        unit_id : str
            The unit id to read dials for.

        Returns
        -------
        dict[str, Any]
            The merged hyperparameters. Empty for a unit with no dials.
        """
        base = dict(self.decided_hyperparameters.get(unit_id, ()))
        base.update(dict(self.hyperparameter_overrides.get(unit_id, ())))
        return base

    def with_hyperparameters(
        self, unit_id: str, hyperparameters: dict[str, Any] | None
    ) -> ImputationRecipe:
        """Return a new recipe overriding named hyperparameters on one unit.

        A per-key merge onto the unit's resolved base: the named keys are
        overridden and every other decided dial is kept. Successive edits to
        the same unit accumulate per key. Passing ``None`` resets the unit to
        its decided values by clearing its delta.

        Parameters
        ----------
        unit_id : str
            ID of the unit to override. Must already be part of the routing.
        hyperparameters : dict[str, Any] or None
            The keys to override and their replacement values, or ``None`` to
            reset the unit to its decided base.

        Returns
        -------
        ImputationRecipe
            A new recipe with the override applied.

        Raises
        ------
        KeyError
            If ``unit_id`` is not part of the routing's derived units.
        ValueError
            If ``hyperparameters`` names a key that is not one of the unit's
            strategy's dials.
        """
        from ._units import derive_units

        unit = next(
            (u for u in derive_units(self.routing) if u.unit_id == unit_id), None
        )
        if unit is None:
            raise KeyError(f"Unit '{unit_id}' is not part of this recipe's routing.")

        new_overrides = dict(self.hyperparameter_overrides)
        if hyperparameters is None:
            new_overrides.pop(unit_id, None)
        else:
            dial_keys = set(_dial_defaults(unit.strategy))
            for key in hyperparameters:
                if key not in dial_keys:
                    raise ValueError(
                        f"Unit '{unit_id}' ({unit.strategy}) has no dial '{key}' "
                        f"to override."
                    )
            merged_delta = dict(new_overrides.get(unit_id, ()))
            merged_delta.update(hyperparameters)
            new_overrides[unit_id] = tuple(merged_delta.items())

        return ImputationRecipe(
            routing=self.routing,
            decided_hyperparameters=self.decided_hyperparameters,
            hyperparameter_overrides=new_overrides,
            column_estimates=self.column_estimates,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def with_estimates(self, column: str, **fields: Any) -> ImputationRecipe:
        """Return a new recipe with named estimate fields overwritten for one column.

        Parameters
        ----------
        column : str
            Column to edit. Must already carry a :class:`ColumnEstimates` entry.
        **fields : Any
            Named fields of :class:`ColumnEstimates` to overwrite
            (``center1``, ``center2``, ``feature_cols``,
            ``domain_snap_bounds``).

        Returns
        -------
        ImputationRecipe
            A new recipe with the edit applied.

        Raises
        ------
        ValueError
            If ``column`` carries no estimates entry, or ``fields`` names a
            key that is not one of :class:`ColumnEstimates`'s fields.
        """
        if column not in self.column_estimates:
            raise ValueError(
                f"Column '{column}' carries no estimates entry on this recipe. "
                f"Only a column whose strategy uses estimates (the bimodal "
                f"strategies, or a BoundedDiscrete model-based one) has one."
            )
        unknown = set(fields) - _ESTIMATE_FIELDS
        if unknown:
            raise ValueError(
                f"Column '{column}': unknown estimate field(s) "
                f"{sorted(unknown)}. Known fields: {sorted(_ESTIMATE_FIELDS)}."
            )
        from dataclasses import replace

        new_estimates = dict(self.column_estimates)
        new_estimates[column] = replace(new_estimates[column], **fields)
        return ImputationRecipe(
            routing=self.routing,
            decided_hyperparameters=self.decided_hyperparameters,
            hyperparameter_overrides=self.hyperparameter_overrides,
            column_estimates=new_estimates,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

    def to_markdown(self) -> str:
        """Render the whole recipe as a Markdown document.

        Returns
        -------
        str
            Markdown document with a summary, the hyperparameter table, and
            the per-column estimates.
        """
        lines = ["# Imputation Recipe\n"]

        lines.append("## Summary\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| columns | {len(self.routing.column_routings)} |")
        lines.append(f"| column_estimates | {len(self.column_estimates)} |")
        lines.append("")

        lines.append("## Hyperparameters\n")
        lines.append("| Unit | Decided base | Override delta |")
        lines.append("|---|---|---|")
        unit_ids = list(
            dict.fromkeys(
                list(self.decided_hyperparameters) + list(self.hyperparameter_overrides)
            )
        )
        if unit_ids:
            for unit_id in unit_ids:
                decided = dict(self.decided_hyperparameters.get(unit_id, ()))
                override = dict(self.hyperparameter_overrides.get(unit_id, ()))
                lines.append(
                    f"| `{unit_id}` | {_md_cell(decided)} | {_md_cell(override)} |"
                )
        else:
            lines.append("| none | | |")
        lines.append("")

        lines.append("## Column Estimates\n")
        if self.column_estimates:
            for col, estimates in self.column_estimates.items():
                lines.append(f"### `{col}`\n")
                lines.append(estimates.to_markdown())
                lines.append("")
        else:
            lines.append("none")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Imputation Recipe document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()

    def to_dict(self) -> dict:
        """Serialise the recipe to a plain, JSON-friendly dictionary.

        Saves its routing inline (which omits any live estimator), both
        hyperparameter maps (decided base and override delta), the per-column
        estimates, and the declared sentinel maps.

        Returns
        -------
        dict
            The recipe's fields with nested objects serialised to dicts.
        """
        return {
            "routing": self.routing.to_dict(),
            "decided_hyperparameters": {
                unit_id: dict(dials)
                for unit_id, dials in self.decided_hyperparameters.items()
            },
            "hyperparameter_overrides": {
                unit_id: dict(dials)
                for unit_id, dials in self.hyperparameter_overrides.items()
            },
            "column_estimates": {
                col: est.to_dict()
                for col, est in self.column_estimates.items()
            },
            "numeric_sentinels": {
                col: list(v) for col, v in self.numeric_sentinels.items()
            },
            "string_sentinels": {
                col: list(v) for col, v in self.string_sentinels.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImputationRecipe:
        """Reconstruct an ``ImputationRecipe`` from a plain dictionary.

        Loading is strict: gap-fill is deleted (ADR-0096). The decided base's
        unit ids must match the units derived from the inline routing, and every
        unit's dial row must be complete and contain only valid dials for that
        unit's strategy. A missing or unknown dial row or key, or a unit-id
        mismatch against ``derive_units(routing)``, raises ``ValueError``
        naming what is wrong and advising to resolve again.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        ImputationRecipe
            Reconstructed recipe instance.

        Raises
        ------
        ValueError
            If ``data`` lacks a valid routing, or if the decided-base unit ids
            mismatch ``derive_units(routing)``, or if any dial row has missing
            or unknown dial keys.
        """
        from ._units import derive_units

        if "routing" not in data:
            raise ValueError(
                "Recipe payload is missing 'routing'. Resolve the recipe again."
            )

        routing = ImputationRouting.from_dict(data["routing"])
        derived_units = derive_units(routing)
        unit_strategy_map = {
            u.unit_id: u.strategy
            for u in derived_units
            if u.strategy in _STRATEGY_DIALS
        }

        raw_decided = data.get("decided_hyperparameters")
        if raw_decided is None:
            raise ValueError(
                "Recipe payload is missing 'decided_hyperparameters'. "
                "Resolve the recipe again."
            )

        base_unit_ids = set(raw_decided.keys())
        expected_unit_ids = set(unit_strategy_map.keys())

        if base_unit_ids != expected_unit_ids:
            missing = expected_unit_ids - base_unit_ids
            extra = base_unit_ids - expected_unit_ids
            details = []
            if missing:
                details.append(f"missing unit(s) {sorted(missing)}")
            if extra:
                details.append(f"unexpected unit(s) {sorted(extra)}")
            raise ValueError(
                f"Decided base unit ids mismatch freshly derived units ({'; '.join(details)}). "
                f"Expected unit ids: {sorted(expected_unit_ids)}, got: {sorted(base_unit_ids)}. "
                f"Resolve the recipe again."
            )

        decided_hyperparameters: dict[str, tuple[tuple[str, Any], ...]] = {}
        for unit_id, dials in raw_decided.items():
            strategy = unit_strategy_map[unit_id]
            expected_dials = _dial_defaults(strategy)
            expected_keys = set(expected_dials.keys())
            actual_keys = set(dials.keys())

            missing_keys = expected_keys - actual_keys
            if missing_keys:
                raise ValueError(
                    f"Unit '{unit_id}' ({strategy}) is missing required dial key(s): "
                    f"{sorted(missing_keys)}. Resolve the recipe again."
                )

            unknown_keys = actual_keys - expected_keys
            if unknown_keys:
                raise ValueError(
                    f"Unit '{unit_id}' ({strategy}) has unknown dial key(s): "
                    f"{sorted(unknown_keys)}. Resolve the recipe again."
                )

            decided_hyperparameters[unit_id] = tuple(dials.items())

        raw_overrides = data.get("hyperparameter_overrides", {})
        hyperparameter_overrides: dict[str, tuple[tuple[str, Any], ...]] = {}
        for unit_id, dials in raw_overrides.items():
            if unit_id not in unit_strategy_map:
                raise ValueError(
                    f"Hyperparameter override delta contains unknown unit '{unit_id}'. "
                    f"Resolve the recipe again."
                )
            strategy = unit_strategy_map[unit_id]
            expected_dials = _dial_defaults(strategy)
            expected_keys = set(expected_dials.keys())
            actual_keys = set(dials.keys())
            unknown_keys = actual_keys - expected_keys
            if unknown_keys:
                raise ValueError(
                    f"Unit '{unit_id}' ({strategy}) override delta has unknown dial key(s): "
                    f"{sorted(unknown_keys)}. Resolve the recipe again."
                )
            hyperparameter_overrides[unit_id] = tuple(dials.items())

        column_estimates = {
            col: ColumnEstimates.from_dict(raw)
            for col, raw in data.get("column_estimates", {}).items()
        }

        numeric_sentinels = {
            col: list(v) for col, v in data.get("numeric_sentinels", {}).items()
        }
        string_sentinels = {
            col: list(v) for col, v in data.get("string_sentinels", {}).items()
        }

        return cls(
            routing=routing,
            decided_hyperparameters=decided_hyperparameters,
            hyperparameter_overrides=hyperparameter_overrides,
            column_estimates=column_estimates,
            numeric_sentinels=numeric_sentinels,
            string_sentinels=string_sentinels,
        )


def resolve_recipe(
    routing: ImputationRouting,
    profile: StructuralProfileResult,
    config: PipelineConfig | None = None,
) -> ImputationRecipe:
    """Resolve every dial, hyperparameter map, estimate, and sentinel — once.

    The Recipe half of the layered imputation door (ADR-0089): everything a
    unit needs to train except what training learns, resolved off ``profile``
    in one call. Never raises over a missing estimate — every column whose
    strategy uses one gets a :class:`ColumnEstimates` entry, with ``None``
    fields when the profile has nothing, so :meth:`ImputationRecipe.with_estimates`
    always has a target to edit.

    The joint MICE and KNN blocks' dials are resolved here too, whenever the
    routing carries one: ``decided_hyperparameters["mice"]`` /
    ``["knn"]`` are present exactly when the routing has a column of that
    strategy, both widened past their own block membership to the full
    active-numeric predictor pool (ADR-0079, ADR-0093).

    Parameters
    ----------
    routing : ImputationRouting
        The routing to resolve a recipe for.
    profile : StructuralProfileResult
        The profile to resolve estimates and sentinels off of. Checked for
        columns only: a routed, non-synthetic column absent from it raises.
    config : PipelineConfig, optional
        Pipeline configuration supplying the imputation thresholds. Defaults
        to ``PipelineConfig()`` when omitted.

    Returns
    -------
    ImputationRecipe
        The resolved recipe.

    Raises
    ------
    ValueError
        If a routed column (other than a synthetic ``{col}_missing`` indicator
        entry) is absent from ``profile``, naming every offending column.
    """
    config = config or PipelineConfig()
    numeric_cfg = config.imputation.numeric

    missing = [
        col
        for col, col_routing in routing.column_routings.items()
        if col_routing.strategy != ImputationStrategy.Indicator
        and col not in profile.columns
    ]
    if missing:
        names = ", ".join(f"'{c}'" for c in sorted(missing))
        raise ValueError(
            f"Column(s) {names} are part of this routing but absent from the "
            f"profile handed to resolve_recipe(). Resolve against a profile "
            f"that covers every routed column."
        )

    feature_correlation = profile.dataset.feature_correlation

    decided_hyperparameters: dict[str, tuple[tuple[str, Any], ...]] = {}
    column_estimates: dict[str, ColumnEstimates] = {}

    for col, col_routing in routing.column_routings.items():
        strategy = col_routing.strategy
        if strategy == ImputationStrategy.Indicator:
            continue
        cp = profile.columns[col]

        if strategy == ImputationStrategy.MNAR:
            decided_hyperparameters[f"{strategy}:{col}"] = tuple(
                {"central_tendency": _mnar_central_tendency(cp)}.items()
            )
        elif strategy == ImputationStrategy.ClusterConditional:
            decided_hyperparameters[f"{strategy}:{col}"] = tuple(
                {
                    "central_tendency": _cluster_conditional_central_tendency(cp)
                }.items()
            )

        if strategy in _BIMODAL_STRATEGIES:
            center1, center2 = _bimodal_centers(cp, strategy)
            feature_cols = (
                _correlated_feature_cols(col, feature_correlation, numeric_cfg)
                if strategy == ImputationStrategy.ClusterConditional
                else None
            )
            domain_snap_bounds = _resolve_domain_snap_bounds(cp, strategy)
            column_estimates[col] = ColumnEstimates(
                center1=center1,
                center2=center2,
                feature_cols=feature_cols,
                domain_snap_bounds=domain_snap_bounds,
            )
        elif strategy in _DOMAIN_SNAP_STRATEGIES:
            # KNN / MICE on a BoundedDiscrete column: no bimodal centres, but
            # still needs its predictions snapped into the observed domain.
            domain_snap_bounds = _resolve_domain_snap_bounds(cp, strategy)
            if domain_snap_bounds is not None:
                column_estimates[col] = ColumnEstimates(
                    domain_snap_bounds=domain_snap_bounds
                )

    active_numeric_cols = _active_numeric_columns(routing)

    mice_cols = [
        col
        for col, r in routing.column_routings.items()
        if r.strategy == ImputationStrategy.MICE
    ]
    if mice_cols:
        mice_stats = [_column_stats(profile.columns[c]) for c in mice_cols]
        winning_tag = _mice_winning_tag(
            [_resolve_nonlinearity_tag(s) for s in mice_stats]
        )
        mice_max_iter = _compute_mice_max_iter(
            winning_tag,
            mice_stats,
            _block_miss_fraction(profile, mice_cols),
            _max_pairwise_pearson(feature_correlation, mice_cols),
            _complete_row_fraction(profile),
            numeric_cfg,
        )
        mice_tol = _compute_mice_tol(winning_tag, mice_stats)
        mice_initial_strategy = _mice_initial_strategy(mice_stats)
        mice_n_nearest, _ = _compute_mice_n_nearest_features(
            feature_correlation, mice_cols, active_numeric_cols, numeric_cfg
        )
        decided_hyperparameters["mice"] = tuple(
            {
                "max_iter": mice_max_iter,
                "tol": mice_tol,
                "initial_strategy": mice_initial_strategy,
                "n_nearest_features": mice_n_nearest,
            }.items()
        )

    knn_cols = [
        col
        for col, r in routing.column_routings.items()
        if r.strategy == ImputationStrategy.KNN
    ]
    if knn_cols:
        knn_n_neighbors, knn_weights = _compute_knn_params(
            len(active_numeric_cols),
            _block_miss_fraction(profile, active_numeric_cols),
            _complete_row_fraction(profile),
            numeric_cfg,
        )
        decided_hyperparameters["knn"] = tuple(
            {"n_neighbors": knn_n_neighbors, "weights": knn_weights}.items()
        )

    return ImputationRecipe(
        routing=routing,
        decided_hyperparameters=decided_hyperparameters,
        column_estimates=column_estimates,
        numeric_sentinels={k: list(v) for k, v in profile.numeric_sentinels.items()},
        string_sentinels={k: list(v) for k, v in profile.string_sentinels.items()},
    )
