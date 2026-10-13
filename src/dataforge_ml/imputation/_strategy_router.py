"""
_StrategyRouter — pure strategy routing for numeric column imputation.

Accepts profiling signals and returns a routing decision: the strategy, the
Routing Reason that fixed it, and the typed Routing Evidence walk
(ADR-0105).  No DataFrame
access; all inputs are column profiles, configuration, and scalar context
values derived from the training set by the caller.

Every routing path that would otherwise run its own "is a model worth it
here" check — MAR-suspect Severe/High+correlated, multi-MAR, MCAR High/Severe,
and the Bimodal Imputation Framework's branch 2 — calls
:func:`~dataforge_ml.imputation._escalation._escalation_seam` instead of
picking ``KNN`` or ``MICE`` itself. That seam is the single model-based
escalation point: Feasibility Floor → Signal Score → Capability Ladder
(ADR-0091, ADR-0092, ADR-0094).

``Unpredictable`` and ``NearConstant`` are profile labels only here — neither
is a router predicate (ADR-0092 supersedes ADR-0016's guard and the former
``NearConstant`` cap): the Signal Score's bottom tier reaches the same scalar
fill for a column either flag describes, so routing asks the Signal Score
instead of shortcutting on the label directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag, MissingSeverity
from ..profiling._numeric_config import (
    KurtosisTag,
    NumericFlag,
    NumericStats,
    SkewSeverity,
)
from ._config import ImputationStrategy, NumericImputationConfig
from ._escalation import _escalation_seam, _forced_feasibility, _Routed
from ._evidence import EntryCause, RoutingReason

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile
    from ..profiling._correlation_config import CorrelationProfileResult


class _StrategyRouter:
    """Pure strategy router for numeric column imputation.

    Consumes profiling signals — column profile, dataset-level context, and
    configuration — and returns the imputation strategy together with its
    Routing Reason and typed Routing Evidence (ADR-0105). No
    ``pl.DataFrame`` parameter is accepted; all computation is over scalar
    values derived from Phase 1 profiling.

    The caller is responsible for computing actual fill values (mean, median,
    mode) from the training DataFrame after receiving the strategy.
    """

    def route(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        mnar_columns: set[str],
        feature_correlation: CorrelationProfileResult | None = None,
        per_column_strategy: dict[str, ImputationStrategy] | None = None,
        per_column_constant_fill: dict[str, float] | None = None,
    ) -> _Routed:
        """Route a single column to its imputation strategy.

        Parameters
        ----------
        col : str
            Column name.
        cp : ColumnProfile
            Phase 1 profile for the column.
        config : NumericImputationConfig
            Imputation configuration supplying size guards and thresholds.
        n_rows : int
            Number of rows the routing is performed against.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        mnar_columns : set[str]
            Columns declared MNAR by user configuration.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed Pearson correlation matrix from Phase 1.  Used for
            the escalation point's Signal Score and the Bimodal Imputation
            Framework's branch selection.
        per_column_strategy : dict[str, ImputationStrategy], optional
            Explicit per-column strategy overrides.  When ``col`` is present,
            the declared strategy is returned immediately at Priority 1.5,
            bypassing all routing priorities 2–7.
        per_column_constant_fill : dict[str, float], optional
            User-declared constant fill values.  When ``col`` is present, the
            column is routed to ``Constant`` at Priority 1.5, bypassing all
            routing priorities 2–7.  Checked before ``per_column_strategy`` so
            that a column in ``per_column_constant_fill`` always produces a
            ``Constant`` record regardless of any other override.

        Returns
        -------
        _Routed
            The strategy, its Routing Reason, and the evidence walk. A column
            forced onto ``KNN``, ``MICE``, ``ClusterConditional`` or
            ``GMMSampling`` carries one ``feasibility`` fact for the forced
            strategy, ``passed`` false when it was forced past a term
            (ADR-0091) — never a block.
        """
        missingness = cp.missingness

        # Priority 1: DropCandidate — >50% missing
        if missingness and missingness.has_flag(MissingnessFlag.DropCandidate):
            return _Routed(ImputationStrategy.Dropped, RoutingReason.drop_candidate)

        # Priority 1.5: per_column_constant_fill override — fires before per_column_strategy
        if per_column_constant_fill and col in per_column_constant_fill:
            return _Routed(
                ImputationStrategy.Constant, RoutingReason.constant_declared
            )

        # Priority 1.5: per_column_strategy override — fires after DropCandidate, before MNAR
        if per_column_strategy and col in per_column_strategy:
            declared = per_column_strategy[col]
            fact = _forced_feasibility(
                col,
                cp,
                config,
                n_rows,
                n_features,
                declared,
                feature_correlation=feature_correlation,
            )
            return _Routed(
                declared,
                RoutingReason.strategy_forced,
                (fact,) if fact is not None else (),
            )

        # Priority 2: MNAR declared by user
        if col in mnar_columns:
            return _Routed(ImputationStrategy.MNAR, RoutingReason.mnar_declared)

        # No effective missingness → Passthrough
        if missingness is None or missingness.effective_null_count == 0:
            return _Routed(
                ImputationStrategy.Passthrough, RoutingReason.no_missing_values
            )

        # Priority 3: BoundedDiscrete gate — model-aware sub-chain with domain-snap
        if cp.numeric_kind == NumericKind.BoundedDiscrete:
            return self._route_bounded_discrete(
                col=col,
                cp=cp,
                config=config,
                missingness=missingness,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                feature_correlation=feature_correlation,
            )

        stats = cp.stats if isinstance(cp.stats, NumericStats) else None

        # Priority 3.5: Bimodal Imputation Framework (non-BoundedDiscrete only)
        if stats is not None and stats.has_flag(NumericFlag.Bimodal):
            return self._route_bimodal(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                bounded_discrete=False,
            )

        # Priority 4 (MARSuspect) — full fallback chain. Unpredictable and
        # NearConstant are profile labels only (ADR-0092): the Signal Score's
        # bottom tier reaches the same scalar fill for such a column, so
        # routing no longer shortcuts on either flag directly.
        if missingness.has_flag(MissingnessFlag.MARSuspect):
            return self._mar_strategy(
                col=col,
                cp=cp,
                severity=missingness.severity,
                corrs=missingness.correlated_with,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                feature_correlation=feature_correlation,
                kurtosis_tag=stats.kurtosis_tag if stats is not None else None,
                skewness_severity=(
                    stats.skewness_severity if stats is not None else None
                ),
            )

        # Priority 5: MCAR routing by severity and distribution shape
        severity = missingness.severity
        skew_sev = stats.skewness_severity if stats else None
        kurtosis_tag = stats.kurtosis_tag if stats else None

        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                (EntryCause.mcar_high_severity,),
            )

        # MCAR Minor: Leptokurtic escalates to model-based regardless of skew
        if (
            severity == MissingSeverity.Minor
            and kurtosis_tag == KurtosisTag.Leptokurtic
        ):
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                (EntryCause.mcar_minor_leptokurtic,),
            )

        # Minor + Normal skew → Mean (Platykurtic does not escalate)
        if severity == MissingSeverity.Minor and skew_sev in (
            None,
            SkewSeverity.Normal,
        ):
            return _Routed(ImputationStrategy.Mean, RoutingReason.mcar_minor_unskewed)

        # MCAR Moderate: Leptokurtic or Severe skew escalates to model-based
        if severity == MissingSeverity.Moderate and (
            kurtosis_tag == KurtosisTag.Leptokurtic or skew_sev == SkewSeverity.Severe
        ):
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                (EntryCause.mcar_moderate_shape,),
            )

        # Minor/Moderate + skew >= Moderate → Median
        return _Routed(ImputationStrategy.Median, RoutingReason.mcar_skewed)

    def _route_bimodal(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        feature_correlation: CorrelationProfileResult | None,
        *,
        bounded_discrete: bool,
    ) -> _Routed:
        """Route a bimodal column through the Bimodal Imputation Framework.

        Branch 4 takes GMM Sampling, or Mode for a BoundedDiscrete column.
        """
        if col in config.bimodal_grouping_variables:
            return _Routed(
                ImputationStrategy.ClusterConditional,
                RoutingReason.bimodal_grouping_declared,
            )

        if feature_correlation is not None:
            col_corrs = feature_correlation.pearson_matrix.get(col, {})
            count = sum(
                1
                for c, r in col_corrs.items()
                if c != col and abs(r) > config.bimodal_correlation_threshold
            )

            if count >= config.bimodal_min_correlated_features:
                entries = (EntryCause.bimodal_many_correlated,)
                if bounded_discrete:
                    entries = (EntryCause.bounded_discrete, *entries)
                return _escalation_seam(
                    col,
                    cp,
                    config,
                    n_rows,
                    n_features,
                    feature_correlation,
                    entries,
                    bimodal=True,
                )
            if count > 0:
                return _Routed(
                    ImputationStrategy.ClusterConditional,
                    RoutingReason.bimodal_few_correlated,
                )

        return _Routed(
            ImputationStrategy.Mode
            if bounded_discrete
            else ImputationStrategy.GMMSampling,
            RoutingReason.bimodal_uncorrelated,
        )

    def _route_bounded_discrete(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        missingness: object,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        feature_correlation: CorrelationProfileResult | None = None,
    ) -> _Routed:
        """Route a BoundedDiscrete column through the model-aware sub-chain.

        Every scalar terminal is Mode — Mean and Median are not valid domain
        members — under the Routing Reason ``bounded_discrete_scalar``.

        Parameters
        ----------
        col : str
            Column name.
        cp : ColumnProfile
            Phase 1 profile for the column.
        config : NumericImputationConfig
            Imputation configuration.
        missingness : ColumnMissingnessProfile or None
            Missingness profile for the column.
        n_rows : int
            Number of rows the routing is performed against.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed Pearson correlation matrix from Phase 1.

        Returns
        -------
        _Routed
            The strategy, its Routing Reason, and the evidence walk.
        """
        stats = cp.stats if isinstance(cp.stats, NumericStats) else None

        # (c) Bimodal — BoundedDiscrete variant (branch 4 → Mode, not GMMSampling).
        # Unpredictable and NearConstant are profile labels only (ADR-0092):
        # the escalation seam's Feasibility Floor and Signal Score already
        # reach Mode for a column either flag describes.
        if stats is not None and stats.has_flag(NumericFlag.Bimodal):
            return self._route_bimodal(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                bounded_discrete=True,
            )

        # MARSuspect → domain-snapped MAR sub-chain (Mode replaces Median as terminal)
        if missingness is not None and missingness.has_flag(MissingnessFlag.MARSuspect):
            routed = self._mar_strategy(
                col=col,
                cp=cp,
                severity=missingness.severity,
                corrs=missingness.correlated_with,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                feature_correlation=feature_correlation,
                entry_prefix=(EntryCause.bounded_discrete,),
            )
            if routed.strategy == ImputationStrategy.Median:
                return _Routed(
                    ImputationStrategy.Mode,
                    RoutingReason.bounded_discrete_scalar,
                    routed.evidence,
                )
            return routed

        # MCAR by severity — same routing as non-BoundedDiscrete; the seam's
        # bottom tier already fills a BoundedDiscrete column with Mode.
        severity = missingness.severity if missingness is not None else None
        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                (EntryCause.bounded_discrete, EntryCause.mcar_high_severity),
            )

        # Every remaining case → Mode (BoundedDiscrete scalar-fill rule)
        return _Routed(ImputationStrategy.Mode, RoutingReason.bounded_discrete_scalar)

    def _mar_strategy(
        self,
        col: str,
        cp: ColumnProfile,
        severity: MissingSeverity | None,
        corrs: list[str],
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        feature_correlation: CorrelationProfileResult | None = None,
        kurtosis_tag: KurtosisTag | None = None,
        skewness_severity: SkewSeverity | None = None,
        entry_prefix: tuple[EntryCause, ...] = (),
    ) -> _Routed:
        """Route a MAR-suspect column, deferring every model-worth-it check to the seam.

        Parameters
        ----------
        col : str
            Column name.
        cp : ColumnProfile
            Phase 1 profile for the column.
        severity : MissingSeverity or None
            Missingness severity for the column.
        corrs : list[str]
            Columns whose missingness is correlated with this column.
        config : NumericImputationConfig
            Imputation configuration.
        n_rows : int
            Number of rows the routing is performed against.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed pairwise Pearson correlation matrix from Phase 1,
            read by the escalation seam's Signal Score.
        kurtosis_tag : KurtosisTag, optional
            Kurtosis classification from Phase 1.  Used for distribution shape
            escalation at Minor/Moderate severity.
        skewness_severity : SkewSeverity, optional
            Skewness severity from Phase 1.  Used for distribution shape
            escalation at Minor/Moderate severity.
        entry_prefix : tuple[EntryCause, ...], default ()
            Entry causes recorded before the MAR cause, such as
            ``bounded_discrete``.

        Returns
        -------
        _Routed
            The strategy, its Routing Reason, and the evidence walk.
        """

        def seam(cause: EntryCause) -> _Routed:
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                (*entry_prefix, cause),
            )

        if multi_mar:
            return seam(EntryCause.mar_multi)
        if severity == MissingSeverity.Severe:
            return seam(EntryCause.mar_severe)
        if severity == MissingSeverity.High and corrs:
            return seam(EntryCause.mar_high_correlated)
        # High with empty correlations → MCAR High fallback chain
        if severity == MissingSeverity.High:
            return seam(EntryCause.mar_high_uncorrelated)
        # Minor/Moderate: Leptokurtic or Severe skew → attempt model-based
        if severity in (MissingSeverity.Minor, MissingSeverity.Moderate) and (
            kurtosis_tag == KurtosisTag.Leptokurtic
            or skewness_severity == SkewSeverity.Severe
        ):
            return seam(EntryCause.mar_shape_escalation)

        return _Routed(ImputationStrategy.Median, RoutingReason.mar_unescalated)
