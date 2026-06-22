"""
_StrategyRouter — pure strategy routing for numeric column imputation.

Accepts profiling signals and returns a routing decision.  No DataFrame
access; all inputs are column profiles, configuration, and scalar context
values derived from the training set by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..profiling._config import NumericKind
from ..profiling._missingness_config import MissingnessFlag, MissingSeverity
from ..profiling._numeric_config import (
    KurtosisTag,
    NonlinearityTag,
    NumericFlag,
    NumericStats,
    SkewSeverity,
)
from ._config import ImputationStrategy, NumericImputationConfig

if TYPE_CHECKING:
    from ..profiling._config import ColumnProfile
    from ..profiling._correlation_config import CorrelationProfileResult


class _StrategyRouter:
    """Pure strategy router for numeric column imputation.

    Consumes profiling signals — column profile, dataset-level context, and
    configuration — and returns the imputation strategy together with a list
    of human-readable signals that explain the routing decision.  No
    ``pl.DataFrame`` parameter is accepted; all computation is over scalar
    values derived from Phase 1 profiling.

    The caller is responsible for computing actual fill values (mean, median,
    mode) from the training DataFrame after receiving the strategy.
    """

    def route(
        self,
        col: str,
        cp: "ColumnProfile",
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        mnar_columns: set[str],
        feature_correlation: "Optional[CorrelationProfileResult]" = None,
    ) -> tuple[ImputationStrategy, list[str]]:
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
            Number of rows in the training split.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        mnar_columns : set[str]
            Columns declared MNAR by user configuration.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed Pearson correlation matrix from Phase 1.  Used only
            for the MCAR feature-predictability check.

        Returns
        -------
        tuple[ImputationStrategy, list[str]]
            ``(strategy, signals)`` where ``signals`` records every routing
            decision in order.
        """
        missingness = cp.missingness
        signals: list[str] = []

        # Priority 1: DropCandidate — >50% missing
        if missingness and missingness.has_flag(MissingnessFlag.DropCandidate):
            signals.append(
                f"drop_candidate: {missingness.effective_null_ratio:.1%} effective missing"
            )
            return ImputationStrategy.Dropped, signals

        # Priority 2: MNAR declared by user
        if col in mnar_columns:
            signals.append("declared MNAR by user configuration")
            if cp.numeric_kind == NumericKind.BoundedDiscrete:
                signals.append("mnar_fill: mode")
            else:
                mnar_stats = cp.stats if isinstance(cp.stats, NumericStats) else None
                skew_sev = mnar_stats.skewness_severity if mnar_stats is not None else None
                fill_stat = "mean" if skew_sev == SkewSeverity.Normal else "median"
                signals.append(f"mnar_fill: {fill_stat} (skew={skew_sev or 'unknown'})")
            return ImputationStrategy.MNAR, signals

        # No effective missingness → Passthrough
        if missingness is None or missingness.effective_null_count == 0:
            signals.append("no missing values in full-dataset profile")
            return ImputationStrategy.Passthrough, signals

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
                signals=signals,
                feature_correlation=feature_correlation,
            )

        stats = cp.stats if isinstance(cp.stats, NumericStats) else None

        # Priority 4: Unpredictable guard — non-BoundedDiscrete columns with no predictive signal
        if stats is not None and stats.nonlinearity_tag == NonlinearityTag.Unpredictable:
            mar_suspect = missingness is not None and missingness.has_flag(
                MissingnessFlag.MARSuspect
            )
            signals.append(
                f"unpredictable_guard: nonlinearity_tag=Unpredictable, mar_suspect={mar_suspect}"
            )
            return ImputationStrategy.Median, signals

        # Priority 5: MARSuspect — full fallback chain
        if missingness.has_flag(MissingnessFlag.MARSuspect):
            corrs = missingness.correlated_with
            signals.append(f"mar_suspect: correlated missingness with {corrs}")
            strategy, signal = self._mar_strategy(
                severity=missingness.severity,
                corrs=corrs,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                kurtosis_tag=stats.kurtosis_tag if stats is not None else None,
                skewness_severity=stats.skewness_severity if stats is not None else None,
            )
            signals.append(signal)
            return strategy, signals

        # Priority 6: MCAR routing by severity and distribution shape
        severity = missingness.severity
        skew_sev = stats.skewness_severity if stats else None
        kurtosis_tag = stats.kurtosis_tag if stats else None

        # NearConstant cap — model-based escalation is wasteful when 90%+ share the mode
        if stats is not None and stats.has_flag(NumericFlag.NearConstant):
            signals.append("near_constant: model-based escalation suppressed")
            return ImputationStrategy.Median, signals

        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            strategy, signal = self._mcar_model_strategy(
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                col=col,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # MCAR Minor: Leptokurtic escalates to model-based regardless of skew
        if severity == MissingSeverity.Minor and kurtosis_tag == KurtosisTag.Leptokurtic:
            signals.append(
                "mcar minor + leptokurtic: heavy-tailed distribution escalation to model-based"
            )
            strategy, signal = self._mcar_model_strategy(
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                col=col,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # Minor + Normal skew → Mean (Platykurtic noted but does not escalate)
        if severity == MissingSeverity.Minor and skew_sev in (None, SkewSeverity.Normal):
            if kurtosis_tag == KurtosisTag.Platykurtic:
                signals.append(
                    "mcar minor + platykurtic: thin-tailed distribution, scalar fill representative"
                )
            signals.append(f"mcar minor + skew={skew_sev or 'normal'}: mean imputation")
            return ImputationStrategy.Mean, signals

        # MCAR Moderate: Leptokurtic or Severe skew escalates to model-based
        if severity == MissingSeverity.Moderate and (
            kurtosis_tag == KurtosisTag.Leptokurtic or skew_sev == SkewSeverity.Severe
        ):
            escalation_reasons = []
            if kurtosis_tag == KurtosisTag.Leptokurtic:
                escalation_reasons.append("leptokurtic")
            if skew_sev == SkewSeverity.Severe:
                escalation_reasons.append("skew=severe")
            signals.append(
                f"mcar moderate + {'+'.join(escalation_reasons)}: distribution shape escalation to model-based"
            )
            strategy, signal = self._mcar_model_strategy(
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                col=col,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # Minor/Moderate + skew >= Moderate → Median
        signals.append(f"mcar {severity} + skew={skew_sev or 'unknown'}: median imputation")
        return ImputationStrategy.Median, signals

    def _route_bounded_discrete(
        self,
        col: str,
        cp: "ColumnProfile",
        config: NumericImputationConfig,
        missingness: "object",
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        signals: list[str],
        feature_correlation: "Optional[CorrelationProfileResult]" = None,
    ) -> tuple[ImputationStrategy, list[str]]:
        """Route a BoundedDiscrete column through the model-aware sub-chain.

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
            Number of rows in the training split.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        signals : list[str]
            Signal list to append routing decisions to. Extended in place and
            also returned.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed Pearson correlation matrix from Phase 1.

        Returns
        -------
        tuple[ImputationStrategy, list[str]]
            ``(strategy, signals)`` where ``signals`` records every routing
            decision in order.
        """
        stats = cp.stats if isinstance(cp.stats, NumericStats) else None
        snap_min = stats.min if stats is not None else None
        snap_max = stats.max if stats is not None else None

        # 1. Unpredictable → Mode (terminal; model-based uplift unavailable)
        if stats is not None and stats.nonlinearity_tag == NonlinearityTag.Unpredictable:
            signals.append("bounded_discrete: unpredictable_guard → mode")
            return ImputationStrategy.Mode, signals

        # 2. NearConstant → Mode (terminal; fitting cost wasted when 90%+ share the same value)
        if stats is not None and stats.has_flag(NumericFlag.NearConstant):
            signals.append("bounded_discrete: near_constant → mode")
            return ImputationStrategy.Mode, signals

        # 3. Bimodal → falls through (future scope; no special handling)

        # 4. MARSuspect → domain-snapped MAR sub-chain (Mode replaces Median as terminal)
        if missingness is not None and missingness.has_flag(MissingnessFlag.MARSuspect):
            corrs = missingness.correlated_with
            signals.append(
                f"bounded_discrete: mar_suspect: correlated missingness with {corrs}"
            )
            strategy, signal = self._mar_strategy(
                severity=missingness.severity,
                corrs=corrs,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
            )
            signals.append(signal)
            if strategy == ImputationStrategy.Median:
                signals.append(
                    "bounded_discrete: terminal → mode (median not valid domain member)"
                )
                return ImputationStrategy.Mode, signals
            signals.append(
                f"bounded_discrete: domain_snap_bounds=({snap_min}, {snap_max})"
            )
            return strategy, signals

        # 5. MCAR by severity — same routing as non-BoundedDiscrete; Mode replaces Median
        severity = missingness.severity if missingness is not None else None
        skew_sev = stats.skewness_severity if stats is not None else None

        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            strategy, signal = self._mcar_model_strategy(
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                col=col,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            if strategy == ImputationStrategy.Median:
                signals.append(
                    "bounded_discrete: terminal → mode (median not valid domain member)"
                )
                return ImputationStrategy.Mode, signals
            signals.append(
                f"bounded_discrete: domain_snap_bounds=({snap_min}, {snap_max})"
            )
            return strategy, signals

        # Minor + Normal skew → Mode (BoundedDiscrete scalar-fill rule; Mean is not a valid domain member)
        if severity == MissingSeverity.Minor and skew_sev in (None, SkewSeverity.Normal):
            signals.append("bounded_discrete: mcar minor + normal skew → mode")
            return ImputationStrategy.Mode, signals

        # Terminal fallback → Mode (replaces Median for all remaining cases)
        signals.append(
            f"bounded_discrete: terminal → mode (severity={severity}, skew={skew_sev})"
        )
        return ImputationStrategy.Mode, signals

    def _mar_strategy(
        self,
        severity: "MissingSeverity | None",
        corrs: list[str],
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        kurtosis_tag: "KurtosisTag | None" = None,
        skewness_severity: "SkewSeverity | None" = None,
    ) -> tuple[ImputationStrategy, str]:
        """Full fallback chain for MAR-suspect columns: MICE → Regression → KNN → Median.

        Parameters
        ----------
        severity : MissingSeverity or None
            Missingness severity for the column.
        corrs : list[str]
            Columns whose missingness is correlated with this column.
        config : NumericImputationConfig
            Imputation configuration supplying size guards.
        n_rows : int
            Number of rows in the training split.
        n_features : int
            Total number of numeric columns being imputed.
        multi_mar : bool
            ``True`` when two or more non-dropped, non-MNAR columns carry
            ``MissingnessFlag.MARSuspect``.
        kurtosis_tag : KurtosisTag, optional
            Kurtosis classification from Phase 1.  Used for distribution shape
            escalation at Minor/Moderate severity.
        skewness_severity : SkewSeverity, optional
            Skewness severity from Phase 1.  Used for distribution shape
            escalation at Minor/Moderate severity.

        Returns
        -------
        tuple[ImputationStrategy, str]
            ``(strategy, signal)`` where ``signal`` records the routing decision.
        """
        # Multi-MAR or Severe → MICE
        if multi_mar:
            return ImputationStrategy.MICE, "mice: ≥2 MAR-suspect columns (multi-MAR)"
        if severity == MissingSeverity.Severe:
            return ImputationStrategy.MICE, "mice: MAR-suspect + severe missingness"

        # High with correlations → Regression → KNN → Median
        if severity == MissingSeverity.High and corrs:
            if n_rows >= config.regression_min_rows:
                return (
                    ImputationStrategy.Regression,
                    f"regression: MAR high + correlations, {n_rows:,} rows >= regression_min_rows={config.regression_min_rows:,}",
                )
            if n_rows <= config.knn_max_rows and n_features <= config.knn_max_features:
                return (
                    ImputationStrategy.KNN,
                    f"knn: regression size guard failed ({n_rows:,} rows < {config.regression_min_rows:,})",
                )
            return (
                ImputationStrategy.Median,
                f"median: all size guards failed (rows={n_rows:,}, features={n_features})",
            )

        # High with empty correlations → MCAR High fallback chain (KNN → Regression → Median)
        if severity == MissingSeverity.High and not corrs:
            strategy, inner_signal = self._mcar_model_strategy(
                severity=MissingSeverity.High,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
            )
            return (
                strategy,
                f"knn/regression: MAR high + no missingness correlations detected, applying MCAR High fallback chain | {inner_signal}",
            )

        # Minor/Moderate: distribution shape escalation — Leptokurtic or Severe skew → attempt KNN
        if severity in (MissingSeverity.Minor, MissingSeverity.Moderate):
            if kurtosis_tag == KurtosisTag.Leptokurtic or skewness_severity == SkewSeverity.Severe:
                escalation_reasons = []
                if kurtosis_tag == KurtosisTag.Leptokurtic:
                    escalation_reasons.append("leptokurtic")
                if skewness_severity == SkewSeverity.Severe:
                    escalation_reasons.append("skew=severe")
                strategy, inner_signal = self._mcar_model_strategy(
                    severity=severity,
                    config=config,
                    n_rows=n_rows,
                    n_features=n_features,
                )
                return (
                    strategy,
                    f"mar {severity} + {'+'.join(escalation_reasons)}: distribution shape escalation to model-based | {inner_signal}",
                )

        return (
            ImputationStrategy.Median,
            "median: MAR-suspect fallback (low severity or no correlations)",
        )

    def _mcar_model_strategy(
        self,
        severity: "MissingSeverity",
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        col: "Optional[str]" = None,
        feature_correlation: "Optional[CorrelationProfileResult]" = None,
    ) -> tuple[ImputationStrategy, str]:
        """Full fallback chain for MCAR High/Severe: KNN → Regression → Median (High); MICE (Severe).

        Parameters
        ----------
        severity : MissingSeverity
            Missingness severity; only ``High`` and ``Severe`` are expected.
        config : NumericImputationConfig
            Imputation configuration supplying size guards and the
            ``mcar_feature_predictability_threshold``.
        n_rows : int
            Number of rows in the training split.
        n_features : int

            Total number of numeric columns being imputed.
        col : str, optional
            Column name; required when ``feature_correlation`` is provided so
            that the predictability check can look up the correct row.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed Pearson correlation matrix from Phase 1.  When
            provided and ``col`` is set, the feature-predictability check is
            applied before routing to KNN or Regression.

        Returns
        -------
        tuple[ImputationStrategy, str]
            ``(strategy, signal)`` where ``signal`` records the routing decision.
        """
        if severity == MissingSeverity.Severe:
            return ImputationStrategy.MICE, "mice: MCAR severe missingness"

        # Feature-predictability check: skip KNN/Regression when no predictor carries useful signal
        if feature_correlation is not None and col is not None:
            col_corrs = feature_correlation.pearson_matrix.get(col, {})
            abs_rs = [abs(r) for c, r in col_corrs.items() if c != col]
            if abs_rs:
                max_abs_r = max(abs_rs)
                threshold = config.mcar_feature_predictability_threshold
                if max_abs_r < threshold:
                    return (
                        ImputationStrategy.Median,
                        f"median: feature-predictability check failed (max |r|={max_abs_r:.2f} < threshold={threshold})",
                    )

        # KNN → Regression → Median
        if n_rows <= config.knn_max_rows and n_features <= config.knn_max_features:
            return (
                ImputationStrategy.KNN,
                f"knn: MCAR {severity}, rows={n_rows:,} <= {config.knn_max_rows:,}, features={n_features} <= {config.knn_max_features}",
            )
        if n_rows >= config.regression_min_rows:
            return (
                ImputationStrategy.Regression,
                f"regression: knn size guard failed, {n_rows:,} rows >= regression_min_rows={config.regression_min_rows:,}",
            )
        return (
            ImputationStrategy.Median,
            f"median: all size guards failed (rows={n_rows:,}, features={n_features})",
        )
