"""
_StrategyRouter — pure strategy routing for numeric column imputation.

Accepts profiling signals and returns a routing decision.  No DataFrame
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
from ._escalation import _escalation_seam, _forced_model_signal

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
        cp: ColumnProfile,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        mnar_columns: set[str],
        feature_correlation: CorrelationProfileResult | None = None,
        per_column_strategy: dict[str, ImputationStrategy] | None = None,
        per_column_constant_fill: dict[str, float] | None = None,
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
        tuple[ImputationStrategy, list[str]]
            ``(strategy, signals)`` where ``signals`` records every routing
            decision in order. A user-declared strategy is recorded only as the
            ``per_column_strategy_override`` signal, for human readers; no
            machine predicate for it survives on the routing. A column forced
            onto ``KNN`` or ``MICE`` also carries the Feasibility Floor's
            failed terms, if any, as an additional informational signal
            (ADR-0091) — never a block.
        """
        missingness = cp.missingness
        signals: list[str] = []

        # Priority 1: DropCandidate — >50% missing
        if missingness and missingness.has_flag(MissingnessFlag.DropCandidate):
            signals.append(
                f"drop_candidate: {missingness.effective_null_ratio:.1%} effective missing"
            )
            return ImputationStrategy.Dropped, signals

        # Priority 1.5: per_column_constant_fill override — fires before per_column_strategy
        if per_column_constant_fill and col in per_column_constant_fill:
            signals.append(
                "per_column_constant_fill_override: user declared constant fill"
            )
            return ImputationStrategy.Constant, signals

        # Priority 1.5: per_column_strategy override — fires after DropCandidate, before MNAR
        if per_column_strategy and col in per_column_strategy:
            declared = per_column_strategy[col]
            signals.append(
                f"per_column_strategy_override: user forced strategy={declared}"
            )
            forced_signal = _forced_model_signal(
                col,
                cp,
                config,
                n_rows,
                n_features,
                declared,
                feature_correlation=feature_correlation,
            )
            if forced_signal is not None:
                signals.append(forced_signal)
            return declared, signals

        # Priority 2: MNAR declared by user
        if col in mnar_columns:
            signals.append("declared MNAR by user configuration")
            if cp.numeric_kind == NumericKind.BoundedDiscrete:
                signals.append("mnar_fill: mode")
            else:
                mnar_stats = cp.stats if isinstance(cp.stats, NumericStats) else None
                skew_sev = (
                    mnar_stats.skewness_severity if mnar_stats is not None else None
                )
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

        # Priority 3.5: Bimodal Imputation Framework (non-BoundedDiscrete only)
        if stats is not None and stats.has_flag(NumericFlag.Bimodal):
            return self._route_bimodal(
                col,
                cp,
                config,
                n_rows,
                n_features,
                multi_mar,
                signals,
                feature_correlation,
            )

        # Priority 4 (MARSuspect) — full fallback chain. Unpredictable and
        # NearConstant are profile labels only (ADR-0092): the Signal Score's
        # bottom tier reaches the same scalar fill for such a column, so
        # routing no longer shortcuts on either flag directly.
        if missingness.has_flag(MissingnessFlag.MARSuspect):
            corrs = missingness.correlated_with
            signals.append(f"mar_suspect: correlated missingness with {corrs}")
            strategy, signal = self._mar_strategy(
                col=col,
                cp=cp,
                severity=missingness.severity,
                corrs=corrs,
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
            signals.append(signal)
            return strategy, signals

        # Priority 5: MCAR routing by severity and distribution shape
        severity = missingness.severity
        skew_sev = stats.skewness_severity if stats else None
        kurtosis_tag = stats.kurtosis_tag if stats else None

        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            strategy, signal = self._mcar_model_strategy(
                col=col,
                cp=cp,
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # MCAR Minor: Leptokurtic escalates to model-based regardless of skew
        if (
            severity == MissingSeverity.Minor
            and kurtosis_tag == KurtosisTag.Leptokurtic
        ):
            signals.append(
                "mcar minor + leptokurtic: heavy-tailed distribution escalation to model-based"
            )
            strategy, signal = self._mcar_model_strategy(
                col=col,
                cp=cp,
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # Minor + Normal skew → Mean (Platykurtic noted but does not escalate)
        if severity == MissingSeverity.Minor and skew_sev in (
            None,
            SkewSeverity.Normal,
        ):
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
                col=col,
                cp=cp,
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            return strategy, signals

        # Minor/Moderate + skew >= Moderate → Median
        signals.append(
            f"mcar {severity} + skew={skew_sev or 'unknown'}: median imputation"
        )
        return ImputationStrategy.Median, signals

    def _route_bimodal(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        signals: list[str],
        feature_correlation: CorrelationProfileResult | None = None,
    ) -> tuple[ImputationStrategy, list[str]]:
        """Route a bimodal column to the Bimodal Imputation Framework."""
        if col in config.bimodal_grouping_variables:
            signals.append("bimodal_branch_1: grouping variable declared")
            return ImputationStrategy.ClusterConditional, signals

        if feature_correlation is not None:
            col_corrs = feature_correlation.pearson_matrix.get(col, {})
            corr_list = [
                c
                for c, r in col_corrs.items()
                if c != col and abs(r) > config.bimodal_correlation_threshold
            ]
            count = len(corr_list)

            if count >= config.bimodal_min_correlated_features:
                signals.append("bimodal_branch_2: N correlated features >= threshold")
                strategy, signal = self._mar_strategy(
                    col=col,
                    cp=cp,
                    severity=MissingSeverity.High,
                    corrs=corr_list,
                    config=config,
                    n_rows=n_rows,
                    n_features=n_features,
                    multi_mar=False,
                    feature_correlation=feature_correlation,
                    bimodal=True,
                )
                signals.append(signal)
                return strategy, signals
            elif count > 0:
                signals.append("bimodal_branch_3: N correlated features < threshold")
                return ImputationStrategy.ClusterConditional, signals

        signals.append("bimodal_branch_4: no correlated features, gmm_sampling")
        return ImputationStrategy.GMMSampling, signals

    def _route_bimodal_bounded_discrete(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        signals: list[str],
        feature_correlation: CorrelationProfileResult | None = None,
    ) -> tuple[ImputationStrategy, list[str]]:
        """Route a BoundedDiscrete bimodal column to the Bimodal Imputation Framework."""
        if col in config.bimodal_grouping_variables:
            signals.append("bimodal_branch_1: grouping variable declared")
            return ImputationStrategy.ClusterConditional, signals

        if feature_correlation is not None:
            col_corrs = feature_correlation.pearson_matrix.get(col, {})
            corr_list = [
                c
                for c, r in col_corrs.items()
                if c != col and abs(r) > config.bimodal_correlation_threshold
            ]
            count = len(corr_list)

            if count >= config.bimodal_min_correlated_features:
                signals.append("bimodal_branch_2: N correlated features >= threshold")
                strategy, signal = self._mar_strategy(
                    col=col,
                    cp=cp,
                    severity=MissingSeverity.High,
                    corrs=corr_list,
                    config=config,
                    n_rows=n_rows,
                    n_features=n_features,
                    multi_mar=False,
                    feature_correlation=feature_correlation,
                    bimodal=True,
                )
                signals.append(signal)
                return strategy, signals
            elif count > 0:
                signals.append("bimodal_branch_3: N correlated features < threshold")
                return ImputationStrategy.ClusterConditional, signals

        signals.append("bimodal_branch_4: no correlated features, mode")
        return ImputationStrategy.Mode, signals

    def _route_bounded_discrete(
        self,
        col: str,
        cp: ColumnProfile,
        config: NumericImputationConfig,
        missingness: object,
        n_rows: int,
        n_features: int,
        multi_mar: bool,
        signals: list[str],
        feature_correlation: CorrelationProfileResult | None = None,
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
            Number of rows the routing is performed against.
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

        # (c) Bimodal — BoundedDiscrete variant (branch 4 → Mode, not GMMSampling).
        # Unpredictable and NearConstant are profile labels only (ADR-0092):
        # the escalation seam's Feasibility Floor and Signal Score already
        # reach Mode for a column either flag describes.
        if stats is not None and stats.has_flag(NumericFlag.Bimodal):
            return self._route_bimodal_bounded_discrete(
                col,
                cp,
                config,
                n_rows,
                n_features,
                multi_mar,
                signals,
                feature_correlation,
            )

        # MARSuspect → domain-snapped MAR sub-chain (Mode replaces Median as terminal)
        if missingness is not None and missingness.has_flag(MissingnessFlag.MARSuspect):
            corrs = missingness.correlated_with
            signals.append(
                f"bounded_discrete: mar_suspect: correlated missingness with {corrs}"
            )
            strategy, signal = self._mar_strategy(
                col=col,
                cp=cp,
                severity=missingness.severity,
                corrs=corrs,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                multi_mar=multi_mar,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            if strategy == ImputationStrategy.Median:
                signals.append(
                    "bounded_discrete: terminal → mode (median not valid domain member)"
                )
                return ImputationStrategy.Mode, signals
            return strategy, signals

        # MCAR by severity — same routing as non-BoundedDiscrete; Mode replaces Median
        severity = missingness.severity if missingness is not None else None
        skew_sev = stats.skewness_severity if stats is not None else None

        if severity in (MissingSeverity.High, MissingSeverity.Severe):
            strategy, signal = self._mcar_model_strategy(
                col=col,
                cp=cp,
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
            )
            signals.append(signal)
            if strategy == ImputationStrategy.Median:
                signals.append(
                    "bounded_discrete: terminal → mode (median not valid domain member)"
                )
                return ImputationStrategy.Mode, signals
            return strategy, signals

        # Minor + Normal skew → Mode (BoundedDiscrete scalar-fill rule; Mean is not a valid domain member)
        if severity == MissingSeverity.Minor and skew_sev in (
            None,
            SkewSeverity.Normal,
        ):
            signals.append("bounded_discrete: mcar minor + normal skew → mode")
            return ImputationStrategy.Mode, signals

        # Terminal fallback → Mode (replaces Median for all remaining cases)
        signals.append(
            f"bounded_discrete: terminal → mode (severity={severity}, skew={skew_sev})"
        )
        return ImputationStrategy.Mode, signals

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
        bimodal: bool = False,
    ) -> tuple[ImputationStrategy, str]:
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
        bimodal : bool, default False
            Whether this call originates from the Bimodal Imputation
            Framework's branch 2 — passed through to the escalation seam so
            its bottom tier picks GMM Sampling.

        Returns
        -------
        tuple[ImputationStrategy, str]
            ``(strategy, signal)`` where ``signal`` records the routing decision.
        """
        if multi_mar:
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                f"mar: multi-mar (>=2 MAR-suspect columns, {n_rows:,} rows)",
                bimodal=bimodal,
            )

        if severity == MissingSeverity.Severe:
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                f"mar: severe missingness ({n_rows:,} rows)",
                bimodal=bimodal,
            )

        if severity == MissingSeverity.High and corrs:
            return _escalation_seam(
                col,
                cp,
                config,
                n_rows,
                n_features,
                feature_correlation,
                f"mar: high severity + correlated missingness with {corrs}",
                bimodal=bimodal,
            )

        # High with empty correlations → MCAR High fallback chain
        if severity == MissingSeverity.High and not corrs:
            strategy, inner_signal = self._mcar_model_strategy(
                col=col,
                cp=cp,
                severity=MissingSeverity.High,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
                bimodal=bimodal,
            )
            return (
                strategy,
                f"mar high + no missingness correlations detected, applying MCAR High fallback chain | {inner_signal}",
            )

        # Minor/Moderate: distribution shape escalation — Leptokurtic or Severe skew → attempt model-based
        if severity in (MissingSeverity.Minor, MissingSeverity.Moderate) and (
            kurtosis_tag == KurtosisTag.Leptokurtic
            or skewness_severity == SkewSeverity.Severe
        ):
            escalation_reasons = []
            if kurtosis_tag == KurtosisTag.Leptokurtic:
                escalation_reasons.append("leptokurtic")
            if skewness_severity == SkewSeverity.Severe:
                escalation_reasons.append("skew=severe")
            strategy, inner_signal = self._mcar_model_strategy(
                col=col,
                cp=cp,
                severity=severity,
                config=config,
                n_rows=n_rows,
                n_features=n_features,
                feature_correlation=feature_correlation,
                bimodal=bimodal,
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
        col: str,
        cp: ColumnProfile,
        severity: MissingSeverity,
        config: NumericImputationConfig,
        n_rows: int,
        n_features: int,
        feature_correlation: CorrelationProfileResult | None = None,
        bimodal: bool = False,
    ) -> tuple[ImputationStrategy, str]:
        """Route an MCAR High/Severe column through the escalation seam.

        The former feature-predictability terminal check is deleted (ADR-0092):
        the Signal Score grades a column's predictor correlation directly,
        so every MCAR High/Severe column defers to the seam uniformly.

        Parameters
        ----------
        col : str
            Column name.
        cp : ColumnProfile
            Phase 1 profile for the column.
        severity : MissingSeverity
            Missingness severity; only ``High`` and ``Severe`` are expected.
        config : NumericImputationConfig
            Imputation configuration.
        n_rows : int
            Number of rows the routing is performed against.
        n_features : int
            Total number of numeric columns being imputed.
        feature_correlation : CorrelationProfileResult, optional
            Pre-computed pairwise Pearson correlation matrix from Phase 1,
            read by the escalation seam's Signal Score.
        bimodal : bool, default False
            Whether this call originates from the Bimodal Imputation
            Framework's branch 2 — passed through to the escalation seam.

        Returns
        -------
        tuple[ImputationStrategy, str]
            ``(strategy, signal)`` where ``signal`` records the routing decision.
        """
        return _escalation_seam(
            col,
            cp,
            config,
            n_rows,
            n_features,
            feature_correlation,
            f"mcar: {severity} severity ({n_rows:,} rows, {n_features} features)",
            bimodal=bimodal,
        )
