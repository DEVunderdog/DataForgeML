# ADR 0032: Bimodal columns follow a four-branch imputation framework ordered by available cluster evidence

When `NumericFlag.Bimodal` is set, imputation is routed through a dedicated four-branch framework that escalates from the richest available cluster evidence to the poorest:

1. **Grouping variable available** — the user has declared (or the system has auto-detected) a categorical column whose values cleanly separate the two clusters. Fill with the cluster-conditional statistic (mean or median within each group, skew-driven).
2. **Many correlated features** — `≥ bimodal_min_correlated_features` numeric features with `|r| > bimodal_correlation_threshold` are available. Use MICE or KNN. Regression and MICE with tree-based estimators are preferred over KNN because KNN averages neighbor values and can produce valley fills when neighbors span both clusters.
3. **Few correlated features** — fewer than `bimodal_min_correlated_features` features with `|r| > bimodal_correlation_threshold`. Use cluster-conditional imputation: assign each missing row to its nearest cluster centroid (derived from GMM centers and cluster-conditional feature means computed at fit time), then fill with that cluster's statistic.
4. **No correlated features** — GMM Sampling: sample fill values from the fitted 2-component GMM. Requires `PipelineConfig.random_seed` for determinism.

`bimodal_min_correlated_features: int = 3` and `bimodal_correlation_threshold: float = 0.2` in `NumericImputationConfig`. The threshold is separate from `mcar_feature_predictability_threshold` — bimodal feature-counting ("does this feature know which cluster a row belongs to") and MCAR predictability ("can a model predict this column") are different questions that deserve separate knobs. User-declared grouping variables go in `NumericImputationConfig.bimodal_grouping_variables: dict[str, str]` and take priority over auto-detection.

## Priority position

For non-BoundedDiscrete columns, the Bimodal Imputation Framework fires at **Priority 3.5** in the Numeric Imputation Decision Priority — after the BoundedDiscrete gate (Priority 3) but before the Unpredictable guard (Priority 4). This means a bimodal column with no correlated features routes to GMM Sampling even when `NonlinearityTag.Unpredictable` is set (the guard never fires for bimodal columns). For BoundedDiscrete columns, the framework fires inside the BoundedDiscrete gate at step (c), after the Unpredictable guard at step (a) — so BoundedDiscrete + Bimodal + Unpredictable → Mode (the Unpredictable guard fires first).

## feature_correlation is None

When `feature_correlation` is `None` (correlation profiling was not run), branches 2 and 3 cannot be evaluated. Branch 1 (grouping variable) is checked first; if no grouping variable is declared, branch 4 applies directly. For non-BoundedDiscrete columns this means GMM Sampling; for BoundedDiscrete columns this means Mode. Falling through to Median (as if no bimodal structure existed) was rejected: Median lands in the valley between the two peaks, the worst possible fill for a confirmed bimodal column.

## NonlinearityTag.Linear override for bimodal columns

When `NonlinearityTag.Linear` is detected for a bimodal column, the Regression estimator is overridden to `RandomForestRegressor` regardless of the tag. A linear model (BayesianRidge) fits a linear function of the features and produces near-mean predictions for ambiguous rows — effectively a valley fill for bimodal targets. Tree-based models learn cluster-specific predictions from feature splits and are the correct choice regardless of apparent feature-target linearity. `NonlinearityTag.MonotonicNonlinear` and `ComplexNonlinear` already map to tree-based estimators and are unaffected.

## GMM Sampling overrides the Unpredictable guard

`NonlinearityTag.Unpredictable` is an unconditional pre-routing guard for non-bimodal columns (ADR-0016): no model achieves R² > 0.1, so model-based imputation is indistinguishable from a scalar fill. For bimodal columns in branch 4, the guard is overridden: GMM Sampling fills missing values from the distribution shape rather than predicting from features. It makes no R² claim and is not subject to the R² guard. Median (the Unpredictable fallback) lands in the valley between the two peaks; GMM Sampling preserves the bimodal marginal distribution in the imputed output. The override applies exclusively to the bimodal + no-correlated-features combination. The Unpredictable guard remains unconditional for all non-bimodal columns.

## Bimodal discrete

For `NumericKind.BoundedDiscrete` columns, the Bimodal Imputation Framework applies inside the BoundedDiscrete gate (Priority 3, step c). Branches 1–3 are available with normal strategy assignments; model-based predictions are domain-snapped (ADR-0018) and all scalar fills use Mode (not mean/median — ADR-0035). Branch 4 routes to **Mode**, not GMM Sampling. Rejected alternative for branch 4: domain-constrained GMM Sampling (draw from the 2-component GMM, snap to nearest valid discrete value). This was rejected because sampling a continuous distribution into a finite discrete domain changes the semantics of the fill — the GMM was fitted to float observations and its mass between valid integer values is wasted or misleadingly rounded. For a column with values in `{0, 1, 2, 3}`, a GMM sample of 1.7 rounds to 2 regardless of whether the column's actual second cluster peaks at 2 or 3, producing fills that depend on the rounding artifact rather than the cluster structure. Mode per cluster is a cleaner, domain-native fallback that makes no continuous-domain assumptions.

## The rejected alternative

The rejected design routed bimodal columns directly into the existing distribution shape escalation at Priority 7 (KNN → Regression → Median) without a dedicated framework. This is insufficient because: (1) KNN averages neighbor values and risks valley fills when neighbors span both clusters; (2) the existing escalation has no concept of grouping variables or cluster-conditional fills; (3) the Unpredictable guard would send bimodal + no-feature columns to Median even though GMM Sampling is available and strictly better.

## random_seed

GMM Sampling introduces stochasticity. `PipelineConfig.random_seed: Optional[int]` is the single seed for all stochastic operations across the pipeline. `None` produces non-deterministic output. All phases that introduce randomness read from this field — it is not scoped per-phase.
