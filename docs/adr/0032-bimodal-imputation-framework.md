# ADR 0032: Bimodal columns follow a four-branch imputation framework ordered by available cluster evidence

When `NumericFlag.Bimodal` is set, imputation is routed through a dedicated four-branch framework that escalates from the richest available cluster evidence to the poorest:

1. **Grouping variable available** — the user has declared (or the system has auto-detected) a categorical column whose values cleanly separate the two clusters. Fill with the cluster-conditional statistic (mean or median within each group, skew-driven).
2. **Many correlated features** — `≥ bimodal_min_correlated_features` numeric features with `|r| > 0.2` are available. Use MICE or KNN. Regression and MICE with tree-based estimators are preferred over KNN because KNN averages neighbor values and can produce valley fills when neighbors span both clusters.
3. **Few correlated features** — fewer than `bimodal_min_correlated_features` features with `|r| > 0.2`. Use cluster-conditional imputation: assign each missing row to its nearest cluster centroid (derived from GMM centers and cluster-conditional feature means computed at fit time), then fill with that cluster's statistic.
4. **No correlated features** — GMM Sampling: sample fill values from the fitted 2-component GMM. Requires `PipelineConfig.random_seed` for determinism.

`bimodal_min_correlated_features: int = 3` in `NumericImputationConfig`. User-declared grouping variables go in `NumericImputationConfig.bimodal_grouping_variables: dict[str, str]` and take priority over auto-detection.

## NonlinearityTag.Linear override for bimodal columns

When `NonlinearityTag.Linear` is detected for a bimodal column, the Regression estimator is overridden to `RandomForestRegressor` regardless of the tag. A linear model (BayesianRidge) fits a linear function of the features and produces near-mean predictions for ambiguous rows — effectively a valley fill for bimodal targets. Tree-based models learn cluster-specific predictions from feature splits and are the correct choice regardless of apparent feature-target linearity. `NonlinearityTag.MonotonicNonlinear` and `ComplexNonlinear` already map to tree-based estimators and are unaffected.

## GMM Sampling overrides the Unpredictable guard

`NonlinearityTag.Unpredictable` is an unconditional pre-routing guard for non-bimodal columns (ADR-0016): no model achieves R² > 0.1, so model-based imputation is indistinguishable from a scalar fill. For bimodal columns in branch 4, the guard is overridden: GMM Sampling fills missing values from the distribution shape rather than predicting from features. It makes no R² claim and is not subject to the R² guard. Median (the Unpredictable fallback) lands in the valley between the two peaks; GMM Sampling preserves the bimodal marginal distribution in the imputed output. The override applies exclusively to the bimodal + no-correlated-features combination. The Unpredictable guard remains unconditional for all non-bimodal columns.

## Bimodal discrete

For `NumericKind.BoundedDiscrete` columns, model-based strategies produce continuous predictions that fall outside the finite domain (ADR-0018). The Bimodal Imputation Framework applies, but the fill value is always replaced with the **cluster mode** — the most frequent valid domain value within the assigned cluster. Domain-constrained GMM Sampling is the no-features fallback: samples are drawn from the 2-component GMM and snapped to the nearest valid discrete value, preserving the bimodal shape while keeping fills within the domain.

## The rejected alternative

The rejected design routed bimodal columns directly into the existing distribution shape escalation at Priority 7 (KNN → Regression → Median) without a dedicated framework. This is insufficient because: (1) KNN averages neighbor values and risks valley fills when neighbors span both clusters; (2) the existing escalation has no concept of grouping variables or cluster-conditional fills; (3) the Unpredictable guard would send bimodal + no-feature columns to Median even though GMM Sampling is available and strictly better.

## random_seed

GMM Sampling introduces stochasticity. `PipelineConfig.random_seed: Optional[int]` is the single seed for all stochastic operations across the pipeline. `None` produces non-deterministic output. All phases that introduce randomness read from this field — it is not scoped per-phase.
