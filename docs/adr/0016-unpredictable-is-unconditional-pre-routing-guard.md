# ADR 0016: NonlinearityTag.Unpredictable is an unconditional pre-routing guard

`NonlinearityTag.Unpredictable` short-circuits model-based imputation routing regardless of missingness mechanism (MCAR, MAR) and regardless of severity level. A column flagged Unpredictable always routes to Median.

## The alternative

The rejected alternative was: MAR `High` and MAR `Severe` always force a model-based attempt, with `Unpredictable` only acting as a guard on MCAR paths. The argument for this was that MAR confirms a structural missingness mechanism, and the library should always honour that signal by attempting model-based imputation.

## Why the guard wins unconditionally

`NonlinearityTag.Unpredictable` is defined by four independent statistical tests all agreeing that no model family achieves R² > 0.1 against the available numeric predictors. This is not a weak signal — it means both linear and tree-based models, applied to the full available numeric feature set, produce predictions indistinguishable from the column mean.

The MAR flag confirms that *something* drives the missingness. It does not confirm that the available *numeric* features at Phase 2 time can model the column's values. The cause of missingness may be a categorical column (not yet encoded), a latent variable not captured in the data, or a mechanism that generates missing values without leaving a predictable signal in the feature space. In all three cases, routing to KNN or Regression produces a model that performs at or below Median on the held-out evaluation, adds compute cost, and presents the user with false precision.

The correct response is to route to Median, record `NonlinearityTag.Unpredictable` and the original MAR flag in `ColumnImputationRecord.signals`, and let the user decide whether to supply additional features or override the strategy via `per_column_strategy`. Silently fitting a model with near-zero predictive power is worse than a transparent scalar fill with full audit visibility.

## Scope

This decision applies to the strategy routing phase only. The `NonlinearityTag` is still computed and stored in `NumericStats` for all columns — it remains visible in the Phase 1 profile output regardless of which imputation strategy is ultimately chosen.

## Amendment (Scope 16): GMM Sampling exception for bimodal columns

When `NumericFlag.Bimodal` is set and no correlated features are available (all `|r| < 0.2` against available numeric predictors), the column routes to GMM Sampling instead of Median. GMM Sampling fills missing values by drawing from the fitted 2-component GMM — it does not predict from features and is therefore not subject to the R² guard that defines the Unpredictable classification. Median remains the correct fallback for non-bimodal Unpredictable columns; the exception applies exclusively to the bimodal + no-correlated-features combination. See ADR-0032.
