# ADR 0033: Tail asymmetry ratio and outlier density extend the universal distribution shape escalation from ADR-0017

ADR-0017 established that `KurtosisTag` and `SkewSeverity` are universal routing signals that escalate scalar fills to KNN at every severity level and on both MCAR and MAR paths. Its scope boundary explicitly deferred `tail asymmetry ratio` and `outlier density` as future signals requiring new Phase 1 computation. This ADR resolves that deferral.

## Tail asymmetry ratio

`tail_asymmetry_ratio = (p99 - p95) / (p5 - p1)` is derived from the existing `PercentileSnapshot` — all four values are already computed in Phase 1. No new raw computation is required. It is stored as `NumericStats.tail_asymmetry_ratio` (consistent with `mean_median_ratio`) and classified into `TailAsymmetryTag`:

- `RightHeavy` — right extreme tail is significantly heavier than left (ratio > `tail_asymmetry_right_threshold`)
- `LeftHeavy` — left extreme tail is significantly heavier than right (ratio < `tail_asymmetry_left_threshold`)
- `Symmetric` — tails are balanced

`TailAsymmetryTag.RightHeavy` or `LeftHeavy` **upgrades the effective `SkewSeverity` by one level** for routing purposes only — the stored `SkewSeverity` on `NumericStats` is unchanged. The upgrade closes the gap that `SkewSeverity` alone leaves: a column with `SkewSeverity.Moderate` and a disproportionately heavy extreme tail receives the same KNN escalation as `SkewSeverity.Severe`. `SkewSeverity.Severe` with a heavy tail is already at the top level — no further upgrade.

Division-by-zero (when `p5 == p1`, a flat left tail) is handled at compute time; `tail_asymmetry_ratio` is stored as `None` and `TailAsymmetryTag` is not set for that column.

Thresholds in `NumericProfileConfig`: `tail_asymmetry_right_threshold: float`, `tail_asymmetry_left_threshold: float`.

## Outlier density

`outlier_density = count(|x - mean| > outlier_sigma_threshold × std) / n_rows` requires a new pass over the column's values to count values beyond the σ band. This is a cheap filter-and-count, not a model fit. It is stored as `NumericStats.outlier_density`. `NumericFlag.HighOutlierDensity` fires when outlier density exceeds `high_outlier_density_threshold`.

`KurtosisTag.Leptokurtic` captures heavy tails through the fourth moment — driven by the magnitude of extreme values. `HighOutlierDensity` captures the same phenomenon as a directly interpretable fraction — driven by the count of extreme values. A column can be Leptokurtic from a handful of very extreme values (low density, high magnitude) or exhibit high outlier density without extreme kurtosis (many moderate outliers). They are independent signals and both warrant escalation independently.

`NumericFlag.HighOutlierDensity` is an independent trigger in the Priority 7 distribution shape escalation condition, alongside `KurtosisTag.Leptokurtic`, `SkewSeverity.Severe`, and `NumericFlag.Bimodal`.

Thresholds in `NumericProfileConfig`: `outlier_sigma_threshold: float = 3.0`, `high_outlier_density_threshold: float = 0.05`. For a normal distribution, ~0.27% of values fall beyond 3σ; the 5% default means the column has roughly 18× more extreme values than normal.

## The rejected alternative

The rejected design treated both signals as refinements of existing signals — tail asymmetry as a kurtosis sub-signal, outlier density as a kurtosis proxy — and proposed consulting them only when kurtosis or skewness was already present. This understates their independent information content: a `KurtosisTag.Mesokurtic` column can have high outlier density; a `SkewSeverity.Normal` column can have `TailAsymmetryTag.RightHeavy`. Under the accuracy-over-speed principle, independent signals must be evaluated independently.
