# ADR 0017: Distribution shape (KurtosisTag + SkewSeverity) is a universal routing signal across all missingness paths

Distribution shape signals — `KurtosisTag` and `SkewSeverity` — act as escalation signals at every severity level and for both MCAR and MAR paths, not only for specific cases like MCAR Moderate.

## The routing implication

Wherever the routing would produce a scalar fill (Mean or Median), distribution shape is consulted first:

- `KurtosisTag.Leptokurtic` OR `SkewSeverity.Severe` → escalate to KNN (under size guards) before falling back to Median. Applies at Minor and Moderate severity, on both MCAR and MAR paths.
- `KurtosisTag.Platykurtic` → de-escalate: scalar fills are more representative because the distribution has thin tails. Can favour Mean over Median even at Moderate skew.
- `KurtosisTag.Mesokurtic` → neutral: current severity-based routing is unchanged.

`NumericFlag.NearConstant` overrides distribution shape escalation entirely — when 90%+ of values share the mode, model-based imputation learns near-constant predictions regardless of tail shape, so the escalation is wasted.

## The rejected alternative

The rejected design applied distribution shape only to MCAR Moderate — treating it as a narrow patch for one underserved path. The argument was that Minor severity (< 1% missing) does not warrant model-based imputation regardless of tail shape, and that MAR paths already have their own model-based routing.

## Why the universal principle is correct

The cost of a poor scalar fill is proportional to the severity of the distribution anomaly, not the severity of the missingness. A column with extreme fat tails (`Leptokurtic`) suffers from a systematically biased median fill whether 1% or 5% of its values are missing. At 1% missing the aggregate effect is small, but under the accuracy-over-speed principle the library should use the best available strategy unconditionally — user config (`per_column_strategy`) is the escape hatch for cases where the model-based cost is not justified.

Keeping distribution shape as a case-by-case patch would require re-evaluating the same gap each time a new severity / mechanism combination was introduced. A single universal principle — "check shape before producing a scalar fill" — is easier to reason about, easier to test, and closed to partial coverage gaps.

## Scope boundary

Kurtosis and skewness are the only distribution shape signals in this decision. Bimodality, tail asymmetry ratio, and outlier density are deferred to a future scope because they require new Phase 1 computation. When those signals are added, they inherit the same universal escalation principle established here.
