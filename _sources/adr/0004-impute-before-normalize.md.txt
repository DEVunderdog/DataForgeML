# ADR 0004: Impute missing values before normalizing distributions

**Status:** Accepted

## Context

Phase 2 (Imputation) runs before Phase 4 (Normalization) in the pipeline. A reasonable question arises: when a numeric column is heavily skewed, should we normalize its distribution first to get a better central estimate, and then impute?

## Decision

Imputation always runs on the raw, pre-normalization data. The pipeline phase order (Imputation → Outlier Detection → Normalization) is fixed.

## Rationale

- Normalization transforms (Log, Box-Cox) are undefined on null values. There is no viable "normalize-first" path — the column must be complete before any transform can operate on it.
- The skewness signal from Phase 1 (`SkewSeverity`) is computed on the raw distribution and is the correct signal for choosing imputation strategy. On a right-skewed column, Mean is pulled up by outliers; Median is the appropriate fill regardless of what normalization will do later.
- Imputed values are transformed along with the rest of the column during normalization. A Median fill on a skewed column lands in a reasonable position in the normalized space. A Mean fill (inflated by outliers) remains an outlier after normalization — the phase ordering does not rescue a bad fill choice.

## Consequences

- Imputation strategy selection uses `SkewSeverity` from `StructuralProfileResult` to choose Mean vs Median on the raw distribution. `SkewSeverity >= Moderate` routes to Median.
- The pipeline phase order (Profiling → Imputation → Outlier Detection → Normalization → Encoding → Scaling) is a hard constraint, not a configurable preference.
