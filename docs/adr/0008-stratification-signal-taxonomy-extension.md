# ADR 0008: Stratification signal taxonomy extension

**Status:** Accepted
**Extends:** ADR 0007

ADR 0007 established the profile-stratified split and its original three signal families. This ADR records five targeted extensions made after auditing the full profile output against what was actually wired into the label matrix.

## Decisions

### 1. Kurtosis extends the extreme value threshold, independently of skewness

The extreme value signal uses `p99` (not `p95`) when `SkewSeverity.Severe` OR `KurtosisTag.Leptokurtic`. Either condition alone is sufficient — they are not ANDed. A symmetric but leptokurtic column (excess kurtosis > 3, skewness ≈ 0) has heavier-than-normal tails that p95 under-captures just as severely as a skewed column does. Requiring both conditions would miss this case entirely.

### 2. DropCandidate columns are excluded from the missingness signal

Columns with `MissingnessFlag.DropCandidate` (>50% missing) are dropped by Phase 2 before any downstream phase sees them. Including their missingness as a stratification signal consumes a label slot from the 50-label budget while protecting a column that will not exist after imputation. The slot is better spent on signals that protect live downstream inputs.

### 3. NearConstant numeric minority signal added

`NumericFlag.NearConstant` (mode frequency > 90%) columns can have their rare non-mode rows cluster in one split. The existing extreme value signal does not catch this for symmetric near-constant distributions — if 98% of values are `5.0`, p5 and p95 are both `~5.0`, so no rows qualify as extreme. A dedicated signal is added: one label per NearConstant numeric column, row gets `1` if its value differs from the mode. Minority definition is exact equality for `NumericKind.Discrete` columns; `|value - mode| > 0.5 * IQR` for `NumericKind.Continuous`.

### 4. Datetime future-date signal added

Datetime columns previously produced no stratification signals at all. A column with `DatetimeFlag.FutureDates` has rows that will produce out-of-distribution encoded values in Phase 5 (temporal feature extraction: year, month, day_of_week). If future-dated rows cluster in the test split, the model never sees those encoded values during training. One label per such column; row gets `1` if value > current timestamp at split time.

### 5. Compound missingness row signal added

The per-column missingness signals (one label per missing column) balance each column's missing rate independently. They do not protect rows that are simultaneously missing across many columns — the joint sparsity structure that MICE depends on. A compound signal is added: one label for the whole dataset when `MissingnessProfileResult.row_missingness_p90 > 0`; row gets `1` if its per-row missing column count exceeds `row_missingness_p90`. This fires only when compound missingness is meaningfully concentrated (p90 = 0 means almost no rows are multiply missing).

## Required profile changes

These decisions require two new fields to be persisted in the profile:

- `NumericStats.numeric_kind: Optional[NumericKind]` — the profiler already computes discrete/continuous internally but discards it. Persisting it ensures the NearConstant minority signal uses the same definition as imputation strategy routing (ADR 0005), preventing drift between phases. `NumericKind` lives in `models/_data_types.py` because both Phase 1 and Phase 2 consume it.
- `MissingnessProfileResult.row_missingness_p75: Optional[float]` and `row_missingness_p90: Optional[float]` — the 75th and 90th percentile of per-row missing column count. Computed during Phase 1 and stored so the compound signal threshold is profile-driven rather than re-derived from the raw DataFrame at split time.
