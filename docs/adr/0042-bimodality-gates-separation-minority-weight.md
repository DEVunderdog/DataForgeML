# ADR 0042: Bimodality detection requires separation and minority weight gates

Bimodality detection relies on three compound gates evaluated with an AND composition: Hartigan's Dip Test p-value, Ashman's D for component separation, and minimum component weight.

## The problem

The `diptest` package's default table-interpolation p-value method saturates and returns exactly `0.0` once the scaled dip statistic exceeds the tabulated ceiling (a ~1e-5 resolution floor). This saturation is expected mathematical behavior, not a defect. However, this alone caused too many columns to be flagged.

In a worked example on `life_expectancy.csv` (~2900 rows, 19 numeric columns), 14 out of 19 columns were flagged as bimodal under the old single-gate logic (p-value only).

## The new gates

To correct this over-sensitivity while accepting the p-value saturation, we added two new gates to the bimodality check. A column is now flagged as `NumericFlag.Bimodal` only if all three gates pass:

1. **Dip-test p-value** (`p < bimodal_dip_p_value_threshold`): The original non-parametric test.
2. **Component separation** via Ashman's D (`cluster_separation > bimodal_separation_threshold`): The default threshold of `2.0` represents the standard convention for "visibly separated" distributions.
3. **Minimum component weight** (`minority_weight > bimodal_minority_weight_threshold`): The default threshold of `0.05` guards against single-outlier or few-outlier components being mistaken for true bimodality.

Applying these two new gates at their default thresholds to the `life_expectancy.csv` dataset reduces the flagged columns from 14 down to 2 (`Life expectancy`, `BMI`), both of which have plausible real-world explanations for being bimodal.

## The alternatives considered

- **Switching to `boot_pval=True` (bootstrap p-values):** Rejected. The p-values still saturate at these sample sizes (~2900 rows), and bootstrap introduces significant extra compute cost.
- **Patching the p-value threshold alone:** Rejected. Adjusting the p-value threshold does not address the fundamental resolution ceiling of the dip test tables.

## Future scope

True multi-modality (3+ peaks) detection is explicitly deferred as future scope. The current architecture expects at most a 2-component GMM for imputation logic (using `center1`, `center2`).
