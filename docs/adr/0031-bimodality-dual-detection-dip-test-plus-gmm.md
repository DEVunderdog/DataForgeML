# ADR 0031: Bimodality is detected with Hartigan's Dip Test for the signal and 2-component GMM for peak locations

Bimodality detection uses two methods together: Hartigan's Dip Test confirms whether a column's distribution departs from unimodality; a 2-component Gaussian Mixture Model is fitted when the dip test fires to extract the two cluster centers.

## The alternatives

**Dip test alone:** Non-parametric and robust to any peak shape — it makes zero assumptions about whether the peaks are Gaussian, skewed, or sharp. Returns a p-value only. Requires the `diptest` package (not in scipy or sklearn). Gives no peak locations.

**GMM alone (BIC comparison):** Fits 1-component and 2-component Gaussian Mixture Models and declares bimodal when the 2-component BIC improvement exceeds a threshold. Returns peak locations. Uses only sklearn (already a dependency). Fails on non-Gaussian peaks: a heavily right-skewed unimodal distribution may yield a large BIC improvement for the 2-component fit (false positive); sharp non-Gaussian spikes may not yield enough BIC improvement to cross the threshold (false negative).

**KDE with valley detection:** Non-parametric and bandwidth-sensitive. Does not return peak locations cleanly. Fragile against bandwidth choice.

## Why both

Neither method alone provides the complete output Phase 2 requires:

- The dip test gives a reliable, shape-agnostic detection signal. GMM's Gaussian assumption makes it unsuitable as the sole detection method when peak shapes are unknown — the exact case that motivated departing from GMM-alone.
- The GMM gives `center1` and `center2` — the two cluster means required by the Bimodal Imputation Framework for cluster-conditional imputation and domain-constrained GMM Sampling. The dip test provides no location information.

Under the library's accuracy-over-speed principle, both computations run. The dip test owns the routing decision: `NumericFlag.Bimodal` is set when `p < bimodal_dip_p_value_threshold`. The GMM runs only when the dip test fires; its centers are stored in `BimodalStats`. A column that does not pass the dip test incurs no GMM cost.

## Detection threshold

`bimodal_dip_p_value_threshold: float = 0.05` in `NumericProfileConfig`. The standard α = 0.05 provides high power for clearly bimodal distributions. At this threshold a false negative (missed bimodality) produces a worse imputation outcome — valley fill — than a false positive (unnecessary escalation). The dip test's strong power at 0.05 makes false negatives rare for genuinely bimodal columns. Users working with domains where bimodality is common and subtle (e.g. medical imaging features) may lower the threshold.

## Scope

Both computations live in Phase 1 (`NumericProfiler`). The dip test result and GMM centers are stored in `NumericStats` — in `NumericFlag.Bimodal` and `NumericStats.bimodal_stats` respectively — and consumed by Phase 2 routing without re-computation.
