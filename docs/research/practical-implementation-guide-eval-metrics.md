# Practical implementation guide: six evaluation metrics

An **implementation guide**, not a literature review. The theory for these six metrics is already
settled in `docs/learning/designing-the-evaluation-and-tuning-layers.md` (sections A5–A7, E1–E6) and
the three sibling research docs. This file answers one question per metric: **how do you actually write
the Python code**, including exact imports, exact install commands, and where the gaps are.

Every version number and function signature below was re-verified against the live source on
**2026-08-24** — PyPI JSON, an installed wheel inspected with `inspect.signature`/`help()`, a CRAN
manual PDF, a vignette source file, or the paper itself. Where a claim could not be re-verified, that is
stated plainly rather than carried over from the design doc.

---

## 1. C2ST (Classifier Two-Sample Test)

### What it computes

Train a binary classifier to distinguish two samples (real rows vs. imputed rows, or fully-observed
rows vs. imputed-touched rows). If the classifier can't do better than chance on held-out data, the two
samples are indistinguishable — the imputation looks like real data. Lopez-Paz & Oquab's contribution is
the **closed-form null**: under `H0: P=Q`, `n_te·t̂ ~ Binomial(n_te, 1/2)`, so for large `n_te` the test
accuracy `t̂ ~ N(1/2, 1/(4·n_te))`. That gives a z-score with no p-value floor and no permutation test.

### Implementation path

Re-verified directly from the paper (arXiv:1610.06545v4, HTML rendering fetched 2026-08-24), §3:

> "First, construct the dataset `D = {(xi,0)}∪{(yi,1)}`. Second, shuffle `D` at random, and split it
> into the disjoint training and testing subsets `Dtr` and `Dte`. Third, train a binary classifier
> `f: X → [0,1]` on `Dtr`... Fourth, return the classification accuracy on `Dte`."
>
> "Under the null hypothesis `H0: P=Q`... `nte·t̂` follows a `Binomial(nte, p=1/2)` distribution.
> Therefore, for large `nte`, we can use the central limit theorem to approximate the null distribution
> ... by `N(1/2, 1/(4nte))`."

The paper studies exactly two classifiers in its own experiments (§4.1): **C2ST-NN** (one hidden layer,
20 ReLU units, Adam, 100 epochs) and **C2ST-KNN** (`k = ⌊n_tr^(1/2)⌋` nearest neighbours). It does not
declare either "the" default — it presents both as equally valid instantiations of the same test. **The
paper gives no recommendation for the "simplest" classifier**; that is a gap in the source, not
something omitted here.

For DataForgeML, the practical choice is a **tree ensemble** — `sklearn.ensemble.
HistGradientBoostingClassifier` or `RandomForestClassifier` — over k-NN or a hand-rolled NN, for a reason
the design doc's own A5 flags: Snoke et al.'s under-specified-propensity-model failure mode (a main-effects-only
model is blind to dependence; the standardized pMSE stayed near 0 for a broken synthesis until
first-order interactions were added). Tree ensembles capture interactions with no feature-engineering
step, need no scaling, and `HistGradientBoostingClassifier` accepts native categorical columns by
default — verified via `inspect.signature` on the pinned scikit-learn **1.9.0**
(`pyproject.toml: scikit-learn>=1.9,<1.10`): `categorical_features='from_dtype'` is the constructor's
default, so a Polars/pandas frame with proper categorical dtypes needs no manual encoding. k-NN is worth
keeping as a lightweight alternative because it is what the paper itself validated the null against.

**CV setup.** The paper's own algorithm is a single stratified train/test split — not k-fold — because
`n_te` enters the null variance directly and must be a fixed, known number at the moment the z-score is
computed. Two legitimate options:
1. **Single stratified split** (`sklearn.model_selection.train_test_split(..., stratify=y)`), matching
   the paper's Algorithm 1 exactly. `n_te` = size of the held-out set.
2. **Out-of-fold accuracy via `cross_val_predict`** (`StratifiedKFold`, `method="predict"`), so every row
   contributes one out-of-sample prediction and `n_te = n` (the whole dataset). This uses more of the
   data per accuracy estimate at the cost of `k` classifier fits instead of one — a reasonable trade for
   a diagnostic that only runs occasionally, not inside an inner loop.

Either is a legitimate reading of "accuracy on held-out data"; the paper does not adjudicate between
them, so this is an implementation choice DataForgeML has to make, not something to source further.

### Python recipe

```
pip install scikit-learn scipy   # already DataForgeML dependencies
```

```python
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def c2st(X_real: np.ndarray, X_imputed: np.ndarray, random_state: int = 0) -> dict:
    X = np.vstack([X_real, X_imputed])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_imputed))])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=random_state
    )
    clf = HistGradientBoostingClassifier(random_state=random_state).fit(X_tr, y_tr)
    n_te = len(y_te)
    acc = clf.score(X_te, y_te)
    z = (acc - 0.5) / np.sqrt(1 / (4 * n_te))
    p_two_sided = 2 * (1 - norm.cdf(abs(z)))
    return {"accuracy": acc, "n_te": n_te, "z": z, "p_value": p_two_sided}
```

### Edge cases (design decisions, not literature facts)

- **Small `n_te`.** The `N(1/2, 1/(4n_te))` approximation is a CLT limit on a Binomial and degrades below
  roughly `n_te ≈ 30–50` per standard CLT practice. For small held-out sets, use the exact test instead:
  `scipy.stats.binomtest(k=int(acc*n_te), n=n_te, p=0.5)` — exact, no approximation, same null.
- **Unequal real/imputed row counts.** The paper's own construction assembles `D` from **equal-sized**
  groups (`{(xi,0)}_{i=1}^n ∪ {(yi,1)}_{i=1}^n`, same `n` for both). If DataForgeML's two groups are
  unequal size (e.g. comparing all rows against only the subset with ≥1 imputed cell), the
  `Binomial(n_te, 1/2)` null silently stops being correct — a classifier that always predicts the
  majority class would score above 0.5 even under the true null. **This is not resolved by the paper**;
  the two defensible fixes are (a) subsample the larger group to match before assembling `D` (loses
  data, keeps the null exact), or (b) train and evaluate on the full imbalanced set but replace raw
  accuracy with **balanced accuracy** and derive its own null empirically (baseline-ratio style, §2
  below) rather than reusing `N(1/2, 1/(4n_te))`, which no longer applies. State this as an open
  implementation choice, not a verified formula.
- **What the two groups actually are.** The paper is agnostic; DataForgeML has to decide. The
  ground-truth-free default (per A2/A5) is: label 1 = rows containing at least one imputed cell, label 0
  = rows that were fully observed before imputation, both read off the *same* completed table. This is
  the row-level analogue of Snoke's pMSE construction and is what makes §5's shared-infrastructure
  argument (closing section) work.
- **Memorisation/copying.** A discriminator at chance is also achieved by an imputer that copies
  observed rows verbatim (SDMetrics' own documented warning, reused per A5). Any C2ST-based objective
  needs a companion nearest-neighbour-distance check; that check is not part of C2ST itself.

---

## 2. Energy Distance (joint, multivariate)

### What it computes

The two-sample energy distance between the real-row distribution and the imputed-row distribution,
computed jointly over all active numeric columns at once — not per column. Näf, Scornet & Josse use it
as their primary distributional criterion because its empirical sample complexity is `O(n^-1/2)`
regardless of dimension `d`, unlike joint Wasserstein's `O(n^-1/d)`, and it needs no bandwidth/kernel
choice.

### Implementation path — verified against the installed wheels, 2026-08-24

```
pip install dcor hyppo
```

Both packages are unchanged from the design doc's prior verification:

| Package | PyPI version (`pip index versions`, 2026-08-24) |
|---|---|
| `dcor` | **0.7** |
| `hyppo` | **0.5.2** |

**`dcor.energy_distance` — exact signature, read via `inspect.signature`:**

```python
dcor.energy_distance(
    x, y, *,
    average: Callable[[Array], Array] | None = None,
    exponent: float = 1,
    estimation_stat: EstimationStatisticLike = EstimationStatistic.V_STATISTIC,
) -> Array
```

`x`/`y` are `(n, p)` arrays — rows are instances, columns are variables, multivariate by construction.
Verified from the wheel's own source (`dcor/_energy.py`, unpacked): `energy_distance` internally calls
`distances.pairwise_distances(x, exponent=exponent)` for all three of the `xx`, `yy`, `xy` distance
matrices — **always Euclidean** (optionally raised to a power via `exponent`). **There is no public hook
in `dcor.energy_distance` for a precomputed or custom (e.g. Gower) distance matrix** — the function does
not accept one, and the private helper that does (`_energy_distance_from_distance_matrices`) is not
exported from the `dcor` top-level namespace. This narrows the design doc's claim: `dcor` handles numeric
columns natively but has **no built-in mixed-type path**.

`dcor.homogeneity.energy_test(*args, num_resamples=0, exponent=1, random_state=None, average=None,
estimation_stat=..., n_jobs=1)` gives a permutation p-value; `num_resamples` defaults to **0** (no
p-value unless you set it explicitly).

**`hyppo.ksample.Energy` — exact signature:**

```python
Energy(compute_distance="euclidean", bias=False, **kwargs)
Energy.statistic(x, y)
Energy.test(x, y, reps=1000, workers=1, auto=True, random_state=None)
```

Verified from the class docstring (`hyppo` 0.5.2, installed wheel): `compute_distance` accepts the usual
sklearn/scipy metric strings, a callable, **or `None`/`"precomputed"` "if x and y are already distance
matrices."** This is the confirmed hook for a Gower-style mixed-type distance — compute your own
mixed-type distance matrices for `x` and `y` first, then pass them straight through. `hyppo` has this
capability; `dcor` does not.

### Baseline-ratio standardization (mandatory — A7 rules out raw distances)

Per the design doc's A7, a raw energy distance is meaningless without a null to compare against, and
energy distance has no closed-form null (only pMSE and C2ST do). The literature's answer (Shadbahr et
al., Methods §C) is an **empirical baseline**: compute the same distance between two random halves of the
*real* data, and report the ratio. `1.0` means "as different as real data is from itself."

```python
import numpy as np
import dcor

def energy_distance_ratio(
    real: np.ndarray, imputed: np.ndarray, n_splits: int = 10, random_state: int = 0
) -> dict:
    rng = np.random.default_rng(random_state)
    d_obs = dcor.energy_distance(real, imputed)

    baseline = []
    n = len(real)
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        baseline.append(dcor.energy_distance(real[idx[:half]], real[idx[half:]]))
    baseline = np.array(baseline)

    return {
        "energy_distance": d_obs,
        "baseline_mean": baseline.mean(),
        "baseline_sd": baseline.std(ddof=1),
        "ratio": d_obs / baseline.mean(),
    }
```

`n_splits=10` matches Shadbahr et al.'s own `P=10`; this is roughly 10x the compute of the raw distance,
which A7 already names as the cost of this standardization move.

### Unverified / gaps

- No package on PyPI computes a **Gower distance matrix** ready-made for `hyppo`'s `compute_distance="precomputed"`
  hook — that matrix has to be hand-rolled (numeric columns: normalized Manhattan; categorical: simple
  mismatch indicator; combine with per-column weights). This is a real build item, not a one-line install.
- `dcor`'s lack of a mixed-type path means the "one line" energy distance call (`dcor.energy_distance(A,
  B)`) is numeric-only in practice; mixed-type tables need either (a) one-hot/dummy-encode categoricals
  first and accept that this changes the effective metric on those columns, matching the CRAN `Iscores`
  package's own approach (§4 below — "mixed-type data (here the score is calculated on dummy variables)"),
  or (b) switch to `hyppo.ksample.Energy(compute_distance="precomputed")` with a hand-built Gower matrix.

---

## 3. Per-column W1 (1-D Wasserstein / earth mover's distance)

### What it computes

The integrated CDF gap `∫|U−V|` between the observed and imputed marginal distribution of one column, in
the column's own units. Diagnostic only — per A6/A7, a per-column score is provably gameable (an imputer
that samples each column's own observed marginal, ignoring every other column, scores perfectly here
while destroying joint structure) and must never be the sole or primary tuning objective.

### Implementation path — verified against the installed scipy wheel, 2026-08-24

```
pip install scipy   # already a DataForgeML dependency
```

Confirmed via `inspect.signature` on scipy **1.18.1** (current PyPI release as of 2026-08-24, one patch
ahead of the design doc's 1.18.0):

```python
scipy.stats.wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None)
```

Docstring, verbatim: *"Compute the Wasserstein-1 distance between two 1D discrete distributions."* No `p`
parameter — **this is the correct function**, and it is Wasserstein-1 only, matching the design doc's
claim. `scipy.stats.wasserstein_distance_nd(u_values, v_values, u_weights=None, v_weights=None)` exists
as a separate function for the joint case and is explicitly the wrong tool here (per A6, joint Wasserstein
hits the `O(n^-1/d)` sample-complexity wall; use energy distance for the joint score, §2 above).

### Python recipe

```python
import numpy as np
from scipy.stats import wasserstein_distance

def per_column_w1(observed: dict[str, np.ndarray], imputed: dict[str, np.ndarray]) -> dict[str, float]:
    """observed/imputed: column name -> 1D array of that column's values."""
    return {
        col: wasserstein_distance(observed[col], imputed[col])
        for col in observed
        if col in imputed
    }
```

### Aggregation guidance — keep it diagnostic, never an objective

Two rules, both sourced from the design doc's A6/A7 and the underlying papers, not invented here:

1. **Report per-column, never collapse to a single number that feeds the tuner.** A6 is explicit: pairwise
   and marginal-only scores are `O(k²)` or gameable respectively, and the whole reason energy distance
   (§2) exists as the *scored* object is so nothing downstream needs to average W1 across columns to get
   a tuning signal.
2. **If a summary table is wanted for a human**, report each column's W1 next to a **baseline** — the W1
   between two random halves of the column's own *observed* values, same idea as §2's ratio — so a reader
   can see "column X's W1 is 3x its own noise floor" instead of a bare unitless number. Do not average the
   ratios into one scalar; that reintroduces the SDMetrics gotcha where an aggregate average is swamped by
   many near-zero, uninformative columns (design doc A6, citing SDMetrics' own 0.27.0 "discard weak
   pairs" fix for the analogous pairwise case). A per-column table, sorted by ratio descending, is the
   right shape for a diagnostic report — not a mean.
3. **W1 is in the column's own units**, so cross-column comparison on the raw statistic is already
   meaningless without a per-column scale correction (e.g. divide by the column's observed IQR or SD)
   before sorting a mixed-unit table.

### Unverified / gaps

None — this metric's Python path is fully settled and simple. The only design risk is social, not
technical: a per-column W1 table is exactly the kind of number a future contributor will be tempted to
`np.mean()` into a tuning objective. The guard is procedural (code review / a docstring warning on the
function), not something scipy can enforce.

---

## 4. I-Score (`energy_IScore` and `DR_IScore`)

### What it computes

A no-ground-truth-required score: for each column with missing values, it repeatedly re-imputes the
*already-observed* cells of that column (masking them internally, using the imputer's own predictor set),
and scores the imputer with a **1-D energy-score-style discrepancy** — dispersion among repeated draws
minus their distance to the true (known) observed value — then aggregates across columns with a weight
proportional to how much of that column is missing.

### The exact formula — verified from the CRAN vignette source, not the (silent) manual

Re-fetched directly (`curl` to `https://CRAN.R-project.org/web/packages/Iscores/vignettes/About_IScore.Rmd`,
2026-08-24) — this is the "Energy-I-Score: Implementation Details" vignette, and it **does** carry the
formula the design doc flagged as absent from the CRAN manual. Quoted verbatim.

**Notation.** `X ∈ R^{n×p}` original data with missing values, `X̃` an imputed dataset, `I` the
imputation function, `N` the number of draws from `I`. For each variable `j` with missing values: `L_j` =
row indices where `X_{i,j}` is **observed**, `L_j^c` = row indices where it's missing, `O_j` = the set of
predictor columns fully observed on `L_j`. `S = {j : L_j^c ≠ ∅}` — the set of columns worth scoring.

**Algorithm, six steps, verbatim from the vignette:**

1. **Predictor selection.** `O_j = ⋂_{m∈L_j} {l : m_l = 0}` (fully-observed predictors over the rows
   where `j` is observed). If empty, fall back to the single variable `k* = argmax_{k≠j} |{i : row i ∈
   L_j ∩ L_k}|` — the predictor with the most overlap.
2. **Partition.** Build a synthetic train/test split *within the observed part of column `j`*: `Train` =
   the predictors `O_j` plus column `j` masked to `NA` on rows `L_j`, and column `j`'s truly-missing rows
   `L_j^c` with their own known predictors; `Test` = the true (known) values `X_{L_j, j}`.
3. **Multiple imputation.** Impute the masked training column `N` times with `I`:
   `X̃_{i,j}^{(1)}, …, X̃_{i,j}^{(N)} ~ H_{X_j | X_{O_j}, M_j=1}`.
4. **Per-row score.**
   ```
   Ŝ^j_NA(H,P) = (1/|L_j|) Σ_{i∈L_j} [
       (1/(2N²)) ΣΣ_{l,ℓ} |X̃_{i,j}^{(l)} − X̃_{i,j}^{(ℓ)}|
       − (1/N) Σ_l |X̃_{i,j}^{(l)} − x_{i,j}|
   ]
   ```
   First term = internal dispersion among the `N` draws (this is the 1-D pairwise-average-distance form
   of an energy statistic); second term = mean distance from each draw to the true observed value. This
   is structurally the 1-D energy score / CRPS: dispersion minus accuracy, exactly Gneiting & Raftery's
   proper-scoring-rule shape.
5. **Weighting.** `w_j = (1/n²) · |L_j| · |L_j^c|` — larger for columns with more missingness (and more
   observed rows to validate against).
6. **Final score.** `Ŝ_NA(H,P) = (1/|S|) Σ_{j∈S} w_j · Ŝ^j_NA(H,P)`.

Citation on the vignette: *"This approach follows the methodology proposed by Näf, Grzesiak, and Scornet
(2025) in 'How to rank imputation methods?' (arXiv:2507.11297)."* Matches the design doc's E1 pointer.

**Package facts, verified from the CRAN manual PDF (fetched 2026-08-24, `Iscores` 1.2.0, 2026-06-08):**

```
energy_IScore(X, imputation_func, X_imp = NULL, multiple = TRUE, N = 50,
              max_length = NULL, skip_if_needed = TRUE, scale = FALSE,
              n_cores = 1, silent = TRUE)
```

`N = 50` is confirmed as the default (not just "N=50" from the design doc's memory — read directly from
the manual's `Usage` block). `imputation_func` is a **function**, not a pre-imputed frame — confirming
the design doc's "the door takes an imputation function" observation; `X_imp` is an optional
pre-computed alternative. `DR_IScore(X, imputation_func=NULL, X_imp=NULL, m=5, n_proj=100,
n_trees_per_proj=5, min_node_size=10, n_cores=1, projection_function=NULL, ...)` — `Imports:` field lists
`energy, kernlab, pbapply, pbmcapply, ranger, scoringRules, stats`, confirming `ranger` (random forest)
and `scoringRules` as hard R dependencies for `DR_IScore` specifically (`energy_IScore` does not need
them). `edistance(X, X_imp, rescale=FALSE)` is the separate ground-truth *required* function (`"a
complete original dataset (without missing values)"`) — not the no-ground-truth path, confirmed
unchanged from the design doc.

License, verified from the CRAN package page: **GPL-3**. Maintainer: Krystyna Grzesiak
(`krygrz11@gmail.com`).

### (b) Is a Python reimplementation of `energy_IScore` realistic?

**Yes.** The formula above needs nothing but repeated calls to whatever Python imputer is under test plus
`numpy` for the pairwise-absolute-difference sums — no `dcor`/`scipy` special-purpose function is even
required, since the per-row term is a plain 1-D statistic. The real cost is not the math, it's the
**re-fit machinery**: for each column `j` with missing values, you need to (1) select a predictor subset
`O_j` from the *currently fitted* plan, (2) mask the observed part of column `j` on a subset of rows, (3)
re-run the imputer `N=50` times with different seeds, (4) restore. That is close to what DataForgeML's
`fit_unit` already knows how to do for a single unit, but the internal masking-and-restoring loop over
`L_j` is new work, not a reuse of existing plumbing.

```python
import numpy as np

def energy_iscore_column(
    x_true: np.ndarray,          # observed values of column j, shape (n_obs,)
    draws: np.ndarray,           # shape (n_obs, N) — N re-imputations of the masked observed cells
) -> float:
    n_obs, N = draws.shape
    # dispersion term: mean pairwise |draw_l - draw_l'| over l,l' (vectorised, all i at once)
    disp = np.abs(draws[:, :, None] - draws[:, None, :]).mean(axis=(1, 2)) / 2
    # accuracy term: mean |draw_l - x_true| over l
    acc = np.abs(draws - x_true[:, None]).mean(axis=1)
    return float((disp - acc).mean())

def energy_iscore(per_column_scores: dict[str, float], n_obs: dict[str, int],
                   n_mis: dict[str, int], n_total: int) -> float:
    weights = {j: (n_obs[j] * n_mis[j]) / n_total**2 for j in per_column_scores}
    num = sum(weights[j] * per_column_scores[j] for j in per_column_scores)
    return num / len(per_column_scores)
```

`disp.mean(axis=(1,2))` computes the full `N×N` sum including the `l=ℓ` diagonal (which is 0), matching
the vignette's un-excluded double sum exactly (`1/(2N²) ΣΣ_{l,ℓ}`, not a U-statistic that drops the
diagonal) — worth flagging as a detail future implementers could get subtly wrong by "fixing" it to an
unbiased form that no longer matches the R package's number.

### (c) Is `rpy2` a viable shortcut?

Re-verified 2026-08-24. **Requirements:**

- A system **R** runtime (rpy2 embeds R; it does not ship one). `pip install rpy2` alone is not
  sufficient.
- The `Iscores` R package installed inside that R environment (`install.packages("Iscores")` or
  `devtools::install_github(...)`), which itself pulls in `energy`, `kernlab`, `pbapply`, `pbmcapply`,
  `ranger`, `scoringRules` as transitive R dependencies for full functionality.
- `rpy2` itself: current PyPI version **3.6.7** (`pip index versions rpy2`, 2026-08-24),
  `requires-python: >=3.9`, **license `GPL-2.0-or-later`** (verified from the PyPI JSON `info.license`
  field).

**Licensing note, stated plainly rather than as legal advice.** DataForgeML is MIT-licensed
(`pyproject.toml`). `rpy2` is GPL-2.0-or-later; the `Iscores` R package it would call is GPL-3. Neither
of those licenses is automatically triggered onto DataForgeML merely by having them as an *optional*
runtime dependency that a user opts into (the standard "mere aggregation" argument), but bundling,
vendoring, or distributing DataForgeML together with a copy of GPL code — as opposed to an optional pip
extra that fetches it separately — is the kind of thing that should get an actual license review before
shipping, not a judgment call made in a research doc. Flagging the fact, not resolving it.

**Practically:** this is real friction for a Python library's install story — a working R toolchain is a
much heavier ask than any pure-Python dependency in this whole guide, and it is not something `uv` (the
project's own dependency/build tool, per `docs/research/uv-migration` decisions referenced in memory)
manages at all. `rpy2` is viable as a documented, fully-optional escape hatch for a user who already has
R installed and wants `DR_IScore` specifically (which is not reimplementable — see below) — not as
something DataForgeML installs by default or tests in CI without an R-provisioned runner.

### `DR_IScore` — confirmed not cheaply portable

Confirmed from the `Imports:` field above: `ranger` (Breiman-style random forest, C++-backed R package)
and `scoringRules` (proper scoring rules for probabilistic forecasts, including the specific KL/density-ratio
machinery `DR_IScore` needs). Neither has a drop-in Python equivalent that reproduces the exact
projections-and-forests construction described in Näf, Grzesiak & Scornet (2025), arXiv:2507.11297 (the
`DR_IScore` signature — `n_proj=100, n_trees_per_proj=5, min_node_size=10, projection_function=NULL` —
confirms it is a random-projection-plus-forest density-ratio estimator, a genuinely different and heavier
algorithm than `energy_IScore`'s repeated-draw energy statistic). Reimplementing it faithfully in Python
would mean re-deriving the projection scheme from the arXiv paper's equations directly — out of scope for
"practical" and not attempted here.

---

## 5. Propensity Conditional comparison (Bondarenko & Raghunathan)

### What it computes

Fits a model of `P(row/cell affected by missingness | completed predictors)`, stratifies rows by that
propensity, then tests — **within each stratum** — whether the observed and imputed values of a target
column come from the same distribution via a two-way ANOVA with an interaction term. The interaction is
the load-bearing part: under MAR, the *marginal* observed-vs-imputed difference is allowed to be nonzero,
but the difference *within a matched-propensity stratum* should not be, so testing the interaction (not
the main effect) is what makes the diagnostic legitimate under MAR rather than MCAR.

### The stratum count — re-attempted from scratch, still unresolved from the primary source

The design doc flagged Bondarenko & Raghunathan (2016), *Stat Med* 35(17):3007–3020, DOI 10.1002/sim.6926,
as unreadable everywhere it tried. Re-attempted here via five additional/different routes, all
2026-08-24:

1. **Semantic Scholar API** (`api.semanticscholar.org/graph/v1/paper/DOI:10.1002/sim.6926`) — this
   *did* surface a direct open-access bitstream URL not found before:
   `http://deepblue.lib.umich.edu/bitstream/2027.42/122409/1/sim6926_am.pdf` (the accepted-manuscript
   PDF, "openAccessPdf... status: GREEN"). Fetched directly with `curl` (bypassing WebFetch, in case the
   earlier failure was tool-specific): still **HTTP 403**, and the response headers show
   `cf-mitigated: challenge` — a live Cloudflare bot-challenge, not a dead link. Confirms the design doc's
   finding rather than just repeating it.
2. **SSRN precursor** (Raghunathan & Bondarenko 2007, "Diagnostics for Multiple Imputations",
   `papers.ssrn.com/sol3/papers.cfm?abstract_id=1031750`) — HTTP 403.
3. **"Diagnostics for Multiple Imputation in Stata"** (Stata Journal 12(3), SAGE,
   `journals.sagepub.com/doi/pdf/10.1177/1536867X1201200301`) — a Stata implementation that would plausibly
   state a stratum count if it reproduces B&R's procedure — HTTP 403.
4. **Citing papers that quote B&R's method in detail** — "Diagnosing missing always at random in
   multivariate data" (arXiv:1710.06891) and "Choosing Imputation Models" (arXiv:2107.05427) — both
   **cite** Bondarenko & Raghunathan (2016) in their reference lists but neither paper's fetched text
   states a stratum count in the passages retrieved.
5. **ResearchGate, NBER, Google/web search** for any secondary source stating a specific number
   (quintiles / deciles / any other count) tied specifically to B&R — none found.

**The stratum count remains unverified from the primary source.** This is not a re-statement of the
design doc's flag; it is a fresh attempt across five routes that reached the same wall, including one new
route (the Semantic Scholar-sourced direct bitstream link) that still failed for a documented, live reason
(a Cloudflare challenge page, confirmed via response headers) rather than a stale/dead URL.

### The fallback, verified against the actual Rosenbaum & Rubin (1984) paper

Per the task brief's instructed fallback: Rosenbaum & Rubin, "Reducing Bias in Observational Studies
Using Subclassification on the Propensity Score," *JASA* 79(387):516–524. Fetched and text-extracted
directly (PDF at `www2.stat-athens.aueb.gr/~jpan/Rosenbuam-JASA1984(516-524).pdf`, 2026-08-24) —
verbatim, from the paper's own §1.1:

> "Cochran (1968) shows that five subclasses are often sufficient to remove over 90% of the bias due to
> the subclassifying variable or covariate."

and, describing their own worked example:

> "we may expect approximately a 90% reduction in bias for each of the 74 variables when we subclassify
> at the quintiles of the distribution of the population propensity score. Consequently we subclassified
> at the quintiles of the distribution of the estimated propensity score."

This is Rosenbaum & Rubin's own citation of **Cochran (1968)**, not their own novel result, and it is
about causal treatment-effect bias reduction, not missing-data imputation diagnostics — a genuinely
different problem that happens to use the same propensity-subclassification mechanic. **Use 5 quantile
strata (quintiles) as a documented, sourced fallback for the propensity-conditional diagnostic — explicitly
labelled as the Rosenbaum & Rubin / Cochran general convention, not the Bondarenko & Raghunathan-specific
number**, which remains unknown. If B&R ever becomes readable, replace this default without hesitation;
until then this is the best-sourced number available.

### Python recipe

**Propensity model: `sklearn.linear_model.LogisticRegression`, not `statsmodels`.** DataForgeML's own
codebase already uses `sklearn` throughout for this kind of fit (`_regression_estimator_factory.py`
imports `BayesianRidge`, `RandomForestRegressor`, `GradientBoostingRegressor`; the splitting module uses
`sklearn.model_selection`) — and it has an explicit, checked-in precedent for *avoiding* `statsmodels`:
`profiling/_nonlinearity_profiler.py`'s Breusch–Pagan check is documented as "implemented with `scipy`
(no `statsmodels` dependency)." Fitting propensities only needs `predict_proba`, which
`LogisticRegression` gives directly with no extra dependency:

```python
from sklearn.linear_model import LogisticRegression

def fit_propensity(X_predictors, missingness_indicator):
    model = LogisticRegression(max_iter=1000).fit(X_predictors, missingness_indicator)
    return model.predict_proba(X_predictors)[:, 1]
```

**Stratification into quantile bins — pandas, standard idiom:**

```python
import pandas as pd

strata = pd.qcut(propensity_scores, q=5, labels=False, duplicates="drop")
```

**Two-way ANOVA with interaction — `statsmodels`, no substitute exists.** Verified from the statsmodels
docs (`statsmodels.org/stable/anova.html`, fetched 2026-08-24) and from `inspect.signature` on the
installed **statsmodels 0.14.6** wheel (current PyPI release, `pip index versions statsmodels`,
2026-08-24):

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

df["stratum"] = strata
df["status"] = np.where(is_imputed, "imputed", "observed")

model = ols("target_col ~ C(stratum) * C(status)", data=df).fit()
table = sm.stats.anova_lm(model, typ=2)
p_interaction = table.loc["C(stratum):C(status)", "PR(>F)"]
```

`anova_lm(*args, scale=None, test="F", typ="I"/"II"/"III" or 1/2/3, robust=None)` — confirmed exact
signature from the installed wheel's docstring. `typ=2` is the conventional default for a balanced-ish
factorial design; the design doc's own reading of the FIMD R code (`glm`+`xyplot`, no ANOVA call shown
there at all) does not pin down a `typ`, so `typ=2` here is a documented choice, not a re-verified one.

**A genuine dependency tension, stated plainly.** `statsmodels` is **not currently a DataForgeML
dependency** (`pyproject.toml` lists `scikit-learn`, `numpy`, `scipy`, `joblib`, `polars`, `pandas`,
`chardet`, `iterative-stratification`, `diptest`, `ruff` — no `statsmodels`), and the codebase has one
existing comment explicitly justifying *not* adding it for a similar-weight statistical test
(Breusch–Pagan). Two-way ANOVA with a clean, correctly-typed sum-of-squares partition is exactly the kind
of thing that's error-prone to hand-roll from scratch and well-tested in `statsmodels` — so this metric
either (a) breaks the no-`statsmodels` precedent and adds it (as an optional extra, mirroring how the
HPO doc already proposes `optuna` as a lazily-imported `[tuning]` extra, and how §4 above proposes
`rpy2`/R as optional for `DR_IScore`), or (b) hand-rolls a two-way ANOVA with interaction over `scipy`/`numpy`
primitives. This is a real decision the implementer has to make, flagged here rather than silently
resolved.

**Per target variable, or one shared row-level propensity?** The literature is not fully consistent on
this and it matters for the recipe above. FIMD's own worked R example (§6.6.2) fits **one shared
propensity** — `glm(ici(imp) ~ age + bmi + hyp + chl, family=binomial)`, where `ici()` is `mice`'s
"incomplete case indicator," i.e. *any* missingness in the row — and reuses it across different columns'
diagnostic plots. Nguyen, Carlin & Lee's description (which the design doc's E5 built the assembled
procedure from) instead reads as **one propensity model per target variable**, since MAR is a
per-variable concept (column `j`'s missingness depends on other observed columns) and mixing multiple
columns' missingness into one shared indicator blurs mechanisms that may differ across columns. This
guide recommends **per-column propensity** (a `missing_j ~ other_completed_columns` logistic fit, refit
for each column being diagnosed) as the methodologically cleaner reading, consistent with the design
doc's E5 procedure — but this is a judgment call resolving an ambiguity the sources themselves don't
close, not a verified fact.

---

## 6. PPC (Posterior Predictive Check, Nguyen, Carlin & Lee 2017)

### What it computes, from the primary source directly

PMC5569512 is fully open access; fetched directly, 2026-08-24 (*Emerging Themes in Epidemiology* 14:8).
Quoted verbatim from the "Posterior predictive checking" subsection:

> "This is a Bayesian model checking technique that involves simulating 'replicated' datasets from the
> proposed imputation model... We created 2000 replications and calculated means of the test quantities in
> the replicated and completed data."
>
> "The posterior predictive p-values can be estimated as the proportion of replications in which the
> estimate of the test quantity from the replicated data is larger than that estimated from the completed
> data."
>
> "The posterior predictive p-value for the regression coefficient for harsh discipline was 0.026; thus,
> in 2.6% of the 2000 replications, the estimate in the replicated dataset was larger than that obtained
> from the actual data, suggesting poor model fit."

**Their discrepancy statistic `T(data, θ)`** was a **logistic regression coefficient** from their actual
analysis model (the coefficient on "harsh discipline" predicting a binary conduct-problem outcome) — not
a generic marginal statistic. This matters: PPC is specifically checking whether a **named downstream
analysis** (a regression coefficient) is stable across the imputation model's own posterior, which is
exactly why it is the one diagnostic that catches uncongeniality (per the design doc's A3). Their
diagnosis, also verbatim:

> "These PPC results drew our attention to a problem with the proposed imputation model; the imputation
> model was incompatible with the logistic regression analysis, in the sense that the continuous version
> of the outcome variable (conduct) was included in the imputation model rather than the binary outcome
> that was used in the analysis."

Fixing that incompatibility brought PPP from 0.026 back to a non-extreme value — an uncongeniality that
no marginal-distribution check would have caught (design doc A3/A5).

### Practical Python tooling: plain Monte Carlo, no PPL required

**This is not an MCMC problem in the way `PyMC`/`arviz` usually imply.** Nguyen et al.'s "2000
replications" is 2000 independent redraws of the imputation model's posterior for the *already-missing*
cells — structurally identical to producing 2000 extra `sample_posterior=True`-style imputations, not
2000 MCMC steps of a single chain, and not a redraw of the *observed* cells (PPC here only replicates the
missing part; the observed cells are the fixed conditioning data in every replication). The loop is:

```python
def posterior_predictive_check(
    fit_imputation_replicate,   # callable: () -> one new completed frame, redrawn missing cells only
    compute_T,                  # callable: completed frame -> scalar test statistic
    T_completed: float,         # T computed on the actual analysis-ready (observed + imputed) frame
    n_replications: int = 2000,
) -> dict:
    T_replicated = [compute_T(fit_imputation_replicate()) for _ in range(n_replications)]
    T_replicated = np.array(T_replicated)
    ppp = float((T_replicated > T_completed).mean())
    return {"ppp": ppp, "T_completed": T_completed, "T_replicated": T_replicated}
```

No probabilistic-programming framework is needed **if** the imputer already supports repeated posterior
draws of the missing cells with `compute_T` being an ordinary Python function (e.g. `statsmodels`'s
`.fit().params["harsh_discipline"]` on a completed frame). The Monte Carlo loop itself is the entire
implementation once those two pieces exist.

### Why this is architecturally blocked in DataForgeML today, not just unbuilt

Two independent blockers, both already decided against in the design doc (A3, A4) — restated here as
concrete requirements this specific recipe needs and does not have:

1. **`compute_T` needs a user-supplied estimand `Q`.** The design doc's A3 explicitly rejected shipping a
   default estimand set (`(b) declare broad scope … (a) rejected`) — DataForgeML by design never
   presumes what downstream analysis a user will run, and PPC's entire value (catching uncongeniality) is
   *specifically* about testing fit against a **named** analysis. Without a user-supplied `Q`, there is no
   `compute_T` to write.
2. **`fit_imputation_replicate` needs repeated posterior draws, i.e. `m` frames.** The design doc's A4
   decided this is a real architectural change (`IterativeImputer.transform` cannot change row count, so
   the `m`-axis has to be an outer loop of `m` fits with distinct seeds and `sample_posterior=True`, "not
   yet built"). PPC additionally needs this outer loop run at a much larger count (2000, per Nguyen et
   al.) than the `m` used for Rubin's-rules pooling (typically 5–20 per FIMD's own rules) — a second,
   separate replication count, since PPC's 2000 is diagnostic sampling, not inferential `m`.

So: not "nobody has researched how to build PPC" — the algorithm above is fully specified and simple. It
is that **both of its two inputs are things DataForgeML has explicitly decided not to build yet**
(A3's opt-in `Q`, A4's `m`-frame outer loop). Building PPC without also building those two things first
is not possible; building it after they exist is a genuinely small addition — the Monte Carlo loop itself
is ~10 lines.

---

## Sequencing recommendation

**Build order: C2ST → joint energy distance → per-column W1 (diagnostic-only) → defer I-Score and PPC.**
This matches the design doc's A5 with one specific instrument substitution (C2ST in place of pMSE, since
E2 confirms nothing in Python implements pMSE and `SDMetrics`' `LogisticDetection` is a rescaled ROC-AUC,
not pMSE) and one addition below: whether to bundle the propensity-conditional comparison (§5) with the
C2ST work.

**Should the propensity-conditional comparison (§5) be bundled with C2ST/discriminator work?**
Partially, not wholesale. Both fit *a* classifier of `P(indicator | features)` and get calibrated
probabilities out — that shared utility (fit an indicator classifier, extract `predict_proba`, optionally
split train/test) is worth writing once and reusing. But they are not, on inspection, literally the same
fitted model the way the design doc's A5 implies for pMSE specifically (pMSE and van Buuren's FIMD
worked-example propensity are both row-level "any missingness" indicators — genuinely the same
construction). C2ST as scoped here (§1) is also row-level: `imputed-row vs fully-observed-row`. But §5's
propensity-conditional comparison, on the methodologically cleaner per-column reading recommended above,
needs a **separate model per target column** (`missing_j ~ other completed columns`), because MAR is
defined per variable and B&R's whole point is testing column-`j`-specific propensity-conditional
similarity. Bundling them into one implementation effort makes sense at the *utility-function* level
(one `fit_propensity_classifier(X, indicator) -> probabilities` helper shared by both), but building them
as a single feature would conflate two different indicators (row-level "touched by imputation" vs.
column-level "this specific value is missing") that happen to reuse the same three lines of
`LogisticRegression` code. Recommendation: implement C2ST first as scoped (§1) with its fitted classifier
built on a small, swappable internal helper; when §5 is built, have it call the *same* helper with a
different indicator and predictor set, rather than either duplicating the fit logic or forcing both
diagnostics to share one classifier instance. That gets the real, honest infrastructure win (one
well-tested "fit an indicator model, return probabilities" function) without overstating how much of the
two features is actually shared.

I-Score and PPC stay deferred for different reasons: I-Score's `energy_IScore` is a real, scoped,
reimplementable build (§4b) but a second, separate build after the discriminator and energy distance are
shipped and trusted — it needs its own masking-and-refit loop that nothing else in this list requires.
PPC is not a build-order question at all right now; it is blocked on two prerequisite architectural
decisions (`m`-frame support, opt-in estimand `Q`) that live outside the evaluation layer entirely, so it
cannot be scheduled relative to the other four until those land.
