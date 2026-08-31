# Distributional fidelity metrics for evaluating imputed datasets

A **resource guide**, not a literature review. The goal is to get from "Wasserstein distance sounds promising" to a defensible, tunable objective for judging whether an imputed dataset preserves the data's distribution, uncertainty and dependence structure.

Every URL below was fetched and the claim checked against the live page or the PDF. Quotations are verbatim. Where a source contradicts or complicates the framing in the brief, it is flagged with **⚠ Complicates the framing**.

Scope of the framing being tested:

> Evaluate imputation by how well the *imputed dataset* preserves distribution, uncertainty and randomness — not by how accurately the imputer predicts held-out cells. Must work on mixed-type tabular data, per-column and jointly, and must reduce to a scalar usable as an HPO objective.

**Headline finding:** the framing is well supported and is, as of 2024, the explicitly stated position of the primary methodological literature. But three things complicate it, all of which have first-party sources below:

1. The metric you want is almost certainly **not marginal per-column Wasserstein** — that is exactly the degenerate optimum you already worry about, and two papers demonstrate it empirically (§4).
2. **Empirical Wasserstein is the worst-behaved of the candidate metrics in dimension.** Its sample complexity is `O(n^{-1/d})`; MMD/energy is `O(n^{-1/2})` regardless of `d`. This is a first-party result in the standard reference (§1). It is the single strongest argument for energy distance over joint Wasserstein.
3. The literature's distributional scores mostly compare **imputed data against ground-truth complete data** (a benchmarking setting). Comparing *imputed values against observed values* — which is what a library can actually do at runtime — is a different and genuinely ambiguous quantity under MAR (§5). There is one paper that solves the no-ground-truth case (§5).

---

## Reading order

If you read four things, read them in this order:

1. **van Buuren FIMD §2.6** (10 min) — why per-cell accuracy is the wrong objective, with numbers.
2. **Näf, Scornet & Josse §4–§5** (40 min) — the modern answer: energy distance, and why not RMSE.
3. **Peyré & Cuturi §8.4** (20 min) — the sample-complexity wall that decides Wasserstein vs. energy/MMD.
4. **Shadbahr et al., Fig. 1 + Methods C** (20 min) — the degenerate-optimum failure demonstrated on real data, and the sliced-Wasserstein fix.

Everything else is depth on a specific question.

---

## Q1. Wasserstein: definition, the 1-D CDF formula, and the dimension problem

### R1. Cédric Villani, *Optimal Transport: Old and New* — Chapter 6, "The Wasserstein distances"

*Springer, Grundlehren der mathematischen Wissenschaften 338, 2009. Author's near-final draft PDF: <https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O&N.pdf> (547 pp.).*

**What it is.** The canonical mathematical reference. **What it answers:** what `W_p` *is*, stated once, cleanly, with the metric axioms proved.

**Read exactly:** Chapter 6, pp. 75–77 — Definition 6.1, Example 6.2, the two-page proof that `W_p` satisfies the distance axioms, Definition 6.3 (Wasserstein space `P_p`), Remark 6.4 (Kantorovich–Rubinstein duality). **Time: 20 minutes.** Stop at Remark 6.5; the rest of the chapter is topology you do not need.

Definition 6.1, verbatim:

> Let (X, d) be a metric space, and let p ∈ [1, +∞). For any two probability measures µ, ν on X, the Wasserstein distance of order p between µ and ν is defined by the formula
> `W_p(µ, ν) = ( inf_{π ∈ Π(µ,ν)} ∫ d(x,y)^p dπ(x,y) )^{1/p} = inf { E d(X,Y)^p }^{1/p}`

The proof of the triangle inequality (via the Gluing Lemma) is on p. 76 and is short enough to actually follow. This is the source for the claim "W_p is a proper metric" — Villani proves symmetry, triangle inequality, and `W_p(µ,ν) = 0 ⟹ µ = ν` in about fifteen lines. Note the caveat stated in the text: "At the present level of generality, `W_p` is not a distance in the strict sense, because it might take the value +∞" — hence Definition 6.3 restricting to measures with finite `p`-th moment.

Also worth two minutes: Remark 6.4 gives the dual form `W_1(µ,ν) = sup_{‖ψ‖_Lip ≤ 1} { ∫ψ dµ − ∫ψ dν }`. This is what makes `W_1` an *integral probability metric*, which is the bridge to MMD and energy distance in R3.

⚠ **Note on the URL:** the Dauphine PDF is titled "September 27, 2006" and is Villani's own pre-publication draft, not the 2009 Springer text. Numbering of Definition 6.1 / 6.3 matches the published book (Springer's chapter landing page for the published Ch. 6 is at `link.springer.com/chapter/10.1007/978-3-540-71050-9_6`, paywalled). Cite the book, read the draft.

### R2. Peyré & Cuturi, *Computational Optimal Transport* — §2.4, §2.6, §8.4

*Foundations and Trends in Machine Learning 11(5–6):355–607, 2019. Free: <https://arxiv.org/abs/1803.00567> (v4, 2020), companion site <https://optimaltransport.github.io/>.*

**What it is.** The computational counterpart to Villani. **Why it earns its place:** it is the only source here that tells you, with a number, how badly empirical Wasserstein degrades in dimension — which is the decisive fact for the whole design.

**Read exactly, in this order. Total: 45 minutes.**

**§2.6, Remark 2.30 ("1-D case — Generic case"), pp. 31–32. (10 min.)** This is the 1-D closed form. Verbatim:

> `W_p(α,β)^p = ‖C_α^{-1} − C_β^{-1}‖^p_{L^p([0,1])} = ∫_0^1 |C_α^{-1}(r) − C_β^{-1}(r)|^p dr` (2.36)
>
> This means that through the map α ↦ C_α^{-1}, the Wasserstein distance is isometric to a linear space equipped with the L^p norm or, equivalently, that the Wasserstein distance for measures on the real line is a Hilbertian metric.

and, for `p = 1`, equation (2.37):

> `W_1(α,β) = ‖C_α − C_β‖_{L^1(R)} = ∫_R |C_α(x) − C_β(x)| dx`

**This equation is the entire answer to "why does W₁ differ from KS."** Both are functionals of the same CDF difference `|C_α − C_β|`. KS takes its **sup**; `W_1` takes its **integral**. So KS sees only the single worst point of disagreement and is invariant to how wide that disagreement is; `W_1` accumulates disagreement across the whole support and is expressed in the units of the variable. A variance-collapsed imputation that is slightly wrong everywhere and badly wrong nowhere is nearly invisible to KS and clearly visible to `W_1`.

**§2.6, Remark 2.31 ("Distance between Gaussians"), p. 33. (5 min.)** Equation (2.41):

> `W_2^2(α,β) = ‖m_α − m_β‖^2 + B(Σ_α, Σ_β)^2`, where `B(Σ_α,Σ_β)^2 = tr(Σ_α + Σ_β − 2(Σ_α^{1/2} Σ_β Σ_α^{1/2})^{1/2})`

This is the cleanest statement anywhere that `W_2` **explicitly and separately penalises covariance mismatch** via the Bures metric. For the variance-collapse question (Q3) this is the theoretical hook: a conditional-mean imputer shrinks `Σ`, and `W_2` charges for it in a named term. `W_1` has no such decomposition.

**§8.4, "Empirical Estimators for OT, MMD and φ-divergences", pp. 128–131. (20 min.)** The single most important page in this document. Verbatim:

> **Rates for OT.** For X = R^d and measure supported on bounded domain, it is shown by [Dudley, 1969] that for d > 2, and 1 ≤ p < +∞,
> `E(|W_p(α̂_n, β̂_n) − W_p(α,β)|) = O(n^{-1/d})`
> where the expectation E is taken with respect to the random samples. This rate is tight in R^d if one of the two measures has a density with respect to the Lebesgue measure.
>
> **Rates for MMD.** For weak norms ‖·‖_k which are dual of RKHS norms (also called MMD) […] and contrary to Wasserstein distances, the sample complexity does not depend on the ambient dimension
> `E(|‖α̂_n − β̂_n‖_k − ‖α − β‖_k|) = O(n^{-1/2})`

**⚠ Complicates the framing.** Read `n^{-1/d}` concretely. On a 20-column dataset, halving the estimation error of a joint empirical Wasserstein requires `2^20 ≈ 10^6`× more rows. Joint Wasserstein over a whole tabular dataset is not an estimator; it is a number that mostly measures your sample size. Two consequences:

- **Per-column `W_1` is statistically fine** (`d = 1`, rate `n^{-1}` by Fournier–Guillin/Dudley refinements), which is why it is the right *diagnostic* per column — but it is exactly the metric with the degenerate optimum (§4).
- **Joint fidelity should be measured with energy distance / MMD, or with a sliced (projection-based) Wasserstein**, both of which sidestep the `n^{-1/d}` wall. This is not an aesthetic preference; it is the reason both Näf et al. and Shadbahr et al. independently avoided plain joint Wasserstein.

Same section, one more line that kills a whole family of candidates:

> Note that for `D(α,β) = ‖·‖_TV`, since the TV norm does not metrize the weak convergence, `‖α̂_n − β̂_n‖_TV` is not a consistent estimator […] Indeed, with probability 1, `‖α̂_n − β̂_n‖_TV = 2` since the support of the two discrete measures does not overlap. Similar issues arise with other φ-divergences, which cannot be estimated using divergences between empirical distributions.

and §8.4.2:

> It is not possible to approximate `D_φ(α|β)` […] from discrete samples using `D_φ(α̂_n | β̂_n)`. Indeed, this quantity is either +∞ (for instance, for the KL divergence) or is not converging […] Instead, it is required to use a density estimator.

**This rules out KL, JS and TV as continuous-column metrics without an arbitrary binning or bandwidth choice.** They are computable on *categorical* columns (finite shared support) and nowhere else without a smoothing decision you would then have to defend and tune.

**Optional, 10 min: §2.4, Proposition 2.2 and its proof, pp. 19–21** for the discrete-histogram metric proof, and **§8.2, Examples 8.7–8.12, pp. 121–125** for the IPM framing that puts `W_1`, MMD and energy distance on one axis. §8.2's Example 8.12 states energy distance *is* an MMD with kernel `k_ED(x,y) = −d(x,y)^p`, and notes its chief practical advantage: "it is scale-free and does not depend on a bandwidth parameter σ."

### R3. SciPy first-party API docs (v1.18.0 — current release as of this writing)

*<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html> · <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance_nd.html>*

**What it answers:** exactly what SciPy will and will not give you. **Time: 10 minutes for both.** Read the Notes section of each.

The 1-D page (added 1.0.0) states the CDF formulation verbatim:

> If U and V are the respective CDFs of u and v, this distance also equals to: `l_1(u,v) = ∫_{-∞}^{+∞} |U−V|`

and confirms it takes raw samples:

> The input distributions can be empirical, therefore coming from samples whose values are effectively inputs of the function, or they can be seen as generalized functions, in which case they are weighted sums of Dirac delta functions located at the specified values.

**Two hard limits to know before designing around SciPy:**

- **SciPy ships Wasserstein-1 only.** Both functions are titled "Compute the Wasserstein-1 distance…", and neither takes a `p` parameter. There is no `W_2` in SciPy. If you want the covariance-penalising `W_2` of Remark 2.31 you must use POT.
- `wasserstein_distance_nd` (added 1.13.0) solves the full LP: the Notes derive the Monge problem as a linear program with constraint matrix `A`, `b = [u; v]`, and state the distance is recovered as `b^T y*` from the dual solution. No complexity bound is documented, but this is an exact `n × m` transport solve — treat it as `O(n^3)`-ish and unusable above a few thousand rows.

Also on the shelf: `scipy.stats.energy_distance` (1-D only). Its Notes give

> `D(u,v) = sqrt( 2E|X−Y| − E|X−X'| − E|Y−Y'| )`

with the useful caveat "Only the square root definition satisfies metric axioms, though some literature refers to the squared quantity as 'energy distance'", and the 1-D identity `D(u,v) = √2 · l_2(u,v)` relating it to the **Cramér–von Mises** distance. References Székely (2002, BGSU TR 02-16) and Rizzo & Székely, *WIREs Comput. Stat.* 8(1):27–38 (2015).

### R4. POT — Python Optimal Transport (v0.9.7)

*<https://pythonot.github.io/quickstart.html> · API index <https://pythonot.github.io/all.html>*

**What it answers:** what to use when SciPy is not enough — `W_2`, sliced Wasserstein, entropic/Sinkhorn approximations.

**Read exactly:** the Quickstart "Optimal transport and Wasserstein distance" section and the "Entropic regularized OT" section. **Time: 20 minutes.**

Key facts verified on the page:

- `ot.emd` returns the transport matrix, `ot.emd2` "returns the Wasserstein distance" (the optimal value). Complexity is documented as **`O(n^3)`** for the network-simplex solver, "quite efficient and uses sparsity of the solution".
- **Memory:** "the memory cost for an OT problem is always `O(n^2)`". This, not runtime, is what will kill you on a 50k-row dataset.
- For `W_2` with squared-Euclidean cost you must **take the square root of `ot.emd2` yourself**.
- `ot.dist` computes the ground cost matrix — this is the hook for mixed-type data: supply your own metric (e.g. Gower-style) rather than assuming Euclidean.
- `ot.wasserstein_1d` — "Solves the Earth Movers distance problem between 1d measures and returns the loss". Vectorised, and unlike SciPy it takes `p`.
- **`ot.sliced_wasserstein_distance`** — the escape hatch from `n^{-1/d}`, and the function Shadbahr et al. (§4) built their score on.
- `ot.sinkhorn2` for the entropic-regularised value.

Pair this with **Peyré & Cuturi §8.5** (pp. 131–132, 5 min), which proves the Sinkhorn divergence *interpolates* between the two poles: Proposition 8.3 states `W̃_{p,ε} → 2W_p` as `ε → 0` and `W̃_{p,ε}^p → ‖α−β‖²_ED` as `ε → +∞`. So "Wasserstein vs. energy distance" is literally a single knob `ε`, and its effect on sample complexity is the subject of Genevay et al. (2019), cited there. If you ever want one tunable joint-fidelity metric with the OT-vs-MMD tradeoff exposed as a parameter, that is the object.

---

## Q2. The alternatives — comparison table

Everything in this table was verified against the source named in the last column. "Mixed-type" means: does it work on a table containing continuous, bounded-discrete and categorical columns *without* a preprocessing decision you would have to defend and tune.

| Metric | Sensitive to | Blind to | Proper metric? | Kernel/bandwidth/binning needed? | Mixed-type tabular? | Sample complexity in `d` | Python | Verified against |
|---|---|---|---|---|---|---|---|---|
| **Wasserstein-1** (1-D, per column) | the whole integrated CDF gap `∫\|U−V\|`; location and spread; in the variable's own units | dependence between columns entirely | Yes (Villani Def. 6.1, p ≥ 1) | No | Numeric/ordinal yes; categorical needs a ground metric on labels | `d=1`: fast | `scipy.stats.wasserstein_distance`; `ot.wasserstein_1d` | Villani Def 6.1; P&C (2.37); SciPy 1.18.0 |
| **Wasserstein-2** (joint) | mean shift **and** covariance mismatch as separate terms (Gaussian case: `‖Δm‖² + Bures²`) | little — this is the most sensitive on the list | Yes | No, but needs a ground cost on mixed types | Only via a user-supplied `ot.dist` metric | **`O(n^{-1/d})` — the wall** | `ot.emd2` + sqrt | P&C (2.41), §8.4; POT quickstart |
| **Sliced Wasserstein** (joint) | joint structure along random 1-D projections; recovers dependence that marginals miss | structure orthogonal to all sampled directions (mitigated by `M ≥ d`) | Yes | No; needs `M` (number of projections) | Numeric only (needs a vector space to project in) | Escapes `n^{-1/d}` by construction | `ot.sliced_wasserstein_distance` | POT API; Shadbahr et al. Methods C |
| **Kolmogorov–Smirnov** (2-sample) | the single largest CDF gap | width of the discrepancy; anything in the tails where the CDF is flat; dependence | The statistic is a valid metric on CDFs; distribution-free | No | Continuous only ("underlying *continuous* distributions") | `d=1` only | `scipy.stats.ks_2samp` | SciPy 1.18.0 |
| **Cramér–von Mises** (2-sample) | integrated *squared* CDF gap, rank-based | dependence; less tail-sensitive than Anderson–Darling | Rank statistic, not a metric on measures | No | Continuous only | `d=1` only | `scipy.stats.cramervonmises_2samp` | SciPy 1.18.0 |
| **Energy distance** (Székely & Rizzo) | joint distributional difference in any `d`; in 1-D equals `√2 ×` Cramér distance | nothing structural; weaker than `W_p` at large-displacement mass movement | Yes — **but only the square-root form** | **No — scale-free, no bandwidth.** This is its selling point | Numeric via Euclidean; mixed-type via a valid negative-definite metric | **`O(n^{-1/2})`, dimension-free** (it is an MMD) | `scipy.stats.energy_distance` (1-D); `dcor` 0.7; R `energy` | SciPy 1.18.0; P&C Ex. 8.12, §8.4 |
| **MMD** (Gretton et al.) | whatever the kernel is sensitive to; with a universal kernel, everything (metrizes weak convergence) | anything the kernel's bandwidth washes out | Yes, for characteristic kernels | **Yes — kernel and bandwidth `σ`.** P&C: "an important issue […] is that one needs to select the bandwidth parameter σ" | Yes, via a product/sum of per-type kernels | `O(n^{-1/2})`, dimension-free | `hyppo` 0.5.2; kernels via sklearn; `ot` | JMLR 13:723–773; P&C §8.2.2, §8.4 |
| **KL divergence** | density-ratio mismatch; punishes support gaps infinitely | symmetric behaviour (it is not symmetric) | **No** — not symmetric, no triangle inequality | **Yes, unavoidably** — density estimator or binning | Categorical only, in practice | **Not estimable from empirical measures at all** | `scipy.stats.entropy` (needs pmf) | P&C §8.4.2 |
| **Jensen–Shannon** | as KL but symmetrised and bounded | fine detail smoothed by binning | `√JS` is a metric | **Yes** — same problem as KL | Categorical only, in practice | Same problem as KL | `scipy.spatial.distance.jensenshannon` — takes "probability vector", not samples | P&C §8.4.2; SciPy 1.18.0 |
| **Total variation** | any probability-mass difference | geometry entirely (no notion of "close values") | Yes | Only meaningful on a shared discrete support | **Categorical: yes, and it is the natural choice.** Continuous: no | Empirically **always 2** on continuous samples — inconsistent | Hand-rolled; `sdmetrics` `TVComplement` | P&C §8.4; SDMetrics docs |
| **pMSE** (propensity) | *anything a classifier can detect*, jointly, incl. dependence | whatever the propensity model cannot represent | Not a metric; a fitted statistic | Yes — the propensity model is a choice | **Yes, natively — this is its strength** | Model-dependent | `synthpop` (R); trivially reimplemented | Snoke et al. §4 |
| **C2ST** (classifier accuracy) | as pMSE; learns its own representation | as pMSE | Not a metric | Yes — the classifier | **Yes, natively** | Model-dependent | Any sklearn classifier | Lopez-Paz & Oquab §3 |

Two rows deserve their own note:

**Energy distance is an MMD.** P&C Example 8.12: "The energy distance (or Cramer distance when d = 1) [Székely and Rizzo, 2004] associated to a distance d is defined as `‖α−β‖_{ED(X,d^p)} = ‖α−β‖_{k_ED}` where `k_ED(x,y) = −d(x,y)^p` for 0 < p < 2. It is a valid MMD norm over measures if d is negative definite". So energy distance inherits MMD's dimension-free `O(n^{-1/2})` rate **while having no bandwidth to pick**. That combination is why it is the modern default for this exact task. Caveat from the same passage: `p` must satisfy `0 < p < 2`, and "for p = 2, it degenerates to the distance between the means" — at `p = 2` it stops being a norm and becomes blind to everything except the mean. Do not set `p = 2`.

**KS is what SDMetrics actually ships**, so the industry-standard synthetic-data score is built on the metric with the weakest per-column sensitivity on this list. Worth knowing when comparing against it.

### R5. Gretton, Borgwardt, Rasch, Schölkopf & Smola, "A Kernel Two-Sample Test"

*JMLR 13(25):723–773, 2012. <https://www.jmlr.org/papers/v13/gretton12a.html>*

**Read exactly:** §1 (introduction) and §2 (definition of MMD, the biased/unbiased estimators). **Time: 30 minutes.** Skip §3–§8 unless you are implementing a test.

MMD is "the largest difference in expectations over functions in the unit ball of a reproducing kernel Hilbert space (RKHS)", and — the sentence that connects it to everything else here — "Our statistic is an instance of an integral probability metric, and various classical metrics on distributions are obtained when alternative function classes are used in place of an RKHS." Quadratic-time exact, linear-time approximations available.

**Why it earns its place despite the bandwidth problem:** the *unbiased* estimator matters if you plan to optimise. P&C §8.4 spells it out: the plain plug-in `‖α̂_n − β̂_n‖²_k` "is a slightly biased estimate", and the unbiased form drops the diagonal terms:

> `MMD_k(α̂_n, β̂_n)² = 1/(n(n−1)) Σ_{i≠i'} k(x_i,x_{i'}) + 1/(n(n−1)) Σ_{j≠j'} k(y_j,y_{j'}) − 2/n² Σ_{i,j} k(x_i,y_j)`
>
> It satisfies `E(MMD_k(α̂_n,β̂_n)²) = ‖α−β‖²_k`; see [Gretton et al., 2012].

If you feed a *biased* distance to an optimiser you are optimising the bias. Use the unbiased form. Note it can go negative — that is correct behaviour, not a bug, and it means the "perfect score" is 0 with symmetric noise around it, which is exactly the well-behaved-optimum property you want.

### R6. Ramdas, García Trillos & Cuturi, "On Wasserstein Two Sample Testing and Related Families of Nonparametric Tests"

*<https://arxiv.org/abs/1509.02237>*

**What it is.** The map that connects everything in the table above. **What it answers:** "how do KS, PP/QQ plots, ROC curves, energy statistics, MMD and Wasserstein relate to each other?" The abstract: they "tie together many of these tests, drawing connections between seemingly very different statistics", via a **smoothed Wasserstein distance** and a **distribution-free Wasserstein test**.

**Read exactly:** the abstract and the introduction's roadmap figure. **Time: 20 minutes** for orientation; the full paper is a research read you probably do not need. Cited by SciPy's own `wasserstein_distance` docs (reference [3], "for a proof of the equivalence of both definitions" of `W_1`), and by P&C for Proposition 8.3.

---

## Q3. Variance collapse: the metric that ranks conditional-mean imputers backwards

This is the best-documented question in the set. There are three independent demonstrations.

### R7. van Buuren, *Flexible Imputation of Missing Data*, 2nd ed. — §2.6 "Imputation is not prediction"

*<https://stefvanbuuren.name/fimd/sec-true.html>*

**What it is.** Two pages. Read them first. **Time: 10 minutes.** This is the highest value-per-minute source in the entire document.

Verbatim, the argument:

> It is well known that the minimum RMSE is attained by predicting the missing `ẏ_i` by the linear model with the regression weights set to their least squares estimates. According to this reasoning the "best" method replaces each missing value by its most likely value under the model. However, this will find the same values over and over, and is single imputation. This method ignores the inherent uncertainty of the missing values (and acts as if they were known after all), resulting in biased estimates and invalid statistical inferences. **Hence, the method yielding the lowest RMSE is bad for imputation.** More generally, measures based on similarity between the true and imputed values do not separate valid from invalid imputation methods.

And the numbers, from a 1000-run simulation with 50% MCAR missingness in `x`:

| Method | RMSE (§2.6) | Percent bias | Coverage (nominal 0.95) |
|---|---|---|---|
| `norm.predict` (deterministic regression) | **0.725** ← "better" | 34.3 | **0.364** |
| `norm.nob` (stochastic regression) | 1.025 | 0.5 | 0.925 |

Read §2.6 alongside **§2.5.2 "Evaluation criteria"** (<https://stefvanbuuren.name/fimd/sec-evaluation.html>, 10 min), which supplies the coverage/bias table above and closes with: "While the RMSE is widely used, we will see in Section 2.6 that it is not a suitable metric to evaluate multiple imputation methods."

**This is the exact ranking reversal the brief asked for**, from the field's standard textbook, with a reproducible R simulation on the page. The metric that ranks correctly here is *coverage*, not a distributional distance — which is a subtlety worth carrying: van Buuren's answer to "RMSE is wrong" is inferential validity, not distributional fidelity. The distributional-fidelity answer comes from R8.

### R8. Näf, Scornet & Josse, "What Is a Good Imputation Under MAR Missingness?"

*arXiv:2403.19196 (v5, Jan 2026), submitted to Statistical Science. <https://arxiv.org/abs/2403.19196> — 25 pp.*

**This is the keystone resource.** If you read one paper end to end, read this one. It is a 2024–26 methodological paper whose entire third contribution is "how should imputation methods be evaluated?" and whose answer is: with a distributional distance, specifically the energy distance.

**Read exactly:**
- **§1, the paragraph beginning "Currently, imputation methods are largely benchmarked…" (p. 2).** 5 min.
- **§4, from "Requirement (1) suggests…" through the energy distance formula.** 10 min.
- **§5.1 and §5.2 with Figures 8 and 10.** 25 min.
- **§6 Discussion, "RMSE should not be used".** 5 min.

**Total: 45 minutes.** §2–§3 are a careful measure-theoretic treatment of MAR identification — excellent, but a separate read.

The verdict, verbatim from §1:

> However, when comparing imputation methods, one should refrain from using measures such as the RMSE, as already pointed out (Van Buuren, 2018; Hong and Lynn, 2020). Indeed, measures like RMSE favor methods that impute conditional means, instead of draws from the conditional distribution. Hence, **using RMSE as a validation criterion tends to favor methods that artificially strengthen the dependence between variables** and lead to severe biases in parameter estimates and uncertainty quantification. […] we emphasize here that **imputation is a distributional prediction task and needs to be evaluated as such.** Thus, we advocate the use of a distributional metric or score (Gneiting and Raftery, 2007; Székely, 2003) between actual and imputed data sets. In particular, we focus on the energy distance (Székely, 2003) **which is simple to calculate and does not require to choose any tuning parameters.**

Their score, §4 verbatim:

> `d̃²(H, P_X) = 2 E_{X∼H, Y∼P_X}‖X−Y‖₂ − E_{X,X'∼H}‖X−X'‖₂ − E_{Y,Y'∼P_X}‖Y−Y'‖₂`

— i.e. the **joint** energy distance over the full `d`-dimensional row, computed with the R `energy` package (Rizzo & Székely).

The empirical result (§5.1, Figure 8) is the ranking reversal in its cleanest form. Comparing mice-cart, mice-DRF, missForest, mice-norm.predict, mice-norm.nob, GAIN and MIWAE against a known ground truth:

> As expected, the true imputation is ranked highest in terms of energy distance, closely followed by mice-cart mice-DRF and mice-norm.nob. […] **Despite this success all three distributional mice imputations attain the worst score in terms of (negative) RMSE.**

The *oracle* imputer — sampling from the true conditional distributions — ranks **first** by energy distance and among the **worst** by RMSE. That is as decisive as this kind of evidence gets.

§5.2 adds the downstream check: "The ordering induced by the energy distance matches well with the performance on the Wasserstein downstream task", i.e. the energy-distance ranking predicts which imputer preserves a real analysis result. And §6: "**RMSE is not a sensible way of evaluating imputations.** Dropping RMSE as an evaluation method likely has important implications. For instance, the recommendation of papers to use single imputation methods such as k-NN imputation […] or missForest […] appears to rest entirely on the use of RMSE."

Also note §1.2 for the relationship to R9: "Independently, Shadbahr et al. (2023) propose a similar approach but using the sliced Wasserstein distance (Bonneel et al., 2015). Their procedure is designed for high-dimensional data and rather complicated as it involves randomly partitioning the data and projecting to the real numbers multiple times. In contrast, we simply propose to calculate the energy distance between the imputed and real data."

**⚠ Complicates the framing.** Näf et al.'s score is computed against **`P_X`, the true complete data** — explicitly: "assume that a sample from `P_X` is observed, i.e. the complete data is available, **as is the case in benchmarking studies**." This is a benchmarking criterion, not a runtime one. To use it inside a library you must either (a) mask-and-restore known cells to synthesise a ground truth, or (b) use R13's I-Score, which is designed to avoid exactly this requirement.

### R9. Shadbahr, Roberts, Stanczuk, Gilbey, Teare et al., "The impact of imputation quality on machine learning classifiers for datasets with missing values"

*Communications Medicine 3, 139 (2023). Open-access PDF: <https://api.repository.cam.ac.uk/server/api/core/bitstreams/b6722ae8-8e81-47ad-94de-46bba7792628/content>*

**What it answers:** all three of Q3, Q4 and "does any of this predict anything downstream", on real clinical data (MIMIC-III) plus simulations.

**Read exactly:** Methods sections A, B and C (pp. 4–6, ~2 pages) — especially **Figure 1**, which is the degenerate-optimum picture — and the Discussion's first three paragraphs. **Time: 25 minutes.**

The ranking reversal, verbatim from Results:

> **MissForest performs best by sample-wise RMSE and MAE** […] **MICE imputation, which performed worst by the sample-wise discrepancy scores, is the best-performing method by the Kolmogorov-Smirnoff statistic and Wasserstein distance for all missingness rates** across all the minimum, median and maximum discrepancies.

And from the Discussion:

> In our experiments, we find that the sample-wise discrepancy scores are not sufficient to assess the quality of the reconstruction of the distribution of imputed values. **In fact, for MSE, we have seen that it takes an optimal value for imputations that give a very poor distribution match.** Using MSE to evaluate the imputation quality is only statistically justified in the case of imputing data from a Gaussian distribution (minimising MSE corresponds to maximising the log-likelihood).

Their taxonomy is worth stealing wholesale — it is the cleanest three-level framing of this problem anywhere:

- **Class A — sample-wise:** RMSE, MAE, R². Per-cell accuracy.
- **Class B — feature-wise distribution:** per-column KL, KS, 2-Wasserstein (crediting Thurow, Dumpert, Ramosaj & Pauly, arXiv:2101.07532).
- **Class C — whole-distribution:** their contribution. Verbatim: "we strongly believe that the class of measures which would be of most practical value to practitioners is (C) measures of discrepancy for imputed and true data across the whole data distribution. **In the literature, we were unable to find any examples of discrepancy measures of this type and we propose such a class in this paper.**"

The load-bearing finding for the HPO question, from the Discussion:

> We also find that not only are the common discrepancy scores used by the community, i.e. the sample-wise statistics (RMSE, MAE, R²), **uncorrelated from the distributional discrepancy metrics (of classes B and C)** but that they are also **disconnected from the downstream classification performance** of the model. […] Importantly, we find **a correlation between the proposed class of sliced Wasserstein discrepancy scores and the downstream model performance**.

So: per-cell metrics are orthogonal to distributional metrics *and* to downstream utility; the joint distributional metric is the one that tracks utility. That is the empirical warrant for the whole design.

---

## Q4. The degenerate optimum: marginal-sampling scores perfectly and is useless

### R9 again — Methods §C, and Figure 1

Shadbahr et al. state the failure mode in one sentence and draw it:

> In Figure 1, we show that **simply considering the feature-by-feature marginal distributions is not sufficient to quantify how well a high-dimensional data structure has been imputed. The marginal distributions of the imputed data (directions 1 and 2) match that of the original data perfectly but do not identify the discrepancy of the distributions** shown in Figures 1(a) and (b).

**This is precisely the degenerate optimum in the brief.** An imputer that samples the observed marginal per column attains a perfect per-column score and an arbitrarily bad joint one.

Their fix is the **sliced Wasserstein** construction (Methods C, Steps 1–3), and the design reasoning is worth reading because it names both problems at once:

> Modelling the discrepancy between imputed and true data in high dimensions is challenging for two key reasons. Firstly, the **curse of dimensionality** results in computations that are infeasible for high-dimensional datasets. Secondly, high-dimensional (complete) datasets are **very sparse** […] unless there are an unrealistically large number of samples. We address both of these issues by repeatedly projecting the entire data distribution to random one-dimensional subspaces.

The procedure: choose `M ≥ d` random unit directions `n_r`; choose `P` random half-splits of the rows; for each `(r, p)` compute the 2-Wasserstein between projected original and projected imputed data, **normalised by the standard deviation of the projected data**; and crucially, also compute a **baseline** `w(r,p)` between two halves of the *original* data. Their reported score is the **ratio** `ŵ(r,p)/w(r,p)`.

**That baseline ratio is the design idea to steal.** It converts a raw distance — whose floor depends on `n`, `d` and the data — into a quantity whose meaning is stable: 1.0 means "as different as two halves of the real data are from each other". That is a **known degenerate optimum with known noise behaviour**, obtained empirically instead of asymptotically. Their parameters: `P = 10`, `M = 50` or `90`.

⚠ One caveat they flag themselves and Näf et al. echo: the procedure "is designed for high-dimensional data and rather complicated". If you do not need the ratio, the joint energy distance of R8 gets you most of the joint sensitivity in one line.

### R10. Snoke, Raab, Nowok, Dibben & Slavković, "General and specific utility measures for synthetic data"

*arXiv:1604.06651v2 (2017); published JRSS-A 181(3):663–688, 2018. <https://arxiv.org/abs/1604.06651>*

**What it answers:** all three of "how do you build a joint criterion", "what is a known degenerate optimum" and "how do you standardise a distance so an optimiser can use it". Uniquely on this list, it does all three at once.

**Read exactly:** §3 (the propensity-score method and Algorithm 1, ~1 page), §4.1 including Equations (1) and (2), and **§4.3 with Table 1** — the simulation. **Time: 40 minutes.** Skip the disclosure-risk material.

**The statistic** (§3, Algorithm 1), verbatim:

> The `n_1` rows of the original and `n_2` rows of the masked datasets are combined with the addition of an indicator variable `I` giving the source of the data (0 for original data and 1 for altered). A propensity score `p̂_i` is estimated for each of the `N = n_1 + n_2` rows […] The mean squared difference between these estimated probabilities and the true proportion of records from the masked data in the combined data (denoted `c = n_2/N`, usually ½), gives the utility statistic `1/N Σ(p̂_i − c)²` (the propensity score mean-squared error, henceforth referred to as pMSE).

And, crucially for Q4:

> If we can model the propensity scores well, **this general measure should capture relationships among the data that methods such as the empirical CDF may miss.**

**The known null** (§4.1.1), which is the answer to "known degenerate optimum" — verbatim:

> the null pMSE is distributed as a multiple of a chi-squared distribution with `(k−1)` degrees of freedom and expectation and standard deviation given by
> `E[pMSE] = (k−1)((n_1/N)²(n_2/N))/N = (k−1)(1−c)²c/N`  (1)
> `StDev(pMSE) = √2 (k−1)(1−c)²c/N`  (2)
> […] In the most common case when `n_1` and `n_2` are equal, the expectation becomes `(k−1)/(8N)`.

From this they define two derived scalars: the **pMSE ratio** (expected value **1** under correct synthesis) and the **standardized pMSE** (expected value **0**, standard deviation **1**). Verbatim: "In both cases, increased values of these statistics will be expected if CS does not hold."

**This is the single most HPO-ready construction in the whole literature.** A scalar with an analytically known optimum, an analytically known standard deviation at the optimum, and a clear direction of badness. An optimiser can be told "stop at 1" or "anything under ~2 standard deviations is indistinguishable from correct."

**The degenerate-optimum experiment** (§4.3) is worth quoting because it is *your* scenario, deliberately constructed:

> For the correct synthesis we use the variance matrix fitted to the Real data to generate synthetic multivariate Normal data. **For the incorrect synthesis we use the sample means and a variance matrix with its off-diagonal elements set to 0.** […] **This emulates synthesis that fails to account for correlations between the variables.**

Zeroed off-diagonal covariance, correct marginals — the independent-marginal sampler. And the propensity model that catches it: "a logistic regression model including all main effects and **first-order interactions** for the variables, but omitting the quadratic terms, giving us `k = 56` parameters." With `n = 5000`: `E[pMSE] = 55 × 0.5³/10000 = 0.000688`. In Table 11 (the CART variant), the standardized pMSE for the incorrect synthesis climbs from ~0 at population covariance 0.0 to **29.7** at covariance 0.9. Correct synthesis stays at ~0 throughout.

**⚠ Two things that complicate using pMSE naively**, both stated in the paper:

- **The propensity model must be able to see dependence, or the metric cannot.** Main effects only would score the independent-marginal sampler as perfect. Interactions are load-bearing, and `k` enters the null expectation directly.
- **Over-flexible models saturate.** Verbatim: "it is important to not use overly large trees, since it will make it harder to discern between worse syntheses. **As the number of splits (effective parameters) in the tree approaches the number of observations, the expected pMSE ratio will be limited at 2, no matter how bad the synthesis.** This is because the maximum value for the pMSE is 0.25, and the maximum expectation under the null is 0.125 (1/8)". A saturating objective is death for an optimiser — it flattens exactly where you need gradient.

Implemented in R's `synthpop`; trivial to reimplement (stack, label, fit, mean-squared-deviation-from-`c`).

### R11. SDMetrics — the Quality Report's two-property split

*SDMetrics 0.28.2. Docs: <https://docs.sdv.dev/sdmetrics/>. Full doc corpus (useful, fetchable in one go): <https://docs.sdv.dev/sdmetrics/llms-full.txt>*

**What it is.** The de-facto industry implementation of exactly this evaluation, for synthetic tabular data. **Why it earns its place:** it is a *worked answer to the mixed-type problem*, and its structure encodes the marginal/joint distinction as a first-class design decision.

**Read exactly:** the "What's included in the Quality Report" page (Column Shapes / Column Pair Trends methodology tables), plus the four metric pages `KSComplement`, `TVComplement`, `CorrelationSimilarity`, `ContingencySimilarity`. **Time: 30 minutes.** All verified against the doc corpus.

The type-routing table — this is the mixed-type answer, verbatim from the docs:

| Property | Column type(s) | Metric |
|---|---|---|
| Column Shapes | numerical, datetime | `KSComplement` |
| Column Shapes | boolean, categorical | `TVComplement` |
| Column Pair Trends | numerical × numerical | `CorrelationSimilarity` |
| Column Pair Trends | categorical × categorical | `ContingencySimilarity` |
| Column Pair Trends | numerical × categorical | "Discretize the numerical columns into bins, then apply `ContingencySimilarity`" |

Details worth noting:

- `KSComplement` returns `1 − KS statistic`; `TVComplement` returns `1 − TVD` where `δ(R,S) = ½ Σ_ω |R_ω − S_ω|`. Both are in `[0,1]`, higher is better, **best = 1.0** — a fixed, known optimum, which is HPO-friendly. Both docs state "This metric ignores missing values."
- `CorrelationSimilarity` = `1 − |S_{A,B} − R_{A,B}|/2`, supporting Pearson or Spearman. This is the "does the imputed data preserve the correlation structure" check in its most literal form.
- **The Column Shapes / Column Pair Trends split is exactly the marginal-vs-joint split of Q4**, and SDV shipped it because marginal scores alone were misleading. The example in the `CorrelationSimilarity` docs is a synthetic dataset where "The real data has a strongly positive correlation of 0.93 but the synthetic data has a weak correlation of 0.22."
- **A gotcha for aggregate scoring**, verbatim: "*Starting from SDMetrics version 0.27.0, the Quality Report discards pairs that do not exhibit a strong pattern in the real data to begin with. A strong correlation is defined as a Pearson correlation of >0.5 or <−0.5, or a Cramer's association of >0.3.*" Without this filter the pairwise average is dominated by thousands of genuinely-uncorrelated pairs and the score becomes uninformative. If you build a pairwise aggregate, you will hit this and need the same fix.
- Scaling: "if your dataset contains over 50K rows, the quality report may subsample your data to compute specific metrics such as `ContingencySimilarity`."

⚠ **Complicates the framing.** Column Pair Trends is `O(k²)` pairs and only ever *pairwise*. It cannot see three-way structure. Energy distance and pMSE can. Pairwise is a pragmatic middle, not the joint criterion.

---

## Q5. The MAR ambiguity — the hardest part, and the least resolved

### R12. van Buuren, FIMD §6.6 "Diagnostics"

*<https://stefvanbuuren.name/fimd/sec-diagnostics.html>*

**Read exactly:** §6.6 intro, §6.6.1 "Model fit versus distributional discrepancy", §6.6.2 "Diagnostic graphs" in full. **Time: 15 minutes.** It is short.

This is the first-hand source requested, and it is more nuanced than the framing suggests. Three passages, verbatim.

**(a) The distributional criterion is endorsed, and preferred over model fit.** §6.6 intro:

> Conventional model evaluation concentrates on the fit between the data and the model. **In imputation it is often more informative to focus on distributional discrepancy, the difference between the observed and imputed data.**

§6.6.1 makes it concrete with the worm-plot example: "despite the fact that the model does not fit the data, the distributions of the observed and imputed data are similar. **This distributional similarity is more relevant for the final inferences than model fit per se.**"

**(b) The ambiguity, stated plainly.** §6.6.2:

> One of the best tools to assess the plausibility of imputations is to study the discrepancy between the observed and imputed data. **The idea is that good imputations have a distribution similar to the observed data.** In other words, the imputations could have been real values had they been observed. **Except under MCAR, the distributions do not need to be identical, since strong MAR mechanisms may induce systematic differences between the two distributions.** However, any dramatic differences between the imputed and observed data should certainly alert us to the possibility that something is wrong.

**(c) The resolution — condition on the response propensity.** Also §6.6.2:

> **Bondarenko and Raghunathan (2016) proposed a more refined diagnostic tool that aims to compare the distributions of observed and imputed data conditional on the missingness probability. The idea is that under MAR the conditional distributions should be similar if the assumed model for creating multiple imputations has a good fit.**

and the closing paragraph, which is the honest statement of where the field is:

> The marginal distributions of the observed and imputed data may differ because the missing data are MAR or MNAR. The diagnostics tell us in what way they differ, and hopefully also suggest whether these differences are expected and sensible in light of what we know about the data. **Under MAR, any distributions that are conditional on the missing data process should be the same.** If our diagnostics suggest otherwise […] there might be something wrong with the imputations that we created. Alternatively, it could be the case that the observed differences are justified, and that the missing data process is MNAR. **The art of imputation is to distinguish between these two explanations.**

**⚠ This is the deepest complication in the brief, and it is not fully resolvable.** Concretely:

1. **A raw observed-vs-imputed marginal distance is not a loss function.** Under MAR its correct value is *not zero*, and its correct value is unknown. Minimising it towards zero actively rewards imputers that *ignore* the MAR mechanism — the mean-imputation-shaped failure in a different disguise.
2. **The formalised fix is conditional comparison.** van Buuren's own recipe, given as runnable R on the page: fit `glm(is-incomplete ~ all variables)`, average the fitted propensities across imputations, then compare observed and imputed values *at matched propensity*. His caveat: "the comparison is only as good as the propensity score is. If important predictors are omitted from the response model, then we may not be able to see the potential misfit."
3. **Note the convergence.** van Buuren's conditional diagnostic and Snoke et al.'s pMSE both fit a propensity model and both compare distributions through it. They arrive from opposite directions (imputation diagnostics vs. disclosure control) at the same construct. If you build one thing, that is the thing to build.

Full citation for the resolution paper: **Bondarenko, I. & Raghunathan, T. (2016), "Graphical and numerical diagnostic tools to assess suitability of multiple imputations and imputation models", *Statistics in Medicine* 35(17):3007–3020.** PMID 26952693. (I verified the citation and abstract via search-result metadata and van Buuren's first-hand citation of it; the publisher page is paywalled and PubMed blocked automated fetch, so I did not read the paper itself — treat the tool description as van Buuren's characterisation, which is quoted verbatim above.)

### R13. Näf, Spohn, Michel & Meinshausen, "Imputation Scores"

*Annals of Applied Statistics 17(3):2452–2472, 2023. <https://arxiv.org/abs/2106.03742>*

**Why it earns its place: it is the only source here that addresses the no-ground-truth case**, which is the case a library actually faces.

**Read exactly:** the abstract and the definition of the DR-I-Score. **Time: 30 minutes** for orientation.

Verified from the abstract page: the paper develops "I-Scores" for assessing imputations, gives a concrete score "based on density ratios and projections" applicable to discrete and continuous data, and — the distinguishing property — it "**does not require to mask additional observations for evaluations and is also applicable if there are no complete observations.**"

It states the problem the whole family exists to fix: "imputations based on the conditional mean will rank highest if predictive accuracy is measured with quadratic loss", whereas the goal is to rank highest an imputation that samples from the true conditional distributions.

**⚠ Complicates the framing, usefully.** This is the paper to read if you do *not* want to build a mask-and-restore harness. Everything else on this list (energy distance, sliced Wasserstein, pMSE-against-real-data) needs either complete rows or an artificial-missingness simulation. If the library's evaluation must run on the user's actual incomplete data with no held-out truth, this is the one design that is built for it.

---

## Q6. Two-sample test vs. distance-as-score

There is no single paper titled "don't feed a p-value to an optimiser". The answer is assembled from four verified facts.

**1. A p-value is a function of the distance *and* of `n`, and only the distance is the thing you care about.** SciPy's `ks_2samp` docs state the mechanism plainly: "If the KS statistic is large, then the p-value will be small, and this may be taken as evidence against the null hypothesis." For a *fixed* true discrepancy, growing `n` drives the p-value to 0 regardless. An HPO objective built on a p-value therefore changes meaning when the row count changes — across folds, across subsampling, across datasets. The statistic does not.

**2. Floors and saturation.** SciPy's `ks_2samp` notes: for the exact two-sided computation "the minimum probability it can return is about 1e-16", and "numerical errors may accumulate for large sample sizes". A objective that bottoms out at `1e-16` gives an optimiser no gradient across the entire region where all your candidates live. `cramervonmises_2samp` has a related discreteness: "If `method='auto'`, the exact approach is used if both samples contain equal to or less than 20 observations" — the p-value is computed by a *different procedure* depending on `n`, so the objective is not even a continuous function of the data.

**3. The literature's own answer is "standardise the statistic against its known null", not "report a p-value".** Snoke et al. build the pMSE **ratio** (null expectation 1) and the **standardized** pMSE (null mean 0, null SD 1) — precisely so that a raw distance becomes comparable across `n` and `k` while remaining a continuous, unbounded, monotone score. Shadbahr et al. do the same thing empirically with the `ŵ/w` baseline ratio. **Both convert a distance into a scale-free score; neither converts it into a p-value.** That is the design pattern to copy.

**4. The classifier literature agrees, and says why.** Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests* (ICLR 2017, arXiv:1610.06545), §3, verbatim on the statistic:

> Fourth, return the classification accuracy on `D_te`: `t̂ = (1/n_te) Σ_{(z_i,l_i)∈D_te} I[ I(f(z_i) > ½) = l_i ]`

and on its null, §3.1:

> under the null hypothesis `H_0 : P = Q`, the samples […] follow the same distribution, leading to an impossible binary classification problem. In that case, `n_te · t̂` follows a `Binomial(n_te, p = ½)` distribution. Therefore, for large `n_te`, we can use the central limit theorem to approximate the null distribution of (2) by **`N(½, 1/(4n_te))`**.

Their stated motivation for preferring accuracy over the classical alternatives, from §1: existing tests "return test statistics in units that are difficult to interpret", whereas C2ST "return[s] test statistics in interpretable units, have a simple null distribution, and their predictive uncertainty allow to interpret where P and Q differ."

**`N(½, 1/(4·n_te))` is exactly what the brief asked for: a scalar with a known degenerate optimum (½) and known noise behaviour (variance `1/(4n_te)`, closed-form in the sample size).** You can compute, before running anything, how much of an observed excess over ½ is signal. That is a strictly better property than any p-value has, and it is a *statistic*, not a test.

**Summary of the answer to Q6:** consume the **distance/statistic**, never the p-value. Where the distance's scale is awkward, standardise it against its null — analytically (pMSE Eq. 1–2; C2ST's `N(½,1/(4n_te))`) or empirically by resampling a baseline (Shadbahr's `ŵ/w`; Snoke's resampled null for CART propensities, §4.3.1). Reserve the hypothesis test for the human-facing diagnostic, where "is this difference real?" is genuinely the question being asked.

---

## Q7. What transfers from synthetic-data evaluation

Almost everything, because it is the same problem: *did this generated table come from the right distribution?* Three things transfer directly.

**1. The property split (R11, SDMetrics).** Column Shapes vs. Column Pair Trends, with per-type metric routing and a strong-pattern filter on the pairwise aggregate. This is a shipped, battle-tested answer to the mixed-type problem. It is also the weakest statistically (KS marginals, pairwise-only joint) — take the *structure*, upgrade the *metrics*.

**2. The propensity/discriminator idea (R10 pMSE, R14 C2ST).** These are the same idea twice: *if a model can tell imputed rows from real rows, the imputation is wrong.* Properties that make this the strongest candidate for a joint criterion:
- Natively mixed-type — the classifier handles categorical, bounded-discrete and continuous with no distance metric to invent.
- Natively joint, at whatever order the model can represent (main effects → marginals; interactions → dependence; trees → arbitrary structure).
- Known null: pMSE `E = (k−1)(1−c)²c/N`; C2ST `N(½, 1/(4n_te))`.
- Failure modes are documented and avoidable: under-specified model → blind to dependence; over-flexible model → saturates.

**3. R14. Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests"** — *ICLR 2017, arXiv:1610.06545.* **Read exactly:** §2 (two-sample testing background, ~1.5 pp.) and §3–§3.1 (the C2ST statistic and its null). **Time: 25 minutes.** §5's use of C2ST to evaluate GAN sample quality is the direct analogue of "evaluate imputer output quality" and is worth another 10 minutes.

Verbatim on why classical tests do not transfer to tabular data: "most of these tests are only applicable to one-dimensional examples, require the prescription of a fixed representation of the data, return test statistics in units that are difficult to interpret, or do not explain how the two samples under comparison differ." And on kernel methods specifically: "kernel two-sample tests require the prescription of a manually-engineered representation of the data under study, and return values in units that are difficult to interpret."

The interpretability property is worth flagging: because C2ST is a fitted model, you get *per-row* predicted probabilities. Rows the classifier confidently calls "imputed" localise *where* the imputation failed. Neither Wasserstein nor energy distance offers that, and SDMetrics' `LogisticDetection`/`SVCDetection` ship exactly this, with `score = 1 − (max(ROC AUC, 0.5)×2 − 1)`.

SDMetrics also ships the honest warning about this family, verbatim: "**Be careful when interpreting the score.** A score of 1 may indicate high quality but it could also be a clue that the synthetic data is leaking privacy (for example, if the synthetic data is copying the rows in the real data)." The imputation analogue: a discriminator score at chance is also achieved by an imputer that *memorises and copies* observed rows. Any discriminator-based objective needs a companion check that the imputer is not merely a nearest-neighbour copier.

---

## What this adds up to

Nothing below is a recommendation about the library's design — just the shape the evidence takes.

- **Q3 is settled.** Per-cell RMSE ranks conditional-mean imputers first and correct-sampling imputers last. Three independent demonstrations (FIMD §2.6; Näf et al. Fig. 8, where the *oracle* imputer scores worst by RMSE; Shadbahr et al., where MissForest wins on RMSE and MICE wins on every distributional measure). The distributional family is the right family.
- **Wasserstein is the right *intuition* and only partly the right *metric*.** Per column (`d = 1`) it is excellent, cheap, interpretable in the variable's units, and strictly more informative than KS (`∫|U−V|` vs. `sup|U−V|`). Jointly it hits `O(n^{-1/d})`. The two papers that actually did this job avoided joint Wasserstein: one used **energy distance** (dimension-free `O(n^{-1/2})`, no tuning parameter), one used **sliced Wasserstein** (random 1-D projections). Straight joint `W_p` over a wide table is the one option with a first-party argument against it.
- **The degenerate optimum is real, is documented, and has three known answers:** joint energy distance (R8), sliced Wasserstein with a real-data baseline ratio (R9), and a propensity/discriminator score (R10, R14). All three are joint by construction. A per-column-only score, however good the per-column metric, is provably gameable.
- **The MAR ambiguity is the genuine open problem.** Observed-vs-imputed marginal divergence is *supposed* to be non-zero under MAR, so driving it to zero is itself a failure mode. The formalised fix — van Buuren §6.6.2 citing Bondarenko & Raghunathan — is to compare **conditional on the response propensity**. If ground-truth complete rows are unavailable, the I-Score of R13 is the only construction on this list built for that case.
- **For HPO, prefer statistics with known nulls over p-values.** pMSE (`E = (k−1)(1−c)²c/N`, SD in closed form), C2ST (`N(½, 1/(4n_te))`), unbiased MMD² (`E = ‖α−β‖²_k`, correctly signed around 0) all have the "known degenerate optimum with known noise" property the brief asked for. p-values do not, and floor at `1e-16`.

---

## Sources consulted

Fetched and verified against the live page or PDF.

**Books / monographs**
- Villani, C. (2009), *Optimal Transport: Old and New*, Springer Grundlehren 338 — read Ch. 6, Definitions 6.1/6.3, Remark 6.4, via author's draft PDF at <https://www.ceremade.dauphine.fr/~mischler/articles/VBook-O&N.pdf>
- Peyré, G. & Cuturi, M. (2019), *Computational Optimal Transport*, FnT ML 11(5–6):355–607 — full ToC + §2.4, §2.6 (Rem. 2.29–2.31), §8.1, §8.2 (Ex. 8.7–8.12), §8.3, §8.4, §8.5 (Prop. 8.3), via <https://arxiv.org/abs/1803.00567> v4 PDF; site <https://optimaltransport.github.io/>
- van Buuren, S., *Flexible Imputation of Missing Data*, 2nd ed. — §2.5 (<https://stefvanbuuren.name/fimd/sec-evaluation.html>), §2.6 (<https://stefvanbuuren.name/fimd/sec-true.html>), §6.6 (<https://stefvanbuuren.name/fimd/sec-diagnostics.html>), all read in full text

**Papers**
- Näf, Scornet & Josse (2026), "What Is a Good Imputation Under MAR Missingness?", arXiv:2403.19196v5 — full PDF, §1, §4, §5, §6
- Shadbahr et al. (2023), *Communications Medicine* 3:139 — full PDF via Cambridge repository, Methods A/B/C, Results, Discussion
- Snoke, Raab, Nowok, Dibben & Slavković (2017/2018), arXiv:1604.06651v2, JRSS-A 181(3):663–688 — full PDF, §3, §4.1–4.3, Table 11
- Lopez-Paz & Oquab (2017), "Revisiting Classifier Two-Sample Tests", ICLR, arXiv:1610.06545v4 — full PDF, §1, §2, §3, §3.1
- Gretton, Borgwardt, Rasch, Schölkopf & Smola (2012), "A Kernel Two-Sample Test", JMLR 13(25):723–773 — <https://www.jmlr.org/papers/v13/gretton12a.html> (abstract page)
- Näf, Spohn, Michel & Meinshausen (2023), "Imputation Scores", *AoAS* 17(3):2452–2472, arXiv:2106.03742 — abstract page
- Ramdas, García Trillos & Cuturi, arXiv:1509.02237 — abstract page
- Fournier & Guillin, arXiv:1312.2128 — abstract page (empirical-measure convergence rates, non-asymptotic `L^p` bounds for all `p>0`, `d≥1`)
- Weed & Bach, arXiv:1707.00087 — abstract page (sharp asymptotic/finite-sample rates; note the abstract emphasises **intrinsic**-dimension-dependent multi-scale behaviour, not a single `n^{-1/d}` law)

**First-party library docs (versions verified against PyPI on the day of writing)**
- SciPy **1.18.0** (current): `wasserstein_distance`, `wasserstein_distance_nd`, `energy_distance`, `ks_2samp`, `cramervonmises_2samp`, `spatial.distance.jensenshannon`
- POT **0.9.7**: quickstart + full API index
- SDMetrics **0.28.2**: full documentation corpus via `docs.sdv.dev/sdmetrics/llms-full.txt`
- Package existence/versions checked: `dcor` 0.7 ("distance correlation and energy statistics in Python"), `hyppo` 0.5.2

**Cited but not read first-hand (flagged in text)**
- Bondarenko & Raghunathan (2016), *Statistics in Medicine* 35(17):3007–3020 — paywalled; PubMed blocked automated fetch. Described via van Buuren's verbatim first-hand citation in FIMD §6.6.2 plus verified bibliographic metadata.
- Thurow, Dumpert, Ramosaj & Pauly, arXiv:2101.07532 — citation verified from Shadbahr et al.'s reference list only.

---

## Discarded

- **`scipy.stats.wasserstein_distance` as a joint metric.** It is 1-D only, and `wasserstein_distance_nd` solves the exact LP with no documented complexity bound and `O(n²)` memory in any comparable implementation. Fine for a per-column diagnostic; wrong tool for a whole-dataset score.
- **KL and Jensen–Shannon on continuous columns.** Peyré & Cuturi §8.4.2 is unambiguous: φ-divergences cannot be estimated from empirical measures — the value is `+∞` or does not converge, and you must interpose a density estimator with a bandwidth `σ` "adapted to the number `n` of samples and to the dimension `d`". That is a tuning parameter inside your tuning objective. SciPy's `jensenshannon` confirms it: it takes "probability vector", not samples. Retained only as a legitimate option on genuinely categorical columns with shared support. (Shadbahr et al. do report per-feature KL, which is why it is in the table — but they binned to get it.)
- **Total variation on continuous columns.** Same section: `‖α̂_n − β̂_n‖_TV = 2` with probability 1 on continuous samples. Kept for categorical columns, where SDMetrics' `TVComplement` shows it is the right primitive.
- **Kolmogorov–Smirnov as the *primary* per-column score.** Not wrong — it is what SDMetrics ships, and Shadbahr et al. found it ranked MICE correctly. But it uses only `sup|U−V|` where `W_1` uses `∫|U−V|` from the same object, it is documented for "continuous distributions" only, and it has no units. Retained as a cheap secondary and as the compatibility point with SDMetrics scores.
- **p-values as the optimisation objective.** See Q6. Floors at `1e-16` (SciPy's documented limit), changes meaning with `n`, and switches computational method at `n = 20` for Cramér–von Mises. The standardised-statistic pattern (pMSE ratio, C2ST accuracy vs. `N(½,1/(4n_te))`) does the same job without any of these problems.
- **Raw MMD with a Gaussian kernel as the default joint metric.** Sound and dimension-free, but P&C §8.2.2 names the blocker: "one needs to select the bandwidth parameter σ. This bandwidth should match the 'typical scale' between observations […] If the measures have multiscale features […] a Gaussian kernel is thus not well adapted." Tabular data is multiscale by nature. Energy distance is the same object with the bandwidth problem removed (Ex. 8.12: "scale-free and does not depend on a bandwidth parameter σ"). Kept MMD in the table because a *bespoke mixed-type kernel* is the natural route to a principled categorical+continuous joint metric — but then the kernel is a design decision to defend.
- **Energy distance with `p = 2`.** P&C Ex. 8.12: at `p = 2` "it degenerates to the distance between the means […] so it is not a norm anymore". Blind to variance collapse — the exact failure the whole exercise exists to catch. Use `p = 1` (Euclidean), i.e. the `‖·‖₂` form Näf et al. use.
- **The Springer/Nature landing pages for Villani (2009), Bondarenko & Raghunathan (2016) and Shadbahr et al. (2023).** All redirect through identity providers or paywalls; substituted the author's draft, van Buuren's first-hand citation, and the Cambridge open-access PDF respectively.
- **arXiv:1610.03883.** Fetched on a guess that it was a Wasserstein sample-complexity paper. It is "A method for obtaining Fibonacci identities". Discarded; noted here so nobody re-chases it.
