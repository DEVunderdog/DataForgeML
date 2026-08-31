# Evaluating imputation: multiple imputation theory, simulation design, and ground-truth-free diagnostics

A **resource guide**, not a literature review. Each entry says what the thing is, why it earns a place,
which question it answers, roughly how long it takes, and the exact section to read.

Framing being tested (the author's position):

> Imputation must not be evaluated by the predictive performance of the imputation model. The imputation
> model is disposable machinery, never reused downstream. The goal is to fill missing cells so the
> *resulting dataset* is consistent with the data's distribution, uncertainty and randomness. Want `m`
> imputed datasets, not one. Named instruments: propensity models, Monte-Carlo simulation studies, raw
> bias, percent bias, coverage rate, average width, RMSE.

**The sources back the core of this strongly and complicate it in six specific places.** Those are collected
in [Where the sources push back](#where-the-sources-push-back) and flagged inline with 🚩.

Everything quoted below was fetched and read on the live page. Paid books are marked **[PAID]** with a free
substitute where one exists.

---

## Q1. Multiple imputation theory — Rubin's rules, "proper", congeniality, choosing `m`

### 1.1 FIMD §2.3 "Why and when multiple imputation works" — **the load-bearing section**
<https://stefvanbuuren.name/fimd/sec-whyandwhen.html> · free · ~45 min

This is the one section to read if you read only one. It is *not* introductory — §2.3.2 and §2.3.3 carry the
♠ "advanced" marker in the book.

- **§2.3.1** defines the goal precisely, and it is not "a realistic dataset". It is: find an estimate `Q̂`
  of a *scientific estimand* `Q` that is **unbiased** (`E(Q̂|Y) = Q`, eq. 2.12) and **confidence valid**
  (`E(U|Y) ≥ V(Q̂|Y)`, eq. 2.13). Note the explicit exclusion: "Examples of quantities that are **not**
  scientific estimands are sample means, standard errors and test statistics."
- **§2.3.2** derives Rubin's rules from `P(Q|Y_obs) = ∫ P(Q|Y_obs,Y_mis) P(Y_mis|Y_obs) dY_mis` (eq. 2.14).
  The three formulas you need:
  - `Q̄ = (1/m) Σ Q̂_ℓ` (eq. 2.16) — pooled point estimate
  - `Ū = (1/m) Σ Ū_ℓ` (eq. 2.18) — within-imputation variance
  - `B = 1/(m−1) Σ (Q̂_ℓ − Q̄)(Q̂_ℓ − Q̄)'` (eq. 2.19) — between-imputation variance
  - `T = Ū + (1 + 1/m) B` (eq. 2.20) — total variance. **"The procedure to combine the repeated-imputation
    results by Equations (2.16) and (2.20) is referred to as Rubin's rules."**

  The `B/m` term is the whole trick: "The addition of the latter term is critical to make multiple imputation
  work at low values of `m`. Not including it would result in `p`-values that are too low, or confidence
  intervals that are too short."
- **§2.3.3 "Proper imputation"** — the exact answer to "what is proper (Rubin 1987)". An imputation procedure
  is **confidence proper** for complete-data statistics `(Q̂, U)` if at large `m`, approximately:

  ```
  E(Q̄ | Y) = Q̂                      (2.21)   pooled estimate recovers the hypothetically-complete estimate
  E(Ū | Y) = U                       (2.22)   pooled within-variance recovers the complete-data variance
  (1 + 1/m) E(B | Y) ≥ V(Q̄)          (2.23)   between-variance covers the extra uncertainty from missingness
  ```

  Replace `≥` with `>` in (2.23) and it is *proper* (stricter). "In practice, being confidence proper is
  enough to obtain valid inferences." **A method is improper when it violates any of the three** — typically
  (2.22)/(2.23), by not propagating parameter uncertainty. Deterministic regression imputation
  (`norm.predict`) is the canonical improper method: it sets `B ≈ 0`.

  🚩 **The sentence that most complicates the framing:** *"Note a procedure may be proper for the estimand
  pair `(Q̂, U)`, while being improper for another pair `(Q̂', U')`. Also, a procedure may be proper with
  respect to one response mechanism `P(R)`, but improper for an alternative mechanism `P(R')`."* Propriety is
  not a property of the imputer alone.
- **§2.3.4 "Scope of the imputation model"** — the escape hatch for a general-purpose library. Broad /
  intermediate / narrow scope. A library shipping one imputation for unknown downstream use is committing to
  **broad scope**, which van Buuren says is "preferable" for public/reusable data but weaker than narrow
  scope when "the parameters of interest have high-stakes consequences". "It is the responsibility of the
  imputer to indicate the scope of the generated imputations."
- **§2.3.5** gives `λ` (proportion of variance attributable to missing data), `r` (relative increase in
  variance), `γ` (fraction of missing information). `γ` is the input to every `m` rule below.

### 1.2 FIMD §2.4 "Statistical intervals and tests" — turning `T` into an interval
<https://stefvanbuuren.name/fimd/sec-inference.html> · free · ~10 min

Short. Gives `(Q − Q̄)/√T ~ t_ν` (eq. 2.34) and the interval `Q̄ ± t_{ν,1−α/2} √T` (eq. 2.35). You need this
to compute coverage and average width at all — those two criteria are undefined without an interval.

### 1.3 FIMD §4.5.3–4.5.4 — compatibility vs congeniality, disentangled
<https://stefvanbuuren.name/fimd/sec-FCS.html> · free · ~25 min (skip to §4.5.3)

The best short treatment of Meng's congeniality anywhere, and it explicitly untangles two words that get
used interchangeably:

- **Compatibility** = the conditionals `p(Y₁|Y₂)`, `p(Y₂|Y₁)` correspond to *some* joint distribution.
  Property of the imputation model alone, a Gibbs-sampler requirement. Reassuring news for chained equations:
  "incompatibility seems like a relatively minor problem in practice, especially if the missing data rate is
  modest and the imputation models fit the data well."
- **Congeniality** (Meng 1994) = the relation between the **imputation model and the substantive/analysis
  model**. "It is widely accepted that the imputation model should be more general than the substantive
  model." Bartlett et al. (2015) connected the two: an imputation model is congenial to the substantive model
  if the two are compatible; "Models that are incompatible may lead to biased estimates of parameters in the
  substantive model."

🚩 Congeniality is the framing's sharpest problem: it makes imputation quality **partly a function of an
analysis the library does not know about**. See also FIMD §12.1.1, which lists uncongeniality as one of the
three "major dangers": "Uncongeniality can occur if the imputation model is specified as more restrictive
than the complete-data model, or if it fails to account for important factors in the missing data mechanism.
Both types of omissions introduce biased and possibly inefficient estimates."

### 1.4 Meng (1994), "Multiple-Imputation Inferences with Uncongenial Sources of Input"
<https://projecteuclid.org/journals/statistical-science/volume-9/issue-4/Multiple-Imputation-Inferences-with-Uncongenial-Sources-of-Input/10.1214/ss/1177010269.full> ·
*Statistical Science* 9(4):538–558 · **marked Open Access on Project Euclid** · ~2 h for the whole paper

The primary source for congeniality. Read §1–§2 (setup and the definition) and the discussion of what happens
when the analyst's procedure "cannot be derived from (is 'uncongenial' to) the model adopted for multiple
imputation". The line most relevant to a library that imputes once for many downstream users, verbatim from
the abstract:

> "Given sensible imputations and complete-data analysis procedures, inferences from standard
> multiple-imputation combining rules are typically superior to, and thus different from, users'
> incomplete-data analyses."

So uncongeniality is not automatically a defect of the imputer — it can be the imputer being *better
informed*. That is the nuance FIMD §12.1.1 calls "superefficiency" (Rubin 1996).

### 1.5 FIMD §2.8 "How many imputations?" — every competing `m` rule with its justification
<https://stefvanbuuren.name/fimd/sec-howmany.html> · free · ~20 min

Directly answers "how is `m` chosen and what are the competing rules". Read the whole section; it is three
pages and it *is* the survey. Summary of the ladder, all verified on the page:

| Rule | `m` | Justification |
|---|---|---|
| Classic (Rubin 1987 p.114; Schafer 1997 p.107) | 3–5 | `T_m = (1 + γ₀/m) T_∞` (eq. 2.38). At `γ₀=0.3`, `m=5` inflates variance only 6%, the CI only 3%. |
| Royston (2004) | "at least 20 and possibly more" | Requires CV of `ln(t_ν √T)` < 0.05, i.e. CI uncertainty within ~10%. |
| Graham, Olchowski & Gilreath (2007) | 20 / 20 / 40 / 100 / >100 for `γ` = 0.1/0.3/0.5/0.7/0.9 | Statistical **power** within 1% of theoretical. At `γ₀=0.3`, `m=5` gives 73.1% power vs 78.4% theoretical. |
| Bodner (2008), linear rule | 3, 6, 12, 24, 59, 114, 258 for `γ₀` = .05/.1/.2/.3/.5/.7/.9 | CI width within 10% of its true value 95% of the time. |
| Von Hippel (2018), quadratic rule | eq. 2.39, two-step using the upper CI limit of `γ₀` | Lower `m` than Bodner when `γ₀ < 0.5`, substantially higher when `γ₀ > 0.5`. |
| **White, Royston & Wood (2011)** — *the de-facto standard* | `m ≈ 100λ` — "the number of imputations should be similar to the percentage of cases that are incomplete" | Reproducibility: Monte Carlo error of `β̂` ≈ 10% of its SE; of the test statistic ≈ 0.1; of the p-value ≈ 0.01 at a true p of 0.05. Applies up to `γ ≤ 0.5`. |

van Buuren's own operational advice, verbatim: *"It is convenient to set `m=5` during model building, and
increase `m` only after being satisfied with the model for the 'final' round of imputation. So if calculation
is not prohibitive, we may set `m` to the average percentage of missing data."*

🚩 And the caveat that complicates "I want `m` datasets, not one": *"The substantive conclusions are unlikely
to change as a result of raising `m` beyond `m=5`"*, and *"setting `m` high may not be worth the extra wait
if the primary interest is on the point estimates (and not on standard errors, p-values, and so on). In that
case using `m = 5-20` will be enough under moderate missingness."* The value of large `m` is almost entirely
in the **variance/interval**, not the point estimate. If the library's downstream consumer is a point
prediction, `m` buys much less than the framing implies.

### 1.6 `mice::pool` reference — Rubin's rules as shipped code
<https://amices.org/mice/reference/pool.html> · free · ~5 min

Worth five minutes because it names the failure mode a naive implementation walks into:

> "A common error is to reverse steps 2 and 3, i.e., to pool the multiply-imputed data instead of the
> estimates. Doing so may severely bias the estimates of scientific interest and yield incorrect statistical
> intervals and p-values. The `pool()` function will detect this case."

`rule = "rubin1987"` is the default; `"reiter2003"` is the synthetic-data variant. `dfcom` (complete-data
degrees of freedom) must be supplied or it warns "Large sample assumed". FIMD §5.1.2–5.1.3 covers the same
two anti-patterns ("Not recommended workflow: Averaging the data", "…Stack imputed data").

### 1.7 Carpenter, Bartlett, Morris, Wood, Quartagno & Kenward, *Multiple Imputation and its Application*, 2nd ed (2023) **[PAID]**
Wiley, ISBN 978-1-119-75608-8, $72 e-book · exact pointers below · ~3 h

Read **§2.8.1 "Proper imputation"**, **§2.8.2 "Congenial imputation and substantive model"**, **§2.8.3
"Uncongenial imputation and substantive models"**, **§2.9 "Constructing congenial imputation models"**
(pp. 64–73), plus **§2.6 "Choosing the number of imputations"** (p. 56). This is a more frequentist,
more proof-forward treatment than FIMD's; useful only if FIMD §2.3/§4.5.4 left you unconvinced.

**Free substitute:** FIMD §2.3.3 + §4.5.4 + Meng 1994 cover the same ground at no cost. Chapter 1 and the
full TOC are free PDFs from the Wiley product page. Do **not** use `missingdata.org.uk`, the URL widely cited
as this book's companion site — see [Discarded](#discarded).

### 1.8 Rubin (1987), *Multiple Imputation for Nonresponse in Surveys* **[PAID]**
Wiley Classics reissue, ISBN 978-0-471-65574-9, $174.95 paperback / $140 e-book · ~1 day

The original. You almost certainly do not need it — FIMD restates every result you'd use and cites the exact
pages. If you do go there, the pointers FIMD itself uses:

- **pp. 118–128** — definition of *proper* multiple imputation methods (the source of eq. 2.21–2.23)
- **p. 114** — `T_m = (1 + γ₀/m) T_∞`
- **p. 75** — the `Q̂ ~ N(Q, U)` complete-data assumption
- **p. 124** — the approximate Bayesian bootstrap (needed for Q4)
- **eq. 3.1.7 / 3.1.10 / 3.3.5** — `r`, `γ`, and the `B_∞/m` term

**Free substitute:** FIMD §2.3–§2.4 in full.

---

## Q2. Designing the Monte-Carlo simulation study

### 2.1 FIMD §2.5 "How to evaluate imputation methods" — **the exact five criteria, with a runnable worked example**
<https://stefvanbuuren.name/fimd/sec-evaluation.html> · free · ~30 min including running the R

This section *is* the author's named list. Read all three subsections.

**§2.5.1 — three simulation designs.** The choice matters more than the metrics:

1. *Sampling mechanism only.* Choose `Q`, take samples `Y^(s)`, fit, aggregate over `s`. Does not address
   missing data at all; use it "to study whether any problems are attributable to the complete-data model."
2. *Sampling and missing-data mechanisms combined.* Choose `Q`, sample, **generate incomplete data**, impute,
   estimate, aggregate over `s` and `t`. "A popular procedure… with settings `s = 1,…,1000` and `t = 1`."
   Its weakness, verbatim: "As this design does not separate the two mechanisms, any problems found may
   result from both the sampling and the missing-data mechanism."
3. *Missing-data mechanism only.* Choose `(Q̂, U)`, generate incomplete data, impute, aggregate over `t`.
   "Design 3 addresses the missing-data mechanism only, and thus allows for a more detailed assessment of any
   problem caused by the imputation step. **An advantage of this procedure is that no population model is
   needed.**" (Brand et al. 2003.)

   👉 **Design 3 is the one a library evaluating imputation on a user's real dataset should use.** It is also
   where the Vink & Van Buuren (2014) simplification applies: "we may simplify evaluation by defining the
   sample equal to the population, and set the within-variance `Ū = 0`."

**§2.5.2 — the criteria, verbatim formulas:**

| Measure | Formula | Tolerance (van Buuren, verbatim) |
|---|---|---|
| Raw bias | `RB = E(Q̄) − Q` | "RB should be close to zero." |
| Percent bias | `PB = 100 × \|(E(Q̄) − Q) / Q\|` | **"For acceptable performance we use an upper limit for PB of 5%."** (attributed to Demirtas, Freels & Yucel 2008) |
| Coverage rate | proportion of CIs containing the true value | "The actual rate should be equal to or exceed the nominal rate… **A CR below 90 percent for a nominal 95 percent interval indicates poor quality.** A high CR (e.g., 0.99) may indicate that confidence interval is too wide, so the method is inefficient… Inferences that are 'too conservative' are generally regarded a lesser sin than 'too optimistic'." |
| Average width | mean of `upper − lower` | "The length should be as small as possible, but not so small that the CR will fall below the nominal level." |
| RMSE | `RMSE = √( E(Q̄) − Q )²` — **of the estimate, not of the cells** | "a compromise between bias and variance". But: "While the RMSE is widely used, we will see in Section 2.6 that it is not a suitable metric to evaluate multiple imputation methods." |

The synthesis sentence: *"If all is well, then RB should be close to zero, and the coverage should be near
0.95. Methods having no bias and proper coverage are called **randomization-valid** (Rubin 1987b). If two
methods are both randomization-valid, the method with the shorter confidence intervals is more efficient."*

🚩 **Two things the author's list conflates.** First, `PB` carries an absolute value — it is **unsigned**, so
it cannot tell you the *direction* of the bias; you need `RB` alongside it. Second, and much more important:
the `RMSE` in this table is `√(E(Q̄) − Q)²`, the RMSE **of the pooled estimate against the true parameter**.
That is a different object from the cell-level RMSE that §2.6 demolishes. Bundling "RMSE" into a list next to
coverage and average width only makes sense for this estimate-level version — and even that one van Buuren
flags as unsuitable.

**§2.5.3 — the worked example.** ~40 lines of R (`create.data`, `make.missing`, `test.impute`, `simulate`)
that generate 1000 runs of `y = x + ε`, blow 50% MCAR holes in `x`, and compare `norm.predict` vs `norm.nob`.
This is the template to copy. Its output table:

```
                 RB   PB    CR    AW  RMSE
norm.predict  0.343 34.3 0.364 0.555 0.409
norm.nob     -0.005  0.5 0.925 0.693 0.201
```

Note what each column catches that the others miss: deterministic regression imputation is **34.3% biased**
and covers the truth **36.4%** of the time against a nominal 95%. And note the interpretive rule van Buuren
applies: *"Because of this, the estimates for AW and RMSE are not relevant."* — **AW and RMSE are only
meaningful once RB and CR have passed.** They are tiebreakers between randomization-valid methods, not
primary criteria.

### 2.2 Morris, White & Crowther (2019), "Using simulation studies to evaluate statistical methods"
*Statistics in Medicine* 38(11):2074–2102 · **free preprint** <https://arxiv.org/pdf/1712.03198> · ~2 h

The general design reference, and it supplies three things FIMD does not.

- **§3 — ADEMP.** The planning structure: **A**ims, **D**ata-generating mechanisms, **E**stimands,
  **M**ethods, **P**erformance measures. Use it as the checklist for any evaluation harness spec. §3.2 is the
  longest for a reason: "it is usual to spend more time deciding on data-generating mechanisms than any other
  element of ADEMP."
- **§5.2 + Table 6 — every performance measure with its estimator *and its Monte Carlo SE*.** This is the
  single most valuable table in the paper. It defines Bias, EmpSE, MSE, relative % increase in precision,
  Average ModSE, relative % error in ModSE, **Coverage**, **Bias-eliminated coverage**, and Rejection %.

  🚩 **"Bias-eliminated coverage"** — `Pr(θ̂_low ≤ θ̄ ≤ θ̂_upp)` — is missing from the author's list and it is
  the measure that *diagnoses which problem you have*. Plain coverage tells you the interval is wrong;
  bias-eliminated coverage (which substitutes the mean estimate `θ̄` for the truth `θ`) tells you whether that
  is because the point estimate is biased or because the variance is misestimated. §5.2 spells out the
  decomposition: "Under-coverage is to be expected if, for example, i) Bias ≠ 0, ii) ModSE < EmpSE, iii) the
  distribution of θ̂ is not normal."
- **§5.2 + §5.3 — Monte Carlo SE is mandatory, and it sets `n_sim`.** "In our review of simulation studies in
  *Statistics in Medicine* Volume 34, 93 did not mention Monte Carlo SEs for estimated performance." The
  sizing formula (eq. 1):

  ```
  n_sim = E(Coverage) × (1 − E(Coverage)) / (MonteCarloSE_req)²
  ```

  Worked: for coverage 95% and a required SE of 0.5%, `n_sim = 95×5/0.5² = 1,900` repetitions. Worst case
  (coverage 50%) needs 10,000. So FIMD's habitual `runs = 1000` is *slightly under* what Morris et al. would
  ask for a coverage-primary study.
- 🚩 **§5.2's warning on MSE**, which the author's list should absorb: "for method comparisons, the relative
  influence of bias and of variance on the MSE tends to vary with `n_obs` (except when all methods are
  unbiased), making generalisation of results difficult… **when MSE is a performance measure,
  data-generating mechanisms should include a range of values of `n_obs`.**" Figure 1 shows the crossover
  explicitly: method B beats method A on root MSE below `n_obs = 60` and loses above it. An MSE/RMSE ranking
  computed at one dataset size is not a ranking.
- **§5.2 on coverage semantics.** "Neyman's original description of confidence intervals defined the property
  of randomisation validity as *exactly* 100(1−α)% of intervals containing θ… Confidence validity is the
  property that the true percentage is *at least* 100(1−α)." Worth reading against van Buuren's "too
  conservative is a lesser sin", because Morris et al. note the two definitions disagree about whether
  over-coverage is a fault.

Read §3 (ADEMP) and §5.2 (measures + Table 6) first — about 45 minutes. §6–7 are reporting guidance and a
worked example.

---

## Q3. Why cell-level RMSE against known truth is the wrong objective

### 3.1 FIMD §2.6 "Imputation is not prediction" — **read this first-hand, it is two pages**
<https://stefvanbuuren.name/fimd/sec-true.html> · free · ~10 min

The whole section verbatim in its load-bearing parts.

The metric he is attacking (eq. 2.37), explicitly at the **cell** level:

```
RMSE = √( (1/n_mis) · Σ_{i=1}^{n_mis} ( y_i^mis − ẏ_i )² )
```

> "where `y_i^mis` represents the true (removed) data value for unit `i` and where `ẏ_i` is imputed value for
> unit `i`. For multiply imputed data we calculate RMSE for each imputed dataset, and average these."

The argument, verbatim:

> "It is well known that the minimum RMSE is attained by predicting the missing `ẏ_i` by the linear model with
> the regression weights set to their least squares estimates. According to this reasoning the 'best' method
> replaces each missing value by its most likely value under the model. However, this will find the same
> values over and over, and is single imputation. This method ignores the inherent uncertainty of the missing
> values (and acts as if they were known after all), resulting in biased estimates and invalid statistical
> inferences. **Hence, the method yielding the lowest RMSE is bad for imputation. More generally, measures
> based on similarity between the true and imputed values do not separate valid from invalid imputation
> methods.**"

**His simulation numbers** (1000 runs, `simulate2()`, same DGP as §2.5.3):

```
              RMSE
norm.predict 0.725
norm.nob     1.025
```

The flawed method wins on RMSE by ~30%. Cross-referenced against the §2.5.3 table, that same `norm.predict`
is 34.3% biased with 36.4% coverage. **That pair of tables is the entire argument, and it is the single most
useful artifact in this whole guide for justifying the design.**

Closing, verbatim: *"On the contrary, selecting such methods may be harmful as these might increase the rate
of false positives. Imputation is not prediction."*

The deeper source it names: **Gleason & Staelin (1975)** is cited as "an early paper developing this idea"
— i.e. as the *origin of the mistake*, not as support. Do not chase it expecting a defence of the practice.

### 3.2 FIMD §3.2.3 "Performance" — **the confirming simulation at `n_sim` = 10,000, and the second explicit RMSE dismissal**
<https://stefvanbuuren.name/fimd/sec-linearnormal.html> · free · ~20 min

Read this immediately after §2.6. It runs the §2.5.2 criteria over five methods at `n_sim = 10000`, and it is
where the "RMSE is misleading" claim gets stated a second time — this time about the **estimate-level** RMSE,
sitting in the same table as Bias / % Bias / Coverage / CI Width.

**Table 3.1** — 50% MCAR missingness in the *outcome* `y`, `m = 5`:

| Method | Bias | % Bias | Coverage | CI Width | RMSE |
|---|---|---|---|---|---|
| `norm.predict` | 0.0000 | 0.0 | **0.652** | 0.114 | **0.063** |
| `norm.nob` | −0.0001 | 0.0 | 0.908 | 0.226 | 0.064 |
| `norm` | −0.0001 | 0.0 | **0.951** | 0.314 | 0.066 |
| `norm.boot` | −0.0001 | 0.0 | 0.941 | 0.299 | 0.066 |
| Listwise deletion | 0.0001 | 0.0 | 0.946 | 0.251 | 0.063 |

**Table 3.2** — same, but missingness moved into the *predictor* `x`:

| Method | Bias | % Bias | Coverage | CI Width | RMSE |
|---|---|---|---|---|---|
| `norm.predict` | −0.1007 | **34.7** | **0.359** | 0.160 | 0.118 |
| `norm.nob` | 0.0006 | 0.2 | 0.924 | 0.202 | 0.056 |
| `norm` | 0.0075 | 2.6 | **0.955** | 0.254 | 0.058 |
| `norm.boot` | −0.0014 | 0.5 | 0.946 | 0.238 | 0.058 |
| Listwise deletion | −0.0001 | 0.0 | 0.946 | 0.251 | 0.063 |

Four things worth extracting:

- **Bias alone would have cleared `norm.predict` in Table 3.1.** All five methods are unbiased for `β₁` when
  only the outcome is missing. Only **coverage** separates them — 0.652 vs a nominal 0.95, "leading to
  substantial undercoverage and `p`-values that are 'too significant.'" This is the cleanest demonstration
  available that **coverage is the load-bearing criterion and bias cannot substitute for it.**
- **Moving missingness from outcome to predictor is what turns undercoverage into bias**: "Method
  `norm.predict` is now severely biased, whereas the other methods remain unbiased." Any evaluation harness
  that only amputes the target column will systematically under-report this failure mode.
- 🚩 **The RMSE verdict, verbatim and stated twice.** Table 3.1: *"Note that the RMSE values are uninformative
  for separating correct and incorrect methods, and are in fact misleading."* Table 3.2: *"The RMSE values are
  uninformative, and are only shown to illustrate that point."* Look at the columns and you can see why —
  `norm.predict` ties for the *lowest* RMSE in Table 3.1 while covering the truth 65% of the time, and the
  RMSE spread across all five methods (0.063–0.066) is smaller than Monte Carlo noise would justify caring
  about. **This is the estimate-level RMSE, not the cell-level one from §2.6.** Both are dismissed.
- **A humbling control:** listwise deletion is "in fact the most efficient choice for this problem as it
  yields the shortest confidence interval". van Buuren immediately adds "This result does not hold more
  generally. In realistic situations involving more covariates multiple imputation will rapidly catch up and
  pass complete-case analysis." Worth including complete-case as a baseline arm in any harness — if MI does
  not beat it, that is information.

### 3.3 FIMD §3.1 "How to generate multiple imputations" — the three-rung ladder
<https://stefvanbuuren.name/fimd/how-to-generate-multiple-imputations.html> · free · ~20 min

§3.1.1 *Predict* → §3.1.2 *Predict + noise* → §3.1.3 *Predict + noise + parameter uncertainty*. Only the third
rung is proper. §3.1.6 Conclusion, verbatim:

> "In summary, **prediction methods are not suitable to create multiple imputations. Both the inherent
> prediction error and the parameter uncertainty should be incorporated into the imputations.**"

This is the mechanism behind the §2.6 and §3.2.3 results, and it is the most compact statement of the
author's own position found in the literature. §3.1.5 "Drawing from the observed data" introduces the hot-deck
alternative that PMM (§3.4) generalises.

### 3.4 FIMD §9.3.2 "Don't count on predictions" — the geometric intuition
<https://stefvanbuuren.name/fimd/sec-prevalence.html> · free · ~15 min

A real case study (correcting obesity prevalence from self-reported BMI) showing *why* prediction-based
filling is biased for a distributional quantity, in pictures rather than algebra. The mechanism: because the
density is asymmetric around a threshold, "even if a symmetric normal distribution around the regression line
is correct, `n₂` is on average larger than `n₁`. This yields bias in the predictive equation." And the
scaling law: *"this effect will be stronger if the regression line becomes more shallow, or equivalently, if
the spread around the regression line increases… **Thus, predictive equations only work well if the
predictability is very high, but they are systematically biased in general.**"*

This is the best short answer to "but our imputation model has high R², so surely predictions are fine" —
high predictability is exactly the *only* regime where they are fine.

### 3.5 FIMD §12.1.2–12.1.3 — the do/don't list, as a one-page design checklist
<https://stefvanbuuren.name/fimd/sec-limitations.html> · free · ~5 min

Reads like a spec review. Directly relevant don'ts, verbatim: *"Take predictions as imputations"*,
*"Average the multiply imputed data"*, *"Create imputations using a model that is more restrictive than
needed"*, *"Uncritically accept imputations that are very different from the observed data"*, and — worth
sitting with — *"Use multiple imputation if simpler methods are valid"*.

### 3.6 scikit-learn User Guide §8.4.3.2 "Multiple vs. Single Imputation"
<https://scikit-learn.org/stable/modules/impute.html> · free · ~5 min

The first-party statement of where the engine actually stands, verbatim:

> "Our implementation of `IterativeImputer` was inspired by the R MICE package… but differs from it by
> returning a single imputation instead of multiple imputations. However, `IterativeImputer` can also be used
> for multiple imputations by applying it repeatedly to the same dataset with different random seeds when
> `sample_posterior=True`."

So `m` datasets **are** reachable from the existing engine: `m` fits with distinct `random_state` and
`sample_posterior=True`. Note also: "a call to the `transform` method of `IterativeImputer` is not allowed to
change the number of samples. Therefore multiple imputations cannot be achieved by a single call to
`transform`" — the `m` axis has to live outside the estimator.

🚩 And the sentence that most directly complicates the framing, from the same section:

> "It is still an open problem as to how useful single vs. multiple imputation is in the context of
> **prediction and classification** when the user is not interested in measuring uncertainty due to missing
> values."

The entire MI literature evaluates against a *scientific estimand* under repeated sampling. A downstream
consumer that wants one feature matrix for one model fit is a case scikit-learn declines to claim MI wins.

### 3.7 statsmodels `mice` module — the Python precedent that *does* pool
<https://www.statsmodels.org/stable/imputation.html> · free · ~10 min

Worth reading as an existence proof: `statsmodels.imputation.mice.MICE` / `MICEData` fits models across `m`
imputed datasets and "provides rigorous standard errors for the fitted parameters", using PMM by default
("even when the imputation model is linear, the PMM procedure preserves the domain of each variable"). Also
`bayes_mi.MI` / `BayesGaussMI`. This is the closest existing Python API shape to what the framing describes.

---

## Q4. Propensity models — three distinct things wearing one name

The distinctions matter, and two of the three are traps.

### (a) Propensity score for the **missingness mechanism**, used as a **diagnostic instrument** — ✅ this is the relevant one

### 4.1 FIMD §6.6.2 (second half) — the propensity-conditional diagnostic, with runnable code
<https://stefvanbuuren.name/fimd/sec-diagnostics.html> · free · ~10 min

The exact recipe, verbatim from the book:

> "Bondarenko and Raghunathan (2016) proposed a more refined diagnostic tool that aims to compare the
> distributions of observed and imputed data **conditional on the missingness probability**. The idea is that
> under MAR the conditional distributions should be similar if the assumed model for creating multiple
> imputations has a good fit."

The R, verbatim:

```r
fit <- with(imp, glm(ici(imp) ~ age + bmi + hyp + chl, family = binomial))
ps  <- rep(rowMeans(sapply(fit$analyses, fitted.values)), imp$m + 1)
xyplot(imp, bmi ~ ps | as.factor(.imp),
       xlab = "Probability that record is incomplete", ylab = "BMI",
       pch = c(1, 19), col = mdc(1:2))
```

Note the mechanics that a library implementation must reproduce: the response model is fitted **within each
imputed dataset**, and the propensities are **averaged over the `m` datasets "to obtain stability"**.

🚩 The book's own limitation, verbatim: *"Realize that the comparison is only as good as the propensity score
is. If important predictors are omitted from the response model, then we may not be able to see the potential
misfit."* A propensity-based diagnostic can silently pass a bad imputation.

### 4.2 Nguyen, Carlin & Lee (2017), "Model checking in multiple imputation: an overview and case study"
*Emerging Themes in Epidemiology* 14:8 · **fully open access**
<https://link.springer.com/article/10.1186/s12982-017-0062-6> · ~1 h

**The single best paper for both Q4(a) and Q5.** It is an explicit catalogue of ground-truth-free imputation
diagnostics, each demonstrated on one real dataset (LSAC, n=5107). Read the "Model checking methods"
subsections in order; each is 2–4 paragraphs. What it contains that FIMD does not:

- **Formalised numeric thresholds for observed-vs-imputed comparison.** "Stuart et al. proposed comparing the
  means and variances of observed and imputed values. They suggested **flagging variables if the ratio of
  variances of the observed and imputed values is less than 0.5 or greater than 2, or if the absolute
  difference in means is greater than two standard deviations.** Abayomi et al. proposed using the
  **Kolmogorov–Smirnov test**… flagged variables as potentially concerning if they had a p value below 0.05."
  With the caveat: "the results can be difficult to interpret, because the magnitude of the p-values depends
  on both the sample size and the proportion of missing values."
- **The formalised version of the propensity diagnostic** — this is the answer to "is there a numeric form of
  the Bondarenko–Raghunathan plot?" Yes: *"Bondarenko and Raghunathan propose checking continuous variables
  using **analysis of variance (ANOVA)** where the outcome variable is the variable being imputed and the
  factors are the **response stratum**, the **indicator for observed/imputed status** and their
  **interaction**. Based on empirical results from simulations, Bondarenko and Raghunathan suggest **rejecting
  an imputation model if the ANOVA test is rejected in 2 of 5 imputed datasets** (using an alpha level of
  0.05)."* That is a shippable acceptance rule.
- **Standard regression diagnostics on the imputation model**, fitted to the observed data before imputing
  (Marchenko & Eddings); and residual plots per completed dataset after (White et al.): "if problems (e.g.
  outliers) occurred in only a few of the residual plots, then this might indicate a problem with the
  imputation model. If, however, the extreme values were consistent across all datasets, then the problems
  could be attributed to the analysis model."
- **Leave-one-out cross-validation of the imputer** — note this is *cross-validation of imputation intervals*,
  not point accuracy: the plot shows median imputed value with 5th–95th percentile error bars against the
  observed value. Their read: "the prediction intervals do not always contain the observed values… the
  imputation model has poorer predictive performance at the extreme values." A calibration check, not an
  accuracy score.
- **Posterior predictive checking (PPC)**, with the framing sentence that most supports the author's position:
  *"An important feature of PPC is that it is designed to investigate the potential effect of model
  inadequacies **on the ultimate results of interest** (rather than focussing on the intermediate step of the
  quality of the imputed data values)."* Method: simulate replicated datasets from the imputation model
  (He & Zaslavsky give a practical recipe using standard MI routines), compute a test quantity in both
  completed and replicated data, and report the **posterior predictive p-value** — "the proportion of
  replications in which the estimate of the test quantity from the replicated data is larger than that
  estimated from the completed data. Posterior predictive p-values that are close to 0 or 1 indicate
  systematic differences." Their case study: PPP = 0.026 over 2000 replications, which *correctly detected an
  uncongeniality* (they imputed a continuous outcome but analysed its dichotomised version).

🚩 And the honesty flag: *"It is also important to recognise that discrepancies between observed and imputed
data are **not necessarily problematic**, since under MAR we may expect such differences to arise."*

### (b) Propensity score as an **imputation method** — ⚠️ documented as inappropriate for anything but marginals

### 4.3 SAS/STAT `PROC MI` — "Propensity Score Method for Monotone Missing Data"
<https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_mi_sect021.htm> · free · ~10 min

First-party documentation of the propensity-score *imputation* method (Lavori, Dawson & Shera 1995), and it
is unusually candid about its own limits. The algorithm: build a missingness indicator, fit a logistic
regression for `P(observed | covariates)`, sort into (typically five) propensity strata, then apply an
**approximate Bayesian bootstrap** (Rubin 1987, p.124) within each stratum — draw `n_obs` observations with
replacement to form `Y*`, then draw the imputations with replacement from `Y*`. The ABB is what makes it
proper: "This is a nonparametric analog of drawing parameters from the posterior predictive distribution."

The warning, verbatim, and it is the reason this belongs in the "trap" column:

> "The method uses only the covariate information that is associated with whether the imputed variable values
> are missing. **It does not use correlations among variables.** It is effective for inferences about the
> distributions of individual imputed variables, such as a univariate analysis, **but it is not appropriate
> for analyses involving relationship among variables, such as a regression analysis** (Schafer 1999, p. 11).
> It can also produce **badly biased estimates of regression coefficients when data on predictor variables
> are missing** (Allison 2000)."

🚩 This is a direct, first-party contradiction of a naive reading of "the goal is a dataset consistent with
the data's distribution". Propensity-stratified ABB imputation is *very good* at exactly that goal —
per-variable marginals are preserved essentially by construction, since imputations are resampled observed
values — **and it still wrecks regression coefficients**, because marginal fidelity does not imply joint
fidelity. Any library-level acceptance criterion phrased purely over marginal distributions will pass this
method.

### (c) Propensity-score **matching in causal inference** — ❌ not relevant here

Rosenbaum & Rubin (1983) propensity scores balance treatment assignment given covariates. The SAS doc above
inherits the definition ("the conditional probability of assignment to a particular treatment given a vector
of observed covariates") and then repurposes it for the response indicator. Same estimator, entirely
different estimand. FIMD Chapter 8 ("Individual causal effects") is where causal inference lives in that
book, and it does not connect to imputation evaluation. Do not let the shared vocabulary pull the causal
literature into scope.

### (d) IPW as the alternative to MI — adjacent, worth one hour of orientation

Seaman & White (2013), "Review of inverse probability weighting for dealing with missing data", *Statistical
Methods in Medical Research* 22(3):278–95, DOI `10.1177/0962280210395740` — **paywalled at SAGE**; PubMed
record PMID 21220355. Same propensity model, used to *weight complete cases* rather than to impute. Relevant
only for the framing "MI is one of two ways to spend a missingness model"; skip unless that choice is live.
FIMD §1.6.2 ("Weighting procedures") gives a one-paragraph free orientation and explicitly excludes the topic
from the book.

---

## Q5. Ground-truth-free diagnostics

### 5.1 FIMD §6.6 "Diagnostics" — **the conceptual pivot, and the shortest thing here worth reading twice**
<https://stefvanbuuren.name/fimd/sec-diagnostics.html> · free · ~20 min

**§6.6.1 "Model fit versus distributional discrepancy"** contains the reframe. Verbatim:

> "Conventional model evaluation concentrates on the fit between the data and the model. In imputation it is
> often more informative to focus on **distributional discrepancy**, the difference between the observed and
> imputed data."

And the worm-plot demonstration that these come apart in practice: after PMM imputation of body weight, "The
fit between the observed data and the imputation model is bad… In contrast to this, the red and blue worms
are generally close… **Thus, despite the fact that the model does not fit the data, the distributions of the
observed and imputed data are similar. This distributional similarity is more relevant for the final
inferences than model fit per se.**"

That paragraph is the strongest single endorsement in the primary literature of the author's core position.

**§6.6.2 "Diagnostic graphs"** gives the standard toolkit — `bwplot()`, `stripplot()`, `densityplot()`,
`xyplot()` — plus a taxonomy of what to look for (different means / spreads / scales / relations /
non-overlapping-and-defying-common-sense), each cross-referenced to a specific figure.

🚩 **But read the caveat carefully, because it undercuts a naive "match the distribution" objective**:

> "The idea is that good imputations have a distribution similar to the observed data… **Except under MCAR,
> the distributions do not need to be identical, since strong MAR mechanisms may induce systematic
> differences between the two distributions.** However, any dramatic differences between the imputed and
> observed data should certainly alert us to the possibility that something is wrong."

And the closing sentence names the irreducible ambiguity: "Alternatively, it could be the case that the
observed differences are justified, and that the missing data process is MNAR. **The art of imputation is to
distinguish between these two explanations.**" No ground-truth-free diagnostic resolves that.

### 5.2 Nguyen, Carlin & Lee (2017) — see [§4.2](#42-nguyen-carlin--lee-2017-model-checking-in-multiple-imputation-an-overview-and-case-study)
It is the formalised, thresholded version of FIMD §6.6, and it is the paper to implement from.

### 5.3 FIMD §6.5.2 "Convergence" and §4.5.6 "Number of iterations"
<https://stefvanbuuren.name/fimd/sec-algoptions.html> · <https://stefvanbuuren.name/fimd/sec-FCS.html> · free · ~15 min

The other ground-truth-free check, and the one specific to chained equations: trace plots of the mean and SD
of imputed values per stream per iteration, inspected for trend and for between-stream separation. FIMD
§4.5.7 gives the worked slow-convergence example. Non-convergence is a defect you can detect with no truth
and no downstream model at all — the cheapest diagnostic in the set.

### 5.4 `mice` JSS paper (van Buuren & Groothuis-Oudshoorn 2011)
*Journal of Statistical Software* 45(3) · **free** <https://www.jstatsoft.org/article/view/v045i03> · ~1.5 h

First-party, peer-reviewed, and it ships the R example code as a separate download. Its value here is as the
API-design reference for what a diagnostics surface looks like when someone has already built it: "diagnostic
graphs", "specialized pooling routines", "the proper setup of the predictor matrix". Read the sections on
diagnostics and pooling; skip the multilevel material.

---

## Q6. Simulating missingness, and the honest limits of amputation

### 6.1 `mice::ampute` vignette — "Generate missing values with ampute"
<https://rianneschouten.github.io/mice_ampute/vignette/ampute.html> · free · ~45 min

The first-party tutorial by the method's author, and the most efficient path into the procedure. Read
straight through; every argument gets its own worked section with plots.

The procedure (verbatim summary from the vignette): split the complete data into `k` subsets, one per
**missingness pattern**; within each subset compute a **weighted sum score**

```
wss_i = w_{k,1}·y_{1,i} + w_{k,2}·y_{2,i} + … + w_{k,m}·y_{m,i}
```

then map that score to an amputation probability through one of four logistic distribution **types**, and
ampute `prop` of the rows. Six arguments carry the whole design:

| Arg | What it controls |
|---|---|
| `prop` | proportion of incomplete **rows** by default; set `bycases = FALSE` for proportion of missing **cells** |
| `patterns` | `k × m` 0/1 matrix — 0 = becomes missing, 1 = stays complete. Default: one pattern per variable, so *no case has missingness on more than one variable* |
| `freq` | relative size of each pattern's subset; must sum to 1 |
| `mech` | `"MCAR"` / `"MAR"` / `"MNAR"` |
| `weights` | `k × m` matrix of `wss` coefficients — **the actual mechanism knob**; a 0 excludes a variable from driving missingness. "The weights matrix can also be used to switch from MAR to MNAR missingness, or even to create a combined mechanism" |
| `type` | `RIGHT` / `LEFT` / `MID` / `TAIL` logistic — high / low / average / extreme sum scores get the higher amputation probability. `cont = FALSE` + `odds` gives manual per-quantile probabilities instead |

How `mech` actually works, which is the thing worth internalising: it is just a default `weights` matrix.
MAR ⇒ weights of 1 on the variables that stay **complete** in that pattern; MNAR ⇒ weights of 1 on the
variables that **become missing**. Same machinery, different column mask.

**Limits stated in the vignette itself:**
- 🚩 "Generation of both MCAR and MAR missingness (or any other form of weak MAR) is currently **not directly
  possible** with `ampute`." The workaround is to ampute twice and splice rows by a random allocation vector
  — code is given.
- The default pattern matrix produces **no multivariate missingness at all** (one variable missing per case).
  A realistic simulation must override `patterns`.
- `prop` means rows, not cells, unless you say otherwise — an easy way to design a study at the wrong severity.

**Python:** the vignette points to `pyampute` (`MultivariateAmputation`, plus `mdPatterns` and `MCARTest`) —
<https://rianneschouten.github.io/pyampute/build/html/index.html>. Same author. This is the direct dependency
candidate if the library builds an amputation harness.

### 6.2 Schouten, Lugtig & Vink (2018), "Generating missing values for simulation purposes: a multivariate amputation procedure"
*Journal of Statistical Computation and Simulation* 88(15):2909–2930 · **paywalled at Taylor & Francis**
(DOI `10.1080/00949655.2018.1491577`) · ~1.5 h

The method paper. Read it only if you need the derivation of the logistic type functions or the justification
for pattern-wise subsetting; the vignette above is by the same author and covers the operational surface.
Builds on Brand (1999). **Free substitute: the vignette in §6.1** — it is not a summary, it is a full tutorial
with the formulas.

### 6.3 Schouten & Vink (2021), "The Dance of the Mechanisms" — 🚩 **the honest-limits paper, and it is open access**
*Sociological Methods & Research* 50(3):1243–1258 · **open access**
<https://journals.sagepub.com/doi/10.1177/0049124118799376> · ~1 h

This is the source that most sharply limits what an amputation study can prove. Abstract, verbatim:

> "we simulate complete data and generate missing values according several types of MCAR, MAR, and MNAR
> mechanisms. **We demonstrate that in scenarios where the data correlations are either low or very
> substantial, strictly different mechanisms yield equivalent statistical inferences.**"

The practical consequence for an evaluation harness: a study that amputes under MAR and under MNAR and finds
no difference has **not** shown the imputer is robust to MNAR. It may only have shown the dataset's
correlation structure sits in a regime where the mechanisms are inferentially indistinguishable. Mechanism
sensitivity must be reported *conditional on the correlation regime*, or it is uninterpretable.

### 6.4 FIMD §3.2.4–3.2.5 "Generating MAR missing data" (univariate and multivariate)
<https://stefvanbuuren.name/fimd/sec-linearnormal.html> · free · ~15 min

The pre-`ampute` hand-rolled recipe, useful as a sanity reference for what a minimal MAR generator looks like
before reaching for a dependency.

### 6.5 Vink & van Buuren (2014), "Pooling multiple imputations when the sample happens to be the population"
<https://arxiv.org/abs/1409.8542> · **free preprint, 6 pages** · ~20 min

Short, and it fixes a specific bug an amputation harness will otherwise ship. When you ampute a dataset you
already hold in full, you are in FIMD's **design 3** (missing-data mechanism only) — the sample *is* the
population and there is no sampling variance. Abstract, verbatim:

> "Using the standard pooling rules in situations where sampling variance should not be considered, leads to
> **overestimation of the variance of the estimates of interest**, especially when the amount of missingness
> is not very large. As a result, populations estimates are **over-covered**, which may lead to a loss of
> statistical power… The simplified pooling rules can be easily implemented to obtain valid inference in
> cases where we have observed essentially all units and **in simulation studies addressing the missingness
> mechanism only**."

🚩 So: if the harness amputes a held dataset, computes `T = Ū + (1+1/m)B` with the usual `Ū`, and then reports
coverage, **the coverage number is biased upward and the method will look better than it is.** Set `Ū = 0`
(FIMD §2.5.1 says exactly this) or use the simplified rules from this paper.

### 6.6 FIMD §2.5.3, closing line — the ceiling on all of this
> "Of course, simulation studies do not guarantee fitness for a particular application. However, if simulation
> studies illustrate limitations in simple examples, we may expect these will also be present in
> applications."

Amputation studies are **falsifiers, not certifiers**. They can prove a method is broken; they cannot prove
one is fit for a user's data.

---

## Where the sources push back

Ordered by how much they should change the design.

1. **🚩 "Proper" is not a property of an imputer — it is a property of an imputer *paired with an estimand and
   a response mechanism.*** FIMD §2.3.3, verbatim: "a procedure may be proper for the estimand pair `(Q̂, U)`,
   while being improper for another pair `(Q̂', U')`. Also, a procedure may be proper with respect to one
   response mechanism `P(R)`, but improper for an alternative mechanism `P(R')`." A library that evaluates
   imputation without naming an estimand cannot report bias, coverage, or width at all — those four of the
   five named criteria are all functions of `Q`. The available moves are: (a) name a default estimand set
   (column means, variances, pairwise correlations, coefficients of a reference regression) and report per
   estimand; or (b) declare **broad scope** in van Buuren's §2.3.4 sense and say so explicitly, which he
   frames as the imputer's responsibility.

2. **🚩 Congeniality means imputation quality is partly a function of a downstream analysis the library cannot
   see.** Meng 1994; FIMD §4.5.4; FIMD §12.1.1 lists uncongeniality among the "major dangers"; Carpenter et
   al. §2.8.2–2.8.3. The framing's "the imputation model is disposable machinery, never reused downstream" is
   correct about the *artifact* and incomplete about the *evaluation*. Nguyen et al.'s PPC case study is the
   concrete demonstration: their diagnostic fired (PPP = 0.026) precisely because the imputation model was
   uncongenial to the analysis model, and no marginal-distribution check would have caught it. **Mitigation:
   PPC is the one diagnostic in this guide that tests congeniality without ground truth, and it can be run
   against a user-supplied reference analysis.**

3. **🚩 Marginal distributional fidelity is neither necessary nor sufficient.** *Not necessary*: FIMD §6.6.2,
   "Except under MCAR, the distributions do not need to be identical, since strong MAR mechanisms may induce
   systematic differences." Nguyen et al.: "discrepancies between observed and imputed data are not
   necessarily problematic." *Not sufficient*: SAS's propensity-score/ABB method resamples observed values, so
   it preserves marginals almost by construction, and SAS still documents that it "is not appropriate for
   analyses involving relationship among variables" and "can also produce badly biased estimates of regression
   coefficients." The correct conditional form is the Bondarenko–Raghunathan one — compare observed vs imputed
   **conditional on response propensity** — and even that is "only as good as the propensity score is."

4. **🚩 "RMSE" in the author's list is two different metrics, and van Buuren rejects both — for different
   reasons.** Cell-level RMSE (FIMD eq. 2.37) is actively *anti*-correlated with validity: `norm.predict`
   scores 0.725 vs `norm.nob`'s 1.025 while being 34.3% biased with 36.4% coverage. Estimate-level RMSE
   (`√(E(Q̄)−Q)²`, FIMD §2.5.2) fares no better in his hands: §2.5.2 calls it "not a suitable metric to
   evaluate multiple imputation methods"; §2.5.3 declares "the estimates for AW and RMSE are not relevant"
   once coverage has failed; and §3.2.3, at `n_sim = 10,000`, says it outright twice — "the RMSE values are
   **uninformative** for separating correct and incorrect methods, and are in fact **misleading**" and "The
   RMSE values are uninformative, and are only shown to illustrate that point." In his Table 3.1 the
   improper method ties for the *lowest* estimate-level RMSE while covering the truth 65.2% of the time.
   Morris et al. add an independent objection: MSE rankings flip with `n_obs` (their Figure 1), so an MSE
   comparison at a single dataset size is not a ranking. **Design implication: gate on RB and CR first;
   report AW and RMSE only for methods that already pass — and never let RMSE break a tie by itself.**

5. **🚩 The `m`-datasets requirement buys standard errors, not point estimates — and the framing's downstream
   consumer may only want point estimates.** FIMD §2.8: "The substantive conclusions are unlikely to change as
   a result of raising `m` beyond `m=5`"; "setting `m` high may not be worth the extra wait if the primary
   interest is on the point estimates." scikit-learn §8.4.3.2, first-party: "It is still an open problem as to
   how useful single vs. multiple imputation is in the context of prediction and classification when the user
   is not interested in measuring uncertainty due to missing values." Both statements are compatible with
   wanting `m` — but they mean the *justification* has to be "so we can compute and report the uncertainty",
   not "so the point estimates are better." Getting the justification right also picks the `m` rule: for
   reproducible intervals, White et al.'s `m ≈ 100λ` is the de-facto standard.

6. **🚩 "The imputation model is never reused downstream" is false in the train/serve setting this library
   actually occupies.** `IterativeImputer` stores per-feature estimators at `fit` and replays them at
   `transform` (its `imputation_sequence_`) — the machinery is explicitly *not* disposable in inductive mode.
   Combined with the fact that `transform` "is not allowed to change the number of samples", the `m` axis
   cannot live inside a single fitted estimator and must be an outer loop of `m` fits with distinct seeds and
   `sample_posterior=True`. Worth stating explicitly in the design so the two claims don't silently conflict.

**Two smaller gaps in the named-criteria list, both cheap to close:**

- **Bias-eliminated coverage** (Morris et al. Table 6) — `Pr(θ̂_low ≤ θ̄ ≤ θ̂_upp)`. Plain coverage says the
  interval is wrong; this one says *why* (biased point estimate vs misestimated variance). One extra column.
- **Monte Carlo SE on every reported performance number** (Morris et al. §5.2, and their finding that 93
  studies in one volume omitted it). Also sets `n_sim`: 1,900 repetitions for a coverage MCSE of 0.5%, which
  is above FIMD's habitual 1,000.

---

## Sources consulted

Every URL below was fetched and the quoted claim verified against the live page on 2026-08-16.

**van Buuren, *Flexible Imputation of Missing Data*, 2nd ed — free at <https://stefvanbuuren.name/fimd/>**
| Section | URL | Used for |
|---|---|---|
| §2.3 Why and when MI works | [sec-whyandwhen](https://stefvanbuuren.name/fimd/sec-whyandwhen.html) | Rubin's rules eq. 2.16–2.20; proper imputation eq. 2.21–2.23; scope; `λ`, `r`, `γ` |
| §2.4 Statistical intervals and tests | [sec-inference](https://stefvanbuuren.name/fimd/sec-inference.html) | `t_ν` interval eq. 2.34–2.36 |
| §2.5 How to evaluate imputation methods | [sec-evaluation](https://stefvanbuuren.name/fimd/sec-evaluation.html) | three simulation designs; RB/PB/CR/AW/RMSE formulas + 5% and 90% tolerances; worked R example and its output table |
| §2.6 Imputation is not prediction | [sec-true](https://stefvanbuuren.name/fimd/sec-true.html) | eq. 2.37; the 0.725 vs 1.025 simulation; Gleason & Staelin 1975 attribution |
| §2.8 How many imputations? | [sec-howmany](https://stefvanbuuren.name/fimd/sec-howmany.html) | all six `m` rules with numbers |
| §3.1 How to generate multiple imputations | [how-to-generate…](https://stefvanbuuren.name/fimd/how-to-generate-multiple-imputations.html) | the three-rung ladder; §3.1.6 "prediction methods are not suitable to create multiple imputations" |
| §3.2 Imputation under the normal linear normal | [sec-linearnormal](https://stefvanbuuren.name/fimd/sec-linearnormal.html) | §3.2.3 Tables 3.1 and 3.2 at `n_sim`=10,000; both "RMSE is uninformative/misleading" statements; §3.2.4–3.2.5 MAR generation |
| §4.5 Fully conditional specification | [sec-FCS](https://stefvanbuuren.name/fimd/sec-FCS.html) | §4.5.3 compatibility, §4.5.4 congeniality vs compatibility |
| §6.5 Algorithmic options | [sec-algoptions](https://stefvanbuuren.name/fimd/sec-algoptions.html) | §6.5.2 Convergence (trace-line inspection) |
| §6.3 Model form and predictors | [sec-modelform](https://stefvanbuuren.name/fimd/sec-modelform.html) | predictor-selection context (also used in `regression-vs-mice-literature.md`) |
| §6.6 Diagnostics | [sec-diagnostics](https://stefvanbuuren.name/fimd/sec-diagnostics.html) | model fit vs distributional discrepancy; worm plot; `bwplot`/`stripplot`/`densityplot`/`xyplot`; Bondarenko–Raghunathan propensity plot + R code |
| §9.3 Correct prevalence from self-report | [sec-prevalence](https://stefvanbuuren.name/fimd/sec-prevalence.html) | §9.3.2 "Don't count on predictions" geometric argument |
| §12.1 Dangers, do's and don'ts | [sec-limitations](https://stefvanbuuren.name/fimd/sec-limitations.html) | uncongeniality as a major danger; the don't-list |

**Papers**
- Morris, White & Crowther (2019), *Stat Med* 38(11):2074–2102 — free preprint <https://arxiv.org/pdf/1712.03198>. Verified: ADEMP (§3), Table 4 (measure prevalence in the reviewed volume), Table 6 (definitions + Monte Carlo SEs), the MSE/`n_obs` caveat, bias-eliminated coverage, `n_sim` formula (eq. 1) and the 1,900 / 10,000 figures, and the Neyman randomisation-vs-confidence-validity distinction.
- Meng (1994), *Statistical Science* 9(4):538–558 — <https://projecteuclid.org/journals/statistical-science/volume-9/issue-4/Multiple-Imputation-Inferences-with-Uncongenial-Sources-of-Input/10.1214/ss/1177010269.full>. Verified: marked **Open Access**, page range, abstract quoted verbatim.
- Nguyen, Carlin & Lee (2017), *Emerging Themes in Epidemiology* 14:8 — <https://link.springer.com/article/10.1186/s12982-017-0062-6>. Fully open access. Verified: Stuart et al. variance-ratio 0.5/2 and 2-SD mean thresholds; Abayomi KS test at p<0.05; Bondarenko–Raghunathan ANOVA design and the "reject if 2 of 5" rule; LOOCV plot; PPC method, posterior-predictive p-value definition, PPP = 0.026 result and the uncongeniality it exposed; the "discrepancies are not necessarily problematic" caveat.
- Schouten & Vink (2021), *Sociological Methods & Research* 50(3):1243–1258 — <https://journals.sagepub.com/doi/10.1177/0049124118799376>. Verified open access; abstract quoted verbatim.
- Vink & van Buuren (2014) — <https://arxiv.org/abs/1409.8542>. Verified: abstract quoted verbatim; 6 pages, free.
- van Buuren & Groothuis-Oudshoorn (2011), *JSS* 45(3) — <https://www.jstatsoft.org/article/view/v045i03>. Verified free, abstract read, code supplement listed.

**Package / first-party documentation**
- `mice::ampute` vignette — <https://rianneschouten.github.io/mice_ampute/vignette/ampute.html>. Verified: `wss` formula; all six arguments; the MAR-vs-MNAR default weights behaviour; `bycases`; the `odds`/`cont=FALSE` path; the explicit "MCAR+MAR not directly possible" limitation and its workaround code; the `pyampute` pointer.
- `mice::ampute` reference page — <https://amices.org/mice/reference/ampute.html>.
- `mice::pool` reference page — <https://amices.org/mice/reference/pool.html>. Verified: `rule="rubin1987"` default, `dfcom` behaviour, the pool-the-estimates-not-the-data warning.
- scikit-learn User Guide, Imputation — <https://scikit-learn.org/stable/modules/impute.html>. Verified §8.4.3.2 verbatim, including the `sample_posterior=True` multiple-imputation route and the open-problem statement.
- statsmodels MICE — <https://www.statsmodels.org/stable/imputation.html>. Verified `MICE`/`MICEData`/`bayes_mi.MI`/`BayesGaussMI` and the PMM-preserves-domain claim.
- SAS/STAT `PROC MI`, Propensity Score Method — <https://support.sas.com/documentation/cdl/en/statug/63347/HTML/default/statug_mi_sect021.htm>. Verified the full algorithm (logistic response model → 5 strata → approximate Bayesian bootstrap) and the "not appropriate for analyses involving relationship among variables" / "badly biased estimates of regression coefficients" warning, with its Schafer 1999 and Allison 2000 attributions.
- `pyampute` — <https://rianneschouten.github.io/pyampute/build/html/index.html>. Verified `MultivariateAmputation`, `mdPatterns`, `MCARTest`.

**Paid books — pointers verified from publisher pages, contents not read first-hand**
- Little & Rubin, *Statistical Analysis with Missing Data*, 3rd ed (2019), Wiley ISBN 978-0-470-52679-8, $87 e-book / $108.95 hardcover, 462pp — <https://www.wiley.com/en-us/Statistical+Analysis+with+Missing+Data%2C+3rd+Edition-p-9780470526798>. Free Chapter 1, TOC and Index PDFs available from that page. The 3rd edition's new material explicitly includes "diagnostic methods, and sensitivity analysis".
- Carpenter, Bartlett, Morris, Wood, Quartagno & Kenward, *Multiple Imputation and its Application*, 2nd ed (2023), Wiley ISBN 978-1-119-75608-8, $72 e-book, 464pp. **TOC PDF read** — confirmed §2.6 (choosing `m`, p.56), §2.8.1 Proper imputation (p.64), §2.8.2 Congenial imputation and substantive model (p.64), §2.8.3 Uncongenial imputation and substantive models (p.65), §2.9 Constructing congenial imputation models (p.72).
- Rubin (1987), *Multiple Imputation for Nonresponse in Surveys*, Wiley Classics reissue ISBN 978-0-471-65574-9, $174.95 paperback, 320pp. Wiley Online Library TOC returned HTTP 402; the page pointers listed in §1.8 above are **taken from FIMD's citations**, not verified against the book itself. Flagged as such.
- Schafer (1997), *Analysis of Incomplete Multivariate Data*, Chapman & Hall. Not accessed. The p.107 quote used in §1.5 ("the additional resources that would be required to create and store more than a few imputations would not be well spent") is quoted **from FIMD §2.8**, which cites it directly. Flagged as second-hand.
- Seaman & White (2013), *Stat Methods Med Res* 22(3):278–95, DOI `10.1177/0962280210395740` — paywalled at SAGE; abstract verified via search, full text not read.
- Schouten, Lugtig & Vink (2018), *J Stat Comput Simul* 88(15):2909–2930, DOI `10.1080/00949655.2018.1491577` — Taylor & Francis returned HTTP 403. Content characterised from the author's own vignette (§6.1), which is first-party and covers the same procedure.

---

## Discarded

- **`missingdata.org.uk`** — the URL widely cited as the companion site for Carpenter & Kenward. Fetched: the
  domain now serves a UK medical-cannabis marketing blog. **Do not cite it.** Use the Wiley product page for
  free chapter/TOC/index PDFs instead.
- **`https://amices.org/mice/articles/ampute.html`** — the `ampute` vignette URL on the canonical `mice`
  pkgdown site returns HTTP 404. The live copy is at `rianneschouten.github.io/mice_ampute/vignette/ampute.html`
  (the author's own site) and is what the `mice` docs link to.
- **Gleason & Staelin (1975)** — cited by FIMD §2.6 as "an early paper developing this idea". Not pursued: it
  is cited as the *origin of the error being refuted*, so reading it adds nothing to the case against
  cell-level RMSE and risks being mistaken for support.
- **Wiley Online Library TOC for Rubin (1987)** (`doi/book/10.1002/9780470316696`) — HTTP 402 Payment
  Required. Chapter/section titles could not be verified; FIMD's page-level citations are used instead and
  labelled as such.
- **Citation correction for the sibling doc.** `docs/research/regression-vs-mice-literature.md` attributes the
  `norm.predict` numbers — coverage 0.652, bias 34.7%, "a confidence interval of method `norm.predict` is much
  too short, leading to substantial undercoverage", "It is always better to include parameter uncertainty" —
  to **FIMD §1.3**. Fetched and checked: `sec-simplesolutions.html` (§1.3) contains none of them. They are in
  **§3.2.3 "Performance"** (`sec-linearnormal.html`, Tables 3.1 and 3.2), and the parameter-uncertainty line is
  a paraphrase of **§3.1.6**, whose actual wording is "prediction methods are not suitable to create multiple
  imputations. Both the inherent prediction error and the parameter uncertainty should be incorporated into
  the imputations." The claims are all correct; only the pointers are wrong. Worth fixing there.
- **The pirated full-text FIMD PDF** surfaced by search on a `.dstu.dp.ua` host — ignored. The 2nd edition is
  legitimately free on the author's own site and that is what was used throughout.
- **FIMD Chapter 8 "Individual causal effects"** — checked for relevance to the propensity question and
  rejected. It concerns imputing individual causal effects under the Rubin causal model; it does not connect
  to propensity models as an *evaluation* instrument, and pulling it in would re-import the causal-inference
  sense of "propensity score" that Q4 explicitly separates out.
- **FIMD §5.3 (`D₁`/`D₂`/`D₃` multi-parameter tests)** — real and correct, but out of scope. It matters when
  the estimand is a vector and you need one simultaneous test; every criterion in the author's list is
  scalar-per-estimand. Noted here so it can be found later if vector estimands come into scope.
- **ResearchGate and Semantic Scholar copies** of Bondarenko & Raghunathan (2016) and Seaman & White (2013) —
  not used as sources. The Deep Blue (UMich) repository copy of Bondarenko & Raghunathan returned HTTP 403.
  The method is therefore reported **through** Nguyen et al. (2017), which is open access, describes the
  procedure in operational detail, and states the 2-of-5 acceptance rule explicitly — flagged as second-hand
  in §4.2.
