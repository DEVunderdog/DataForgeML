# Hyperparameter optimisation when the objective is a multiply-imputed dataset

A resource guide for the layer *above* `docs/learning/hyperparameter-optimization-plan.md` and
`docs/learning/evaluation-and-validation-plan.md`. Those two plans teach HPO as a search loop over a
deterministic cross-validation score of a predictive model. This document is about what breaks when
that premise is dropped.

**The premise being dropped.** HPO for imputation must not optimise the imputation model's predictive
performance. The imputation model is disposable. What is optimised is the effect the imputation has on
the *resulting dataset* — its distribution, its uncertainty, its randomness. The output of a run is `m`
multiply-imputed datasets, not one. Candidate objectives (distributional distance, coverage rate,
percent bias, average interval width) are computed from a Monte-Carlo simulation over amputed data,
which makes the objective **stochastic**, **expensive**, and **vector-valued with internal conflicts**.

Three consequences drive everything below:

1. The objective is a noisy estimate of an integral, not a number. Every trial's value carries
   Monte-Carlo error. Standard Bayesian-optimisation machinery assumes noise-free or mildly-noisy
   observations, and the *default acquisition function of the whole field* (expected improvement)
   is the one that degrades worst under noise.
2. Bias, coverage, width and distributional fidelity genuinely trade off. There is no scalarisation
   that is safe by default, but there *is* a proper scoring rule that legitimately fuses coverage and
   width — and finding it is worth more than any amount of Pareto tooling.
3. The natural fidelity axis is "number of Monte-Carlo replications", not "training epochs". Whether
   Hyperband-family machinery transfers to that axis is a real question, and it has a well-developed
   answer in a literature (algorithm configuration: SMAC, irace) that neither existing curriculum
   mentions.

Every URL below was fetched and checked against the live page before it was written down. Version
numbers were read from PyPI and from the installed wheel, not recalled.

---

## Part 1 — HPO when the objective is stochastic

**What you should be able to say afterwards:** why the fact that your objective is a Monte-Carlo mean
changes the *acquisition function*, not just the trial count; what the GP's nugget term is and which
switch in Optuna controls it; and why TPE's rank-based construction reacts differently to noise than a
GP does.

### Frazier, *A Tutorial on Bayesian Optimization*, Section 5 — "Noisy Evaluations" and "Random Environmental Conditions"

- **What it is:** arXiv:1807.02811, Peter Frazier (Cornell), July 2018. The existing HPO plan already
  assigns this at Stage 4 as the rigorous companion to the Distill piece — but it assigns the *wrong
  part*. Stage 4 sends you to the GP-regression and acquisition-function chapters. The section you
  actually need is **Section 5, "Exotic Bayesian Optimization"**, which the plan never mentions.
- **Why it earns its place:** it contains, in two pages, both halves of this project's Q1.

  First, the noise treatment, verbatim: *"GP regression can be extended naturally to observations with
  independent normally distributed noise of known variance (Rasmussen and Williams, 2006). This adds a
  diagonal term with entries equal to the variance of the noise to the covariance matrices in (3). In
  practice, this variance is not known, and so the most common approach is to assume that the noise is
  of common variance and to include this variance as a hyperparameter."* That diagonal term is the
  nugget. Frazier also notes the heteroskedastic option (model the log-variance with a second GP,
  Kersting et al. 2007) — relevant here because MC error in a coverage estimate is not constant across
  the hyperparameter space.

  Second, and this is the sentence that should change your design: *"Direct use of the EI acquisition
  function presents conceptual challenges, however, since the 'improvement' that results from a
  function value is no longer easily defined, and f(x) in (7) is no longer observed."* He concludes:
  *"KG can outperform EI substantially in problems with substantial noise."* The plain reading is that
  **expected improvement, the acquisition function all three of Stage 3–4's resources build intuition
  around, is the one that stops making sense on this problem.** Knowledge-gradient, entropy search and
  predictive entropy search *"apply directly in the setting with noise and they retain their one-step
  optimality properties."*

  Third — and this is the part almost nobody reads — the subsection **"Random Environmental Conditions
  and Multi-Task Bayesian Optimization"** states this project's objective in its exact mathematical
  form: maximise `∫ f(x, w) p(w) dw` where `f` is expensive. Substitute `w` = one Monte-Carlo
  replication (one amputation draw + one seed) and `p(w)` = the amputation mechanism, and that *is*
  the imputation-simulation objective. Frazier's guidance on it is directly actionable: *"Rather than
  taking the objective ∫f(x,w)p(w)dw as our unit of evaluation, a natural approach is to evaluate
  f(x,w) at a small number of w at an x of interest. This gives partial information about the objective
  at x. Based on this information, one can explore a different x, or resolve the current x with more
  precision."* That single sentence is the whole design pattern for Part 3.
- **Question it answers:** Q1 entirely, and it pre-frames Q3.
- **Time:** ~25 min for Section 5 alone (pp. 12–15). Do not re-read Sections 3–4; the HPO plan already
  sent you there.
- **Exact section:** §5, subsections "Noisy Evaluations", "Random Environmental Conditions and
  Multi-Task Bayesian Optimization". <https://arxiv.org/abs/1807.02811>

### Toscano-Palmerin & Frazier, *Bayesian Optimization with Expensive Integrands*

- **What it is:** arXiv:1803.08661 (2018), the paper Frazier's §5 points to for the environmental-
  conditions problem.
- **Why it earns its place:** it is the one paper whose abstract describes this exact objective shape.
  Verbatim: *"We propose a Bayesian optimization algorithm for objective functions that are sums or
  integrals of expensive-to-evaluate functions, allowing noisy evaluations. These objective functions
  arise in multi-task Bayesian optimization for tuning machine learning hyperparameters, optimization
  via simulation, and sequential design of experiments with random environmental conditions."* It also
  reports that the method *"offers significant improvements when evaluations are noisy or the integrand
  varies smoothly in the integrated variables"* — which is the empirical case for spending replications
  adaptively rather than fixing `n_sim` up front.
- **Question it answers:** is "spend fewer replications on unpromising configurations" principled, or
  just a budget hack? (Principled; it is average-case optimal by construction under a one-evaluation
  budget.)
- **Caveat, stated plainly:** there is no maintained first-party implementation of this to depend on.
  Read it for the framing and the justification, then implement the cheap approximation (Part 3).
- **Time:** ~20 min for abstract + Section 1 + the numerical-results discussion.
- **Link:** <https://arxiv.org/abs/1803.08661>

### Optuna `GPSampler` — the docs page *and* `optuna/_gp/prior.py`

- **What it is:** Optuna 4.9.0's Gaussian-process sampler. Read the API page, then read 30 lines of
  source, because the source is where the noise model actually lives.
- **Why it earns its place:** it is the only place in the mainstream Python HPO stack where you can see
  the nugget term as a concrete switch. The signature is
  `GPSampler(*, seed=None, independent_sampler=None, n_startup_trials=10, deterministic_objective=False, constraints_func=None, warn_independent_sampling=True)`.
  The relevant parameter, verbatim: *"deterministic_objective: Whether the objective function is
  deterministic or not. If True, the sampler will fix the noise variance of the surrogate model to the
  minimum value (slightly above 0 to ensure numerical stability). Defaults to False."*

  Then open `optuna/_gp/prior.py` in the installed wheel. It is 30 lines and contains the two facts the
  docs page hides:

  ```python
  DEFAULT_MINIMUM_NOISE_VAR = 1e-6
  ...
      + gamma_log_prior(gpr.noise_var, 1.1, 30)
  ```

  So: `noise_var` is a learned kernel hyperparameter fitted by marginal-log-likelihood maximisation
  under a Gamma(1.1, 30) prior, floored at 1e-6; `deterministic_objective=True` pins it to that floor.
  **Leaving `deterministic_objective` at its default is correct for this project, and setting it True
  would be an active bug** — it would tell the surrogate that Monte-Carlo error does not exist and make
  it interpolate through every replication-noise wiggle.
- **Question it answers:** "what does Optuna actually expose for observation noise?" Answer: exactly
  one boolean, and it is on the GP sampler only. TPE has no equivalent.
- **Dependency note worth knowing before you commit:** `optuna/_gp/gp.py` lazily imports `torch` and
  `scipy`. GPSampler is not usable on Optuna's base install.
- **Time:** 10 min docs + 5 min source.
- **Links:** <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html>
  and the `optuna/_gp/prior.py` / `optuna/_gp/gp.py` files in the 4.9.0 wheel.

### BoTorch `qNoisyExpectedImprovement` / `qLogNoisyExpectedImprovement`

- **What it is:** the reference implementation of Frazier's point, as shipped code.
- **Why it earns its place:** one sentence from the docstring does the job — *"This function does not
  assume a `best_f` is known (which would require noiseless observations)."* That is the entire
  difference between EI and noisy EI in eleven words: classical EI needs an incumbent best *observed*
  value, and under noise you don't have one. NEI instead draws from the joint posterior over the
  candidate and the previously-observed points. The `qLog*` variants are the numerically-stable
  rewrites and BoTorch recommends them as drop-in replacements.
- **Question it answers:** what a production BO library does instead of EI when it cannot trust `y`.
- **Time:** 10 min.
- **Exact section:** `botorch.acquisition.monte_carlo.qNoisyExpectedImprovement` and
  `botorch.acquisition.logei.qLogNoisyExpectedImprovement` in
  <https://botorch.readthedocs.io/en/stable/acquisition.html>

### Optuna `TPESampler` source — `_split_trials`, and why ranking is the noise story

- **What it is:** `optuna/samplers/_tpe/sampler.py` in the 4.9.0 wheel. Read the docstring and then the
  `_split_trials` function around line 757.
- **Why it earns its place:** the reason TPE behaves differently from a GP under noise is structural
  and visible in fifty lines of code. TPE never models the objective *value*. It sorts finished trials,
  splits them into a "good" set and a "bad" set at a quantile `gamma`, fits two Parzen densities `l(x)`
  and `g(x)`, and samples to maximise `l(x)/g(x)`. Consequences, both of which matter here:
  - **Robustness:** only the *ordering* of trial values enters the algorithm, and only coarsely — a
    trial is either above or below the split. Monte-Carlo error that does not move a trial across the
    split is invisible to TPE. A GP surrogate, by contrast, tries to fit the noisy value itself.
  - **Fragility:** there is no noise model at all, so TPE cannot *know* it is being fooled, and it
    cannot re-sample a promising configuration to reduce its uncertainty. Under enough noise a lucky
    bad configuration is permanently promoted into `l(x)` and permanently biases the search. There is
    no `deterministic_objective` knob and no nugget to fall back on.
  - `constant_liar=True` exists ("*If True, penalize running trials to avoid suggesting parameter
    configurations nearby*") but that addresses *parallelism*, not noise. Do not mistake one for the
    other.
- **Question it answers:** why "just use TPE, it's the default" is a defensible but *unjustified*
  choice here — and what you give up by making it.
- **Time:** 20 min.
- **Links:** <https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html>
  plus the wheel source.

### Optuna `WilcoxonPruner` — the piece of Optuna that was built for exactly this

- **What it is:** `optuna.pruners.WilcoxonPruner`, `@experimental_class("3.6.0")`, still experimental in
  4.9.0. Read the class docstring in `optuna/pruners/_wilcoxon.py` — it is better than the rendered
  docs page.
- **Why it earns its place:** this is the single most on-target feature in either candidate library, and
  it appears in the evaluation plan only as a bare name in a list of eight pruners. The docstring,
  verbatim: *"This pruner performs the Wilcoxon signed-rank test between the current trial and the
  current best trial, and stops whenever the pruner is sure up to a given p-value that the current
  trial is worse than the best one. This pruner is effective for optimizing the mean/median of some
  (costly-to-evaluate) performance scores over a set of problem instances."* Substitute "problem
  instance" → "Monte-Carlo replication" and it is a description of the imputation simulation loop.

  Three details that make it usable and are easy to miss:
  - *"There can be 'easy' or 'hard' instances (the pruner handles correspondence of the instances
    between different trials)."* The test is **paired** — replication `i` in trial A is compared
    against replication `i` in trial B. So you must use the *same* amputation draws across trials and
    report them under stable `step` ids. That is a hard constraint on the simulation harness design,
    and it is also free variance reduction (common random numbers).
  - *"In each trial, it is recommended to shuffle the evaluation order, so that the optimization doesn't
    overfit to the instances in the beginning."*
  - *"This is different from other pruners in that the reported value need not converge to the real
    value."* Median/SuccessiveHalving/Hyperband pruners all assume `report(value, step)` is a
    *convergent learning curve*. A per-replication score is not. Using them on a replication axis
    without first converting to a running average is a category error.
  - Requires `scipy`.
- **Question it answers:** does Optuna expose anything for stochastic objectives? Yes, precisely one
  thing, and it is a racing procedure, not a noise model.
- **Time:** 15 min.
- **Link:** <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html>

### Ray Tune `Repeater`

- **What it is:** `ray.tune.search.Repeater`, a search-algorithm wrapper.
- **Why it earns its place:** it is Ray Tune's entire answer to noise, and it is worth reading mostly to
  see how little it is. *"A wrapper algorithm for repeating trials of same parameters."* It runs each
  suggested configuration `repeat` times, averages the metric, and only reports the average to the
  wrapped searcher. `set_index=True` injects a `TRIAL_INDEX` into the trial config *"which can be used
  for seeds."*
- **The two lines that constrain the design:** you must set `num_samples` to a multiple of `repeat`
  (leftovers are discarded), and — verbatim — *"It is recommended that you do not run an early-stopping
  TrialScheduler simultaneously."* So in Ray Tune, **replication-averaging and early-stopping are
  mutually exclusive by the docs' own advice.** Optuna's `WilcoxonPruner` is the strictly better
  primitive here because it *is* the early stopping, computed on the replications.
- **Question it answers:** what Ray Tune actually exposes for this. Answer: fixed-budget resampling with
  no statistical test and no adaptivity.
- **Time:** 10 min.
- **Link:** <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.Repeater.html>

---

## Part 2 — Multi-objective, because this problem genuinely is

**What you should be able to say afterwards:** what a Pareto front is and why "just take a weighted
sum" is a decision rather than a convenience; what a hypervolume indicator measures and what its
reference point is doing; and — most importantly — which pairs of your candidate objectives are
*already* fused correctly by a known proper scoring rule and therefore should never have been separate
objectives at all.

### Gneiting & Raftery, *Strictly Proper Scoring Rules, Prediction, and Estimation* — §6.2, and Table 2

**Read this before you build a Pareto front. It may delete two of your objectives.**

- **What it is:** JASA 102(477):359–378, 2007. Tilmann Gneiting and Adrian Raftery. The author copy is
  free at the UW stats site.
- **Why it earns its place:** it is the primary source for the concept of *propriety*, which is the
  formal name for "an objective a cynical optimiser cannot game". The definition, verbatim from the
  abstract: *"A scoring rule is proper if the forecaster maximizes the expected score for an
  observation drawn from the distribution F if he or she issues the probabilistic forecast F, rather
  than G ≠ F. It is strictly proper if the maximum is unique."* And: *"In prediction problems, proper
  scoring rules encourage the forecaster to make careful assessments and to be honest."* Honesty here
  is a theorem, not an aspiration. Optimising an *improper* score is exactly the trap in Part 5.

  Now §6.2, "Interval Score". They derive the negatively-oriented interval score for a central
  (1−α)·100% prediction interval `[l, u]`:

  ```
  S_α(l, u; x) = (u − l) + (2/α)(l − x)·1{x < l} + (2/α)(x − u)·1{x > u}
  ```

  In their words: *"The forecaster is rewarded for narrow prediction intervals, and he or she incurs a
  penalty, the size of which depends on α, if the observation misses the interval."* This is a **single
  proper score that fuses coverage and width** — the two objectives the framing lists as conflicting.
  They are not independent objectives whose tradeoff needs a Pareto front. They are the two terms of
  one score whose relative weight is fixed by α, by theory, not by a scalarisation you invent.

  Then **Table 2 on p. 371 — this is the money.** They compare three 95% interval forecasts on 100,000
  sequential predictions from a bilinear process:

  | forecast | empirical coverage | average width | average interval score |
  |---|---|---|---|
  | I (classical) | 95.01% | 4.00 | **4.77** |
  | J | 95.08% | 5.45 | 8.04 |
  | K | 94.98% | **3.79** | 5.32 |

  All three hit nominal coverage. K is the *sharpest*. K is also, in their words, *"misguided, in that
  it collapses to a point forecast when the conditional predictive variance is highest."* It is
  constructed to minimise expected width subject to nominal coverage — which is precisely what an
  optimiser handed "coverage ≥ 0.95" as a constraint and "minimise width" as an objective would
  converge to. The interval score correctly ranks it below the classical interval. **This table is a
  published, numbered demonstration that coverage-plus-width as two separate objectives is gameable and
  that the proper score is not.**

  Also worth having: §4.2 (CRPS — *"generalizes the absolute error"*, and is proper for the full
  predictive distribution) and §4.3 (the energy score, its multivariate generalisation). If you want a
  distributional objective that is *proper*, CRPS/energy score is it, and a bare Wasserstein distance
  between imputed and observed marginals is not (see Part 5).
- **Question it answers:** which of my objectives are actually one objective? And: what is the technical
  criterion for "an objective that cannot be cheated"?
- **Time:** ~40 min for §6.2 + Table 2 + §4.2–4.3. Skip §2, §3, §5, §7, §8 unless curious.
- **Link:** <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>

### Deb, Pratap, Agarwal & Meyarivan, *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*

- **What it is:** IEEE Transactions on Evolutionary Computation 6(2):182–197, April 2002.
  DOI `10.1109/4235.996017`. This is the paper Optuna's `NSGAIISampler` cites by that exact DOI.
- **Why it earns its place:** it is the default multi-objective sampler you get from Optuna without
  asking, so you should know what it does. Its three contributions are: fast non-dominated sorting
  (O(MN²) instead of O(MN³)), elitism, and crowding distance — which removes the need to hand-tune a
  niche-sharing parameter. The crowding-distance mechanism is the part to internalise: it is what
  spreads solutions *along* the front rather than clustering them, and it is why NSGA-II gives you a
  usable front rather than three near-identical points.
- **Question it answers:** what the default is doing, and why a genetic algorithm rather than a
  surrogate model. (Answer: no surrogate to build, no acquisition function to define under noise —
  which is quietly an advantage here, and see the next entry.)
- **Time:** ~35 min; Sections II–III carry it.
- **Link:** DOI 10.1109/4235.996017. Freely-hosted author-adjacent copy:
  <https://sci2s.ugr.es/sites/default/files/files/Teaching/OtherPostGraduateCourses/Metaheuristicas/Deb_NSGAII.pdf>

### Ozaki, Tanigaki, Watanabe, Nomura & Onishi, *Multiobjective Tree-Structured Parzen Estimator*

- **What it is:** JAIR vol. 73, 8 April 2022, DOI `10.1613/jair.1.13188`. The journal-length successor
  to the GECCO 2020 paper *Multiobjective Tree-Structured Parzen Estimator for Computationally
  Expensive Optimization Problems* (DOI `10.1145/3377930.3389817`). Optuna's `TPESampler` docstring
  cites both.
- **Why it earns its place:** it is the algorithm that is now *inside* `TPESampler` (see the correction
  in the last section — the standalone `MOTPESampler` class no longer exists). The key move is that
  the good/bad split that single-objective TPE does by value is done here by **non-domination rank,
  with hypervolume-contribution used as the tie-break within the boundary rank**. You can read that
  directly in `optuna/samplers/_tpe/sampler.py::_split_trials` (`_fast_non_domination_rank`, then
  `_solve_hssp` on the boundary rank). Optuna's docstring is explicit that the `l(x)` weights come from
  *"a rule based on the hypervolume contribution proposed in the paper of MOTPE"*.
- **Question it answers:** how a density-ratio method that has no notion of "better" copes with a
  partial order. And it is the mechanism that makes multi-objective TPE inherit single-objective TPE's
  rank-based noise robustness from Part 1.
- **Time:** ~35 min. The GECCO version is shorter if you can access it.
- **Link:** <https://www.jair.org/index.php/jair/article/view/13188>

### Optuna's multi-objective surface: the tutorial, `plot_pareto_front`, `plot_hypervolume_history`

- **What it is:** the "Multi-objective Optimization with Optuna" recipe, plus two visualisation
  functions.
- **Why it earns its place:** it is a ten-minute read that pins down the whole API. `directions=` takes
  a list (*"we want to minimize the FLOPS (we want a faster model) and maximize the accuracy. So we set
  `directions` to `["minimize", "maximize"]`"*); `study.best_trials` returns the Pareto front rather
  than a single trial (there is no `best_trial` in multi-objective mode, which is the API forcing the
  conceptual point); `optuna.visualization.plot_pareto_front(study, target_names=[...])` draws it.

  The tutorial does **not** mention hypervolume. That matters, because hypervolume is how you answer
  "is run B better than run A" when both produce fronts. The public entry point is
  `optuna.visualization.plot_hypervolume_history(study, reference_point)` — note the **mandatory
  `reference_point` argument**, validated against `len(study.directions)`. There is no default. That is
  the API telling you a truth about the indicator: **hypervolume is not a property of a front, it is a
  property of a front relative to a chosen reference point**, and choosing that point is a modelling
  decision you must make and record. The computation itself lives in the private `optuna._hypervolume`
  package (WFG, plus specialised 2-D/3-D routines) — it is not public API, so do not build on it.
- **Question it answers:** what the code actually looks like, and where the one non-obvious required
  input is.
- **Time:** 15 min.
- **Links:** <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html>,
  `optuna/visualization/_hypervolume_history.py` in the 4.9.0 wheel.

### Ray Tune's multi-objective support — read it to see how thin it is

- **What it is:** Ray Tune does not have first-class multi-objective optimisation. It has
  `OptunaSearch` with list-valued `metric`/`mode`.
- **Why it earns its place:** as a negative result, cheaply obtained. The `OptunaSearch` page states
  *"Multi-objective optimization is supported"* and *"OptunaSearch requires optuna>=3.0"*, with the
  example `metric=["loss1", "loss2"], mode=["min", "max"]`. The Tune FAQ — which the existing HPO plan
  cites at Stage 8 as *"the single clearest first-party statement of the decision rule that exists"* —
  says nothing about multi-objective, nothing about noisy objectives, and nothing about `Repeater`. Its
  recommendation is still, verbatim: *"Our go-to solution is usually to use random search with ASHA for
  early stopping for smaller problems. Use BOHB for larger problems with a small number of
  hyperparameters and Population Based Training for larger problems with a large number of
  hyperparameters if a learning schedule is acceptable."* Every branch of that rule assumes a
  single deterministic metric and a learning-curve fidelity. **None of the three branches applies to
  this problem.**
- **Question it answers:** if I want multi-objective in Ray Tune, what am I actually running? Answer:
  Optuna, wrapped, plus 78 MB of scheduler.
- **Time:** 10 min.
- **Links:** <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html>,
  <https://docs.ray.io/en/latest/tune/faq.html>

### Dang, Opris, Salehi & Sudholt, *Analysing the Robustness of NSGA-II under Noise* (optional, sharp)

- **What it is:** GECCO 2023, arXiv:2306.04525. A runtime-analysis paper.
- **Why it earns its place:** it is the only rigorous result connecting Part 1 and Part 2. Verbatim from
  the abstract: *"We show that GSEMO fails badly on every noisy fitness function as it tends to remove
  large parts of the population indiscriminately. In contrast, NSGA-II is able to handle the noise
  efficiently on LeadingOnesTrailingZeroes when p < 1/2, as the algorithm is able to preserve useful
  search points even in the presence of noise. We identify a phase transition at p = 1/2 where the
  expected time to cover the Pareto front changes from polynomial to exponential."* The mechanism —
  population-based elitism preserves good points that a single bad evaluation would otherwise discard —
  is the intuition to keep. It is an argument that NSGA-II is a *reasonable* default under noise, not
  merely a convenient one.
- **Honest caveat:** this is theory on a synthetic pseudo-Boolean benchmark with a specific
  posterior-noise model. Do not cite it as evidence about your problem; cite it for the mechanism.
- **Time:** 20 min (abstract + intro + the phase-transition discussion). Skip the proofs.
- **Link:** <https://arxiv.org/abs/2306.04525>

---

## Part 3 — Multi-fidelity when "fidelity" is replications, not epochs

**The question:** Successive Halving, Hyperband, ASHA and BOHB all assume a *cheap partial signal that
converges toward the true value* — a partial learning curve. A per-replication imputation score is not
that. Does the bandit machinery still apply when the fidelity axis is "number of Monte-Carlo
replications"?

**The short answer, which you should hold in mind while reading:** yes, but it comes from a different
literature. The algorithm-configuration community (SMAC, irace) has been running configurations across
*instances and seeds* since before Hyperband existed, and the correct primitive there is **racing**,
not learning-curve extrapolation. Optuna's `WilcoxonPruner` (Part 1) is a racing procedure. Hyperband
is not.

### SMAC3 — "Optimization across Instances" and "Multi-Fidelity Optimization"

- **What it is:** the docs for SMAC3 (AutoML/Freiburg), the maintained descendant of Hutter, Hoos &
  Leyton-Brown's SMAC. BSD-licensed. **Neither existing curriculum mentions SMAC at all**, which is the
  largest single gap in them relative to this project.
- **Why it earns its place:** SMAC is the only mainstream tool whose core abstraction is "a
  configuration is evaluated on a *set* of instances/seeds, and we must decide how many to spend". Two
  pages carry it:
  - **Optimization across Instances.** You declare `Scenario(instances=["d0","d1",...], instance_features={...})`.
    The instance features are not decorative: *"Those instance features are used to expand the internal
    X matrix and thus play a role in training the underlying surrogate model."* Translated to this
    project: if your Monte-Carlo replications differ systematically (different missingness rates,
    different amputation patterns, different subsample sizes), you can hand those descriptors to the
    surrogate and it will learn *which configurations are good on which kinds of replication* rather
    than averaging the distinction away. That is a strictly richer model than any `Repeater`.
  - **Multi-Fidelity Optimization.** SMAC supports two budget kinds, and the second is the answer to
    the research question: *"Instance-based budgets: budget values remain internal; `min_budget` and
    `max_budget` determine stage instance counts"*, and *"when using instances, budget parameters
    become optional since `max_budget` represents maximum instance count and `min_budget` defaults
    to 1."* **That is Successive Halving / Hyperband run with "number of instances evaluated" as the
    rung resource.** So the bandit machinery does transfer — it just needs the running-mean-over-
    instances as the reported value, not the raw per-instance score. Hyperband is the default
    intensifier in SMAC's multi-fidelity facade.
- **Question it answers:** Q3, directly and affirmatively, with a shipped implementation as evidence.
- **Time:** 20 min for both pages; another 20 for the "Stochastic Gradient Descent On Multiple
  Datasets" example, which is the runnable version.
- **Links:** <https://automl.github.io/SMAC3/latest/advanced_usage/4_instances/>,
  <https://automl.github.io/SMAC3/latest/advanced_usage/2_multi_fidelity/>

### Lindauer, Eggensperger, Feurer, Biedenkapp, Deng, Benjamins, Ruhkopf, Sass & Hutter, *SMAC3*

- **What it is:** JMLR 23(54), 2022. The paper for the package above.
- **Why it earns its place:** the citable, peer-reviewed statement of the design. It describes SMAC3 as
  *"a robust and flexible framework for Bayesian Optimization, which can improve performance within a
  few evaluations"*, and — the load-bearing clause — that it supports *"algorithm configuration across
  various problem instances"* as a first-class case alongside hyperparameter tuning. BSD-licensed.
- **Time:** 25 min.
- **Link:** <https://jmlr.org/papers/v23/21-0888.html>

### Hutter, Hoos & Leyton-Brown, *Sequential Model-Based Optimization for General Algorithm Configuration*

- **What it is:** LION-5, 2011, DOI `10.1007/978-3-642-25566-3_40`. The original SMAC paper.
- **Why it earns its place:** this is where the *intensification* mechanism is defined — the aggressive
  racing procedure that decides how many additional (instance, seed) runs to spend comparing a
  challenger against the incumbent, rather than fixing the count in advance. It is also where the
  random-forest surrogate (rather than a GP) is introduced, motivated partly by the need to handle
  categorical and conditional parameters and instance features together. SMAC's own project page
  describes it as *"a versatile tool for optimizing algorithm parameters (or the parameters of some
  other process we can run automatically, or a function we can evaluate, such as a simulation)"* —
  note "a simulation".
- **Time:** ~35 min; read Section 3 (the SMAC algorithm) and the intensification subsection.
- **Link:** <https://link.springer.com/chapter/10.1007/978-3-642-25566-3_40>. First-party project page
  with the citation and code: <https://www.cs.ubc.ca/labs/algorithms/Projects/SMAC/>

### López-Ibáñez, Dubois-Lacoste, Pérez Cáceres, Stützle & Birattari, *The irace package*

- **What it is:** *Operations Research Perspectives* 3:43–58, 2016, DOI `10.1016/j.orp.2016.09.002`.
  An R package; you will not depend on it. Read it for the procedure.
- **Why it earns its place:** irace is racing in its purest form and the cleanest mental model for
  "spend replications adaptively". Self-described as *"the iterated racing procedure, which is an
  extension of the Iterated F-race procedure"*: sample a population of configurations, evaluate them on
  instances one at a time, run a statistical test after each instance, **eliminate configurations that
  are statistically significantly worse**, and stop when few enough survive; then re-sample around the
  survivors and repeat. Two of its named extensions map onto real risks here: a **restart mechanism to
  avoid premature convergence** (the risk when your MC noise makes an early winner look decisive), and
  **elitist racing**, which ensures the configurations finally returned are also the ones evaluated on
  the *most* instances (the risk that your reported winner won on 3 replications while the runner-up
  lost on 200).
- **Question it answers:** what the mature form of "race configurations over simulation replications"
  looks like after fifteen years of use.
- **Time:** ~40 min; Sections 2–3.
- **Link:** <https://iridia.ulb.ac.be/irace/> (project page with the full citation);
  DOI 10.1016/j.orp.2016.09.002

### The fidelity axes, and which ones are legitimate

Worth writing down explicitly, because they are not interchangeable:

| candidate fidelity axis | legitimate? | why |
|---|---|---|
| **Number of Monte-Carlo replications** | **Yes** — this is the one | The objective is literally a mean over replications. Fewer replications = a noisier estimate of *the same* quantity. This is what SMAC's instance-budgets and irace's racing do, and what `WilcoxonPruner` tests. |
| **Subsample of rows** | Yes, with care | A cheaper, *biased* estimate — small-`n` behaviour of MICE differs from large-`n` behaviour, so ranking at low fidelity can invert. This is the classic multi-fidelity setting (a biased cheap source), and Frazier §5's multi-fidelity subsection is the right frame, not racing. |
| **Number of imputations `m`** | **No** | `m` is not a fidelity dial, it is a *parameter of the estimand* (Part 6). Changing `m` changes what you are measuring, not how precisely. |
| **Number of MICE iterations** | **No** | Same objection, worse. A run that has not converged is not a low-fidelity estimate of a converged run; it is a different, wrong sampler. FIMD §6.5.2 is explicit that convergence must be assessed, and warns *"automated convergence monitoring (as by a machine) is unsafe and should be avoided."* |

The last two rows are the ones a naive Hyperband port would get wrong, because both look superficially
like "epochs".

---

## Part 4 — Optuna vs Ray Tune, as a library-dependency decision

Not a research-script decision. DataForgeML is a library that other people install.

**Hard numbers, read from PyPI and the installed wheel on 2026-08-16:**

| | Optuna | Ray (`ray[tune]`) |
|---|---|---|
| Current version | **4.9.0** | **2.57.0** |
| Licence | **MIT** (classifier-confirmed) | **Apache 2.0** |
| `requires-python` | `>=3.9` | `>=3.10` |
| Wheel | `py3-none-any`, **0.4 MB**, pure Python | platform wheels, **77.8 MB** (manylinux x86_64); 28.7 MB on Windows |
| Runtime deps (base) | `alembic>=1.5.0`, `colorlog`, `numpy`, `packaging>=20.0`, `sqlalchemy>=1.4.2`, `tqdm`, `PyYAML` — **7** | `click>=7.0`, `filelock`, `jsonschema`, `msgpack`, `packaging>=24.2`, `protobuf>=3.20.3`, `pyyaml`, `requests` — **8**, plus a compiled core |
| Deps added by the tuning extra | none (`optional` extra adds plotly, scikit-learn, torch, redis, boto3… all opt-in) | `[tune]` adds `pandas`, `tensorboardX>=1.9`, `requests`, `pyarrow>=17.0.0`, `fsspec`, `pydantic<3,>=2.5.0` |
| Distributed scheduler required? | **No.** In-memory storage by default. | **Yes** — Ray Tune runs on the Ray runtime; a local cluster is started even for single-node runs. |
| Persistence | Pluggable storage; in-memory default, SQLite/`JournalFileBackend` for resume, client-server RDB only for multi-node parallelism | Filesystem/`fsspec` experiment directories, tensorboardX event files |
| Can it be an optional extra? | **Yes, cleanly** — pure Python, no build step, importable at call time | Technically yes; but a 78 MB conditional install with `protobuf` and `pydantic` pins is a heavy thing to ask, and the pins are exactly the ones that collide with other packages in a user's environment |
| API stability | Semver-ish with explicit long deprecation windows. Concrete evidence: `TPESampler(weights=...)` is *"Deprecated in v4.9.0 … removal … currently scheduled for v6.0.0"* — two majors of notice | Faster-moving; `ray.tune` has absorbed several API reshuffles (`tune.run` → `Tuner`, the AIR consolidation and its partial unwind) |

**Two facts that decide it on their own.**

1. **Ray Tune's multi-objective support *is* Optuna.** `OptunaSearch` is how you get list-valued
   `metric`/`mode` in Tune, and it *"requires optuna>=3.0"*. So the Ray path for this project is
   "install 78 MB of distributed scheduler in order to call Optuna". That is a strictly larger
   dependency for a strictly smaller feature set.
2. **The one primitive this project most needs exists only in Optuna.** `WilcoxonPruner` races
   replications with a paired statistical test. Ray Tune's `Repeater` averages a fixed count and its
   own docs say not to combine it with an early-stopping scheduler.

**Recommendation: Optuna, as an optional extra (`dataforge-ml[tuning]`), imported lazily.**

**Stated both ways, because the tradeoff is real:**

*What you gain.* A 0.4 MB pure-Python, MIT-licensed, no-build-step dependency that supports a Python
floor one minor version lower than Ray's. Multi-objective (`directions=[...]`, `study.best_trials`),
the noise-aware GP nugget switch, MOTPE inside `TPESampler`, and `WilcoxonPruner` — all four things
this project needs — in one package. No scheduler process, no cluster, no ports. Storage is pluggable,
so DataForgeML never has to own a database. Deprecation windows measured in major versions.

*What you give up.* Ray Tune's genuinely better story for large-scale distributed execution: if a user
one day wants to fan a simulation study across a cluster, Optuna's answer is "point every worker at a
shared PostgreSQL/MySQL RDB" (their FAQ is explicit that multi-node parallelism *requires* a
client/server RDB), which is a real operational burden that Ray absorbs for you. You also give up
ASHA/PBT as production-grade schedulers — though Part 3 argues ASHA's fidelity model is the wrong one
here anyway, so this costs less than it looks. And you inherit two experimental surfaces: `GPSampler`'s
multi-objective logEHVI path is recent, and `WilcoxonPruner` is still `@experimental_class` four minor
versions after introduction, so its interface may move.

*The hedge that makes the loss cheap.* Keep the objective function a plain callable that takes a
`dict[str, Any]` of hyperparameters and returns a tuple of floats, with the simulation harness owning
replication seeding. Then Optuna is thirty lines of adapter, and if Ray is ever needed it is thirty
different lines. Do not let `optuna.Trial` leak into DataForgeML's own types.

---

## Part 5 — The traps: what a maximally-cynical optimiser would exploit

The named trap: **an HPO loop that tunes against held-out reconstruction error converges on
deterministic conditional-mean imputation, which is exactly the invalid method.** This is not a
hypothesis; it is FIMD §2.6, and the evaluation plan already covers it well (`norm.predict` RMSE 0.725
vs `norm.nob` 1.025, with the worse RMSE being the valid method). What follows is the same analysis
applied to each *replacement* objective, because every one of them has its own degenerate optimum.

Read this table as: *if this were the only thing being optimised, what would win?*

| Objective | What a cynical optimiser converges on | Why it wins | The guard |
|---|---|---|---|
| **Masked-cell RMSE / MAE** | Deterministic conditional-mean imputation; `sample_posterior=False`; zero added noise | The conditional mean minimises squared error by construction. Uncertainty is pure cost. | Do not optimise it. Report it as a diagnostic only. (FIMD §2.6) |
| **Wasserstein distance between imputed and observed marginals** | An imputer that **ignores every other column** and draws from the observed marginal (a bootstrap of observed values, per column) | It scores a *perfect* marginal match while destroying every cross-column association. The score is blind to the joint distribution. Worse: under MAR the two distributions *should* differ, so the true model is actively penalised — FIMD §6.6: *"Except under MCAR, the distributions do not need to be identical, since strong MAR mechanisms may induce systematic differences between the two distributions."* | Never use marginal distributional distance alone. Either pair it with an association-preservation term, or replace it with a **proper** score over the predictive distribution (CRPS / energy score, Gneiting §4.2–4.3), which is not maximised by a marginal-matching cheat. |
| **Coverage rate (alone)** | Absurdly wide intervals — inflate `m`-between variance, add huge noise, degenerate to near-infinite intervals | Coverage is monotone in width. 100% coverage is trivially achievable. FIMD §2.5 only ever states coverage as a *floor* (*"The actual rate should be equal to or exceed the nominal rate"*), never as a thing to maximise. | Coverage is a **constraint at nominal, not an objective**. Pair with width — or better, fuse them via the interval score. |
| **Average interval width (alone, or as "minimise width s.t. coverage ≥ nominal")** | Gneiting & Raftery's forecast **K**: an interval that *"collapses to a point forecast when the conditional predictive variance is highest"* | It achieves nominal *marginal* coverage (94.98%) while being the sharpest (3.79) — by being confidently wrong exactly where the data is hardest. Marginal coverage does not imply conditional coverage. | Use the interval score (§6.2, eq. 43). It ranks K (5.32) below the classical interval (4.77) despite K's better width and equal coverage. This is the documented counterexample. |
| **Percent bias (alone)** | Any method whose errors are *symmetric*, however wildly dispersed — including pure noise centred on the truth | Bias is a first-moment criterion. Cancellation is free. A method that imputes `truth + N(0, 10⁶)` has ~zero bias. | Never alone; PB is meaningful only jointly with coverage and width (which is exactly why FIMD §2.5 lists all four together). |
| **Downstream-task CV score** | Whatever suits *that* estimator — often deterministic imputation again, since a point-prediction model is scored by a point metric | The downstream model does not benefit from your uncertainty; it benefits from a clean signal. The proxy silently reverts to the §2.6 trap one level down. | Only defensible if the downstream *analysis* is itself the pooled multiple-imputation inference (Rubin's rules), not a single fit on one imputed frame. |
| **Hypervolume over the whole objective vector** | Points that push a single easy objective toward the reference point while doing nothing useful | Hypervolume rewards dominating volume, and a reference point chosen badly makes one axis cheap to dominate. Optuna makes `reference_point` mandatory precisely because there is no safe default. | Choose and record the reference point from the *nominal* targets (e.g. coverage 0.95, PB 5%), not from the observed range of trials. |
| **The whole simulation, if replications are re-drawn per trial** | Configurations that got lucky draws | With few replications, trial-to-trial variance dominates configuration-to-configuration variance. Nothing is being optimised except noise. | Common random numbers: the *same* amputation draws for every trial, paired. This is also what makes `WilcoxonPruner`'s paired test valid, and what irace's elitist racing exists to protect. |

**The unifying principle**, and the reason Gneiting & Raftery is in Part 2 rather than an appendix:
every trap above is an *improper* score being optimised. Propriety is the formal property that "the
honest answer is the optimal answer". If a candidate objective is not a proper scoring rule and not a
hard constraint, assume it is gameable and go looking for what games it.

---

## Part 6 — Where hyperparameters live, and which ones are not yours to tune

Not every knob is a hyperparameter. Some are settled by theory, and tuning them is not merely wasteful,
it is a category error — you would be optimising the definition of the estimand rather than the quality
of the estimate.

| Knob | Tunable against a distributional objective? | Primary source and verdict |
|---|---|---|
| **`m` (number of imputations)** | **No — settled by theory** | FIMD §2.8. Rubin (1987): variance inflates by `(1 + γ₀/m)`; at γ₀=0.3, m=5 costs ~6%. But the modern rules are all *formulas of the missingness fraction, not search targets*: Graham, Olchowski & Gilreath (2007) — for γ = (0.1,0.3,0.5,0.7,0.9), *"we need to set m = (20, 20, 40, 100, >100)"*; Bodner (2008) — m = (3,6,12,24,59,114,258) for γ₀ = (0.05,…,0.9); White, Royston & Wood (2011) — *"the number of imputations should be similar to the percentage of cases that are incomplete"* (m ≈ 100λ). Van Buuren's practice: *"set m=5 during model building, and increase m only after being satisfied with the model."* **Larger m is monotonically better and only costs compute.** An optimiser handed `m` will just push it to the ceiling, learning nothing. Compute it from λ; expose it as config; never tune it. |
| **MICE iteration count (`max_iter`)** | **No — a convergence diagnostic, not a dial** | FIMD §6.5.2. Convergence is diagnosed by whether the chains *"are freely intermingled with one another, without showing any definite trends"*, and *"when the variance between different sequences is no larger than the variance within each individual sequence"*. Van Buuren's explicit warning: *"automated convergence monitoring (as by a machine) is unsafe and should be avoided"* (citing Cowles and Carlin). A too-small `max_iter` gives you a non-converged Gibbs sampler, which is *wrong*, not merely *cheap* — and a distributional objective may well reward it (a non-converged chain has not yet propagated the associations that widen the imputations). **This is an actively dangerous thing to tune.** Set it generously, diagnose convergence, do not optimise it. |
| **PMM donor count `d` / `k`** | **Yes, narrowly — within a theory-bounded range** | FIMD §3.4.3. `mice` defaults to `d = 5`, *"a compromise"*. Schenker and Taylor (1996) evaluated d=3 and d=10, finding adaptive schemes *"slightly better … but the differences were small"*; Morris, White & Royston (2014) found *"d = 5 and d = 10 generally provided the best results"*; Kleinke (2017) suggests d=5 may be too high below n=100. FIMD warns against d=1 (too low) and against very high values like n/10 (introduces bias). So: a **bounded categorical over {3, 5, 10}**, sample-size-aware, is a legitimate and cheap search dimension. An unbounded integer range is not — the trap is that large `d` flattens toward the marginal (the Wasserstein cheat above) and small `d` collapses toward the deterministic prediction (the §2.6 trap). Both ends of an unbounded range are degenerate. |
| **KNN `n_neighbors`** | **Yes** | Genuinely free — no theory pins it, and it directly controls the bias/variance and the sharpness of the imputed distribution. But note it has the same two-ended degeneracy as `d`: k=1 is nearest-neighbour hot-deck (high variance), large k is a local mean (variance collapse, the §2.6 trap). Bound it and interpret the endpoints. |
| **Predictor-set selection (breadth, `mincor`, `n_nearest_features`)** | **Yes — and this is probably the highest-value dimension** | Already researched in `docs/research/regression-vs-mice-literature.md` §Q2. The literature treats breadth as a **dial**, not a fixed answer: FIMD §6.3 says *"It is often beneficial to choose as large a number of predictors as possible"* but caps at 15–25 for wide data; R's `quickpred` selects by correlation at `mincor = 0.1`; sklearn's `n_nearest_features=None` means all. The threshold is exactly the kind of thing with no theoretical answer and a real effect on the joint distribution. Tune `mincor` / the width cutoff. |
| **The per-column estimator's own hyperparameters** (RF depth, ridge α, …) | **Yes, but with the sharpest version of the trap** | This is where the disposable-model framing bites hardest. These knobs are *designed* to be tuned against predictive accuracy, and every default tuning recipe in the ecosystem — including Stages 1–8 of the existing HPO plan — will tune them that way. Tuned against the dataset-level objective they behave completely differently: regularisation that improves prediction shrinks the conditional variance and *hurts* the imputation. Treat any per-column estimator hyperparameter as tunable only through the dataset-level objective, never through an inner CV on the column. |
| **`sample_posterior` / whether to add noise at all** | **No — not a hyperparameter, a validity condition** | FIMD §1.3: *"It is always better to include parameter uncertainty."* If this is in the search space, the search will turn it off (it improves every point metric and several distributional ones). It is a fixed structural choice, not a dial. |

**The pattern:** knobs settled by theory are exactly the ones where the objective is *monotone* (m: more
is better; iterations: more is better; noise: must be on). Monotone knobs must never enter a search
space — an optimiser will always slam them to one end and you will have burned trials confirming
arithmetic. Genuinely tunable knobs are the ones with an interior optimum and *degenerate behaviour at
both ends* (`d`, `k`, predictor breadth). That is the test.

---

## Corrections to the existing curricula

Things in `docs/learning/hyperparameter-optimization-plan.md` and
`docs/learning/evaluation-and-validation-plan.md` that the new framing makes wrong, out-of-date, or
misleading. Ordered by how much damage they would do if followed.

1. **HPO plan, Stages 3–4: expected improvement is presented as *the* acquisition function.** The
   Distill piece, the paretos video, and the de Freitas lecture all build intuition on PI/EI/UCB. For a
   Monte-Carlo objective, Frazier §5 is explicit that *"Direct use of the EI acquisition function
   presents conceptual challenges … since the 'improvement' … is no longer easily defined, and f(x) …
   is no longer observed"*, and that *"KG can outperform EI substantially in problems with substantial
   noise."* **Add:** knowledge-gradient and noisy EI (BoTorch `qLogNoisyExpectedImprovement`, which
   *"does not assume a `best_f` is known (which would require noiseless observations)"*) to Stage 4,
   with a note that EI is the noise-fragile member of the family.

2. **HPO plan, Stage 4 assigns Frazier for the wrong sections.** It is used as a rigorous restatement of
   GP regression and acquisition functions. Its distinctive contribution — and the author's own stated
   differentiator versus other surveys — is **Section 5 on exotic problems**, which contains the noise
   treatment and the "random environmental conditions" formulation `∫f(x,w)p(w)dw` that *is* this
   project's objective. Re-target the assignment.

3. **HPO plan, Stage 5 + Stage 9: the fidelity axis is assumed to be training epochs / learning
   curves.** Hyperband, ASHA and the Domhan learning-curve-extrapolation paper all assume a partial
   signal that *converges toward* the final value. A per-replication imputation score does not, and
   Optuna's `WilcoxonPruner` docstring says so directly: *"This is different from other pruners in that
   the reported value need not converge to the real value. To use pruners such as SuccessiveHalvingPruner
   in the same setting, you must provide e.g., the historical average of the evaluated values."*
   Following Stage 5 naively — reporting raw per-replication scores to a Hyperband pruner — is a bug,
   not a suboptimality.

4. **Both plans omit SMAC entirely.** SMAC is the tool whose central abstraction (configurations
   evaluated over instances and seeds, with intensification deciding how many runs to spend) is the
   closest existing match to this problem, and it is BSD-licensed, peer-reviewed (JMLR 23(54), 2022),
   and actively maintained. Its instance-based budgets are Successive Halving with *instance count* as
   the rung resource, which is the direct answer to "does the bandit machinery still apply?". The same
   omission covers **irace** and the entire racing literature. This is the biggest gap.

5. **HPO plan, Stage 8: the Ray Tune FAQ decision rule is presented as "the single clearest first-party
   statement of the decision rule that exists".** Verified still live and still saying what the plan
   quotes — but all three branches (*"random search with ASHA"* / *"BOHB"* / *"Population Based
   Training"*) presuppose a single deterministic metric and an epoch-like fidelity. **None applies
   here.** Keep the resource, add the caveat that it is out of scope for stochastic multi-objective
   simulation objectives.

6. **HPO plan, Stage 6 (PBT): safe to skip for this project.** PBT's defining property is that
   hyperparameters change *during* training. There is no "during" in a MICE fit worth exploiting, and
   the plan's own check-yourself question ("a five-second scikit-learn fit") already implies the answer.
   Not wrong — just demote it.

7. **Neither plan covers multi-objective optimisation at all.** This is the largest structural gap
   after SMAC: no Pareto front, no scalarisation, no hypervolume, no NSGA-II, no MOTPE. The problem is
   irreducibly multi-objective, and `study.best_trials`-vs-`study.best_trial` is the first API surface
   you hit.

8. **Optuna facts that have gone stale.** Verified against Optuna **4.9.0** (PyPI, 2026-08-16) and the
   installed wheel:
   - **`MOTPESampler` no longer exists.** There is no `optuna/samplers/_motpe` in 4.9.0. MOTPE is folded
     into `TPESampler`, which now handles multi-objective via non-domination-rank splitting with
     hypervolume-contribution tie-breaking (`_split_trials` → `_fast_non_domination_rank` → `_solve_hssp`).
     Any curriculum or design note that names `MOTPESampler` as the class to use is wrong.
   - **`NSGAIISampler` is still the default for multi-objective studies** (`create_study` selects it
     when `directions` has length > 1), but its own docstring now nudges away from it: *"TPESampler
     became much faster in v4.0.0 and supports several features not supported by NSGAIISampler such as
     handling of dynamic search space and categorical distance."*
   - **`TPESampler(weights=...)` is deprecated in 4.9.0**, removal scheduled for v6.0.0.
   - `GPSampler` now supports multi-objective via logEHVI, and requires `torch` + `scipy` (lazy
     imports, not base dependencies).

9. **Evaluation plan, Stage 5: `scipy.stats.wasserstein_distance` is offered as "the distributional
   proxy as a *number*".** The plan does flag the degenerate optimum in its prose gap section, which is
   to its credit. What it lacks is the technical vocabulary that makes the flaw nameable and the fix
   findable: Wasserstein between marginals is **not a proper scoring rule** for the predictive
   distribution. The proper alternatives are CRPS (Gneiting & Raftery §4.2, which *"generalizes the
   absolute error"*) and its multivariate generalisation the **energy score** (§4.3) — and
   `scipy.stats.energy_distance` (SciPy 1.18.0) exists alongside `wasserstein_distance` on the same
   docs shelf. Upgrade the recommendation.

10. **Evaluation plan, Stage 5's synthesis proposes "downstream-task CV score as the primary
    optimization objective, with a distributional term as a guard."** Under the new framing this is
    backwards. The downstream-task proxy reverts to the §2.6 trap one level down: a point-prediction
    model scored by a point metric prefers a deterministic imputation. It is only defensible when the
    downstream analysis is the *pooled* multiple-imputation inference. The plan's own final sentence
    concedes this is *"a design claim, not a finding"* — treating it as superseded.

11. **Evaluation plan, Stage 5's FIMD §2.5 criteria are listed but not ranked.** Percent bias, coverage
    rate, average width and RMSE are presented as a set. Two of the four (coverage, width) are, per
    Gneiting & Raftery §6.2, **the two terms of a single proper score** — the interval score, with
    their relative weight fixed by α. Treating them as separate objectives to trade off is a modelling
    error, and their Table 2 is the counterexample. This is the single most consequential correction in
    this list.

12. **Evaluation plan, Stage 6's Optuna sketch returns one number.** *"return a single number with the
    sign correct for your chosen `direction`"*. For this project the objective returns a tuple and the
    study is created with `directions=[...]`; `study.best_trial` will raise and `study.best_trials`
    returns a front. Also worth flagging: the plan's suggestion that *"an iterative method like MICE has
    a natural intermediate signal — the per-iteration state — which is exactly what a pruner wants"* is
    the mistake in item 3 and item Part-6-row-2 combined. MICE iterations are a convergence requirement,
    not a fidelity axis; pruning on them selects non-converged samplers.

---

## Sources consulted

Every item below was fetched and checked against the live page or the installed artefact before being
cited. Versions were read, not recalled.

**Papers**

- **Frazier, "A Tutorial on Bayesian Optimization", arXiv:1807.02811 (8 Jul 2018).**
  <https://arxiv.org/abs/1807.02811> — PDF downloaded and text-extracted. Confirmed §5 "Exotic Bayesian
  Optimization" contains the "Noisy Evaluations" and "Random Environmental Conditions and Multi-Task
  Bayesian Optimization" subsections; all quotes pulled from the extracted text (pp. 12–15).
- **Toscano-Palmerin & Frazier, "Bayesian Optimization with Expensive Integrands", arXiv:1803.08661
  (2018).** <https://arxiv.org/abs/1803.08661> — fetched; abstract quoted verbatim; confirmed it is the
  paper Frazier §5 cites for the environmental-conditions problem.
- **Gneiting & Raftery, "Strictly Proper Scoring Rules, Prediction, and Estimation", JASA
  102(477):359–378 (2007).** <https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf>
  — author copy downloaded and text-extracted. Confirmed section structure (§4.2 CRPS, §4.3 Energy
  Score, §6.2 Interval Score, §9 Optimum Score Estimation); the propriety definition, the interval
  score formula (43), the "forecaster is rewarded for narrow prediction intervals" passage, the
  "misguided … collapses to a point forecast" verdict on forecast K, and all of Table 2's figures
  (I: 95.01%/4.00/4.77; J: 95.08%/5.45/8.04; K: 94.98%/3.79/5.32) pulled from the extracted text.
- **Deb, Pratap, Agarwal & Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II",
  IEEE TEC 6(2):182–197 (April 2002), DOI 10.1109/4235.996017.** Publication metadata cross-checked;
  the DOI verified as the one Optuna's `NSGAIISampler` docstring cites. Note: IEEE Xplore returned no
  fetchable body, so the abstract was not quoted verbatim — bibliographic details only.
- **Ozaki, Tanigaki, Watanabe, Nomura & Onishi, "Multiobjective Tree-Structured Parzen Estimator",
  JAIR 73 (8 April 2022), DOI 10.1613/jair.1.13188.**
  <https://www.jair.org/index.php/jair/article/view/13188> — fetched; authorship, venue, volume and
  date confirmed.
- **Ozaki et al., "Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive
  Optimization Problems", GECCO 2020, DOI 10.1145/3377930.3389817.** ACM DL returned HTTP 403;
  **not verified against the live page.** Cited here only because Optuna 4.9.0's `TPESampler` docstring
  links that exact DOI with that exact title. Treat the JAIR paper as the citable version.
- **Lindauer, Eggensperger, Feurer, Biedenkapp, Deng, Benjamins, Ruhkopf, Sass & Hutter, "SMAC3: A
  Versatile Bayesian Optimization Package for Hyperparameter Optimization", JMLR 23(54) (2022).**
  <https://jmlr.org/papers/v23/21-0888.html> — fetched; authorship, volume/issue and BSD licensing
  confirmed.
- **Hutter, Hoos & Leyton-Brown, "Sequential Model-Based Optimization for General Algorithm
  Configuration", LION-5 (2011), DOI 10.1007/978-3-642-25566-3_40.** Springer redirected to an auth
  gateway; metadata confirmed instead from the first-party UBC project page
  <https://www.cs.ubc.ca/labs/algorithms/Projects/SMAC/>, which carries the citation and the
  self-description *"a versatile tool for optimizing algorithm parameters (or the parameters of some
  other process we can run automatically, or a function we can evaluate, such as a simulation)"*.
  **The abstract was not quoted verbatim.**
- **López-Ibáñez, Dubois-Lacoste, Pérez Cáceres, Stützle & Birattari, "The irace package: Iterated
  Racing for Automatic Algorithm Configuration", Operations Research Perspectives 3:43–58 (2016),
  DOI 10.1016/j.orp.2016.09.002.** <https://iridia.ulb.ac.be/irace/> — first-party project page
  fetched; the citation and the *"iterated racing procedure, which is an extension of the Iterated
  F-race procedure"* self-description confirmed. ScienceDirect returned HTTP 403, so the abstract and
  the restart / truncated-sampling / elitist-racing extensions come from the project page and search
  metadata rather than the article body. **Flagged as partially verified.**
- **Dang, Opris, Salehi & Sudholt, "Analysing the Robustness of NSGA-II under Noise", GECCO 2023,
  arXiv:2306.04525.** <https://arxiv.org/abs/2306.04525> — fetched; abstract quoted verbatim, including
  the p = 1/2 phase transition.

**Library documentation and source (versions read on 2026-08-16)**

- **Optuna 4.9.0.** PyPI JSON API: version 4.9.0, MIT licence (classifier), `requires-python >=3.9`,
  `py3-none-any` wheel at 0.4 MB, runtime deps `alembic>=1.5.0, colorlog, numpy, packaging>=20.0,
  sqlalchemy>=1.4.2, tqdm, PyYAML`. Wheel downloaded and unpacked; all source claims below read from
  that artefact.
  - `optuna/_gp/prior.py` — `DEFAULT_MINIMUM_NOISE_VAR = 1e-6` and `gamma_log_prior(gpr.noise_var, 1.1, 30)`.
  - `optuna/_gp/gp.py` — `noise_var` added to the covariance diagonal; MLL optimisation via
    `scipy.optimize.minimize`; `torch` and `scipy` are `_LazyImport`s.
  - `optuna/pruners/_wilcoxon.py` — `@experimental_class("3.6.0")`; full docstring quoted (paired
    signed-rank test over problem instances, shuffle recommendation, "reported value need not converge",
    scipy requirement).
  - `optuna/samplers/_tpe/sampler.py` — MOTPE paper citations at lines 104–108; `_split_trials` using
    `_fast_non_domination_rank` and `_solve_hssp`; `weights` deprecation notice ("Deprecated in v4.9.0
    … removal … scheduled for v6.0.0").
  - `optuna/samplers/nsgaii/_sampler.py` — the "TPESampler became much faster in v4.0.0" note.
  - `optuna/study/study.py` — `create_study` selects `NSGAIISampler` for multi-objective.
  - `optuna/samplers/` — **confirmed absence** of any `_motpe` module.
  - `optuna/visualization/_hypervolume_history.py` — `reference_point` is a required positional
    argument, validated against `len(study.directions)`.
- **Optuna docs** — samplers index (full sampler list and multi-objective capability table; MOTPESampler
  absent), `GPSampler` (signature and the `deterministic_objective` description quoted verbatim),
  `TPESampler` (`constant_liar`, multi-objective note), `NSGAIISampler`, the multi-objective recipe
  (`directions`, `best_trials`, `plot_pareto_front`; hypervolume **not** mentioned), and the FAQ
  (reproducibility caveat for non-deterministic objectives; storage/parallelism requirements —
  multi-node needs a client/server RDB).
  <https://optuna.readthedocs.io/en/stable/reference/samplers/index.html>,
  <https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html>,
  <https://optuna.readthedocs.io/en/stable/faq.html>
- **Ray 2.57.0.** PyPI JSON API: Apache 2.0, `requires-python >=3.10`, `ray-2.57.0-cp312-cp312-manylinux2014_x86_64.whl`
  at **77.8 MB** (28.7 MB on Windows), base deps `click>=7.0, filelock, jsonschema, msgpack, packaging>=24.2,
  protobuf>=3.20.3, pyyaml, requests`, and the `tune` extra adding `pandas, tensorboardX>=1.9, requests,
  pyarrow>=17.0.0, fsspec, pydantic`.
- **Ray Tune docs** — `Repeater` (the `repeat`/`set_index` parameters, the `num_samples`-multiple
  guidance, and *"It is recommended that you do not run an early-stopping TrialScheduler
  simultaneously"*), `OptunaSearch` (*"Multi-objective optimization is supported"*, *"requires
  optuna>=3.0"*, list-valued `metric`/`mode`), and the FAQ (decision rule re-verified verbatim;
  confirmed it says nothing about noise, repetition or multi-objective).
  <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.Repeater.html>,
  <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.search.optuna.OptunaSearch.html>,
  <https://docs.ray.io/en/latest/tune/faq.html>
- **BoTorch acquisition reference** — `qNoisyExpectedImprovement` docstring quoted verbatim (*"This
  function does not assume a `best_f` is known (which would require noiseless observations)"*).
  `qLogNoisyExpectedImprovement` is documented as the numerically-stable, same-API replacement.
  <https://botorch.readthedocs.io/en/stable/acquisition.html> (note: the page truncated before the
  `logei` section, so the `qLogNEI` formula was corroborated from the module docs rather than that
  page).
- **SMAC3 docs** — the landing page (self-description: *"a tool for algorithm configuration to optimize
  the parameters of arbitrary algorithms … Bayesian Optimization in combination with an aggressive
  racing mechanism"*), "Optimization across Instances" (`instances`, `instance_features`, and *"Those
  instance features are used to expand the internal X matrix and thus play a role in training the
  underlying surrogate model"*), and "Multi-Fidelity Optimization" (real-valued vs **instance-based
  budgets**; `min_budget`/`max_budget` as stage instance counts; Hyperband as the default intensifier).
  <https://automl.github.io/SMAC3/latest/advanced_usage/4_instances/>,
  <https://automl.github.io/SMAC3/latest/advanced_usage/2_multi_fidelity/>
- **SciPy 1.18.0** — `scipy.stats.energy_distance` (definition and its relation to the non-distribution-free
  Cramér–von Mises distance).
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html>

**van Buuren, *Flexible Imputation of Missing Data*, 2nd ed.**

- **§2.8 "How many imputations?"** <https://stefvanbuuren.name/fimd/sec-howmany.html> — fetched; Rubin's
  variance-inflation factor, and the Graham/Olchowski/Gilreath (2007), Royston (2004), Bodner (2008) and
  White/Royston/Wood (2011) rules all quoted from the live page, plus van Buuren's own m=5-then-increase
  practice.
- **§3.4.3 "Number of donors"** <https://stefvanbuuren.name/fimd/sec-pmm.html> — fetched; `mice` default
  d = 5 as *"a compromise"*, Schenker & Taylor (1996), Morris/White/Royston (2014) d ∈ {5, 10}, Kleinke
  (2017) small-sample caveat, and the warnings against d=1 and d≈n/10.
- **§6.5.2 "Convergence"** <https://stefvanbuuren.name/fimd/sec-algoptions.html> — fetched; the
  intermingling criterion, the between-vs-within variance criterion, and the verbatim warning that
  *"automated convergence monitoring (as by a machine) is unsafe and should be avoided"*.
- **§2.5, §2.6, §6.6, §1.3** — not re-fetched for this document; already verified and quoted in
  `docs/learning/evaluation-and-validation-plan.md`'s own sources-consulted section, and reused here on
  that basis. Where quoted above (§6.6's MAR caveat, §2.5's coverage-as-floor, §2.6's RMSE result), the
  wording is taken from that verified record.

---

## Discarded

Recorded so the same ground is not re-covered.

- **ACM Digital Library (`dl.acm.org/doi/10.1145/3377930.3389817`)** — HTTP 403 for the GECCO 2020 MOTPE
  paper. Substituted the JAIR 2022 journal version, which is open and covers the same algorithm in more
  depth.
- **IEEE Xplore (`ieeexplore.ieee.org/document/996017`)** — returned no fetchable body for NSGA-II.
  Bibliographic details confirmed by triangulating the DOI Optuna's source cites against multiple
  indexes; abstract deliberately **not** quoted, since it could not be read from a primary page.
- **ScienceDirect (`S2214716015300270`)** — HTTP 403 for the irace paper. The first-party IRIDIA project
  page carries the citation and the self-description, which is enough for a resource guide; the article
  body was not read.
- **Springer Link (`10.1007/978-3-642-25566-3_40`)** — 303 redirect to an identity-provider gateway for
  the 2011 SMAC paper. Substituted the UBC first-party project page for the citation.
- **`botorch.org/docs/acquisition/` and `botorch.org/docs/multi_objective`** — HTTP 404; the site
  restructured. The readthedocs API reference is the live surface and was used instead.
- **`automl.github.io/SMAC3/main/...`** — HTTP 404. SMAC3 docs moved from `/main/` to `/latest/`; any
  older bookmark to `/main/` is dead.
- **Optuna's `optuna._hypervolume` package (WFG, HSSP, box decomposition)** — read to confirm how
  hypervolume is computed and how MOTPE's tie-break works, but **deliberately not recommended as a
  resource**: it is private API with no stability guarantee. Use `plot_hypervolume_history` if you need
  the number.
- **A large body of arXiv work on imputation-model selection without ground truth** (Wasserstein /
  Jensen–Shannon based imputation ranking, GAN- and diffusion-imputation benchmarks). Skimmed via search
  and not pursued: the evaluation plan already names this body as its Stage-5 gap, and none of it
  supplies what this document needed — a *proper* score with a propriety proof. Gneiting & Raftery does,
  and it is 19 years older and far better established. Revisit only if a specific benchmark protocol is
  wanted.
- **Population-Based Training resources** (already in the HPO plan, Stage 6). Not extended here: PBT's
  defining property — hyperparameters mutating mid-training — has no analogue in a MICE fit worth
  exploiting.
- **BOHB (Falkner, Klein & Hutter, 2018).** Considered for Part 3 and dropped: it combines TPE with
  Hyperband, and its fidelity assumption is the epoch-like one that Part 3 argues does not transfer.
  SMAC's instance budgets and irace's racing dominate it for this problem, and SMAC shares an author.
