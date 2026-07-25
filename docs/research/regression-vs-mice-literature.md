# Industry / literature standard: per-column regression imputation vs joint chained equations

Research ticket [#404](https://github.com/DEVunderdog/DataForgeML/issues/404) (part of map [#403](https://github.com/DEVunderdog/DataForgeML/issues/403)).
Grounds the decision to collapse the per-column single-target `Regression` strategy into a unified chained-equations strategy.

Primary sources consulted:

- Stef van Buuren, *Flexible Imputation of Missing Data*, 2nd ed. (FIMD), free online at <https://stefvanbuuren.name/fimd/>.
- scikit-learn User Guide & API reference for `IterativeImputer` (<https://scikit-learn.org/stable/modules/impute.html>, <https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html>).
- R `mice` package reference: `quickpred` (<https://amices.org/mice/reference/quickpred.html>).

Where a source does not clearly state something, it is flagged as such rather than inferred.

---

## Q1. Is per-column single-target regression a recognised named pattern, or subsumed by MICE / chained equations?

**Both are true, at different levels.** Single-variable "regression imputation" is a *named, textbook method* — but only as a **building block** for the univariate (one-column-missing) case. Once multiple columns are missing, the literature routes that building block through Fully Conditional Specification (FCS) / MICE rather than treating N independent per-column regressions as a first-class multivariate method.

- van Buuren names and describes **"regression imputation"** (FIMD §1.3.4) and **"stochastic regression imputation"** (§1.3.5) as distinct methods (`norm.predict`, `predict + noise`, `predict + noise + parameter uncertainty`). So the pattern itself has a name.
- Deterministic regression imputation is treated as *statistically deficient*, not merely a simpler variant: it shrinks confidence intervals ("a confidence interval of method `norm.predict` is much too short, leading to substantial undercoverage", coverage 0.652 vs target 0.95, FIMD §1.3), and "when missing data occur in predictors rather than outcomes... Method `norm.predict` is now severely biased" (bias 34.7% in his simulation). His conclusion: "It is always better to include parameter uncertainty" (FIMD §1.3).
- For the **multivariate** case, van Buuren frames FCS as the generalization of the single-column method, not a competitor to it: "FCS is a natural generalization of univariate imputation" (FIMD §4.5.1), with the univariate methods of Chapter 3 acting as the per-variable building blocks.
- scikit-learn's `IterativeImputer` is exactly this generalization mechanized: "It models each feature with missing values as a function of other features... in an iterated round-robin fashion: at each step, a feature column is designated as output `y` and the other feature columns are treated as inputs `X`. A regressor is fit on `(X, y)`... Then, the regressor is used to predict the missing values of `y`. This is done for each feature... repeated for `max_iter` imputation rounds." (sklearn User Guide, Multivariate feature imputation.) It states it "was inspired by the R MICE package".

**Takeaway for Q1:** per-column single-target regression is a recognised *univariate* method, but there is **no recognised named multivariate pattern of "fit N independent per-column regressions, each keeping one target."** The literature's answer to "many columns missing" is chained equations (FCS/MICE) — the same `IterativeImputer` algorithm the library already uses for both strategies.

---

## Q2. Predictor selection / breadth — all columns, or a selected subset?

The stated *ideal* is breadth; the *practical norm at scale* is deliberate correlation-based selection. "Use every numeric column" is the theoretical default but is explicitly walked back for wide data.

- **van Buuren (FIMD §6.3, "Building the imputation model" / predictors):** "It is often beneficial to choose as large a number of predictors as possible" — this minimizes bias and maximizes efficiency. **But**: "For datasets containing hundreds or thousands of variables, using all predictors may not be feasible (because of multicollinearity and computational problems)," and there he recommends a subset of **"no more than 15 to 25 variables."** He gives a four-step selection strategy: (1) include all analysis-model variables and outcomes; (2) include variables related to nonresponse; (3) include variables that explain considerable variance in the target (by correlation strength); (4) drop predictors with too many missing values within the incomplete cases. Tool: `quickpred()`.
- **R `mice::quickpred`:** builds the `predictorMatrix` automatically by correlation. For each target it computes, on complete cases, the correlation of each candidate predictor with (a) the target values and (b) the target's *missingness indicator*; if the larger absolute value exceeds `mincor` the predictor is included. Defaults: **`mincor = 0.1`**, `minpuc = 0` (minimum proportion of usable cases), `include = ""`, `exclude = ""`. So R's canonical helper is a **correlation-thresholded subset selector**, not "everything."
- **scikit-learn `IterativeImputer`:** default `n_nearest_features=None` means "all features will be used" as predictors for each target — i.e. full breadth is the sklearn default. But it provides `n_nearest_features` precisely to restrict breadth: "Nearness between features is measured using the absolute correlation coefficient... the neighbor features are... drawn with probability proportional to correlation for each imputed target feature. Can provide significant speed-up when the number of features is huge." (API reference.)

**Takeaway for Q2:** Breadth is good *in principle* (van Buuren) and is sklearn's default, but **deliberate, correlation-driven predictor selection is the recognised norm once the column count is large** — van Buuren's 15–25 cap and four-step rule, and R's `quickpred` at `mincor = 0.1`. There is no primary source endorsing "always use every numeric column regardless of width." The library's current `Regression` = all-numeric vs `MICE` = block-only split maps onto the two ends of a single **predictor-breadth dial**, which is exactly how the literature treats it (a tunable, not two algorithms).

---

## Q3. Joint chained-equations pass vs independent per-column fits when many columns are missing

The standard is **one joint chained-equations (Gibbs) pass over all incomplete columns together**, iterated — not independent per-column fits. The accuracy rationale is consistency of the joint distribution and letting cross-column associations propagate.

- **van Buuren (FIMD §4.5):** "The MICE algorithm is a Markov chain Monte Carlo (MCMC) method" and acts as "a Gibbs sampler" when the conditionals are compatible (§4.5.2). Iteration is the point: fitting each column *conditional on current fills of the others*, cycling, lets "observed correlations percolate into the imputations." Independent (single-pass, conditionally-independent) fits distort associations — his worked example: under conditional independence the induced correlation is 0.9 × 0.9 = 0.81, "whereas the true value equals 0.7" (§4.5.7). Incompatibility of conditionals is noted as "a relatively minor problem in practice" (§4.5.3).
- **scikit-learn** operationalizes this as the round-robin over `max_iter` rounds (Q1 quote): every incomplete feature is re-modelled against the others each round within a *single* fitted imputer, not N separate imputers.

**Takeaway for Q3:** The literature squarely favours a **single joint chained pass** over N independent per-column fits when several columns are missing. This is both the accuracy argument (associations stay coherent; uncertainty handled by iteration) and, as a *consequence*, the efficiency argument — one fit instead of the library's current N separate full `IterativeImputer` fits. Note: `IterativeImputer` returns a **single** imputation, "differ[ing] from [mice] by returning a single imputation instead of multiple imputations" — so the library does not get MICE's between-imputation uncertainty accounting for free; that is a known limitation of the sklearn engine, independent of the joint-vs-single collapse.

---

## Q4. Predictor NULLs at predict / serve time (the transform-time round-robin problem)

scikit-learn has a concrete, documented answer; the classical MICE literature largely does **not** address a train/serve split, because `mice` is designed to impute a *fixed* dataset, not to be fitted once and applied to new rows.

- **scikit-learn `IterativeImputer.transform`:** inductive mode is explicit — "To support imputation in inductive mode we store each feature's estimator during the `fit` phase, and predict without refitting (in order) during the `transform` phase." (API reference.) The stored `imputation_sequence_` is a list of `(feat_idx, neighbor_feat_idx, estimator)` tuples of length `n_features_with_missing_ * n_iter_`, replayed in the same order at transform time. So when new data has NULLs in *predictor* columns, transform re-runs the same round-robin — each stored per-feature estimator fills its target using the current (possibly just-filled) values of its neighbours, for the same `n_iter_` rounds seen in fit. The docstring warns this is stochastic unless `random_state` is fixed.
- **MICE literature (van Buuren):** FIMD does not present a canonical "apply the fitted imputation model to brand-new observations with missing predictors" procedure — MICE's target is multiple imputation of the analysis dataset in hand, with uncertainty carried into pooled estimates. (Not found: any FIMD prescription equivalent to sklearn's stored-estimator replay for serving new rows.) The relevant MICE-adjacent construct is that each column's conditional model is defined given the others, and the Gibbs cycle already tolerates other columns being partly imputed — which is the same structural idea sklearn's transform replay leans on.

**Takeaway for Q4:** The transform-time round-robin (predictor NULLs at serve time) is a **solved problem in the sklearn engine the library already builds on** — store per-feature estimators, replay the sequence for `n_iter_` rounds. A unified joint chained unit inherits this behaviour directly and consistently; N independent per-column `Regression` fits each carry their *own* internal round-robin (the ADR-0067 train/serve skew), so collapsing to one joint unit **reduces** the number of independent serve-time replays rather than complicating it.

---

## Bottom line for DataForgeML

- **Collapsing per-column `Regression` into joint chained equations is supported by the literature.** There is no recognised named multivariate method of "N independent single-target regressions." The recognised generalization of single-column regression to multi-column missingness *is* chained equations / FCS (FIMD §4.5.1), which is the very same `IterativeImputer` algorithm both current strategies wrap. A single joint chained pass is the standard for accuracy (coherent joint distribution, associations propagate via Gibbs iteration; FIMD §4.5) and, as a consequence, for cost (one fit vs N).
- **Predictor breadth should be a dial, not a strategy identity — and its default should not be a naïve "all numeric columns always."** Breadth is beneficial in principle (FIMD §6.3; sklearn default `n_nearest_features=None` = all features), but every primary source provides an explicit escape hatch for wide data: van Buuren caps at 15–25 predictors with a correlation-strength selection rule, and R's canonical `quickpred` selects by `mincor = 0.1`. So a defensible default is **full numeric breadth at modest width, correlation-thresholded selection past a width threshold** — matching the `Regression`(all-numeric) ↔ `MICE`(block-only) span as the two ends of one predictor-breadth dial.
- **Serve-time predictor NULLs are already handled** by the sklearn stored-estimator round-robin replay; a unified joint unit inherits it cleanly and collapses N independent replays into one.
- **One caveat to keep in the ADR:** `IterativeImputer` yields a *single* imputation, not MICE's multiple imputations, so the collapse does not by itself buy proper between-imputation uncertainty — that limitation exists today and is orthogonal to the joint-vs-single decision.
