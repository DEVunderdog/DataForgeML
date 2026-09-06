# `score_accuracy` fold stand-ins replicate M's recipe but re-learn its parameters

`score_accuracy` cannot honestly grade the delivered model M directly — M trained on every complete cell, so scoring it in-sample is optimistically biased (ADR-0058). It must grade a **stand-in**: the same training recipe fit on a fold's training rows, which serves as a conservative baseline for M (M sees more data, so it is expected to do at least as well). This ADR fixes *how faithful* that stand-in must be, because "make it resemble M" pulls in two opposite directions depending on which part of M you copy.

We split every model into two parts and treat them oppositely inside the fold loop:

- **Recipe (configuration / hyperparameters)** — the choices made *before* seeing the answers: estimator type, regularization, `max_iter`, `tol`, `n_neighbors`, `weights`, feature set. → **Frozen from M.** The stand-in must be the same recipe, differing from M only in that it trained on fewer rows.
- **Learned parameters** — everything M *fit from the data*: regression coefficients, KNN feature scaling (`col_means`/`col_stds`), MICE fills, centroids. → **Re-learned on each fold's training rows only.** These were computed from all rows *including the held-out fold*; freezing them leaks the answers back in and re-introduces the exact in-sample bias CV exists to remove.

Rule: **freeze the recipe, re-learn the parameters; never freeze anything M learned from the data.**

## Status

superseded by ADR-0087 — refined ADR-0058, which is itself superseded. There is no fold loop left to govern: C2ST needs no refit, no fold stand-in and no complete-row basis, so **Fold Stand-in Fidelity** ceases to be a term. The freeze-the-recipe / re-learn-the-parameters rule is recorded here should any future metric need a held-out refit.

## Considered Options

- **Replicate M as fully as possible, including its learned parameters.** Rejected as leakage: reusing M's full-data scaling or coefficients means the stand-in has already "seen" the cells it is being graded on, so the grade is no longer held-out.
- **Rebuild each fold's recipe from scratch on the fold's data (the prior Regression behaviour).** Rejected: re-deriving `max_iter`/estimator per fold lets the stand-in drift into a *different* recipe than M, so its grade describes a fold-tuned variant rather than a faithful baseline for M.

## Consequences

- **Regression** was too loose on the recipe: `_score_regression_cv` rebuilt a fresh estimator sized to the fold and recomputed `max_iter`/`tol`. It now reads M's exact `estimator`, `max_iter`, and `tol` straight off `FittedRegression.model` and only refits on the fold rows — mirroring how MICE already reuses its fitted model's parameters. The `profile` nonlinearity tag is no longer consulted for Regression scoring.
- **KNN** was too tight on the learned part: `_score_knn_cv` scaled every row (including each validation fold) by M's full-data `col_means`/`col_stds`. It now recomputes scaling from each fold's training rows only, while still freezing M's `n_neighbors` and `weights` (those are recipe).
- **MICE** already followed this rule (freezes estimator/`max_iter`/`tol`, refits per fold) and is unchanged — it is the template the other two now match.
- The reported `r2_cv`/`rmse`/`mae` are honest conservative baselines for the delivered M, with no held-out leakage in any strategy.
