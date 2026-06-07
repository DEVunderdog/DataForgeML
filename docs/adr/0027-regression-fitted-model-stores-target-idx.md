# ADR 0027: Regression fitted model stores `target_idx` explicitly alongside the fitted `IterativeImputer`

**Status:** Accepted

## Context

`_fit_regression` constructs a joint numpy array as `[col] + feat_cols`, placing the imputation target at column index 0. At inference time, `FittedImputer` extracts the imputed target values from the `IterativeImputer` output by this same index. Previously this was an implicit convention — `output[:, 0]` — with no stored record of which index is the target.

MICE does not have this problem: it imputes all assigned columns jointly and inference extracts any column by looking it up in the stored `model_cols["mice"]` list. No single column is a designated target; all are equally targets and predictors for each other.

Regression is structurally asymmetric — `col` is the imputation target and `feat_cols` are the predictors. This asymmetry requires knowing the target index both at training time and at inference time. The risk of the implicit convention: if a future call site reorders `[col] + feat_cols` (e.g. alphabetical sort, deduplication that shifts index), target and features are silently swapped. The model trains on the wrong target with no error and no warning.

## Decision

`_fit_regression` returns a `FittedRegression` dataclass containing:

- `model` — the fitted `IterativeImputer`
- `target_idx` — the index of the target column in the joint array (always `0` by construction, but stored explicitly)
- `all_cols` — the full column list `[col] + feat_cols`

A runtime assertion fires at the top of `_fit_regression` confirming `all_cols[0] == col` before the array is constructed. At inference time `target_idx` is read from the stored `FittedRegression` rather than assumed to be `0`.

`model_cols["regression:{col}"]` changes from `feat_cols` to `all_cols` — the full column order including the target — consistent with `model_cols["mice"]` already storing the full MICE column list.

## Considered Options

**Named constant only** (`_TARGET_COL_IDX = 0`) — names the convention but does not make inference self-contained. A future refactor could change the column construction order without triggering any failure at inference.

**Assertion only** — catches construction-time reordering but inference still carries a bare `[:, 0]` magic number. A reader must trace back to the call site to understand what index 0 means.

**Separate arrays at training time** — extract target and features into separate numpy arrays, avoiding the joint array. Rejected: `IterativeImputer` (introduced in Scope 0) requires a single joint matrix input. Separate arrays would still need to be concatenated before fit, reintroducing the index dependency at inference with no net gain.

## Rationale

- **MICE alignment** — MICE already extracts imputed values by column-list lookup, not by position convention. Storing `all_cols` and `target_idx` brings Regression into the same pattern.
- **Self-documenting inference** — reading `fitted_regression.target_idx` makes the extraction intent explicit. A bare `[:, 0]` requires the reader to know the construction convention; a stored index does not.
- **Belt-and-suspenders** — the assertion catches call-site reordering at training time; the stored index protects inference even if the assertion is bypassed (e.g. `python -O`).

## Consequences

- `FittedImputer.from_dict()` needs a migration path for legacy regression entries that stored only `feat_cols` under `model_cols`. Legacy entries can be identified by the absence of a `target_idx` field and migrated by prepending the target column name derived from the key (`"regression:{col}"` → `col`) to the stored `feat_cols`.
- `_fit_regression` becomes a pure function: returns `FittedRegression | None` (`None` signals fallback to Median). The caller (`NumericImputer.fit()`) assembles the dicts and calls `_fallback_to_median` explicitly. This is a consequence of the broader Scope 12 decision to eliminate mutation side effects in `_fit_regression`.
- `_fallback_to_median` also becomes pure: returns a modified `ColumnImputationRecord`; the caller writes it back into `records`. Consistent with the above.
- The caller raises on a duplicate `model_key` rather than silently overwriting.
