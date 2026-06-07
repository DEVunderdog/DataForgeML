# ADR 0009: Phase 2 effective null boundary normalization

**Status:** Accepted

Phase 1 defines *effective null* as a dtype-driven concept covering string sentinels (`"NA"`, `"NAN"`, `"NULL"`, `"NONE"`, `"?"`), empty strings, float `NaN`, and float `Inf` — in addition to Polars-native nulls. The `ImputationResult` contract (CONTEXT.md) states that the output DataFrame has all effective nulls filled. Phase 2's internals, however, used ad-hoc null checks at six separate sites (split-imbalance check, Passthrough violation check, indicator expressions, scalar fill, `_df_to_numpy`, and `_clean()`), each catching only Polars nulls and NaN — silently passing through sentinels, empty strings, and `Inf`.

## Decision

Phase 2 normalizes the entire DataFrame to Polars-native null at the boundary — at the very top of `ImputationOrchestrator.fit()` and `FittedImputer.transform()`, before any other operation. A single `_resolve_effective_nulls(df)` function converts every effective null (sentinels, empty strings, `NaN`, `Inf`) to Polars `null` using the same dtype-driven rules as Phase 1. All downstream Phase 2 code then works exclusively with Polars nulls, simplifying every internal site to a plain `null_count()` check or `fill_null()` expression.

The null-detection primitives (`_SENTINEL_STRINGS`, `_sentinel_eligible`, `_inf_eligible`) are moved from `profiling/_null_detection.py` to `utils/_null_detection.py` so both Phase 1 and Phase 2 import from a single source of truth. The `_resolve_effective_nulls` function itself lives in `imputation/` as a Phase 2 boundary operation.

## Considered Options

**Site-by-site detection** — teach each of the six Phase 2 sites to detect the full effective null set inline. Rejected because: it requires six independent fixes, any new site added to Phase 2 will get it wrong by default, and the scattered NaN-specific branches (`fill_nan()`, `is_nan()`) remain scattered rather than collapsed.

**Cross-package private import** — keep primitives in `profiling/_null_detection.py` and import them directly into `imputation/`. Rejected because importing private names across sibling packages is a non-obvious dependency that breaks package encapsulation.
