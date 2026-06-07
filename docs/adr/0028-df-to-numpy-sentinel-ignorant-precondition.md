# ADR 0028: `_df_to_numpy` is sentinel-ignorant; normalization is the caller's precondition

**Status:** Accepted

## Context

`_df_to_numpy` (in `imputation/_utils.py`) converts a Polars DataFrame to a float64 numpy array, replacing Polars-native nulls with NaN via `fill_null(float("nan"))`. Scope 13 identified that numeric sentinel values (e.g. `-999` stored as an actual float) pass through unchanged — they become `-999.0` in the numpy array, not NaN. KNN, MICE, and BayesianRidge then treat the sentinel as a real observation, silently corrupting model training and inference.

The sentinel map (which column holds which sentinel value) lives in `StructuralProfileResult`, produced by Phase 1. Scope 5 owns end-to-end numeric sentinel support: declaration in `ProfileConfig`, storage in `StructuralProfileResult`, and normalization of sentinel values to Polars nulls before Phase 2 consumes the DataFrame.

## Decision

`_df_to_numpy` does not accept or apply a sentinel map. Its documented contract is: all effective nulls — including numeric sentinels — must already be normalized to Polars `null` before calling. Sentinel normalization is not this function's responsibility.

The function's docstring states the precondition explicitly. No runtime guard is added — the precondition is satisfied structurally once Scope 5 ships.

## Considered Options

**Accept `sentinels: dict[str, float | int] | None = None` parameter** — `_df_to_numpy` replaces sentinel values with NaN internally, closing the gap in Scope 13 independently of Scope 5.

Rejected because: `_df_to_numpy` is a narrow array-conversion utility. Accepting a sentinel map introduces a cross-layer dependency — a leaf-node function gains knowledge of the profile contract (`StructuralProfileResult`) that belongs one layer up. Scope 5 must wire the normalization at every call site regardless; adding partial sentinel handling here duplicates that responsibility without eliminating it.

## Rationale

- **Single responsibility** — `_df_to_numpy` converts Polars nulls to NaN. Sentinel normalization is a separate concern; conflating them makes both harder to test and reason about independently.
- **Scope boundary** — Scope 5 owns the full end-to-end sentinel path. Splitting sentinel awareness between a utility function and Scope 5 creates two partial implementations with no clear owner.
- **Dependency direction** — utility functions are leaf nodes. Pulling `StructuralProfileResult` into `_utils.py` inverts that direction.

## Consequences

- Until Scope 5 ships, any call site that passes a DataFrame with un-normalized numeric sentinels produces silently corrupted numpy inputs. The precondition docstring is the only guard.
- Scope 5 must satisfy the `_df_to_numpy` precondition at all call sites — every path that constructs a numpy array for KNN, MICE, or Regression must normalize sentinels to Polars nulls first.
- If Scope 5 is deferred indefinitely, the sentinel contamination gap remains open. This risk is accepted explicitly in Scope 13.
