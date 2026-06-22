# ADR 0035: BoundedDiscrete model-based routing with domain-snap

Amends ADR-0018.

## Context

ADR-0018 established `NumericKind.BoundedDiscrete → Mode, unconditionally` on the grounds that model-based strategies (KNN, Regression, MICE) produce continuous predictions that are not valid members of a column's finite domain. The rule was simple and correct for that premise.

The premise is incomplete. A continuous prediction `3.47` for a `{1,2,3,4,5}` column is not a valid domain member — but `clip(round(3.47), 1, 5) = 3` is. The domain constraint is a post-processing concern, not a routing blocker. Routing unconditionally to Mode ignores all column correlations and missingness mechanism signals, producing a less accurate fill for MAR-suspect and High/Severe MCAR BoundedDiscrete columns.

## Decision

`NumericKind.BoundedDiscrete` is promoted from an unconditional Mode route at Priority 5 to a **dedicated routing gate at Priority 3** (after MNAR at Priority 2, before the Unpredictable guard and MARSuspect). Inside the gate the column runs its own sub-chain and exits; the outer routing priorities 4–7 are never reached for BoundedDiscrete columns.

### BoundedDiscrete sub-chain (first match wins)

1. `NonlinearityTag.Unpredictable` → **Mode**. No model-based uplift is available; Mode is the correct scalar fill for a discrete bounded column.
2. `NumericFlag.NearConstant` → **Mode**. 90%+ of values share the same value; model-based is wasteful and the prediction collapses to the mode anyway.
3. `NumericFlag.Bimodal` → deferred (Bimodal Imputation Framework for BoundedDiscrete is a future scope).
4. `MARSuspect` → domain-snapped MAR sub-chain (MICE → Regression → KNN → Mode terminal, same size guards as Priority 5).
5. MCAR by severity → domain-snapped MCAR sub-chain (same severity routing as Priority 7, Mode as terminal fallback replacing Median).

### Domain-snap rule

Every model-based prediction produced inside the gate is post-processed as:

```python
int(clip(round(prediction), NumericStats.min, NumericStats.max))
```

Domain-snap applies **only to model-based predictions** (KNN, Regression, MICE). Scalar fills inside the gate are always Mode — no Mean or Median is used, so no snap is needed for scalar paths.

### Snap bound storage

`ColumnImputationRecord` gains an optional field:

```python
domain_snap_bounds: Optional[tuple[float, float]] = None
```

Set to `(NumericStats.min, NumericStats.max)` for BoundedDiscrete columns assigned a model-based strategy (KNN, Regression, MICE). `None` for all other columns. `to_dict()` serialises it as a two-element list or `null`; `from_dict()` reads it back. `transform()` reads this field and applies the snap after model inference.

### Mode as the only scalar fill

Mode is the scalar fill for **all** routing outcomes inside the BoundedDiscrete gate where a model-based strategy is not used. This includes:

- `NonlinearityTag.Unpredictable` and `NumericFlag.NearConstant` (unconditional)
- All MAR and MCAR severity paths where model-based size guards are not met (terminal fallback)
- MCAR Minor + Normal skew — previously routed to Mean; now Mode. Mean introduces no statistical advantage for a discrete bounded column, and eliminating it removes the special-case Mean-snapping logic from `_resolve_fill_value`.

Median is never used inside the gate. Median is not guaranteed to be a valid domain member (e.g. median of `{1,2,3,4,5}` on an even-row sample is `2.5`), and the snapped alternative (`clip(round(median), min, max)`) offers no principled advantage over Mode for a discrete bounded column with no feature context.

### MNAR BoundedDiscrete

When a declared MNAR column has `NumericKind.BoundedDiscrete`, the fill computed in `_resolve_fill_value` is Mode (not the skew-driven mean/median used for all other MNAR columns). The `ImputationStrategy.MNAR` is preserved — the missingness indicator is still added. This is the only case where the BoundedDiscrete scalar-fill rule reaches inside Priority 2; the BoundedDiscrete gate at Priority 3 is never reached for MNAR columns.

### Runtime model-based fallback (`_fallback_to_mode`)

When a model-based fit fails at runtime (e.g. `_fit_regression` returns `None`), `_fallback_to_median` is the standard recovery path for non-BoundedDiscrete columns. For BoundedDiscrete columns, `_fallback_to_mode` is called instead. The call site detects BoundedDiscrete via `record.domain_snap_bounds is not None` — this field is set by `_resolve_domain_snap_bounds` for all BoundedDiscrete model-based columns and is `None` for all others.

### Unpredictable guard for non-BoundedDiscrete columns

The `NonlinearityTag.Unpredictable` pre-routing guard is also added to `_fit_one` for non-BoundedDiscrete columns (firing at Priority 4, having previously been missing from the implementation despite being codified in ADR-0016 and CONTEXT.md).

## Rejected alternative

Keep `BoundedDiscrete → Mode, unconditionally` (ADR-0018 as written). Rejected because:
- Mode ignores all column correlations and missingness mechanism signals.
- For a MAR-suspect or High-severity BoundedDiscrete column, Mode fills every missing row with the same value regardless of what correlated features indicate — a provably less accurate fill when the column's missingness is non-random.
- The domain constraint is correctly enforced by post-processing (domain-snap), not by routing away from model-based strategies.

## What ADR-0018 said vs. what is now true

| Claim in ADR-0018 | Status |
|---|---|
| "Model-based strategies produce continuous predictions that are not valid members of the domain" | Still true — domain-snap is the fix, not a contradiction |
| "Mode fires regardless of missingness severity or mechanism" | Amended: Mode is the scalar fill for all paths inside the gate; model-based strategies (KNN, Regression, MICE) are attempted where signal exists and size guards allow |
| BoundedDiscrete at Priority 5 (after MARSuspect) | Amended: Priority 3 gate (before MARSuspect) |
| Mean used for Minor + Normal skew inside the gate | Amended: Mode replaces Mean for all scalar fills inside the gate |
| Scalar fills snapped at fit time | Amended: no scalar fills require snapping — all scalar fills inside the gate are Mode, which is always a valid domain member |
| MNAR columns bypass the BoundedDiscrete gate | Still true — MNAR is Priority 2; BoundedDiscrete gate is Priority 3. MNAR BoundedDiscrete columns use Mode as fill value (amendment to Priority 2 MNAR behaviour) |

## Scope boundary

The Bimodal Imputation Framework for BoundedDiscrete (domain-constrained GMM Sampling, Cluster-Conditional fills snapped to the nearest valid discrete value) remains a future scope and is not changed by this ADR. The existing ADR-0032 bimodal provisions for BoundedDiscrete are unchanged.

`_LegacyRegressionModel` (a private migration shim for pre-Issue-#141 serialised `(BayesianRidge, feat_means)` tuples) is removed in this scope: the class, its migration block in `FittedImputer.from_dict()`, and `test_regression_legacy_migration_in_from_dict` are deleted. All serialised models produced after Issue #141 carry `FittedRegression` directly.
