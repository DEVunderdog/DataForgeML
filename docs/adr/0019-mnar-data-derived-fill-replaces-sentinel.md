# ADR 0019 — MNAR fill is data-derived (mean/median), not a user-configured sentinel

## Status
Accepted

## Context

MNAR columns were filled with a hardcoded sentinel value (`mnar_constant_fill`, default `-1`) alongside a binary missingness indicator. The sentinel was a user-configurable constant applied uniformly to all MNAR columns regardless of their distribution.

Three problems with this approach:

1. **Outlier phantom mode.** `-1` is almost always outside the observed range of a numeric column. Linear models and distance-based models treat it as a real extreme value, creating a directional bias for the missing rows that has nothing to do with the MNAR mechanism.

2. **Redundant but misleading for tree models.** Tree-based models learn a split at `col == -1` which is equivalent to `col_missing == 1`. The indicator already carries this signal — the sentinel fill adds no information but pollutes the feature space.

3. **`mnar_constant_fill` is a global override** applied identically to all MNAR columns, ignoring each column's distribution. A column ranging 0–100 and a column ranging 0–1,000,000 both receive `-1`, producing inconsistent distances from the column centre.

## Decision

Replace the sentinel fill with a **data-derived fill** computed from the non-missing rows of each MNAR column:

- `SkewSeverity.Normal` → observed mean
- Any other severity → observed median

The fill value centres the MNAR-missing rows within the observed distribution. It is the least-information state for the missing rows: linear models see "at the mean/median," adding no directional signal. Tree models ignore the fill anyway because the indicator dominates. The Phase 2 output remains null-free in all cases.

`ImputationStrategy.Constant` is renamed to `ImputationStrategy.MNAR` to reflect that this strategy is exclusively for declared MNAR columns, not for arbitrary sentinel fills. `mnar_constant_fill` is removed from `NumericImputationConfig` — the fill is always data-derived and is not user-configurable at the strategy level.

`FittedImputer.from_dict()` migrates legacy serialised `"constant"` strategy values to `"mnar"` automatically.

## Consequences

- Linear and distance-based models no longer receive an outlier-constant for MNAR-missing rows.
- The `mnar_constant_fill` config parameter is a breaking removal. No safe migration value exists (the default `-1` was the problem being fixed).
- The serialisation format changes from `"constant"` to `"mnar"`. Old `FittedImputer` dicts are migrated transparently in `from_dict()`.
- Users who deliberately set `mnar_constant_fill` to a domain-specific sentinel (e.g. `-999`) lose that capability. If a column-level fill override is needed in future, it belongs in a `per_column_mnar_fill` dict in `NumericImputationConfig`, consistent with the per-column override pattern introduced in Scope 3.
