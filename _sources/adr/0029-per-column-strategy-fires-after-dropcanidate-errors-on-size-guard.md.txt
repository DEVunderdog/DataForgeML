# ADR 0029: `per_column_strategy` fires after `DropCandidate` and raises at fit time when size guards fail

**Status:** Accepted

## Context

Scope 14 adds `per_column_strategy` to `NumericImputationConfig` — an explicit per-column strategy override that lets users bypass the automatic routing engine when they have domain knowledge the engine cannot derive. Two structural decisions had to be made about how this override interacts with the existing priority chain in `_fit_one`.

**Decision 1 — Where in the priority chain does the override fire?**

Three options existed:
- Priority 0 (before `DropCandidate`): user can fill any column regardless of missingness fraction.
- Priority 1.5 (after `DropCandidate`, before MNAR): user overrides routing but DropCandidate remains a hard gate.
- Priority 2.5 (after MNAR): both DropCandidate and MNAR are inviolable.

**Decision 2 — What happens when a model-based override (KNN, Regression, MICE) is declared but the dataset does not meet the size guard for that strategy?**

Three options existed:
- Bypass size guards entirely: user gets what they asked for regardless of dataset size.
- Silent fallback to Median: same behaviour as the routing engine's fallback.
- Raise `ValueError` at fit time: user is told exactly which guard was not met and what to change.

## Decision

**Priority 1.5** — `per_column_strategy` fires after `DropCandidate` and before MNAR routing. `DropCandidate` (>50% missing) is a hard gate that cannot be bypassed by a strategy override.

**Raise `ValueError` at fit time** when a model-based override is declared but the dataset does not meet the size guard for that strategy (`n_rows < regression_min_rows` for Regression; `n_rows > knn_max_rows` or `n_features > knn_max_features` for KNN). No silent fallback to Median.

## Considered Options

**Priority 0 (before DropCandidate):** rejected because `DropCandidate` is a data-quality gate, not a routing preference. A column that is >50% missing should not be silently fillable via a general-purpose override field. Users who genuinely need to rescue a DropCandidate column require a dedicated escape hatch (future scope) — not a side effect of the strategy override mechanism.

**Priority 2.5 (after MNAR):** rejected because one of the three motivating use cases for `per_column_strategy` is filling a column with a domain-specific constant (e.g. `0` for a transaction-count column) without the MNAR semantics (no indicator, no skew-driven fill). If the override fired after MNAR, these users would have no clean path — they would be forced to declare the column MNAR and accept semantics that are wrong for their use case.

**Silent fallback to Median on size guard failure:** rejected because it silently contradicts user intent. The `ColumnImputationRecord` would show `Median` while the config declares `Regression` — the audit trail would disagree with the config. This is the worst outcome for a feature whose primary purpose is explicit, inspectable user control over strategy selection. A `ValueError` is actionable (lower the threshold or choose a different strategy) and impossible to miss.

**Bypass size guards entirely:** considered but rejected as too permissive. Size guards exist because model fitting on very small datasets produces unstable, unreliable imputed values. While the user is taking responsibility by overriding, giving no signal at all means they may not realise they are in an unsafe regime. A `ValueError` is a one-time friction cost that surfaces a real data constraint.

## Rationale

- **DropCandidate is a data-quality gate, not a routing preference.** The "routing decision tree" that `per_column_strategy` bypasses is the MNAR/MAR/MCAR/Discrete chain — the choices about how to fill. The DropCandidate gate answers whether filling is meaningful at all. These are different questions and should not be collapsed into one override.
- **Silent fallbacks corrupt the audit trail.** The `ColumnImputationRecord` is the library's contract with the user about what happened to each column. A record that shows `Median` when the user declared `Regression` is a lie in the audit trail. Errors that force the user to make their intent explicit are preferable to silent deviations.
- **Construction-time and fit-time errors catch mistakes at the earliest possible moment.** Conflict validation (column in both `mnar_columns` and `per_column_strategy`) fires at construction — before any data is touched. Size guard validation fires at fit — as early as the dataset is available. Neither defers failure to transform time.

## Consequences

- A user who wants to fill a >50%-missing column cannot use `per_column_strategy` to do so. A dedicated escape hatch must be added in a future scope if this use case arises.
- A user who forces a model-based strategy on a small dataset must either lower the relevant size threshold (`regression_min_rows`, `knn_max_rows`, `knn_max_features`) or choose a different strategy. The `ValueError` message names the specific guard and current values.
- MNAR columns must not be added to `per_column_strategy` — they do not go through the routing engine and their R² is not a meaningful signal; doing so fails the construction-time conflict validation.
