# The pure ImputationDecision object separates the data-free plan from execution

Phase 2 currently welds routing (which strategy per column) and parameter estimation (fitting MICE/KNN/Regression/GMM/cluster models) into one fused `fit()`. There is no way to inspect the plan, edit it, run models selectively, or resume later. This ADR specifies the keystone that unwelds them: a pure `ImputationDecision` object — the data-free plan produced from `(profile, shape, config)` alone — that execution consumes, persistence serializes, and the user can inspect and edit before anything trains. It is the reference instance of the Decision/Execution split charted in the wayfinder map (#333); the phase-agnostic contract is abstracted from it later.

The decision is a pure function of **`(profile, shape, config)`**, where `shape = (n_rows, n_features, column set)`. It is **value-free by construction**: it holds no value learned by the execution layer from `train_df` — no fill values, no fitted coefficients. It *may* hold profile-provided descriptors (including `min`/`max` as `domain_snap_bounds`), because those arrive through the same sanctioned profile channel as every other routing signal. Provenance of those inputs (full-dataset vs train-only) is governed entirely by the Split-Profiling Topology doctrine (ADR-0051); this object introduces no new leakage surface. Model *training* — the only step that can leak — still happens exclusively in the execution layer, on `train_df`.

Locked decisions:

- **New pure per-column type `ColumnImputationDecision`.** The existing `ColumnImputationRecord` becomes *decision + learned values*: it composes the decision (`record.decision`) plus `fill_value` and fit metadata. The pure type structurally cannot hold a fill value, so "what was decided" versus "what was learned" is un-blurrable. Per-column fields: `column`, `semantic_type`, `strategy`, `signals` (the routing rationale), `model_choice`, `domain_snap_bounds`, `indicator_flag`, `mnar`, `drop`. Out: `fill_value`, fitted model, convergence/`n_iter`.
- **Model choice is resolved at decide-time.** The concrete estimator family (`BayesianRidge` / `RandomForestRegressor` / `GradientBoostingRegressor`) is chosen from `(NonlinearityTag, n_rows, config)` when the plan is built — no longer deferred to fit time. This is the "Recipe" half of the Recipe/Learned split (ADR-0059) pulled one layer earlier; it makes the plan a complete, inspectable, overridable artifact.
- **The plan materializes its execution units.** The joint MICE block, the joint KNN block, and one unit per independent column (Regression / GMM-Sampling / Cluster-Conditional / scalar) are enumerated as `ImputationUnit` records — a frozen projection of the per-column map, computed once at build. Units are what execution resumes on and persistence keys checkpoints against. Unit ids: `"mice"` / `"knn"` for the joint blocks (exactly one of each), `"{strategy}:{column}"` for per-column units.
- **The plan is immutable.** Edits return a new `ImputationDecision` (`with_strategy(...)`, `with_model_choice(...)`); nothing is mutated in place. Units are re-derived at every construction, so a stale unit list is structurally impossible, and every plan that exists is valid by construction. Edit-time validates *legality* (declarable strategy for the column's semantic type — rejecting the output-only labels `Dropped`/`Passthrough`/`Indicator`/`MNAR`/`Constant`/`ClusterConditional`/`GMMSampling` with the same redirect messages the config already uses); *data-size feasibility* (size guards) is deferred to execution, keeping the object a function of shape without re-running guard arithmetic on every edit.
- **Whole-object shape.** `column_decisions` (editable map) + `units` (materialized tuple) + `decided_for_shape` + `config_snapshot` + `profile_provenance` + `dropped_columns` (convenience projection). The exact hashing/identity/serialization mechanics of `config_snapshot` and `profile_provenance` are deferred to the serialization/versioning (#337) and store-identity (#338) tickets.
- **`route()` stays the routing kernel; a new assembler builds the plan.** `_StrategyRouter.route()` keeps its one job — return `(strategy, signals)` — and is reused as-is. A new internal assembler calls `route()` per column, resolves `model_choice`, surfaces `domain_snap_bounds`, folds in the indicator/mnar/drop flags, builds each `ColumnImputationDecision`, derives the units, captures shape/config/profile provenance, and returns the immutable `ImputationDecision`.

The public-facing entry point (`decide()` placement at the package root, retiring the fused `fit_transform` / `ImputationOrchestrator` / `FittedImputer` seam per ADR-0050) is deliberately *not* decided here — it graduates from the map's fog after the execution and store tickets settle.

## Status

accepted

## Considered Options

- **Widen `ColumnImputationRecord` in place** with a "these fields are pure" convention instead of a separate type. Rejected: the boundary stays a convention rather than a structural guarantee, and the split is the whole point.
- **Carry only the strategy; pick the estimator family in execution.** Rejected: the plan would not be fully executable or editable — a user could not see or override that a column will train with `GradientBoostingRegressor`, which the ticket requires.
- **Mutating setters that recompute units in place.** Rejected: every setter must remember to re-derive the unit list, so the object can transiently be inconsistent, and a persisted plan could silently drift from its in-memory copy. Immutability makes stale units impossible and gives versioning a stable value.

## Consequences

- The keystone that the execution layer (#335), hyperparameter classification (#336), serialization/versioning (#337), and store identity (#338) all consume is now specified; those tickets can proceed against a fixed object shape.
- The Recipe choice (estimator family) moves out of `RegressionEstimatorFactory`'s fit-time call into the decide-time assembler; `RegressionEstimatorFactory` is reused by the assembler rather than by the fitters directly.
- `ColumnImputationRecord` readers (`transform`, evaluation, `to_dict`) reach the plan fields through `record.decision.*`. Acceptable churn: pre-release, no backward-compat constraint.
