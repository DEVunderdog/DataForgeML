# Routing rationale is a reason code plus typed evidence

Today the only answer to "why did the library route this column this way?" is `ColumnRouting.signals`, a tuple of prose strings such as `"signal_score: 0.514 (base=0.422, latent=0.851, …) | signal_tier: middle -> knn"`. The router computes the Signal Score, the Signal Tier and every Feasibility term as numbers, then formats them into text and drops the numbers (G5). A non-numeric column is routed `Passthrough` with no signals at all, so `Alley` at 93.8% missing keeps its nulls without a word (G2). An Agent can read these strings but must not parse them, which leaves it nothing to branch on. This ADR decides how the library tells an Agent why it decided something.

## Status

accepted. Resolves [How does the library tell an Agent why it decided something?](https://github.com/DEVunderdog/DataForgeML/issues/577) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0088 (what a routing holds), ADR-0074 and ADR-0102 (where the forced-past-feasibility caveat is reported). Planning only: no code changes yet.

## Decisions

- **The prose is deleted.** `ColumnRouting.signals` goes, and no rationale string replaces it. Typed evidence is the only rationale. A Harness that wants a sentence renders one from it. Gained: one source, so the text and the numbers cannot drift, and it follows ADR-0102's rule that the library emits JSON only. Given up: a ready-made readable line, which the research called compact and readable.
- **Each column carries one `reason` and an ordered `evidence` list.** The `reason` is the **terminal rule**: the rule that fixed the strategy, not the cause that started the walk. The `evidence` list is the walk itself, as tagged facts:
  - `entry {cause}`: why the column reached the escalation seam (MAR suspect, MCAR high severity, leptokurtic escalation, moderate-severity shape escalation, bimodal with many correlated features, bounded discrete). A closed enum of its own.
  - `signal_score {score, base, latent, breadth, predictors, degraded}`.
  - `tier {tier}`.
  - `feasibility {strategy, passed, failed_terms: [{term, value, limit, dial}]}`.

  Gained: an Agent branches on one field and still gets the numbers behind it, including how close a column came to the next tier. `dial` names the config field that would change the outcome. A column that tier-picked KNN reads the same whatever brought it to the seam, which is what an Agent branches on. Given up: a few small records per model-routed column, and the MAR-versus-MCAR story sits one level down in `entry`.
- **`RoutingReason` is a closed enum in the output schema.** It is sent by value and versioned by library version through ADR-0101's snapshot test. It is separate from the Error Code registry, because a reason describes a successful decision, not a failure or a caveat. The values:
  - declared or structural: `drop_candidate`, `constant_declared`, `strategy_forced`, `mnar_declared`, `no_missing_values`, `soft_excluded`, `not_imputed_type`, `authored`;
  - shape rules: `mcar_minor_unskewed` (Mean), `mcar_skewed` (Median), `bounded_discrete_scalar` (Mode), `bimodal_grouping_declared`, `bimodal_few_correlated` (Cluster-Conditional), `bimodal_uncorrelated` (GMM Sampling, or Mode for BoundedDiscrete);
  - seam outcomes: `signal_tier_pick`, `bottom_tier_fill`, `no_feasible_model` (the tier wanted a model and none was feasible), `no_predictor_correlation` (no score could be computed).

  Gained: a stable value to branch on. Given up: a new router branch is a visible schema change.
- **Every `Passthrough` has a reason (G2).** It is one of `no_missing_values`, `soft_excluded`, `not_imputed_type` or `authored`. Gained: no column is left without an answer. Given up: nothing.
- **Non-numeric missingness earns one Result Warning on `route`.** When non-numeric columns with missing values are routed `Passthrough`, the `route` result carries one `non_numeric_missing_kept` warning that lists them, with a hint pointing at `author` as `Dropped` or a Soft Exclusion. Gained: an Agent that reads only `warnings` still learns the nulls will survive, before C2ST reports those columns `Unfilled`. Given up: almost every real dataset gets this warning, so it risks becoming noise. One warning per call, not one per column, keeps it to a single item.
- **`route` returns the reason. The evidence is read through a detail Operation.** The `route` result holds strategy and `reason` per column. The full `evidence` list is stored on the routing Handle and read by `explain_routing(columns=null, offset, limit)`, capped at about 15 columns per call, which `omitted` points at. Gained: `route` stays inside the Result Budget, and an Agent pays for the numbers only on the columns it asks about. It is the same digest-then-detail split as ADR-0104. Given up: one extra call whenever the Agent wants the numbers.
- **The MICE estimator's reason is inline on `route`.** The estimator is chosen once for the whole block, by the winning Nonlinearity Tag among its columns. `route` carries `mice_estimator` and `estimator_reason {tag, column}` beside the per-column list. Gained: the question an Agent asks most ("why random forest?") is answered without a call. Given up: one evidence item sits on the concise result.
- **Authored columns carry `authored` and no evidence.** A column a person named in `author()` has nothing the library decided, so its `evidence` is empty. A column copied from `base=` keeps its reason and evidence. Gained: the routing says honestly which decisions are the library's. Given up: nothing.
- **The evidence is persisted.** It is part of the routing, so it is saved on the routing Handle and in the recipe's routing copy (ADR-0096), and strict load checks its schema. FitSignals stays ephemeral, because it describes an event, while evidence describes the decision. Gained: `explain_routing` works on a routing made in another session or by another Agent. Given up: saved routings are larger.
- **A caveat is reported once, where it was decided.**
  - Forcing a column past the Feasibility Floor produces a `forced_past_feasibility` Result Warning on `route` and a `feasibility {passed: false}` fact. Fit no longer repeats it. Fit warnings keep only facts about the run, such as `mice_not_converged`.
  - A profiling fallback (for example a failed mutual-information computation or a missing R² pair) is a Result Warning with its own code on the `profile` result. Downstream, the routing evidence shows only its effect, as `degraded` on `signal_score`. Each of today's 13 `warnings.warn` sites in profiling gets a code in the registry.

  Gained: each fact is said once, in the call that caused it. Given up: an Agent that fits from a stored routing in a later session gets no fit-time warning for a forced column, and one that missed the `profile` warning sees `degraded` without its cause. Both must read the evidence.
- **Every typed field is branchable. Only `message` and `hint` are read-only text.** Codes, enums, numbers and evidence facts are part of the contract and follow the schema's versioning. The wording of `message` and `hint` on errors and warnings may change in any release. Gained: one clear line for Agents. Given up: changing what the Signal Score is made of becomes a schema-visible change. A formula change already changes routing, so it should be visible.

## Considered Options

- **Keep the prose beside the typed record.** Rejected: two copies of every fact, and the strings are exactly what Agents must not parse.
- **A list of reason codes with no numbers, or a single code with no evidence.** Rejected: neither can say why KNN and not MICE, or how close a column came to the next tier.
- **The entry cause as the reason, the seam outcome as evidence.** Rejected: "why this strategy?" would then need the evidence list.
- **Reasons in the Error Code registry under a third marker.** Rejected: it mixes successful decisions with failures and caveats.
- **An open, documented string instead of an enum.** Rejected: an Agent cannot trust it for branching.
- **Evidence inline on `route`, capped.** Rejected: every call pays for it, and a cap cuts evidence unevenly across columns.
- **Ephemeral evidence, like FitSignals.** Rejected: an older routing Handle would have nothing to explain.
- **Report forcing at both route and fit.** Rejected: the same fact twice.
- **Mark the evidence numbers "informational, may change meaning".** Rejected: a field an Agent cannot rely on is a field it should not be shown.

## Consequences

- ADR-0088 is amended: a routing holds `reason` and `evidence` per column and `estimator_reason` for the block, and no longer holds `signals`.
- ADR-0074 and ADR-0102 are amended: the forced-past-threshold caveat leaves fit and is reported on `route`.
- `_forced_model_signal`, `_signal_score`'s note string and `_escalation_seam`'s joined notes become typed fact builders.
- The inventory ticket inherits `explain_routing`. The schema snapshot inherits `RoutingReason` and the evidence union.
- `CONTEXT.md`'s Soft Exclusion and `per_column_strategy` entries now describe a reason and a feasibility fact, not a signal.
