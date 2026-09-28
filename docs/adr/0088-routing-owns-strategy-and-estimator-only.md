# Routing owns strategy and estimator only

`decide()` did five jobs in one call. It routed each column, picked the MICE estimator, computed the block dials at one `n_rows`, lowered profile estimates onto the plan, and materialised units. A user who wanted routing got everything else too. The dials were frozen at the one row count the plan was decided for, so a per-fold fit either used the wrong size or re-decided and broke unit-keyed overrides.

We narrow routing to the **choice of approach**. `route(profile, config) -> ImputationRouting` returns, per column, a `ColumnRouting` with the strategy, the signals, and the two config declarations that complete a strategy (`constant_fill`, `grouping_variable`), plus one block-level `mice_model_choice`. Everything computed *from* that choice (dials, profile estimates, units) belongs to later layers.

## Status

accepted — partly supersedes ADR-0060 (the plan is no longer one object), ADR-0066 (`n_rows` is no longer an argument), ADR-0068 (sentinels leave the routing output) and ADR-0073 (both hyperparameter maps leave it). Amends ADR-0071 and ADR-0074 for one case: forcing a column past a routing gate is recorded as a routing signal, not a fit-time warning. ADR-0077 (exclusion enforcement) and ADR-0082 (strategy declared before routing) stand.

Amended by ADR-0095 ([Whether per_column_strategy accepts every trainable strategy](https://github.com/DEVunderdog/DataForgeML/issues/540)): a column forced to `ClusterConditional` or `GMMSampling` records each unmet condition routing can see (not flagged Bimodal; no grouping variable and no correlated features) as a signal, the same way.

Amended by ADR-0090 ([What author becomes](https://github.com/DEVunderdog/DataForgeML/issues/533)): routing also holds the user's own estimator object beside `mice_model_choice`, and `with_model_choice` accepts a `ModelChoice` or an estimator instance. `author` returns `ImputationRouting`.

## Decisions

- **`n_rows` is read from the profile.** Under the hybrid topology the routing profile is the dev profile, so its row count is dev's. Every other routing input already comes from the profile. An explicit count from a different frame made a mixed-shape decision. Given up: a profile-first user who routes on a full profile routes at full size. The topology is documented, not enforced.
- **The estimator stays in routing and is block-level.** It is part of the approach: it changes cost, parallelism and determinism, and it is what a custom estimator replaces. One field, `mice_model_choice`, because the block trains one estimator. The old per-column copy let two MICE columns disagree through `with_model_choice(column, …)`. Given up: the estimator is fixed at dev's size, so a fold below `gradient_boost_min_rows` still trains the one dev chose. CV then scores what ships.
- **The indicator, MNAR and drop flags are derived from `strategy`.** They were never anything else, and stored copies could disagree with it. `add_indicator_columns` is deleted: it set nothing and only fed the contradictory-config check. An indicator on a non-MNAR column would return as its own feature, with a real field.
- **Config declarations stay; profile estimates leave.** `constant_fill` and `grouping_variable` are typed by the user and complete the strategy (`Constant` without a value is not a strategy). The bimodal centres, `feature_cols` and `domain_snap_bounds` are estimates validation rows influence, so the routing output carries no leak surface.
- **No dials, units, sentinels or `config_snapshot` on the output.** Routing must be obtainable without them. `config_snapshot`'s one reader was the fit-time "KNN forced past its routing threshold" warning, which rebuilt config at fit time. Routing is where the gate is checked and the override applied, so it records the condition as a signal. Given up: the signal compares dev's size, not the fold's, and a signal is quieter than a Python warning.
- **The function is `route`, not `decide`.** `decide` is left as an umbrella word; whether it names a convenience chain is decided elsewhere. Settled by [The one-call convenience chain](https://github.com/DEVunderdog/DataForgeML/issues/534): no chain exists, on either the decide side or the fit side, and `decide` is deleted with no successor.

## Considered Options

- **Slim `ImputationDecision` in place.** Rejected: the name meant "fit-ready plan", and keeping it on an object that cannot fit would change its meaning silently. Breaking loudly is wanted.
- **Move the estimator pick next to the dials.** Rejected: it would split "which approach" across two layers, and a feasibility-driven estimator pick is routing's kind of logic.
- **Keep `n_rows` explicit.** Rejected: it is the one routing input that can disagree with every other routing input.
