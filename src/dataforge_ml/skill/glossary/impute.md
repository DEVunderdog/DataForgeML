# impute

The terms the impute commands use. Look one up with `dataforge glossary <term>`.

## Routing Reason

Aliases:
Related: Routing Evidence, Signal Tier, Soft Exclusion
Commands: route, explain-routing

The one rule that fixed a column's strategy, as a fixed code to branch on, such as `signal_tier_pick`, `bottom_tier_fill`, `authored` or `soft_excluded`. It names the last rule applied, not what started the decision. Every routed column has one, `passthrough` columns included.

## Routing Evidence

Aliases:
Related: Routing Reason, Signal Score, Signal Tier, Feasibility Floor
Commands: route, explain-routing

The facts behind a column's Routing Reason, in a fixed order: why it reached the Signal Score check, the Signal Score and its parts, the Signal Tier, and each Feasibility check with its failed terms and the dial that would change them. The routing Handle stores it. A column you set in `strategies` has none, except one `feasibility` check for `knn`, `mice`, `cluster_conditional` or `gmm_sampling`.

## Signal Score

Aliases: Signal
Related: Signal Tier, Feasibility Floor
Commands: route, explain-routing

How strongly a column's other columns predict it, from 0 to 1: is a model worth it over a plain fill? It is null when no other column correlates with it at all. Routing reads it after the Feasibility Floor, and its Signal Tier picks the strategy. The `signal_score_*` dials of `route` weight its parts and set its tiers.

## Signal Tier

Aliases:
Related: Signal Score, Feasibility Floor
Commands: route, explain-routing

The band a column's Signal Score falls in, which picks its strategy. Bottom: a plain fill, such as the mean, median or mode, or GMM sampling for a bimodal column. Middle: the simplest model that passes the Feasibility Floor and Resource Ceiling, `knn` before `mice`. Top: the strongest such model. With no such model, the column takes the plain fill. `route`'s dials `signal_score_middle_tier_min` and `signal_score_top_tier_min` set the bands.

## Feasibility Floor

Aliases: Feasibility, Feasibility check
Related: Resource Ceiling, Signal Tier
Commands: route, explain-routing

The check of whether the data can train a model for one column at all, before its Signal Tier is read. A column that fails it is left off that model; a plain fill always stays open. `mice` needs `mice_min_rows_per_predictor` usable rows per predictor column. `knn` has no floor, only its Resource Ceiling. Forcing a model past it still trains, with a `forced_past_feasibility` warning.

## Resource Ceiling

Aliases:
Related: Feasibility Floor
Commands: route, explain-routing

A limit on what a model costs the machine, not on whether it can be trained. It reads the raw row count, because memory grows with the whole table. `knn` has one, the `knn_max_rows` dial of `route`; `mice` has none. A model over it is left off the column's choices, like one that fails the Feasibility Floor.

## Estimator Ladder

Aliases:
Related: Signal Tier
Commands: route

The order in which routing picks the estimator the `mice` block trains with: `bayesian_ridge`, then `random_forest_regressor`. When every `mice` column looks linear or unpredictable it takes `bayesian_ridge`; when any looks nonlinear, `random_forest_regressor`. `gradient_boosting_regressor` is never picked; set it with `route --model-choice`. The estimator never changes a column's strategy.

## Drop Candidate

Aliases:
Related: Routing Reason
Commands: profile, route, impute

A column missing more of its values than the `col_drop_threshold` dial of `profile` allows. Routing marks it `dropped` with reason `drop_candidate`, but `impute` keeps it in the filled frame with its missing values and lists it under `drop_candidates` with that dial. Dropping it is your call: pass it to `route` as `--exclude-columns`, or set it to `dropped` in `strategies`.

## Soft Exclusion

Aliases:
Related: Routing Reason
Commands: route, impute

A column one phase leaves as it is while it stays in the data. Pass it to `route` as `--skip-columns`: it routes as `passthrough` with reason `soft_excluded`, keeps its missing values, and is never used to predict another column. `impute` also makes each indicator column it adds a Soft Exclusion for later phases. Unlike `--exclude-columns`, the column is not removed.
