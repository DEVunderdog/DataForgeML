# The authoring door writes routing

ADR-0083 opened a manual end to the automated↔manual spectrum: `author()` built a full fit-ready `ImputationDecision` from column names alone. ADR-0088 and ADR-0089 then split that plan into **routing** (`ImputationRouting`) and a **recipe** (`ImputationRecipe`, resolved off a profile). The door had to follow. Most of what it wrote — dials, override deltas, profile estimates, sentinels — no longer lives on the object a person chooses.

We narrow the door to **manual routing**. A person writes the choice of approach. Everything computed from that choice comes from the same layers a routed plan goes through.

```python
def author(
    columns_map: Mapping[str, ImputationStrategy | AuthoredColumn],
    *,
    profile: StructuralProfileResult | None = None,
    base: ImputationRouting | None = None,
    default: ImputationStrategy = ImputationStrategy.Passthrough,
) -> ImputationRouting
```

## Status

accepted — resolves [What author becomes](https://github.com/DEVunderdog/DataForgeML/issues/533) on map #528. Partly supersedes ADR-0083 (return type, `columns=`, the sentinel keywords, `estimators=`, the three estimate fields on `AuthoredColumn`, and the carry-over rules of `base=`). Amends ADR-0088: the custom estimator object joins `mice_model_choice` on routing, and `with_model_choice` accepts it. ADR-0083's Level 2 scope, column-keyed map, strategy vocabulary and "dials do not enter here" stand. Amended by ADR-0095: config can now force `ClusterConditional` and `GMMSampling`, so those two no longer need `author(base=)`; `base=` stays for `Constant`, `MNAR`, `Dropped` and `Passthrough`. A forced bimodal column with a missing estimate follows this ADR's `fit_unit` refusal.

## Decisions

- **`author` returns `ImputationRouting`.** A hand-written routing reaches a recipe the one way any routing does: `resolve_recipe(routing, profile, config)`. There is no data-free recipe. Given up: a fit-ready plan from column names alone, in CI, before any data exists. Gained: one recipe layer, and hand-written plans get the same profile-computed dials as routed ones instead of the neutral table.
- **The sentinel keywords are deleted.** The recipe copies sentinels off the profile. Given up: declaring a sentinel at the door; it is declared in profile config instead.
- **`profile=` replaces `columns=`.** The door reads the column universe and each column's semantic type off the profile — names and types only, never statistics. Naming a non-numeric column with anything other than `Passthrough` or `Dropped` raises. This fixes a defect: the old door stamped `Numeric` on every column, so a string column defaulted to `Passthrough` joined the MICE block's predictor set. Given up: the door is no longer data-free in name, since `author` and `route` read the same profile.
- **Exactly one of `profile=` and `base=`.** Both carry the universe and the types; two sources could disagree. Given up: re-authoring with a newer profile's types — author from the profile instead.
- **No soft exclusion at the door.** Hard-excluded columns never reach a profile. Imputation soft exclusion is config, and the door reads no config; a column the map does not name takes `default`. Given up: an exclusion in config does not reach a hand-written routing, so under `default=Median` a soft-excluded column is imputed unless named.
- **`AuthoredColumn` keeps `strategy`, `constant_fill` and `grouping_variable`.** The three estimates (`center1`/`center2`, `feature_cols`, `domain_snap_bounds`) are set with `recipe.with_estimates`. The door still refuses a bare `Constant`, `Indicator`, and MICE on one column. Given up: a user who wrote centres at the door now makes two calls on two objects. The twin-type drift cost remains, now against `ColumnRouting`'s two declarations.
- **A missing estimate is caught by `fit_unit`, not the door or the recipe.** `resolve_recipe` gives every column whose strategy uses estimates a `ColumnEstimates` entry, with `None` fields when the profile has nothing, so `with_estimates` always has a target. `fit_unit` raises `UnitNotTrainableError` for a bimodal column with no centres, and for a `ClusterConditional` column with no grouping variable and no `feature_cols` (previously it fitted and filled nothing). The message names the `with_estimates` fix. Resolving cannot raise here, or the user could never reach `with_estimates`. Given up: in a loop the failure arrives when that unit is reached, possibly after a long MICE fit.
- **The custom estimator object lives on routing, beside `mice_model_choice`.** One field, because the choice is block-level and there is one MICE block. `fit_unit` reads it through `recipe.routing`; `core_budget(routing, …)` still sees `Custom` there. Label and object cannot disagree, except after a reload, where `Custom` keeps an empty slot and the fit raises as before. Held by identity, never cloned, never persisted, kept in `==` (ADR-0083). Given up: routing is no longer pure data.
- **One door for the estimator: `with_model_choice(choice: ModelChoice | estimator)`.** An instance sets the label to `Custom`; the bare `ModelChoice.Custom` label raises (no object); a routing with no MICE block raises. `author` loses `estimators=`. Gained: one act, one spelling, and a routed user can now supply their own estimator, which was impossible. Given up: a hand-written plan with a custom estimator is two calls, and `Custom` becomes a label you read, never pass.
- **`base=` survives, much smaller.** Unnamed columns keep their `ColumnRouting` verbatim, signals included. Named columns are rebuilt from the map with empty signals and keep only their semantic type. If MICE columns remain, `mice_model_choice` and the estimator carry; if the edit creates the block, it is stamped `BayesianRidge`; if the edit dissolves it, both go. Routing holds no dials, deltas, estimates, sentinels or snapshot, so ADR-0083's gap-fill and orphan-removal rules have nothing to act on. Given up: re-authoring means resolving again, which drops the override delta (ADR-0089); and a carried estimator was picked for the old membership.

## Considered Options

- **A data-free recipe (`resolve_recipe(routing, profile=None)`).** Rejected: two ways to build a recipe, neutral dials on one of them, and no profile for the estimates `with_estimates` exists to fill.
- **The estimator on the recipe.** Rejected: the label (routing) and the object (recipe) could disagree, and `resolve_recipe` would need an `estimator=` argument to check against.
- **Keep `author(estimators=)` beside `with_model_choice`.** Rejected: two spellings of one act, #451's own rule.
- **Keep `columns=` and check semantic types in `resolve_recipe`.** Rejected: under `default=Passthrough` every mixed-type frame would raise, and exempting Passthrough keeps the predictor defect.
- **Replace `AuthoredColumn` with door-level maps (`constant_fills=`).** Rejected: a `Constant` and its fill belong together, and splitting them loses the author-time check.
- **An explicit `recipe.missing_estimates()` check.** Rejected: a second check the user can skip, where the fit-time raise cannot be skipped.
