# Config forces every trainable strategy

`per_column_strategy` refused `ClusterConditional` and `GMMSampling` as "internal-only". Only `author` could force them, so ADR-0082's promise (strategy is set before routing) had a gap for config users. Both were refused because each needs data: centres, and a grouping variable or `feature_cols`. The layered design already has a place for each: the grouping variable is a config declaration on `ColumnRouting`, and centres and `feature_cols` are profile estimates on the recipe (ADR-0089). So we widen config to every trainable strategy. A column that lacks the data it needs is refused by the rule ADR-0090 set for hand-written routing.

## Status

accepted — resolves [Whether per_column_strategy accepts every trainable strategy](https://github.com/DEVunderdog/DataForgeML/issues/540) on map #528. Closes the known gap in ADR-0082's amendment. Amends ADR-0088 (forcing a bimodal strategy records its unmet conditions as routing signals), ADR-0089 (`domain_snap_bounds` is also resolved for a BoundedDiscrete column forced to `GMMSampling`) and ADR-0090 (one reason for `author(base=)` goes; the door keeps it).

## Decisions

- **`per_column_strategy` accepts `ClusterConditional` and `GMMSampling`.** They leave the output-only set. Config's vocabulary is now `Mean`, `Median`, `Mode`, `KNN`, `MICE`, `ClusterConditional`, `GMMSampling`, plus `Constant` through `per_column_constant_fill`. Given up: config can now produce a column that cannot train, which only `author` could do before.
- **A missing estimate is refused by `fit_unit`, not routing or the recipe.** Same rule as ADR-0090. `resolve_recipe` gives the forced column a `ColumnEstimates` entry with `None` fields when the profile has nothing, and `fit_unit` raises `UnitNotTrainableError` naming `with_estimates`. Routing raising would make routing read estimates it does not own. The recipe raising would leave `with_estimates` unreachable. Given up: the failure arrives at fit time, not at routing.
- **Forcing records every unmet condition routing can see, as signals.** "Not flagged Bimodal" for both strategies. For `ClusterConditional`, also "no grouping variable declared and no correlated features". Recording, never blocking, like forcing past the Feasibility Floor (ADR-0088, ADR-0091). Given up: routing reads the flag and correlations for a forced column it otherwise short-circuits, and the signal partly repeats the later fit error.
- **A grouping variable declared for a column forced to anything but `ClusterConditional` raises at config time.** Both are the user's own typed declarations, and they contradict each other. The profile never computes a grouping variable, so no profile finding can trigger this raise. Forcing a strategy over profile estimates (a bimodal column forced to MICE) never raises: the recipe resolves only the estimates the strategy uses. Checked in both setters and in `validate()`, like the `mnar_columns` clash, so the order of the setters does not matter. An unforced column with a grouping variable still routes normally.
- **A BoundedDiscrete column forced to `GMMSampling` trains, with a signal, and its samples are domain-snapped.** The router never picks this (ADR-0035), but forcing is informed consent, and the snap keeps the output in the domain. Given up: the recipe resolves snap bounds in one more case, and the GMM fitter gains a snap step.
- **`author(base=)` stays.** The door still has strategies config spells through other surfaces or not at all: `Constant` with its fill, `MNAR`, `Dropped`, `Passthrough`. Editing those on top of a routed result needs `base=`. Given up: `base=` rests on a narrower case.

## Considered Options

- **Fit the centres from the fold when the recipe has none.** Rejected: a silent fallback, and it breaks ADR-0089's "estimates resolved once off the profile".
- **Raise in routing when a forced column is not flagged Bimodal.** Rejected: routing would stand in for the estimate check, and `with_estimates` could never rescue a column the user knows is bimodal.
- **Ignore, or carry, a grouping variable on a column forced elsewhere.** Rejected: both let a contradictory declaration survive silently.
- **Refuse `GMMSampling` on a BoundedDiscrete column.** Rejected: the one forcing case that would block, against the informed-consent rule.
