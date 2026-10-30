# impute

Filling missing values takes five commands, each writing one Handle:

1. `route` the profile → routing: one strategy per column, each with a Routing Reason, the rule that fixed it.
2. `resolve-recipe` the routing → recipe: the units to fit, each with the hyperparameters it trains with.
3. `fit-unit` once per unit the recipe lists → unit. Pass a fold's `train` frame as `--frame`; with none, the unit trains on the frame the profile was made from.
4. `compose-imputer` the recipe and one unit Handle per unit → imputer.
5. `impute` each frame to fill, such as `train`, `validation` and `test` → a filled frame.

## Routing

- Routing reads only the profile, never the rows. Columns you list in `--strategies` take your strategy with reason `authored`. Every other column is routed from its profile.
- `explain-routing` shows how close a column came to another strategy: its Signal Score, Signal Tier and Feasibility checks, with the dial that would change each. `route`'s own result already has every strategy and reason.
- Forcing `knn`, `mice`, `cluster_conditional` or `gmm_sampling` past the Feasibility Floor still trains, with a `forced_past_feasibility` warning.
- Non-numeric columns keep their missing values.

## Fitting

- There is no command that fits every unit. Fit each one yourself, in any order. They are independent, so you can run them side by side.
- `fit-unit` is a (background) command: on large data, run it in the background or with a longer shell timeout.
- Read `fit-unit`'s `warnings`. A model that hit its iteration cap still fits, and its warning names the dial to turn.
- To tune a unit, run `resolve-recipe` again with `--hyperparameters` or `--estimates`, then fit that recipe's units. Units from another recipe are refused by `compose-imputer`.
- To refit one unit on another frame or seed, fit it again and compose again with the new unit Handle in place of the old one.

## Filling

- The imputer learns nothing from the frame it fills, so filling `test` leaks nothing.
- The filled frame drops hard-excluded columns and the `dropped` columns you declared, and adds an indicator column per indicated column.
- A Drop Candidate stays in the filled frame with its missing values. `drop_candidates` lists each one with its dial; dropping one is your call.
- An `mnar` column keeps its nulls beside its `<column>_missing` indicator. Its fill is on record in `columns` and is not applied. `passthrough` columns keep their missing values too.
