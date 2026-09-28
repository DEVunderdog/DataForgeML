# MNAR exposes its fill rather than applying it

A user declares a column MNAR because its gaps carry meaning: a missing `income` means "declined to answer", not "not recorded". The library kept that meaning in a `{col}_missing` indicator, then filled the gap with a central tendency anyway. The user never chose that fill, and it hides the gap inside the observed distribution. Now the library still computes the fill but does not apply it. The column leaves `FittedImputer.transform` with its nulls, next to its indicator, and applying the fill is the user's choice.

## Status

accepted — resolves [MNAR should expose its fill value, not apply it](https://github.com/DEVunderdog/DataForgeML/issues/459). Amends ADR-0087 (the C2ST outcome vocabulary gains a seventh member, `Unfilled`). Its companion [#460](https://github.com/DEVunderdog/DataForgeML/issues/460) was settled separately: `add_indicator_columns` was deleted (ADR-0088), so MNAR is still the only way to get an indicator. This is a breaking change on the automatic path, taken with no compatibility shim.

## Decisions

- **`FittedImputer.transform` skips MNAR in the scalar-fill loop.** The indicator is built before that loop, so it is unchanged. An MNAR column comes out null wherever it was missing. Gained: the declared meaning survives, and the user decides what happens to the gap. Given up: "Phase 2 output is null-free" no longer holds. Any consumer of an imputed frame must now expect nulls in declared MNAR columns.
- **The fill is still computed, and it is exposed twice.** `fit_scalar_unit` is unchanged. The value lands on `ColumnImputationRecord.fill_value` through `compose`, as before. The MNAR unit is still a `FittedScalar`, and its own `transform` still fills, so calling it is how a user opts in. Nothing new is exported. Gained: no new surface, and the `central_tendency` dial, persistence and the recipe are all untouched. Given up: one fit pass per MNAR column for a value that is not applied by default. It is cheap.
- **Discoverability is a `FitSignals` note, not a warning.** The MNAR unit's signals now say the fill was computed and not applied, and name both ways to apply it. Gained: the fact is visible wherever signals are read, with no noise. Given up: a user who never reads signals learns it only from the nulls. An `ImputationFitWarning` would be louder, but it would fire on every fit for a choice the user made on purpose.
- **C2ST refuses a column that is still null where it was missing: `C2STOutcome.Unfilled`.** The adapter sees only the profile and the config, not the routing, so it cannot tell a column is MNAR. Without this outcome, the filled pile would hold nulls and the observed pile would not. HGB reads NaN natively, so it would separate the piles perfectly and report a strong failure for a column left unfilled on purpose. The check is on the data: if any imputed frame still has a null among the mask rows, the column is refused. It runs after the pre-pile refusals (`TypeNotTestable`, `NoFilledCells`, `NoObservedCells`) and before the sample floor. Like every refusal, it has no score and no verdict. Gained: no false failure, and a table from any imputer that passed a column through gets the same honest answer. The adapter's signature is unchanged. Given up: the vocabulary grows to seven members, which ADR-0087 names as a breaking change. A partly filled column is refused whole rather than tested on the cells that were filled.

## Considered Options

- **Keep filling and document the fill as optional.** No break, but the default still destroys the signal the declaration exists to keep.
- **Stop computing the fill.** Removes the fit pass, but `fill_value` goes empty and the user loses a value they may want.
- **Pass the routing into `imputation_score_c2st` and skip MNAR by strategy.** More precise, but it widens a public signature settled in #537, and ADR-0087's amendment kept routing out on purpose so a table from any imputer can be scored.
- **Test a partly filled column on its filled cells only.** It would keep a number, but it would mix up "the imputer filled badly" with "the imputer did not fill", and a partial fill has no current producer.

## Consequences

- CONTEXT.md no longer says MNAR output is null-free. It says the fill is exposed, not applied.
- Any downstream phase that reads an imputed frame must handle nulls in declared MNAR columns. The indicator marks exactly those rows.
- The C2ST integration fixture `imputed_frame` used `map_elements`, which skips nulls, so its "median fill" never filled a null. The earlier C2ST tests were scoring nulls against observed values. The new check exposed this, and the fixture now fills with a vectorised `when/then`.
- The version bump is the maintainer's to make.
