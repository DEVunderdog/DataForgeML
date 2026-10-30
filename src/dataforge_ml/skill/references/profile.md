# profile

`profile` computes every statistic of a frame once and stores it as a profile Handle. Its result is the Dataset Digest: the dataset at low resolution, one row per column, flagged columns first. Every later step reads the profile, never the raw rows.

## Declarations

Declarations are facts you tell `profile` instead of letting it detect them: `--targets`, `--semantic-types`, `--numeric-kinds`, `--sentinels`, `--datetime-formats`, `--exclude-columns` and `--skip-columns`. They are recorded on the profile and passed down to every frame made from it, so declare them before `split`.

To change a declaration or a dial, run `profile` again. That makes a new profile Handle and leaves the old one as it was.

## Reading past the Digest

Each detail command reads the stored profile, so it is cheap and never recomputes:

- `describe-columns`: every statistic of a column, its flags and how its type was decided. `--flagged-only` narrows to the columns worth a look.
- `value-summary`: a column's top values, rare labels, bounds and percentiles. Use it to spot sentinels such as `-999` or `"?"`, then run `profile` again with them in `--sentinels`.
- `missingness`: missing values per row, and the column pairs that go missing together. Columns that go missing together suggest the values are missing for a reason.
- `top-associations`: the most associated column pairs. Set `--column` to a target to rank features against it.
- `describe-target`: a declared target's problem type, class balance or distribution.

Page any long list with `--offset` and `--limit`. The same arguments always return the same page.

## A loop that works

1. `profile` the frame with no declarations.
2. Read the Digest's flagged columns, then `value-summary` on any column whose flag hints at a sentinel or a wrong type.
3. `profile` again with the declarations you found, plus `--targets`.
4. Move on to `split` with that last profile Handle.
