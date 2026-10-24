# split

`split` splits the frame behind a profile into frame Handles. Every scheme returns one shape: an optional `test` frame and a list of `folds`, each with a `train` and an optional `validation` frame. A train/test split is one fold with no validation frame.

## Picking a scheme

`--scheme` takes JSON with a `kind`. `dataforge schema split scheme` lists each kind's fields.

| The data | Scheme |
|---|---|
| Rows are independent, one train/test split | `profile_stratified` (balances every signal the profile flags), or `random` |
| Rows are independent, cross-validation | `profile_stratified_kfold`, `kfold`, `repeated_kfold`, or `holdout_cv` for a test frame plus folds |
| Rows are ordered in time | `time`, with `time_column`; the latest rows are `test` |
| Rows belong to groups, such as many rows per customer | `group` or `group_kfold`, with `group_column`; each group stays whole on one side |

## After the split

- Each new frame keeps the profile's declarations. Profiling a fold later may add declarations but never contradict them.
- Read `fallbacks`. Each entry names a step a threshold forced, such as rows split at random, and the Dial Pointers to turn.
- Train on a fold's `train` frame and hold `test` back until the end. To check two frames come from the same distribution, run `c2st` on them.
- Seeds default to a fixed value, so the same call gives the same frames. Pass `--seed` for a different draw.
