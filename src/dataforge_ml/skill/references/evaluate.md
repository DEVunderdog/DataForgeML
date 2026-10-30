# evaluate

Both evaluate commands run a Classifier Two-Sample Test (C2ST): a classifier learns to tell two sets of rows apart. If it does no better than chance on held-out rows, the two are indistinguishable. Each column's `z` measures how far above chance it did.

| Question | Command |
|---|---|
| Does a fill look like the observed data? | `evaluate-imputation --profile <profile> --imputed <frame>` |
| Do two frames come from the same distribution, such as `train` and `test`? | `c2st --reference <frame> --candidate <frame>` |

## evaluate-imputation

- Pass the profile the imputer was routed on, even for a filled fold. The imputed frame is scored against the frame it was filled from.
- Read `census` first, then the `flagged` columns.
- A column filled with one value (`mean`, `median`, `mode` or `constant`) is expected to be flagged. Judge a model-based column against those.
- Under missing at random, the filled and observed rows are meant to differ. A flag cannot tell a bad fill from strongly MAR missingness: the `z` is true, but it may not be the fill's fault. Read `missingness` before blaming the strategy.
- The result writes a c2st Handle. `describe-c2st-columns --report <c2st>` reads each column's per-repeat `z`, accuracies and provenance from it, without recomputing.

## c2st

- Columns either frame excludes or soft-excludes are left out. Every other column must be in both frames.
- String columns cannot be tested: exclude them on `profile` before `split`.
- `c2st` writes no Handle.

Both commands are (background) commands. Seeds default to a fixed value, so the same call gives the same score.
