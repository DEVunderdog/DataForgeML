# evaluate

The terms the evaluate commands use. Look one up with `dataforge glossary <term>`.

## Classifier Two-Sample Test

Aliases: C2ST
Related: Dial Pointer
Commands: c2st, evaluate-imputation, describe-c2st-columns

A test of whether two sets of rows come from the same distribution: a classifier learns to tell them apart, and each column's `z` measures how far above chance it scores on held-out rows. Near zero, the two are indistinguishable. `evaluate-imputation` compares filled rows with observed ones; `c2st` compares two frames, such as `train` and `test`.
