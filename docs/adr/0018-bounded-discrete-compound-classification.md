# ADR 0018: BoundedDiscrete replaces Discrete — compound four-signal classification

`NumericKind.Discrete` is renamed `NumericKind.BoundedDiscrete` and its classification rule is replaced with a compound four-signal test. A column must pass all four signals to be classified `BoundedDiscrete`; any failure routes it to `Continuous`.

## The classification rule

All four conditions must hold:

1. **Tight sequence** — `range_span == n_unique` (no gaps between min and max; every integer slot is occupied)
2. **Small range** — `max - min ≤ 20`
3. **Low cardinality** — `n_unique / n_rows < 0.05` OR `n_unique ≤ 10` (the floor protects small datasets where the ratio inflates)
4. **Standard origin** — `min == 0 or min == 1`

For float columns, a pre-check is required before applying the four signals: all non-null values must be whole numbers (`value % 1 == 0`). Float columns with any fractional values are always `Continuous` — the tight-sequence check assumes integer steps and is undefined for decimal spacing.

`n_rows` is passed into `_classify_numeric_kind` (Phase 1) to support signal 3. `EncodedCategory` columns remain exempt from `_classify_numeric_kind` — they are `SemanticType.Categorical` and belong to a future `CategoricalImputer` scope.

## The routing implication

`NumericKind.BoundedDiscrete` → Mode imputation, unconditionally, at Priority 5 in `_fit_one`. This is the same priority position as the old `Discrete → Mode` routing. Mode fires regardless of missingness severity or mechanism because `BoundedDiscrete` columns have a finite closed domain — model-based strategies (KNN, Regression, MICE) produce continuous predictions that are not valid members of that domain.

Integer-typed or integer-valued float columns that fail any of the four signals are classified `Continuous` and fall to the MCAR severity chain (Priority 7), which may route them to Mean, Median, KNN, Regression, or MICE based on severity and distribution shape.

## The rejected alternative

The rejected design kept `Discrete` as-is (integer dtype OR < 20 unique values) and moved the compound check into `_fit_one` in Phase 2, where `n_rows` is already available without parameter threading. The argument was that Phase 2 already has all the signals it needs to make a better routing decision without touching Phase 1.

## Why Phase 1 is the correct location

Type classification belongs in Phase 1. Phase 2 should consume a decision, not remake one. Moving classification logic into the routing function conflates two responsibilities — "what kind of column is this" and "what strategy should I apply" — that the architecture deliberately separates. The `n_rows` thread-through to `_classify_numeric_kind` is a small, clean change.

## Why conservative (all four must pass)

The alternative was permissive: any single signal passing is sufficient for `BoundedDiscrete`. The conservative rule is correct because:

- A false `BoundedDiscrete` classification (continuous column gets Mode) is actively wrong — the mode of `[18, 22, 35, 42, 55]` is not a principled age fill.
- A false `Continuous` classification (bounded discrete column gets Mean/Median) is suboptimal but recoverable — the mean of `{1,2,3,4,5}` is 3.0, close to the true central tendency.

When in doubt, default to Continuous.

## Scope boundary

Bimodality detection for `BoundedDiscrete` columns — where Mode is a poor fill because the distribution has two peaks — is deferred. It requires new Phase 1 computation (Hartigan's Dip Test or GMM) and a new `Stochastic` imputation strategy (random sampling from the observed distribution). The compound classification fix (Gap 1 and Gap 2 from Scope 7) ships without bimodality handling. Gap 3 is a future scope.

## Amendment (Scope 16): Cluster-conditional mode for bimodal BoundedDiscrete columns

When `NumericFlag.Bimodal` is set on a `BoundedDiscrete` column, Mode is replaced by Cluster-Conditional Imputation following the Bimodal Imputation Framework (ADR-0032). The fill is always the **mode within the assigned cluster** — a valid member of the finite domain. Domain-constrained GMM Sampling (samples drawn from the 2-component GMM and snapped to the nearest valid discrete value) is the no-features fallback. The `BoundedDiscrete → Mode unconditionally` rule holds for all non-bimodal `BoundedDiscrete` columns without exception.
