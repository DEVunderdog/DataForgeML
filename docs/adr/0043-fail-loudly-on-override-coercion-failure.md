# Fail loudly when a column override yields zero usable values

**Status:** accepted

## Context

`PipelineConfig.column_overrides` lets a caller force a column's `SemanticType`, bypassing the type detector's verdict. Overriding does route the column to a real per-type profiler, but nothing checks that the override actually made sense for the underlying data. Three profilers can silently produce a non-signal for a mismatched override instead of erroring:

- `DatetimeProfiler._coerce_to_datetime` returns `None` for any dtype it can't parse as a date (e.g. an `Int64` "Year" column overridden to `Datetime`). The column is then dropped from `analysed_columns` entirely — `semantic_type=datetime` but `stats=None`, with no warning.
- `BooleanProfiler._to_bool_series` returns an empty series when none of the column's string values match the known true/false vocabulary (e.g. a free-text column overridden to `Boolean`). `non_null_count` ends up `0` and an empty `BooleanStats()` is returned silently.
- `NumericProfiler` casts to `Float64` with `strict=False`; if every non-null value fails to cast, the column still reports a full `NumericStats` object, just filled with nulls/NaNs.

`CategoricalProfiler` and `TextProfiler` have no equivalent failure mode — any dtype can produce meaningful value-counts or be treated as text, so they're out of scope for this decision.

## Decision

When a column carries `TypeFlag.UserOverride` and the target profiler's coercion step yields **zero** usable (non-null, successfully-coerced) values from a column that had at least one non-null value to begin with, the profiler raises `OverrideCoercionError` instead of returning empty/degenerate stats or dropping the column.

- Applies to `DatetimeProfiler`, `BooleanProfiler`, `NumericProfiler`. `CategoricalProfiler`/`TextProfiler` are exempt — no coercion step in either can produce zero usable values.
- Only triggers on **total** failure. Partial failures (e.g. a numeric override where 95% of values cast but 5% are typos) remain silent nulls — that's data-quality noise, not an override mismatch, and doesn't warrant a hard stop.
- Only triggers for `UserOverride` columns. Auto-detected types are self-consistent with the data by construction (the detector inspected the values before assigning the type), so this failure mode cannot occur without an override.
- Raised at profiling time (inside each profiler's `_run`), not when `set_column_type`/`set_columns_type` is called — the DataFrame isn't available yet at config time to check coercibility.
- No backward-compatibility shim: this is a breaking behavior change for any caller currently relying on the silent-empty/silent-drop outcome, accepted deliberately since the library has no external users yet.

## Consequences

Callers who override a column's type incompatibly with its actual dtype now get an explicit, actionable error at profile time instead of a quietly wrong or missing profile. `OverrideCoercionError` follows the existing exception-per-failure-mode convention (`UnfittedColumnError`, `UnseenColumnError`, `FittedColumnAbsentError`, `UnsupportedFormatError`).
