# Native epoch → Datetime coercion via declared per-column unit

**Status:** accepted

## Context

`DatetimeProfiler._coerce_to_datetime` only accepts `Utf8/String` (via `str.to_datetime`) or native `Date/Datetime/Duration/Time` dtypes. A bare numeric column (epoch seconds/ms/us/ns) has no embedded unit signal the way a string format or a native dtype does — `1700000000` could be epoch seconds, or an unrelated integer quantity like Population. Overriding such a column to `Datetime` today fails coercion entirely and, after ADR-0043, raises `OverrideCoercionError`.

Polars already exposes `pl.from_epoch(col, time_unit=...)` as a trivial pre-cast a caller could run themselves before handing data to DataForgeML, which raised the question of whether the library needs to reimplement epoch handling at all.

## Decision

DataForgeML adds native support rather than deferring to caller pre-casting, consistent with the one-stop-solution vision (users shouldn't need to hand-roll Polars calls before profiling). It does this by wrapping Polars' own `pl.from_epoch`, not reimplementing epoch-to-datetime math:

- New field `ProfileConfig.datetime_epoch_units: dict[str, EpochUnit]`, declared per column via `set_datetime_epoch_unit(column: str | list[str], unit: str | EpochUnit)` (see ADR-0044 for the setter-lockdown mechanics this field follows).
- `EpochUnit(StrEnum)` mirrors `pl.from_epoch`'s `time_unit` values (`s`, `ms`, `us`, `ns`, `d`) — matches this codebase's existing convention of a `StrEnum` for every fixed-value-set field rather than a raw `Literal`.
- `DatetimeProfiler._coerce_to_datetime` is extended to also accept numeric dtypes when the column has a declared unit in `datetime_epoch_units`; it calls `pl.from_epoch(col, time_unit=unit.value)`.
- A numeric column overridden to `Datetime` with no declared unit in `datetime_epoch_units` is not a separate error path — it falls straight into `OverrideCoercionError` (ADR-0043), since coercion still yields zero usable values.
- Scoped to Phase 1 / structural profiling / type detection, not the separately-planned time-series work — time-series logic should consume already-correctly-typed `Datetime` columns and shouldn't have to re-solve type-detection ambiguity itself.

## Consequences

Column-level epoch ambiguity is resolved once, at declaration time, by the caller who actually knows the column's semantics — the same operating principle as `numeric_sentinels`/`string_sentinels`. No epoch-math is maintained in DataForgeML itself; correctness rides on Polars' own implementation.
