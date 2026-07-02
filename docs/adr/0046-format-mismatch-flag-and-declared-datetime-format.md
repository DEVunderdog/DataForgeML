# Format Mismatch flag and declared per-column datetime format

**Status:** accepted

## Context

The three coercing profilers (`NumericProfiler`, `BooleanProfiler`, `DatetimeProfiler`) all coerce input with a non-strict cast and then `drop_nulls()`. This silently collapses two different situations into one: a value that is genuinely *absent* (an Effective Null) and a value that is *present but uncoercible* (`"banana"` in a numeric column, `"maybe"` in a boolean, `"12/2015"` in a datetime column). ADR-0043's Context already noted this blind spot; auto-detection makes it worse, because `datetime_coerce_threshold = 0.80` lets an auto-detected `Datetime` column carry up to 20% uncoercible values that vanish into the null count.

Separately, Polars format inference cannot guess a format for some legitimate patterns at all — a bare-year column (`"2015"`, `"2014"`) raises `ComputeError` from `str.to_datetime`, which `_coerce_to_datetime` turns into total failure and, for an overridden column, `OverrideCoercionError` (ADR-0043). The caller has no way to say "read this column as `%Y`."

## Decision

Two related additions, deliberately kept minimal.

**1. A `FormatMismatch` flag — detection only, no count.** A new flag (`NumericFlag.FormatMismatch`, `DatetimeFlag.FormatMismatch`, and a new `BooleanFlag` enum with `flags`/`has_flag()` machinery added to `BooleanStats` to match Numeric/Datetime) fires on any of the three profilers when at least one value that is neither a genuine null nor an Effective Null fails to coerce. It is a boolean signal with **no count and no per-value detail** — quantifying "how much is dirty" was rejected as complexity that the caller did not want. Uncoercible values continue to be treated as null for all statistics; the null-based signals (including `MnarSuspected`) are computed unchanged. A column may therefore carry both a missingness signal and `FormatMismatch`, read as "some of what looks missing here is actually dirty data." The library never guesses, counts, locates, or repairs mismatches.

To honour the Effective Null model without breaking sub-processor purity (ADR-0002/0030 — profilers hold no config), the **orchestrator** resolves *string* Effective Nulls (empty/whitespace strings and declared/default `string_sentinels`) to genuine nulls on the frame handed to the typed profiling pass. This changes no existing statistic — those strings already became null via coercion — it only makes the new flag ignore recognized missing-markers. Numeric sentinels are left alone (deferred to Scope 5, and a numeric sentinel coerces to a valid number so it never trips the flag).

**2. A declared per-column datetime format** — `ProfileConfig.datetime_formats: dict[str, str]`, set via `set_datetime_format(column, format)`, mirroring `datetime_epoch_units`/`set_datetime_epoch_unit` (ADR-0045) and the setter-lockdown mechanics of ADR-0044. `_coerce_to_datetime` uses the declared format with `strict=False`, applicable to any column profiled as `Datetime` (override or auto-detected). Rows that don't match become null → `FormatMismatch` if any non-missing value fails; a format that matches *zero* rows is still total failure → `OverrideCoercionError` for overrides, unchanged from ADR-0043. Format strings are not validated at set-time — validation surfaces at profiling time, consistent with `set_column_type` and `set_datetime_epoch_unit`. The `DatetimeProfiler`'s `OverrideCoercionError` message points the caller at `set_datetime_format`; the Numeric and Boolean raise sites are unchanged.

## Considered Options

- **A per-value count (`format_mismatch_count`) and clean MNAR separation** — recomputing `null_ratio` from pre-coercion nulls only, so "missing" and "malformed" never mix. Rejected: it forces the pre-coercion series through `_profile_column` and asks the caller to reason about two overlapping ratios, for precision the caller explicitly did not want. The standalone flag beside the unchanged null signal conveys the same actionable fact ("go clean this column") without the machinery.
- **A format-declaration escape hatch for all three profilers** (`set_numeric_format`, etc.). Rejected: only datetime coercion is format-shaped. Numeric corruption is unfixable by any format string, and boolean coercion is membership against a fixed vocabulary. The detection *signal* is universal; the *escape hatch* is datetime-only by nature.
- **A declared format as a strict contract** (any non-matching row raises). Rejected: contradicts the flag-not-raise philosophy — partial dirt is non-fatal everywhere else in this design.

## Consequences

The library ships a documented asymmetry: `set_datetime_format` exists but `set_numeric_format` does not, because numeric/boolean dirt cannot be rescued by a declaration — those profilers get detection only. `FormatMismatch` is intentionally imprecise (no count, null-ratio untouched), trading numeric exactness for a simple, cheap, at-scale signal that is uniform across all three coercing profilers.
