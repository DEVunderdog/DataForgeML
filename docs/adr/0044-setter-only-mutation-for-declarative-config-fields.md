# Setter-only mutation for declarative per-column config fields

**Status:** accepted

## Context

`PipelineConfig`, `ProfileConfig`, and `ImputationConfig` all carry per-column declarative fields — dicts or lists keyed/valued by column name that a caller populates before running the pipeline (`column_overrides`, `numeric_kind_overrides`, `numeric_sentinels`, `string_sentinels`, `mnar_columns`, `add_indicator_columns`, `per_column_strategy`, `per_column_constant_fill`, `per_column_max_iter`, `bimodal_grouping_variables`, `exclude_columns`, `phase_exclusions`). Several already have setter methods (`set_column_type`, `set_numeric_kind`, `add_exclusions`) that validate input before storing it. But the backing fields are still plain public dataclass attributes — `config.column_overrides["age"] = "not_a_real_type"` or `config.numeric.per_column_strategy["age"] = ImputationStrategy.Dropped` writes directly to the dict/list, bypassing whatever validation the setter (or, for `NumericImputationConfig`/`ImputationConfig`, `__post_init__`) would have applied. The `__post_init__` checks in particular only run once, at construction — any mutation afterward silently skips them, even though their docstrings and error messages imply the rule is always enforced.

## Decision

Every field in this category is private-backed with a read-only accessor and mutable only through a validating setter:

- Storage moves to a `_`-prefixed attribute (e.g. `_column_overrides`).
- The public name becomes a read-only property. Dict-shaped fields return `types.MappingProxyType(self._x)` (a live view — reflects mutation through the setter, but rejects writes). List-shaped fields (`exclude_columns`, `mnar_columns`, `add_indicator_columns`) return `tuple(self._x)`, a fresh order-preserving snapshot on every access.
- Every field gets exactly one setter, named with the field's singular concept (`set_column_type`, `set_numeric_sentinel`, `add_mnar_column`, `add_phase_exclusion`, etc.) — no separate plural method. The column argument accepts `str | list[str]` and loops internally when given a list. This retires the existing `set_columns_type`/`set_columns_numeric_kind` plural methods and renames `add_exclusions` to `add_exclusion` for the same reason.
- Setters validate only what they can see in their own object. Cross-object rules (`mnar_columns` vs. `numeric.per_column_strategy` — a column can't be in both) are not re-derivable at set time from the nested `NumericImputationConfig` side, since it holds no reference to its parent `ImputationConfig`. Rather than adding a parent back-reference, the full cross-field check remains a single `validate()` method (today's `__post_init__` logic, made callable) that the relevant orchestrator invokes before running — same coverage as today, just re-armed on every run instead of only at construction.
- No removal/unset methods. These fields feed a one-directional pipeline (profile → impute → split); there's no valid scenario for retracting a declaration mid-run, and starting over means constructing a fresh `PipelineConfig`, which is cheap.
- `from_dict()` reconstructs these fields by calling the same validating setter per entry, not by writing straight into the private backing field — a hand-edited config file with an invalid entry fails loudly at load time instead of silently loading bad state.
- Dict-valued fields whose value is itself a fixed-choice type get a matching `StrEnum` rather than a raw `Literal`, matching every other fixed-value-set field in the codebase (`SemanticType`, `NumericKind`, `ImputationStrategy`, ...). This is why `datetime_epoch_units` (ADR-0045) gets `EpochUnit(StrEnum)` instead of a bare string literal.

New/changed setters by field:

| Field | Owner | Setter |
|---|---|---|
| `column_overrides` | `PipelineConfig` | `set_column_type` (extended, replaces `set_columns_type`) |
| `numeric_kind_overrides` | `PipelineConfig` | `set_numeric_kind` (extended, replaces `set_columns_numeric_kind`) |
| `exclude_columns` | `PipelineConfig` | `add_exclusion` (renamed from `add_exclusions`) |
| `phase_exclusions` | `PipelineConfig` | `add_phase_exclusion` (new) |
| `numeric_sentinels` | `ProfileConfig` | `set_numeric_sentinel` (new) |
| `string_sentinels` | `ProfileConfig` | `set_string_sentinel` (new) |
| `datetime_epoch_units` | `ProfileConfig` | `set_datetime_epoch_unit` (new; see ADR-0045) |
| `per_column_strategy` | `NumericImputationConfig` | `set_per_column_strategy` (new) |
| `per_column_constant_fill` | `NumericImputationConfig` | `set_per_column_constant_fill` (new) |
| `per_column_max_iter` | `NumericImputationConfig` | `set_per_column_max_iter` (new) |
| `bimodal_grouping_variables` | `NumericImputationConfig` | `set_bimodal_grouping_variable` (new) |
| `mnar_columns` | `ImputationConfig` | `add_mnar_column` (new) |
| `add_indicator_columns` | `ImputationConfig` | `add_indicator_column` (new) |

## Consequences

This is a breaking change to the two existing plural methods and to `add_exclusions`' name — accepted deliberately, same rationale as ADR-0043 (no external users yet). Every in-scope config class picks up a `validate()` method callable ahead of orchestrator execution, replacing reliance on `__post_init__` alone. Per CLAUDE.md's documentation rule (ADR-0034), all new/changed setters and properties need numpy-style docstrings, and touching any of these three files means bringing the whole file's docstrings up to date.
