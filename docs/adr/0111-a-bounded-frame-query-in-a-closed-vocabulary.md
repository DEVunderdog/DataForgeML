# A bounded frame query in a closed vocabulary

ADR-0107 settled that literal values travel only in Literal Fields, and it left room for "rows or query results". This ADR decides whether that query exists. In Tool Calls the Agent has no pandas, so `value_summary` (at most 20 of each kind) is the only way a cell value reaches the model. In Code Mode, pandas reaches only the Declared Input. Folds and imputed frames live in the Workspace, and reading one means `export_frame` and a CSV re-read, which may change the dtypes. The query runs over every row, and only the bounded answer comes back.

## Status

accepted. Resolves [Should the library offer a bounded query over raw data?](https://github.com/DEVunderdog/DataForgeML/issues/586) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0110 (the inventory grows to 20 Operations) and ADR-0099 (the Harness contract). Planning only: no code changes yet.

## Decisions

- **One Operation, `query_frame`, in both deliveries.** It sits in the `workspace` phase next to `export_frame` and `list_handles`. It writes nothing, its cost grows with the data, and it is not in `load_first`. Gained: the same behaviour in both deliveries (ADR-0099). Under Tool Calls it is the only way rows reach the model, and every cell it returns goes through the Literal Policy. Derived frames can be read without an export. Given up: a 20th Operation, and a vocabulary to keep small. In Code Mode it overlaps with pandas on the Declared Input.
- **It reads any `frame_` Handle**: the loaded frame, split folds and `impute` output. It does not take a `profile_`, because the Agent already holds the frame that profile was made from.
- **It never makes a Handle.** It returns values only. A "filtered frame" Handle would be a data change config cannot express, and ADR-0103 rules that out. Given up: profiling a subset of rows. Subsets still come only from `split`.
- **A tagged-union `mode`: `rows` or `aggregate`.** The arguments for each mode live inside its variant, so a rows call cannot carry `group_by`. This follows ADR-0110's `split`. Both modes share the filter and one result shape.
- **The filter is an OR across AND-lists of predicates `{column, op, value}`.** That is two fixed levels, so the Strict Subset's no-recursion rule holds.
  - The `op` values are `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`, `not_in`, `is_null` and `not_null`.
  - `value` is a tagged union of number, string or bool, or a list of one of them.
  - There is no string matching and no regex. Regex can hang on a crafted pattern, and the sentinel and label checks need only `eq`.
  - Gained: any boolean filter can be expressed. Given up: a heavier schema than an AND-only list.
- **Aggregates.** `group_by` takes up to 3 columns. An empty list means the whole frame as one row. The measures are `count`, `null_count`, `nunique`, `sum`, `mean`, `std`, `min`, `max` and `median`. There are no arbitrary quantiles and no `mode`, because `value_summary` covers those.
- **In rows mode, a tagged `order` picks the rows.** `dataset` (the default) returns the first matches. `random` is a seeded sample with ADR-0106's fixed default seed. `sort` orders by one column, ascending or descending. Each row carries its position in the frame. The library never takes that position back as an id.
- **It shows raw cells as stored.** Declared sentinels are not turned into nulls, so the Agent can find and check the rows its declaration targets. Given up: `is_null` can disagree with the profile's null ratio. The docstring says so, and `missingness` gives the profile's view.
- **One result shape, laid out by column.** The fields are `mode`, `total_rows`, `matched_rows` (an exact count over every row), `row_positions` (rows mode), `group_count` (aggregate mode, counted before the cap) and `columns`, plus `omitted` and `warnings`. Each column record is a tagged union, so the Literal Field marks stay static per variant:
  - `cells` `{name, dtype, values}`: a raw column or a group key. A **Literal Field**. Values are typed by `dtype`, and datetimes are ISO strings.
  - `measure` `{column, fn, values}`: `count`, `null_count`, `nunique`, `sum`, `mean`, `std`. Not literal.
  - `literal_measure` `{column, fn, values}`: `min`, `max`, `median`. A **Literal Field**, because each can equal a cell.

  Gained: no `Any`, each column name appears once, and no type tag per cell. Given up: an Agent reads a row by indexing across arrays.
- **Caps are counts.** Output caps cut the result and list the cut in `omitted`. Input caps raise `InvalidInput` and name the cap.

  | Cap | Value | Kind |
  |---|---|---|
  | `limit` (rows mode) | default 20, max 50 | output |
  | `columns` selected (rows mode) | max 10; null means the first 10 in dataset order | output |
  | string cell or group key | 64 chars, reason `truncated` | output |
  | groups returned | max 100, by count descending, ties in key order | output |
  | `group_by` | max 3 | input |
  | measures | max 5 | input |
  | filter | max 5 OR branches × 10 predicates; `in` lists max 100 values | input |

  The worst cases are estimated at about 8k tokens (rows mode) and 7k tokens (aggregate mode), both under the 10k Result Budget. ADR-0102's fixture test is the real check, and if it fails the caps come down.
- **Under `deny`, the query still runs, and it is not a wall.** `cells` and `literal_measure` values come back null and are listed in `omitted` with reason `literal_policy`. `matched_rows`, `row_positions` and `measure` values stay. The filter takes literal values and returns exact counts, so an Agent can binary-search a withheld value. This is ADR-0107's containment, not privacy, the same trade as means of tiny groups. A Harness that wants a real wall leaves `query_frame` out of its tool set.
- **Errors.**
  - An unknown column is `UnknownName`, with close-match candidates.
  - A filter value whose type doesn't match the column is `InvalidInput`.
  - `sum`, `mean`, `std`, `median`, `min` or `max` on a column that isn't numeric is `InvalidInput`.
  - An input cap that is exceeded is `InvalidInput`.

## Considered Options

- **No query Operation.** Rejected: an Agent on Tool Calls could never see a row, and every look at a fold in Code Mode costs an export.
- **Tool Calls only.** Rejected: it breaks ADR-0099's rule that both deliveries behave the same, and Code Mode loses the cheap read of derived frames.
- **Three Operations (`sample_rows`, `filter_rows`, `aggregate_frame`) or two.** Rejected: more tools and more result shapes for one idea.
- **An AND-only filter.** It would be simpler, but it was rejected for not expressing OR across columns.
- **An expression string.** Rejected: it is free code, and the Strict Subset forbids it.
- **Declarations applied, so sentinels read as null.** Rejected: the Agent could never check its own sentinel declaration.
- **Rows laid out one array per row.** Rejected: every cell would need a type tag, since `Any` is banned.
- **Refusing the query under `deny`, or refusing filters on non-numeric columns.** Rejected: the first makes the policy do more than withhold fields. The second still leaks numeric values through ranges.

## Consequences

- ADR-0110's inventory has 20 Operations, and ADR-0109's catalogue has 20 entries.
- ADR-0099's Harness contract states that `query_frame` under `deny` still leaks through counts.
- ADR-0107's containment test covers `query_frame` in both modes, under `allow` and under `deny`.
