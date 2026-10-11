# Every failure is a Library Error with a stable code

An Agent has to act on a failure, not just read it. Today only three of the library's exceptions carry attributes. The rest are bare classes or plain `ValueError`/`KeyError`/`TypeError`, with column names buried in the message. One sklearn error dumped every singleton class value into a single message. Typos in column names are ignored silently. So every failure now follows one contract, the **Library Error**. It is raised at the source, identified by a stable **Error Code**, capped in size, and carries a hint the library writes itself.

## Status

accepted; the `truncated` count is replaced by `omitted` (ADR-0102). Resolves [What is the one error contract every library failure follows?](https://github.com/DEVunderdog/DataForgeML/issues/572) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It closes gaps G1c, G7, G8, G12, G16 and G17 of `dataforgeml-agent-surface.md` §6. Implemented for every internal raise site by #591; the two codes this ADR left unnamed are named below.

## Decisions

- **The contract applies at every raise, and the Operation boundary enforces it.** Library code raises Library Errors where the facts are known. Each Operation boundary has one catch-all that turns any foreign exception (sklearn, polars, numpy) into `Unexpected`, with `code="unexpected"`, the original class name and a truncated message. `MemoryError` and `KeyboardInterrupt` pass through untouched, because the Harness owns resources (ADR-0099). A foreign error that comes up often gets its own code at its source. Gained: no raw stack text ever reaches the model, and each error is defined next to the code that knows its columns. Given up: internal helpers depend on the error module, and debugging an `unexpected` error needs the logs.
- **The shape is flat.** It has `code` and `message`, both required. The message is one or two sentences saying what to do. The optional fields are `param` (an Operation argument path such as `config.exclusions[2]`), `params`, `columns`, `allowed`, `candidates`, `hint` and `omitted`. We dropped RFC 9457's `type` URI, `status` and `instance`, because there is no HTTP and no doc site to point at. Gained: a small record that is easy to make strict-schema. Given up: conformance to a named standard.
- **Code Mode raises, and Tool Calls return.** The exception carries the record (`err.code`, `err.columns`, `err.to_dict()`). The Tool Calls delivery catches it and returns `is_error` with the same JSON. Gained: normal Python control flow in the Agent's glue code, and one payload for both deliveries. Given up: there is no "result or error" union.
- **One base class with six categories. The code makes the finer split.** The base class `DataForgeError` subclasses plain `Exception`, with no builtin parent. Its categories are `InvalidInput`, `UnknownName`, `Conflict`, `Infeasible`, `IncompatibleArtifact` and `Unexpected`. The category is the advice:
  - `InvalidInput` and `UnknownName`: fix the call.
  - `Conflict`: change one of the two declarations.
  - `Infeasible`: change the plan or the data.
  - `IncompatibleArtifact`: retrain (ADR-0065).
  - `Unexpected`: stop and report.

  Gained: coarse `except` blocks without a class explosion, and `except ValueError` never catches a library error by accident. Given up: catching one code by class. Agents match on `err.code`.
- **Today's named exceptions are deleted and become codes.** That covers `UnseenColumnError`, `FittedColumnAbsentError`, `OverrideCoercionError`, `UnsupportedFormatError`, `IncompatibleArtifactError`, `UnitNotTrainableError`, `C2STDtypeError` and `C2STSampleFloorError`. Warnings are untouched: they belong to results. Gained: one identity per failure. Given up: `except UnitNotTrainableError` becomes `except Infeasible` plus a code check.
- **Codes live in one registry.** Each code has one entry: its category, its hint and the fields it fills. The Skill and the catalogue list codes generated from that registry. A test fails any raise site with no entry. Renaming or removing a code is a breaking change. Adding a code or an optional field is not. Gained: every way the library can fail is listed and tested. Given up: an ad-hoc raise needs a registry entry first.
- **Hints live in the library, not the Harness.** The research gave hints to "the adapter". ADR-0099 then made the library generate its own MCP delivery, so the adapter is the library. A generic Harness cannot know DataForgeML. Gained: one correct hint for every Harness. Given up: tuning hints per model is left to a Harness that rewrites them.
- **No silent acceptance.** Every name an Agent supplies is checked: columns, dials, enum values and unit ids. An unknown name raises `UnknownName` with up to 5 close-match `candidates`. Names that need no data are checked at once. Column names are checked in the first Operation that holds both the config and the dataset handle. Gained: no silent typos. Given up: no instant column check while a config is built before any data exists.
- **One error per call.** Its list fields hold every offender for that code, so three unknown columns make one `unknown_column` error. There is no nested `errors[]`. A schema-validation failure becomes `InvalidInput` with `code="invalid_argument"`, where `param`/`params` come from the validator's location paths and `allowed` from the schema enum. An ill-typed dial value is `invalid_dial_value`, and that includes sklearn's `InvalidParameterError` mapped at fit time. Gained: a flat record whatever layer caught the mistake. Given up: two independent mistakes cost two calls.
- **The library caps every record.** `message` is at most about 500 characters. Each list field holds at most 20 items. A cut list is recorded in `omitted` as `[{field, shown, total, operation}]`, the same shape results use (ADR-0102). Foreign messages are cut to the same limit. ADR-0099's Harness output cap stays as the backstop. Gained: bounded errors in both deliveries. Given up: the full lists. The Agent asks a describe Operation for them.
- **Errors carry names, never cell values.** They hold columns, params and dials, and follow the literal-value policy with no special case. A coercion failure reports a count, not the bad value. Gained: no data leaks through the error path. Given up: messages like "value 'abc' in row 12".

## Considered Options

- **The Operation boundary translates everything, and internal code raises anything.** Rejected because the boundary would have to guess at columns that the raise site already knew.
- **Return an error value instead of raising in Code Mode.** Rejected because every call site would have to check the result.
- **One class per code.** Rejected as a class explosion. **One class only:** rejected because it gives up coarse `except` blocks.
- **Code map, hints and truncation in the Harness or an adapter** (the research's split, §7.3). Rejected for the reasons above.
- **Keep builtin parents** (`ValueError`, `RuntimeError`). Rejected because there is no backward compatibility to keep.

## Consequences

- ADR-0050's clause on exported exceptions is superseded: the freeze moves from class names to Error Codes.
- The C2ST "two named refusals" lines (ADR-0087 as amended) become the codes `c2st_string_column` and `c2st_sample_floor` under `Infeasible`. "Refusal" keeps one meaning: a C2ST outcome carried in a result.
- G7 (`serialize` of an unsupported object) becomes an `InvalidInput` naming the supported kinds. G8 (foreign bytes) becomes `incompatible_artifact`. G1c (stratifying on a continuous target) gets its own code at its source.
- Whether dial values are checked at edit time is left to the configuration ticket. This ADR only fixes the code they raise.
- The two codes left to implementation are named (#591):
  - `stratify_on_continuous_target` (`Infeasible`, fields `param`, `columns`). Every stratified split checks the target before sklearn does. It raises when the target is continuous, or when it has too few rows per value for that splitter: a shuffle split needs every value in at least 2 rows, and k-fold needs at least one value in `k` rows. The message gives counts of values, never the values.
  - `missing_fitted_columns` (`InvalidInput`, field `columns`). A frame lacks columns that a fitted imputer or unit was fitted to fill. It replaces `FittedColumnAbsentError`.
- The registry test reads the source's syntax tree, so it catches a raise of anything but a registered Library Error. A few guards are exempt. They protect the error machinery itself or an Operation's definition at import time, so they fire only on a library bug.
