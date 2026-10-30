# Operation schemas come from Pydantic wire records in a strict subset

ADR-0099 made the Python Operation the single source: the MCP server, the catalogue data and the tool definitions are all generated from it. This ADR decides what turns an Operation into a JSON Schema, and the rules every such schema follows. The [schema generation probe](../research/schema-generation-probe.md) found that a generator already works on flat records. It also found that no schema is strict-portable as generated. Anthropic's `transform_schema` silently turns every open per-column map into "accepts only `{}`". `to_dict()` writes enums by name while the derived schema expects values. The private fields behind the config setters leak in as schema properties. And `Any`, live estimators and frames have no schema at all.

## Status

accepted. Resolves [What is the source of truth for schemas, and what rules must they follow?](https://github.com/DEVunderdog/DataForgeML/issues/573) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It builds on ADR-0099 (the source) and ADR-0100 (the validator feeds `invalid_argument`). Planning only: no code changes yet.

**Amended by [ADR-0112](0112-agents-reach-the-library-through-a-cli-and-a-skill.md) (2026-10-11).** The Strict Subset stays, but the test that runs schemas through Anthropic's and OpenAI's transformers is deleted, since no vendor tool schema is sent any more. Its rules now have two reasons. Design rules (no `Any`, strict values, `null` means the default, enums by value, keyed arrays) shape the data. Printing rules make `dataforge schema` output read well for an Agent and keep MCP a cheap add-back. Output lists of flat records go on the wire in Table Layout.

## Decisions

- **Pydantic v2 is a core dependency.** One definition gives the JSON Schema, checks inputs at the Operation boundary in both deliveries, and writes the output JSON. Gained: schema, validation and output cannot drift, so the probe's two drifts (enum name vs value, private vs public keys) cannot come back. ADR-0100's `param` paths come from the validator's location paths for free. Given up: a new compiled dependency (`pydantic-core`) that is capped and bumped under ADR-0080.
- **Operations take and return Wire Records, not domain types.** A Wire Record is a frozen Pydantic model that rejects extra fields and is importable at runtime (no `TYPE_CHECKING`-only names). Internal domain types convert to and from Wire Records at the boundary. Gained: the internals stay free to hold frames, estimators and setters, and the wire is clean by construction. Given up: a converter per type and about 20–30 more classes.
- **Inputs are keyword arguments.** The args model Pydantic builds from the Operation's signature is the Tool Call input object. Gained: natural Python in Code Mode, and the signature the Agent reads is the schema. Given up: nothing material. The probe already built args models from signatures.
- **Descriptions are written once, in docstrings.** Parameter descriptions are parsed from the Operation docstring's `Parameters` section. Record field descriptions come from attribute docstrings. Constraint text (such as "between 0 and 1") is appended by the generator. A test fails any parameter or field without a description, and any documented parameter that is not in the signature. Gained: the text the Agent reads in the file and the text in the tool schema are the same text. Given up: a small numpy-section parser to maintain. How the text is written belongs to [How are Operation docstrings and the Skill written for Agents?](https://github.com/DEVunderdog/DataForgeML/issues/580).
- **In Code Mode an Operation returns the Wire Record instance.** `.to_dict()` gives exactly the Tool Call JSON, the same rule ADR-0100 gives errors. Gained: attribute access and helpers such as `.get(column)`. Given up: the Agent's glue code depends on Pydantic model behaviour. The envelope around the record belongs to [What does every library result look like to an Agent?](https://github.com/DEVunderdog/DataForgeML/issues/574).
- **Every Operation schema follows the Strict Subset, and the library emits it itself.** A post-pass turns each Pydantic schema into the subset, and a test checks every Operation schema against a checker of our own. The test also runs each schema through Anthropic's and OpenAI's transformers and fails if either one changes it. The rules:
  - the input and output roots are objects (a bare list such as `derive_units`'s tuple is wrapped);
  - no extra fields (`additionalProperties: false`);
  - every field is required, and a field the caller does not care about is `null`;
  - no recursion, no fixed tuples (`prefixItems`), no `**kwargs`;
  - constraints are checked at the boundary and written into the description, never emitted as keywords;
  - depth at most 10 and at most 5,000 properties (OpenAI's published limits; `StructuralProfileResult` is depth 6 with 317 today).

  Gained: one strict schema for every vendor, and a test that proves it. Given up: we maintain the post-pass and the checker, and we may be stricter than one vendor needs.
- **No open maps.** A keyed collection is an array of records with the key as a named field, for example `[{"column": "age", "strategy": "median"}]`. A duplicate key raises `InvalidInput` with `duplicate_key`, and an unknown key raises `UnknownName` (ADR-0100). A fixed tuple becomes a named record: `domain_snap_bounds` becomes `{"low": …, "high": …}`. Gained: the only shape that survives every transformer. Given up: a lookup by column in Code Mode is a list search, which the Wire Record's `.get(column)` helper hides.
- **`null` always means "use the library's default".** An omitted field counts as `null`. No parameter may use `null` for a real choice. A real "none" is an explicit value, such as `stratify` set to `"none"`. The `_UNSET` sentinels disappear. Gained: one meaning everywhere, the same behaviour in both deliveries, and an old cached tool schema that lacks a newly added field still validates. Given up: a few parameters get a wordier explicit "none".
- **No `Any`.** A value whose shape depends on a kind is a tagged union: `anyOf` branches, each with a `kind` constant. Each case today maps like this:
  - `data`, `recipe`, `results`, `units`, `serialize(obj)` → a typed handle field (the handle form belongs to [How does an Agent refer to large objects across calls?](https://github.com/DEVunderdog/DataForgeML/issues/575));
  - `mice_estimator`, `ModelChoice | Any`, C2ST `classifier` → a **closed enum of named estimators** per slot, plus a typed parameter record per estimator;
  - `with_hyperparameters(dict[str, Any])` → a tagged union with one typed parameter record per strategy;
  - `time_split`'s `cutoff` → a tagged union of an ISO 8601 datetime or a number;
  - `fill_value`, `TopValueEntry.value`, `rare_label_values` → a scalar union (string, number, boolean, ISO datetime, null). Whether these values appear at all belongs to [Which literal data values may a library result carry by default?](https://github.com/DEVunderdog/DataForgeML/issues/579).

  Gained: every estimator an Agent can name is one we test, and nothing is imported from a string. Given up: foreign estimators, for good. Adding an estimator needs a library release.
- **Values are checked strictly, shapes loosely.** `"5"` is never `5`, and `"true"` is never `True`. An int is accepted for a float, a plain dict for a Wire Record, and a tuple for an array. In Tool Calls an ISO 8601 string is accepted for a datetime, since JSON has none. Gained: the schema means exactly what it says. Given up: an Agent that quotes a number gets `invalid_argument` instead of a silent fix.
- **Enums go on the wire by value, never by name.** An output-only enum (`TypeFlag`, `EpochUnit` and the like) is inlined as a string `enum` with a description. Its values are contract, and its Python class stays internal and is not importable from the Operation surface. Gained: Agents get a closed, documented value set, and we can still move the class. Given up: Code Mode compares strings instead of using `isinstance` or autocomplete on those enums.
- **Schemas are versioned by the library version.** The catalogue and the MCP server carry the library version, and the `[mcp]` extra ships at the same version (ADR-0099). A snapshot test commits every generated schema, so any change shows in review. These changes are breaking: renaming or removing a field, an enum value or an Operation. These are not: adding an input field that defaults to `null`, adding an output field, or adding an output enum value. `format_schema_version` stays only for saved artifacts, which outlive a process. Gained: no version handshake for a skew ADR-0099 already prevents. Given up: a Harness that caches tool definitions across upgrades finds out from an `invalid_argument` error, not from a version check.

## Considered Options

- **msgspec as the generator.** Smaller and faster. Rejected because it never did better than Pydantic in the probe and failed on three more types (`InitVar`, unions of several dataclasses).
- **Hand-written schemas with a contract test.** Rejected because they make a second source of truth, which ADR-0099 rules out, and the validator would have to be written by hand.
- **Fix the domain types until they are strict-portable, and expose them directly.** Rejected because live estimators and frames can never be made to fit, so the result would be a mix of both layers anyway.
- **Emit plain Pydantic schemas and let vendor transformers fix them.** Rejected because Anthropic's transformer silently corrupts open maps.
- **Keep open maps and drop strict mode on those Operations.** Rejected because strict mode is the reason for the schema.
- **`null` as a real value case by case, plus an "unset" marker.** Rejected because it makes the two deliveries harder to keep the same.
- **An open estimator spec (`class_path` plus `params`, loaded with `importlib`).** It keeps foreign estimators. Rejected because it is an open map plus a path to running arbitrary code in the Harness's sandbox.
- **Pydantic's lax mode.** Rejected because it hides the Agent's mistakes and makes Tool Calls accept what the schema calls wrong.
- **A schema version per Operation, sent with each call.** Rejected because the same-version `[mcp]` extra already prevents the skew.
- **Returning plain dicts in Code Mode.** Rejected for attribute access and one rule shared with errors.

## Consequences

- `ModelChoice.Custom`, `with_model_choice(estimator)`, `custom_estimators` and C2ST classifier injection are deleted. This supersedes the user-supplied estimator half of ADR-0090 and the injection half of the C2ST classifier pins (ADR-0087 as amended). `C2STProvenance.classifier_injected` goes. `classifier_params` becomes a typed record of the pinned classifier's parameters. The degeneracy check stays, now for the library's own classifier only.
- The setter lockdown of ADR-0043/0044 no longer touches the wire. Private backing fields stay internal and never reach a schema. Whether configs keep setters belongs to [How does an Agent configure the pipeline?](https://github.com/DEVunderdog/DataForgeML/issues/578).
- ADR-0034's docstrings become the source of schema descriptions for Operations and Wire Records. Their writing style is amended by the docstrings ticket.
- The Error Code registry (ADR-0100) gains `duplicate_key`. The catalogue ([What fields does the Operation catalogue carry?](https://github.com/DEVunderdog/DataForgeML/issues/581)) reads the same generated schemas.
- Persistence keeps `format_schema_version` for artifacts. It is not used for Operation schemas.
