# Literal values are contained in marked fields under a Harness-set policy

The profile leaks literal values by design: top values, every rare label, exact bounds, percentiles and the mode. ADR-0104 already keeps them off the Dataset Digest and `describe_columns`, and parks `value_summary` until this decision. They still reach the wire in a few other places: target class labels, `fill_value` (a mode is a real label), and `domain_snap_bounds` (the exact min and max). In Code Mode the library runs inside the sandbox, so the Agent's code can read the Declared Input with pandas and print any row. Whatever the library does, it cannot be the privacy boundary there. This ADR decides what the library promises about literal values, and who can restrict them.

## Status

accepted. Resolves [Which literal data values may a library result carry by default?](https://github.com/DEVunderdog/DataForgeML/issues/579) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It extends ADR-0100's "names, never cell values" rule to Result Warnings and Dial Pointers, and settles ADR-0104's open `value_summary`. Planning only: no code changes yet.

## Decisions

- **The library promises containment, not privacy.** Literal values appear only in **Literal Fields**, which are marked in the output schema with an annotation. Every other field is literal-free, and a test checks this. The Harness decides whether Literal Fields reach the model, and uses the marks to find them. Gained: a claim that holds in both deliveries, and filtering a Harness can do by machine. Given up: the library can't tell anyone "your data stays private". That claim belongs to the Harness.
- **A field is a Literal Field if it can hold a cell's value.** Marking is per field, not per value. Literal: category and class labels, min/max, percentiles, median, mode, `domain_snap_bounds`, and `fill_value`. Not literal: means, std, bimodal centres, skew, ratios, counts, correlations, and Routing Evidence terms. Gained: a rule that is easy to test, and scale statistics stay visible under `deny`. Given up: no protection against means of tiny groups. Only differential privacy fixes that, and it is deferred.
- **The default is full values.** Under the default, Literal Fields hold real labels, exact bounds and real counts. There is no masking, bucketing or small-count suppression. The Agent needs real literals to do its job: it must see `-999` or `"?"` to declare them as sentinels, and it needs real labels to write Code Mode code. Since the digest and `describe_columns` are literal-free, the Agent only gets literals from an Operation it calls for them. Gained: accuracy and one simple rule. Given up: no safe default. A Harness that does nothing sends real values to the model whenever the Agent calls `value_summary`.
- **The Literal Policy is `allow | deny`, set by the Harness outside the Agent's reach.** It is set like the Workspace (ADR-0103): through an environment variable or an MCP server startup argument. The default is `allow`. No Operation argument changes it, because the Agent is the party it restricts. Gained: a control the Agent can't override, in both deliveries, for everything the library returns. Given up: one more Harness setting, and a second result shape to test.
- **Under `deny`, a Literal Field is withheld.** It comes back `null`, it is listed in `omitted` with reason `literal_policy`, and the call carries one Result Warning. Counts and shares next to a withheld label stay, because they are aggregates. Gained: one simple rule, with no reverse token map in the Workspace. Given up: under `deny` the Agent can't act on a value it can't see, such as declaring a sentinel nobody told it about.
- **There are two states, not three.** No middle tier suppresses small counts or rounds extremes. Gained: one switch, with no arbitrary *k*. Given up: no middle ground. A Harness that wants one builds it from the marks. A third state can be added later without breaking anything.
- **Result Warnings, Dial Pointers and Library Errors carry no literal values.** A warning that would name a class label gives a count instead, and points at the Operation that holds the literal values. Gained: `deny` needs no special cases outside Literal Fields. Given up: messages like "class 'X' has one row".
- **Values the Agent supplied are not Literal Fields.** Declared sentinels, MNAR declarations and type overrides that are echoed on a result stay visible under `deny`. Gained: echoing them leaks nothing, because the Agent already has them. Given up: nothing.
- **Column names are schema, not literal values.** The library never masks them. Gained: a library that works at all, since every Operation, error and Dial Pointer names columns. Given up: no protection for a sensitive name. A Harness renames columns before declaring the input.
- **No literal value ever enters a schema.** Schemas are static per library version (ADR-0101), so no column name or label becomes an `enum`, `const` or `pattern`. Close-match candidates and column lists travel as data. Gained: provider schema caches never hold dataset content. Given up: per-dataset enum validation of column names. `UnknownName` covers that already.
- **`value_summary` is offered.** It is the main Operation that carries Literal Fields. Any later Operation that returns cell values, including rows or query results, returns them as Literal Fields under the same policy.

## Considered Options

- **A privacy promise ("library results don't leak your data").** Rejected: it is false in Code Mode, and in Tool Calls whenever the Agent has another tool that reads files.
- **Masked by default, with opt-in reveal (pseudonym tokens, rounded extremes, counts of 1–10 hidden).** Rejected: it breaks sentinel discovery and code written against real labels. Pseudonymised data is still personal data under GDPR Recital 26, so the masking sells protection it doesn't give.
- **A per-call argument for opting in.** Rejected: the Agent would be lifting its own restriction.
- **Nothing in the library, with the Harness stripping fields itself.** Rejected: every Harness would rebuild the classification, with no marks to guide it.
- **Pseudonym tokens under `deny`, accepted back as arguments.** Rejected: it needs a reverse map in the Workspace, and it gives the same false sense of anonymity.
- **"On the column's scale" as the Literal Field rule.** Rejected: it would withhold means and std under `deny` while buying little protection.

## Consequences

- ADR-0099's Harness contract gains the Literal Policy setting. It also states that `deny` is a real boundary only when the Agent's code can't read the Declared Inputs, which means Tool Calls, or a sandbox that doesn't mount the inputs. The library doesn't try to detect which case it is in.
- A containment test seeds a fixture with canary values. It asserts that no canary appears outside a Literal Field in any Operation's result, warnings or errors, and that under `deny` none appears anywhere.
- The split fallback that names singleton class labels changes to report a count.
- The exact schema annotation keyword is chosen at implementation.
- The map's fog item "bounded lazy querying of raw data" is unblocked. Its results are Literal Fields by this ADR.
