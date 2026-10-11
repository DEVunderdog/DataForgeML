# Agents reach the library through typed Operations

The Agent is the library's only user, so we had to decide how an Agent reaches the library at all. We follow Anthropic's "code execution with MCP" and "advanced tool use" patterns. The library ships a small set of typed **Operations**. An Agent's code calls them inside a sandbox the Harness owns (**Code Mode**), or the Harness hands them to the model as tools (**Tool Calls**). Both deliveries come from one source: the Python Operation itself. The goal is fewer tokens, deterministic decisions, nothing for the model to invent, and accurate results.

## Status

accepted. Resolves [How does an Agent reach the library?](https://github.com/DEVunderdog/DataForgeML/issues/571), the root decision of [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It rests on the [access-routes research](https://github.com/DEVunderdog/DataForgeML/issues/569). Planning only: no code changes yet. This is a breaking change with no compatibility shim, because the library has no users yet.

**Amended by [ADR-0112](0112-agents-reach-the-library-through-a-cli-and-a-skill.md) (2026-10-11).** The CLI is now the primary delivery and Code Mode the second. Tool Calls, the `[mcp]` extra and the "no library CLI" decision are withdrawn. The Harness contract shrinks to a shell with Python ≥ 3.11, `dataforge` on the PATH and the Skill installed. The Workspace, Literal Policy, Declared Inputs and core budget become Workspace Settings: Operator-set guardrails, not a boundary. The Operations as the whole public surface, JSON-able inputs and Handles, and one Python source stand.

## Decisions

- **The library runs inside a sandbox the Harness owns.** It is installed ahead of time and needs no network. The library never depends on a vendor's hosted sandbox. Gained: every route stays open, and no dependency surgery is needed. Given up: an Agent with only a vendor code-execution tool cannot run the library on its own. Anthropic's container has no network and no polars, and a Skill upload is capped at 30 MB.
- **The primary route is Code Mode over typed Operations, not plain code over the library.** Gained: the library makes every decision inside a typed Operation, results are bounded, inputs are schema-checked, and errors are typed. The model writes only the glue between Operations, so CodeAct's accuracy edge is kept. Given up: the Agent's code sits next to the raw CSV and can print anything. What reaches the model is bounded by the library only by default. The Harness's sandbox owns the final guarantee.
- **There are two deliveries: Code Mode and Tool Calls. There is no library CLI.** Gained: one less generated surface to keep in sync. Given up: an Agent with a shell but no Python gets nothing. Any shell that has the library installed also has Python, so this case is rare.
- **The Operations replace the public Python API.** Today's door becomes internal. That means `route`, `resolve_recipe`, `fit_unit`, the configs and their setters, and the `to_markdown` reports. Gained: one surface to document, test and evaluate, and no raw path in Code Mode that skips bounded results and typed errors. Given up: in-sandbox code has no supported path to the raw objects. Phases 3–6 must be written as Operations from the start.
- **Operations take and return only JSON-able values and handles, in both deliveries.** A frame or a fitted unit is never passed by value. Gained: Code Mode and Tool Calls behave the same, so one test covers both. Given up: Code Mode cannot hand the library a frame it changed with pandas. Today's user-owned steps, such as dropping a column, need an Operation or a way to register a changed frame as a new handle.
- **The Python Operation is the single source. MCP is a generated delivery, never the hub.** The MCP server, the catalogue data and the tool definitions are all generated from it. Gained: Code Mode needs no MCP at all. Given up: we generate the MCP delivery ourselves, instead of getting stubs, a CLI and a Skill free from FastMCP's MCP-centred chain.
- **Packaging: the core package plus one optional extra.** The core package holds the Operations and the Skill, with no LLM or MCP dependency. `dataforge-ml[mcp]` holds the MCP server, in the same repo and at the same version. Gained: Tool Calls can never drift from the Operations. Given up: MCP adapter releases are tied to library releases.
- **What an Agent gets first is a Skill shipped inside the package.** `SKILL.md` holds the workflow and points into the package. Each Operation lives in its own Python file, and the Agent reads that file in place. This is the post's progressive-disclosure file tree: about 100 tokens up front, then one file per Operation actually used. The catalogue data for tool search is generated from the same files. Gained: the Agent reads the real signature, so nothing drifts. Given up: the package layout becomes part of the contract, so moving an Operation's file breaks Agents. Operation docstrings now serve Agents first, which changes ADR-0034's maintainer style for Operations.
- **The Harness contract.** The Harness provides:
  - a sandbox with the library installed, Python ≥ 3.11 and no network;
  - the input CSV as a file path inside the sandbox;
  - a cap on how much Code Mode output reaches the model, and a Tool Calls output cap at least the largest `result_budget_tokens` in the Operation Catalogue (added by ADR-0109);
  - all resource limits (memory, time, CPU);
  - a writable working directory, if handles turn out to need one;
  - the package's Skill directory (`dataforge_ml/`) exposed to the Agent, by copying or linking it into the Harness's skills location (added by ADR-0108).

  The library promises only that its own results are bounded. Gained: a short contract that can be tested. Given up: the library makes no promises about time or memory. A MICE fit that runs out of memory is the Harness's to contain.
- **We decide from the research now, and measure later.** No source has measured these routes on a data-preparation library. Gained: the tickets for results, handles, schemas and docs can start. Given up: no empirical check before committing. An evaluation rig needs a typed surface to measure, so it is deferred to a separate effort after implementation. It is out of scope for this map ([How do we measure whether Agents use the Operations well?](https://github.com/DEVunderdog/DataForgeML/issues/582)), because the priority is making the surface work.

## Considered Options

- **Plain typed tool calls only, with no code.** This has the most portable wire format and typed errors. Rejected because every intermediate result passes through the model, and the Agent loses code as glue.
- **Plain code over the library's normal Python API, taught by a Skill (A1).** This is the most flexible option. Rejected because nothing bounds what the Agent prints (one profile is about 394k tokens), and results, errors and the literal-value policy stop being something the library controls.
- **MCP as the hub (typed functions → MCP → stubs, CLI and Skill).** This is the chain that works with today's tooling. Rejected because the library is already installed in the sandbox, so Code Mode can import the real functions and needs no MCP hop.
- **A separate `dataforge-ml-mcp` package.** Rejected because a second version line can drift.
- **Targeting vendor-hosted sandboxes.** Rejected because polars, `diptest` and `iterative-stratification` cannot be installed there, and the Harness runs its own Docker sandbox.

## Consequences

- ADR-0050's export rule now applies to Operations, not to today's door. The Public API entry in `CONTEXT.md` points at the Operations.
- ADR-0034's docstring rule needs amending for Operations. That is [How are Operation docstrings and the Skill written for Agents?](https://github.com/DEVunderdog/DataForgeML/issues/580), resolved by ADR-0108.
- The schema ticket inherits the source decision: schemas are derived from the Python Operation.
- The handles ticket inherits one fact: handles exist in both deliveries.
- Two ideas from the Anthropic post stay with the Harness: saving reusable code to `./skills/`, and tokenising PII before it reaches the model.
- The version bump is the maintainer's to make.
