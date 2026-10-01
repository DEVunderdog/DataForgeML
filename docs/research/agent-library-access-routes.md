# How Agents call a Python library today, and what each route costs

Research ticket [#569](https://github.com/DEVunderdog/DataForgeML/issues/569) (part of map #568). Researched 2026-10-01.

**Question.** How do LLM Agents call a Python library today, and what does each route cost? Four routes are compared, Anthropic first, then OpenAI and the open frameworks:

1. **Code in a sandbox.** The Agent writes Python; only what it prints comes back.
2. **Typed tool calls.** Native tool use or an MCP server, with JSON Schema in and out.
3. **A CLI.** Subcommands the Agent runs from a shell.
4. **Agent Skills.** A `SKILL.md` plus scripts, loaded on demand.

For each route: what the model sees (up front and per call), how large data stays out of context, how state is held between calls, error and retry behaviour, sandbox and resource needs, provider portability, and which Python data/ML libraries already ship that way. Then: can one source of typed Python functions produce several routes at once?

**This note decides nothing.** It feeds the decision ticket "How does an Agent reach the library?". The last section lists considerations, not a recommendation.

**Builds on** two earlier notes in the `dataforgeml-agent` repo (`docs/research/`). They are not repeated here:
- [P1] *What makes a Python library's API consumable by an LLM agent* (`agent-consumable-api.md`). It covers the tool-definition contracts (MCP, Anthropic, OpenAI), the portable strict-schema subset, output shaping, error payloads, handles, schema generation from code (Pydantic, PydanticAI, OpenAI Agents SDK, MCP Python SDK), and the per-entry-point gap table for DataForgeML.
- [P2] *How agents route a request to predefined capabilities* (`capability-routing.md`). It covers tool-count accuracy thresholds, tool search and deferred loading, Agent Skills' three loading levels, and MCP `tools/list`.
- Also used for context: `agent-frameworks.md` [P3], `data-context-channels.md` [P4] (Claude code execution + Files API, Julius containers), and `dataforgeml-agent-surface.md` [P5] (profile ~394k tokens, a fitted MICE unit 127 MB).

**Sources and method.** Primary sources only, read on 2026-10-01: Claude platform docs and Anthropic engineering posts, the Claude Code and Agent SDK docs, the MCP specification **2026-07-28**, OpenAI API docs, and the official docs or repos of FastMCP, PydanticAI/Monty, smolagents, LangChain, Cyclopts and Typer. One benchmark (Scalekit) is a vendor blog and is marked **secondary**. DataForgeML facts come from `pyproject.toml` at `origin/main` (v3.3.0) and PyPI's JSON API. Anything not checked is marked **unverified**. References are in [Sources](#sources).

---

## 0. Summary

- **"Code in a sandbox" is really two routes, and the split matters more than any other in this note.**
  - **A1. The code imports the library.** The library must be installed inside the sandbox. Examples: Claude's code execution tool, OpenAI Code Interpreter and hosted shell, smolagents `CodeAgent`, Jupyter MCP servers.
  - **A2. The code calls tools.** The library stays in the host process, and sandboxed code calls it through generated stubs. Examples: Anthropic programmatic tool calling, OpenAI programmatic tool calling, Cloudflare Code Mode, Anthropic's "code execution with MCP", PydanticAI Code Mode on Monty, FastMCP `CodeMode`.
  - A2 gets most of A1's context savings without putting the library in the sandbox. Every A2 sandbox measured here forbids third-party imports, so **the library can only be reached as a tool under A2**.
- **Anthropic's hosted sandbox cannot import DataForgeML today.** The container has no network, and "only the pre-installed libraries are available" [A1]. The list has pandas, numpy, scipy and scikit-learn, but not polars, `diptest` or `iterative-stratification`, all of which DataForgeML requires (D1). Skills on the API have the same limit [A4]. The polars runtime wheel alone is 49.9 MB, more than the 30 MB Skill upload cap [A5][D2]. OpenAI's hosted shell can `pip install`, but only with a network allowlist [O4].
- **Typed tools cost the most up front and the least per call to reason about.** The definitions sit in every request: a tool-use system prompt of 286 tokens on Opus 5.5, plus the definitions themselves [A7]. Tool search removes most of that once a client has 10+ tools [A8][P2]. Each call returns a typed result, an `isError` the model can act on, and a schema-validated input.
- **A CLI costs almost nothing up front but needs a shell.** The model sees `--help` text only when it asks for it, and output comes back as text, capped by the harness (Claude Code: 30,000 characters by default) [C2]. One vendor benchmark measured 4–32× fewer tokens than an MCP server, but it ran on a model without tool search, and the MCP failures were network timeouts (**secondary**) [X1].
- **Skills are a delivery format for instructions, not a call mechanism.** A Skill tells the Agent how to use code, a CLI or tools. It costs ~100 tokens per Skill up front and under 5k when triggered [A3]. The format is an open standard listed by 46 clients on agentskills.io, OpenAI Codex, Gemini CLI and Cursor among them [S1]. On the Claude API and OpenAI's hosted shell a Skill runs inside the vendor sandbox, so it carries route A1's install problem.
- **State has the same three homes everywhere:**
  - a live interpreter (A1 REPLs, Jupyter kernels, PydanticAI Code Mode within one run);
  - explicit handles passed between stateless calls (MCP's normative "no protocol-level session" [M1]);
  - files on disk (CLI, Skills, Anthropic's `./skills/` pattern [A9]).
- **One source can produce several routes, and the tooling exists.**
  - Typed, docstring-ed Python functions → MCP tools with input and output schemas (FastMCP, MCP SDK).
  - The same functions → native tools (Anthropic `@beta_tool`, OpenAI `function_tool`, PydanticAI, LangChain `@tool`).
  - The same functions → a CLI (Cyclopts parses NumPy docstrings; Typer reads only the summary).
  - An MCP server → a CLI **plus** a generated `SKILL.md` (FastMCP `generate-cli`, v3.0.0).
  - An MCP server → code mode (FastMCP `CodeMode`; PydanticAI `CodeModeToolset` wraps any toolset).
  - So "typed functions → MCP server → {native tools, code-mode stubs, CLI, Skill}" is a working chain today, with the MCP schema as the hub.

---

## 1. Comparison table

"Up front" means tokens in every request before work starts. "Per call" means what one operation adds to context.

| | **A1. Code imports the library** | **A2. Code calls tools** | **B. Typed tool calls (native / MCP)** | **C. CLI** | **D. Agent Skills** |
|---|---|---|---|---|---|
| **Up front** | The code tool's description and the sandbox tool. The model must already know the library's API, or read docs. | The code tool plus **typed stubs** for each callable tool (TS/Python signatures). FastMCP and Anthropic's MCP post load stubs on demand [F2][A9]. | Tool-use system prompt (286 tokens, Opus 5.5) **plus every definition** [A7]. With tool search: only the search tool and 3–5 hot tools [A8]. | A shell tool only. Help text costs nothing until the Agent runs `--help`. | ~100 tokens per Skill (name + description) [A3]. |
| **Per call** | The code the model writes (output tokens) plus whatever it prints. | The program (output tokens) plus only the **final** output. "Tool results from programmatic calls are not added to Claude's context" [A2]. | One `tool_use` + one `tool_result` per operation; every intermediate result passes through the model. | The command line plus stdout/stderr, truncated by the harness [C2]. | <5k tokens when triggered, then whatever the scripts print [A3]. |
| **Large data out of context** | Yes: data lives in the sandbox; the model sees only prints. | Yes: tool results stay in the program; "the agent sees five rows instead of 10,000" [A9]. | Only by design: handles, pagination, `resource_link`s, field selectors [P1][M1]. | Yes if the CLI writes files and prints a summary. | Yes: "scripts run through bash, and only their output enters context" [A3]. |
| **State between calls** | Live interpreter. Claude: REPL persists across requests that reuse the container (`code_execution_20260120`+); 30-day cap, checkpointed after ~5 min idle [A1]. OpenAI: 20 min idle expiry [O3]. | Anthropic PTC: same container REPL [A1][A2]. OpenAI PTC: **no** state between programs [O5]. PydanticAI Code Mode: state persists within a run [F5]. | None at the protocol level; explicit handles [M1]. Native tools: whatever the host process keeps. | Files and IDs on disk; each invocation is a fresh process. | Files on disk; Skills can save scripts (`./skills/`) [A9]. |
| **Errors and retry** | A Python traceback in stderr; the model rewrites the code. | Tool errors arrive inside the program as strings; Anthropic PTC raises `TimeoutError` after ~4 min waiting for a result [A2]. | `isError` / `is_error` results; Claude "will retry 2-3 times" on invalid input [P1]; SDKs turn exceptions into error results [A6][F4]. | Exit code plus stderr; quality depends on the CLI's messages. | Whatever the script prints. |
| **Sandbox / resources** | Required. Claude: 5 GiB RAM, 5 GiB disk, 1 CPU, no internet [A1]. OpenAI: 1–64 GB tiers [O3]. Local: Docker, E2B, Modal, Blaxel, or smolagents' AST interpreter [F6]. | Required but light: V8 isolates "a few megabytes" (Cloudflare) [X2]; Monty <1 ms start, 256 MiB / 30 s default (PydanticAI) [F5][F7]; FastMCP 100 MB / 30 s [F2]. Anthropic PTC uses the full container; 90 s per cell [A1]. | None beyond the host process (or the MCP server's process). | A shell, so the harness's own sandbox and permissions. | A filesystem plus code execution [A3]. |
| **Portability** | The pattern is universal; the hosted sandboxes are vendor-specific (and the library must be installed in each). | Hosted PTC is vendor-specific (Anthropic: Claude API, AWS, Foundry; **not** Bedrock or Vertex [A2]). PydanticAI and FastMCP code mode are provider-agnostic [F5][F2]. | The most portable: every provider has tool calling; MCP is read by every major client. | Any Agent with a shell (all coding agents). Not available to plain chat API calls without a shell tool. | 46 clients listed on agentskills.io [S1]; vendor-sandbox runtimes differ (network, installs) [A3]. |
| **Data/ML precedents** | Claude container ships pandas/numpy/scipy/sklearn/statsmodels [A1]; Jupyter MCP servers [L1]; smolagents. | MotherDuck MCP + any code-mode client; Cloudflare's MCP servers. No data library ships an A2 surface of its own (found). | MotherDuck/DuckDB MCP [P1]; Jupyter MCP [L1]; Hugging Face MCP (**unverified**). | `duckdb`, `hf`, `gh`, `kaggle`; HF's `hf-cli` Skill drives the `hf` CLI [L3]. | Anthropic `xlsx` [L2]; HF Skills [L3]; Scientific Agent Skills (scikit-learn, polars, statsmodels, …) [L4]. |

---

## 2. Route A: code in a sandbox

### 2.1 A1: the code imports the library

**Anthropic code execution tool** [A1].
- Versions: `code_execution_20250825` (bash + files), `code_execution_20260120` (adds REPL state persistence and programmatic tool calling), `code_execution_20260521` (same runtime; the description tells Claude about the 90 s per-cell limit). GA; no beta header.
- Platforms: Claude API, Claude Platform on AWS, Microsoft Foundry (Anthropic-hosted only). **Not on Amazon Bedrock or Google Cloud.** Not ZDR-eligible.
- Runtime: Python 3.11, Linux x86_64, **5 GiB RAM, 5 GiB disk, 1 CPU**. "Internet access: Completely disabled". "Claude can't download or install additional packages at runtime: only the pre-installed libraries are available".
- Pre-installed data stack: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, pyarrow, openpyxl, joblib, among others. **Polars is not listed.** Library versions are not documented (**unverified**).
- State: pass the container ID back to reuse it. "Containers expire 30 days after creation. After about 5 minutes of inactivity a container is checkpointed", and a later request restores it. With `20260120`+, Python variables persist too.
- Output: bash results carry `stdout`, `stderr`, `return_code` and file IDs from `$OUTPUT_DIR` [P4]. Error codes include `output_file_too_large` and `execution_time_exceeded`. The output size cap is not stated as a number (**unverified**).
- Price: billed by execution time, 5-minute minimum, **1,550 free hours per org per month, then $0.05 per container-hour**. Free when the request also uses web search or web fetch (`*_20260209`+). If files are attached, time is billed even when the tool is not called.

**OpenAI Code Interpreter and hosted shell** [O3][O4].
- Code Interpreter memory tiers: `1g` (default), `4g`, `16g`, `64g`, priced at **$0.03 / $0.12 / $0.48 / $1.92 per 20-minute session per container**, billed by the minute with a 5-minute minimum [O6]. "A container expires if it is not used for 20 minutes", and an expired container cannot be reactivated [O3]. The page does not say whether packages can be installed (**unverified**).
- Hosted shell (Responses API only): Debian 12 with Python 3.11 and Node 22, among others. "Hosted containers don't have outbound network access" by default; a `network_policy` allowlist opens named domains, with a warning about prompt-injection exfiltration. The docs show `pip install` in examples. Output is capped by `max_output_length`; commands take `timeout_ms` [O4].

**smolagents `CodeAgent`** [F6].
- By default it runs generated code in-process through `LocalPythonExecutor`, an AST interpreter. Imports are refused unless listed in `additional_authorized_imports`; submodules need explicit authorisation; loops stop after 1,000,000 iterations.
- The docs say plainly that "no local python sandbox can ever be completely secure", and offer `executor_type` values `e2b`, `modal`, `docker` and `blaxel`. In those, "only the output will be returned".
- Rationale cited: CodeAct (ICML 2024) reports "up to 20% higher success rate" than JSON or text actions across 17 LLMs [R1].

**What A1 costs a library like DataForgeML.**
- **Install.** The library and its dependencies must be inside the sandbox. DataForgeML needs `scikit-learn>=1.9,<1.10`, `numpy>=2.4,<2.6`, polars, pandas, `diptest`, `iterative-stratification` and others (D1). That rules out Anthropic's hosted container as it stands. Whether an uploaded wheel could be `pip install`ed offline there is not documented (**unverified**).
- **Discovery.** The model has to know the API. Nothing is declared up front, so the model relies on training data or reads docs (`help()`, a Skill, `llms.txt`). DataForgeML is not in any model's training data to a useful degree (assumption, **unverified**).
- **Output discipline.** Nothing stops the model printing a 394k-token profile [P5]. The harness caps protect the context, but the library has no say over what gets printed.

### 2.2 A2: the code calls tools (library stays host-side)

**Anthropic programmatic tool calling (PTC)** [A2].
- GA. Opt-in per tool with `allowed_callers: ["code_execution_20260120"]`. Same platforms as code execution (not Bedrock or Vertex). Not ZDR-eligible. Haiku 4.5 does not support it.
- Mechanism: tools "are exposed to Claude's code as async Python functions… Each function takes a single dict of arguments and returns a string". The response pauses with a `tool_use` block whose `caller` names the code run. **Your process executes the tool**, returns a `tool_result`, and the code resumes.
- What enters context: "Tool results from programmatic calls are not added to Claude's context — only the final code output is". These tool results are not billed as input tokens.
- Numbers (Anthropic-run):
  - "calling 10 tools directly uses ~10x the tokens of calling them programmatically and returning a summary";
  - a 75-tool project-management benchmark: billed input tokens down ~38%, accuracy unchanged;
  - τ²-bench (one or two sequential calls per turn): scores unchanged, **cost ~8% more**;
  - production traffic with 10–49 tools: typical savings 20–40%;
  - the engineering post: 43,588 → 27,297 tokens (−37%) on complex research tasks, and GIA 46.5% → 51.2% [A10].
- Stated weak fit: "Strictly sequential workflows where each call depends on Claude reasoning over the previous result", and "A small number of tool calls with small responses".
- Limits: tools with `strict: true` cannot be called programmatically; no forcing via `tool_choice`; schemas with a recursive `$ref` are refused; MCP-connector tools cannot be called programmatically. A pending call times out after **~4 minutes** (raising `TimeoutError` in the code), idle containers are reclaimed after **~5 minutes**, and each REPL cell has a **90 s** wall clock [A1][A2].
- Error path: a tool error is just a string result that "Claude's code receives… and can handle". Anthropic advises documenting the output format (JSON shape, field types), because the model parses it in code.

**OpenAI programmatic tool calling** [O5].
- The model writes **JavaScript** with top-level `await`, run in "a fresh, isolated V8 runtime" with no Node.js, packages, network, filesystem, subprocesses or console.
- Eligible tools: function/custom, MCP, `apply_patch`, local and hosted shell, `code_interpreter`, each opted in with `allowed_callers: ["programmatic"]`.
- **No persistent JavaScript state between programs.** Output is emitted with `text(...)`/`image(...)`. ZDR is possible when enabled for the org.
- The OpenAI Agents SDK exposes this as `ProgrammaticToolCallingTool` [F4].

**Cloudflare Code Mode** (2025-09-26, updated 2026-07-15) [X2].
- MCP tool schemas are converted to a TypeScript API with JSDoc; the model writes TS against it.
- The code runs in V8 isolates that start in "merely milliseconds" and use "only a few megabytes". `globalOutbound` blocks network. MCP access goes through bindings, so "the AI cannot possibly write code that leaks any keys".
- Rationale: models have seen far more real TypeScript than synthetic tool calls. No accuracy numbers are published.

**Anthropic, "Code execution with MCP"** (2025-11-04) [A9].
- MCP servers become a file tree (`servers/<server>/<tool>.ts`); the Agent explores it and loads only the definitions it needs.
- Example: "from 150,000 tokens to 2,000 tokens… 98.7%" (vendor example, not a benchmark).
- Data stays in the sandbox and can be tokenised (PII) before reaching the model. State and reusable code are saved to a `./skills/` folder.
- Stated cost: "Running agent-generated code requires a secure execution environment with appropriate sandboxing, resource limits, and monitoring."

**PydanticAI Code Mode / Monty** [F5][F7].
- `CodeModeToolset` wraps any toolset. "Every eligible regular tool becomes callable from inside `run_code`", and the model sees Python stubs.
- Monty is a Rust interpreter for a Python subset: "**No third-party imports**"; stdlib modules such as `asyncio`, `json` and `re` are allowed.
- Defaults: **30 s per snippet, 256 MiB heap, 100 tool calls per `run_code`**, 1,000 suspensions per session. Syntax errors count as retries (default 3). Exceeding a limit resets the session.
- State "persists between `run_code` calls within the same agent run". Works with any provider PydanticAI supports. Start-up "under 1ms from a running pool", against "around 1500ms for a sandbox service" (Pydantic's numbers).
- Status: Monty is labelled experimental in other Pydantic material (**unverified** against the README, which the fetch did not show).

**FastMCP `CodeMode` transform** (v3.1, experimental) [F2].
- Replaces a server's tool list with meta-tools: `search` (BM25), `get_schema` and `execute` (Python that chains `call_tool()`). The stages can be collapsed for small catalogues.
- Sandbox: Monty by default, with **30 s, 100 MB, recursion depth 1,000, 50 tool calls per execute**.
- Works with **any MCP client**, because to the client it is just three tools.

**What A2 costs a library like DataForgeML.**
- **The library must still be a tool.** A2 needs the same typed tool surface as route B. The model writes code against stubs generated from the tool schemas. So A2 adds to B; it does not replace it.
- **Gains depend on workload shape.** PTC pays off for fan-out and filtering, and loses (~8%) on short sequential chains [A2]. A DataForgeML Run is mostly sequential (profile → route → recipe → fit → score) with fan-out inside phases (one `fit_unit` per unit; per-column queries). Where its gains would land is **unmeasured**.
- **Output contract matters more.** Results reach code as strings, so the output format must be documented and parseable [A2]. That favours JSON with a declared schema, which is MCP `outputSchema`'s job [M1].

---

## 3. Route B: typed tool calls (native or MCP)

[P1] covers contracts, the portable schema subset, output shaping, errors and handles. [P2] covers tool counts and tool search. New here are costs and client limits.

**Up-front cost.**
- Tool-use system prompt: **286 tokens** on Opus 5.5 and Sonnet 5.5 (auto/none); 286–675 across current models, and up to 804 for `any`/`tool` on Opus 4.7 [A7]. Then every tool's name, description and schema.
- Tool search cuts this. Anthropic: a five-server setup "can consume ~55k tokens in definitions", and tool search "typically reduces this by over 85 percent". Up to **10,000 deferred tools**; 5 results per search by default; use it at 10+ tools or >10k tokens; keep 3–5 tools non-deferred; caching preserved; strict mode composes with it [A8].
- **Claude Code and the Agent SDK turn tool search on by default** for MCP tools: the model sees names in a compact list and loads schemas on demand [C1][C3]. It is off by default behind a non-first-party `ANTHROPIC_BASE_URL` and for models before the 4.5 generation [C1][C4].

**Per-call cost.** Each operation is one model turn: `tool_use` out, `tool_result` in. Intermediate results pass through the model, which is the cost A2 removes. Claude Code warns when an MCP tool result exceeds **10,000 tokens** and caps it at **25,000 by default** (`MAX_MCP_OUTPUT_TOKENS`); a tool can raise its own cap with `_meta["anthropic/maxResultSizeChars"]` up to 500,000 characters [C1].

**Structured output.** MCP 2026-07-28 lets `structuredContent` be "any JSON value", validated against `outputSchema`, with a text twin "for backwards compatibility" [M1]. **Is it actually read by models?**
- Claude Agent SDK: "When `structuredContent` is set, Claude receives the JSON plus any image or resource blocks from `content`. Text blocks in `content` are not forwarded" [C3]. That is first-party evidence that a Claude harness uses the structured channel.
- But the SDK's Python `@tool` decorator forwards only `content` and `is_error`; Python needs a standalone MCP server for `structuredContent` [C3].
- OpenAI Agents SDK: `use_structured_content=True` makes it prefer structured payloads over text (default not stated in the fetch, **unverified**) [F8].
- [P1] found that no surveyed adapter used `structuredContent` yet.

**State.** "MCP has no protocol-level session"; state lives behind explicit handles with a stated lifetime and an "expired or unknown handle" error [M1]. In-process native tools can hold Python objects in the host, but the model still names them by handle.

**Errors and retry.**
- Exceptions become `is_error` results in every SDK checked:
  - Anthropic tool runner: "the exception's message (in Python, its type and message), not the full stack trace" [A6];
  - Agent SDK: "the raw exception message" [C3];
  - FastMCP: `ToolError` is always passed through; other errors can be masked with `mask_error_details` [F1];
  - OpenAI Agents SDK: `default_tool_error_function` [F4];
  - LangChain: `@wrap_tool_call` middleware [F9].
- Strict modes guarantee schema-valid inputs [P1], but strict tools cannot be called programmatically on Anthropic [A2].

**Sandbox.** None needed for the model's actions. The library runs in the host (native tools, in-process SDK MCP servers) or in a separate server process (stdio or HTTP MCP). Claude Code's stdio idle timeout defaults to 30 min, and HTTP to 5 min [C1].

**Hosted MCP from the API.** Anthropic's MCP connector (beta `mcp-client-2025-11-20`) supports **only tool calls**, and the server "must be publicly exposed through HTTP… Local STDIO servers cannot be connected directly". Not ZDR-eligible [A11]. A local-library MCP server can therefore reach Claude only through a client that runs it (Claude Code, the Agent SDK, Desktop) or by being exposed over HTTPS.

**Measured comparison with CLI (secondary).** Scalekit (2026-03-11) ran 5 read-only GitHub tasks × 5 runs on Claude Sonnet 4: the CLI used 1,365 tokens against 44,026 for GitHub's MCP server on the simplest task (4–32× across tasks), with success rates of 25/25 (CLI) and 18/25 (MCP). Every MCP failure was a TCP timeout to the remote server [X1]. **Caveats:** Sonnet 4 does not support tool search [A8], the server was remote, and n = 25.

---

## 4. Route C: a CLI

**What the model sees.** Only a shell tool up front. It discovers commands with `--help`, from a Skill or `AGENTS.md`, or from training data. Per call it sees the command it wrote and the output.

**Harness limits (Claude Code)** [C4]:
- bash output read back: **30,000 characters** by default, 150,000 at most (`BASH_MAX_OUTPUT_LENGTH`);
- foreground timeout: **2 min** by default, **10 min** at most (`BASH_DEFAULT_TIMEOUT_MS`, `BASH_MAX_TIMEOUT_MS`);
- longer jobs can run in the background.

**Large data.** It stays out of context if the CLI writes files (parquet, JSON) and prints a summary and a path. Nothing forces that: an unbounded `print` is truncated at the harness cap, not shaped.

**State.** Each invocation is a new process. State lives in files (a run directory, IDs printed back) or a resident daemon. The library's import cost is paid on every call: sklearn, polars and scipy imports take seconds (**unmeasured** for DataForgeML).

**Errors.** Exit code plus stderr. There is no typed error channel unless the CLI prints a JSON error object (for example a `--json` flag).

**Sandbox.** The Agent's shell, so the harness's permission system and sandbox apply. The CLI itself needs no sandbox.

**Portability.** Any Agent with a shell: Claude Code, Codex, Gemini CLI, Cursor, OpenHands and Goose among them. **Not reachable** from a plain chat API call without a shell tool, and not from Anthropic's hosted container unless the CLI is installed there (same install problem as A1).

**Precedents for data/ML.** `duckdb`, `gh`, Hugging Face `hf` (driven by HF's `hf-cli` Skill) [L3], `kaggle`. The Scalekit result above is the only measured CLI-versus-MCP comparison found, and it is secondary [X1].

---

## 5. Route D: Agent Skills

[P2] covers the three loading levels. New here: runtime constraints, sizes, the open standard and data precedents.

**Format and cost.** `name` ≤ 64 characters, `description` ≤ 1,024; ~100 tokens per Skill always loaded, the body under 5k tokens on trigger, bundled files free until read; "Scripts run through bash, and only their output enters context" [A3].

**A Skill is not a call mechanism.** It is instructions plus scripts. It tells the Agent to run code (A1), a CLI (C) or tools (B). So a Skill inherits the costs of whichever route its body uses.

**Runtimes differ by surface** [A3][A5][O7]:

| Surface | Network | Installs | Limits |
|---|---|---|---|
| Claude API | **none** | **none** ("Only pre-installed packages") | ≤ 20 Skills per request; upload < **30 MB** uncompressed; workspace-scoped; not ZDR |
| claude.ai | varies with settings | (not stated) | per-user only |
| Claude Code | "Full network access" | local installs only ("Global package installation discouraged") | filesystem, no upload |
| OpenAI hosted shell | off unless allowlisted | `pip` shown in examples | zip ≤ **50 MB**, ≤ **500** files, ≤ 25 MB uncompressed per file; local shell Skills by path |

**Custom Skills do not sync across surfaces** on Anthropic [A3].

**Open standard.** Anthropic released Skills as an open standard (agentskills.io). The client list includes Claude/Claude Code, ChatGPT & Codex, Gemini CLI, Cursor, GitHub Copilot/VS Code, JetBrains Junie, Goose, OpenHands, Kiro, Databricks Genie Code and Snowflake Cortex Code: 46 in all on the site's client list (2026-10-01) [S1]. LangChain Deep Agents also read the format [F9b].

**What it costs DataForgeML.**
- **On Claude Code and other local Agents** a Skill can install and import the library locally, so A1 or C works.
- **On the Claude API** a Skill cannot carry DataForgeML: polars' runtime wheel is 49.9 MB [D2], above the 30 MB cap, and runtime installs are blocked [A3][A5].
- **Security.** Anthropic warns that Skills are "like installing software" and should come only from trusted sources [A3].

---

## 6. Who already ships which route (data/ML)

| Library / project | Route(s) | What it is | Link |
|---|---|---|---|
| MotherDuck / DuckDB | B (MCP, FastMCP ≥ 3.2) | Separate server package over `duckdb`; JSON-in-text envelope; covered in [P1] | https://github.com/motherduckdb/mcp-server-motherduck |
| DuckDB | C | The `duckdb` shell, run by coding agents | https://duckdb.org/docs/stable/clients/cli/overview |
| Jupyter (Datalayer) | B wrapping A1 | MCP server that edits and runs notebook cells; **kernel state persists** between calls; BSD-3; ~1.3k stars | [L1] https://github.com/datalayer/jupyter-mcp-server |
| Anthropic code execution | A1 | Hosted container with pandas, numpy, scipy, scikit-learn, statsmodels | [A1] |
| Anthropic `xlsx` Skill | D | Powers Claude's spreadsheet features; **source-available**, not open source | [L2] https://github.com/anthropics/skills |
| Hugging Face Skills | D driving C and A1 | 25+ Skills; `hf-cli` teaches the `hf` CLI; trainers drive TRL/Unsloth; Apache-2.0; for Claude Code, Codex, Gemini CLI, Cursor | [L3] https://github.com/huggingface/skills |
| Scientific Agent Skills (K-Dense) | D driving A1 | 181 Skills incl. scikit-learn, statsmodels, polars, Dask, SHAP, PyMC, UMAP; MIT; renamed from "Claude Scientific Skills" for the open standard | [L4] https://github.com/K-Dense-AI/claude-scientific-skills |
| smolagents | A1 | `CodeAgent` imports whatever is authorised | [F6] |
| OpenAI Code Interpreter | A1 | Hosted container, 1–64 GB | [O3] |
| PandasAI | A1-like | Pastes skill signatures into a code-gen prompt; covered in [P1] | — |
| pandas, polars, scikit-learn themselves | none first-party | No first-party MCP server, CLI-for-agents or Skill found from these projects (search, not exhaustive, **unverified**) | — |

**Pattern.** Data libraries reach Agents mostly through **third parties**: a hosted sandbox that pre-installs them (A1), a Skill that documents them (D), or a separate MCP server (B). Only the database engines (DuckDB/MotherDuck) ship a first-party Agent surface, and it is an adapter outside the core package [P1].

---

## 7. Can one source produce several routes?

Yes. The building blocks exist and are documented; the chains below use only shipped features.

| From | To | Tool | What carries over | Gaps |
|---|---|---|---|---|
| Typed function + NumPy docstring | MCP tool (input **and output** schema) | FastMCP `@mcp.tool`; MCP Python SDK | Signature → `inputSchema`; NumPy/Google/Sphinx docstrings → descriptions; return annotation (dataclass, Pydantic, TypedDict) → `outputSchema` + `structuredContent`; primitives wrapped as `{"result": …}` | `ToolError` vs masked errors; `Any` params unusable [F1][P1] |
| Same function | Anthropic native tool | `@beta_tool` / `@beta_async_tool` + tool runner (Python, TS, Go, Java, C#, PHP, Ruby; beta) | "inspects the function arguments and docstring to derive the JSON schema"; exceptions → `is_error` with type and message [A6] | Provider-specific loop |
| Same function | OpenAI native tool | Agents SDK `function_tool` | `inspect` + griffe (NumPy supported) + Pydantic; strict by default [F4][P1] | Provider-specific |
| Same function | Provider-agnostic tool | PydanticAI tools; LangChain `@tool` | griffe docstrings (PydanticAI); type hints required (LangChain); `content_and_artifact` keeps an artifact **out of the model's context** [F9][F10] | Framework-bound |
| Same function | Claude Agent SDK in-process MCP | `@tool(name, desc, schema)` + `create_sdk_mcp_server` | **Does not derive from type hints**: the schema is a dict or JSON Schema; Python forwards no `structuredContent` [C3] | Manual schema |
| Same function | CLI | **Cyclopts** | Type hints → flags; **NumPy**, Google and reST docstrings → help; dataclass/Pydantic/attrs params; `Literal` choices [F11] | Output format is up to the function |
| Same function | CLI | Typer | Type hints → flags; the docstring gives command help only; parameter help needs `typer.Option(help=…)` [F12] | Duplicates descriptions |
| MCP server | CLI **+ `SKILL.md`** | FastMCP `generate-cli` (v3.0.0) | Each tool → a Cyclopts subcommand with typed flags and `--help`; objects as JSON strings; also writes a `SKILL.md` "documenting every tool's exact invocation syntax" (`--no-skill` to skip). The CLI **connects to the server on every invocation** [F3] | Needs fastmcp at runtime; per-call connection cost |
| MCP server | Code mode (A2) | FastMCP `CodeMode`; Cloudflare Code Mode; Anthropic PTC (`allowed_callers`, non-MCP-connector tools); PydanticAI `CodeModeToolset` | Schemas → typed stubs (TS or Python) [F2][X2][A2][F5] | Experimental (FastMCP, Monty); PTC vendor-specific |
| MCP server | Native tools in any framework | OpenAI Agents SDK, PydanticAI, LangChain, smolagents `ToolCollection.from_mcp` | Consumed as-is [F8][P3] | — |

**The hub.** In every chain, the MCP tool schema (generated from type hints and docstrings) is the shared artifact. From it, current tooling derives native tools, code-mode stubs, a CLI and a Skill. The route-specific parts that do **not** derive automatically:
- output shaping (truncation, handles, concise/detailed) [P1];
- the Skill's workflow prose beyond invocation syntax;
- CLI ergonomics such as file outputs and `--json`.

**Precedent for a library doing this itself.** No data/ML library found ships all four routes from one source. Stripe (in [P1]) is the nearest: the MCP server is the source, and the toolkit converts it per framework, plus Skills for coding agents.

---

## 8. Implications for the access-mode decision (considerations, not a recommendation)

1. **Where the library runs decides most of the rest.** If DataForgeML runs in the Agent's host process or on the user's machine, every route is open. If it must run in a vendor's hosted sandbox, today it cannot (Anthropic: no installs, polars absent; Skills cap 30 MB) and OpenAI needs a network allowlist [A1][A3][O4]. A2 avoids the question by keeping the library host-side.
2. **A2 code mode builds on B; it does not replace it.** Every code-mode system found calls a typed tool surface. Choosing code mode still means designing the typed tools, and then adds a sandbox and stub generation.
3. **Run shape vs. where PTC helps.** A DataForgeML Run is a mostly sequential chain with fan-out inside phases. Anthropic measured PTC as cost-neutral or slightly worse (~+8%) on sequential chains and 20–40% cheaper with fan-out and big intermediate results [A2]. Which dominates for a Run is **unmeasured**.
4. **Large artifacts force a handle design under every route** except A1 with a persistent interpreter. Profile (~394k tokens), frames, and fitted units (up to 127 MB) [P5] must stay out of context: as variables (A1), as tool results kept in code (A2), as handles (B), or as files (C, D). MCP gives no session, so B needs explicit handles with lifetimes [M1].
5. **Up-front token cost is a solved problem for B on Anthropic, but not everywhere.** Tool search is default in Claude Code and the Agent SDK [C1][C3]; OpenAI's `tool_search` needs gpt-5.4+ [P2]. On providers or proxies without it, every definition is paid on every turn. A CLI or Skill pays nothing up front, but needs a shell.
6. **Portability ranking differs by axis.**
   - Wire format: B (every provider) > D (46 listed clients) > C (agents with shells) > hosted A1/A2 (per vendor).
   - Install footprint: B and A2 need the library only on the host; A1, C and D need it wherever the code or shell runs.
7. **Errors.** B is the only route with a typed error channel (`isError` plus structured payloads) [M1][P1]. A1, C and D give the model tracebacks or stderr. A2 delivers tool errors as strings inside code. The upstream asks in [P1] §6 (error attributes, typed subclasses) matter most for B and A2.
8. **One source is feasible.** Typed functions with NumPy docstrings feed FastMCP or the MCP SDK (B), which feeds code mode (A2), `generate-cli` (C) and a generated `SKILL.md` (D). The cost is the [P1] gap list: `Any` params, maintainer-facing docstrings, missing `to_dict()`. DataForgeML's NumPy docstrings (ADR-0034) are already the input Cyclopts, griffe and FastMCP parse.
9. **Security posture differs.**
   - A1 runs model-written code next to the data; smolagents says no local sandbox "can ever be completely secure" [F6].
   - A2 sandboxes block imports and network by default.
   - B exposes only declared operations.
   - C runs under the harness's shell permissions.
   - D is "like installing software" [A3].
10. **Evaluation.** No source measured these routes on a data-preparation library. The only cross-route numbers are vendor-run (Anthropic PTC, Cloudflare, Scalekit). A small in-house eval on one Run would be the first such measurement.

---

## 9. Unverified or open

- Library versions in Anthropic's code execution container (sklearn/numpy versions are not on the page), and whether an uploaded wheel can be `pip install`ed offline there.
- Whether OpenAI Code Interpreter (not hosted shell) allows `pip install`.
- The default of `use_structured_content` in the OpenAI Agents SDK.
- Whether other Claude surfaces (Claude Code, Desktop) forward `structuredContent` the same way the Agent SDK does.
- Monty's "experimental" status, per its README.
- No first-party Agent surface from pandas, polars or scikit-learn: absence from a search, not proof.
- DataForgeML's cold-import time, which bounds the per-call cost of route C.
- The Hugging Face MCP server: not read for this note.

---

## Sources

Prior notes (repo `DEVunderdog/dataforgeml-agent`, `docs/research/`)
- P1. *What makes a Python library's API consumable by an LLM agent* (#23). `agent-consumable-api.md`
- P2. *How agents route a request to predefined capabilities* (#22). `capability-routing.md`
- P3. *Which agent frameworks fit…* (#3). `agent-frameworks.md`
- P4. *How data-analysis products get a user's data into the model's context* (#20). `data-context-channels.md`
- P5. *What DataForgeML exposes that the agent must wrap* (#2). `dataforgeml-agent-surface.md`

Anthropic (Claude platform docs and engineering, read 2026-10-01)
- A1. *Code execution tool* (versions, limits, libraries, container reuse, pricing, error codes). https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
- A2. *Programmatic tool calling* (`allowed_callers`, context rules, token-efficiency numbers, limits, timeouts). https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling
- A3. *Agent Skills* overview (levels, surfaces, runtime constraints, open-source Skills, security). https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
- A5. *Using Agent Skills with the API* (20 Skills per request, 30 MB upload). https://platform.claude.com/docs/en/build-with-claude/skills-guide
- A6. *Tool runner (SDK)* (`@beta_tool`, error wrapping). https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner
- A7. *Tool use with Claude* (pricing; tool-use system prompt tokens per model). https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
- A8. *Tool search tool* (~55k-token example, 85%, 10,000 deferred tools, thresholds, model support). https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
- A9. Anthropic Engineering, *Code execution with MCP* (2025-11-04). https://www.anthropic.com/engineering/code-execution-with-mcp
- A10. Anthropic Engineering, *Introducing advanced tool use* (2025-11-24; PTC 43,588 → 27,297 tokens; tool-use examples 72% → 90%). https://www.anthropic.com/engineering/advanced-tool-use
- A11. *MCP connector* (beta `mcp-client-2025-11-20`; tools only; public HTTP only). https://platform.claude.com/docs/en/agents-and-tools/mcp-connector

Claude Code and Agent SDK
- C1. *Connect Claude Code to tools via MCP* (output limits 10k/25k, `maxResultSizeChars`, tool search default, transports, timeouts). https://code.claude.com/docs/en/mcp
- C3. *Give Claude custom tools* (Agent SDK `@tool`, `create_sdk_mcp_server`, error handling, `structuredContent` forwarding, Python limitation, tool search default). https://code.claude.com/docs/en/agent-sdk/custom-tools
- C4. *Environment variables* (`BASH_MAX_OUTPUT_LENGTH`, Bash timeouts, `ENABLE_TOOL_SEARCH` with proxies). https://code.claude.com/docs/en/env-vars
- C2. = C4 (the bash output cap, cited as C2 in the CLI rows).

Model Context Protocol
- M1. MCP specification **2026-07-28**, *Tools* (`structuredContent` any JSON value, `outputSchema`, `resource_link`, Stateful Tools "no protocol-level session", error handling). https://modelcontextprotocol.io/specification/2026-07-28/server/tools

OpenAI
- O3. *Code Interpreter* (memory tiers, 20-minute expiry, `container_file_citation`). https://developers.openai.com/api/docs/guides/tools-code-interpreter
- O4. *Shell* (hosted vs local, no outbound network by default, allowlist, runtimes, `max_output_length`, `timeout_ms`). https://developers.openai.com/api/docs/guides/tools-shell
- O5. *Programmatic Tool Calling* (JavaScript in isolated V8, eligible tools, no state between programs). https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
- O6. *Pricing* (containers $0.03–$1.92 per 20-minute session; 5-minute minimum). https://developers.openai.com/api/docs/pricing
- O7. *Skills* (hosted and local attachment, 50 MB zip, 500 files, 25 MB per file). https://developers.openai.com/api/docs/guides/tools-skills

Frameworks
- F1. FastMCP, *Tools* (schema from type hints and docstrings, `outputSchema` from return annotations, `ToolError`, `mask_error_details`). https://gofastmcp.com/servers/tools
- F2. FastMCP, *Code Mode* transform (v3.1; meta-tools; Monty; 30 s / 100 MB / depth 1,000 / 50 calls). https://gofastmcp.com/servers/transforms/code-mode
- F3. FastMCP, *Generate CLI* (v3.0.0; Cyclopts CLI + `SKILL.md`; connects per invocation). https://gofastmcp.com/cli/generate-cli
- F4. OpenAI Agents SDK, *Tools* (`function_tool` with griffe and Pydantic; `default_tool_error_function`; hosted tools incl. `ProgrammaticToolCallingTool`, `ToolSearchTool`). https://openai.github.io/openai-agents-python/tools/
- F5. PydanticAI, *Code Mode* (`CodeModeToolset`, Monty limits, no third-party imports, state within a run). https://pydantic.dev/docs/ai/harness/code-mode/
- F6. smolagents, *Secure code execution* (`LocalPythonExecutor`, authorised imports, E2B/Modal/Docker/Blaxel). https://huggingface.co/docs/smolagents/tutorials/secure_code_execution
- F7. `pydantic/monty` repository (Rust Python-subset interpreter; <1 ms start from a pool vs ~1,500 ms sandbox service). https://github.com/pydantic/monty
- F8. OpenAI Agents SDK, *MCP* (`use_structured_content`, tool filtering, `cache_tools_list`, hosted vs local). https://openai.github.io/openai-agents-python/mcp/
- F9. LangChain, *Tools* (`@tool`, type hints required, `ToolRuntime`, `@wrap_tool_call`). https://docs.langchain.com/oss/python/langchain/tools
- F9b. LangChain, *Deep Agents: Skills* (Agent Skills standard, three levels, body < 5k tokens). https://docs.langchain.com/oss/python/deepagents/skills
- F10. LangChain reference, `BaseTool.response_format` (`content_and_artifact`; artifact "not meant to be sent to the model"). https://reference.langchain.com/python/langchain-core/tools/base/BaseTool/response_format and https://reference.langchain.com/python/langchain-core/messages/tool/ToolMessage
- F11. Cyclopts documentation (type hints; NumPy/Google/reST docstrings; dataclass/Pydantic/attrs; `Literal`). https://cyclopts.readthedocs.io/en/latest/
- F12. Typer, *Command help* (docstring → command help; parameter help via `typer.Option(help=…)`). https://typer.tiangolo.com/tutorial/commands/help/

Standards and ecosystem
- S1. Agent Skills open standard, overview and client list. https://agentskills.io (spec: https://agentskills.io/specification)
- L1. `datalayer/jupyter-mcp-server`. https://github.com/datalayer/jupyter-mcp-server
- L2. `anthropics/skills` (document Skills source-available). https://github.com/anthropics/skills
- L3. `huggingface/skills`. https://github.com/huggingface/skills
- L4. `K-Dense-AI/claude-scientific-skills` ("Scientific Agent Skills"). https://github.com/K-Dense-AI/claude-scientific-skills
- X2. Cloudflare, *Code Mode: the better way to use MCP* (2025-09-26, updated 2026-07-15). https://blog.cloudflare.com/code-mode/
- R1. Wang et al., *Executable Code Actions Elicit Better LLM Agents* (CodeAct), ICML 2024, arXiv:2402.01030. https://arxiv.org/abs/2402.01030
- X1. **Secondary**, vendor blog: Scalekit, *MCP vs CLI: Benchmarking AI Agent Cost & Reliability* (2026-03-11). https://www.scalekit.com/blog/mcp-vs-cli-use

DataForgeML
- D1. `pyproject.toml` at `origin/main` (v3.3.0): dependencies `scikit-learn>=1.9,<1.10`, `numpy>=2.4,<2.6`, `scipy>=1.17,<1.19`, `joblib`, `polars>=1.0.0`, `pandas>=2.0.0`, `chardet`, `iterative-stratification`, `diptest`, `ruff`. https://github.com/DEVunderdog/DataForgeML/blob/main/pyproject.toml
- D2. PyPI JSON API, `polars` 1.44.2 / `polars-runtime-32` 1.44.2: manylinux x86_64 wheel 49.9 MB (read 2026-10-01). https://pypi.org/pypi/polars-runtime-32/json
