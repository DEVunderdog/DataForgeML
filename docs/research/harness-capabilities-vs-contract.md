# What off-the-shelf Harnesses expose, measured against the Harness contract

Research ticket [#607](https://github.com/DEVunderdog/DataForgeML/issues/607) (part of map #568). Researched 2026-10-10.

**Question.** What do today's off-the-shelf Harnesses (Claude Code, OpenAI Codex CLI, plus one or two of Cursor and Gemini CLI) actually expose, measured against each line of the Harness contract in ADR-0099 as amended by ADR-0103, 0107, 0108 and 0109? Six facts per Harness: skill discovery, MCP server config, Code Mode env, output caps, tool fields, tool search.

**This note decides nothing.** It lists which contract assumptions hold, which look wrong, and which could not be verified. Section 5 flags items for the human. No ADR is edited.

**Harness choice.** Claude Code and Codex CLI were required. Gemini CLI was added at full depth because it is open source: its config schema and code were read. Cursor was added at documentation depth only. It is closed source, and its public docs are silent on facts 4, 5 and 6. Those cells are marked UNVERIFIED.

**Sources and method.** Primary sources only, all read on 2026-10-10:
- The MCP specification and its `schema.ts` files, at stated revisions.
- Each vendor's own docs (Markdown twins fetched directly).
- Open-source repositories, cloned at a stated commit: Codex CLI and Gemini CLI.
- The shipped Claude Code binary (v2.1.296). Its JavaScript is embedded as minified text, so strings and some functions were readable. This is a shipped artifact, not a vendor doc. Claims from it are marked "binary".

No blog posts or aggregator write-ups are used as the basis for any claim. Confidence labels:
- **VERIFIED (doc)**: stated in a vendor or spec document.
- **VERIFIED (source)**: read in repository code or the shipped binary.
- **INFERRED**: derived by combining read facts. The inference is stated.
- **UNVERIFIED**: not found in a primary source.

References such as [CC-MCP] are listed in [Sources](#sources) with URL, version and date.

---

## 1. Summary

- **Skill discovery.** No Harness finds a skill inside `site-packages`. All four need one manual step. Codex, Gemini CLI and Cursor read `~/.agents/skills/`. Claude Code reads `~/.claude/skills/` and does not document `.agents/skills`. Symlinked skill folders are followed by Claude Code and Codex. Every Harness has a bundle that ships a skill and an MCP config together.
- **`input_examples` is not in the MCP `Tool` spec.** It is absent from every revision, including the draft. It is an optional field of the Anthropic Messages API tool definition. None of the three open or inspectable Harnesses forwards it from an MCP server to the model.
- **The `_meta` prefix `dataforge_ml/` is not a valid MCP key prefix.** The spec allows only letters, digits and hyphens inside a label. An underscore is not allowed.
- **Env vars as a Harness-only setting hold for Tool Calls and break for Code Mode.** An MCP server's env block is outside the Agent's reach. In Code Mode the library runs in the process the Agent controls, so the Agent can set any env var per command.
- **Output caps are soft in Claude Code and Gemini CLI.** Over-limit output is saved to a file and the Agent can read the file. Only Codex truncates without a spill file.
- **The 10k-token Result Budget fits Claude Code and Codex, and is borderline for Gemini CLI.** Gemini CLI cuts MCP text at 40,000 characters. Claude Code also persists any text result over 50,000 characters, which ADR-0109 does not mention.
- **Model-facing tool text is cut in two places the ADRs do not mention.** Claude Code truncates each tool description at 2,048 characters. Codex strips `description` keys from any input schema over 5,000 bytes, and it drops most JSON Schema keywords entirely.
- **Tool search.** Claude Code and Codex defer MCP tools by default. Gemini CLI does not. No Harness reads a catalogue file. Claude Code reads per-tool `_meta` keys under `anthropic/`, which is the in-protocol way to say "load this one first".
- **Unknown.** Cursor facts 4 to 6. Whether Claude Code forwards `structuredContent` from external servers to the model. Hot reload of MCP env in all four.

---

## 2. The contract lines being measured

Numbered so the tables and matrix can refer to them. Quotes are exact where marked.

| # | Contract line | ADR |
|---|---|---|
| C1 | The Harness exposes the package's Skill directory (`dataforge_ml/`) to the Agent, "by copying or linking it into its own skills location". `SKILL.md` sits at the package root. | 0099, 0108 |
| C2 | "The Harness sets the Workspace once, outside the Agent's reach. That means an environment variable, or a startup argument to the MCP server." Only the library and the Harness write to it. | 0103 |
| C3 | `load` reads only input paths the Harness declares, "set up the same way as the Workspace". "The Harness mounts declared inputs read-only." | 0103 |
| C4 | "The Literal Policy is `allow \| deny`, set by the Harness outside the Agent's reach ... through an environment variable or an MCP server startup argument." `deny` is a real boundary only "when the Agent's code can't read the Declared Inputs". | 0107 |
| C5 | The Harness provides "a cap on how much Code Mode output reaches the model". | 0099 |
| C6 | The Harness provides "a Tool Calls output cap at least the largest `result_budget_tokens` in the Operation Catalogue". The default Result Budget is 10k tokens, which "matches the point where Claude Code warns ... and stays far under its 25k cutoff". | 0099, 0102, 0109 |
| C7 | Each catalogue entry "is an MCP `Tool` object". Model-facing fields: `name`, `title`, `description`, `inputSchema`, `outputSchema`, `annotations`, "plus `input_examples`". Library-only fields go in `_meta` "under the `dataforge_ml/` prefix". | 0109 |
| C8 | `description` is the docstring summary, `Use when`, `Returns`, `Errors` and `See Also`. `Parameters` go into the input schema's property descriptions. `Example` becomes `input_examples`. | 0109, 0101 |
| C9 | Hints derive from `effect`: `readOnlyHint` and `idempotentHint` true only for `read`; `destructiveHint` and `openWorldHint` always false. "A Harness can auto-approve or run reads in parallel." | 0109 |
| C10 | The catalogue is "for the Harness". The Harness uses it for tool search and deferred loading. `load_first` marks 3 to 5 entry Operations to keep non-deferred, and "the Harness has the final say". | 0109 |
| C11 | Tool Calls are delivered by a `dataforge-ml[mcp]` server generated from the Operations. | 0099 |

Not measured here: the sandbox with Python 3.11 and no network, and resource limits (CPU, memory, time). These are in ADR-0099 but outside the six facts of the ticket.

---

## 3. Findings by fact

### 3.1 Spec and API facts that frame the Harness rows

| Finding | Source | Confidence |
|---|---|---|
| MCP `Tool` (spec 2026-07-28) has exactly these members: `name`, `title?`, `icons?`, `description?`, `inputSchema`, `outputSchema?`, `annotations?`, `_meta?`. `ToolAnnotations` has `title?`, `readOnlyHint?`, `destructiveHint?`, `idempotentHint?`, `openWorldHint?`. | [MCP-TOOLS], [MCP-SCHEMA] | VERIFIED (doc, source) |
| `input_examples` and `inputExamples` do not occur in `schema.ts` for 2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, 2026-07-28 or `draft`. Not part of the MCP `Tool` spec. | [MCP-SCHEMA] (grep, 0 hits in each) | VERIFIED (source) |
| An MCP issue asking for `examples` in the standard schema is open since 2024-12-22 and not adopted. | [MCP-125] | VERIFIED (doc) |
| `inputSchema` and `outputSchema` default to JSON Schema 2020-12. JSON Schema's own `examples` keyword is therefore valid inside `inputSchema`. | [MCP-TOOLS] | VERIFIED (doc) |
| Spec: "clients MUST consider tool annotations to be untrusted unless they come from trusted servers". | [MCP-TOOLS] | VERIFIED (doc) |
| `_meta` key names: an optional prefix, then a name. A prefix is dot-separated labels ending in `/`. "Labels MUST start with a letter and end with a letter or digit; interior characters can be letters, digits, or hyphens". "Implementations SHOULD use reverse DNS notation". The same rule is in 2025-06-18. `dataforge_ml/` has an underscore inside a label. | [MCP-META] | VERIFIED (doc); the conclusion about `dataforge_ml/` is INFERRED |
| `input_examples` is an optional field of the Anthropic Messages API tool definition. Each example must validate against `input_schema`, else HTTP 400. It is not supported on server tools or client toolsets. It was introduced under the beta header `advanced-tool-use-2025-11-20`. The current define-tools page lists it as a plain optional field and shows no beta header. | [A-DEFINE], [A-ADV] | VERIFIED (doc). Whether any beta header is still required is UNVERIFIED |
| Cost of examples: about 20 to 50 tokens for a simple example, 100 to 200 for a nested one. | [A-DEFINE] | VERIFIED (doc) |

### 3.2 Fact 1: skill discovery, and bundles of skill plus MCP config

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code | Skill locations: enterprise (managed settings dir), personal `~/.claude/skills/<name>/SKILL.md`, project `.claude/skills/` (also parents up to the repo root; nested dirs load lazily), a `--add-dir` directory's `.claude/skills/`, and `<plugin>/skills/<name>/SKILL.md`. | [CC-SKILLS] | VERIFIED (doc) |
| Claude Code | "A `<skill-name>` entry in the enterprise, personal, or project location can be a symlink to a directory elsewhere on disk." | [CC-SKILLS] | VERIFIED (doc) |
| Claude Code | `.agents/skills` is not documented. The strings `.agents/skills` occur 0 times in the full docs corpus (9.4 MB). | [CC-FULL] | VERIFIED (doc, absence) |
| Claude Code | `site-packages` is never scanned. A pip-installed package is not found without a step. Minimal step: a symlink from `~/.claude/skills/<name>` to the package directory. | [CC-SKILLS] | INFERRED |
| Claude Code | A plugin bundles skills, `.mcp.json` (or inline `mcpServers`), hooks, `bin/`, LSP config and agents. Plugin MCP entries accept `command`, `args`, `env`, with `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PLUGIN_DATA}`, `${CLAUDE_PROJECT_DIR}` substitution. Plugin `settings.json` honours only `agent` and `subagentStatusLine`, so a plugin cannot set `env`. | [CC-PLUGCOMP], [CC-MCP] | VERIFIED (doc) |
| Claude Code | "A `SKILL.md` at the plugin root loads as one skill" when there is no `skills/` directory and no `skills` manifest key. A manifest is optional. | [CC-PLUGCOMP], [CC-PLUGMAN] | VERIFIED (doc) |
| Claude Code | Plugin source types: relative path, `github`, `url`, `git-subdir`, `npm`, `archive`, `command`. There is no pip source, and the names `pip` and `uv` are reserved. A `command` source runs a shell command that prints a plugin directory's absolute path. `mode: link` uses that directory in place. The user must install the plugin and accept the command (`--yes` in scripts). | [CC-MARKET], [CC-PLUGINSTALL] | VERIFIED (doc) |
| Claude Code | A session flag `--plugin-dir <path>` loads a plugin from a local directory for one session. | [CC-FULL] (CLI reference) | VERIFIED (doc) |
| Claude Code | Putting these together: a `command` source or `--plugin-dir` that points at `site-packages/dataforge_ml` could load the package root as one plugin, with its root `SKILL.md` and a `.mcp.json`. Not tested end to end. | [CC-MARKET], [CC-PLUGCOMP] | INFERRED |
| Codex CLI | Skill locations: `$CWD/.agents/skills`, `$CWD/../.agents/skills`, `$REPO_ROOT/.agents/skills`, `$HOME/.agents/skills`, `/etc/codex/skills`, and bundled system skills. "Codex supports symlinked skill folders." | [X-SKILLS] | VERIFIED (doc) |
| Codex CLI | `[[skills.config]] path / enabled` only enables or disables a skill. It does not add a location. `skills.max_context_tokens` caps the initial skill catalog (default 2% of the context window, explicit values capped at 10,000 tokens). | [X-SKILLS], [X-CONFIG] | VERIFIED (doc) |
| Codex CLI | `site-packages` is not scanned. Minimal step: a symlink into `~/.agents/skills/<name>`. | [X-SKILLS] | INFERRED |
| Codex CLI | A plugin bundles skills, MCP servers (registered or bundled config), apps and hooks. Manifest path `.codex-plugin/plugin.json`. Codex also reads `.claude-plugin/plugin.json`, `.cursor-plugin/plugin.json` and an Agent Plugins `plugin.json` at the plugin root. | [X-SKILLS], [X-SRC-PLUGINS] | VERIFIED (doc, source) |
| Codex CLI | Marketplace plugin sources in code: local path, git, npm. No pip and no `command` source. | [X-SRC-PLUGINS] (`MarketplacePluginSource`) | VERIFIED (source) |
| Codex CLI | A skill can declare an MCP dependency in `agents/openai.yaml`. Codex prompts to install it. For a stdio dependency the generated server config has `args: []` and `env: None`, so a skill cannot carry the Workspace or Literal Policy settings. | [X-SRC-DEPS] | VERIFIED (source) |
| Gemini CLI | Skill tiers, low to high: built-in, extension skills, user (`~/.gemini/skills/` or alias `~/.agents/skills/`), workspace (`.gemini/skills/` or alias `.agents/skills/`). | [G-SKILLS] | VERIFIED (doc) |
| Gemini CLI | `gemini skills install <git URL or local dir> [--scope] [--path]`, and `/skills link <path>` ("Links skills from a local directory"). `gemini extensions install <path>` makes a copy. `gemini extensions link <path>` makes a symlink. | [G-SKILLS], [G-EXT] | VERIFIED (doc) |
| Gemini CLI | An extension is a directory with `gemini-extension.json`. It bundles `mcpServers`, a context file (`GEMINI.md`), `settings` (env var names), `skills/`, `commands/`, `hooks/hooks.json`, `agents/`. `${extensionPath}` is expanded in MCP config. | [G-EXT] | VERIFIED (doc) |
| Gemini CLI | `site-packages` is not scanned. A step is needed (`skills link`, `extensions link`, or a symlink). | [G-SKILLS] | INFERRED |
| Cursor | Skill locations: project `.agents/skills/` and `.cursor/skills/`; user `~/.agents/skills/` and `~/.cursor/skills/`. It also loads `.claude/skills/`, `.codex/skills/`, `~/.claude/skills/`, `~/.codex/skills/`. Nested project dirs are scoped to their subtree. | [U-SKILLS] | VERIFIED (doc) |
| Cursor | Plugins: Agent Plugins (`plugin.json` at the root, skills and MCP servers) and Cursor Plugins (`.cursor-plugin/plugin.json`, adds rules, agents, commands, hooks). "Cursor does not expand the standard's `${PLUGIN_ROOT}` and `${PLUGIN_DATA}`"; it uses `${CURSOR_PLUGIN_ROOT}`. Skills are not imported on their own; package them in a plugin and publish through a marketplace. | [U-PLUGINS], [U-SKILLS] | VERIFIED (doc) |
| Cursor | `site-packages` is not scanned. A step is needed. | [U-SKILLS] | INFERRED |
| All | Agent Skills spec: `name` is 1 to 64 characters of lowercase alphanumerics and hyphens, with no leading, trailing or doubled hyphen, and "must match the parent directory name". A directory named `dataforge_ml` cannot match a conforming `name`. Claude Code treats `name` as optional and defaults it to the directory name. | [S-SKILLS], [CC-SKILLS] | VERIFIED (doc); the effect on `dataforge_ml` is INFERRED |
| All | Agent Plugins v1: the fixed skill location is `skills/`, and "each immediate child directory containing" `SKILL.md` is a skill. MCP config is `mcp.json` at the plugin root. A `SKILL.md` at the package root is not that layout. | [S-PLUGINS] | VERIFIED (doc); the mismatch is INFERRED |

### 3.3 Fact 2: MCP server config, and whether the Agent can change it

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code | Config sources: project `.mcp.json`, local and user scope in `~/.claude.json`, plugin `.mcp.json`, `--mcp-config`, managed config. CLI: `claude mcp add --env K=V --transport stdio <name> -- <cmd> <args...>`. Fields: `command`, `args`, `env`, `url`, `headers`, `alwaysLoad`, timeout. `${VAR}` and `${VAR:-default}` expand in `command`, `args`, `env`, `url`, `headers`. Claude Code sets `CLAUDE_PROJECT_DIR` in a stdio server's environment. | [CC-MCP] | VERIFIED (doc) |
| Claude Code | `.mcp.json` and `.claude.json` are protected files. A write to a protected path is prompted in `default` and `acceptEdits`, denied in `dontAsk`, allowed in `bypassPermissions`. Sandboxed Bash cannot write `.mcp.json`, `.claude/*` or `~/.claude.json`, and "there is no way to exempt one of these paths". | [CC-PERMMODES], [CC-SANDBOX] | VERIFIED (doc) |
| Claude Code | Servers in a project `.mcp.json` need user approval before they connect. A cloned repo cannot approve its own servers. | [CC-MCP] | VERIFIED (doc) |
| Claude Code | The Agent can therefore change MCP config from inside a session only through a permission prompt, or freely under `bypassPermissions` or with the sandbox off. | [CC-PERMMODES], [CC-SANDBOX] | INFERRED |
| Claude Code | Whether a changed `env` for an already connected stdio server applies without a reconnect or restart: not documented. `/mcp` has Reconnect. | [CC-MCP] | UNVERIFIED |
| Codex CLI | Config: `~/.codex/config.toml` and project `.codex/config.toml` (trusted projects only). `[mcp_servers.<id>]` keys: `command`, `args`, `env` (table), `env_vars` (names forwarded from Codex's env), `cwd`, `startup_timeout_sec` (10), `tool_timeout_sec` (60), `enabled`, `required`, `enabled_tools`, `disabled_tools`, `default_tools_approval_mode`, `tools.<tool>.approval_mode`, `tools.<tool>.output_token_limit`. CLI: `codex mcp add <name> --env K=V -- <cmd>`. | [X-MCP], [X-CONFIG] | VERIFIED (doc) |
| Codex CLI | Plugin-provided MCP servers: "user config doesn't set their transport command". User config controls on or off and tool policy. | [X-MCP] | VERIFIED (doc) |
| Codex CLI | Under `workspace-write`, `<writable_root>/.codex` is read-only. `~/.codex` is outside the writable roots. `codex mcp add` is an ordinary shell command, so it runs under the sandbox and approval policy. | [X-APPROVALS], [X-PERMS] | VERIFIED (doc); the conclusion is INFERRED |
| Gemini CLI | Config: `settings.json` `mcpServers.<name>` with `command`, `args`, `env` (expands `$VAR`, `${VAR}`, `%VAR%`), `cwd`, `url`, `httpUrl`, `headers`, `timeout` (600,000 ms), `trust`, `includeTools`, `excludeTools`. CLI: `gemini mcp add -e KEY=value <name> <command> [args]`. An extension carries `mcpServers` in its manifest. | [G-MCP] | VERIFIED (doc) |
| Gemini CLI | Env sanitization: names matching `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*KEY*`, `*AUTH*`, `*CREDENTIAL*` are redacted from the inherited env. Variables set explicitly in the server's `env` (or by an extension's `settings[].envVar`) pass through. | [G-MCP], [G-EXT] | VERIFIED (doc) |
| Gemini CLI | Whether the Agent can edit `mcpServers` in a session, and whether it hot reloads: not documented. The shell tool can run `gemini mcp add`. | [G-MCP] | UNVERIFIED (reload), INFERRED (shell) |
| Cursor | Config: `.cursor/mcp.json` (project), `~/.cursor/mcp.json` (global). Stdio fields: `command`, `args`, `env`, `envFile`. Interpolation in `command`, `args`, `env`, `url`, `headers`: `${env:NAME}`, `${workspaceFolder}`, `${userHome}`. | [U-MCP] | VERIFIED (doc) |
| Cursor | Whether the Agent's file edits to `mcp.json` are gated: not documented. | [U-MCP] | UNVERIFIED |

### 3.4 Fact 3: Code Mode env, and sandboxes that hide inputs

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code | "The Bash tool runs each command in a separate process." "Environment variables don't persist" between commands. Persistence is via `CLAUDE_ENV_FILE` or a SessionStart hook. | [CC-TOOLS] | VERIFIED (doc) |
| Claude Code | Settings `env` "reach every subprocess Claude Code starts". That includes Bash commands and stdio MCP servers. An `.mcp.json` `env` block reaches only that server. | [CC-SETTINGS], [CC-MCP] | VERIFIED (doc) |
| Claude Code | The Agent can set any env var for a command it writes, for example `VAR=x python ...` or `os.environ[...]` in its own Python. | [CC-TOOLS] | INFERRED (ordinary shell semantics) |
| Claude Code | The sandbox is off by default. When on: writes to the working dir and temp; reads "most of the machine"; no direct network. `sandbox.filesystem.denyRead` hides paths from sandboxed commands, and `allowRead` re-opens a narrower path. `denyRead` "doesn't stop the Read tool"; the Read tool follows permission rules. `sandbox.filesystem.denyWrite` makes a path read-only to commands. | [CC-SANDBOX] | VERIFIED (doc) |
| Claude Code | `sandbox.credentials.envVars` with `mode: deny` unsets named variables "before each sandboxed command runs". `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` strips credentials, matched by name or value, from Bash, hooks and stdio MCP servers. | [CC-SANDBOX], [CC-ENV] | VERIFIED (doc) |
| Claude Code | "Local MCP servers" and hooks "run with your full access", outside the sandbox. So a stdio MCP server can read a path that a sandboxed Bash command cannot. | [CC-SANDBOX] | VERIFIED (doc) |
| Claude Code | A library running in Code Mode sits inside the same sandboxed Bash command as the Agent's code. Any `denyRead` that hides the input from the Agent's code also hides it from the library. | [CC-SANDBOX] | INFERRED |
| Codex CLI | `shell_environment_policy`: `inherit = all \| core \| none`, `filters` (include or exclude patterns), `set` (explicit values injected after exclusions). It applies to every shell command. The Agent can still prefix its own command. | [X-CONFIG] | VERIFIED (doc); Agent override INFERRED |
| Codex CLI | Sandbox modes: `read-only`, `workspace-write`, `danger-full-access`. Permission profiles map paths to `read`, `write` or `deny`, with `deny` winning and glob deny-read rules (`"**/*.env" = "deny"`). Network is off by default in `workspace-write`. | [X-CONFIG], [X-PERMS], [X-APPROVALS] | VERIFIED (doc) |
| Codex CLI | "Local and remote MCP servers use their own process or transport", outside the permission profile. | [X-PERMS] | VERIFIED (doc) |
| Codex CLI | Codex has its own experimental "code mode" feature (`features.code_mode.*`, "under development and off by default"). It is a different thing from the ADR's Code Mode. | [X-CONFIG] | VERIFIED (doc) |
| Gemini CLI | The shell tool sets `GEMINI_CLI=1`. `security.environmentVariableRedaction` (default `false`) redacts secret-like names. | [G-SHELL], [G-CONFIG] | VERIFIED (doc) |
| Gemini CLI | Sandbox: Seatbelt profiles on macOS, Docker or Podman containers elsewhere. The current directory is mounted at the same absolute path. Seatbelt `strict-*` profiles restrict reads and writes. "Ensure MCP servers are available within the sandbox environment", so MCP servers run in the same sandbox as the shell. | [G-SANDBOX], [G-MCP] | VERIFIED (doc); "same sandbox" INFERRED |
| Gemini CLI | Whether a mount can be read-only, and any per-path read deny: not documented. | [G-SANDBOX] | UNVERIFIED |
| Cursor | Sandbox: Landlock and seccomp on Linux, Seatbelt on macOS. `sandbox.json` keys include `readBoundary` and `additionalReadPaths`. A sandboxed command reads the workspace, the Read Allowlist and system paths. Reads outside the workspace ask first. In Auto-review, allowlisted shell, Fetch and MCP calls "run outside the sandbox". | [U-RUNMODES] | VERIFIED (doc) |
| Cursor | Env vars injected into sandboxed processes: `CURSOR_SANDBOX`, `CURSOR_ORIG_UID`, `CURSOR_ORIG_GID`, `CURSOR_SANDBOX_LANDLOCK_STATUS`. Whether the Agent can set env for the library: not documented. | [U-RUNMODES] | UNVERIFIED |

### 3.5 Fact 4: output caps and warning points

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code (MCP) | Warns when an MCP result exceeds 10,000 tokens (fixed). Default limit 25,000 tokens (`MAX_MCP_OUTPUT_TOKENS`). Over the limit, the result is saved to a file under `~/.claude/projects/` and replaced in the conversation by a message that names the path, "so Claude reads the file when it needs the content". | [CC-MCP], [CC-ENV] | VERIFIED (doc) |
| Claude Code (MCP) | A successful text result over 50,000 characters is saved to a file "whatever its token count". `MAX_MCP_OUTPUT_TOKENS` does not change it. A tool can raise its own threshold with `_meta["anthropic/maxResultSizeChars"]`, up to 500,000 characters. | [CC-MCP] | VERIFIED (doc, binary) |
| Claude Code (MCP) | Error results longer than about 11,000 characters keep the first and last 5,000. Each tool description and each server's `instructions` are truncated at 2,048 characters (`CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`, v2.1.280 or later). HTTP and SSE responses over 16 MB fail. | [CC-MCP], [CC-ENV] | VERIFIED (doc) |
| Claude Code (Bash) | Valid output inline up to about 30,000 characters. Past that the Agent gets a file path and a 2,000-character preview. `BASH_MAX_OUTPUT_LENGTH` default 30,000, ceiling 150,000. `bashOutputMaxChars` setting up to 128,000. Failure output about 10,000 characters. A command whose output passes 5 GB is killed. | [CC-TOOLS], [CC-ENV] | VERIFIED (doc) |
| Codex CLI (all tools) | Each model has a `truncation_policy`. All 11 models in the bundled `models.json` set tokens, limit 10,000. A 20% serialization allowance is added. `tool_output_token_limit` overrides it globally. Tokens are estimated as bytes divided by 4. Truncation keeps the head and tail and drops the middle. | [X-SRC-MODELS], [X-SRC-TRUNC] | VERIFIED (source) |
| Codex CLI (MCP) | `mcp_servers.<id>.tools.<tool>.output_token_limit` sets a per-tool budget "before the standard 20% serialization allowance". It is user config, not server metadata. When a result has `structuredContent`, the model sees that JSON serialized and the `content` array is not used. The cut is a middle cut of the serialized text. | [X-MCP], [X-SRC-MODELS] | VERIFIED (doc, source); "can cut mid-JSON" INFERRED |
| Codex CLI (shell) | Default 10,000 output tokens per exec call (`DEFAULT_MAX_OUTPUT_TOKENS`). The effective cap is the smaller of the requested value and the model policy. Raw output buffer 1 MiB. No spill-to-file was found in the truncation code. | [X-SRC-TRUNC] | VERIFIED (source); "no spill" INFERRED from grep |
| Gemini CLI | `tools.truncateToolOutputThreshold`, default 40,000 characters. Applies to the shell tool and to an MCP result that is a single text part. Over the limit, the full output is saved to a file in the project temp dir and a truncated text with the path is returned. A separate "context management" path has `maxOutputTokens` 10,000 and an LLM-summarization threshold of 20,000 tokens. MCP request timeout 600,000 ms. | [G-CONFIG], [G-SRC-EXEC] | VERIFIED (doc, source); whether context management is on by default UNVERIFIED |
| Cursor | "Large outputs are truncated automatically" in the CLI shell mode. No number is given. No MCP result cap or warning point is documented. | [U-SHELL] | UNVERIFIED |

### 3.6 Fact 5: tool fields reaching the model

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code | Maps each MCP tool as follows (v2.1.296 binary). `description` from `description`, truncated to 2,048 characters. `inputSchema` becomes the API input schema after checks. `title` comes from **`annotations.title`**, not from top-level `title`. `readOnlyHint` sets read-only and concurrency-safe. `destructiveHint` and `openWorldHint` are read. `idempotentHint` is not read in the mapping function. `execution` is read. | [CC-BIN] | VERIFIED (binary) |
| Claude Code | `_meta` keys read: `anthropic/maxResultSizeChars`, `anthropic/requiresUserInteraction`, `anthropic/alwaysLoad`, `anthropic/searchHint`, and a `ui` entry. No `dataforge_ml/` key is read. | [CC-BIN], [CC-MCP] | VERIFIED (binary, doc) |
| Claude Code | `outputSchema` is not used in the tool-mapping function. | [CC-BIN] | VERIFIED (binary) for that function; whether it is used elsewhere is UNVERIFIED |
| Claude Code | `input_examples` and `inputExamples` occur 0 times in the binary. Claude Code does not forward them from an MCP server. | [CC-BIN] | VERIFIED (binary) |
| Claude Code | Schema checks: top-level property names must be 1 to 64 characters of ASCII letters, digits, `_`, `.`, `-`; the schema must validate against the 2020-12 meta-schema. A root-level `anyOf`, `oneOf` or `allOf` is flattened and a sentence is prepended to the description. A failing tool is dropped and the user is told. | [CC-MCP] | VERIFIED (doc) |
| Claude Code | `structuredContent`: documented as forwarded to Claude (text blocks in `content` then dropped) for in-process SDK servers. For external servers it is exposed on `tool_use_result`, and the doc says "what Claude receives doesn't change". Which of `content` or `structuredContent` Claude sees from an external server is not stated. | [CC-FULL] (Agent SDK custom tools, `tool_use_result`) | UNVERIFIED |
| Codex CLI | The model-facing function tool has `name`, `description` (full, no limit for ordinary servers) and `parameters`. `outputSchema` is kept in a field marked `serde(skip)`, so it is not serialized into the request. `title` and `_meta` are not forwarded. | [X-SRC-TOOLS] | VERIFIED (source) |
| Codex CLI | Input schemas pass through a typed struct. Only these keys survive: `$ref`, `type`, `description`, `enum`, `items`, `minItems`, `properties`, `required`, `additionalProperties`, `anyOf`, `oneOf`, `allOf`, `$defs`, `definitions`. `const` is rewritten to a one-value `enum`. All other keywords (`minimum`, `maximum`, `pattern`, `format`, `default`, `examples`, `title`, `maxItems`) are dropped. | [X-SRC-SCHEMA] | VERIFIED (source) |
| Codex CLI | Input schemas over 5,000 bytes are compacted. The first pass removes every `description`. Then definitions, deep objects and compositions are dropped. The limit is per server: `tool_input_schema_max_bytes` in config source. It is not in the public config reference. | [X-SRC-SCHEMA], [X-SRC-CONFIG] | VERIFIED (source) |
| Codex CLI | Annotations drive approval, not model text. In Auto mode a tool needs approval unless `readOnlyHint` is true, or `destructiveHint` and `openWorldHint` are both false. `idempotentHint` is not read. Parallel tool calls follow a per-server `supports_parallel_tool_calls` setting, not an annotation. | [X-SRC-APPROVAL], [X-SRC-CONFIG] | VERIFIED (source) |
| Codex CLI | `input_examples` and `inputExamples` occur 0 times in `codex-rs`. Generic `_meta` is not read. The `_meta` keys read are specific `openai/*` and `codex/*` keys for Apps. | [X-SRC-TOOLS] | VERIFIED (source) |
| Gemini CLI | The model sees `mcp_<server>_<tool>`, `description` and the input schema. The schema is sanitized: `$schema` and `additionalProperties` are removed, and a `default` inside `anyOf` is removed. `annotations.readOnlyHint` sets `isReadOnly`, and the annotations are kept for the policy engine. Top-level `title`, `outputSchema` and `_meta` are not used. | [G-MCP], [G-SRC-MCP] | VERIFIED (doc, source) |
| Gemini CLI | `input_examples` and `inputExamples` occur 0 times in `packages/`. | [G-SRC-MCP] | VERIFIED (source) |
| Cursor | Which MCP `Tool` fields reach the model: not documented. (Release notes describe `annotations` and `outputSchema` for the Cursor SDK's own custom tools. That is not an external MCP server.) | [U-RELNOTES] | UNVERIFIED |

### 3.7 Fact 6: tool search, deferral, and custom catalogue files

| Harness | Finding | Source | Confidence |
|---|---|---|---|
| Claude Code | Tool search is on by default. "Only tool names and server instructions load at session start." `ENABLE_TOOL_SEARCH`: unset (all deferred), `true`, `auto` or `auto:N` (defer once definitions reach N% of context, default 10%), `false`. Falls back to upfront loading when `ANTHROPIC_BASE_URL` is a non-first-party host, on Foundry on Azure, and on some older models on Google Cloud. | [CC-MCP] | VERIFIED (doc) |
| Claude Code | Per-server `alwaysLoad` in config, and per-tool `_meta["anthropic/alwaysLoad"]` (`true` loads upfront, and the user's server-level setting can override it). `_meta["anthropic/searchHint"]` is a one-line phrase for the deferred-tool list. | [CC-MCP], [CC-BIN], [CC-FULL] | VERIFIED (doc, binary) |
| Claude Code | No custom catalogue file is read. The tool source is the server's `tools/list`. Server `instructions` are the documented place for "when Claude should search for your tools". | [CC-MCP] | VERIFIED (doc, absence) |
| Codex CLI | MCP tools are deferred and searched when the model has `supports_search_tool`. All 11 bundled models set it true. The `tool_search` feature flag is removed ("always enabled"). Search text is the tool name, the name with underscores as spaces, the description, and parameter names and descriptions. | [X-SRC-EXPOSURE], [X-SRC-MODELS], [X-SRC-SEARCH] | VERIFIED (source) |
| Codex CLI | No per-tool "always load" was found in config or in code. Per-server `omit_tools_from` and `enabled_tools` filter. Server `instructions` are read, and "keep the first 512 characters self-contained". | [X-MCP], [X-SRC-CONFIG] | VERIFIED (doc, source); absence INFERRED |
| Codex CLI | No catalogue file is read. | [X-SRC-EXPOSURE] | VERIFIED (source, absence) |
| Gemini CLI | No tool search or deferral in docs or code (searches for `tool search`, `deferred`, `defer_loading`, `lazy` found nothing relevant). All discovered MCP tools are registered. `includeTools` and `excludeTools` filter. | [G-SRC-MCP], [G-MCP] | VERIFIED (doc, source); absence INFERRED |
| Cursor | Deferral of MCP tools: not documented. Release notes say cloud agents "cap the MCP tools they send" when a model's tool limit is exceeded. | [U-RELNOTES] | UNVERIFIED |

---

## 4. Contract line vs reality

Legend. **holds**: supported as stated by documented means. **partially**: works, with a stated caveat. **breaks**: documented or read behaviour contradicts the line. **unknown**: not documented. "TC" is Tool Calls (MCP). "CM" is Code Mode.

| Line | Claude Code | Codex CLI | Gemini CLI | Cursor |
|---|---|---|---|---|
| **C1** Skill dir exposed by copy or link | partially: symlink works; no site-packages discovery; a `command`-source plugin or `--plugin-dir` could load the root `SKILL.md` | partially: symlink into `~/.agents/skills`; no pip or command plugin source | partially: `gemini skills link` or `extensions link`; one manual step | partially: user skill dirs; one manual step |
| **C2** Workspace set outside the Agent's reach | TC holds (server `env`, config write gated). CM breaks: the env var lives in the Agent's own process | TC holds (server `env` in `config.toml`, `.codex` read-only). CM breaks (`shell_environment_policy.set` is overridable per command) | TC partially: server and shell share one sandbox when sandboxed. CM breaks | TC unknown (edit gating not documented). CM unknown |
| **C3** Declared inputs read-only | partially: `sandbox.filesystem.denyWrite` exists; sandbox is off by default | holds: profile `read` entries | unknown: read-only mount not documented | partially: `additionalReadPaths` is read access; workspace is writable |
| **C4** Literal Policy set by the Harness; `deny` real only if Agent code can't read inputs | TC partially: needs `denyRead` plus a `Read(...)` deny; MCP server is outside the sandbox. CM breaks: library and Agent code share the sandbox | TC partially: profile `deny` hides inputs; MCP server outside the profile. CM breaks | TC unknown: MCP server inside the sandbox. CM breaks | unknown |
| **C5** Cap on Code Mode output | partially: about 30,000 characters inline, the rest saved to a file the Agent can read | holds: 10,000 tokens per call, middle cut, no spill file found | partially: 40,000 characters, rest saved to a file | unknown: truncation unspecified |
| **C6** TC cap at least 10k tokens | holds: warn at 10k, cut at 25k tokens; but any text over 50,000 characters is persisted to file | partially: default equals 10k tokens, plus 20% allowance; no margin; per-tool override is user config | partially: 40,000 characters cut; a 10k-token JSON result is near that; applies to single-text results only | unknown |
| **C7** MCP `Tool` plus `input_examples` plus `_meta` `dataforge_ml/` | breaks: `input_examples` not in spec and not forwarded; `dataforge_ml/` invalid prefix; only `anthropic/*` keys read | breaks: same; `_meta` and `title` not forwarded | breaks: same; `_meta`, `title`, `outputSchema` unused | unknown |
| **C8** `description` carries the 5 parts; `Parameters` in property descriptions | partially: description cut at 2,048 characters | partially: description full; property descriptions stripped over 5,000 bytes; most keywords dropped | holds: no limit found | unknown |
| **C9** Hints from `effect` | partially: `readOnlyHint` drives read-only and parallel; `idempotentHint` unread | partially: `readOnlyHint` and the two false hints skip approval; parallel needs server config | partially: `readOnlyHint` read; others stored for policy | unknown |
| **C10** Harness uses catalogue for search and deferral; `load_first` | partially: defers by default; per-tool `anthropic/alwaysLoad` is the carrier; catalogue file not read | breaks: defers by default; no per-tool always-load; catalogue file not read | breaks: no deferral at all | unknown |
| **C11** `[mcp]` stdio server | holds | holds | holds | holds |

---

## 5. Surprises and consequences for the ADRs

Flags for the human. None of these is a decision.

1. **`input_examples` is not an MCP field.** ADR-0109 says each catalogue entry "is an MCP `Tool` object" and carries `input_examples`. The field is absent from every spec revision and from the draft. It exists on the Anthropic Messages API tool definition. Claude Code v2.1.296, Codex `806d973` and Gemini CLI `9b6e026` do not read it from an MCP server. So over MCP it reaches no off-the-shelf Harness. The in-spec carrier is the JSON Schema `examples` keyword in `inputSchema`. Codex drops `examples` before the model sees it (see item 4).
2. **`dataforge_ml/` is not a valid `_meta` prefix.** The spec rule forbids an underscore inside a label and recommends reverse DNS. This applies to ADR-0109's "library-only fields in `_meta`". A strict client may reject or ignore such keys. No Harness read here uses them.
3. **Description truncation in Claude Code.** ADR-0109 builds `description` from five docstring parts, and ADR-0108 says docstrings are two to three times longer. Claude Code cuts each tool description at 2,048 characters and says to "put critical details near the start". The parts at the end (`Errors`, `See Also`) are cut first. Whether real descriptions exceed 2,048 characters is INFERRED, not measured.
4. **Codex drops most schema keywords.** ADR-0101's strict subset and its "appends default and range" survive in Codex only as text in `description`. `minimum`, `maximum`, `default`, `pattern`, `format`, `examples`, `maxItems` and `title` are dropped. Over 5,000 bytes, every property `description` is stripped, which is where ADR-0101 puts `Parameters`. The limit is per server and is a source-only setting.
5. **The env-var boundary holds only over MCP.** ADR-0103 and ADR-0107 say the Harness sets Workspace and Literal Policy outside the Agent's reach through an env var or a startup argument. For an MCP server this is true on Claude Code and Codex, because the server env block is separate from Bash and config writes are gated. In Code Mode the library runs in the Agent's process, so any env var is Agent-settable per command, and the Workspace directory is writable by the same sandbox. ADR-0107 states the `deny` caveat for reading inputs. It does not state the env-var caveat for Code Mode, and ADR-0103's "written only by the library and the Harness" has no mechanism in Code Mode on these Harnesses.
6. **Caps are soft in two Harnesses.** ADR-0099 asks for "a cap on how much Code Mode output reaches the model". Claude Code and Gemini CLI save the full output to a file and return the path, so the Agent can read it in pieces. Only Codex truncates with no spill file found. The 394k-token profile print that ADR-0099 worries about is bounded inline, not unreachable.
7. **Result Budget margins.** ADR-0109 compares 10k tokens to Claude Code's 10k warning and 25k cutoff. It does not mention Claude Code's separate 50,000-character persist threshold, Gemini CLI's 40,000-character cut, or that Codex's default is also 10,000 tokens (plus 20%) with a bytes-divided-by-4 estimator. The library counts `o200k_base` tokens. Whether a 10k-token JSON result stays under each threshold is INFERRED from typical bytes per token, not measured.
8. **Gemini CLI has no deferral, and its sandbox contains the MCP server.** With about 19 Operations (ADR-0110) all definitions load upfront there. Under its sandbox the MCP server shares the container with the shell, so ADR-0107's "Tool Calls give a real `deny` boundary" does not hold in that configuration.
9. **No Harness reads the catalogue.** ADR-0109's `_harness/catalogue.json`, `load_first` and `definition_tokens` have no consumer among the four. What exists in-protocol: Claude Code's per-tool `_meta` keys under `anthropic/` (`alwaysLoad`, `searchHint`, `maxResultSizeChars`, `requiresUserInteraction`), server `instructions` (cut at 2,048 characters in Claude Code, first 512 matter in Codex), and Codex's user-side per-tool `output_token_limit`.
10. **Skill layout and naming.** `SKILL.md` at the package root with a directory called `dataforge_ml` does not fit the Agent Skills `name` rules (hyphens only, must match the parent directory) or the Agent Plugins v1 `skills/<name>/` convention. It does fit Claude Code's "single skill at the plugin root". ADR-0099 says "no library CLI", and every Harness needs one manual step, so the install helper's form is open. ADR-0108 mentions only an `importlib.resources` lookup.
11. **One cross-harness location.** `~/.agents/skills/` is read by Codex, Gemini CLI and Cursor. Claude Code does not document it. Cursor also reads `~/.claude/skills/` and `~/.codex/skills/`. Codex also reads `.claude-plugin/plugin.json`, `.cursor-plugin/plugin.json` and Agent Plugins `plugin.json`.
12. **Title.** Claude Code uses `annotations.title` for display, not top-level `title`. Codex and Gemini CLI use neither.
13. **`outputSchema` is not model-facing in these Harnesses.** Codex does not serialize it into the request. It was not found in Claude Code's tool-mapping function or in Gemini CLI's MCP code. ADR-0107 marks Literal Fields with an annotation in the output schema. Through these Harnesses that annotation is therefore machine-readable only to a Harness-side filter, not to the model. This is INFERRED from the code read.
14. **Consistent with the ADRs.** Codex Auto mode does not prompt for a tool with `destructiveHint: false` and `openWorldHint: false`, so `create` Operations run without a per-tool prompt there. `readOnlyHint` is honoured by Claude Code, Codex and Gemini CLI.

---

## 6. Open items

| Item | Why open | What would verify it |
|---|---|---|
| Cursor facts 4, 5, 6 (MCP result cap, tool fields, deferral) | Closed source; docs and release notes silent | A probe MCP server run inside Cursor with a request-capture proxy, or inspection of the Cursor app bundle |
| Whether Claude Code sends `structuredContent` or `content` text to the model for an external server | The SDK doc covers only in-process servers | Probe server returning different `content` and `structuredContent`, with a capture proxy. Note that a non-first-party `ANTHROPIC_BASE_URL` turns tool search off unless `ENABLE_TOOL_SEARCH` is set |
| Real description lengths against the 2,048-character cut | ADRs have no entries yet | Generate one catalogue entry per Operation and count characters, once `catalogue.json` exists |
| Real result sizes against 50,000 characters (Claude Code), 40,000 characters (Gemini CLI), 12,000-token estimate (Codex) | Budget measured in `o200k_base` only | Measure bytes of the largest budgeted result on the House Prices fixture |
| Whether a changed MCP `env` applies mid-session in each Harness | Not documented | Change `env` on a live stdio server and observe the server environment, per Harness |
| Whether a `command`-source plugin or `--plugin-dir` over `site-packages/dataforge_ml` loads a root `SKILL.md` plus `.mcp.json` | Composition of two documented behaviours, untested | Build a stub package and run `claude --plugin-dir` against it |
| Codex skill-root resolution in code | The root list came from docs; the loader crate was not located | Read the skills extension crate in `openai/codex` |
| Gemini CLI: read-only mounts, which schema keywords survive to the model, and hot reload of `mcpServers` | Docs silent | Read `packages/core` function-declaration conversion; run a container-sandbox probe |
| Gemini CLI: whether context management (distillation) is on by default | Setting listed, default not read | Read the `isContextManagementEnabled` default in `packages/core/src/config` |
| Claude Code `outputSchema` use outside the tool-mapping function | Binary is minified; one function read | Search the binary for the other `outputSchema` call sites, or run a probe |
| Whether the Messages API still needs a beta header for `input_examples` | The define-tools page shows none; the 2025-11-24 post shows `advanced-tool-use-2025-11-20` | Send one request without the header |
| Model-side effect of examples, descriptions and deferral on Operation-calling accuracy | Out of scope here | Ticket #582 |

---

## Sources

All read on 2026-10-10 unless stated.

**MCP specification and Agent standards**
- [MCP-TOOLS] MCP specification 2026-07-28, *Tools*. https://modelcontextprotocol.io/specification/2026-07-28/server/tools
- [MCP-SCHEMA] `schema.ts` per revision in `modelcontextprotocol/modelcontextprotocol` (`main` at `c518f7a`, 2026-10-08). 2026-07-28 file last changed at `271ecc9` (2026-07-28). https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema . Revisions checked for `input_examples`: 2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, 2026-07-28, draft.
- [MCP-META] `_meta` key rules, *Base protocol* 2026-07-28 and 2025-06-18. https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/basic/index.mdx and https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2025-06-18/basic/index.mdx
- [MCP-125] Issue "Add `examples` or similar to the standard schema" (open). https://github.com/modelcontextprotocol/modelcontextprotocol/issues/125
- [S-SKILLS] Agent Skills specification. https://agentskills.io/specification
- [S-PLUGINS] Agent Plugins specification 1.0.0 (repo pushed 2026-09-28). https://github.com/agentplugins/agent-plugins-spec/blob/main/spec/1.0.0.md

**Anthropic API**
- [A-DEFINE] *Define tools*, `input_examples` and the tool parameter table. https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools
- [A-ADV] Anthropic Engineering, *Introducing advanced tool use* (2025-11-24; beta header `advanced-tool-use-2025-11-20`). https://www.anthropic.com/engineering/advanced-tool-use

**Claude Code** (docs at `code.claude.com/docs/en/<page>.md`; binary `@anthropic-ai/claude-code-linux-x64` 2.1.296, latest on npm)
- [CC-MCP] https://code.claude.com/docs/en/mcp.md
- [CC-SKILLS] https://code.claude.com/docs/en/skills.md
- [CC-PLUGCOMP] https://code.claude.com/docs/en/plugins/components.md
- [CC-PLUGMAN] https://code.claude.com/docs/en/plugins/manifest-reference.md
- [CC-MARKET] https://code.claude.com/docs/en/plugins/marketplace-reference.md
- [CC-PLUGINSTALL] https://code.claude.com/docs/en/plugins/install.md
- [CC-TOOLS] https://code.claude.com/docs/en/tools-reference.md
- [CC-ENV] https://code.claude.com/docs/en/env-vars.md
- [CC-SETTINGS] https://code.claude.com/docs/en/settings-reference.md
- [CC-SANDBOX] https://code.claude.com/docs/en/sandboxing.md
- [CC-PERMMODES] https://code.claude.com/docs/en/permission-modes.md
- [CC-FULL] https://code.claude.com/docs/llms-full.txt (9.4 MB; CLI reference, Agent SDK custom tools, Agent SDK `tool_use_result`)
- [CC-BIN] Strings and the MCP tool-mapping function read from the shipped binary, v2.1.296: https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-2.1.296.tgz . Minified JS; not a vendor doc.

**OpenAI Codex CLI** (repo `openai/codex` at `806d9732c974`, 2026-10-10; latest release `rust-v0.162.1`, 2026-10-09)
- [X-SKILLS] https://learn.chatgpt.com/docs/build-skills.md
- [X-MCP] https://learn.chatgpt.com/docs/extend/mcp.md
- [X-CONFIG] https://learn.chatgpt.com/docs/config-file/config-reference.md
- [X-PERMS] https://learn.chatgpt.com/docs/permissions.md
- [X-APPROVALS] https://learn.chatgpt.com/docs/agent-approvals-security.md
- [X-SRC-MODELS] `codex-rs/models-manager/models.json`, `codex-rs/models-manager/src/model_info.rs`
- [X-SRC-TRUNC] `codex-rs/utils/output-truncation/src/lib.rs`, `codex-rs/utils/string/src/truncate.rs`, `codex-rs/core/src/tools/context.rs`, `codex-rs/core/src/unified_exec/mod.rs`
- [X-SRC-TOOLS] `codex-rs/tools/src/mcp_tool.rs`, `codex-rs/tools/src/responses_api.rs`, `codex-rs/protocol/src/models.rs` (`CallToolResult`)
- [X-SRC-SCHEMA] `codex-rs/tools/src/json_schema.rs`, `json_schema/types.rs`, `json_schema/compaction.rs`
- [X-SRC-CONFIG] `codex-rs/config/src/mcp_types.rs` (`McpServerConfig`)
- [X-SRC-APPROVAL] `codex-rs/core/src/mcp_tool_call.rs` (`requires_mcp_tool_approval`)
- [X-SRC-EXPOSURE] `codex-rs/core/src/mcp_tool_exposure.rs`, `codex-rs/core/src/tools/spec_plan.rs`
- [X-SRC-SEARCH] `codex-rs/tools/src/tool_search.rs`
- [X-SRC-PLUGINS] `codex-rs/core-plugins/src/marketplace.rs`, `codex-rs/core-plugins/src/manifest.rs`, `codex-rs/utils/plugins/src/plugin_namespace.rs`, `codex-rs/exec-server-protocol/src/protocol.rs`
- [X-SRC-DEPS] `codex-rs/core/src/mcp_skill_dependencies.rs`

**Gemini CLI** (repo `google-gemini/gemini-cli` at `9b6e0265d16b`, 2026-10-09; `package.json` 0.65.0-nightly; latest release v0.63.0, 2026-10-06)
- [G-SKILLS] `docs/cli/skills.md`
- [G-EXT] `docs/extensions/reference.md`
- [G-MCP] `docs/tools/mcp-server.md`
- [G-CONFIG] `docs/reference/configuration.md`
- [G-SANDBOX] `docs/cli/sandbox.md`
- [G-SHELL] `docs/tools/shell.md`
- [G-SRC-MCP] `packages/core/src/tools/mcp-client.ts`, `mcp-tool.ts`, `mcp-compliance-transport.ts`
- [G-SRC-EXEC] `packages/core/src/scheduler/tool-executor.ts`, `packages/core/src/config/config.ts`

**Cursor** (docs only; release notes 3.23)
- [U-SKILLS] https://cursor.com/docs/skills.md
- [U-MCP] https://cursor.com/docs/mcp.md
- [U-PLUGINS] https://cursor.com/docs/plugins.md
- [U-RUNMODES] https://cursor.com/docs/agent/security/run-modes.md
- [U-SHELL] https://cursor.com/docs/cli/shell-mode.md
- [U-RELNOTES] https://cursor.com/docs/release-notes.md
