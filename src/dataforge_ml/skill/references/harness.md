# The Harness contract

The Harness is the program that runs the Agent and gives it a shell, such as Claude Code, Codex, Gemini CLI or Cursor. The Operator is the person who installs the library, sets up the Harness and writes the Workspace Settings.

## What the Harness provides

- A shell with Python 3.11 or later.
- The library installed, with `dataforge` on the PATH.
- This Skill installed: the package's `skill/` folder, `SKILL.md` and `references/`, copied into the Harness's skills folder under the name `dataforge`. `dataforge install --skills` copies it into `.claude/skills/dataforge/` and `.agents/skills/dataforge/` under the working directory, or under the home folder with `--global`. The copy is stamped with the library version; after an upgrade, the first command warns once on stderr until the install is run again.

Resource limits (CPU, memory, disk, shell timeouts) stay the Harness's. The library assumes no sandbox, no network limit and no read-only mount.

## Workspace Settings are guardrails, not a boundary

The four Workspace Settings (`workspace`, `literal_policy`, `declared_inputs`, `core_budget`) resolve in one order: a flag beats an env var, which beats `.dataforge/config.toml`, which beats the default. In a shell the Agent can type any flag, set any env var and edit any file, so it can change every setting. The library does not enforce them. They hold for an Agent that leaves them alone, and for a Harness that restricts the shell.

Real enforcement is the Harness's own permission rules:

- Limit the Agent's shell to `dataforge` commands, as the Skill's `allowed-tools: Bash(dataforge:*)` asks.
- Deny reading the raw CSV files and the `.dataforge/` folder.
- Deny the settings flags (`--workspace`, `--literal-policy`, `--declared-inputs`, `--core-budget`).

## What Literal Policy `deny` hides, and what it does not

`deny` nulls every field that holds a cell value and lists it in `omitted`. It is a real boundary only where the Harness permission rules above hold. Without them, the Agent can read the CSV directly or set `--literal-policy allow`.

Even with them, `query-frame` still leaks values through counts. Its `--filter` takes literal values and `matched_rows` is an exact count, so an Agent can binary-search a withheld value one query at a time. Means of tiny groups leak the same way. This is containment, not privacy. A Harness that needs a wall denies `dataforge query-frame` in its permission rules.

## Declared Inputs are opt-in

With no Declared Inputs, `load` reads any path and gives no warning. Once the Operator declares any, `load` refuses every other path with `input_not_declared`.
