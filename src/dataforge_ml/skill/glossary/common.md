# common

The terms every phase uses. Look one up with `dataforge glossary <term>`.

## Agent

Aliases:
Related: Operator, Harness, Skill
Commands:

You: the program, driven by a language model, that runs the `dataforge` commands. The library is built for Agents only. A sub-agent the Harness starts is an Agent too, and it shares the same Workspace.

## Operator

Aliases:
Related: Agent, Harness, Workspace Settings, Declared Inputs, Literal Policy
Commands: load

The person who installs the library, sets up the Harness and writes the Workspace Settings. The Operator chooses the guardrails; the Agent runs the commands. Leave the Operator's settings as they are, and ask the Operator when one stops you, such as a file that is not among the Declared Inputs.

## Harness

Aliases:
Related: Agent, Operator, Skill, Workspace Settings
Commands: install

The program that runs the Agent and gives it a shell, such as Claude Code, Codex, Gemini CLI or Cursor. It owns the shell, its permissions, resource limits and timeouts. Its permission rules, not the library, are what make the Workspace Settings hold. `dataforge install --skills` copies the Skill into its skills folder.

## Skill

Aliases:
Related: Harness, Agent
Commands: install, glossary

The `dataforge` instructions the Agent reads first: `SKILL.md`, its `references/` files, one per phase, and these `glossary/` files. `dataforge install --skills` copies it into the Harness's skills folder, stamped with the library version; after an upgrade, the first command warns until the install is run again.

## Operation

Aliases:
Related: Handle, Library Error, Dial
Commands: list-handles

One typed entry point of the library, such as `fit_unit`. The CLI runs it as the command of the same name in kebab-case (`dataforge fit-unit`), and Python imports it as a function. Both take the same arguments and return the same result. Results name Operations in snake_case, such as in `omitted` and in a Dial Pointer.

## Handle

Aliases:
Related: Workspace, Lineage, Operation
Commands: list-handles

A string that names one object the library stores for you, a kind prefix plus an id, such as `frame_7f3a9c2e`. The kinds are `frame`, `profile`, `routing`, `recipe`, `unit`, `imputer` and `c2st`. A Handle is write-once: it always names the same object, and a command that changes something writes a new Handle. Handles never expire.

## Workspace

Aliases:
Related: Handle, Workspace Settings, Lineage
Commands: list-handles, export-frame

The folder that stores every Handle's object, `.dataforge/` in the working directory by default. It also holds `config.toml`, the Workspace Settings file, and `exports/`, where `export-frame` writes. Several processes and sub-agents may share one. It is storage, not a run: nothing in it tracks progress.

## Workspace Settings

Aliases:
Related: Operator, Literal Policy, Declared Inputs, Harness
Commands:

The four guardrails the Operator sets for how the library runs: `workspace`, `literal_policy`, `declared_inputs` and `core_budget`. A flag beats an env var, which beats `.dataforge/config.toml`, which beats the default. Their flags would override the Operator, so leave them out. They are guardrails, not a boundary: only the Harness's permissions enforce them.

## Library Error

Aliases:
Related: Error Code
Commands:

The one JSON shape every failure prints: `code`, `message`, and fields such as `param`, `issues`, `allowed`, `candidates` and `hint`. Its category sets the exit code, such as `2` for `InvalidInput` or `3` for `UnknownName`; the table of errors in `SKILL.md` says what each asks. In Python it is raised as an exception of that category.

## Error Code

Aliases:
Related: Library Error
Commands:

The stable name of one kind of Library Error, such as `unknown_column` or `unit_not_trainable`, in its `code` field. Branch on the code and the exit code, never on `message`. Each command's `--help` lists the Error Codes it can return.

## Dial

Aliases:
Related: Dial Pointer, Operation
Commands: profile, split, route, c2st, evaluate-imputation

One threshold inside a command's `--thresholds` JSON, with a stated default and range, such as `missingness.severity_high` of `profile`. Dials are how you tune a step: run it, read the result, set a dial, run it again. `dataforge schema <command> thresholds` lists them. A dial you leave out keeps its default.

## Dial Pointer

Aliases:
Related: Dial
Commands: profile, describe-columns, missingness, top-associations, split, explain-routing, impute, evaluate-imputation

The link from an outcome in a result, such as a flag on a column, to the Dial that decided it: `operation`, `dial` (its path, such as `thresholds.missingness.severity_high`), `value`, and `source`, which says whether the value was the default or yours. To change the outcome, run that Operation again with the dial set.

## Pipeline Event

Aliases:
Related: Operation
Commands:

One step of a running command, such as a column profiled or a unit fitted, reported as it happens. With `--progress`, a command prints one line per Pipeline Event to stderr, so stdout keeps its one JSON value. Use it to watch a (background) command on large data.

## Table Layout

Aliases:
Related: Operation
Commands:

How a list of records prints in JSON: `{"fields": [...], "rows": [[...], ...]}`, the field names once, then each record as a row of values in that order. A record wider than about a dozen fields nests instead. In Python, the same field is a sequence of records with named attributes.

## Literal Field

Aliases:
Related: Literal Policy
Commands: value-summary, describe-target, query-frame, resolve-recipe, impute

A result field that can hold a value from a cell of the data, such as a category label, a bound, a percentile, the mode or a fill value. Values you passed in, such as sentinels, and column names are not Literal Fields. Under the Literal Policy `deny`, every Literal Field comes back `null`.

## Literal Policy

Aliases:
Related: Literal Field, Workspace Settings, Operator
Commands: value-summary, describe-target, query-frame, resolve-recipe, impute

The Operator's Workspace Setting for Literal Fields. `allow`, the default, returns real values; `deny` returns each Literal Field as `null` and lists it in `omitted` with reason `literal_policy`. Work from the counts and shares then. It is containment, not privacy: `query-frame` counts can still reveal values.

## Lineage

Aliases:
Related: Handle, Workspace
Commands: list-handles

What a Handle was made from: the Operation that made it, the arguments it was passed, and the Handles it read. `list-handles` returns it with every Handle. Read it to tell two Handles of the same kind apart, or to find your Handles again after a restart.
