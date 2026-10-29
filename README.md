# DataForgeML

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DEVunderdog/DataForgeML)

DataForgeML prepares a tabular dataset for modelling. It profiles the columns, splits the rows,
fills missing values, and checks whether the fill kept the data's distribution.

It is built for **AI agents**, not for people writing Python. An agent such as Claude Code,
Codex, Gemini CLI or Cursor drives it through the `dataforge` command:

- Every command prints one JSON value, on success and on failure.
- Large objects (frames, profiles, fitted imputers) never pass through the model. They stay
  on disk, and the agent passes short **Handles** such as `frame_7f3a9c2e`.
- Every result fits a fixed budget (10k tokens, 25,000 characters). Detail comes from narrower
  commands, not from huge outputs.
- Every failure has a stable error code, an exit code and a hint on what to do next.
- A packaged **Skill** teaches the agent the workflow, so you don't write a prompt for it.

You set it up once. After that, you ask your agent for things like *"profile `train.csv` and
fill its missing values"* and it runs the commands itself.

## Contents

- [Requirements](#requirements)
- [Setup for Claude Code](#setup-for-claude-code)
- [Setup for other agents](#setup-for-other-agents)
- [Workspace Settings](#workspace-settings)
- [Locking it down](#locking-it-down)
- [What the agent can do](#what-the-agent-can-do)
- [The workflow](#the-workflow)
- [Results and errors](#results-and-errors)
- [Using it from Python](#using-it-from-python)
- [Upgrading](#upgrading)
- [Troubleshooting](#troubleshooting)
- [Developing DataForgeML](#developing-dataforgeml)

## Requirements

- Python 3.11 or later.
- An agent with a shell: Claude Code, Codex, Gemini CLI or Cursor.
- Input data as a CSV file.

## Setup for Claude Code

**1. Install the `dataforge` command.** It must be on the `PATH` of the shell the agent uses.
The simplest way is a [uv](https://docs.astral.sh/uv/) tool install, which puts the command
on your `PATH` in its own environment:

```bash
uv tool install dataforge-ml
```

`pipx install dataforge-ml` works the same way. Plain `pip install dataforge-ml` into an
active virtual environment also works, as long as Claude Code starts with that environment
active.

To install from a clone of this repository instead:

```bash
uv tool install /path/to/DataForgeML
```

Check it:

```bash
dataforge --help
```

**2. Install the Skill into your project.** Run this in the folder where you run Claude Code:

```bash
cd my-project
dataforge install --skills
```

It copies the Skill into `.claude/skills/dataforge/` (read by Claude Code) and
`.agents/skills/dataforge/` (read by Codex, Gemini CLI and Cursor). Each copy is stamped with
the library version. To install it once for every project, use `--global`, which copies it
under your home folder instead.

**3. Allow the command.** The Skill asks for `Bash(dataforge:*)`, so the agent can run
`dataforge` commands without a permission prompt each time. To make that a project rule, add
it to `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["Bash(dataforge:*)"]
  }
}
```

**4. Ask for what you want.** Start Claude Code in the project and describe the task:

> Profile `data/train.csv`, split it into 5 folds, fill the missing values on each fold, and
> tell me how well the fill kept the distribution.

Claude Code loads the Skill when the task matches it. You can also call it directly with
`/dataforge`.

## Setup for other agents

Codex, Gemini CLI and Cursor read `.agents/skills/`, which `dataforge install --skills`
fills too. Steps 1 and 2 above are the same. Use your agent's own permission settings in
place of step 3.

For an agent with no Skill support, give it the content of
`.agents/skills/dataforge/SKILL.md` as instructions. Every command also documents itself
through `--help`, so a capable agent can learn the rest from there.

## Workspace Settings

The library keeps all its state in the **Workspace**, a folder named `.dataforge/` in the
working directory. You, the Operator, control four settings:

| Setting | What it does | Default |
|---|---|---|
| `workspace` | Where Handles are stored. | `.dataforge` |
| `literal_policy` | `allow` returns cell values (top values, labels, bounds). `deny` returns `null` in their place and lists them in `omitted`, so the agent works from counts and shares. | `allow` |
| `declared_inputs` | When set, `load` reads only these paths and refuses all others. | none: any path |
| `core_budget` | The most CPU cores one command uses. | all available cores |

Each setting is resolved in this order, first match wins:

1. A command-line flag: `--workspace`, `--literal-policy`, `--declared-inputs`, `--core-budget`.
2. An environment variable: `DATAFORGE_ML_WORKSPACE`, `DATAFORGE_ML_LITERAL_POLICY`,
   `DATAFORGE_ML_DECLARED_INPUTS` (paths separated by `:` on Linux and macOS, `;` on
   Windows), `DATAFORGE_ML_CORE_BUDGET`.
3. The file `.dataforge/config.toml`.
4. The default.

An example `.dataforge/config.toml`:

```toml
literal_policy = "deny"
declared_inputs = ["data/train.csv", "data/test.csv"]
core_budget = 4
```

Relative paths are resolved from the working directory. A bad value in any source fails with
the error code `invalid_setting`, which names the setting and where it came from. It never
falls back silently.

## Locking it down

The settings are **guardrails, not a security boundary.** An agent with a free shell can pass
any flag, set any variable, edit any file, or just `cat` the CSV. The settings hold for an
agent that follows the Skill, which tells it to leave them alone.

If the data must stay hidden from the model, enforce that with your agent's permission rules.
In Claude Code, `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["Bash(dataforge:*)"],
    "deny": [
      "Read(./data/**)",
      "Read(./.dataforge/**)",
      "Bash(dataforge * --workspace*)",
      "Bash(dataforge * --literal-policy*)",
      "Bash(dataforge * --declared-inputs*)",
      "Bash(dataforge * --core-budget*)",
      "Bash(dataforge query-frame:*)"
    ]
  }
}
```

Set `literal_policy = "deny"` in `.dataforge/config.toml` as well. Then:

- The agent can only run `dataforge`, and can't read the raw CSVs or the Workspace files.
- It can't override your settings with flags.
- It can't use `query-frame`. With `deny`, this command still leaks values through exact
  counts, because a filter on a value plus a row count can find that value one query at a
  time. Deny it whenever you need a real wall.

Keep in mind this is containment, not privacy. Column names, counts and statistics still
reach the model. Resource limits (CPU, memory, disk, timeouts) stay with your agent's
environment. Full details are in the Skill's
[Harness contract](src/dataforge_ml/skill/references/harness.md).

## What the agent can do

Twenty commands, in five phases. `dataforge --help` prints this list.

| Phase | Command | What it does |
|---|---|---|
| load | `load` | Read a CSV into a frame Handle and report its shape. |
| profile | `profile` | Profile a frame once and return its Dataset Digest: column types, missingness, skew, outliers, flags. |
| | `describe-columns` | Full column profiles, flagged columns first. |
| | `describe-target` | One declared target's problem type and statistics. |
| | `missingness` | How missing values spread across rows, and which columns go missing together. |
| | `top-associations` | The most strongly associated column pairs. |
| | `value-summary` | One column's top values, rare labels, bounds and percentiles. |
| split | `split` | Split into a test frame and folds: random, time, group, k-fold, repeated k-fold, holdout-CV, or profile-stratified. |
| impute | `route` | Pick an imputation strategy per column. |
| | `explain-routing` | Show the evidence behind each column's strategy. |
| | `resolve-recipe` | Turn a routing into a recipe that lists the units to fit. |
| | `fit-unit` | Train one unit of a recipe on a frame. |
| | `compose-imputer` | Join a recipe's fitted units into one imputer. |
| | `impute` | Fill a frame's missing values with an imputer. |
| evaluate | `evaluate-imputation` | Score whether filled values look like real ones, with a Classifier Two-Sample Test (C2ST). |
| | `c2st` | Test whether any two frames come from the same distribution. |
| | `describe-c2st-columns` | Full per-column C2ST results. |
| workspace | `list-handles` | List the Workspace's Handles, newest first, with how each was made. |
| | `query-frame` | Read raw cells, or per-group measures, from any frame. |
| | `export-frame` | Write a frame as a CSV file in the Workspace. |

Three more commands help the agent learn:

- `dataforge <command> --help`: every flag with its default and range, the result fields, the
  error codes and an example.
- `dataforge schema <command> <argument>`: opens a nested JSON argument one level at a time.
- `dataforge glossary [term]`: what a term such as *signal score* or *Handle* means.

## The workflow

Each step reads the Handle before it and writes a new one. Handles are write-once: nothing is
overwritten, so every earlier result stays readable.

```
load ─▶ frame ─▶ profile ─▶ split ─▶ folds (train / validation) and test
                    │
                    └─▶ route ─▶ resolve-recipe ─▶ fit-unit (per unit, on a train frame)
                                                        │
                                  compose-imputer ◀─────┘
                                        │
                        impute (any frame) ─▶ filled frame ─▶ evaluate-imputation
```

This is what the agent runs. You can run the same commands yourself to try it out:

```bash
dataforge load --path train.csv
# {"frame": "frame_a2e1bcf5", "n_rows": 1460, ...}

dataforge profile --frame frame_a2e1bcf5
# {"profile": "profile_6f8602e5", "columns": {...}, ...}

dataforge split --profile profile_6f8602e5 --scheme '{"kind": "kfold", "k": 3}'
# {"folds": {"fields": ["index", "repeat", "train", "validation", ...], "rows": [...]}, ...}

dataforge route --profile profile_6f8602e5
# {"routing": "routing_0090fb89", ...}

dataforge resolve-recipe --routing routing_0090fb89
# {"recipe": "recipe_1d2b84bd", "units": {"rows": [["knn", ...], ["mice", ...]]}, ...}

dataforge fit-unit --recipe recipe_1d2b84bd --unit-id knn  --frame <train frame of fold 0>
dataforge fit-unit --recipe recipe_1d2b84bd --unit-id mice --frame <train frame of fold 0>

dataforge compose-imputer --recipe recipe_1d2b84bd --units unit_05c6305c,unit_045d0dc5
dataforge impute --imputer imputer_c620518c --frame <validation frame of fold 0>
dataforge evaluate-imputation --profile profile_6f8602e5 --imputed <filled frame>
```

### Tuning

Every decision the library makes is driven by a threshold, called a **dial**. When a dial
decides an outcome, such as a flag on a column, the result includes a **Dial Pointer** that
names it, for example `thresholds.missingness.severity_high`. The agent changes the outcome
by running the command again with that dial set:

```bash
dataforge profile --frame frame_a2e1bcf5 --thresholds '{"missingness": {"severity_high": 0.3}}'
```

Dials it leaves out keep their defaults. `dataforge schema <command> thresholds` lists them all.
Seeds are fixed by default, so the same call always gives the same result.

### Long-running commands

Commands marked `(background)` in the Skill (`load`, `profile`, `split`, `fit-unit`,
`impute`, `evaluate-imputation`, `c2st` and the frame commands) take longer as data grows. On
large data, the agent should run them in the background or with a longer shell timeout.
`--progress` prints one line per step to stderr. Stopping a command is safe: a Handle is
written completely or not at all.

## Results and errors

**Results.** stdout carries exactly one JSON value. Every result has `warnings` (read first)
and `omitted` (lists that were cut, each naming the command that returns the rest). Lists of
records use a compact table form:

```json
{"fields": ["column", "semantic_type", "null_ratio"], "rows": [["age", "numeric", 0.0], ["city", "categorical", 0.17]]}
```

**Errors.** A failure prints a Library Error with a `code`, a `message`, a `hint`, and fields
such as `param` or `candidates`. The exit code tells the agent what kind of failure it is:

| Exit | Category | What to do |
|---|---|---|
| 0 | success | Read `warnings`. |
| 2 | InvalidInput | Fix the argument named in `param` or `issues`. |
| 3 | UnknownName | Use one of `candidates`. |
| 4 | Conflict | Two declarations contradict: change one. |
| 5 | Infeasible | The call can't run on this data: change the plan or the data. |
| 6 | IncompatibleArtifact | Another library version wrote the Handle: run the command that made it again. |
| 1 | Unexpected | A bug. Stop and report the `code` and `message`. |

## Using it from Python

Every command is also a Python function, for glue code such as a loop over folds. This is
called **Code Mode**. The CLI and Python run the same function, so arguments, results,
Handles and error codes are identical.

Install the package into the environment your code runs in:

```bash
uv add dataforge-ml        # or: pip install dataforge-ml
```

The function name is the command name in snake_case, and arguments are keywords:

```python
from dataforge_ml import (
    compose_imputer, evaluate_imputation, fit_unit, impute, load, profile,
    resolve_recipe, route, split,
)

frame = load(path="train.csv").frame
prof = profile(frame=frame).profile
parts = split(profile=prof, scheme={"kind": "kfold", "k": 3})
resolved = resolve_recipe(routing=route(profile=prof).routing)

for fold in parts.folds:
    units = [
        fit_unit(recipe=resolved.recipe, unit_id=u.unit_id, frame=fold.train).unit
        for u in resolved.units
    ]
    imputer = compose_imputer(recipe=resolved.recipe, units=units).imputer
    filled = impute(imputer=imputer, frame=fold.validation).frame
    print(evaluate_imputation(profile=prof, imputed=filled).to_dict()["census"])
```

- Results are frozen records. Read fields as attributes; `.to_dict()` returns the exact JSON
  the CLI prints.
- Failures raise a subclass of `dataforge_ml.records.DataForgeError`, one per category
  (`InvalidInput`, `UnknownName`, and so on). Branch on `err.code`.
- Python has no settings flags. Workspace Settings come from the environment variables, then
  `.dataforge/config.toml`.

## Upgrading

```bash
uv tool upgrade dataforge-ml
dataforge install --skills        # add --global if you installed it globally
```

Run the Skill install again after every upgrade. Until you do, the first command prints a
one-line warning on stderr that the installed Skill doesn't match the library version.

Handles written by an older version may be refused with `IncompatibleArtifact` (exit 6).
Run the command that made them again.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dataforge: command not found` inside the agent | The command isn't on the agent's `PATH`. Use `uv tool install`, or start the agent from the virtual environment you installed into. |
| The agent doesn't use the Skill | Check `.claude/skills/dataforge/SKILL.md` exists in the folder you start Claude Code from, or install with `--global`. Ask for it by name with `/dataforge`. |
| A permission prompt on every command | Add `Bash(dataforge:*)` to `permissions.allow` in `.claude/settings.json`. |
| `input_not_declared` | `declared_inputs` is set and the file isn't listed. Add it to `.dataforge/config.toml`. |
| `invalid_setting` | A Workspace Setting has a bad value. The error names which one and where it came from. |
| A command times out | Raise the agent's shell timeout, run it in the background, or lower the data size. |
| Want to start over | Delete the `.dataforge/` folder. Every Handle in it is gone, and your settings file with it. |

## Developing DataForgeML

To work on the library itself, see [CONTRIBUTING.md](CONTRIBUTING.md) for the setup, tests and
the checklist for adding an Operation. Design decisions are in [docs/adr/](docs/adr/), and the
project's vocabulary is in [CONTEXT.md](CONTEXT.md).

## License

MIT
