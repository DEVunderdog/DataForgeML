# Operation docstrings and the package Skill are written for Agents

ADR-0099 decided that an Agent learns the library in two steps. First it reads `SKILL.md` inside the package. Then it reads one Python file per Operation, in place. ADR-0101 decided that schema descriptions are parsed from those docstrings. So the docstrings now have a new main reader: an Agent, not a human. ADR-0034's rule was written for human users of a public Python API. Under ADR-0099 that API becomes internal, so its scope now covers nothing. This ADR decides how Operation docstrings and the Skill are written, which parts are generated, and what else ships.

## Status

accepted. Resolves [How are Operation docstrings and the Skill written for Agents?](https://github.com/DEVunderdog/DataForgeML/issues/580) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0034 (scope and audience), supersedes ADR-0038 (the Sphinx toolchain is deleted), and adds one line to ADR-0099's Harness contract. Planning only: no code changes yet.

**Amended by [ADR-0112](0112-agents-reach-the-library-through-a-cli-and-a-skill.md) (2026-10-11).** The Package Skill becomes a CLI cheat-sheet: workflow, one usage line per command, dial iteration, error categories and exit codes, and which commands to run in the background. `references/` holds one file per phase plus `python.md`. `dataforge install --skills` copies it into `.claude/skills/dataforge/` and `.agents/skills/dataforge/`, stamped with the library version. Per Operation, the Agent reads layered `--help` (generated from the seven-part docstring) and `dataforge schema`, not the thin Operation file. The file stays the source and the Code Mode contract. The docstring's Example is a shell command. The ~1.5k-token cap becomes a signal to split a layer, never to cut it.

**Amended by [ADR-0113](0113-an-agent-glossary-behind-a-layered-command.md) (2026-10-28).** A shipped Agent glossary is no longer rejected. The library ships one, read one term at a time through `dataforge glossary`, with one markdown file per phase, plus a `common` file, in the Skill folder as the fallback. The rule that a docstring defines a term in one clause where it is first used stands.

**Amended on 2026-10-28 by [the SKILL.md size budget research](../research/skill-md-size-budget.md).** The Skill's cap is 8,000 UTF-8 bytes, not about 2k tokens. The 2k figure had no source. Codex cuts an explicitly invoked Skill to its first 8,000 bytes, so a longer file loses its tail there. The References line sits near the top of `SKILL.md`, because a Harness that cuts the file cuts its end. Gained: the cap matches a real cut and needs no tokenizer. Given up: it rests on a Codex constant its docs do not state, and each new command line takes about 145 of the bytes.

## Decisions

- **There are two docstring rules in one format.** Operations and Wire Records follow the Agent rule below, and a test enforces it. Internal code keeps ADR-0034's numpy rule. Its scope is redrawn to the internal seams Operations call: `route`, `resolve_recipe`, `fit_unit`, the profiler, the splitter and `c2st`. The format is numpy everywhere, because the ADR-0101 description parser reads numpy sections. Gained: coding Agents that maintain the library still get contracts at the seams, and there is one format. Given up: two scopes to keep straight in `CLAUDE.md`, and docstring upkeep on code no user sees.
- **An Operation docstring has seven parts, in this order:**
  1. a one-sentence summary, which is also the tool-search catalogue line;
  2. `Use when`, a paragraph saying when to call it, when not to, and which Operation to use instead;
  3. `Parameters`, where a dial states its meaning and the generator appends its default and range (ADR-0101);
  4. `Returns`, naming the Wire Record, what it can omit, and which detail Operations return the rest;
  5. `Errors`, listing Error Codes instead of exception classes;
  6. `See Also`, naming the Operations before and after it in the workflow;
  7. `Example`, one short Code Mode snippet, run as a test against a fixture.

  Gained: the file alone tells the Agent when to call the Operation, how, what comes back and what to call next. Given up: docstrings two to three times longer, and a doctest fixture to maintain.
- **Docstrings define the domain terms they use.** A term is explained in one clause where it is first used. ADR-0034's "name terms, never define them" rule does not apply to Operations, because `CONTEXT.md` does not ship. In Tool Calls the model may see only the schema description and no Skill. There is no shipped Agent glossary. Gained: every Operation explains itself in both deliveries, and no second glossary drifts from `CONTEXT.md`. Given up: the same clause repeats across files, and two files may word a term differently. No test catches that.
- **`SKILL.md` holds the workflow, not the contract.** It contains:
  - frontmatter whose `description` says when to use the Skill (about 100 tokens up front);
  - the workflow as a chain of which Operation makes which Handle and which Operation reads it next;
  - an Operation index giving name, summary line and file path;
  - the iteration loop: read the Dial Pointer, turn the dial, run again;
  - how to act on each error category;
  - what is outside the Agent's reach (the Workspace and the Literal Policy).

  It never holds signatures, parameters, dial defaults or record fields. A test caps the body at about 2k tokens. Gained: the Skill cannot drift on a signature it never states. Given up: any real call costs at least one file read past the Skill.
- **Hand-written vs generated:**
  - Hand-written: Operation docstrings (they are the source), and the `SKILL.md` workflow prose.
  - Generated: the `SKILL.md` Operation index, written between markers from summary lines and file paths. A test fails if it is stale. Tool schemas, the catalogue data and constraint text are generated too (ADR-0101).
  - `Errors` lists are hand-written. A test checks that every listed code exists in the registry.

  Gained: nothing an Agent reads in two places is maintained twice. Given up: no test can prove an `Errors` list is complete, only that each code in it is real.
- **Agent-facing text is written with the `writing-for-agents` skill.** The maintainer is Claude Code. It loads `writing-for-agents` before writing or editing `SKILL.md` or any Operation docstring. The seven-part structure is fixed and wins on any conflict. Gained: every text an Agent reads is written under one discipline. Given up: that skill targets skills and `CLAUDE.md` files, so some of its advice does not fit numpy sections.
- **An Operation file is thin.** It holds the signature, the docstring, its own Wire Records, and one call into internal code. It has no logic. A test caps each file at about 1.5k tokens. Gained: one read costs about as much as the docstring. Given up: an extra layer for maintainers, and the boundary checks live in shared internal code.
- **A Wire Record lives with the only Operation that uses it.** Shared records (Library Error, Omission, Dial Pointer, Result Warning) are documented once in `records.py`, and `SKILL.md` points to it. Gained: one read usually covers an Operation's whole contract, both call and result. Given up: the per-file cap must fit the record docstrings too, and the shared file is a second read the first time.
- **`SKILL.md` sits at the package root.** Its paths are relative to `dataforge_ml/`. Gained: paths resolve the same way in every install. Given up: the root layout becomes part of the contract. ADR-0099 already accepted this for Operation files.
- **Nothing else ships.** There is no `llms.txt`, since that is for websites and the Agent reads the installed package offline. The Sphinx site, `docs/conf.py`, `docs/api/`, the `docs` dependency group and the GitHub Pages deploy are deleted. The README covers people. Gained: one less generated surface, and no docs build dependencies. Given up: no browsable API site for people, including recruiters. They read the README and the code.

## Considered Options

- **No mandatory docstrings on internal code.** Rejected: the maintainers are coding Agents, and they read the internal seams the way users read Operations.
- **One rule everywhere.** Rejected: the Agent rule's `Use when`, `See Also` and `Example` sections cost a lot and buy little on internal helpers.
- **Keep numpy's `Raises`.** Rejected: ADR-0100 turns every failure into an Error Code, and exception classes are not part of the contract.
- **A shipped Agent glossary.** Rejected: Tool Calls may never show it, and it would drift from `CONTEXT.md`. Reversed by [ADR-0113](0113-an-agent-glossary-behind-a-layered-command.md): ADR-0112 cut Tool Calls, and a contract test and keeping each term in exactly one file contain the drift.
- **Generate `SKILL.md` whole (FastMCP `generate-cli` style).** Rejected: generated text can list Operations but cannot explain the workflow or the iteration loop.
- **All records in one file, or one file per record.** Rejected: both turn one Operation's contract into several reads.
- **Ship `llms.txt`, or keep a Sphinx showcase site.** Rejected: humans are not users, and both are extra generated surfaces to keep in sync.

## Consequences

- ADR-0034 is amended: its numpy rule now covers only the internal seams listed above, and Operations follow this ADR.
- ADR-0038 is superseded. The toolchain it chose is deleted.
- ADR-0099's Harness contract gains a line: the Harness exposes the package's Skill directory (`dataforge_ml/`) to the Agent, by copying or linking it into its own skills location. A short `importlib.resources` lookup that finds it is documented.
- `CLAUDE.md` is rewritten when the Operations land. It states the two scopes, the seven-part rule, and that `writing-for-agents` must be loaded before writing Agent-facing text. Until then today's rule stays, because today's code is unchanged.
- Removing the Sphinx build and its Pages deploy is a standalone ticket, [Remove the Sphinx docs build and its GitHub Pages deploy](https://github.com/DEVunderdog/DataForgeML/issues/587). It covers `deploy-docs.yml`, the `docs` dependency group, `uv.lock`, `docs/conf.py`, `docs/api/`, and a manual checklist for the `gh-pages` branch and the repo's Pages setting.
- New tests: docstring sections present and in order; `Example` runs; every `Errors` code exists; the Skill index is fresh; the Skill body and each Operation file stay under their token caps.
