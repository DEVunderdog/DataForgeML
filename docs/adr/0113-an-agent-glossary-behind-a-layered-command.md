# An Agent glossary behind a layered command

The Skill's references use terms they never define, such as Signal Score and Feasibility Floor. An Agent meets them in `references/impute.md` and in `--help`, and `CONTEXT.md`, which defines them, does not ship. ADR-0108 rejected "A shipped Agent glossary" for two reasons. Tool Calls might never show it, and it would drift from `CONTEXT.md`. Both reasons have changed. ADR-0112 cut Tool Calls, so every Agent has a shell and the Skill. Drift is now contained by a contract test and by keeping each term in exactly one file. This ADR decides how the library ships a glossary for Agents, and how an Agent reads it without reading all of it.

## Status

accepted. Decided on 2026-10-28 in a design session on [Glossary Entries for subjective and common terminologies](https://github.com/DEVunderdog/DataForgeML/issues/613). It amends ADR-0108: its rejection of a shipped Agent glossary is reversed. Its rule that Operation docstrings define a term in one clause where it is first used stands. Planning only: no code changes yet.

## Decisions

- **Lookups are selective.** An Agent looks up one term at a time and never reads every definition at once. Gained: a lookup costs what the question costs, not what the whole glossary costs, however large the glossary grows. Given up: an Agent that wants to learn many terms makes many lookups.
- **The primary path is a `dataforge glossary` command.** It is not an Operation. Like `schema` and `install`, it is a helper command, and like them it never imports the heavy dependencies. It has three layers:
  1. No arguments: each phase and its term count, and the Common Terms count as a separate group.
  2. `--phase <phase>` or `--common`: that phase's term names, or the Common Terms' names, with no definitions.
  3. `<term...>`: one entry, with its term, its phase (or `common`), definition, aliases, related terms and the commands whose output uses it. The term can be passed as several words, so quotes are optional. Matching ignores case and plurals, and accepts aliases.

  Gained: it works under `Bash(dataforge:*)`, with no file-read permission, and each layer is small. Given up: a new helper command to maintain, with its own help and tests.
- **Output is one JSON value, like every other command.** Gained: the Skill's one rule, that every command prints one JSON value on stdout, has no exception, and `cmd | jq` works the same way. Given up: a definition reads as a JSON string, not as prose.
- **A missing term is an `unknown_term` error.** It is a Library Error of category `UnknownName`, so the command exits with `3`. Its `candidates` list close matches, and its hint points to the fallback. An unknown phase is an `invalid_argument` error whose `allowed` lists the phases. Gained: a miss is handled like every other unknown name, and the Agent learns where to look next. Given up: one more Error Code in the registry.
- **The fallback is reading one phase's file.** Entries live in one markdown file per pipeline phase, `glossary/<phase>.md` inside the Package Skill folder: `load`, `profile`, `split`, `impute` and `evaluate`. Each term lives in exactly one file. An entry is a `## Term` heading, then `Aliases:`, `Related:` and `Commands:` lines, then the definition. The command parses these same files. Gained: the command and the fallback never disagree, because they share one source, and an Agent whose command misses still reads only one phase's terms. Given up: the file format is a contract the parser and the coverage test both depend on.
- **Common Terms are not a phase.** Terms the whole library uses, whatever the phase, live in `glossary/common.md`: Agent, Handle, Workspace, Error Code, Dial, Lineage and the like. Workspace terms count as common, because every phase reads and writes Handles in the Workspace, so there is no `workspace` glossary file. `--phase` accepts only the pipeline phases, and `--common` lists the Common Terms. The fallback is the one phase's file plus `common.md`. Gained: a phase file holds only that step's own terms, and a shared term has one obvious home. Given up: one more flag, and the Agent may read two files on the fallback.
- **`SKILL.md` stays an index.** It gains one link, to a reference guide `references/glossary.md` on how to look terms up, and nothing more. The guide says when to look a term up, the three layers and the fallback. Gained: `SKILL.md` stays within its ~2k-token budget, and it points the way rather than holding definitions. Given up: an Agent reads one more short file before its first lookup.
- **Definitions are written for Agents.** They name no ADR numbers and no internal class names. They use only terms that are themselves entries. Each entry is at most about 120 `o200k_base` tokens, and a test checks it. Gained: an Agent can follow every definition without reading `CONTEXT.md` or the code. Given up: Agent definitions are maintained by hand alongside `CONTEXT.md`, and two files can word the same term differently.
- **A contract test keeps the glossary complete.** It finds terms the way this codebase marks them, by capitalising them. It collects runs of Title-case words that sit mid-sentence, plus runs of two or more such words at the start of a sentence after a leading article is dropped, in `SKILL.md`, every Skill reference, every generated `--help` and schema page, and the glossary itself. Every term it finds must match exactly one entry, by name or alias, singular or plural. Every entry must be used somewhere in that text. Every `Related` term must be an entry, and every `Commands` value a real command. Proper names and error category names sit on an allow-list, each with its reason. Gained: a new term in Agent-facing text fails the build until it has an entry, and no dead entry lingers. Given up: the test misses a one-word term used only at the start of a sentence. An entry named only in another entry still counts as used. The allow-list needs upkeep.
- **`CONTEXT.md` stays the developer glossary.** It keeps its ADR links and internal names, and it does not ship. Operation docstrings keep ADR-0108's rule: define a term in one clause where it is first used. Gained: developers and Agents each get a glossary written for them, and a docstring still explains itself without a lookup. Given up: a term is described in up to three places. They are `CONTEXT.md`, its glossary entry and the docstring clause.

## Considered Options

- **A single glossary file.** Rejected: one read grows with the project, so every lookup pays for every term.
- **Inline definitions only, as ADR-0108 decided.** Rejected: the Skill references already use terms they never define, and a one-clause definition in a docstring does not reach a reference file.
- **Plain-text output for `dataforge glossary`.** Rejected: it breaks the rule that every command prints one JSON value.
- **Ship `CONTEXT.md` itself.** Rejected: it is written for developers, with ADR numbers and internal class names an Agent cannot use, and it is one file that grows with the project.
- **A glossary Operation.** Rejected: the glossary is about the library, not about the Agent's data. Like `schema`, it reads shipped files and needs no Workspace or Handle.

## Consequences

- ADR-0108 is amended: its "A shipped Agent glossary" rejection is reversed by this ADR. Its docstring rule stands.
- `CONTEXT.md` gains an **Agent Glossary** entry, and its **Package Skill** entry mentions the glossary files.
- The Error Code registry gains `unknown_term` (`UnknownName`).
- The `dataforge --help` overview gains a `dataforge glossary [term]` line in its "Go deeper" section, and the committed help is regenerated.
- The Skill contract test accepts `references/glossary.md` and counts `glossary` as a real command.
- `CONTRIBUTING.md`'s "Adding an Operation" checklist gains a row: a new capitalised term needs a glossary entry.
- New tests: the three layers and the `unknown_term` error; the command imports no heavy dependency; the term-coverage contract test; the per-entry token cap.
- The work is ticketed under [#613](https://github.com/DEVunderdog/DataForgeML/issues/613): the command with the impute entries first, then the remaining entries with the coverage test, and the Skill and help wiring.
