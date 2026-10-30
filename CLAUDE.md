# DataForgeML — Claude Code Instructions

Agents are the only users. They reach the Operations two ways (ADR-0112):

1. **The CLI** first: `dataforge <command>` runs one Operation, taught by the package Skill (`src/dataforge_ml/skill/`).
2. **Code Mode** second: the same Operations imported from `dataforge_ml`.

An Operation's docstring is the source of its `dataforge <command> --help`. Editing a docstring or signature means regenerating the committed help; the commands are in `CONTRIBUTING.md`, "Regenerate the derived files".

Adding an Operation? Follow the checklist in `CONTRIBUTING.md`, section "Adding an Operation".
Writing or reviewing code? Follow the judgement-call rules in `CODING_STANDARDS.md`, including the docstring rule.
