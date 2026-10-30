# glossary

The commands, their `--help` and these references name terms by capitalising them, such as Signal Score or Feasibility Floor. Look a capitalised term up before you act on it: before you turn the dial it names, branch on it, or report it to the user.

## Looking a term up

`dataforge glossary` answers in three layers. Each prints one JSON value, and each is small, so read only the layer you need:

1. `dataforge glossary`: each phase and how many terms it has, with the count of terms every phase uses apart.
2. `dataforge glossary --phase <phase>`: that phase's term names, with no definitions. `--common` lists the names of the terms every phase uses, such as Handle and Dial.
3. `dataforge glossary <term>`: one entry. It gives the definition, the aliases, the related terms and the commands whose output uses the term.

Pass a term as several words, with or without quotes: `dataforge glossary signal score`. Case and plurals do not matter, and an alias finds its term. Follow a `related` term with another lookup when the definition leans on it.

## When the lookup misses

A term the glossary lacks fails with `unknown_term`, exit 3. Look up one of its `candidates` first. When none fits, read the glossary file of the phase you met the term in, `glossary/<phase>.md` in this Skill's folder, plus `glossary/common.md`. Each entry there is a `## Term` heading, then its `Aliases:`, `Related:` and `Commands:` lines, then its definition.
