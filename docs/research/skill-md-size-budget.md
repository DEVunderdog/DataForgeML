# How big may the package `SKILL.md` be?

Researched 2026-10-28. No ticket. Asked while `SKILL.md` sat at 1,995 of its 2,000-token cap.

**Question.** What size do the Agent Skills format and the four Harnesses we install into recommend or enforce for `SKILL.md`? Is the cap in `test_the_skill_is_about_2k_tokens` (2,000 `o200k_base` tokens, whole file, `SKILL_BUDGET_TOKENS` in `src/dataforge_ml/_boundary/_skill_build.py`) grounded, and what would a defensible cap be?

**This note does not edit the test or any ADR.** Section 5 gives a recommendation for the human to decide.

**Sources and method.** Primary sources only, read on 2026-10-28:
- Anthropic's Agent Skills docs, the Skill authoring best practices page and the Claude Code skills page.
- The agentskills.io open specification and its best-practices page.
- The `anthropics/skills` repo, cloned at `dbd4588f9e10`.
- OpenAI's Codex skills docs, and the `openai/codex` repo cloned at `fc916a22b57e`.
- Gemini CLI's skills docs, and the `google-gemini/gemini-cli` repo cloned at `b8fa81715573`.
- Cursor's skills docs (closed source, docs only).

Token counts are `tiktoken` `o200k_base`, measured with `uv run python` in this repo. Confidence labels follow [harness-capabilities-vs-contract.md](harness-capabilities-vs-contract.md):
- **VERIFIED (doc)**: stated in a vendor or spec document.
- **VERIFIED (source)**: read in repository code.
- **INFERRED**: derived from read facts. The inference is stated.
- **UNVERIFIED**: not found in a primary source.

---

## 1. Summary

- **The 2k cap cites no source.** ADR-0108 says "A test caps the body at about 2k tokens" with no reference. ADR-0113 only repeats "~2k-token budget". It is a house number.
- **The published guidance is 500 lines and 5,000 tokens.** The agentskills.io spec says SKILL.md instructions are "< 5000 tokens recommended" and "under 500 lines". Anthropic's docs say the same. Both are guidance, not limits.
- **One Harness truncates hard, and it is tighter than 2k tokens for our file.** Codex cuts an explicitly invoked skill's main prompt to 8,000 bytes. Our file is 8,238 bytes. The cut would drop the last line, which holds the links to five of the eight reference files.
- **Claude Code keeps only the first 5,000 tokens of a skill after compaction.** That is a hard cut too, but far above our size.
- **Gemini CLI and Cursor state no size limit.** Gemini CLI injects the whole body and never masks it later.
- **Our file is median-sized.** The median `SKILL.md` in `anthropics/skills` is 1,983 tokens. Ours is 1,995.
- **Recommendation.** Change the unit to UTF-8 bytes and cap the whole file at 8,000 bytes. That cap is set by a real truncation. It sits close to today's 2k tokens, so the file must lose about 240 bytes now. See section 5 for the options and tradeoffs.

---

## 2. Where the 2k came from

- ADR-0108 sets the cap: "A test caps the body at about 2k tokens." It cites nothing. It does cite "about 100 tokens up front" for the description, which matches the spec's metadata figure. [ADR-0108]
- ADR-0112 restates the cap as a split signal: "The ~1.5k-token cap becomes a signal to split a layer, never to cut it." That sentence is about Operation files. [ADR-0112]
- ADR-0113 says "`SKILL.md` stays within its ~2k-token budget". It cites nothing. [ADR-0113]
- The code says only "Most `o200k_base` tokens the whole Skill file may have (ADR-0108)". [SKILL-BUILD]

So the number is a design choice. No vendor or spec figure backs it. **VERIFIED (source).**

---

## 3. What each source says

### 3.1 The open spec (agentskills.io)

| Item | Figure | Kind | Source |
|---|---|---|---|
| `name` | 1 to 64 characters | hard (spec "must") | [AS-SPEC] |
| `description` | 1 to 1,024 characters | hard (spec "must") | [AS-SPEC] |
| `compatibility` | 1 to 500 characters | hard (spec "must") | [AS-SPEC] |
| Metadata at startup | ~100 tokens | estimate | [AS-SPEC] |
| SKILL.md instructions | "< 5000 tokens recommended" | guidance | [AS-SPEC] |
| SKILL.md length | "Keep your main `SKILL.md` under 500 lines" | guidance | [AS-SPEC] |

- The spec says "the agent will load this entire file once it's decided to activate a skill". It says to consider "splitting longer `SKILL.md` content into referenced files". [AS-SPEC]
- The best-practices page combines the two: "keeping `SKILL.md` under 500 lines and 5,000 tokens — just the core instructions the agent needs on every run". [AS-BP]
- It also warns against size from the other side: "Overly comprehensive skills can hurt more than they help." [AS-BP]
- It says to tell the Agent *when* to load each reference file. [AS-BP]
- The spec does not say which tokenizer its 5,000 counts with. **UNVERIFIED.**

**VERIFIED (doc).**

### 3.2 Anthropic (API docs, best practices, `anthropics/skills`)

- The overview table gives three levels. Metadata is "~100 tokens per Skill", always loaded. Instructions are "Under 5k tokens", loaded "When Skill is triggered". Resources cost "None until accessed". [A-OVERVIEW]
- `name` is at most 64 characters. `description` is at most 1,024 characters. [A-OVERVIEW], [A-BP]
- The best-practices page has a "Token budgets" section, but it gives lines, not tokens: "Keep the `SKILL.md` body under 500 lines for optimal performance." [A-BP]
- It says the body is read only when the Skill is relevant, and then "every token competes with conversation history and other context". [A-BP]
- It says keep references one level deep from `SKILL.md`. It says to give a reference file over 100 lines a table of contents. [A-BP]
- The `skill-creator` skill gives the levels in words and lines: metadata "~100 words", body "<500 lines ideal". It adds: "These word counts are approximate and you can feel free to go longer if needed." It sets the table-of-contents point at 300 lines. [A-SKILLCREATOR]

All of these are guidance. **VERIFIED (doc).**

### 3.3 Claude Code

- The skill listing caps each entry's `description` plus `when_to_use` at 1,536 characters. The whole listing is budgeted at 1% of the context window. Both are configurable. [CC-SKILLS]
- "Keep `SKILL.md` under 500 lines." [CC-SKILLS]
- Once loaded, a skill "stays in context across turns, so every line is a recurring token cost". [CC-SKILLS]
- After compaction, Claude Code "re-attaches the most recent invocation of each skill after the summary, keeping the first 5,000 tokens of each". All re-attached skills share 25,000 tokens. The docs tell authors to "put the most important instructions near the top of `SKILL.md`". [CC-SKILLS]

The 5,000-token cut is hard, but only after compaction. It counts in Claude's tokenizer, not `o200k_base`. **VERIFIED (doc).**

### 3.4 OpenAI Codex

Docs:
- Codex starts with each skill's name and description, then loads "the full `SKILL.md` instructions when they decide to use that skill". [X-SKILLS]
- The initial skills list uses at most 2% of the context window, or 8,000 characters if the window is unknown. It can be raised to 10,000 tokens. "Codex shortens skill descriptions first." [X-SKILLS]
- The docs give no limit on `SKILL.md` length. [X-SKILLS]

Source:
- The SKILL.md parser rejects a `name` over 64 characters. It only checks that `description` is non-empty. [X-SRC-PARSER]
- The listing cuts each description at 1,024 characters. [X-SRC-RENDER]
- **An injected skill's main prompt is cut at 8,000 bytes.** `MAX_SKILL_PROMPT_BYTES = 8_000`, and `truncate_main_prompt_contents` cuts at a character boundary. Codex then warns: "Skill `…` exceeded the main prompt context limit and was truncated." [X-SRC-RENDER], [X-SRC-EXT]
- That cut runs on every skill the skills extension injects for an explicit mention, host skills included. [X-SRC-EXT] The Codex CLI installs that extension. [X-SRC-CLI]
- Core's own fallback path cuts only agent-plugin skills, not plain host skills. [X-SRC-HOSTPROMPT]
- When the model reads `SKILL.md` itself through the shell, the normal tool-output cap applies instead: 10,000 tokens for every listed model, counted as 4 bytes per token. [X-SRC-MODELS], [X-SRC-TRUNC]

**VERIFIED (source)** for the constants and the code paths. **INFERRED:** a `$dataforge` mention in the Codex CLI goes through the extension path, so our file gets the 8,000-byte cut. I did not run Codex to confirm this. The docs do not mention the cut.

The bundled Codex `skill-creator` gives no number. It says: "A large upper bound is not a target: move conditional detail into references when doing so improves clarity or context use." [X-SRC-SKILLCREATOR]

### 3.5 Gemini CLI

- Only name and description load at first. On a match the model calls `activate_skill`, and "The `SKILL.md` body and folder structure is added to the conversation history." The docs state no size limit. [G-SKILLS]
- Source: `activate_skill` inserts `${skill.body}` whole, with no cut. [G-SRC-ACTIVATE]
- Its output is exempt from tool-output masking, so the body stays in history for the whole session. [G-SRC-MASK]

**VERIFIED (doc + source).** No hard limit.

### 3.6 Cursor

- "Keep your main `SKILL.md` focused and move detailed reference material to separate files." `name` must match the folder. No numbers are given for any field or for the body. [U-SKILLS]

**VERIFIED (doc)** for the guidance. Any truncation is **UNVERIFIED**, since Cursor is closed source.

---

## 4. Measurements

### 4.1 Our file

| Measure | Value |
|---|---|
| Whole file, `o200k_base` | 1,995 tokens |
| Frontmatter | 74 tokens |
| Body | 1,922 tokens |
| Generated command index | 669 tokens |
| `description` | 252 characters, 57 tokens |
| Lines | 113 |
| UTF-8 bytes | 8,238 (nine `→` take 3 bytes each) |
| Bytes per token | 4.13 |
| Tokens per line | 17.7 |

At 8,000 bytes Codex keeps the file up to `[profile](references/profile.md`. It drops the rest of the last line: the links to `split`, `impute`, `evaluate`, `workspace`, `python.md` and `glossary.md`. **INFERRED** from the byte offsets and [X-SRC-RENDER].

### 4.2 The `anthropics/skills` examples

The 19 `skills/*/SKILL.md` files at `dbd4588f9e10`, whole file: [A-SKILLS-REPO]

| Skill | Lines | Tokens | Tokens/line |
|---|---|---|---|
| internal-comms | 32 | 321 | 10.0 |
| brand-guidelines | 73 | 518 | 7.1 |
| theme-factory | 59 | 659 | 11.2 |
| web-artifacts-builder | 74 | 699 | 9.4 |
| webapp-testing | 96 | 884 | 9.2 |
| docx | 91 | 1,762 | 19.4 |
| academy-guide | 147 | 1,808 | 12.3 |
| frontend-design | 71 | 1,892 | 26.6 |
| mcp-builder | 236 | 1,938 | 8.2 |
| slack-gif-creator | 254 | 1,983 | 7.8 |
| pdf | 314 | 2,082 | 6.6 |
| xlsx | 99 | 2,137 | 21.6 |
| canvas-design | 130 | 2,353 | 18.1 |
| discernment-nudge | 209 | 2,422 | 11.6 |
| doc-coauthoring | 375 | 3,254 | 8.7 |
| algorithmic-art | 405 | 4,151 | 10.2 |
| pptx | 238 | 5,320 | 22.4 |
| skill-creator | 485 | 7,241 | 14.9 |
| claude-api | 616 | 26,696 | 43.3 |

- The median is 1,983 tokens. Ours, at 1,995, sits right on it.
- Three of 19 pass 5,000 tokens. One passes 500 lines.
- 11 of 19 pass 8,000 bytes, so Anthropic's own examples would hit the Codex cut.
- Tokens per line runs from 6.6 to 43.3. A line count says little about context cost.

### 4.3 The published limits in `o200k_base` tokens

| Limit | Kind | ≈ `o200k_base` tokens for our file |
|---|---|---|
| Codex injected prompt, 8,000 bytes | hard | ≈ 1,940 (at 4.13 bytes/token) |
| Current test | house | 2,000 |
| Spec and Anthropic, 5,000 tokens | guidance | 5,000 (tokenizer unstated) |
| Claude Code after compaction, 5,000 tokens | hard, after compaction only | ≈ 5,000 (Claude tokenizer) |
| 500 lines | guidance | ≈ 8,850 (at 17.7 tokens/line) |
| Codex shell read, 10,000 "tokens" = 40,000 bytes | hard | ≈ 9,700 |

**INFERRED** from the measured ratios. Our lines are dense, so 500 lines is the loosest figure here, and the line rule never binds first.

---

## 5. Recommendation

The cap should come from a real limit, not a house number. The tightest real limit on any Harness we install into is Codex's 8,000-byte cut. The published 5,000-token figure is guidance that every Harness ignores in code.

**Recommended: cap the whole file at 8,000 UTF-8 bytes.**
- Count `len(text.encode("utf-8"))`. Rename the constant and the test to bytes, and cite the Codex constant in the comment.
- Today's file is 238 bytes over. Trim or move about 240 bytes. Changing the nine `→` to `->` saves 18 bytes, so most of the cut has to be words.
- Put the reference links near the top, not on the last line. Both cuts we found (Codex's byte cut and Claude Code's compaction cut) drop the tail. Claude Code's docs ask for the most important text at the top. [CC-SKILLS]
- Gained: the cap matches a measured truncation, so passing the test means no Harness we read cuts the file. The unit needs no tokenizer and is exact. It sits near the old cap, so the house discipline barely changes.
- Given up: about 240 bytes now, and less room each time an Operation adds a line to the command index (about 35 tokens, or about 145 bytes, per command). The number is tied to one Codex constant that could change without notice, since no doc states it. Non-ASCII characters cost more than they look.

**Alternative A: keep 2,000 `o200k_base` tokens.**
- Gained: no change.
- Given up: the cap stays unsourced. It does not stop the Codex cut, because our file already passes 8,000 bytes at 1,995 tokens. The next Operation breaks it anyway.

**Alternative B: raise to the spec's 5,000 tokens, or a margin under it such as 4,000.**
- Gained: backed by the spec, Anthropic and Claude Code's compaction window. Room for about 60 to 85 more index lines.
- Given up: an explicit `$dataforge` in Codex would cut the file to its first 8,000 bytes. That loses the tail, which today holds the reference links. Gemini CLI never masks the body, so each extra token costs every later turn in every session that uses the Skill.

**Alternative C: count lines (500).**
- Given up: at our density 500 lines is about 8,850 tokens. That passes every hard limit above. Lines are the least useful unit here.

**Alternative D: move the generated command index to `references/commands.md`.**
- Pairs with any cap above. Frees about 670 tokens (about 2,700 bytes) and stops index growth from touching the cap.
- Given up: the Agent reads one more file before its first command. The best-practices pages say to keep only what is needed "on every run" in `SKILL.md` [AS-BP], and the command index arguably is.

If the 8,000-byte cap is adopted, confirm the Codex cut once by hand with `codex` and `$dataforge` before writing the ADR amendment. That code path is read, not run.

---

## Sources

**This repo**
- [ADR-0108] `docs/adr/0108-operation-docstrings-and-the-package-skill-are-written-for-agents.md`
- [ADR-0112] `docs/adr/0112-agents-reach-the-library-through-a-cli-and-a-skill.md`
- [ADR-0113] `docs/adr/0113-an-agent-glossary-behind-a-layered-command.md`
- [SKILL-BUILD] `src/dataforge_ml/_boundary/_skill_build.py` (`SKILL_BUDGET_TOKENS`), `tests/contract/test_skill.py` (`test_the_skill_is_about_2k_tokens`)

**Agent Skills open spec**
- [AS-SPEC] https://agentskills.io/specification (sections "Frontmatter", "Body content", "Progressive disclosure")
- [AS-BP] https://agentskills.io/skill-creation/best-practices (section "Spending context wisely")

**Anthropic**
- [A-OVERVIEW] https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview (sections "How Skills work", "Skill structure")
- [A-BP] https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices (sections "Concise is key", "Progressive disclosure patterns", "Token budgets")
- [A-SKILLCREATOR] https://github.com/anthropics/skills/blob/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills/skill-creator/SKILL.md
- [A-SKILLS-REPO] https://github.com/anthropics/skills/tree/dbd4588f9e1033efb41dad4bef2f7947c8993d44/skills
- [CC-SKILLS] https://code.claude.com/docs/en/skills (sections "Skill content lifecycle", "Claude stops following a skill", "Skill descriptions are cut short")

**OpenAI Codex** (repo `openai/codex` at `fc916a22b57e9d7b17988f044a54794d2222ae2c`, 2026-10-28)
- [X-SKILLS] https://learn.chatgpt.com/docs/build-skills (redirected from https://developers.openai.com/codex/skills)
- [X-SRC-PARSER] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/skills/src/parser.rs (`MAX_NAME_LEN`, line 4; check, line 82)
- [X-SRC-RENDER] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/ext/skills/src/render.rs (lines 23 to 29: `MAX_SKILL_PROMPT_BYTES = 8_000`, `MAX_CATALOG_SKILL_DESCRIPTION_CHARS = 1_024`; `truncate_main_prompt_contents`, line 1206)
- [X-SRC-EXT] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/ext/skills/src/extension.rs (explicit mentions, line 380; cut and warning, lines 448 to 455; `read_main_prompt` reads `Host` skills, lines 533 to 544)
- [X-SRC-HOSTPROMPT] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/ext/skills/src/host_prompt.rs (lines 78 to 82: cut only `is_agent_plugin_skill`)
- [X-SRC-CLI] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/cli/src/main.rs (line 2163: `codex_skills_extension::install`)
- [X-SRC-MODELS] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/models-manager/models.json (`truncation_policy` `tokens` 10000 on every model)
- [X-SRC-TRUNC] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/utils/string/src/truncate.rs (`APPROX_BYTES_PER_TOKEN = 4`)
- [X-SRC-SKILLCREATOR] https://github.com/openai/codex/blob/fc916a22b57e9d7b17988f044a54794d2222ae2c/codex-rs/skills/src/assets/samples/skill-creator/SKILL.md (line 60)

**Gemini CLI** (repo `google-gemini/gemini-cli` at `b8fa817155731c783d47dc6eaeee63163c0ec723`, 2026-10-28)
- [G-SKILLS] https://geminicli.com/docs/cli/skills/
- [G-SRC-ACTIVATE] https://github.com/google-gemini/gemini-cli/blob/b8fa817155731c783d47dc6eaeee63163c0ec723/packages/core/src/tools/activate-skill.ts (lines 142 to 151)
- [G-SRC-MASK] https://github.com/google-gemini/gemini-cli/blob/b8fa817155731c783d47dc6eaeee63163c0ec723/packages/core/src/context/toolOutputMaskingService.ts (`EXEMPT_TOOLS`, lines 32 to 41)

**Cursor** (docs only)
- [U-SKILLS] https://cursor.com/docs/context/skills
