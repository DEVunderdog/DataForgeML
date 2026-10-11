# The Operation inventory for today's phases

ADR-0099 says Operations replace the public Python API. ADRs 0103–0106 already named some of them: `load`, `export_frame`, `list_handles`, `profile` and its five detail Operations, `explain_routing`, and "no frame edits". This ADR lists every Operation for load → profile → split → impute → evaluate. For each one it gives the grain and the inputs and outputs in Handle terms. It also says what happens to every capability of today's door: either an Operation reaches it, or it is cut on purpose.

## Status

accepted, amended by ADR-0111 (`query_frame` is a 20th Operation). Resolves [What is the Operation inventory for today's phases?](https://github.com/DEVunderdog/DataForgeML/issues/583) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0090 (`author` folds into `route`), ADR-0084 (`derive_units` is no longer a public step), ADR-0076 and ADR-0077 (exclusions after imputation ride on the frame Handle), and ADR-0081 (the core budget is set by the Harness). Planning only: no code changes yet.

**Amended by [ADR-0112](0112-agents-reach-the-library-through-a-cli-and-a-skill.md) (2026-10-11).** The core budget is a Workspace Setting, set like the Workspace. Each Operation is also a CLI command, named in kebab-case.

## Decisions

- **19 Operations and 7 Handle kinds.** The Handle kinds are `frame`, `profile`, `routing`, `recipe`, `unit`, `imputer` and `c2st`. The `phase` values follow ADR-0109.

  | Phase | Operation | In → Out |
  |---|---|---|
  | load | `load(path)` | declared path → `frame_`, plus rows, columns, and names with dtypes (at most 100, then `omitted`) |
  | workspace | `export_frame(frame)` | → a CSV path in `exports/` |
  | workspace | `list_handles(kind)` | → paged Handles with Lineage |
  | profile | `profile(frame, declarations…, seed, thresholds)` | → `profile_` + Dataset Digest |
  | profile | `describe_columns`, `top_associations`, `missingness`, `describe_target`, `value_summary` | `profile_` → bounded detail (ADR-0104) |
  | split | `split(profile, scheme, seed, thresholds)` | → `{test, folds[]}`, each partition a `frame_` |
  | impute | `route(profile, strategies, default_strategy, model_choice, …)` | → `routing_` + one Routing Reason per column |
  | impute | `explain_routing(routing, columns)` | → Routing Evidence |
  | impute | `resolve_recipe(routing, hyperparameters, estimates)` | → `recipe_` + its units by value |
  | impute | `fit_unit(recipe, unit_id, frame, seed)` | → `unit_` + Fit Signals |
  | impute | `compose_imputer(recipe, units)` | → `imputer_` (a manifest, ADR-0103) |
  | impute | `impute(imputer, frame)` | → `frame_` + per-column records + Drop Candidates |
  | evaluate | `evaluate_imputation(profile, imputed, seed, thresholds)` | → `c2st_` + a bounded summary |
  | evaluate | `describe_c2st_columns(report, names, offset, limit)` | → per-frame results and provenance |
  | evaluate | `c2st(reference, candidate, repeats, scheme, seed, thresholds)` | → a C2ST result by value |

  Gained: every capability of today's door is reachable or cut below, and Phases 3–6 have a pattern to copy. Given up: 19 tools is more than a Harness should keep loaded at once. ADR-0109's `load_first` and the Package Skill's index carry that.

- **The imputation chain stays step by step, with one step fewer.** `resolve_recipe` lists its units by value (`unit_id`, `strategy`, `columns`), so `derive_units` is no longer an Operation. `fit_unit` is still called once per unit, as ADR-0075 says. There is no convenience chain (Scope 534 stands). `FittedImputer.compose` becomes `compose_imputer`, and `transform` becomes `impute`. Gained: one fewer call, and the unit list sits on the record it came from. Given up: the `strategy=` filter. The Agent filters the list itself.
- **`author` folds into `route`.** `route` takes one keyed array, `strategies=[{column, strategy, constant_fill, grouping_variable}]`, in the manual door's full vocabulary (`Passthrough`, `Dropped`, `MNAR`, `Constant` and the rest; `Indicator` is still refused). `default_strategy=null` auto-routes every column not listed. A value authors them instead. The separate MNAR and constant-fill arguments go away. Gained: one way to make a routing, one Routing Reason path (`authored`), and no second vocabulary to drift. Given up: routing without a profile. That costs nothing, because `resolve_recipe` needs one anyway, and it makes the Numeric-stamp bug from ADR-0090 impossible.
- **One `split` Operation with a tagged-union `scheme`.** It covers all nine of today's schemes: random, time, group, k-fold, group k-fold, repeated k-fold, holdout-CV, and the profile-stratified split and k-fold. There is one result shape, `{test: frame|null, folds: [{index, repeat, train, validation}]}`. The split result has no Handle of its own. Gained: one tool, one result shape, and holdout-CV needs no special case. Given up: a bigger schema with a nine-way union, and a docstring that covers every scheme.
- **A profile Handle carries the frame it was made from.** So `split`, `route`, `resolve_recipe` and `evaluate_imputation` take a profile and no separate frame. `fit_unit` and `impute` take a frame, because you fit on one fold and apply to another. Gained: fewer arguments, and the "profile of frame A, rows of frame B" mismatch cannot happen. Given up: profiling a fold means calling `profile` on that fold's frame first.
- **`fit_unit(frame=null)` fits on the frame behind the recipe's profile.** Another frame is allowed only if it has the same columns. Otherwise the call fails before any work. Gained: the common call has no frame to get wrong. Given up: this is the one place a different frame may still be passed, and its columns must match.
- **Every frame an Operation makes inherits its source's declarations.** That covers split folds and `impute` output. Targets, type overrides, sentinels and exclusions carry over. `profile` on such a frame may add to them but never remove them, and a contradiction (such as a different type for the same column) fails with a Library Error. `impute` adds the indicator columns as Soft Exclusions for later phases. Gained: a fold cannot silently lose its target, folds cannot disagree, and Phases 3–6 read exclusions off the frame they are given. `apply_exclusions` and its "nothing records that the call was made" gap are gone. Given up: a fold cannot re-declare a column differently. To do that, profile the source frame again. A frame Handle also carries a small declarations record.
- **`impute` returns a frame ready to model on.** It leaves out hard-excluded and `Dropped` columns and includes the indicator columns. `DroppedColumnAbsentWarning` is deleted, because a pre-trimmed frame can no longer exist. `FittedColumnAbsentError` becomes an Error Code for a frame that is missing columns. Gained: exclusion is honoured inside the library, which replaces ADR-0077's user drop step. Given up: excluded raw columns are not in the output. They stay in the source frame Handle.
- **Evaluation is a digest plus one detail Operation, and the generic seam stays.** `evaluate_imputation` writes a `c2st_` Handle and returns each column's outcome and score, for at most 100 columns. `describe_c2st_columns` returns the per-frame results. `c2st` over two frame Handles stays as an Operation, for example to check train-vs-test drift after a split. Gained: the report follows ADR-0104's pattern, and the generic check stays reachable. Given up: two more Operations, and one that is not imputation-specific for the Skill to explain.
- **Cut on purpose:**
  - `serialize`, `deserialize` and `inspect` are internal (ADR-0103).
  - `DataLoader(fmt=…)` becomes `load(path)`, CSV only.
  - `FittedUnit.transform` is cut. To tune one unit, compose again with the other unit Handles unchanged. That is cheap, because Handles are write-once.
  - `apply_exclusions`, the `with_*` methods, the config setters and `to_markdown` are deleted (ADR-0102, ADR-0106).
  - `stderr_observer` and `PipelineEvent` get no Operation. They wait on the map's progress question.
  - `n_jobs_inner` and the core budget leave the wire. The Harness sets the core budget the same way it sets the Workspace. Gained: CPU stays the Harness's resource (ADR-0099), and sub-agents cannot oversubscribe the machine. Given up: an Agent cannot trade inner parallelism for outer parallelism itself.

## Considered Options

- **Keep `derive_units` as its own Operation.** Rejected: it is a pure projection of a record the Agent already holds.
- **Keep `author` as a second routing Operation.** Rejected: two routing doors with two vocabularies, which the glossary already called a twin that must not drift.
- **One `split` Operation per scheme, or `split` plus `cv_folds`.** Rejected: more tools and more result shapes for one idea.
- **No declaration inheritance on derived frames.** Rejected: the Agent must declare the target again on every fold, and one omission silently changes the profile.
- **A per-unit `transform` Operation.** Rejected: re-composing does the same thing through one path.
- **`n_jobs_inner` as an argument.** Rejected: it puts a machine resource in the Agent's hands.

## Consequences

- The Package Skill's workflow follows the table above.
- ADR-0109's catalogue has 19 entries. `load_first` stays `load`, `profile`, `list_handles`.
- The glossary gains **Inherited Declarations**. The Manual Authoring Door, `derive_units` and `apply_exclusions` entries are marked as being replaced.
- The map's "Sub-agent fit" question now covers a known list: 19 Operations, and only `create` and `export` Operations write.
