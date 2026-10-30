# Configuration is Operation arguments, and results point at the dials behind them

Today configuration is the one mutable object in an otherwise immutable flow. `PipelineConfig` nests the profiling, imputation and split configs. Setters mutate it in place and return nothing. Unknown column names pass silently (G12). `FittedImputer.apply_exclusions` writes into the caller's config (G11). `PipelineConfig.random_seed` is read by nothing, and profiling hard-codes `42` in places (G10). The large dial surface exists so an Agent can tune the pipeline the way hyperparameter search tunes a model: run a step, read what happened, turn a knob, run it again. This ADR decides how an Agent configures the pipeline, and how each result tells the Agent which knob to turn.

## Status

accepted. Resolves [How does an Agent configure the pipeline?](https://github.com/DEVunderdog/DataForgeML/issues/578) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0100 (all argument problems in one error), ADR-0101 (setters leave the wire entirely) and ADR-0105 (`dial` becomes a Dial Pointer). For Agents it supersedes ADR-0043/0044 (setter-only Declarative Config Fields) and ADR-0023's exclusion propagation. Planning only: no code changes yet.

## Decisions

- **There is no config object.** Each Operation takes the decisions it reads as typed keyword arguments. A declaration made upstream travels down the Handle chain: `route` reads the type overrides and exclusions from the profile Handle it is given, and they are never passed again. There are no setters, no `configure` Operation and no config diff. Changing a decision is calling the Operation again, which writes a new Handle, and Lineage (ADR-0103/0104) is the record. Gained: configuration costs zero tools against the visible-tool limit, each decision has one source, and `profile` and `route` cannot disagree about an exclusion. G11 is gone because nothing is mutable. Given up: the Agent cannot build and review a whole config before data exists, and changing an upstream declaration means re-running everything below it.
- **Each declaration belongs to the first Operation that reads it.**
  - `profile`: semantic type and numeric kind (D1), sentinels and datetime formats (D3), target columns (D5). The split Operations read the target from the profile Handle.
  - `route`: MNAR columns (D7), per-column strategy, constant fill and bimodal grouping (D8), and `model_choice`.
  - `resolve_recipe`: per-unit `hyperparameters` and per-column `estimates`, as keyed arrays.
  - The split Operations: the scheme (D6).
  - Every phase Operation: `exclude_columns`, added to what it inherits, never removing it; and its own `skip_columns` for that phase only. The `PipelinePhase`-keyed soft-exclusion map is deleted.

  Gained: each schema holds only what that Operation reads, and soft exclusion stops being an open map. Given up: an imputation-only skip cannot be declared before profiling.
- **The `with_*` edit methods are deleted.** `with_model_choice`, `with_hyperparameters` and `with_estimates` become arguments on `route` and `resolve_recipe`. The recipe still keeps a decided base plus a sparse override delta inside. Gained: one way to change anything, and comparing two iterations compares plain arguments. Given up: the Agent passes all its overrides on every call, not only the new one.
- **Drop Candidates are reported, not applied.** The imputation result lists them with a Dial Pointer to the threshold that flagged them. The Agent excludes them downstream by passing `exclude_columns`. Gained: every drop is a decision the Agent made. Given up: one more thing the Agent must act on.
- **One `seed` argument per stochastic Operation.** `profile`, the split Operations, `fit_unit` and `c2st` each take `seed`. `null` means a fixed library default, not "random". `PipelineConfig.random_seed`, `NumericProfileConfig.random_state` and the hard-coded `42`s are deleted. Gained: the same call on the same data gives the same answer, and one step can be re-run with a new seed alone. Given up: random-by-default, and no single run-wide seed ties the steps together (Lineage shows each).
- **Every argument is checked before any work.** Names are checked against the frame or Handle, dials against their ranges, and cross-field rules together. On failure no Handle is written. All argument problems come back in one Library Error as a capped list of per-argument issues. Gained: a tuning loop fixes everything in one round trip. Given up: a nested list in the error record.
- **Every dial stays on the wire.** Dials sit in one nullable, nested `thresholds` record per Operation, grouped by sub-config. Declarations stay at the top level. The schema description states each dial's default and range. Gained: the Agent's whole action space is visible, and the common call passes `thresholds=null`. Given up: large schemas; `profile` alone has about 110 dials.
- **Every constant on a decision path is a dial.** A hard-coded literal there is a defect. The exception is a constant that defines what a thing is rather than how strictly it is judged, such as the Binomial ½ null in C2ST. Gained: new knobs arrive by rule, not by debate. Given up: a growing schema.
- **Results carry Dial Pointers.** Wherever a result reports an outcome a threshold decided (a profiling flag, a split fallback, a C2ST outcome, a feasibility failure, a Drop Candidate), it carries `{operation, dial, value, source}`. `dial` is the argument path in the Operation schema, in the same notation as a Library Error's `param` (for example `thresholds.numeric.skew_threshold`). `source` is `default` or `argument`. Gained: the Agent's feedback signal names its action, and the resolved value of a default is visible after all, which ADR-0104's Lineage gave up. Given up: a few bytes per flagged outcome, a flag-to-dial mapping the library must maintain and test, and renaming a dial path becomes a breaking change.

## Considered Options

- **A config Handle with a `configure(config, edits[])` Operation returning a diff.** Rejected: a second record of every choice beside Lineage, one more Handle kind and tool, and a cross-phase object each Operation drags in to read a sliver.
- **A whole config Wire Record passed to each Operation.** Rejected: huge schemas on every call, and the Agent can pass different exclusions to different phases.
- **Keep only dials some result points at.** Rejected: dials are the Agent's action space, and trimming them limits what it can explore.
- **Apply Drop Candidates downstream automatically.** Rejected: the library would make a drop the Agent never made.
- **Keep the `with_*` methods as an edit Operation.** Rejected: two ways to reach the same recipe, and stacked edits make Lineage hard to read.
- **Non-deterministic `null` seed.** Rejected: identical calls could disagree, and Lineage could not reproduce a result.

## Consequences

- ADR-0100 is amended: an argument-validation Library Error may carry a capped list of per-argument issues.
- ADR-0105 is amended: Routing Evidence's `dial` becomes a Dial Pointer.
- `PipelineConfig`, the config setters, `apply_exclusions`, `resolve_active_columns` as a public method, and the `with_*` methods become internal or are deleted. The Phase Active-Columns Contract now reads each phase's own `exclude_columns`/`skip_columns` plus what it inherits.
- An audit that turns today's hard-coded literals on decision paths into dials is an implementation follow-on.
- There is no compare Operation. The Agent compares two iterations of a step by diffing their results and Lineage itself.
