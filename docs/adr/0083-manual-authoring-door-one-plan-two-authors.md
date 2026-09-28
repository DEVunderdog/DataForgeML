# A manual authoring door for imputation — one plan, two authors

> **Amended by [Where dial defaults and profile estimates are resolved](https://github.com/DEVunderdog/DataForgeML/issues/531) (map #528, ADR-0089).** The bimodal centres, `feature_cols` and `domain_snap_bounds` leave the column decision. They are profile estimates on `ImputationRecipe.column_estimates`, edited with `with_estimates(column, **fields)`. `constant_fill` and `grouping_variable` stay on routing (ADR-0088). GMMSampling's `central_tendency` fact is deleted: no fitter read it. What `author` becomes is decided by #533.

> **Partly superseded by [What author becomes](https://github.com/DEVunderdog/DataForgeML/issues/533) (map #528, ADR-0090).** `author(map, *, profile=None, base=None, default=Passthrough)` returns `ImputationRouting`. `columns=` becomes `profile=` (universe and semantic types, never statistics). `estimators=` and both sentinel keywords are deleted: the estimator is set with `routing.with_model_choice(estimator)` and sentinels come off the profile. `AuthoredColumn` keeps `strategy`, `constant_fill` and `grouping_variable`. A missing estimate is refused by `fit_unit`, not the door. `base=` takes a routing and its dial and delta carry-over rules are gone. Level 2 scope, the column-keyed map and the strategy vocabulary stand.

> **Amended by [Persistence of the layered artifacts](https://github.com/DEVunderdog/DataForgeML/issues/535) (map #528, ADR-0096).** `from_dict` no longer gap-fills. A loaded recipe with a missing or unknown dial raises, and that strict load is what licenses the fitters' hard subscripts now.

`decide()` was the only way an `ImputationDecision` could come into existence, and ADR-0082 closed the one incremental door (`with_strategy`) that would have gotten a hand-author there by editing. The automated↔manual spectrum therefore had no manual end: a user who knew their data better than the router could not say so. This ADR opens that end. **A free function `author()` builds a real `ImputationDecision` from a column→strategy map, and the plan it returns is indistinguishable to everything downstream from one `decide()` produced.** One execution layer, one artifact, one evaluation surface, two authors.

Nothing below the plan needed inventing. The generic unit layer a manual path requires — `ImputationUnit`, `fit_unit`, the structural `FittedUnit` protocol, `FittedImputer.compose` — already existed (ADR-0060, ADR-0071). The friction was one level up: `ImputationDecision` was shaped as `decide()`'s *output*, so every field it carried was something `decide()` happened to know (`decided_for_shape`, `profile_provenance`, `forced`, `config_snapshot`, `signals`, `decided_hyperparameters`). A hand-author knows almost none of them and would have had to fake all of them.

The repeated finding across the effort is that the asymmetry was **residue, not design**. Three of the four contested fields had no readers in `src/` at all; three "dials" the fitters read were facts about the data misfiled as hyperparameters; two `NumericImputationConfig` overrides were a second spelling of `with_hyperparameters`. The manual door did not need special cases carved for it — it needed dead weight removed, and removing that weight fixed pre-existing defects on the automatic path (a bare MICE plan that trains nothing, a `ClusterConditional` unit that fills zero nulls, `Passthrough` raising on ordinary categorical blanks). **Every deletion below is a deletion outright — nothing was rehomed, defaulted or shimmed.**

Scope is **numeric only** and **Level 2**: library strategies, plus a user's own sklearn-compatible estimator inside a model-based strategy. Level 3 — arbitrary user fill logic admitted as a first-class unit — was considered seriously and cut (see *Out of scope*). Decided across [#447](https://github.com/DEVunderdog/DataForgeML/issues/447)–[#461](https://github.com/DEVunderdog/DataForgeML/issues/461) under map [#446](https://github.com/DEVunderdog/DataForgeML/issues/446); the code change is an unticketed follow-on.

## The door

```python
def author(
    columns_map: Mapping[str, ImputationStrategy | AuthoredColumn],
    *,
    columns: Sequence[str],
    default: ImputationStrategy = ImputationStrategy.Passthrough,
    estimators: Mapping[str, Any] | None = None,
    numeric_sentinels: Mapping[str, Sequence[float]] | None = None,
    string_sentinels: Mapping[str, Sequence[str]] | None = None,
    base: ImputationDecision | None = None,
) -> ImputationDecision: ...

@dataclass(frozen=True)
class AuthoredColumn:
    strategy: ImputationStrategy
    center1: float | None = None
    center2: float | None = None
    feature_cols: tuple[str, ...] | None = None
    grouping_variable: str | None = None
    constant_fill: float | None = None
    domain_snap_bounds: tuple[float, float] | None = None
```

`author` sits at the same altitude and in the same grammar as `decide` — both verbs, both taking a description and returning an `ImputationDecision`, one from a profile and one from a person ([#451](https://github.com/DEVunderdog/DataForgeML/issues/451)).

## Locked decisions

### 1. The plan sheds what only `decide()` could know ([#447](https://github.com/DEVunderdog/DataForgeML/issues/447))

- **`decided_for_shape`, `profile_provenance` and `forced` are deleted outright** — not made optional. All three had **no readers in `src/`**. No field, no two-author asymmetry, and frameless authoring (from a schema, in CI, against next month's extract) works for free. Knowingly given up: `decided_for_shape` was the only record of the row count a plan was valid for (ADR-0066), so enforcing that rule later needs a second format change; `forced` was ADR-0066's "forced raises instead of degrading" predicate, already dead since #377/#389 removed degradation wholesale.
- **`config_snapshot` stays and may be empty.** `PipelineConfig.from_dict({})` yields full defaults, so **empty means "library defaults"** — truthful for a plan no config produced.
- **`signals` returns to being purely descriptive and is legally empty.** A hand-authored decision explains no routing because there was none; absence of rationale *is* the encoding.
- **Consequence: `Passthrough` tolerates nulls — a behaviour change for every plan.** `signals` being descriptive forces `_EXCLUSION_SIGNAL` off the transform path, and with it the Passthrough violation check. `Passthrough` now means *leave this column completely alone, nulls included*. This fixes a pre-existing defect (a categorical column with blanks is normal data and the library errored on it) and removes a real safety net — "I thought this column would get filled and it didn't" now passes silently. It also leaves the exported `UnfittedColumnError` with **no raise site**; whether the export is retired is an ADR-0050 question this map did not rule. *Ruled in implementation ([#465](https://github.com/DEVunderdog/DataForgeML/issues/465)): removed.* An exception no code path can raise fails the ADR-0050 test — the user never needs to reference it to handle the pipeline — so the class and both its exports are deleted rather than kept as a name that can only ever produce a dead `except` clause. Backward compatibility is a non-concern at this stage.

### 2. `AuthoredColumn` carries facts; dials keep their one grammar ([#451](https://github.com/DEVunderdog/DataForgeML/issues/451), [#453](https://github.com/DEVunderdog/DataForgeML/issues/453), [#458](https://github.com/DEVunderdog/DataForgeML/issues/458))

- **The line, applying to both authors: hyperparameters are knobs the library has a default for; facts about the data live on the column.** `center1`/`center2`, `feature_cols`, the grouping variable and `Constant`'s fill are facts, kin to `domain_snap_bounds`. `central_tendency` is a dial — computed from the profile like `max_iter`, not measured.
- **A new input type, deliberately not `ColumnImputationDecision`** — no `column`, `semantic_type`, `drop` or `signals` to restate. The accepted cost is a **twin type that must not drift**: a new field on the plan's per-column type must be mirrored here. Taken because drift lands on maintainers once per field, where ceremony lands on every author on every column.
- **Flat optional fields, not one `strategy_facts` dict** — nine fields to fourteen on the per-column decision, most `None`. Keeps every fact typed and greppable, and avoids a second lookalike dict beside the dial map decision 4 licenses hard subscripts on.
- **The door takes estimators but not dials.** An estimator has no other way in (`with_model_choice` takes a *label*; there is no `with_estimator`), so `estimators={"mice": lgbm}` enters unit-keyed — `ImputationStrategy` is a `StrEnum`, so unit ids are strings an author can predict. Dials keep `with_hyperparameters`, one grammar and one merge rule (ADR-0073) whichever author made the plan. Fully dialling a hand-written plan is therefore two calls, not one.
- **Sentinels are plan-level keywords, not per-column fields** — they describe how the *frame* encodes missingness, and under `default=Passthrough` a sentinel-bearing column nobody has an opinion about has no `AuthoredColumn` to put them on. This takes `author()` from four parameters to six, against the keep-it-small rule, because a manual plan that cannot express `-999` is not indistinguishable downstream.
- **`columns` is required, keyword-only, an ordered `Sequence[str]` — never a DataFrame.** `transform` raises `UnseenColumnError` on any frame column absent from `records` and `compose` builds `records` from `column_decisions`, so a partial map without the universe builds an imputer that rejects its own author's frame ([#452](https://github.com/DEVunderdog/DataForgeML/issues/452)). Names, not a frame, because a decide-time door does not touch data.
- **`default` parameterises the `Passthrough` stamp**, buying repetition relief that does not grow with the frame. It **flips which case is silent**: under `default=Median` a forgotten column is quietly imputed rather than left alone. Unknown *keys* still raise; a forgotten column cannot, by construction.
- **The door stamps `SemanticType.Numeric` and refuses any other value.** `semantic_type` is not decoration — `_fit_context` builds `feature_columns` from every `Numeric` decision and MICE widens its predictors to all of them (ADR-0079), so the field *chooses MICE's predictor set*.

### 3. What the door refuses ([#453](https://github.com/DEVunderdog/DataForgeML/issues/453), [#458](https://github.com/DEVunderdog/DataForgeML/issues/458))

Two hazards **dissolved**: overlapping units are impossible because the input is column-keyed and `_derive_units` is a pure projection of it (ADR-0070's order-independence invariant is structural, not policed), and uncovered columns are impossible because `default` stamps every unnamed column.

The rest split on *the door checks exactly what the door can see*:

- **Authoring time raises** — an unknown column (a map key absent from `columns`); MICE with a single entry (one target, no predictors); bare `Constant` with no fill; either bimodal strategy without centres; `ClusterConditional` with neither a grouping column nor a non-empty `feature_cols`.
- **Fit time raises** via the existing `UnitNotTrainableError` — strategy/dtype mismatch and size guards, both of which need data.
- **Accepted asymmetry:** a manual author gets a hard `UnitNotTrainableError` where a *forced* automatic one gets a `FitSignals` warning. Parity would mean handing the door a frame, reversing the decide-time-doors-do-not-touch-data rule for one diagnostic.
- **Vocabulary**, by the test *does this answer "what happens to this column?"* — `Passthrough`, `Dropped`, `MNAR`, `Constant` (paired with a fill), `ClusterConditional` and `GMMSampling` are **allowed**; `Indicator` is **refused**, being the label of an appended column that arrives as a consequence, never as a choice. The door's vocabulary is deliberately a strict superset of `per_column_strategy`'s.
- **`MNAR` is a strategy, not a flag.** The router *returns* it and `indicator_flag` derives solely from `strategy == MNAR`, so the map is its one declaration site — no `author(config=)`, no `AuthoredColumn.mnar`.
- **Two silent no-ops in existing code are closed by the door's defaults**, both reachable only by hand: `ClusterConditional` with no grouping column and no `feature_cols` fits clean and fills **zero** nulls, and a bare MICE plan — the plainest thing an author would write — trains nothing because `_block_model_choice` finds no family. The door raises on the first and **defaults `BayesianRidge`** on the second, which is `decide()`'s own neutral branch.
- **A pre-fitted estimator is explicitly not refused** (see decision 5).

### 4. One dial table, read by both authors ([#456](https://github.com/DEVunderdog/DataForgeML/issues/456), [#457](https://github.com/DEVunderdog/DataForgeML/issues/457))

- **Three profile-derived "dials" are deleted from the decided bases outright** — they were facts about the data misfiled under decision 2's line, a pre-existing category error the manual door merely exposed. Verified: no fit path branches on any of them; they wrote one `FitSignals` note each and **nothing in `src/` reads `notes`**, so five keys die rather than three (MICE's `miss_frac`/`complete_frac` have no reader anywhere). Rejected: placeholders (a plan stating invented profile facts as truth) and `.get()`-plus-"unknown" — **this repo already ran that experiment**, printed `complete_frac=0.00` as though measured, recorded it as a defect and stripped it. `decide()`'s arithmetic is untouched; only the stashed copy left on the plan dies. Given up: a saved plan no longer explains why `n_neighbors` came out as it did.
- **The door writes a complete base eagerly, from one table `decide()` reads too.** There is no second table to drift from: `decide()` starts from the same row and overwrites what it computed, and neither side types a key literal — so adding a fifth dial is the same act as giving it a default. What made this cheap: **every one of `decide()`'s formulas already degenerates to the sklearn default at neutral inputs**, and the library had written those values down (`base_max_iter` = 10 = sklearn's `max_iter`; `knn_min_neighbors` = 5 = sklearn's `n_neighbors`). The table is the first set of opinions evaluated at neutral inputs, not a second set.
- **Four rows.** MICE (`max_iter`, `tol`, `initial_strategy`, `n_nearest_features`), KNN (`n_neighbors`, `weights`), and `MNAR` / `ClusterConditional` at `central_tendency = "median"`. **`GMMSampling` and `Constant` have no row at all**, so dialling them raises on every key for *both* authors — correct, since neither unit has anything to dial.
- **The invariant is made true of the type, not of two functions: `from_dict` gap-fills on load** — fill what is missing, never overwrite. An old artifact meeting a newly-added dial keeps running instead of dying on a bare `KeyError`, at the knowingly-taken cost of silently gaining a dial it was never saved with. That is what licenses the fitters' hard subscripts.
- **Rejected: the lazy shape** (empty map, table consulted at `_derive_units` and inside `with_hyperparameters`) — more truthful, but decision 6's case for amending ADR-0082 rests on the base being *carried*, not inferred.

### 5. A foreign estimator is held, never copied, never saved ([#449](https://github.com/DEVunderdog/DataForgeML/issues/449), [#450](https://github.com/DEVunderdog/DataForgeML/issues/450))

- **`ModelChoice` gains a `Custom` member**, and it is a *value* rather than `None` — `None` already carries the load-bearing "no estimator family, cannot train" meaning on the fit path. As a value, existing readers stay correct without learning anything. Given up: `ModelChoice` stops being a closed enumeration of families the library can *build*; one member means "look elsewhere".
- **The instance is the author's own object, never a clone**, on ingest or on any derived copy. Half the hazard dissolved first: `IterativeImputer` does `clone(self.estimator)` and clones again per column (verified, sklearn 1.9.0), and MICE is the only path an estimator reaches — so **executing a plan cannot mutate the estimator it holds**. Cloning was rejected because it breaks identity, makes `get_params` a *construction-time* requirement (stricter than sklearn, where validation defers to `fit`), and silently strips fitted state. What remains is deliberate post-authoring mutation, which moves every plan holding the object with nothing recording it — closed by documentation, the ADR-0075 informed-consent posture.
- **A pre-fitted estimator is tolerated.** Its state is inert (sklearn refits from scratch), and any check for it recognises only sklearn's trailing-underscore spelling — a guard catching one spelling reads as a guarantee it is not. So the Decision/Execution promise narrows from a structural guarantee to a statement about the library: **the library never writes learned state onto a plan.**
- **The estimator is never saved.** Both `to_dict` and `serialize` drop it; the plan reloads as `Custom` with the slot empty, and **`build_from_choice` raises** so a reloaded plan cannot silently impute with the median. Rejected: refusing to serialize such a plan (one class of plan that cannot leave its process is too high a price) and pickling it in (an artifact that opens only where xgboost is installed). Nothing learned is lost — the estimator is a line of the user's own code.
- **It stays in `==`**, so a save/load round trip on a `Custom` decision compares unequal. That is the truth: the restored plan cannot fit. The round-trip test gains a carve-out rather than the field being excluded to keep it green.
- **`core_budget` prices `Custom` at `1` in every branch** — no introspection, no new knob. The library never sets a foreign estimator's parameters, so `1` is the truthful count of cores *the library* spends; `total_cores` is already the reservation hatch. The budget is therefore blind to parallelism the user configured, so the documented pattern is to fit a self-parallelising unit outside the pool where it has the machine to itself. **No determinism guard and no warning**: wrapping breaks `clone`, and an `n_jobs` probe would fire on correct plans while missing `nthread`, `num_threads`, and every internal pool not exposed as a parameter.
- **Placement, reconciled.** [#449](https://github.com/DEVunderdog/DataForgeML/issues/449) put the instance in a per-column `custom_estimator` field; [#458](https://github.com/DEVunderdog/DataForgeML/issues/458) then **dissolved that field** on the finding that `build_from_choice` is called once in the whole file — MICE is the only strategy with an estimator slot, so the unit-keyed `estimators=` channel covers every case and a per-column spelling would let two estimators be named for one block. The plan therefore holds one unit-keyed estimator map, alongside `decided_hyperparameters` / `override_hyperparameters`; `ImputationUnit` itself stays estimator-free, and the block-coherence error class #449 deferred is structurally impossible rather than checked.

### 6. Re-authoring a decided plan: `author(map, base=plan)` ([#454](https://github.com/DEVunderdog/DataForgeML/issues/454))

**ADR-0082 is amended, not overturned.** Its case against `with_strategy` was a *defect* — the method carried `decided_hyperparameters` through untouched, so a new unit got no base and the fitter raised a bare `KeyError` — and decision 4's mandatory static base removes it. **The dividing rule stands** (a structural edit invalidates what is derived downstream of the unit list) while its consequence changes: **it must re-derive that state, not necessarily via `decide()`**. `with_strategy` stays deleted; the door rebuilds what it did not.

Carry-over is ruled field by field:

- `column_decisions` — carried, then overwritten by the map.
- Sentinel maps and `config_snapshot` — **verbatim**, since a re-authoring user may no longer hold the profile or the config.
- Tuning deltas — carried for surviving units, orphans **actively removed**. Not left inert: a silently-dead delta survives `to_dict` and a save/load and **springs back to life** when a column re-enters that unit.
- The decided base — **gap-filled, never overwritten**. `decide()`'s dials are profile-*computed*, so blanket recalculation would mirror ADR-0082's own bug as a silent quality regression instead of a loud crash.

Costs taken knowingly: **two provenances of settings in one plan**, and a knowingly-stale `n_neighbors` when a block's membership changes — kept because a structural edit hands dial responsibility to the author. `base=` is **mutually exclusive with `columns=`/`default=`** (two sources for one fact). **Mixed provenance gets no field** — `forced` was deleted for having no readers and a replacement would repeat that. **Re-authoring a deserialized plan is allowed, unguarded and documented**: the units saved beside it were fitted under the old strategies, so it yields a plan to fit fresh, never one to pair with earlier units.

### 7. Config retreats to decide-time ([#458](https://github.com/DEVunderdog/DataForgeML/issues/458), [#461](https://github.com/DEVunderdog/DataForgeML/issues/461))

- **`decide()` lowers `bimodal_grouping_variables` and `per_column_constant_fill` onto the column, and the fitters stop reading config.** One declaration site per fact, so a mixed `base=` plan needs no precedence rule and a saved plan fits without the config that produced it. Given up — **a behaviour change on the router path**: editing config between `decide()` and `fit_unit()` takes effect today and will silently stop.
- **`mice_max_iter` and `knn_n_neighbors` are deleted from `NumericImputationConfig`.** They say "always 30" — **the same sentence `with_hyperparameters` already says**, in the one grammar made canonical for dials. Two spellings for one meaning, one of them working on only one authoring path, *is* the asymmetry; deleting the path-bound spelling dissolves it rather than patching it.
- **The other `NumericImputationConfig` fields stay, untouched and decide-time-only.** They are not the same kind of thing: `base_max_iter` is the number `_compute_mice_max_iter` *starts from*, `knn_min_neighbors`/`knn_max_neighbors` are a floor and a cap on a computed `k` — all **operands in arithmetic `author()` does not run**, and `knn_max_neighbors` has no table row to apply to at all. Applying them at the door would redefine each field from "operand" to "the answer" depending on who wrote the plan.
- **The door stays config-free, and takes no `config=` parameter.** Decisive: after the lowering, **nothing in `src/` reads a plan's `config_snapshot`** — evaluation reads the orchestrator's config — so a config parameter would re-weaponise a field three tickets de-weaponised. Nothing is silently dropped either: a door that never receives a config cannot ignore one.
- **`from_dict` raises on either retired key.** Because it reads keys with `.get()` and silently ignores unknowns, plain deletion would manufacture this ticket's own complaint. Warn-and-drop was rejected as kind-looking and useless. A guard, **not a migration path** — backward compatibility is explicitly a non-concern, so no deprecation window, shim or translation.

## The guarantee boundary

Which library guarantees hold for *every* plan regardless of author, and which hold only where the library chose the estimator:

| Guarantee | Holds for a hand-authored plan? |
|---|---|
| Frame coverage — every column in `columns` gets a decision | **Yes**, by construction (`default` stamps the rest) |
| Observed-Value Preservation (ADR-0078) | **Yes** — enforced inside each unit's own `transform`, below the plan |
| Composition (`compose`'s exact-coverage gate, ADR-0071) | **Yes** — units are a pure projection of `column_decisions` |
| Execution (`fit_unit`, single-track raise, unit independence) | **Yes** |
| Persistence of the plan and of each fitted unit (ADR-0072) | **Yes**, except a `Custom` estimator, which is dropped and re-supplied |
| Sentinel normalisation at fit and transform | **Yes**, via the plan-level sentinel keywords |
| Dial validation (`with_hyperparameters` against a complete base) | **Yes** — the door writes the same base `decide()` does |
| `core_budget` arithmetic | **Yes for library estimators**; a `Custom` unit is priced at `1` and the budget cannot see parallelism the user configured |
| Determinism across `n_jobs` (ADR-0069's RandomForest pinning) | **Only where the library built the estimator** |
| Routing rationale (`signals`) | **No** — legally empty; there was no routing to explain |
| Size-guard warnings before fit | **No** — the door has no frame; a plan that cannot train raises at fit |
| Reload-and-fit without further input | **No** for a `Custom` plan — `build_from_choice` raises until the estimator is re-supplied |
| Tuning of the estimator's own dials | **No** for a foreign estimator — the library cannot enumerate dials it does not define |

## Consequences

- **Breaking changes on the automatic path**, all deliberate: `Passthrough` no longer raises on nulls; `mice_max_iter` and `knn_n_neighbors` are gone and `from_dict` raises on them; a config edit between `decide()` and `fit_unit()` stops taking effect; three plan fields and five hyperparameter keys disappear from the serialized format.
- **ADR-0082 is amended** — the structural/dial dividing rule stands, its "re-decide" consequence widens to "re-derive". ADR-0037 is **partly superseded** (both retired override fields were its contribution). ADR-0066 loses `decided_for_shape` and `forced`. ADR-0060's value-free claim narrows to a promise about the library. ADR-0050 owed a ruling on the now-raise-less `UnfittedColumnError`, settled at implementation in favour of deletion (see §1).
- **`_read_model_metadata`'s `knn_n_neighbors_override` parameter dies and `k_capped` becomes always computable** for a fitted KNN unit — trimming [#435](https://github.com/DEVunderdog/DataForgeML/issues/435)'s evaluation audit rather than adding to it.
- **Docstring scope** for `author`, `AuthoredColumn` and `ModelChoice.Custom` under ADR-0034 is not yet ruled.
- Three questions are left open by name: generalising the door to other modalities (categorical, boolean, temporal do not exist yet), authoring-time observability (nothing at the door warns today), and **provenance of authorship** — now green-field, since `forced` was its assumed anchor and was deleted; it introduces a correctly-named field only if a *reader* appears.

## Out of scope

- **Foreign strategies (Level 3)** — arbitrary user fill logic (interpolation, a domain formula, an external lookup) as a first-class unit. The argument for it is real: without it a user must impute those columns outside and join, which is the manual wrangling the library exists to eliminate, and it carries a silent train/serve skew hazard since the user's learned state never enters the serialized artifact. It lost anyway: that logic is code the user already owns and can re-run, and foreign units would break `core_budget`'s `model_choice` cross-reference, sit outside the enforcement point for Observed-Value Preservation, be untunable for want of enumerable dials, and force a two-tier guarantee surface. If it returns, it returns informed.
- **Structural re-routing by anything other than an author.** ADR-0082 stands: a strategy is declared, never inferred from an edit.
- **What a manual plan means to evaluation.** Authoring is upstream of scoring; the evaluation surface is being replaced by [#435](https://github.com/DEVunderdog/DataForgeML/issues/435). The two sub-questions worked before that ruling both returned *the manual door changes nothing, evaluation changes*.
- **MNAR exposing its fill rather than applying it** — [#459](https://github.com/DEVunderdog/DataForgeML/issues/459), since resolved by ADR-0098. A breaking behaviour change on the automatic path needing its own ADR; it concerns what a strategy *does*, not who wrote the plan. Its companion defect, `add_indicator_columns` being dead config, is [#460](https://github.com/DEVunderdog/DataForgeML/issues/460).
- **Hyperparameter tuning at large**, and **control over profiling** (already steerable through `ProfileConfig`).
