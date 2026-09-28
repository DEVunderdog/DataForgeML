# Hyperparameter overrides are a per-key plan-edit merge onto an always-complete decided base

> **Amended by [Where dial defaults and profile estimates are resolved](https://github.com/DEVunderdog/DataForgeML/issues/531) (map #528, ADR-0089).** The per-key merge, the table-as-schema check and `None`-resets stand. Both maps and `with_hyperparameters` move from the plan to `ImputationRecipe`. Units carry no merge; `recipe.hyperparameters(unit_id)` returns it. Resolving the recipe again starts with an empty delta.

ADR-0062 made hyperparameters a decide-time fact carried on the plan: each unit holds the concrete dials (`max_iter`, `tol`, `n_neighbors`, …) resolved from `(profile, shape, config)`. But the fitters still read those dials defensively — `hyp.get("tol", 1e-3)`, `hyp.get("max_iter", ctx.config.base_max_iter)` — treating the carried set as possibly-incomplete and re-deriving missing keys from config, and there was no first-class way for a user to override a single dial without replacing the whole set. This ADR specifies the override channel, resolving [Hyperparameter override channel (plan-edit merge) (#382)](https://github.com/DEVunderdog/DataForgeML/issues/382) on map [#377](https://github.com/DEVunderdog/DataForgeML/issues/377). It sits alongside the surviving pure-decision layer (ADR-0060) and does not touch routing.

The channel is a **plan edit**, consistent with the immutable-decision doctrine: an override returns a new `ImputationDecision`, never mutates one. The design turns on one choice — override is a **per-key merge onto a complete base**, not a replace — which is what finally lets the fitters stop guessing.

Locked decisions:

- **The plan holds two maps; the unit carries their merge.** `decide()` computes the **decided base** — always complete for the unit's strategy — and only `decide()` writes it. A separate **sparse override delta** is written only by the editor. `_derive_units` stamps `merged = decided ⊕ delta` (per-key, delta wins) onto `unit.hyperparameters`, so a fitter reads one complete dict and the two-map split is invisible below the plan surface. Overriding `max_iter` leaves the decided `tol` intact — this per-key merge *is* the "keep most, change one" case that a full-replace could not express.
- **API: `plan.with_hyperparameters(unit_id, dict | None)`.** Renamed from `with_unit_hyperparameters`. It takes a **dict** of overrides (not kwargs) and is **per-unit only** (not per-column — a joint unit shares one estimator, so per-column overrides on it are meaningless). Passing `None` **resets** that unit to its decided base by clearing the unit's delta. Like every plan edit it returns a new immutable `ImputationDecision`.
- **The decided base is the schema; any decided key is overridable.** There is **no** dial-vs-diagnostic allowlist — the set of keys `decide()` populated for a strategy *is* the legal override surface. An override to a key the decided base does not carry **raises at edit time** (`with_hyperparameters`), because "a key the strategy doesn't use" and "an unknown key" are the same case: the base is the authority on what the strategy uses. There is **no value-type check** — an ill-typed value (a string where sklearn wants an int) is rejected by sklearn at fit time, and duplicating sklearn's parameter validation in the editor is a maintenance liability with no added safety.
- **The unit exposes the merged set only; decided-vs-override is a plan-level query.** `unit.hyperparameters` is the merged dict — what actually fits. "Which of these did the user override?" is answered at the plan level by reading the delta map, not by a second field on every unit. Inspection stays cheap and the unit stays a single source of truth for the fitter.

## Status

accepted

Consumes the charting leans of #382. Complements ADR-0062 (decide-time hyperparameters) by giving them an override channel and removing the fitter-side fallbacks that treated the carried set as incomplete.

## Considered Options

- **Full-replace override** (`with_hyperparameters` replaces the unit's whole dict). Rejected: it forces a user changing one dial to re-supply every other, re-introducing exactly the incompleteness the fitters were defending against — and a user who forgets one key silently gets a fitter fallback instead of the decided value. Per-key merge onto a complete base is what lets the fitters stop guessing.
- **A dial-vs-diagnostic allowlist** naming which keys a user may override. Rejected: the decided base already enumerates the strategy's real parameters, so a separate allowlist is a second thing to maintain and drift; "any decided key" is the same surface with no extra bookkeeping.
- **Value-type validation at edit time.** Rejected: it duplicates sklearn's own parameter validation, must track sklearn's evolving type rules, and adds no safety a clear fit-time error does not already give.
- **A per-unit `overrides` field on the unit** (decided and override kept side by side on every unit). Rejected: the fitter only ever wants the merged set, so the split belongs on the plan (which owns edits) rather than duplicated onto every materialised unit where it would have to be re-merged on read.

## Consequences

- **Strip every fallback in `_fitters.py`.** Because the merged set is now guaranteed complete for the unit's strategy, `hyp.get("tol", 1e-3)`, `hyp.get("max_iter", ctx.config.base_max_iter)`, `hyp.get("initial_strategy", "mean")`, `hyp.get("n_neighbors", 5)`, and every sibling become plain `hyp["..."]`. A missing key is now a bug in `decide()`'s base construction, surfaced loudly as a `KeyError`, rather than a silent config re-read that could disagree with the plan.
- **Close the KNN `complete_frac` gap.** The KNN fitter reads `hyp.get("complete_frac", 0.0)` but the decided base never populates it — the fitter was silently defaulting. `decide()`'s KNN base must now populate `complete_frac`, so the stripped `hyp["complete_frac"]` resolves. This is a real behaviour fix uncovered by removing the fallback.
- **The delta must persist in the decision envelope** (ADR-0072): the sparse override map is plan state not reconstructible from `decide()`, so `serialize(decision)` carries both maps and `deserialize` restores a plan that re-stamps the same merged units.
- **Editing an unknown key fails fast and locally** at the `with_hyperparameters` call, naming the unit and the offending key, rather than surfacing as a confusing fit-time error three steps later.
