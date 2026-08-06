# Units are selected by strategy and passed by value, never named by id

The user-orchestrated door hands the caller a plan and asks them to drive their own fit loop (ADR-0071, ADR-0075). Concurrency is deliberately theirs — the library plans the work and prices the inner layer (ADR-0081), but owns no pool. That bargain only works if the caller can see what the units *are*, because a pool cannot be sized against work you cannot enumerate.

Enumeration was never the missing piece: `ImputationDecision.units` has always been there. What was missing was a way to *select* from it. A caller who wanted the MICE block wrote `next(u for u in plan.units if u.unit_id == "mice")` and then handed `"mice"` back to `fit_unit` as a bare string. The id was the currency of the whole surface, and the id is a hand-typed literal that nothing checks.

The failure mode is worse than a typo raising. `unit.unit_id == "MICE"` is a silent miss inside an `if`: the block is skipped, the loop completes, and the plan trains short with no error at all. The library's own docs shipped this pattern (`docs/api/imputation.md`), so it was not merely permitted — it was taught.

Locked decisions:

- **`ImputationDecision.units_for(strategy) -> tuple[ImputationUnit, ...]` is the selection surface.** Selection keys on `ImputationStrategy`, not on the id, because "I want the MICE unit" is a question about strategy — `"mice"` is only how that strategy's block happens to be named. The enum is checked by the type checker, so the whole class of misspelling stops existing. The id returns to being an internal name for the plan's own bookkeeping and for serialisation, which is all it was ever meant to be.

- **It always returns a tuple, and the empty case is a zero-length one.** A plan that routed nothing to MICE — few rows, all-categorical, every numeric column on a scalar — is an ordinary outcome, not an error. Returning `None` would push a guard onto every call site and reward forgetting it with `AttributeError: 'NoneType' has no attribute 'unit_id'`, a *worse* message than the `KeyError` the string produced. Raising would make an ordinary question into control flow. The tuple needs no guard: a loop over `()` runs zero times.

- **There is no singular `block(strategy)` accessor.** `MICE` and `KNN` yield at most one unit each, and a singular accessor would read better at the call site for exactly those two. It was rejected: it is a second method whose validity depends on its argument, and it reintroduces the `None`-or-raise question that the tuple just settled. The at-most-one guarantee stays an invariant the plan enforces internally rather than one the API has to advertise. Given up — `for unit in plan.units_for(MICE)` reads oddly when you know there is one, and `(unit,) = plan.units_for(MICE)` is the idiom for when you want to assert it.

- **`fit_unit` takes the `ImputationUnit`, not its id.** A selector alone would have removed one hand-typed string and left the other: the caller would pull a unit out and pass `.unit_id` straight back. Taking the unit closes the round trip, and the two changes together mean no unit id is spelled at any call site in a correct program.

- **A union `ImputationUnit | str` was rejected.** It would have been additive and forced no version bump, and that is precisely the problem: it leaves the hand-typed literal legal forever, so the API would document one way and permit two. The stated goal was to make the error unavailable, and the union does not deliver it. The one case that would have justified `str` — resuming from a stored id — is not live, having died with the executor (ADR-0071).

- **The passed unit is re-resolved against the plan by id; the plan always wins.** `fit_unit` reads only `unit_id` off the argument and looks the recipe up in `decision.units`. Training *from* the passed object would make it a second source of truth for the hyperparameters, so a unit held from before a `with_hyperparameters` edit would silently train the stale recipe. Resolving by id makes a stale unit harmless. Given up — the signature now looks like it accepts a recipe when it only accepts an identity, which is documented on the parameter rather than expressed in the type.

- **A string argument raises `TypeError` naming the replacement.** Without it the pre-4.x call shape dies on `'str' object has no attribute 'unit_id'` deep inside a private helper — the worst possible message for the one person guaranteed to hit it, the user migrating off the old signature.

- **`ImputationUnit` is exported from `dataforge_ml`.** The export principle is that a symbol is public iff the user must reference it to configure, drive, or handle the library (ADR-0050). Once the user holds a unit and passes it back, they must be able to name its type.

Consequences:

- Breaking change to `fit_unit`, the one training primitive. Every call site changes; the migration is mechanical, and the `TypeError` catches any that are missed at the first call rather than at import.
- `core_budget` still returns `dict[str, int]` keyed by unit id, and that is deliberate. `budget[unit.unit_id]` reads the id off an object the plan handed over — the value is never spelled in user code, so it carries none of the hazard this ADR removes. Re-keying the budget by unit would have bought nothing and cost hashability guarantees.
- `ImputationDecision.with_hyperparameters(unit_id, ...)` still takes an id string and is left as found. It is on the safe side of the same line by accident rather than design: it validates the id eagerly and raises `KeyError` at edit time, so a typo is loud and immediate rather than a silent skip. Worth revisiting for consistency; not a defect.
- ADR-0075 is unamended. Batch scheduling stays user-owned — this ADR changes what the user passes, not who owns the loop.
