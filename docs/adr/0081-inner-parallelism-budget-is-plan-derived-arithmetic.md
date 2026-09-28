# Inner-parallelism budget is plan-derived arithmetic, not a scheduler

ADR-0075 handed batch scheduling to the user and deleted `fit_many`, leaving `fit_unit`'s docstring to carry the whole concurrency story: *"A self-parallelised drive — units fitted side by side in your own thread pool — must pass `n_jobs_inner=1` on every call"* (`_unit_fit.py:180-188`). That advice is a measured **5.84x regression**, and this ADR replaces it with a free function that computes the budget from the plan.

The advice assumes a batch with enough units to fill the machine. A real plan does not have one. `_derive_units` emits **at most two heavy units** — the single joint `mice` block and the single joint `knn` block (`_JOINT_BLOCK_STRATEGIES`, `_config.py:1331`) — and after ADR-0079's widening the scalar columns are pulled into the MICE block as *predictors* rather than surviving as units at all. At 2352 rows the measured plan was **two units**, one of which cost 0.01 s. Pinning every call to `1` therefore runs the entire fit on 1 of 12 cores and leaves 11 idle. Measured (#427, 12 cores, `max_workers=8`): the all-ones pool takes **364.92 s** where the heavy-aware budget takes **62.45 s** and a plain sequential drive takes **64.24 s**.

The shape of that result is what this ADR is built on: the budget does not *beat* the sequential drive, it **recovers** it (within 3%, both directions — noise). The defect being fixed is not "we lack concurrency"; it is that the library's own documented concurrency advice is slower than doing nothing.

Deriving the right number by hand requires three library-internal facts, which is why it cannot stay the user's problem: that only MICE forwards `n_jobs_inner` (`_fitters.py:303`), that only the RandomForest branch has an `n_jobs` to receive it (ADR-0069 — `BayesianRidge` and `GradientBoostingRegressor` have none), and that `KNNImputer` has no `n_jobs` at all.

Locked decisions:

- **The library owns the arithmetic; the user keeps the pool and the loop.** ADR-0075 stands **unamended** and this is explicitly not a revival of `fit_many`. What moves into the library is one computation — "how many cores does this unit want" — and nothing else: no pool, no submission order, no exception policy, no straggler handling. The user still writes the `ThreadPoolExecutor` loop and still owns the failure semantics ADR-0071 gave them.

- **The arithmetic is heavy-aware, not degree-proportional.** `mice` and `knn` are heavy; every other unit is dust. `mice` receives `max(1, total_cores − 1_if_knn_present)`; every other unit receives `1`. When `max_workers == 1` the drive is sequential and outer-degree-one, so `mice` receives `-1` per ADR-0069's fourth locked decision. The reserved core for `knn` reflects that the KNN block, while cheap, is a real concurrent claimant when the pool runs both blocks side by side.

- **`max_workers=None` is read as a pool of unknown degree**, taking the same branch as `max_workers > 1`. `ThreadPoolExecutor(max_workers=None)` is legal and common, and it is emphatically not sequential; reading it as `1` would hand `mice` a `-1` while other units fit alongside it, oversubscribing exactly as today's un-pinned pool does. It is not rejected with a `ValueError`, because the number the user does not know is the number they came here to avoid computing.

- **Core detection is `joblib.cpu_count()`, with an optional `total_cores` override.** joblib is already a direct dependency (`pyproject.toml:24`). It respects cgroup quotas — `os.cpu_count()` returns 64 inside a 2-CPU container — and it is the same detector sklearn uses for its own `n_jobs=-1`, so our budget and sklearn's actual fan-out agree on how big the box is rather than disagreeing by the quota factor. The override exists for callers who are subdividing a box across processes and know something the process cannot detect.

- **A free function, `core_budget(decision, max_workers, total_cores=None) -> dict[str, int]`**, living in `_unit_fit.py` beside `fit_unit` and exported. It is deliberately **not** a method or property on `ImputationDecision`: the plan is documented as derived purely from `(profile, shape, config)`, and a machine fact hung on it would make the same serialised plan answer differently on a different box. This follows ADR-0072's bare-free-function precedent.

- **It returns a whole-plan mapping of `unit_id → n_jobs_inner`, not a per-unit query.** The arithmetic *is* whole-plan: the `1_if_knn_present` term is a fact about the plan, not about the unit being asked after. A per-unit signature would rescan the plan on every call and would invite the reading that units are independently priced — which is the naive `cores / outer_degree` model this ADR rejects. The mapping is also inspectable: a user can log or eyeball the whole budget before committing to a drive.

- **`core_budget` reads `model_choice` itself and folds the answer into the number.** A MICE block whose columns route to `GradientBoostingRegressor` or `BayesianRidge` gets `1`, because those estimators have no `n_jobs` to spend (ADR-0069).

- **One new field: `ImputationUnit.is_block`** — a pure fact derivable from `strategy` alone (membership in `_JOINT_BLOCK_STRATEGIES`). There is deliberately **no `wants_inner_jobs` companion**: that would depend on `model_choice`, which `ImputationUnit`'s own docstring says "lives on its `ColumnImputationDecision` … so it is never duplicated here" (`_config.py:1411-1414`). The unit stays value-free and estimator-agnostic; `core_budget` does the cross-referencing.

- **`fit_unit` keeps `n_jobs_inner=-1` as its default.** ADR-0075's reasoning survives `core_budget`'s arrival intact: the sequential one-at-a-time drive is the primary shape, it is outer-degree-one, and per ADR-0069 it deserves wide inner parallelism. `core_budget` returns `-1` for `mice` at `max_workers=1` anyway, so a sequential driver gets the same number whether or not they call it — the function is purely additive, and flipping the default would make the plain loop, which every example uses, quietly single-core.

- **No frame-size floor.** Charting suspected that on a small frame the per-column-per-iteration fits might be too small for threading to pay for itself. Measured false: at 600 rows the wide budget is already **2.87x** faster than the all-ones pool, and the margin *grows* with frame size (2.87x → 5.84x). Any floor that exists is below 600 rows, where the whole fit costs 2 seconds.

- **`fit_unit`'s concurrency docstring paragraph is rewritten to point at `core_budget`.** "Pin `n_jobs_inner=1` on every call" becomes wrong advice the moment `core_budget` exists and must not survive as a second, slower story. The replacement: a sequential loop takes the default; a self-parallelised drive calls `core_budget` once and passes the per-unit value. The oversubscription warning stays — it is still what happens if you parallelise and pass nothing — but it now names the fix instead of prescribing the pin.

## Status

accepted; supersedes the pin-when-parallel documentation clause of ADR-0075 (the locked decision that "a user who parallelises `fit_unit` calls themselves must pass `n_jobs_inner=1`"). ADR-0075's removal of `fit_many` and its retention of the `-1` default are unaffected. ADR-0069's layer rule is unamended and is the authority `core_budget` implements.

Amended by [What fit_unit consumes](https://github.com/DEVunderdog/DataForgeML/issues/532) (map #528): the signature is `core_budget(routing, max_workers, total_cores=None)`. Every input (the units via `derive_units`, whether a KNN unit exists, `mice_model_choice` including `Custom`) is on `ImputationRouting`, and no dial changes the answer. A caller holding a recipe passes `recipe.routing`. Gained: the machine can be priced before a recipe is resolved, and the budget cannot look dial-dependent. Given up: one attribute hop. It relies on `mice_model_choice == Custom` staying on routing wherever the custom estimator object ends up. The free-function reasoning below stands, with "plan" read as "routing".

## Considered Options

- **Leave the arithmetic to the user and improve the documentation.** The zero-surface option, and consistent with #377's handing scheduling back. Rejected because the three facts the derivation needs — MICE is the only `n_jobs_inner` forward, RandomForest is the only receiver, `KNNImputer` has none — are all library internals, two of which are undocumented implementation details subject to change. Documenting them would freeze them as public contract, which is a *larger* commitment than exporting one function that reads them.

- **Naive degree-proportional budgeting: `total_cores / outer_degree` per unit.** The obvious model, and it fixes nothing: with two units and `max_workers=8` it hands `mice` a `1` and reproduces today's regression almost exactly. It prices a `Median` fit and a MICE block as equal claimants on a core — over a crowd that, when it exists at all, exists for 5 ms of a 400 s fit.

- **Revive `fit_many` as a heavy-aware scheduler.** Rejected during charting. It would have to own the pool, the exception policy and the submission order, all of which #377 deliberately handed back to the user and ADR-0075 removed three weeks before this effort began. The measured complaint is the *arithmetic*, not the loop — the user's `ThreadPoolExecutor` was never the problem.

- **Reorder `_derive_units` to emit heavy units first.** Submission order genuinely matters at small `max_workers` (a `mice` unit submitted behind a `knn` block costs the whole KNN runtime before it starts). Rejected in favour of exposing `is_block` and letting the user order their own submissions: `_derive_units` emitting in column order mirrors the frame and is what makes an inspected plan legible, and reordering it to serve a scheduling concern would make the plan's own presentation a concurrency artifact.

- **Flip `fit_unit`'s default to `n_jobs_inner=1`** so a hand-rolled pool cannot thrash. Rejected for the reason ADR-0069's fourth locked decision and ADR-0075 both give: it makes the sequential drive single-core at both layers unless the user knows to pass `-1`, punishing the common shape to protect the rare one. `core_budget` protects the rare one directly.

- **Warn when the heavy unit cannot spend a budget** (a MICE block routed to `GradientBoostingRegressor`). Rejected: `core_budget` returns `1` there because `1` is the *truthful* answer, so the warning would fire on a plan with nothing wrong with it. Noise on a correct result is worse than a documented condition.

- **Raise `gradient_boost_min_rows` so RandomForest reaches further up** and the budget stays live on large frames. Rejected as out of this map's authority: that threshold is an accuracy decision (GBR is judged the better model on large complex-nonlinear data), and moving it to win cores is choosing a worse model to go faster.

- **`total_cores` from `os.cpu_count()`.** Rejected: it ignores cgroup quotas, so a containerised budget would be wrong by the quota factor and would disagree with what sklearn's own `n_jobs=-1` does inside the same container.

## Consequences

- **The concurrency story becomes one story.** A user calls `core_budget` once before their loop and passes the per-unit value; a sequential user calls nothing. `fit_unit`'s docstring stops prescribing a number and starts naming a function.

- **The budget is a snapshot, and a stale one is silent.** `core_budget` is computed against a plan; a `with_hyperparameters` edit (ADR-0073) or any other plan edit between the call and the loop yields numbers that no longer describe the plan being fitted. Nothing detects this. The mitigation is placement — compute it immediately before the loop — not enforcement.

- **A plan with very many dust units oversubscribes briefly.** The arithmetic gives `mice` a wide budget regardless of how many GMM/cluster units are in flight beside it, so a plan with a large crowd of them momentarily runs more threads than cores. Accepted: too many cheap threads degrades gracefully, and starving MICE does not. The measurement additionally found the crowd is largely hypothetical — ADR-0079's widening absorbs scalar columns into the MICE block as predictors, and at 2352 rows no dust unit existed at all.

- **On default config the budget goes dead for `ComplexNonlinear` frames above 10 000 rows.** `_regression_estimator_factory.py:118` routes that branch to `GradientBoostingRegressor` at `n_rows >= config.gradient_boost_min_rows` (default `10_000`, `_config.py:284`), and GBR has no `n_jobs`. `core_budget` returns all ones, correctly, with no error and no speedup. This is **documented, not warned or worked around**: the condition is named in `core_budget`'s docstring. It is branch-specific — `MonotonicNonlinear` takes RandomForest at any row count — so large frames can still spend a budget, just not through the `ComplexNonlinear` path, which is the one that gets expensive enough to care.

- **`ImputationUnit` gains a field and therefore a serialisation surface.** `is_block` is derivable, so `from_dict` must reconstruct rather than trust it, keeping ADR-0063's contract honest against payloads written before the field existed.

- **ADR-0056's determinism promise is untouched.** `n_jobs_inner` never moves a result (ADR-0069): the RandomForest fit is bit-identical across the switch and `_CoreInvariantRandomForest` pins prediction. `core_budget` changes wall-clock shape and nothing else, so it needs no determinism test of its own beyond the existing core-invariance gate.

- **Planning-only.** This ADR locks the design; the code change — `core_budget`, `ImputationUnit.is_block`, the `fit_unit` docstring rewrite and the exports — is a follow-on effort. `CONTEXT.md` is not updated until that lands, since it documents live behaviour.
