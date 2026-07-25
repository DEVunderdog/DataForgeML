# Per-unit fit observability is a structured, ephemeral `FitSignals` record returned from `fit_unit`, with warnings dual-channelled

Under ADR-0061 the executor reported fit-time facts as a Warning-Event stream on an `Emitter`, and the fitters returned loose `signals: tuple[str, ...]` of human-readable strings appended onto records. ADR-0071 removed the executor and made `fit_unit` return a `UnitFitResult`; this ADR specifies what rides in that result's observability slot and how warnings reach the user, resolving [Per-unit observability (#383)](https://github.com/DEVunderdog/DataForgeML/issues/383) on map [#377](https://github.com/DEVunderdog/DataForgeML/issues/377).

The governing choice is that observability becomes a **structured value the caller reads off the return**, not an event stream and not opaque strings — while genuine warnings *also* travel the standard Python `warnings` channel so an inattentive caller is not silently deprived of them.

Locked decisions:

- **`FitSignals` is a frozen structured record, not a string tuple.** It carries a typed **core** — `unit_id`, `strategy`, `estimator` (`str | None`), `converged` (`bool | None`), `n_iter` (`int | None`), `duration_s` (`float`), `warnings` (tuple of strings) — plus a free-form `notes` tuple for strategy-specific minutiae that do not earn a typed field. A per-strategy typed detail union was rejected: the typed core covers what any consumer branches on, and forcing every strategy's incidentals into the type system buys nothing a `notes` bag does not. `UnitFitResult` bundles `(unit: FittedUnit, signals: FitSignals)`.
- **`warnings` is split from `notes`.** They are distinct fields so a consumer can answer the cheap question "did anything go wrong with this fit?" by reading `warnings` alone, without scanning free-form notes for concerning prose. `notes` is informational; `warnings` is actionable.
- **`duration_s` is the whole-`fit_unit` wall time.** Measured with `perf_counter` around the entire `fit_unit` call, normalisation included. Under `fit_many`, which normalises once for the whole batch (ADR-0071), the per-unit `duration_s` therefore excludes the shared normalisation and slightly undercounts relative to a standalone `fit_unit` — this skew is documented, not corrected, because attributing shared normalisation cost per-unit would be arbitrary.
- **Warnings are dual-channelled: recorded *and* emitted.** Every warning is both stored in `FitSignals.warnings` and raised via `warnings.warn(..., category=ImputationFitWarning)`. Record-only was rejected: a caller who never inspects `signals` would miss the warning entirely, whereas the `warnings.warn` half pushes it to stderr by default and plugs into the standard `filterwarnings` toolkit (suppress, escalate-to-error, route to logging) with no library-specific API.
- **The forced-oversize warning is reconstructed structurally in the fitters.** It fires when a user forced a strategy past its routing threshold — KNN over `knn_max_rows` / `knn_max_features`, or regression under `regression_min_rows` — which ADR-0071 no longer blocks. There is **no forced flag** consulted; the condition is detected purely from the unit's shape against the config thresholds at fit time (the one genuinely new computation this ADR adds). It warns; it never raises.
- **Failure is payload-only; no `FitSignals` on a raise.** An untrainable unit raises `UnitNotTrainableError` with its structured payload (ADR-0071) and produces **no** `FitSignals` — there is no successful fit to describe. Observability describes fits that happened; failures are the exception's job.
- **`fit_many` keeps signals intact and stays fail-fast.** Its `{unit_id: UnitFitResult}` return carries each unit's `FitSignals` unchanged. On the first failure it raises clean (ADR-0071) — no partial dict, no signals attached to the exception. There is **no** live-progress channel: `on_unit_complete` callbacks and a revived `Emitter` were both rejected; a caller wanting per-unit progress writes the `fit_unit` loop.
- **`FitSignals` is ephemeral — never serialized.** It describes one fit's runtime, not the fitted state, so it is not part of any persisted artifact (ADR-0072). Consequently the in-unit observability fields that leaked fit-runtime facts into the fitted objects — `FittedRegression.signals`, `FittedRegression.max_iter_used`, and their siblings — are **stripped**: those facts live on `FitSignals` at fit time and nowhere after.

## Status

accepted; the `fit_many` clauses (normalise-once `duration_s` skew, batch fail-fast return) are superseded by ADR-0075 (`fit_many` removed) — every `duration_s` now spans its own normalisation

Consumes the charting leans of #383. Depends on ADR-0071 (`fit_unit`/`UnitFitResult` shape) and ADR-0072 (ephemerality — nothing here is persisted).

## Considered Options

- **Opaque `tuple[str, ...]` signals** (the shipped shape). Rejected: a consumer must string-parse to learn whether a fit converged or which estimator ran, and the strings drift with no schema. A typed core makes the common facts first-class.
- **Per-strategy typed detail union** (a distinct signals subtype per strategy). Rejected: over-engineered — no consumer branches on strategy-specific minutiae, and a free-form `notes` bag beside the typed core carries them at zero type-system cost.
- **Record-only warnings** (store in `FitSignals`, do not emit). Rejected: silently deprives an inattentive caller of warnings; the `warnings.warn` half gives stderr visibility and the standard filtering toolkit for free.
- **A forced flag carried on the unit** to detect oversize. Rejected: ADR-0066's `forced` flag was about degrade-vs-raise, now gone with degradation (ADR-0071); the oversize condition is fully determined by shape-vs-threshold and needs no stored flag.
- **A live-progress callback / revived `Emitter` in `fit_many`.** Rejected: it re-introduces the observer-threading machinery ADR-0071 removed, to serve a case the `fit_unit` loop already covers.
- **Emit `FitSignals` on failure too.** Rejected: there is no successful fit to describe, and splitting the failure story across an exception payload *and* a signals record is two places to look for one thing.

## Consequences

- **`FitSignals` and `ImputationFitWarning` are exported** from the Public API and documented (ADR-0034): the user reads `FitSignals` off every `UnitFitResult` and filters `ImputationFitWarning` through the standard toolkit.
- **`UnitFitOutcome`/the fitter return shape changes** from `(fitted, signals: tuple[str,...], fallback_reason)` to carrying a populated `FitSignals`; `fit_unit` wraps it into `UnitFitResult`. The fitters gain the `perf_counter` timing and the structural oversize check.
- **Fitted objects lose their observability fields** (`FittedRegression.signals`, `max_iter_used`, and siblings on `_fitted_units.py`), shrinking the persisted payload and removing runtime facts from state that outlives the fit.
- **`ColumnImputationRecord.signals`** — the append target the old string signals fed — is re-examined: routing rationale (a decide-time fact) stays on the decision; fit-time signals no longer flow into records, they ride on `FitSignals`.
- **The imputation-execution `Emitter`/Warning-Event usage is fully retired** (begun in ADR-0071); profiling's observability event stream (ADR-0054) is untouched.
