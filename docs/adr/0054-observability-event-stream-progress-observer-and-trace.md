# Observability: a unified Pipeline Event stream feeding a per-call Progress Observer and a named-logger Trace

Long-running orchestrator calls (profiling, imputation fitting, correlation/nonlinearity) previously ran silently for minutes, leaving users unsure whether the run was progressing or hung, and leaving no diagnostic trail. We add observability as a **single stream of Pipeline Events** (`phase`, `stage`, `column`, `index`, `total`, `message`, `event_type`) emitted at phase/stage boundaries and per-item inside the expensive loops. Every Event goes to two sinks, always: the `dataforge_ml` stdlib logger (Trace, silent by default via `NullHandler`, level derived from `event_type`) and — if supplied — a user-provided **Progress Observer** passed as a live `observer=` argument on each orchestrator (Progress). The library never writes to a stream on its own except through a shipped default observer the user opts into, which writes to stderr.

## Status

accepted — the "solver-iteration granularity is deferred / column-level is the item floor" consequence is amended by ADR-0055, which lowers the floor to Substep granularity (still above solver-iteration) via a `substep` event type and an explicit Emitter.

## Considered Options

- **`verbose=True` and the library prints its own progress.** Rejected: hard-codes the UI and output stream, adds a dependency (e.g. tqdm), and cannot be redirected to a file, notebook, or web UI. Fights the advanced-developer / full-configurability principle.
- **Observer as a field on `PipelineConfig`** (read once, used everywhere). Rejected: a live callable cannot be serialized, breaking the config's `to_json`/`from_dict` and setter-only declarative contract (ADR-0044). The observer is a runtime collaborator, not declarative config, so it lives on the orchestrator constructor instead — accepting that the user passes `observer=` to each of the three orchestrators.
- **Declarative observability knobs (`ObservabilityConfig` sub-config) to toggle event tiers/verbosity.** Rejected as unnecessary machinery: because the observer receives structured Events and filters by `event_type` itself, and Trace verbosity is controlled by standard `logging` on the `dataforge_ml` logger, no library-side switch is needed. The library always emits; consumers filter.
- **Hand the observer a preformatted string instead of a structured Event.** Rejected: a string caps consumers at printing — a real progress bar or structured log needs `index`/`total`, not a sentence to parse. Events carry both the raw fields and a ready-made `.message`.
- **A third-party logging library (loguru, structlog) for the Trace sink.** Rejected: a *library* must not impose a logging stack on the applications that consume it. Trace uses stdlib `logging` only, with no new logging dependency. Because we emit through the named `dataforge_ml` logger, a consumer who prefers loguru/structlog routes our logger into it on their side in one line — they lose nothing, and we force nothing. Structure lives in the `PipelineEvent` object itself, so structlog buys us nothing here. A future contributor should not "upgrade" this to a third-party logger.
- **Two separate streams for progress vs. trace.** Rejected in favour of one Event stream with two sinks: identical Events, each sink filtering its own subset, no duplicate plumbing.

## Consequences

- **No global/whole-pipeline progress.** The library is driven as discrete stateless calls with no run object spanning them, so progress is honestly scoped **per orchestrator call** ("column 7/12 in this fit"), never "% of the entire pipeline."
- **Solver-iteration granularity is deferred.** The item floor is column-level; MICE/KNN internal rounds are not reported yet. The Event shape (with `index`/`total`) is designed so an iteration counter can be added later without a breaking change.
- **Recoverable soft fallbacks now surface** as `warning`-type Events (e.g. non-convergence → median). Hard failures continue to raise exceptions.
- Public API additions to keep stable: the `observer=` constructor argument on each orchestrator, the `dataforge_ml` logger name, and the Event's public fields and `event_type` vocabulary. `PipelineEvent` and an `EventType` enum are exported from `dataforge_ml.__init__` (the observer is a public extension point, so the type a user implements against must be referenceable — ADR-0050) and `event_type` is an enum, not bare strings, matching every other categorical in the API (`SemanticType`, `ImputationStrategy`).
