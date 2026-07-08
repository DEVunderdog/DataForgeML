# Thread-based concurrency as the standing model, with an independence rule

Orchestrator fitting runs sequentially today, summing the cost of every independent unit. We make it concurrent, governed by one **independence rule**: parallelize across units that share no data dependency — the mutually-independent strategy blocks (MICE, KNN, the Regression set, GMM, cluster), and the independent columns within a per-column strategy — and keep sequential only where one step consumes another's output (routing → fit) or where the algorithm is internally coupled (a joint model's columns, a solver's iterations). Concurrency never changes a result: same fixed seeds, same fold splits, executed in parallel.

We commit to **threads**, not processes, as the standing model, for two independent reasons: (1) the heavy work (RandomForest / GradientBoosting / BayesianRidge fits inside `IterativeImputer`) is native sklearn/NumPy/BLAS code that releases the GIL, so threads genuinely parallelize it; and (2) threads share memory, so the Progress Observer stays reachable and Substep progress (ADR-0055) survives concurrency. sklearn's own `n_jobs` is a free inner layer. To avoid core oversubscription from nesting, we run outer parallelism across our independent units and pin the innermost estimator to `n_jobs=1` when nested, reserving `n_jobs=-1` for a block running alone.

## Status

accepted

## Considered Options

- **Processes (`multiprocessing` / loky) for maximum CPU scaling.** Rejected as the default: worker processes cannot see the Observer callable, so progress would require a `multiprocessing.Queue` + listener bridge (worker enqueues picklable `PipelineEvent`s, parent drains them to the real Observer), plus shipping/serializing the DataFrame to each worker (often costing more than the compute saved) and losing clean event ordering. Kept as a **documented exception** for a proven pure-Python CPU bottleneck that cannot be vectorized; if taken, the Observer crosses via the queue-listener bridge, never by pickling the Observer itself.
- **asyncio.** Rejected: asyncio is concurrency for I/O waiting, not parallelism for CPU work. A `fit()` call has no `await` points and would block the single-threaded event loop for its whole duration, yielding serial execution. Offloading to `run_in_executor` just wraps a thread pool — the parallelism comes from threads, and asyncio adds `async`/`await` coloring for no benefit on CPU-bound math.

## Consequences

- Wall-clock for the imputation fit collapses toward the slowest single block (usually MICE) instead of the sum of all blocks; per-column strategies with many columns fan out.
- Threads will not scale a *pure-Python* hot spot (e.g. GradientBoosting's Python-level boosting loop) across cores; the escape hatch is to push that specific piece into NumPy, not to reach for processes.
- The independence rule and the thread model are the standing philosophy for all phases, current and future, not an imputation-only choice.
