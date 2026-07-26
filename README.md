# DataForgeML

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DEVunderdog/DataForgeML)

DataForgeML is an automated, end-to-end pipeline for turning raw tabular datasets into ML-ready data — without hand-writing schema logic, imputation rules, or splitting code for every new dataset.

The library is built as a one-stop solution: the goal is that a user should not need to manually assemble fragmented libraries steps or write custom glue code around it. Every phase — profiling, imputation, splitting — is designed to compose directly, driven by configuration objects rather than bespoke per-dataset scripts.

Supported input formats: CSV, TSV, Parquet, JSON, NDJSON, JSONL, XLSX, XLS, Arrow, and Feather, with automatic encoding and delimiter detection.

## Installation

Requires Python 3.11+.

```bash
uv add dataforge-ml
```

or, from PyPI with pip:

```bash
pip install dataforge-ml
```

## Quick start — profile a dataset

Structurally profile a raw dataset: inferred semantic types, per-column stats,
missingness, correlations, and nonlinearity signals — all driven by config.

```python
from dataforge_ml import (
    PipelineConfig,
    ProfileConfig,
    SemanticType,
    StructuralProfiler,
)

pipeline_config = PipelineConfig(
    profiling=ProfileConfig(
        compute_correlation=True,
        compute_nonlinearity=True,
    ),
    random_seed=42,
)

# Nudge the profiler where you know better than the type detector.
pipeline_config.set_column_type(column="Year", semantic_type=SemanticType.Categorical)

profiler = StructuralProfiler(config=pipeline_config)
result = profiler.profile(data=df)          # df is a polars.DataFrame

print(result.to_markdown())                 # human-readable profile report
```

## End-to-end — profile, split, impute (no leakage)

Profile the dataset, carve a stratified train/test split, fit numeric imputation
**on train only**, then apply the fitted imputer to test — so no test statistics
ever leak into training.

Imputation is user-orchestrated in three layers: `decide()` produces the pure
plan, your own loop trains each planned unit with `fit_unit()`, and
`FittedImputer.compose()` aggregates the trained units into the whole-frame
imputer. There is no fused entry point — batch scheduling is yours to own.

```python
from dataforge_ml import (
    DataSplitter,
    FittedImputer,
    PipelineConfig,
    ProfileConfig,
    SemanticType,
    StructuralProfiler,
    decide,
    fit_unit,
)

pipeline_config = PipelineConfig(profiling=ProfileConfig(), random_seed=42)
pipeline_config.set_column_type(column="Year", semantic_type=SemanticType.Categorical)

profiler = StructuralProfiler(config=pipeline_config)

# Stratified split, guided by the full-dataset profile.
full_profile = profiler.profile(data=df)
split = DataSplitter(df=df, target="Life expectancy", random_seed=42).profile_stratified_split(
    profile=full_profile,
    test_size=0.2,
)
train_df, test_df = split.train, split.test

# 1. Plan — a pure function of (profile, shape, config). Trains nothing.
train_profile = profiler.profile(data=train_df)
plan = decide(train_profile, len(train_df), pipeline_config)

# 2. Train — one call per planned unit; the loop is yours.
results = {
    unit.unit_id: fit_unit(
        plan, unit.unit_id, train_df, random_seed=pipeline_config.random_seed
    )
    for unit in plan.units
}

# 3. Compose — the whole-frame imputer, then transform test with the SAME state.
fitted = FittedImputer.compose(plan, results)
test_result = fitted.transform(df=test_df)
```

Only the cells that were missing receive fill values: every observed cell comes
back bit-for-bit, with its original dtype.

### Inspecting a fit

`fit_unit()` returns a `UnitFitResult` — the trained unit plus a structured
record of the fit, so "did this converge? which estimator ran? how long did it
take?" is answered without parsing strings:

```python
res = results["mice"]
res.signals.converged      # bool | None
res.signals.duration_s     # float
res.signals.warnings       # tuple[str, ...]
```

Every recorded warning is also raised through `ImputationFitWarning`, so it is
visible on stderr and filterable with the standard `warnings` toolkit. A unit
that cannot train raises `UnitNotTrainableError` rather than silently degrading
to a weaker strategy.

### Parallelising the loop

Parallelism lives in exactly one layer, and you choose which. A sequential loop
takes the default `n_jobs_inner=-1` (each fit fans out to every core). If you
fit units side by side in your own thread pool, call `core_budget` once before
the loop and pass each unit's value as that call's `n_jobs_inner`, so the two
layers do not oversubscribe the machine:

```python
from dataforge_ml import core_budget

budget = core_budget(plan, max_workers=4)
# inside your pool, for each unit:
res = fit_unit(plan, unit.unit_id, df, n_jobs_inner=budget[unit.unit_id])
```

A parallelised drive that passes nothing leaves every concurrent fit fanning out
to every core. The value never changes the result — only how the cores are
spent.

### Overriding hyperparameters

Estimator hyperparameters are resolved at decide time and carried on the plan,
so they are edited on the plan — before any training happens:

```python
plan = plan.with_hyperparameters("mice", {"max_iter": 25})
```

It is a per-key merge onto the decided base: named keys are overridden, every
other decided dial is kept, and unknown keys are rejected immediately. Pass
`None` to reset a unit to its decided values.

## Persistence

Fitted units, plans, and profiles persist through one pair of bare-bytes
functions — you decide where the bytes live (disk, object store, database):

```python
from dataforge_ml import deserialize, inspect, serialize

blob = serialize(results["mice"].fitted)
inspect(blob)          # header only — kind, versions, provenance; no unpickling
unit = deserialize(blob)
```

`inspect()` reads only the JSON header, so an artifact's `kind`, format/library
versions, and `produced_with` provenance stamp can be checked *before*
`deserialize()` unpickles anything. A whole `FittedImputer` has no aggregate
format on purpose: persist its plan and its units, then rehydrate through
`FittedImputer.compose()`.

## Evaluation

Fit-quality diagnostics are opt-in and live in their own stateless orchestrator.
Nothing is retained between calls — you hand the fitted imputer and the data
back in:

```python
from dataforge_ml import EvaluationOrchestrator

evaluator = EvaluationOrchestrator(config=pipeline_config)

report = evaluator.inspect(fitted, train_df)                    # cheap, no retraining
accuracy = evaluator.score_accuracy(fitted, train_df, train_profile)  # cross-validated
```

`inspect()` reuses the models the fit already learned and compares filled cells
against observed values. `score_accuracy()` is the expensive, honest held-out
measurement per model-based column, and refits.

## Examples

To see how to use DataForgeML end to end, browse the full set of runnable,
self-contained example scripts — each with bundled datasets and step-by-step
walkthroughs — in the dedicated examples repository:

**https://github.com/DEVunderdog/dataforgeml-examples**

## Observability

Profiling and imputation calls can run for minutes by design (the library computes
every available indicator). To see live progress and keep a diagnostic trail, each
long-running call emits a single stream of **Pipeline Events**. The library is
**silent by default** — you opt in per call.

### 1. Live progress in one line

Pass the shipped `stderr_observer` as `observer=` to watch progress on stderr:

```python
from dataforge_ml import StructuralProfiler, stderr_observer

profiler = StructuralProfiler(config=pipeline_config, observer=stderr_observer)
profiler.profile(data=df)
# [profiling] modality started
# [profiling] column_profiling: age (1/7)
# [profiling] correlation: feature-feature matrices (1/1)
# [profiling] nonlinearity: income (2/3)
# ...
```

The same `observer=` argument is available on `EvaluationOrchestrator`, and you
can pass a different observer (or none) to each call independently.

### 2. Your own observer

An observer is just a `Callable[[PipelineEvent], None]`. Each event carries raw
`index`/`total` fields (drive a progress bar without parsing text) and an
`event_type` you can filter on:

```python
from dataforge_ml import EventType, EvaluationOrchestrator, PipelineEvent

def my_observer(event: PipelineEvent) -> None:
    # Keep only per-item heartbeats; ignore stage boundaries.
    if event.event_type is EventType.item:
        pct = 100 * event.index / event.total
        print(f"{event.stage}: {event.column} — {pct:.0f}%")

EvaluationOrchestrator(config=pipeline_config, observer=my_observer)
```

`EventType` values: `stage_start`, `stage_end`, `item`, `decision`, `warning`.

### 3. Durable trace via standard logging

Independently of any observer, every event is also written to the named
`dataforge_ml` stdlib logger (level derived from its `event_type`:
`item` → DEBUG, `decision` → INFO, `warning` → WARNING). It stays silent until
you attach a handler, so you integrate it with your existing logging without the
library forcing a stack on you:

```python
import logging

logging.getLogger("dataforge_ml").addHandler(logging.StreamHandler())
logging.getLogger("dataforge_ml").setLevel(logging.INFO)
```

Progress is scoped **per orchestrator call** — there is no global "% of the whole
pipeline", because the library is driven as discrete, stateless calls.

## Documentation

Full documentation — including the complete API reference, configuration guide, and the domain concepts behind profiling, imputation, and splitting — is available at:

**https://devunderdog.github.io/DataForgeML/**

## License

MIT
