# DataForgeML

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DEVunderdog/DataForgeML)

DataForgeML is an automated, end-to-end pipeline for turning raw tabular datasets into ML-ready data — without hand-writing schema logic, imputation rules, or splitting code for every new dataset.

The library is built as a one-stop solution: the goal is that a user should not need to manually assemble fragmented libraries steps or write custom glue code around it. Every phase — profiling, imputation, splitting — is designed to compose directly, driven by configuration objects rather than bespoke per-dataset scripts.

Supported input formats: CSV, TSV, Parquet, JSON, NDJSON, JSONL, XLSX, XLS, Arrow, and Feather, with automatic encoding and delimiter detection.

## Installation

Install from PyPI

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

```python
from dataforge_ml import (
    DataSplitter,
    ImputationExecutor,
    PipelineConfig,
    ProfileConfig,
    SemanticType,
    StructuralProfiler,
    decide,
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

# Fit imputation on train, then transform test with the SAME fitted state.
# Imputation is layered: decide() plans, the executor trains, build() aggregates.
train_profile = profiler.profile(data=train_df)
plan = decide(train_profile, len(train_df), pipeline_config)

executor = ImputationExecutor(
    plan,
    train_df,
    pipeline_config.imputation.numeric,
    random_seed=pipeline_config.random_seed,
)
executor.execute_all_pending()
fitted = executor.build()

test_result = fitted.transform(df=test_df)
```

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

The same `observer=` argument is available on `ImputationExecutor`, and you
can pass a different observer (or none) to each call independently.

### 2. Your own observer

An observer is just a `Callable[[PipelineEvent], None]`. Each event carries raw
`index`/`total` fields (drive a progress bar without parsing text) and an
`event_type` you can filter on:

```python
from dataforge_ml import EventType, ImputationExecutor, PipelineEvent

def my_observer(event: PipelineEvent) -> None:
    # Keep only per-item heartbeats; ignore stage boundaries.
    if event.event_type is EventType.item:
        pct = 100 * event.index / event.total
        print(f"{event.stage}: {event.column} — {pct:.0f}%")

ImputationExecutor(plan, train_df, pipeline_config.imputation.numeric, observer=my_observer)
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
