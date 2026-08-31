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

# The lossless Profile Report — every to_dict() field rendered as Markdown.
# On a wide dataset (~82 columns) this is roughly a megabyte of text; for a
# bounded view, read result.to_dict() and select the fields you care about.
print(result.to_markdown())
```

## Examples

To see how to use DataForgeML end to end, browse the full set of runnable,
self-contained example scripts — each with bundled datasets and step-by-step
walkthroughs — in the dedicated examples repository:

**https://github.com/DEVunderdog/dataforgeml-examples**

## Video

[![Watch the video](https://img.youtube.com/vi/8WrMEOKIEUE/maxresdefault.jpg)](https://www.youtube.com/watch?v=8WrMEOKIEUE)

## Documentation

Full documentation — including the complete API reference, configuration guide, and the domain concepts behind profiling, imputation, and splitting — is available at:

**https://devunderdog.github.io/DataForgeML/**

## License

MIT
