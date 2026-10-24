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

The package root exports the Operations: typed functions, one per `dataforge`
CLI command. Each takes and returns Handles to artifacts kept in the
Workspace. Profile a raw dataset into inferred semantic types, per-column
stats, missingness, correlations and nonlinearity signals:

```python
from dataforge_ml import load, profile

frame = load(path="train.csv").frame

# Declare what you know better than the type detector.
result = profile(
    frame=frame,
    semantic_types=[{"column": "Year", "semantic_type": "categorical"}],
    seed=42,
)

# The Dataset Digest: one row per column, flagged columns first.
print(result.to_dict())
```

## Examples

To see how to use DataForgeML end to end, browse the full set of runnable,
self-contained example scripts — each with bundled datasets and step-by-step
walkthroughs — in the dedicated examples repository:

**https://github.com/DEVunderdog/dataforgeml-examples**

## Video

[![Watch the video](https://img.youtube.com/vi/8WrMEOKIEUE/maxresdefault.jpg)](https://www.youtube.com/watch?v=8WrMEOKIEUE)

## License

MIT
