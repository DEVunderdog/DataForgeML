# DataForgeML

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DEVunderdog/DataForgeML)

DataForgeML is an automated, end-to-end pipeline for turning raw tabular datasets into ML-ready data — without hand-writing schema logic, imputation rules, or splitting code for every new dataset.

The library is built as a one-stop solution: the goal is that a user should not need to manually assemble fragmented libraries steps or write custom glue code around it. Every phase — profiling, imputation, splitting — is designed to compose directly, driven by configuration objects rather than bespoke per-dataset scripts.

Supported input formats: CSV, TSV, Parquet, JSON, NDJSON, JSONL, XLSX, XLS, Arrow, and Feather, with automatic encoding and delimiter detection.

## Documentation

Full documentation — including the complete API reference, configuration guide, and the domain concepts behind profiling, imputation, and splitting — is available at:

**https://devunderdog.github.io/DataForgeML/**

## Installation

Install from PyPI
```
pip install dataforge-ml
```

## License

MIT
