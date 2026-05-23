# DataForgeML

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/DEVunderdog/DataForgeML)

Automated data profiling and splitting pipeline for ML datasets.

DataForgeML inspects your dataset, detects each column's semantic type (numeric, categorical, boolean, text, datetime, or identifier), computes per-column statistics and missingness, and produces a structured result ready for downstream feature engineering — no manual schema wrangling required.

## Installation

```bash
pip install dataforge-ml
```

## Quick Start

```python
from dataforge_ml import DataLoader, PipelineConfig, StructuralProfiler

df = DataLoader().load("titanic.csv")

config = PipelineConfig()
result = StructuralProfiler(config).profile(df)

print(result.columns["Age"].semantic_type)  # SemanticType.Numeric
print(result.dataset.row_count)             # total rows
```

`DataLoader` auto-detects encoding and delimiter. Supported formats: CSV, TSV, Parquet, JSON, NDJSON, JSONL, XLSX, XLS, Arrow, Feather.

## Column Type Overrides

Override the auto-detected type for any column before profiling:

```python
config = PipelineConfig()
config.set_column_type("PassengerId", "identifier")           # skip stats entirely
config.set_columns_type(["Survived", "Pclass"], "categorical")

result = StructuralProfiler(config).profile(df)
```

To drop a column from all processing entirely, use `exclude_columns`:

```python
config = PipelineConfig(exclude_columns=["PassengerId", "Name"])
```

## Splitting

```python
from dataforge_ml import DataLoader, DataSplitter

df = DataLoader().load("titanic.csv")
splitter = DataSplitter(df, target="Survived", random_seed=42)

# Random train/test split (stratified by default when target is set)
split = splitter.random_split(test_size=0.2)
print(split.train.shape, split.test.shape)

# Chronological split (no temporal leakage)
split = splitter.time_split(time_column="date", test_size=0.2)

# K-fold cross-validation
for fold in splitter.kfold(k=5):
    print(f"Fold {fold.fold_index}: train={fold.train_size}, val={fold.val_size}")
```

## License

MIT
