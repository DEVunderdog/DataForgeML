# Compact Profile Report — `to_markdown()` is human-readable, `to_full_markdown()` is lossless

`StructuralProfileResult` previously had a single `to_markdown()` method documented as "lossless," producing ~1 MB of Markdown for an 82-column dataset. This makes it unreadable for human inspection while adding nothing over `to_json()` for machine consumers.

We rename the lossless method to `to_full_markdown()` and replace `to_markdown()` with a compact, human-oriented view. The complete machine-readable serialization remains in `to_dict()` / `to_json()`; downstream phases always use those, never Markdown.

## What the compact view keeps and drops

**Dropped entirely:** histogram bins, `missingness_matrix` (N×N pairwise missingness correlations), `memory_breakdown` (per-column byte counts), `total_rows` on `ColumnMissingnessProfile` (redundant with `DatasetStats.row_count`).

**List-heavy fields capped:** `top_values` capped at 3 entries; feature and target correlation matrices replaced by top-5 highest absolute Pearson correlations and top-5 highest absolute Spearman correlations per column.

**Scalars kept in full:** all scalar raw numbers (`skewness`, `kurtosis`, `std`, `variance`, `mean`, `median`, `mean_median_ratio`, `effective_null_ratio`, `standard_null_ratio`, percentile snapshot, etc.) are preserved — they are interpretable on their own and cheap. Redundant scalar pairs (e.g. `std` + `variance`) are kept because the redundancy is cheap and both values are independently readable.

**`correlated_with` kept in full:** this field contains only column names, not values, so it carries no bulk.

## Two-tier column rendering

All columns appear in a top-level Column Summary table (type, missing %, severity, flags). Columns meeting the **clean threshold** — no `MissingnessFlag`, no `NumericFlag`, `MissingSeverity` is `None` or `Minor`, `NonlinearityTag` is `None` or `Linear` — appear in that table only. All other columns receive a full per-column detail section in the **Flagged Columns** section, ordered by descending severity (`DropCandidate`/`FullyNull` first, then `Severe`, `High`, `Moderate`, `Minor`-but-flagged, then numeric/nonlinearity-only flags), alphabetical within each tier.

## Document structure

```
# Structural Profile Report (Compact)
## Dataset Overview        ← scalar fields only; no memory_breakdown
## Column Summary          ← one table row per column, all columns
## Flagged Columns         ← full detail sections, severity-first ordering
## Target Analysis         ← top-5 Pearson + Spearman per feature, per target
## Sentinels               ← numeric_sentinels + string_sentinels unchanged
```

## Considered options

- **Keep `to_markdown()` lossless, add `to_compact_markdown()`** — rejected: two markdown methods with confusing overlap; the lossless markdown had no consumer that couldn't use `to_json()`.
- **`to_markdown(compact=False)` parameter** — rejected: a boolean mode flag on a serialization method is harder to read and document than two named methods with clear intent.
