# KNN and MICE block overrides are scalars, not per-column dicts

`per_column_n_neighbors` was typed as `dict[str, int]` and `per_column_max_iter` was used for both Regression and MICE columns. Both were wrong for KNN and MICE because those strategies fit one shared model for the entire block — there is no such thing as a different `n_neighbors` per KNN column or a different `max_iter` per MICE column. The code that read these values iterated over the column list and picked the first dict entry that matched, silently discarding all other entries. This produced the right outcome by accident for the common case (one entry) but promised per-column semantics that the implementation could never honour.

`per_column_n_neighbors` is replaced by `knn_n_neighbors: Optional[int]` (scalar, KNN block only). `per_column_max_iter` is retained for Regression, where each column genuinely has its own `IterativeImputer` and a per-column value is meaningful. A new `mice_max_iter: Optional[int]` scalar is added for the MICE block.

## Consequences

- **Breaking config change**: existing `NumericImputationConfig` instances with `per_column_n_neighbors` set will no longer have that field; callers must migrate to `knn_n_neighbors`.
- `ImputationFitDiagnostic` gains `n_neighbors_used: Optional[int]` and `k_capped: Optional[bool]` for the KNN block. `k_capped` is `None` when `knn_n_neighbors` is set (adaptive formula was bypassed) and `True`/`False` based on whether `k_raw` exceeded `n_rows − 1`.
