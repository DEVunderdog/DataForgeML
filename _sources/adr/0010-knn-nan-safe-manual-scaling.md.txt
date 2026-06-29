# ADR 0010: NaN-safe manual scaling for KNN rather than sklearn Pipeline

**Status:** Accepted

`KNNImputer` requires NaN values in its input — that is the entire mechanism by which it identifies which cells to impute. `sklearn.preprocessing.StandardScaler` does not accept NaN inputs and raises `ValueError` when any cell is NaN. A naive `Pipeline([StandardScaler(), KNNImputer()])` therefore fails at fit time.

Feature scaling is nonetheless required before KNN: `nan_euclidean` distances are scale-sensitive, and a column with values in the thousands dominates distances over a column in the 0–1 range regardless of the number of neighbors chosen.

## Decision

Compute scaling parameters manually at fit time using `np.nanmean` and `np.nanstd` on the KNN training matrix. Apply the scale transformation directly to the array (NaN cells remain NaN after scaling). Fit `KNNImputer` on the scaled array. At transform time, scale the input, call `KNNImputer.transform()`, and inverse-scale the result.

Store `(model, col_means, col_stds)` together in a `_FittedKNN` dataclass under `"knn"` in `FittedImputer.models`. The existing joblib serialisation path handles `_FittedKNN` without modification.

## Considered Options

**Custom NaN-tolerant transformer in a sklearn Pipeline** — wrapping the scaling in a `TransformerMixin` that ignores NaN would keep the sklearn Pipeline idiom and allow `GridSearchCV` over KNN hyperparameters. Rejected: the added transformer class is complexity with no benefit at this scope — KNN hyperparameters are set adaptively by the library, not via grid search, so Pipeline composability provides nothing.

**Fit StandardScaler on complete rows only, then transform the full matrix** — `StandardScaler.fit(arr[complete_mask])` would work, then `StandardScaler.transform(arr)` on the full matrix. Rejected: `StandardScaler.transform()` still raises on NaN inputs because it applies `check_array` with `force_all_finite=True` by default. Overriding that flag is an undocumented workaround that may break across sklearn versions.

**Skip scaling entirely** — rely on `nan_euclidean`'s built-in handling. Rejected: `nan_euclidean` does not normalise column magnitudes; it only handles missing values in the distance computation. Scale dominance remains.
