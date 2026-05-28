"""
FittedImputer — stateless, serialisable object returned by ImputationOrchestrator.fit().

transform(df) applies train-time fill parameters and fitted models to any DataFrame.
to_dict() / from_dict() round-trips all scalar strategies and model-based strategies
(sklearn objects stored as base64-encoded joblib bytes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..config import SemanticType
from ._config import (
    ColumnImputationRecord,
    ImputationResult,
    ImputationStrategy,
)
from ..models._data_types import _INT_DTYPES, _FLOAT_DTYPES


class UnfittedColumnError(Exception):
    """
    Raised by FittedImputer.transform() when the input DataFrame contains
    missing values in a column for which no fill strategy was learned during
    fit() — i.e. the column had zero missing values in the training split.

    Typically indicates that a non-profile-stratified split was used.
    """


@dataclass
class FittedImputer:
    """
    Stores per-column imputation records and fitted models; applies them to any DataFrame.

    Parameters
    ----------
    records : dict[str, ColumnImputationRecord]
        One record per column processed during fit().
    models : dict[str, Any]
        Fitted sklearn model objects keyed by model name.
        - "knn"           : KNNImputer for KNN-assigned columns
        - "mice"          : IterativeImputer for MICE-assigned columns
        - "regression:{col}" : (BayesianRidge, feat_means ndarray) tuple per column
    model_cols : dict[str, list[str]]
        Ordered column lists for each model entry in models.
    """

    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    model_cols: dict[str, list[str]] = field(default_factory=dict)

    def transform(self, df: pl.DataFrame) -> ImputationResult:
        """
        Apply train-time fill parameters and models to df.

        Raises
        ------
        UnfittedColumnError
            If df has missing values in a column that had no missingness during
            fit() (strategy == Passthrough).
        """
        # --- Passthrough violation check ---
        violating_cols: list[str] = []
        for col, rec in self.records.items():
            if rec.strategy != ImputationStrategy.Passthrough:
                continue
            if col not in df.columns:
                continue
            series = df[col]
            has_missing = series.null_count() > 0
            if not has_missing and series.dtype in _FLOAT_DTYPES:
                has_missing = series.is_nan().any()
            if has_missing:
                violating_cols.append(col)

        if violating_cols:
            cols_str = ", ".join(f"'{c}'" for c in violating_cols)
            raise UnfittedColumnError(
                f"Column(s) {cols_str} have missing values but no fill strategy was "
                f"learned during fit() because the training split had no missing "
                f"values. Consider using DataSplitter.profile_stratified_split() "
                f"to ensure missingness is represented in training data."
            )

        # --- Drop columns ---
        dropped_cols = [
            col
            for col, rec in self.records.items()
            if rec.strategy == ImputationStrategy.Dropped and col in df.columns
        ]
        result_df = df.drop(dropped_cols)

        # --- Build indicator expressions before filling ---
        indicator_exprs = []
        for col, rec in self.records.items():
            if not rec.indicator_added:
                continue
            if col not in result_df.columns:
                continue
            dtype = result_df.schema[col]
            if dtype in _FLOAT_DTYPES:
                null_expr = pl.col(col).is_null() | pl.col(col).is_nan()
            else:
                null_expr = pl.col(col).is_null()
            indicator_exprs.append(null_expr.cast(pl.Int8).alias(f"{col}_missing"))

        if indicator_exprs:
            result_df = result_df.with_columns(indicator_exprs)

        # --- Apply scalar fill values ---
        fill_exprs = []
        for col, rec in self.records.items():
            if rec.strategy in (
                ImputationStrategy.Dropped,
                ImputationStrategy.Passthrough,
                ImputationStrategy.KNN,
                ImputationStrategy.MICE,
                ImputationStrategy.Regression,
            ):
                continue
            if col not in result_df.columns:
                continue
            if rec.fill_value is None:
                continue
            dtype = result_df.schema[col]
            fill_val = rec.fill_value
            if dtype in _INT_DTYPES:
                fill_val = int(round(float(fill_val)))
                fill_exprs.append(pl.col(col).fill_null(fill_val))
            elif dtype in _FLOAT_DTYPES:
                fv = float(fill_val)
                fill_exprs.append(pl.col(col).fill_nan(fv).fill_null(fv))
            else:
                fill_exprs.append(pl.col(col).fill_null(fill_val))

        if fill_exprs:
            result_df = result_df.with_columns(fill_exprs)

        # --- Apply model-based strategies (MICE → KNN → Regression) ---
        if self.models:
            result_df = _apply_block_model(
                result_df, "mice", self.models, self.model_cols
            )
            result_df = _apply_block_model(
                result_df, "knn", self.models, self.model_cols
            )
            for col, rec in self.records.items():
                if rec.strategy == ImputationStrategy.Regression:
                    result_df = _apply_regression(
                        result_df, col, self.models, self.model_cols
                    )

        return ImputationResult(
            dataframe=result_df,
            records=dict(self.records),
            dropped_columns=dropped_cols,
        )

    def to_dict(self) -> dict:
        import base64
        import io
        import joblib

        serialized_models: dict[str, str] = {}
        for key, model in self.models.items():
            buf = io.BytesIO()
            joblib.dump(model, buf)
            serialized_models[key] = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "records": {col: rec.to_dict() for col, rec in self.records.items()},
            "models": serialized_models,
            "model_cols": {k: list(v) for k, v in self.model_cols.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> FittedImputer:
        import base64
        import io
        import joblib
        from ._config import ColumnImputationRecord, ImputationStrategy

        records: dict[str, ColumnImputationRecord] = {}
        for col, rec_data in data.get("records", {}).items():
            records[col] = ColumnImputationRecord(
                column=rec_data["column"],
                semantic_type=SemanticType(rec_data["semantic_type"]),
                strategy=ImputationStrategy(rec_data["strategy"]),
                fill_value=rec_data.get("fill_value"),
                indicator_added=bool(rec_data.get("indicator_added", False)),
                signals=list(rec_data.get("signals", [])),
            )

        models: dict[str, Any] = {}
        for key, b64str in data.get("models", {}).items():
            buf = io.BytesIO(base64.b64decode(b64str))
            models[key] = joblib.load(buf)

        model_cols: dict[str, list[str]] = {
            k: list(v) for k, v in data.get("model_cols", {}).items()
        }

        return cls(records=records, models=models, model_cols=model_cols)


# ---------------------------------------------------------------------------
# Model application helpers
# ---------------------------------------------------------------------------


def _apply_block_model(
    df: pl.DataFrame,
    model_key: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a block model (KNN or MICE) to its assigned columns."""
    if model_key not in models:
        return df
    model = models[model_key]
    cols = [c for c in model_cols.get(model_key, []) if c in df.columns]
    if not cols:
        return df
    arr = _df_to_numpy(df, cols)
    arr_filled = model.transform(arr)
    return _numpy_to_df(df, cols, arr_filled)


def _apply_regression(
    df: pl.DataFrame,
    col: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a per-column BayesianRidge model to fill nulls in col."""
    model_key = f"regression:{col}"
    if model_key not in models or col not in df.columns:
        return df

    reg, feat_means = models[model_key]
    feat_cols = [c for c in model_cols.get(model_key, []) if c in df.columns]

    target_arr = _df_to_numpy(df, [col]).ravel().copy()
    null_mask = np.isnan(target_arr)
    if not null_mask.any():
        return df

    if feat_cols:
        X_full = _df_to_numpy(df, feat_cols)
        nan_in_X = np.isnan(X_full)
        if nan_in_X.any():
            X_full = X_full.copy()
            for j in range(X_full.shape[1]):
                col_nans = nan_in_X[:, j]
                if col_nans.any():
                    fill = float(feat_means[j]) if j < len(feat_means) else 0.0
                    X_full[col_nans, j] = fill
        y_pred = reg.predict(X_full[null_mask])
    else:
        y_pred = np.zeros(null_mask.sum())

    target_arr[null_mask] = y_pred
    return _numpy_to_df(df, [col], target_arr.reshape(-1, 1))


def _df_to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract columns as float64 numpy array, converting Polars nulls to NaN."""
    return (
        df.select([pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in cols])
        .to_numpy()
        .astype(np.float64)
    )


def _numpy_to_df(df: pl.DataFrame, cols: list[str], arr: np.ndarray) -> pl.DataFrame:
    """Replace column values in df with values from arr, preserving original dtypes."""
    new_cols = []
    for i, col in enumerate(cols):
        dtype = df.schema[col]
        col_arr = arr[:, i]
        if dtype in _INT_DTYPES:
            new_cols.append(
                pl.Series(col, np.round(col_arr), dtype=pl.Float64).cast(dtype)
            )
        elif dtype in _FLOAT_DTYPES:
            new_cols.append(pl.Series(col, col_arr, dtype=pl.Float64).cast(dtype))
        else:
            new_cols.append(pl.Series(col, col_arr, dtype=pl.Float64).cast(dtype))
    return df.with_columns(new_cols)
