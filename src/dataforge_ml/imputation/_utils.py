"""Shared array-conversion utilities for the imputation package."""

from __future__ import annotations

import numpy as np
import polars as pl

from ..models._data_types import _FLOAT_DTYPES, _INT_DTYPES


def _df_to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Extract columns as a float64 numpy array, converting Polars nulls to NaN.

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame.
    cols : list[str]
        Ordered column names to extract.

    Returns
    -------
    np.ndarray
        Shape ``(len(df), len(cols))`` float64 array.  Each Polars ``null``
        becomes ``NaN``; all other values are cast to ``float64``.

    Notes
    -----
    **Sentinel precondition (ADR-0028):** this function only converts
    Polars-native nulls.  Numeric sentinel values (e.g. ``-999`` stored as a
    real float) pass through unchanged.  All effective nulls — including
    numeric sentinels — *must already be normalised to Polars ``null``* before
    calling this function.  Sentinel normalisation is the responsibility of the
    call site (Scope 5, Issue #94).
    """
    return (
        df.select([pl.col(c).cast(pl.Float64).fill_null(float("nan")) for c in cols])
        .to_numpy()
        .astype(np.float64)
    )


def _numpy_to_df(df: pl.DataFrame, cols: list[str], arr: np.ndarray) -> pl.DataFrame:
    """Replace column values in df with values from arr, preserving original dtypes.

    Parameters
    ----------
    df : pl.DataFrame
        Source DataFrame whose schema determines the output dtypes.
    cols : list[str]
        Ordered column names corresponding to columns of arr.
    arr : np.ndarray
        Shape ``(len(df), len(cols))`` float64 array of imputed values.

    Returns
    -------
    pl.DataFrame
        df with the named columns replaced by values from arr cast to each
        column's original dtype.

    Raises
    ------
    AssertionError
        If arr contains NaN for a column whose original dtype is an integer type.
        Post-imputation NaN in an integer column is a call-site bug.
    ValueError
        If a column's original dtype is neither numeric (integer or float) nor
        castable from float64 — e.g. ``pl.Utf8`` or ``pl.Boolean``.
    """
    new_cols = []
    for i, col in enumerate(cols):
        dtype = df.schema[col]
        col_arr = arr[:, i]
        if dtype in _INT_DTYPES:
            rounded = np.round(col_arr)
            assert not np.isnan(col_arr).any(), (
                f"NaN in integer column '{col}' after imputation — "
                "all nulls must be filled before writing back to an integer dtype"
            )
            arr_int = rounded.astype(np.int64)
            new_cols.append(pl.Series(col, arr_int, dtype=dtype))
        elif dtype in _FLOAT_DTYPES:
            new_cols.append(pl.Series(col, col_arr, dtype=pl.Float64).cast(dtype))
        else:
            raise ValueError(
                f"_numpy_to_df: column '{col}' has non-numeric dtype {dtype!r}; "
                "only integer and float columns are supported"
            )
    return df.with_columns(new_cols)
