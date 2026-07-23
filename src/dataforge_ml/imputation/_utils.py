"""Shared array-conversion and scalar-statistic utilities for the imputation package.

Deliberately dependency-free with respect to the fitting engines and the fitters
so anything in the package can reach these without a cycle.
"""

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


def _preserve_observed(
    original: pl.DataFrame, output: pl.DataFrame, cols: list[str]
) -> pl.DataFrame:
    """Restore observed cells of ``cols`` from ``original`` into ``output``.

    The observed-value preservation seam: a fitted unit's ``transform`` may
    rewrite a whole column (model write-back, domain snap), so as its final
    step it hands the frame it received and the frame it produced through
    here.  Every cell that was *not* missing in ``original`` is restored
    bit-for-bit, and each touched column is cast back to its original dtype
    (integer casts round, matching ``_numpy_to_df``).  Cells that *were*
    missing keep the unit's fill.

    A cell counts as missing under the Effective Null predicate:
    ``is_null() OR is_nan() OR is_infinite()`` for float columns,
    ``is_null()`` alone otherwise — matching what ``_df_to_numpy`` actually
    hands the estimator as a hole.  Only the named columns are touched.
    """
    corrected = []
    for col in cols:
        dtype = original.schema[col]
        src = original[col]
        if dtype in _FLOAT_DTYPES:
            observed = ~(src.is_null() | src.is_nan() | src.is_infinite())
        else:
            observed = src.is_not_null()
        out = output[col]
        if out.dtype != dtype:
            if dtype in _INT_DTYPES and out.dtype in _FLOAT_DTYPES:
                out = out.round(0)
            out = out.cast(dtype)
        corrected.append(src.zip_with(observed, out).alias(col))
    return output.with_columns(corrected)


def _clean(series: pl.Series) -> pl.Series:
    return series.drop_nulls()


def _compute_mean(df: pl.DataFrame, col: str) -> float:
    val = _clean(df[col]).mean()
    return float(val) if val is not None else 0.0


def _compute_median(df: pl.DataFrame, col: str) -> float:
    val = _clean(df[col]).median()
    return float(val) if val is not None else 0.0


def _compute_mode(df: pl.DataFrame, col: str) -> float:
    clean = _clean(df[col])
    if len(clean) == 0:
        return 0.0
    modes = clean.mode().sort()
    return float(modes[0])
