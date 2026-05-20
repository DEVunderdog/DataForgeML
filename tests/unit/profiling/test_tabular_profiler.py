"""
Tests for issue #67: TabularProfiler is pipeline-agnostic.

Verifies:
  - No constructor arguments required.
  - column_count reflects the total number of columns in the DataFrame,
    not a filtered subset.
  - overall_sparsity is computed over all columns present in the DataFrame.
  - No config attribute on the instance.
"""

import polars as pl
import pytest

from dataforge_ml.profiling._tabular import TabularProfiler
from dataforge_ml.profiling.config import DatasetStats


# ---------------------------------------------------------------------------
# No-argument instantiation
# ---------------------------------------------------------------------------


def test_tabular_profiler_no_args():
    """TabularProfiler must be instantiable with no arguments."""
    TabularProfiler()


def test_tabular_profiler_no_config_attribute():
    """TabularProfiler instance must not expose a config attribute."""
    profiler = TabularProfiler()
    assert not hasattr(profiler, "config"), (
        "TabularProfiler must not hold a config attribute"
    )


# ---------------------------------------------------------------------------
# column_count reflects the full DataFrame
# ---------------------------------------------------------------------------


def test_column_count_equals_full_dataframe_width():
    """column_count must equal df.width regardless of any pipeline-level exclusion."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3], dtype=pl.Int64),
            "b": pl.Series([4, 5, 6], dtype=pl.Int64),
            "c": pl.Series(["x", "y", "z"], dtype=pl.Utf8),
        }
    )
    result = TabularProfiler().profile(df)
    assert isinstance(result, DatasetStats)
    assert result.column_count == df.width  # 3


def test_column_count_single_column():
    """column_count works correctly for a single-column frame."""
    df = pl.DataFrame({"only": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64)})
    result = TabularProfiler().profile(df)
    assert result.column_count == 1


def test_column_count_many_columns():
    """column_count tracks all columns even when the frame is wide."""
    n_cols = 20
    data = {f"col_{i}": pl.Series(list(range(10)), dtype=pl.Int64) for i in range(n_cols)}
    df = pl.DataFrame(data)
    result = TabularProfiler().profile(df)
    assert result.column_count == n_cols


# ---------------------------------------------------------------------------
# overall_sparsity is computed over all columns
# ---------------------------------------------------------------------------


def test_overall_sparsity_includes_all_columns():
    """
    overall_sparsity must cover every column — not a pipeline-scoped subset.

    We build a 2-column frame where col_a is fully populated and col_b is
    entirely null.  With both columns counted the sparsity is 0.5; if only
    col_a were counted it would be 0.0.
    """
    df = pl.DataFrame(
        {
            "col_a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
            "col_b": pl.Series([None, None, None, None, None], dtype=pl.Int64),
        }
    )
    result = TabularProfiler().profile(df)
    # 5 nulls out of 10 cells (5 rows × 2 cols)
    assert result.overall_sparsity == pytest.approx(0.5, abs=1e-9)


def test_overall_sparsity_all_populated_is_zero():
    """overall_sparsity is 0.0 when no column has any null or effective-null."""
    df = pl.DataFrame(
        {
            "x": pl.Series([1, 2, 3], dtype=pl.Int64),
            "y": pl.Series([10, 20, 30], dtype=pl.Int64),
        }
    )
    result = TabularProfiler().profile(df)
    assert result.overall_sparsity == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# row_count is always correct
# ---------------------------------------------------------------------------


def test_row_count_equals_dataframe_height():
    df = pl.DataFrame({"v": pl.Series(list(range(42)), dtype=pl.Int64)})
    result = TabularProfiler().profile(df)
    assert result.row_count == 42
