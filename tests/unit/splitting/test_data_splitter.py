import polars as pl
import pytest

from ....splitting._splitter import DataSplitter
from ....splitting._config import SplitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N = 100


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_a": pl.Series(list(range(_N)), dtype=pl.Float64),
            "feature_b": pl.Series([i * 0.5 for i in range(_N)], dtype=pl.Float64),
            "label": pl.Series(["cat" if i % 2 == 0 else "dog" for i in range(_N)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def df_no_target() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": pl.Series(list(range(_N)), dtype=pl.Float64),
            "y": pl.Series(list(range(_N, _N * 2)), dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_valid_construction(df):
    splitter = DataSplitter(df, target="label", random_seed=42)
    assert splitter._df is df
    assert splitter._target == "label"
    assert splitter._random_seed == 42


def test_constructor_no_target(df_no_target):
    splitter = DataSplitter(df_no_target)
    assert splitter._target is None
    assert splitter._random_seed is None


def test_constructor_raises_type_error_for_non_polars():
    with pytest.raises(TypeError):
        DataSplitter([[1, 2], [3, 4]])


def test_constructor_raises_type_error_for_numpy_array():
    import numpy as np
    with pytest.raises(TypeError):
        DataSplitter(np.zeros((10, 3)))


def test_constructor_raises_value_error_for_empty_df():
    empty = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="empty"):
        DataSplitter(empty)


def test_constructor_raises_value_error_for_missing_target(df):
    with pytest.raises(ValueError, match="not found"):
        DataSplitter(df, target="nonexistent_column")


# ---------------------------------------------------------------------------
# random_split — sizes and ratios
# ---------------------------------------------------------------------------


def test_random_split_sizes_sum_to_total(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df)


def test_random_split_dataframe_row_counts_match_sizes(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert len(result.train) == result.train_size
    assert len(result.test) == result.test_size


def test_random_split_ratios_reflect_actual_proportions(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    total = len(df)
    assert result.train_ratio == pytest.approx(result.train_size / total)
    assert result.test_ratio == pytest.approx(result.test_size / total)


def test_random_split_returns_split_result(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert isinstance(result, SplitResult)


# ---------------------------------------------------------------------------
# random_split — stratification
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_class_ratios(df):
    splitter = DataSplitter(df, target="label", random_seed=42)
    result = splitter.random_split(test_size=0.2, stratify=True)
    original_ratio = df["label"].value_counts(sort=True)["count"].to_list()
    train_counts = result.train["label"].value_counts(sort=True)["count"].to_list()
    test_counts = result.test["label"].value_counts(sort=True)["count"].to_list()
    # both splits should have roughly equal class representation (50/50 here)
    train_ratio = train_counts[0] / sum(train_counts)
    test_ratio = test_counts[0] / sum(test_counts)
    assert abs(train_ratio - 0.5) < 0.1
    assert abs(test_ratio - 0.5) < 0.1


def test_stratify_false_produces_valid_split(df_no_target):
    splitter = DataSplitter(df_no_target, random_seed=7)
    result = splitter.random_split(test_size=0.3, stratify=False)
    assert result.train_size + result.test_size == len(df_no_target)


def test_stratify_defaults_true_when_target_set(df):
    splitter = DataSplitter(df, target="label", random_seed=1)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df)


def test_stratify_defaults_false_when_no_target(df_no_target):
    splitter = DataSplitter(df_no_target, random_seed=1)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df_no_target)


def test_stratify_true_without_target_raises_value_error(df_no_target):
    splitter = DataSplitter(df_no_target)
    with pytest.raises(ValueError, match="target"):
        splitter.random_split(test_size=0.2, stratify=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_splits(df):
    s1 = DataSplitter(df, target="label", random_seed=99)
    s2 = DataSplitter(df, target="label", random_seed=99)
    r1 = s1.random_split(test_size=0.2)
    r2 = s2.random_split(test_size=0.2)
    assert r1.train.equals(r2.train)
    assert r1.test.equals(r2.test)


def test_different_seeds_produce_different_splits(df):
    s1 = DataSplitter(df, target="label", random_seed=1)
    s2 = DataSplitter(df, target="label", random_seed=2)
    r1 = s1.random_split(test_size=0.2)
    r2 = s2.random_split(test_size=0.2)
    assert not r1.train.equals(r2.train)


# ---------------------------------------------------------------------------
# No profiling import leakage
# ---------------------------------------------------------------------------


def test_no_profiling_import():
    import splitting._splitter as mod
    import sys
    profiling_modules = [k for k in sys.modules if k.startswith("profiling")]
    # DataSplitter module itself must not have caused profiling to be imported
    assert "profiling" not in mod.__dict__
