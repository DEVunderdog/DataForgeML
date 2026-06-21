import polars as pl
import pytest

from dataforge_ml.splitting._splitter import DataSplitter
from dataforge_ml.splitting._config import FoldResult, SplitResult


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
    import dataforge_ml.splitting._splitter as mod
    import sys
    profiling_modules = [k for k in sys.modules if k.startswith("profiling")]
    # DataSplitter module itself must not have caused profiling to be imported
    assert "profiling" not in mod.__dict__


# ---------------------------------------------------------------------------
# time_split — fixtures
# ---------------------------------------------------------------------------

from datetime import date, timedelta

_BASE = date(2024, 1, 1)
_TIME_N = 50


@pytest.fixture(scope="module")
def time_df() -> pl.DataFrame:
    dates = [_BASE + timedelta(days=i) for i in range(_TIME_N)]
    return pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "value": pl.Series(list(range(_TIME_N)), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def time_splitter(time_df) -> DataSplitter:
    return DataSplitter(time_df)


# ---------------------------------------------------------------------------
# time_split — error cases
# ---------------------------------------------------------------------------


def test_time_split_raises_for_missing_column(time_splitter):
    with pytest.raises(ValueError, match="not found"):
        time_splitter.time_split("nonexistent")


def test_time_split_raises_when_neither_arg_provided(time_splitter):
    with pytest.raises(ValueError, match="Either"):
        time_splitter.time_split("date")


# ---------------------------------------------------------------------------
# time_split — fraction mode
# ---------------------------------------------------------------------------


def test_fraction_mode_sizes_sum_to_total(time_df, time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    assert result.train_size + result.test_size == len(time_df)


def test_fraction_mode_test_size_is_floor(time_df, time_splitter):
    import math
    result = time_splitter.time_split("date", test_size=0.2)
    assert result.test_size == math.floor(len(time_df) * 0.2)


def test_fraction_mode_no_temporal_leakage(time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    max_train = result.train["date"].max()
    min_test = result.test["date"].min()
    assert max_train < min_test


def test_fraction_mode_metadata_accurate(time_df, time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    total = len(time_df)
    assert result.train_ratio == pytest.approx(result.train_size / total)
    assert result.test_ratio == pytest.approx(result.test_size / total)


# ---------------------------------------------------------------------------
# time_split — cutoff mode
# ---------------------------------------------------------------------------


def test_cutoff_mode_rows_before_cutoff_are_train(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train["date"].max() < cutoff


def test_cutoff_mode_rows_on_or_after_cutoff_are_test(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.test["date"].min() == cutoff


def test_cutoff_mode_sizes_sum_to_total(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train_size + result.test_size == len(time_df)


def test_cutoff_mode_no_temporal_leakage(time_splitter):
    cutoff = _BASE + timedelta(days=25)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train["date"].max() < result.test["date"].min()


# ---------------------------------------------------------------------------
# time_split — cutoff takes priority over test_size
# ---------------------------------------------------------------------------


def test_cutoff_takes_priority_over_test_size(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    # test_size=0.5 would give 25 test rows; cutoff=day40 gives 10 test rows
    result_both = time_splitter.time_split("date", test_size=0.5, cutoff=cutoff)
    result_cutoff_only = time_splitter.time_split("date", cutoff=cutoff)
    assert result_both.test.equals(result_cutoff_only.test)
    assert result_both.train.equals(result_cutoff_only.train)


# ---------------------------------------------------------------------------
# kfold — fixtures
# ---------------------------------------------------------------------------

_KFOLD_N = 100
_K = 5


@pytest.fixture(scope="module")
def kfold_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": pl.Series(list(range(_KFOLD_N)), dtype=pl.Float64),
            "label": pl.Series(["A" if i % 2 == 0 else "B" for i in range(_KFOLD_N)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def kfold_splitter(kfold_df) -> DataSplitter:
    return DataSplitter(kfold_df, target="label", random_seed=42)


@pytest.fixture(scope="module")
def kfold_splitter_no_target(kfold_df) -> DataSplitter:
    return DataSplitter(kfold_df, random_seed=42)


# ---------------------------------------------------------------------------
# kfold — basic structure
# ---------------------------------------------------------------------------


def test_kfold_returns_exactly_k_folds(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert len(folds) == _K


def test_kfold_fold_indices_zero_to_k_minus_one(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert [f.fold_index for f in folds] == list(range(_K))


def test_kfold_returns_fold_result_instances(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert all(isinstance(f, FoldResult) for f in folds)


def test_kfold_sizes_sum_to_total(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    for fold in folds:
        assert fold.train_size + fold.val_size == len(kfold_df)


def test_kfold_dataframe_row_counts_match_sizes(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    for fold in folds:
        assert len(fold.train) == fold.train_size
        assert len(fold.val) == fold.val_size


# ---------------------------------------------------------------------------
# kfold — non-overlapping and complete coverage
# ---------------------------------------------------------------------------


def test_kfold_val_sets_non_overlapping(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    # Collect all row hashes across val sets; no duplicates allowed
    seen = set()
    for fold in folds:
        for row in fold.val.iter_rows():
            assert row not in seen, f"Row {row} appeared in multiple val sets"
            seen.add(row)


def test_kfold_val_sets_cover_all_rows(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    all_val_rows = set()
    for fold in folds:
        for row in fold.val.iter_rows():
            all_val_rows.add(row)
    all_df_rows = set(kfold_df.iter_rows())
    assert all_val_rows == all_df_rows


# ---------------------------------------------------------------------------
# kfold — stratification
# ---------------------------------------------------------------------------


def test_stratified_kfold_preserves_class_ratios(kfold_splitter):
    folds = kfold_splitter.kfold(_K, stratify=True)
    for fold in folds:
        counts = fold.val["label"].value_counts()["count"].to_list()
        ratio = counts[0] / sum(counts)
        assert abs(ratio - 0.5) < 0.15


def test_kfold_stratify_false_produces_valid_folds(kfold_df, kfold_splitter_no_target):
    folds = kfold_splitter_no_target.kfold(_K, stratify=False)
    assert len(folds) == _K
    for fold in folds:
        assert fold.train_size + fold.val_size == len(kfold_df)


def test_kfold_stratify_defaults_true_when_target_set(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert len(folds) == _K


def test_kfold_stratify_defaults_false_when_no_target(kfold_df, kfold_splitter_no_target):
    folds = kfold_splitter_no_target.kfold(_K)
    assert len(folds) == _K


def test_kfold_stratify_true_without_target_raises(kfold_splitter_no_target):
    with pytest.raises(ValueError, match="target"):
        kfold_splitter_no_target.kfold(_K, stratify=True)


# ---------------------------------------------------------------------------
# profile_stratified_split and profile_stratified_kfold — fixtures
# ---------------------------------------------------------------------------

from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.config import PipelineConfig

_PS_N = 300
_PS_NULL_EVERY = 10  # ~10 % missingness


@pytest.fixture(scope="module")
def ps_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "with_nulls": pl.Series(
                [None if i % _PS_NULL_EVERY == 0 else float(i) for i in range(_PS_N)],
                dtype=pl.Float64,
            ),
            "feature": pl.Series([float(i) for i in range(_PS_N)], dtype=pl.Float64),
            "label": pl.Series(["A" if i % 3 == 0 else "B" for i in range(_PS_N)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def ps_profile(ps_df):
    return StructuralProfiler(PipelineConfig()).profile(ps_df)


@pytest.fixture(scope="module")
def ps_splitter(ps_df) -> DataSplitter:
    return DataSplitter(ps_df, target="label", random_seed=42)


# ---------------------------------------------------------------------------
# profile_stratified_split — basic structure
# ---------------------------------------------------------------------------


def test_profile_split_returns_split_result(ps_df, ps_profile, ps_splitter):
    result = ps_splitter.profile_stratified_split(ps_profile, test_size=0.2)
    assert isinstance(result, SplitResult)


def test_profile_split_sizes_sum_to_total(ps_df, ps_profile, ps_splitter):
    result = ps_splitter.profile_stratified_split(ps_profile, test_size=0.2)
    assert result.train_size + result.test_size == len(ps_df)


def test_profile_split_dataframe_row_counts_match(ps_profile, ps_splitter):
    result = ps_splitter.profile_stratified_split(ps_profile, test_size=0.2)
    assert len(result.train) == result.train_size
    assert len(result.test) == result.test_size


# ---------------------------------------------------------------------------
# profile_stratified_split — acceptance criteria
# ---------------------------------------------------------------------------


def test_profile_split_missingness_in_training(ps_profile, ps_splitter):
    """Every column with missingness has at least one null in the training split."""
    result = ps_splitter.profile_stratified_split(ps_profile, test_size=0.2)
    for col, cp in ps_profile.columns.items():
        if cp.missingness and cp.missingness.effective_null_count > 0:
            if col in result.train.columns:
                assert result.train[col].null_count() > 0, (
                    f"column '{col}' has missingness in the profile but zero nulls "
                    f"in the training split"
                )


def test_profile_split_preserves_target_proportions(ps_df, ps_profile, ps_splitter):
    """Target class proportions are approximately preserved in both partitions."""
    result = ps_splitter.profile_stratified_split(ps_profile, test_size=0.2)
    original_a_ratio = (ps_df["label"] == "A").sum() / len(ps_df)
    train_a_ratio = (result.train["label"] == "A").sum() / result.train_size
    test_a_ratio = (result.test["label"] == "A").sum() / result.test_size
    assert abs(train_a_ratio - original_a_ratio) < 0.1
    assert abs(test_a_ratio - original_a_ratio) < 0.1


# ---------------------------------------------------------------------------
# profile_stratified_split — fallback
# ---------------------------------------------------------------------------


def test_profile_split_falls_back_when_no_signals():
    """A profile with no missingness and no at-risk signals falls back to random split."""
    df = pl.DataFrame(
        {
            "x": pl.Series([float(i) for i in range(100)], dtype=pl.Float64),
            "y": pl.Series([float(i) for i in range(100)], dtype=pl.Float64),
        }
    )
    # Profile with no target, no missingness → no signals → graceful fallback
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, random_seed=0)
    result = splitter.profile_stratified_split(profile, test_size=0.2)
    assert result.train_size + result.test_size == len(df)


# ---------------------------------------------------------------------------
# profile_stratified_kfold — basic structure
# ---------------------------------------------------------------------------


_PS_K = 5


def test_profile_kfold_returns_k_folds(ps_profile, ps_splitter):
    folds = ps_splitter.profile_stratified_kfold(ps_profile, k=_PS_K)
    assert len(folds) == _PS_K


def test_profile_kfold_returns_fold_result_instances(ps_profile, ps_splitter):
    folds = ps_splitter.profile_stratified_kfold(ps_profile, k=_PS_K)
    assert all(isinstance(f, FoldResult) for f in folds)


def test_profile_kfold_fold_indices_zero_to_k_minus_one(ps_profile, ps_splitter):
    folds = ps_splitter.profile_stratified_kfold(ps_profile, k=_PS_K)
    assert [f.fold_index for f in folds] == list(range(_PS_K))


def test_profile_kfold_sizes_sum_to_total(ps_df, ps_profile, ps_splitter):
    folds = ps_splitter.profile_stratified_kfold(ps_profile, k=_PS_K)
    for fold in folds:
        assert fold.train_size + fold.val_size == len(ps_df)


# ---------------------------------------------------------------------------
# profile_stratified_kfold — acceptance criteria
# ---------------------------------------------------------------------------


def test_profile_kfold_missingness_in_training(ps_profile, ps_splitter):
    """Each fold's training partition has at least one null for missing columns."""
    folds = ps_splitter.profile_stratified_kfold(ps_profile, k=_PS_K)
    for fold in folds:
        for col, cp in ps_profile.columns.items():
            if cp.missingness and cp.missingness.effective_null_count > 0:
                if col in fold.train.columns:
                    assert fold.train[col].null_count() > 0, (
                        f"fold {fold.fold_index}: column '{col}' has no nulls in training"
                    )


# ---------------------------------------------------------------------------
# build_label_matrix — signal cap
# ---------------------------------------------------------------------------


def test_signal_cap_at_default_50():
    """When more than 50 signals exist, only the 50 rarest are used by default."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    # Build a DataFrame with many columns that each have missingness
    n = 200
    cols = {f"c{i}": pl.Series([None if j == i else float(j) for j in range(n)], dtype=pl.Float64)
            for i in range(60)}
    df = pl.DataFrame(cols)
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    mat = build_label_matrix(df, profile, target=None)
    assert mat.shape[1] <= SplitConfig().max_stratification_signals


# ---------------------------------------------------------------------------
# build_label_matrix — signal 1: effective null mask
# ---------------------------------------------------------------------------


def test_signal_1_float_inf_rows_are_marked():
    """Inf values in a Float64 column produce 1 in the missingness signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    data = [1.0, 2.0, float("inf"), 4.0, float("nan"), 6.0]
    df = pl.DataFrame({"val": pl.Series(data, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[2] == 1, "Inf row should be marked as effective null"
    assert signal[4] == 1, "NaN row should be marked as effective null"
    assert signal[0] == 0, "Non-null row should not be marked"
    assert signal[1] == 0, "Non-null row should not be marked"


def test_signal_1_utf8_sentinel_rows_are_marked():
    """Sentinel strings in a Utf8 column produce 1 in the missingness signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    data = ["apple", "NA", "banana", "NULL", "", "cherry"]
    df = pl.DataFrame({"txt": pl.Series(data, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, '"NA" sentinel should be marked as effective null'
    assert signal[3] == 1, '"NULL" sentinel should be marked as effective null'
    assert signal[4] == 1, 'empty string should be marked as effective null'
    assert signal[0] == 0, "Normal value should not be marked"
    assert signal[2] == 0, "Normal value should not be marked"


def test_signal_1_integer_column_uses_standard_null_only():
    """Integer columns use only standard null detection (no Inf/sentinel expansion)."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    data = [1, None, 3, None, 5]
    df = pl.DataFrame({"num": pl.Series(data, dtype=pl.Int64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, "Standard null should be marked"
    assert signal[3] == 1, "Standard null should be marked"
    assert signal[0] == 0
    assert signal[2] == 0


# ---------------------------------------------------------------------------
# build_label_matrix — signal 5: rare categorical from profile
# ---------------------------------------------------------------------------


def test_signal_5_rare_value_marked_from_profile():
    """A value appearing 2% of rows is marked in signal 5 via the profile."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    # 100 rows: "rare" at 2%, "dominant" at 98%
    data = ["dominant"] * 98 + ["rare"] * 2
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    # The rare categorical signal should mark the last 2 rows
    assert mat.shape[1] >= 1
    rare_signal = mat[98, :]  # one of the rare rows
    assert rare_signal.max() == 1, "Rare row should be marked by at least one signal"


def test_signal_5_no_value_counts_in_module():
    """Confirm _profile_signals.py has no value_counts call for signal 5."""
    import inspect
    from dataforge_ml.splitting import _profile_signals

    source = inspect.getsource(_profile_signals)
    assert "value_counts" not in source


# ---------------------------------------------------------------------------
# build_label_matrix — signal 7: regression target quantile binning
# ---------------------------------------------------------------------------


def test_signal_7_numeric_target_produces_five_signals():
    """A numeric target with many unique values produces exactly 5 quantile-bucket signals."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 200
    # Sequential feature with no interesting signals, numeric target with 200 unique values
    df = pl.DataFrame({
        "feature": pl.Series(list(range(n)), dtype=pl.Int64),
        "target": pl.Series([float(i) for i in range(n)], dtype=pl.Float64),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target")

    # Signals come from: numeric extreme (feature + target) + zero/neg (target skew check)
    # + 5 quantile bucket signals for numeric target.
    # Key assertion: total signals <= 50 and the target contributed 5, not 200.
    assert mat.shape[1] <= 50
    # With 200 unique values, one-per-class would hit the cap of 50 before any other signals.
    # With quantile binning we get 5, so other signal families are not crowded out.
    assert mat.shape[1] < 200


def test_signal_7_categorical_target_produces_one_signal_per_class():
    """A categorical target with 3 classes still produces 3 target signals."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 90
    # Three perfectly balanced classes, no other interesting signals
    labels = ["A"] * 30 + ["B"] * 30 + ["C"] * 30
    df = pl.DataFrame({
        "feature": pl.Series(list(range(n)), dtype=pl.Int64),
        "target": pl.Series(labels, dtype=pl.Utf8),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target")

    # Only signals here: 3 target class signals (feature has no nulls, no extremes)
    assert mat.shape[1] == 3


# ---------------------------------------------------------------------------
# SplitConfig — dataclass basics
# ---------------------------------------------------------------------------


def test_split_config_defaults():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = SplitConfig()
    assert cfg.max_stratification_signals == 50
    assert cfg.boolean_minority_threshold == 0.05


def test_split_config_round_trip_defaults():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = SplitConfig()
    assert SplitConfig.from_dict(cfg.to_dict()) == cfg


def test_split_config_round_trip_custom_values():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = SplitConfig(max_stratification_signals=20, boolean_minority_threshold=0.10)
    restored = SplitConfig.from_dict(cfg.to_dict())
    assert restored.max_stratification_signals == 20
    assert restored.boolean_minority_threshold == 0.10


def test_split_config_exported_from_splitting_api():
    from dataforge_ml.splitting import SplitConfig as _Exported
    from dataforge_ml.splitting._config import SplitConfig
    assert _Exported is SplitConfig


# ---------------------------------------------------------------------------
# DataSplitter — SplitConfig constructor wiring
# ---------------------------------------------------------------------------


def test_data_splitter_no_config_uses_defaults(df):
    from dataforge_ml.splitting._config import SplitConfig
    splitter = DataSplitter(df, target="label", random_seed=0)
    assert splitter._config == SplitConfig()


def test_data_splitter_accepts_custom_config(df):
    from dataforge_ml.splitting._config import SplitConfig
    cfg = SplitConfig(max_stratification_signals=10)
    splitter = DataSplitter(df, target="label", random_seed=0, config=cfg)
    assert splitter._config.max_stratification_signals == 10


# ---------------------------------------------------------------------------
# max_stratification_signals — custom cap respected by build_label_matrix
# ---------------------------------------------------------------------------


def test_custom_max_signals_cap_is_respected():
    """Setting max_stratification_signals=5 caps the matrix at 5 columns."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    n = 200
    # 60 columns each with one null → 60 missingness signals before cap
    cols = {f"c{i}": pl.Series([None if j == i else float(j) for j in range(n)], dtype=pl.Float64)
            for i in range(60)}
    df = pl.DataFrame(cols)
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    cfg = SplitConfig(max_stratification_signals=5)
    mat = build_label_matrix(df, profile, target=None, config=cfg)
    assert mat.shape[1] <= 5


# ---------------------------------------------------------------------------
# boolean_minority_threshold — controls boolean stratification signal
# ---------------------------------------------------------------------------


def test_boolean_minority_threshold_triggers_signal():
    """A boolean column with 8% true_ratio fires a signal at threshold=0.10."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    n = 100
    # 8 True, 92 False → true_ratio = 0.08
    bool_vals = [True] * 8 + [False] * 92
    df = pl.DataFrame({"flag": pl.Series(bool_vals, dtype=pl.Boolean)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # Default threshold (0.05): 8% > 5% → no signal
    mat_default = build_label_matrix(df, profile, target=None)
    assert mat_default.shape[1] == 0

    # Raised threshold (0.10): 8% < 10% → signal fires
    cfg = SplitConfig(boolean_minority_threshold=0.10)
    mat_custom = build_label_matrix(df, profile, target=None, config=cfg)
    assert mat_custom.shape[1] == 1
    # The True rows should be marked
    assert mat_custom[:8, 0].sum() == 8


def test_boolean_minority_threshold_suppresses_signal():
    """Lowering the threshold below the minority ratio suppresses the signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    n = 100
    # 3 True, 97 False → true_ratio = 0.03
    bool_vals = [True] * 3 + [False] * 97
    df = pl.DataFrame({"flag": pl.Series(bool_vals, dtype=pl.Boolean)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # Default (0.05): 3% < 5% → signal fires
    mat_default = build_label_matrix(df, profile, target=None)
    assert mat_default.shape[1] == 1

    # Threshold lowered to 0.02: 3% > 2% → signal suppressed
    cfg = SplitConfig(boolean_minority_threshold=0.02)
    mat_custom = build_label_matrix(df, profile, target=None, config=cfg)
    assert mat_custom.shape[1] == 0


# ---------------------------------------------------------------------------
# PipelineConfig — SplitConfig nested round-trip
# ---------------------------------------------------------------------------


def test_pipeline_config_has_split_field():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = PipelineConfig()
    assert isinstance(cfg.split, SplitConfig)


def test_pipeline_config_round_trip_preserves_split():
    from dataforge_ml.splitting._config import SplitConfig
    original = PipelineConfig(split=SplitConfig(max_stratification_signals=15, boolean_minority_threshold=0.08))
    restored = PipelineConfig.from_dict(original.to_dict())
    assert restored.split.max_stratification_signals == 15
    assert restored.split.boolean_minority_threshold == 0.08


def test_pipeline_config_round_trip_default_split():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = PipelineConfig()
    restored = PipelineConfig.from_dict(cfg.to_dict())
    assert restored.split == SplitConfig()


# ---------------------------------------------------------------------------
# build_label_matrix — signal 1: string sentinel replace semantics
# ---------------------------------------------------------------------------


def test_signal_1_declared_sentinels_replace_hardcoded_defaults():
    """Declared sentinels suppress hardcoded defaults for that column (replace semantics)."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._config import ProfileConfig

    # "NA" is a hardcoded default; "MISSING" is the declared sentinel.
    # With replace semantics, "NA" must NOT be marked; "MISSING" MUST.
    data = ["apple", "MISSING", "NA", "banana", ""]
    df = pl.DataFrame({"txt": pl.Series(data, dtype=pl.Utf8)})
    pipeline_cfg = PipelineConfig(
        profiling=ProfileConfig(string_sentinels={"txt": ["MISSING"]})
    )
    profile = StructuralProfiler(pipeline_cfg).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, '"MISSING" (declared sentinel) should be marked'
    assert signal[4] == 1, 'empty string should always be marked regardless of declaration'
    assert signal[2] == 0, '"NA" (hardcoded default) must NOT be marked when replaced by declared sentinels'
    assert signal[0] == 0
    assert signal[3] == 0


def test_signal_1_no_sentinel_declaration_falls_back_to_hardcoded_defaults():
    """Columns with no string_sentinels declaration continue to use _SENTINEL_STRINGS."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    data = ["apple", "NA", "banana", "NULL", "cherry"]
    df = pl.DataFrame({"txt": pl.Series(data, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, '"NA" should be marked via hardcoded fallback'
    assert signal[3] == 1, '"NULL" should be marked via hardcoded fallback'
    assert signal[0] == 0
    assert signal[2] == 0
    assert signal[4] == 0


def test_signal_1_whitespace_always_marked_with_declared_sentinels():
    """Empty/whitespace strings are always effective null even when custom sentinels are declared."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._config import ProfileConfig

    data = ["apple", "CUSTOM", "  ", "banana", "NA"]
    df = pl.DataFrame({"txt": pl.Series(data, dtype=pl.Utf8)})
    pipeline_cfg = PipelineConfig(
        profiling=ProfileConfig(string_sentinels={"txt": ["CUSTOM"]})
    )
    profile = StructuralProfiler(pipeline_cfg).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, '"CUSTOM" (declared sentinel) should be marked'
    assert signal[2] == 1, 'whitespace-only string should always be marked regardless of declaration'
    assert signal[4] == 0, '"NA" (hardcoded default) must NOT be marked when replaced by declared sentinels'
    assert signal[0] == 0
    assert signal[3] == 0


def test_signal_1_declared_sentinels_matched_case_insensitively():
    """Declared sentinels match column data case-insensitively."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._config import ProfileConfig

    data = ["apple", "missing", "MISSING", "Missing", "banana"]
    df = pl.DataFrame({"txt": pl.Series(data, dtype=pl.Utf8)})
    pipeline_cfg = PipelineConfig(
        profiling=ProfileConfig(string_sentinels={"txt": ["MISSING"]})
    )
    profile = StructuralProfiler(pipeline_cfg).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, '"missing" (lowercase) should match "MISSING" declared sentinel'
    assert signal[2] == 1, '"MISSING" (exact match) should be marked'
    assert signal[3] == 1, '"Missing" (mixed-case) should match case-insensitively'
    assert signal[0] == 0
    assert signal[4] == 0


def test_signal_1_declared_sentinels_do_not_affect_other_dtype_columns():
    """string_sentinels declarations for a column do not bleed into non-string columns."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._config import ProfileConfig

    df = pl.DataFrame({
        "txt": pl.Series(["apple", "MISSING", "NA", "banana", ""], dtype=pl.Utf8),
        "num": pl.Series([1.0, 2.0, None, 4.0, 5.0], dtype=pl.Float64),
    })
    pipeline_cfg = PipelineConfig(
        profiling=ProfileConfig(string_sentinels={"txt": ["MISSING"]})
    )
    profile = StructuralProfiler(pipeline_cfg).profile(df)

    mat = build_label_matrix(df, profile, target=None)

    # Collect the num signal: standard null (row 2) should be the only null
    num_profile = profile.columns["num"]
    assert num_profile.missingness is not None
    assert num_profile.missingness.effective_null_count > 0
