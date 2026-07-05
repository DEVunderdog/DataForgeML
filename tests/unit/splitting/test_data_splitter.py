import polars as pl
import pytest

from dataforge_ml.splitting._splitter import DataSplitter
from dataforge_ml.splitting._config import FoldResult, SplitConfig, SplitResult


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


def test_profile_split_computes_shuffle_floor(monkeypatch):
    """profile_stratified_split threads ceil(2 / min(test_size, 1 - test_size)) as min_positives."""
    import dataforge_ml.splitting._profile_signals as ps_mod

    captured = {}
    real = ps_mod.build_label_matrix

    def spy(*args, **kwargs):
        captured["min_positives"] = kwargs.get("min_positives")
        return real(*args, **kwargs)

    monkeypatch.setattr(ps_mod, "build_label_matrix", spy)

    df = pl.DataFrame(
        {
            "with_nulls": pl.Series(
                [None if i % 10 == 0 else float(i) for i in range(200)], dtype=pl.Float64
            ),
            "label": pl.Series(["A" if i % 3 == 0 else "B" for i in range(200)], dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, target="label", random_seed=0)
    splitter.profile_stratified_split(profile, test_size=0.2)

    # ceil(2 / min(0.2, 0.8)) == ceil(2 / 0.2) == 10
    assert captured["min_positives"] == 10


def test_profile_kfold_computes_k_floor(monkeypatch):
    """profile_stratified_kfold threads k as min_positives."""
    import dataforge_ml.splitting._profile_signals as ps_mod

    captured = {}
    real = ps_mod.build_label_matrix

    def spy(*args, **kwargs):
        captured["min_positives"] = kwargs.get("min_positives")
        return real(*args, **kwargs)

    monkeypatch.setattr(ps_mod, "build_label_matrix", spy)

    df = pl.DataFrame(
        {
            "with_nulls": pl.Series(
                [None if i % 10 == 0 else float(i) for i in range(200)], dtype=pl.Float64
            ),
            "label": pl.Series(["A" if i % 3 == 0 else "B" for i in range(200)], dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, target="label", random_seed=0)
    splitter.profile_stratified_kfold(profile, k=5)

    assert captured["min_positives"] == 5


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
    """The retained-signal count never exceeds the configured maximum."""
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
# build_label_matrix — importance-ranked cap + rows-per-signal guard (ADR-0047)
# ---------------------------------------------------------------------------


def test_cap_keeps_target_and_drops_missingness_first():
    """Over budget, target survives while low-priority missingness is dropped."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    # 300 rows: a 3-class categorical target (3 class signals, collapsed to 2
    # after the dummy-drop) plus many single-null columns producing low-priority
    # missingness signals that a tight cap must shed first.
    n = 300
    target_vals = (["a"] * 100 + ["b"] * 100 + ["c"] * 100)
    cols = {"target": pl.Series(target_vals, dtype=pl.Utf8)}
    for i in range(20):
        cols[f"m{i}"] = pl.Series(
            [None if j == i else float(j) for j in range(n)], dtype=pl.Float64
        )
    df = pl.DataFrame(cols)
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # Cap of 2 → only the two surviving target class signals fit; every
    # missingness signal is evicted, but the target is never dropped.
    cfg = SplitConfig(max_stratification_signals=2)
    mat = build_label_matrix(df, profile, target="target", config=cfg)
    assert mat.shape[1] == 2
    # Both retained columns are target-class indicators (never a missingness
    # signal, which would mark at most a single row).
    class_cols = {
        tuple((df["target"] == cls).cast(pl.Int8).to_numpy().tolist())
        for cls in ("a", "b", "c")
    }
    for j in range(mat.shape[1]):
        assert tuple(mat[:, j].tolist()) in class_cols


def test_rows_per_signal_reduces_wide_but_short_input():
    """A wide-but-short dataset is reduced to the rows-per-signal budget."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    # 100 rows, 40 single-null missingness columns → 40 candidate signals, but
    # the row budget (100 / 20 = 5) caps retention well below both 40 and the
    # generous max of 50.
    n = 100
    cols = {f"c{i}": pl.Series([None if j == i else float(j) for j in range(n)], dtype=pl.Float64)
            for i in range(40)}
    df = pl.DataFrame(cols)
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    cfg = SplitConfig(max_stratification_signals=50, rows_per_signal=20)
    mat = build_label_matrix(df, profile, target=None)  # default budget 10 -> 10
    assert mat.shape[1] == 10
    mat_tight = build_label_matrix(df, profile, target=None, config=cfg)
    assert mat_tight.shape[1] == 5


def test_too_few_rows_to_support_target_degrades_to_empty():
    """When the budget cannot fit the full target set the matrix is empty."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.splitting._config import SplitConfig

    # 3-class target → 2 signals after dummy-drop, but a row budget of
    # 12 / 10 = 1 cannot support both, so the whole matrix is emptied.
    target_vals = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    df = pl.DataFrame({"target": pl.Series(target_vals, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target", min_positives=1)
    assert mat.shape[1] == 0


def test_too_small_dataset_falls_back_to_random_split():
    """profile_stratified_split degrades to a random split on a tiny dataset."""
    from dataforge_ml.splitting._config import SplitConfig

    target_vals = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    df = pl.DataFrame({"target": pl.Series(target_vals, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    splitter = DataSplitter(df, target="target", random_seed=0)
    result = splitter.profile_stratified_split(profile, test_size=0.25)
    # A valid split is still produced (via the random fallback), covering all rows.
    assert result.train_size + result.test_size == len(df)


# ---------------------------------------------------------------------------
# build_label_matrix — redundancy gate (ADR-0047 gate 3): correlation collapse
# ---------------------------------------------------------------------------


def _num_null(nulls: set, n: int, val: float = 9.0) -> pl.Series:
    """Constant-valued Float64 column null on ``nulls``: a pure missingness signal
    (a constant non-null region yields no extreme/near-constant signal)."""
    return pl.Series([None if j in nulls else val for j in range(n)], dtype=pl.Float64)


def test_redundancy_collapses_near_identical_signals():
    """Two columns that are always missing together collapse to a single signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 100
    # x and y are null on exactly the same rows → identical missingness signals
    # (correlation +1); z is null on a disjoint set and must survive independently.
    df = pl.DataFrame({
        "x": _num_null({0, 1, 2, 3, 4}, n),
        "y": _num_null({0, 1, 2, 3, 4}, n),
        "z": _num_null({10, 11, 12, 13, 14}, n),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(
        df, profile, target=None, config=SplitConfig(rows_per_signal=1)
    )
    cols = [tuple(mat[:, j].tolist()) for j in range(mat.shape[1])]
    xy_pat = tuple([1] * 5 + [0] * 95)
    z_pat = tuple([0] * 10 + [1] * 5 + [0] * 85)
    # The identical x/y missingness signal appears exactly once, not twice.
    assert cols.count(xy_pat) == 1
    # The distinct z signal is retained.
    assert z_pat in cols


def test_redundancy_collapses_mirror_image_binary_target():
    """A binary target's two mirror-image class flags collapse to one column."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 200
    df = pl.DataFrame({
        "feat": pl.Series(list(range(n)), dtype=pl.Int64),
        "target": pl.Series(["a"] * 100 + ["b"] * 100, dtype=pl.Utf8),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target")
    # The two class flags are perfect mirror images (correlation −1); the gate
    # collapses them to a single column.
    assert mat.shape[1] == 1


def test_redundancy_keeps_higher_priority_family():
    """On a redundant pair the higher-priority family's signal survives."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 100
    # Boolean minority (True on rows 0–3, priority 2) and a missingness signal
    # that is null on rows 4–99 (priority 4) are perfect mirror images. The
    # higher-priority boolean signal must survive, the missingness one drop.
    df = pl.DataFrame({
        "flag": pl.Series([True] * 4 + [False] * 96, dtype=pl.Boolean),
        "m": _num_null(set(range(4, 100)), n),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(
        df, profile, target=None, config=SplitConfig(rows_per_signal=1)
    )
    cols = [tuple(mat[:, j].tolist()) for j in range(mat.shape[1])]
    bool_pat = tuple([1] * 4 + [0] * 96)
    miss_pat = tuple([0] * 4 + [1] * 96)
    assert bool_pat in cols, "boolean minority (higher priority) must survive"
    assert miss_pat not in cols, "mirror-image missingness (lower priority) must drop"


def test_redundancy_runs_before_the_cap():
    """Duplicates collapse before the cap, so the cap does not spend its budget
    on redundant copies and evict a distinct signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 100
    # Five identical single-null signals plus one distinct signal, with a cap of
    # 2. If the cap ran first it would fill both slots from the (rarer) identical
    # cluster and then dedupe to one column, dropping the distinct signal.
    # Because redundancy runs first, the five collapse to one, leaving room for
    # the distinct signal → two columns, distinct retained.
    cols = {f"a{i}": _num_null({0}, n) for i in range(5)}
    cols["dist"] = _num_null({0, 1, 2}, n)
    df = pl.DataFrame(cols)
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(
        df,
        profile,
        target=None,
        config=SplitConfig(max_stratification_signals=2, rows_per_signal=1),
    )
    col_tuples = [tuple(mat[:, j].tolist()) for j in range(mat.shape[1])]
    assert mat.shape[1] == 2
    assert tuple([1, 1, 1] + [0] * 97) in col_tuples


def test_redundancy_threshold_is_configurable():
    """The redundancy threshold is conservative by default but configurable."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 100
    # p and q overlap heavily (absolute correlation ≈ 0.82) but are not
    # near-perfect duplicates. The conservative default (0.95) keeps both; a
    # lowered threshold collapses them.
    df = pl.DataFrame({
        "p": _num_null(set(range(0, 10)), n),
        "q": _num_null(set(range(0, 7)), n),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    default_mat = build_label_matrix(
        df, profile, target=None, config=SplitConfig(rows_per_signal=1)
    )
    lowered_mat = build_label_matrix(
        df,
        profile,
        target=None,
        config=SplitConfig(rows_per_signal=1, redundancy_correlation_threshold=0.7),
    )
    assert default_mat.shape[1] == 2
    assert lowered_mat.shape[1] == 1


# ---------------------------------------------------------------------------
# build_label_matrix — viability gate (ADR-0047 gate 2)
# ---------------------------------------------------------------------------


def test_viability_gate_drops_single_positive_signal():
    """A signal with fewer than min_positives ones is excluded."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    # One null with constant non-null values → the only surviving signal is the
    # per-column missingness signal, which has exactly one positive (extremes and
    # skew signals produce all-zeros on a constant column and are dropped).
    n = 50
    data = [None if i == 0 else 5.0 for i in range(n)]
    df = pl.DataFrame({"val": pl.Series(data, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # min_positives=1 admits the single-positive missingness signal.
    mat_default = build_label_matrix(df, profile, target=None, min_positives=1)
    assert mat_default.shape[1] == 1
    assert int(mat_default[:, 0].sum()) == 1

    # min_positives=2 excludes it (only one 1 present).
    mat_gated = build_label_matrix(df, profile, target=None, min_positives=2)
    assert mat_gated.shape[1] == 0


def test_viability_gate_drops_all_ones_signal():
    """A signal with fewer than min_positives zeros (near-all-ones) is excluded."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    # All-but-one null → the missingness signal has exactly one zero.
    n = 50
    data = [1.0 if i == 0 else None for i in range(n)]
    df = pl.DataFrame({"val": pl.Series(data, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # min_positives=2 excludes it (only one 0 present).
    mat_gated = build_label_matrix(df, profile, target=None, min_positives=2)
    assert mat_gated.shape[1] == 0


# ---------------------------------------------------------------------------
# build_label_matrix — signal 1: effective null mask
# ---------------------------------------------------------------------------


def test_signal_1_float_inf_rows_are_marked():
    """Inf values in a Float64 column produce 1 in the missingness signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    data = [1.0, 2.0, float("inf"), 4.0, float("nan"), 6.0]
    df = pl.DataFrame({"val": pl.Series(data, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # rows_per_signal=1 keeps the tiny-fixture budget from evicting the signal.
    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

    assert mat.shape[1] >= 1
    signal = mat[:, 0]
    assert signal[1] == 1, "Standard null should be marked"
    assert signal[3] == 1, "Standard null should be marked"
    assert signal[0] == 0
    assert signal[2] == 0


# ---------------------------------------------------------------------------
# build_label_matrix — signal 2: Joint MAR pair effective null mask
# ---------------------------------------------------------------------------


def test_signal_2_joint_mar_string_sentinel_receives_label_one():
    """A row with a string sentinel in a MAR-correlated column receives label 1 in the joint signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    # Rows 0–69: valid values in both columns.
    # Rows 70–99: string sentinels in both columns simultaneously → Pearson r = 1.0
    # between effective-null indicators → MAR pair detected → joint signal column added.
    col_a = ["valid"] * 70 + ["NA"] * 30
    col_b = ["good"] * 70 + ["NULL"] * 30
    df = pl.DataFrame(
        {
            "col_a": pl.Series(col_a, dtype=pl.Utf8),
            "col_b": pl.Series(col_b, dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    # Precondition: MAR correlation detected between the pair.
    mp_a = profile.columns["col_a"].missingness
    assert mp_a is not None and "col_b" in mp_a.correlated_with

    mat = build_label_matrix(df, profile, target=None)

    # per-col_a missingness, per-col_b missingness and the joint MAR signal are
    # all the identical [0]*70 + [1]*30 vector, so the ADR-0047 redundancy gate
    # collapses the three near-identical signals into a single column.
    assert mat.shape[1] == 1

    # Every sentinel row (70–99) must be marked (label 1) in at least one signal.
    assert (mat[70:, :].sum(axis=1) > 0).all(), (
        "Rows with string sentinels in a MAR-correlated pair must receive label 1"
    )
    # Valid rows (0–69) must not be marked in any signal.
    assert mat[:70, :].sum() == 0


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


def test_signal_5_one_column_per_rare_value():
    """The matrix has exactly one rare-categorical column per rare value."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    # 100 rows: three distinct rare values (3 rows each) below the dominant.
    data = ["dominant"] * 91 + ["r1"] * 3 + ["r2"] * 3 + ["r3"] * 3
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    rare_vals = profile.columns["cat"].stats.rare_categories.rare_label_values
    assert len(rare_vals) >= 2, "test requires multiple rare values"

    mat = build_label_matrix(df, profile, target=None)

    # One binary signal column per rare value (no other signals for a clean col).
    assert mat.shape[1] == len(rare_vals)
    # Each column is 1 iff the row equals that specific rare value.
    for j, val in enumerate(rare_vals):
        expected = (df["cat"] == val).cast(pl.Int8).to_numpy()
        assert (mat[:, j] == expected).all()


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


def test_signal_7_categorical_target_produces_class_set_minus_dummy():
    """A categorical target with 3 classes yields 3 - 1 = 2 signals after the
    ADR-0047 gate-3 dummy-drop of the mutually-exclusive set's last member."""
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

    # 3 target class signals minus the dummy-dropped last member (feature has no
    # nulls, no extremes).
    assert mat.shape[1] == 2


def test_signal_7_discrete_rating_target_produces_one_signal_per_class():
    """A non-numeric discrete-rating target emits one label per class."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 300
    # Ratings 1..5 land as SemanticType.Categorical → the class branch fires.
    ratings = [(i % 5) + 1 for i in range(n)]
    df = pl.DataFrame({"target": pl.Series(ratings, dtype=pl.Int64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target", min_positives=1)

    # Five ratings → five class signals minus the dummy-dropped last member.
    assert mat.shape[1] == 4


def test_signal_7_bounded_discrete_target_produces_one_signal_per_class():
    """A numeric target classified BoundedDiscrete emits one label per class."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.config import SemanticType
    from dataforge_ml.profiling._config import NumericKind

    n = 300
    ratings = [(i % 5) + 1 for i in range(n)]
    df = pl.DataFrame({"target": pl.Series(ratings, dtype=pl.Int64)})
    # Force SemanticType.Numeric + NumericKind.BoundedDiscrete so the two-tier
    # signal reuses the Phase 1 classification rather than a local heuristic.
    cfg = PipelineConfig()
    cfg.set_column_type("target", SemanticType.Numeric)
    cfg.set_numeric_kind("target", NumericKind.BoundedDiscrete)
    profile = StructuralProfiler(cfg).profile(df)

    from dataforge_ml.profiling._numeric_config import NumericFlag

    assert profile.columns["target"].numeric_kind == NumericKind.BoundedDiscrete
    mat = build_label_matrix(df, profile, target="target", min_positives=1)

    # BoundedDiscrete routes to the class branch: one label per rating value,
    # minus the dummy-dropped last member of the mutually-exclusive set — four
    # class signals. The evenly-spread 1..5 column also trips the dip test, so
    # the numeric column contributes one extra bimodal minority-cluster signal
    # (the feature-numeric loops do not skip the target), for five in total.
    assert profile.columns["target"].stats.has_flag(NumericFlag.Bimodal)
    assert mat.shape[1] == 5


def test_signal_7_continuous_target_buckets_min_of_five_and_nunique():
    """A continuous target with 3 unique values yields min(5, 3) = 3 buckets."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    n = 300
    base = [1.5, 2.5, 3.5]  # fractional values keep the column Continuous
    df = pl.DataFrame(
        {"target": pl.Series([base[i % 3] for i in range(n)], dtype=pl.Float64)}
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target", min_positives=1)

    # min(5, n_unique) == 3 quantile buckets minus the dummy-dropped last bucket.
    assert mat.shape[1] == 2


def test_signal_7_continuous_target_emits_target_missing_label():
    """A continuous target with nulls emits a dedicated 'target missing' label."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    import numpy as np

    n = 300
    # Every 20th row is null → 15 null-target rows.
    tvals = [None if i % 20 == 0 else float(i) * 1.37 for i in range(n)]
    df = pl.DataFrame(
        {
            "feature": pl.Series([float(i) for i in range(n)], dtype=pl.Float64),
            "target": pl.Series(tvals, dtype=pl.Float64),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    mat = build_label_matrix(df, profile, target="target", min_positives=1)

    # A column equal to the null-target mask must be present so null rows balance.
    null_mask = df["target"].is_null().cast(pl.Int8).to_numpy()
    assert int(null_mask.sum()) == 15
    found = any(
        np.array_equal(mat[:, j], null_mask) for j in range(mat.shape[1])
    )
    assert found, "a dedicated 'target missing' label must be emitted for null-target rows"


# ---------------------------------------------------------------------------
# build_label_matrix — NearConstant minority + zero/negative gate
# ---------------------------------------------------------------------------


def test_near_constant_column_produces_minority_signal():
    """A NearConstant numeric column emits an off-mode minority signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._numeric_config import NumericFlag
    import numpy as np

    n = 300
    # 95% of rows share the mode (5.0); the remaining 15 are off-mode. Fractional
    # values keep the column NumericKind.Continuous, so the band branch fires.
    off_mode = [1.25, 2.75, 3.5, 4.1, 6.9, 7.3, 8.8, 9.2, 0.5, 10.5,
                11.5, 12.5, 13.5, 14.5, 15.5]
    vals = [5.0] * (n - len(off_mode)) + off_mode
    df = pl.DataFrame({"nc": pl.Series(vals, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    # Guard: the flag must actually be set, otherwise the signal branch is dead.
    assert profile.columns["nc"].stats.has_flag(NumericFlag.NearConstant)

    mat = build_label_matrix(df, profile, target=None, min_positives=1)

    minority = (df["nc"] != 5.0).cast(pl.Int8).to_numpy()
    assert int(minority.sum()) == len(off_mode)
    found = any(
        np.array_equal(mat[:, j], minority) for j in range(mat.shape[1])
    )
    assert found, "NearConstant column must emit an off-mode minority signal"


def test_bounded_discrete_near_constant_uses_exact_equality():
    """A BoundedDiscrete NearConstant column marks the minority by exact inequality."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.config import SemanticType
    from dataforge_ml.profiling._config import NumericKind
    from dataforge_ml.profiling._numeric_config import NumericFlag
    import numpy as np

    n = 300
    # 96% at value 3; the rest spread across the bounded 0..5 domain.
    off_mode = [0, 1, 2, 4, 5, 0, 1, 2, 4, 5, 0, 1]
    vals = [3] * (n - len(off_mode)) + off_mode
    df = pl.DataFrame({"nc": pl.Series(vals, dtype=pl.Int64)})
    cfg = PipelineConfig()
    cfg.set_column_type("nc", SemanticType.Numeric)
    cfg.set_numeric_kind("nc", NumericKind.BoundedDiscrete)
    profile = StructuralProfiler(cfg).profile(df)
    assert profile.columns["nc"].stats.has_flag(NumericFlag.NearConstant)
    assert profile.columns["nc"].numeric_kind == NumericKind.BoundedDiscrete

    mat = build_label_matrix(df, profile, target=None, min_positives=1)

    minority = (df["nc"] != 3).cast(pl.Int8).to_numpy()
    assert int(minority.sum()) == len(off_mode)
    found = any(
        np.array_equal(mat[:, j], minority) for j in range(mat.shape[1])
    )
    assert found, "BoundedDiscrete NearConstant column must use exact-equality minority"


def test_zero_negative_signal_absent_for_strictly_positive_skewed_column():
    """A strictly-positive right-skewed column produces no zero/negative signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._numeric_config import SkewSeverity
    import numpy as np

    n = 300
    # Right-skewed but strictly positive (min > 0): a heavy right tail on a body
    # confined to [1, 7].
    vals = [1.0 + float(i % 7) for i in range(n)]
    for i in range(20):
        vals[i] = 200.0 + float(i)
    df = pl.DataFrame({"x": pl.Series(vals, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    stats = profile.columns["x"].stats
    # Guards: strictly positive, right-skewed, and severe enough that the
    # zero/negative branch would fire if the min<=0 gate were absent.
    assert stats.min > 0
    assert stats.skewness is not None and stats.skewness > 0
    assert stats.skewness_severity in (SkewSeverity.High, SkewSeverity.Severe)

    # min_positives=0 lets even all-zeros signals through the viability gate, so
    # an all-zeros zero/negative label would surface here if it were generated.
    mat = build_label_matrix(df, profile, target=None, min_positives=0)
    zero_mask = (df["x"] <= 0).cast(pl.Int8).to_numpy()  # all zeros
    assert not any(
        np.array_equal(mat[:, j], zero_mask) for j in range(mat.shape[1])
    ), "strictly-positive column must not produce a zero/negative signal"


# ---------------------------------------------------------------------------
# build_label_matrix — Bimodal minority-cluster signal
# ---------------------------------------------------------------------------


def _bimodal_frame(seed: int = 0):
    """Build a two-cluster numeric column plus its nearest-center minority mask.

    240 rows form a wide majority cluster near 0 and 60 rows a tight minority
    cluster near 8. The minority mode sits inside the ``[p5, p95]`` body, so the
    numeric extreme-value signal cannot see most of it — the case the
    bimodal-cluster signal exists to cover.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    vals = np.concatenate([rng.normal(0.0, 1.0, 240), rng.normal(8.0, 0.3, 60)])
    df = pl.DataFrame({"x": pl.Series(vals, dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    bs = profile.columns["x"].stats.bimodal_stats
    d1 = np.abs(vals - bs.center1)
    d2 = np.abs(vals - bs.center2)
    # Cluster 2 (near 8) is the less-populous cluster, so it is the minority.
    minority = (d1 > d2).astype("int8")
    return df, profile, minority, vals


def test_bimodal_minority_signal_invisible_to_extreme_value_signal():
    """A minority mode inside the body emits a cluster signal the extreme misses."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    from dataforge_ml.profiling._numeric_config import NumericFlag, SkewSeverity
    import numpy as np

    df, profile, minority, vals = _bimodal_frame()
    stats = profile.columns["x"].stats
    # Guard: the flag and its stats must be set, else the branch is dead.
    assert stats.has_flag(NumericFlag.Bimodal)
    assert stats.bimodal_stats is not None
    assert int(minority.sum()) == 60

    mat = build_label_matrix(df, profile, target=None, min_positives=1)
    found = any(np.array_equal(mat[:, j], minority) for j in range(mat.shape[1]))
    assert found, "Bimodal column must emit a less-populous-cluster minority signal"

    # The extreme-value signal only reaches the tail past p95, so most minority
    # rows sit inside the body and are invisible to it — the value the cluster
    # signal adds.
    p5 = stats.percentiles.p5
    p_high = (
        stats.percentiles.p99
        if stats.skewness_severity == SkewSeverity.Severe
        else stats.percentiles.p95
    )
    extreme = ((vals < p5) | (vals > p_high)).astype("int8")
    missed = int(((minority == 1) & (extreme == 0)).sum())
    assert missed > 0, "extreme signal alone should miss body-side minority rows"
    assert not np.array_equal(minority, extreme)


def test_bimodal_signal_uses_split_counts_not_minority_weight():
    """The positive class is the cluster with fewer split rows, ignoring weight."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    import numpy as np

    df, profile, minority, _ = _bimodal_frame()
    mat = build_label_matrix(df, profile, target=None, min_positives=1)
    # The emitted signal marks exactly the 60-row (less-populous) cluster, never
    # the 240-row majority — the count decides the sign, not center ordering.
    majority = 1 - minority
    assert any(np.array_equal(mat[:, j], minority) for j in range(mat.shape[1]))
    assert not any(np.array_equal(mat[:, j], majority) for j in range(mat.shape[1]))


def test_bimodal_signal_evicted_below_viability_floor():
    """A minority cluster smaller than the floor is dropped before capping."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    import numpy as np

    df, profile, minority, _ = _bimodal_frame()
    # 60 minority rows: a floor of 61 cannot place that many positives in both
    # partitions, so the signal fails the two-sided viability gate.
    mat = build_label_matrix(df, profile, target=None, min_positives=61)
    assert not any(
        np.array_equal(mat[:, j], minority) for j in range(mat.shape[1])
    ), "sub-floor bimodal minority signal must be evicted"


def test_bimodal_signal_collapses_against_correlated_higher_priority_signal():
    """A perfectly correlated target class collapses the lower-priority cluster signal."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    import numpy as np

    df, profile, minority, _ = _bimodal_frame()
    # Control: with no correlated partner the cluster signal is present.
    base = build_label_matrix(df, profile, target=None, min_positives=1)
    assert any(np.array_equal(base[:, j], minority) for j in range(base.shape[1]))

    # A binary target that mirrors cluster membership yields a target-class
    # signal (priority 0) perfectly correlated with the bimodal cluster signal
    # (priority numeric). The redundancy gate keeps the higher-priority target
    # and drops the cluster signal.
    tgt = np.where(minority == 1, "min", "maj")
    df2 = df.with_columns(pl.Series("target", tgt, dtype=pl.Utf8))
    profile2 = StructuralProfiler(PipelineConfig()).profile(df2)
    mat2 = build_label_matrix(df2, profile2, target="target", min_positives=1)
    # The two-class target collapses to one class signal (dummy-drop), which is
    # perfectly correlated (±1) with the cluster membership. Count columns in the
    # cluster family: without a collapse there would be two (surviving target
    # class + bimodal cluster); the redundancy gate leaves exactly one.
    complement = 1 - minority
    family = sum(
        1
        for j in range(mat2.shape[1])
        if np.array_equal(mat2[:, j], minority)
        or np.array_equal(mat2[:, j], complement)
    )
    assert family == 1, "cluster signal must collapse against the correlated target class"


# ---------------------------------------------------------------------------
# SplitConfig — dataclass basics
# ---------------------------------------------------------------------------


def test_split_config_defaults():
    from dataforge_ml.splitting._config import SplitConfig
    cfg = SplitConfig()
    assert cfg.max_stratification_signals == 50
    assert cfg.rows_per_signal == 10
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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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

    mat = build_label_matrix(df, profile, target=None, config=SplitConfig(rows_per_signal=1))

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


# ---------------------------------------------------------------------------
# build_label_matrix — signal 8: compound row missingness
# ---------------------------------------------------------------------------


def test_signal_8_absent_when_p90_is_zero():
    """No compound row signal when dataset has no effective nulls (p90 == 0)."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    df = pl.DataFrame({
        "a": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64),
        "b": pl.Series([10, 20, 30, 40, 50], dtype=pl.Int64),
    })
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    assert profile.dataset.row_distribution.row_missingness_p90 == 0
    mat = build_label_matrix(df, profile, target=None)
    assert mat.shape[1] == 0


def _make_compound_df() -> pl.DataFrame:
    """12 rows, 5 columns. Rows 0-9 fully present. Row 10 is null in cols a & b;
    row 11 is null in cols c & d. Per-row null counts: [0]*10 + [2, 2], so p90 = 1
    and the compound signal marks rows 10 and 11 (count 2 > p90).

    The two globally-sparse rows are null in *disjoint* column pairs, so the
    compound signal is not near-identical to (nor a mirror image of) any single
    per-column missingness signal and therefore survives the ADR-0047 redundancy
    gate — unlike a fixture where a lone sparse row's nulls all fall in one column.
    """
    return pl.DataFrame({
        "a": pl.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, 11], dtype=pl.Int64),
        "b": pl.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, 11], dtype=pl.Int64),
        "c": pl.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None], dtype=pl.Int64),
        "d": pl.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None], dtype=pl.Int64),
        "e": pl.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=pl.Int64),
    })


def test_signal_8_present_when_p90_is_positive():
    """Compound row signal column is present and non-zero when p90 > 0."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix

    df = _make_compound_df()
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    p90 = profile.dataset.row_distribution.row_missingness_p90
    assert p90 > 0, f"precondition: p90 must be positive, got {p90}"

    # rows_per_signal=1 keeps the gate-4 cap from evicting signals on this tiny
    # fixture, isolating the compound signal's survival of the redundancy gate.
    mat = build_label_matrix(
        df, profile, target=None, config=SplitConfig(rows_per_signal=1)
    )
    # The compound signal marks rows 10 and 11 (count 2 > p90); it must be
    # non-zero, so it survives the viability and redundancy gates.
    assert mat.shape[1] >= 1
    # At least one column has a 1 in each globally-sparse row.
    assert mat[10, :].sum() > 0, "row 10 (globally sparse) should be marked in some signal"
    assert mat[11, :].sum() > 0, "row 11 (globally sparse) should be marked in some signal"


def test_signal_8_rows_above_p90_receive_label_one():
    """Rows with effective-null count > p90 receive label 1; others receive 0."""
    from dataforge_ml.splitting._profile_signals import build_label_matrix
    import numpy as np

    # 12 rows, 5 columns. Per-row null counts: [0]*10 + [2, 2] (rows 10 and 11).
    # p90 = 1, so the compound signal flags rows with count > 1 → rows 10 and 11.
    df = _make_compound_df()
    profile = StructuralProfiler(PipelineConfig()).profile(df)

    p90 = profile.dataset.row_distribution.row_missingness_p90
    assert p90 > 0, f"precondition: p90 must be positive, got {p90}"

    mat = build_label_matrix(
        df, profile, target=None, config=SplitConfig(rows_per_signal=1)
    )

    per_row_null = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2])
    expected_compound = (per_row_null > p90).astype(np.int8)

    # The compound signal must appear as one column in the matrix.
    found = any(np.array_equal(mat[:, j], expected_compound) for j in range(mat.shape[1]))
    assert found, (
        f"compound signal column not found in matrix; p90={p90}, "
        f"expected {expected_compound.tolist()}, matrix shape {mat.shape}"
    )


# ---------------------------------------------------------------------------
# Unsplittable rare-row routing into the training split (ADR-0048)
# ---------------------------------------------------------------------------


def test_unsplittable_mask_flags_sub_floor_rare_categorical():
    """unsplittable_train_mask marks rows of a rare value below the floor."""
    from dataforge_ml.splitting._profile_signals import unsplittable_train_mask

    n = 300
    # "Z" occurs once (0.33%) → rare and below any reasonable floor.
    cat = ["A"] * 150 + ["B"] * 149 + ["Z"]
    df = pl.DataFrame({"cat": pl.Series(cat, dtype=pl.Utf8)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    assert "Z" in profile.columns["cat"].stats.rare_categories.rare_label_values

    mask = unsplittable_train_mask(df, profile, target=None, min_positives=10)
    assert mask.sum() == 1
    assert mask[299]


def test_unsplittable_mask_empty_when_no_sub_floor_rows():
    """No rare/sparse labels → an all-False mask (routing is a no-op)."""
    from dataforge_ml.splitting._profile_signals import unsplittable_train_mask

    n = 300
    df = pl.DataFrame(
        {"label": pl.Series(["A" if i % 2 == 0 else "B" for i in range(n)], dtype=pl.Utf8)}
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    mask = unsplittable_train_mask(df, profile, target="label", min_positives=10)
    assert mask.sum() == 0


def test_profile_split_routes_sub_floor_rare_categorical_to_train():
    """A lone rare categorical value lands entirely in the training split."""
    n = 300
    cat = ["A"] * 150 + ["B"] * 149 + ["Z"]
    label = ["x" if i % 2 == 0 else "y" for i in range(n)]
    df = pl.DataFrame(
        {
            "cat": pl.Series(cat, dtype=pl.Utf8),
            "label": pl.Series(label, dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, target="label", random_seed=7)
    result = splitter.profile_stratified_split(profile, test_size=0.2)

    assert (result.test["cat"] == "Z").sum() == 0
    assert (result.train["cat"] == "Z").sum() == 1
    assert result.train_size + result.test_size == n


def test_profile_split_routes_sub_floor_target_class_to_train():
    """A lone target class lands entirely in the training split."""
    n = 300
    # "C" is a single-row class, unsplittable at any floor >= 2.
    label = ["A"] * 150 + ["B"] * 149 + ["C"]
    df = pl.DataFrame(
        {
            "feature": pl.Series([float(i) for i in range(n)], dtype=pl.Float64),
            "label": pl.Series(label, dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, target="label", random_seed=7)
    result = splitter.profile_stratified_split(profile, test_size=0.2)

    assert (result.test["label"] == "C").sum() == 0
    assert (result.train["label"] == "C").sum() == 1
    assert result.train_size + result.test_size == n


def test_profile_kfold_routes_lone_rare_row_to_every_train():
    """A lone rare categorical row is in every fold's train, never in val."""
    n = 300
    cat = ["A"] * 150 + ["B"] * 149 + ["Z"]
    label = ["x" if i % 2 == 0 else "y" for i in range(n)]
    df = pl.DataFrame(
        {
            "cat": pl.Series(cat, dtype=pl.Utf8),
            "label": pl.Series(label, dtype=pl.Utf8),
        }
    )
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    splitter = DataSplitter(df, target="label", random_seed=7)
    folds = splitter.profile_stratified_kfold(profile, k=5)

    for fold in folds:
        assert (fold.val["cat"] == "Z").sum() == 0, (
            f"fold {fold.fold_index}: unsplittable rare row leaked into val"
        )
        assert (fold.train["cat"] == "Z").sum() == 1
