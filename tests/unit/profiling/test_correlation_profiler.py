import polars as pl

from dataforge_ml.profiling._correlation_profiler import CorrelationProfiler
from dataforge_ml.profiling._correlation_config import (
    CorrelationProfileConfig,
    CorrelationProfileResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df_three_cols():
    """Three numeric columns with moderate (non-trivial) correlations."""
    n = 50
    a = [float(i % 7) for i in range(n)]
    b = [float((i * 3) % 11) for i in range(n)]
    c = [float(i % 13) for i in range(n)]
    return pl.DataFrame(
        {
            "a": pl.Series(a, dtype=pl.Float64),
            "b": pl.Series(b, dtype=pl.Float64),
            "c": pl.Series(c, dtype=pl.Float64),
        }
    )


def _make_df_with_duplicate():
    """Two identical columns plus one independent column."""
    n = 40
    vals = [float(i) for i in range(n)]
    return pl.DataFrame(
        {
            "x": pl.Series(vals, dtype=pl.Float64),
            "x_copy": pl.Series(vals, dtype=pl.Float64),
            "y": pl.Series([v * 0.3 + 7.0 for v in vals], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Pearson matrix symmetry  (profile_features)
# ---------------------------------------------------------------------------


def test_pearson_matrix_is_symmetric():
    df = _make_df_three_cols()
    cols = ["a", "b", "c"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for col_x in cols:
        for col_y in cols:
            assert result.pearson_matrix[col_x][col_y] == result.pearson_matrix[col_y][col_x]


# ---------------------------------------------------------------------------
# Spearman matrix symmetry  (profile_features)
# ---------------------------------------------------------------------------


def test_spearman_matrix_is_symmetric():
    df = _make_df_three_cols()
    cols = ["a", "b", "c"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for col_x in cols:
        for col_y in cols:
            assert result.spearman_matrix[col_x][col_y] == result.spearman_matrix[col_y][col_x]


# ---------------------------------------------------------------------------
# Every group column appears in at least one near-redundant pairwise entry
# (profile_features)
# ---------------------------------------------------------------------------


def test_near_redundancy_group_columns_have_redundant_pairs():
    df = _make_df_with_duplicate()
    cols = ["x", "x_copy", "y"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for group in result.near_redundancy_groups:
        for col in group.columns:
            has_pair = any(
                (p.col_a == col or p.col_b == col) and p.near_redundant
                for p in result.pairwise
            )
            assert has_pair


# ---------------------------------------------------------------------------
# suggested_drop is a strict subset of the group's columns  (profile_features)
# ---------------------------------------------------------------------------


def test_suggested_drop_is_strict_subset_of_group_columns():
    df = _make_df_with_duplicate()
    cols = ["x", "x_copy", "y"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for group in result.near_redundancy_groups:
        drop_set = set(group.suggested_drop)
        col_set = set(group.columns)
        assert drop_set < col_set  # non-empty and strictly smaller


# ---------------------------------------------------------------------------
# Identical columns produce a near-redundant pair; profile_target also works
# ---------------------------------------------------------------------------


def test_identical_columns_produce_near_redundant_pair():
    df = _make_df_with_duplicate()
    feature_cols = ["x", "x_copy", "y"]
    profiler = CorrelationProfiler(numeric_columns=feature_cols)

    # profile_features: identical pair must be flagged near-redundant
    feature_result = profiler.profile_features(df, feature_cols)
    assert any(
        {p.col_a, p.col_b} == {"x", "x_copy"}
        for p in feature_result.near_redundant_pairs
    )

    # profile_target: second entry point must run without error and attach target info
    target_result = profiler.profile_target(
        df, feature_result, feature_cols, [], "y"
    )
    assert isinstance(target_result, CorrelationProfileResult)
    assert target_result.target_column == "y"


# ---------------------------------------------------------------------------
# Cramér's V — categorical ↔ categorical
# ---------------------------------------------------------------------------


def test_perfectly_correlated_categoricals_near_redundant():
    # Perfect 1:1 mapping between two categoricals → Cramér's V == 1.0
    n = 60
    col_a = ["A", "B", "C"] * (n // 3)
    col_b = ["X", "Y", "Z"] * (n // 3)  # perfect correspondence
    df = pl.DataFrame({
        "cat1": pl.Series(col_a, dtype=pl.Utf8),
        "cat2": pl.Series(col_b, dtype=pl.Utf8),
    })
    profiler = CorrelationProfiler(numeric_columns=[], categorical_columns=["cat1", "cat2"])
    result = profiler.profile_features(df, [], categorical_cols=["cat1", "cat2"])

    assert len(result.cramer_v_pairs) == 1
    pair = result.cramer_v_pairs[0]
    assert pair.cramer_v is not None
    assert pair.cramer_v > 0.8
    assert pair.near_redundant is True
    assert len(result.near_redundant_cramer_v_pairs) == 1


def test_independent_categoricals_not_near_redundant():
    import random
    rng = random.Random(42)
    n = 100
    col_a = [rng.choice(["A", "B", "C"]) for _ in range(n)]
    col_b = [rng.choice(["X", "Y", "Z"]) for _ in range(n)]
    df = pl.DataFrame({
        "cat1": pl.Series(col_a, dtype=pl.Utf8),
        "cat2": pl.Series(col_b, dtype=pl.Utf8),
    })
    profiler = CorrelationProfiler(numeric_columns=[], categorical_columns=["cat1", "cat2"])
    result = profiler.profile_features(df, [], categorical_cols=["cat1", "cat2"])

    assert len(result.cramer_v_pairs) == 1
    assert result.cramer_v_pairs[0].near_redundant is False


# ---------------------------------------------------------------------------
# Eta-squared — numeric ↔ categorical
# ---------------------------------------------------------------------------


def test_numeric_perfectly_separates_groups_near_redundant():
    # Numeric values are perfectly separated by the categorical groups.
    df = pl.DataFrame({
        "group": pl.Series(["A"] * 30 + ["B"] * 30, dtype=pl.Utf8),
        "value": pl.Series([1.0] * 30 + [100.0] * 30, dtype=pl.Float64),
    })
    profiler = CorrelationProfiler(
        numeric_columns=["value"], categorical_columns=["group"]
    )
    result = profiler.profile_features(df, ["value"], categorical_cols=["group"])

    assert len(result.eta_squared_pairs) == 1
    pair = result.eta_squared_pairs[0]
    assert pair.eta_squared is not None
    assert pair.eta_squared > 0.5
    assert pair.near_redundant is True
    assert len(result.near_redundant_eta_squared_pairs) == 1


def test_existing_numeric_pearson_behaviour_unchanged():
    df = _make_df_with_duplicate()
    feature_cols = ["x", "x_copy", "y"]
    profiler = CorrelationProfiler(numeric_columns=feature_cols)
    result = profiler.profile_features(df, feature_cols)

    assert any(
        {p.col_a, p.col_b} == {"x", "x_copy"}
        for p in result.near_redundant_pairs
    )
    assert result.cramer_v_pairs == []
    assert result.eta_squared_pairs == []


# ---------------------------------------------------------------------------
# Cramér's V — degenerate case: near-saturated contingency table
# ---------------------------------------------------------------------------


def test_cramer_v_near_saturated_does_not_raise():
    # When n_unique ≈ n_rows for a categorical column (e.g. a Name column that
    # slipped through type detection), the bias-corrected denominator collapses
    # to ≤ 0. The profiler must skip the pair silently rather than crashing.
    n = 50
    # col_a: 50 fully unique strings — r == n triggers the degenerate case
    col_a = [f"Name_{i}" for i in range(n)]
    col_b = ["A", "B", "C"] * 16 + ["A", "B"]
    df = pl.DataFrame({
        "name": pl.Series(col_a, dtype=pl.Utf8),
        "group": pl.Series(col_b, dtype=pl.Utf8),
    })
    profiler = CorrelationProfiler(
        numeric_columns=[], categorical_columns=["name", "group"]
    )
    # Must not raise; the degenerate pair should have cramer_v=None
    result = profiler.profile_features(df, [], categorical_cols=["name", "group"])
    assert len(result.cramer_v_pairs) == 1
    assert result.cramer_v_pairs[0].cramer_v is None


# ---------------------------------------------------------------------------
# CorrelationProfileConfig — threshold override tests
# ---------------------------------------------------------------------------


def test_pearson_threshold_override_flags_borderline_pair():
    # x = [0..99]; y = x with indices 10, 30, 50, 70, 90 set to 0.
    # Pearson |r| ≈ 0.91 — above 0.85 but below the default 0.95.
    n = 100
    x = [float(i) for i in range(n)]
    y = [0.0 if i in (10, 30, 50, 70, 90) else float(i) for i in range(n)]
    df = pl.DataFrame({
        "x": pl.Series(x, dtype=pl.Float64),
        "y": pl.Series(y, dtype=pl.Float64),
    })
    cols = ["x", "y"]

    # Default threshold (0.95): not near-redundant
    default_result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    pair = default_result.pairwise[0]
    assert pair.pearson_r is not None
    assert abs(pair.pearson_r) < 0.95
    assert pair.near_redundant is False

    # Lowered threshold (0.85): pair is now near-redundant
    config = CorrelationProfileConfig(near_redundant_pearson_threshold=0.85)
    override_result = CorrelationProfiler(
        numeric_columns=cols, config=config
    ).profile_features(df, cols)
    assert override_result.pairwise[0].near_redundant is True


def test_cramer_v_threshold_override_flags_borderline_pair():
    # 2×2 contingency table with 75/25 split → Cramér's V ≈ 0.50.
    # Default threshold (0.80): NOT near-redundant.
    # Lowered threshold (0.45): IS near-redundant.
    col_a = ["A"] * 100 + ["B"] * 100
    col_b = ["X"] * 75 + ["Y"] * 25 + ["X"] * 25 + ["Y"] * 75
    df = pl.DataFrame({
        "cat1": pl.Series(col_a, dtype=pl.Utf8),
        "cat2": pl.Series(col_b, dtype=pl.Utf8),
    })
    cat_cols = ["cat1", "cat2"]

    # Default threshold (0.80): not near-redundant
    default_profiler = CorrelationProfiler(
        numeric_columns=[], categorical_columns=cat_cols
    )
    default_result = default_profiler.profile_features(df, [], categorical_cols=cat_cols)
    pair = default_result.cramer_v_pairs[0]
    assert pair.cramer_v is not None
    assert pair.cramer_v < 0.80
    assert pair.near_redundant is False

    # Lowered threshold (0.45): near-redundant
    config = CorrelationProfileConfig(near_redundant_cramer_v_threshold=0.45)
    override_profiler = CorrelationProfiler(
        numeric_columns=[], categorical_columns=cat_cols, config=config
    )
    override_result = override_profiler.profile_features(df, [], categorical_cols=cat_cols)
    assert override_result.cramer_v_pairs[0].near_redundant is True


def test_correlation_profile_config_round_trip():
    cfg = CorrelationProfileConfig(
        near_redundant_pearson_threshold=0.85,
        near_redundant_cramer_v_threshold=0.70,
        near_redundant_eta_squared_threshold=0.40,
        mi_min_rows=20,
    )
    restored = CorrelationProfileConfig.from_dict(cfg.to_dict())
    assert restored.near_redundant_pearson_threshold == 0.85
    assert restored.near_redundant_cramer_v_threshold == 0.70
    assert restored.near_redundant_eta_squared_threshold == 0.40
    assert restored.mi_min_rows == 20
