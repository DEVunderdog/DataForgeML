import polars as pl
import pytest
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType
from dataforge_ml.profiling._config import ProfileConfig, StructuralProfileResult
from dataforge_ml.profiling._numeric_config import NumericStats, SkewSeverity
from dataforge_ml.profiling._categorical_config import CategoricalStats
from dataforge_ml.profiling._datetime_config import DatetimeStats
from dataforge_ml.profiling._boolean_config import BooleanStats
from dataforge_ml.profiling._text_config import TextStats
from dataforge_ml.profiling._target_config import TargetProfileResult
from dataforge_ml.profiling._missingness_config import MissingSeverity, MissingnessProfileConfig
from dataforge_ml.profiling._numeric_config import NumericProfileConfig


def test_happy_path(mixed_df):
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=True))
    result = StructuralProfiler(config).profile(mixed_df)

    assert isinstance(result, StructuralProfileResult)
    assert set(result.columns.keys()) == set(mixed_df.columns)
    for col_profile in result.columns.values():
        assert (
            col_profile.semantic_type is not None
        ), f"column '{col_profile.name}' has no semantic_type"


    assert result.dataset.row_count == mixed_df.height
    assert result.dataset.feature_correlation is not None


def test_no_correlation(mixed_df):
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=False))
    result = StructuralProfiler(config).profile(mixed_df)

    assert result.dataset.feature_correlation is None


def test_boolean_handoff(mixed_df):
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)

    cp = result.columns["is_active"]
    assert cp.semantic_type == SemanticType.Boolean
    assert cp.stats is not None
    assert isinstance(cp.stats, BooleanStats)
    assert cp.stats.mode in (True, False, None)


def test_text_handoff(text_df):
    result = StructuralProfiler(PipelineConfig()).profile(text_df)

    cp = result.columns["review"]
    assert cp.semantic_type == SemanticType.Text
    assert cp.stats is not None
    assert isinstance(cp.stats, TextStats)

    assert cp.stats.vocabulary_size > 0
    assert cp.stats.char_length_max >= cp.stats.char_length_min
    assert cp.stats.avg_token_count > 0
    assert 0.0 <= cp.stats.empty_ratio <= 1.0


def test_correlation_consistency(mixed_df):
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=True))
    result = StructuralProfiler(config).profile(mixed_df)

    fc = result.dataset.feature_correlation
    assert fc is not None

    # age and income are correlated by construction — forward invariant must not be vacuous
    assert len(fc.near_redundant_pairs) >= 1, (
        "expected at least one near-redundant pair (age/income are strongly correlated)"
    )

    # Forward invariant: every near_redundant pair must have both columns co-located
    # in the same NearRedundancyGroup
    for pair in fc.pairwise:
        if not pair.near_redundant:
            continue
        assert any(
            pair.col_a in group.columns and pair.col_b in group.columns
            for group in fc.near_redundancy_groups
        ), (
            f"near_redundant pair ({pair.col_a}, {pair.col_b}) "
            f"not co-located in any NearRedundancyGroup"
        )

    # Backward invariant: every column in a redundancy group must have at least
    # one near_redundant=True pair in pairwise
    for group in fc.near_redundancy_groups:
        for col in group.columns:
            assert any(
                (p.col_a == col or p.col_b == col) and p.near_redundant
                for p in fc.pairwise
            ), (
                f"column '{col}' is in a NearRedundancyGroup but has no "
                f"near_redundant=True pair in pairwise"
            )

    # Matrix symmetry — Pearson
    for col_a, row in fc.pearson_matrix.items():
        for col_b, val in row.items():
            mirror = fc.pearson_matrix.get(col_b, {}).get(col_a)
            assert mirror is not None and abs(val - mirror) < 1e-10, (
                f"Pearson matrix asymmetry: [{col_a}][{col_b}]={val} "
                f"vs [{col_b}][{col_a}]={mirror}"
            )

    # Matrix symmetry — Spearman
    for col_a, row in fc.spearman_matrix.items():
        for col_b, val in row.items():
            mirror = fc.spearman_matrix.get(col_b, {}).get(col_a)
            assert mirror is not None and abs(val - mirror) < 1e-10, (
                f"Spearman matrix asymmetry: [{col_a}][{col_b}]={val} "
                f"vs [{col_b}][{col_a}]={mirror}"
            )

    # Suggested drop is a strict subset of its group's columns
    for group in fc.near_redundancy_groups:
        group_cols = set(group.columns)
        drop_cols = set(group.suggested_drop)
        assert drop_cols < group_cols, (
            f"suggested_drop {drop_cols} is not a strict subset of "
            f"group columns {group_cols}"
        )


def test_column_handoffs(mixed_df):
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)

    stats_type_for = {
        SemanticType.Numeric: NumericStats,
        SemanticType.Categorical: CategoricalStats,
        SemanticType.Datetime: DatetimeStats,
        SemanticType.Boolean: BooleanStats,
    }

    for name, cp in result.columns.items():
        expected_type = stats_type_for.get(cp.semantic_type)
        if expected_type is None:
            continue

        assert cp.stats is not None, (
            f"column '{name}' has semantic_type={cp.semantic_type} but stats is None"
        )
        assert isinstance(cp.stats, expected_type), (
            f"column '{name}' has semantic_type={cp.semantic_type} "
            f"but stats type is {type(cp.stats).__name__}, expected {expected_type.__name__}"
        )


# ---------------------------------------------------------------------------
# Override: numeric column forced to Categorical via column_overrides
# ---------------------------------------------------------------------------


def test_column_override_changes_stats_type(override_df):
    config = (cfg := PipelineConfig(), cfg.set_column_type("score", SemanticType.Categorical))[0]
    result = StructuralProfiler(config).profile(override_df)
    cp = result.columns["score"]
    assert isinstance(cp.stats, CategoricalStats)


# ---------------------------------------------------------------------------
# Target profiling integration
# ---------------------------------------------------------------------------


def test_target_profiling_integration(target_df):
    config = PipelineConfig(profiling=ProfileConfig(target_columns=["label"]))
    result = StructuralProfiler(config).profile(target_df)
    assert "label" in result.targets
    assert isinstance(result.targets["label"], TargetProfileResult)


# ---------------------------------------------------------------------------
# Empty DataFrame does not crash
# ---------------------------------------------------------------------------


def test_empty_dataframe_does_not_crash(empty_df):
    result = StructuralProfiler(PipelineConfig()).profile(empty_df)
    assert isinstance(result, StructuralProfileResult)


# ---------------------------------------------------------------------------
# Numeric handoff: float column produces NumericStats on ColumnProfile
# ---------------------------------------------------------------------------


def test_numeric_handoff(mixed_df):
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)
    cp = result.columns["income"]
    assert cp.stats is not None
    assert isinstance(cp.stats, NumericStats)


# ---------------------------------------------------------------------------
# Datetime handoff: date column produces DatetimeStats on ColumnProfile
# ---------------------------------------------------------------------------


def test_datetime_handoff(mixed_df):
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)
    cp = result.columns["joined"]
    assert cp.stats is not None
    assert isinstance(cp.stats, DatetimeStats)


# ---------------------------------------------------------------------------
# Missingness surfaced at column level for columns with nulls
# ---------------------------------------------------------------------------


def test_missingness_surfaced(mixed_df):
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)
    cp = result.columns["salary"]  # salary has ~10 % nulls by construction
    assert cp.missingness is not None
    assert cp.missingness.standard_null_count > 0


# ---------------------------------------------------------------------------
# Issue #75: NumericKind propagated from ColumnTypeInfo to ColumnProfile
# ---------------------------------------------------------------------------


def test_numeric_kind_set_for_numeric_columns(mixed_df):
    from dataforge_ml.profiling._config import NumericKind

    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)

    for name, cp in result.columns.items():
        if cp.semantic_type == SemanticType.Numeric:
            assert cp.numeric_kind in (NumericKind.BoundedDiscrete, NumericKind.Continuous), (
                f"column '{name}' is Numeric but numeric_kind={cp.numeric_kind!r}"
            )
        else:
            assert cp.numeric_kind is None, (
                f"column '{name}' has semantic_type={cp.semantic_type} "
                f"but numeric_kind={cp.numeric_kind!r} (expected None)"
            )


def test_numeric_kind_in_to_dict(mixed_df):
    from dataforge_ml.profiling._config import NumericKind

    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)

    for name, cp in result.columns.items():
        d = cp.to_dict()
        assert "numeric_kind" in d, f"column '{name}' to_dict() missing 'numeric_kind' key"
        if cp.semantic_type == SemanticType.Numeric:
            assert d["numeric_kind"] in (
                NumericKind.Continuous.value,
                NumericKind.BoundedDiscrete.value,
            ), f"column '{name}' to_dict()['numeric_kind']={d['numeric_kind']!r}"
        else:
            assert d["numeric_kind"] is None, (
                f"non-numeric column '{name}' to_dict()['numeric_kind']={d['numeric_kind']!r}"
            )


# ---------------------------------------------------------------------------
# Scope 15: sub-config override E2E tests
# ---------------------------------------------------------------------------


def test_default_profile_config_no_regression(mixed_df):
    # ProfileConfig() with all defaults must produce the same result as
    # an explicitly constructed PipelineConfig — no behaviour change from Scope 15.
    result_default = StructuralProfiler(PipelineConfig()).profile(mixed_df)
    result_explicit = StructuralProfiler(
        PipelineConfig(profiling=ProfileConfig())
    ).profile(mixed_df)

    assert set(result_default.columns.keys()) == set(result_explicit.columns.keys())
    for col in result_default.columns:
        cp_d = result_default.columns[col]
        cp_e = result_explicit.columns[col]
        assert cp_d.semantic_type == cp_e.semantic_type
        assert cp_d.missingness is None or (
            cp_d.missingness.severity == cp_e.missingness.severity
        )


def test_missingness_severity_override_via_profile_config():
    # Column with 12 % null ratio.
    # Default severity_high=0.20 → 0.12 < 0.20 → MissingSeverity.High
    # Override severity_high=0.10 → 0.12 >= 0.10 → MissingSeverity.Severe
    n = 100
    values = [None] * 12 + [1.0] * 88
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64)})

    result_default = StructuralProfiler(PipelineConfig()).profile(df)
    assert result_default.columns["x"].missingness.severity == MissingSeverity.High

    custom_profile = ProfileConfig(
        missingness=MissingnessProfileConfig(severity_high=0.10)
    )
    result_custom = StructuralProfiler(
        PipelineConfig(profiling=custom_profile)
    ).profile(df)
    assert result_custom.columns["x"].missingness.severity == MissingSeverity.Severe


def test_skew_high_override_via_profile_config():
    # exp(x/10) for x in [1,100] produces skewness ≈ 2.58 (verified empirically).
    # Default skew_high=2.0 → 2.58 > 2.0 → SkewSeverity.Severe.
    # Override skew_high=3.0 → 2.58 <= 3.0 AND > skew_moderate(1.0) → SkewSeverity.High.
    import math
    values = [math.exp(x / 10.0) for x in range(1, 101)]
    df = pl.DataFrame({"val": pl.Series(values, dtype=pl.Float64)})

    result_default = StructuralProfiler(PipelineConfig()).profile(df)
    stats_default = result_default.columns["val"].stats
    assert isinstance(stats_default, NumericStats)
    assert stats_default.skewness_severity == SkewSeverity.Severe

    custom_profile = ProfileConfig(
        numeric=NumericProfileConfig(skew_high=3.0)
    )
    result_custom = StructuralProfiler(
        PipelineConfig(profiling=custom_profile)
    ).profile(df)
    stats_custom = result_custom.columns["val"].stats
    assert isinstance(stats_custom, NumericStats)
    assert stats_custom.skewness_severity == SkewSeverity.High


def test_row_drop_threshold_override_via_profile_config():
    # 5 rows, each missing exactly 2 out of 4 columns (50 % each row).
    # Default row_drop_threshold=0.50 → ceil(4 * 0.50)=2 → rows with ≥2 missing are candidates
    # Override row_drop_threshold=0.60 → ceil(4 * 0.60)=3 → rows with ≥3 missing are candidates
    df = pl.DataFrame({
        "a": pl.Series([None, None, 1.0, 1.0, 1.0], dtype=pl.Float64),
        "b": pl.Series([None, None, 1.0, 1.0, 1.0], dtype=pl.Float64),
        "c": pl.Series([1.0, 1.0, None, None, 1.0], dtype=pl.Float64),
        "d": pl.Series([1.0, 1.0, None, None, 1.0], dtype=pl.Float64),
    })

    result_default = StructuralProfiler(PipelineConfig()).profile(df)
    assert result_default.dataset.row_distribution.drop_candidate_row_count == 4

    custom_profile = ProfileConfig(row_drop_threshold=0.60)
    result_custom = StructuralProfiler(
        PipelineConfig(profiling=custom_profile)
    ).profile(df)
    assert result_custom.dataset.row_distribution.drop_candidate_row_count == 0


# ---------------------------------------------------------------------------
# numeric_sentinels propagation — ProfileConfig → StructuralProfileResult
# ---------------------------------------------------------------------------


def test_profiler_propagates_numeric_sentinels_to_result():
    df = pl.DataFrame({
        "age": pl.Series([25, -999, 30], dtype=pl.Int64),
        "score": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
    })
    sentinels = {"age": [-999.0]}
    config = PipelineConfig(profiling=ProfileConfig(numeric_sentinels=sentinels))
    result = StructuralProfiler(config).profile(df)

    assert result.numeric_sentinels == sentinels


def test_profiler_numeric_sentinels_empty_when_none_declared():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3], dtype=pl.Int64)})
    result = StructuralProfiler(PipelineConfig()).profile(df)

    assert result.numeric_sentinels == {}


def test_profiler_numeric_sentinels_multi_column_propagation():
    df = pl.DataFrame({
        "income": pl.Series([50000, -1, 75000], dtype=pl.Int64),
        "age": pl.Series([25, 9999, 30], dtype=pl.Int64),
    })
    sentinels = {"income": [-1.0], "age": [9999.0]}
    config = PipelineConfig(profiling=ProfileConfig(numeric_sentinels=sentinels))
    result = StructuralProfiler(config).profile(df)

    assert result.numeric_sentinels == sentinels


# ---------------------------------------------------------------------------
# string_sentinels propagation — ProfileConfig → StructuralProfileResult
# ---------------------------------------------------------------------------


def test_profiler_propagates_string_sentinels_to_result():
    df = pl.DataFrame({
        "status": pl.Series(["active", "N/A", "inactive"], dtype=pl.String),
        "score": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
    })
    sentinels = {"status": ["N/A", "missing"]}
    config = PipelineConfig(profiling=ProfileConfig(string_sentinels=sentinels))
    result = StructuralProfiler(config).profile(df)

    assert result.string_sentinels == sentinels


def test_profiler_string_sentinels_empty_when_none_declared():
    df = pl.DataFrame({"status": pl.Series(["active", "inactive"], dtype=pl.String)})
    result = StructuralProfiler(PipelineConfig()).profile(df)

    assert result.string_sentinels == {}


def test_profiler_string_sentinels_multi_column_propagation():
    df = pl.DataFrame({
        "status": pl.Series(["active", "N/A", "inactive"], dtype=pl.String),
        "grade": pl.Series(["A", "?", "B"], dtype=pl.String),
    })
    sentinels = {"status": ["N/A", "missing"], "grade": ["?"]}
    config = PipelineConfig(profiling=ProfileConfig(string_sentinels=sentinels))
    result = StructuralProfiler(config).profile(df)

    assert result.string_sentinels == sentinels


def test_profiler_numeric_and_string_sentinels_propagate_independently():
    df = pl.DataFrame({
        "age": pl.Series([25, -999, 30], dtype=pl.Int64),
        "status": pl.Series(["active", "N/A", "inactive"], dtype=pl.String),
    })
    num_sentinels = {"age": [-999.0]}
    str_sentinels = {"status": ["N/A"]}
    config = PipelineConfig(profiling=ProfileConfig(
        numeric_sentinels=num_sentinels,
        string_sentinels=str_sentinels,
    ))
    result = StructuralProfiler(config).profile(df)

    assert result.numeric_sentinels == num_sentinels
    assert result.string_sentinels == str_sentinels
