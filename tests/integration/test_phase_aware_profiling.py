"""
Integration tests for phase-aware column skipping through StructuralProfiler.

Verifies end-to-end exclusion behaviour: hard-excluded columns must be absent
from the result entirely; soft-excluded columns must be present but unprofiled
(stats=None).
"""

import polars as pl
import pytest

from dataforge_ml.profiling import StructuralProfiler, PipelineConfig, PipelinePhase


@pytest.fixture(scope="module")
def sample_df() -> pl.DataFrame:
    n = 60
    return pl.DataFrame({
        "id":      pl.Series(list(range(n)), dtype=pl.Int64),
        "age":     pl.Series([25 + i % 50 for i in range(n)], dtype=pl.Int64),
        "income":  pl.Series([30_000.0 + i * 500.0 for i in range(n)], dtype=pl.Float64),
        "country": pl.Series(["US", "UK", "CA", "AU", "DE"] * (n // 5), dtype=pl.Utf8),
    })


# ---------------------------------------------------------------------------
# Hard exclusions
# ---------------------------------------------------------------------------


def test_hard_excluded_column_absent_from_result(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_exclusion("id"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "id" not in result.columns


def test_hard_excluded_column_does_not_affect_other_columns(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_exclusion("id"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "age" in result.columns
    assert "income" in result.columns
    assert "country" in result.columns


def test_multiple_hard_exclusions_all_absent(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_exclusion(["id", "age"]))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "id" not in result.columns
    assert "age" not in result.columns
    assert "income" in result.columns


# ---------------------------------------------------------------------------
# Soft exclusions (Profiling phase)
# ---------------------------------------------------------------------------


def test_soft_excluded_column_present_in_result(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_phase_exclusion(PipelinePhase.Profiling, "age"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "age" in result.columns


def test_soft_excluded_column_has_stats_none(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_phase_exclusion(PipelinePhase.Profiling, "age"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert result.columns["age"].stats is None


def test_soft_excluded_column_not_type_detected(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_phase_exclusion(PipelinePhase.Profiling, "age"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert result.columns["age"].semantic_type is None


def test_soft_exclusion_does_not_affect_other_columns(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_phase_exclusion(PipelinePhase.Profiling, "age"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert result.columns["income"].stats is not None
    assert result.columns["country"].stats is not None


# ---------------------------------------------------------------------------
# Hard takes precedence over soft
# ---------------------------------------------------------------------------


def test_column_in_both_hard_and_soft_is_absent(sample_df):
    cfg = PipelineConfig()
    cfg.add_exclusion("age")
    cfg.add_phase_exclusion(PipelinePhase.Profiling, "age")
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "age" not in result.columns


# ---------------------------------------------------------------------------
# PipelineConfig.exclude_columns applied as hard exclusions (replaces AC8 —
# the backward-compat ProfileConfig wrapper was removed by design; this test
# verifies the equivalent behaviour through the current API)
# ---------------------------------------------------------------------------


def test_pipeline_config_exclude_columns_act_as_hard_exclusions(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_exclusion("id"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert "id" not in result.columns
    remaining = set(result.columns.keys())
    assert remaining == {"age", "income", "country"}


# ---------------------------------------------------------------------------
# DatasetStats.column_count reflects the raw DataFrame, not the active column set
# ---------------------------------------------------------------------------


def test_dataset_column_count_reflects_raw_dataframe_with_hard_exclusion(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_exclusion("id"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert result.dataset.column_count == sample_df.width


def test_dataset_column_count_reflects_raw_dataframe_with_soft_exclusion(sample_df):
    cfg = (cfg := PipelineConfig(), cfg.add_phase_exclusion(PipelinePhase.Profiling, "age"))[0]
    result = StructuralProfiler(cfg).profile(sample_df)

    assert result.dataset.column_count == sample_df.width


def test_dataset_column_count_reflects_raw_dataframe_no_exclusions(sample_df):
    result = StructuralProfiler(PipelineConfig()).profile(sample_df)

    assert result.dataset.column_count == sample_df.width


# ---------------------------------------------------------------------------
# Declared per-column datetime format (set_datetime_format) — orchestrator seam
# ---------------------------------------------------------------------------


def _bare_year_df() -> pl.DataFrame:
    return pl.DataFrame({
        "Year": pl.Series([str(2000 + (i % 20)) for i in range(60)], dtype=pl.Utf8),
        "value": pl.Series([float(i) for i in range(60)], dtype=pl.Float64),
    })


def test_bare_year_override_with_declared_format_profiles_successfully():
    from dataforge_ml.config import SemanticType

    df = _bare_year_df()
    cfg = PipelineConfig()
    cfg.set_column_type("Year", SemanticType.Datetime)
    cfg.profiling.set_datetime_format("Year", "%Y")

    result = StructuralProfiler(cfg).profile(df)

    assert "Year" in result.columns
    stats = result.columns["Year"].stats
    assert stats is not None
    assert stats.min_date is not None
    assert stats.min_date.year == 2000


def test_bare_year_override_without_declared_format_raises():
    from dataforge_ml.config import SemanticType
    from dataforge_ml.profiling import OverrideCoercionError

    df = _bare_year_df()
    cfg = PipelineConfig()
    cfg.set_column_type("Year", SemanticType.Datetime)

    with pytest.raises(OverrideCoercionError, match="set_datetime_format"):
        StructuralProfiler(cfg).profile(df)


def test_wrong_declared_format_still_raises_with_hint():
    from dataforge_ml.config import SemanticType
    from dataforge_ml.profiling import OverrideCoercionError

    df = _bare_year_df()
    cfg = PipelineConfig()
    cfg.set_column_type("Year", SemanticType.Datetime)
    cfg.profiling.set_datetime_format("Year", "%H:%M:%S")

    with pytest.raises(OverrideCoercionError, match="set_datetime_format"):
        StructuralProfiler(cfg).profile(df)


# ---------------------------------------------------------------------------
# FormatMismatch flag through the orchestrator (effective-null normalization)
# ---------------------------------------------------------------------------


def _datetime_override_cfg(column: str = "d") -> PipelineConfig:
    from dataforge_ml.config import SemanticType

    cfg = PipelineConfig()
    cfg.set_column_type(column, SemanticType.Datetime)
    return cfg


def test_recognized_missing_markers_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._datetime_config import DatetimeFlag

    # "NA" / "?" / empty strings are Effective Nulls, not dirt.
    values = ["2024-01-01", "NA", "?", "", "2024-02-01"] * 4
    df = pl.DataFrame({"d": pl.Series(values, dtype=pl.Utf8)})
    cfg = _datetime_override_cfg("d")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["d"].stats
    assert stats is not None
    assert not stats.has_flag(DatetimeFlag.FormatMismatch)


def test_declared_string_sentinels_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._datetime_config import DatetimeFlag

    values = ["2024-01-01", "MISSING", "2024-02-01"] * 8
    df = pl.DataFrame({"d": pl.Series(values, dtype=pl.Utf8)})
    cfg = _datetime_override_cfg("d")
    cfg.profiling.set_string_sentinel("d", ["MISSING"])

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["d"].stats
    assert stats is not None
    assert not stats.has_flag(DatetimeFlag.FormatMismatch)


def test_genuine_dirt_trips_format_mismatch_through_orchestrator():
    from dataforge_ml.profiling._datetime_config import DatetimeFlag

    values = ["2024-01-01", "banana", "2024-02-01"] * 8
    df = pl.DataFrame({"d": pl.Series(values, dtype=pl.Utf8)})
    cfg = _datetime_override_cfg("d")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["d"].stats
    assert stats is not None
    assert stats.has_flag(DatetimeFlag.FormatMismatch)


def test_substantially_missing_and_dirty_carries_both_signals():
    from dataforge_ml.profiling._datetime_config import DatetimeFlag

    # Mostly recognized missing-markers, a few valid dates, and one dirty value.
    values = ["NA"] * 45 + ["2024-01-01"] * 10 + ["banana"] * 5
    df = pl.DataFrame({"d": pl.Series(values, dtype=pl.Utf8)})
    cfg = _datetime_override_cfg("d")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["d"].stats
    assert stats is not None
    assert stats.has_flag(DatetimeFlag.MnarSuspected)
    assert stats.has_flag(DatetimeFlag.FormatMismatch)


# ---------------------------------------------------------------------------
# FormatMismatch flag on numeric columns through the orchestrator
# ---------------------------------------------------------------------------


def _numeric_override_cfg(column: str = "n") -> PipelineConfig:
    from dataforge_ml.config import SemanticType

    cfg = PipelineConfig()
    cfg.set_column_type(column, SemanticType.Numeric)
    return cfg


def test_numeric_recognized_missing_markers_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._numeric_config import NumericFlag

    # "NA" / "?" / empty are Effective Nulls resolved by the orchestrator.
    values = ["1", "NA", "?", "", "2", "3"] * 4
    df = pl.DataFrame({"n": pl.Series(values, dtype=pl.Utf8)})
    cfg = _numeric_override_cfg("n")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["n"].stats
    assert stats is not None
    assert not stats.has_flag(NumericFlag.FormatMismatch)


def test_numeric_declared_string_sentinels_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._numeric_config import NumericFlag

    values = ["1", "MISSING", "2", "3"] * 6
    df = pl.DataFrame({"n": pl.Series(values, dtype=pl.Utf8)})
    cfg = _numeric_override_cfg("n")
    cfg.profiling.set_string_sentinel("n", ["MISSING"])

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["n"].stats
    assert stats is not None
    assert not stats.has_flag(NumericFlag.FormatMismatch)


def test_numeric_genuine_dirt_trips_format_mismatch_through_orchestrator():
    from dataforge_ml.profiling._numeric_config import NumericFlag

    values = ["1", "banana", "2", "3"] * 6
    df = pl.DataFrame({"n": pl.Series(values, dtype=pl.Utf8)})
    cfg = _numeric_override_cfg("n")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["n"].stats
    assert stats is not None
    assert stats.has_flag(NumericFlag.FormatMismatch)


# ---------------------------------------------------------------------------
# FormatMismatch flag on boolean columns through the orchestrator
# ---------------------------------------------------------------------------


def _boolean_override_cfg(column: str = "b") -> PipelineConfig:
    from dataforge_ml.config import SemanticType

    cfg = PipelineConfig()
    cfg.set_column_type(column, SemanticType.Boolean)
    return cfg


def test_boolean_recognized_missing_markers_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._boolean_config import BooleanFlag

    values = ["yes", "NA", "?", "", "no", "yes"] * 4
    df = pl.DataFrame({"b": pl.Series(values, dtype=pl.Utf8)})
    cfg = _boolean_override_cfg("b")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["b"].stats
    assert stats is not None
    assert not stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_declared_string_sentinels_do_not_trip_format_mismatch():
    from dataforge_ml.profiling._boolean_config import BooleanFlag

    values = ["yes", "MISSING", "no", "yes"] * 6
    df = pl.DataFrame({"b": pl.Series(values, dtype=pl.Utf8)})
    cfg = _boolean_override_cfg("b")
    cfg.profiling.set_string_sentinel("b", ["MISSING"])

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["b"].stats
    assert stats is not None
    assert not stats.has_flag(BooleanFlag.FormatMismatch)


def test_boolean_genuine_dirt_trips_format_mismatch_through_orchestrator():
    from dataforge_ml.profiling._boolean_config import BooleanFlag

    values = ["yes", "maybe", "no", "yes"] * 6
    df = pl.DataFrame({"b": pl.Series(values, dtype=pl.Utf8)})
    cfg = _boolean_override_cfg("b")

    result = StructuralProfiler(cfg).profile(df)
    stats = result.columns["b"].stats
    assert stats is not None
    assert stats.has_flag(BooleanFlag.FormatMismatch)
