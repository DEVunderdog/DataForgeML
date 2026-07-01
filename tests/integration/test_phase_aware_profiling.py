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
