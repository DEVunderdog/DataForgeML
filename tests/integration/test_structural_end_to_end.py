import pytest
from ...profiling.structural import StructuralProfiler
from ...profiling.config import ProfileConfig, StructuralProfileResult


def test_happy_path(mixed_df):
    config = ProfileConfig(compute_correlation=True)
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
    config = ProfileConfig(compute_correlation=False)
    result = StructuralProfiler(config).profile(mixed_df)

    assert result.dataset.feature_correlation is None
