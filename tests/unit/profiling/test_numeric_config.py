"""
Unit tests for NonlinearityTag enum and NumericStats extension (Issue #138).

Pure structural tests — no profiler runs, no DataFrames.
"""

from dataforge_ml.profiling._numeric_config import (
    NonlinearityTag,
    NumericStats,
)


# ---------------------------------------------------------------------------
# NonlinearityTag enum
# ---------------------------------------------------------------------------


def test_nonlinearity_tag_has_four_members():
    assert len(NonlinearityTag) == 4


def test_nonlinearity_tag_linear():
    assert NonlinearityTag.Linear == "linear"


def test_nonlinearity_tag_monotonic_nonlinear():
    assert NonlinearityTag.MonotonicNonlinear == "monotonic_nonlinear"


def test_nonlinearity_tag_complex_nonlinear():
    assert NonlinearityTag.ComplexNonlinear == "complex_nonlinear"


def test_nonlinearity_tag_unpredictable():
    assert NonlinearityTag.Unpredictable == "unpredictable"


# ---------------------------------------------------------------------------
# NumericStats — new fields default to None
# ---------------------------------------------------------------------------


def test_numeric_stats_nonlinearity_tag_defaults_to_none():
    stats = NumericStats()
    assert stats.nonlinearity_tag is None


def test_numeric_stats_spearman_pearson_discrepancy_defaults_to_none():
    stats = NumericStats()
    assert stats.spearman_pearson_discrepancy is None


def test_numeric_stats_mean_mutual_information_defaults_to_none():
    stats = NumericStats()
    assert stats.mean_mutual_information is None


def test_numeric_stats_r2_gap_defaults_to_none():
    stats = NumericStats()
    assert stats.r2_gap is None


def test_numeric_stats_heteroscedasticity_p_value_defaults_to_none():
    stats = NumericStats()
    assert stats.heteroscedasticity_p_value is None


# ---------------------------------------------------------------------------
# NumericStats — new fields accept values
# ---------------------------------------------------------------------------


def test_numeric_stats_accepts_nonlinearity_tag():
    stats = NumericStats(nonlinearity_tag=NonlinearityTag.ComplexNonlinear)
    assert stats.nonlinearity_tag == NonlinearityTag.ComplexNonlinear


def test_numeric_stats_accepts_signal_floats():
    stats = NumericStats(
        spearman_pearson_discrepancy=0.35,
        mean_mutual_information=0.12,
        r2_gap=0.18,
        heteroscedasticity_p_value=0.03,
    )
    assert stats.spearman_pearson_discrepancy == 0.35
    assert stats.mean_mutual_information == 0.12
    assert stats.r2_gap == 0.18
    assert stats.heteroscedasticity_p_value == 0.03


# ---------------------------------------------------------------------------
# NumericStats.to_dict — all five new fields are present
# ---------------------------------------------------------------------------


def test_to_dict_includes_nonlinearity_tag_key():
    d = NumericStats().to_dict()
    assert "nonlinearity_tag" in d


def test_to_dict_includes_spearman_pearson_discrepancy_key():
    d = NumericStats().to_dict()
    assert "spearman_pearson_discrepancy" in d


def test_to_dict_includes_mean_mutual_information_key():
    d = NumericStats().to_dict()
    assert "mean_mutual_information" in d


def test_to_dict_includes_r2_gap_key():
    d = NumericStats().to_dict()
    assert "r2_gap" in d


def test_to_dict_includes_heteroscedasticity_p_value_key():
    d = NumericStats().to_dict()
    assert "heteroscedasticity_p_value" in d


def test_to_dict_new_fields_are_none_when_unset():
    d = NumericStats().to_dict()
    assert d["nonlinearity_tag"] is None
    assert d["spearman_pearson_discrepancy"] is None
    assert d["mean_mutual_information"] is None
    assert d["r2_gap"] is None
    assert d["heteroscedasticity_p_value"] is None


def test_to_dict_serialises_nonlinearity_tag_as_string():
    stats = NumericStats(nonlinearity_tag=NonlinearityTag.MonotonicNonlinear)
    d = stats.to_dict()
    assert d["nonlinearity_tag"] == "monotonic_nonlinear"


def test_to_dict_serialises_signal_float_values():
    stats = NumericStats(
        spearman_pearson_discrepancy=0.4,
        mean_mutual_information=0.2,
        r2_gap=0.1,
        heteroscedasticity_p_value=0.05,
    )
    d = stats.to_dict()
    assert d["spearman_pearson_discrepancy"] == 0.4
    assert d["mean_mutual_information"] == 0.2
    assert d["r2_gap"] == 0.1
    assert d["heteroscedasticity_p_value"] == 0.05
