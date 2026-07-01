"""
Tests for StructuralProfileResult.to_full_markdown() and to_markdown().

to_full_markdown() is the renamed lossless Markdown serialization (formerly
to_markdown()) — see ADR-0040. It must remain content-equivalent to
to_dict(): every leaf value reachable in to_dict() must also be discoverable
as text in the Markdown output, including histogram bins, full correlation
matrices, memory breakdown, and all per-column fields.

to_markdown() is the compact, human-oriented view (ADR-0040). It covers the
Dataset Overview (scalar dataset-level fields, no memory_breakdown /
missingness_matrix), the Column Summary table (one row per column), the
Flagged Columns section (full detail subsections for columns exceeding the
clean threshold, severity-first then alphabetical ordering, with top-5
Pearson/Spearman correlations in place of the full matrices), the Target
Analysis section (top-5 Pearson + top-5 Spearman per feature per target),
and the Sentinels section (numeric_sentinels / string_sentinels unchanged).
"""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, SemanticType
from dataforge_ml.profiling._config import (
    ColumnProfile,
    DatasetStats,
    MemoryBreakdown,
    ProfileConfig,
    StructuralProfileResult,
)
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import (
    BimodalStats,
    HistogramBin,
    NonlinearityTag,
    NumericFlag,
    NumericStats,
    NumericTopValueEntry,
    PercentileSnapshot,
)
from dataforge_ml.profiling._categorical_config import CategoricalStats, TopValueEntry
from dataforge_ml.profiling.orchestrator import StructuralProfiler


def _leaf_values(obj):
    """Yield every non-container leaf value found anywhere within obj."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaf_values(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _leaf_values(item)
    else:
        yield obj


@pytest.fixture(scope="module")
def rich_profile_result() -> StructuralProfileResult:
    """
    A StructuralProfileResult with every field family populated:
    a continuous numeric column (histogram), a categorical column with
    missing values (alongside score, giving two columns with missingness
    so the missingness correlation matrix is computed), a boolean column,
    a datetime column, a correlated numeric target (feature + target
    correlation matrices), and an injected memory breakdown (the real
    threshold for this is 500MB of data, impractical to construct here).
    """
    n = 100
    base_score = [math.sin(i / 3.0) * 50.0 + i * 0.7 for i in range(n)]
    score = [None if i % 9 == 0 else base_score[i] for i in range(n)]
    target = [base_score[i] * 2.5 + (i % 5) * 1.3 for i in range(n)]
    visits = [i % 20 for i in range(n)]
    categories = ["alpha", "beta", "gamma", "delta"]
    category = [None if i % 11 == 0 else categories[i % len(categories)] for i in range(n)]
    active = [i % 2 == 0 for i in range(n)]
    base_date = date(2022, 1, 1)
    event_date = [base_date + timedelta(days=i) for i in range(n)]

    df = pl.DataFrame(
        {
            "score": pl.Series(score, dtype=pl.Float64),
            "visits": pl.Series(visits, dtype=pl.Int64),
            "category": pl.Series(category, dtype=pl.Utf8),
            "active": pl.Series(active, dtype=pl.Boolean),
            "event_date": pl.Series(event_date, dtype=pl.Date),
            "target": pl.Series(target, dtype=pl.Float64),
        }
    )

    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_correlation=True,
            compute_nonlinearity=True,
            target_columns=["target"],
        )
    )
    result = StructuralProfiler(config).profile(df)
    result.dataset.memory_breakdown = MemoryBreakdown(
        column_bytes={"score": 800, "visits": 800, "category": 1200}
    )
    return result


# ---------------------------------------------------------------------------
# Fixture sanity — confirms the scenario actually exercises every field family
# ---------------------------------------------------------------------------


def test_fixture_exercises_all_target_field_families(rich_profile_result):
    result = rich_profile_result
    assert result.columns["score"].stats.histogram, "expected histogram bins on 'score'"
    assert result.dataset.feature_correlation is not None
    assert result.dataset.feature_correlation.pearson_matrix
    assert result.dataset.feature_correlation.spearman_matrix
    assert result.dataset.target_correlations
    assert result.dataset.memory_breakdown is not None
    assert result.dataset.memory_breakdown.column_bytes
    assert result.dataset.missingness_matrix


# ---------------------------------------------------------------------------
# Losslessness — every to_dict() leaf value must appear in to_full_markdown()
# ---------------------------------------------------------------------------


def test_to_full_markdown_contains_every_to_dict_leaf_value(rich_profile_result):
    data = rich_profile_result.to_dict()
    markdown = rich_profile_result.to_full_markdown()

    missing = []
    for leaf in _leaf_values(data):
        if leaf is None:
            continue
        text = str(leaf)
        if text == "" or text in markdown:
            continue
        missing.append(text)

    assert not missing, f"{len(missing)} to_dict() leaf values missing from to_full_markdown(): {missing[:10]}"


def test_to_full_markdown_contains_histogram_bins(rich_profile_result):
    markdown = rich_profile_result.to_full_markdown()
    for b in rich_profile_result.columns["score"].stats.histogram:
        assert str(b.lower_bound) in markdown
        assert str(b.upper_bound) in markdown
        assert str(b.count) in markdown


def test_to_full_markdown_contains_full_correlation_matrices(rich_profile_result):
    markdown = rich_profile_result.to_full_markdown()
    fc = rich_profile_result.dataset.feature_correlation
    for row in fc.pearson_matrix.values():
        for value in row.values():
            assert str(value) in markdown
    for row in fc.spearman_matrix.values():
        for value in row.values():
            assert str(value) in markdown


def test_to_full_markdown_contains_memory_breakdown(rich_profile_result):
    markdown = rich_profile_result.to_full_markdown()
    for col_name, byte_count in rich_profile_result.dataset.memory_breakdown.column_bytes.items():
        assert col_name in markdown
        assert str(byte_count) in markdown


def test_to_full_markdown_contains_per_column_fields(rich_profile_result):
    markdown = rich_profile_result.to_full_markdown()
    for col_name, col_profile in rich_profile_result.columns.items():
        assert f"`{col_name}`" in markdown
        assert str(col_profile.semantic_type) in markdown
        assert col_profile.original_dtype in markdown
        assert col_profile.inferred_dtype in markdown


def test_to_full_markdown_returns_string(rich_profile_result):
    assert isinstance(rich_profile_result.to_full_markdown(), str)


def test_to_full_markdown_on_empty_result_does_not_raise():
    result = StructuralProfileResult()
    markdown = result.to_full_markdown()
    assert isinstance(markdown, str)
    assert "Structural Profile Report" in markdown


# ---------------------------------------------------------------------------
# to_markdown() compact view — Dataset Overview + Column Summary (ADR-0040)
# ---------------------------------------------------------------------------


def test_to_markdown_returns_string(rich_profile_result):
    assert isinstance(rich_profile_result.to_markdown(), str)


def test_to_markdown_on_empty_result_does_not_raise():
    result = StructuralProfileResult()
    markdown = result.to_markdown()
    assert isinstance(markdown, str)
    assert "## Dataset Overview" in markdown
    assert "## Column Summary" in markdown


def test_to_markdown_contains_dataset_overview_scalars(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    assert "## Dataset Overview" in markdown
    ds = rich_profile_result.dataset
    assert str(ds.row_count) in markdown
    assert str(ds.memory_bytes) in markdown
    assert str(ds.duplicate_ratio) in markdown
    assert str(ds.overall_sparsity) in markdown


def test_to_markdown_dataset_overview_excludes_memory_breakdown_and_missingness_matrix(
    rich_profile_result,
):
    markdown = rich_profile_result.to_markdown()
    overview = markdown.split("## Dataset Overview", 1)[1].split("## Column Summary", 1)[0]

    assert "memory_breakdown" not in overview
    assert "missingness_matrix" not in overview
    assert "column_bytes" not in overview
    for col_name, byte_count in rich_profile_result.dataset.memory_breakdown.column_bytes.items():
        assert f"{col_name}: {byte_count}" not in overview


def test_to_markdown_column_summary_has_one_row_per_column(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    summary_section = markdown.split("## Column Summary", 1)[1]
    table_rows = [
        line for line in summary_section.splitlines() if line.startswith("| `")
    ]
    assert len(table_rows) == len(rich_profile_result.columns)
    for col_name in rich_profile_result.columns:
        assert f"`{col_name}`" in summary_section


def test_to_markdown_column_summary_row_shows_key_fields(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    summary_section = markdown.split("## Column Summary", 1)[1]
    col = rich_profile_result.columns["score"]
    assert str(col.semantic_type) in summary_section
    assert f"{col.missingness.effective_null_ratio * 100:.2f}%" in summary_section


# ---------------------------------------------------------------------------
# to_markdown() Flagged Columns — two-tier rendering, severity ordering
# ---------------------------------------------------------------------------


def _missingness(
    name: str,
    *,
    ratio: float = 0.0,
    severity=None,
    flags=None,
    correlated_with=None,
) -> ColumnMissingnessProfile:
    return ColumnMissingnessProfile(
        column=name,
        total_rows=100,
        effective_null_count=int(ratio * 100),
        effective_null_ratio=ratio,
        severity=severity,
        flags=flags or [],
        correlated_with=correlated_with or [],
    )


def _col(name: str, *, missingness=None, stats=None) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        semantic_type=SemanticType.Numeric,
        missingness=missingness,
        stats=stats,
    )


def _flagged_section(markdown: str) -> str:
    return markdown.split("## Flagged Columns", 1)[1]


def test_to_markdown_all_clean_result_has_no_flagged_column_sections():
    columns = {
        "alpha": _col("alpha", stats=NumericStats()),
        "beta": _col(
            "beta",
            missingness=_missingness("beta", ratio=0.005, severity=MissingSeverity.Minor),
        ),
    }
    result = StructuralProfileResult(columns=columns)
    markdown = result.to_markdown()

    assert "## Flagged Columns" in markdown
    section = _flagged_section(markdown)
    assert "### `" not in section


def test_to_markdown_mixed_result_only_flagged_columns_get_sections():
    clean = _col("clean_col", stats=NumericStats())
    flagged = _col(
        "flagged_col",
        missingness=_missingness("flagged_col", ratio=0.3, severity=MissingSeverity.Severe),
    )
    result = StructuralProfileResult(columns={"clean_col": clean, "flagged_col": flagged})
    markdown = result.to_markdown()

    section = _flagged_section(markdown)
    assert "### `flagged_col`" in section
    assert "### `clean_col`" not in section


def test_to_markdown_flagged_columns_severity_ordering():
    columns = {
        "zz_drop": _col(
            "zz_drop",
            missingness=_missingness(
                "zz_drop",
                ratio=0.6,
                severity=MissingSeverity.Severe,
                flags=[MissingnessFlag.DropCandidate],
            ),
        ),
        "yy_fullynull": _col(
            "yy_fullynull",
            missingness=_missingness(
                "yy_fullynull",
                ratio=1.0,
                severity=MissingSeverity.Severe,
                flags=[MissingnessFlag.FullyNull],
            ),
        ),
        "ww_severe": _col(
            "ww_severe",
            missingness=_missingness("ww_severe", ratio=0.25, severity=MissingSeverity.Severe),
        ),
        "vv_high": _col(
            "vv_high",
            missingness=_missingness("vv_high", ratio=0.1, severity=MissingSeverity.High),
        ),
        "uu_moderate": _col(
            "uu_moderate",
            missingness=_missingness("uu_moderate", ratio=0.03, severity=MissingSeverity.Moderate),
        ),
        "tt_minor_flagged": _col(
            "tt_minor_flagged",
            missingness=_missingness(
                "tt_minor_flagged",
                ratio=0.005,
                severity=MissingSeverity.Minor,
                flags=[MissingnessFlag.MARSuspect],
                correlated_with=["other"],
            ),
        ),
        "ss_numeric_only": _col(
            "ss_numeric_only",
            stats=NumericStats(flags=[NumericFlag.NearConstant]),
        ),
        "rr_nonlinearity_only": _col(
            "rr_nonlinearity_only",
            stats=NumericStats(nonlinearity_tag=NonlinearityTag.MonotonicNonlinear),
        ),
        "aa_clean": _col("aa_clean", stats=NumericStats()),
    }
    result = StructuralProfileResult(columns=columns)
    section = _flagged_section(result.to_markdown())

    expected_order = [
        "yy_fullynull",  # tier 0 (DropCandidate/FullyNull), alphabetical before zz_drop
        "zz_drop",  # tier 0
        "ww_severe",  # tier 1
        "vv_high",  # tier 2
        "uu_moderate",  # tier 3
        "tt_minor_flagged",  # tier 4
        # tier 5 (numeric/nonlinearity-only), alphabetical
        "rr_nonlinearity_only",
        "ss_numeric_only",
    ]

    assert "### `aa_clean`" not in section
    positions = [section.index(f"### `{name}`") for name in expected_order]
    assert positions == sorted(positions), (
        f"expected severity-first, alphabetical-within-tier ordering, got: {expected_order}"
    )


def test_to_markdown_flagged_columns_alphabetical_within_same_tier():
    columns = {
        "zebra": _col(
            "zebra",
            missingness=_missingness("zebra", ratio=0.25, severity=MissingSeverity.Severe),
        ),
        "apple": _col(
            "apple",
            missingness=_missingness("apple", ratio=0.3, severity=MissingSeverity.Severe),
        ),
    }
    result = StructuralProfileResult(columns=columns)
    section = _flagged_section(result.to_markdown())

    assert section.index("### `apple`") < section.index("### `zebra`")


# ---------------------------------------------------------------------------
# to_markdown() Flagged Columns — per-column field rules (caps, drops, scalar
# preservation; ADR-0040 field inclusion rules)
# ---------------------------------------------------------------------------


def _numeric_stats_rich(**overrides) -> NumericStats:
    defaults = dict(
        mean=10.5,
        median=10.0,
        mean_median_ratio=1.05,
        mode=9.0,
        mode_frequency=0.12,
        top_values=[
            NumericTopValueEntry(value=float(i), count=100 - i, percentage=(100 - i) / 1000)
            for i in range(5)
        ],
        histogram=[
            HistogramBin(lower_bound=float(i), upper_bound=float(i + 1), count=10, percentage=0.1)
            for i in range(5)
        ],
        std=3.3,
        variance=10.89,
        min=0.0,
        max=99.0,
        percentiles=PercentileSnapshot(p1=1.0, p5=5.0, p25=25.0, p50=50.0, p75=75.0, p95=95.0, p99=99.0),
        skewness=0.42,
        kurtosis=2.1,
        flags=[NumericFlag.Bimodal],
        spearman_pearson_discrepancy=0.07,
        mean_mutual_information=0.33,
        r2_gap=0.15,
        heteroscedasticity_p_value=0.02,
        bimodal_stats=BimodalStats(dip_statistic=0.05, dip_p_value=0.01, center1=2.0, center2=8.0, cluster_separation=3.0, minority_weight=0.4),
        tail_asymmetry_share=1.8,
        outlier_density=0.04,
    )
    defaults.update(overrides)
    return NumericStats(**defaults)


def _categorical_stats_rich(**overrides) -> CategoricalStats:
    defaults = dict(
        cardinality=4,
        unique_ratio=0.04,
        mode_frequency=0.4,
        top_values=[
            TopValueEntry(value=f"cat_{i}", count=50 - i, percentage=(50 - i) / 100)
            for i in range(5)
        ],
    )
    defaults.update(overrides)
    return CategoricalStats(**defaults)


def test_to_markdown_flagged_column_drops_histogram_bins():
    col = _col("rich_numeric", stats=_numeric_stats_rich())
    result = StructuralProfileResult(columns={"rich_numeric": col})
    section = _flagged_section(result.to_markdown())

    assert "### `rich_numeric`" in section
    assert "histogram" not in section
    assert "lower_bound" not in section
    assert "upper_bound" not in section


def test_to_markdown_flagged_column_caps_numeric_top_values():
    col = _col("rich_numeric", stats=_numeric_stats_rich())
    result = StructuralProfileResult(columns={"rich_numeric": col})
    section = _flagged_section(result.to_markdown())

    detail = section.split("### `rich_numeric`", 1)[1]
    kept = col.stats.top_values[:3]
    dropped = col.stats.top_values[3:]
    assert dropped, "fixture must declare more than 3 top_values entries"
    for entry in kept:
        assert str(entry.value) in detail
    for entry in dropped:
        assert f"'value': {entry.value}" not in detail


def test_to_markdown_flagged_column_caps_categorical_top_values():
    col = _col(
        "rich_categorical",
        missingness=_missingness(
            "rich_categorical", ratio=0.02, severity=MissingSeverity.Minor,
            flags=[MissingnessFlag.MARSuspect], correlated_with=["other_col"],
        ),
        stats=_categorical_stats_rich(),
    )
    result = StructuralProfileResult(columns={"rich_categorical": col})
    section = _flagged_section(result.to_markdown())

    assert "### `rich_categorical`" in section
    detail = section.split("### `rich_categorical`", 1)[1]
    kept = col.stats.top_values[:3]
    dropped = col.stats.top_values[3:]
    assert dropped, "fixture must declare more than 3 top_values entries"
    for entry in kept:
        assert str(entry.value) in detail
    for entry in dropped:
        assert str(entry.value) not in detail


def test_to_markdown_flagged_column_missingness_drops_total_rows():
    col = _col(
        "flagged_missing",
        missingness=_missingness(
            "flagged_missing", ratio=0.3, severity=MissingSeverity.Severe,
        ),
    )
    result = StructuralProfileResult(columns={"flagged_missing": col})
    section = _flagged_section(result.to_markdown())

    detail = section.split("### `flagged_missing`", 1)[1]
    assert "total_rows" not in detail


def test_to_markdown_flagged_column_keeps_all_numeric_scalars():
    stats = _numeric_stats_rich()
    col = _col("rich_numeric", stats=stats)
    result = StructuralProfileResult(columns={"rich_numeric": col})
    section = _flagged_section(result.to_markdown())
    detail = section.split("### `rich_numeric`", 1)[1]

    for field_name in (
        "mean", "median", "std", "variance", "skewness", "kurtosis",
        "min", "max", "mode", "mode_frequency", "mean_median_ratio",
        "outlier_density", "tail_asymmetry_share",
    ):
        assert f"**{field_name}**" in detail, f"missing scalar field: {field_name}"

    for value in (
        stats.mean, stats.median, stats.std, stats.variance, stats.skewness,
        stats.kurtosis, stats.min, stats.max, stats.mode, stats.mode_frequency,
        stats.mean_median_ratio, stats.outlier_density, stats.tail_asymmetry_share,
    ):
        assert str(value) in detail

    for pct_value in (
        stats.percentiles.p1, stats.percentiles.p5, stats.percentiles.p25,
        stats.percentiles.p50, stats.percentiles.p75, stats.percentiles.p95,
        stats.percentiles.p99,
    ):
        assert str(pct_value) in detail


def test_to_markdown_flagged_column_keeps_missingness_null_ratio_pair():
    col = _col(
        "flagged_missing",
        missingness=_missingness(
            "flagged_missing", ratio=0.3, severity=MissingSeverity.Severe,
        ),
    )
    col.missingness.standard_null_ratio = 0.28
    result = StructuralProfileResult(columns={"flagged_missing": col})
    section = _flagged_section(result.to_markdown())
    detail = section.split("### `flagged_missing`", 1)[1]

    assert "**standard_null_ratio**: 0.28" in detail
    assert "**effective_null_ratio**: 0.3" in detail
    assert str(col.missingness.effective_null_count) in detail
    assert str(col.missingness.standard_null_count) in detail


def test_to_markdown_flagged_column_keeps_correlated_with_in_full():
    correlated = [f"col_{i}" for i in range(8)]
    col = _col(
        "flagged_mar",
        missingness=_missingness(
            "flagged_mar", ratio=0.02, severity=MissingSeverity.Minor,
            flags=[MissingnessFlag.MARSuspect], correlated_with=correlated,
        ),
    )
    result = StructuralProfileResult(columns={"flagged_mar": col})
    section = _flagged_section(result.to_markdown())
    detail = section.split("### `flagged_mar`", 1)[1]

    for name in correlated:
        assert name in detail


def test_to_markdown_flagged_column_keeps_bimodal_stats_when_flagged():
    stats = _numeric_stats_rich()
    assert NumericFlag.Bimodal in stats.flags
    col = _col("rich_numeric", stats=stats)
    result = StructuralProfileResult(columns={"rich_numeric": col})
    section = _flagged_section(result.to_markdown())
    detail = section.split("### `rich_numeric`", 1)[1]

    assert "bimodal_stats" in detail
    assert str(stats.bimodal_stats.dip_statistic) in detail
    assert str(stats.bimodal_stats.dip_p_value) in detail
    assert str(stats.bimodal_stats.center1) in detail
    assert str(stats.bimodal_stats.center2) in detail


# ---------------------------------------------------------------------------
# to_markdown() — feature correlations top-5, Target Analysis, Sentinels
# (ADR-0040, final slice)
# ---------------------------------------------------------------------------


def _one_sided_matrix(
    col_name: str, partners: dict[str, float]
) -> dict[str, dict[str, float]]:
    """A matrix dict with a full row for ``col_name`` and degenerate rows
    for every partner (only used as the lookup target for other columns,
    never asserted on)."""
    row = {col_name: 1.0}
    row.update(partners)
    matrix = {col_name: row}
    for other, value in partners.items():
        matrix[other] = {other: 1.0, col_name: value}
    return matrix


def test_to_markdown_column_correlations_capped_at_five_entries():
    partners = {f"col_{i}": (i + 1) / 10.0 for i in range(9)}  # 9 partners, 0.1..0.9
    pearson_matrix = _one_sided_matrix("flagged_col", partners)
    spearman_matrix = _one_sided_matrix(
        "flagged_col", {k: -v for k, v in partners.items()}
    )

    flagged = _col(
        "flagged_col",
        missingness=_missingness("flagged_col", ratio=0.3, severity=MissingSeverity.Severe),
    )
    result = StructuralProfileResult(
        columns={"flagged_col": flagged},
        dataset=DatasetStats(
            feature_correlation=CorrelationProfileResult(
                analysed_numeric_columns=["flagged_col", *partners.keys()],
                pearson_matrix=pearson_matrix,
                spearman_matrix=spearman_matrix,
            )
        ),
    )
    section = _flagged_section(result.to_markdown())
    detail = section.split("### `flagged_col`", 1)[1]

    assert "top_pearson" in detail
    assert "top_spearman" in detail
    pearson_block = detail.split("**top_pearson**", 1)[1].split("**top_spearman**", 1)[0]
    spearman_block = detail.split("**top_spearman**", 1)[1]

    pearson_entries = [line for line in pearson_block.splitlines() if "col_" in line]
    spearman_entries = [line for line in spearman_block.splitlines() if "col_" in line]
    assert len(pearson_entries) == 5
    assert len(spearman_entries) == 5

    expected_top5 = sorted(partners.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    excluded = sorted(partners.items(), key=lambda kv: abs(kv[1]), reverse=True)[5:]
    for name, value in expected_top5:
        assert f"{name}: {value}" in pearson_block
    for name, value in excluded:
        assert f"{name}: {value}" not in pearson_block


def test_to_markdown_full_correlation_matrix_does_not_appear():
    partners = {f"col_{i}": (i + 1) / 100.0 for i in range(9)}  # 0.01..0.09
    pearson_matrix = _one_sided_matrix("flagged_col", partners)
    spearman_matrix = _one_sided_matrix("flagged_col", partners)

    flagged = _col(
        "flagged_col",
        missingness=_missingness("flagged_col", ratio=0.3, severity=MissingSeverity.Severe),
    )
    result = StructuralProfileResult(
        columns={"flagged_col": flagged},
        dataset=DatasetStats(
            feature_correlation=CorrelationProfileResult(
                analysed_numeric_columns=["flagged_col", *partners.keys()],
                pearson_matrix=pearson_matrix,
                spearman_matrix=spearman_matrix,
            )
        ),
    )
    markdown = result.to_markdown()

    # Only the top-5 (col_4..col_8) may appear; the bottom 4 must not.
    for name in ("col_0", "col_1", "col_2", "col_3"):
        assert f"{name}: {partners[name]}" not in markdown


def test_to_markdown_target_analysis_top5_ranking_correct():
    pearson_row = {
        "feat_1": 0.9,
        "feat_2": -0.85,
        "feat_3": 0.7,
        "feat_4": 0.6,
        "feat_5": 0.5,
        "feat_6": 0.1,
        "target": 0.3,
    }
    matrix = {"feat_0": {"feat_0": 1.0, **pearson_row}}
    corr = CorrelationProfileResult(
        analysed_numeric_columns=["feat_0", "feat_1", "feat_2", "feat_3", "feat_4",
                                   "feat_5", "feat_6", "target"],
        pearson_matrix=matrix,
        spearman_matrix=matrix,
        target_column="target",
    )
    result = StructuralProfileResult(
        dataset=DatasetStats(target_correlations={"target": corr})
    )
    markdown = result.to_markdown()

    assert "## Target Analysis" in markdown
    assert "### Target: `target`" in markdown
    assert "#### `feat_0`" in markdown
    # The target itself is not a "feature column" under its own analysis.
    assert "#### `target`" not in markdown

    target_section = markdown.split("## Target Analysis", 1)[1]
    feat0_block = target_section.split("#### `feat_0`", 1)[1].split("#### `feat_1`", 1)[0]

    for name in ("feat_1", "feat_2", "feat_3", "feat_4", "feat_5"):
        assert f"{name}: {pearson_row[name]}" in feat0_block
    for name in ("feat_6", "target"):
        assert f"{name}: {pearson_row[name]}" not in feat0_block


def test_to_markdown_no_target_analysis_section_when_target_correlations_empty():
    result = StructuralProfileResult()
    markdown = result.to_markdown()

    assert "## Target Analysis" not in markdown
    assert "## Sentinels" in markdown


def test_to_markdown_sentinels_section_renders_both_dicts():
    result = StructuralProfileResult(
        numeric_sentinels={"col_a": [-999.0, -1.0]},
        string_sentinels={"col_b": ["N/A", "unknown"]},
    )
    markdown = result.to_markdown()

    assert "## Sentinels" in markdown
    sentinels_section = markdown.split("## Sentinels", 1)[1]
    assert "numeric_sentinels" in sentinels_section
    assert "col_a" in sentinels_section
    assert "-999.0" in sentinels_section
    assert "string_sentinels" in sentinels_section
    assert "col_b" in sentinels_section
    assert "N/A" in sentinels_section
    assert "unknown" in sentinels_section
