"""
Tests for StructuralProfileResult.to_markdown() — the lossless Profile Report.

to_markdown() is the single renderer bound by the Rendering Contract
(ADR-0086, superseding ADR-0040's two-tier split). It must remain
content-equivalent to to_dict(): every leaf value reachable in to_dict()
must also be discoverable as text in the Markdown output, including
histogram bins, full correlation matrices, memory breakdown, and all
per-column fields. ``str(result)`` returns exactly the same bytes.
"""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig
from dataforge_ml.profiling._config import (
    MemoryBreakdown,
    ProfileConfig,
    StructuralProfileResult,
)
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
# Losslessness — every to_dict() leaf value must appear in to_markdown()
# ---------------------------------------------------------------------------


def test_to_markdown_contains_every_to_dict_leaf_value(rich_profile_result):
    data = rich_profile_result.to_dict()
    markdown = rich_profile_result.to_markdown()

    missing = []
    for leaf in _leaf_values(data):
        if leaf is None:
            continue
        text = str(leaf)
        if text == "" or text in markdown:
            continue
        missing.append(text)

    assert not missing, f"{len(missing)} to_dict() leaf values missing from to_markdown(): {missing[:10]}"


def test_to_markdown_contains_histogram_bins(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    for b in rich_profile_result.columns["score"].stats.histogram:
        assert str(b.lower_bound) in markdown
        assert str(b.upper_bound) in markdown
        assert str(b.count) in markdown


def test_to_markdown_contains_full_correlation_matrices(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    fc = rich_profile_result.dataset.feature_correlation
    for row in fc.pearson_matrix.values():
        for value in row.values():
            assert str(value) in markdown
    for row in fc.spearman_matrix.values():
        for value in row.values():
            assert str(value) in markdown


def test_to_markdown_contains_memory_breakdown(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    for col_name, byte_count in rich_profile_result.dataset.memory_breakdown.column_bytes.items():
        assert col_name in markdown
        assert str(byte_count) in markdown


def test_to_markdown_contains_per_column_fields(rich_profile_result):
    markdown = rich_profile_result.to_markdown()
    for col_name, col_profile in rich_profile_result.columns.items():
        assert f"`{col_name}`" in markdown
        assert str(col_profile.semantic_type) in markdown
        assert col_profile.original_dtype in markdown
        assert col_profile.inferred_dtype in markdown


def test_to_markdown_returns_string(rich_profile_result):
    assert isinstance(rich_profile_result.to_markdown(), str)


def test_to_markdown_on_empty_result_does_not_raise():
    result = StructuralProfileResult()
    markdown = result.to_markdown()
    assert isinstance(markdown, str)
    assert "Structural Profile Report" in markdown


# ---------------------------------------------------------------------------
# Rendering Contract rule 2 — __str__ delegates to to_markdown()
# ---------------------------------------------------------------------------


def test_str_equals_to_markdown(rich_profile_result):
    assert str(rich_profile_result) == rich_profile_result.to_markdown()
