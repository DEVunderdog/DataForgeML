"""The C2ST report's rendering — the census, the framing, and the fixed caveat.

The Rendering Contract's mechanics (one renderer, ``__str__`` delegation, the
fragment heading rule) are asserted once for every Result Type in
``tests/unit/test_rendering_contract.py``. What is asserted here is the
*content* the report owes on top of them: an outcome census leading the
fragment so 190 refusals are two rows rather than a wall, the baseline framing
that stops a wall of flagged scalar columns reading as a wall of defects, and
the MAR caveat, which prints whenever anything is flagged and which no
configuration suppresses.
"""

from dataclasses import fields, replace

import pytest

from dataforge_ml.evaluation import (
    C2STConfig,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
)
from dataforge_ml.evaluation.imputation._records import (
    _BASELINE_FRAMING,
    _MAR_CAVEAT,
)

CENSUS_HEADING = "**Outcome census**"


def _tested(column: str, p_value: float) -> ColumnC2STResult:
    return ColumnC2STResult(
        column=column,
        outcome=C2STOutcome.Tested,
        n_observed=300,
        n_filled=100,
        score=C2STScore(mean_frame_z=6.0, p_value=p_value),
    )


def _refused(column: str, outcome: C2STOutcome) -> ColumnC2STResult:
    return ColumnC2STResult(
        column=column, outcome=outcome, n_observed=400, n_filled=0
    )


def _report(*records: ColumnC2STResult, config: C2STConfig = None) -> C2STReport:
    return C2STReport(
        columns={record.column: record for record in records},
        provenance=C2STProvenance(
            classifier_class="sklearn.dummy.DummyClassifier",
            config=config or C2STConfig(),
        ),
    )


@pytest.fixture
def flagged_report() -> C2STReport:
    """A report whose single tested column comes out of BH flagged."""
    return _report(
        _tested("c", p_value=1e-9),
        _refused("note", C2STOutcome.TypeNotTestable),
        _refused("empty", C2STOutcome.NoFilledCells),
    )


@pytest.fixture
def clean_report() -> C2STReport:
    """A report where the tested column sits at chance, so nothing is flagged."""
    return _report(
        _tested("c", p_value=0.8),
        _refused("note", C2STOutcome.TypeNotTestable),
    )


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def test_the_census_leads_the_fragment(flagged_report):
    markdown = flagged_report.to_markdown()
    # After the heading, before anything else the report says.
    assert markdown.index(CENSUS_HEADING) < markdown.index("| Column |")


def test_the_census_counts_every_active_column(flagged_report):
    census = flagged_report.to_markdown().split("| Column |")[0]

    assert "3 active columns" in census
    # Every outcome present in the census, with its count.
    assert "| tested | 1 |" in census
    assert "| no_filled_cells | 1 |" in census
    assert "| type_not_testable | 1 |" in census
    # And the verdict half, on the same rows.
    assert "flagged | 1 |" in census
    assert "no_verdict | 2 |" in census


def test_a_wide_refusing_report_does_not_grow_its_census(clean_report):
    wide = _report(
        _tested("c", p_value=0.8),
        *[
            _refused(f"col{i}", C2STOutcome.BelowSampleFloor)
            for i in range(190)
        ],
    )
    census = wide.to_markdown().split("| Column |")[0]

    assert "191 active columns" in census
    assert "| below_sample_floor | 190 |" in census
    # The whole point: 190 refusals are one row, not 190.
    assert census.count("below_sample_floor") == 1


def test_an_empty_report_still_renders_a_census():
    census = C2STReport().to_markdown()
    assert CENSUS_HEADING in census
    assert "no active columns" in census


# ---------------------------------------------------------------------------
# The baseline framing
# ---------------------------------------------------------------------------


def test_the_framing_prints_whenever_anything_is_flagged(flagged_report):
    markdown = flagged_report.to_markdown()
    assert _BASELINE_FRAMING in markdown
    # It reads as the baseline a flag is judged against, not as a defect list.
    assert "baseline" in _BASELINE_FRAMING
    assert "Median" in _BASELINE_FRAMING


def test_the_framing_is_absent_when_nothing_is_flagged(clean_report):
    assert _BASELINE_FRAMING not in clean_report.to_markdown()


def test_the_framing_leads_the_per_column_table(flagged_report):
    markdown = flagged_report.to_markdown()
    assert markdown.index(_BASELINE_FRAMING) < markdown.index("| Column |")


# ---------------------------------------------------------------------------
# The MAR caveat — printed, and suppressible by nothing
# ---------------------------------------------------------------------------


def test_the_mar_caveat_prints_whenever_anything_is_flagged(flagged_report):
    markdown = flagged_report.to_markdown()
    assert _MAR_CAVEAT in markdown
    assert "MAR" in _MAR_CAVEAT
    # A footer: it is the last thing the fragment says.
    assert markdown.rstrip().endswith(_MAR_CAVEAT)


def test_the_mar_caveat_is_absent_when_nothing_is_flagged(clean_report):
    assert _MAR_CAVEAT not in clean_report.to_markdown()


@pytest.mark.parametrize(
    "field_name", [f.name for f in fields(C2STConfig) if f.name != "classifier"]
)
def test_no_dial_suppresses_the_mar_caveat(field_name):
    """Moving any dial off its default leaves the caveat exactly where it was.

    The disclosure of a known limitation of the library's own number is not a
    threshold, so the full-configurability principle does not reach it
    (ADR-0087). The parametrisation is over the config's own fields so a dial
    added later is covered without anyone remembering to add a case.
    """
    default = C2STConfig()
    current = getattr(default, field_name)
    moved = {
        int: lambda v: v + 1,
        float: lambda v: min(0.99, v + 0.1),
        type(None): lambda v: 0,
    }.get(type(current), lambda v: v)(current)
    config = replace(default, **{field_name: moved})

    report = _report(_tested("c", p_value=1e-9), config=config)

    assert report.columns["c"].verdict is C2STVerdict.Flagged
    assert _MAR_CAVEAT in report.to_markdown()


def test_print_and_to_markdown_agree_on_the_new_content(flagged_report):
    assert str(flagged_report) == flagged_report.to_markdown()
