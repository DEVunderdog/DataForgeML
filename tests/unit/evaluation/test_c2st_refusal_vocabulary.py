"""The refusal vocabulary, verdicts and Benjamini-Hochberg (#512 / ADR-0087).

Record-level tests, driven by hand-built records rather than by a classifier:
BH is arithmetic over a set of p-values, and a verdict is a property of that
set, so neither needs a fit to be pinned. What the classifier actually does is
asserted in ``tests/integration/test_evaluate_imputation.py``.

The structural claim under test throughout is that **refusal is structural, not
a sentinel number**: a refused column has no :class:`C2STScore` at all, so a
refusal cannot arithmetic-collapse into a passing figure.
"""

from __future__ import annotations

import pytest

from dataforge_ml.evaluation import (
    C2STAnnotation,
    C2STConfig,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
)
from dataforge_ml.evaluation.imputation._records import _benjamini_hochberg

ALPHA = 0.05


def _tested(column: str, p_value: float) -> ColumnC2STResult:
    """A record that ran, carrying nothing but the p-value BH reads."""
    return ColumnC2STResult(
        column=column,
        outcome=C2STOutcome.Tested,
        n_observed=300,
        n_filled=300,
        score=C2STScore(mean_frame_z=3.0, p_value=p_value),
    )


def _refused(column: str, outcome: C2STOutcome) -> ColumnC2STResult:
    return ColumnC2STResult(
        column=column, outcome=outcome, n_observed=10, n_filled=0
    )


def _report(*records: ColumnC2STResult, alpha: float = ALPHA) -> C2STReport:
    return C2STReport(
        columns={record.column: record for record in records},
        provenance=C2STProvenance(
            classifier_class="x.Y", config=C2STConfig(fdr_alpha=alpha)
        ),
    )


# ---------------------------------------------------------------------------
# The closed vocabulary
# ---------------------------------------------------------------------------


def test_the_outcome_vocabulary_is_closed_at_six():
    # A seventh member is a breaking change, so the list is pinned here.
    assert [m.value for m in C2STOutcome] == [
        "tested",
        "no_filled_cells",
        "no_observed_cells",
        "type_not_testable",
        "below_sample_floor",
        "uninformative",
    ]


def test_the_verdict_is_three_valued_and_never_a_bool():
    assert [m.value for m in C2STVerdict] == [
        "flagged",
        "not_flagged",
        "no_verdict",
    ]
    for member in C2STVerdict:
        assert not isinstance(member, bool)


def test_the_annotation_vocabulary_names_all_four_facts():
    assert {m.value for m in C2STAnnotation} == {
        "low_power",
        "imbalanced_piles",
        "degenerate",
        "unseen_category",
    }


# ---------------------------------------------------------------------------
# Refusal is structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome",
    [
        C2STOutcome.NoFilledCells,
        C2STOutcome.NoObservedCells,
        C2STOutcome.TypeNotTestable,
        C2STOutcome.BelowSampleFloor,
        C2STOutcome.Uninformative,
    ],
)
def test_a_refused_column_has_no_score_object_at_all(outcome):
    record = _refused("c", outcome)

    assert record.score is None
    with pytest.raises(AttributeError):
        _ = record.score.mean_frame_z


def test_a_refused_column_has_no_verdict_even_inside_a_report():
    report = _report(_tested("a", 0.0001), _refused("b", C2STOutcome.Uninformative))

    assert report["b"].verdict is C2STVerdict.NoVerdict
    assert report["a"].verdict is C2STVerdict.Flagged


def test_a_record_outside_a_report_carries_no_verdict():
    # The verdict's meaning depends on the set it came from, so a bare record
    # has not got one.
    assert _tested("a", 0.0).verdict is C2STVerdict.NoVerdict


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_bh_matches_the_step_up_formula_by_hand():
    # k = 4, ascending p: 0.01 0.02 0.03 0.04
    #   raw * k / rank:  0.04 0.04 0.04 0.04, monotone already.
    assert _benjamini_hochberg([0.01, 0.02, 0.03, 0.04]) == pytest.approx(
        [0.04, 0.04, 0.04, 0.04]
    )


def test_bh_enforces_monotonicity_with_a_running_minimum():
    # k = 3, ascending p: 0.001 0.5 0.6 -> 0.003 0.75 0.6, and the last value
    # pulls the middle one down to 0.6 rather than leaving the sequence
    # non-monotone.
    assert _benjamini_hochberg([0.001, 0.5, 0.6]) == pytest.approx(
        [0.003, 0.6, 0.6]
    )


def test_no_adjusted_value_ever_exceeds_one():
    # The running minimum is bounded above by the largest raw p, so the cap is
    # a belt-and-braces guard rather than a live path — assert the property,
    # not a value that would only appear if the arithmetic were wrong.
    assert _benjamini_hochberg([0.9, 0.95, 0.99]) == pytest.approx(
        [0.99, 0.99, 0.99]
    )
    assert all(p <= 1.0 for p in _benjamini_hochberg([0.9, 0.95, 0.99]))


def test_bh_is_the_identity_on_a_single_test():
    assert _benjamini_hochberg([0.017]) == pytest.approx([0.017])


def test_the_report_stamps_the_adjusted_p_value_onto_the_score():
    report = _report(_tested("a", 0.001), _tested("b", 0.5), _tested("c", 0.6))

    assert report["a"].score.p_adjusted == pytest.approx(0.003)
    assert report["b"].score.p_adjusted == pytest.approx(0.6)


def test_bh_runs_across_the_tested_set_only():
    # The two refusals must not inflate k: with k = 1 the single tested
    # column's adjusted p equals its raw p.
    report = _report(
        _tested("a", 0.02),
        _refused("b", C2STOutcome.BelowSampleFloor),
        _refused("c", C2STOutcome.TypeNotTestable),
    )

    assert report["a"].score.p_adjusted == pytest.approx(0.02)
    assert report["a"].verdict is C2STVerdict.Flagged


def test_bh_leaves_the_raw_score_uncorrected():
    # The z *is* the score, so BH moves the shortlist and nothing else.
    report = _report(*[_tested(f"c{i}", 0.4) for i in range(20)])

    for name in report:
        assert report[name].score.mean_frame_z == 3.0
        assert report[name].score.p_value == pytest.approx(0.4)
        assert report[name].score.p_adjusted == pytest.approx(0.4)


def test_the_verdict_reads_the_configured_fdr_alpha():
    lenient = _report(_tested("a", 0.08), alpha=0.10)
    strict = _report(_tested("a", 0.08), alpha=0.05)

    assert lenient["a"].verdict is C2STVerdict.Flagged
    assert strict["a"].verdict is C2STVerdict.NotFlagged


def test_bh_shrinks_the_shortlist_a_raw_threshold_would_have_kept():
    # Twenty columns is twenty tests, so roughly one lands under a raw 5% by
    # chance alone. Here "borderline" would pass an uncorrected threshold and
    # BH keeps it off the shortlist; the genuine signal survives.
    records = (
        [_tested("real", 0.0001), _tested("borderline", 0.04)]
        + [_tested(f"noise{i}", 0.6) for i in range(18)]
    )

    report = _report(*records)

    assert report["borderline"].score.p_value < ALPHA
    flagged = [n for n in report if report[n].verdict is C2STVerdict.Flagged]
    assert flagged == ["real"]


def test_a_report_with_no_tested_column_stamps_nothing():
    report = _report(
        _refused("a", C2STOutcome.NoFilledCells),
        _refused("b", C2STOutcome.TypeNotTestable),
    )

    assert all(report[n].verdict is C2STVerdict.NoVerdict for n in report)


def test_the_census_order_survives_the_stamping():
    report = _report(
        _tested("z", 0.001), _refused("a", C2STOutcome.NoFilledCells),
        _tested("m", 0.4),
    )

    assert list(report) == ["z", "a", "m"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_verdict_and_annotations_reach_the_column_fragment():
    record = ColumnC2STResult(
        column="score",
        outcome=C2STOutcome.Tested,
        n_observed=300,
        n_filled=40,
        score=C2STScore(mean_frame_z=3.0, p_value=0.001),
        annotations=(C2STAnnotation.LowPower, C2STAnnotation.ImbalancedPiles),
    )

    text = _report(record)["score"].to_markdown()

    assert "| verdict | flagged |" in text
    assert "low_power, imbalanced_piles" in text


def test_the_summary_table_carries_the_verdict_and_adjusted_p():
    text = _report(_tested("a", 0.001), _refused("b", C2STOutcome.Uninformative)).to_markdown()

    assert "| Column | Outcome | Verdict |" in text
    assert "no_verdict" in text
