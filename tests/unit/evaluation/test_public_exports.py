"""What the evaluation module ships from the package root.

The export rule is ADR-0050's: a symbol lands at the root iff the user must
reference it to configure, drive, or handle the pipeline. Two things about the
evaluation module's list need a test rather than a reading.

The three outcome enums are a **stated exception** (ADR-0087): they classify
output and are never user-supplied — the exact test that keeps ``TypeFlag``
internal — but reading the report *is* branching on them, so the handling half
of the rule reaches them. The generic seam ``c2st`` is Public as of ADR-0087's
amendment (#537): the umbrella that justified its privacy is gone, so it lands
at the package root alongside the two errors it raises (spec #563).
"""

import dataforge_ml
from dataforge_ml import evaluation

EXPECTED = [
    "imputation_score_c2st",
    "C2STConfig",
    "C2STScheme",
    "C2STReport",
    "ColumnC2STResult",
    "C2STScore",
    "C2STResult",
    "C2STProvenance",
    "C2STOutcome",
    "C2STAnnotation",
    "C2STVerdict",
    "c2st",
    "C2STDtypeError",
    "C2STSampleFloorError",
]


def test_every_evaluation_symbol_lands_at_the_package_root():
    missing = [name for name in EXPECTED if name not in dataforge_ml.__all__]
    assert not missing, f"not exported from the package root: {missing}"


def test_the_exported_names_are_reachable_and_are_the_module_s_own():
    for name in EXPECTED:
        assert getattr(dataforge_ml, name) is getattr(evaluation, name)


def test_the_three_outcome_enums_are_the_stated_adr_0050_exception():
    # Never user-supplied, exported anyway because reading the report is
    # comparing against them.
    for name in ("C2STOutcome", "C2STAnnotation", "C2STVerdict"):
        assert name in dataforge_ml.__all__


def test_the_generic_seam_and_its_errors_are_public_at_the_root():
    assert callable(dataforge_ml.c2st)
    for err in ("C2STSampleFloorError", "C2STDtypeError"):
        assert issubclass(getattr(dataforge_ml, err), Exception)


def test_the_root_export_list_is_wholly_reachable():
    missing = [n for n in dataforge_ml.__all__ if not hasattr(dataforge_ml, n)]
    assert not missing, f"declared in __all__ but absent: {missing}"
