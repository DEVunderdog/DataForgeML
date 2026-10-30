"""What the evaluation module ships, and what the package root does not.

The package root exports the Operations only (ADR-0099). The evaluation
module's records and the generic C2ST seam stay importable from
``dataforge_ml.evaluation``; the root ``c2st`` is the Operation that runs the
seam over two frame Handles (ADR-0110).
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
]


def test_every_evaluation_symbol_ships_from_the_evaluation_module():
    missing = [name for name in EXPECTED if name not in evaluation.__all__]
    assert not missing, f"not exported from dataforge_ml.evaluation: {missing}"


def test_no_evaluation_symbol_reaches_the_package_root():
    leaked = [name for name in EXPECTED if hasattr(dataforge_ml, name)]
    assert not leaked, f"reachable from the package root: {leaked}"


def test_the_root_c2st_is_the_operation_and_its_errors_are_codes():
    # The generic seam stays in ``dataforge_ml.evaluation``; the root name is
    # the ``c2st`` Operation, which runs it over two frame Handles (ADR-0110).
    from dataforge_ml.operations.c2st import c2st as operation

    assert dataforge_ml.c2st is operation
    assert callable(evaluation.c2st) and evaluation.c2st is not operation
    # The two named refusals became Error Codes (ADR-0100); no class is exported.
    for err in ("C2STSampleFloorError", "C2STDtypeError"):
        assert not hasattr(dataforge_ml, err)


def test_the_root_export_list_is_wholly_reachable():
    missing = [n for n in dataforge_ml.__all__ if not hasattr(dataforge_ml, n)]
    assert not missing, f"declared in __all__ but absent: {missing}"
