"""The imputation consumer of the evaluation module.

The first adapter written against the generic metric layer: it turns a
pre-imputation frame, its imputed table(s) and their profile into an
:class:`~dataforge_ml.evaluation.EvaluationReport`. Everything
imputation-shaped lives here; everything arithmetic lives one level up.
"""

from __future__ import annotations

from ._adapter import evaluate_imputation
from ._records import (
    C2STAnnotation,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
    EvaluationReport,
)

__all__ = [
    "C2STAnnotation",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STScore",
    "C2STVerdict",
    "ColumnC2STResult",
    "EvaluationReport",
    "evaluate_imputation",
]
