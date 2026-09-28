"""The imputation consumer of the evaluation module.

Turns a pre-imputation frame, its imputed table(s) and their profile into a
:class:`~dataforge_ml.evaluation.C2STReport`. Everything
imputation-shaped lives here; everything arithmetic lives one level up.
"""

from __future__ import annotations

from ._adapter import imputation_score_c2st
from ._records import (
    C2STAnnotation,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
)

__all__ = [
    "C2STAnnotation",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STScore",
    "C2STVerdict",
    "ColumnC2STResult",
    "imputation_score_c2st",
]
