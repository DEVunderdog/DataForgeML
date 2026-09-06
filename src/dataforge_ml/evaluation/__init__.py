"""The evaluation module — scoring the data, never the imputation model.

An independent, opt-in package (ADR-0087) whose metrics answer one question:
does a table produced by a pipeline phase still carry the distribution and
dependence structure of the data behind it? The model that produced the table
is disposable and is never graded.

The package is layered. A **generic** layer holds the metric arithmetic and
knows nothing about any one phase — :func:`~dataforge_ml.evaluation._c2st.c2st`
takes two frames and returns a
:class:`~dataforge_ml.evaluation._c2st.C2STResult`. Consumer sub-packages
adapt a phase's artefacts to that layer; ``evaluation/imputation/`` is the
first of them.

Public exports land with the entry point ``evaluate_imputation``; the generic
seam stays private under ADR-0050.
"""

from __future__ import annotations

from ._c2st import C2STResult, C2STScheme
from ._config import C2STConfig, EvaluationConfig, EvaluationMetric
from .imputation import (
    C2STAnnotation,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    C2STVerdict,
    ColumnC2STResult,
    EvaluationReport,
    evaluate_imputation,
)

__all__ = [
    "C2STAnnotation",
    "C2STConfig",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STResult",
    "C2STScheme",
    "C2STScore",
    "C2STVerdict",
    "ColumnC2STResult",
    "EvaluationConfig",
    "EvaluationMetric",
    "EvaluationReport",
    "evaluate_imputation",
]
