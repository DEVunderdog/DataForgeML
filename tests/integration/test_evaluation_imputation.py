"""Purity seam for the Evaluation teardown (#331 / ADR-0058).

After the split into :meth:`EvaluationOrchestrator.inspect` and
:meth:`EvaluationOrchestrator.score_accuracy`, fitting returns to doing exactly
one thing — route + learn models — and pays no evaluation tax.  This module
asserts that externally observable purity: the imputation fit path carries no
diagnostics and runs no cross-validated fold work, and ``NumericImputer.fit`` no
longer accepts a ``collect_diagnostics`` argument.

The two-method Evaluation surface itself is exercised in
``test_inspection_imputation.py`` and ``test_accuracy_imputation.py``.
"""

from __future__ import annotations

import inspect as _inspect

import polars as pl
import pytest

from dataforge_ml import (
    EventType,
    PipelineConfig,
    PipelineEvent,
    StructuralProfiler,
)
from dataforge_ml.profiling._config import ProfileConfig

from tests.conftest import fit_imputer


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# A wide, correlated numeric frame so several columns route to model-based
# strategies (MICE) whose cross-validated folds would emit
# substeps — if fit() ran them (it must not).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_df(rng):
    rng = rng(seed=44)
    n = 300
    base = rng.normal(0.0, 1.0, n)
    cols = {}
    for name in ("a", "b", "c", "d", "e"):
        vals = (base + rng.normal(0.0, 0.5, n)).tolist()
        for i in range(n):
            if rng.random() < 0.10:
                vals[i] = None
        cols[name] = pl.Series(vals, dtype=pl.Float64)
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def eval_profile(eval_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(eval_df)


# ---------------------------------------------------------------------------
# Purity: fit() carries no diagnostics and runs no diagnostics fold work
# ---------------------------------------------------------------------------


def test_fit_records_carry_no_diagnostics(eval_df, eval_profile):
    fitted = fit_imputer(eval_df, eval_profile)
    for rec in fitted.records.values():
        assert not hasattr(rec, "diagnostic")
    # And the serialisable record form carries no diagnostic key.
    for rec in fitted.records.values():
        assert "diagnostic" not in rec.to_dict()


def test_fit_event_stream_has_no_diagnostics_fold_substeps(eval_df, eval_profile):
    """Fitting only learns models — no cross-validated diagnostics fold work happens."""
    recorder = _Recorder()
    fit_imputer(eval_df, eval_profile, observer=recorder)
    fold_msgs = [
        e.message
        for e in recorder.events
        if e.event_type == EventType.substep and "diagnostics fold" in (e.message or "")
    ]
    assert not fold_msgs, "the fit path must not run diagnostics folds"


