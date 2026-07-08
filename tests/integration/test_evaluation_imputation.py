"""Purity seam for the Evaluation teardown (#331 / ADR-0058).

After the split into :meth:`EvaluationOrchestrator.inspect` and
:meth:`EvaluationOrchestrator.score_accuracy`, ``fit()`` returns to doing exactly
one thing — route + learn models — and pays no evaluation tax.  This module
asserts that externally observable purity: ``ImputationOrchestrator.fit()``
carries no diagnostics and runs no cross-validated fold work, and
``NumericImputer.fit`` no longer accepts a ``collect_diagnostics`` argument.

The two-method Evaluation surface itself is exercised in
``test_inspection_imputation.py`` and ``test_accuracy_imputation.py``.
"""

from __future__ import annotations

import inspect as _inspect

import polars as pl
import pytest

from dataforge_ml import (
    EventType,
    ImputationOrchestrator,
    PipelineConfig,
    PipelineEvent,
    StructuralProfiler,
)
from dataforge_ml.imputation._numeric_imputer import NumericImputer
from dataforge_ml.profiling._config import ProfileConfig


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# A wide, correlated numeric frame so several columns route to model-based
# strategies (MICE / Regression) whose cross-validated folds would emit
# substeps — if fit() ran them (it must not).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def eval_df(rng):
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
    fitted = ImputationOrchestrator().fit(eval_df, eval_profile)
    for rec in fitted.records.values():
        assert not hasattr(rec, "diagnostic")
    # And serialization round-trips cleanly without any diagnostic key.
    payload = fitted.to_dict()
    for rec_data in payload["records"].values():
        assert "diagnostic" not in rec_data


def test_fit_event_stream_has_no_diagnostics_fold_substeps(eval_df, eval_profile):
    """fit() only learns models — no cross-validated diagnostics fold work happens."""
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit(eval_df, eval_profile)
    fold_msgs = [
        e.message
        for e in recorder.events
        if e.event_type == EventType.substep and "diagnostics fold" in (e.message or "")
    ]
    assert not fold_msgs, "fit() must not run diagnostics folds"


def test_numeric_fit_no_longer_accepts_collect_diagnostics(eval_df, eval_profile):
    """External evidence the diagnostics leak is gone from the fit path (ADR-0058)."""
    assert "collect_diagnostics" not in _inspect.signature(NumericImputer.fit).parameters

    with pytest.raises(TypeError):
        NumericImputer().fit(
            train_df=eval_df,
            columns=["a", "b", "c", "d", "e"],
            profile=eval_profile,
            config=PipelineConfig().imputation.numeric,
            mnar_columns=set(),
            collect_diagnostics=True,
        )


def test_numeric_fit_bundle_has_no_diagnostics_field(eval_df, eval_profile):
    from dataforge_ml.utils._null_normalization import _resolve_effective_nulls

    df = _resolve_effective_nulls(
        eval_df,
        numeric_sentinels=eval_profile.numeric_sentinels,
        string_sentinels=eval_profile.string_sentinels,
    )
    bundle = NumericImputer().fit(
        train_df=df,
        columns=["a", "b", "c", "d", "e"],
        profile=eval_profile,
        config=PipelineConfig().imputation.numeric,
        mnar_columns=set(),
    )
    assert not hasattr(bundle, "diagnostics")
