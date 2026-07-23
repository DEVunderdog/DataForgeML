"""Expensive, held-out accuracy path for Phase 2 imputation (#330 / ADR-0058).

Drives the public seam — ``EvaluationOrchestrator.score_accuracy(fitted_imputer,
train_df, profile)`` — and asserts on externally observable behaviour: a
per-column ``AccuracyReport`` holding only the held-out accuracy numbers for
model-based columns, accuracy fields that go ``None`` when complete rows fall
below ``refit_r2_min_complete_rows``, ``substep`` heartbeats emitted for the
fold work, and reuse of the recorded strategies (no re-routing).
"""

from __future__ import annotations

import inspect as _inspect

import numpy as np
import polars as pl
import pytest

from dataforge_ml import (
    AccuracyDiagnostic,
    AccuracyReport,
    EvaluationOrchestrator,
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
# strategies (KNN), with one independent column forced to Regression and a
# scalar (Median) / no-missing (Passthrough) column that must never earn an
# entry in the accuracy report.
# ---------------------------------------------------------------------------


def _build_setup(config: PipelineConfig):
    rng = np.random.default_rng(11)
    n = 600
    base = rng.normal(0.0, 1.0, n)
    cols: dict[str, pl.Series] = {}
    for name in ("a", "b", "c", "d"):
        vals = (base + rng.normal(0.0, 0.3, n)).tolist()
        for i in range(n):
            if rng.random() < 0.10:
                vals[i] = None
        cols[name] = pl.Series(vals, dtype=pl.Float64)

    noise = rng.normal(0.0, 1.0, n).tolist()
    for i in range(n):
        if rng.random() < 0.20:
            noise[i] = None
    cols["noise"] = pl.Series(noise, dtype=pl.Float64)

    med = rng.normal(5.0, 2.0, n).tolist()
    for i in range(n):
        if rng.random() < 0.10:
            med[i] = None
    cols["med"] = pl.Series(med, dtype=pl.Float64)

    cols["full"] = pl.Series(rng.normal(0.0, 1.0, n).tolist(), dtype=pl.Float64)

    df = pl.DataFrame(cols)
    config.imputation.numeric.set_per_column_strategy("noise", "regression")
    config.imputation.numeric.set_per_column_strategy("med", "median")

    profile = StructuralProfiler(config).profile(df)
    fitted = fit_imputer(df, profile, config)
    return config, df, profile, fitted


@pytest.fixture(scope="module")
def accuracy_setup():
    return _build_setup(PipelineConfig(profiling=ProfileConfig()))


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_accuracy_types_are_public_api():
    from dataforge_ml import AccuracyReport as R, AccuracyDiagnostic as D

    assert R is AccuracyReport
    assert D is AccuracyDiagnostic


def test_score_accuracy_signature_takes_profile():
    params = list(_inspect.signature(EvaluationOrchestrator.score_accuracy).parameters)
    assert params == ["self", "fitted_imputer", "train_df", "profile"]


def test_accuracy_diagnostic_holds_only_held_out_fields():
    fields = set(AccuracyDiagnostic.__dataclass_fields__)
    assert fields == {"r2_cv", "rmse", "mae"}


# ---------------------------------------------------------------------------
# The report — entries only for model-based columns, focused fields
# ---------------------------------------------------------------------------


def test_score_accuracy_returns_report_keyed_by_model_based_columns(accuracy_setup):
    config, df, profile, fitted = accuracy_setup
    report = EvaluationOrchestrator(config).score_accuracy(fitted, df, profile)

    assert isinstance(report, AccuracyReport)
    # Model-based columns present; scalar (med) and Passthrough (full) absent.
    assert set(report.columns) == {"a", "b", "c", "d", "noise"}
    assert "med" not in report
    assert "full" not in report

    for col in report.columns:
        diag = report[col]
        assert isinstance(diag, AccuracyDiagnostic)


def test_score_accuracy_fields_populated_when_complete_rows_sufficient(accuracy_setup):
    config, df, profile, fitted = accuracy_setup
    report = EvaluationOrchestrator(config).score_accuracy(fitted, df, profile)

    for col in ("a", "b", "c", "d", "noise"):
        diag = report[col]
        assert diag.r2_cv is not None
        assert diag.rmse is not None and diag.rmse >= 0.0
        assert diag.mae is not None and diag.mae >= 0.0


# ---------------------------------------------------------------------------
# The guard — accuracy fields go None below refit_r2_min_complete_rows
# ---------------------------------------------------------------------------


def test_score_accuracy_fields_none_when_complete_rows_insufficient():
    config = PipelineConfig(profiling=ProfileConfig())
    # Set the floor above the total row count so no column ever clears it.
    config.imputation.numeric.refit_r2_min_complete_rows = 100_000
    config, df, profile, fitted = _build_setup(config)

    report = EvaluationOrchestrator(config).score_accuracy(fitted, df, profile)

    # Entries still exist for every model-based column, but every metric is None.
    assert set(report.columns) == {"a", "b", "c", "d", "noise"}
    for col in report.columns:
        diag = report[col]
        assert diag.r2_cv is None
        assert diag.rmse is None
        assert diag.mae is None


# ---------------------------------------------------------------------------
# Observability — substep heartbeats fire during the fold work
# ---------------------------------------------------------------------------


def test_score_accuracy_emits_substep_heartbeats_for_fold_work(accuracy_setup):
    config, df, profile, fitted = accuracy_setup
    recorder = _Recorder()
    EvaluationOrchestrator(config, observer=recorder).score_accuracy(fitted, df, profile)

    types = [e.event_type for e in recorder.events]
    assert EventType.substep in types, "score_accuracy must emit substep heartbeats"

    # A fold substep carries its progression on index/total.
    fold_substeps = [
        e
        for e in recorder.events
        if e.event_type == EventType.substep
        and e.index is not None
        and e.total is not None
    ]
    assert fold_substeps, "expected fold substeps with index/total progression"


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_accuracy_report_round_trips(accuracy_setup):
    config, df, profile, fitted = accuracy_setup
    report = EvaluationOrchestrator(config).score_accuracy(fitted, df, profile)
    restored = AccuracyReport.from_dict(report.to_dict())
    assert set(restored.columns) == set(report.columns)
    for col in report.columns:
        assert restored[col].r2_cv == report[col].r2_cv
        assert restored[col].rmse == report[col].rmse
        assert restored[col].mae == report[col].mae
