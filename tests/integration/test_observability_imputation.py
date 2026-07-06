"""Observability for Phase 2: ImputationOrchestrator observer + fit heartbeat.

Covers Issue #319 over the real profile → fit → transform harness (no stubs).
"""

import logging

import polars as pl
import pytest

from dataforge_ml import (
    EventType,
    ImputationOrchestrator,
    ImputationStrategy,
    PipelineConfig,
    PipelineEvent,
    StructuralProfiler,
)
from dataforge_ml.imputation import ImputationConfig, NumericImputationConfig
from dataforge_ml.profiling._config import ProfileConfig


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Fixtures — three numeric columns get fitted; the categorical passes through.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imp_df(rng):
    n = 300
    a = rng.normal(50.0, 10.0, n).tolist()
    b = rng.normal(200.0, 30.0, n).tolist()
    c = [float(v) for v in rng.integers(1, 6, n).tolist()]
    for col in (a, b, c):
        for i in range(n):
            if rng.random() < 0.10:
                col[i] = None
    return pl.DataFrame(
        {
            "score": pl.Series(a, dtype=pl.Float64),
            "revenue": pl.Series(b, dtype=pl.Float64),
            "rating": pl.Series(c, dtype=pl.Float64),
            "label": pl.Series(["A" if i % 2 else "B" for i in range(n)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def imp_profile(imp_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(imp_df)


@pytest.fixture(scope="module")
def fallback_df(rng):
    # A lone numeric column with ~10% missing.  Forcing Regression on it leaves
    # no feature columns, so the model-based fit is unusable and the router
    # takes a recoverable soft fallback to median (ADR-0054).
    n = 200
    vals = rng.normal(50.0, 10.0, n).tolist()
    for i in range(n):
        if rng.random() < 0.10:
            vals[i] = None
    return pl.DataFrame({"x": pl.Series(vals, dtype=pl.Float64)})


@pytest.fixture(scope="module")
def fallback_profile(fallback_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(
        fallback_df
    )


def _force_regression_config() -> PipelineConfig:
    """A config forcing Regression on column ``x`` with a permissive size guard."""
    cfg = PipelineConfig(
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                regression_min_rows=10,
                _per_column_strategy={"x": ImputationStrategy.Regression},
            )
        )
    )
    return cfg


# ---------------------------------------------------------------------------
# Observer wiring + fit event sequence (#319)
# ---------------------------------------------------------------------------


def test_fit_emits_stage_and_item_events(imp_df, imp_profile):
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit(imp_df, imp_profile)

    stream = [e for e in recorder.events if e.stage == "column_fitting"]
    assert stream[0].event_type == EventType.stage_start
    assert stream[-1].event_type == EventType.stage_end
    assert all(e.phase == "imputation" for e in stream)

    items = [e for e in stream if e.event_type == EventType.item]
    assert items, "no per-column fit item events emitted"

    total = items[0].total
    assert total == len(items)
    # Only the three numeric columns are fitted; the categorical passes through.
    assert total == 3

    fitted_cols = []
    for i, e in enumerate(items, start=1):
        assert e.index == i
        assert e.total == total
        assert e.column in imp_df.columns
        fitted_cols.append(e.column)

    assert set(fitted_cols) == {"score", "revenue", "rating"}
    assert "label" not in fitted_cols  # passthrough column raises no item


def test_fit_transform_emits_fit_events(imp_df, imp_profile):
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit_transform(imp_df, imp_profile)

    items = [
        e for e in recorder.events
        if e.event_type == EventType.item and e.stage == "column_fitting"
    ]
    assert {e.column for e in items} == {"score", "revenue", "rating"}


def test_no_observer_is_harmless(imp_df, imp_profile):
    fitted = ImputationOrchestrator().fit(imp_df, imp_profile)
    result = fitted.transform(imp_df)
    for col in ("score", "revenue", "rating"):
        assert result.dataframe[col].null_count() == 0


# ---------------------------------------------------------------------------
# Two-Sink Rule: DEBUG log alongside the observer (#319)
# ---------------------------------------------------------------------------


def test_fit_events_log_at_debug(imp_df, imp_profile, caplog):
    with caplog.at_level(logging.DEBUG, logger="dataforge_ml"):
        ImputationOrchestrator().fit(imp_df, imp_profile)

    fit_records = [
        r for r in caplog.records
        if r.name == "dataforge_ml" and "[imputation] column_fitting" in r.message
    ]
    assert fit_records, "no imputation fit records captured"
    assert all(r.levelno == logging.DEBUG for r in fit_records)


# ---------------------------------------------------------------------------
# decision events: routing rationale (#320)
# ---------------------------------------------------------------------------


def test_decision_event_per_routed_column(imp_df, imp_profile):
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit(imp_df, imp_profile)

    decisions = [e for e in recorder.events if e.event_type == EventType.decision]
    # One decision per routed (fitted) column — the three numerics.
    assert {e.column for e in decisions} == {"score", "revenue", "rating"}
    for e in decisions:
        assert e.phase == "imputation"
        # message carries both the chosen strategy and a "because" rationale.
        assert f"{e.column} ->" in e.message
        assert "because" in e.message


def test_decision_events_log_at_info(imp_df, imp_profile, caplog):
    with caplog.at_level(logging.INFO, logger="dataforge_ml"):
        ImputationOrchestrator().fit(imp_df, imp_profile)

    decision_records = [
        r for r in caplog.records
        if r.name == "dataforge_ml" and "because" in r.message
    ]
    assert decision_records, "no decision records captured at INFO"
    assert all(r.levelno == logging.INFO for r in decision_records)


def test_decisions_absent_at_warning_but_visible_at_info(imp_df, imp_profile, caplog):
    # decision events sit at INFO: prominent above the DEBUG heartbeats, yet
    # still below WARNING so they stay quiet unless the consumer opts in.
    with caplog.at_level(logging.WARNING, logger="dataforge_ml"):
        ImputationOrchestrator().fit(imp_df, imp_profile)
    assert [r for r in caplog.records if r.name == "dataforge_ml"] == []


def test_observer_can_filter_by_event_type(imp_df, imp_profile):
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit(imp_df, imp_profile)

    # A consumer interested only in decisions filters the one stream by type.
    decisions = [e for e in recorder.events if e.event_type == EventType.decision]
    heartbeats = [e for e in recorder.events if e.event_type == EventType.item]
    assert decisions and heartbeats
    assert all(e.event_type == EventType.decision for e in decisions)
    # Filtering out decisions leaves the heartbeat stream untouched.
    non_decisions = [e for e in recorder.events if e.event_type != EventType.decision]
    assert heartbeats == [e for e in non_decisions if e.event_type == EventType.item]


# ---------------------------------------------------------------------------
# warning events: recoverable soft fallbacks (#321)
# ---------------------------------------------------------------------------


def test_soft_fallback_emits_warning_event(fallback_df, fallback_profile):
    recorder = _Recorder()
    fitted = ImputationOrchestrator(
        _force_regression_config(), observer=recorder
    ).fit(fallback_df, fallback_profile)

    # The routed Regression became unusable and recovered to median.
    assert fitted.records["x"].strategy == ImputationStrategy.Median

    warnings = [e for e in recorder.events if e.event_type == EventType.warning]
    assert warnings, "recoverable fallback emitted no warning event"
    warn = warnings[0]
    assert warn.column == "x"
    assert warn.phase == "imputation"
    assert "fell back" in warn.message
    assert "fallback_to_median" in warn.message


def test_warning_events_log_at_warning_level(fallback_df, fallback_profile, caplog):
    with caplog.at_level(logging.WARNING, logger="dataforge_ml"):
        ImputationOrchestrator(_force_regression_config()).fit(
            fallback_df, fallback_profile
        )

    warn_records = [
        r for r in caplog.records
        if r.name == "dataforge_ml" and "fell back" in r.message
    ]
    assert warn_records, "no warning records captured at WARNING level"
    assert all(r.levelno == logging.WARNING for r in warn_records)


def test_clean_fit_emits_no_warning(imp_df, imp_profile):
    recorder = _Recorder()
    ImputationOrchestrator(observer=recorder).fit(imp_df, imp_profile)
    assert [e for e in recorder.events if e.event_type == EventType.warning] == []


def test_hard_failure_still_raises_and_is_not_softened(fallback_profile):
    # Regression forced on a column with fewer rows than the size guard is a
    # HARD failure: it must raise, never degrade into a warning event.
    df = pl.DataFrame({"x": pl.Series([1.0, None, 3.0], dtype=pl.Float64)})
    profile = StructuralProfiler(PipelineConfig()).profile(df)
    cfg = PipelineConfig(
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                regression_min_rows=10_000,
                _per_column_strategy={"x": ImputationStrategy.Regression},
            )
        )
    )
    recorder = _Recorder()
    with pytest.raises(ValueError):
        ImputationOrchestrator(cfg, observer=recorder).fit(df, profile)
    # No warning event was emitted in place of the exception.
    assert [e for e in recorder.events if e.event_type == EventType.warning] == []


def test_observer_can_isolate_warnings_by_event_type(fallback_df, fallback_profile):
    recorder = _Recorder()
    ImputationOrchestrator(_force_regression_config(), observer=recorder).fit(
        fallback_df, fallback_profile
    )
    warnings = [e for e in recorder.events if e.event_type == EventType.warning]
    assert warnings and all(e.event_type == EventType.warning for e in warnings)


# ---------------------------------------------------------------------------
# One observer instance reused across profiling and imputation (#319)
# ---------------------------------------------------------------------------


def test_same_observer_across_profile_fit_transform(imp_df):
    recorder = _Recorder()

    profile = StructuralProfiler(
        PipelineConfig(profiling=ProfileConfig()), observer=recorder
    ).profile(imp_df)
    ImputationOrchestrator(observer=recorder).fit_transform(imp_df, profile)

    phases = {e.phase for e in recorder.events}
    assert phases == {"profiling", "imputation"}

    # Each orchestrator's stages are honestly attributed to its own phase.
    profiling_stages = {e.stage for e in recorder.events if e.phase == "profiling"}
    imputation_stages = {e.stage for e in recorder.events if e.phase == "imputation"}
    assert "column_profiling" in profiling_stages
    assert "column_fitting" in imputation_stages
