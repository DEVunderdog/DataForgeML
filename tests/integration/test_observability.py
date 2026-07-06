"""Observability spine: Pipeline Event stream, Trace logger, and observers.

Covers the backbone (Issue #317) and the per-column item heartbeat (#318),
both proven end-to-end through ``StructuralProfiler``.
"""

import logging

import pytest

import dataforge_ml
from dataforge_ml import (
    EventType,
    PipelineConfig,
    PipelineEvent,
    StructuralProfiler,
    stderr_observer,
)
from dataforge_ml.profiling._config import ProfileConfig


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Public interface / exports
# ---------------------------------------------------------------------------

def test_exports_present():
    for name in ("PipelineEvent", "EventType", "stderr_observer"):
        assert name in dataforge_ml.__all__
        assert hasattr(dataforge_ml, name)


def test_event_type_is_enum_not_str():
    # event_type carries the enum, never a bare string.
    assert isinstance(EventType.item, EventType)
    assert EventType.stage_start.value == "stage_start"


# ---------------------------------------------------------------------------
# Backbone (#317): stage-event sequence via a recording observer
# ---------------------------------------------------------------------------

def test_stage_events_reach_observer(mixed_df):
    recorder = _Recorder()
    config = PipelineConfig(profiling=ProfileConfig(compute_correlation=True))
    StructuralProfiler(config, observer=recorder).profile(mixed_df)

    stage_events = [
        e for e in recorder.events
        if e.event_type in (EventType.stage_start, EventType.stage_end)
    ]
    assert stage_events, "no stage events emitted"

    # Every stage_start is matched by a later stage_end for the same stage.
    opened: list[str] = []
    for e in stage_events:
        assert e.phase == "profiling"
        assert e.message
        if e.event_type == EventType.stage_start:
            opened.append(e.stage)
        else:
            assert opened and opened[-1] == e.stage, (
                f"stage_end for {e.stage!r} without matching open"
            )
            opened.pop()
    assert not opened, f"stages left open: {opened}"

    # The always-run stages are present.
    started = {e.stage for e in stage_events if e.event_type == EventType.stage_start}
    assert {"modality", "type_detection", "column_profiling"} <= started


def test_no_observer_is_harmless(mixed_df):
    # observer defaults to None; profiling must still succeed and be silent.
    result = StructuralProfiler(PipelineConfig()).profile(mixed_df)
    assert set(result.columns.keys()) == set(mixed_df.columns)


# ---------------------------------------------------------------------------
# Two-Sink Rule (#317): Trace logger level mapping + silence-by-default
# ---------------------------------------------------------------------------

def test_stage_events_log_at_debug(mixed_df, caplog):
    with caplog.at_level(logging.DEBUG, logger="dataforge_ml"):
        StructuralProfiler(PipelineConfig()).profile(mixed_df)

    stage_records = [
        r for r in caplog.records
        if r.name == "dataforge_ml" and "started" in r.message
    ]
    assert stage_records, "no stage records captured"
    assert all(r.levelno == logging.DEBUG for r in stage_records)


def test_silent_by_default(mixed_df, caplog):
    # A NullHandler is attached at import; with no propagation capture at
    # WARNING (pytest's default), DEBUG stage/item events emit nothing.
    with caplog.at_level(logging.WARNING, logger="dataforge_ml"):
        StructuralProfiler(PipelineConfig()).profile(mixed_df)
    assert [r for r in caplog.records if r.name == "dataforge_ml"] == []


def test_logger_has_null_handler():
    logger = logging.getLogger("dataforge_ml")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


# ---------------------------------------------------------------------------
# Default observer (#317): stderr-only, never stdout
# ---------------------------------------------------------------------------

def test_stderr_observer_writes_stderr_only(capsys):
    event = PipelineEvent(
        event_type=EventType.stage_start,
        phase="profiling",
        stage="modality",
        message="[profiling] modality started",
    )
    stderr_observer(event)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "modality started" in captured.err


# ---------------------------------------------------------------------------
# Item heartbeat (#318): stage_start -> item 1..N -> stage_end
# ---------------------------------------------------------------------------

def test_column_profiling_item_heartbeat(mixed_df):
    recorder = _Recorder()
    StructuralProfiler(PipelineConfig(), observer=recorder).profile(mixed_df)

    # Slice out just the column_profiling stage's events, in order.
    stage_stream = [e for e in recorder.events if e.stage == "column_profiling"]
    assert stage_stream[0].event_type == EventType.stage_start
    assert stage_stream[-1].event_type == EventType.stage_end

    items = [e for e in stage_stream if e.event_type == EventType.item]
    assert items, "no item events emitted"

    total = items[0].total
    assert total == len(items), "total must equal the number of item events"

    # index is 1-based and strictly monotonic; total is constant.
    for i, e in enumerate(items, start=1):
        assert e.index == i
        assert e.total == total
        assert e.column is not None
        assert e.column in mixed_df.columns
        assert e.event_type is EventType.item

    # All items lie strictly between the stage boundaries.
    assert all(e.event_type == EventType.item for e in stage_stream[1:-1])

    # Column identity is honest and unique across the heartbeat.
    columns = [e.column for e in items]
    assert len(set(columns)) == len(columns)


def test_cheap_stages_emit_no_items(mixed_df):
    recorder = _Recorder()
    StructuralProfiler(PipelineConfig(), observer=recorder).profile(mixed_df)

    item_stages = {e.stage for e in recorder.events if e.event_type == EventType.item}
    # Only the expensive per-column loop emits item events.
    assert item_stages == {"column_profiling"}


# ---------------------------------------------------------------------------
# Item events across correlation / nonlinearity (#322)
# ---------------------------------------------------------------------------

def _stage_stream(events, stage):
    """Events for one stage, in emission order."""
    return [e for e in events if e.stage == stage]


def _assert_monotonic_items(items):
    """item events are 1-based, strictly increasing, with a constant total."""
    assert items, "no item events emitted"
    total = items[0].total
    assert total == len(items), "total must equal the number of item events"
    for i, e in enumerate(items, start=1):
        assert e.event_type is EventType.item
        assert e.index == i
        assert e.total == total


def test_correlation_and_nonlinearity_emit_items(mixed_df):
    recorder = _Recorder()
    config = PipelineConfig(
        profiling=ProfileConfig(compute_correlation=True, compute_nonlinearity=True)
    )
    StructuralProfiler(config, observer=recorder).profile(mixed_df)

    # Both expensive stages now report progress rather than running silently.
    item_stages = {e.stage for e in recorder.events if e.event_type == EventType.item}
    assert {"correlation", "nonlinearity"} <= item_stages

    # ── Correlation ──────────────────────────────────────────────────────
    corr = _stage_stream(recorder.events, "correlation")
    assert corr[0].event_type == EventType.stage_start
    assert corr[-1].event_type == EventType.stage_end
    corr_items = [e for e in corr if e.event_type == EventType.item]
    _assert_monotonic_items(corr_items)
    # With no declared target, the sole unit is the feature-feature matrices;
    # it is not tied to a single column.
    assert corr_items[0].column is None
    assert corr_items[0].total == 1

    # ── Nonlinearity ─────────────────────────────────────────────────────
    nl = _stage_stream(recorder.events, "nonlinearity")
    assert nl[0].event_type == EventType.stage_start
    assert nl[-1].event_type == EventType.stage_end
    nl_items = [e for e in nl if e.event_type == EventType.item]
    _assert_monotonic_items(nl_items)
    # Every nonlinearity item names a real column, and columns are unique.
    cols = [e.column for e in nl_items]
    assert all(c in mixed_df.columns for c in cols)
    assert len(set(cols)) == len(cols)
    # All items lie strictly between the stage boundaries.
    assert all(e.event_type == EventType.item for e in nl[1:-1])


def test_correlation_items_cover_each_target(mixed_df):
    recorder = _Recorder()
    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_correlation=True, target_columns=["income"]
        )
    )
    StructuralProfiler(config, observer=recorder).profile(mixed_df)

    corr_items = [
        e
        for e in recorder.events
        if e.stage == "correlation" and e.event_type == EventType.item
    ]
    _assert_monotonic_items(corr_items)
    # One unit for the feature matrices plus one per declared target present.
    assert corr_items[0].total == 2
    assert corr_items[0].column is None
    assert corr_items[1].column == "income"


def test_correlation_nonlinearity_items_log_at_debug(mixed_df, caplog):
    config = PipelineConfig(
        profiling=ProfileConfig(compute_correlation=True, compute_nonlinearity=True)
    )
    with caplog.at_level(logging.DEBUG, logger="dataforge_ml"):
        StructuralProfiler(config).profile(mixed_df)

    # Two-Sink Rule: item events from both stages also reach the Trace logger
    # at DEBUG.
    item_records = [
        r
        for r in caplog.records
        if r.name == "dataforge_ml"
        and ("correlation:" in r.message or "nonlinearity:" in r.message)
    ]
    assert item_records, "no correlation/nonlinearity item records captured"
    assert all(r.levelno == logging.DEBUG for r in item_records)
