"""Observability spine — the Pipeline Event stream, Trace logger, and observers.

Every long-running orchestrator emits a single stream of :class:`PipelineEvent`
records. Each event is delivered to two sinks (the *Two-Sink Rule*, ADR-0054):
the named ``dataforge_ml`` stdlib logger (Trace, silent by default via a
``NullHandler``, level derived from :class:`EventType`) and — when supplied — a
user-provided Progress Observer, a plain ``Callable[[PipelineEvent], None]``
passed as ``observer=`` on the orchestrator. The library never writes to a
stream on its own; only the shipped :func:`stderr_observer`, which a user opts
into, writes to stderr.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Trace sink: the named library logger, silent by default.
#
# A library must not configure logging for the applications that consume it, so
# the ``dataforge_ml`` logger carries a NullHandler and emits nothing until the
# consumer attaches their own handler (ADR-0054). Attached once, at import.
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger("dataforge_ml")
logger.addHandler(logging.NullHandler())


class EventType(StrEnum):
    """The kind of a :class:`PipelineEvent`, used to route and filter the stream.

    The value also determines the Trace log level: ``stage_start``,
    ``stage_end`` and ``item`` log at ``DEBUG``, ``decision`` at ``INFO``, and
    ``warning`` at ``WARNING``.
    """

    stage_start = "stage_start"
    stage_end = "stage_end"
    item = "item"
    decision = "decision"
    warning = "warning"


@dataclass(frozen=True)
class PipelineEvent:
    """A single structured observability record emitted by an orchestrator.

    Progress and Trace share this one shape. Consumers filter on
    ``event_type`` and read the raw ``index``/``total`` fields to drive a
    progress bar, or print the ready-made ``message`` as-is.

    Parameters
    ----------
    event_type : EventType
        The kind of event; also fixes the Trace log level.
    phase : str
        The pipeline phase emitting the event (e.g. ``"profiling"``).
    stage : str
        The stage within the phase (e.g. ``"column_profiling"``).
    message : str
        A ready-made human-readable sentence describing the event.
    column : str, optional
        The column an ``item`` event concerns; ``None`` for stage boundaries.
    index : int, optional
        The 1-based position of the current item within its stage; ``None`` for
        stage boundaries.
    total : int, optional
        The total number of items the stage will process; ``None`` for stage
        boundaries.
    """

    event_type: EventType
    phase: str
    stage: str
    message: str
    column: Optional[str] = None
    index: Optional[int] = None
    total: Optional[int] = None


# Progress Observer form: a plain callable handed each event as it happens.
Observer = Callable[[PipelineEvent], None]

_LEVEL_BY_EVENT_TYPE: dict[EventType, int] = {
    EventType.stage_start: logging.DEBUG,
    EventType.stage_end: logging.DEBUG,
    EventType.item: logging.DEBUG,
    EventType.decision: logging.INFO,
    EventType.warning: logging.WARNING,
}


def _emit(event: PipelineEvent, observer: Observer | None) -> None:
    """Deliver one event to both sinks: the Trace logger and the observer.

    Honours the Two-Sink Rule — the event always reaches the logger (at the
    level derived from ``event_type``) and, when supplied, the observer.
    """
    logger.log(_LEVEL_BY_EVENT_TYPE.get(event.event_type, logging.INFO), event.message)
    if observer is not None:
        observer(event)


class _ObservabilityMixin:
    """Shared event-emission helpers for orchestrators.

    Subclasses set the class attribute ``_PHASE`` and the instance attribute
    ``_observer``; the mixin turns stage/item calls into :class:`PipelineEvent`
    records dispatched through the Two-Sink Rule.
    """

    _PHASE: str = ""
    _observer: Observer | None = None

    def _emit_stage_start(self, stage: str) -> None:
        _emit(
            PipelineEvent(
                event_type=EventType.stage_start,
                phase=self._PHASE,
                stage=stage,
                message=f"[{self._PHASE}] {stage} started",
            ),
            self._observer,
        )

    def _emit_stage_end(self, stage: str) -> None:
        _emit(
            PipelineEvent(
                event_type=EventType.stage_end,
                phase=self._PHASE,
                stage=stage,
                message=f"[{self._PHASE}] {stage} completed",
            ),
            self._observer,
        )

    def _emit_item(
        self,
        stage: str,
        column: Optional[str],
        index: int,
        total: int,
        message: Optional[str] = None,
    ) -> None:
        # ``column`` is None for a work unit that is not tied to a single column
        # (e.g. the one-time feature-feature correlation matrices); callers pass
        # an explicit ``message`` in that case since the default sentence reads
        # off ``column``.
        _emit(
            PipelineEvent(
                event_type=EventType.item,
                phase=self._PHASE,
                stage=stage,
                message=message
                or f"[{self._PHASE}] {stage}: {column} ({index}/{total})",
                column=column,
                index=index,
                total=total,
            ),
            self._observer,
        )

    def _emit_decision(self, stage: str, column: str, message: str) -> None:
        _emit(
            PipelineEvent(
                event_type=EventType.decision,
                phase=self._PHASE,
                stage=stage,
                message=message,
                column=column,
            ),
            self._observer,
        )

    def _emit_warning(self, stage: str, column: str, message: str) -> None:
        _emit(
            PipelineEvent(
                event_type=EventType.warning,
                phase=self._PHASE,
                stage=stage,
                message=message,
                column=column,
            ),
            self._observer,
        )


def stderr_observer(event: PipelineEvent) -> None:
    """Write an event's ``message`` to stderr; the shipped opt-in observer.

    Pass this as ``observer=`` to an orchestrator to see progress on stderr
    without wiring up a custom callback. Writes only to stderr, never stdout.

    Parameters
    ----------
    event : PipelineEvent
        The event to render.
    """
    print(event.message, file=sys.stderr)
