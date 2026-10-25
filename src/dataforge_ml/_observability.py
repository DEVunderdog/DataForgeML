"""Observability spine — the progress-only Pipeline Event stream (internal).

Every long-running orchestrator emits a single stream of :class:`PipelineEvent`
records. Each event is delivered to two sinks (the *Two-Sink Rule*, ADR-0054):
the named ``dataforge_ml`` stdlib logger (Trace, silent by default via a
``NullHandler``, at ``DEBUG``) and — when set — a Progress Observer, a plain
``Callable[[PipelineEvent], None]``. Events are progress only (ADR-0102,
ADR-0112): caveats are Result Warnings on the result, never events. Nothing
here is public; the CLI's ``--progress`` flag is the only way an Agent sees the
stream, through :func:`stderr_observer`.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

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

    Every kind is progress; all of them log to Trace at ``DEBUG``.

    ``substep`` is a Substep heartbeat *below* the column (a strategy-block fit,
    a diagnostics fold, a per-column model fit within a set); it is additive, so
    an observer that ignores it is unaffected, and a consumer drives a coarse
    progress bar off ``item`` while reading a live detail line off ``substep``
    (ADR-0055).
    """

    stage_start = "stage_start"
    stage_end = "stage_end"
    item = "item"
    substep = "substep"


@dataclass(frozen=True)
class PipelineEvent:
    """A single structured observability record emitted by an orchestrator.

    Progress and Trace share this one shape. Consumers filter on
    ``event_type`` and read the raw ``index``/``total`` fields to drive a
    progress bar, or print the ready-made ``message`` as-is.

    Parameters
    ----------
    event_type : EventType
        The kind of progress event.
    phase : str
        The pipeline phase emitting the event (e.g. ``"profiling"``).
    stage : str
        The stage within the phase (e.g. ``"column_profiling"``).
    message : str
        A ready-made human-readable sentence describing the event.
    column : str, optional
        The column an ``item`` or ``substep`` event concerns; ``None`` for
        stage boundaries and column-independent work.
    index : int, optional
        The 1-based position of the current item within its stage, or the
        Substep sub-progression (e.g. fold ``3`` of ``5``) on a ``substep``
        event; ``None`` for stage boundaries.
    total : int, optional
        The total number of items the stage will process, or the Substep
        sub-total (e.g. ``5`` folds) on a ``substep`` event; ``None`` for stage
        boundaries.
    """

    event_type: EventType
    phase: str
    stage: str
    message: str
    column: str | None = None
    index: int | None = None
    total: int | None = None


# Progress Observer form: a plain callable handed each event as it happens.
Observer = Callable[[PipelineEvent], None]


def _emit(event: PipelineEvent, observer: Observer | None) -> None:
    """Deliver one event to both sinks: the Trace logger and the observer.

    Honours the Two-Sink Rule — the event always reaches the logger (at
    ``DEBUG``) and, when supplied, the observer.
    """
    logger.debug(event.message)
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
        column: str | None,
        index: int,
        total: int,
        message: str | None = None,
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


class Emitter:
    """Threaded event source that lets deep fit methods emit progress.

    An orchestrator's sub-processors (the imputation fitters and their
    helpers) are otherwise observer-blind: the Progress Observer lives only on
    the orchestrator, so the methods doing the slow work cannot report activity.
    The orchestrator builds exactly one :class:`Emitter` — carrying the
    ``phase``, the ``stage``, and the Progress Observer — and threads it
    explicitly into those methods (never as an ambient global). The Emitter owns
    the ``item`` index bookkeeping and applies the Two-Sink Rule in one place, so
    a call site just says what happened, e.g. ``emitter.substep("MICE block
    fitting")`` (ADR-0055).

    A single Emitter is shared by the concurrent fitting threads (ADR-0056); it
    is thread-safe, serialising the ``item`` index bookkeeping and event
    delivery behind an internal lock so the "column k of N" heartbeat stays
    monotonic no matter which worker thread reports first.

    Parameters
    ----------
    phase : str
        The pipeline phase the emitted events belong to (e.g. ``"imputation"``).
    stage : str
        The stage within the phase the emitted events belong to (e.g.
        ``"column_fitting"``).
    observer : Callable[[PipelineEvent], None], optional
        The Progress Observer to deliver events to, or ``None`` to emit to the
        Trace logger only.
    total : int, optional
        The total number of ``item`` heartbeats the stage will emit, stamped on
        every :meth:`item` event; ``None`` when the total is unknown.
    """

    def __init__(
        self,
        phase: str,
        stage: str,
        observer: Observer | None,
        total: int | None = None,
    ) -> None:
        self._phase = phase
        self._stage = stage
        self._observer = observer
        self._total = total
        self._index = 0
        self._lock = threading.Lock()

    def stage_start(self) -> None:
        """Emit the ``stage_start`` boundary event for the Emitter's stage.

        The stage boundaries live on :class:`_ObservabilityMixin` for the
        orchestrators that are classes; an entry point that is a *free function*
        inherits nothing, so the Emitter carries them too (ADR-0087). Purely
        additive — the mixin is unchanged and its users are unaffected.
        """
        _emit(
            PipelineEvent(
                event_type=EventType.stage_start,
                phase=self._phase,
                stage=self._stage,
                message=f"[{self._phase}] {self._stage} started",
            ),
            self._observer,
        )

    def stage_end(self) -> None:
        """Emit the ``stage_end`` boundary event for the Emitter's stage.

        The closing half of :meth:`stage_start`; see that method for why the
        boundaries exist on the Emitter as well as on
        :class:`_ObservabilityMixin`.
        """
        _emit(
            PipelineEvent(
                event_type=EventType.stage_end,
                phase=self._phase,
                stage=self._stage,
                message=f"[{self._phase}] {self._stage} completed",
            ),
            self._observer,
        )

    def item(self, column: str | None, message: str | None = None) -> None:
        """Emit an ``item`` heartbeat for a column, auto-incrementing the index.

        Each call advances the Emitter's internal 1-based counter and stamps it,
        together with the ``total`` supplied at construction, onto the event so a
        watching observer sees monotonic "column k of N" progress.

        Parameters
        ----------
        column : str, optional
            The column being fitted; ``None`` for a work unit not tied to a
            single column, in which case ``message`` should be supplied.
        message : str, optional
            An explicit human-readable sentence; defaults to a ``"{column}
            (index/total)"`` rendering when omitted.
        """
        with self._lock:
            self._index += 1
            _emit(
                PipelineEvent(
                    event_type=EventType.item,
                    phase=self._phase,
                    stage=self._stage,
                    message=message
                    or f"[{self._phase}] {self._stage}: {column} "
                    f"({self._index}/{self._total})",
                    column=column,
                    index=self._index,
                    total=self._total,
                ),
                self._observer,
            )

    def substep(
        self,
        message: str,
        column: str | None = None,
        index: int | None = None,
        total: int | None = None,
    ) -> None:
        """Emit a ``substep`` heartbeat for sub-column work.

        Used to surface continuous activity during the long inner stretches (a
        strategy-block fit, a diagnostics fold, a per-column model fit within a
        set) that were previously silent. Sub-progression such as "fold 3 of 5"
        rides the existing ``index``/``total`` fields; ``PipelineEvent`` gains no
        new fields.

        Parameters
        ----------
        message : str
            What is currently in flight (e.g. ``"MICE block fitting"``). The
            phase and stage are prefixed automatically.
        column : str, optional
            The column the Substep concerns; ``None`` for block-level work
            spanning several columns.
        index : int, optional
            The 1-based position of this Substep within its group (e.g. the
            current fold); ``None`` when there is no sub-progression to report.
        total : int, optional
            The size of the Substep group (e.g. the fold count); ``None`` when
            unknown or not applicable.
        """
        rendered = f"[{self._phase}] {self._stage}: {message}"
        if index is not None and total is not None:
            rendered += f" ({index}/{total})"
        with self._lock:
            _emit(
                PipelineEvent(
                    event_type=EventType.substep,
                    phase=self._phase,
                    stage=self._stage,
                    message=rendered,
                    column=column,
                    index=index,
                    total=total,
                ),
                self._observer,
            )


def stderr_observer(event: PipelineEvent) -> None:
    """Write an event's ``message`` to stderr; the shipped opt-in observer.

    The CLI installs it for ``--progress``. Writes only to stderr, never
    stdout, which carries the result.

    Parameters
    ----------
    event : PipelineEvent
        The event to render.
    """
    print(event.message, file=sys.stderr)
