"""Structured, ephemeral observability for one imputation unit's fit (ADR-0074).

A fit reports what happened through a :class:`FitSignals` value the caller reads
straight off the :class:`~dataforge_ml.imputation.UnitFitResult` it gets back —
not an event stream and not opaque strings. The record is **ephemeral**: it
describes one fit's runtime, never the fitted state, so it is never part of any
persisted artifact (ADR-0072).

Caveats come back as coded :class:`~dataforge_ml.records.ResultWarning`
records in :attr:`FitSignals.warnings`, never through the standard
:mod:`warnings` machinery (ADR-0102): fitting emits no Python warnings.

This module carries no dependency on the training surface; both the fitters and
the surface import from it, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..records import ResultWarning
from ._config import ImputationStrategy, _md_cell

__all__ = [
    "FitSignals",
]


@dataclass(frozen=True)
class FitSignals:
    """Structured record of one imputation unit's fit (ADR-0074).

    The observability value :func:`~dataforge_ml.imputation.fit_unit` returns on
    every :class:`~dataforge_ml.imputation.UnitFitResult`, letting the caller
    answer "did this fit converge? which estimator ran?" without parsing
    strings. A typed core carries the facts any consumer branches on; ``notes``
    is a free-form bag for strategy-specific minutiae that earn no typed field,
    kept separate from ``warnings`` so the cheap question "did anything go
    wrong?" is answered by reading ``warnings`` alone.

    The record is ephemeral — it describes the fit's runtime, not the fitted
    state, and is never serialized (ADR-0072). An untrainable unit raises
    an ``unit_not_trainable`` Library Error and produces no
    record: observability describes fits that happened (ADR-0074).

    Attributes
    ----------
    unit_id : str
        The plan id of the trained unit.
    strategy : ImputationStrategy
        The strategy that was executed.
    estimator : str or None
        Human-readable name of the estimator that ran (e.g.
        ``"RandomForestRegressor"``, ``"KNNImputer"``); ``None`` for strategies
        that fit no estimator, such as the scalar fills.
    converged : bool or None
        Whether an iterative model stopped before its iteration cap; ``None`` for
        strategies with no iterative-convergence notion.
    n_iter : int or None
        The iteration count an iterative model settled on; ``None`` when there is
        none.
    warnings : tuple[ResultWarning, ...]
        Caveats on the fit, each a registered ``warning`` Error Code — such as
        ``mice_not_converged`` for a model that hit its iteration cap. Branch on
        ``code``, not ``message``. Never also raised as a Python warning
        (ADR-0102).
    notes : tuple[str, ...]
        Informational fit-time detail (resolved dials, scaling, the routing
        branch taken) that does not earn a typed field.
    """

    unit_id: str
    strategy: ImputationStrategy
    estimator: str | None = None
    converged: bool | None = None
    n_iter: int | None = None
    warnings: tuple[ResultWarning, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        """Render the fit record as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so the owning
        :class:`~dataforge_ml.imputation.UnitFitResult` document composes it
        without a heading collision. The typed core renders as a field table;
        ``warnings`` and ``notes`` render as two separate ``####`` sections,
        preserving the separation ADR-0074 built into the type — "did anything
        go wrong?" is answered by reading the warnings section alone.

        Returns
        -------
        str
            Markdown subsection headed by ``### Fit Signals`` with the typed
            core, then a warnings section and a notes section.
        """
        lines = [
            f"### Fit Signals — `{self.unit_id}`\n",
            "| Field | Value |",
            "|---|---|",
            f"| strategy | {_md_cell(self.strategy)} |",
            f"| estimator | {_md_cell(self.estimator)} |",
            f"| converged | {_md_cell(self.converged)} |",
            f"| n_iter | {_md_cell(self.n_iter)} |",
            "",
            "#### Warnings\n",
        ]
        if self.warnings:
            lines.extend(f"- `{w.code}`: {w.message}" for w in self.warnings)
        else:
            lines.append("none")
        lines.append("")
        lines.append("#### Notes\n")
        if self.notes:
            lines.extend(f"- {note}" for note in self.notes)
        else:
            lines.append("none")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
