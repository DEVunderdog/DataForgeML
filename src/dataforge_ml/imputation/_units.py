"""``derive_units(routing, strategy=None) -> tuple[ImputationUnit, ...]`` (ADR-0084).

The public selection surface over a routing's execution units: a pure
projection of ``routing.column_routings`` into the joint ``MICE`` block, the
joint ``KNN`` block, and one unit per independent column. Structural
strategies (Dropped / Passthrough / Indicator) train nothing and produce no
unit.
"""

from __future__ import annotations

from ._config import (
    _STRUCTURAL_STRATEGIES,
    ImputationRouting,
    ImputationStrategy,
    ImputationUnit,
)

__all__ = ["derive_units"]


def _project_units(routing: ImputationRouting) -> tuple[ImputationUnit, ...]:
    """Project ``routing.column_routings`` into its execution units."""
    mice_cols = tuple(
        c
        for c, r in routing.column_routings.items()
        if r.strategy == ImputationStrategy.MICE
    )
    knn_cols = tuple(
        c
        for c, r in routing.column_routings.items()
        if r.strategy == ImputationStrategy.KNN
    )

    units: list[ImputationUnit] = []
    mice_emitted = False
    knn_emitted = False
    for column, col_routing in routing.column_routings.items():
        strategy = col_routing.strategy
        if strategy == ImputationStrategy.MICE:
            if not mice_emitted:
                units.append(
                    ImputationUnit(
                        unit_id="mice",
                        strategy=ImputationStrategy.MICE,
                        columns=mice_cols,
                        is_block=True,
                    )
                )
                mice_emitted = True
        elif strategy == ImputationStrategy.KNN:
            if not knn_emitted:
                units.append(
                    ImputationUnit(
                        unit_id="knn",
                        strategy=ImputationStrategy.KNN,
                        columns=knn_cols,
                        is_block=True,
                    )
                )
                knn_emitted = True
        elif strategy in _STRUCTURAL_STRATEGIES:
            continue
        else:
            unit_id = f"{strategy}:{column}"
            units.append(
                ImputationUnit(
                    unit_id=unit_id,
                    strategy=strategy,
                    columns=(column,),
                    is_block=False,
                )
            )
    return tuple(units)


def derive_units(
    routing: ImputationRouting,
    strategy: ImputationStrategy | None = None,
) -> tuple[ImputationUnit, ...]:
    """Derive the execution units a routing implies.

    A pure projection of ``routing.column_routings`` (ADR-0084): the joint
    ``MICE`` block, the joint ``KNN`` block, and one unit per independent
    column (GMM-Sampling / Cluster-Conditional / scalar / Constant / MNAR).
    Structural strategies (Dropped / Passthrough / Indicator) train nothing and
    produce no unit.

    Parameters
    ----------
    routing : ImputationRouting
        The routing to project. Read-only.
    strategy : ImputationStrategy, optional
        When given, only units executing this strategy are returned. Keys on
        the enum rather than a hand-typed ``unit_id`` string, so a typo cannot
        silently miss a unit. Always returns a tuple; an empty one is an
        ordinary outcome when the routing sent nothing to ``strategy``.

    Returns
    -------
    tuple[ImputationUnit, ...]
        The derived units, in routing order. Empty when ``routing`` has no
        columns, or when ``strategy`` matched none of them.
    """
    units = _project_units(routing)
    if strategy is None:
        return units
    return tuple(u for u in units if u.strategy == strategy)
