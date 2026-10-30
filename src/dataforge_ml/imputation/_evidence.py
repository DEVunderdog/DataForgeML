"""Routing Reason and Routing Evidence: why ``route`` decided a column (ADR-0105).

Each :class:`~dataforge_ml.imputation.ColumnRouting` carries one
:class:`RoutingReason` — the **terminal** rule that fixed its strategy — and
an ordered ``evidence`` tuple of typed facts: the walk that led there. The
facts are Wire Records, so they persist with the routing and with the
recipe's routing copy, and an Agent branches on their fields, never on prose.

- :class:`EntryFact` — why the column reached the escalation seam.
- :class:`SignalScoreFact` — the Signal Score and the terms it was built from.
- :class:`TierFact` — the Signal Tier the score fell in.
- :class:`FeasibilityFact` — one candidate's Feasibility Floor verdict, each
  failed term carrying a :class:`~dataforge_ml.records.DialPointer` to the
  dial that decided it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from ..records import DataForgeError, DialPointer, WireRecord

__all__ = [
    "EntryCause",
    "EntryFact",
    "FailedTerm",
    "FeasibilityFact",
    "RoutingEvidence",
    "RoutingReason",
    "SignalScoreFact",
    "SignalTier",
    "TierFact",
]


class RoutingReason(StrEnum):
    """The terminal rule that fixed a column's strategy (ADR-0105).

    Declared or structural reasons: ``drop_candidate``,
    ``constant_declared``, ``strategy_forced``, ``mnar_declared``,
    ``no_missing_values``, ``soft_excluded``, ``not_imputed_type``,
    ``authored``.

    Shape rules: ``mcar_minor_unskewed`` (Mean), ``mcar_skewed`` (Median),
    ``mar_unescalated`` (Median for a MAR-suspect column that met no
    escalation rule), ``bounded_discrete_scalar`` (Mode),
    ``bimodal_grouping_declared`` and ``bimodal_few_correlated``
    (Cluster-Conditional), ``bimodal_uncorrelated`` (GMM Sampling, or Mode
    for a BoundedDiscrete column).

    Seam outcomes: ``signal_tier_pick``, ``bottom_tier_fill``,
    ``no_feasible_model`` (the tier wanted a model and none was feasible),
    ``no_predictor_correlation`` (no Signal Score could be computed).
    """

    drop_candidate = "drop_candidate"
    constant_declared = "constant_declared"
    strategy_forced = "strategy_forced"
    mnar_declared = "mnar_declared"
    no_missing_values = "no_missing_values"
    soft_excluded = "soft_excluded"
    not_imputed_type = "not_imputed_type"
    authored = "authored"
    mcar_minor_unskewed = "mcar_minor_unskewed"
    mcar_skewed = "mcar_skewed"
    mar_unescalated = "mar_unescalated"
    bounded_discrete_scalar = "bounded_discrete_scalar"
    bimodal_grouping_declared = "bimodal_grouping_declared"
    bimodal_few_correlated = "bimodal_few_correlated"
    bimodal_uncorrelated = "bimodal_uncorrelated"
    signal_tier_pick = "signal_tier_pick"
    bottom_tier_fill = "bottom_tier_fill"
    no_feasible_model = "no_feasible_model"
    no_predictor_correlation = "no_predictor_correlation"


class EntryCause(StrEnum):
    """Why a column reached the escalation seam.

    ``bounded_discrete`` precedes the missingness cause for a BoundedDiscrete
    column. The ``mar_*`` causes split the MAR-suspect walk:
    ``mar_multi`` (two or more MAR-suspect columns), ``mar_severe``,
    ``mar_high_correlated``, ``mar_high_uncorrelated`` (no missingness
    correlation, so the MCAR High chain applies) and ``mar_shape_escalation``
    (Leptokurtic or Severe skew at Minor/Moderate severity). The ``mcar_*``
    causes are ``mcar_high_severity`` (High or Severe),
    ``mcar_minor_leptokurtic`` and ``mcar_moderate_shape``.
    ``bimodal_many_correlated`` is the Bimodal Imputation Framework's
    branch 2.
    """

    bounded_discrete = "bounded_discrete"
    mar_multi = "mar_multi"
    mar_severe = "mar_severe"
    mar_high_correlated = "mar_high_correlated"
    mar_high_uncorrelated = "mar_high_uncorrelated"
    mar_shape_escalation = "mar_shape_escalation"
    mcar_high_severity = "mcar_high_severity"
    mcar_minor_leptokurtic = "mcar_minor_leptokurtic"
    mcar_moderate_shape = "mcar_moderate_shape"
    bimodal_many_correlated = "bimodal_many_correlated"


class SignalTier(StrEnum):
    """The Signal Tier a Signal Score falls in (ADR-0092)."""

    bottom = "bottom"
    middle = "middle"
    top = "top"


class EntryFact(WireRecord):
    """Why the column reached the escalation seam."""

    kind: Literal["entry"] = "entry"
    """Fact tag, always ``entry``."""
    cause: EntryCause
    """The routing rule that sent the column to the seam."""


class SignalScoreFact(WireRecord):
    """The Signal Score and the terms it was built from (ADR-0092)."""

    kind: Literal["signal_score"] = "signal_score"
    """Fact tag, always ``signal_score``."""
    score: float
    """The Signal Score, in [0, 1]."""
    base: float
    """Explainable Variance: the larger of the best R² and the squared largest |r|."""
    latent: float
    """Latent Structure, from the largest mutual information."""
    breadth: float
    """Signal Breadth, ``predictors / (predictors + 1)``."""
    predictors: int
    """Predictors whose |r| is above ``mice_correlation_threshold``."""
    degraded: bool
    """True when the R² pair was unavailable, so ``base`` is the squared largest |r| alone."""


class TierFact(WireRecord):
    """The Signal Tier the column's score fell in."""

    kind: Literal["tier"] = "tier"
    """Fact tag, always ``tier``."""
    tier: SignalTier
    """``bottom``, ``middle`` or ``top``."""


class FailedTerm(WireRecord):
    """One Feasibility Term or condition a candidate failed."""

    term: str
    """The term's name, such as ``rows_per_predictor`` or ``row_count``."""
    value: bool | int | float | None
    """The column's value for the term, or null when it is undefined."""
    limit: bool | int | float | None
    """The bound the value failed, or null when the term has none."""
    dial: DialPointer | None
    """The dial that set ``limit``, or null when no dial changes this outcome."""


class FeasibilityFact(WireRecord):
    """One candidate strategy's Feasibility Floor verdict (ADR-0091)."""

    kind: Literal["feasibility"] = "feasibility"
    """Fact tag, always ``feasibility``."""
    strategy: str
    """The candidate strategy, by value, such as ``mice`` or ``knn``."""
    passed: bool
    """True when the candidate passed every term."""
    failed_terms: list[FailedTerm]
    """The terms the candidate failed. Empty when ``passed`` is true."""


RoutingEvidence = Annotated[
    EntryFact | SignalScoreFact | TierFact | FeasibilityFact,
    Field(discriminator="kind"),
]
"""One Routing Evidence fact, tagged by ``kind``."""

_EVIDENCE_LIST = TypeAdapter(list[RoutingEvidence])


def _evidence_to_list(evidence: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Serialise an evidence tuple to JSON-ready dictionaries."""
    return [fact.to_dict() for fact in evidence]


def _evidence_from_list(column: str, raw: Any) -> tuple[Any, ...]:
    """Strictly rebuild an evidence tuple, raising ``incompatible_artifact`` on any mismatch."""
    try:
        return tuple(_EVIDENCE_LIST.validate_json(json.dumps(raw)))
    except (ValidationError, TypeError, ValueError) as exc:
        raise DataForgeError.from_code(
            "incompatible_artifact",
            f"Column '{column}': saved routing evidence does not match this "
            f"library's schema. Route again.",
        ) from exc


def _reason_from_value(column: str, raw: Any) -> RoutingReason:
    """Strictly rebuild a Routing Reason, raising ``incompatible_artifact`` on any mismatch."""
    try:
        return RoutingReason(raw)
    except ValueError as exc:
        raise DataForgeError.from_code(
            "incompatible_artifact",
            f"Column '{column}': saved routing reason {raw!r} is not one this "
            f"library knows. Route again.",
        ) from exc
