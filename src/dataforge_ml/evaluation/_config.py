"""Configuration for the evaluation module — an umbrella over per-metric dials.

:class:`EvaluationConfig` **stands alone and is never nested on**
``PipelineConfig``, deliberately breaking ADR-0030's Phase Sub-Config topology
(ADR-0087): ``PipelineConfig`` is built once for a run, whereas evaluation is
opt-in, stateless, and re-configured freely between calls while poking at
results. Each metric owns one nested config — :class:`C2STConfig` today — so a
later metric lands as a sibling field rather than a rename.

:class:`EvaluationMetric` is the input enum ``metrics=`` selects with. It is
deliberately *not* a nullable nested config: making ``EvaluationConfig.c2st is
None`` mean "off" would collapse *absence of configuration* and *refusal of the
metric* into one sentinel and put a control-flow switch inside a declarative
object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from ._c2st import DEFAULT_MIN_SAMPLES_LEAF, MIN_SAMPLE_FLOOR, C2STScheme

__all__ = [
    "C2STConfig",
    "EvaluationConfig",
    "EvaluationMetric",
]

# The repeat count a bare ``C2STConfig`` declares. Repeats are what make
# balancing's answer stable rather than the artefact of one lucky subsample, so
# the default is above 1; it is low enough that the default call does not cost
# five fits per column the way ``CrossValidation`` would.
_DEFAULT_REPEATS: int = 5

# The annotation band's upper edge, measured in ADR-0087: the degenerate
# imputer is caught 9% of the time at 60 rows per pile, 40% at 100, 81% at 150
# and 100% at 250. Between the hard floor and this number the test is honest
# but weak.
_DEFAULT_LOW_POWER_BELOW: int = 250

# The pile ratio below which the effective sample size is worth annotating.
# Its own dial, deliberately not the profile's ``DropCandidate`` threshold
# despite both sitting at 50%.
_DEFAULT_IMBALANCE_WARN_RATIO: float = 0.5

# The false discovery rate Benjamini-Hochberg controls when it turns the
# tested columns' p-values into verdicts. The conventional 5%: BH answers
# "which columns should a human go look at", and the number is what makes that
# shortlist arguable rather than a hidden constant.
_DEFAULT_FDR_ALPHA: float = 0.05


class EvaluationMetric(StrEnum):
    """The metric selector passed as ``metrics=`` to ``evaluate_imputation``.

    ``None`` — the default — means *every metric the library has*, so the bare
    call honours the umbrella name and nobody needs to know the list exists.
    Narrowing is scope selection on an opt-in diagnostic handed to a human, not
    the staging of a library decision to save cost, so it does not contradict
    the accuracy-over-speed principle (ADR-0087).
    """

    C2ST = "c2st"


@dataclass
class C2STConfig:
    """Dials for the Classifier Two-Sample Test.

    Every threshold the refusal machinery uses is here, so a user moves the
    floor for their own data rather than arguing with a constant. Two values
    are *not* dials and never will be: balancing, because unequal piles break
    the ``Binomial(n_test, ½)`` null outright, and the classifier's
    ``early_stopping`` pin, which is a structural correction to a dataset-size
    assumption rather than a threshold — a user wanting different classifier
    internals has the wider :attr:`classifier` door.

    Parameters
    ----------
    repeats : int
        The number of balanced subsampling repeats, ``R``. A single balanced
        draw is a random subsample whose ``z`` depends on which rows were
        drawn; the repeats are what make that answer stable. Never
        auto-downgraded when :attr:`scheme` is ``CrossValidation``.
    scheme : C2STScheme
        ``Split`` (one fit per repeat, the default) or ``CrossValidation``
        (five). ``Split`` is the default because repeats are already mandatory
        and a default that silently costs 5× is a bad default.
    min_samples_leaf : int
        The leaf-size floor of the library's default classifier. sklearn's 20
        cannot split fewer than 40 training rows, so the classifier
        constant-predicts and reports a constant fill as a perfect pass. Fixed,
        never scaled to pile size — a scaled setting would make the score a
        function of the column's missingness rate. Ignored when
        :attr:`classifier` is supplied, since injection is total.
    min_filled_cells : int
        The hard refusal floor, applied to the **smaller** pile
        (``min(n_observed, n_filled)``) before balancing, because every
        measured number is in rows per pile *after* balancing. Below it the
        column's outcome is ``BelowSampleFloor``.
    low_power_below : int
        The annotation band's upper edge, in rows per pile. An annotation and
        never a refusal, so a pass in that band reads "not enough data to tell"
        rather than "clean".
    imbalance_warn_ratio : float
        The pile ratio ``min(a, b) / max(a, b)`` below which the result is
        annotated as imbalanced, naming the smaller pile as the cap on the
        effective sample size.
    fdr_alpha : float
        The false discovery rate Benjamini-Hochberg controls across one
        report's tested columns. A column whose adjusted p-value is at or
        below it is ``Flagged``. It moves the **shortlist** only: the raw
        ``z`` is never corrected, because the ``z`` is the score.
    random_state : int, optional
        Seed for the subsampling, the train/test carve and the default
        classifier. The same seed on the same frames gives the same report.
    classifier : Any, optional
        An unfitted classifier to use **verbatim** for every column of the run
        — no ``set_params`` reach-in and no clone, mirroring how
        ``ImputationDecision.custom_estimators`` holds the user's live object.
        ``None`` builds the library's default, pins included. Stated hole
        (ADR-0087): an injected *bare*
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier` reintroduces
        the constant-predictor pathology at 30–60 rows per pile, and only
        ``classifier_injected`` in the provenance warns the reader. Being a
        live object it is the one field :meth:`to_dict` cannot carry.
    """

    repeats: int = _DEFAULT_REPEATS
    scheme: C2STScheme = C2STScheme.Split
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF
    min_filled_cells: int = MIN_SAMPLE_FLOOR
    low_power_below: int = _DEFAULT_LOW_POWER_BELOW
    imbalance_warn_ratio: float = _DEFAULT_IMBALANCE_WARN_RATIO
    fdr_alpha: float = _DEFAULT_FDR_ALPHA
    random_state: Optional[int] = None
    classifier: Optional[Any] = None

    def to_dict(self) -> dict:
        """Serialise the dials to a plain dictionary.

        :attr:`classifier` is deliberately absent: it is a live estimator
        instance, and a config that is serialisable except for one field that
        silently degrades on the round trip is worse than one that states the
        omission. The report's ``C2STProvenance`` is where a run's classifier
        is recorded.

        Returns
        -------
        dict
            Every dial except :attr:`classifier`, keyed by field name.
        """
        return {
            "repeats": self.repeats,
            "scheme": str(self.scheme),
            "min_samples_leaf": self.min_samples_leaf,
            "min_filled_cells": self.min_filled_cells,
            "low_power_below": self.low_power_below,
            "imbalance_warn_ratio": self.imbalance_warn_ratio,
            "fdr_alpha": self.fdr_alpha,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> C2STConfig:
        """Reconstruct a ``C2STConfig`` from a plain dictionary.

        Missing keys fall back to field defaults, so a serialised config
        written against an earlier version loads rather than raising. The
        reconstructed config carries ``classifier=None``; see :meth:`to_dict`.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`.

        Returns
        -------
        C2STConfig
            Reconstructed config instance.
        """
        default = cls()
        scheme = data.get("scheme")
        return cls(
            repeats=data.get("repeats", default.repeats),
            scheme=C2STScheme(scheme) if scheme is not None else default.scheme,
            min_samples_leaf=data.get(
                "min_samples_leaf", default.min_samples_leaf
            ),
            min_filled_cells=data.get(
                "min_filled_cells", default.min_filled_cells
            ),
            low_power_below=data.get(
                "low_power_below", default.low_power_below
            ),
            imbalance_warn_ratio=data.get(
                "imbalance_warn_ratio", default.imbalance_warn_ratio
            ),
            fdr_alpha=data.get("fdr_alpha", default.fdr_alpha),
            random_state=data.get("random_state", default.random_state),
        )


@dataclass
class EvaluationConfig:
    """The evaluation module's configuration object, one nested config per metric.

    Handed to ``evaluate_imputation`` as ``config=``. It holds no observer:
    a live callable would break the serialisable, setter-only contract every
    config in the library keeps (ADR-0044), so ``observer=`` rides the call
    instead.

    Parameters
    ----------
    c2st : C2STConfig
        Dials for the Classifier Two-Sample Test, the module's first metric.
    """

    c2st: C2STConfig = field(default_factory=C2STConfig)

    def to_dict(self) -> dict:
        """Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name, with ``c2st`` nested.
        """
        return {"c2st": self.c2st.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> EvaluationConfig:
        """Construct an ``EvaluationConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by :meth:`to_dict`. Missing keys fall back to
            field defaults.

        Returns
        -------
        EvaluationConfig
            Reconstructed config instance.
        """
        return cls(c2st=C2STConfig.from_dict(data.get("c2st", {})))
