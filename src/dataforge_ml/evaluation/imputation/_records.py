"""The records ``imputation_score_c2st`` returns.

One record per **active** column, always: *untestable is a result, never an
absence*. A 200-column table yields 200 records, and the ones the test could
not run on carry a reason rather than going silently missing. ``report[col]``
and ``col in report`` work; a column the user excluded raises ``KeyError``,
because it is not in the census at all.

Refusal is **structural, not a sentinel number**: a refused column has no
:class:`C2STScore` object, so ``result.score.mean_frame_z`` raises
``AttributeError`` on the spot instead of feeding a ``0.0`` into a mean, a plot
or a comparison. The ``m`` frames nest *inside* the column record rather than
forming a second key, because the missingness mask is ``m``-invariant — pile
sizes, floors and annotations are column-level facts that cannot differ
between frames, so they are stated once and cannot disagree with themselves.

A tested column's :class:`C2STVerdict` is stamped on by :class:`C2STReport`
after Benjamini-Hochberg runs across the tested set, because BH is a property
of the *set* and no single record can compute it. The verdict is three-valued
and never a bool: a refused column has ``NoVerdict``, where ``False`` would
read as "checked, fine".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum

from .._c2st import C2STResult
from .._config import C2STConfig

__all__ = [
    "C2STAnnotation",
    "C2STOutcome",
    "C2STProvenance",
    "C2STReport",
    "C2STScore",
    "C2STVerdict",
    "ColumnC2STResult",
]


class C2STOutcome(StrEnum):
    """What happened to one column — the closed seven-member vocabulary.

    A further member is a breaking change: reading the report *is* comparing
    against it (``rec.outcome is C2STOutcome.Tested``).

    ``Tested`` is the only member carrying a :class:`C2STScore`; the other six
    are refusals, each naming why the test could not produce an honest number.
    ``Unfilled`` is the column an imputed frame still carries nulls in where
    the original was missing — a declared MNAR column, which is left unfilled
    by design (ADR-0098), or any column the imputation passed through. The
    null alone would separate the piles, so a ``z`` there would score the
    null, not a fill.
    """

    Tested = "tested"
    NoFilledCells = "no_filled_cells"
    NoObservedCells = "no_observed_cells"
    Unfilled = "unfilled"
    TypeNotTestable = "type_not_testable"
    BelowSampleFloor = "below_sample_floor"
    Uninformative = "uninformative"


class C2STAnnotation(StrEnum):
    """A fact riding on a result the test actually ran — never a refusal.

    Annotations do not suppress the ``z``; they qualify it. A column carrying
    ``LowPower`` was tested honestly and simply had too little data for the
    answer to mean much, which is why a pass there reads "not enough data to
    tell" rather than "clean".

    Two members are column-level facts read off the pile sizes
    (:attr:`LowPower`, :attr:`ImbalancedPiles`) and two are rolled up from the
    per-frame :class:`~dataforge_ml.evaluation.C2STResult` flags
    (:attr:`Degenerate`, :attr:`UnseenCategory`). Unlike
    :class:`C2STOutcome` this vocabulary is open: a further outcome is a
    breaking change, a further annotation is additive.

    Attributes
    ----------
    LowPower
        The balanced pile size is under ``C2STConfig.low_power_below``.
    ImbalancedPiles
        ``min(n_observed, n_filled) / max(...)`` is under
        ``C2STConfig.imbalance_warn_ratio``, so the smaller pile caps the
        effective sample size.
    Degenerate
        At least one — but not every — frame's classifier predicted a single
        class in every repeat. Every frame degenerating is the ``Uninformative``
        outcome instead, since there is then no honest number left to report.
    UnseenCategory
        Some ``Categorical``/``Boolean`` level appears among the filled rows and
        never among the observed ones. Computed as a set difference before the
        test; the ``z`` is reported regardless, because inventing an unseen
        *category* is always wrong whereas an unseen *numeric* value is
        legitimate interpolation.
    """

    LowPower = "low_power"
    ImbalancedPiles = "imbalanced_piles"
    Degenerate = "degenerate"
    UnseenCategory = "unseen_category"


class C2STVerdict(StrEnum):
    """Whether a human should go look at this column — three-valued, never a bool.

    Stamped onto the record by :class:`C2STReport` after Benjamini-Hochberg
    runs, because BH is a property of the *set* of tested columns and no single
    record can compute it. The third member is the point: a column the test
    could not run on has ``NoVerdict``, where ``False`` would read as "checked,
    fine".

    A consequence worth stating: a :class:`ColumnC2STResult` lifted out of its
    report carries a verdict whose meaning depends on the set it came from.

    Attributes
    ----------
    Flagged
        The BH-adjusted p-value is at or below ``C2STConfig.fdr_alpha``.
    NotFlagged
        The column was tested and its adjusted p-value is above the level.
    NoVerdict
        The column was never tested, so there is nothing to correct and nothing
        to conclude.
    """

    Flagged = "flagged"
    NotFlagged = "not_flagged"
    NoVerdict = "no_verdict"


@dataclass(frozen=True)
class C2STScore:
    """Every number that exists only when the test actually ran.

    Nullability is the whole point: a refused column holds ``score = None``, so
    a refusal can never arithmetic-collapse into a passing figure. A refused
    column and a chance-scoring column differ by *having or not having this
    object*.

    The headline is :attr:`mean_frame_z` — the mean ``z`` over the ``m`` frames
    **against the single-run null**. Dividing that null's SD by ``√m`` is
    forbidden (ADR-0087): the frames share pile A entirely, so shrinking the
    null manufactures significance. No field at any level is named plain ``z``.

    Attributes
    ----------
    mean_frame_z : float
        The mean of the per-frame :attr:`C2STResult.mean_repeat_z`, read
        against the single-run null. The score.
    frame_z_spread : float
        The sample standard deviation of the per-frame ``z`` (``0.0`` at
        ``m = 1``).
    p_value : float
        The one-sided, **upper-tail** normal p-value of :attr:`mean_frame_z`.
        Present but never the score: a p-value is a function of the
        discrepancy *and* of ``n``, so it floors out on large tables and
        changes meaning across datasets. Below-chance accuracy therefore reads
        as noise rather than as evidence of excellent imputation.
    p_adjusted : float, optional
        The Benjamini–Hochberg adjusted p-value, stamped on by the report after
        correction runs across the tested set. ``None`` until it does.
    frames : tuple[C2STResult, ...]
        One generic result per imputed frame, in the order the frames were
        supplied.
    """

    mean_frame_z: float
    frame_z_spread: float = 0.0
    p_value: float = 1.0
    p_adjusted: float | None = None
    frames: tuple[C2STResult, ...] = ()


@dataclass(frozen=True)
class ColumnC2STResult:
    """One active column's result — an outcome first, a number only sometimes.

    The two pile sizes are stated whatever the outcome, because they are the
    facts that explain a refusal.

    Attributes
    ----------
    column : str
        The column the record describes.
    outcome : C2STOutcome
        What happened. Only ``Tested`` carries a :attr:`score`.
    annotations : tuple[C2STAnnotation, ...]
        Facts qualifying a result the test actually ran, in a fixed order.
        Empty on a column refused before the test ever ran, since there is
        then no number for them to qualify. They never suppress the ``z``.
    verdict : C2STVerdict
        Whether a human should go look at this column, stamped on by
        :class:`C2STReport` once Benjamini-Hochberg has run across the tested
        set. ``NoVerdict`` on every refusal — and on a record that has not been
        through a report at all.
    n_observed : int
        Pile A's raw size — the rows where the column was observed, before
        balancing.
    n_filled : int
        Pile B's raw size — the rows where the column was filled, before
        balancing. Rows carrying a declared sentinel are here, not in pile A.
    score : C2STScore, optional
        The numbers, present only on a ``Tested`` outcome. ``None`` on every
        refusal, so ``result.score.mean_frame_z`` raises ``AttributeError``
        rather than reading a stand-in.
    """

    column: str
    outcome: C2STOutcome
    n_observed: int
    n_filled: int
    score: C2STScore | None = None
    annotations: tuple[C2STAnnotation, ...] = ()
    verdict: C2STVerdict = C2STVerdict.NoVerdict


@dataclass(frozen=True)
class C2STProvenance:
    """What ran, so a number can still be argued with six months later.

    One object per call: a single evaluation uses one classifier and one
    resolved :class:`~dataforge_ml.evaluation.C2STConfig` for every column, so
    it hangs off the report rather than off each record.

    Attributes
    ----------
    classifier_class : str
        The fully qualified class name of the classifier that ran.
    classifier_params : dict[str, str]
        The classifier's **full** ``get_params()``, values stringified.
        Deliberately not its ``repr``, which elides every parameter left at its
        default — exactly the fact most needed when arguing with a report.
        Stringified because ``get_params()`` can hold live nested estimators.
    sklearn_version : str
        The scikit-learn version that produced the numbers, so a figure that
        changes after an upgrade is explicable.
    config : C2STConfig
        The resolved dials. A report that cannot say what ``min_filled_cells``
        was when it refused a column is a report you cannot argue with.
    """

    classifier_class: str
    classifier_params: dict[str, str] = field(default_factory=dict)
    sklearn_version: str = ""
    config: C2STConfig = field(default_factory=C2STConfig)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return the BH step-up adjusted p-values, in the input's own order.

    ``p_adj_(i) = min over j >= i of  p_(j) * k / j`` over the ascending sort,
    the running minimum being what keeps the adjusted sequence monotone. Each
    value is capped at 1.
    """
    k = len(p_values)
    ascending = sorted(range(k), key=lambda i: p_values[i])
    adjusted = [1.0] * k
    running = 1.0
    for rank, index in enumerate(reversed(ascending)):
        j = k - rank
        running = min(running, p_values[index] * k / j)
        adjusted[index] = min(1.0, running)
    return adjusted


@dataclass(frozen=True)
class C2STReport:
    """The column-keyed C2ST report for one evaluation call.

    Holds one :class:`ColumnC2STResult` per **active** column, in the frame's
    own column order, plus the run's :class:`C2STProvenance`.

    Constructing the report is what runs Benjamini-Hochberg: the tested
    columns' verdicts and adjusted p-values are stamped on in
    :meth:`__post_init__`, so a report never exists in an uncorrected state and
    a record can never be corrected against a set it did not come from.

    Attributes
    ----------
    columns : dict[str, ColumnC2STResult]
        The census: every active column, keyed by name.
    provenance : C2STProvenance
        What ran and under which dials.
    """

    columns: dict[str, ColumnC2STResult] = field(default_factory=dict)
    provenance: C2STProvenance = field(
        default_factory=lambda: C2STProvenance(classifier_class="")
    )

    def __post_init__(self) -> None:
        """Run Benjamini-Hochberg across the tested columns and stamp the records.

        The correction lives here, and not in the adapter's per-column loop,
        because BH is a property of the **set**: the adjusted p-value of one
        column is a function of every other column's. Assembling the report is
        therefore the only moment it can be computed, and doing it in
        ``__post_init__`` means no ``C2STReport`` can exist in an uncorrected
        state.

        Only :attr:`ColumnC2STResult.score` and
        :attr:`ColumnC2STResult.verdict` are rewritten. The raw
        ``mean_frame_z`` is left exactly as measured: BH moves the shortlist,
        never the score.
        """
        tested = [
            name
            for name, record in self.columns.items()
            if record.score is not None
        ]
        if not tested:
            return
        alpha = self.provenance.config.fdr_alpha
        adjusted = _benjamini_hochberg(
            [self.columns[name].score.p_value for name in tested]
        )
        stamped = dict(self.columns)
        for name, p_adjusted in zip(tested, adjusted):
            record = self.columns[name]
            stamped[name] = replace(
                record,
                score=replace(record.score, p_adjusted=p_adjusted),
                verdict=(
                    C2STVerdict.Flagged
                    if p_adjusted <= alpha
                    else C2STVerdict.NotFlagged
                ),
            )
        object.__setattr__(self, "columns", stamped)

    def __getitem__(self, column: str) -> ColumnC2STResult:
        """Return one column's record.

        Parameters
        ----------
        column : str
            The column name.

        Returns
        -------
        ColumnC2STResult
            The record for that column.

        Raises
        ------
        KeyError
            When the column was not evaluated — a column excluded by the
            Phase Active-Columns Contract is not in the census at all, which is
            what keeps exclusion and refusal distinguishable.
        """
        return self.columns[column]

    def __contains__(self, column: object) -> bool:
        """Report whether a column has a record.

        Parameters
        ----------
        column : object
            The column name to look for.

        Returns
        -------
        bool
            ``True`` when the column was evaluated; ``False`` when it was
            excluded.
        """
        return column in self.columns

    def __iter__(self) -> Iterator[str]:
        """Iterate the evaluated column names in report order.

        Returns
        -------
        Iterator[str]
            The census's column names.
        """
        return iter(self.columns)

    def __len__(self) -> int:
        """Return the number of evaluated columns.

        Returns
        -------
        int
            The census size.
        """
        return len(self.columns)
