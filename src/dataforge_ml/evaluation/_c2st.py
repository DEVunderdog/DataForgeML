"""The generic Classifier Two-Sample Test — two frames in, one result out.

The arithmetic half of C2ST (ADR-0087), written against two samples and
nothing else. It trains a classifier to tell a ``reference`` sample from a
``candidate`` sample and reads its held-out accuracy as a standardised ``z``
against the closed-form ``Binomial(n_test, 1/2)`` null: if nothing can tell the
two samples apart, they are distributionally indistinguishable.

**What this layer owns, exhaustively:** balancing by repeated subsampling, the
``R`` repeat loop, the size floor (a pure predicate the caller may ask first,
asserted internally so a direct caller cannot bypass it), the default
classifier factory and its two pins, the pooling rule, and the repeat Substep
emit.

**What it does not know:** ``m``, masks, column semantics, imputation. There is
no column-type argument either — **dtype is the whole type contract**. A
``pl.String`` column raises :class:`C2STDtypeError` naming the column, and an
int-coded categorical sitting as ``pl.Int64`` is treated as an ordered numeric,
a stated and unenforceable precondition on the caller's typing.

The one imported dependency beyond the numeric stack is
:mod:`dataforge_ml.observability`, because the repeat Substep can only be
emitted from here (ADR-0087). ``Emitter`` takes ``phase`` and ``stage`` as plain
strings and is phase-agnostic infrastructure, so the seam's rule bans
*imputation*-shaped knowledge, not observability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

from ..observability import Emitter

__all__ = [
    "C2STDtypeError",
    "C2STResult",
    "C2STScheme",
]

# ---------------------------------------------------------------------------
# The three module-level constants the generic side owns.
# ---------------------------------------------------------------------------

# Pin 1 of 2 (ADR-0087). sklearn's default of 20 cannot split fewer than 40
# training rows, so the classifier constant-predicts, scores exactly 0.5
# against a balanced test set and reports z = 0.000 — a constant fill returned
# as a perfect pass. This constant is also ``C2STConfig.min_samples_leaf``'s
# field default, so the configured value and the value a bare
# ``c2st(classifier=None)`` builds can never drift apart.
DEFAULT_MIN_SAMPLES_LEAF: int = 5

# The hard refusal floor, applied to the *smaller* sample before balancing.
# Also ``C2STConfig.min_filled_cells``'s field default. Measured: the one-sided
# false-positive rate is 9.0% at 20 rows per pile and 4.5% at 30.
MIN_SAMPLE_FLOOR: int = 30

# The train/test proportion of the ``Split`` scheme, and the split
# ``early_stopping='auto'`` was measured against in ADR-0087. Not a dial: both
# schemes exist to keep ``n_test`` exactly known, and a third way of carving the
# piles would only make two reports incomparable.
_TEST_FRACTION: float = 0.3

# The fold count of the ``CrossValidation`` scheme — five fits per repeat.
_CV_FOLDS: int = 5


class C2STDtypeError(TypeError):
    """A column's dtype cannot be handed to a C2ST classifier.

    Raised by :func:`c2st` when a sample carries a ``pl.String`` column, which
    :class:`~sklearn.ensemble.HistGradientBoostingClassifier` rejects with a
    bare ``could not convert string to float`` naming a *value* rather than a
    column. Under ADR-0085's dtype floor no semantic categorical arrives as a
    string, and ``Text``/``Identifier`` columns are dropped by the imputation
    adapter before it ever calls here, so this is a diagnosis for a direct
    caller of the generic seam rather than a path the adapter can trip.
    """


class C2STScheme(StrEnum):
    """The train/test scheme of one C2ST repeat — a dial; balancing is not.

    ``Split`` fits once per repeat on a 70/30 stratified carve of the balanced
    piles; ``CrossValidation`` fits five times and pools the out-of-fold
    predictions. Both keep ``n_test`` exactly known, which is what the
    closed-form null requires. ``Split`` is the default a config declares:
    repeats are already mandatory, so a default that silently costs 5× is a bad
    default. The repeat count is never auto-downgraded when cross-validation is
    on.
    """

    Split = "split"
    CrossValidation = "cross_validation"


def _md_cell(value: Any) -> str:
    """Render one value as a Markdown table cell.

    Absence is stated rather than left as a bare ``None``, enums render by their
    string form, and any pipe or newline is neutralised so it cannot break the
    surrounding table.
    """
    if value is None:
        return "none"
    if isinstance(value, Enum):
        text = str(value)
    elif isinstance(value, (list, tuple)):
        text = ", ".join(_md_cell(v) for v in value) if value else "none"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


@dataclass(frozen=True)
class C2STResult:
    """What the generic test returns for one pair of samples.

    Every number here is pooled over the ``R`` repeats of a single ``c2st``
    call. The headline is :attr:`mean_repeat_z` — the **mean ``z`` against the
    single-run null**. Dividing the null's SD by ``√R`` is forbidden (ADR-0087):
    the repeats share the candidate sample entirely, so shrinking the null
    manufactures significance. No field is ever named plain ``z``; that is the
    field someone averages along the wrong axis.

    Attributes
    ----------
    mean_repeat_z : float
        The mean of :attr:`repeat_z`, read against the single-run null.
    repeat_z : tuple[float, ...]
        One ``z`` per repeat, in repeat order.
    repeat_z_spread : float
        The sample standard deviation of :attr:`repeat_z` (``0.0`` for a single
        repeat) — the guard that says whether the mean is stable or the artefact
        of one lucky draw.
    train_accuracy : float
        Mean training accuracy over the repeats. Reported beside
        :attr:`test_accuracy` so a 1.000/0.500 pair reads as memorisation —
        which fails *safe*, yielding false negatives, never false positives.
    test_accuracy : float
        Mean held-out accuracy over the repeats; the quantity the ``z``
        standardises.
    n_test : int
        The held-out row count of one repeat, exactly known and identical across
        repeats: ``2 × round(m × 0.3)`` under ``Split`` and ``2 × m`` under
        ``CrossValidation``, where ``m`` is the balanced pile size.
    degenerate : bool
        ``True`` when *every* repeat's classifier predicted a single class — a
        runtime safety net for an injected classifier, explicitly not the size
        floor.
    unseen_category : bool
        ``True`` when some ``pl.Categorical``/``pl.Boolean`` level appears in
        the candidate sample and never in the reference sample. Computed as a
        set difference before the test; the ``z`` is reported regardless.
    """

    mean_repeat_z: float
    repeat_z: tuple[float, ...] = field(default_factory=tuple)
    repeat_z_spread: float = 0.0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    n_test: int = 0
    degenerate: bool = False
    unseen_category: bool = False

    def to_markdown(self) -> str:
        """Render the result as a ``###``-rooted Markdown fragment.

        A fragment per rule 5 of the Rendering Contract (ADR-0086): it carries
        no ``#`` or ``##`` heading, so an owning document composes it without a
        heading collision.

        Returns
        -------
        str
            Markdown subsection headed by ``### C2ST Sample-Pair Result``,
            holding the pooled figures and the two runtime flags.
        """
        return "\n".join(
            [
                "### C2ST Sample-Pair Result\n",
                "| Field | Value |",
                "|---|---|",
                f"| mean_repeat_z | {self.mean_repeat_z:.4f} |",
                f"| repeat_z_spread | {self.repeat_z_spread:.4f} |",
                f"| repeats | {len(self.repeat_z)} |",
                "| repeat_z | " + _md_cell([f"{z:.4f}" for z in self.repeat_z]) + " |",
                f"| train_accuracy | {self.train_accuracy:.4f} |",
                f"| test_accuracy | {self.test_accuracy:.4f} |",
                f"| n_test | {self.n_test} |",
                f"| degenerate | {_md_cell(self.degenerate)} |",
                f"| unseen_category | {_md_cell(self.unseen_category)} |",
            ]
        )

    def __str__(self) -> str:
        """Return the fragment, per rule 2 of the Rendering Contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


def _default_c2st_classifier(
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    random_state: int | None = None,
) -> Any:
    """Build the library's default C2ST classifier, pins included.

    The single place a default classifier is constructed, so
    ``c2st(classifier=None)`` can never build a bare sklearn-default
    ``HistGradientBoostingClassifier``. Exactly two settings are pinned, under
    the rule *pin only where sklearn's default encodes a dataset-size assumption
    balanced C2ST piles violate* (ADR-0087): ``min_samples_leaf`` reads
    :data:`DEFAULT_MIN_SAMPLES_LEAF`, and ``early_stopping`` is ``False`` rather
    than the ``'auto'`` that flips at 10,000 training rows. Everything else —
    ``max_iter``, ``learning_rate``, ``max_leaf_nodes``, ``l2_regularization``,
    ``class_weight`` — is inherited; tuning those is what the injection channel
    is for.

    Parameters
    ----------
    min_samples_leaf : int, optional
        The leaf-size floor to pin; defaults to :data:`DEFAULT_MIN_SAMPLES_LEAF`,
        which is also ``C2STConfig.min_samples_leaf``'s field default, so the
        configured value and the value a bare ``c2st(classifier=None)`` builds
        read one number in one place.
    random_state : int, optional
        Seed handed to the estimator.

    Returns
    -------
    Any
        An unfitted ``HistGradientBoostingClassifier``.
    """
    return HistGradientBoostingClassifier(
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        random_state=random_state,
    )


def _meets_sample_floor(
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    min_rows: int = MIN_SAMPLE_FLOOR,
) -> bool:
    """Report whether two samples clear the size floor, without running a test.

    A pure predicate, so a caller describing 190 ordinary refusals on a wide
    table does not have to raise 190 exceptions to learn what it already knows.
    The floor is applied to the **smaller** sample and checked *before*
    balancing, since every number ADR-0087 measured is in rows per pile after
    balancing.

    :func:`c2st` asserts the same condition internally against its own
    ``min_rows``, so a direct caller who skips this predicate is refused rather
    than handed a five-row ``z``; a consumer whose dial has moved the floor
    states that number in both places.

    Parameters
    ----------
    reference : polars.DataFrame
        The reference sample.
    candidate : polars.DataFrame
        The candidate sample.
    min_rows : int, optional
        The floor to test against; defaults to :data:`MIN_SAMPLE_FLOOR`, which
        is also the field default of the configured dial.

    Returns
    -------
    bool
        ``True`` when ``min(reference.height, candidate.height) >= min_rows``.
    """
    return min(reference.height, candidate.height) >= min_rows


def _check_dtypes(frame: pl.DataFrame, side: str) -> None:
    """Raise :class:`C2STDtypeError` for the first ``pl.String`` column."""
    for name, dtype in frame.schema.items():
        if dtype == pl.String:
            raise C2STDtypeError(
                f"C2ST cannot test a pl.String column: {side} column "
                f"{name!r} has dtype {dtype}. Cast it to pl.Categorical, or "
                f"drop it from both samples."
            )


def _align(reference: pl.DataFrame, candidate: pl.DataFrame) -> pl.DataFrame:
    """Stack the two samples into one frame, reference rows first."""
    if list(reference.columns) != list(candidate.columns):
        raise ValueError(
            "C2ST needs the same columns in the same order in both samples: "
            f"reference has {list(reference.columns)}, "
            f"candidate has {list(candidate.columns)}."
        )
    return pl.concat([reference, candidate], how="vertical")


def _has_unseen_category(reference: pl.DataFrame, candidate: pl.DataFrame) -> bool:
    """Report whether a categorical or boolean level is candidate-only."""
    for name, dtype in candidate.schema.items():
        if dtype not in (pl.Categorical, pl.Boolean) and not isinstance(dtype, pl.Enum):
            continue
        seen = set(reference[name].drop_nulls().cast(pl.String).to_list())
        used = set(candidate[name].drop_nulls().cast(pl.String).to_list())
        if used - seen:
            return True
    return False


def _fold_sizes(m: int, folds: int) -> list[int]:
    """Split ``m`` rows into ``folds`` contiguous blocks, largest first."""
    base, remainder = divmod(m, folds)
    return [base + 1] * remainder + [base] * (folds - remainder)


def _fit_predict(
    classifier: Any,
    pooled: pl.DataFrame,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit on the training rows; return the test predictions and train accuracy."""
    train_rows = pooled[train_idx.tolist()]
    classifier.fit(train_rows, labels[train_idx])
    train_accuracy = float((classifier.predict(train_rows) == labels[train_idx]).mean())
    predictions = np.asarray(classifier.predict(pooled[test_idx.tolist()]))
    return predictions, train_accuracy


def c2st(
    reference: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    repeats: int,
    scheme: C2STScheme,
    classifier: Any | None = None,
    random_state: int | None = None,
    min_rows: int = MIN_SAMPLE_FLOOR,
    emitter: Emitter | None = None,
) -> C2STResult:
    """Run a Classifier Two-Sample Test on two samples.

    Two frames in, one result out — never a frame plus a label vector, so a
    mislabelled or misaligned ``y`` is unrepresentable. The larger sample is
    subsampled to the smaller sample's size, ``repeats`` times with different
    draws; each repeat fits a classifier under ``scheme`` and scores its
    held-out accuracy as ``z = (accuracy − ½) · 2 · √n_test`` against the
    ``Binomial(n_test, ½)`` null. **Balancing is mandatory and not a dial**:
    unequal samples break that null outright, since at 850 against 150 a
    classifier that always answers "reference" scores 85% having learned
    nothing.

    The repeats are pooled by the **mean ``z`` against the single-run null**;
    the null's SD is never divided by ``√repeats`` (ADR-0087).

    The samples must carry the same columns in the same order, and dtype is the
    whole type contract: ``pl.Categorical`` is read natively, nulls are handled
    natively, and ``pl.String`` raises. An int-coded categorical sitting as
    ``pl.Int64`` is treated as an ordered numeric — a stated and unenforceable
    precondition on the caller's typing.

    Nothing here knows about masks, ``m``, column semantics or imputation. A
    train/test drift check (reference = training rows, candidate = holdout) and
    an outlier-clipping check (reference = rows left alone, candidate = rows
    clipped) are served by this call as-is.

    Parameters
    ----------
    reference : polars.DataFrame
        The reference sample. Which sample is which is a naming convenience:
        accuracy does not care.
    candidate : polars.DataFrame
        The candidate sample, carrying the same columns in the same order.
    repeats : int
        The number of balanced subsampling repeats, ``R``. At least 1.
    scheme : C2STScheme
        ``Split`` for one fit per repeat, ``CrossValidation`` for five.
    classifier : Any, optional
        An unfitted classifier to use **verbatim** — no ``set_params``, no
        clone, so its parameters are exactly what the caller set. ``None``
        builds the library's default through
        :func:`_default_c2st_classifier`. Stated hole (ADR-0087): an injected
        *bare* ``HistGradientBoostingClassifier`` reintroduces the
        constant-predictor pathology at 30–60 rows per pile.
    random_state : int, optional
        Seed for the subsampling, the train/test carve and the default
        classifier. The same seed on the same samples gives the same result.
    min_rows : int, optional
        The size floor asserted internally, applied to the **smaller** sample
        before balancing; defaults to :data:`MIN_SAMPLE_FLOOR`. Mechanism
        rather than configuration, and the same number
        :func:`_meets_sample_floor` takes, so a consumer whose own dial has
        moved the floor states it once and the assertion cannot then refuse
        what the consumer deliberately allowed.
    emitter : Emitter, optional
        Observability sink; one ``substep`` heartbeat is emitted per repeat.

    Returns
    -------
    C2STResult
        The pooled result of the ``repeats`` repeats.

    Raises
    ------
    C2STDtypeError
        When either sample carries a ``pl.String`` column; the message names the
        column.
    ValueError
        When ``repeats`` is below 1, when the two samples do not carry the same
        columns in the same order, or when the smaller sample falls below
        ``min_rows`` — the internal assertion of the floor
        :func:`_meets_sample_floor` exposes, so a direct caller cannot bypass
        it.
    """
    if repeats < 1:
        raise ValueError(f"C2ST needs at least 1 repeat, got {repeats}.")
    _check_dtypes(reference, "reference")
    _check_dtypes(candidate, "candidate")
    if not _meets_sample_floor(reference, candidate, min_rows=min_rows):
        raise ValueError(
            "C2ST refuses samples below the size floor: the smaller of "
            f"{reference.height} reference and {candidate.height} candidate "
            f"rows is under {min_rows}. Ask _meets_sample_floor "
            "before calling to describe such a pair rather than raise on it."
        )

    pooled = _align(reference, candidate)
    n_reference = reference.height
    candidate_height = candidate.height
    labels = np.concatenate(
        [
            np.zeros(n_reference, dtype=np.int64),
            np.ones(candidate_height, dtype=np.int64),
        ]
    )
    unseen_category = _has_unseen_category(reference, candidate)

    model = (
        _default_c2st_classifier(random_state=random_state)
        if classifier is None
        else classifier
    )
    rng = np.random.default_rng(random_state)
    balanced = min(n_reference, candidate_height)

    repeat_z: list[float] = []
    train_accuracies: list[float] = []
    test_accuracies: list[float] = []
    degenerate_repeats = 0
    n_test = 0

    for repeat in range(1, repeats + 1):
        # Balancing: an independent draw of ``balanced`` rows from each sample.
        drawn_reference = rng.choice(n_reference, balanced, replace=False)
        drawn_candidate = (
            rng.choice(candidate_height, balanced, replace=False) + n_reference
        )
        rng.shuffle(drawn_reference)
        rng.shuffle(drawn_candidate)

        predictions, truth, fold_train_accuracies = _run_repeat(
            model, pooled, labels, drawn_reference, drawn_candidate, scheme
        )
        n_test = predictions.size
        accuracy = float((predictions == truth).mean())
        repeat_z.append((accuracy - 0.5) * 2.0 * float(np.sqrt(n_test)))
        test_accuracies.append(accuracy)
        train_accuracies.append(float(np.mean(fold_train_accuracies)))
        if np.unique(predictions).size == 1:
            degenerate_repeats += 1
        if emitter is not None:
            emitter.substep("c2st repeat", index=repeat, total=repeats)

    return C2STResult(
        mean_repeat_z=float(np.mean(repeat_z)),
        repeat_z=tuple(repeat_z),
        repeat_z_spread=(float(np.std(repeat_z, ddof=1)) if len(repeat_z) > 1 else 0.0),
        train_accuracy=float(np.mean(train_accuracies)),
        test_accuracy=float(np.mean(test_accuracies)),
        n_test=n_test,
        degenerate=degenerate_repeats == repeats,
        unseen_category=unseen_category,
    )


def _run_repeat(
    model: Any,
    pooled: pl.DataFrame,
    labels: np.ndarray,
    drawn_reference: np.ndarray,
    drawn_candidate: np.ndarray,
    scheme: C2STScheme,
) -> tuple[np.ndarray, np.ndarray, Sequence[float]]:
    """Fit one repeat under ``scheme`` and return its held-out predictions.

    The test set is balanced by construction — the same number of rows is taken
    from each sample — which is what makes ``n_test`` exactly known and the
    ``Binomial(n_test, ½)`` null the right one.
    """
    balanced = drawn_reference.size
    if scheme is C2STScheme.Split:
        held = max(1, round(balanced * _TEST_FRACTION))
        test_idx = np.concatenate([drawn_reference[:held], drawn_candidate[:held]])
        train_idx = np.concatenate([drawn_reference[held:], drawn_candidate[held:]])
        predictions, train_accuracy = _fit_predict(
            model,
            pooled,
            labels,
            train_idx,
            test_idx,
        )
        return predictions, labels[test_idx], [train_accuracy]

    predictions: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    train_accuracies: list[float] = []
    start = 0
    for size in _fold_sizes(balanced, _CV_FOLDS):
        stop = start + size
        fold = np.concatenate(
            [drawn_reference[start:stop], drawn_candidate[start:stop]]
        )
        rest = np.concatenate(
            [
                drawn_reference[:start],
                drawn_reference[stop:],
                drawn_candidate[:start],
                drawn_candidate[stop:],
            ]
        )
        fold_predictions, train_accuracy = _fit_predict(
            model, pooled, labels, rest, fold
        )
        predictions.append(fold_predictions)
        truth.append(labels[fold])
        train_accuracies.append(train_accuracy)
        start = stop
    return np.concatenate(predictions), np.concatenate(truth), train_accuracies
