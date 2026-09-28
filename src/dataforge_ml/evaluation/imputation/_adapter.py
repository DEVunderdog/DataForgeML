"""The imputation adapter — the first consumer of the generic C2ST seam.

Everything phase-shaped lives here, exhaustively (ADR-0087): resolving
``PipelinePhase.Imputation``'s active columns under the Phase Active-Columns
Contract, deriving the mask through effective-null resolution, the
``Text``/``Identifier`` gate, column choice and order, dropping 100%-null
columns from the feature matrix, building the two piles, the ``m`` loop with
its frame pooling, and the refusal floors and annotations. Benjamini-Hochberg
is adapter-owned too, but runs one level along in :class:`C2STReport`, since it
is a property of the assembled *set* rather than of any one column. Everything
not on that list is generic and lives in :mod:`dataforge_ml.evaluation._c2st`.

**Column-conditional construction.** For a column ``c``, pile A is the rows
where ``c`` was observed and pile B the rows where ``c`` was filled. The
classifier sees **whole rows** of the imputed table and ``c`` is itself among
the features, so a median fill's one-value spike is visible rather than hidden
and an imputer that matches the marginal while dissolving the dependence
structure is still caught.

**No pool anywhere.** ``HistGradientBoostingClassifier`` has no ``n_jobs`` and
already saturates the box over OpenMP, so the column, frame and repeat loops
are plain sequential ``for`` loops; determinism comes free and the library owns
zero pools.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import sklearn

from ...config import PipelineConfig, PipelinePhase, SemanticType
from ...observability import Emitter, Observer
from ...profiling._config import StructuralProfileResult
from ...utils._dtype_floor import _apply_dtype_floor
from ...utils._null_normalization import _resolve_effective_nulls
from .._c2st import (
    C2STResult,
    _c2st,
    _default_c2st_classifier,
    _meets_sample_floor,
)
from .._config import C2STConfig
from ._records import (
    C2STAnnotation,
    C2STOutcome,
    C2STProvenance,
    C2STReport,
    C2STScore,
    ColumnC2STResult,
)

__all__ = ["imputation_score_c2st"]

# The Pipeline Event ``phase`` this module stamps. A bare string with no
# ``PipelinePhase`` member behind it: that enum's job is feeding
# ``resolve_active_columns`` for the six sequential phases, and evaluation is
# opt-in, out of band, and resolves its columns against *the imputation phase's*
# set (ADR-0087).
_PHASE: str = "evaluation"

# The stage, named after the function that owns the loop, so an observer can
# tell it from a direct c2st call and later metrics land as sibling stages
# under the same phase (ADR-0087).
_STAGE: str = "imputation_score_c2st"

# The semantic types that are out as features and as targets alike. Text
# imputation therefore remains unevaluated by anything (ADR-0087).
_UNTESTABLE_TYPES: frozenset[SemanticType] = frozenset(
    {SemanticType.Text, SemanticType.Identifier}
)


def _upper_tail_p(z: float) -> float:
    """Return the one-sided, upper-tail normal p-value of ``z``."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _phase_entry(
    df: pl.DataFrame,
    profile: StructuralProfileResult,
) -> pl.DataFrame:
    """Normalise one frame at evaluation's phase entry.

    Effective nulls first, then the Dtype Floor (ADR-0085) — the order is
    fixed, because the floor casts away the string namespace the sentinel
    rules need. The caller's frame is never mutated.
    """
    return _apply_dtype_floor(
        _resolve_effective_nulls(
            df,
            numeric_sentinels=profile.numeric_sentinels,
            string_sentinels=profile.string_sentinels,
        ),
        {name: cp.semantic_type for name, cp in profile.columns.items()},
    )


def _resolve_classifier(config: C2STConfig) -> tuple[Any, bool]:
    """Return the classifier for the run and whether the user supplied it."""
    if config.classifier is not None:
        return config.classifier, True
    return (
        _default_c2st_classifier(
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        ),
        False,
    )


def _provenance(classifier: Any, injected: bool, config: C2STConfig) -> C2STProvenance:
    """Record what ran, from ``get_params()`` rather than from ``repr``."""
    get_params = getattr(classifier, "get_params", None)
    params = get_params() if callable(get_params) else {}
    return C2STProvenance(
        classifier_class=(
            f"{type(classifier).__module__}.{type(classifier).__qualname__}"
        ),
        classifier_params={k: str(v) for k, v in params.items()},
        classifier_injected=injected,
        sklearn_version=sklearn.__version__,
        config=config,
    )


def _pool_frames(frames: Sequence[C2STResult]) -> C2STScore:
    """Pool the ``m`` frames by the **mean ``z`` against the single-run null**.

    The same rule the repeat axis uses. Dividing the null's SD by ``√m`` is
    forbidden (ADR-0087): the frames share pile A entirely, so shrinking the
    null manufactures significance.
    """
    per_frame = [frame.mean_repeat_z for frame in frames]
    mean_frame_z = float(np.mean(per_frame))
    return C2STScore(
        mean_frame_z=mean_frame_z,
        frame_z_spread=(
            float(np.std(per_frame, ddof=1)) if len(per_frame) > 1 else 0.0
        ),
        p_value=_upper_tail_p(mean_frame_z),
        frames=tuple(frames),
    )


def _refusal(
    semantic_type: SemanticType | None,
    n_observed: int,
    n_filled: int,
) -> C2STOutcome | None:
    """Return the outcome refusing this column on the facts known before piling."""
    if semantic_type in _UNTESTABLE_TYPES:
        return C2STOutcome.TypeNotTestable
    if n_filled == 0:
        return C2STOutcome.NoFilledCells
    if n_observed == 0:
        return C2STOutcome.NoObservedCells
    return None


def _annotations(
    results: Sequence[C2STResult],
    n_observed: int,
    n_filled: int,
    config: C2STConfig,
) -> tuple[C2STAnnotation, ...]:
    """Qualify a result the test actually ran — never suppress its ``z``.

    Two of the four are column-level facts read off the raw pile sizes, and
    two are rolled up from the per-frame flags. ``Degenerate`` appears only
    when *some* frame degenerated: every frame degenerating leaves no honest
    number at all and is the ``Uninformative`` outcome instead.
    """
    smaller, larger = min(n_observed, n_filled), max(n_observed, n_filled)
    found: list[C2STAnnotation] = []
    if smaller < config.low_power_below:
        found.append(C2STAnnotation.LowPower)
    if larger and smaller / larger < config.imbalance_warn_ratio:
        found.append(C2STAnnotation.ImbalancedPiles)
    if any(result.degenerate for result in results):
        found.append(C2STAnnotation.Degenerate)
    if any(result.unseen_category for result in results):
        found.append(C2STAnnotation.UnseenCategory)
    return tuple(found)


def imputation_score_c2st(
    original: pl.DataFrame,
    imputed_frames: Sequence[pl.DataFrame],
    profile: StructuralProfileResult,
    *,
    pipeline_config: PipelineConfig,
    config: C2STConfig | None = None,
    observer: Observer | None = None,
) -> C2STReport:
    """Score an imputed table against the frame it was imputed from.

    The imputation C2ST entry point: stateless, opt-in, and free to be
    re-run with different dials while poking at results. It grades **the
    data**, never the imputer (ADR-0087) — the model that produced the table is
    disposable, and the question is whether the table still carries the
    distribution and dependence structure of the data behind it.

    For every active column, the Classifier Two-Sample Test trains a classifier
    to tell the rows where that column was *observed* from the rows where it
    was *filled*, and reports its held-out accuracy as a standardised ``z``.
    Every active column gets a record, always: *untestable is a result, never
    an absence*. A column the test could not run on carries a
    ``C2STOutcome`` naming why and no ``score`` object at all, so
    ``record.score.mean_frame_z`` raises ``AttributeError`` rather than
    reading a stand-in that could collapse into a passing mean.

    Once every column has run, Benjamini-Hochberg turns the tested columns'
    p-values into three-valued ``C2STVerdict``s controlling the false
    discovery rate across the report. The raw ``z`` is never corrected: BH
    moves the shortlist, not the score.

    Expect scalar strategies (Mean/Median/Mode/Constant) to be caught at close
    to 100% — that is the **baseline** which makes a model-based number
    interpretable, not forty defects.

    A declared MNAR column comes back ``Unfilled``: the imputer leaves its
    nulls in place by design (ADR-0098), and a column still null where the
    original was missing has no fill to score.

    Parameters
    ----------
    original : polars.DataFrame
        The **pre-imputation** frame. It is what identifies the filled cells:
        the mask is derived from it through effective-null resolution, so rows
        carrying a declared sentinel land in the filled pile where they belong.
    imputed_frames : Sequence[polars.DataFrame]
        The imputed table(s), row-aligned with ``original``. A sequence from
        day one — length 1 today — so calling code never has to change when
        multiple imputation arrives. The ``m`` frames are pooled by the mean
        ``z`` against the single-run null; the null is never shrunk by ``√m``.
    profile : StructuralProfileResult
        The profile the imputation was routed on. Needed, not optional: the mask
        needs its sentinel maps, the type gate needs its ``SemanticType`` values,
        and the Dtype Floor is driven off it. A table handed over cold, without
        its profile, cannot be evaluated. There is deliberately no row-count
        check against a val slice, because semantic types re-inferred on a small
        slice can flip and silently change the tested column set.
    pipeline_config : PipelineConfig
        The run's config, read only for the Phase Active-Columns Contract:
        evaluation resolves its own column set against
        ``PipelinePhase.Imputation``. Excluded columns are honoured on
        governance grounds — an excluded column is absent from the census
        entirely, so ``report[col]`` raises ``KeyError``.
    config : C2STConfig, optional
        Dials for the Classifier Two-Sample Test. ``None``, the default, builds
        a default ``C2STConfig()``.
    observer : Callable[[PipelineEvent], None], optional
        A Progress Observer receiving the event stream. It rides the call and
        is never a config field, because a live callable would break the
        config's serialisable contract. Events are progress only:
        ``phase="evaluation"``, ``stage="imputation_score_c2st"``, one ``item``
        per active column — refused ones included, so the bar reaches its own
        total — and one ``substep`` per repeat.

    Returns
    -------
    C2STReport
        The report holding the per-column C2ST census and the run's provenance.

    Raises
    ------
    ValueError
        When ``imputed_frames`` is empty, when a frame's row count differs from
        ``original``'s, or when an active column is missing from an imputed
        frame.
    """
    if not imputed_frames:
        raise ValueError(
            "imputation_score_c2st needs at least one imputed frame; "
            "imputed_frames was empty."
        )
    for index, frame in enumerate(imputed_frames):
        if frame.height != original.height:
            raise ValueError(
                "Each imputed frame must be row-aligned with original: "
                f"frame {index} has {frame.height} rows against "
                f"{original.height}."
            )

    if config is None:
        config = C2STConfig()

    orig = _phase_entry(original, profile)
    frames = [_phase_entry(frame, profile) for frame in imputed_frames]

    active = pipeline_config.resolve_active_columns(
        PipelinePhase.Imputation, list(orig.columns)
    )
    missing = sorted({c for c in active for frame in frames if c not in frame.columns})
    if missing:
        names = ", ".join(f"'{c}'" for c in missing)
        raise ValueError(
            "Every active column must be present in each imputed frame; "
            f"missing from at least one frame: {names}. Exclude a column the "
            "imputation dropped via PipelineConfig rather than handing over a "
            "frame that has lost it."
        )

    semantic_types = {name: cp.semantic_type for name, cp in profile.columns.items()}
    # Columns under test and columns used as features are the same set, minus
    # the endpoint case: a 100%-null column is pure imputer output with no
    # observed anchor and is dropped as a feature. The rule is specifically
    # 100%-null and *not* "untestable" — a column with no filled cells is
    # entirely observed data and is the most trustworthy feature available, and
    # a 60%-missing column still carries 40% real data.
    testable = [c for c in active if semantic_types.get(c) not in _UNTESTABLE_TYPES]
    features = [
        c for c in testable if orig.get_column(c).null_count() < orig.height
    ]

    classifier, injected = _resolve_classifier(config)
    emitter = Emitter(_PHASE, _STAGE, observer, total=len(active))
    emitter.stage_start()

    records: dict[str, ColumnC2STResult] = {}
    for column in active:
        # A refused column emits its ``item`` and nothing else: ``total`` counts
        # the active columns, so a silent refusal would leave the bar short of
        # its own total. The reason lives on the record only — the event stream
        # is progress, never decisions.
        emitter.item(column)
        mask = orig.get_column(column).is_null().to_numpy()
        n_filled = int(mask.sum())
        n_observed = int(orig.height - n_filled)
        outcome = _refusal(semantic_types.get(column), n_observed, n_filled)
        # A frame still null where the original was missing has no fill to
        # score: the null alone separates the piles. Refused structurally so a
        # declared MNAR column, left unfilled by design (ADR-0098), is not
        # reported as a failed imputation.
        if outcome is None and any(
            frame.get_column(column).filter(mask).null_count() > 0
            for frame in frames
        ):
            outcome = C2STOutcome.Unfilled
        piles = (
            []
            if outcome is not None
            else [
                (
                    frame.select(features).filter(~mask),
                    frame.select(features).filter(mask),
                )
                for frame in frames
            ]
        )
        # The floor is a pure predicate the adapter asks first, because on a
        # wide table refusal is the *common* path and raising 190 exceptions to
        # describe the ordinary case is control flow standing on its head. It
        # is applied to the smaller pile before balancing, since every measured
        # number is in rows per pile after it.
        if outcome is None and not _meets_sample_floor(
            *piles[0], min_rows=config.min_filled_cells
        ):
            outcome = C2STOutcome.BelowSampleFloor
        if outcome is not None:
            records[column] = ColumnC2STResult(
                column=column,
                outcome=outcome,
                n_observed=n_observed,
                n_filled=n_filled,
            )
            continue

        results = [
            _c2st(
                reference,
                candidate,
                repeats=config.repeats,
                scheme=config.scheme,
                classifier=classifier,
                random_state=config.random_state,
                min_rows=config.min_filled_cells,
                emitter=emitter,
            )
            for reference, candidate in piles
        ]
        annotations = _annotations(results, n_observed, n_filled, config)
        # The runtime degeneracy roll-up: every frame's classifier predicting a
        # single class in every repeat scores exactly 0.5 against a balanced
        # test set and would report z = 0.000 — a constant fill returned as a
        # perfect pass. There is no honest number left, so the score object is
        # withheld and the refusal is structural like every other. It is a
        # safety net for an injected classifier and explicitly *not* the floor.
        degenerate = bool(results) and all(result.degenerate for result in results)
        records[column] = ColumnC2STResult(
            column=column,
            outcome=(C2STOutcome.Uninformative if degenerate else C2STOutcome.Tested),
            n_observed=n_observed,
            n_filled=n_filled,
            score=None if degenerate else _pool_frames(results),
            annotations=annotations,
        )

    emitter.stage_end()
    return C2STReport(
        columns=records,
        provenance=_provenance(classifier, injected, config),
    )
