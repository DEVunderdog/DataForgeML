"""
FittedImputer — stateless object assembled by FittedImputer.compose().

transform(df) applies train-time fill parameters and fitted models to any
DataFrame. The imputer has no aggregate serialize format of its own (ADR-0072):
a whole imputer is persisted as its decision plus its fitted units — each a
single ``dataforge_ml.serialize`` blob — and rehydrated through
:meth:`FittedImputer.compose`, whose exact-coverage check guarantees the
reconstructed imputer is structurally complete.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from ..config import PipelineConfig, PipelinePhase
from ..models._data_types import _FLOAT_DTYPES, _INT_DTYPES
from ..utils._dtype_floor import _apply_dtype_floor
from ..utils._null_normalization import _resolve_effective_nulls
from ._config import (
    ColumnImputationRecord,
    ImputationResult,
    ImputationStrategy,
)
from ._fitted_units import FittedScalar


def _normalize_results(decision: Any, results: Any) -> dict[str, Any]:
    """Reduce ``compose``'s tolerant input to a ``{unit_id: fitted_unit}`` map.

    Accepts a mapping or an iterable, and within it either raw fitted units or
    ``UnitFitResult`` bundles. A ``UnitFitResult`` names its own unit id; a raw
    fitted unit is matched to its plan unit by the columns it owns.
    """
    by_columns = {frozenset(u.columns): u.unit_id for u in decision.units}

    def _resolve(unit_id: Optional[str], item: Any) -> tuple[str, Any]:
        fitted = getattr(item, "fitted", None)
        if fitted is not None and hasattr(item, "unit_id"):
            # A UnitFitResult bundle.
            return item.unit_id, fitted
        # A raw fitted unit: trust an explicit key, else match by owned columns.
        if unit_id is not None:
            return unit_id, item
        key = frozenset(item.target_columns)
        matched = by_columns.get(key)
        if matched is None:
            raise ValueError(
                f"A supplied fitted unit owns columns {sorted(item.target_columns)}, "
                f"which match no unit in the plan; it cannot be composed."
            )
        return matched, item

    normalized: dict[str, Any] = {}
    if hasattr(results, "items"):
        for key, item in results.items():
            unit_id, fitted = _resolve(key, item)
            normalized[unit_id] = fitted
    else:
        for item in results:
            unit_id, fitted = _resolve(None, item)
            normalized[unit_id] = fitted
    return normalized


class DroppedColumnAbsentWarning(UserWarning):
    """
    Emitted by FittedImputer.transform() when a column recorded as
    ``ImputationStrategy.Dropped`` during fit() is already absent from the
    input DataFrame.

    This typically means the caller pre-removed the column before calling
    transform(). Transform continues normally; the warning is the only signal.
    Suppress with
    ``warnings.filterwarnings("ignore", category=DroppedColumnAbsentWarning)``.
    """


class UnseenColumnError(Exception):
    """
    Raised by FittedImputer.transform() when the input DataFrame contains
    columns that were not present in the training DataFrame during fit().

    Fires before any DataFrame mutations regardless of whether the unknown
    columns contain missing values, so schema drift is caught at transform
    entry rather than silently propagating downstream (ADR 0026).

    All unknown column names are reported in a single raise so the caller
    can resolve all schema mismatches at once.
    """


class FittedColumnAbsentError(Exception):
    """
    Raised by FittedImputer.transform() when a column that received an active
    imputation strategy during fit() is absent from the input DataFrame.

    Active strategies are any strategy other than ``Dropped`` or ``Indicator``.
    Absence of such a column is always a pipeline bug — imputation was never
    applied — so this is escalated to an error rather than a warning.

    All absent column names are reported in a single raise.
    """


from typing import Protocol, runtime_checkable


@runtime_checkable
class FittedUnit(Protocol):
    """Standalone fitted imputation unit that can transform its own columns.

    This unit can transform its own columns immediately, with no completeness
    requirement (it carries only the 'my own columns absent' contract).

    A trained unit always executed the strategy the plan asked for: a unit that
    cannot train raises :class:`~dataforge_ml.imputation.UnitNotTrainableError`
    rather than becoming a divergent artifact (single-track failure, ADR-0071).
    The strategy the unit executed is a decide-time fact the plan holds under the
    same ``unit_id``, so a fitted unit carries only its fitted state (ADR-0074)
    and does not restate it; a fit's runtime observability rides on the
    :class:`~dataforge_ml.imputation.FitSignals` the fit returned.

    **Observed-Value Preservation (ADR-0078).** A unit fills holes and never
    edits a value the user supplied: after ``transform``, every cell that was
    not missing in the input comes back bit-for-bit with its original dtype.
    "Missing" is the Effective Null predicate — null, NaN, or infinite for
    float columns; null alone otherwise. The guarantee is a property of the
    unit itself, so it holds when a bare unit is transformed directly with no
    ``FittedImputer`` involved (the stateless door, ADR-0071).

    **On the stateless door, the caller owns sentinel normalisation.** A bare
    unit resolves float ``NaN``/``Inf`` as missing with no configuration, but
    it cannot resolve sentinel-encoded Effective Nulls (e.g. ``-999``): the
    declared sentinel maps live on the ``ImputationDecision`` (ADR-0068), and
    a bare unit does not hold them. Normalise sentinels before calling a bare
    unit's ``transform``, or transform through the composed
    :meth:`FittedImputer.transform`, which normalises off the plan's maps.
    """

    @property
    def target_columns(self) -> list[str]:
        """The columns this unit owns — the ones its ``transform`` may fill.

        A unit's *targets* are not its *inputs*: a regression unit reads every
        feature column but owns only the one column it predicts. Every column is
        routed to exactly one strategy and therefore to exactly one unit, so
        targets are disjoint across units — which is what lets
        :meth:`FittedImputer.transform` merge unit outputs in any order and get
        the same frame (ADR-0070).

        Returns
        -------
        list[str]
            The owned column names.
        """
        ...

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply this unit's imputation to the DataFrame.

        The unit reads whatever features it needs from ``df`` but only its
        :attr:`target_columns` are meaningful in the result; the caller merges
        those and discards the rest (ADR-0070).

        Observed-Value Preservation (ADR-0078): only the cells that were
        missing in ``df`` receive fill values; every observed cell of the
        target columns returns bit-for-bit with its original dtype.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pl.DataFrame
            The transformed DataFrame.
        """
        ...

@dataclass
class FittedMICE:
    """Fitted MICE block.

    A trained MICE block is the happy path by construction — a block that could
    not train raises rather than becoming one of these (single-track failure,
    ADR-0071) — so it carries only its fitted state (ADR-0074).

    The block's inputs and its targets are no longer the same set (ADR-0079):
    it fits one solver over ``all_cols`` — every active numeric column, not just
    the block's own — but writes back only ``columns``, the columns it owns.
    This generalizes the ``[target] + feature_columns`` / ``target_idx`` split
    the former per-column regression unit carried, block-wide.

    Parameters
    ----------
    model : Any
        Fitted ``IterativeImputer`` trained over ``all_cols``.
    columns : list[str]
        The columns this block owns and writes back. A strict subset of
        ``all_cols`` in the widened shape; equal to it when ``all_cols`` is
        left unset.
    all_cols : list[str], optional
        Full column list, in joint-array order, the block reads at fit and
        transform time. Defaults to ``columns`` when omitted — the pre-widening
        shape, where the block's inputs and targets coincided.
    domain_snap_bounds : dict[str, tuple[float, float]]
        Per-owned-column domain-snap bounds, keyed by column name.
    """
    model: Any
    columns: list[str]
    all_cols: Optional[list[str]] = None
    domain_snap_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.all_cols is None:
            self.all_cols = list(self.columns)

    @property
    def target_columns(self) -> list[str]:
        """The block's owned columns — the ones its ``transform`` may fill.

        A strict subset of :attr:`all_cols`, the full set the block reads as
        predictors (ADR-0079): the block is the block-wide generalization of
        the input/target split the former per-column regression unit carried.

        Returns
        -------
        list[str]
            The owned column names.
        """
        return list(self.columns)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill the block's missing cells, preserving observed cells.

        Reads every column in :attr:`all_cols` present in ``df`` as a
        predictor (a column absent from ``df`` contributes an all-missing
        column to the joint solver, same as a column that was never observed),
        but writes back only :attr:`columns` — the block never overwrites a
        column it merely read as a predictor. The joint solver and the
        per-column domain snaps rewrite whole columns; observed-value
        preservation is applied as the final step, so every cell that was not
        an Effective Null in ``df`` comes back bit-for-bit with its original
        dtype (ADR-0078).

        Parameters
        ----------
        df : pl.DataFrame
            Frame to impute. Returned unchanged when none of the block's
            owned columns are present.

        Returns
        -------
        pl.DataFrame
            ``df`` with the block's owned columns' missing cells filled.
        """
        cols = [c for c in self.columns if c in df.columns]
        if not cols:
            return df
        import polars as pl

        from ._utils import _df_to_numpy, _numpy_to_df, _preserve_observed

        n_df_rows = len(df)
        arr = np.full((n_df_rows, len(self.all_cols)), np.nan, dtype=np.float64)
        for j, c in enumerate(self.all_cols):
            if c in df.columns:
                arr[:, j] = _df_to_numpy(df, [c]).ravel()

        arr_filled = self.model.transform(arr)
        owned_idx = [self.all_cols.index(c) for c in cols]
        out_df = _numpy_to_df(df, cols, arr_filled[:, owned_idx])

        snap_exprs = []
        for col in cols:
            if bounds := self.domain_snap_bounds.get(col):
                lo, hi = bounds
                # Deliberately snaps the whole column; _preserve_observed below
                # restores every observed cell bit-for-bit (#400).
                snap_exprs.append(pl.col(col).round(0).clip(lo, hi).alias(col))
        if snap_exprs:
            out_df = out_df.with_columns(snap_exprs)
        return _preserve_observed(df, out_df, cols)

@dataclass
class _FittedKNN:
    """Fitted KNN state including scaling parameters.

    Stores the model together with the
    ``nanmean``/``nanstd`` statistics used to scale the training matrix so
    that ``_apply_knn`` can inverse-scale the imputed output back to original
    units.

    Parameters
    ----------
    model : Any
        Fitted ``KNNImputer`` trained on the NaN-safe scaled training matrix.
    col_means : np.ndarray
        Per-column means computed with ``nanmean`` from the KNN training
        matrix.  Shape ``(n_knn_features,)``.
    col_stds : np.ndarray
        Per-column standard deviations computed with ``nanstd`` from the KNN
        training matrix, with zero values replaced by ``1.0``.
        Shape ``(n_knn_features,)``.
    """

    model: Any
    col_means: np.ndarray
    col_stds: np.ndarray
    columns: list[str] = field(default_factory=list)
    domain_snap_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def target_columns(self) -> list[str]:
        """The block's columns, which are both its inputs and its targets.

        The block trains one solver over exactly these columns and fills all of
        them, so it reads no column it does not own.

        Returns
        -------
        list[str]
            The owned column names.
        """
        return list(self.columns)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill the block's missing cells, preserving observed cells.

        The scale/inverse-scale round trip and the per-column domain snaps
        rewrite whole columns; observed-value preservation is applied as the
        final step, so every cell that was not an Effective Null in ``df``
        comes back bit-for-bit with its original dtype (ADR-0078).

        Parameters
        ----------
        df : pl.DataFrame
            Frame to impute. Returned unchanged when none of the block's
            columns are present.

        Returns
        -------
        pl.DataFrame
            ``df`` with the block's missing cells filled.
        """
        cols = [c for c in self.columns if c in df.columns]
        if not cols:
            return df
        import polars as pl

        from ._utils import _df_to_numpy, _numpy_to_df, _preserve_observed
        arr = _df_to_numpy(df, cols)
        arr_scaled = (arr - self.col_means) / self.col_stds
        arr_imputed = self.model.transform(arr_scaled)
        arr_unscaled = arr_imputed * self.col_stds + self.col_means
        out_df = _numpy_to_df(df, cols, arr_unscaled)

        snap_exprs = []
        for col in cols:
            if bounds := self.domain_snap_bounds.get(col):
                lo, hi = bounds
                # Deliberately snaps the whole column; _preserve_observed below
                # restores every observed cell bit-for-bit (#400).
                snap_exprs.append(pl.col(col).round(0).clip(lo, hi).alias(col))
        if snap_exprs:
            out_df = out_df.with_columns(snap_exprs)
        # Also undoes the float drift of the scale/inverse-scale round trip on
        # observed cells (#400).
        return _preserve_observed(df, out_df, cols)


@dataclass
class FittedImputer:
    """Stores the structural manifest and fitted units; applies them to any DataFrame.

    Assembled by :meth:`compose` from a plan and its trained units (ADR-0071).

    Parameters
    ----------
    records : dict[str, ColumnImputationRecord]
        The structural manifest: one record per column the plan describes. Drives
        the whole-frame concerns a single unit has no standing to own — the
        schema guards, the Dropped / Passthrough / Indicator projection — and
        carries each scalar column's learned ``fill_value``, which ``transform``
        applies directly.
    units : list[FittedUnit]
        The trained model-based units (KNN, MICE, and the bimodal
        strategies) in the plan's application order (ADR-0067), never
        thread-completion order. Scalar fills live on ``records`` instead. The
        order is not load-bearing on the result: ``transform`` applies every unit
        against one shared pre-model snapshot and merges each unit's own columns
        back, and those columns are disjoint across units (ADR-0070); it is kept
        because it is the order a human reads the plan in.
    numeric_sentinels : dict[str, list[float]]
        Per-column numeric sentinel declarations copied from
        ``StructuralProfileResult.numeric_sentinels``.  Keys are column names;
        values are float-compatible sentinel values that are normalized to
        Polars-native null before any imputation operation.  Defaults to an
        empty dict — columns with no declaration are completely unaffected.
        Carried from the plan by :meth:`compose`.
    string_sentinels : dict[str, list[str]]
        Per-column string sentinel declarations copied from
        ``StructuralProfileResult.string_sentinels``.  Uses **replace
        semantics**: when a column name is present, only the declared values
        are matched (case-insensitive) and the hardcoded defaults
        (``"NA"``, ``"NAN"``, ``"NULL"``, ``"NONE"``, ``"?"``) are suppressed
        for that column.  Empty/whitespace strings are always treated as
        effective null regardless of any declaration.  Defaults to an empty
        dict — columns with no declaration continue to use the hardcoded
        defaults.  Carried from the plan by :meth:`compose`.
    """

    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    units: list[Any] = field(default_factory=list)
    numeric_sentinels: dict[str, list[float]] = field(default_factory=dict)
    string_sentinels: dict[str, list[str]] = field(default_factory=dict)
    random_seed: Optional[int] = None

    @classmethod
    def compose(
        cls,
        decision: Any,
        results: Any,
    ) -> "FittedImputer":
        """Assemble trained units into a whole-frame imputer (ADR-0071).

        The named constructor of the user-orchestrated flow: given the plan and
        the units the caller's :func:`~dataforge_ml.imputation.fit_unit` loop
        trained (batch scheduling is user-owned, ADR-0075), build the aggregate
        that holds the whole frame. ``compose`` owns exactly the concerns a
        single unit cannot — the full-schema manifest every column is looked up
        in (and with it the whole-frame safety guards), and the structural
        projection of the Dropped / Passthrough / Indicator columns, which have
        no unit precisely because they learn nothing.

        Coverage must be exact: the supplied units must cover the plan's units
        one-for-one — no missing unit (which would leave a column unfilled) and
        no extra one (which describes no column the plan planned).

        Parameters
        ----------
        decision : ImputationDecision
            The plan the units were trained against. Supplies the column
            decisions, the unit order, and the sentinel maps.
        results : Mapping or Iterable
            The trained units, tolerant of either raw fitted units or
            :class:`~dataforge_ml.imputation.UnitFitResult` bundles, given as a
            mapping keyed by unit id or as a plain iterable. A raw fitted unit is
            matched to its plan unit by the columns it owns.

        Returns
        -------
        FittedImputer
            The aggregate: the structural manifest, the trained units in plan
            order, and the plan's sentinel maps.

        Raises
        ------
        ValueError
            If the supplied units do not exactly cover the plan's units, or a
            raw fitted unit matches no plan unit.
        """
        fitted_by_id = _normalize_results(decision, results)

        expected = {u.unit_id for u in decision.units}
        supplied = set(fitted_by_id)
        missing = expected - supplied
        extra = supplied - expected
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing units {sorted(missing)}")
            if extra:
                parts.append(f"unexpected units {sorted(extra)}")
            raise ValueError(
                "compose() requires the supplied units to cover the plan's units "
                f"exactly: {'; '.join(parts)}. Train every planned unit (and only "
                "those) before composing."
            )

        # A scalar unit's learned fill lands on the structural manifest, which is
        # where ``transform`` applies it; a model-based unit joins the ordered
        # ``units`` list (ADR-0071). Every column is routed to exactly one
        # strategy, so a scalar fill and a model unit never collide on a column.
        scalar_fill: dict[str, Any] = {}
        units: list[Any] = []
        for unit in decision.units:
            fitted = fitted_by_id[unit.unit_id]
            if isinstance(fitted, FittedScalar):
                scalar_fill[fitted.target_col] = fitted.fill_value
            else:
                # Application order is the plan's unit order, never the order the
                # units were trained in (ADR-0067).
                units.append(fitted)

        records: dict[str, ColumnImputationRecord] = {}
        for col, col_decision in decision.column_decisions.items():
            records[col] = ColumnImputationRecord(
                decision=col_decision,
                fill_value=scalar_fill.get(col),
                indicator_added=col_decision.indicator_flag,
            )

        return cls(
            records=records,
            units=units,
            numeric_sentinels=dict(decision.numeric_sentinels),
            string_sentinels=dict(decision.string_sentinels),
        )

    def apply_exclusions(self, config: PipelineConfig) -> None:
        """Propagate dropped and indicator columns into the pipeline config.

        Hard Exclusions (ADR 0023): columns recorded with
        ``ImputationStrategy.Dropped`` are added to ``config.exclude_columns``
        via ``add_exclusions``, removing them from every downstream phase.

        Soft Exclusions: columns recorded with ``ImputationStrategy.Indicator``
        are registered in ``config.phase_exclusions`` for Phases 3–6
        (OutlierDetection, Normalization, Encoding, Scaling), so those phases
        skip indicator columns without removing them from the dataset.

        Safe to invoke unconditionally — it is a no-op when no dropped or
        indicator columns exist, so callers need not branch.

        Propagation is caller-initiated: fitting does not touch
        ``PipelineConfig``, preserving re-fit idempotency. A fresh call is
        required against any config that has not already received this
        imputer's exclusions, including after rehydrating via :meth:`compose`.
        Duplicate calls are safe — both hard and soft exclusion registrations
        deduplicate automatically. Nothing verifies that this method was
        called; enforcement is deferred to Phase 3 (ADR-0076).

        Parameters
        ----------
        config : PipelineConfig
            Pipeline config to update.
        """
        dropped = [
            col for col, rec in self.records.items()
            if rec.decision.strategy == ImputationStrategy.Dropped
        ]
        config.add_exclusion(dropped)

        indicator_cols = [
            col for col, rec in self.records.items()
            if rec.decision.strategy == ImputationStrategy.Indicator
        ]
        _soft_phases = [
            PipelinePhase.OutlierDetection,
            PipelinePhase.Normalization,
            PipelinePhase.Encoding,
            PipelinePhase.Scaling,
        ]
        for phase in _soft_phases:
            existing = set(config.phase_exclusions.get(phase, ()))
            new_cols = [c for c in indicator_cols if c not in existing]
            if new_cols:
                config.add_phase_exclusion(phase, new_cols)

    def transform(self, df: pl.DataFrame) -> ImputationResult:
        """
        Apply train-time fill parameters and models to df.

        Sentinel-encoded Effective Nulls are normalised off the plan's
        declared maps before any fill (ADR-0068), and each model-based unit's
        Observed-Value Preservation (ADR-0078) guarantees that only cells
        missing in the normalised frame receive fill values. A ``Passthrough``
        column is left completely alone, nulls included (ADR-0083).

        The **Dtype Floor** is enforced on the working copy immediately after
        that normalisation (ADR-0085), so every consumer inside this call reads
        a column at the dtype its semantic type names. It stops at the boundary:
        the columns it casts come back at their input dtype, observed cells
        bit-for-bit (ADR-0078).

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame to impute. Its columns must be a subset of the
            train-time schema manifest.

        Returns
        -------
        ImputationResult
            The imputed DataFrame together with the per-column records and
            the list of dropped columns.

        Raises
        ------
        UnseenColumnError
            If df contains any column absent from ``self.records``. Fires before
            any DataFrame mutations regardless of whether the unknown column has
            missing values. All unknown column names are reported in one raise.
        FittedColumnAbsentError
            If a column with an active imputation strategy (any strategy other
            than ``Dropped`` or ``Indicator``) is absent from df. All absent
            column names are reported in one raise.

        Warns
        -----
        DroppedColumnAbsentWarning
            If a column recorded as Dropped during fit() is already absent from
            df. One warning is emitted per absent column. Transform continues
            normally.
        """
        # --- UnseenColumnError check (before any mutations) ---
        unseen = [col for col in df.columns if col not in self.records]
        if unseen:
            cols_str = ", ".join(f"'{c}'" for c in unseen)
            raise UnseenColumnError(
                f"Column(s) {cols_str} were not present in the training DataFrame "
                f"during fit() and have no entry in the schema manifest. Schema "
                f"drift between fit and transform is not permitted."
            )

        # --- FittedColumnAbsentError check (before any mutations) ---
        _absent_exempt = {ImputationStrategy.Dropped, ImputationStrategy.Indicator}
        absent = [
            col for col, rec in self.records.items()
            if rec.decision.strategy not in _absent_exempt and col not in df.columns
        ]
        if absent:
            cols_str = ", ".join(f"'{c}'" for c in absent)
            raise FittedColumnAbsentError(
                f"Column(s) {cols_str} were fitted with an active imputation "
                f"strategy but are absent from the input DataFrame. Imputation "
                f"cannot be applied to absent columns."
            )

        # Phase entry: effective nulls first, then the Dtype Floor off the
        # fitted records' decisions (ADR-0085). The floor exists for the
        # consumers *inside* this call; the columns it casts are the
        # non-numeric ones, every one of which is Passthrough, so they are
        # handed back at their pre-floor dtype on the way out and the cast
        # stays invisible at the public boundary (ADR-0078).
        df = _resolve_effective_nulls(
            df,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )
        pre_floor = df
        df = _apply_dtype_floor(
            df,
            {
                name: rec.decision.semantic_type
                for name, rec in self.records.items()
            },
        )
        floored_cols = [
            name
            for name, dtype in df.schema.items()
            if pre_floor.schema[name] != dtype
        ]

        # --- Warn about already-absent dropped columns ---
        for col, rec in self.records.items():
            if rec.decision.strategy == ImputationStrategy.Dropped and col not in df.columns:
                warnings.warn(
                    f"Column '{col}' was recorded as Dropped during fit() but is "
                    f"already absent from the input DataFrame. The drop is a no-op.",
                    DroppedColumnAbsentWarning,
                    stacklevel=2,
                )

        # --- Drop columns ---
        dropped_cols = [
            col
            for col, rec in self.records.items()
            if rec.decision.strategy == ImputationStrategy.Dropped and col in df.columns
        ]
        result_df = df.drop(dropped_cols)

        # --- Build indicator expressions before filling ---
        indicator_exprs = []
        for col, rec in self.records.items():
            if not rec.indicator_added:
                continue
            if col not in result_df.columns:
                continue
            indicator_exprs.append(
                pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_missing")
            )

        if indicator_exprs:
            result_df = result_df.with_columns(indicator_exprs)

        # --- Apply scalar fill values ---
        # Scalar fills come off the structural manifest: a model-based unit reads
        # whatever features it needs off a single pre-model snapshot, and that
        # snapshot must already carry the scalar fills so a regression predicting
        # from a Median-filled feature sees the filled value.
        fill_exprs = []
        for col, rec in self.records.items():
            if rec.decision.strategy in (
                ImputationStrategy.Dropped,
                ImputationStrategy.Passthrough,
                ImputationStrategy.KNN,
                ImputationStrategy.MICE,
            ):
                continue
            if col not in result_df.columns:
                continue
            if rec.fill_value is None:
                continue
            dtype = result_df.schema[col]
            fill_val = rec.fill_value
            if dtype in _INT_DTYPES:
                fill_val = int(round(float(fill_val)))
                fill_exprs.append(pl.col(col).fill_null(fill_val))
            elif dtype in _FLOAT_DTYPES:
                fill_exprs.append(pl.col(col).fill_null(float(fill_val)))
            else:
                fill_exprs.append(pl.col(col).fill_null(fill_val))

        if fill_exprs:
            result_df = result_df.with_columns(fill_exprs)

        # --- Apply model-based units in plan order (ADR-0067) ---
        # Every model unit reads the same pre-model frame and contributes only
        # the columns it owns, so no unit ever sees a sibling's output and the
        # iteration order below cannot reach the result (ADR-0070).
        if self.units:
            pre_model_df = result_df
            for unit in self.units:
                produced = unit.transform(pre_model_df)
                result_df = result_df.with_columns(
                    [
                        produced[col]
                        for col in unit.target_columns
                        if col in produced.columns
                    ]
                )

        # Undo the phase-entry floor cast on the way out. The floor is for the
        # consumers inside this call, not a change to what the caller gets
        # back: every observed cell is restored from the pre-floor column
        # bit-for-bit in its original dtype, and a cell that was missing keeps
        # whatever fill landed on it, rendered back into that dtype (ADR-0078).
        restore_cols = [c for c in floored_cols if c in result_df.columns]
        if restore_cols:
            from ._utils import _preserve_observed

            result_df = _preserve_observed(pre_floor, result_df, restore_cols)

        return ImputationResult(
            dataframe=result_df,
            records=dict(self.records),
            dropped_columns=dropped_cols,
        )


# ---------------------------------------------------------------------------
