"""
FittedImputer — stateless, serialisable object returned by ImputationOrchestrator.fit().

transform(df) applies train-time fill parameters and fitted models to any DataFrame.
to_dict() / from_dict() round-trips all scalar strategies and model-based strategies
(sklearn objects stored as base64-encoded joblib bytes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import polars as pl

from ..config import PipelineConfig, PipelinePhase, SemanticType
from ._config import (
    ColumnImputationRecord,
    ImputationResult,
    ImputationStrategy,
)
from ._numeric_imputer import FittedRegression
from ..models._data_types import _INT_DTYPES, _FLOAT_DTYPES
from ..utils._null_normalization import _resolve_effective_nulls
from ._utils import _df_to_numpy, _numpy_to_df


class UnfittedColumnError(Exception):
    """
    Raised by FittedImputer.transform() when the input DataFrame contains
    missing values in a column for which no fill strategy was learned during
    fit() — i.e. the column had zero missing values in the training split.

    Typically indicates that a non-profile-stratified split was used.
    """


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


@dataclass
class _FittedKNN:
    """Fitted KNN state including scaling parameters.

    Replaces the bare ``KNNImputer`` previously stored under ``"knn"`` in
    ``FittedImputer.models``.  Stores the model together with the
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

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary for JSON-safe storage.

        Returns
        -------
        dict
            Keys: ``"model"`` (base64-encoded joblib bytes), ``"col_means"``
            and ``"col_stds"`` (plain Python lists).
        """
        import base64
        import io
        import joblib

        buf = io.BytesIO()
        joblib.dump(self.model, buf)
        return {
            "model": base64.b64encode(buf.getvalue()).decode("ascii"),
            "col_means": self.col_means.tolist(),
            "col_stds": self.col_stds.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> _FittedKNN:
        """Reconstruct a ``_FittedKNN`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``.

        Returns
        -------
        _FittedKNN
            Reconstructed instance with a deserialised model and numpy arrays.
        """
        import base64
        import io
        import joblib

        buf = io.BytesIO(base64.b64decode(data["model"]))
        model = joblib.load(buf)
        return cls(
            model=model,
            col_means=np.array(data["col_means"]),
            col_stds=np.array(data["col_stds"]),
        )


@dataclass
class FittedImputer:
    """Stores per-column imputation records and fitted models; applies them to any DataFrame.

    Parameters
    ----------
    records : dict[str, ColumnImputationRecord]
        One record per column processed during fit().
    models : dict[str, Any]
        Fitted sklearn model objects keyed by model name.

        - ``"knn"``           : ``_FittedKNN`` for KNN-assigned columns.
        - ``"mice"``          : ``IterativeImputer`` for MICE-assigned columns.
        - ``"regression:{col}"`` : ``FittedRegression`` containing a fitted
          ``IterativeImputer``.
    model_cols : dict[str, list[str]]
        Ordered column lists for each model entry in models.  Regression
        entries store the full ``[col] + feat_cols`` list (including the
        target at index 0).
    numeric_sentinels : dict[str, list[float]]
        Per-column numeric sentinel declarations copied from
        ``StructuralProfileResult.numeric_sentinels``.  Keys are column names;
        values are float-compatible sentinel values that are normalized to
        Polars-native null before any imputation operation.  Defaults to an
        empty dict — columns with no declaration are completely unaffected.
        Survives ``to_dict()`` / ``from_dict()`` round-trips.
    string_sentinels : dict[str, list[str]]
        Per-column string sentinel declarations copied from
        ``StructuralProfileResult.string_sentinels``.  Uses **replace
        semantics**: when a column name is present, only the declared values
        are matched (case-insensitive) and the hardcoded defaults
        (``"NA"``, ``"NAN"``, ``"NULL"``, ``"NONE"``, ``"?"``) are suppressed
        for that column.  Empty/whitespace strings are always treated as
        effective null regardless of any declaration.  Defaults to an empty
        dict — columns with no declaration continue to use the hardcoded
        defaults.  Survives ``to_dict()`` / ``from_dict()`` round-trips.
    """

    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    model_cols: dict[str, list[str]] = field(default_factory=dict)
    numeric_sentinels: dict[str, list[float]] = field(default_factory=dict)
    string_sentinels: dict[str, list[str]] = field(default_factory=dict)
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._exclusions_applied: bool = False

    def apply_exclusions(self, config: PipelineConfig) -> None:
        """Propagate dropped and indicator columns into the pipeline config.

        Hard Exclusions (ADR 0023): columns recorded with
        ``ImputationStrategy.Dropped`` are added to ``config.exclude_columns``
        via ``add_exclusions``, removing them from every downstream phase.

        Soft Exclusions: columns recorded with ``ImputationStrategy.Indicator``
        are registered in ``config.phase_exclusions`` for Phases 3–6
        (OutlierDetection, Normalization, Encoding, Scaling), so those phases
        skip indicator columns without removing them from the dataset.

        Sets ``_exclusions_applied`` to ``True`` regardless of whether any
        excluded columns exist, so callers can invoke this method
        unconditionally without branching.

        Propagation is caller-initiated: ``fit()`` does not touch
        ``PipelineConfig``, preserving re-fit idempotency. A fresh call is
        required after deserialising via ``from_dict()`` because
        ``_exclusions_applied`` is not persisted across serialisation.
        Duplicate calls are safe — both hard and soft exclusion registrations
        deduplicate automatically.

        Parameters
        ----------
        config : PipelineConfig
            Pipeline config to update.
        """
        dropped = [
            col for col, rec in self.records.items()
            if rec.strategy == ImputationStrategy.Dropped
        ]
        config.add_exclusion(dropped)

        indicator_cols = [
            col for col, rec in self.records.items()
            if rec.strategy == ImputationStrategy.Indicator
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

        self._exclusions_applied = True

    def transform(self, df: pl.DataFrame) -> ImputationResult:
        """
        Apply train-time fill parameters and models to df.

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
        UnfittedColumnError
            If df has missing values in a column that had no missingness during
            fit() (strategy == Passthrough).

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
            if rec.strategy not in _absent_exempt and col not in df.columns
        ]
        if absent:
            cols_str = ", ".join(f"'{c}'" for c in absent)
            raise FittedColumnAbsentError(
                f"Column(s) {cols_str} were fitted with an active imputation "
                f"strategy but are absent from the input DataFrame. Imputation "
                f"cannot be applied to absent columns."
            )

        df = _resolve_effective_nulls(
            df,
            numeric_sentinels=self.numeric_sentinels,
            string_sentinels=self.string_sentinels,
        )

        # --- Passthrough violation check ---
        violating_cols: list[str] = []
        for col, rec in self.records.items():
            if rec.strategy != ImputationStrategy.Passthrough:
                continue
            if col not in df.columns:
                continue
            if df[col].null_count() > 0:
                violating_cols.append(col)

        if violating_cols:
            cols_str = ", ".join(f"'{c}'" for c in violating_cols)
            raise UnfittedColumnError(
                f"Column(s) {cols_str} have missing values but no fill strategy was "
                f"learned during fit() because the training split had no missing "
                f"values. Consider using DataSplitter.profile_stratified_split() "
                f"to ensure missingness is represented in training data."
            )

        # --- Warn about already-absent dropped columns ---
        for col, rec in self.records.items():
            if rec.strategy == ImputationStrategy.Dropped and col not in df.columns:
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
            if rec.strategy == ImputationStrategy.Dropped and col in df.columns
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
        fill_exprs = []
        for col, rec in self.records.items():
            if rec.strategy in (
                ImputationStrategy.Dropped,
                ImputationStrategy.Passthrough,
                ImputationStrategy.KNN,
                ImputationStrategy.MICE,
                ImputationStrategy.Regression,
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

        # --- Apply model-based strategies (MICE → KNN → Regression) with domain-snap ---
        if self.models:
            result_df = _apply_block_model(
                result_df, "mice", self.models, self.model_cols
            )
            result_df = _apply_domain_snap(
                result_df, self.model_cols.get("mice", []), self.records
            )
            result_df = _apply_knn(result_df, self.models, self.model_cols)
            result_df = _apply_domain_snap(
                result_df, self.model_cols.get("knn", []), self.records
            )
            for col, rec in self.records.items():
                if rec.strategy == ImputationStrategy.Regression:
                    result_df = _apply_regression(
                        result_df, col, self.models, self.model_cols
                    )
                    if rec.domain_snap_bounds is not None:
                        lo, hi = rec.domain_snap_bounds
                        result_df = result_df.with_columns(
                            pl.col(col).round(0).clip(lo, hi).alias(col)
                        )

            for col, rec in self.records.items():
                if rec.strategy == ImputationStrategy.ClusterConditional:
                    result_df = _apply_cluster_conditional(
                        result_df, col, self.models[f"cluster:{col}"]
                    )
                elif rec.strategy == ImputationStrategy.GMMSampling:
                    result_df = _apply_gmm_sampling(
                        result_df, col, self.models[f"gmm:{col}"], self.random_seed, self.records
                    )


        return ImputationResult(
            dataframe=result_df,
            records=dict(self.records),
            dropped_columns=dropped_cols,
            exclusions_applied=self._exclusions_applied,
        )

    def to_dict(self) -> dict:
        """Serialize the FittedImputer instance to a dictionary.

        Returns
        -------
        dict
            Dictionary containing serialized records, models (as base64-encoded
            joblib bytes), model column lists, numeric sentinel declarations,
            and string sentinel declarations.
        """
        import base64
        import io
        import joblib

        import json
        serialized_models: dict[str, str] = {}
        for key, model in self.models.items():
            if type(model).__name__ in ("FittedClusterConditional", "FittedGMMSampling"):
                serialized_models[key] = json.dumps({
                    "_type": type(model).__name__,
                    "data": model.to_dict()
                })
            else:
                buf = io.BytesIO()
                joblib.dump(model, buf)
                serialized_models[key] = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "records": {col: rec.to_dict() for col, rec in self.records.items()},
            "models": serialized_models,
            "model_cols": {k: list(v) for k, v in self.model_cols.items()},
            "numeric_sentinels": {k: list(v) for k, v in self.numeric_sentinels.items()},
            "string_sentinels": {k: list(v) for k, v in self.string_sentinels.items()},
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FittedImputer:
        """Deserialize a dictionary back into a FittedImputer instance.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``.  Payloads without a
            ``"numeric_sentinels"`` key (serialised before Scope 5 / issue #94)
            default to an empty dict for backwards compatibility.  Payloads
            without a ``"string_sentinels"`` key (serialised before issue #182)
            also default to an empty dict for backwards compatibility.
            Pre-Scope 8 payloads may contain ``"constant"`` as a strategy
            value; those records deserialise as ``ImputationStrategy.Constant``
            (audit label only — fill behaviour is unchanged).  Payloads without
            a ``"diagnostic"`` key on individual records (serialised before
            Scope 3 / issue #214) default to ``None`` for backwards
            compatibility.

        Returns
        -------
        FittedImputer
            The deserialized FittedImputer instance.
        """
        import base64
        import io
        import joblib
        from ._config import ColumnImputationRecord, ImputationFitDiagnostic, ImputationStrategy

        records: dict[str, ColumnImputationRecord] = {}
        for col, rec_data in data.get("records", {}).items():
            raw_bounds = rec_data.get("domain_snap_bounds")
            snap_bounds = tuple(raw_bounds) if raw_bounds is not None else None
            strategy_str = rec_data["strategy"]
            raw_diag = rec_data.get("diagnostic")
            diagnostic = ImputationFitDiagnostic.from_dict(raw_diag) if raw_diag is not None else None
            records[col] = ColumnImputationRecord(
                column=rec_data["column"],
                semantic_type=SemanticType(rec_data["semantic_type"]),
                strategy=ImputationStrategy(strategy_str),
                fill_value=rec_data.get("fill_value"),
                indicator_added=bool(rec_data.get("indicator_added", False)),
                signals=list(rec_data.get("signals", [])),
                domain_snap_bounds=snap_bounds,
                diagnostic=diagnostic,
            )

        model_cols: dict[str, list[str]] = {
            k: list(v) for k, v in data.get("model_cols", {}).items()
        }

        import json
        models: dict[str, Any] = {}
        for key, b64str in data.get("models", {}).items():
            if b64str.startswith('{"_type":'):
                payload = json.loads(b64str)
                if payload["_type"] == "FittedClusterConditional":
                    from ._numeric_imputer import FittedClusterConditional
                    models[key] = FittedClusterConditional.from_dict(payload["data"])
                elif payload["_type"] == "FittedGMMSampling":
                    from ._numeric_imputer import FittedGMMSampling
                    models[key] = FittedGMMSampling.from_dict(payload["data"])
            else:
                buf = io.BytesIO(base64.b64decode(b64str))
                models[key] = joblib.load(buf)

        numeric_sentinels: dict[str, list[float]] = {
            k: [float(v) for v in vals]
            for k, vals in data.get("numeric_sentinels", {}).items()
        }

        string_sentinels: dict[str, list[str]] = {
            k: [str(v) for v in vals]
            for k, vals in data.get("string_sentinels", {}).items()
        }

        return cls(
            records=records,
            models=models,
            model_cols=model_cols,
            numeric_sentinels=numeric_sentinels,
            string_sentinels=string_sentinels,
            random_seed=data.get("random_seed"),
        )


# ---------------------------------------------------------------------------
# Model application helpers
# ---------------------------------------------------------------------------


def _apply_block_model(
    df: pl.DataFrame,
    model_key: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a block model (MICE) to its assigned columns."""
    if model_key not in models:
        return df
    model = models[model_key]
    cols = [c for c in model_cols.get(model_key, []) if c in df.columns]
    if not cols:
        return df
    arr = _df_to_numpy(df, cols)
    arr_filled = model.transform(arr)
    return _numpy_to_df(df, cols, arr_filled)


def _apply_knn(
    df: pl.DataFrame,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply NaN-safe scaling, KNN imputation, and inverse scaling.

    Retrieves the ``_FittedKNN`` stored at ``models["knn"]``, scales the KNN
    feature columns using the stored ``col_means``/``col_stds``, runs
    ``KNNImputer.transform()`` on the scaled matrix, then inverse-scales the
    output back to original units before writing the imputed values into ``df``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the KNN columns.
    models : dict[str, Any]
        Fitted models dictionary; the ``"knn"`` entry must be a ``_FittedKNN``.
    model_cols : dict[str, list[str]]
        Ordered column lists for each model.

    Returns
    -------
    pl.DataFrame
        DataFrame with KNN-imputed values, inverse-scaled to original units.
    """
    if "knn" not in models:
        return df
    fitted: _FittedKNN = models["knn"]
    cols = [c for c in model_cols.get("knn", []) if c in df.columns]
    if not cols:
        return df
    arr = _df_to_numpy(df, cols)
    arr_scaled = (arr - fitted.col_means) / fitted.col_stds
    arr_imputed = fitted.model.transform(arr_scaled)
    arr_unscaled = arr_imputed * fitted.col_stds + fitted.col_means
    return _numpy_to_df(df, cols, arr_unscaled)


def _apply_regression(
    df: pl.DataFrame,
    col: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a per-column IterativeImputer to fill nulls in col.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    col : str
        Target column name.
    models : dict[str, Any]
        Fitted models dictionary.
    model_cols : dict[str, list[str]]
        Ordered column lists for each model.

    Returns
    -------
    pl.DataFrame
        DataFrame with imputed values in col.
    """
    model_key = f"regression:{col}"
    if model_key not in models or col not in df.columns:
        return df

    fitted_reg: FittedRegression = models[model_key]
    all_cols = model_cols.get(model_key, [])

    target_arr = _df_to_numpy(df, [col]).ravel()
    if not np.isnan(target_arr).any():
        return df

    # Build the joint array ([col] + feat_cols); absent columns stay NaN.
    n_df_rows = len(df)
    arr = np.full((n_df_rows, len(all_cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(all_cols):
        if c in df.columns:
            arr[:, j] = _df_to_numpy(df, [c]).ravel()

    arr_filled = fitted_reg.model.transform(arr)
    target_filled = arr_filled[:, fitted_reg.target_idx]
    return _numpy_to_df(df, [col], target_filled.reshape(-1, 1))


def _apply_domain_snap(
    df: pl.DataFrame,
    cols: list[str],
    records: dict[str, ColumnImputationRecord],
) -> pl.DataFrame:
    """Apply clip(round(p), lo, hi) after model inference for BoundedDiscrete columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame after model inference.
    cols : list[str]
        Columns processed by the preceding model block.
    records : dict[str, ColumnImputationRecord]
        Per-column imputation records; only columns with ``domain_snap_bounds`` set are snapped.

    Returns
    -------
    pl.DataFrame
        DataFrame with snapped values for any column that has ``domain_snap_bounds``.
    """
    snap_exprs = []
    for col in cols:
        rec = records.get(col)
        if rec is None or rec.domain_snap_bounds is None:
            continue
        lo, hi = rec.domain_snap_bounds
        snap_exprs.append(pl.col(col).round(0).clip(lo, hi).alias(col))
    if snap_exprs:
        df = df.with_columns(snap_exprs)
    return df



def _apply_gmm_sampling(
    df: pl.DataFrame, col: str, model: Any, random_seed: Optional[int], records: dict
) -> pl.DataFrame:
    s = df[col]
    null_mask = s.is_null()
    n_missing = null_mask.sum()
    if n_missing == 0:
        return df

    rng = np.random.default_rng(random_seed)
    
    choices = rng.choice([0, 1], p=[model.weight1, model.weight2], size=n_missing)
    samples = np.where(
        choices == 0,
        rng.normal(model.center1, model.std1, size=n_missing),
        rng.normal(model.center2, model.std2, size=n_missing)
    )
    
    rec = records.get(col)
    if rec and rec.domain_snap_bounds is not None:
        lo, hi = rec.domain_snap_bounds
        samples = np.clip(np.round(samples), lo, hi)
        
    s_arr = s.to_numpy().copy()
    s_arr[null_mask.to_numpy()] = samples
    
    return df.with_columns(pl.Series(col, s_arr))

def _apply_cluster_conditional(
    df: pl.DataFrame, col: str, model: Any
) -> pl.DataFrame:
    s = df[col]
    null_mask = s.is_null()
    n_missing = null_mask.sum()
    if n_missing == 0:
        return df
        
    import pandas as pd
    s_arr = s.to_numpy().copy()
    null_idx = np.where(null_mask.to_numpy())[0]

    if model.grouping_variable:
        group_s = df[model.grouping_variable].to_numpy()
        for idx in null_idx:
            group_val = group_s[idx]
            if pd.isna(group_val):
                continue
            fill_val = model.group_fills.get(group_val)
            if fill_val is not None:
                s_arr[idx] = fill_val
    else:
        if model.feature_cols:
            feat_arr = df.select(model.feature_cols).to_numpy()
            for idx in null_idx:
                row_feats = feat_arr[idx]
                row_feats = np.nan_to_num(row_feats)
                
                dist1 = float('inf')
                dist2 = float('inf')
                if model.feature_centroid_1 is not None:
                    dist1 = np.linalg.norm(row_feats - model.feature_centroid_1)
                if model.feature_centroid_2 is not None:
                    dist2 = np.linalg.norm(row_feats - model.feature_centroid_2)
                    
                if dist1 <= dist2 and model.fill_1 is not None:
                    s_arr[idx] = model.fill_1
                elif model.fill_2 is not None:
                    s_arr[idx] = model.fill_2
                    
    return df.with_columns(pl.Series(col, s_arr))
