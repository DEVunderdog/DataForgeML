"""
FittedImputer — stateless, serialisable object returned by ImputationOrchestrator.fit().

transform(df) applies train-time fill parameters and fitted models to any DataFrame.
to_dict() / from_dict() round-trips all scalar strategies and model-based strategies
(sklearn objects stored as base64-encoded joblib bytes).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

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


class _LegacyRegressionModel(tuple):
    """Wrapper for legacy (BayesianRidge, feat_means) models.

    Inherits from tuple to maintain backwards-compatibility with tests checking
    for tuple type, while providing a ``transform`` method to unify the
    inference path and eliminate the mean-patching loop from the caller.
    """

    def __new__(cls, reg: Any, feat_means: np.ndarray) -> _LegacyRegressionModel:
        """Create a new instance of _LegacyRegressionModel.

        Parameters
        ----------
        reg : Any
            The fitted BayesianRidge regressor.
        feat_means : np.ndarray
            The mean feature values computed during training.

        Returns
        -------
        _LegacyRegressionModel
            The legacy model wrapper.
        """
        return super().__new__(cls, (reg, feat_means))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply legacy regression prediction with mean-patching.

        Parameters
        ----------
        X : np.ndarray
            Joint array of target and feature columns.

        Returns
        -------
        np.ndarray
            Filled array.
        """
        X = X.copy()
        target_arr = X[:, 0]
        null_mask = np.isnan(target_arr)
        if not null_mask.any():
            return X

        # Feature columns are at indices 1 to end
        X_feats = X[:, 1:]
        nan_in_X = np.isnan(X_feats)
        if nan_in_X.any():
            X_feats = X_feats.copy()
            for j in range(X_feats.shape[1]):
                if nan_in_X[:, j].any():
                    fill = float(self[1][j]) if j < len(self[1]) else 0.0
                    X_feats[nan_in_X[:, j], j] = fill

        if X_feats.shape[1] > 0:
            y_pred = self[0].predict(X_feats[null_mask])
        else:
            y_pred = np.zeros(null_mask.sum())

        target_arr[null_mask] = y_pred
        X[:, 0] = target_arr
        return X


@dataclass
class FittedImputer:
    """Stores per-column imputation records and fitted models; applies them to any DataFrame.

    Parameters
    ----------
    records : dict[str, ColumnImputationRecord]
        One record per column processed during fit().
    models : dict[str, Any]
        Fitted sklearn model objects keyed by model name.

        - ``"knn"``           : ``KNNImputer`` for KNN-assigned columns.
        - ``"mice"``          : ``IterativeImputer`` for MICE-assigned columns.
        - ``"regression:{col}"`` : ``FittedRegression`` containing a fitted
        ``IterativeImputer``.  Legacy entries serialised before Issue #141
        carried a ``(BayesianRidge, feat_means)`` tuple directly; the
        ``from_dict()`` migration path wraps these in a ``FittedRegression``
        transparently.
    model_cols : dict[str, list[str]]
        Ordered column lists for each model entry in models.  Regression
        entries store the full ``[col] + feat_cols`` list (including the
        target at index 0).
    """

    records: dict[str, ColumnImputationRecord] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    model_cols: dict[str, list[str]] = field(default_factory=dict)

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
        config.add_exclusions(dropped)

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
            existing = set(config.phase_exclusions.get(phase, []))
            new_cols = [c for c in indicator_cols if c not in existing]
            if new_cols:
                config.phase_exclusions.setdefault(phase, []).extend(new_cols)

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

        df = _resolve_effective_nulls(df)

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

        # --- Apply model-based strategies (MICE → KNN → Regression) ---
        if self.models:
            result_df = _apply_block_model(
                result_df, "mice", self.models, self.model_cols
            )
            result_df = _apply_block_model(
                result_df, "knn", self.models, self.model_cols
            )
            for col, rec in self.records.items():
                if rec.strategy == ImputationStrategy.Regression:
                    result_df = _apply_regression(
                        result_df, col, self.models, self.model_cols
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
            joblib bytes), and model column lists.
        """
        import base64
        import io
        import joblib

        serialized_models: dict[str, str] = {}
        for key, model in self.models.items():
            buf = io.BytesIO()
            joblib.dump(model, buf)
            serialized_models[key] = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "records": {col: rec.to_dict() for col, rec in self.records.items()},
            "models": serialized_models,
            "model_cols": {k: list(v) for k, v in self.model_cols.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> FittedImputer:
        """Deserialize a dictionary back into a FittedImputer instance.

        Also performs migration of legacy regression entries transparently.

        Parameters
        ----------
        data : dict
            Dictionary containing serialized FittedImputer state.

        Returns
        -------
        FittedImputer
            The deserialized and migrated FittedImputer instance.
        """
        import base64
        import io
        import joblib
        from ._config import ColumnImputationRecord, ImputationStrategy

        records: dict[str, ColumnImputationRecord] = {}
        for col, rec_data in data.get("records", {}).items():
            records[col] = ColumnImputationRecord(
                column=rec_data["column"],
                semantic_type=SemanticType(rec_data["semantic_type"]),
                strategy=ImputationStrategy(rec_data["strategy"]),
                fill_value=rec_data.get("fill_value"),
                indicator_added=bool(rec_data.get("indicator_added", False)),
                signals=list(rec_data.get("signals", [])),
            )

        model_cols: dict[str, list[str]] = {
            k: list(v) for k, v in data.get("model_cols", {}).items()
        }

        models: dict[str, Any] = {}
        for key, b64str in data.get("models", {}).items():
            buf = io.BytesIO(base64.b64decode(b64str))
            loaded = joblib.load(buf)

            # Migration: legacy regression entries are identified by the absence of
            # target_idx in the stored object (or model_cols entry not starting with target).
            if key.startswith("regression:"):
                target_col = key[len("regression:"):]
                feat_cols = list(model_cols.get(key, []))
                # 1. Detect legacy format: model_cols entry does not start with target
                if not feat_cols or feat_cols[0] != target_col:
                    # 2. Derive col from the key (target_col)
                    # 3. Prepend col to the stored list to produce all_cols
                    all_cols = [target_col] + feat_cols
                    model_cols[key] = all_cols
                    # 4. Wrap the loaded (BayesianRidge, feat_means) tuple in a FittedRegression
                    # with target_idx=0 and the migrated all_cols.
                    legacy_tuple = loaded if isinstance(loaded, tuple) else (loaded[0], loaded[1])
                    loaded = FittedRegression(
                        model=_LegacyRegressionModel(legacy_tuple[0], legacy_tuple[1]),
                        target_idx=0,
                        all_cols=all_cols,
                    )

            models[key] = loaded

        return cls(records=records, models=models, model_cols=model_cols)


# ---------------------------------------------------------------------------
# Model application helpers
# ---------------------------------------------------------------------------


def _apply_block_model(
    df: pl.DataFrame,
    model_key: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a block model (KNN or MICE) to its assigned columns."""
    if model_key not in models:
        return df
    model = models[model_key]
    cols = [c for c in model_cols.get(model_key, []) if c in df.columns]
    if not cols:
        return df
    arr = _df_to_numpy(df, cols)
    arr_filled = model.transform(arr)
    return _numpy_to_df(df, cols, arr_filled)


def _apply_regression(
    df: pl.DataFrame,
    col: str,
    models: dict[str, Any],
    model_cols: dict[str, list[str]],
) -> pl.DataFrame:
    """Apply a per-column IterativeImputer (or migrated legacy model) to fill nulls in col.

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


