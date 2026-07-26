"""
Fitted imputation units — standalone, independently transform-capable state.

Each class here satisfies the ``FittedUnit`` protocol (``_fitted_imputer``):
it transforms only its own columns and carries no whole-frame completeness
requirement.

Every unit's ``transform`` upholds Observed-Value Preservation (ADR-0078): it
fills holes and never edits a value the user supplied — after ``transform``,
every cell that was not an Effective Null in the input is identical, same
value, same dtype. The model-based units enforce it as the final step of
their own ``transform``; a bare unit therefore carries the guarantee onto the
stateless door (ADR-0071), where the caller owns sentinel normalisation.

This module is deliberately dependency-free with respect to the fitting
surface. The ``_fitters`` functions produce these types, so it must not import
them — that edge is what keeps the fitted types free of a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import polars as pl
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

if TYPE_CHECKING:
    from ._fitted_imputer import FittedUnit


class FittedScalar:
    """Standalone unit for scalar imputation (Mean, Median, Mode, Constant).

    A scalar is only ever reached one way now — a plan that routed the column to
    a scalar strategy in the first place — since a model-based unit that cannot
    train raises rather than degrading to one (single-track failure, ADR-0071).
    The unit therefore carries only its fitted state: the column and the learned
    fill value. Which scalar strategy produced it is a decide-time fact the plan
    holds under the same ``unit_id``, and its fit-time observability rides on the
    :class:`~dataforge_ml.imputation.FitSignals` the fit returned (ADR-0074), not
    on the persisted unit.

    Parameters
    ----------
    target_col : str
        The column this unit fills.
    fill_value : Any
        The learned fill value.
    """

    def __init__(
        self,
        target_col: str,
        fill_value: Any,
    ):
        self.target_col = target_col
        self.fill_value = fill_value

    @property
    def target_columns(self) -> list[str]:
        """The single column this unit fills.

        Returns
        -------
        list[str]
            The owned column name.
        """
        return [self.target_col]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill the target column's null cells with the learned scalar.

        Fills via ``fill_null``, which writes only null cells, so observed
        cells are untouched by construction (ADR-0078). On the stateless door
        a raw float ``NaN`` is not a null and is deliberately left unfilled;
        the composed :meth:`FittedImputer.transform` normalises it to null
        before the fill.

        Parameters
        ----------
        df : pl.DataFrame
            Frame to impute. Returned unchanged when the target column is
            absent or no fill value was learned.

        Returns
        -------
        pl.DataFrame
            ``df`` with the target column's null cells filled.
        """
        if self.target_col not in df.columns or self.fill_value is None:
            return df
        # An integer-typed column takes a whole-number fill: a Median of 3.5 on
        # an ``Int64`` column would otherwise fail to cast (or silently upcast
        # the whole column to float). Float columns fill with the value as-is.
        from ..models._data_types import _FLOAT_DTYPES, _INT_DTYPES

        dtype = df.schema[self.target_col]
        fill_val = self.fill_value
        if dtype in _INT_DTYPES:
            fill_val = int(round(float(fill_val)))
        elif dtype in _FLOAT_DTYPES:
            fill_val = float(fill_val)
        return df.with_columns(pl.col(self.target_col).fill_null(fill_val))


@dataclass
class FittedClusterConditional:
    """Fitted state for Cluster-Conditional Imputation (branches 1 and 3).

    Parameters
    ----------
    grouping_variable : str or None
        Name of the categorical column used for grouping (branch 1).
    group_fills : dict of Any to float or None
        Mapping from group values to fill values (branch 1).
    fill_1 : float or None
        Fill value for cluster 1 (branch 3).
    fill_2 : float or None
        Fill value for cluster 2 (branch 3).
    feature_centroid_1 : np.ndarray or None
        Feature centroid for cluster 1 (branch 3).
    feature_centroid_2 : np.ndarray or None
        Feature centroid for cluster 2 (branch 3).
    feature_cols : list of str or None
        Names of the feature columns used to compute distances (branch 3).
    center1 : float
        Mean of cluster 1 from the univariate bimodal fit.
    center2 : float
        Mean of cluster 2 from the univariate bimodal fit.
    """
    grouping_variable: Optional[str]
    group_fills: Optional[dict[Any, float]]
    fill_1: Optional[float]
    fill_2: Optional[float]
    feature_centroid_1: Optional[np.ndarray]
    feature_centroid_2: Optional[np.ndarray]
    feature_cols: Optional[list[str]]
    center1: float
    center2: float
    target_col: str = ""
    domain_snap_bounds: Optional[tuple[float, float]] = None

    @property
    def target_columns(self) -> list[str]:
        """The single column this unit fills.

        The grouping variable and the distance features are read, never
        written, so they are inputs rather than targets.

        Returns
        -------
        list[str]
            The owned column name.
        """
        return [self.target_col]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill the target column's missing cells, preserving observed cells.

        Cluster assignment and the domain snap operate on a whole-column
        array; observed-value preservation is applied as the final step, so
        every cell that was not an Effective Null in ``df`` comes back
        bit-for-bit with its original dtype (ADR-0078).

        Parameters
        ----------
        df : pl.DataFrame
            Frame to impute. Returned unchanged when the target column (or,
            in branch 1, the grouping variable) is absent or the column
            carries no missing cells.

        Returns
        -------
        pl.DataFrame
            ``df`` with the target column's missing cells filled.
        """
        col = self.target_col
        if col not in df.columns:
            return df
        s = df[col]
        null_mask = s.is_null()
        n_missing = null_mask.sum()
        if n_missing == 0:
            return df

        import pandas as pd
        s_arr = s.to_numpy().copy()
        null_idx = np.where(null_mask.to_numpy())[0]

        if self.grouping_variable:
            if self.grouping_variable not in df.columns:
                return df
            group_s = df[self.grouping_variable].to_numpy()
            for idx in null_idx:
                group_val = group_s[idx]
                if pd.isna(group_val):
                    continue
                fill_val = self.group_fills.get(group_val)
                if fill_val is not None:
                    s_arr[idx] = fill_val
        else:
            if self.feature_cols:
                feat_arr = df.select([c for c in self.feature_cols if c in df.columns]).to_numpy()
                for idx in null_idx:
                    row_feats = feat_arr[idx]
                    row_feats = np.nan_to_num(row_feats)
                    
                    dist1 = float('inf')
                    dist2 = float('inf')
                    if self.feature_centroid_1 is not None:
                        dist1 = np.linalg.norm(row_feats - self.feature_centroid_1)
                    if self.feature_centroid_2 is not None:
                        dist2 = np.linalg.norm(row_feats - self.feature_centroid_2)
                        
                    if dist1 <= dist2 and self.fill_1 is not None:
                        s_arr[idx] = self.fill_1
                    elif self.fill_2 is not None:
                        s_arr[idx] = self.fill_2
                        
        if self.domain_snap_bounds is not None:
            lo, hi = self.domain_snap_bounds
            # Deliberately snaps the whole column; _preserve_observed below
            # restores every observed cell bit-for-bit (#400).
            s_arr = np.clip(np.round(s_arr), lo, hi)

        from ..models._data_types import _INT_DTYPES
        from ._utils import _preserve_observed

        out = pl.Series(col, s_arr)
        if df.schema[col] in _INT_DTYPES:
            # An integer column's holes surfaced as NaN in the numpy round trip;
            # any still unfilled (e.g. a null group value) must go back as null
            # so the integer cast in _preserve_observed holds.
            out = out.fill_nan(None)
        return _preserve_observed(df, df.with_columns(out), [col])



@dataclass
class FittedGMMSampling:
    """Fitted state for GMM Sampling Imputation (branch 4).

    Parameters
    ----------
    center1 : float
        Mean of the first GMM component.
    center2 : float
        Mean of the second GMM component.
    std1 : float
        Standard deviation of the first GMM component.
    std2 : float
        Standard deviation of the second GMM component.
    weight1 : float
        Mixing weight of the first GMM component.
    weight2 : float
        Mixing weight of the second GMM component.
    """
    center1: float
    center2: float
    std1: float
    std2: float
    weight1: float
    weight2: float
    target_col: str = ""
    domain_snap_bounds: Optional[tuple[float, float]] = None
    random_seed: Optional[int] = None

    @property
    def target_columns(self) -> list[str]:
        """The single column this unit fills.

        The unit samples from its own fitted mixture and reads no other column,
        so it is order-independent by construction.

        Returns
        -------
        list[str]
            The owned column name.
        """
        return [self.target_col]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill the target column's missing cells, preserving observed cells.

        Samples one fill per missing cell from the fitted two-component
        mixture; observed-value preservation is applied as the final step, so
        every cell that was not an Effective Null in ``df`` comes back
        bit-for-bit with its original dtype (ADR-0078).

        Parameters
        ----------
        df : pl.DataFrame
            Frame to impute. Returned unchanged when the target column is
            absent or carries no missing cells.

        Returns
        -------
        pl.DataFrame
            ``df`` with the target column's missing cells filled.
        """
        col = self.target_col
        if col not in df.columns:
            return df
        s = df[col]
        null_mask = s.is_null()
        n_missing = null_mask.sum()
        if n_missing == 0:
            return df

        rng = np.random.default_rng(self.random_seed)
        
        choices = rng.choice([0, 1], p=[self.weight1, self.weight2], size=n_missing)
        samples = np.where(
            choices == 0,
            rng.normal(self.center1, self.std1, size=n_missing),
            rng.normal(self.center2, self.std2, size=n_missing)
        )
        
        if self.domain_snap_bounds is not None:
            lo, hi = self.domain_snap_bounds
            # Deliberately snaps the samples destined for the whole column's
            # write-back; _preserve_observed below restores every observed cell
            # bit-for-bit (#400).
            samples = np.clip(np.round(samples), lo, hi)

        from ._utils import _preserve_observed

        s_arr = s.to_numpy().copy()
        s_arr[null_mask.to_numpy()] = samples
        return _preserve_observed(df, df.with_columns(pl.Series(col, s_arr)), [col])

