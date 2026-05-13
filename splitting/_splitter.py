"""
DataSplitter: constructor and random_split implementation.
"""

from __future__ import annotations

from typing import Optional

import polars as pl
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from ._config import SplitResult

_UNSET = object()


class DataSplitter:
    """
    Splits a Polars DataFrame into train/test or cross-validation folds.

    Parameters
    ----------
    df : pl.DataFrame
        Source data. Must be non-empty.
    target : str, optional
        Name of the target column. Required for stratified splits.
    random_seed : int, optional
        Seed forwarded to sklearn splitters for reproducibility.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        target: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        if not isinstance(df, pl.DataFrame):
            raise TypeError(f"df must be a polars DataFrame, got {type(df).__name__}")
        if df.is_empty():
            raise ValueError("df must not be empty")
        if target is not None and target not in df.columns:
            raise ValueError(f"target column '{target}' not found in df")

        self._df = df
        self._target = target
        self._random_seed = random_seed

    def random_split(self, test_size: float, stratify=_UNSET) -> SplitResult:
        """
        Return a single randomised train/test split.

        Parameters
        ----------
        test_size : float
            Fraction of rows to reserve for the test set (0 < test_size < 1).
        stratify : bool, optional
            Whether to stratify on the target column.
            Defaults to True when a target was provided, False otherwise.

        Returns
        -------
        SplitResult
        """
        if stratify is _UNSET:
            stratify = self._target is not None
        if stratify and self._target is None:
            raise ValueError(
                "stratify=True requires a target column; "
                "pass target= when constructing DataSplitter"
            )

        if stratify:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self._random_seed
            )
            y = self._df[self._target].to_numpy()
            train_idx, test_idx = next(splitter.split(self._df, y))
        else:
            splitter = ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self._random_seed
            )
            train_idx, test_idx = next(splitter.split(self._df))

        train_df = self._df[train_idx]
        test_df = self._df[test_idx]
        total = len(self._df)

        return SplitResult(
            train=train_df,
            test=test_df,
            train_size=len(train_df),
            test_size=len(test_df),
            train_ratio=len(train_df) / total,
            test_ratio=len(test_df) / total,
        )
