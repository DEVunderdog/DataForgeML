"""
DataSplitter: constructor, random_split, time_split, kfold, and
profile_stratified_split / profile_stratified_kfold implementations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional

import polars as pl
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit

from ._config import FoldResult, SplitResult

if TYPE_CHECKING:
    from ..profiling._config import StructuralProfileResult

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

    def time_split(
        self,
        time_column: str,
        test_size: Optional[float] = None,
        cutoff: Optional[Any] = None,
    ) -> SplitResult:
        """
        Return a chronological train/test split with no temporal leakage.

        The DataFrame is sorted ascending by ``time_column`` before splitting.
        ``cutoff`` takes priority over ``test_size`` when both are supplied.

        Parameters
        ----------
        time_column : str
            Column to sort by. Must exist in the DataFrame.
        test_size : float, optional
            Fraction of rows (from the end of the sorted series) to use as
            the test set.  ``floor(len(df) * test_size)`` rows are taken.
        cutoff : scalar, optional
            Threshold value.  Rows where ``time_column >= cutoff`` go to
            test; all earlier rows go to train.

        Returns
        -------
        SplitResult
        """
        if time_column not in self._df.columns:
            raise ValueError(f"time_column '{time_column}' not found in df")
        if test_size is None and cutoff is None:
            raise ValueError("Either test_size or cutoff must be provided")

        sorted_df = self._df.sort(time_column)
        total = len(sorted_df)

        if cutoff is not None:
            train_df = sorted_df.filter(pl.col(time_column) < cutoff)
            test_df = sorted_df.filter(pl.col(time_column) >= cutoff)
        else:
            n_test = math.floor(total * test_size)
            n_train = total - n_test
            train_df = sorted_df[:n_train]
            test_df = sorted_df[n_train:]

        return SplitResult(
            train=train_df,
            test=test_df,
            train_size=len(train_df),
            test_size=len(test_df),
            train_ratio=len(train_df) / total,
            test_ratio=len(test_df) / total,
        )

    def kfold(self, k: int, stratify=_UNSET) -> List[FoldResult]:
        """
        Return a list of ``k`` cross-validation folds.

        Parameters
        ----------
        k : int
            Number of folds.
        stratify : bool, optional
            Whether to stratify on the target column.
            Defaults to True when a target was provided, False otherwise.

        Returns
        -------
        list[FoldResult]
            Exactly ``k`` folds with zero-based ``fold_index``.
        """
        if stratify is _UNSET:
            stratify = self._target is not None
        if stratify and self._target is None:
            raise ValueError(
                "stratify=True requires a target column; "
                "pass target= when constructing DataSplitter"
            )

        if stratify:
            folder = StratifiedKFold(
                n_splits=k, shuffle=True, random_state=self._random_seed
            )
            y = self._df[self._target].to_numpy()
            splits = folder.split(self._df, y)
        else:
            folder = KFold(
                n_splits=k, shuffle=True, random_state=self._random_seed
            )
            splits = folder.split(self._df)

        folds: List[FoldResult] = []
        for fold_index, (train_idx, val_idx) in enumerate(splits):
            train_df = self._df[train_idx]
            val_df = self._df[val_idx]
            folds.append(
                FoldResult(
                    train=train_df,
                    val=val_df,
                    fold_index=fold_index,
                    train_size=len(train_df),
                    val_size=len(val_df),
                )
            )

        return folds

    def profile_stratified_split(
        self,
        profile: StructuralProfileResult,
        test_size: float,
    ) -> SplitResult:
        """
        Return a train/test split stratified across all at-risk signals derived
        from the Phase 1 profile (missingness, extremes, rare categories, target).

        Falls back to an unstratified random split when the profile yields no
        usable signals.

        Parameters
        ----------
        profile : StructuralProfileResult
            Output of StructuralProfiler.profile() run on the same DataFrame.
        test_size : float
            Fraction of rows to reserve for the test set (0 < test_size < 1).

        Returns
        -------
        SplitResult
        """
        from ._profile_signals import build_label_matrix
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        label_matrix = build_label_matrix(self._df, profile, self._target)

        if label_matrix.shape[1] == 0:
            return self.random_split(test_size, stratify=False)

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=self._random_seed,
        )
        import numpy as np
        X_dummy = np.zeros((len(self._df), 1))
        train_idx, test_idx = next(splitter.split(X_dummy, label_matrix))

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

    def profile_stratified_kfold(
        self,
        profile: StructuralProfileResult,
        k: int,
    ) -> List[FoldResult]:
        """
        Return k cross-validation folds stratified across all at-risk signals
        derived from the Phase 1 profile.

        Falls back to unstratified KFold when the profile yields no usable signals.

        Parameters
        ----------
        profile : StructuralProfileResult
            Output of StructuralProfiler.profile() run on the same DataFrame.
        k : int
            Number of folds.

        Returns
        -------
        list[FoldResult]
            Exactly k folds with zero-based fold_index.
        """
        from ._profile_signals import build_label_matrix
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        label_matrix = build_label_matrix(self._df, profile, self._target)

        if label_matrix.shape[1] == 0:
            return self.kfold(k, stratify=False)

        import numpy as np
        X_dummy = np.zeros((len(self._df), 1))
        folder = MultilabelStratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=self._random_seed,
        )

        folds: List[FoldResult] = []
        for fold_index, (train_idx, val_idx) in enumerate(folder.split(X_dummy, label_matrix)):
            train_df = self._df[train_idx]
            val_df = self._df[val_idx]
            folds.append(
                FoldResult(
                    train=train_df,
                    val=val_df,
                    fold_index=fold_index,
                    train_size=len(train_df),
                    val_size=len(val_df),
                )
            )

        return folds
