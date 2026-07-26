"""
DataSplitter: constructor, random_split, time_split, group_split, kfold,
group_kfold, repeated_kfold, holdout_cv, and profile_stratified_split /
profile_stratified_kfold implementations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Optional

import polars as pl
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from ._config import FoldResult, HoldoutCVResult, SplitConfig, SplitResult

if TYPE_CHECKING:
    from ..profiling._config import StructuralProfileResult

_UNSET = object()


def _route_unsplittable_indices(train_idx, test_idx, mask):
    """Move every row flagged in ``mask`` out of ``test_idx`` and into ``train_idx``.

    Implements the split-side reassignment of ADR-0048: rows belonging to a
    sub-floor target class or rare categorical value are deterministically
    routed into the training partition after the base split decides the rest.
    A no-op when ``mask`` flags nothing in ``test_idx``.
    """
    import numpy as np

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    if not mask.any():
        return train_idx, test_idx
    move = mask[test_idx]
    if not move.any():
        return train_idx, test_idx
    moved = test_idx[move]
    kept = test_idx[~move]
    return np.concatenate([train_idx, moved]), kept


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
    config : SplitConfig, optional
        Splitting thresholds.  Defaults to ``SplitConfig()`` (library defaults).
    """

    def __init__(
        self,
        df: pl.DataFrame,
        target: Optional[str] = None,
        random_seed: Optional[int] = None,
        config: Optional[SplitConfig] = None,
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
        self._config = config if config is not None else SplitConfig()

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

    def group_split(self, test_size: float, group_column: str) -> SplitResult:
        """
        Return a single group-disjoint train/test split.

        Every row belonging to a given ``group_column`` value lands entirely on
        one side of the split, so no group value appears in both partitions.
        Group-disjointness is a hard guarantee: the grouping is the user's
        domain knowledge and is never inferred from the data.

        Parameters
        ----------
        test_size : float
            Fraction of groups to reserve for the test set (0 < test_size < 1).
        group_column : str
            Name of the column identifying the group each row belongs to.
            Must exist in the DataFrame.

        Returns
        -------
        SplitResult

        Raises
        ------
        ValueError
            If ``group_column`` is not a column of the DataFrame.
        """
        if group_column not in self._df.columns:
            raise ValueError(f"group_column '{group_column}' not found in df")

        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=self._random_seed
        )
        groups = self._df[group_column].to_numpy()
        train_idx, test_idx = next(splitter.split(self._df, groups=groups))

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

    def group_kfold(
        self, k: int, group_column: str, stratify=_UNSET
    ) -> List[FoldResult]:
        """
        Return ``k`` group-disjoint cross-validation folds.

        Every row belonging to a given ``group_column`` value stays together in
        one partition, so no group value spans a train/validation boundary in
        any fold. Group-disjointness is a hard constraint that is never traded
        away. When a target is available and ``stratify`` resolves true, target
        class ratios are balanced across folds as a best-effort layer on top of
        the group constraint (only the target is balanced, never a wider signal
        matrix); the balance is approximate, not exact, because whole groups
        cannot be broken to equalise ratios.

        Parameters
        ----------
        k : int
            Number of folds.
        group_column : str
            Name of the column identifying the group each row belongs to.
            Must exist in the DataFrame.
        stratify : bool, optional
            Whether to balance the target class distribution across folds.
            Defaults to True when a target was provided, False otherwise.

        Returns
        -------
        list[FoldResult]
            Exactly ``k`` folds with zero-based ``fold_index``.

        Raises
        ------
        ValueError
            If ``group_column`` is not a column of the DataFrame, or if
            ``stratify`` is True (explicitly or by default) but no target
            column was provided.
        """
        if group_column not in self._df.columns:
            raise ValueError(f"group_column '{group_column}' not found in df")
        if stratify is _UNSET:
            stratify = self._target is not None
        if stratify and self._target is None:
            raise ValueError(
                "stratify=True requires a target column; "
                "pass target= when constructing DataSplitter"
            )

        groups = self._df[group_column].to_numpy()
        if stratify:
            folder: Any = StratifiedGroupKFold(
                n_splits=k, shuffle=True, random_state=self._random_seed
            )
            y = self._df[self._target].to_numpy()
            splits = folder.split(self._df, y, groups=groups)
        else:
            folder = GroupKFold(
                n_splits=k, shuffle=True, random_state=self._random_seed
            )
            splits = folder.split(self._df, groups=groups)

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

    def repeated_kfold(
        self, k: int, n_repeats: int, stratify=_UNSET
    ) -> List[FoldResult]:
        """
        Return ``k * n_repeats`` cross-validation folds from repeated k-fold.

        Runs the fold assignment ``n_repeats`` times with a different seed each
        time to lower the variance of the cross-validated estimate. Each
        returned fold carries its originating repeat via ``repeat_index``; the
        validation partitions within a single repeat cover every row exactly
        once.

        Parameters
        ----------
        k : int
            Number of folds per repeat.
        n_repeats : int
            Number of times the k-fold assignment is repeated.
        stratify : bool, optional
            Whether to stratify on the target column.
            Defaults to True when a target was provided, False otherwise.

        Returns
        -------
        list[FoldResult]
            Exactly ``k * n_repeats`` folds. ``fold_index`` runs 0..k-1 within
            each repeat and ``repeat_index`` runs 0..n_repeats-1.

        Raises
        ------
        ValueError
            If ``stratify`` is True (explicitly or by default) but no target
            column was provided.
        """
        if stratify is _UNSET:
            stratify = self._target is not None
        if stratify and self._target is None:
            raise ValueError(
                "stratify=True requires a target column; "
                "pass target= when constructing DataSplitter"
            )

        if stratify:
            folder: Any = RepeatedStratifiedKFold(
                n_splits=k, n_repeats=n_repeats, random_state=self._random_seed
            )
            y = self._df[self._target].to_numpy()
            splits = folder.split(self._df, y)
        else:
            folder = RepeatedKFold(
                n_splits=k, n_repeats=n_repeats, random_state=self._random_seed
            )
            splits = folder.split(self._df)

        folds: List[FoldResult] = []
        for position, (train_idx, val_idx) in enumerate(splits):
            repeat_index = position // k
            fold_index = position % k
            train_df = self._df[train_idx]
            val_df = self._df[val_idx]
            folds.append(
                FoldResult(
                    train=train_df,
                    val=val_df,
                    fold_index=fold_index,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    repeat_index=repeat_index,
                )
            )

        return folds

    def holdout_cv(
        self, test_size: float, k: int, stratify=_UNSET
    ) -> HoldoutCVResult:
        """
        Reserve a hold-out test set, then cross-validate the remainder.

        First reserves a test partition of ``test_size``, then runs ``k``-fold
        cross-validation over the remaining rows. The held-out test rows are
        disjoint from every fold, so the final evaluation is untouched by the
        cross-validation. This is the hold-out-then-CV protocol; it is
        deliberately not nested CV — the library runs no models, so there is no
        inner hyperparameter loop.

        Parameters
        ----------
        test_size : float
            Fraction of rows to reserve for the test set (0 < test_size < 1).
        k : int
            Number of cross-validation folds over the training remainder.
        stratify : bool, optional
            Whether to stratify the hold-out and the inner folds on the target
            column. Defaults to True when a target was provided, False
            otherwise.

        Returns
        -------
        HoldoutCVResult
            The held-out test partition and the ``k`` folds over the remainder.

        Raises
        ------
        ValueError
            If ``stratify`` is True (explicitly or by default) but no target
            column was provided.
        """
        if stratify is _UNSET:
            stratify = self._target is not None
        if stratify and self._target is None:
            raise ValueError(
                "stratify=True requires a target column; "
                "pass target= when constructing DataSplitter"
            )

        holdout = self.random_split(test_size=test_size, stratify=stratify)
        remainder = DataSplitter(
            holdout.train,
            target=self._target,
            random_seed=self._random_seed,
            config=self._config,
        )
        folds = remainder.kfold(k, stratify=stratify)

        return HoldoutCVResult(test=holdout.test, folds=folds)

    def profile_stratified_split(
        self,
        profile: StructuralProfileResult,
        test_size: float,
    ) -> SplitResult:
        """
        Return a train/test split stratified across all at-risk signals derived
        from the Phase 1 profile (missingness, extremes, rare categories, target).

        Falls back to an unstratified random split when the profile yields no
        usable signals.  When exactly one signal survives the viability and
        redundancy gates, that lone signal is stratified on directly (iterstrat
        needs two or more label columns), rather than being discarded.

        After the base split is decided, rows belonging to a target class or
        rare categorical value too sparse to satisfy the viability floor are
        deterministically routed into the training partition (ADR-0048), so the
        label is learned at fit time and never surfaces unseen at transform
        time.  This routing is applied silently in every path, including the
        random fallback.

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
        import math

        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        from ._profile_signals import build_label_matrix, unsplittable_train_mask

        min_positives = math.ceil(2 / min(test_size, 1 - test_size))
        label_matrix = build_label_matrix(
            self._df, profile, self._target, config=self._config, min_positives=min_positives
        )
        unsplittable = unsplittable_train_mask(
            self._df, profile, self._target, min_positives, config=self._config
        )

        import numpy as np
        if label_matrix.shape[1] == 0:
            splitter = ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self._random_seed
            )
            train_idx, test_idx = next(splitter.split(self._df))
        else:
            X_dummy = np.zeros((len(self._df), 1))
            if label_matrix.shape[1] == 1:
                # A lone surviving signal is not a multilabel-indicator (iterstrat
                # rejects a single column as 'binary'), so stratify on it directly
                # with the single-label splitter rather than discarding it.
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=self._random_seed
                )
                train_idx, test_idx = next(splitter.split(X_dummy, label_matrix[:, 0]))
            else:
                splitter = MultilabelStratifiedShuffleSplit(
                    n_splits=1,
                    test_size=test_size,
                    random_state=self._random_seed,
                )
                train_idx, test_idx = next(splitter.split(X_dummy, label_matrix))

        train_idx, test_idx = _route_unsplittable_indices(train_idx, test_idx, unsplittable)

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

        Falls back to unstratified KFold when the profile yields no usable
        signals.  When exactly one signal survives the viability and redundancy
        gates, that lone signal is stratified on directly (iterstrat needs two
        or more label columns), rather than being discarded.

        In every fold, rows belonging to a target class or rare categorical
        value too sparse to satisfy the viability floor are deterministically
        routed into that fold's training partition (ADR-0048), so the label is
        always learned at fit time and never surfaces unseen at transform time.
        This routing is applied silently in every path, including the KFold
        fallback.

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
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        from ._profile_signals import build_label_matrix, unsplittable_train_mask

        label_matrix = build_label_matrix(
            self._df, profile, self._target, config=self._config, min_positives=k
        )
        unsplittable = unsplittable_train_mask(
            self._df, profile, self._target, k, config=self._config
        )

        import numpy as np
        if label_matrix.shape[1] == 0:
            folder: Any = KFold(
                n_splits=k, shuffle=True, random_state=self._random_seed
            )
            splits = folder.split(self._df)
        else:
            X_dummy = np.zeros((len(self._df), 1))
            if label_matrix.shape[1] == 1:
                # A lone surviving signal is not a multilabel-indicator (iterstrat
                # rejects a single column as 'binary'), so stratify on it directly
                # with the single-label folder rather than discarding it.
                folder = StratifiedKFold(
                    n_splits=k, shuffle=True, random_state=self._random_seed
                )
                splits = folder.split(X_dummy, label_matrix[:, 0])
            else:
                folder = MultilabelStratifiedKFold(
                    n_splits=k,
                    shuffle=True,
                    random_state=self._random_seed,
                )
                splits = folder.split(X_dummy, label_matrix)

        folds: List[FoldResult] = []
        for fold_index, (train_idx, val_idx) in enumerate(splits):
            train_idx, val_idx = _route_unsplittable_indices(train_idx, val_idx, unsplittable)
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
