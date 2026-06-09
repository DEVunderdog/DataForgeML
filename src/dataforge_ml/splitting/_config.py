from __future__ import annotations

from dataclasses import dataclass

import polars as pl


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    """
    Threshold configuration for the splitting sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``SplitConfig()`` produces identical behaviour to the
    pre-config implementation.

    Parameters
    ----------
    max_stratification_signals : int
        Maximum number of binary stratification signals retained by
        ``build_label_matrix``.  When more signals exist, only the
        ``max_stratification_signals`` rarest (smallest proportion of 1s) are
        kept to bound the multilabel-stratification cost.
    boolean_minority_threshold : float
        Minority-class ratio below which a boolean column contributes a
        stratification signal.  A column whose ``true_ratio`` or ``false_ratio``
        falls below this value is treated as imbalanced and receives a signal.
    """

    max_stratification_signals: int = 50
    boolean_minority_threshold: float = 0.05

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "max_stratification_signals": self.max_stratification_signals,
            "boolean_minority_threshold": self.boolean_minority_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SplitConfig:
        """
        Construct a ``SplitConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        SplitConfig
            Reconstructed config instance.
        """
        return cls(
            max_stratification_signals=int(
                data.get("max_stratification_signals", 50)
            ),
            boolean_minority_threshold=float(
                data.get("boolean_minority_threshold", 0.05)
            ),
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplitResult:
    """
    Attributes
    ----------
    train : pl.DataFrame
        Training partition.
    test : pl.DataFrame
        Test/hold-out partition.
    train_size : int
        Number of rows in the training partition.
    test_size : int
        Number of rows in the test partition.
    train_ratio : float
        Fraction of total rows assigned to training (0.0–1.0).
    test_ratio : float
        Fraction of total rows assigned to testing (0.0–1.0).
    """

    train: pl.DataFrame
    test: pl.DataFrame
    train_size: int
    test_size: int
    train_ratio: float
    test_ratio: float


@dataclass
class FoldResult:
    """
    Attributes
    ----------
    train : pl.DataFrame
        Training partition for this fold.
    val : pl.DataFrame
        Validation partition for this fold.
    fold_index : int
        Zero-based index of this fold within the CV run.
    train_size : int
        Number of rows in the training partition.
    val_size : int
        Number of rows in the validation partition.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    fold_index: int
    train_size: int
    val_size: int
