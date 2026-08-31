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
        Upper bound on the number of binary stratification signals retained by
        ``build_label_matrix``.  The effective cap (ADR-0047 gate 4) is
        ``min(max_stratification_signals, n_rows / rows_per_signal)``; when more
        signals survive the viability and redundancy gates than the cap allows,
        signals are retained by importance rank (target first, missingness last)
        rather than by rarity, to bound the multilabel-stratification cost.
    rows_per_signal : int
        Rows-per-signal budget (ADR-0047 gate 4).  The data can support at most
        ``n_rows / rows_per_signal`` simultaneous stratification constraints, so
        this term co-bounds the cap alongside ``max_stratification_signals``.
        The default of 10 is a standard per-constraint rule of thumb; a
        wide-but-short dataset is reduced to fewer columns, and one too small to
        support even the target degrades to a random split.
    boolean_minority_threshold : float
        Minority-class ratio below which a boolean column contributes a
        stratification signal.  A column whose ``true_ratio`` or ``false_ratio``
        falls below this value is treated as imbalanced and receives a signal.
    redundancy_correlation_threshold : float
        Absolute-correlation strength at or above which two stratification
        signals are treated as redundant (ADR-0047 gate 3) and collapsed to the
        higher-priority one.  Correlation magnitude catches both near-identical
        (``+1``) and mirror-image (``−1``) pairs.  The default is conservative so
        that only near-perfect duplicates are removed, never merely-similar
        signals.
    """

    max_stratification_signals: int = 50
    rows_per_signal: int = 10
    boolean_minority_threshold: float = 0.05
    redundancy_correlation_threshold: float = 0.95

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
            "rows_per_signal": self.rows_per_signal,
            "boolean_minority_threshold": self.boolean_minority_threshold,
            "redundancy_correlation_threshold": self.redundancy_correlation_threshold,
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
            rows_per_signal=int(data.get("rows_per_signal", 10)),
            boolean_minority_threshold=float(
                data.get("boolean_minority_threshold", 0.05)
            ),
            redundancy_correlation_threshold=float(
                data.get("redundancy_correlation_threshold", 0.95)
            ),
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


def _frame_lines(frame: pl.DataFrame, n_rows: int) -> list[str]:
    """Render a Payload Frame's shape and dtypes, never its rows (ADR-0086)."""
    lines = [
        f"Shape: {n_rows:,} rows x {frame.width:,} columns\n",
        "| Column | Dtype |",
        "|---|---|",
    ]
    if frame.width:
        for name, dtype in frame.schema.items():
            lines.append(f"| `{name}` | {dtype} |")
    else:
        lines.append("| none | |")
    return lines


@dataclass
class SplitResult:
    """
    A single train/test partition of a dataset.

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

    def to_markdown(self) -> str:
        """Render the train/test split as a Markdown document.

        A document per rule 5 of the Rendering Contract (ADR-0086). Both
        partitions are Payload Frames: their shape and dtypes are reported and
        their rows never are, so the output stays bounded in the number of rows
        the split carries.

        Returns
        -------
        str
            Markdown document with the partition sizes and ratios, followed by
            the shape and dtypes of the train and test frames.
        """
        total = self.train_size + self.test_size
        lines = ["# Train/Test Split\n"]
        lines.append("| Partition | Rows | Ratio |")
        lines.append("|---|---|---|")
        lines.append(f"| train | {self.train_size:,} | {self.train_ratio:.2%} |")
        lines.append(f"| test | {self.test_size:,} | {self.test_ratio:.2%} |")
        lines.append(f"| total | {total:,} | |")
        lines.append("")
        lines.append("## Train Frame\n")
        lines.extend(_frame_lines(self.train, self.train_size))
        lines.append("")
        lines.append("## Test Frame\n")
        lines.extend(_frame_lines(self.test, self.test_size))
        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Train/Test Split document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()


@dataclass
class FoldResult:
    """
    One cross-validation fold: its train/validation partitions and position.

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
    repeat_index : int
        Zero-based index of the repeat this fold belongs to. Defaults to ``0``
        for single-pass schemes (``kfold``, ``profile_stratified_kfold``); only
        ``repeated_kfold`` sets it to the originating repeat number.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    fold_index: int
    train_size: int
    val_size: int
    repeat_index: int = 0

    def to_markdown(self) -> str:
        """Render the fold as a Markdown document.

        A document, not a fragment: ``kfold``, ``group_kfold``,
        ``repeated_kfold`` and ``profile_stratified_kfold`` hand a
        ``list[FoldResult]`` back directly, so a single fold is something the
        library returns and ``print(folds[0])`` must read as a complete report
        (ADR-0086). Both partitions are Payload Frames: shape and dtypes are
        reported, rows never are.

        Returns
        -------
        str
            Markdown document with the fold's position, its partition sizes,
            and the shape and dtypes of the train and validation frames.
        """
        total = self.train_size + self.val_size
        lines = [f"# Fold {self.fold_index} (repeat {self.repeat_index})\n"]
        lines.append("| Partition | Rows |")
        lines.append("|---|---|")
        lines.append(f"| train | {self.train_size:,} |")
        lines.append(f"| validation | {self.val_size:,} |")
        lines.append(f"| total | {total:,} |")
        lines.append("")
        lines.append("## Train Frame\n")
        lines.extend(_frame_lines(self.train, self.train_size))
        lines.append("")
        lines.append("## Validation Frame\n")
        lines.extend(_frame_lines(self.val, self.val_size))
        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Fold document, per rule 2 of the contract.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()



@dataclass
class HoldoutCVResult:
    """
    A held-out test partition plus cross-validation folds over the remainder.

    Attributes
    ----------
    test : pl.DataFrame
        Held-out test partition, disjoint from every fold in ``folds``.
    folds : list[FoldResult]
        Cross-validation folds over the training remainder (all rows not in
        ``test``).
    """

    test: pl.DataFrame
    folds: list[FoldResult]

    def to_markdown(self) -> str:
        """Render the holdout-plus-CV split as a Markdown document.

        The fold summary table is formatted inline rather than delegated to
        :meth:`FoldResult.to_markdown`, because ``FoldResult`` is itself a
        document and rule 5 of the Rendering Contract (ADR-0086) does not let
        one type be both document and fragment. The test partition is a Payload
        Frame: its shape and dtypes are reported and its rows never are.

        Returns
        -------
        str
            Markdown document with the test frame's shape and dtypes followed
            by a one-row-per-fold summary table.
        """
        lines = ["# Holdout + Cross-Validation Split\n"]
        lines.append("## Test Frame\n")
        lines.extend(_frame_lines(self.test, self.test.height))
        lines.append("")
        lines.append(f"## Folds ({len(self.folds)})\n")
        lines.append("| Fold | Repeat | Train rows | Validation rows | Columns |")
        lines.append("|---|---|---|---|---|")
        if self.folds:
            for fold in self.folds:
                lines.append(
                    f"| {fold.fold_index} | {fold.repeat_index} "
                    f"| {fold.train_size:,} | {fold.val_size:,} "
                    f"| {fold.train.width:,} |"
                )
        else:
            lines.append("| none | | | | |")
        return "\n".join(lines).strip() + "\n"

    def __str__(self) -> str:
        """Return the Holdout + Cross-Validation document, per rule 2.

        Returns
        -------
        str
            The output of :meth:`to_markdown`.
        """
        return self.to_markdown()
