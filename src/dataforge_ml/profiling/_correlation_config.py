"""
Result dataclasses for correlation and information-structure profiling.

Populated by CorrelationProfiler, which is opt-in via
ProfileConfig.correlation_target_column (and implicitly by passing
numeric/categorical column lists that are already resolved upstream).

Design notes
------------
- Pearson / Spearman : linear / monotonic relationships between numeric columns.
- Cramér's V         : association between categorical column pairs [0, 1].
- Eta-squared        : numeric-categorical association via ANOVA [0, 1].
- Near-redundancy    : Pearson/Spearman |r| > 0.95, Cramér's V > 0.80,
                    or eta² > 0.50 flagged — near-identical signal.
- Feature–target     : Pearson (numeric target), ANOVA/eta² (categorical target).
- Mutual information : MI for all features vs target (classif or regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

# ---------------------------------------------------------------------------
# Sub-config
# ---------------------------------------------------------------------------


@dataclass
class CorrelationProfileConfig:
    """
    Threshold configuration for the correlation sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``CorrelationProfileConfig()`` produces identical behaviour to
    the pre-config implementation.

    Parameters
    ----------
    near_redundant_pearson_threshold : float
        Maximum absolute Pearson or Spearman ``|r|`` below which a numeric column
        pair is *not* flagged ``near_redundant``.  Pairs whose
        ``max(|pearson_r|, |spearman_r|)`` exceeds this value are flagged.
    near_redundant_cramer_v_threshold : float
        Cramér's V above which a categorical column pair is flagged
        ``near_redundant``.
    near_redundant_eta_squared_threshold : float
        Eta-squared (η²) above which a numeric-categorical pair is flagged
        ``near_redundant``.
    mi_min_rows : int
        Minimum number of complete-case rows required for a k-NN mutual
        information estimate to be computed.  Columns with fewer valid rows
        are silently skipped.
    """

    near_redundant_pearson_threshold: float = 0.95
    near_redundant_cramer_v_threshold: float = 0.80
    near_redundant_eta_squared_threshold: float = 0.50
    mi_min_rows: int = 10

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "near_redundant_pearson_threshold": self.near_redundant_pearson_threshold,
            "near_redundant_cramer_v_threshold": self.near_redundant_cramer_v_threshold,
            "near_redundant_eta_squared_threshold": self.near_redundant_eta_squared_threshold,
            "mi_min_rows": self.mi_min_rows,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CorrelationProfileConfig:
        """
        Construct a ``CorrelationProfileConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        CorrelationProfileConfig
            Reconstructed config instance.
        """
        return cls(
            near_redundant_pearson_threshold=float(
                data.get("near_redundant_pearson_threshold", 0.95)
            ),
            near_redundant_cramer_v_threshold=float(
                data.get("near_redundant_cramer_v_threshold", 0.80)
            ),
            near_redundant_eta_squared_threshold=float(
                data.get("near_redundant_eta_squared_threshold", 0.50)
            ),
            mi_min_rows=int(data.get("mi_min_rows", 10)),
        )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CorrelationMethod(StrEnum):
    Pearson = "pearson"
    Spearman = "spearman"


class TargetType(StrEnum):
    Numeric = "numeric"  # numeric target  → Pearson + MI regression
    Categorical = "categorical"  # categorical target → ANOVA/eta² + MI classif


# ---------------------------------------------------------------------------
# Pairwise correlation result
# ---------------------------------------------------------------------------


@dataclass
class CorrelationPair:
    """
    A single entry in the pairwise correlation results.

    Attributes
    ----------
    col_a, col_b : str
        The two column names (col_a < col_b lexicographically,
        so each pair appears exactly once).
    pearson_r : float | None
        Pearson r.  None when fewer than 3 non-null paired observations.
    spearman_r : float | None
        Spearman r.  None under the same condition.
    near_redundant : bool
        True when max(|pearson_r|, |spearman_r|) > threshold (default 0.95).
    """

    col_a: str
    col_b: str
    pearson_r: Optional[float] = None
    spearman_r: Optional[float] = None
    near_redundant: bool = False

    def to_dict(self) -> dict:
        """
        Serialise this pair to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``col_a``, ``col_b``, ``pearson_r``, ``spearman_r``,
            ``near_redundant``.
        """
        return {
            "col_a": self.col_a,
            "col_b": self.col_b,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "near_redundant": self.near_redundant,
        }


@dataclass
class CramerVPair:
    """
    Cramér's V association between two categorical columns.

    Attributes
    ----------
    col_a, col_b : str
    cramer_v : float | None
        Cramér's V in [0, 1]. None when computation fails or sample too small.
    near_redundant : bool
        True when cramer_v exceeds the near-redundancy threshold (default 0.80).
    """

    col_a: str = ""
    col_b: str = ""
    cramer_v: Optional[float] = None
    near_redundant: bool = False

    def to_dict(self) -> dict:
        """
        Serialise this pair to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``col_a``, ``col_b``, ``cramer_v``, ``near_redundant``.
        """
        return {
            "col_a": self.col_a,
            "col_b": self.col_b,
            "cramer_v": self.cramer_v,
            "near_redundant": self.near_redundant,
        }


@dataclass
class EtaSquaredPair:
    """
    Eta-squared (η²) association between a numeric and a categorical column.

    Attributes
    ----------
    numeric_col : str
    categorical_col : str
    eta_squared : float | None
        Effect size in [0, 1]. None when computation fails.
        Rule of thumb: 0.01 small, 0.06 medium, 0.14 large.
    near_redundant : bool
        True when eta_squared exceeds the near-redundancy threshold (default 0.50).
    """

    numeric_col: str = ""
    categorical_col: str = ""
    eta_squared: Optional[float] = None
    near_redundant: bool = False

    def to_dict(self) -> dict:
        """
        Serialise this pair to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``numeric_col``, ``categorical_col``, ``eta_squared``,
            ``near_redundant``.
        """
        return {
            "numeric_col": self.numeric_col,
            "categorical_col": self.categorical_col,
            "eta_squared": self.eta_squared,
            "near_redundant": self.near_redundant,
        }


# ---------------------------------------------------------------------------
# Feature–target entries
# ---------------------------------------------------------------------------


@dataclass
class NumericTargetCorrelation:
    """
    Pearson r between one numeric feature and a numeric target.

    Attributes
    ----------
    feature : str
    pearson_r : float | None
    """

    feature: str
    pearson_r: Optional[float] = None

    def to_dict(self) -> dict:
        """
        Serialise this entry to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``feature``, ``pearson_r``.
        """
        return {"feature": self.feature, "pearson_r": self.pearson_r}


@dataclass
class CategoricalTargetCorrelation:
    """
    ANOVA-based association between one categorical feature and a numeric
    target (or a numeric feature vs a categorical target when the roles
    are reversed — see CorrelationProfiler docs).

    Attributes
    ----------
    feature : str
    f_statistic : float | None
        One-way ANOVA F-statistic.  Higher F → stronger group separation.
    p_value : float | None
        p-value for the F-test.
    eta_squared : float | None
        Effect size: SS_between / SS_total.  Ranges [0, 1].
        Rule of thumb: 0.01 small, 0.06 medium, 0.14 large.
    """

    feature: str
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None
    eta_squared: Optional[float] = None

    def to_dict(self) -> dict:
        """
        Serialise this entry to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``feature``, ``f_statistic``, ``p_value``, ``eta_squared``.
        """
        return {
            "feature": self.feature,
            "f_statistic": self.f_statistic,
            "p_value": self.p_value,
            "eta_squared": self.eta_squared,
        }


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------


@dataclass
class MutualInformationEntry:
    """
    MI score for one feature vs the target.

    Attributes
    ----------
    feature : str
    mi_score : float
        Raw MI value (nats, sklearn default).  Not directly comparable
        across datasets — use rank ordering within this dataset.
    rank : int
        1 = highest MI (most informative).
    """

    feature: str
    mi_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        """
        Serialise this entry to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``feature``, ``mi_score``, ``rank``.
        """
        return {"feature": self.feature, "mi_score": self.mi_score, "rank": self.rank}


# ---------------------------------------------------------------------------
# Near-redundancy summary
# ---------------------------------------------------------------------------


@dataclass
class NearRedundancyGroup:
    """
    A cluster of mutually near-redundant columns.

    All pairs within the group exceed the |r| > 0.95 threshold.
    The suggested_drop list contains every column except the first
    alphabetically — a simple, deterministic heuristic.
    """

    columns: list[str] = field(default_factory=list)
    suggested_drop: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Serialise this group to a plain dictionary.

        Returns
        -------
        dict
            Keys: ``columns``, ``suggested_drop``.
        """
        return {
            "columns": list(self.columns),
            "suggested_drop": list(self.suggested_drop),
        }


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class CorrelationProfileResult:
    """
    Full correlation and information-structure profile.

    Attributes
    ----------
    analysed_numeric_columns : list[str]
        Numeric columns actually included in the pairwise matrices.
    pairwise : list[CorrelationPair]
        All (col_a, col_b) pairs, each carrying Pearson and Spearman r.
    near_redundant_pairs : list[CorrelationPair]
        Subset of *pairwise* where near_redundant is True.
    near_redundancy_groups : list[NearRedundancyGroup]
        Union-find clusters of near-redundant columns.

    target_column : str | None
        The target column supplied by the caller (may be None when no
        target is provided — only pairwise matrices are then computed).
    target_type : TargetType | None

    feature_target_numeric : list[NumericTargetCorrelation]
        Populated when target is numeric.  Top-10 by |Pearson r|.
    feature_target_categorical : list[CategoricalTargetCorrelation]
        Populated when target is categorical.  Top-10 by eta².
    mutual_information : list[MutualInformationEntry]
        All features ranked by MI vs target.  Empty when no target.

    pearson_matrix : dict[str, dict[str, float]]
        Full symmetric Pearson matrix (numeric columns only).
    spearman_matrix : dict[str, dict[str, float]]
        Full symmetric Spearman matrix (numeric columns only).
    """

    # Column scope
    analysed_numeric_columns: list[str] = field(default_factory=list)
    analysed_categorical_columns: list[str] = field(default_factory=list)

    # Pairwise matrices
    pearson_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    spearman_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # Pairwise summaries — numeric ↔ numeric
    pairwise: list[CorrelationPair] = field(default_factory=list)
    near_redundant_pairs: list[CorrelationPair] = field(default_factory=list)
    near_redundancy_groups: list[NearRedundancyGroup] = field(default_factory=list)

    # Pairwise summaries — categorical ↔ categorical (Cramér's V)
    cramer_v_pairs: list[CramerVPair] = field(default_factory=list)
    near_redundant_cramer_v_pairs: list[CramerVPair] = field(default_factory=list)

    # Pairwise summaries — numeric ↔ categorical (eta-squared)
    eta_squared_pairs: list[EtaSquaredPair] = field(default_factory=list)
    near_redundant_eta_squared_pairs: list[EtaSquaredPair] = field(default_factory=list)

    # Target info
    target_column: Optional[str] = None
    target_type: Optional[TargetType] = None

    # Feature–target correlations (top-10 each)
    feature_target_numeric: list[NumericTargetCorrelation] = field(default_factory=list)
    feature_target_categorical: list[CategoricalTargetCorrelation] = field(
        default_factory=list
    )

    # Mutual information (all features, ranked)
    mutual_information: list[MutualInformationEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def top_mi(self, n: int = 10) -> list[MutualInformationEntry]:
        """
        Return the top-n features by mutual information score.

        Parameters
        ----------
        n : int
            Number of top entries to return.  Defaults to 10.

        Returns
        -------
        list[MutualInformationEntry]
            The first ``n`` entries from ``mutual_information``, which is
            already sorted by descending MI score (rank 1 = highest).
        """
        return self.mutual_information[:n]

    def get_pearson(self, col_a: str, col_b: str) -> Optional[float]:
        """
        Look up the Pearson r between two numeric columns.

        Parameters
        ----------
        col_a : str
            First column name.
        col_b : str
            Second column name.

        Returns
        -------
        float or None
            Pearson r from the symmetric matrix, or ``None`` if the pair is
            not present.
        """
        return self.pearson_matrix.get(col_a, {}).get(col_b)

    def get_spearman(self, col_a: str, col_b: str) -> Optional[float]:
        """
        Look up the Spearman r between two numeric columns.

        Parameters
        ----------
        col_a : str
            First column name.
        col_b : str
            Second column name.

        Returns
        -------
        float or None
            Spearman r from the symmetric matrix, or ``None`` if the pair is
            not present.
        """
        return self.spearman_matrix.get(col_a, {}).get(col_b)

    def to_dict(self) -> dict:
        """
        Serialise the full correlation profile to a plain dictionary.

        Returns
        -------
        dict
            All fields serialised recursively; nested objects call their own
            ``to_dict`` methods.  Enum values are converted to their string
            representations.
        """
        return {
            "analysed_numeric_columns": list(self.analysed_numeric_columns),
            "analysed_categorical_columns": list(self.analysed_categorical_columns),
            "pearson_matrix": {k: dict(v) for k, v in self.pearson_matrix.items()},
            "spearman_matrix": {k: dict(v) for k, v in self.spearman_matrix.items()},
            "pairwise": [p.to_dict() for p in self.pairwise],
            "near_redundant_pairs": [p.to_dict() for p in self.near_redundant_pairs],
            "near_redundancy_groups": [
                g.to_dict() for g in self.near_redundancy_groups
            ],
            "cramer_v_pairs": [p.to_dict() for p in self.cramer_v_pairs],
            "near_redundant_cramer_v_pairs": [
                p.to_dict() for p in self.near_redundant_cramer_v_pairs
            ],
            "eta_squared_pairs": [p.to_dict() for p in self.eta_squared_pairs],
            "near_redundant_eta_squared_pairs": [
                p.to_dict() for p in self.near_redundant_eta_squared_pairs
            ],
            "target_column": self.target_column,
            "target_type": str(self.target_type) if self.target_type else None,
            "feature_target_numeric": [
                f.to_dict() for f in self.feature_target_numeric
            ],
            "feature_target_categorical": [
                f.to_dict() for f in self.feature_target_categorical
            ],
            "mutual_information": [m.to_dict() for m in self.mutual_information],
        }
