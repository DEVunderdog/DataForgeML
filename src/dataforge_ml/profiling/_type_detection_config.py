"""
Config dataclass for the type-detection sub-processor.

Populated by TypeDetector, which is always run as part of
StructuralProfiler (non-optional Phase 1 component).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TypeDetectionConfig:
    """
    Threshold configuration for the type-detection sub-processor.

    All fields default to the library's original hard-coded constants so that
    constructing ``TypeDetectionConfig()`` produces identical behaviour to the
    pre-config implementation.

    Parameters
    ----------
    numeric_coerce_threshold : float
        Minimum fraction of non-null values that must cast successfully to
        ``Float64`` for a ``Utf8`` column to be reclassified as
        ``SemanticType.Numeric``.
    datetime_coerce_threshold : float
        Minimum fraction of non-null values that must parse successfully as a
        datetime for a ``Utf8`` column to be reclassified as
        ``SemanticType.Datetime``.
    encoded_category_max_unique : int
        Maximum absolute number of unique values for a non-tight-sequence
        integer column to be labelled ``TypeFlag.EncodedCategory`` and
        classified as ``SemanticType.Categorical``.
    encoded_category_max_ratio : float
        Maximum ratio of unique values to non-null rows for an integer column
        to be labelled ``TypeFlag.EncodedCategory``.
    identifier_unique_ratio : float
        Minimum ratio of unique values to total rows above which a column is
        flagged as ``TypeFlag.IdentifierColumn`` and classified as
        ``SemanticType.Identifier``.
    identifier_max_median_length : int
        Maximum median character length allowed for a ``Utf8`` column to be
        classified as ``SemanticType.Identifier``. Columns with longer median
        length are treated as free text instead.
    discrete_nunique_threshold : int
        Maximum number of unique values for a non-integer numeric column to be
        treated as discrete (top-value counts) rather than continuous
        (histogram).
    free_text_avg_words : int
        Median word count above which a ``Utf8`` column is flagged as
        ``TypeFlag.FreeTextCandidate``.
    free_text_median_chars : int
        Median character length above which a multi-word ``Utf8`` column is
        flagged as ``TypeFlag.FreeTextCandidate``.
    free_text_p90_chars : int
        90th-percentile character length above which a high-cardinality
        ``Utf8`` column is flagged as ``TypeFlag.FreeTextCandidate``.
    free_text_min_unique_ratio : float
        Minimum unique ratio required (alongside ``free_text_p90_chars``) to
        flag a ``Utf8`` column as ``TypeFlag.FreeTextCandidate``.
    free_text_high_unique_with_spaces : float
        Unique ratio above which a multi-token ``Utf8`` column is flagged as
        ``TypeFlag.FreeTextCandidate`` regardless of character length.
    """

    numeric_coerce_threshold: float = 0.95
    datetime_coerce_threshold: float = 0.80
    encoded_category_max_unique: int = 15
    encoded_category_max_ratio: float = 0.05
    identifier_unique_ratio: float = 0.99
    identifier_max_median_length: int = 40
    discrete_nunique_threshold: int = 20
    free_text_avg_words: int = 3
    free_text_median_chars: int = 20
    free_text_p90_chars: int = 35
    free_text_min_unique_ratio: float = 0.40
    free_text_high_unique_with_spaces: float = 0.70

    def to_dict(self) -> dict:
        """
        Serialise the config to a plain dictionary.

        Returns
        -------
        dict
            All field values keyed by field name.
        """
        return {
            "numeric_coerce_threshold": self.numeric_coerce_threshold,
            "datetime_coerce_threshold": self.datetime_coerce_threshold,
            "encoded_category_max_unique": self.encoded_category_max_unique,
            "encoded_category_max_ratio": self.encoded_category_max_ratio,
            "identifier_unique_ratio": self.identifier_unique_ratio,
            "identifier_max_median_length": self.identifier_max_median_length,
            "discrete_nunique_threshold": self.discrete_nunique_threshold,
            "free_text_avg_words": self.free_text_avg_words,
            "free_text_median_chars": self.free_text_median_chars,
            "free_text_p90_chars": self.free_text_p90_chars,
            "free_text_min_unique_ratio": self.free_text_min_unique_ratio,
            "free_text_high_unique_with_spaces": self.free_text_high_unique_with_spaces,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TypeDetectionConfig:
        """
        Construct a ``TypeDetectionConfig`` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Mapping produced by ``to_dict()``. Missing keys fall back to field
            defaults.

        Returns
        -------
        TypeDetectionConfig
            Reconstructed config instance.
        """
        return cls(
            numeric_coerce_threshold=float(data.get("numeric_coerce_threshold", 0.95)),
            datetime_coerce_threshold=float(data.get("datetime_coerce_threshold", 0.80)),
            encoded_category_max_unique=int(data.get("encoded_category_max_unique", 15)),
            encoded_category_max_ratio=float(data.get("encoded_category_max_ratio", 0.05)),
            identifier_unique_ratio=float(data.get("identifier_unique_ratio", 0.99)),
            identifier_max_median_length=int(data.get("identifier_max_median_length", 40)),
            discrete_nunique_threshold=int(data.get("discrete_nunique_threshold", 20)),
            free_text_avg_words=int(data.get("free_text_avg_words", 3)),
            free_text_median_chars=int(data.get("free_text_median_chars", 20)),
            free_text_p90_chars=int(data.get("free_text_p90_chars", 35)),
            free_text_min_unique_ratio=float(data.get("free_text_min_unique_ratio", 0.40)),
            free_text_high_unique_with_spaces=float(
                data.get("free_text_high_unique_with_spaces", 0.70)
            ),
        )
