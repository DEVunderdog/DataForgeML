from __future__ import annotations

from enum import StrEnum


class SemanticType(StrEnum):
    """The ML-level interpretation assigned to a column by the type detector.

    Used throughout the pipeline to route columns to the correct sub-processors
    and to determine which statistical operations apply. See CONTEXT.md §SemanticType
    for the full type taxonomy and the Text vs Categorical distinction.
    """

    Numeric = "numeric"
    Categorical = "categorical"
    Datetime = "datetime"
    Boolean = "boolean"
    Text = "text"
    Identifier = "identifier"


class Modality(StrEnum):
    """The data modality the pipeline operates on.

    Currently only ``Tabular`` is supported. Reserved for future expansion to
    additional modalities (time-series, image, etc.).
    """

    Tabular = "tabular"
