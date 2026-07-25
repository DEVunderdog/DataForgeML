"""
Unit tests for serializing / deserializing a ``StructuralProfileResult`` through
the bare-bytes persistence boundary (ADR-0072).

The profile serialises as a single JSON envelope. ``deserialize(serialize(x))``
must be structurally equal to ``x`` across every column-stats family (numeric,
categorical, datetime, boolean, text), the dataset-level correlation and
missingness structures, and target profiles — with every enum reconstructed from
its serialised value so reordering an enum's members can never corrupt a
previously-serialized envelope.
"""

import json
from datetime import date, timedelta
from enum import StrEnum

import numpy as np
import polars as pl
import pytest

from dataforge_ml import deserialize, inspect, serialize
from dataforge_ml.config import PipelineConfig
from dataforge_ml.profiling._config import ProfileConfig, StructuralProfileResult
from dataforge_ml.profiling._missingness_config import MissingSeverity
from dataforge_ml.profiling.orchestrator import StructuralProfiler


@pytest.fixture(scope="module")
def rich_profile_result() -> StructuralProfileResult:
    """A profile exercising every column-stats family plus dataset-level
    correlation and a numeric target, so the round-trip covers the whole
    object graph."""
    rng = np.random.default_rng(7)
    n = 300

    age = rng.integers(18, 75, size=n)
    income = age * 1200 + rng.normal(0, 5000, size=n)
    salary = rng.normal(50_000, 15_000, size=n).tolist()
    null_mask = rng.random(n) < 0.10
    salary = [None if null_mask[i] else salary[i] for i in range(n)]
    country_choices = ["US", "UK", "CA", "AU", "DE"]
    country = [country_choices[i % len(country_choices)] for i in range(n)]
    is_active = [bool(v) for v in rng.integers(0, 2, size=n)]
    base = date(2020, 1, 1)
    joined = [base + timedelta(days=int(d)) for d in rng.integers(0, 1460, size=n)]
    topics = ["science", "art", "history", "technology", "nature"]
    review = [
        f"A detailed description covering {topics[i % len(topics)]} "
        f"with several words comfortably past the free-text threshold {i}"
        for i in range(n)
    ]

    df = pl.DataFrame(
        {
            "age": pl.Series(age.tolist(), dtype=pl.Int64),
            "income": pl.Series(income.tolist(), dtype=pl.Float64),
            "salary": pl.Series(salary, dtype=pl.Float64),
            "country": pl.Series(country, dtype=pl.Utf8),
            "is_active": pl.Series(is_active, dtype=pl.Boolean),
            "joined": pl.Series(joined, dtype=pl.Date),
            "review": pl.Series(review, dtype=pl.Utf8),
        }
    )

    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_correlation=True,
            target_columns=["income"],
            correlation_target_column="income",
        )
    )
    return StructuralProfiler(config).profile(df)


def test_serialize_deserialize_round_trip_structural_equality(rich_profile_result) -> None:
    loaded = deserialize(serialize(rich_profile_result))

    assert loaded == rich_profile_result
    assert loaded.columns == rich_profile_result.columns
    assert loaded.dataset == rich_profile_result.dataset
    assert loaded.targets == rich_profile_result.targets


def test_serialized_envelope_is_json_native(rich_profile_result) -> None:
    blob = serialize(rich_profile_result)
    # A pure-data envelope holds no opaque payload: json.loads succeeding is the
    # assertion that only JSON-native types crossed the boundary.
    document = json.loads(blob.decode("utf-8"))
    assert document["kind"] == "profile"
    loaded = deserialize(blob)
    assert loaded == rich_profile_result


def test_inspect_reports_kind_and_versions_without_the_payload(
    rich_profile_result,
) -> None:
    header = inspect(serialize(rich_profile_result))
    assert header["kind"] == "profile"
    assert "format_schema_version" in header
    assert "library_version" in header
    assert "data" not in header


def test_enums_are_serialised_by_value_not_ordinal(rich_profile_result) -> None:
    document = json.loads(serialize(rich_profile_result).decode("utf-8"))
    payload = document["data"]

    for col_data in payload["columns"].values():
        assert isinstance(col_data["semantic_type"], (str, type(None)))
        missingness = col_data.get("missingness")
        if missingness and missingness.get("severity") is not None:
            assert missingness["severity"] in {s.value for s in MissingSeverity}


def test_reordering_enum_members_does_not_corrupt_a_saved_document() -> None:
    """
    Demonstrates the mechanism every profiling ``from_dict`` relies on:
    lookup by member *value* (``EnumType(value)``) is invariant to
    declaration order, unlike a hypothetical ordinal/index-based encoding.
    """

    class SeverityV1(StrEnum):
        Minor = "minor"
        Moderate = "moderate"
        High = "high"

    class SeverityV2(StrEnum):
        High = "high"
        Minor = "minor"
        Moderate = "moderate"

    assert list(SeverityV1).index(SeverityV1.High) != list(SeverityV2).index(
        SeverityV2.High
    )

    saved_value = str(SeverityV1.High)
    assert SeverityV2(saved_value) == SeverityV2.High
    assert SeverityV2(saved_value).value == SeverityV1.High.value
