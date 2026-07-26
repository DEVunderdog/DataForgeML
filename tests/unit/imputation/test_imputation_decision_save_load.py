"""
Unit tests for serializing / deserializing an ``ImputationDecision`` through the
bare-bytes persistence boundary (ADR-0072).

The plan serialises as a single JSON envelope. ``deserialize(serialize(x))``
must be structurally equal to ``x``, with ``units`` re-derived rather than read
back, every enum reconstructed by name so reordering an enum's members can never
corrupt a previously-serialized envelope, and the hyperparameter-override delta
surviving the round-trip.
"""

import json
from enum import StrEnum

import pytest

from dataforge_ml import (
    ImputationDecision,
    ModelChoice,
    PipelineConfig,
    deserialize,
    inspect,
    serialize,
)
from dataforge_ml.imputation import ImputationStrategy, decide
from dataforge_ml.profiling._config import (
    ColumnProfile,
    NumericKind,
    StructuralProfileResult,
)
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import NonlinearityTag, NumericStats


def _numeric_cp(
    name: str,
    *,
    null_count: int = 20,
    total_rows: int = 100,
    nonlinearity_tag: NonlinearityTag | None = None,
    numeric_kind: NumericKind = NumericKind.Continuous,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ColumnProfile:
    missingness = ColumnMissingnessProfile(
        column=name,
        total_rows=total_rows,
        effective_null_count=null_count,
        effective_null_ratio=null_count / total_rows,
        severity=MissingSeverity.Moderate,
        flags=[],
        correlated_with=[],
    )
    stats = NumericStats(nonlinearity_tag=nonlinearity_tag, min=min_value, max=max_value)
    from dataforge_ml.config import SemanticType

    return ColumnProfile(
        name=name,
        semantic_type=SemanticType.Numeric,
        numeric_kind=numeric_kind,
        missingness=missingness,
        stats=stats,
    )


def _rich_plan() -> ImputationDecision:
    """A plan exercising MICE, KNN, MNAR/indicator, and a
    domain-snapped BoundedDiscrete column, so the round-trip covers every
    field family on ``ColumnImputationDecision``."""
    profile = StructuralProfileResult()
    profile.columns.update(
        {
            "a": _numeric_cp("a", nonlinearity_tag=NonlinearityTag.Linear),
            "b": _numeric_cp("b", nonlinearity_tag=NonlinearityTag.Linear),
            "k": _numeric_cp("k"),
            "r": _numeric_cp("r", nonlinearity_tag=NonlinearityTag.ComplexNonlinear),
            "d": _numeric_cp(
                "d",
                numeric_kind=NumericKind.BoundedDiscrete,
                min_value=0.0,
                max_value=9.0,
            ),
            "n": _numeric_cp("n"),
        }
    )
    profile.dataset.row_count = 10_000

    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(["a", "b", "r"], ImputationStrategy.MICE)
    config.imputation.numeric.set_per_column_strategy("k", ImputationStrategy.KNN)
    config.imputation.numeric.set_per_column_strategy("d", ImputationStrategy.KNN)
    config.imputation.add_mnar_column("n")

    plan = decide(profile, profile.dataset.row_count, config)
    return plan.with_hyperparameters("knn", {"n_neighbors": 7})


def test_serialize_deserialize_round_trip_structural_equality() -> None:
    plan = _rich_plan()

    loaded = deserialize(serialize(plan))

    assert loaded == plan
    assert loaded.units == plan.units
    assert (
        loaded.column_decisions["r"].model_choice
        == ModelChoice.GradientBoostingRegressor
    )
    assert loaded.column_decisions["d"].domain_snap_bounds == (0.0, 9.0)
    assert loaded.column_decisions["n"].mnar is True


def test_override_delta_survives_the_round_trip() -> None:
    plan = _rich_plan()
    assert plan.override_hyperparameters  # the knn override applied in _rich_plan

    loaded = deserialize(serialize(plan))

    assert loaded.override_hyperparameters == plan.override_hyperparameters
    knn_unit = next(u for u in loaded.units if u.unit_id == "knn")
    assert dict(knn_unit.hyperparameters)["n_neighbors"] == 7


def test_serialized_envelope_is_json_native() -> None:
    plan = _rich_plan()
    blob = serialize(plan)
    # A pure-data envelope holds no opaque payload: the whole thing is JSON,
    # so json.loads succeeding is the assertion that only JSON-native types
    # crossed the boundary.
    document = json.loads(blob.decode("utf-8"))
    assert document["kind"] == "decision"
    loaded = deserialize(blob)
    assert loaded == plan


def test_inspect_reports_kind_and_versions_without_the_payload() -> None:
    plan = _rich_plan()
    header = inspect(serialize(plan))
    assert header["kind"] == "decision"
    assert "format_schema_version" in header
    assert "library_version" in header
    assert "data" not in header


def test_enums_are_serialised_by_name_not_ordinal() -> None:
    plan = _rich_plan()
    document = json.loads(serialize(plan).decode("utf-8"))
    payload = document["data"]

    from dataforge_ml.config import SemanticType

    for column, decision in payload["column_decisions"].items():
        assert isinstance(decision["strategy"], str)
        assert decision["strategy"] in ImputationStrategy.__members__
        assert isinstance(decision["semantic_type"], str)
        assert decision["semantic_type"] in SemanticType.__members__
        if decision["model_choice"] is not None:
            assert decision["model_choice"] in ModelChoice.__members__


def test_reordering_enum_members_does_not_corrupt_a_saved_document() -> None:
    """
    Demonstrates the mechanism ``ColumnImputationDecision.from_dict`` /
    ``ImputationUnit.from_dict`` rely on: lookup by member *name*
    (``EnumType[name]``) is invariant to declaration order, unlike a
    hypothetical ordinal/index-based encoding.

    Two StrEnum classes with the same members declared in a different
    order simulate "the enum was reordered after the document was saved".
    A name-keyed lookup still resolves to the correct member and value in
    both; an ordinal-keyed lookup would not.
    """

    class StrategyV1(StrEnum):
        Median = "median"
        MICE = "mice"
        KNN = "knn"

    class StrategyV2(StrEnum):
        KNN = "knn"
        Median = "median"
        MICE = "mice"

    # Ordinal position of MICE differs across the two "versions".
    assert list(StrategyV1).index(StrategyV1.MICE) != list(StrategyV2).index(
        StrategyV2.MICE
    )

    saved_name = StrategyV1.MICE.name
    # A document saved under V1 and read back under V2 (the "reordered"
    # definition) must still resolve to the same logical member and value —
    # exactly what ImputationDecision.to_dict()/from_dict() rely on via
    # `.name` / `EnumType[name]`.
    assert StrategyV2[saved_name] == StrategyV2.MICE
    assert StrategyV2[saved_name].value == StrategyV1.MICE.value
