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

import polars as pl
import pytest
from sklearn.tree import DecisionTreeRegressor

from dataforge_ml import (
    ImputationDecision,
    ModelChoice,
    PipelineConfig,
    author,
    deserialize,
    fit_unit,
    inspect,
    serialize,
)
from dataforge_ml.imputation import (
    ImputationStrategy,
    ImputationUnit,
    UnitNotTrainableError,
    decide,
)
from dataforge_ml.imputation._config import _STRATEGY_DIALS
from dataforge_ml.imputation._regression_estimator_factory import (
    RegressionEstimatorFactory,
)
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
    """Structural equality across the round trip.

    Carve-out: a plan carrying a **user-supplied estimator** is the one
    exception, and deliberately so — the estimator is never written, so the
    reloaded plan is genuinely a different (unfittable) plan and compares
    unequal (ADR-0083). See
    ``test_a_custom_plan_compares_unequal_to_its_own_round_trip``. ``_rich_plan``
    carries no such estimator, so equality holds here.
    """
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


def test_retired_fields_are_absent_from_the_payload() -> None:
    """The three unread fields leave no trace on the wire (ADR-0083).

    The round trip itself is unchanged — asserted by the tests above — so what
    is left to pin is that the plan no longer writes a shape, a provenance, or a
    per-column forced flag anyone could come to depend on.
    """
    document = json.loads(serialize(_rich_plan()).decode("utf-8"))
    data = document["data"]

    assert "decided_for_shape" not in data
    assert "profile_provenance" not in data
    assert all("forced" not in d for d in data["column_decisions"].values())


# ---------------------------------------------------------------------------
# A user-supplied estimator is never persisted (#469, ADR-0083)
# ---------------------------------------------------------------------------


def _custom_plan() -> ImputationDecision:
    """A hand-authored MICE plan holding the author's own estimator."""
    return author(
        {"a": ImputationStrategy.MICE, "b": ImputationStrategy.MICE},
        columns=["a", "b"],
        estimators={"mice": DecisionTreeRegressor(max_depth=3, random_state=0)},
    )


def test_the_estimator_is_absent_from_to_dict_and_from_the_wire() -> None:
    plan = _custom_plan()
    assert plan.custom_estimators  # it is on the plan to begin with

    assert "custom_estimators" not in plan.to_dict()
    document = json.loads(serialize(plan).decode("utf-8"))
    assert "custom_estimators" not in document["data"]
    # And the envelope is still pure JSON — nothing was pickled in instead.
    assert json.dumps(document)


def test_a_reloaded_plan_keeps_the_custom_label_with_an_empty_slot() -> None:
    loaded = deserialize(serialize(_custom_plan()))

    assert loaded.column_decisions["a"].model_choice == ModelChoice.Custom
    assert loaded.column_decisions["b"].model_choice == ModelChoice.Custom
    assert loaded.custom_estimators == {}


def test_build_from_choice_raises_on_custom() -> None:
    """The backstop: nothing constructs an estimator for a label it did not build."""
    with pytest.raises(ValueError, match="user-supplied estimator"):
        RegressionEstimatorFactory.build_from_choice(ModelChoice.Custom)


def test_fitting_a_reloaded_custom_plan_fails_loudly_rather_than_falling_back() -> None:
    loaded = deserialize(serialize(_custom_plan()))
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0, 6.0, None, 8.0],
            "b": [2.0, 4.0, 6.0, None, 10.0, 12.0, 14.0, 16.0],
        }
    )

    with pytest.raises(UnitNotTrainableError, match="estimators="):
        fit_unit(loaded, "mice", df)


def test_a_custom_plan_compares_unequal_to_its_own_round_trip() -> None:
    """The one round-trip carve-out, and it is the truth (ADR-0083).

    The restored plan cannot fit, so calling it equal to the original would be
    the lie. Everything *else* about it survives — which is what makes the
    inequality attributable to the dropped estimator alone.
    """
    plan = _custom_plan()
    loaded = deserialize(serialize(plan))

    assert loaded != plan
    assert loaded.column_decisions == plan.column_decisions
    assert loaded.units == plan.units
    assert loaded.decided_hyperparameters == plan.decided_hyperparameters


def test_from_dict_gap_fills_a_dial_the_payload_is_missing() -> None:
    """An old artifact meeting a newly-added dial keeps running (#464).

    Gap-fill on load is what makes "the base is complete for the unit's
    strategy" true of the type rather than of the authoring functions, which is
    what licenses the fitters' bare ``hyp["max_iter"]`` subscripts. The cost is
    knowingly taken: the reloaded plan silently gains a dial it was never saved
    with.
    """
    document = _rich_plan().to_dict()
    del document["decided_hyperparameters"]["mice"]["max_iter"]

    loaded = ImputationDecision.from_dict(document)

    assert (
        dict(loaded.decided_hyperparameters["mice"])["max_iter"]
        == _STRATEGY_DIALS[ImputationStrategy.MICE]["max_iter"]
    )


def test_from_dict_never_overwrites_a_dial_the_payload_carries() -> None:
    """Fill what is missing; never touch what is carried (#464)."""
    plan = _rich_plan()
    document = plan.to_dict()
    document["decided_hyperparameters"]["mice"]["max_iter"] = 99

    loaded = ImputationDecision.from_dict(document)

    assert dict(loaded.decided_hyperparameters["mice"])["max_iter"] == 99
    # A complete payload round-trips untouched — gap-fill is a no-op on it.
    assert ImputationDecision.from_dict(plan.to_dict()) == plan


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


# ---------------------------------------------------------------------------
# is_block is written but re-derived, never trusted off the wire (#431)
# ---------------------------------------------------------------------------


def test_to_dict_writes_is_block_for_every_unit() -> None:
    plan = _rich_plan()
    document = json.loads(serialize(plan).decode("utf-8"))
    units = {u["unit_id"]: u for u in document["data"]["units"]}
    assert units["mice"]["is_block"] is True
    assert units["knn"]["is_block"] is True
    assert all("is_block" in u for u in units.values())
    assert not any(
        u["is_block"] for uid, u in units.items() if uid not in {"mice", "knn"}
    )


def test_is_block_absent_from_payload_round_trips_to_derived_value() -> None:
    """A plan written before the field existed still loads with it correct."""
    plan = _rich_plan()
    mice = next(u for u in plan.units if u.unit_id == "mice")
    legacy = mice.to_dict()
    del legacy["is_block"]

    assert ImputationUnit.from_dict(legacy) == mice


def test_wrong_is_block_in_payload_is_ignored_on_load() -> None:
    """A derivable fact cannot be poisoned by a hand-edited artifact."""
    plan = _rich_plan()
    mice = next(u for u in plan.units if u.unit_id == "mice")
    scalar = next(u for u in plan.units if u.unit_id.startswith("mnar:"))

    tampered_block = mice.to_dict() | {"is_block": False}
    tampered_scalar = scalar.to_dict() | {"is_block": True}

    assert ImputationUnit.from_dict(tampered_block).is_block is True
    assert ImputationUnit.from_dict(tampered_scalar).is_block is False
