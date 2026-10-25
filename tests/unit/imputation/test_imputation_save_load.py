"""Unit tests for saving and loading ImputationRouting and ImputationRecipe (ADR-0096).

Persisting routing or recipe uses bare-bytes serialization:
- Four independent kinds: profile, routing, recipe, fitted unit.
- Schema version 3 is strictly checked across every kind.
- Gap-fill is deleted: missing or unknown dial keys/rows or unit-id mismatches fail loudly at load.
- Live custom estimators are dropped at save, reloading as ModelChoice.Custom with an empty slot.
- Header inspection stays lightweight and header-only.
"""

from __future__ import annotations

import dataclasses
import json

import polars as pl
import pytest
from sklearn.tree import DecisionTreeRegressor

from dataforge_ml._serialization import deserialize, inspect, serialize
from dataforge_ml.imputation import (
    FittedImputer,
    ImputationConfig,
    ImputationRecipe,
    ImputationRouting,
    ImputationStrategy,
    ModelChoice,
    author,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
)
from dataforge_ml.profiling._config import StructuralProfileResult
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from tests.conftest import raises_code

# --------------------------------------------------------------------------- #
# Fixtures and Helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def rich_profile_and_df() -> tuple[StructuralProfileResult, pl.DataFrame]:
    """DataFrame and profile exercising multiple strategies including MICE, KNN, and scalars."""
    df = pl.DataFrame(
        {
            "m1": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, None] * 10,
            "m2": [2.0, None, 3.0, 4.0, 5.0, None, 7.0, 8.0] * 10,
            "k1": [10.0, 20.0, None, 40.0, 50.0, 60.0, 70.0, 80.0] * 10,
            "k2": [5.0, 15.0, 25.0, None, 45.0, 55.0, 65.0, 75.0] * 10,
            "s1": [1.0, None, 1.0, 1.0, None, 1.0, None, 1.0] * 10,
            "s2": [100.0, 200.0, 300.0, 400.0, 500.0, None, 700.0, 800.0] * 10,
        }
    )
    profile = StructuralProfiler().profile(df)
    return profile, df


@pytest.fixture
def rich_routing(rich_profile_and_df) -> ImputationRouting:
    profile, _ = rich_profile_and_df
    return author(
        {
            "m1": ImputationStrategy.MICE,
            "m2": ImputationStrategy.MICE,
            "k1": ImputationStrategy.KNN,
            "k2": ImputationStrategy.KNN,
            "s1": ImputationStrategy.Mode,
            "s2": ImputationStrategy.Mean,
        },
        profile=profile,
    )


@pytest.fixture
def rich_recipe(rich_routing, rich_profile_and_df) -> ImputationRecipe:
    profile, _ = rich_profile_and_df
    recipe = resolve_recipe(rich_routing, profile)
    # Add an override delta as well to test override persistence
    return dataclasses.replace(
        recipe, hyperparameter_overrides={"mice": (("max_iter", 25),)}
    )


@pytest.fixture
def evidenced_routing() -> tuple[ImputationRouting, StructuralProfileResult]:
    """A routed (not authored) routing whose columns reach the escalation seam."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 400
    base = rng.normal(size=n)
    cols = {
        f"c{i}": (base * (i + 1) + rng.normal(scale=0.3, size=n)).tolist()
        for i in range(4)
    }
    for i, name in enumerate(cols):
        for row in range(i, n, 3):
            cols[name][row] = None
    cols["label"] = (["a", None, "b"] * n)[:n]
    profile = StructuralProfiler().profile(pl.DataFrame(cols))
    return route(profile), profile


def _tamper_json_header(blob: bytes, mutator) -> bytes:
    """Mutate the JSON header of a serialized envelope without touching payload."""
    header, sep, tail = blob.partition(b"\n")
    doc = json.loads(header.decode("utf-8"))
    mutator(doc)
    return json.dumps(doc).encode("utf-8") + (sep + tail if sep else b"")


# --------------------------------------------------------------------------- #
# Acceptance Criterion 1: serialize(routing) -> deserialize round-trips
# --------------------------------------------------------------------------- #


def test_serialize_routing_round_trips_correctly(rich_routing: ImputationRouting) -> None:
    blob = serialize(rich_routing)
    reloaded = deserialize(blob)

    assert isinstance(reloaded, ImputationRouting)
    assert reloaded == rich_routing
    assert reloaded.column_routings == rich_routing.column_routings
    assert reloaded.mice_model_choice == rich_routing.mice_model_choice
    assert reloaded.mice_estimator is None


def test_serialize_recipe_round_trips_correctly(rich_recipe: ImputationRecipe) -> None:
    blob = serialize(rich_recipe)
    reloaded = deserialize(blob)

    assert isinstance(reloaded, ImputationRecipe)
    assert reloaded == rich_recipe
    assert reloaded.routing == rich_recipe.routing
    assert reloaded.decided_hyperparameters == rich_recipe.decided_hyperparameters
    assert reloaded.hyperparameter_overrides == rich_recipe.hyperparameter_overrides
    assert reloaded.column_estimates == rich_recipe.column_estimates
    assert reloaded.numeric_sentinels == rich_recipe.numeric_sentinels
    assert reloaded.string_sentinels == rich_recipe.string_sentinels


def test_routing_and_recipe_envelopes_are_pure_json_native(
    rich_routing: ImputationRouting, rich_recipe: ImputationRecipe
) -> None:
    for obj, expected_kind in [(rich_routing, "routing"), (rich_recipe, "recipe")]:
        blob = serialize(obj)
        # No joblib tail: json.loads succeeding on the full bytes verifies pure JSON
        doc = json.loads(blob.decode("utf-8"))
        assert doc["kind"] == expected_kind
        assert doc["format_schema_version"] == 3
        assert "library_version" in doc
        assert "data" in doc


def test_inspect_routing_and_recipe_is_header_only(
    rich_routing: ImputationRouting, rich_recipe: ImputationRecipe
) -> None:
    for obj, expected_kind in [(rich_routing, "routing"), (rich_recipe, "recipe")]:
        header = inspect(serialize(obj))
        assert header["kind"] == expected_kind
        assert header["format_schema_version"] == 3
        assert "library_version" in header
        assert "data" not in header


def test_routed_evidence_survives_routing_and_recipe_round_trip(evidenced_routing) -> None:
    """Routing Evidence persists on the routing and the recipe's routing copy (ADR-0105)."""
    routing, profile = evidenced_routing
    seam_facts = [
        fact.kind
        for r in routing.column_routings.values()
        for fact in r.evidence
    ]
    assert {"entry", "tier", "feasibility"} <= set(seam_facts)

    reloaded = deserialize(serialize(routing))
    assert reloaded == routing
    for col, r in routing.column_routings.items():
        assert reloaded.column_routings[col].reason == r.reason
        assert reloaded.column_routings[col].evidence == r.evidence

    recipe = resolve_recipe(routing, profile)
    reloaded_recipe = deserialize(serialize(recipe))
    assert reloaded_recipe.routing == routing


def test_routing_with_tampered_evidence_refuses_to_load(evidenced_routing) -> None:
    routing, _ = evidenced_routing
    blob = serialize(routing)

    def corrupt(doc):
        for r in doc["data"]["column_routings"].values():
            r["evidence"] = [{"kind": "tier", "tier": "sideways"}]

    with raises_code("incompatible_artifact"):
        deserialize(_tamper_json_header(blob, corrupt))


# --------------------------------------------------------------------------- #
# Acceptance Criterion 2: v1-schema or kind="decision" raises incompatible_artifact
# --------------------------------------------------------------------------- #


def test_v1_schema_payload_raises_incompatible_artifact_error(
    rich_routing: ImputationRouting, rich_recipe: ImputationRecipe
) -> None:
    for obj in [rich_routing, rich_recipe]:
        blob = serialize(obj)
        tampered = _tamper_json_header(
            blob, lambda h: h.update(format_schema_version=1)
        )
        with raises_code("incompatible_artifact") as exc_info:
            deserialize(tampered)
        msg = str(exc_info.value).lower()
        assert "1" in msg
        assert "obsolete" in msg or "profile" in msg or "resolve" in msg or "fit" in msg


def test_kind_decision_payload_raises_incompatible_artifact_error(
    rich_routing: ImputationRouting,
) -> None:
    blob = serialize(rich_routing)
    tampered = _tamper_json_header(
        blob, lambda h: h.update(kind="decision", format_schema_version=1)
    )
    with raises_code("incompatible_artifact") as exc_info:
        deserialize(tampered)
    msg = str(exc_info.value).lower()
    assert "decision" in msg
    assert "obsolete" in msg
    assert "profile" in msg or "route" in msg or "resolve" in msg or "fit" in msg


def test_kind_decision_schema_v2_payload_also_raises(
    rich_routing: ImputationRouting,
) -> None:
    blob = serialize(rich_routing)
    tampered = _tamper_json_header(blob, lambda h: h.update(kind="decision"))
    with raises_code("incompatible_artifact") as exc_info:
        deserialize(tampered)
    assert "decision" in str(exc_info.value).lower()


# --------------------------------------------------------------------------- #
# Acceptance Criterion 3: Hand-edited recipe missing a dial key raises at load
# --------------------------------------------------------------------------- #


def test_hand_edited_recipe_missing_dial_key_raises_naming_key(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    # Delete 'max_iter' from MICE's decided hyperparameters
    del data["decided_hyperparameters"]["mice"]["max_iter"]

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "max_iter" in err
    assert "missing" in err.lower()
    assert "resolve" in err.lower()


def test_hand_edited_recipe_missing_dial_key_via_deserialize(
    rich_recipe: ImputationRecipe,
) -> None:
    blob = serialize(rich_recipe)
    tampered = _tamper_json_header(
        blob, lambda h: h["data"]["decided_hyperparameters"]["mice"].pop("tol")
    )

    with raises_code("incompatible_artifact") as exc_info:
        deserialize(tampered)

    err = str(exc_info.value)
    assert "tol" in err
    assert "missing" in err.lower()
    assert "resolve" in err.lower()


def test_hand_edited_recipe_unknown_dial_key_raises_naming_key(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    data["decided_hyperparameters"]["mice"]["bogus_dial"] = 999

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "bogus_dial" in err
    assert "unknown" in err.lower()
    assert "resolve" in err.lower()


def test_hand_edited_recipe_unknown_override_dial_key_raises(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    data["hyperparameter_overrides"]["mice"] = {"non_existent": 1}

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "non_existent" in err
    assert "override" in err.lower()


# --------------------------------------------------------------------------- #
# Acceptance Criterion 4: Decided-base unit-id mismatch against derived units raises
# --------------------------------------------------------------------------- #


def test_decided_base_unit_id_mismatch_missing_unit_raises(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    # Delete the entire 'knn' dial row
    del data["decided_hyperparameters"]["knn"]

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "knn" in err
    assert "mismatch" in err.lower()
    assert "resolve" in err.lower()


def test_decided_base_unit_id_mismatch_unexpected_unit_raises(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    data["decided_hyperparameters"]["unexpected_unit"] = {"max_iter": 10}

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "unexpected_unit" in err
    assert "mismatch" in err.lower()
    assert "resolve" in err.lower()


def test_decided_base_unit_id_mismatch_when_routing_is_altered(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    # Change routing so 'm1' and 'm2' are Mean instead of MICE.
    # Now derived units will NOT have 'mice', but decided_hyperparameters has 'mice'.
    data["routing"]["column_routings"]["m1"]["strategy"] = "Mean"
    data["routing"]["column_routings"]["m2"]["strategy"] = "Mean"

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    err = str(exc_info.value)
    assert "mismatch" in err.lower()
    assert "mice" in err


def test_decided_base_missing_entire_hyperparameters_block_raises(
    rich_recipe: ImputationRecipe,
) -> None:
    data = rich_recipe.to_dict()
    del data["decided_hyperparameters"]

    with raises_code("incompatible_artifact") as exc_info:
        ImputationRecipe.from_dict(data)

    assert "decided_hyperparameters" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Acceptance Criterion 5: Round-trip test for a custom MICE estimator
# --------------------------------------------------------------------------- #


def test_custom_estimator_round_trip(
    rich_routing: ImputationRouting,
    rich_profile_and_df: tuple[StructuralProfileResult, pl.DataFrame],
) -> None:
    profile, df = rich_profile_and_df

    custom_estimator = DecisionTreeRegressor(max_depth=3, random_state=42)
    routing_custom = dataclasses.replace(
        rich_routing, mice_model_choice=ModelChoice.Custom, mice_estimator=custom_estimator
    )

    assert routing_custom.mice_model_choice == ModelChoice.Custom
    assert routing_custom.mice_estimator is custom_estimator

    # 1. Round-trip routing:
    blob_routing = serialize(routing_custom)
    reloaded_routing = deserialize(blob_routing)

    assert reloaded_routing.mice_model_choice == ModelChoice.Custom
    assert reloaded_routing.mice_estimator is None

    # 2. Round-trip recipe:
    recipe_custom = resolve_recipe(routing_custom, profile)
    assert recipe_custom.routing.mice_model_choice == ModelChoice.Custom
    assert recipe_custom.routing.mice_estimator is custom_estimator

    blob_recipe = serialize(recipe_custom)
    reloaded_recipe = deserialize(blob_recipe)

    assert reloaded_recipe.routing.mice_model_choice == ModelChoice.Custom
    assert reloaded_recipe.routing.mice_estimator is None

    # 3. Fitting the reloaded recipe fails loudly with clear message:
    (mice_unit,) = derive_units(reloaded_recipe.routing, strategy=ImputationStrategy.MICE)
    with raises_code("unit_not_trainable") as exc_info:
        fit_unit(reloaded_recipe, mice_unit, df)

    reason = exc_info.value.message
    assert "ModelChoice.Custom" in reason
    assert "no estimator is carried" in reason
    assert "mice_estimator" in reason

    # 4. Re-supplying the custom estimator on the routing allows fitting to succeed:
    fixed_routing = dataclasses.replace(
        reloaded_recipe.routing, mice_estimator=custom_estimator
    )
    fixed_recipe = resolve_recipe(fixed_routing, profile)
    (fixed_mice_unit,) = derive_units(fixed_recipe.routing, strategy=ImputationStrategy.MICE)
    fit_result = fit_unit(fixed_recipe, fixed_mice_unit, df)
    assert fit_result.fitted is not None


# --------------------------------------------------------------------------- #
# Whole Imputer Persistence via Recipe + Units
# --------------------------------------------------------------------------- #


def test_whole_imputer_round_trips_via_recipe_plus_units(
    rich_profile_and_df: tuple[StructuralProfileResult, pl.DataFrame],
) -> None:
    profile, df = rich_profile_and_df
    cfg = ImputationConfig()
    routing = route(profile, cfg)
    recipe = resolve_recipe(routing, profile, cfg)
    units = derive_units(routing)

    results = {
        unit.unit_id: fit_unit(recipe, unit, df, seed=7)
        for unit in units
    }

    original = FittedImputer.compose(recipe, results)

    # Persist the recipe itself and each unit as bare bytes
    recipe_bytes = serialize(recipe)
    unit_bytes = {uid: serialize(r.fitted) for uid, r in results.items()}

    restored_recipe = deserialize(recipe_bytes)
    restored_units = {uid: deserialize(b) for uid, b in unit_bytes.items()}
    restored = FittedImputer.compose(restored_recipe, restored_units)

    assert original.transform(df).dataframe.equals(restored.transform(df).dataframe)
