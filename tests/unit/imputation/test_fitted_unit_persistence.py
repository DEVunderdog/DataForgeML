"""Unit tests for fitted-unit bare-bytes persistence (ADR-0072).

A fitted unit serializes as a readable JSON header (unit type, owned columns,
the ``produced_with`` six-version stamp, and the payload's SHA-256 checksum)
followed by a *separate* opaque joblib payload — never base64-inlined into the
header. ``deserialize(serialize(unit)).transform(X)`` is bit-identical to
``unit.transform(X)`` on equal input, and any reconstruction-critical version
mismatch (or a corrupt payload) refuses loudly. ``inspect`` reads the header
without unpickling.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

from dataforge_ml import (
    ArtifactPythonVersionWarning,
    IncompatibleArtifactError,
    deserialize,
    inspect,
    serialize,
)
from dataforge_ml.imputation._config import ImputationStrategy
from dataforge_ml.imputation._fitted_imputer import FittedMICE, _FittedKNN
from dataforge_ml.imputation._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)

# --------------------------------------------------------------------------- #
# Byte-level tampering helpers: a serialized unit is a JSON header line, a
# newline, then the opaque joblib payload.
# --------------------------------------------------------------------------- #


def _split(blob: bytes) -> tuple[dict, bytes]:
    header, _, payload = blob.partition(b"\n")
    return json.loads(header.decode("utf-8")), payload


def _rejoin(header: dict, payload: bytes) -> bytes:
    return json.dumps(header).encode("utf-8") + b"\n" + payload


def _tamper_header(blob: bytes, mutate) -> bytes:
    header, payload = _split(blob)
    mutate(header)
    return _rejoin(header, payload)


# --------------------------------------------------------------------------- #
# Unit factories: each returns a genuinely fitted unit plus a DataFrame that
# exercises its transform (with missing values to fill).
# --------------------------------------------------------------------------- #


def _knn_unit() -> tuple[_FittedKNN, pl.DataFrame]:
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(60, 3))
    cols = ["a", "b", "c"]
    col_means = np.nanmean(arr, axis=0)
    col_stds = np.nanstd(arr, axis=0)
    col_stds[col_stds == 0.0] = 1.0
    model = KNNImputer(n_neighbors=3).fit((arr - col_means) / col_stds)
    unit = _FittedKNN(
        model=model, col_means=col_means, col_stds=col_stds, columns=cols
    )
    df = pl.DataFrame(
        {
            "a": [1.0, None, 3.0, None, 5.0],
            "b": [2.0, 4.0, None, 8.0, 10.0],
            "c": [None, 1.5, 2.5, 3.5, None],
        }
    )
    return unit, df


def _gmm_unit() -> tuple[FittedGMMSampling, pl.DataFrame]:
    unit = FittedGMMSampling(
        center1=1.0,
        center2=9.0,
        std1=0.5,
        std2=0.7,
        weight1=0.4,
        weight2=0.6,
        target_col="g",
        random_seed=7,
    )
    df = pl.DataFrame({"g": [1.2, None, 8.9, None, None]})
    return unit, df


def _scalar_unit() -> tuple[FittedScalar, pl.DataFrame]:
    unit = FittedScalar(
        target_col="s", fill_value=42.0
    )
    df = pl.DataFrame({"s": [1.0, None, 3.0, None]})
    return unit, df


def _mice_unit() -> tuple[FittedMICE, pl.DataFrame]:
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(50, 2))
    model = IterativeImputer(estimator=BayesianRidge(), random_state=0).fit(arr)
    unit = FittedMICE(model=model, columns=["a", "b"])
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [2.0, 4.0, None]})
    return unit, df


def _cluster_unit() -> tuple[FittedClusterConditional, pl.DataFrame]:
    unit = FittedClusterConditional(
        grouping_variable="g",
        group_fills={"x": 1.0, "y": 2.0},
        fill_1=None,
        fill_2=None,
        feature_centroid_1=None,
        feature_centroid_2=None,
        feature_cols=None,
        center1=1.0,
        center2=2.0,
        target_col="c",
    )
    df = pl.DataFrame({"c": [None, 1.0, None], "g": ["x", "y", "y"]})
    return unit, df


_FACTORIES = {
    "knn": _knn_unit,
    "gmm": _gmm_unit,
    "scalar": _scalar_unit,
    "mice": _mice_unit,
    "cluster": _cluster_unit,
}


@pytest.fixture(params=list(_FACTORIES), ids=list(_FACTORIES))
def unit_and_df(request):
    return _FACTORIES[request.param]()


# --------------------------------------------------------------------------- #
# Envelope shape
# --------------------------------------------------------------------------- #


def test_serialize_produces_json_header_and_separate_opaque_payload(unit_and_df) -> None:
    unit, _ = unit_and_df
    blob = serialize(unit)
    header, payload = _split(blob)
    assert header["kind"] == "fitted_unit"
    # The learned state lives in the opaque tail, not the header.
    assert payload
    assert "model" not in header


def test_inspect_header_is_readable_and_carries_no_base64_model(unit_and_df) -> None:
    unit, _ = unit_and_df
    header = inspect(serialize(unit))
    # The header is fully inspectable without touching the payload, and no field
    # inlines the model as base64 (the pattern this replaces).
    assert "model" not in header
    assert "base64" not in json.dumps(header)
    assert isinstance(header["columns"], list)
    assert header["unit_type"] == type(unit).__name__


def test_inspect_carries_produced_with_six_version_block(unit_and_df) -> None:
    unit, _ = unit_and_df
    pw = inspect(serialize(unit))["produced_with"]
    assert set(pw) == {
        "format_schema_version",
        "dataforge_ml",
        "scikit_learn",
        "numpy",
        "joblib",
        "python",
    }
    assert pw["scikit_learn"] != "unknown"
    assert pw["numpy"] != "unknown"


def test_header_records_payload_checksum(unit_and_df) -> None:
    unit, _ = unit_and_df
    header, payload = _split(serialize(unit))
    assert header["payload_sha256"] == hashlib.sha256(payload).hexdigest()


def test_persistence_module_documents_trust_boundary() -> None:
    """The trust boundary is documented alongside the decode path (issue #360).

    Reconstruction unpickles arbitrary code, so both the module and the decode
    function must state the boundary: only deserialize trusted artifacts, and the
    checksum is an integrity — not authenticity — guard.
    """
    import dataforge_ml._serialization as persistence

    for doc in (
        persistence.__doc__,
        persistence._decode_fitted_unit.__doc__,
        deserialize.__doc__,
    ):
        assert doc is not None
        lowered = " ".join(doc.lower().split())
        assert "trust" in lowered
        assert "arbitrary code" in lowered


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_deserialize_reconstructs_bit_identical_transform(unit_and_df) -> None:
    unit, df = unit_and_df
    loaded = deserialize(serialize(unit))
    assert type(loaded) is type(unit)
    assert loaded.transform(df).equals(unit.transform(df))


# --------------------------------------------------------------------------- #
# Strict load gates
# --------------------------------------------------------------------------- #


def test_corrupt_payload_refuses_to_deserialize() -> None:
    unit, _ = _scalar_unit()
    blob = serialize(unit)
    with pytest.raises(IncompatibleArtifactError, match="checksum"):
        deserialize(blob + b"garbage")


def test_dependency_version_mismatch_refuses_to_deserialize() -> None:
    unit, _ = _scalar_unit()
    tampered = _tamper_header(
        serialize(unit), lambda h: h["produced_with"].update(numpy="0.0.0-fake")
    )
    with pytest.raises(IncompatibleArtifactError, match="numpy"):
        deserialize(tampered)


def test_newer_format_schema_refuses_to_deserialize() -> None:
    unit, _ = _scalar_unit()
    tampered = _tamper_header(
        serialize(unit),
        lambda h: h["produced_with"].update(format_schema_version=999),
    )
    with pytest.raises(IncompatibleArtifactError, match="format-schema"):
        deserialize(tampered)


def test_python_version_drift_warns_but_loads() -> None:
    unit, df = _scalar_unit()
    # Python drift is tolerated (pickle protocol is stable across CPython
    # minors), so the load succeeds and transforms identically — but it warns.
    tampered = _tamper_header(
        serialize(unit), lambda h: h["produced_with"].update(python="1.2.3")
    )
    with pytest.warns(ArtifactPythonVersionWarning, match="1.2.3"):
        loaded = deserialize(tampered)
    assert loaded.transform(df).equals(unit.transform(df))


def test_unrecognised_kind_refuses_to_deserialize() -> None:
    unit, _ = _scalar_unit()
    tampered = _tamper_header(serialize(unit), lambda h: h.update(kind="mystery"))
    with pytest.raises(IncompatibleArtifactError, match="kind"):
        deserialize(tampered)


def test_genuine_joblib_load_failure_propagates() -> None:
    unit, _ = _scalar_unit()
    header, _ = _split(serialize(unit))
    # Replace the payload with bytes that pass the checksum gate (header updated
    # to match) but are not a valid joblib stream. The reconstruction gates let
    # it through; the raw joblib.load failure must propagate rather than be
    # swallowed or masquerade as IncompatibleArtifactError.
    garbage = b"not a joblib payload"
    header["payload_sha256"] = hashlib.sha256(garbage).hexdigest()
    tampered = _rejoin(header, garbage)

    with pytest.raises(Exception) as excinfo:
        deserialize(tampered)
    assert not isinstance(excinfo.value, IncompatibleArtifactError)


# --------------------------------------------------------------------------- #
# The whole imputer: no aggregate format — decision + units re-composed
#
# There is no aggregate serialize format (ADR-0072). A whole imputer persists as
# its decision plus its units, each a single serialize blob, and rehydrates
# through FittedImputer.compose.
# --------------------------------------------------------------------------- #


def test_whole_imputer_round_trips_via_decision_plus_units() -> None:
    from dataforge_ml import PipelineConfig, StructuralProfiler
    from dataforge_ml.imputation import FittedImputer, decide, fit_unit

    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0, 5.0, None, 7.0, 8.0] * 8,
            "b": [10.0, None, 30.0, 40.0, 50.0, 60.0, None, 80.0] * 8,
        }
    )
    cfg = PipelineConfig()
    profile = StructuralProfiler(cfg).profile(df)
    plan = decide(profile, len(df), cfg)
    results = {
        unit.unit_id: fit_unit(plan, unit.unit_id, df, random_seed=7)
        for unit in plan.units
    }

    original = FittedImputer.compose(plan, results)

    # Persist the plan and each unit as bytes, then rehydrate through compose.
    plan_bytes = serialize(plan)
    unit_bytes = {uid: serialize(r.fitted) for uid, r in results.items()}
    restored_plan = deserialize(plan_bytes)
    restored_units = {uid: deserialize(b) for uid, b in unit_bytes.items()}
    restored = FittedImputer.compose(restored_plan, restored_units)

    assert original.transform(df).dataframe.equals(restored.transform(df).dataframe)
