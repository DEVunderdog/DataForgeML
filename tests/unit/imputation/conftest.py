"""Shared fixtures for the imputation unit tests.

Since ADR-0072 persistence is bare bytes: a fitted unit round-trips through
``dataforge_ml.serialize`` / ``deserialize``, and a whole imputer has no
aggregate format — it is persisted as its decision plus its units and rehydrated
through ``FittedImputer.compose``. The ``round_trip`` fixture below mirrors that:
it puts each of an imputer's model-based units through the real serialize/
deserialize boundary (the only binary format) and reassembles the aggregate the
way ``compose`` would, carrying the structural records and sentinel maps across
in-process the way the decision carries them.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def round_trip():
    """Round-trip a FittedImputer through the bare-bytes persistence boundary.

    Each model-based unit is serialized and deserialized via the public
    ``dataforge_ml.serialize`` / ``deserialize`` door; the structural manifest
    (records) and sentinel maps are carried across the way ``compose`` receives
    them from the decision.
    """

    def _round_trip(imputer, key: str = "imputer"):
        from dataforge_ml import deserialize, serialize
        from dataforge_ml.imputation._fitted_imputer import FittedImputer

        restored_units = [deserialize(serialize(unit)) for unit in imputer.units]
        return FittedImputer(
            records=dict(imputer.records),
            units=restored_units,
            numeric_sentinels={k: list(v) for k, v in imputer.numeric_sentinels.items()},
            string_sentinels={k: list(v) for k, v in imputer.string_sentinels.items()},
            random_seed=imputer.random_seed,
        )

    return _round_trip
