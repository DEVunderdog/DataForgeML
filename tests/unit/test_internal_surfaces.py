"""Persistence, ``DataLoader(fmt=)``, ``FittedUnit.transform`` and the event stream are internal.

Persistence is the Workspace's on-disk format (ADR-0103): no Operation hands
out or takes in artifact bytes, so ``serialize``/``deserialize``/``inspect``
live only in the private ``dataforge_ml._serialization`` module. A Fitted Unit
is applied only through ``FittedImputer.transform``. Pipeline Events reach an
Agent only through the CLI's ``--progress`` stream.
"""

import importlib
import inspect

import pytest

import dataforge_ml
from dataforge_ml.imputation._fitted_imputer import FittedMICE, _FittedKNN
from dataforge_ml.imputation._fitted_units import (
    FittedClusterConditional,
    FittedGMMSampling,
    FittedScalar,
)
from dataforge_ml.utils.data_loader import DataLoader


def test_persistence_functions_are_not_reachable_from_the_package_root():
    for name in ("serialize", "deserialize", "inspect"):
        assert name not in dataforge_ml.__all__
        assert not hasattr(dataforge_ml, name)


def test_data_loader_takes_no_fmt():
    assert "fmt" not in inspect.signature(DataLoader.load).parameters


def test_no_fitted_unit_has_a_public_transform():
    for unit_type in (
        FittedScalar,
        FittedClusterConditional,
        FittedGMMSampling,
        FittedMICE,
        _FittedKNN,
    ):
        assert not hasattr(unit_type, "transform"), unit_type.__name__
        assert callable(unit_type._transform)


def test_pipeline_events_are_not_public_surface():
    for name in ("stderr_observer", "PipelineEvent", "observability"):
        assert name not in dataforge_ml.__all__
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dataforge_ml.observability")
