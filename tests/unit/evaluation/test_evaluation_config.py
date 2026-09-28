"""Round-trip tests for ``C2STConfig``.

Follows the shape of ``tests/unit/imputation/test_imputation_config.py``: the
field list is asserted once so a new dial cannot be added without a decision,
and every dial survives ``to_dict`` -> ``from_dict``.
"""

from dataclasses import fields

from sklearn.dummy import DummyClassifier

from dataforge_ml.evaluation import (
    C2STConfig,
    C2STScheme,
)


def test_c2st_config_field_list_is_the_declared_dial_set():
    assert {f.name for f in fields(C2STConfig)} == {
        "repeats",
        "scheme",
        "min_samples_leaf",
        "min_filled_cells",
        "low_power_below",
        "imbalance_warn_ratio",
        "fdr_alpha",
        "random_state",
        "classifier",
    }


def test_c2st_config_defaults_are_the_measured_ones():
    config = C2STConfig()
    assert config.scheme is C2STScheme.Split
    assert config.min_samples_leaf == 5
    assert config.min_filled_cells == 30
    assert config.low_power_below == 250
    assert config.imbalance_warn_ratio == 0.5
    assert config.fdr_alpha == 0.05
    assert config.classifier is None


def test_c2st_config_round_trips_every_dial():
    config = C2STConfig(
        repeats=9,
        scheme=C2STScheme.CrossValidation,
        min_samples_leaf=7,
        min_filled_cells=40,
        low_power_below=300,
        imbalance_warn_ratio=0.25,
        fdr_alpha=0.10,
        random_state=11,
    )

    restored = C2STConfig.from_dict(config.to_dict())

    assert restored == config


def test_from_dict_falls_back_to_defaults_on_missing_keys():
    assert C2STConfig.from_dict({}) == C2STConfig()


def test_the_live_classifier_is_not_carried_across_the_round_trip():
    config = C2STConfig(classifier=DummyClassifier())

    assert "classifier" not in config.to_dict()
    assert C2STConfig.from_dict(config.to_dict()).classifier is None


def test_c2st_config_carries_no_observer_field():
    # A live callable would break the serialisable, setter-only contract
    # (ADR-0044), so ``observer=`` rides the call instead.
    assert "observer" not in {f.name for f in fields(C2STConfig)}
