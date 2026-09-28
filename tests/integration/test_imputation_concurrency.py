"""Concurrency configuration surface (#326 / ADR-0056).

Thread-based concurrent fitting under the independence rule was previously
exercised here against forced MICE/KNN blocks. Model-based strategies (KNN,
MICE, and the RandomForest core-invariance determinism claim) are the
escalation-point ticket's concern — no routing this ticket's ``route()`` or
``author()`` produces ever carries one — so those cases are retired pending
that ticket. What remains is the concurrency knob itself: a real, validated,
user-facing configuration surface regardless of which strategy consumes it.
"""

from __future__ import annotations

import pytest

from dataforge_ml.imputation import NumericImputationConfig


def test_max_workers_rejects_zero_and_negative():
    with pytest.raises(ValueError):
        NumericImputationConfig(max_workers=0)
    with pytest.raises(ValueError):
        NumericImputationConfig(max_workers=-2)


def test_max_workers_survives_config_round_trip():
    cfg = NumericImputationConfig(max_workers=3)
    assert NumericImputationConfig.from_dict(cfg.to_dict()).max_workers == 3
    # ``None`` (auto) round-trips as well.
    cfg_auto = NumericImputationConfig()
    assert NumericImputationConfig.from_dict(cfg_auto.to_dict()).max_workers is None
