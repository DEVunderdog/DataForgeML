from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> Callable[..., np.random.Generator]:
    # A factory, deliberately not a generator.  A shared generator is stateful:
    # every draw advances it, so the data a test receives depends on how many
    # tests ran before it and a test can pass in isolation but fail in the full
    # suite.  Handing out a factory makes each caller's stream start from a
    # known seed, so fixture data is a function of the seed alone.  Callers
    # wanting data distinct from another fixture's pass an explicit seed.
    def _make(seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _make


def fit_imputer(train_df, profile, config=None, observer=None):
    """Fit a FittedImputer over the user-orchestrated decide → fit_unit → compose path.

    The imputation door is layered by design (ADR-0060, ADR-0071), so a test that
    only wants "the fitted imputer for this frame" would otherwise repeat the
    three-step drive verbatim. This helper is that drive and nothing more: it
    plans against the *train* row count, trains every unit in a plain sequential
    loop (batch scheduling is user-owned, ADR-0075), and composes the aggregate.
    Tests asserting on the layers themselves (plan contents, per-unit training)
    should drive the steps directly rather than call this.
    """
    from dataforge_ml import FittedImputer, PipelineConfig, decide, fit_unit

    config = config or PipelineConfig()
    plan = decide(profile, len(train_df), config)
    results = {
        unit.unit_id: fit_unit(
            plan, unit, train_df, random_seed=config.random_seed
        )
        for unit in plan.units
    }
    return FittedImputer.compose(plan, results)
