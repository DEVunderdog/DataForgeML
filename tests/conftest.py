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
    """Fit a FittedImputer over the user-orchestrated route → fit_unit → compose path.

    The imputation door is layered by design (ADR-0071, ADR-0088/0089), so a
    test that only wants "the fitted imputer for this frame" would otherwise
    repeat the four-step drive verbatim. This helper is that drive and nothing
    more: it routes and resolves a recipe off ``profile``, trains every unit in
    a plain sequential loop (batch scheduling is user-owned, ADR-0075), and
    composes the aggregate. Tests asserting on the layers themselves (routing
    contents, per-unit training) should drive the steps directly rather than
    call this.
    """
    from dataforge_ml import (
        FittedImputer,
        PipelineConfig,
        derive_units,
        fit_unit,
        resolve_recipe,
        route,
    )

    config = config or PipelineConfig()
    routing = route(profile, config)
    recipe = resolve_recipe(routing, profile, config)
    units = derive_units(routing)
    results = {
        unit.unit_id: fit_unit(
            recipe, unit, train_df, random_seed=config.random_seed
        )
        for unit in units
    }
    return FittedImputer.compose(recipe, results)
