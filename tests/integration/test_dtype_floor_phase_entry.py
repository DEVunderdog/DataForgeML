"""The Dtype Floor is enforced at every phase entry (#508 / ADR-0085).

The floor is *constructive*: nothing asserts it and there is no public
predicate, so the guarantee is only as good as the wiring. Each test below
drives one phase through its real public door and inspects the frame the
normaliser was handed and the frame it returned, pinning both halves of the
contract at that entry — the cast is ordered **after** effective-null
resolution, and every non-string semantic type arrives materialised.

``FittedImputer.transform`` additionally has a public boundary to defend: the
floor is a detail of the working copy, so the caller gets its columns back at
their original dtype (ADR-0078).
"""

from __future__ import annotations

import polars as pl
import pytest

from dataforge_ml import (
    PipelineConfig,
    StructuralProfiler,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
)
from tests.conftest import fit_imputer

# ---------------------------------------------------------------------------
# One frame covering every branch of the rule: a string Categorical carrying a
# sentinel (the ordering probe), a boolean-string column, a datetime-string
# column, free text, an int-coded categorical, and numerics with holes so the
# imputation phases have something to fit.
# ---------------------------------------------------------------------------

_N = 90


@pytest.fixture(scope="module")
def floor_setup():
    import numpy as np

    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 1.0, _N)

    def _holed(vals):
        out = list(vals)
        for i in range(3, _N, 9):
            out[i] = None
        return out

    df = pl.DataFrame(
        {
            "cat": pl.Series(
                [("NA" if i % 10 == 0 else "abc"[i % 3]) for i in range(_N)],
                dtype=pl.String,
            ),
            "flag": pl.Series(
                ["yes" if i % 2 else "no" for i in range(_N)], dtype=pl.String
            ),
            "when": pl.Series(
                [f"2024-{(i % 12) + 1:02d}-05" for i in range(_N)], dtype=pl.String
            ),
            "notes": pl.Series(
                [f"a free text note number {i} with words" for i in range(_N)],
                dtype=pl.String,
            ),
            "coded": pl.Series([i % 4 for i in range(_N)], dtype=pl.Int64),
            "x": pl.Series(_holed(base), dtype=pl.Float64),
            "y": pl.Series(
                _holed(base + rng.normal(0.0, 0.2, _N)), dtype=pl.Float64
            ),
        }
    )
    config = PipelineConfig()
    profile = StructuralProfiler(config).profile(df)
    return config, df, profile


def _semantic_types(profile):
    return {c: cp.semantic_type for c, cp in profile.columns.items()}


def _assert_floor_holds(frame_in: pl.DataFrame, frame_out: pl.DataFrame) -> None:
    """The two halves of the contract at one phase entry."""
    # Ordering: the sentinel "NA" was already resolved before the cast — had
    # the floor run first it would survive as a Categorical value.
    assert frame_in["cat"].dtype == pl.String
    assert frame_in["cat"].null_count() > 0
    assert "NA" not in frame_in["cat"].to_list()

    # Round trip: each non-string semantic type arrives materialised.
    assert frame_out["cat"].dtype == pl.Categorical
    assert frame_out["flag"].dtype == pl.Boolean
    assert frame_out["when"].dtype == pl.Datetime
    # Text stays String; the int-coded categorical stays pl.Int*.
    assert frame_out["notes"].dtype == pl.String
    assert frame_out["coded"].dtype == pl.Int64


@pytest.fixture
def capture_floor(monkeypatch):
    """Wrap ``_apply_dtype_floor`` in one module and record what flowed through."""

    def _capture(module_path: str):
        import importlib

        from dataforge_ml.utils._dtype_floor import _apply_dtype_floor

        module = importlib.import_module(module_path)
        calls: list[tuple[pl.DataFrame, pl.DataFrame]] = []

        def _spy(df, semantic_types):
            out = _apply_dtype_floor(df, semantic_types)
            calls.append((df, out))
            return out

        monkeypatch.setattr(module, "_apply_dtype_floor", _spy)
        return calls

    return _capture


# ---------------------------------------------------------------------------
# The four phase entries
# ---------------------------------------------------------------------------


def test_profiling_orchestrator_applies_the_floor_after_null_resolution(
    floor_setup, capture_floor
):
    config, df, _ = floor_setup
    calls = capture_floor("dataforge_ml.profiling.orchestrator")

    StructuralProfiler(config).profile(df)

    assert calls, "the profiling orchestrator never applied the floor"
    _assert_floor_holds(*calls[0])


def test_fit_unit_applies_the_floor_after_null_resolution(
    floor_setup, capture_floor
):
    config, df, profile = floor_setup
    calls = capture_floor("dataforge_ml.imputation._unit_fit")

    routing = route(profile, config)
    recipe = resolve_recipe(routing, profile, config)
    units = derive_units(routing)
    fit_unit(recipe, units[0], df, random_seed=config.random_seed)

    assert calls, "fit_unit never applied the floor"
    _assert_floor_holds(*calls[0])


def test_fitted_imputer_transform_applies_the_floor_after_null_resolution(
    floor_setup, capture_floor
):
    config, df, profile = floor_setup
    fitted = fit_imputer(df, profile, config)
    calls = capture_floor("dataforge_ml.imputation._fitted_imputer")

    fitted.transform(df)

    assert calls, "FittedImputer.transform never applied the floor"
    _assert_floor_holds(*calls[0])


def test_splitting_profile_signals_applies_the_floor_after_null_resolution(
    floor_setup, capture_floor
):
    _, df, profile = floor_setup
    calls = capture_floor("dataforge_ml.splitting._profile_signals")

    from dataforge_ml.splitting._profile_signals import build_label_matrix

    build_label_matrix(df, profile, target=None)

    assert calls, "build_label_matrix never applied the floor"
    _assert_floor_holds(*calls[0])


# ---------------------------------------------------------------------------
# ADR-0078: the floor stops at the public boundary
# ---------------------------------------------------------------------------


def test_transform_returns_floored_columns_at_their_original_dtype(floor_setup):
    config, df, profile = floor_setup
    fitted = fit_imputer(df, profile, config)

    out = fitted.transform(df).dataframe

    for col in ("cat", "flag", "when", "notes"):
        assert out[col].dtype == df[col].dtype
    assert out["coded"].dtype == pl.Int64


def test_transform_returns_observed_cells_of_floored_columns_bit_for_bit(
    floor_setup,
):
    config, df, profile = floor_setup
    fitted = fit_imputer(df, profile, config)

    out = fitted.transform(df).dataframe

    # "NA" is an Effective Null and is normalised; every other cell is the
    # caller's own value, untouched by the round trip through pl.Categorical.
    expected = [None if v == "NA" else v for v in df["cat"].to_list()]
    assert out["cat"].to_list() == expected
    assert out["flag"].to_list() == df["flag"].to_list()
    assert out["when"].to_list() == df["when"].to_list()
