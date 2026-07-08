"""Thread-based concurrent fitting under the independence rule (#326 / ADR-0056).

Drives the highest available seam — ``ImputationOrchestrator.fit()`` — and asserts
on externally observable behaviour: a concurrent fit is result-identical to a
forced-sequential one on the same seed, the concurrency knob is honoured, and
Substep progress still reaches the observer while work runs in parallel.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dataforge_ml import (
    EventType,
    ImputationOrchestrator,
    ImputationStrategy,
    PipelineConfig,
    PipelineEvent,
    StructuralProfiler,
)
from dataforge_ml.imputation import ImputationConfig, NumericImputationConfig
from dataforge_ml.profiling._config import ProfileConfig


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list[PipelineEvent] = []

    def __call__(self, event: PipelineEvent) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# A wide, mixed-strategy dataset: several mutually-independent units of work
# (a joint MICE block plus multiple per-column Regression fits) so the outer
# thread parallelism actually engages.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wide_df(rng):
    n = 300
    base = rng.normal(0.0, 1.0, n)
    cols = {}
    for name in ("a", "b", "c", "d", "e"):
        vals = (base + rng.normal(0.0, 0.5, n)).tolist()
        for i in range(n):
            if rng.random() < 0.10:
                vals[i] = None
        cols[name] = pl.Series(vals, dtype=pl.Float64)
    return pl.DataFrame(cols)


@pytest.fixture(scope="module")
def wide_profile(wide_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(wide_df)


def _mixed_config(max_workers) -> PipelineConfig:
    """Force a joint MICE block and three per-column Regression fits.

    Yields four mutually-independent units of work so both kinds of concurrency
    (independent blocks and independent columns) are exercised at once.
    """
    numeric = NumericImputationConfig(
        regression_min_rows=10,
        max_workers=max_workers,
        _per_column_strategy={
            "a": ImputationStrategy.MICE,
            "b": ImputationStrategy.MICE,
            "c": ImputationStrategy.Regression,
            "d": ImputationStrategy.Regression,
            "e": ImputationStrategy.Regression,
        },
    )
    return PipelineConfig(
        random_seed=7,
        imputation=ImputationConfig(numeric=numeric),
    )


# ---------------------------------------------------------------------------
# Result identity: concurrent vs forced-sequential on the same seed
# ---------------------------------------------------------------------------


def test_concurrent_fit_is_result_identical_to_sequential(wide_df, wide_profile):
    fitted_seq = ImputationOrchestrator(_mixed_config(max_workers=1)).fit(
        wide_df, wide_profile
    )
    fitted_par = ImputationOrchestrator(_mixed_config(max_workers=4)).fit(
        wide_df, wide_profile
    )

    # Same strategy and scalar fill for every column.
    assert set(fitted_seq.records) == set(fitted_par.records)
    for col, rec_seq in fitted_seq.records.items():
        rec_par = fitted_par.records[col]
        assert rec_seq.strategy == rec_par.strategy
        assert rec_seq.fill_value == rec_par.fill_value

    # Same fitted models: transforming the same frame yields identical output
    # (byte-identical imputed values ⇒ identical model parameters).
    out_seq = fitted_seq.transform(wide_df).dataframe
    out_par = fitted_par.transform(wide_df).dataframe
    assert_frame_equal(out_seq, out_par)


def test_concurrent_fit_matches_default_auto_sized(wide_df, wide_profile):
    # The default (max_workers=None) auto-sizes to the CPU count; its result must
    # still match a forced-sequential fit on the same seed.
    fitted_seq = ImputationOrchestrator(_mixed_config(max_workers=1)).fit(
        wide_df, wide_profile
    )
    fitted_auto = ImputationOrchestrator(_mixed_config(max_workers=None)).fit(
        wide_df, wide_profile
    )
    assert_frame_equal(
        fitted_seq.transform(wide_df).dataframe,
        fitted_auto.transform(wide_df).dataframe,
    )


# ---------------------------------------------------------------------------
# The observer stays reachable during concurrent fitting
# ---------------------------------------------------------------------------


def test_substeps_still_emit_during_concurrent_fit(wide_df, wide_profile):
    recorder = _Recorder()
    ImputationOrchestrator(_mixed_config(max_workers=4), observer=recorder).fit(
        wide_df, wide_profile
    )

    substeps = [e for e in recorder.events if e.event_type == EventType.substep]
    assert substeps, "no substep events emitted during the concurrent fit"
    assert all(e.phase == "imputation" for e in substeps)
    assert all(e.stage == "column_fitting" for e in substeps)


def test_items_are_monotonic_and_complete_under_concurrency(wide_df, wide_profile):
    # Even when units fit in parallel, the coarse "k of N" bar stays honest:
    # exactly one item per fitted column, contiguous 1..N index, stable total.
    recorder = _Recorder()
    ImputationOrchestrator(_mixed_config(max_workers=4), observer=recorder).fit(
        wide_df, wide_profile
    )

    items = [
        e
        for e in recorder.events
        if e.event_type == EventType.item and e.stage == "column_fitting"
    ]
    assert {e.column for e in items} == {"a", "b", "c", "d", "e"}
    total = items[0].total
    assert total == len(items) == 5
    assert sorted(e.index for e in items) == list(range(1, 6))
    assert all(e.total == total for e in items)


# ---------------------------------------------------------------------------
# The concurrency knob is a real, validated, user-facing configuration surface
# ---------------------------------------------------------------------------


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
