"""Cheap, retrain-free inspection path for Phase 2 imputation (#329 / ADR-0058).

Drives the public seam — ``EvaluationOrchestrator.inspect(fitted_imputer, train_df)``
— and asserts on externally observable behaviour: a per-column
``InspectionReport`` holding distributional and model-metadata diagnostics for
model-based columns only, a ``variance_ratio`` that catches a collapsed-variance
column, and equivalence to ``FittedImputer.transform`` (proving ``inspect`` is
built on the real application path, not a re-implementation).
"""

from __future__ import annotations

import inspect as _inspect

import polars as pl
import pytest

from dataforge_ml import (
    EvaluationOrchestrator,
    InspectionDiagnostic,
    InspectionReport,
    PipelineConfig,
    StructuralProfiler,
)
from dataforge_ml.profiling._config import ProfileConfig
from tests.conftest import fit_imputer

# ---------------------------------------------------------------------------
# A wide, correlated numeric frame so several columns route to model-based
# strategies (KNN). One independent column is forced to MICE (the unified
# chained-equations block, ADR-0079) to collapse its imputed variance; a scalar
# (Median) and a no-missing (Passthrough) column are present to prove they never
# earn a report entry.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inspect_setup():
    import numpy as np

    rng = np.random.default_rng(7)
    n = 600
    base = rng.normal(0.0, 1.0, n)
    cols: dict[str, pl.Series] = {}
    for name in ("a", "b", "c", "d"):
        vals = (base + rng.normal(0.0, 0.3, n)).tolist()
        for i in range(n):
            if rng.random() < 0.10:
                vals[i] = None
        cols[name] = pl.Series(vals, dtype=pl.Float64)

    # Independent noise, forced to MICE -> near-constant fills -> collapse.
    noise = rng.normal(0.0, 1.0, n).tolist()
    for i in range(n):
        if rng.random() < 0.30:
            noise[i] = None
    cols["noise"] = pl.Series(noise, dtype=pl.Float64)

    # A scalar-strategy column (forced Median) with missingness.
    med = rng.normal(5.0, 2.0, n).tolist()
    for i in range(n):
        if rng.random() < 0.10:
            med[i] = None
    cols["med"] = pl.Series(med, dtype=pl.Float64)

    # A column with no missingness at all -> Passthrough.
    cols["full"] = pl.Series(rng.normal(0.0, 1.0, n).tolist(), dtype=pl.Float64)

    df = pl.DataFrame(cols)

    config = PipelineConfig(profiling=ProfileConfig())
    config.imputation.numeric.set_per_column_strategy("noise", "mice")
    config.imputation.numeric.set_per_column_strategy("med", "median")

    profile = StructuralProfiler(config).profile(df)
    fitted = fit_imputer(df, profile, config)
    return config, df, fitted


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


def test_inspection_types_are_public_api():
    from dataforge_ml import InspectionDiagnostic as D
    from dataforge_ml import InspectionReport as R

    assert R is InspectionReport
    assert D is InspectionDiagnostic


def test_inspect_signature_takes_no_profile():
    params = list(_inspect.signature(EvaluationOrchestrator.inspect).parameters)
    assert params == ["self", "fitted_imputer", "train_df"]
    assert "profile" not in params


# ---------------------------------------------------------------------------
# The report — entries only for model-based columns, populated fields
# ---------------------------------------------------------------------------


def test_inspect_returns_report_keyed_by_model_based_columns(inspect_setup):
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)

    assert isinstance(report, InspectionReport)
    # Model-based columns present; scalar (med) and Passthrough (full) absent.
    assert set(report.columns) == {"a", "b", "c", "d", "noise"}
    assert "med" not in report
    assert "full" not in report

    for col in report.columns:
        diag = report[col]
        assert isinstance(diag, InspectionDiagnostic)
        assert isinstance(diag.imputed_mean, float)
        assert isinstance(diag.imputed_std, float)
        assert isinstance(diag.observed_mean, float)
        assert isinstance(diag.observed_std, float)
        assert isinstance(diag.variance_ratio, float)


def test_inspect_variance_ratio_catches_collapsed_column(inspect_setup):
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)

    # The independent, forced-MICE column collapses to near-constant fills.
    assert report["noise"].variance_ratio < 0.1
    # The genuinely correlated columns retain most of their spread.
    for col in ("a", "b", "c", "d"):
        assert report[col].variance_ratio > 0.5


def test_inspect_model_metadata_read_from_fitted_models(inspect_setup):
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)

    # MICE column exposes IterativeImputer convergence metadata,
    # and no KNN neighbour count.
    noise = report["noise"]
    assert isinstance(noise.converged, bool)
    assert isinstance(noise.n_iter, int)
    assert noise.n_neighbors_used is None

    # KNN columns expose the neighbour count and no convergence metadata.
    knn = report["a"]
    assert isinstance(knn.n_neighbors_used, int)
    assert knn.converged is None
    assert knn.n_iter is None
    assert knn.k_capped in (True, False)


# ---------------------------------------------------------------------------
# Reuse seam — inspect is built on FittedImputer.transform
# ---------------------------------------------------------------------------


def test_inspect_reuses_transform_filled_values(inspect_setup):
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)

    import numpy as np

    from dataforge_ml.utils._null_normalization import _resolve_effective_nulls

    filled = fitted.transform(df).dataframe
    resolved = _resolve_effective_nulls(
        df,
        numeric_sentinels=fitted.numeric_sentinels,
        string_sentinels=fitted.string_sentinels,
    )

    for col in report.columns:
        null_mask = resolved[col].is_null().to_numpy()
        if not null_mask.any():
            continue
        imputed_vals = filled[col].to_numpy()[null_mask]
        assert report[col].imputed_mean == pytest.approx(float(np.mean(imputed_vals)))
        assert report[col].imputed_std == pytest.approx(float(np.std(imputed_vals)))


def test_inspect_needs_no_profile_argument(inspect_setup):
    """inspect runs from the fitted imputer and data alone — no profile."""
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)
    assert isinstance(report, InspectionReport)


def test_inspect_report_round_trips(inspect_setup):
    config, df, fitted = inspect_setup
    report = EvaluationOrchestrator(config).inspect(fitted, df)
    restored = InspectionReport.from_dict(report.to_dict())
    assert set(restored.columns) == set(report.columns)
    for col in report.columns:
        assert restored[col].variance_ratio == report[col].variance_ratio
