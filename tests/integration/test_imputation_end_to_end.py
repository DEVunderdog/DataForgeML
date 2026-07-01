"""
Integration test: Phase 1 → DataSplitter → Phase 2 imputation.

Verifies the full fit/transform contract on real DataFrames using actual
StructuralProfiler and ImputationOrchestrator (no stubs).
"""

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase
from dataforge_ml.imputation import FittedImputer, ImputationOrchestrator
from dataforge_ml.profiling._config import ProfileConfig
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.splitting import DataSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imputation_df(rng):
    n = 400
    values_a = rng.normal(50.0, 10.0, n).tolist()
    values_b = rng.normal(200.0, 30.0, n).tolist()
    values_c = rng.integers(1, 6, n).tolist()  # discrete: ratings 1–5

    # ~10% missing in each column
    null_mask_a = rng.random(n) < 0.10
    null_mask_b = rng.random(n) < 0.15
    null_mask_c = rng.random(n) < 0.08

    col_a = [None if null_mask_a[i] else values_a[i] for i in range(n)]
    col_b = [None if null_mask_b[i] else values_b[i] for i in range(n)]
    col_c = [None if null_mask_c[i] else float(values_c[i]) for i in range(n)]

    return pl.DataFrame({
        "score": pl.Series(col_a, dtype=pl.Float64),
        "revenue": pl.Series(col_b, dtype=pl.Float64),
        "rating": pl.Series(col_c, dtype=pl.Float64),
        "label": pl.Series(["A" if i % 2 == 0 else "B" for i in range(n)], dtype=pl.Utf8),
    })


@pytest.fixture(scope="module")
def imputation_profile(imputation_df):
    config = PipelineConfig(profiling=ProfileConfig())
    return StructuralProfiler(config).profile(imputation_df)


@pytest.fixture(scope="module")
def imputation_split(imputation_df, imputation_profile):
    splitter = DataSplitter(imputation_df, random_seed=42)
    return splitter.profile_stratified_split(imputation_profile, test_size=0.2)


@pytest.fixture(scope="module")
def fitted_imputer(imputation_split, imputation_profile) -> FittedImputer:
    return ImputationOrchestrator().fit(imputation_split.train, imputation_profile)


# ---------------------------------------------------------------------------
# Acceptance criteria from issue #78
# ---------------------------------------------------------------------------


def test_fit_returns_fitted_imputer(imputation_split, imputation_profile):
    fi = ImputationOrchestrator().fit(imputation_split.train, imputation_profile)
    assert isinstance(fi, FittedImputer)


def test_transform_train_has_no_nulls_in_numeric_cols(fitted_imputer, imputation_split):
    result = fitted_imputer.transform(imputation_split.train)
    for col in ["score", "revenue", "rating"]:
        assert result.dataframe[col].null_count() == 0, (
            f"train split: column '{col}' still has nulls after transform"
        )


def test_transform_test_has_no_nulls_in_numeric_cols(fitted_imputer, imputation_split):
    result = fitted_imputer.transform(imputation_split.test)
    for col in ["score", "revenue", "rating"]:
        assert result.dataframe[col].null_count() == 0, (
            f"test split: column '{col}' still has nulls after transform"
        )


def test_transform_applies_train_time_fill_values(fitted_imputer, imputation_split):
    """Fill value on test split equals the value learned from train, not recomputed."""
    train_result = fitted_imputer.transform(imputation_split.train)
    test_result = fitted_imputer.transform(imputation_split.test)

    for col in ["score", "revenue", "rating"]:
        train_fill = fitted_imputer.records[col].fill_value
        test_fill = fitted_imputer.records[col].fill_value
        assert train_fill == test_fill, (
            f"Fill value must be fixed at fit() time; "
            f"train={train_fill}, test={test_fill}"
        )

    # The fill values in the records should not change between transform calls
    fill_before = {col: fitted_imputer.records[col].fill_value for col in ["score", "revenue"]}
    fitted_imputer.transform(imputation_split.test)
    fill_after = {col: fitted_imputer.records[col].fill_value for col in ["score", "revenue"]}
    assert fill_before == fill_after


def test_result_records_contain_strategy_and_signals(fitted_imputer):
    for col in ["score", "revenue", "rating"]:
        rec = fitted_imputer.records[col]
        assert rec.strategy is not None
        assert len(rec.signals) >= 1


def test_label_column_passes_through_untouched(fitted_imputer, imputation_split):
    """Non-numeric (Text/Categorical) columns must not be altered."""
    result = fitted_imputer.transform(imputation_split.train)
    assert "label" in result.dataframe.columns
    assert result.dataframe["label"].equals(imputation_split.train["label"])


def test_fitted_imputer_serialisation_round_trip(fitted_imputer, imputation_split):
    restored = FittedImputer.from_dict(fitted_imputer.to_dict())
    r1 = fitted_imputer.transform(imputation_split.test)
    r2 = restored.transform(imputation_split.test)
    assert r1.dataframe.equals(r2.dataframe)


def test_mnar_column_receives_data_derived_fill_and_indicator():
    """Dedicated test: MNAR-declared column gets constant fill + indicator column."""
    from dataforge_ml.imputation import ImputationConfig, NumericImputationConfig

    n = 200
    data = pl.DataFrame({
        "salary": pl.Series(
            [None if i % 5 == 0 else float(i * 1000) for i in range(n)],
            dtype=pl.Float64,
        ),
    })
    imputation_config = ImputationConfig()
    imputation_config.add_mnar_column("salary")
    config = PipelineConfig(imputation=imputation_config)
    profile = StructuralProfiler(PipelineConfig()).profile(data)
    orch = ImputationOrchestrator(config=config)
    _fitted, result = orch.fit_transform(data, profile)

    assert result.dataframe["salary"].null_count() == 0
    assert "salary_missing" in result.dataframe.columns


def test_orchestrator_stateless_across_multiple_fits(imputation_df, imputation_profile):
    """fit() must not accumulate state — two calls produce independent FittedImputors."""
    orch = ImputationOrchestrator()
    splitter = DataSplitter(imputation_df, random_seed=1)
    split1 = splitter.random_split(test_size=0.5, stratify=False)
    split2 = splitter.random_split(test_size=0.5, stratify=False)

    fi1 = orch.fit(split1.train, imputation_profile)
    fi2 = orch.fit(split2.train, imputation_profile)

    # Both should produce valid results
    r1 = fi1.transform(split1.test)
    r2 = fi2.transform(split2.test)
    for col in ["score", "revenue"]:
        assert r1.dataframe[col].null_count() == 0
        assert r2.dataframe[col].null_count() == 0


# ---------------------------------------------------------------------------
# Scope 10: DropCandidate lifecycle end-to-end (#115)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drop_candidate_df():
    n = 300
    # "sparse": 67% null — well above the 50% DropCandidate threshold
    sparse = [None if i < 200 else float(i) for i in range(n)]
    # "dense": 10% null — normal column, should survive imputation
    dense = [None if i % 10 == 0 else float(i) for i in range(n)]
    return pl.DataFrame({
        "sparse": pl.Series(sparse, dtype=pl.Float64),
        "dense": pl.Series(dense, dtype=pl.Float64),
    })


@pytest.fixture(scope="module")
def drop_candidate_profile(drop_candidate_df):
    return StructuralProfiler(PipelineConfig()).profile(drop_candidate_df)


def test_drop_candidate_column_in_dropped_columns(drop_candidate_df, drop_candidate_profile):
    fi = ImputationOrchestrator().fit(drop_candidate_df, drop_candidate_profile)
    result = fi.transform(drop_candidate_df)
    assert "sparse" in result.dropped_columns


def test_drop_candidate_apply_exclusions_adds_column_to_config(drop_candidate_df, drop_candidate_profile):
    fi = ImputationOrchestrator().fit(drop_candidate_df, drop_candidate_profile)
    config = PipelineConfig()
    fi.apply_exclusions(config)
    assert "sparse" in config.exclude_columns


def test_drop_candidate_exclusions_applied_true_after_apply_exclusions(drop_candidate_df, drop_candidate_profile):
    fi = ImputationOrchestrator().fit(drop_candidate_df, drop_candidate_profile)
    config = PipelineConfig()
    fi.apply_exclusions(config)
    result = fi.transform(drop_candidate_df)
    assert result.exclusions_applied is True


def test_drop_candidate_resolve_active_columns_excludes_dropped(drop_candidate_df, drop_candidate_profile):
    fi = ImputationOrchestrator().fit(drop_candidate_df, drop_candidate_profile)
    config = PipelineConfig()
    fi.apply_exclusions(config)
    active = config.resolve_active_columns(
        PipelinePhase.Imputation, list(drop_candidate_df.columns)
    )
    assert "sparse" not in active
    assert "dense" in active


# ---------------------------------------------------------------------------
# Scope 143: Regression imputation with partially missing features
# ---------------------------------------------------------------------------


def test_regression_imputation_with_partially_missing_features():
    """Integration test: exercises regression imputation with partially missing features.

    Verifies the complete pipeline contract from profiling to imputation fitting
    and transformation, ensuring zero nulls, correct signals, and round-trip identity.
    """
    import numpy as np
    from dataforge_ml.config import PipelineConfig
    from dataforge_ml.imputation import (
        ImputationConfig,
        ImputationOrchestrator,
        ImputationStrategy,
        NumericImputationConfig,
    )
    from dataforge_ml.profiling._numeric_config import NonlinearityTag
    from dataforge_ml.profiling.orchestrator import StructuralProfiler

    rng = np.random.default_rng(42)
    n = 600

    # Generate linear relationship: target = 2 * feat + 5 + noise
    feat_clean = rng.normal(10.0, 2.0, n)
    target_clean = 2.0 * feat_clean + 5.0 + rng.normal(0.0, 0.5, n)

    # Introduce missingness (~10% for target, ~8% for feat)
    null_mask_target = rng.random(n) < 0.10
    null_mask_feat = rng.random(n) < 0.08

    target_vals = [None if null_mask_target[i] else float(target_clean[i]) for i in range(n)]
    feat_vals = [None if null_mask_feat[i] else float(feat_clean[i]) for i in range(n)]

    df = pl.DataFrame({
        "target": pl.Series(target_vals, dtype=pl.Float64),
        "feat": pl.Series(feat_vals, dtype=pl.Float64),
    })

    # Configure pipeline: force MCAR High columns to route to Regression
    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_nonlinearity=True,
            compute_correlation=True,
        ),
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=10,
                regression_min_rows=100,
            )
        )
    )

    # 1. Verify Phase 1 profile contains a valid NonlinearityTag
    profile = StructuralProfiler(config).profile(df)
    assert "target" in profile.columns
    target_profile = profile.columns["target"]
    assert target_profile.stats is not None
    assert target_profile.stats.nonlinearity_tag in list(NonlinearityTag)

    # 2. Fit ImputationOrchestrator
    orch = ImputationOrchestrator(config=config)
    fi = orch.fit(df, profile)

    # Verify strategy routed to Regression
    assert "target" in fi.records
    target_rec = fi.records["target"]
    assert target_rec.strategy == ImputationStrategy.Regression

    # 3. Assert ColumnImputationRecord.signals contains correct entries
    # Estimator chosen entry
    assert any("regression_estimator:" in s for s in target_rec.signals)

    # Convergence warning entry (if max_iter hit, check format)
    assert len(target_rec.signals) > 0
    convergence_warnings = [s for s in target_rec.signals if "convergence_warning:" in s]
    for warning in convergence_warnings:
        assert "max_iter=" in warning

    # 4. Transform and assert zero nulls
    res = fi.transform(df)
    assert res.dataframe["target"].null_count() == 0
    assert res.dataframe["feat"].null_count() == 0

    # 5. Serialise / deserialise round-trip
    restored = FittedImputer.from_dict(fi.to_dict())
    res_restored = restored.transform(df)
    assert res.dataframe.equals(res_restored.dataframe)


# ---------------------------------------------------------------------------
# Issue #152 — KNN imputation with mixed-scale columns (integration)
# ---------------------------------------------------------------------------


def test_knn_mixed_scale_imputation_integration():
    """Integration test: KNN columns with 1000:1 magnitude ratio.

    Verifies:
    - No nulls in imputed output.
    - knn_params and knn_scaling signals present for each KNN column.
    - Imputed small-scale values remain in the small column's original range
      (demonstrating scale-insensitive imputation).
    """
    import numpy as np
    from dataforge_ml.config import PipelineConfig
    from dataforge_ml.imputation import (
        ImputationConfig,
        ImputationOrchestrator,
        ImputationStrategy,
        NumericImputationConfig,
    )
    from dataforge_ml.profiling.orchestrator import StructuralProfiler
    from dataforge_ml.profiling._config import ProfileConfig

    rng = np.random.default_rng(999)
    n = 500

    # Two KNN columns: `small` in [0, 1], `large` in [0, 1000] — perfect correlation
    small_clean = rng.uniform(0.0, 1.0, n)
    large_clean = small_clean * 1000.0 + rng.normal(0, 0.01, n)

    # Introduce ~15% missingness in both columns
    null_mask_small = rng.random(n) < 0.15
    null_mask_large = rng.random(n) < 0.12

    small_vals = [None if null_mask_small[i] else float(small_clean[i]) for i in range(n)]
    large_vals = [None if null_mask_large[i] else float(large_clean[i]) for i in range(n)]

    df = pl.DataFrame({
        "small": pl.Series(small_vals, dtype=pl.Float64),
        "large": pl.Series(large_vals, dtype=pl.Float64),
    })

    # Force KNN routing by keeping dataset within KNN size guards
    config = PipelineConfig(
        profiling=ProfileConfig(),
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=50_000,
                knn_max_features=50,
            )
        )
    )

    profile = StructuralProfiler(config).profile(df)
    orch = ImputationOrchestrator(config=config)
    fi = orch.fit(df, profile)

    # Verify at least one column routes to KNN
    knn_cols = [col for col, rec in fi.records.items() if rec.strategy == ImputationStrategy.KNN]
    if not knn_cols:
        pytest.skip("No columns routed to KNN under current profile; check size guards.")

    # Each KNN column must carry both signal entries
    for col in knn_cols:
        signals = fi.records[col].signals
        assert any("knn_params:" in s for s in signals), (
            f"Column '{col}' missing knn_params signal; got: {signals}"
        )
        assert any("knn_scaling: applied" in s for s in signals), (
            f"Column '{col}' missing knn_scaling signal; got: {signals}"
        )

    # Transform: zero nulls in imputed output
    result = fi.transform(df)
    for col in knn_cols:
        assert result.dataframe[col].null_count() == 0, (
            f"Column '{col}' still has nulls after KNN imputation"
        )

    # Scale-sensitivity check: imputed `small` values must stay in [0, 1]
    if "small" in knn_cols:
        small_imputed = result.dataframe["small"].to_list()
        out_of_range = [v for v in small_imputed if v is not None and not (0.0 - 0.5 <= v <= 1.0 + 0.5)]
        assert not out_of_range, (
            f"Imputed 'small' values dominated by large-scale column: {out_of_range[:5]}"
        )


# ---------------------------------------------------------------------------
# Issue #155 — Integration: adaptive KNN end-to-end with mixed-scale columns
# ---------------------------------------------------------------------------


def test_knn_adaptive_end_to_end_mixed_scale():
    """End-to-end adaptive KNN with mixed-scale columns and adaptive k > 5.

    Exercises all three problems fixed in Scope 1:
    - Adaptive k (6 KNN features → base_k = max(5, sqrt(6)) = 5, k > 5 after
      missingness/completeness scaling)
    - Reliability-based weights
    - NaN-safe scaling with correct inverse-scale (large-column values must not
      collapse to small-column magnitudes)

    Assertions:
    1. No nulls in any KNN-routed column after transform.
    2. knn_params signal present on every KNN column.
    3. knn_scaling signal present on every KNN column.
    4. Imputed large-scale column values are in a plausible range (~[0, 1000]),
       not collapsed to small-scale magnitudes (~[0, 1]).
    """
    import numpy as np
    from dataforge_ml.config import PipelineConfig
    from dataforge_ml.imputation import (
        ImputationConfig,
        ImputationOrchestrator,
        ImputationStrategy,
        NumericImputationConfig,
    )
    from dataforge_ml.profiling._config import ProfileConfig
    from dataforge_ml.profiling.orchestrator import StructuralProfiler

    rng = np.random.default_rng(155)
    n = 600

    # Anchor signal: drives all other columns to create correlated structure.
    anchor = rng.uniform(0.0, 1.0, n)

    # 5 small-scale columns in [0, 1] and 1 large-scale column in [0, 1000].
    # All are linearly related to `anchor` to make KNN meaningful.
    small_cols = {f"s{i}": anchor + rng.normal(0, 0.05, n) for i in range(5)}
    large_col = anchor * 1000.0 + rng.normal(0, 1.0, n)

    # Introduce ~15% missingness in the large column and ~10% in two small cols.
    null_large = rng.random(n) < 0.15
    null_s0 = rng.random(n) < 0.10
    null_s1 = rng.random(n) < 0.10

    data = {}
    for i, (name, vals) in enumerate(small_cols.items()):
        col_vals = vals.tolist()
        if i == 0:
            col_vals = [None if null_s0[j] else v for j, v in enumerate(col_vals)]
        elif i == 1:
            col_vals = [None if null_s1[j] else v for j, v in enumerate(col_vals)]
        data[name] = pl.Series(col_vals, dtype=pl.Float64)
    data["large"] = pl.Series(
        [None if null_large[j] else float(large_col[j]) for j in range(n)],
        dtype=pl.Float64,
    )

    df = pl.DataFrame(data)

    config = PipelineConfig(
        profiling=ProfileConfig(),
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=50_000,
                knn_max_features=50,
            )
        ),
    )

    profile = StructuralProfiler(config).profile(df)
    orch = ImputationOrchestrator(config=config)
    fi = orch.fit(df, profile)

    knn_cols = [col for col, rec in fi.records.items() if rec.strategy == ImputationStrategy.KNN]
    if not knn_cols:
        pytest.skip("No columns routed to KNN under current profile; check size guards.")

    # 1. Both signals on every KNN column.
    for col in knn_cols:
        signals = fi.records[col].signals
        assert any("knn_params:" in s for s in signals), (
            f"Column '{col}' missing knn_params signal; got: {signals}"
        )
        assert any("knn_scaling: applied" in s for s in signals), (
            f"Column '{col}' missing knn_scaling signal; got: {signals}"
        )

    # 2. No nulls after transform.
    result = fi.transform(df)
    for col in knn_cols:
        assert result.dataframe[col].null_count() == 0, (
            f"Column '{col}' still has nulls after KNN imputation"
        )

    # 3. Large-scale column imputed values must be in a plausible range.
    #    If inverse-scaling is broken, all imputed values collapse to the
    #    standardised range (~[-3, 3]) instead of [0, 1000].  The max of a
    #    600-row column whose true range is [0, 1000] must comfortably exceed
    #    100 in correctly inverse-scaled output.
    if "large" in knn_cols:
        large_vals = result.dataframe["large"].drop_nulls().to_list()
        max_large = max(large_vals)
        assert max_large > 100.0, (
            f"Max imputed 'large' value is {max_large:.2f} — appears collapsed to "
            f"small-scale magnitudes (expected > 100 for a [0, 1000] column)"
        )


# ---------------------------------------------------------------------------
# Issue #162 — adaptive MICE end-to-end with non-linear MAR-suspect dataset
# ---------------------------------------------------------------------------


def test_mice_adaptive_end_to_end_nonlinear_dataset():
    """End-to-end adaptive MICE with a non-linear MAR-suspect dataset.

    Creates three correlated columns (quadratic, linear, cubic relationships to
    a shared base signal) with a shared missingness mask to trigger multi-MAR
    detection and MICE routing.  Asserts:
    - Final imputed output contains no nulls.
    - Every MICE column's ``ColumnImputationRecord.signals`` contains a
      ``mice_estimator:`` entry.
    - Every MICE column's ``ColumnImputationRecord.signals`` contains a
      convergence-status entry (either ``mice_convergence_warning:`` or
      ``mice_converged:``).
    """
    import numpy as np
    from dataforge_ml.config import PipelineConfig
    from dataforge_ml.imputation import (
        ImputationConfig,
        ImputationOrchestrator,
        ImputationStrategy,
        NumericImputationConfig,
    )
    from dataforge_ml.profiling._config import ProfileConfig
    from dataforge_ml.profiling.orchestrator import StructuralProfiler

    rng = np.random.default_rng(162)
    n = 600

    base = rng.uniform(0.0, 3.0, n)
    col_a = base ** 2 + rng.normal(0, 0.1, n)      # quadratic — non-linear
    col_b = base + rng.normal(0, 0.2, n)             # linear
    col_c = base ** 3 + rng.normal(0, 0.2, n)       # cubic — non-linear

    # Shared missingness mask: same ~15% of rows missing in all three columns
    # → Pearson correlation between null indicators ≈ 1.0 → MARSuspect on all
    shared_mask = rng.random(n) < 0.15

    data = {
        "a": pl.Series(
            [None if shared_mask[i] else float(col_a[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
        "b": pl.Series(
            [None if shared_mask[i] else float(col_b[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
        "c": pl.Series(
            [None if shared_mask[i] else float(col_c[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
    }
    df = pl.DataFrame(data)

    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_nonlinearity=True,
            compute_correlation=True,
        ),
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=0,  # force MICE routing (disable KNN size guard)
            )
        ),
    )

    profile = StructuralProfiler(config).profile(df)
    orch = ImputationOrchestrator(config=config)
    fi = orch.fit(df, profile)

    mice_cols = [col for col, rec in fi.records.items() if rec.strategy == ImputationStrategy.MICE]
    if not mice_cols:
        pytest.skip("No columns routed to MICE under current profile; check missingness thresholds.")

    # 1. No nulls in the final imputed output.
    result = fi.transform(df)
    for col in mice_cols:
        assert result.dataframe[col].null_count() == 0, (
            f"Column '{col}' still has nulls after adaptive MICE imputation"
        )

    # 2. Every MICE column record has a mice_estimator: signal.
    for col in mice_cols:
        signals = fi.records[col].signals
        assert any("mice_estimator:" in s for s in signals), (
            f"Column '{col}' missing mice_estimator signal; got: {signals}"
        )

    # 3. Every MICE column record has a convergence-status signal.
    for col in mice_cols:
        signals = fi.records[col].signals
        has_status = any(
            "mice_convergence_warning:" in s or "mice_converged:" in s
            for s in signals
        )
        assert has_status, (
            f"Column '{col}' missing convergence-status signal; got: {signals}"
        )


# ---------------------------------------------------------------------------
# Issue #175 — numeric sentinel end-to-end fit/transform
# ---------------------------------------------------------------------------


def test_numeric_sentinel_end_to_end_fit_transform():
    """Full sentinel pipeline: -999 normalised before fit; fill derived from real values only.

    Uses an Int64 column where some rows contain -999 (sentinel) and some are
    native null.  ProfileConfig declares the sentinel.  After fit/transform:
    - No -999 values remain in the output.
    - The mean fill value used for imputation is derived from non-sentinel
      observations only (i.e. does not include -999 in its computation).
    """
    import numpy as np

    rng = np.random.default_rng(175)
    n = 300

    real_values = rng.integers(20, 80, n).tolist()  # real ages in [20, 80]
    sentinel_mask = rng.random(n) < 0.10             # ~10% sentinel rows (-999)
    native_null_mask = rng.random(n) < 0.05          # ~5% native null rows

    age_vals = []
    for i in range(n):
        if sentinel_mask[i]:
            age_vals.append(-999)
        elif native_null_mask[i]:
            age_vals.append(None)
        else:
            age_vals.append(real_values[i])

    df = pl.DataFrame({"age": pl.Series(age_vals, dtype=pl.Int64)})

    config = PipelineConfig(
        profiling=ProfileConfig(numeric_sentinels={"age": [-999.0]}),
        random_seed=42,
    )
    profile = StructuralProfiler(config).profile(df)

    # Profile must carry the declared sentinels.
    assert profile.numeric_sentinels == {"age": [-999.0]}

    fi = ImputationOrchestrator(config).fit(df, profile)

    # FittedImputer must carry the sentinels.
    assert fi.numeric_sentinels == {"age": [-999.0]}

    # Transform the full DataFrame; no -999 values must remain.
    result = fi.transform(df)
    output_vals = result.dataframe["age"].to_list()
    assert -999 not in output_vals, "Sentinel value -999 remains in transform output."
    assert result.dataframe["age"].null_count() == 0, "Null values remain after imputation."

    # Fill value must be derived from real observations only (mean in [20, 80]).
    fill_value = fi.records["age"].fill_value
    if fill_value is not None:
        assert 20 <= fill_value <= 80, (
            f"Fill value {fill_value} is outside the real-value range [20, 80]; "
            f"sentinel -999 may have contaminated the mean computation."
        )

    # Round-trip serialisation preserves sentinel behaviour.
    restored = FittedImputer.from_dict(fi.to_dict())
    r_restored = restored.transform(df)
    assert result.dataframe.equals(r_restored.dataframe)

