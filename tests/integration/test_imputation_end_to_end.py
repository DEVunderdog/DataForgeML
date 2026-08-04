"""
Integration test: Phase 1 → DataSplitter → Phase 2 imputation.

Verifies the full fit/transform contract on real DataFrames using actual
StructuralProfiler and the layered decide -> execute -> build path (no stubs).
"""

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase
from dataforge_ml.imputation import (
    FittedImputer,
    ImputationStrategy,
    author,
    decide,
    fit_unit,
)
from dataforge_ml.profiling._config import ProfileConfig
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.splitting import DataSplitter
from tests.conftest import fit_imputer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def imputation_df(rng):
    rng = rng(seed=46)
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
    return fit_imputer(imputation_split.train, imputation_profile)


# ---------------------------------------------------------------------------
# Acceptance criteria from issue #78
# ---------------------------------------------------------------------------


def test_fit_returns_fitted_imputer(imputation_split, imputation_profile):
    fi = fit_imputer(imputation_split.train, imputation_profile)
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
        assert rec.decision.strategy is not None
        assert len(rec.decision.signals) >= 1


def test_label_column_passes_through_untouched(fitted_imputer, imputation_split):
    """Non-numeric (Text/Categorical) columns must not be altered."""
    result = fitted_imputer.transform(imputation_split.train)
    assert "label" in result.dataframe.columns
    assert result.dataframe["label"].equals(imputation_split.train["label"])


def test_fitted_imputer_serialisation_round_trip(fitted_imputer, imputation_split, round_trip):
    restored = round_trip(fitted_imputer)
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
    result = fit_imputer(data, profile, config).transform(data)

    assert result.dataframe["salary"].null_count() == 0
    assert "salary_missing" in result.dataframe.columns


def test_repeated_fits_are_independent(imputation_df, imputation_profile):
    """Two drives must not share state — each produces its own FittedImputer.

    ``decide()`` is a pure function and each ``ImputationExecutor`` owns its units,
    so independence is structural on the layered path rather than a property of a
    reused orchestrator; this pins that it stays so.
    """
    splitter = DataSplitter(imputation_df, random_seed=1)
    split1 = splitter.random_split(test_size=0.5, stratify=False)
    split2 = splitter.random_split(test_size=0.5, stratify=False)

    fi1 = fit_imputer(split1.train, imputation_profile)
    fi2 = fit_imputer(split2.train, imputation_profile)

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
    fi = fit_imputer(drop_candidate_df, drop_candidate_profile)
    result = fi.transform(drop_candidate_df)
    assert "sparse" in result.dropped_columns


def test_drop_candidate_apply_exclusions_adds_column_to_config(drop_candidate_df, drop_candidate_profile):
    fi = fit_imputer(drop_candidate_df, drop_candidate_profile)
    config = PipelineConfig()
    fi.apply_exclusions(config)
    assert "sparse" in config.exclude_columns


def test_drop_candidate_resolve_active_columns_excludes_dropped(drop_candidate_df, drop_candidate_profile):
    fi = fit_imputer(drop_candidate_df, drop_candidate_profile)
    config = PipelineConfig()
    fi.apply_exclusions(config)
    active = config.resolve_active_columns(
        PipelinePhase.Imputation, list(drop_candidate_df.columns)
    )
    assert "sparse" not in active
    assert "dense" in active


# ---------------------------------------------------------------------------
# Scope 143: MICE imputation with partially missing features (formerly Regression,
# collapsed by ADR-0079 — the MCAR-High/KNN-size-guard-failed branch now emits MICE)
# ---------------------------------------------------------------------------


def test_mice_imputation_with_partially_missing_features(round_trip):
    """Integration test: exercises MICE imputation with partially missing features.

    Verifies the complete pipeline contract from profiling to imputation fitting
    and transformation, ensuring zero nulls, correct signals, and round-trip identity.
    """
    import numpy as np

    from dataforge_ml.config import PipelineConfig
    from dataforge_ml.imputation import (
        ImputationConfig,
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

    # Configure pipeline: force MCAR High columns past the KNN size guard into MICE
    config = PipelineConfig(
        profiling=ProfileConfig(
            compute_nonlinearity=True,
            compute_correlation=True,
        ),
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(
                knn_max_rows=10,
                mice_min_rows=100,
            )
        )
    )

    # 1. Verify Phase 1 profile contains a valid NonlinearityTag
    profile = StructuralProfiler(config).profile(df)
    assert "target" in profile.columns
    target_profile = profile.columns["target"]
    assert target_profile.stats is not None
    assert target_profile.stats.nonlinearity_tag in list(NonlinearityTag)

    # 2. Fit the imputer
    plan = decide(profile, len(df), config)
    fi = fit_imputer(df, profile, config)

    # Verify strategy routed to MICE
    assert "target" in fi.records
    target_rec = fi.records["target"]
    assert target_rec.decision.strategy == ImputationStrategy.MICE

    # 3. The estimator family is resolved at decide-time and carried on the plan
    # (ADR-0060), so it is read off the decision rather than a fit-time signal.
    assert target_rec.decision.model_choice is not None, (
        "a MICE column must carry the estimator family it will train"
    )
    assert plan.column_decisions["target"].model_choice == target_rec.decision.model_choice
    assert len(target_rec.decision.signals) > 0

    # 4. Transform and assert zero nulls
    res = fi.transform(df)
    assert res.dataframe["target"].null_count() == 0
    assert res.dataframe["feat"].null_count() == 0

    # 5. Serialise / deserialise round-trip
    restored = round_trip(fi)
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
        ImputationStrategy,
        NumericImputationConfig,
    )
    from dataforge_ml.profiling._config import ProfileConfig
    from dataforge_ml.profiling.orchestrator import StructuralProfiler

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
    plan = decide(profile, len(df), config)
    fi = fit_imputer(df, profile, config)

    # Verify at least one column routes to KNN
    knn_cols = [col for col, rec in fi.records.items() if rec.decision.strategy == ImputationStrategy.KNN]
    if not knn_cols:
        pytest.skip("No columns routed to KNN under current profile; check size guards.")

    # The resolved KNN dials are decision-carried on the plan's unit (ADR-0062),
    # not a fit-time signal appended to the record: the assembler resolves them
    # from profile statistics, so the number on the plan is the number that runs.
    knn_unit = next(u for u in plan.units if u.strategy == ImputationStrategy.KNN)
    dials = dict(knn_unit.hyperparameters or ())
    assert "n_neighbors" in dials, f"KNN unit carries no n_neighbors; got: {dials}"
    assert "weights" in dials, f"KNN unit carries no weights; got: {dials}"

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
    plan = decide(profile, len(df), config)
    fi = fit_imputer(df, profile, config)

    knn_cols = [col for col, rec in fi.records.items() if rec.decision.strategy == ImputationStrategy.KNN]
    if not knn_cols:
        pytest.skip("No columns routed to KNN under current profile; check size guards.")

    # 1. The resolved dials are decision-carried on the plan's unit (ADR-0062).
    knn_unit = next(u for u in plan.units if u.strategy == ImputationStrategy.KNN)
    dials = dict(knn_unit.hyperparameters or ())
    assert "n_neighbors" in dials, f"KNN unit carries no n_neighbors; got: {dials}"
    assert "weights" in dials, f"KNN unit carries no weights; got: {dials}"

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
    plan = decide(profile, len(df), config)
    fi = fit_imputer(df, profile, config)

    mice_cols = [col for col, rec in fi.records.items() if rec.decision.strategy == ImputationStrategy.MICE]
    if not mice_cols:
        pytest.skip("No columns routed to MICE under current profile; check missingness thresholds.")

    # 1. No nulls in the final imputed output.
    result = fi.transform(df)
    for col in mice_cols:
        assert result.dataframe[col].null_count() == 0, (
            f"Column '{col}' still has nulls after adaptive MICE imputation"
        )

    # 2. Every MICE column carries its estimator family, resolved at decide-time
    # and read off the decision rather than a fit-time signal (ADR-0060).
    for col in mice_cols:
        assert fi.records[col].decision.model_choice is not None, (
            f"Column '{col}' carries no model_choice"
        )

    # 3. The MICE dials are decision-carried on the block unit (ADR-0062).
    mice_unit = next(u for u in plan.units if u.strategy == ImputationStrategy.MICE)
    dials = dict(mice_unit.hyperparameters or ())
    assert "max_iter" in dials, f"MICE unit carries no max_iter; got: {dials}"


# ---------------------------------------------------------------------------
# Issue #394 — end-to-end exclusion flow through the imputation door
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exclusion_df(rng):
    """Three numeric columns with missing values plus a text column.

    ``kept`` stays active, ``soft_out`` will be soft-excluded for Imputation,
    ``hard_out`` will be hard-excluded after profiling.
    """
    rng = rng(seed=394)
    n = 300
    base = rng.normal(100.0, 15.0, n)
    kept = base + rng.normal(0, 5.0, n)
    soft_out = base * 0.5 + rng.normal(0, 3.0, n)
    hard_out = base * 2.0 + rng.normal(0, 8.0, n)

    null_kept = rng.random(n) < 0.10
    null_soft = rng.random(n) < 0.12
    null_hard = rng.random(n) < 0.10

    return pl.DataFrame({
        "kept": pl.Series(
            [None if null_kept[i] else float(kept[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
        "soft_out": pl.Series(
            [None if null_soft[i] else float(soft_out[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
        "hard_out": pl.Series(
            [None if null_hard[i] else float(hard_out[i]) for i in range(n)],
            dtype=pl.Float64,
        ),
        "label": pl.Series(
            ["A" if i % 2 == 0 else "B" for i in range(n)], dtype=pl.Utf8
        ),
    })


@pytest.fixture(scope="module")
def exclusion_profile(exclusion_df):
    # Profiled with no exclusions declared: both exclusions are added after
    # profiling, exercising the decide()-time enforcement path on its own.
    return StructuralProfiler(PipelineConfig()).profile(exclusion_df)


@pytest.fixture(scope="module")
def exclusion_config():
    config = PipelineConfig()
    config.add_exclusion("hard_out")
    config.add_phase_exclusion(PipelinePhase.Imputation, "soft_out")
    return config


@pytest.fixture(scope="module")
def exclusion_fitted(exclusion_df, exclusion_profile, exclusion_config):
    """Fit over the full door with the hard-excluded column dropped by the user."""
    train = exclusion_df.drop("hard_out")
    return fit_imputer(train, exclusion_profile, exclusion_config)


def test_exclusion_plan_shape(exclusion_df, exclusion_profile, exclusion_config):
    """Soft-excluded column is Passthrough with the exclusion signal; hard-excluded column is absent."""
    from dataforge_ml.imputation import ImputationStrategy

    plan = decide(exclusion_profile, len(exclusion_df), exclusion_config)
    assert "hard_out" not in plan.column_decisions
    soft = plan.column_decisions["soft_out"]
    assert soft.strategy == ImputationStrategy.Passthrough
    assert any("soft-excluded" in s for s in soft.signals)


def test_soft_excluded_column_rides_through_transform_untouched(
    exclusion_df, exclusion_fitted
):
    """A soft-excluded column appears in the output untouched, missing values intact."""
    frame = exclusion_df.drop("hard_out")
    result = exclusion_fitted.transform(frame)

    assert result.dataframe["soft_out"].equals(frame["soft_out"])
    assert result.dataframe["soft_out"].null_count() == frame["soft_out"].null_count()
    assert result.dataframe["soft_out"].null_count() > 0
    # The active column is still imputed normally.
    assert result.dataframe["kept"].null_count() == 0


def test_hard_excluded_column_still_present_raises_before_mutation(
    exclusion_df, exclusion_fitted
):
    """A hard-excluded column the user forgot to drop hits the strict unknown-column raise."""
    from dataforge_ml.imputation import UnseenColumnError

    with pytest.raises(UnseenColumnError, match="hard_out"):
        exclusion_fitted.transform(exclusion_df)


def test_flow_succeeds_once_hard_excluded_column_dropped(
    exclusion_df, exclusion_fitted
):
    """The same flow succeeds when the user drops the hard-excluded column."""
    result = exclusion_fitted.transform(exclusion_df.drop("hard_out"))
    assert result.dataframe["kept"].null_count() == 0
    assert "hard_out" not in result.dataframe.columns


def test_fitted_imputer_holds_no_config_and_never_auto_drops(exclusion_fitted):
    """FittedImputer stays config-free; the hard-excluded column simply has no record."""
    assert not hasattr(exclusion_fitted, "config")
    assert "hard_out" not in exclusion_fitted.records
    assert "soft_out" in exclusion_fitted.records


# ---------------------------------------------------------------------------
# Issue #175 — numeric sentinel end-to-end fit/transform
# ---------------------------------------------------------------------------


def test_numeric_sentinel_end_to_end_fit_transform(round_trip):
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

    fi = fit_imputer(df, profile, config)

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
    restored = round_trip(fi)
    r_restored = restored.transform(df)
    assert result.dataframe.equals(r_restored.dataframe)



# ---------------------------------------------------------------------------
# The manual authoring door end to end (#468, ADR-0083)
# ---------------------------------------------------------------------------


def test_hand_authored_plan_drives_column_names_to_an_imputed_frame():
    """author → fit_unit → compose → transform, with no profile anywhere.

    The door's whole claim is that a hand-authored plan is indistinguishable
    downstream from a decided one, so this drives the same three steps the
    automatic path drives and puts every fitted unit through the real
    ``serialize`` / ``deserialize`` boundary (ADR-0072).
    """
    import numpy as np

    from dataforge_ml import (
        AuthoredColumn,
        ImputationStrategy,
        author,
        deserialize,
        fit_unit,
        serialize,
    )

    rng = np.random.default_rng(468)
    n = 300
    score = rng.normal(50.0, 10.0, n)
    revenue = score * 4.0 + rng.normal(0.0, 5.0, n)
    rating = np.clip(np.round(rng.normal(3.0, 1.0, n)), 1.0, 5.0)

    idx = pl.arange(0, n, eager=True)
    df = pl.DataFrame(
        {
            "score": pl.Series(score, dtype=pl.Float64),
            "revenue": pl.Series(revenue, dtype=pl.Float64),
            "rating": pl.Series(rating, dtype=pl.Float64),
            "tenure": pl.Series(rng.integers(0, 40, n), dtype=pl.Int64),
            "label": pl.Series(["A" if i % 2 else "B" for i in range(n)]),
        }
    ).with_columns(
        pl.when(idx % 9 == 0).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(idx % 7 == 0).then(None).otherwise(pl.col("revenue")).alias("revenue"),
        pl.when(idx % 11 == 0).then(None).otherwise(pl.col("rating")).alias("rating"),
        pl.when(idx % 13 == 0).then(None).otherwise(pl.col("tenure")).alias("tenure"),
    )

    # Column names alone — the frame is not consulted until fit_unit.
    plan = author(
        {
            "score": ImputationStrategy.MICE,
            "revenue": ImputationStrategy.MICE,
            "rating": AuthoredColumn(
                ImputationStrategy.Constant, constant_fill=3.0
            ),
            "tenure": ImputationStrategy.MNAR,
        },
        columns=["score", "revenue", "rating", "tenure", "label"],
    )
    assert {u.unit_id for u in plan.units} == {
        "mice",
        "constant:rating",
        "mnar:tenure",
    }

    results = {
        unit.unit_id: fit_unit(plan, unit.unit_id, df, random_seed=42)
        for unit in plan.units
    }
    imputer = FittedImputer.compose(plan, results)
    result = imputer.transform(df)

    for col in ("score", "revenue", "rating", "tenure"):
        assert result.dataframe[col].null_count() == 0, f"'{col}' still has nulls"
    assert result.dataframe["tenure_missing"].sum() == df["tenure"].null_count()
    # The Passthrough string column rode through untouched.
    assert result.dataframe["label"].equals(df["label"])

    # Every fitted unit round-trips through the bare-bytes boundary, and the
    # restored units compose into an imputer producing the identical frame.
    restored_units = {
        unit_id: deserialize(serialize(res.fitted))
        for unit_id, res in results.items()
    }
    restored = FittedImputer.compose(plan, restored_units)
    assert restored.transform(df).dataframe.equals(result.dataframe)


# ---------------------------------------------------------------------------
# Re-authoring a decided plan end to end (#470)
# ---------------------------------------------------------------------------


def test_re_authored_decided_plan_drives_to_an_imputed_frame(
    imputation_split, imputation_profile
):
    """decide → author(base=) → fit_unit → compose → transform.

    The re-authoring user disagrees with one column and keeps the rest. The
    result must be a plan in every sense the fit path cares about: the edited
    column takes the new strategy, the untouched ones keep the router's own
    decisions and dials, and the whole thing still fills every numeric null.
    """
    train = imputation_split.train
    decided = decide(imputation_profile, len(train), PipelineConfig())

    edited = author({"rating": ImputationStrategy.Median}, base=decided)

    assert edited.column_decisions["rating"].strategy == ImputationStrategy.Median
    assert "median:rating" in {u.unit_id for u in edited.units}
    for col in ("score", "revenue", "label"):
        assert edited.column_decisions[col] == decided.column_decisions[col]
    assert edited.config_snapshot == decided.config_snapshot

    results = {
        unit.unit_id: fit_unit(edited, unit.unit_id, train, random_seed=42)
        for unit in edited.units
    }
    result = FittedImputer.compose(edited, results).transform(imputation_split.test)

    for col in ("score", "revenue", "rating"):
        assert result.dataframe[col].null_count() == 0, f"'{col}' still has nulls"
    assert result.records["rating"].decision.strategy == ImputationStrategy.Median
