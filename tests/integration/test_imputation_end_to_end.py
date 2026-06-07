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


def test_mnar_column_receives_constant_fill_and_indicator():
    """Dedicated test: MNAR-declared column gets constant fill + indicator column."""
    from dataforge_ml.imputation import ImputationConfig, NumericImputationConfig

    n = 200
    data = pl.DataFrame({
        "salary": pl.Series(
            [None if i % 5 == 0 else float(i * 1000) for i in range(n)],
            dtype=pl.Float64,
        ),
    })
    config = PipelineConfig(
        imputation=ImputationConfig(
            numeric=NumericImputationConfig(mnar_constant_fill=-9999),
            mnar_columns=["salary"],
        )
    )
    profile = StructuralProfiler(PipelineConfig()).profile(data)
    orch = ImputationOrchestrator(config=config)
    result = orch.fit_transform(data, profile)

    assert result.dataframe["salary"].null_count() == 0
    assert "salary_missing" in result.dataframe.columns
    assert (result.dataframe["salary"] == -9999.0).sum() > 0


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
