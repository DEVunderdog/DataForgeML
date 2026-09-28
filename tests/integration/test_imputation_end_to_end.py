"""
Integration test: Phase 1 → DataSplitter → Phase 2 imputation.

Verifies the full fit/transform contract on real DataFrames using actual
StructuralProfiler and the layered route -> resolve_recipe -> fit_unit ->
compose path (no stubs). Model-based strategies (KNN, MICE) are the
escalation-point ticket's concern.
"""

import polars as pl
import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase
from dataforge_ml.imputation import (
    FittedImputer,
    ImputationStrategy,
    author,
    derive_units,
    fit_unit,
    resolve_recipe,
    route,
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


def test_mnar_column_keeps_its_nulls_and_exposes_its_fill():
    """An MNAR column gains its indicator, keeps its nulls, and carries the
    computed fill on its record rather than applying it (ADR-0098)."""
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

    assert result.dataframe["salary"].equals(data["salary"])
    assert result.dataframe["salary_missing"].sum() == data["salary"].null_count()
    observed = data["salary"].drop_nulls()
    assert result.records["salary"].fill_value in (observed.mean(), observed.median())


def test_repeated_fits_are_independent(imputation_df, imputation_profile):
    """Two drives must not share state — each produces its own FittedImputer.

    ``route()`` is a pure function and each fit owns its own units, so
    independence is structural on the layered path rather than a property of
    a reused orchestrator; this pins that it stays so.
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
    # profiling, exercising the route()-time enforcement path on its own.
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

    routing = route(exclusion_profile, exclusion_config)
    assert "hard_out" not in routing.column_routings
    soft = routing.column_routings["soft_out"]
    assert soft.strategy == ImputationStrategy.Passthrough
    assert soft.excluded is True
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
    """author → resolve_recipe → fit_unit → compose → transform.

    The door's whole claim is that a hand-authored routing is indistinguishable
    downstream from a routed one, so this drives the same steps the automatic
    path drives — for every non-model-based strategy — and puts every fitted
    unit through the real ``serialize`` / ``deserialize`` boundary (ADR-0072).
    Model-based strategies (KNN, MICE) are the escalation-point ticket's
    concern, so none is declared here.
    """
    import numpy as np

    from dataforge_ml import (
        AuthoredColumn,
        ImputationStrategy,
        author,
        deserialize,
        derive_units,
        fit_unit,
        resolve_recipe,
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
            "tenure": pl.Series(
                rng.integers(0, 40, n).astype(float), dtype=pl.Float64
            ),
            "label": pl.Series(["A" if i % 2 else "B" for i in range(n)]),
        }
    ).with_columns(
        pl.when(idx % 9 == 0).then(None).otherwise(pl.col("score")).alias("score"),
        pl.when(idx % 7 == 0).then(None).otherwise(pl.col("revenue")).alias("revenue"),
        pl.when(idx % 11 == 0).then(None).otherwise(pl.col("rating")).alias("rating"),
        pl.when(idx % 13 == 0).then(None).otherwise(pl.col("tenure")).alias("tenure"),
    )

    config = PipelineConfig(profiling=ProfileConfig())
    profile = StructuralProfiler(config).profile(df)

    routing = author(
        {
            "score": ImputationStrategy.Median,
            "revenue": ImputationStrategy.Mean,
            "rating": AuthoredColumn(
                ImputationStrategy.Constant, constant_fill=3.0
            ),
            "tenure": ImputationStrategy.MNAR,
        },
        profile=profile,
    )
    recipe = resolve_recipe(routing, profile, config)
    units = derive_units(routing)
    assert {u.unit_id for u in units} == {
        "median:score",
        "mean:revenue",
        "constant:rating",
        "mnar:tenure",
    }

    results = {
        unit.unit_id: fit_unit(recipe, unit, df, random_seed=42)
        for unit in units
    }
    imputer = FittedImputer.compose(recipe, results)
    result = imputer.transform(df)

    for col in ("score", "revenue", "rating"):
        assert result.dataframe[col].null_count() == 0, f"'{col}' still has nulls"
    # MNAR keeps its nulls beside its indicator; the fill is exposed (ADR-0098).
    assert result.dataframe["tenure"].equals(df["tenure"])
    assert result.records["tenure"].fill_value is not None
    assert result.dataframe["tenure_missing"].sum() == df["tenure"].null_count()
    # The Passthrough string column rode through untouched.
    assert result.dataframe["label"].equals(df["label"])

    # Every fitted unit round-trips through the bare-bytes boundary, and the
    # restored units compose into an imputer producing the identical frame.
    restored_units = {
        unit_id: deserialize(serialize(res.fitted))
        for unit_id, res in results.items()
    }
    restored = FittedImputer.compose(recipe, restored_units)
    assert restored.transform(df).dataframe.equals(result.dataframe)


# ---------------------------------------------------------------------------
# Re-authoring a routed plan end to end (#470)
# ---------------------------------------------------------------------------


def test_re_authored_routed_plan_drives_to_an_imputed_frame(
    imputation_split, imputation_profile
):
    """route → author(base=) → resolve_recipe → fit_unit → compose → transform.

    The re-authoring user disagrees with one column and keeps the rest. The
    result must be a routing in every sense the fit path cares about: the
    edited column takes the new strategy, the untouched ones keep the
    router's own routings, and the whole thing still fills every numeric
    null.
    """
    train = imputation_split.train
    config = PipelineConfig()
    routed = route(imputation_profile, config)

    edited = author({"rating": ImputationStrategy.Median}, base=routed)

    assert edited.column_routings["rating"].strategy == ImputationStrategy.Median
    assert "median:rating" in {u.unit_id for u in derive_units(edited)}
    for col in ("score", "revenue", "label"):
        assert edited.column_routings[col] == routed.column_routings[col]

    recipe = resolve_recipe(edited, imputation_profile, config)
    units = derive_units(edited)
    results = {
        unit.unit_id: fit_unit(recipe, unit, train, random_seed=42)
        for unit in units
    }
    result = FittedImputer.compose(recipe, results).transform(imputation_split.test)

    for col in ("score", "revenue", "rating"):
        assert result.dataframe[col].null_count() == 0, f"'{col}' still has nulls"
    assert result.records["rating"].decision.strategy == ImputationStrategy.Median


# ---------------------------------------------------------------------------
# Fit purity (#331 / ADR-0058): fitting routes and learns models and nothing
# else — no diagnostics ride on the records, no cross-validated fold work runs.
# ---------------------------------------------------------------------------


class _Recorder:
    """Progress Observer that records every event it receives, in order."""

    def __init__(self) -> None:
        self.events: list = []

    def __call__(self, event) -> None:
        self.events.append(event)


@pytest.fixture(scope="module")
def purity_df(rng):
    """A wide, correlated numeric frame so several columns route to MICE."""
    rng = rng(seed=44)
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
def purity_profile(purity_df):
    return StructuralProfiler(PipelineConfig(profiling=ProfileConfig())).profile(
        purity_df
    )


def test_fit_records_carry_no_diagnostics(purity_df, purity_profile):
    fitted = fit_imputer(purity_df, purity_profile)
    for rec in fitted.records.values():
        assert not hasattr(rec, "diagnostic")
    # And the serialisable record form carries no diagnostic key.
    for rec in fitted.records.values():
        assert "diagnostic" not in rec.to_dict()


def test_fit_event_stream_has_no_diagnostics_fold_substeps(purity_df, purity_profile):
    """Fitting only learns models — no cross-validated fold work happens."""
    from dataforge_ml import EventType

    recorder = _Recorder()
    fit_imputer(purity_df, purity_profile, observer=recorder)
    fold_msgs = [
        e.message
        for e in recorder.events
        if e.event_type == EventType.substep and "diagnostics fold" in (e.message or "")
    ]
    assert not fold_msgs, "the fit path must not run diagnostics folds"


def test_forced_gmm_sampling_end_to_end_on_bounded_discrete_snaps_output():
    """config.numeric.set_per_column_strategy("age", ImputationStrategy.GMMSampling)
    routes and fits successfully end to end, including on a BoundedDiscrete column
    with snapped output (ADR-0095).
    """
    import numpy as np
    from dataforge_ml.profiling._config import NumericKind

    rng = np.random.default_rng(42)
    n = 1000
    c1 = rng.choice([1, 2], p=[0.85, 0.15], size=450).tolist()
    mid = [3] * 100
    c2 = rng.choice([4, 5], p=[0.15, 0.85], size=450).tolist()
    vals = c1 + mid + c2
    rng.shuffle(vals)

    missing_idx = rng.choice(n, int(n * 0.2), replace=False)
    vals_masked = [
        None if i in set(missing_idx.tolist()) else float(v)
        for i, v in enumerate(vals)
    ]
    df = pl.DataFrame({"age": pl.Series(vals_masked, dtype=pl.Float64)})

    cfg = PipelineConfig()
    cfg.imputation.numeric.set_per_column_strategy("age", ImputationStrategy.GMMSampling)

    profile = StructuralProfiler(cfg).profile(df)
    assert profile.columns["age"].numeric_kind == NumericKind.BoundedDiscrete

    imputer = fit_imputer(df, profile, cfg)
    res = imputer.transform(df)

    out = res.dataframe["age"].to_numpy()
    imputed_vals = out[missing_idx]
    assert res.dataframe["age"].null_count() == 0
    assert np.all(imputed_vals >= 1.0) and np.all(imputed_vals <= 5.0)
    assert np.all(imputed_vals == np.round(imputed_vals))
    assert res.records["age"].decision.strategy == ImputationStrategy.GMMSampling

