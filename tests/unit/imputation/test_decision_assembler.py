"""
Unit tests for the decision assembler — ``decide(profile, n_rows, config)`` and
the whole-object ``ImputationDecision`` (ADR-0060, ADR-0066, issue #345).

Tests assert the external contract a user relies on: the plan is a pure,
immutable function of ``(profile, shape, config)``; per-column ``model_choice``
is resolved at decide-time and visible; the units materialise with the specified
ids and re-derive at every construction. Profiles are minimal
``StructuralProfileResult`` stubs — no real profiler run — with routing driven
deterministically through ``per_column_strategy`` / ``mnar_columns`` so the
assembly (not the already-tested router) is under test.
"""

import dataclasses

import pytest

from dataforge_ml import (
    ImputationDecision as RootImputationDecision,
    ModelChoice,
    PipelineConfig,
    PipelinePhase,
)
from dataforge_ml import decide as root_decide
from dataforge_ml.config import SemanticType
from dataforge_ml.imputation import (
    ImputationDecision,
    ImputationStrategy,
    ImputationUnit,
    decide,
)
from dataforge_ml.profiling._config import (
    ColumnProfile,
    NumericKind,
    StructuralProfileResult,
)
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult
from dataforge_ml.profiling._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingSeverity,
)
from dataforge_ml.profiling._numeric_config import (
    BimodalStats,
    NonlinearityTag,
    NumericFlag,
    NumericStats,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal stubs
# ---------------------------------------------------------------------------


def _numeric_cp(
    name: str,
    *,
    null_count: int = 20,
    total_rows: int = 100,
    severity: MissingSeverity | None = MissingSeverity.Moderate,
    flags: list[MissingnessFlag] | None = None,
    nonlinearity_tag: NonlinearityTag | None = None,
    numeric_kind: NumericKind = NumericKind.Continuous,
    min_value: float | None = None,
    max_value: float | None = None,
) -> ColumnProfile:
    missingness = None
    if null_count > 0 or flags:
        missingness = ColumnMissingnessProfile(
            column=name,
            total_rows=total_rows,
            effective_null_count=null_count,
            effective_null_ratio=null_count / total_rows,
            severity=severity,
            flags=flags or [],
            correlated_with=[],
        )
    stats = NumericStats(nonlinearity_tag=nonlinearity_tag, min=min_value, max=max_value)
    return ColumnProfile(
        name=name,
        semantic_type=SemanticType.Numeric,
        numeric_kind=numeric_kind,
        missingness=missingness,
        stats=stats,
    )


def _profile(
    columns: dict[str, ColumnProfile],
    *,
    row_count: int = 100,
) -> StructuralProfileResult:
    result = StructuralProfileResult()
    result.columns.update(columns)
    result.dataset.row_count = row_count
    return result


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_exported_from_package_root() -> None:
    assert root_decide is decide
    assert RootImputationDecision is ImputationDecision


def test_decide_returns_immutable_decision() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    plan = decide(profile, profile.dataset.row_count, PipelineConfig())
    assert isinstance(plan, ImputationDecision)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.units = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Purity: a pure function of (profile, shape, config)
# ---------------------------------------------------------------------------


def test_decide_is_pure_same_inputs_equal_plans() -> None:
    profile = _profile({"a": _numeric_cp("a"), "b": _numeric_cp("b")})
    config = PipelineConfig()
    assert decide(profile, 100, config) == decide(profile, 100, config)


def test_decide_requires_n_rows() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    with pytest.raises(TypeError):
        decide(profile)  # type: ignore[call-arg]


def test_decide_accepts_n_rows_by_keyword() -> None:
    profile = _profile({"a": _numeric_cp("a")}, row_count=600)
    plan = decide(profile, n_rows=480, config=PipelineConfig())
    assert plan.decided_for_shape[0] == 480


def test_decide_rejects_negative_n_rows() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    with pytest.raises(ValueError, match="n_rows"):
        decide(profile, -1, PipelineConfig())


def test_n_rows_drives_the_plan_not_the_profile_row_count() -> None:
    """Two plans from one profile differ when decided for different splits."""
    columns = {"a": _numeric_cp("a"), "b": _numeric_cp("b")}
    config = PipelineConfig()
    full = decide(_profile(columns, row_count=600), 600, config)
    train = decide(_profile(columns, row_count=600), 480, config)
    assert full.decided_for_shape[0] == 600
    assert train.decided_for_shape[0] == 480


def test_decide_defaults_config_when_omitted() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    plan = decide(profile, profile.dataset.row_count)
    assert plan.config_snapshot == PipelineConfig().to_dict()


def test_decided_for_shape_captures_rows_features_and_column_set() -> None:
    profile = _profile(
        {"a": _numeric_cp("a"), "b": _numeric_cp("b")}, row_count=250
    )
    plan = decide(profile, profile.dataset.row_count, PipelineConfig())
    n_rows, n_features, column_set = plan.decided_for_shape
    assert n_rows == 250
    assert n_features == 2
    assert column_set == ("a", "b")


def test_profile_provenance_carries_row_count() -> None:
    profile = _profile({"a": _numeric_cp("a")}, row_count=100)
    plan = decide(profile, profile.dataset.row_count, PipelineConfig())
    assert plan.profile_provenance["row_count"] == 100


def test_shape_rows_and_provenance_rows_are_different_facts() -> None:
    """"Decided for 480 rows, from a profile of 600" is the honest record."""
    profile = _profile({"a": _numeric_cp("a"), "b": _numeric_cp("b")}, row_count=600)
    plan = decide(profile, 480, PipelineConfig())
    assert plan.decided_for_shape[0] == 480
    assert plan.profile_provenance["row_count"] == 600


# ---------------------------------------------------------------------------
# Model choice resolved at decide-time and visible on the plan
# ---------------------------------------------------------------------------


def test_single_column_mice_model_choice_resolved_at_decide_time() -> None:
    profile = _profile(
        {"r": _numeric_cp("r", nonlinearity_tag=NonlinearityTag.Linear)}
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        "r", ImputationStrategy.MICE
    )
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.column_decisions["r"].model_choice == ModelChoice.BayesianRidge


def test_single_column_mice_complex_nonlinear_large_sample_picks_gradient_boosting() -> None:
    profile = _profile(
        {"r": _numeric_cp("r", nonlinearity_tag=NonlinearityTag.ComplexNonlinear)},
        row_count=10_000_000,
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        "r", ImputationStrategy.MICE
    )
    plan = decide(profile, profile.dataset.row_count, config)
    assert (
        plan.column_decisions["r"].model_choice
        == ModelChoice.GradientBoostingRegressor
    )


def test_scalar_and_knn_columns_carry_no_model_choice() -> None:
    profile = _profile({"m": _numeric_cp("m"), "k": _numeric_cp("k")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("m", ImputationStrategy.Median)
    config.imputation.numeric.set_per_column_strategy("k", ImputationStrategy.KNN)
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.column_decisions["m"].model_choice is None
    assert plan.column_decisions["k"].model_choice is None


def test_mice_block_columns_share_block_model_choice() -> None:
    profile = _profile(
        {
            "a": _numeric_cp("a", nonlinearity_tag=NonlinearityTag.Linear),
            "b": _numeric_cp("b", nonlinearity_tag=NonlinearityTag.ComplexNonlinear),
        },
        row_count=10_000_000,
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    plan = decide(profile, profile.dataset.row_count, config)
    # Winning tag over {Linear, ComplexNonlinear} is ComplexNonlinear → GBR.
    assert (
        plan.column_decisions["a"].model_choice
        == ModelChoice.GradientBoostingRegressor
    )
    assert (
        plan.column_decisions["b"].model_choice
        == plan.column_decisions["a"].model_choice
    )


# ---------------------------------------------------------------------------
# Units: materialised tuple with the specified ids, re-derived at construction
# ---------------------------------------------------------------------------


def test_joint_blocks_collapse_to_single_units() -> None:
    profile = _profile(
        {
            "a": _numeric_cp("a"),
            "b": _numeric_cp("b"),
            "k1": _numeric_cp("k1"),
            "k2": _numeric_cp("k2"),
        }
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    config.imputation.numeric.set_per_column_strategy(
        ["k1", "k2"], ImputationStrategy.KNN
    )
    plan = decide(profile, profile.dataset.row_count, config)
    assert isinstance(plan.units, tuple)
    by_id = {u.unit_id: u for u in plan.units}
    assert by_id["mice"].columns == ("a", "b")
    assert by_id["knn"].columns == ("k1", "k2")
    # exactly one of each joint block
    assert [u.unit_id for u in plan.units].count("mice") == 1
    assert [u.unit_id for u in plan.units].count("knn") == 1


def test_knn_n_neighbors_bounded_by_the_train_split_not_the_profile() -> None:
    """Pins ADR-0066's latent bug: ``n_rows`` is a hyperparameter input too.

    Bounding ``n_neighbors`` against the profile's full-dataset count let a plan
    legally request more neighbours than the train split contains.
    """
    profile = _profile(
        {"k1": _numeric_cp("k1"), "k2": _numeric_cp("k2")}, row_count=600
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["k1", "k2"], ImputationStrategy.KNN
    )
    n_rows = 4
    plan = decide(profile, n_rows, config)
    knn = next(u for u in plan.units if u.unit_id == "knn")
    assert dict(knn.hyperparameters)["n_neighbors"] <= n_rows - 1


def test_knn_decided_base_populates_complete_frac() -> None:
    """The KNN base carries ``complete_frac`` the fitter reads with no fallback.

    Pins the ADR-0073 gap: the fitter previously read
    ``hyp.get("complete_frac", 0.0)`` while the base never populated it, silently
    defaulting. With fallbacks stripped, the key must be present.
    """
    profile = _profile({"k1": _numeric_cp("k1"), "k2": _numeric_cp("k2")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["k1", "k2"], ImputationStrategy.KNN
    )
    plan = decide(profile, profile.dataset.row_count, config)
    knn_base = dict(plan.decided_hyperparameters["knn"])
    assert "complete_frac" in knn_base


def test_decided_base_is_complete_for_every_strategy() -> None:
    """Each unit's decided base carries every dial its fitter reads directly.

    With the fitters reading ``hyp["..."]`` (no fallbacks, ADR-0073), a missing
    key would be a loud ``KeyError`` at fit time; this asserts the base the
    assembler builds never leaves one out.
    """
    profile = _profile(
        {
            "a": _numeric_cp("a", nonlinearity_tag=NonlinearityTag.Linear),
            "b": _numeric_cp("b", nonlinearity_tag=NonlinearityTag.Linear),
            "k1": _numeric_cp("k1"),
            "k2": _numeric_cp("k2"),
            "n": _numeric_cp("n"),
        }
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(["a", "b"], ImputationStrategy.MICE)
    config.imputation.numeric.set_per_column_strategy(["k1", "k2"], ImputationStrategy.KNN)
    config.imputation.add_mnar_column("n")

    plan = decide(profile, profile.dataset.row_count, config)
    base = plan.decided_hyperparameters

    required = {
        "mice": {"max_iter", "tol", "initial_strategy", "n_nearest_features", "nonlinearity_tag"},
        "knn": {"n_neighbors", "weights", "miss_frac", "complete_frac"},
        "mnar:n": {"central_tendency"},
    }
    for unit_id, keys in required.items():
        assert keys <= set(dict(base[unit_id])), unit_id


def test_per_column_unit_ids_use_strategy_column_form() -> None:
    profile = _profile({"m": _numeric_cp("m")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("m", ImputationStrategy.Median)
    plan = decide(profile, profile.dataset.row_count, config)
    assert any(u.unit_id == "median:m" for u in plan.units)


def test_structural_strategies_produce_no_units() -> None:
    profile = _profile(
        {
            "dropme": _numeric_cp(
                "dropme", flags=[MissingnessFlag.DropCandidate], null_count=60
            ),
            "clean": _numeric_cp("clean", null_count=0, severity=None),
            "cat": ColumnProfile(name="cat", semantic_type=SemanticType.Categorical),
        }
    )
    plan = decide(profile, profile.dataset.row_count, PipelineConfig())
    unit_ids = [u.unit_id for u in plan.units]
    assert not any("dropped" in uid for uid in unit_ids)
    assert not any("passthrough" in uid for uid in unit_ids)
    assert plan.dropped_columns == ("dropme",)
    assert plan.column_decisions["clean"].strategy == ImputationStrategy.Passthrough
    assert plan.column_decisions["cat"].strategy == ImputationStrategy.Passthrough


def test_mnar_registers_indicator_decision_without_a_unit() -> None:
    profile = _profile({"n": _numeric_cp("n")})
    config = PipelineConfig()
    config.imputation.add_mnar_column("n")
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.column_decisions["n"].mnar is True
    assert "n_missing" in plan.column_decisions
    assert (
        plan.column_decisions["n_missing"].strategy == ImputationStrategy.Indicator
    )
    unit_ids = [u.unit_id for u in plan.units]
    assert "mnar:n" in unit_ids
    assert not any("indicator" in uid for uid in unit_ids)


def test_units_re_derive_on_edit_no_stale_list() -> None:
    profile = _profile({"a": _numeric_cp("a"), "b": _numeric_cp("b")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    plan = decide(profile, profile.dataset.row_count, config)
    assert {u.unit_id for u in plan.units} == {"mice"}

    edited = plan.with_strategy("a", ImputationStrategy.Median)
    # a leaves the MICE block → its own scalar unit; b remains a (singleton) MICE block
    edited_ids = {u.unit_id for u in edited.units}
    assert "median:a" in edited_ids
    assert "mice" in edited_ids
    # the original plan is untouched (immutability)
    assert {u.unit_id for u in plan.units} == {"mice"}


def test_domain_snap_bounds_surfaced_for_bounded_discrete_model_column() -> None:
    profile = _profile(
        {
            "d": _numeric_cp(
                "d",
                numeric_kind=NumericKind.BoundedDiscrete,
                min_value=0.0,
                max_value=9.0,
            )
        }
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("d", ImputationStrategy.KNN)
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.column_decisions["d"].domain_snap_bounds == (0.0, 9.0)


# ---------------------------------------------------------------------------
# Immutable edits return a new, valid plan
# ---------------------------------------------------------------------------


def test_with_model_choice_returns_new_plan() -> None:
    profile = _profile(
        {"r": _numeric_cp("r", nonlinearity_tag=NonlinearityTag.Linear)}
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        "r", ImputationStrategy.MICE
    )
    plan = decide(profile, profile.dataset.row_count, config)
    edited = plan.with_model_choice("r", ModelChoice.RandomForestRegressor)
    assert plan.column_decisions["r"].model_choice == ModelChoice.BayesianRidge
    assert (
        edited.column_decisions["r"].model_choice
        == ModelChoice.RandomForestRegressor
    )


def test_with_strategy_rejects_output_only_label() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", ImputationStrategy.Median)
    plan = decide(profile, profile.dataset.row_count, config)
    with pytest.raises(ValueError, match="Dropped"):
        plan.with_strategy("a", ImputationStrategy.Dropped)


def test_with_strategy_unknown_column_raises() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    plan = decide(profile, profile.dataset.row_count, PipelineConfig())
    with pytest.raises(KeyError):
        plan.with_strategy("missing", ImputationStrategy.Median)


# ---------------------------------------------------------------------------
# Round-trip serialisation re-derives units from the decision map
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_structural_equality() -> None:
    profile = _profile(
        {"a": _numeric_cp("a"), "b": _numeric_cp("b"), "n": _numeric_cp("n")}
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    config.imputation.add_mnar_column("n")
    plan = decide(profile, profile.dataset.row_count, config)
    assert ImputationDecision.from_dict(plan.to_dict()) == plan


def test_from_dict_ignores_tampered_unit_list() -> None:
    profile = _profile({"a": _numeric_cp("a")})
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("a", ImputationStrategy.Median)
    plan = decide(profile, profile.dataset.row_count, config)
    raw = plan.to_dict()
    raw["units"] = [{"unit_id": "bogus", "strategy": "mice", "columns": [], "model_choice": None}]
    rebuilt = ImputationDecision.from_dict(raw)
    assert rebuilt.units == plan.units
    assert all(isinstance(u, ImputationUnit) for u in rebuilt.units)

# ---------------------------------------------------------------------------
# Exclusion enforcement — decide() resolves its own active column set (#392)
# ---------------------------------------------------------------------------

_EXCLUSION_SIGNAL = "soft-excluded for Imputation phase"


def _all_unit_columns(plan: ImputationDecision) -> set[str]:
    return {c for u in plan.units for c in u.columns}


def test_soft_excluded_numeric_column_is_passthrough_with_exclusion_signal() -> None:
    profile = _profile({"a": _numeric_cp("a"), "x": _numeric_cp("x")})
    config = PipelineConfig()
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.column_decisions["x"].strategy == ImputationStrategy.Passthrough
    assert _EXCLUSION_SIGNAL in plan.column_decisions["x"].signals
    assert "x" not in _all_unit_columns(plan)


def test_hard_excluded_column_omitted_from_plan_snapshot_retains_lists() -> None:
    profile = _profile({"a": _numeric_cp("a"), "x": _numeric_cp("x")})
    config = PipelineConfig()
    config.add_exclusion("x")
    plan = decide(profile, profile.dataset.row_count, config)
    assert "x" not in plan.column_decisions
    assert "x" not in _all_unit_columns(plan)
    assert plan.config_snapshot["exclude_columns"] == ["x"]


def test_hard_exclusion_wins_over_soft_exclusion() -> None:
    profile = _profile({"a": _numeric_cp("a"), "x": _numeric_cp("x")})
    config = PipelineConfig()
    config.add_exclusion("x")
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    plan = decide(profile, profile.dataset.row_count, config)
    assert "x" not in plan.column_decisions


def test_excluded_column_leaves_shape_and_block_membership() -> None:
    profile = _profile(
        {"a": _numeric_cp("a"), "b": _numeric_cp("b"), "x": _numeric_cp("x")},
        row_count=100,
    )
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy(
        ["a", "b"], ImputationStrategy.MICE
    )
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    plan = decide(profile, profile.dataset.row_count, config)
    assert plan.decided_for_shape == (100, 2, ("a", "b"))
    mice = next(u for u in plan.units if u.unit_id == "mice")
    assert mice.columns == ("a", "b")


def test_multi_mar_counting_ignores_excluded_columns() -> None:
    """One of two MAR-suspect columns excluded → no multi-MAR, no MICE block.

    ``n_rows`` is kept above the default ``mice_min_rows`` floor (ADR-0079) so
    the multi-MAR branch itself routes to MICE and this test stays isolated to
    the exclusion-counting invariant it targets, not the floor gate.
    """

    def _mar_col(name: str) -> ColumnProfile:
        return _numeric_cp(name, flags=[MissingnessFlag.MARSuspect])

    columns = {"b": _mar_col("b"), "x": _mar_col("x")}
    plain = decide(_profile(columns), 600, PipelineConfig())
    assert "mice" in {u.unit_id for u in plain.units}

    config = PipelineConfig()
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    plan = decide(_profile(columns), 600, config)
    assert "mice" not in {u.unit_id for u in plan.units}
    # b routes exactly as it would in a profile that never contained x
    without_x = decide(_profile({"b": _mar_col("b")}), 600, PipelineConfig())
    assert plan.units == without_x.units
    assert plan.column_decisions["b"] == without_x.column_decisions["b"]


def test_cluster_conditional_feature_list_excludes_excluded_columns() -> None:
    bimodal = BimodalStats(
        dip_statistic=0.1,
        dip_p_value=0.001,
        center1=0.0,
        center2=10.0,
        cluster_separation=3.0,
        minority_weight=0.4,
    )
    bm = ColumnProfile(
        name="bm",
        semantic_type=SemanticType.Numeric,
        numeric_kind=NumericKind.Continuous,
        missingness=ColumnMissingnessProfile(
            column="bm",
            total_rows=100,
            effective_null_count=20,
            effective_null_ratio=0.2,
            severity=MissingSeverity.Moderate,
            flags=[],
            correlated_with=[],
        ),
        stats=NumericStats(flags=[NumericFlag.Bimodal], bimodal_stats=bimodal),
    )
    profile = _profile(
        {
            "bm": bm,
            "p": _numeric_cp("p", null_count=0, severity=None),
            "x": _numeric_cp("x", null_count=0, severity=None),
        }
    )
    profile.dataset.feature_correlation = CorrelationProfileResult(
        pearson_matrix={
            "bm": {"p": 0.5, "x": 0.9},
            "p": {"bm": 0.5, "x": 0.4},
            "x": {"bm": 0.9, "p": 0.4},
        }
    )
    config = PipelineConfig()
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    plan = decide(profile, profile.dataset.row_count, config)
    assert (
        plan.column_decisions["bm"].strategy == ImputationStrategy.ClusterConditional
    )
    feature_cols = dict(plan.decided_hyperparameters["cluster_conditional:bm"])[
        "feature_cols"
    ]
    assert feature_cols == ("p",)


def test_profiling_soft_excluded_placeholder_behaves_as_before() -> None:
    """A Profiling-phase placeholder (no semantic type) stays skipped."""
    placeholder = ColumnProfile(name="ph", semantic_type=None)
    profile = _profile({"a": _numeric_cp("a"), "ph": placeholder})
    config = PipelineConfig()
    config.add_phase_exclusion(PipelinePhase.Profiling, "ph")
    plan = decide(profile, profile.dataset.row_count, config)
    assert "ph" not in plan.column_decisions
    assert "ph" not in _all_unit_columns(plan)


def test_exclusions_naming_absent_columns_change_nothing() -> None:
    profile = _profile({"a": _numeric_cp("a"), "b": _numeric_cp("b")})
    plain = decide(profile, 100, PipelineConfig())
    config = PipelineConfig()
    config.add_exclusion("ghost_hard")
    config.add_phase_exclusion(PipelinePhase.Imputation, "ghost_soft")
    plan = decide(profile, 100, config)
    assert plan.column_decisions == plain.column_decisions
    assert plan.units == plain.units
    assert plan.decided_for_shape == plain.decided_for_shape
    assert plan.decided_hyperparameters == plain.decided_hyperparameters


# ---------------------------------------------------------------------------
# Contradictory config — excluded column named in imputation config (#393)
# ---------------------------------------------------------------------------


def _two_col_profile() -> StructuralProfileResult:
    return _profile({"a": _numeric_cp("a"), "x": _numeric_cp("x")})


def test_excluded_column_in_mnar_columns_raises() -> None:
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    with pytest.raises(ValueError, match="mnar_columns.*'x'"):
        decide(_two_col_profile(), 100, config)


def test_excluded_column_in_per_column_strategy_raises() -> None:
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_strategy("x", ImputationStrategy.MICE)
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    with pytest.raises(ValueError, match="per_column_strategy.*'x'"):
        decide(_two_col_profile(), 100, config)


def test_excluded_column_in_per_column_constant_fill_raises() -> None:
    config = PipelineConfig()
    config.imputation.numeric.set_per_column_constant_fill("x", 0.0)
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    with pytest.raises(ValueError, match="per_column_constant_fill.*'x'"):
        decide(_two_col_profile(), 100, config)


def test_excluded_column_in_add_indicator_columns_raises() -> None:
    config = PipelineConfig()
    config.imputation.add_indicator_column("x")
    config.add_phase_exclusion(PipelinePhase.Imputation, "x")
    with pytest.raises(ValueError, match="add_indicator_columns.*'x'"):
        decide(_two_col_profile(), 100, config)


def test_hard_exclusion_triggers_the_contradiction_raise() -> None:
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    config.add_exclusion("x")
    with pytest.raises(ValueError, match="'x'"):
        decide(_two_col_profile(), 100, config)


def test_all_offending_columns_reported_in_one_raise() -> None:
    config = PipelineConfig()
    config.imputation.add_mnar_column("m")
    config.imputation.numeric.set_per_column_strategy("s", ImputationStrategy.MICE)
    config.imputation.numeric.set_per_column_constant_fill("c", 1.5)
    config.imputation.add_indicator_column("i")
    config.add_exclusion(["m", "s"])
    config.add_phase_exclusion(PipelinePhase.Imputation, ["c", "i"])
    profile = _profile(
        {name: _numeric_cp(name) for name in ("a", "m", "s", "c", "i")}
    )
    with pytest.raises(ValueError) as excinfo:
        decide(profile, 100, config)
    message = str(excinfo.value)
    for name in ("'m'", "'s'", "'c'", "'i'"):
        assert name in message


def test_contradiction_raise_fires_before_routing() -> None:
    """The raise is config-vs-config: it fires even for unprofiled columns."""
    config = PipelineConfig()
    config.imputation.add_mnar_column("ghost")
    config.add_phase_exclusion(PipelinePhase.Imputation, "ghost")
    with pytest.raises(ValueError, match="'ghost'"):
        decide(_two_col_profile(), 100, config)


def test_other_phase_exclusion_does_not_trigger_the_raise() -> None:
    config = PipelineConfig()
    config.imputation.add_mnar_column("x")
    config.add_phase_exclusion(PipelinePhase.Profiling, "x")
    plan = decide(_two_col_profile(), 100, config)
    assert plan.column_decisions["x"].strategy == ImputationStrategy.MNAR
