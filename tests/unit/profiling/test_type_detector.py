import pytest
import polars as pl

from dataforge_ml.profiling._type_detector import TypeDetector
from dataforge_ml.profiling._config import ColumnTypeInfo, NumericKind, SemanticType, TypeFlag
from dataforge_ml.profiling._type_detection_config import TypeDetectionConfig
from dataforge_ml.profiling.orchestrator import StructuralProfiler
from dataforge_ml.config import PipelineConfig


# ---------------------------------------------------------------------------
# Native pl.Boolean resolves to SemanticType.Boolean
# ---------------------------------------------------------------------------


def test_native_boolean_column_resolves_to_boolean():
    df = pl.DataFrame(
        {"flag": pl.Series([True, False, True, True, False], dtype=pl.Boolean)}
    )
    info = TypeDetector(columns=["flag"]).detect(df)["flag"]
    assert info.semantic_type == SemanticType.Boolean


# ---------------------------------------------------------------------------
# High-cardinality string column resolves to Categorical or Text (not Numeric)
# ---------------------------------------------------------------------------


def test_high_cardinality_string_not_numeric():
    # 80 rows, 40 distinct short strings — high cardinality but below
    # the 99% identifier threshold so it stays Categorical/Text.
    vals = ["item_" + str(i % 40) for i in range(80)]
    df = pl.DataFrame({"name": pl.Series(vals, dtype=pl.Utf8)})
    info = TypeDetector(columns=["name"]).detect(df)["name"]
    assert info.semantic_type in (SemanticType.Categorical, SemanticType.Text)
    assert info.semantic_type != SemanticType.Numeric


# ---------------------------------------------------------------------------
# Identifier space-density guard: multi-word strings must not be Identifier
# ---------------------------------------------------------------------------


def test_unique_sentences_not_classified_as_identifier():
    # 100 unique short sentences — pass the 99% uniqueness threshold but
    # each value contains spaces, so the space-density guard must reject them.
    sentences = [f"The item {i} is ready" for i in range(100)]
    df = pl.DataFrame({"description": pl.Series(sentences, dtype=pl.Utf8)})
    info = TypeDetector(columns=["description"]).detect(df)["description"]
    assert info.semantic_type != SemanticType.Identifier


def test_unique_uuids_classified_as_identifier():
    # 100 unique UUID-like tokens — no spaces, short, 100% unique.
    uuids = [f"3f2a1b-{i:04d}-cd90-ef12" for i in range(100)]
    df = pl.DataFrame({"id": pl.Series(uuids, dtype=pl.Utf8)})
    info = TypeDetector(columns=["id"]).detect(df)["id"]
    assert info.semantic_type == SemanticType.Identifier


def test_unique_short_codes_classified_as_identifier():
    # 100 unique alphanumeric codes — no spaces, 100% unique.
    codes = [f"A{i:03d}" for i in range(100)]
    df = pl.DataFrame({"sku": pl.Series(codes, dtype=pl.Utf8)})
    info = TypeDetector(columns=["sku"]).detect(df)["sku"]
    assert info.semantic_type == SemanticType.Identifier


# ---------------------------------------------------------------------------
# Titanic-style Name column: high-cardinality multi-word strings → Text
# ---------------------------------------------------------------------------


def test_titanic_name_column_classified_as_text():
    # Reproduces the Titanic Name column: near-unique, multi-word, medium length.
    # Previously misclassified as Categorical because thresholds were too high.
    names = [
        "Braund, Mr. Owen Harris",
        "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
        "Heikkinen, Miss. Laina",
        "Futrelle, Mrs. Jacques Heath (Lily May Peel)",
        "Allen, Mr. William Henry",
        "Moran, Mr. James",
        "McCarthy, Mr. Timothy J",
        "Palsson, Master. Gosta Leonard",
        "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)",
        "Nasser, Mrs. Nicholas (Adele Achem)",
    ] * 90  # 900 rows, near-100% unique
    df = pl.DataFrame({"Name": pl.Series(names[:891], dtype=pl.Utf8)})
    info = TypeDetector(columns=["Name"]).detect(df)["Name"]
    assert info.semantic_type == SemanticType.Text


def test_low_cardinality_short_strings_stay_categorical():
    # S/C/Q — exactly the kind of column that must NOT be Text.
    vals = (["S"] * 644 + ["C"] * 168 + ["Q"] * 77 + [None] * 2)[:891]
    df = pl.DataFrame({"Embarked": pl.Series(vals, dtype=pl.Utf8)})
    info = TypeDetector(columns=["Embarked"]).detect(df)["Embarked"]
    assert info.semantic_type == SemanticType.Categorical


# ---------------------------------------------------------------------------
# Unique short names with spaces: high-cardinality-with-spaces path → Text
# ---------------------------------------------------------------------------


def test_short_full_names_classified_as_text():
    # "John Smith"-style names: short, exactly 2 words, near-unique.
    # Caught by the high-unique-with-spaces path.
    first = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Hank"]
    last = ["Smith", "Jones", "Brown", "Davis", "Wilson", "Moore", "Taylor", "Anderson"]
    names = [f"{f} {l}" for f in first for l in last]  # 64 unique names
    df = pl.DataFrame({"full_name": pl.Series(names, dtype=pl.Utf8)})
    info = TypeDetector(columns=["full_name"]).detect(df)["full_name"]
    assert info.semantic_type == SemanticType.Text


# ---------------------------------------------------------------------------
# TypeDetectionConfig — threshold override tests
# ---------------------------------------------------------------------------


def test_numeric_coerce_threshold_override_reclassifies_borderline_column():
    # 90 numeric strings + 10 non-numeric: 90% success rate.
    # Default threshold (0.95) → stays as string → Categorical.
    # Lowered threshold (0.85) → reclassified as Numeric.
    vals = [str(i) for i in range(90)] + ["abc"] * 10
    df = pl.DataFrame({"col": pl.Series(vals, dtype=pl.Utf8)})

    default_info = TypeDetector(columns=["col"]).detect(df)["col"]
    assert default_info.semantic_type != SemanticType.Numeric

    lowered = TypeDetectionConfig(numeric_coerce_threshold=0.85)
    override_info = TypeDetector(columns=["col"], config=lowered).detect(df)["col"]
    assert override_info.semantic_type == SemanticType.Numeric


def test_encoded_category_max_unique_override_reclassifies_integer_column():
    # Non-tight-sequence: 20 unique values at multiples of 10 (0, 10, 20, …, 190)
    # across 1000 rows (ratio = 0.02, well below encoded_category_max_ratio = 0.05).
    # Tight-sequence columns always use a hard limit of 50, so the config field
    # only takes effect on non-tight sequences.
    # Default max_unique (15) → absolute_ok is False (20 >= 15) → Numeric.
    # Raised max_unique (25) → absolute_ok is True  (20 < 25)  → Categorical.
    multiples = [i * 10 for i in range(20)]  # 0, 10, 20, …, 190
    vals = (multiples * 50)  # 1000 rows, 20 unique, non-tight sequence
    df = pl.DataFrame({"label": pl.Series(vals, dtype=pl.Int32)})

    default_info = TypeDetector(columns=["label"]).detect(df)["label"]
    assert default_info.semantic_type == SemanticType.Numeric

    raised = TypeDetectionConfig(encoded_category_max_unique=25)
    override_info = TypeDetector(columns=["label"], config=raised).detect(df)["label"]
    assert override_info.semantic_type == SemanticType.Categorical


def test_identifier_unique_ratio_override_reclassifies_high_cardinality_column():
    # 950 unique short token strings out of 1000 rows → 95% unique ratio.
    # Default identifier_unique_ratio (0.99): 0.95 <= 0.99 → NOT Identifier.
    # Lowered ratio (0.90): 0.95 > 0.90 → IS Identifier.
    tokens = [f"ID{i:04d}" for i in range(950)] + [f"ID{i:04d}" for i in range(50)]
    df = pl.DataFrame({"ref": pl.Series(tokens, dtype=pl.Utf8)})

    default_info = TypeDetector(columns=["ref"]).detect(df)["ref"]
    assert default_info.semantic_type != SemanticType.Identifier

    lowered = TypeDetectionConfig(identifier_unique_ratio=0.90)
    override_info = TypeDetector(columns=["ref"], config=lowered).detect(df)["ref"]
    assert override_info.semantic_type == SemanticType.Identifier


def test_type_detection_config_round_trip():
    cfg = TypeDetectionConfig(
        numeric_coerce_threshold=0.88,
        datetime_coerce_threshold=0.75,
        encoded_category_max_unique=20,
        encoded_category_max_ratio=0.03,
        identifier_unique_ratio=0.97,
        identifier_max_median_length=30,
        free_text_avg_words=4,
        free_text_median_chars=25,
        free_text_p90_chars=40,
        free_text_min_unique_ratio=0.50,
        free_text_high_unique_with_spaces=0.65,
    )
    restored = TypeDetectionConfig.from_dict(cfg.to_dict())
    assert restored.numeric_coerce_threshold == 0.88
    assert restored.datetime_coerce_threshold == 0.75
    assert restored.encoded_category_max_unique == 20
    assert restored.encoded_category_max_ratio == 0.03
    assert restored.identifier_unique_ratio == 0.97
    assert restored.identifier_max_median_length == 30
    assert restored.free_text_avg_words == 4
    assert restored.free_text_median_chars == 25
    assert restored.free_text_p90_chars == 40
    assert restored.free_text_min_unique_ratio == 0.50
    assert restored.free_text_high_unique_with_spaces == 0.65


# ---------------------------------------------------------------------------
# _classify_numeric_kind — four-signal compound test (deep-module unit tests)
# ---------------------------------------------------------------------------

# Helper: call _classify_numeric_kind in isolation and return the resulting kind.
def _classify(series: pl.Series, n_rows: int) -> NumericKind:
    detector = TypeDetector(columns=[series.name])
    info = ColumnTypeInfo(
        column=series.name,
        original_dtype=str(series.dtype),
        inferred_dtype=str(series.dtype),
    )
    detector._classify_numeric_kind(series, info, n_rows)
    return info.numeric_kind


def test_likert_five_level_is_bounded_discrete():
    # {1,2,3,4,5} large n_rows: all four signals pass → BoundedDiscrete
    s = pl.Series("rating", [1, 2, 3, 4, 5], dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.BoundedDiscrete


def test_binary_zero_one_is_bounded_discrete():
    # {0,1}: min=0, tight sequence, range=1 ≤ 20, 2 unique ≤ 10 → BoundedDiscrete
    s = pl.Series("flag", [0, 1, 0, 1, 0], dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.BoundedDiscrete


def test_day_of_week_is_bounded_discrete():
    # {0,...,6}: min=0, tight, range=6 ≤ 20, 7 unique ≤ 10 → BoundedDiscrete
    s = pl.Series("dow", list(range(7)), dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.BoundedDiscrete


def test_gapped_integers_fail_signal_1():
    # {18,22,35,42,55}: range_span=38 ≠ n_unique=5 → Continuous
    s = pl.Series("vals", [18, 22, 35, 42, 55], dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.Continuous


def test_tight_non_zero_origin_fails_signal_4():
    # {18,19,20,21,22}: tight, range=4 ≤ 20, 5 unique ≤ 10, but min=18 ≠ 0/1 → Continuous
    s = pl.Series("vals", list(range(18, 23)), dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.Continuous


def test_year_range_fails_signal_4():
    # {2000,...,2010}: tight, range=10 ≤ 20, 11 unique, but min=2000 ≠ 0/1 → Continuous
    s = pl.Series("year", list(range(2000, 2011)), dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.Continuous


def test_wide_consecutive_range_fails_signal_2():
    # {1,...,25}: tight, origin=1, 25 unique, but range=24 > 20 → Continuous
    s = pl.Series("vals", list(range(1, 26)), dtype=pl.Int64)
    assert _classify(s, n_rows=1000) == NumericKind.Continuous


def test_small_dataset_absolute_floor_saves_bounded_discrete():
    # {1,...,5} in 20-row dataset: ratio=0.25 ≥ 0.05 but n_unique=5 ≤ 10 → BoundedDiscrete
    s = pl.Series("rating", [1, 2, 3, 4, 5], dtype=pl.Int64)
    assert _classify(s, n_rows=20) == NumericKind.BoundedDiscrete


def test_eleven_unique_fails_signal_3_on_ratio_and_floor():
    # {1,...,11} in 20-row dataset: ratio=0.55 ≥ 0.05 AND n_unique=11 > 10 → Continuous
    s = pl.Series("vals", list(range(1, 12)), dtype=pl.Int64)
    assert _classify(s, n_rows=20) == NumericKind.Continuous


def test_whole_number_float_likert_is_bounded_discrete():
    # {1.0,...,5.0}: passes float pre-check (all whole numbers), all four signals pass → BoundedDiscrete
    s = pl.Series("rating", [1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64)
    assert _classify(s, n_rows=1000) == NumericKind.BoundedDiscrete


def test_fractional_float_fails_pre_check():
    # {0.1, 0.5, 1.0, 1.5, 2.0}: fractional values → Continuous immediately
    s = pl.Series("vals", [0.1, 0.5, 1.0, 1.5, 2.0], dtype=pl.Float64)
    assert _classify(s, n_rows=1000) == NumericKind.Continuous


def test_float_binary_is_bounded_discrete():
    # {0.0, 1.0}: passes float pre-check, min=0, tight, 2 unique ≤ 10 → BoundedDiscrete
    s = pl.Series("flag", [0.0, 1.0, 0.0, 1.0], dtype=pl.Float64)
    assert _classify(s, n_rows=1000) == NumericKind.BoundedDiscrete


# ---------------------------------------------------------------------------
# Module 3: numeric_kind_overrides — orchestrator Step 5
# ---------------------------------------------------------------------------


def _numeric_df() -> pl.DataFrame:
    """Minimal DataFrame with one auto-detected Continuous numeric column.

    Duplicates ensure unique_ratio < 0.99 (not Identifier); gaps between
    values ensure the tight-sequence signal fails (not BoundedDiscrete).
    """
    ages = [25, 32, 45, 28, 25, 32, 45, 28, 61, 39, 50, 22, 47, 33, 25, 32]
    return pl.DataFrame({"age": ages})




def test_numeric_kind_override_sets_kind_on_column_profile():
    cfg = PipelineConfig()
    cfg.set_numeric_kind("age", NumericKind.BoundedDiscrete)
    result = StructuralProfiler(cfg).profile(_numeric_df())
    assert result.columns["age"].numeric_kind == NumericKind.BoundedDiscrete


def test_numeric_kind_override_appends_numeric_kind_override_flag():
    cfg = PipelineConfig()
    cfg.set_numeric_kind("age", NumericKind.BoundedDiscrete)
    result = StructuralProfiler(cfg).profile(_numeric_df())
    assert TypeFlag.NumericKindOverride in result.columns["age"].type_flags


def test_numeric_kind_override_does_not_add_user_override_flag():
    cfg = PipelineConfig()
    cfg.set_numeric_kind("age", NumericKind.BoundedDiscrete)
    result = StructuralProfiler(cfg).profile(_numeric_df())
    assert TypeFlag.UserOverride not in result.columns["age"].type_flags


def test_numeric_kind_override_continuous_sets_kind_and_flag():
    # Forcing a Continuous column to Continuous explicitly still appends
    # NumericKindOverride so the audit log records the override was declared.
    cfg = PipelineConfig()
    cfg.set_numeric_kind("age", NumericKind.Continuous)
    result = StructuralProfiler(cfg).profile(_numeric_df())
    assert result.columns["age"].numeric_kind == NumericKind.Continuous
    assert TypeFlag.NumericKindOverride in result.columns["age"].type_flags


def test_numeric_kind_override_non_numeric_column_raises_value_error():
    df = pl.DataFrame({"category": ["a", "b", "c", "a", "b"]})
    cfg = PipelineConfig()
    cfg.set_numeric_kind("category", NumericKind.BoundedDiscrete)
    with pytest.raises(ValueError, match="category"):
        StructuralProfiler(cfg).profile(df)


def test_numeric_kind_override_error_message_names_actual_semantic_type():
    df = pl.DataFrame({"category": ["a", "b", "c", "a", "b"]})
    cfg = PipelineConfig()
    cfg.set_numeric_kind("category", NumericKind.BoundedDiscrete)
    with pytest.raises(ValueError, match="SemanticType.Numeric"):
        StructuralProfiler(cfg).profile(df)


def test_numeric_kind_override_semantic_type_override_to_categorical_then_kind_override_raises():
    # Explicitly push a Numeric column to Categorical via column_overrides, then
    # declare a NumericKind override on it — must raise because the final
    # SemanticType (after column_overrides is applied first) is Categorical.
    df = pl.DataFrame({"age": [25, 32, 45, 28, 25, 32, 45, 28, 61, 39, 50, 22, 47, 33, 25, 32]})
    cfg = PipelineConfig()
    cfg.set_column_type("age", SemanticType.Categorical)
    cfg.set_numeric_kind("age", NumericKind.BoundedDiscrete)
    with pytest.raises(ValueError, match="age"):
        StructuralProfiler(cfg).profile(df)


def test_numeric_kind_override_both_semantic_and_kind_override_sets_both_flags():
    # SemanticType override (to Numeric, which it already is) + NumericKind override.
    cfg = PipelineConfig()
    cfg.set_column_type("age", SemanticType.Numeric)
    cfg.set_numeric_kind("age", NumericKind.BoundedDiscrete)
    result = StructuralProfiler(cfg).profile(_numeric_df())
    cp = result.columns["age"]
    assert TypeFlag.UserOverride in cp.type_flags
    assert TypeFlag.NumericKindOverride in cp.type_flags


def test_numeric_kind_override_absent_column_is_silently_ignored():
    cfg = PipelineConfig()
    cfg.set_numeric_kind("nonexistent_column", NumericKind.BoundedDiscrete)
    # Must not raise; result.columns should not contain the absent column.
    result = StructuralProfiler(cfg).profile(_numeric_df())
    assert "nonexistent_column" not in result.columns
