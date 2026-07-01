"""
Unit tests for PipelineConfig.set_column_type and set_columns_type.

All tests are pure — no DataFrames, no StructuralProfiler.
resolve_active_columns coverage lives in tests/unit/profiling/test_pipeline_config.py.
"""

import pytest

from dataforge_ml.config import PipelineConfig, PipelinePhase, SemanticType


# ---------------------------------------------------------------------------
# set_column_type — single column override
# ---------------------------------------------------------------------------


def test_set_column_type_accepts_enum_value():
    cfg = PipelineConfig()
    cfg.set_column_type("score", SemanticType.Categorical)
    assert cfg.column_overrides["score"] is SemanticType.Categorical


def test_set_column_type_accepts_valid_string():
    cfg = PipelineConfig()
    cfg.set_column_type("score", "numeric")
    assert cfg.column_overrides["score"] is SemanticType.Numeric


def test_set_column_type_raises_on_unknown_string():
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="invalid_type"):
        cfg.set_column_type("score", "invalid_type")


def test_set_column_type_error_message_lists_valid_values():
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="numeric"):
        cfg.set_column_type("score", "not_a_type")


def test_set_column_type_overwrites_existing_override():
    cfg = PipelineConfig()
    cfg.set_column_type("col", SemanticType.Numeric)
    cfg.set_column_type("col", SemanticType.Categorical)
    assert cfg.column_overrides["col"] is SemanticType.Categorical


# ---------------------------------------------------------------------------
# set_column_type — bulk override
# ---------------------------------------------------------------------------


def test_set_columns_type_sets_all_listed_columns():
    cfg = PipelineConfig()
    cfg.set_column_type(["a", "b", "c"], SemanticType.Categorical)
    assert cfg.column_overrides["a"] is SemanticType.Categorical
    assert cfg.column_overrides["b"] is SemanticType.Categorical
    assert cfg.column_overrides["c"] is SemanticType.Categorical


def test_set_columns_type_accepts_valid_string():
    cfg = PipelineConfig()
    cfg.set_column_type(["x", "y"], "boolean")
    assert cfg.column_overrides["x"] is SemanticType.Boolean
    assert cfg.column_overrides["y"] is SemanticType.Boolean


def test_set_columns_type_raises_on_unknown_string():
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="bad_type"):
        cfg.set_column_type(["col"], "bad_type")


def test_set_columns_type_empty_list_is_a_no_op():
    cfg = PipelineConfig()
    cfg.set_column_type([], SemanticType.Text)
    assert cfg.column_overrides == {}


def test_set_columns_type_overwrites_existing_overrides():
    cfg = PipelineConfig()
    cfg.set_column_type("col", SemanticType.Numeric)
    cfg.set_column_type(["col"], SemanticType.Text)
    assert cfg.column_overrides["col"] is SemanticType.Text


def test_set_columns_type_storage_remains_mapping_proxy():
    from types import MappingProxyType
    cfg = PipelineConfig()
    cfg.set_column_type(["p", "q"], SemanticType.Datetime)
    assert isinstance(cfg.column_overrides, MappingProxyType)
    assert all(isinstance(k, str) for k in cfg.column_overrides)
    assert all(isinstance(v, SemanticType) for v in cfg.column_overrides.values())


def test_set_column_type_singular_unaffected_by_bulk_method():
    cfg = PipelineConfig()
    cfg.set_column_type("solo", SemanticType.Identifier)
    cfg.set_column_type(["a", "b"], SemanticType.Categorical)
    assert cfg.column_overrides["solo"] is SemanticType.Identifier
    assert cfg.column_overrides["a"] is SemanticType.Categorical


# ---------------------------------------------------------------------------
# add_exclusion — adding to hard exclusion set
# ---------------------------------------------------------------------------


def test_add_exclusions_single_column_added_to_exclude_columns():
    cfg = PipelineConfig()
    cfg.add_exclusion(["id"])
    assert "id" in cfg.exclude_columns


def test_add_exclusions_input_duplicates_are_deduplicated():
    cfg = PipelineConfig()
    cfg.add_exclusion(["id", "id", "id"])
    assert cfg.exclude_columns.count("id") == 1


def test_add_exclusions_repeated_calls_with_overlapping_lists_do_not_double_add():
    cfg = PipelineConfig()
    cfg.add_exclusion(["id"])
    cfg.add_exclusion(["id", "key"])
    assert cfg.exclude_columns.count("id") == 1
    assert "key" in cfg.exclude_columns


def test_add_exclusions_resolve_active_columns_excludes_added_columns():
    cfg = PipelineConfig()
    cfg.add_exclusion(["id"])
    result = cfg.resolve_active_columns(PipelinePhase.Profiling, ["id", "age", "score"])
    assert "id" not in result
    assert result == ["age", "score"]


def test_add_exclusions_empty_list_is_a_no_op():
    cfg = PipelineConfig()
    cfg.add_exclusion(["id"])
    cfg.add_exclusion([])
    assert cfg.exclude_columns == ("id",)


def test_add_exclusions_empty_list_on_fresh_config_leaves_exclude_columns_empty():
    cfg = PipelineConfig()
    cfg.add_exclusion([])
    assert cfg.exclude_columns == ()

def test_direct_write_attempts_raise_errors():
    cfg = PipelineConfig()
    cfg.add_exclusion("x")
    cfg.set_column_type("y", "numeric")
    cfg.set_numeric_kind("z", "continuous")
    cfg.add_phase_exclusion("scaling", "w")
    
    with pytest.raises(TypeError):
        cfg.column_overrides["x"] = SemanticType.Numeric
        
    with pytest.raises(TypeError):
        cfg.numeric_kind_overrides["x"] = "continuous"
        
    with pytest.raises(AttributeError):
        cfg.exclude_columns.append("y")
        
    with pytest.raises(TypeError):
        cfg.phase_exclusions["scaling"] = ("y",)

def test_from_dict_raises_on_invalid_entry():
    d = {
        "column_overrides": {"score": "not_a_type"},
    }
    with pytest.raises(ValueError, match="not_a_type"):
        PipelineConfig.from_dict(d)
