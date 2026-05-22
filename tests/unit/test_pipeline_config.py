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
# set_columns_type — bulk override
# ---------------------------------------------------------------------------


def test_set_columns_type_sets_all_listed_columns():
    cfg = PipelineConfig()
    cfg.set_columns_type(["a", "b", "c"], SemanticType.Categorical)
    assert cfg.column_overrides["a"] is SemanticType.Categorical
    assert cfg.column_overrides["b"] is SemanticType.Categorical
    assert cfg.column_overrides["c"] is SemanticType.Categorical


def test_set_columns_type_accepts_valid_string():
    cfg = PipelineConfig()
    cfg.set_columns_type(["x", "y"], "boolean")
    assert cfg.column_overrides["x"] is SemanticType.Boolean
    assert cfg.column_overrides["y"] is SemanticType.Boolean


def test_set_columns_type_raises_on_unknown_string():
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="bad_type"):
        cfg.set_columns_type(["col"], "bad_type")


def test_set_columns_type_empty_list_is_a_no_op():
    cfg = PipelineConfig()
    cfg.set_columns_type([], SemanticType.Text)
    assert cfg.column_overrides == {}


def test_set_columns_type_overwrites_existing_overrides():
    cfg = PipelineConfig()
    cfg.set_column_type("col", SemanticType.Numeric)
    cfg.set_columns_type(["col"], SemanticType.Text)
    assert cfg.column_overrides["col"] is SemanticType.Text


def test_set_columns_type_storage_remains_column_to_type_dict():
    cfg = PipelineConfig()
    cfg.set_columns_type(["p", "q"], SemanticType.Datetime)
    assert isinstance(cfg.column_overrides, dict)
    assert all(isinstance(k, str) for k in cfg.column_overrides)
    assert all(isinstance(v, SemanticType) for v in cfg.column_overrides.values())


def test_set_column_type_singular_unaffected_by_bulk_method():
    cfg = PipelineConfig()
    cfg.set_column_type("solo", SemanticType.Identifier)
    cfg.set_columns_type(["a", "b"], SemanticType.Categorical)
    assert cfg.column_overrides["solo"] is SemanticType.Identifier
    assert cfg.column_overrides["a"] is SemanticType.Categorical
