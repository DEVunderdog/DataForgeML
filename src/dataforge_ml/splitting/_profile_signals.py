"""
Profile-driven stratification signal extraction.

Converts a StructuralProfileResult into a binary label matrix (n_rows × n_signals)
suitable for MultilabelStratifiedShuffleSplit / MultilabelStratifiedKFold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from ..config import SemanticType
from ..profiling._config import StructuralProfileResult
from ..profiling._boolean_config import BooleanStats
from ..profiling._categorical_config import CategoricalStats
from ..profiling._numeric_config import NumericStats, SkewSeverity
from dataforge_ml.utils._null_detection import _SENTINEL_STRINGS, _inf_eligible, _sentinel_eligible

_MAX_SIGNALS = 50
_RARE_THRESHOLD = 0.05
_BUCKET_LABELS = ["q1", "q2", "q3", "q4", "q5"]


def build_label_matrix(
    df: pl.DataFrame,
    profile: StructuralProfileResult,
    target: Optional[str],
    max_signals: int = _MAX_SIGNALS,
) -> np.ndarray:
    """
    Return a (n_rows, n_signals) int8 label matrix for multilabel stratification.

    Signals with zero proportion (all zeros) are dropped before capping.
    When total signals exceed max_signals, the rarest signals (smallest
    proportion of 1s) are retained. Returns shape (n_rows, 0) when no
    usable signals exist, signalling the caller to fall back to random splitting.
    """
    n = len(df)
    signals: list[np.ndarray] = []

    # --- 1. Per-column missingness signal ---
    for col, cp in profile.columns.items():
        if cp.missingness is None or cp.missingness.effective_null_count == 0:
            continue
        if col not in df.columns:
            continue
        series = df[col]
        dtype = series.dtype
        std_null = series.is_null()
        if _sentinel_eligible(dtype):
            eff_null = (
                std_null
                | (series.str.strip_chars() == "")
                | series.str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
            )
        elif _inf_eligible(dtype):
            eff_null = std_null | series.is_nan() | series.is_infinite()
        else:
            eff_null = std_null
        s = eff_null.cast(pl.Int8).to_numpy()
        signals.append(s)

    # --- 2. Joint MAR missingness (correlated pairs, each pair once) ---
    seen_pairs: set[frozenset] = set()
    for col, cp in profile.columns.items():
        if cp.missingness is None:
            continue
        for partner in cp.missingness.correlated_with:
            pair: frozenset = frozenset({col, partner})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if col not in df.columns or partner not in df.columns:
                continue
            s = (df[col].is_null() | df[partner].is_null()).cast(pl.Int8).to_numpy()
            signals.append(s)

    # --- 3. Numeric extreme value signal (below p5 or above p95/p99) ---
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Numeric:
            continue
        if not isinstance(cp.stats, NumericStats):
            continue
        p = cp.stats.percentiles
        p_low = p.p5
        p_high = p.p99 if cp.stats.skewness_severity == SkewSeverity.Severe else p.p95
        if p_low is None or p_high is None:
            continue
        if col not in df.columns:
            continue
        col_s = df[col]
        condition = (col_s < p_low) | (col_s > p_high)
        s = condition.fill_null(False).cast(pl.Int8).to_numpy()
        signals.append(s)

    # --- 4. Zero/negative value signal (right-skewed numeric columns) ---
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Numeric:
            continue
        if not isinstance(cp.stats, NumericStats):
            continue
        if cp.stats.skewness is None or cp.stats.skewness <= 0:
            continue
        if cp.stats.skewness_severity in (None, SkewSeverity.Normal):
            continue
        if col not in df.columns:
            continue
        col_s = df[col]
        s = (col_s <= 0).fill_null(False).cast(pl.Int8).to_numpy()
        signals.append(s)

    # --- 5. Rare categorical label signal ---
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Categorical:
            continue
        if not isinstance(cp.stats, CategoricalStats):
            continue
        if col not in df.columns:
            continue
        rare_vals = cp.stats.rare_categories.rare_label_values
        if not rare_vals:
            continue
        s = df[col].is_in(rare_vals).cast(pl.Int8).to_numpy()
        signals.append(s)

    # --- 6. Boolean minority signal ---
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Boolean:
            continue
        if not isinstance(cp.stats, BooleanStats):
            continue
        if col not in df.columns:
            continue
        bs = cp.stats
        if bs.true_ratio < _RARE_THRESHOLD:
            minority_val = True
        elif bs.false_ratio < _RARE_THRESHOLD:
            minority_val = False
        else:
            continue
        s = (df[col] == minority_val).fill_null(False).cast(pl.Int8).to_numpy()
        signals.append(s)

    # --- 7. Target signal ---
    if target and target in df.columns:
        target_cp = profile.columns.get(target)
        if target_cp is not None and target_cp.semantic_type == SemanticType.Numeric:
            buckets = df[target].qcut(5, labels=_BUCKET_LABELS, allow_duplicates=True)
            for label in _BUCKET_LABELS:
                s = (buckets == label).fill_null(False).cast(pl.Int8).to_numpy()
                signals.append(s)
        else:
            for cls in df[target].unique().to_list():
                s = (df[target] == cls).fill_null(False).cast(pl.Int8).to_numpy()
                signals.append(s)

    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    # Drop zero-proportion signals (all zeros — useless for stratification)
    signals = [s for s in signals if s.sum() > 0]
    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    # Cap at max_signals: keep rarest (smallest proportion of 1s) first
    if len(signals) > max_signals:
        proportions = [float(s.mean()) for s in signals]
        ranked = sorted(range(len(signals)), key=lambda i: proportions[i])
        signals = [signals[i] for i in ranked[:max_signals]]

    return np.column_stack(signals).astype(np.int8)
