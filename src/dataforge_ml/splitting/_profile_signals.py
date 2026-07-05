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
from ..profiling._config import NumericKind, StructuralProfileResult
from ..profiling._boolean_config import BooleanStats
from ..profiling._categorical_config import CategoricalStats
from ..profiling._numeric_config import NumericFlag, NumericStats, SkewSeverity
from ._config import SplitConfig
from ..utils._null_normalization import _resolve_effective_nulls

_BUCKET_LABELS = ["q1", "q2", "q3", "q4", "q5"]

# Signal-family priority for the ADR-0047 gate-3 redundancy collapse and the
# gate-4 cap. Lower rank = higher importance = the member that survives a
# redundant pair. Ordering mirrors the contract: target (never evictable) ->
# rare categorical -> boolean minority -> numeric extremes/skew -> missingness.
_PRIORITY_TARGET = 0
_PRIORITY_RARE_CATEGORICAL = 1
_PRIORITY_BOOLEAN = 2
_PRIORITY_NUMERIC = 3
_PRIORITY_MISSINGNESS = 4


def _abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute Pearson correlation between two binary signal vectors.

    Returns ``0.0`` when either vector has zero variance (a constant signal
    cannot be redundant with anything), avoiding the ``nan`` that
    ``np.corrcoef`` would otherwise produce.
    """
    sa = a.std()
    sb = b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def build_label_matrix(
    df: pl.DataFrame,
    profile: StructuralProfileResult,
    target: Optional[str],
    config: Optional[SplitConfig] = None,
    min_positives: int = 1,
) -> np.ndarray:
    """
    Return a (n_rows, n_signals) int8 label matrix for multilabel stratification.

    Each candidate signal must pass the viability gate (ADR-0047): it is
    admitted only when ``min(positives, negatives) >= min_positives``.  This
    two-sided check subsumes both the all-zeros drop and a symmetric all-ones
    drop — a signal that cannot place ``min_positives`` of both classes into
    every partition is not splittable and is excluded before capping.  The
    caller (which knows the split intent) computes ``min_positives``; the
    builder stays ignorant of split modes.

    Surviving signals then pass the redundancy gate (ADR-0047 gate 3), run
    before the cap.  Two signals whose absolute correlation reaches
    ``config.redundancy_correlation_threshold`` impose the same balancing
    constraint — correlation magnitude catches both near-identical (``+1``) and
    mirror-image (``−1``) pairs — and the lower-priority one is dropped, keeping
    the higher-priority family (target → rare categorical → boolean → numeric →
    missingness).  Additionally, a mutually-exclusive class/bucket set that
    survived the viability gate intact drops its last member dummy-style, which
    removes the binary-target double-count at the source.

    Signals that survive both gates are then capped (ADR-0047 gate 4) at
    ``min(config.max_stratification_signals, n_rows / config.rows_per_signal)``.
    Retention is by importance rank — target (never evictable) → rare
    categorical → boolean → numeric extremes/skew → missingness (last), with
    rarer viable signals kept first within a family — not by ascending
    proportion of 1s.  When the budget cannot support even the full target
    signal set, or no usable signals exist at all, the return has shape
    (n_rows, 0), signalling the caller to fall back to random splitting.

    Effective-null resolution (via ``_resolve_effective_nulls``) is applied
    to the full DataFrame once before any signal is computed. This means
    **all** signals — including the per-column missingness signal (signal 1),
    the Joint MAR pair signal (signal 2), and the compound row missingness
    signal (signal 8) — use the same dtype-driven effective-null mask:
    Float32/Float64 NaN/Inf values and String/Utf8 sentinel strings are treated
    as missing for both columns in every MAR pair and in the row null count.

    String/Utf8 effective-null detection uses **replace semantics** when
    ``profile.string_sentinels`` contains an entry for the column: only the
    declared values are matched (case-insensitive) and the hardcoded defaults
    (``_SENTINEL_STRINGS``) are suppressed for that column.  Columns with no
    declaration fall back to the hardcoded defaults unchanged.
    Empty/whitespace strings are always included regardless of any declaration.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame being split.
    profile : StructuralProfileResult
        Full Phase 1 result, including per-column missingness profiles and
        declared string/numeric sentinels.
    target : str, optional
        Name of the target column to generate class-distribution signals for.
        Pass ``None`` when there is no target.
    config : SplitConfig, optional
        Stratification thresholds.  Defaults to ``SplitConfig()`` when omitted.
    min_positives : int, optional
        Viability floor (ADR-0047 gate 2).  A signal is admitted only when it
        carries at least this many ones **and** at least this many zeros.
        Defaults to ``1`` (drop only all-zeros/all-ones signals).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_rows, n_signals)`` with dtype int8.
        Returns shape ``(n_rows, 0)`` when no usable signals exist.
    """
    _config = config if config is not None else SplitConfig()
    df = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    n = len(df)
    # Each signal is collected with its family priority (for the gate-3 redundancy
    # tie-break) and an optional mutually-exclusive group id (target class/bucket
    # sets, whose last member is dropped dummy-style in gate 3).
    signals: list[np.ndarray] = []
    priorities: list[int] = []
    groups: list[Optional[str]] = []

    def add(s: np.ndarray, priority: int, group: Optional[str] = None) -> None:
        signals.append(s)
        priorities.append(priority)
        groups.append(group)

    # --- 1. Per-column missingness signal ---
    for col, cp in profile.columns.items():
        if cp.missingness is None or cp.missingness.effective_null_count == 0:
            continue
        if col not in df.columns:
            continue
        s = df[col].is_null().cast(pl.Int8).to_numpy()
        add(s, _PRIORITY_MISSINGNESS)

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
            add(s, _PRIORITY_MISSINGNESS)

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
        add(s, _PRIORITY_NUMERIC)

    # --- 4. Zero/negative value signal (right-skewed numeric columns) ---
    # Gated on ``min <= 0`` in addition to ``SkewSeverity >= High``: a
    # strictly-positive column has no zero/negative values, so the flag would be
    # all-zeros and carry no information.
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Numeric:
            continue
        if not isinstance(cp.stats, NumericStats):
            continue
        if cp.stats.skewness is None or cp.stats.skewness <= 0:
            continue
        if cp.stats.skewness_severity in (None, SkewSeverity.Normal):
            continue
        if cp.stats.min is None or cp.stats.min > 0:
            continue
        if col not in df.columns:
            continue
        col_s = df[col]
        s = (col_s <= 0).fill_null(False).cast(pl.Int8).to_numpy()
        add(s, _PRIORITY_NUMERIC)

    # --- 4b. NearConstant numeric minority signal ---
    # One label per numeric column flagged ``NumericFlag.NearConstant`` (mode
    # frequency > 90%). A row gets ``1`` when its value falls in the minority
    # (off-mode) region: exact inequality to the mode for ``BoundedDiscrete``
    # columns, and outside a ``mode +/- 0.5 * IQR`` band for ``Continuous``
    # columns. Protects the structurally-rare non-constant rows that the extreme
    # value signal misses in symmetric near-constant distributions.
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Numeric:
            continue
        if not isinstance(cp.stats, NumericStats):
            continue
        if not cp.stats.has_flag(NumericFlag.NearConstant):
            continue
        mode = cp.stats.mode
        if mode is None:
            continue
        if col not in df.columns:
            continue
        col_s = df[col]
        if cp.numeric_kind == NumericKind.BoundedDiscrete:
            condition = col_s != mode
        else:
            iqr = cp.stats.iqr
            if iqr is None:
                continue
            band = 0.5 * iqr
            condition = (col_s < mode - band) | (col_s > mode + band)
        s = condition.fill_null(False).cast(pl.Int8).to_numpy()
        add(s, _PRIORITY_NUMERIC)

    # --- 4c. Bimodal minority-cluster signal ---
    # One label per numeric column flagged ``NumericFlag.Bimodal``. Each row is
    # assigned to its nearest of the two GMM centers (``BimodalStats.center1`` /
    # ``center2``); rows in the less-populous cluster are marked positive. The
    # minority cluster is decided by row counts on the DataFrame being split
    # (not ``minority_weight``, which carries no convention tying it to a
    # specific center). Protects Phase 2's Bimodal Imputation Framework, whose
    # cluster-conditional fill and GMM sampling both fit on ``train_df`` and are
    # blind to a minority mode that sits near the column center (invisible to
    # the numeric extreme-value signal), mirroring the NearConstant rationale.
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Numeric:
            continue
        if not isinstance(cp.stats, NumericStats):
            continue
        if not cp.stats.has_flag(NumericFlag.Bimodal):
            continue
        bimodal = cp.stats.bimodal_stats
        if bimodal is None:
            continue
        if col not in df.columns:
            continue
        col_s = df[col]
        d1 = (col_s - bimodal.center1).abs()
        d2 = (col_s - bimodal.center2).abs()
        in_cluster1 = d1 <= d2
        in_cluster2 = d1 > d2
        # Cluster sizes counted on the split DataFrame's non-null rows; ties in
        # distance fall to cluster 1 so the two counts partition the non-nulls.
        count1 = int(in_cluster1.fill_null(False).sum())
        count2 = int(in_cluster2.fill_null(False).sum())
        condition = in_cluster1 if count1 <= count2 else in_cluster2
        s = condition.fill_null(False).cast(pl.Int8).to_numpy()
        add(s, _PRIORITY_NUMERIC)

    # --- 5. Rare categorical label signals (one per rare value) ---
    # Each rare value gets its own binary signal so the stratification
    # guarantee is per-label presence in both splits, rather than an aggregate
    # rare-row count that could over-represent one rare value and starve
    # another.
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
        col_s = df[col]
        for val in rare_vals:
            s = (col_s == val).cast(pl.Int8).to_numpy()
            add(s, _PRIORITY_RARE_CATEGORICAL)

    # --- 6. Boolean minority signal ---
    for col, cp in profile.columns.items():
        if cp.semantic_type != SemanticType.Boolean:
            continue
        if not isinstance(cp.stats, BooleanStats):
            continue
        if col not in df.columns:
            continue
        bs = cp.stats
        if bs.true_ratio < _config.boolean_minority_threshold:
            minority_val = True
        elif bs.false_ratio < _config.boolean_minority_threshold:
            minority_val = False
        else:
            continue
        s = (df[col] == minority_val).fill_null(False).cast(pl.Int8).to_numpy()
        add(s, _PRIORITY_BOOLEAN)

    # --- 7. Two-tier target signal ---
    # Reuse the Phase 1 ``NumericKind`` classification instead of a local
    # heuristic.  A non-numeric target, or a numeric target classified
    # ``NumericKind.BoundedDiscrete``, emits one binary label per class.  Any
    # other numeric target (Continuous) is quantile-binned into
    # ``min(5, n_unique)`` buckets computed over its non-null values only.  A
    # dedicated "target missing" label is emitted whenever the target carries
    # nulls, so null-target rows are balanced across both partitions.
    #
    # Seam note: the class/bucket set is mutually exclusive, so its last member
    # is fully determined by the others and is therefore dummy-redundant.  Each
    # class/bucket signal is tagged with a shared group id so the ADR-0047 gate-3
    # slice drops that last member dummy-style; the "target missing" label is an
    # independent indicator (group ``None``) and is never part of that drop.
    if target and target in df.columns:
        target_cp = profile.columns.get(target)
        target_s = df[target]
        is_numeric = (
            target_cp is not None
            and target_cp.semantic_type == SemanticType.Numeric
        )
        is_bounded_discrete = (
            target_cp is not None
            and target_cp.numeric_kind == NumericKind.BoundedDiscrete
        )

        if is_numeric and not is_bounded_discrete:
            n_unique = target_s.drop_nulls().n_unique()
            if n_unique > 0:
                n_buckets = min(5, n_unique)
                labels = _BUCKET_LABELS[:n_buckets]
                buckets = target_s.qcut(
                    n_buckets, labels=labels, allow_duplicates=True
                )
                for label in labels:
                    s = (buckets == label).fill_null(False).cast(pl.Int8).to_numpy()
                    add(s, _PRIORITY_TARGET, group="target_bucket")
        else:
            for cls in target_s.drop_nulls().unique().to_list():
                s = (target_s == cls).fill_null(False).cast(pl.Int8).to_numpy()
                add(s, _PRIORITY_TARGET, group="target_class")

        if target_s.null_count() > 0:
            s = target_s.is_null().cast(pl.Int8).to_numpy()
            add(s, _PRIORITY_TARGET)

    # --- 8. Compound row missingness signal ---
    # Rows missing in more columns than the 90th-percentile per-row count are
    # globally sparse and must be proportionally represented in both partitions.
    # Signal is omitted when p90 == 0 (no meaningful multi-column sparsity).
    p90 = profile.dataset.row_distribution.row_missingness_p90
    if p90 > 0:
        profile_cols = [c for c in profile.columns if c in df.columns]
        if profile_cols:
            row_null_count = df.select(profile_cols).select(
                pl.sum_horizontal(pl.all().is_null()).alias("n")
            )["n"]
            s = (row_null_count > p90).cast(pl.Int8).to_numpy()
            add(s, _PRIORITY_MISSINGNESS)

    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    # Original per-group membership counts, captured before the viability gate so
    # gate 3's dummy-drop only fires on a *fully surviving* mutually-exclusive set
    # (dropping the last member is only sound when the set is still exhaustive).
    group_original: dict[str, int] = {}
    for g in groups:
        if g is not None:
            group_original[g] = group_original.get(g, 0) + 1

    # Viability gate (ADR-0047 gate 2): admit a signal only when it carries at
    # least ``min_positives`` ones and at least ``min_positives`` zeros. This
    # two-sided check subsumes both the all-zeros and all-ones drops.
    keep = [
        i
        for i in range(len(signals))
        if min(int(signals[i].sum()), n - int(signals[i].sum())) >= min_positives
    ]
    signals = [signals[i] for i in keep]
    priorities = [priorities[i] for i in keep]
    groups = [groups[i] for i in keep]
    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    # Redundancy gate (ADR-0047 gate 3), before the cap.
    #
    # 3a. Dummy-drop: within each mutually-exclusive class/bucket set that
    # survived viability intact, the last member is fully determined by the
    # others, so drop it. This removes the mirror-image double-count of a binary
    # target at the source (two class flags collapse to one).
    group_survivors: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        if g is not None:
            group_survivors.setdefault(g, []).append(i)
    dropped: set[int] = set()
    for g, idxs in group_survivors.items():
        if len(idxs) >= 2 and len(idxs) == group_original[g]:
            dropped.add(idxs[-1])

    # 3b. Correlation collapse: two signals whose absolute correlation reaches
    # ``redundancy_correlation_threshold`` impose the same balancing constraint
    # (near +1 identical or near -1 mirror-image). Visiting in priority order and
    # keeping only the first representative of each redundant cluster guarantees
    # the higher-priority family's signal survives.
    threshold = _config.redundancy_correlation_threshold
    order = sorted(
        (i for i in range(len(signals)) if i not in dropped),
        key=lambda i: (priorities[i], i),
    )
    kept: list[int] = []
    kept_floats: list[np.ndarray] = []
    for i in order:
        arr = signals[i].astype(np.float64)
        if any(_abs_correlation(arr, ka) >= threshold for ka in kept_floats):
            dropped.add(i)
            continue
        kept.append(i)
        kept_floats.append(arr)

    survivors = [i for i in range(len(signals)) if i not in dropped]
    signals = [signals[i] for i in survivors]
    priorities = [priorities[i] for i in survivors]
    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    # Cap (ADR-0047 gate 4): retain by importance rank, not by rarity. The cap
    # is co-bounded by the configured maximum and the rows-per-signal budget
    # (the data can support at most ``n / rows_per_signal`` simultaneous
    # constraints). Target signals are never evicted; when even the target
    # cannot fit the budget the matrix is emptied so the splitter falls back to
    # a random split.
    max_signals = min(
        _config.max_stratification_signals,
        int(n / _config.rows_per_signal),
    )
    if len(signals) > max_signals:
        n_target = sum(1 for p in priorities if p == _PRIORITY_TARGET)
        if n_target > max_signals:
            # The budget cannot support even the full target signal set; a
            # partial target is worthless, so fall back to a random split.
            return np.empty((n, 0), dtype=np.int8)
        # Rank by family priority (target -> rare categorical -> boolean ->
        # numeric -> missingness) then by rarity within a family (rarer viable
        # signals kept first). Target's priority 0 guarantees it always leads.
        proportions = [float(s.mean()) for s in signals]
        ranked = sorted(
            range(len(signals)),
            key=lambda i: (priorities[i], proportions[i], i),
        )
        keep_idx = sorted(ranked[:max_signals])
        signals = [signals[i] for i in keep_idx]

    if not signals:
        return np.empty((n, 0), dtype=np.int8)

    return np.column_stack(signals).astype(np.int8)


def unsplittable_train_mask(
    df: pl.DataFrame,
    profile: StructuralProfileResult,
    target: Optional[str],
    min_positives: int,
    config: Optional[SplitConfig] = None,
) -> np.ndarray:
    """
    Return a boolean mask of rows to route into the training split (ADR-0048).

    A target class or rare categorical value carrying fewer than
    ``min_positives`` rows falls below the viability floor (ADR-0047) and so
    cannot be balanced across both partitions — a single-row label physically
    cannot be split.  Every such row is marked ``True`` so the caller can
    deterministically reassign it into the training partition after
    ``iterstrat`` decides the rest of the split, guaranteeing the label is
    learned at fit time and preventing an unseen-category failure in Phase 5 at
    transform time.

    Only the discrete target branch is considered — a non-numeric target, or a
    numeric target classified ``NumericKind.BoundedDiscrete`` — mirroring the
    class-signal branch of ``build_label_matrix``.  A continuous quantile-binned
    target has no discrete class that could go unseen and is skipped.

    Effective-null resolution (via ``_resolve_effective_nulls``) is applied
    once before any label is tested, matching ``build_label_matrix`` so the same
    rows are considered.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame being split.
    profile : StructuralProfileResult
        Full Phase 1 result, including per-column stats and declared sentinels.
    target : str, optional
        Name of the target column.  Pass ``None`` when there is no target.
    min_positives : int
        Viability floor (ADR-0047).  A target class or rare categorical value
        with fewer rows than this cannot appear in both partitions and is
        routed to train.
    config : SplitConfig, optional
        Stratification thresholds.  Defaults to ``SplitConfig()`` when omitted.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(n_rows,)``; ``True`` where the row belongs to
        a sub-floor target class or rare categorical value.  All-``False`` when
        no such rows exist.
    """
    _config = config if config is not None else SplitConfig()
    df = _resolve_effective_nulls(
        df,
        numeric_sentinels=profile.numeric_sentinels,
        string_sentinels=profile.string_sentinels,
    )
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    # Rare categorical values too sparse to satisfy the viability floor.
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
        col_s = df[col]
        for val in rare_vals:
            hit = (col_s == val).fill_null(False).to_numpy()
            if int(hit.sum()) < min_positives:
                mask |= hit

    # Target classes too sparse to satisfy the viability floor (discrete branch
    # only; a continuous quantile-binned target has no discrete class).
    if target and target in df.columns:
        target_cp = profile.columns.get(target)
        target_s = df[target]
        is_numeric = (
            target_cp is not None
            and target_cp.semantic_type == SemanticType.Numeric
        )
        is_bounded_discrete = (
            target_cp is not None
            and target_cp.numeric_kind == NumericKind.BoundedDiscrete
        )
        if not (is_numeric and not is_bounded_discrete):
            for cls in target_s.drop_nulls().unique().to_list():
                hit = (target_s == cls).fill_null(False).to_numpy()
                if int(hit.sum()) < min_positives:
                    mask |= hit

    return mask
