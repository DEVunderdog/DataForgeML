"""
CorrelationProfiler  –  Phase 1 extension: Correlation & Information Structure.

profile_features(df, numeric_cols, categorical_cols) computes the pairwise
Pearson + Spearman matrices, Cramér's V and eta-squared pairs, and the
near-redundancy groups. StructuralProfiler runs it once per dataset and stores
the result on ``dataset.feature_correlation``. A target column is one of the
columns in the matrices, so its strength with each feature is read from there.
"""

from __future__ import annotations

import itertools

import polars as pl

from ..models._data_types import _NUMERIC_DTYPES
from ._base import DatasetLevelProfiler
from ._correlation_config import (
    CorrelationPair,
    CorrelationProfileConfig,
    CorrelationProfileResult,
    CramerVPair,
    EtaSquaredPair,
    NearRedundancyGroup,
)
from ._fallbacks import record_fallback

# ---------------------------------------------------------------------------
# Union-Find (unchanged)
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> list[list[str]]:
        from collections import defaultdict

        buckets: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            buckets[self.find(x)].append(x)
        return [sorted(m) for m in buckets.values() if len(m) > 1]


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class CorrelationProfiler(DatasetLevelProfiler[CorrelationProfileResult]):
    """
    Correlation and information-structure profiler.

    Parameters
    ----------
    numeric_columns : list[str]
        Columns to include in pairwise matrices.
    categorical_columns : list[str], optional
        Columns to include in the Cramér's V and eta-squared steps.
    config : CorrelationProfileConfig, optional
        Threshold configuration. Defaults to ``CorrelationProfileConfig()``
        which reproduces the library's original hard-coded behaviour.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str] | None = None,
        config: CorrelationProfileConfig | None = None,
    ) -> None:
        super().__init__()
        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns or []
        self._config = config if config is not None else CorrelationProfileConfig()

    # ------------------------------------------------------------------
    # Concrete implementation of the abstract base method
    # ------------------------------------------------------------------

    def profile(self, data: pl.DataFrame) -> CorrelationProfileResult:  # type: ignore[override]
        return self.profile_features(data, self._numeric_columns)

    # ------------------------------------------------------------------
    # Primary API  (called by StructuralProfiler)
    # ------------------------------------------------------------------

    def profile_features(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str] | None = None,
    ) -> CorrelationProfileResult:
        """
        Compute pairwise feature-feature correlation matrices.

        Pearson + Spearman for numeric pairs, Cramér's V for categorical pairs,
        eta-squared for numeric-categorical pairs.
        """
        result = CorrelationProfileResult()

        resolved_numeric = [
            c
            for c in numeric_cols
            if c in df.columns and df[c].dtype in _NUMERIC_DTYPES
        ]
        result.analysed_numeric_columns = resolved_numeric

        resolved_categorical = [c for c in (categorical_cols or []) if c in df.columns]
        result.analysed_categorical_columns = resolved_categorical

        if len(resolved_numeric) >= 2:
            pearson_mat, spearman_mat = self._compute_matrices(df, resolved_numeric)
            result.pearson_matrix = pearson_mat
            result.spearman_matrix = spearman_mat

            pairs = self._build_pairs(resolved_numeric, pearson_mat, spearman_mat)
            result.pairwise = pairs
            result.near_redundant_pairs = [p for p in pairs if p.near_redundant]
            result.near_redundancy_groups = self._build_redundancy_groups(
                result.near_redundant_pairs
            )

        if len(resolved_categorical) >= 2:
            result.cramer_v_pairs = self._compute_cramer_v_pairs(
                df, resolved_categorical, self._config.near_redundant_cramer_v_threshold
            )
            result.near_redundant_cramer_v_pairs = [
                p for p in result.cramer_v_pairs if p.near_redundant
            ]

        if resolved_numeric and resolved_categorical:
            result.eta_squared_pairs = self._compute_eta_squared_pairs(
                df,
                resolved_numeric,
                resolved_categorical,
                self._config.near_redundant_eta_squared_threshold,
            )
            result.near_redundant_eta_squared_pairs = [
                p for p in result.eta_squared_pairs if p.near_redundant
            ]

        return result

    # ------------------------------------------------------------------
    # Step 1–2: Pearson + Spearman matrices (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_matrices(
        df: pl.DataFrame,
        cols: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        pearson_mat: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        spearman_mat: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}

        pairs = list(itertools.combinations(cols, 2))
        if not pairs:
            return pearson_mat, spearman_mat

        numeric_frame = df.select([pl.col(c).cast(pl.Float64).alias(c) for c in cols])
        rank_frame = numeric_frame.select(
            [pl.col(c).rank(method="average").alias(c) for c in cols]
        )

        pearson_exprs = [
            pl.corr(col_a, col_b, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"p|{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]
        spearman_exprs = [
            pl.corr(col_a, col_b, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"s|{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]

        p_row = numeric_frame.select(pearson_exprs).row(0)
        s_row = rank_frame.select(spearman_exprs).row(0)

        for i, (col_a, col_b) in enumerate(pairs):
            p_r = max(-1.0, min(1.0, float(p_row[i])))
            s_r = max(-1.0, min(1.0, float(s_row[i])))
            pearson_mat[col_a][col_b] = p_r
            pearson_mat[col_b][col_a] = p_r
            spearman_mat[col_a][col_b] = s_r
            spearman_mat[col_b][col_a] = s_r

        return pearson_mat, spearman_mat

    # ------------------------------------------------------------------
    # Step 3–4: Near-redundancy (unchanged)
    # ------------------------------------------------------------------

    def _build_pairs(
        self,
        cols: list[str],
        pearson_mat: dict[str, dict[str, float]],
        spearman_mat: dict[str, dict[str, float]],
    ) -> list[CorrelationPair]:
        pairs: list[CorrelationPair] = []
        for col_a, col_b in itertools.combinations(cols, 2):
            p_r = pearson_mat.get(col_a, {}).get(col_b)
            s_r = spearman_mat.get(col_a, {}).get(col_b)
            near_redundant = (
                p_r is not None
                and abs(p_r) > self._config.near_redundant_pearson_threshold
            ) or (
                s_r is not None
                and abs(s_r) > self._config.near_redundant_pearson_threshold
            )
            pairs.append(
                CorrelationPair(
                    col_a=col_a,
                    col_b=col_b,
                    pearson_r=p_r,
                    spearman_r=s_r,
                    near_redundant=near_redundant,
                )
            )
        return pairs

    @staticmethod
    def _build_redundancy_groups(
        near_redundant_pairs: list[CorrelationPair],
    ) -> list[NearRedundancyGroup]:
        uf = _UnionFind()
        for pair in near_redundant_pairs:
            uf.union(pair.col_a, pair.col_b)
        return [
            NearRedundancyGroup(columns=members, suggested_drop=members[1:])
            for members in uf.groups()
        ]

    # ------------------------------------------------------------------
    # Step 3b: Cramér's V — categorical ↔ categorical
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cramer_v_pairs(
        df: pl.DataFrame,
        cat_cols: list[str],
        threshold: float,
    ) -> list[CramerVPair]:
        import numpy as np
        from scipy.stats import chi2_contingency

        pairs: list[CramerVPair] = []
        for col_a, col_b in itertools.combinations(cat_cols, 2):
            pair_df = df.select(
                [
                    pl.col(col_a).cast(pl.Utf8, strict=False),
                    pl.col(col_b).cast(pl.Utf8, strict=False),
                ]
            ).drop_nulls()
            n = pair_df.height
            if n < 5:
                pairs.append(CramerVPair(col_a=col_a, col_b=col_b))
                continue

            counts = pair_df.group_by([col_a, col_b]).agg(pl.len().alias("count"))
            a_unique = sorted(counts[col_a].unique().to_list())
            b_unique = sorted(counts[col_b].unique().to_list())
            if len(a_unique) < 2 or len(b_unique) < 2:
                pairs.append(CramerVPair(col_a=col_a, col_b=col_b))
                continue

            a_idx = {v: i for i, v in enumerate(a_unique)}
            b_idx = {v: i for i, v in enumerate(b_unique)}
            ct = np.zeros((len(a_unique), len(b_unique)), dtype=int)
            for a_val, b_val, cnt in zip(
                counts[col_a].to_list(),
                counts[col_b].to_list(),
                counts["count"].to_list(),
            ):
                ct[a_idx[a_val], b_idx[b_val]] = cnt

            try:
                chi2, _, _, _ = chi2_contingency(ct)
                r, c = ct.shape
                phi2 = chi2 / n
                # Bergsma & Wicher (2013) bias correction
                phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
                r_corr = r - (r - 1) ** 2 / (n - 1)
                c_corr = c - (c - 1) ** 2 / (n - 1)
                denom = min(r_corr - 1, c_corr - 1)
                if denom <= 0:
                    # Near-saturated contingency table (n_unique ≈ n_rows):
                    # bias correction collapses denominator; skip the pair.
                    pairs.append(CramerVPair(col_a=col_a, col_b=col_b))
                    continue
                v = float(np.sqrt(phi2_corr / denom))
                v = max(0.0, min(1.0, v))
            except Exception:  # noqa: BLE001 - any scipy failure on a pair is a recorded fallback, not a crash
                record_fallback("association_failed", col_a)
                record_fallback("association_failed", col_b)
                pairs.append(CramerVPair(col_a=col_a, col_b=col_b))
                continue

            pairs.append(
                CramerVPair(
                    col_a=col_a,
                    col_b=col_b,
                    cramer_v=v,
                    near_redundant=v > threshold,
                )
            )

        return pairs

    # ------------------------------------------------------------------
    # Step 3c: Eta-squared — numeric ↔ categorical
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_eta_squared_pairs(
        df: pl.DataFrame,
        numeric_cols: list[str],
        cat_cols: list[str],
        threshold: float,
    ) -> list[EtaSquaredPair]:
        from scipy.stats import f_oneway

        pairs: list[EtaSquaredPair] = []
        for num_col in numeric_cols:
            feat = df[num_col].cast(pl.Float64)
            valid_feat = feat.drop_nulls()
            if valid_feat.len() == 0:
                continue
            grand_mean = float(valid_feat.mean())  # type: ignore[arg-type]
            ss_total = float(((valid_feat - grand_mean) ** 2).sum() or 0.0)

            for cat_col in cat_cols:
                target = df[cat_col]
                categories = target.drop_nulls().unique().to_list()
                groups = [
                    feat.filter(target == cat).drop_nulls().to_numpy()
                    for cat in categories
                ]
                non_empty = [g for g in groups if len(g) > 0]
                if len(non_empty) < 2:
                    pairs.append(
                        EtaSquaredPair(numeric_col=num_col, categorical_col=cat_col)
                    )
                    continue
                try:
                    f_oneway(*non_empty)
                    ss_between = sum(
                        len(g) * (float(g.mean()) - grand_mean) ** 2 for g in non_empty
                    )
                    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
                    eta_sq = max(0.0, min(1.0, eta_sq))
                except Exception:  # noqa: BLE001 - any scipy failure on a pair is a recorded fallback, not a crash
                    record_fallback("association_failed", num_col)
                    record_fallback("association_failed", cat_col)
                    pairs.append(
                        EtaSquaredPair(numeric_col=num_col, categorical_col=cat_col)
                    )
                    continue

                pairs.append(
                    EtaSquaredPair(
                        numeric_col=num_col,
                        categorical_col=cat_col,
                        eta_squared=eta_sq,
                        near_redundant=eta_sq > threshold,
                    )
                )

        return pairs
