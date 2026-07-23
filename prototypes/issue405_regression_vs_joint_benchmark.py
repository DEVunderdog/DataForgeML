"""PROTOTYPE — THROWAWAY. Measurement spike for issue #405.

Compares imputation accuracy (RMSE/MAE on imputed cells only) of three shapes
for columns that currently route to the per-column Regression strategy:

  (a) per_column  — today's Regression: N independent IterativeImputer fits,
                    each over [target] + every other numeric column, keeping
                    only the target's imputations (mirrors fit_regression_unit).
  (b) block_only  — one joint IterativeImputer over the missing-block columns
                    alone (today's MICE predictor breadth, fit_mice_unit).
  (c) full_joint  — one joint IterativeImputer over ALL numeric columns
                    (Regression's predictor breadth, single joint fit).

Estimators come from the library's own RegressionEstimatorFactory so they are
bit-identical to production. max_iter / tol mirror _decision_assembler's
_compute_max_iter / _compute_mice_max_iter formulas (r2_gap and pairwise-corr
signals treated as unavailable → no-op, as they degrade to in the library).
n_nearest_features=None throughout (block of 4 <= mice_n_nearest_features_min_cols;
for full_joint "full breadth" means all predictors by definition).

Run:  venv/bin/python prototypes/issue405_regression_vs_joint_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from dataforge_ml.imputation._config import ModelChoice
from dataforge_ml.imputation._regression_estimator_factory import (
    RegressionEstimatorFactory,
)

N_BLOCK = 4  # columns carrying missingness (the ones routing to Regression today)
N_COMPLETE = 8  # fully-observed numeric predictor columns
TARGET_MISS = 0.20


# ---------------------------------------------------------------- data gens
def gen_linear(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Latent-factor Gaussian data: block cols linearly tied to predictors."""
    z = rng.normal(size=(n_rows, 3))
    complete = np.column_stack(
        [
            z @ rng.normal(scale=1.0, size=3) + rng.normal(scale=0.6, size=n_rows)
            for _ in range(N_COMPLETE)
        ]
    )
    block = np.column_stack(
        [
            z @ rng.normal(scale=1.0, size=3) + rng.normal(scale=0.5, size=n_rows)
            for _ in range(N_BLOCK)
        ]
    )
    return np.column_stack([block, complete])


def gen_nonlinear(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    """Block cols are nonlinear functions of the predictors."""
    complete = rng.normal(size=(n_rows, N_COMPLETE))
    p = complete
    noise = lambda s: rng.normal(scale=s, size=n_rows)  # noqa: E731
    block = np.column_stack(
        [
            p[:, 0] * p[:, 1] + 0.5 * p[:, 2] + noise(0.4),
            np.sin(2.0 * p[:, 2]) + 0.6 * p[:, 3] ** 2 + noise(0.4),
            np.abs(p[:, 4]) * p[:, 5] + noise(0.4),
            np.tanh(p[:, 6]) + 0.5 * p[:, 0] * p[:, 7] + noise(0.4),
        ]
    )
    return np.column_stack([block, complete])


# ---------------------------------------------------------------- missingness
def mask_mcar(rng: np.random.Generator, data: np.ndarray) -> np.ndarray:
    masked = data.copy()
    for j in range(N_BLOCK):
        holes = rng.random(data.shape[0]) < TARGET_MISS
        masked[holes, j] = np.nan
    return masked


def mask_mar(rng: np.random.Generator, data: np.ndarray) -> np.ndarray:
    """Missingness in block col j driven by complete predictor j (never masked)."""
    masked = data.copy()
    for j in range(N_BLOCK):
        driver = data[:, N_BLOCK + j]
        # logistic in the driver, calibrated so the mean rate ~= TARGET_MISS
        z = (driver - np.median(driver)) / (np.std(driver) + 1e-12)
        prob = 1.0 / (1.0 + np.exp(-(1.5 * z)))
        prob *= TARGET_MISS / prob.mean()
        holes = rng.random(data.shape[0]) < np.clip(prob, 0, 0.9)
        masked[holes, j] = np.nan
    return masked


# ------------------------------------------------- hyperparameter mirrors
def _reg_max_iter(n_missing_features: int, complete_row_frac: float) -> int:
    # mirrors _decision_assembler._compute_max_iter (BayesianRidge/RF signals
    # for r2_gap and feature-corr treated as unavailable)
    base = 10 + n_missing_features * 2
    if complete_row_frac < 0.2:
        base += 5
    elif complete_row_frac < 0.5:
        base += 3
    return max(1, base)


def _mice_max_iter(block_miss_frac: float, complete_row_frac: float) -> int:
    # mirrors _decision_assembler._compute_mice_max_iter
    base = 10
    if block_miss_frac >= 0.4:
        base += 5
    elif block_miss_frac >= 0.2:
        base += 3
    elif block_miss_frac >= 0.1:
        base += 2
    if complete_row_frac < 0.2:
        base += 5
    elif complete_row_frac < 0.5:
        base += 3
    return max(1, base)


def _tol(cols: np.ndarray) -> float:
    # mirrors _compute_tol/_compute_mice_tol: min IQR across cols * 1e-4
    iqrs = []
    for j in range(cols.shape[1]):
        q75, q25 = np.nanpercentile(cols[:, j], [75, 25])
        if q75 - q25 > 0:
            iqrs.append(q75 - q25)
    return max(1e-7, min(iqrs) * 1e-4) if iqrs else 1e-3


def _estimator(choice: ModelChoice):
    return RegressionEstimatorFactory.build_from_choice(choice, n_jobs=4)


# ---------------------------------------------------------------- the shapes
def shape_per_column(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """(a) N independent fits, each [target] + all other columns, keep target."""
    n_cols = masked.shape[1]
    crf = float(np.mean(~np.isnan(masked).any(axis=1)))
    out = masked.copy()
    for t in range(N_BLOCK):
        order = [t] + [c for c in range(n_cols) if c != t]
        max_iter = _reg_max_iter(
            n_missing_features=N_BLOCK - 1, complete_row_frac=crf
        )
        imp = IterativeImputer(
            estimator=_estimator(choice),
            max_iter=max_iter,
            tol=_tol(masked[:, [t]]),
            random_state=0,
        )
        filled = imp.fit_transform(masked[:, order])
        out[:, t] = filled[:, 0]
    return out


def shape_block_only(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """(b) one joint fit over the block columns alone."""
    block = masked[:, :N_BLOCK]
    crf = float(np.mean(~np.isnan(masked).any(axis=1)))
    miss_frac = float(np.isnan(block).mean())
    imp = IterativeImputer(
        estimator=_estimator(choice),
        random_state=0,
        max_iter=_mice_max_iter(miss_frac, crf),
        tol=_tol(block),
        initial_strategy="mean",
        n_nearest_features=None,
    )
    out = masked.copy()
    out[:, :N_BLOCK] = imp.fit_transform(block)
    return out


def shape_full_joint(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """(c) one joint fit over all columns (full predictor breadth)."""
    crf = float(np.mean(~np.isnan(masked).any(axis=1)))
    miss_frac = float(np.isnan(masked[:, :N_BLOCK]).mean())
    imp = IterativeImputer(
        estimator=_estimator(choice),
        random_state=0,
        max_iter=_mice_max_iter(miss_frac, crf),
        tol=_tol(masked),
        initial_strategy="mean",
        n_nearest_features=None,
    )
    return imp.fit_transform(masked)


SHAPES = {
    "per_column": shape_per_column,
    "block_only": shape_block_only,
    "full_joint": shape_full_joint,
}


# ---------------------------------------------------------------- scoring
def score(truth: np.ndarray, masked: np.ndarray, filled: np.ndarray):
    holes = np.isnan(masked[:, :N_BLOCK])
    err = filled[:, :N_BLOCK][holes] - truth[:, :N_BLOCK][holes]
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def run_scenario(gen, masker, choice, n_rows, seeds):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth = gen(rng, n_rows)
        masked = masker(rng, truth)
        rec = {"seed": seed}
        for name, fn in SHAPES.items():
            t0 = time.perf_counter()
            filled = fn(masked, choice)
            elapsed = time.perf_counter() - t0
            rmse, mae = score(truth, masked, filled)
            rec[name] = {"rmse": rmse, "mae": mae, "time_s": elapsed}
        rows.append(rec)
        print(
            f"  seed {seed:>3}: "
            + " | ".join(
                f"{n} rmse={rec[n]['rmse']:.4f} ({rec[n]['time_s']:.1f}s)"
                for n in SHAPES
            ),
            flush=True,
        )
    return rows


def summarize(rows):
    out = {}
    for name in SHAPES:
        rmses = np.array([r[name]["rmse"] for r in rows])
        maes = np.array([r[name]["mae"] for r in rows])
        times = np.array([r[name]["time_s"] for r in rows])
        out[name] = {
            "rmse_mean": rmses.mean(),
            "rmse_std": rmses.std(),
            "mae_mean": maes.mean(),
            "time_mean": times.mean(),
        }
    # pairwise vs full_joint, ADR-0070 style
    for name in ("per_column", "block_only"):
        a = np.array([r[name]["rmse"] for r in rows])
        c = np.array([r["full_joint"]["rmse"] for r in rows])
        rel = (a - c) / c
        out[f"{name}_vs_full_joint"] = {
            "wins_this": int((a < c).sum()),
            "wins_full_joint": int((a > c).sum()),
            "mean_rel_diff_pct": float(rel.mean() * 100),
            "worst_case_pct": float(np.abs(rel).max() * 100),
        }
    return out


SCENARIOS = {
    "linear_mcar": (gen_linear, mask_mcar, ModelChoice.BayesianRidge, 1000),
    "linear_mar": (gen_linear, mask_mar, ModelChoice.BayesianRidge, 1000),
    "nonlinear_mcar": (gen_nonlinear, mask_mcar, ModelChoice.RandomForestRegressor, 600),
    "nonlinear_mar": (gen_nonlinear, mask_mar, ModelChoice.RandomForestRegressor, 600),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", default="prototypes/issue405_results.json")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    results = {}
    for sc in args.scenarios:
        gen, masker, choice, n_rows = SCENARIOS[sc]
        print(f"\n=== {sc} (estimator={choice}, n_rows={n_rows}, "
              f"{len(seeds)} seeds) ===", flush=True)
        rows = run_scenario(gen, masker, choice, n_rows, seeds)
        results[sc] = {"rows": rows, "summary": summarize(rows)}
        print(json.dumps(results[sc]["summary"], indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
