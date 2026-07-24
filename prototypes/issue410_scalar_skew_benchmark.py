"""PROTOTYPE — THROWAWAY. Measurement spike for issue #410.

Measures the accuracy cost of ADR-0070's still-open *scalar-half* train/serve
skew, now that the Regression->MICE collapse (#407/#408/#409) makes EVERY
MICE-routed column read the full active-numeric matrix as predictors — so the
skew that used to touch only ex-Regression columns now spreads to all of them.

The skew (ADR-0070, consequence 3):
  - scalar-owned predictor columns are median-filled by a SEPARATE scalar unit
    *before* the model loop;
  - the joint IterativeImputer is FIT on the raw train split, where those scalar
    columns still hold their NaNs (imputed by IterativeImputer's own internal
    round-robin during fit);
  - at SERVE (ADR-0070's shared pre-model snapshot) those same columns arrive
    already median-filled.
  => fit sees a *modeled* fill, serve sees a *static median*: train != serve.

#405 never modeled this — it used fit_transform on one array (fit == serve, no
scalar unit at all). This spike builds the genuine mismatch and asks, at
ADR-0070's evidentiary bar (30 seeds, RMSE/MAE on imputed cells only, wins/loss),
whether the delta is noise or large enough to force closing the skew before the
locking ADR can accept it.

Three shapes over ONE dataset (same rows; the skew is a fill-STATE mismatch, not
a held-out-rows question — matches ADR-0070 / #405 methodology, changing exactly
one variable between shapes):

  skewed  — production post-collapse: fit on raw frame (scalar cols NaN),
            serve on frame with scalar cols median-filled. train != serve.
  matched — the "close the skew from the fit side" option (ADR-0070's rejected
            "chained fit" analog for the scalar layer): scalar cols median-filled
            at BOTH fit and serve. train == serve.
  ideal   — reference / no scalar unit at all: fit_transform on the raw frame, so
            IterativeImputer models the scalar columns itself. No skew, no static
            median. Upper-bound grounding, not a shippable shape.

Headline comparison is skewed-vs-matched (does the skew cost accuracy?).
matched-vs-ideal is secondary grounding (does having a scalar unit at all cost?).

Two populations are scored SEPARATELY so the newly-exposed set is represented:
  ex_reg  — ex-Regression-shaped targets: depend strongly on scalar predictors
            (already carried this skew today).
  mice    — pre-existing-MICE-shaped targets: historically predicted by block
            siblings only; the collapse newly exposes them to scalar predictors.

Estimators come from the library's own RegressionEstimatorFactory. max_iter/tol
mirror _decision_assembler's MICE formulas (the collapsed block is MICE-routed).

Run:  venv/bin/python prototypes/issue410_scalar_skew_benchmark.py
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

# ----- column layout -------------------------------------------------------
# [ ex_reg targets | mice targets | scalar preds | complete preds ]
N_EXREG = 2      # ex-Regression-shaped MICE targets (lean hard on scalar preds)
N_MICE = 2       # pre-existing-MICE-shaped targets (lean on siblings + a little scalar)
N_SCALAR = 3     # scalar-owned predictors: skewed, carry NaNs, median-filled at serve
N_COMPLETE = 5   # fully-observed predictors (no skew contribution)

N_TARGET = N_EXREG + N_MICE
SCALAR_SLICE = slice(N_TARGET, N_TARGET + N_SCALAR)
EXREG_IDX = list(range(0, N_EXREG))
MICE_IDX = list(range(N_EXREG, N_TARGET))

TARGET_MISS = 0.20
SCALAR_MISS = 0.18


# ----- data generators -----------------------------------------------------
def _skewed_scalar(rng: np.random.Generator, signal: np.ndarray) -> np.ndarray:
    """Right-skewed scalar-owned column carrying `signal`; median != mean != model."""
    return np.exp(0.6 * signal + rng.normal(scale=0.5, size=signal.shape[0]))


def gen_linear(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    z = rng.normal(size=(n_rows, 3))
    complete = np.column_stack(
        [z @ rng.normal(size=3) + rng.normal(scale=0.6, size=n_rows) for _ in range(N_COMPLETE)]
    )
    scalar = np.column_stack(
        [_skewed_scalar(rng, z @ rng.normal(size=3)) for _ in range(N_SCALAR)]
    )
    # ex-reg targets: strong linear function of the scalar predictors
    exreg = np.column_stack(
        [
            scalar @ rng.normal(scale=0.8, size=N_SCALAR)
            + 0.3 * complete[:, k % N_COMPLETE]
            + rng.normal(scale=0.5, size=n_rows)
            for k in range(N_EXREG)
        ]
    )
    # mice targets: mostly siblings (ex-reg targets + complete) + a modest scalar term
    mice = np.column_stack(
        [
            exreg @ rng.normal(scale=0.7, size=N_EXREG)
            + complete @ rng.normal(scale=0.5, size=N_COMPLETE)
            + 0.25 * scalar[:, k % N_SCALAR]
            + rng.normal(scale=0.5, size=n_rows)
            for k in range(N_MICE)
        ]
    )
    return np.column_stack([exreg, mice, scalar, complete])


def gen_nonlinear(rng: np.random.Generator, n_rows: int) -> np.ndarray:
    complete = rng.normal(size=(n_rows, N_COMPLETE))
    scalar = np.column_stack(
        [_skewed_scalar(rng, complete[:, k % N_COMPLETE]) for k in range(N_SCALAR)]
    )
    noise = lambda s: rng.normal(scale=s, size=n_rows)  # noqa: E731
    exreg = np.column_stack(
        [
            scalar[:, 0] * scalar[:, 1 % N_SCALAR] + 0.5 * np.log1p(scalar[:, 2 % N_SCALAR]) + noise(0.4),
            np.sin(scalar[:, 0]) + 0.6 * scalar[:, 1 % N_SCALAR] + 0.3 * complete[:, 0] + noise(0.4),
        ][:N_EXREG]
    )
    mice = np.column_stack(
        [
            exreg[:, 0] * complete[:, 1] + 0.5 * complete[:, 2] + 0.25 * scalar[:, 0] + noise(0.4),
            np.tanh(exreg[:, 1 % N_EXREG]) + 0.5 * complete[:, 3] + 0.25 * np.log1p(scalar[:, 1 % N_SCALAR]) + noise(0.4),
        ][:N_MICE]
    )
    return np.column_stack([exreg, mice, scalar, complete])


# ----- missingness ---------------------------------------------------------
def apply_missing(rng: np.random.Generator, data: np.ndarray, mar: bool) -> np.ndarray:
    """Holes on target cols (MCAR or MAR) and MCAR holes on scalar-owned cols."""
    masked = data.copy()
    for j in range(N_TARGET):
        if mar:
            driver = data[:, N_TARGET + N_SCALAR + (j % N_COMPLETE)]  # a complete col
            zc = (driver - np.median(driver)) / (np.std(driver) + 1e-12)
            prob = 1.0 / (1.0 + np.exp(-(1.5 * zc)))
            prob *= TARGET_MISS / prob.mean()
            holes = rng.random(data.shape[0]) < np.clip(prob, 0, 0.9)
        else:
            holes = rng.random(data.shape[0]) < TARGET_MISS
        masked[holes, j] = np.nan
    for j in range(N_SCALAR):
        holes = rng.random(data.shape[0]) < SCALAR_MISS
        masked[holes, N_TARGET + j] = np.nan
    return masked


# ----- hyperparameter mirrors (MICE formulas, per _decision_assembler) ------
def _mice_max_iter(block_miss_frac: float, complete_row_frac: float) -> int:
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
    iqrs = []
    for j in range(cols.shape[1]):
        q75, q25 = np.nanpercentile(cols[:, j], [75, 25])
        if q75 - q25 > 0:
            iqrs.append(q75 - q25)
    return max(1e-7, min(iqrs) * 1e-4) if iqrs else 1e-3


def _imputer(masked: np.ndarray, choice: ModelChoice) -> IterativeImputer:
    crf = float(np.mean(~np.isnan(masked).any(axis=1)))
    miss_frac = float(np.isnan(masked[:, :N_TARGET]).mean())
    return IterativeImputer(
        estimator=RegressionEstimatorFactory.build_from_choice(choice, n_jobs=4),
        random_state=0,
        max_iter=_mice_max_iter(miss_frac, crf),
        tol=_tol(masked),
        initial_strategy="mean",
        n_nearest_features=None,
    )


# ----- the three shapes ----------------------------------------------------
def _median_fill_scalar(masked: np.ndarray) -> np.ndarray:
    """Return a copy with scalar-owned columns filled by their observed median."""
    out = masked.copy()
    block = out[:, SCALAR_SLICE]
    med = np.nanmedian(block, axis=0)
    idx = np.isnan(block)
    block[idx] = np.take(med, np.where(idx)[1])
    out[:, SCALAR_SLICE] = block
    return out


def shape_skewed(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """fit on raw (scalar NaN), serve on scalar-median-filled: train != serve."""
    imp = _imputer(masked, choice)
    imp.fit(masked)
    serve = _median_fill_scalar(masked)
    return imp.transform(serve)


def shape_matched(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """scalar median-filled at BOTH fit and serve: train == serve (skew closed)."""
    base = _median_fill_scalar(masked)
    imp = _imputer(base, choice)
    return imp.fit_transform(base)


def shape_ideal(masked: np.ndarray, choice: ModelChoice) -> np.ndarray:
    """no scalar unit: IterativeImputer models the scalar cols itself (reference)."""
    imp = _imputer(masked, choice)
    return imp.fit_transform(masked)


SHAPES = {"skewed": shape_skewed, "matched": shape_matched, "ideal": shape_ideal}


# ----- scoring (imputed target cells only, split by population) -------------
def score(truth: np.ndarray, masked: np.ndarray, filled: np.ndarray):
    out = {}
    for pop, idx in (("ex_reg", EXREG_IDX), ("mice", MICE_IDX), ("all", list(range(N_TARGET)))):
        holes = np.isnan(masked[:, idx])
        err = filled[:, idx][holes] - truth[:, idx][holes]
        out[pop] = {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
        }
    return out


def run_scenario(gen, mar, choice, n_rows, seeds):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        truth = gen(rng, n_rows)
        masked = apply_missing(rng, truth, mar)
        rec = {"seed": seed}
        for name, fn in SHAPES.items():
            t0 = time.perf_counter()
            filled = fn(masked, choice)
            rec[name] = {"score": score(truth, masked, filled), "time_s": time.perf_counter() - t0}
        rows.append(rec)
        print(
            f"  seed {seed:>3}: "
            + " | ".join(f"{n} all_rmse={rec[n]['score']['all']['rmse']:.4f}" for n in SHAPES),
            flush=True,
        )
    return rows


def _pairwise(rows, a_name, b_name, pop):
    a = np.array([r[a_name]["score"][pop]["rmse"] for r in rows])
    b = np.array([r[b_name]["score"][pop]["rmse"] for r in rows])
    rel = (a - b) / b
    return {
        "wins_" + a_name: int((a < b).sum()),
        "wins_" + b_name: int((a > b).sum()),
        "ties": int((a == b).sum()),
        "mean_rel_diff_pct": float(rel.mean() * 100),
        "worst_case_pct": float(np.abs(rel).max() * 100),
    }


def summarize(rows):
    out = {"means": {}, "pairwise": {}}
    for name in SHAPES:
        out["means"][name] = {}
        for pop in ("ex_reg", "mice", "all"):
            rmses = np.array([r[name]["score"][pop]["rmse"] for r in rows])
            maes = np.array([r[name]["score"][pop]["mae"] for r in rows])
            out["means"][name][pop] = {
                "rmse_mean": float(rmses.mean()),
                "rmse_std": float(rmses.std()),
                "mae_mean": float(maes.mean()),
            }
        out["means"][name]["time_mean"] = float(np.mean([r[name]["time_s"] for r in rows]))
    for pop in ("ex_reg", "mice", "all"):
        out["pairwise"][f"skewed_vs_matched::{pop}"] = _pairwise(rows, "skewed", "matched", pop)
        out["pairwise"][f"matched_vs_ideal::{pop}"] = _pairwise(rows, "matched", "ideal", pop)
    return out


SCENARIOS = {
    "linear_mcar": (gen_linear, False, ModelChoice.BayesianRidge, 1000),
    "linear_mar": (gen_linear, True, ModelChoice.BayesianRidge, 1000),
    "nonlinear_mcar": (gen_nonlinear, False, ModelChoice.RandomForestRegressor, 600),
    "nonlinear_mar": (gen_nonlinear, True, ModelChoice.RandomForestRegressor, 600),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--out", default="prototypes/issue410_results.json")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    results = {}
    for sc in args.scenarios:
        gen, mar, choice, n_rows = SCENARIOS[sc]
        print(f"\n=== {sc} (estimator={choice}, n_rows={n_rows}, {len(seeds)} seeds) ===", flush=True)
        rows = run_scenario(gen, mar, choice, n_rows, seeds)
        results[sc] = {"rows": rows, "summary": summarize(rows)}
        print(json.dumps(results[sc]["summary"], indent=2, default=str))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
