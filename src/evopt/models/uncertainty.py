"""Stage 7g - uncertainty: plan stability and honest error bars.

Two different questions, needing two different methods. Conflating them would
answer neither.

(a) Which sites are robust?
    The plan names 120 corners. A planner has no way to tell the ones the model
    insists on from the ones that are a coin flip. `bagged_pu` is an ensemble of
    60 bags, so resampling those bags with replacement gives many equally valid
    viability surfaces; each yields its own greedy plan, and a site's
    **selection frequency** across replicates is the stability score. Per-bag
    predictions are cached once, so a replicate costs only a greedy run.

(b) How precise are the headline metrics really?
    PR-AUC moves between identical pipeline runs because LightGBM's threaded
    histogram build is not bit-deterministic. A bag bootstrap cannot see that:
    it conditions on one fitted ensemble. Only refitting under different seeds
    measures it, so a seed sweep runs alongside and reports mean, sd and a 95%
    interval -- and, more usefully, **the number of decimal places the estimate
    actually supports**.

The bootstrap is checked for bias rather than assumed unbiased: the mean
bootstrap surface is compared against the point estimate in `run()`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED, REPORTS
from .suitability import FEATURES, bagged_pu, _rank_metrics
from .allocate import greedy_mclp, load_grid

N_BOOTSTRAP = 200
N_SEEDS = 15
BASE_SEED = 42
K_PLAN = 120
POOL = 1500
FLOORS = {"Bronx": 18, "Staten Island": 12, "Brooklyn": 22, "Queens": 22}


def _candidates(df: pd.DataFrame) -> pd.DataFrame:
    """The same candidate pool allocate.run() builds, so plans are comparable."""
    grid = load_grid()
    if len(grid):
        df = df.merge(grid, on="h3", how="left")
    cand = df[(df.unmet_kwh > 0) & (df.dist_to_road_m < 800)].copy()
    return cand.nlargest(POOL, "unmet_kwh").reset_index(drop=True)


def bag_matrix(X: np.ndarray, y: np.ndarray, models) -> np.ndarray:
    """(n_bags, n_cells) of per-bag probabilities, computed once."""
    return np.vstack([m.predict_proba(X)[:, 1] for m in models])


def stability(df: pd.DataFrame, P: np.ndarray, c: float,
              n_boot: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Selection frequency per cell over bootstrapped bag ensembles."""
    rng = np.random.default_rng(seed)
    n_bags = P.shape[0]
    counts: dict[str, int] = {}
    gaps: list[float] = []
    ranks: dict[str, list[int]] = {}
    surf_sum = np.zeros(P.shape[1])

    for _ in range(n_boot):
        idx = rng.integers(0, n_bags, n_bags)          # resample bags
        p = np.clip(P[idx].mean(axis=0) / c, 0, 1)
        surf_sum += p
        d = df.copy()
        d["p_viable"] = p
        cand = _candidates(d)
        plan = greedy_mclp(cand, k=K_PLAN, min_per_borough=FLOORS)
        gaps.append(float(sum(s["captured_kwh_day"] for s in plan)))
        for s in plan:
            counts[s["h3"]] = counts.get(s["h3"], 0) + 1
            ranks.setdefault(s["h3"], []).append(s["rank"])

    freq = {h: v / n_boot for h, v in counts.items()}
    return {"freq": freq,
            "rank_median": {h: float(np.median(v)) for h, v in ranks.items()},
            "captured": gaps,
            "mean_surface": surf_sum / n_boot}


def seed_sweep(X: np.ndarray, y: np.ndarray, y_fwd_mask: np.ndarray,
               y_fwd: np.ndarray, n_seeds: int = N_SEEDS) -> dict:
    """Refit the PU ensemble under different seeds; record the spread."""
    rows = []
    for s in range(BASE_SEED, BASE_SEED + n_seeds):
        oob, models, full = bagged_pu(X, y, n_bags=60, seed=s)
        pos = np.flatnonzero(y == 1)
        rng = np.random.default_rng(s)
        hold = rng.choice(pos, size=max(1, int(len(pos) * 0.3)), replace=False)
        c = float(np.clip(np.mean([m.predict_proba(X[hold])[:, 1]
                                   for m in models], axis=0).mean(), 1e-3, 1.0))
        m = _rank_metrics(y_fwd, oob[y_fwd_mask])
        rows.append({"seed": s, "pr_auc": m["pr_auc"],
                     "precision@100": m["precision@100"],
                     "lift@100": m["lift@100"], "ndcg": m["ndcg"],
                     "label_frequency_c": c})
    return {"runs": rows}


def _interval(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    lo, hi = np.percentile(a, [2.5, 97.5])
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    # Decimal places the spread actually supports: report to roughly the scale
    # of the standard deviation, never finer.
    dp = 0 if sd <= 0 else max(0, int(np.ceil(-np.log10(sd))))
    return {"mean": float(a.mean()), "sd": sd, "lo95": float(lo),
            "hi95": float(hi), "n": int(len(a)),
            "supported_decimals": int(min(dp, 4))}


def run() -> dict:
    df = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    stored = pd.read_parquet(PROCESSED / "suitability.parquet")
    df = df.merge(stored, on="h3", how="left")
    X = df[FEATURES].to_numpy(float)
    y = df.y_now.to_numpy(int)

    cmeta = REPORTS / "suitability_metrics.json"
    c_pub = (json.loads(cmeta.read_text())["label_frequency_c"]
             if cmeta.exists() else 1.0)

    _, models, full = bagged_pu(X, y, n_bags=60, seed=BASE_SEED)
    P = bag_matrix(X, y, models)

    point = np.clip(full / c_pub, 0, 1)
    st = stability(df, P, c_pub)

    # Bias guard. Comparing P.mean() against `full` would be tautological --
    # `full` *is* that mean, so it can only ever return zero and tests nothing.
    # The real question is whether averaging over *resampled* ensembles lands
    # back on the point estimate, so the mean across replicates is what gets
    # compared. A non-trivial residual here would mean the resampling is
    # skewing the surface it is supposed to characterise.
    bias = float(np.abs(st["mean_surface"] - point).max())

    # Forward-test slice, matching suitability.run()
    mask = (df.y_now == 0).to_numpy()
    y_fwd = df.loc[mask, "y_future"].to_numpy(int)
    sw = seed_sweep(X, y, mask, y_fwd)

    metrics = {k: _interval([r[k] for r in sw["runs"]])
               for k in ("pr_auc", "precision@100", "lift@100", "ndcg",
                         "label_frequency_c")}

    plan = pd.read_parquet(PROCESSED / "plan_sites.parquet")
    plan["stability"] = plan.h3.map(st["freq"]).fillna(0.0)
    plan["rank_median_boot"] = plan.h3.map(st["rank_median"])
    plan.to_parquet(PROCESSED / "plan_sites.parquet", index=False)

    cap = _interval(st["captured"])
    always = int((plan.stability >= 0.99).sum())
    coin = int((plan.stability < 0.5).sum())

    res = {
        "n_bootstrap": N_BOOTSTRAP, "n_seeds": N_SEEDS,
        "bootstrap_bias_max_abs": round(bias, 6),
        "captured_kwh_day": cap,
        "metric_intervals": metrics,
        "seed_runs": sw["runs"],
        "plan_stability": {
            "mean": round(float(plan.stability.mean()), 4),
            "median": round(float(plan.stability.median()), 4),
            "sites_always_selected": always,
            "sites_below_half": coin,
            "deciles": {str(round(q, 1)): round(float(plan.stability.quantile(q)), 3)
                        for q in np.arange(0.1, 1.0, 0.2)},
        },
        "site_frequency": {h: round(v, 4) for h, v in st["freq"].items()},
    }
    (REPORTS / "uncertainty.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print("Stage 7g - uncertainty")
    print("  bootstrap : %d bag resamples · replicate-mean vs point estimate, "
          "max |dp| %.2e" % (r["n_bootstrap"], r["bootstrap_bias_max_abs"]))
    print("  seeds     : %d refits\n" % r["n_seeds"])

    print("  METRIC INTERVALS (seed sweep)")
    print("    %-18s %10s %10s %22s %s"
          % ("metric", "mean", "sd", "95% interval", "supported dp"))
    for k, v in r["metric_intervals"].items():
        print("    %-18s %10.4f %10.4f  [%8.4f, %8.4f] %6d"
              % (k, v["mean"], v["sd"], v["lo95"], v["hi95"],
                 v["supported_decimals"]))

    c = r["captured_kwh_day"]
    print("\n  PLAN OUTCOME")
    print("    captured  : %s kWh/day  95%% [%s, %s]"
          % (f'{c["mean"]:,.0f}', f'{c["lo95"]:,.0f}', f'{c["hi95"]:,.0f}'))

    p = r["plan_stability"]
    print("\n  SITE STABILITY (share of %d replicates selecting the site)"
          % r["n_bootstrap"])
    print("    mean %.2f · median %.2f" % (p["mean"], p["median"]))
    print("    always selected (>=0.99) : %d of 120" % p["sites_always_selected"])
    print("    coin-flip     (<0.50)    : %d of 120" % p["sites_below_half"])
    print("    deciles: %s" % p["deciles"])
