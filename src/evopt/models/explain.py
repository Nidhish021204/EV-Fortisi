"""Stage 7c - per-site attribution for the siting model (exact TreeSHAP).

Stage 7's exit criterion is that a planner can answer "why this corner" for
every entry on the list. A ranked score cannot do that; an attribution can.

No `shap` dependency
--------------------
LightGBM's `predict(..., pred_contrib=True)` *is* TreeSHAP, computed exactly by
the same polynomial-time algorithm, and it ships with the booster already in
this project. Adding the `shap` package would add a dependency to get the same
numbers more slowly.

Two things this has to get right
--------------------------------
1. **Space.** For a binary objective the contributions are in raw margin
   (log-odds) space, not probability. They sum to the margin, and
   sigmoid(sum) recovers the model's probability. Reporting them as
   "probability points" would be wrong, so everything below is labelled
   log-odds.

2. **The ensemble is nonlinear in its parts.** The PU score is
   mean_k sigmoid(margin_k) over 60 bags, but averaging contributions gives
   mean_k margin_k -- and sigmoid(mean margin) != mean sigmoid(margin) because
   sigmoid is not affine. Averaged attributions therefore explain the *mean
   log-odds*, which is a slightly different quantity from the score that was
   actually ranked. Rather than wave that away, `faithfulness()` measures the
   discrepancy and it is reported alongside the attributions.

The ensemble is rebuilt with the same seed as `suitability.run()`, so it is the
identical set of 60 boosters that produced the published scores; `run()`
verifies that reproduction against the stored `p_viable` instead of assuming it.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROCESSED, REPORTS, DASH_DATA
from .suitability import FEATURES, bagged_pu
from .allocate import greedy_mclp, load_grid

N_BAGS = 60
SEED = 42
TOP_N = 6  # drivers surfaced per site in the inspector

# Feature labels for the dashboard. A planner reading "k2_jobs_area" in a
# public hearing is not being given an explanation.
LABELS: dict[str, str] = {
    "demand_kwh": "Total demand", "demand_taxi_kwh": "Measured taxi demand",
    "demand_fleet_kwh": "Fleet demand", "unmet_kwh": "Unmet demand",
    # The model is scored on a 2019-21 forward window, so it reads the
    # historical surface rather than the Q1 2026 re-weighted one (Stage 8c).
    "demand_kwh_hist": "Total demand (2015-16)",
    "demand_taxi_kwh_hist": "Measured taxi demand (2015-16)",
    "unmet_kwh_hist": "Unmet demand (2015-16)",
    "pickups": "Taxi pickups", "dropoffs": "Taxi dropoffs",
    "dwell_surplus": "Dwell surplus", "flow_imbalance": "Arrival/departure imbalance",
    "mean_trip_miles": "Mean trip length", "pickups_peak": "Peak-hour pickups",
    "aadt_proxy": "Traffic volume", "mean_hourly_vol": "Mean hourly traffic",
    "peak_hourly_vol": "Peak-hour traffic", "night_hourly_vol": "Overnight traffic",
    "p95_hourly_vol": "95th-pct traffic", "peak_ratio": "Peak-to-mean ratio",
    "n_segments": "Road segments", "dist_to_road_m": "Distance to road",
    "access_l2_ex": "Nearby L2 supply", "access_dcfc_ex": "Nearby DC fast supply",
    "dist_station_ex_m": "Distance to nearest station",
    "dist_dcfc_ex_m": "Distance to nearest DC fast",
    "n_intersections": "Intersection density", "n_streets": "Distinct streets",
    "arterial_share": "Arterial road share", "local_share": "Local road share",
    "dir_diversity": "Directional diversity",
    "garage_area": "Parking floor area", "parking_share": "Parking share of floor area",
    "lu_parking_sh": "Land use: parking", "lu_commercial_sh": "Land use: commercial",
    "lu_institutional_sh": "Land use: institutional",
    "lu_openspace_sh": "Land use: open space", "lu_vacant_sh": "Land use: vacant",
    "lu_mixed_sh": "Land use: mixed", "lu_res_high_sh": "Land use: high-density res",
    "lu_industrial_sh": "Land use: industrial", "lu_transport_sh": "Land use: transport",
    "units_res": "Residential units", "jobs_area": "Office/retail floor area",
    "far_built": "Built floor-area ratio", "floors_mean": "Mean building height",
    "lu_entropy": "Land-use mix", "assess_per_sqft": "Assessed value density",
    "res_density": "Residential density", "year_built_med": "Median build year",
    "lots_n": "Tax lots",
    "veh_per_hh": "Vehicles per household",
    "zero_veh_share": "Zero-vehicle household share",
    "renter_share": "Renter share", "hh_total": "Households",
    "veh_total": "Vehicles", "renter_veh_hh": "Renter households with a car",
    "owner_hh": "Owner households", "renter_hh": "Renter households",
    "zero_veh_hh": "Zero-vehicle households",
    "home_charge_cov": "Home-charging coverage",
    "no_home_charge": "Cars with no home charging",
    "no_home_charge_density": "No-home-charging rate",
}


def label(f: str) -> str:
    """Human label, preserving the k-ring qualifier that carries real meaning."""
    if f in LABELS:
        return LABELS[f]
    for k, suffix in (("k1_", " (1-ring)"), ("k2_", " (2-ring)")):
        if f.startswith(k):
            base = f[len(k):]
            return LABELS.get(base, base.replace("_", " ").title()) + suffix
    return f.replace("_", " ").title()


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def contributions(X: np.ndarray, models) -> tuple[np.ndarray, np.ndarray]:
    """Mean TreeSHAP contributions over the bag ensemble, in log-odds.

    Accumulated as a running sum: holding all 60 bags at once would be a
    needless (n x 70 x 60) allocation.
    """
    total = np.zeros((X.shape[0], len(FEATURES) + 1), float)
    for m in models:
        total += m.booster_.predict(X, pred_contrib=True)
    total /= len(models)
    return total[:, :-1], total[:, -1]      # contributions, base (expected) value


def faithfulness(phi: np.ndarray, base: np.ndarray,
                 ensemble_prob: np.ndarray) -> dict:
    """How far sigmoid(mean margin) sits from the ranked mean-probability score.

    Both are legitimate summaries of the ensemble; they are simply not the same
    number. What matters for an explanation is whether they order sites the
    same way, so rank correlation is the headline and the level error is
    reported next to it.
    """
    approx = _sigmoid(phi.sum(axis=1) + base)
    return {
        "spearman_vs_ensemble": float(pd.Series(approx).corr(
            pd.Series(ensemble_prob), method="spearman")),
        "pearson_vs_ensemble": float(np.corrcoef(approx, ensemble_prob)[0, 1]),
        "mean_abs_prob_error": float(np.abs(approx - ensemble_prob).mean()),
        "max_abs_prob_error": float(np.abs(approx - ensemble_prob).max()),
    }


def equity_forced_sites() -> set[str]:
    """Cells the equity floor put in the plan that merit alone would not.

    Exact rather than inferred: greedy is deterministic, so re-running it with
    `min_per_borough={}` gives the plan the optimiser would have chosen on
    demand and viability only. Anything in the constrained plan but absent from
    the free one is there because of the floor.
    """
    df = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    su = pd.read_parquet(PROCESSED / "suitability.parquet")
    df = df.merge(su, on="h3", how="left")
    df["p_viable"] = df.p_viable.fillna(0.0)

    # The counterfactual has to differ from the real plan in exactly one way:
    # the equity floor. Without merging the grid, the "free" run would also be
    # unconstrained by Con Edison capacity, and every site the grid displaced
    # would be miscounted as equity-forced.
    grid = load_grid()
    if len(grid):
        df = df.merge(grid, on="h3", how="left")
    cand = df[(df.unmet_kwh > 0) & (df.dist_to_road_m < 800)].copy()
    if "charge_excluded" in cand:
        cand = cand[~cand.charge_excluded.fillna(False)]
    cand = cand.nlargest(1500, "unmet_kwh").reset_index(drop=True)

    constrained = pd.read_parquet(PROCESSED / "plan_sites.parquet")
    free = greedy_mclp(cand, k=len(constrained), min_per_borough=None)
    return set(constrained.h3) - {s["h3"] for s in free}


def run() -> dict:
    df = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    stored = pd.read_parquet(PROCESSED / "suitability.parquet")
    df = df.merge(stored, on="h3", how="left")
    X = df[FEATURES].to_numpy(float)
    y = df.y_now.to_numpy(int)

    # Same seed and bag count as suitability.run(), so this is the identical
    # ensemble rather than a lookalike. Verified below, not assumed.
    _, models, full = bagged_pu(X, y, n_bags=N_BAGS, seed=SEED)
    repro = float(np.abs(np.clip(full / 0.9708, 0, 1) - df.p_viable).max()) \
        if "p_viable" in df else float("nan")

    phi, base = contributions(X, models)
    faith = faithfulness(phi, base, full)

    # ── global attribution: mean |contribution| ─────────────────────
    mean_abs = np.abs(phi).mean(axis=0)
    order = np.argsort(-mean_abs)
    global_imp = [{"feature": FEATURES[i], "label": label(FEATURES[i]),
                   "mean_abs_shap": round(float(mean_abs[i]), 5),
                   "mean_signed_shap": round(float(phi[:, i].mean()), 5)}
                  for i in order[:25]]

    # ── why a site was SELECTED, which is not why it scored ─────────
    # SHAP explains the viability model. Selection is a different question:
    # the optimiser maximises unmet_kwh x p_viable subject to a per-borough
    # equity floor, so a low-viability cell can be in the plan either because
    # it captures a lot of unmet demand or because the floor required it.
    # Attributing that per site needs a counterfactual, not a heuristic, so
    # greedy is re-run with the floor removed and the two plans diffed.
    forced = equity_forced_sites()

    # ── per-site drivers for the recommended plan ───────────────────
    plan = pd.read_parquet(PROCESSED / "plan_sites.parquet")
    pos = {h: i for i, h in enumerate(df.h3.values)}
    sites = []
    for _, row in plan.iterrows():
        i = pos.get(row.h3)
        if i is None:
            continue
        v = phi[i]
        top = np.argsort(-np.abs(v))[:TOP_N]
        sites.append({
            "rank": int(row["rank"]), "h3": row.h3,
            "margin": round(float(v.sum() + base[i]), 4),
            "base": round(float(base[i]), 4),
            "p_viable": round(float(df.p_viable.iloc[i]), 4),
            "equity_forced": bool(row.h3 in forced),
            "drivers": [{"feature": FEATURES[j], "label": label(FEATURES[j]),
                         "shap": round(float(v[j]), 4),
                         "value": round(float(X[i, j]), 3)} for j in top],
        })

    res = {"n_bags": N_BAGS, "seed": SEED, "space": "log-odds (raw margin)",
           "equity_forced_sites": int(sum(1 for x in sites if x["equity_forced"])),
           "ensemble_reproduction_max_abs_err": round(repro, 6),
           "faithfulness": faith,
           "global_importance": global_imp,
           "n_sites_explained": len(sites)}
    (REPORTS / "shap_global.json").write_text(json.dumps(res, indent=2))

    DASH_DATA.mkdir(parents=True, exist_ok=True)
    (DASH_DATA / "explain.json").write_text(json.dumps(
        {"global": global_imp[:14], "faithfulness": faith,
         "space": "log-odds",
         "equity_forced_sites": res["equity_forced_sites"],
         "sites": sites}, separators=(",", ":")))
    return res


if __name__ == "__main__":
    r = run()
    print("Stage 7c - per-site attribution (exact TreeSHAP via LightGBM)")
    print("  ensemble : %d bags, seed %d, contributions in %s"
          % (r["n_bags"], r["seed"], r["space"]))
    print("  reproduced published scores to max abs err %.2e"
          % r["ensemble_reproduction_max_abs_err"])
    f = r["faithfulness"]
    print("  faithfulness of sigmoid(mean margin) vs the ranked score:")
    print("    spearman %.4f   pearson %.4f   mean |dp| %.4f   max |dp| %.4f"
          % (f["spearman_vs_ensemble"], f["pearson_vs_ensemble"],
             f["mean_abs_prob_error"], f["max_abs_prob_error"]))
    print("  sites explained : %d  (of which %d exist because of the equity "
          "floor)" % (r["n_sites_explained"], r["equity_forced_sites"]))
    print("\n  global drivers (mean |SHAP|, log-odds):")
    for g in r["global_importance"][:12]:
        arrow = "+" if g["mean_signed_shap"] >= 0 else "-"
        print("    %-30s %.4f  (mean %s%.4f)"
              % (g["label"], g["mean_abs_shap"], arrow,
                 abs(g["mean_signed_shap"])))
