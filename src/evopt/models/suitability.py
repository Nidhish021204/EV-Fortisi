"""Stage 2 - siting suitability by Positive-Unlabeled learning.

A cell without a charger is not a bad site; it is an unbuilt one. Treating
unlabeled cells as negatives biases the classifier toward the existing network
and destroys probability calibration, so this uses PU learning:

  * Bagged PU (Mordelet & Vert) -- repeatedly subsample the unlabeled pool as
    pseudo-negatives, average out-of-bag scores over many such fits. Robust,
    and it yields an out-of-bag score for every cell for free.
  * Elkan & Noto -- estimate c = P(labeled | positive) on held-out positives
    and rescale, so the output is a calibrated P(viable) rather than a
    presence score distorted by the labeling rate.

Headline evaluation is a TEMPORAL FORWARD TEST, possible only because AFDC
`Open Date` is 99.86% populated: fit on the network as it stood at 2018-12-31,
then ask whether the model ranked the cells that actually received stations in
2019-2021. Cells that already held a charger at the cutoff are excluded from
scoring -- predicting those is trivial and would flatter the model.

PR-AUC is the headline, not ROC-AUC: at a ~3% positive rate ROC-AUC is
dominated by the vast negative mass and looks impressive while saying little.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import PROCESSED, MODELS, REPORTS

FEATURES = [
    # demand / mobility
    "demand_kwh", "demand_taxi_kwh", "demand_fleet_kwh", "unmet_kwh",
    "pickups", "dropoffs", "dwell_surplus", "flow_imbalance",
    "mean_trip_miles", "pickups_peak",
    # traffic / corridor
    "aadt_proxy", "mean_hourly_vol", "peak_hourly_vol", "night_hourly_vol",
    "p95_hourly_vol", "peak_ratio", "n_segments", "dist_to_road_m",
    # incumbent supply as of the cutoff -- leave-one-cell-out variants only,
    # so the model cannot read the label off its own footprint
    "access_l2_ex", "access_dcfc_ex", "dist_station_ex_m", "dist_dcfc_ex_m",
    # street morphology
    "n_intersections", "n_streets", "arterial_share", "local_share",
    "dir_diversity",
    # built environment -- siting feasibility, not just demand. `garage_area`
    # and `lu_parking_sh` say whether there is physically anywhere to put the
    # hardware; `assess_per_sqft` proxies what the land costs; open space and
    # vacant share mark cells that are either unusable or unusually cheap.
    "garage_area", "parking_share", "lu_parking_sh", "lu_commercial_sh",
    "lu_institutional_sh", "lu_openspace_sh", "lu_vacant_sh", "lu_mixed_sh",
    "lu_res_high_sh", "lu_industrial_sh", "lu_transport_sh",
    "units_res", "jobs_area", "far_built", "floors_mean", "lu_entropy",
    "assess_per_sqft", "res_density", "year_built_med", "lots_n",
    # ACS households and vehicles
    "veh_per_hh", "zero_veh_share", "renter_share", "hh_total", "veh_total",
    "renter_veh_hh", "home_charge_cov", "no_home_charge",
    "no_home_charge_density",
    # spatial lag
    "k1_kwh_day", "k2_kwh_day", "k1_dropoffs", "k2_dropoffs",
    "k1_aadt_proxy", "k2_aadt_proxy", "k1_dwell_surplus", "k2_dwell_surplus",
    "k1_ports", "k2_ports", "k1_n_segments", "k2_n_segments",
    "k1_units_res", "k2_units_res", "k1_jobs_area", "k2_jobs_area",
    "k1_garage_area", "k2_garage_area", "k1_far_built", "k2_far_built",
    "k1_lu_entropy", "k2_lu_entropy",
    "k1_no_home_charge", "k2_no_home_charge", "k1_veh_per_hh", "k2_veh_per_hh",
]

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=2.0, verbose=-1, n_estimators=300)


def bagged_pu(X, y, n_bags: int = 60, seed: int = 42):
    """Average out-of-bag scores over bags of positives + sampled unlabeled."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    unl = np.flatnonzero(y == 0)
    scores = np.zeros(len(y))
    counts = np.zeros(len(y))
    models = []
    for _ in range(n_bags):
        samp = rng.choice(unl, size=min(len(pos), len(unl)), replace=False)
        idx = np.concatenate([pos, samp])
        yy = np.concatenate([np.ones(len(pos)), np.zeros(len(samp))])
        m = lgb.LGBMClassifier(**PARAMS).fit(X[idx], yy)
        oob = np.setdiff1d(unl, samp)
        scores[oob] += m.predict_proba(X[oob])[:, 1]
        counts[oob] += 1
        models.append(m)
    counts[counts == 0] = 1
    oob_scores = scores / counts
    full = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    oob_scores[y == 1] = full[y == 1]
    return oob_scores, models, full


def elkan_noto_c(models, X, y, holdout_frac: float = 0.3, seed: int = 42) -> float:
    """c = P(labeled | positive), estimated on held-out positives."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    hold = rng.choice(pos, size=max(1, int(len(pos) * holdout_frac)), replace=False)
    p = np.mean([m.predict_proba(X[hold])[:, 1] for m in models], axis=0)
    return float(np.clip(p.mean(), 1e-3, 1.0))


def _rank_metrics(y_true, score, ks=(50, 100, 200, 500)) -> dict:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(score))
    yt = y_true[order]
    out = {"pr_auc": float(average_precision_score(y_true, score)),
           "roc_auc": float(roc_auc_score(y_true, score)),
           "base_rate": float(y_true.mean())}
    for k in ks:
        kk = min(k, len(yt))
        out["precision@%d" % k] = float(yt[:kk].mean())
        out["recall@%d" % k] = float(yt[:kk].sum() / max(1, y_true.sum()))
        out["lift@%d" % k] = float(yt[:kk].mean() / max(1e-9, y_true.mean()))
    disc = np.log2(np.arange(2, len(yt) + 2))
    ideal = np.sort(y_true)[::-1] / disc
    out["ndcg"] = float((yt / disc).sum() / ideal.sum()) if ideal.sum() > 0 else 0.0
    return out


def run() -> dict:
    df = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    X = df[FEATURES].to_numpy(float)
    y = df.y_now.to_numpy(int)

    oob, models, full = bagged_pu(X, y)
    c = elkan_noto_c(models, X, y)
    df["p_viable"] = np.clip(full / c, 0, 1)
    df["pu_oob"] = oob

    # Forward test: unbuilt at the cutoff -- did it get built in 2019-2021?
    mask = (df.y_now == 0).to_numpy()
    cand = df.loc[mask]
    y_fwd = cand.y_future.to_numpy(int)

    results = {"label_frequency_c": c,
               "n_train_positives": int(y.sum()),
               "n_forward_candidates": int(mask.sum()),
               "n_forward_positives": int(y_fwd.sum())}

    results["model_pu"] = _rank_metrics(y_fwd, cand.pu_oob.to_numpy())

    # Baselines that a serious accuracy claim has to beat.
    rng = np.random.default_rng(0)
    results["baseline_random"] = _rank_metrics(y_fwd, rng.random(len(cand)))
    results["baseline_demand"] = _rank_metrics(y_fwd, cand.demand_kwh.to_numpy())
    results["baseline_traffic"] = _rank_metrics(y_fwd, cand.aadt_proxy.to_numpy())
    results["baseline_near_existing"] = _rank_metrics(
        y_fwd, -cand.dist_station_ex_m.to_numpy())

    df[["h3", "p_viable", "pu_oob"]].to_parquet(
        PROCESSED / "suitability.parquet", index=False)
    models[0].booster_.save_model(str(MODELS / "suitability_lgbm.txt"))

    imp = np.mean([m.feature_importances_ for m in models], axis=0)
    results["feature_importance"] = sorted(
        zip(FEATURES, imp.round(1).tolist()), key=lambda t: -t[1])[:15]
    (REPORTS / "suitability_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    r = run()
    print("Stage 2 - PU suitability, temporal forward test")
    print("  train positives (<=2018) : %d" % r["n_train_positives"])
    print("  candidates (unbuilt)     : %d" % r["n_forward_candidates"])
    print("  actually built 2019-21   : %d  (base rate %.2f%%)"
          % (r["n_forward_positives"], 100 * r["model_pu"]["base_rate"]))
    print("  Elkan-Noto c             : %.3f\n" % r["label_frequency_c"])
    print("  %-22s%8s%8s%8s%10s%8s"
          % ("method", "PR-AUC", "P@100", "R@100", "lift@100", "NDCG"))
    print("  " + "-" * 62)
    for key, nm in (("model_pu", "PU model"),
                    ("baseline_demand", "demand ranking"),
                    ("baseline_traffic", "traffic ranking"),
                    ("baseline_near_existing", "near existing"),
                    ("baseline_random", "random")):
        m = r[key]
        print("  %-22s%8.3f%7.1f%%%7.1f%%%9.2fx%8.3f"
              % (nm, m["pr_auc"], 100 * m["precision@100"],
                 100 * m["recall@100"], m["lift@100"], m["ndcg"]))
    print("\n  top features: " + ", ".join(k for k, _ in r["feature_importance"][:6]))
