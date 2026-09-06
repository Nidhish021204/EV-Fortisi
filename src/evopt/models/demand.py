"""Stage 1 - demand estimation with a Tweedie objective and spatial block CV.

Leakage guard (important)
-------------------------
`kwh_day` is *derived* from `vmt_terminating`, so training on taxi-derived
columns would be circular and would report a meaningless R^2. The demand model
is therefore restricted to traffic-infrastructure, street-morphology and
built-environment features. That makes it a genuine task -- "estimate charging
demand where no taxi feed exists" -- and it is exactly the form needed to port
the model to another city. No column sourced from the taxi records may appear
in FEATURES, including their k-ring lags.

Tweedie rather than L2-on-log1p: demand is non-negative, zero-inflated and
heavy-tailed. Tweedie (1 < p < 2) models that mixture natively; log1p+L2 biases
the conditional mean and cannot represent the point mass at zero.

Spatial block CV rather than random k-fold: adjacent hexes share features, so
random folds leak and inflate scores. Both are reported so the gap is visible.
"""
from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score

from ..config import PROCESSED, MODELS, REPORTS

# Traffic + form only. No taxi-derived column may appear here.
# Raw lat/lon are deliberately EXCLUDED. Under spatial block CV they act as
# memorisation handles: the tree partitions space, then cannot extrapolate to a
# held-out block. Including them inflates random-fold scores and destroys block
# scores -- the opposite of what a transferable siting model needs.
# Traffic volume and corridor character.
_TRAFFIC = ["aadt_proxy", "mean_hourly_vol", "peak_hourly_vol",
            "night_hourly_vol", "p95_hourly_vol", "peak_ratio", "n_segments"]

# Street-network morphology: hierarchy and connectivity, which volume alone
# does not express.
_MORPH = ["n_intersections", "n_streets", "arterial_share", "local_share",
          "dir_diversity", "dist_to_road_m"]

# Built environment from PLUTO tax lots -- what the floor area is used FOR,
# which is what generates and absorbs trips.
_LANDUSE = [
    "lots_n", "lot_area", "bldg_area", "res_area", "com_area", "office_area",
    "retail_area", "garage_area", "factory_area", "storage_area",
    "units_res", "units_total", "floors_mean", "far_mean", "year_built_med",
    "assess_total", "lu_entropy", "far_built", "jobs_area", "parking_share",
    "assess_per_sqft", "res_density",
] + [f"lu_{k}_sh" for k in
     ["res_low", "res_mid", "res_high", "mixed", "commercial", "industrial",
      "transport", "institutional", "openspace", "parking", "vacant"]]

# ACS households and vehicles. Independent of the taxi VMT target, so the
# Stage-3 leakage guard still holds and the transfer story is unaffected.
_ACS = ['hh_total', 'zero_veh_hh', 'veh_total', 'renter_veh_hh', 'owner_hh', 'renter_hh', 'veh_per_hh', 'zero_veh_share', 'renter_share', 'home_charge_cov', 'no_home_charge', 'no_home_charge_density']

_LAGS = [f"k{k}_{c}" for k in (1, 2)
         for c in ("aadt_proxy", "peak_hourly_vol", "night_hourly_vol",
                   "n_segments", "peak_ratio", "mean_hourly_vol",
                   "units_res", "jobs_area", "garage_area", "far_built",
                   "lu_entropy", "n_intersections", "arterial_share",
                   "no_home_charge", "veh_per_hh")]

FEATURES = _TRAFFIC + _MORPH + _LANDUSE + _ACS + _LAGS
TARGET = "kwh_day"

PARAMS = dict(objective="tweedie", tweedie_variance_power=1.3,
              learning_rate=0.05, num_leaves=31, min_data_in_leaf=25,
              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
              lambda_l2=1.0, verbose=-1, n_estimators=600)


def spatial_folds(df: pd.DataFrame, n: int = 5, seed: int = 42) -> np.ndarray:
    """Contiguous spatial blocks via k-means on cell centroids."""
    km = KMeans(n_clusters=n, random_state=seed, n_init=10)
    return km.fit_predict(df[["lat", "lon"]].to_numpy())


def _cv(df, folds, X, y):
    oof = np.zeros(len(df))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X[tr], y[tr], categorical_feature=[])
        oof[te] = m.predict(X[te])
    return np.clip(oof, 0, None)


def run() -> dict:
    df = pd.read_parquet(PROCESSED / "features_h3.parquet")
    X = df[FEATURES].to_numpy(float)
    y = df[TARGET].to_numpy(float)

    blocks = spatial_folds(df)
    rng = np.random.default_rng(42)
    random_folds = rng.integers(0, 5, len(df))

    oof_block = _cv(df, blocks, X, y)
    oof_rand = _cv(df, random_folds, X, y)

    def sc(p):
        """Decision-relevant metric suite.

        R2 on the raw target is reported for completeness but is a poor guide
        here: kwh_day is dominated by a handful of Manhattan cells, so raw R2
        under spatial blocking is decided almost entirely by whether one block
        happened to hold the tail. What the siting decision actually needs is
        ranking quality -- does the model put the right cells at the top --
        so Spearman rho and top-decile energy capture are the headline numbers.
        """
        k = max(1, len(y) // 10)
        top = np.argsort(-p)[:k]
        capture = float(y[top].sum() / y.sum()) if y.sum() > 0 else 0.0
        return dict(
            r2_raw=float(r2_score(y, p)),
            r2_log=float(r2_score(np.log1p(y), np.log1p(p))),
            mae=float(mean_absolute_error(y, p)),
            spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
            top_decile_capture=capture)

    metrics = {"spatial_block_cv": sc(oof_block), "random_kfold_cv": sc(oof_rand)}
    metrics["leakage_inflation_r2"] = round(
        metrics["random_kfold_cv"]["r2_raw"]
        - metrics["spatial_block_cv"]["r2_raw"], 4)
    # An oracle that ranks perfectly captures this much in its top decile;
    # it bounds what any model can achieve and makes capture interpretable.
    k = max(1, len(y) // 10)
    metrics["oracle_top_decile_capture"] = float(
        np.sort(y)[::-1][:k].sum() / y.sum())
    metrics["baseline_aadt_top_decile_capture"] = float(
        y[np.argsort(-df.aadt_proxy.to_numpy())[:k]].sum() / y.sum())

    # Ablation, measured rather than remembered: refit on traffic + street
    # morphology alone, so the contribution of the built-environment block is
    # a number this pipeline produces on every run instead of a claim.
    ablate = _TRAFFIC + _MORPH
    Xa = df[ablate].to_numpy(float)
    oof_a = _cv(df, blocks, Xa, y)
    metrics["ablation_no_landuse"] = sc(oof_a)
    metrics["ablation_no_landuse"]["n_features"] = len(ablate)
    metrics["n_features"] = len(FEATURES)
    metrics["landuse_capture_gain"] = round(
        metrics["spatial_block_cv"]["top_decile_capture"]
        - metrics["ablation_no_landuse"]["top_decile_capture"], 4)

    final = lgb.LGBMRegressor(**PARAMS).fit(X, y)
    df["demand_pred"] = np.clip(final.predict(X), 0, None)
    # The modelled surface is what the optimiser consumes; observed taxi kWh
    # is kept alongside so the dashboard can show model vs observation.
    df[["h3", "demand_pred"]].to_parquet(PROCESSED / "demand_pred.parquet",
                                         index=False)
    final.booster_.save_model(str(MODELS / "demand_lgbm.txt"))

    imp = sorted(zip(FEATURES, final.feature_importances_.tolist()),
                 key=lambda t: -t[1])
    metrics["feature_importance"] = imp
    (REPORTS / "demand_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    m = run()
    b, r = m["spatial_block_cv"], m["random_kfold_cv"]
    print("Stage 1 - demand model (target: kWh/day)")
    print(f"  {'':18} {'R2raw':>7} {'R2log':>7} {'rho':>6} {'top10%cap':>10}")
    for nm, s_ in (("spatial block CV", b), ("random k-fold CV", r)):
        print(f"  {nm:18} {s_['r2_raw']:+7.3f} {s_['r2_log']:+7.3f} "
              f"{s_['spearman']:6.3f} {s_['top_decile_capture']:9.1%}")
    print(f"  -> naive-CV R2 inflation      : {m['leakage_inflation_r2']:+.3f}")
    print(f"  -> oracle top-decile capture  : {m['oracle_top_decile_capture']:.1%}")
    print(f"  -> AADT-only baseline capture : {m['baseline_aadt_top_decile_capture']:.1%}")
    print("  top features:", ", ".join(k for k, _ in m["feature_importance"][:8]))
    b_cap = m["spatial_block_cv"]["top_decile_capture"]
    base = m["baseline_aadt_top_decile_capture"]
    verdict = "BEATS" if b_cap > base else "does NOT beat"
    print(f"  -> block-CV model {verdict} the AADT baseline "
          f"({b_cap:.1%} vs {base:.1%})")
    a = m["ablation_no_landuse"]
    print(f"  -> ablation, traffic+morphology only ({a['n_features']} feats): "
          f"{a['top_decile_capture']:.1%} capture, rho {a['spearman']:.3f}")
    print(f"  -> built-environment block contributes "
          f"{m['landuse_capture_gain']:+.1%} capture "
          f"({m['n_features']} features total)")
