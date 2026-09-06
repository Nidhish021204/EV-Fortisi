"""Stage 7 - cross-domain transfer validation: Germany, NYC, Connecticut.

The question
-----------
Is siting suitability city-invariant, or does a model fit in one place only
work there? Answering it needs a feature space that exists in every domain,
and that constraint is severe: taxi trips, traffic counts and PLUTO land use
are NYC-only. What all three domains DO have is a georeferenced station network
with ports, power, operator and a commissioning date.

So the transferable feature space is **network morphology plus temporal
state**, and the question this module actually answers is narrower than "does
siting transfer":

    Given the shape and age of an existing charging network, where does the
    next station go?

That tests agglomeration and build-out dynamics. It does NOT test demand-driven
siting, because no demand signal exists outside NYC. Reporting it as the
broader claim would be false, so the limit is stated in the output.

Design
------
Every domain is treated identically:

  universe   H3 res-9 cells within `frontier_m` of at least one station open
             at the cutoff, which do NOT themselves hold a station. This is
             the "expansion frontier" and it is defined by the same rule
             everywhere, so the domains are comparable.
  features   distance to the k-th nearest station, station/port/fast counts in
             radius bands, local power and operator diversity, and recent
             build velocity -- all computed only from stations open at the
             cutoff.
  label      the cell receives at least one station during the forward window.

Because candidate cells hold no station at the cutoff, leave-one-cell-out is
automatic: no feature can read its own label.

The universe rule differs from Stage 4's, so these numbers are NOT comparable
with the Stage 4 PR-AUC. Within-domain runs are included precisely to give
each transfer its own like-for-like reference.
"""
from __future__ import annotations

import json

import h3
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..config import INTERIM, REPORTS, AFDC_NATIONAL, H3_RES
from ..ingest.ghsl import sample as ghsl_sample, GHSL_FEATURES
from .suitability import bagged_pu, _rank_metrics

EARTH_R = 6_371_000.0
RINGS_M = (250, 500, 1000, 2000, 5000)
KTH = (1, 2, 3, 5)
FRONTIER_M = 1000.0
DISK_K = 3  # res-9 grid_disk(3) reaches ~1.1 km, matching FRONTIER_M

# Each domain: cutoff, forward window. Germany's register runs to a partial
# 2023, so its window stops at 2022; NYC and CT use the Stage 4 split.
DOMAINS = {
    "DE": dict(cutoff="2019-12-31", fwd=("2020-01-01", "2022-12-31")),
    # Germany restricted to its densest cities. 7a hypothesised that Germany's
    # weak within-domain score came from nationwide, subsidy-driven expansion
    # into rural areas that network morphology cannot see. Running both scopes
    # turns that story into a measurement: if DE_METRO beats DE, the hypothesis
    # holds; if not, it was wrong.
    "DE_METRO": dict(cutoff="2019-12-31", fwd=("2020-01-01", "2022-12-31"),
                     metro_top_n=7),
    "NYC": dict(cutoff="2018-12-31", fwd=("2019-01-01", "2021-12-31")),
    "CT": dict(cutoff="2018-12-31", fwd=("2019-01-01", "2021-12-31")),
}

FEATURES_NET = (
    [f"dist_k{k}_m" for k in KTH]
    + [f"n_st_{r}m" for r in RINGS_M]
    + [f"n_ports_{r}m" for r in (500, 1000, 2000)]
    + [f"n_fast_{r}m" for r in (1000, 2000)]
    + ["kw_mean_2000m", "kw_max_2000m", "ports_per_station_2000m",
       "fast_share_2000m", "operator_entropy_2000m",
       "recent_2y_1000m", "recent_2y_2000m", "network_age_med_2000m",
       "log_density_1000m"]
)
# Network shape plus a globally-consistent built-form block. The pair is the
# point: 7a could not distinguish "network shape does not transfer" from
# "nothing transfers", and only the delta between these two spaces can.
FEATURES_FULL = FEATURES_NET + GHSL_FEATURES
FEATURES = FEATURES_NET          # back-compat for callers


# ── domain loaders: normalise to one schema ─────────────────────────

def load_domain(name: str) -> pd.DataFrame:
    """-> lat, lon, ports, kw, is_fast, open_date, operator."""
    if name == "NYC":
        d = pd.read_parquet(INTERIM / "stations_nyc.parquet")
        out = pd.DataFrame({
            "lat": d.lat, "lon": d.lon, "ports": d.ports, "kw": d.kw,
            "is_fast": (d.dcfc > 0).astype(int), "open_date": d.open_date,
            "operator": d.network.fillna("Non-Networked")})
    elif name in ("DE", "DE_METRO"):
        d = pd.read_parquet(INTERIM / "stations_de.parquet")
        if name == "DE_METRO":
            # Cities selected by station count, never by hardcoded names: the
            # register's umlauts mis-encode in this environment, so matching on
            # literals like "Munchen" would silently drop the largest cities.
            n = DOMAINS[name].get("metro_top_n", 7)
            top = d.ort.value_counts().head(n).index
            d = d[d.ort.isin(top)].copy()
        out = pd.DataFrame({
            "lat": d.lat, "lon": d.lon, "ports": d.ports, "kw": d.kw,
            "is_fast": d.is_fast, "open_date": d.open_date,
            "operator": d.operator.fillna("unknown")})
    elif name == "CT":
        # From the national AFDC file, not the 385-row state extract: same
        # schema as NYC and it carries the open dates the extract lacks.
        cols = ["Fuel Type Code", "State", "Latitude", "Longitude", "Open Date",
                "EV Network", "EV Level1 EVSE Num", "EV Level2 EVSE Num",
                "EV DC Fast Count"]
        d = pd.read_csv(AFDC_NATIONAL, usecols=cols, low_memory=False)
        d = d[(d["Fuel Type Code"] == "ELEC") & (d.State == "CT")].copy()
        for c in ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        out = pd.DataFrame({
            "lat": pd.to_numeric(d.Latitude, errors="coerce"),
            "lon": pd.to_numeric(d.Longitude, errors="coerce"),
            "ports": d["EV Level1 EVSE Num"] + d["EV Level2 EVSE Num"] + d["EV DC Fast Count"],
            "kw": 1.4 * d["EV Level1 EVSE Num"] + 7.2 * d["EV Level2 EVSE Num"]
                  + 50.0 * d["EV DC Fast Count"],
            "is_fast": (d["EV DC Fast Count"] > 0).astype(int),
            "open_date": pd.to_datetime(d["Open Date"], errors="coerce"),
            "operator": d["EV Network"].fillna("Non-Networked")})
    else:
        raise ValueError(name)

    out = out.dropna(subset=["lat", "lon", "open_date"]).reset_index(drop=True)
    out["h3"] = [h3.latlng_to_cell(a, o, H3_RES)
                 for a, o in zip(out.lat, out.lon)]
    return out


# ── feature construction, identical in every domain ─────────────────

def _entropy(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log(p)).sum() / np.log(len(p)) if len(p) > 1 else 0.0)


def build_domain(name: str) -> pd.DataFrame:
    cfg = DOMAINS[name]
    st = load_domain(name)
    T = pd.Timestamp(cfg["cutoff"])
    f0, f1 = map(pd.Timestamp, cfg["fwd"])

    past = st[st.open_date <= T].reset_index(drop=True)
    future_cells = set(st.loc[st.open_date.between(f0, f1), "h3"])
    occupied = set(past.h3)

    # Expansion frontier: cells near the existing network but not in it.
    cand: set[str] = set()
    for c in occupied:
        cand.update(h3.grid_disk(c, DISK_K))
    cand -= occupied
    cells = sorted(cand)
    if not cells:
        raise RuntimeError(f"{name}: empty candidate frontier")

    ll = np.array([h3.cell_to_latlng(c) for c in cells])
    cell_rad = np.radians(ll)
    st_rad = np.radians(past[["lat", "lon"]].to_numpy())
    tree = BallTree(st_rad, metric="haversine")

    df = pd.DataFrame({"h3": cells, "lat": ll[:, 0], "lon": ll[:, 1]})

    # k-th nearest distances
    kmax = max(KTH)
    kq = min(kmax, len(past))
    dist, _ = tree.query(cell_rad, k=kq)
    dist_m = dist * EARTH_R
    for k in KTH:
        col = dist_m[:, min(k, kq) - 1] if kq >= 1 else np.full(len(cells), 5e4)
        df[f"dist_k{k}_m"] = np.minimum(col, 50_000.0)

    ports = past.ports.to_numpy(float)
    kw = past.kw.to_numpy(float)
    fast = past.is_fast.to_numpy(float)
    age_days = (T - past.open_date).dt.days.to_numpy(float)
    recent = (age_days <= 730).astype(float)
    op_codes = past.operator.astype("category").cat.codes.to_numpy()

    # Radius-band aggregates. query_radius returns neighbour indices per cell.
    band_idx: dict[int, np.ndarray] = {}
    for r in RINGS_M:
        idx = tree.query_radius(cell_rad, r=r / EARTH_R)
        band_idx[r] = idx
        df[f"n_st_{r}m"] = [len(i) for i in idx]
        if r in (500, 1000, 2000):
            df[f"n_ports_{r}m"] = [ports[i].sum() if len(i) else 0.0 for i in idx]
        if r in (1000, 2000):
            df[f"n_fast_{r}m"] = [fast[i].sum() if len(i) else 0.0 for i in idx]

    i2k = band_idx[2000]
    df["kw_mean_2000m"] = [kw[i].mean() if len(i) else 0.0 for i in i2k]
    df["kw_max_2000m"] = [kw[i].max() if len(i) else 0.0 for i in i2k]
    df["ports_per_station_2000m"] = [
        ports[i].sum() / len(i) if len(i) else 0.0 for i in i2k]
    df["fast_share_2000m"] = [
        fast[i].mean() if len(i) else 0.0 for i in i2k]
    df["operator_entropy_2000m"] = [
        _entropy(np.bincount(op_codes[i])) if len(i) else 0.0 for i in i2k]
    df["network_age_med_2000m"] = [
        np.median(age_days[i]) / 365.25 if len(i) else 0.0 for i in i2k]
    df["recent_2y_2000m"] = [recent[i].sum() if len(i) else 0.0 for i in i2k]
    df["recent_2y_1000m"] = [
        recent[i].sum() if len(i) else 0.0 for i in band_idx[1000]]
    # Density spans orders of magnitude between a German village and Midtown;
    # the log keeps that on one comparable scale across domains.
    df["log_density_1000m"] = np.log1p(df["n_st_1000m"].to_numpy(float))

    # Globally-consistent built form, sampled identically in every domain.
    g = ghsl_sample(df[["h3", "lat", "lon"]])
    df = df.merge(g[["h3"] + GHSL_FEATURES], on="h3", how="left")
    df[GHSL_FEATURES] = df[GHSL_FEATURES].fillna(0.0)

    df["y"] = df.h3.isin(future_cells).astype(int)
    df["domain"] = name
    return df


# ── transfer matrix ─────────────────────────────────────────────────

def _fit_score(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42,
               features: list[str] | None = None):
    """Bagged-PU fit on `train`, scored on `test`.

    PU rather than plain classification for the same reason as Stage 4: an
    unbuilt frontier cell is unlabeled, not negative.
    """
    feats = features or FEATURES_NET
    Xtr = train[feats].to_numpy(float)
    ytr = train.y.to_numpy(int)
    _, models, _ = bagged_pu(Xtr, ytr, n_bags=25, seed=seed)
    Xte = test[feats].to_numpy(float)
    return np.mean([m.predict_proba(Xte)[:, 1] for m in models], axis=0)


def run() -> dict:
    doms = {}
    for name in DOMAINS:
        d = build_domain(name)
        doms[name] = d
        print(f"  {name}: {len(d):,} frontier cells, {d.y.sum():,} received a "
              f"station ({d.y.mean():.2%})", flush=True)

    results: dict = {"domains": {
        k: {"frontier_cells": int(len(v)), "positives": int(v.y.sum()),
            "base_rate": float(v.y.mean()), "cutoff": DOMAINS[k]["cutoff"],
            "forward_window": list(DOMAINS[k]["fwd"])}
        for k, v in doms.items()}}

    # Within-domain reference uses out-of-bag scores, so it is not fitting and
    # scoring the same rows.
    spaces = {"net": FEATURES_NET, "full": FEATURES_FULL}
    matrices = {}
    for space, feats in spaces.items():
        print(f"\n  -- feature space: {space} ({len(feats)} features) --",
              flush=True)
        matrix = {}
        for src in DOMAINS:
            for tgt in DOMAINS:
                key = f"{src}->{tgt}"
                if src == tgt:
                    d = doms[src]
                    score, _, _ = bagged_pu(d[feats].to_numpy(float),
                                            d.y.to_numpy(int), n_bags=25)
                else:
                    score = _fit_score(doms[src], doms[tgt], features=feats)
                m = _rank_metrics(doms[tgt].y.to_numpy(int), score)
                matrix[key] = {"pr_auc": m["pr_auc"], "ndcg": m["ndcg"],
                               "precision@100": m["precision@100"],
                               "lift@100": m["lift@100"],
                               "base_rate": m["base_rate"]}
                print(f"  {key:<16} PR-AUC {m['pr_auc']:.3f}  "
                      f"P@100 {m['precision@100']:.1%}", flush=True)
        matrices[space] = matrix
    results["matrices"] = matrices
    matrix = matrices["net"]
    results["matrix"] = matrix          # 7a-compatible view

    # Baselines per target domain, so each transfer has a floor to clear.
    base = {}
    rng = np.random.default_rng(0)
    for tgt, d in doms.items():
        y = d.y.to_numpy(int)
        base[tgt] = {
            "random": _rank_metrics(y, rng.random(len(d)))["pr_auc"],
            "near_existing": _rank_metrics(y, -d.dist_k1_m.to_numpy())["pr_auc"],
            "local_density": _rank_metrics(y, d.n_st_1000m.to_numpy())["pr_auc"],
        }
    results["baselines"] = base

    # Retention: transferred performance as a share of the target's own
    # within-domain reference. 1.0 would mean the foreign model is as good as
    # a locally-fit one.
    ret = {}
    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            own = matrix[f"{tgt}->{tgt}"]["pr_auc"]
            ret[f"{src}->{tgt}"] = round(matrix[f"{src}->{tgt}"]["pr_auc"] / own, 3) \
                if own > 0 else 0.0
    results["retention_vs_local"] = ret
    results["verdict"] = verdicts(results)
    # The comparison is the finding, so both spaces get the same decision rule.
    results["verdict_by_space"] = {
        sp: verdicts({"matrix": mx, "baselines": base})
        for sp, mx in matrices.items()}
    results["feature_counts"] = {sp: len(f) for sp, f in spaces.items()}

    (REPORTS / "transfer_metrics.json").write_text(json.dumps(results, indent=2))
    return results


def verdicts(results: dict) -> dict:
    """Does a transferred model beat the TARGET domain's best trivial baseline?

    Retention against a locally-fit model flatters transfer: a domain whose own
    model is weak makes any import look competitive. The decision-relevant
    question is whether shipping a foreign model beats what the target could
    already do for free, so that is the test recorded here.
    """
    out = {}
    for key, m in results["matrix"].items():
        src, tgt = key.split("->")
        if src == tgt:
            continue
        b = results["baselines"][tgt]
        name = max(b, key=b.get)
        ratio = m["pr_auc"] / b[name] if b[name] > 0 else 0.0
        out[key] = {"model_pr_auc": round(m["pr_auc"], 4),
                    "baseline": name,
                    "baseline_pr_auc": round(b[name], 4),
                    "ratio": round(ratio, 3),
                    "beats_baseline": bool(m["pr_auc"] > b[name])}
    out["_summary"] = {
        "n_transfers": len([k for k in out if not k.startswith("_")]),
        "n_beating_baseline": sum(
            1 for k, v in out.items() if not k.startswith("_") and v["beats_baseline"]),
    }
    return out


if __name__ == "__main__":
    print("Stage 7 - cross-domain transfer validation")
    print("feature space: network morphology + temporal state "
          f"({len(FEATURES)} features)\n")
    r = run()

    print("\n  transfer matrix (PR-AUC), rows = trained on, cols = tested on")
    names = list(DOMAINS)
    print("  " + " " * 10 + "".join(f"{n:>10}" for n in names))
    for src in names:
        row = "".join(f"{r['matrix'][f'{src}->{t}']['pr_auc']:10.3f}" for t in names)
        print(f"  {src:<10}{row}")

    print("\n  target base rates / best baseline PR-AUC")
    for t in names:
        b = r["baselines"][t]
        best = max(b, key=b.get)
        print(f"    {t:<5} base {r['domains'][t]['base_rate']:.2%}   "
              f"best baseline: {best} {b[best]:.3f}   "
              f"local model {r['matrix'][f'{t}->{t}']['pr_auc']:.3f}")

    print("\n  retention vs a locally-fit model")
    for k, v in sorted(r["retention_vs_local"].items(), key=lambda x: -x[1]):
        print(f"    {k:<10} {v:5.1%}")
