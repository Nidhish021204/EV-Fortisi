"""Export model outputs to the compact payload the dashboard reads.

The dashboard renders its own vector geometry -- no tile server, no basemap
imagery, no network at runtime. So geometry ships precomputed: true H3 cell
boundaries projected to Web Mercator and quantised to integer metres relative
to a local origin, which keeps the grid payload small without the frontend
needing an H3 library or a projection step.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import h3

from .config import PROCESSED, INTERIM, REPORTS, DASH_DATA, TRAIN_CUTOFF, RAW


def _raw_corpus_bytes() -> int:
    """Actual size of the raw corpus on disk, walked once per export."""
    return sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file())

R_MERC = 6378137.0


def _merc(lat, lon):
    x = R_MERC * np.radians(np.asarray(lon, float))
    lat = np.clip(np.asarray(lat, float), -85.05, 85.05)
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def build() -> dict:
    DASH_DATA.mkdir(parents=True, exist_ok=True)

    surf = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    suit = pd.read_parquet(PROCESSED / "suitability.parquet")
    df = surf.merge(suit, on="h3", how="left")
    df["p_viable"] = df.p_viable.fillna(0.0)

    st = pd.read_parquet(INTERIM / "stations_nyc.parquet")
    plan = pd.read_parquet(PROCESSED / "plan_sites.parquet")

    # ── grid geometry, quantised to metres about a local origin ─────
    cx, cy = _merc(df.lat.values, df.lon.values)
    ox, oy = float(cx.min()), float(cy.min())

    rings, xs, ys = [], [], []
    for cell in df.h3.values:
        b = h3.cell_to_boundary(cell)          # [(lat, lon), ...]
        bx, by = _merc([p[0] for p in b], [p[1] for p in b])
        rings.append([[int(round(v - ox)) for v in bx],
                      [int(round(v - oy)) for v in by]])
    xs = [int(round(v - ox)) for v in cx]
    ys = [int(round(v - oy)) for v in cy]

    def col(name, scale=1.0, nd=1):
        return [round(float(v) * scale, nd) for v in df[name].values]

    grid = {
        "origin": [ox, oy],
        "count": int(len(df)),
        "fields": ["demand", "served", "unmet", "p_viable", "aadt", "ports",
                   "dwell", "built1921"],
        "cx": xs, "cy": ys,
        "rings": rings,
        "h3": df.h3.tolist(),
        "borough": df.borough.tolist(),
        "demand": col("demand_kwh"),
        "served": col("served_kwh"),
        "unmet": col("unmet_kwh"),
        "p_viable": [round(float(v), 4) for v in df.p_viable.values],
        "aadt": col("aadt_proxy", nd=0),
        "ports": [int(v) for v in df.ports.values],
        "dwell": col("dwell_surplus", nd=0),
        "built1921": [int(v) for v in df.y_future.values],
    }
    (DASH_DATA / "grid.json").write_text(json.dumps(grid, separators=(",", ":")))

    # ── existing network ────────────────────────────────────────────
    sx, sy = _merc(st.lat.values, st.lon.values)
    stations = {
        "origin": [ox, oy],
        "x": [int(round(v - ox)) for v in sx],
        "y": [int(round(v - oy)) for v in sy],
        "lat": [round(float(v), 5) for v in st.lat.values],
        "lon": [round(float(v), 5) for v in st.lon.values],
        "name": st.name.fillna("").tolist(),
        "city": st.city.fillna("").tolist(),
        "network": st.network.fillna("Non-Networked").tolist(),
        "ports": [int(v) for v in st.ports.values],
        "dcfc": [int(v) for v in st.dcfc.values],
        "kw": [round(float(v), 1) for v in st.kw.values],
        "year": [int(v) if pd.notna(v) else 0 for v in st.open_year.values],
    }
    (DASH_DATA / "stations.json").write_text(json.dumps(stations, separators=(",", ":")))

    # ── recommended plan ────────────────────────────────────────────
    px, py = _merc(plan.lat.values, plan.lon.values)
    plan_out = plan.copy()
    plan_out["x"] = [int(round(v - ox)) for v in px]
    plan_out["y"] = [int(round(v - oy)) for v in py]
    (DASH_DATA / "plan.json").write_text(
        plan_out.to_json(orient="records", double_precision=3))

    # ── metrics bundle ──────────────────────────────────────────────
    def rj(p):
        f = REPORTS / p
        return json.loads(f.read_text()) if f.exists() else {}

    demand_m = rj("demand_metrics.json")
    suit_m = rj("suitability_metrics.json")
    transfer_m = rj("transfer_metrics.json")
    cert_m = rj("certification.json")
    feeder_m = rj("feeder_impact.json")
    unc_m = rj("uncertainty.json")
    unc_m.pop("site_frequency", None)   # per-site lives on plan.json
    unc_m.pop("seed_runs", None)

    # Grid coverage is the caveat that bounds every grid claim. Con Edison's
    # 3-phase feeder layer does not reach Manhattan's secondary network, and
    # the network layers publish *generation* hosting capacity rather than
    # charging load headroom -- so the honest report is not "covered / not
    # covered" but why each cell falls where it does. Coverage is also given
    # weighted by built floor area, because cell counts are diluted by river
    # and park cells that have no grid because they have no buildings.
    fpath = INTERIM / "feeders_h3.parquet"
    grid_cov = {}
    if fpath.exists():
        cols = ["h3", "grid_covered", "grid_state", "substation", "feeder_id",
                "headroom_kw", "bank_headroom_kw", "charge_excluded"]
        fh = pd.read_parquet(fpath)
        fh = fh[[c for c in cols if c in fh.columns]]
        gj = fh.merge(df[["h3", "borough", "bldg_area"]], on="h3", how="left")
        by = (gj.pivot_table(index="borough", columns="grid_state", values="h3",
                             aggfunc="size", fill_value=0).reset_index())
        for s in ("feeder", "network_area", "no_data"):
            if s not in by:
                by[s] = 0
        by["cells"] = by[["feeder", "network_area", "no_data"]].sum(axis=1)
        by["pct"] = (100 * by.feeder / by.cells).round(1)
        built = gj.groupby("grid_state").bldg_area.sum()
        tot_built = float(built.sum())
        # City-wide totals from the source tables, not from the cells that
        # happen to be assigned -- otherwise the dashboard would quote a
        # different feeder-vs-bank ratio than the ingest stage and PLAN.md do.
        fc = INTERIM / "feeder_capacity.parquet"
        sub_p = INTERIM / "substations.parquet"
        fdr = (pd.read_parquet(fc) if fc.exists()
               else fh.dropna(subset=["feeder_id"]).drop_duplicates("feeder_id")
                      .rename(columns={"headroom_kw": "charge_kw"}))
        subs_all = (pd.read_parquet(sub_p) if sub_p.exists()
                    else fh.drop_duplicates("substation")
                           .rename(columns={"bank_headroom_kw": "headroom_kw"}))
        grid_cov = {
            "by_borough": by.to_dict(orient="records"),
            "cells_covered": int((fh.grid_state == "feeder").sum()),
            "cells_total": int(len(fh)),
            "cells_network_area": int((fh.grid_state == "network_area").sum()),
            "cells_no_data": int((fh.grid_state == "no_data").sum()),
            "cells_excluded": int(fh.charge_excluded.sum())
            if "charge_excluded" in fh else 0,
            "built_share": {s: round(float(built.get(s, 0.0)) / max(tot_built, 1e-9), 4)
                            for s in ("feeder", "network_area", "no_data")},
            "substations": int(len(subs_all)),
            "feeders": int(len(fdr)),
            # The two readings side by side: the per-feeder charging limit that
            # actually binds, and the bank headroom the plan used to assume.
            "feeder_capacity_total_mw": round(
                float(fdr.charge_kw.sum() / 1000), 1),
            "headroom_total_mw": round(
                float(subs_all.headroom_kw.sum() / 1000), 1),
        }
    alloc_m = rj("allocation_plan.json")
    alloc_m.pop("sites", None)

    boro = (df.groupby("borough")
              .agg(hexes=("h3", "size"), demand=("demand_kwh", "sum"),
                   served=("served_kwh", "sum"), unmet=("unmet_kwh", "sum"),
                   ports=("ports", "sum"))
              .reset_index())
    boro["coverage"] = 100 * boro.served / boro.demand
    boro["kwh_per_port"] = boro.demand / boro.ports.replace(0, np.nan)
    plan_by_boro = plan.groupby("borough").size().to_dict()
    boro["planned"] = boro.borough.map(plan_by_boro).fillna(0).astype(int)

    metrics = {
        "generated": pd.Timestamp.utcnow().isoformat(),
        "cutoff": TRAIN_CUTOFF,
        "corpus": {
            "taxi_trips": 47_400_000,
            "taxi_dropoffs_binned": int(df.dropoffs.sum()),
            "traffic_segments": 14_359,
            "stations_nyc": int(len(st)),
            "stations_ports": int(st.ports.sum()),
            "stations_mw": round(float(st.kw.sum() / 1000), 1),
            "hex_cells": int(len(df)),
            # Measured, not asserted: this was a hardcoded 11.5 GB and went
            # stale the moment Stage 8b added 1.6 GB of for-hire records.
            "raw_bytes": _raw_corpus_bytes(),
        },
        "demand": demand_m,
        "suitability": suit_m,
        "transfer": transfer_m,
        "certification": cert_m,
        "feeder_impact": feeder_m,
        "grid_coverage": grid_cov,
        "uncertainty": unc_m,
        "allocation": alloc_m,
        "by_borough": boro.round(2).to_dict(orient="records"),
        "network_mix": st.network.fillna("Non-Networked")
                         .value_counts().head(8).to_dict(),
        "build_history": st.dropna(subset=["open_year"])
                           .groupby(st.open_year.astype("Int64"))
                           .size().to_dict(),
    }
    metrics["build_history"] = {str(k): int(v)
                                for k, v in metrics["build_history"].items()
                                if k and k >= 2010}
    (DASH_DATA / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                       default=str))
    return metrics


if __name__ == "__main__":
    m = build()
    import os
    print("dashboard payload -> %s" % DASH_DATA)
    for f in sorted(DASH_DATA.glob("*.json")):
        print("  %-16s %8.1f KB" % (f.name, os.path.getsize(f) / 1024))
    print("\ncorpus:", json.dumps(m["corpus"], indent=1))
