"""Stage 7e - Con Edison distribution capacity -> per-cell grid headroom.

Source: Con Edison's published hosting-capacity platform, layer 0 of
`CECONY_EV_Storage_Prod` ("Charging Capacity for 3PH Feeders"), an ArcGIS
FeatureServer carrying ~59,600 feeder polylines inside the NYC envelope.

    TOTAL_XFMR_CAPACITY   substation/bank rating, MW
    SUBST_PEAK            peak load over the past year, MW
    headroom              rating - peak, which is Con Edison's own definition
                          of EV load capacity

The double-count this module exists to avoid
--------------------------------------------
`TOTAL_XFMR_CAPACITY` and `SUBST_PEAK` are **substation-bank** quantities that
repeat on every feeder hanging off that bank. Five separate feeders --
MACQUE_9W69, 5302, 91U1, MACQUE_9W65, 5361 -- all report the identical
264 MW / 202 MW for WASHINGTON STREET. Treating each feeder as owning 62 MW of
headroom would license roughly five times the load the bank can actually carry.

So headroom is deduplicated to the substation and the allocation constraint is
enforced there. This is the same failure mode as the gravity kernel in Stage 5,
where a charger's energy was counted once per cell it could reach.

Cells are assigned to their nearest feeder by BallTree over exploded polyline
vertices -- the same haversine-tree pattern used in `models/transfer.py`.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..config import RAW, INTERIM, PROCESSED, NYC_BBOX

LAYER = ("https://services.arcgis.com/ciPnsNFi1JLWVjva/arcgis/rest/services/"
         "CECONY_EV_Storage_Prod/FeatureServer/0")
CACHE = RAW / "grid" / "coned_feeders.parquet"
PAGE = 2000
EARTH_R = 6_371_000.0

# Beyond this, "nearest feeder" stops meaning "the grid that serves this cell".
# Layer 0 covers 3-phase radial feeders; Manhattan runs largely on Con Edison's
# secondary *network* system, which is published in different layers and is
# therefore sparse here. Without a cap, nearest-neighbour silently assigns
# Manhattan cells to Brooklyn's Brownsville bank across the East River and the
# constraint gets charged to the wrong substation -- worse than no constraint.
# Cells past the cap are marked "no published data" and left unconstrained.
MAX_ASSIGN_M = 1000.0

FIELDS = ["OBJECTID", "FEEDER_ID", "SUBSTATION", "TOTAL_XFMR_CAPACITY",
          "SUBST_PEAK", "SC_FEEDER_RATING", "VOLTAGE_KV"]


def _query(offset: int) -> dict:
    b = NYC_BBOX
    geom = {"xmin": b["lon_min"], "ymin": b["lat_min"],
            "xmax": b["lon_max"], "ymax": b["lat_max"],
            "spatialReference": {"wkid": 4326}}
    params = {
        "where": "1=1",
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(FIELDS),
        "returnGeometry": "true",
        "outSR": "4326",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "f": "json",
    }
    url = LAYER + "/query?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def fetch(force: bool = False) -> pd.DataFrame:
    """Page the layer into a flat table of feeder vertices."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    while True:
        d = _query(offset)
        feats = d.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f.get("attributes", {})
            paths = (f.get("geometry") or {}).get("paths") or []
            # One row per vertex: the tree needs points, and a feeder's route
            # matters more than its centroid for "which grid serves this cell".
            for path in paths:
                for x, y in path[::3]:      # every 3rd vertex is ample at 174 m
                    rows.append({"feeder_id": a.get("FEEDER_ID"),
                                 "substation": a.get("SUBSTATION"),
                                 "cap_mw": a.get("TOTAL_XFMR_CAPACITY"),
                                 "peak_mw": a.get("SUBST_PEAK"),
                                 "kv": a.get("VOLTAGE_KV"),
                                 "lon": x, "lat": y})
        print(f"  fetched {offset + len(feats):,} feeders "
              f"({len(rows):,} vertices)", flush=True)
        if len(feats) < PAGE:
            break
        offset += PAGE

    df = pd.DataFrame(rows)
    for c in ("cap_mw", "peak_mw", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "substation"])
    df.to_parquet(CACHE, index=False)
    return df


def substation_headroom(v: pd.DataFrame) -> pd.DataFrame:
    """One row per substation bank. This is where the double-count is killed.

    `cap_mw` / `peak_mw` repeat across every feeder on a bank, so they are
    reduced with `max` per substation rather than summed: the bank has one
    rating, however many feeders quote it.
    """
    s = (v.groupby("substation", as_index=False)
           .agg(cap_mw=("cap_mw", "max"), peak_mw=("peak_mw", "max"),
                n_feeders=("feeder_id", "nunique")))
    s["headroom_mw"] = (s.cap_mw - s.peak_mw).clip(lower=0.0)
    s["headroom_kw"] = s.headroom_mw * 1000.0
    return s


def build() -> pd.DataFrame:
    v = fetch()
    subs = substation_headroom(v)

    cells = pd.read_parquet(PROCESSED / "features_h3.parquet",
                            columns=["h3", "lat", "lon"])
    tree = BallTree(np.radians(v[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(np.radians(cells[["lat", "lon"]].to_numpy()), k=1)

    cells["substation"] = v.substation.to_numpy()[idx[:, 0]]
    cells["feeder_id"] = v.feeder_id.to_numpy()[idx[:, 0]]
    cells["dist_to_feeder_m"] = dist[:, 0] * EARTH_R

    far = cells.dist_to_feeder_m > MAX_ASSIGN_M
    cells.loc[far, ["substation", "feeder_id"]] = None

    out = cells.merge(subs[["substation", "headroom_kw", "headroom_mw",
                            "cap_mw", "peak_mw", "n_feeders"]],
                      on="substation", how="left")
    # Missing stays missing. Filling with 0 would read "no capacity" and
    # silently exclude every uncovered cell from the plan.
    out["grid_covered"] = out.substation.notna()

    # Diagnostics that make the double-count visible rather than implicit.
    naive = float(v.drop_duplicates("feeder_id")
                   .assign(h=lambda d: (d.cap_mw - d.peak_mw).clip(lower=0))
                   .h.sum())
    out.attrs["headroom_substation_mw"] = float(subs.headroom_mw.sum())
    out.attrs["headroom_naive_per_feeder_mw"] = naive
    out.attrs["n_substations"] = int(len(subs))
    out.attrs["n_feeders"] = int(v.feeder_id.nunique())
    out.attrs["cells_covered"] = int(out.grid_covered.sum())
    out.attrs["cells_total"] = int(len(out))

    out.to_parquet(INTERIM / "feeders_h3.parquet", index=False)
    subs.to_parquet(INTERIM / "substations.parquet", index=False)
    return out


if __name__ == "__main__":
    o = build()
    a = o.attrs
    print(f"feeders           : {a['n_feeders']:,}")
    print(f"substation banks  : {a['n_substations']:,}")
    cov = o[o.grid_covered]
    print(f"cells with published grid data : {a['cells_covered']:,} of "
          f"{a['cells_total']:,} ({100*a['cells_covered']/a['cells_total']:.1f}%), "
          f"cap {MAX_ASSIGN_M:.0f} m")
    print(f"  median distance to feeder    : {cov.dist_to_feeder_m.median():.0f} m")
    print()
    print(f"headroom, deduped to substation : {a['headroom_substation_mw']:>10,.0f} MW")
    print(f"headroom, naive per feeder      : {a['headroom_naive_per_feeder_mw']:>10,.0f} MW")
    print(f"  -> naive overstates capacity by "
          f"{a['headroom_naive_per_feeder_mw'] / max(a['headroom_substation_mw'], 1):.1f}x")
    print()
    s = pd.read_parquet(INTERIM / "substations.parquet")
    print(s.nlargest(6, "headroom_mw")[
        ["substation", "cap_mw", "peak_mw", "headroom_mw", "n_feeders"]]
        .to_string(index=False))
    print()
    import pandas as _pd
    surf = _pd.read_parquet(INTERIM.parent / "processed" / "features_h3.parquet",
                            columns=["h3", "borough"])
    j = o.merge(surf, on="h3", how="left")
    print("grid coverage by borough:")
    print(j.groupby("borough").grid_covered.agg(
        cells="size", covered="sum").assign(
        pct=lambda d: (100 * d.covered / d.cells).round(1)).to_string())
