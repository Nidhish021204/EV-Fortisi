"""Stage 8a - Con Edison per-feeder charging capacity -> per-cell grid limit.

Source: Con Edison's published hosting-capacity platform, service
`CECONY_EV_Storage_Prod` ("Charging and discharging layers for CECONY").

    layer 0   Charging Capacity for 3PH Feeders          82,809 polylines
    layer 2   No Charging Capacity for 1PH and 2PH Feeders  996 polylines

plus `CECONY_NwkSuppliedArea_Prod/0`, the boundary of the secondary-network
territory, which is what explains the holes in layer 0.

Why the constraint column changed
---------------------------------
This module used to derive headroom as `TOTAL_XFMR_CAPACITY - SUBST_PEAK`.
Those are **substation-bank** quantities that repeat on every feeder hanging
off the bank, so they had to be deduplicated with `max` to avoid licensing five
times the load a bank can carry.

But layer 0 also publishes `STORAGE_MAX_CHARGE`: Con Edison's own limit on how
much new charging load each **feeder** can take. It is the right quantity and
it needs no dedup, because it does not repeat. Four feeders on the one
WASHINGTON STREET bank (264 MW rated / 202 MW peak -> a single 62 MW bank
headroom) publish four different charging limits:

    91U1          0.138 MW
    5302          0.209 MW
    MACQUE_9W69   1.345 MW
    MACQUE_9W65   4.244 MW

A quarter of feeders publish exactly zero. Planning against 62 MW of bank
headroom when the feeder that actually serves the site allows 0.138 MW is the
same error as the per-feeder double-count, pointing the other way: it
understates the constraint instead of overstating the capacity.

Both numbers are computed and reported side by side. The feeder limit binds;
the bank figure is retained so the change is visible rather than asserted.

What `STORAGE_MIN_CHARGE` and the `SC_*` columns are not
-------------------------------------------------------
`STORAGE_MIN_CHARGE` is exactly 0.8 x `STORAGE_MAX_CHARGE` in every row
sampled, so it carries no independent information and is not ingested. The
`SC_*` columns (flicker, over-voltage, anti-island, thermal) are *export*
screens belonging to the discharge side; they are not load limits, and
`STORAGE_MAX_CHARGE` is not their minimum (0 of 600 rows matched).

Why Manhattan is still blank, stated properly
---------------------------------------------
Layer 0 covers 3-phase radial feeders. Manhattan runs largely on Con Edison's
secondary *network* system, whose published capacity layers (`CE_NwkHcv_*`)
report **generation** hosting capacity for DER interconnection - not load
headroom for charging. Substituting one for the other would be a category
error, so it is not done. Instead the network-supplied boundary is ingested to
distinguish

    feeder        capacity published, constraint enforced
    network_area  secondary network; charging capacity not published here
    no_data       outside both

Absence of published capacity is still not evidence of available capacity, but
now the reason for the absence is recorded rather than implied.

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

BASE = ("https://services.arcgis.com/ciPnsNFi1JLWVjva/arcgis/rest/services/")
LAYER_CHARGE = BASE + "CECONY_EV_Storage_Prod/FeatureServer/0"
LAYER_NOCHARGE = BASE + "CECONY_EV_Storage_Prod/FeatureServer/2"
LAYER_NWKAREA = BASE + "CECONY_NwkSuppliedArea_Prod/FeatureServer/0"

CACHE = RAW / "grid" / "coned_feeders.parquet"
CACHE_NOCHG = RAW / "grid" / "coned_no_charge.parquet"
CACHE_NWK = RAW / "grid" / "coned_network_areas.json"
PAGE = 2000
EARTH_R = 6_371_000.0

# Beyond this, "nearest feeder" stops meaning "the grid that serves this cell".
# Without a cap, nearest-neighbour silently assigns Manhattan cells to
# Brooklyn's Brownsville bank across the East River and the constraint gets
# charged to the wrong feeder -- worse than no constraint.
MAX_ASSIGN_M = 1000.0

# Unit guard. `STORAGE_MAX_CHARGE` carries no documented unit. Read as MW, a
# feeder taking 0.1-5 MW of new charging load is ordinary; read as kW it would
# mean a whole 13 kV feeder can host less than one Level 2 port, which is
# absurd. So a p99 inside this band is evidence for MW, and outside it the
# stage fails loudly rather than scaling the plan by 1000x.
CHARGE_P99_MW_BAND = (0.1, 50.0)

FIELDS = ["OBJECTID", "FEEDER_ID", "SUBSTATION", "TOTAL_XFMR_CAPACITY",
          "SUBST_PEAK", "SC_FEEDER_RATING", "VOLTAGE_KV", "STORAGE_MAX_CHARGE"]
FIELDS_NOCHG = ["OBJECTID", "FEEDER_ID", "SUBSTATION", "VOLTAGE_KV"]


def _query(layer: str, offset: int, fields: list[str]) -> dict:
    b = NYC_BBOX
    geom = {"xmin": b["lon_min"], "ymin": b["lat_min"],
            "xmax": b["lon_max"], "ymax": b["lat_max"],
            "spatialReference": {"wkid": 4326}}
    params = {
        "where": "1=1",
        "geometry": json.dumps(geom),
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(fields),
        "returnGeometry": "true",
        "outSR": "4326",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE),
        "f": "json",
    }
    url = layer + "/query?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def _page_vertices(layer: str, fields: list[str], label: str,
                   extra: dict | None = None) -> pd.DataFrame:
    """Page a polyline layer into a flat table of vertices.

    One row per vertex: the tree needs points, and a feeder's route matters
    more than its centroid for "which grid serves this cell".
    """
    extra = extra or {}
    rows: list[dict] = []
    offset = 0
    while True:
        d = _query(layer, offset, fields)
        feats = d.get("features", [])
        if not feats:
            break
        for f in feats:
            a = f.get("attributes", {})
            base = {"feeder_id": a.get("FEEDER_ID"),
                    "substation": a.get("SUBSTATION"),
                    "kv": a.get("VOLTAGE_KV")}
            for src, dst in extra.items():
                base[dst] = a.get(src)
            for path in (f.get("geometry") or {}).get("paths") or []:
                for x, y in path[::3]:      # every 3rd vertex is ample at 174 m
                    rows.append({**base, "lon": x, "lat": y})
        print(f"  {label}: fetched {offset + len(feats):,} features "
              f"({len(rows):,} vertices)", flush=True)
        if len(feats) < PAGE:
            break
        offset += PAGE
    return pd.DataFrame(rows)


def fetch(force: bool = False) -> pd.DataFrame:
    """Layer 0 vertices, carrying the per-feeder charging limit."""
    if CACHE.exists() and not force:
        df = pd.read_parquet(CACHE)
        if "charge_mw" in df.columns:
            return df
        print("  cache predates STORAGE_MAX_CHARGE; refetching", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df = _page_vertices(LAYER_CHARGE, FIELDS, "charge",
                        extra={"TOTAL_XFMR_CAPACITY": "cap_mw",
                               "SUBST_PEAK": "peak_mw",
                               "STORAGE_MAX_CHARGE": "charge_mw"})
    for c in ("cap_mw", "peak_mw", "charge_mw", "lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "substation"])
    df.to_parquet(CACHE, index=False)
    return df


def fetch_no_charge(force: bool = False) -> pd.DataFrame:
    """Layer 2 vertices: feeders Con Edison flags as having NO charging capacity."""
    if CACHE_NOCHG.exists() and not force:
        return pd.read_parquet(CACHE_NOCHG)
    CACHE_NOCHG.parent.mkdir(parents=True, exist_ok=True)
    df = _page_vertices(LAYER_NOCHARGE, FIELDS_NOCHG, "no-charge")
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df.to_parquet(CACHE_NOCHG, index=False)
    return df


def fetch_network_areas(force: bool = False) -> list:
    """Secondary-network service-area polygons, as raw ArcGIS rings."""
    if CACHE_NWK.exists() and not force:
        return json.loads(CACHE_NWK.read_text())
    CACHE_NWK.parent.mkdir(parents=True, exist_ok=True)
    rings: list = []
    offset = 0
    while True:
        d = _query(LAYER_NWKAREA, offset, ["OBJECTID"])
        feats = d.get("features", [])
        if not feats:
            break
        for f in feats:
            for r in (f.get("geometry") or {}).get("rings") or []:
                rings.append(r)
        print(f"  network-area: fetched {offset + len(feats):,} polygons "
              f"({len(rings):,} rings)", flush=True)
        if len(feats) < PAGE:
            break
        offset += PAGE
    CACHE_NWK.write_text(json.dumps(rings))
    return rings


def feeder_charge_capacity(v: pd.DataFrame) -> pd.DataFrame:
    """One row per feeder. No dedup arithmetic needed.

    `STORAGE_MAX_CHARGE` is a property of the feeder, constant across its own
    vertices, and genuinely different between feeders on the same bank -- which
    is exactly why it does not need the `max`-per-substation correction that
    `substation_headroom` applies to the bank rating.
    """
    f = (v.dropna(subset=["feeder_id"])
          .groupby("feeder_id", as_index=False)
          .agg(charge_mw=("charge_mw", "max"),
               substation=("substation", "first"),
               kv=("kv", "first")))
    f["charge_kw"] = f.charge_mw * 1000.0
    return f


def substation_headroom(v: pd.DataFrame) -> pd.DataFrame:
    """One row per substation bank -- retained as the comparison figure.

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


def _assert_units(feeders: pd.DataFrame) -> float:
    """Fail loudly if `STORAGE_MAX_CHARGE` cannot plausibly be megawatts."""
    p99 = float(np.nanpercentile(feeders.charge_mw.to_numpy(float), 99))
    lo, hi = CHARGE_P99_MW_BAND
    if not (lo <= p99 <= hi):
        raise ValueError(
            f"STORAGE_MAX_CHARGE p99 = {p99:.4g} is outside the plausible "
            f"megawatt band {CHARGE_P99_MW_BAND}. Con Edison publishes no unit "
            f"for this field; refusing to guess. Inspect the layer before "
            f"trusting the constraint.")
    return p99


def _assert_varies_within_bank(feeders: pd.DataFrame) -> int:
    """Prove the bank double-count does not apply to the per-feeder column."""
    g = feeders.groupby("substation").charge_mw.nunique()
    multi = feeders.groupby("substation").size() > 1
    varying = int((g[multi] > 1).sum())
    if varying == 0:
        raise ValueError(
            "STORAGE_MAX_CHARGE is constant within every multi-feeder bank, "
            "which would mean it repeats like the bank rating and needs the "
            "same dedup. Re-check the layer before using it per feeder.")
    return varying


def _in_network_area(lat: np.ndarray, lon: np.ndarray, rings: list) -> np.ndarray:
    """Point-in-polygon against the secondary-network boundary."""
    if not rings:
        return np.zeros(len(lat), dtype=bool)
    from shapely.geometry import Polygon
    from shapely import STRtree, points as shp_points

    polys = []
    for r in rings:
        if len(r) >= 4:
            try:
                p = Polygon(r)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty:
                    polys.append(p)
            except Exception:
                continue
    if not polys:
        return np.zeros(len(lat), dtype=bool)

    tree = STRtree(polys)
    pts = shp_points(lon, lat)
    hit = np.zeros(len(lat), dtype=bool)
    idx = tree.query(pts, predicate="intersects")
    hit[np.unique(idx[0])] = True
    return hit


def build() -> pd.DataFrame:
    v = fetch()
    nc = fetch_no_charge()
    rings = fetch_network_areas()

    feeders = feeder_charge_capacity(v)
    subs = substation_headroom(v)
    p99 = _assert_units(feeders)
    varying = _assert_varies_within_bank(feeders)

    cells = pd.read_parquet(PROCESSED / "features_h3.parquet",
                            columns=["h3", "lat", "lon"])
    cell_rad = np.radians(cells[["lat", "lon"]].to_numpy())

    tree = BallTree(np.radians(v[["lat", "lon"]].to_numpy()), metric="haversine")
    dist, idx = tree.query(cell_rad, k=1)
    cells["substation"] = v.substation.to_numpy()[idx[:, 0]]
    cells["feeder_id"] = v.feeder_id.to_numpy()[idx[:, 0]]
    cells["dist_to_feeder_m"] = dist[:, 0] * EARTH_R

    # Feeders explicitly flagged as having no charging capacity. A cell is
    # excluded when such a feeder is both close enough to be the grid that
    # serves it and closer than any feeder that does publish capacity.
    if len(nc):
        nc_tree = BallTree(np.radians(nc[["lat", "lon"]].to_numpy()),
                           metric="haversine")
        nd, _ = nc_tree.query(cell_rad, k=1)
        cells["dist_to_nocharge_m"] = nd[:, 0] * EARTH_R
    else:
        cells["dist_to_nocharge_m"] = np.inf
    cells["charge_excluded"] = ((cells.dist_to_nocharge_m <= MAX_ASSIGN_M)
                                & (cells.dist_to_nocharge_m
                                   < cells.dist_to_feeder_m))

    cells["in_network_area"] = _in_network_area(
        cells.lat.to_numpy(), cells.lon.to_numpy(), rings)

    far = cells.dist_to_feeder_m > MAX_ASSIGN_M
    cells.loc[far, ["substation", "feeder_id"]] = None

    # Three-state coverage: why a cell has no constraint is now recorded.
    state = np.where(~far, "feeder",
                     np.where(cells.in_network_area.to_numpy(),
                              "network_area", "no_data"))
    cells["grid_state"] = state
    cells["grid_covered"] = cells.grid_state == "feeder"

    out = cells.merge(feeders[["feeder_id", "charge_kw", "charge_mw", "kv"]],
                      on="feeder_id", how="left")
    out = out.merge(subs[["substation", "headroom_kw", "headroom_mw",
                          "cap_mw", "peak_mw", "n_feeders"]]
                    .rename(columns={"headroom_kw": "bank_headroom_kw",
                                     "headroom_mw": "bank_headroom_mw"}),
                    on="substation", how="left")

    # The binding constraint. Missing stays missing: filling with 0 would read
    # "no capacity" and silently exclude every uncovered cell from the plan.
    out["headroom_kw"] = out.charge_kw
    out.loc[~out.grid_covered, "headroom_kw"] = np.nan
    # Feeders that publish a hard zero are a real constraint, not missing data.
    out.loc[out.charge_excluded, "headroom_kw"] = 0.0

    a = out.attrs
    a["charge_feeder_total_mw"] = float(feeders.charge_mw.sum())
    a["headroom_substation_mw"] = float(subs.headroom_mw.sum())
    a["headroom_naive_per_feeder_mw"] = float(
        v.drop_duplicates("feeder_id")
         .assign(h=lambda d: (d.cap_mw - d.peak_mw).clip(lower=0)).h.sum())
    a["n_feeders"] = int(v.feeder_id.nunique())
    a["n_substations"] = int(len(subs))
    a["n_feeders_zero_charge"] = int((feeders.charge_mw <= 0).sum())
    a["banks_with_varying_charge"] = varying
    a["charge_p99_mw"] = p99
    a["n_no_charge_feeders"] = int(nc.feeder_id.nunique()) if len(nc) else 0
    a["cells_excluded"] = int(out.charge_excluded.sum())
    a["cells_total"] = int(len(out))
    for s in ("feeder", "network_area", "no_data"):
        a[f"cells_{s}"] = int((out.grid_state == s).sum())
    a["cells_covered"] = a["cells_feeder"]

    # Cell counts understate coverage because the land mask admits river and
    # park-interior cells, which have no grid because they have no buildings.
    # Weighting by built floor area says how much of the actual city is
    # classified -- Manhattan's blank is 99% secondary network, not unknown.
    built = pd.read_parquet(PROCESSED / "features_h3.parquet",
                            columns=["h3", "bldg_area"])
    bj = out[["h3", "grid_state"]].merge(built, on="h3", how="left")
    tot_built = float(bj.bldg_area.sum())
    for s in ("feeder", "network_area", "no_data"):
        a[f"built_share_{s}"] = round(
            float(bj.loc[bj.grid_state == s, "bldg_area"].sum())
            / max(tot_built, 1e-9), 4)

    out.to_parquet(INTERIM / "feeders_h3.parquet", index=False)
    subs.to_parquet(INTERIM / "substations.parquet", index=False)
    feeders.to_parquet(INTERIM / "feeder_capacity.parquet", index=False)
    return out


if __name__ == "__main__":
    o = build()
    a = o.attrs

    assert a["cells_feeder"] + a["cells_network_area"] + a["cells_no_data"] \
        == a["cells_total"], "grid_state does not partition the cells"

    print(f"feeders (layer 0)      : {a['n_feeders']:,}")
    print(f"  publishing zero      : {a['n_feeders_zero_charge']:,} "
          f"({100*a['n_feeders_zero_charge']/a['n_feeders']:.1f}%)")
    print(f"  banks where the limit varies between feeders : "
          f"{a['banks_with_varying_charge']:,}")
    print(f"no-charge feeders (layer 2) : {a['n_no_charge_feeders']:,}")
    print(f"substation banks       : {a['n_substations']:,}")
    print()
    print("capacity, the two readings side by side")
    print(f"  per-feeder charging limit (BINDING) : "
          f"{a['charge_feeder_total_mw']:>10,.0f} MW   p99 {a['charge_p99_mw']:.2f} MW/feeder")
    print(f"  substation bank headroom            : "
          f"{a['headroom_substation_mw']:>10,.0f} MW")
    print(f"  naive per-feeder bank headroom      : "
          f"{a['headroom_naive_per_feeder_mw']:>10,.0f} MW")
    ratio = a["headroom_substation_mw"] / max(a["charge_feeder_total_mw"], 1e-9)
    print(f"  -> the bank view is {ratio:,.1f}x the published charging limit")
    print()
    print("cell coverage, by reason        cells          built floor area")
    for s, why in (("feeder", "capacity published, constraint enforced"),
                   ("network_area", "secondary network, capacity not published"),
                   ("no_data", "outside both layers")):
        n = a[f"cells_{s}"]
        print(f"  {s:<13} {n:>6,} ({100*n/a['cells_total']:>5.1f}%)  "
              f"{100*a[f'built_share_{s}']:>5.1f}%   {why}")
    print(f"  cells on a no-charge feeder : {a['cells_excluded']:,}")
    print()
    cov = o[o.grid_covered]
    print(f"median distance to feeder : {cov.dist_to_feeder_m.median():.0f} m "
          f"(cap {MAX_ASSIGN_M:.0f} m)")
    print()
    surf = pd.read_parquet(PROCESSED / "features_h3.parquet",
                           columns=["h3", "borough"])
    j = o.merge(surf, on="h3", how="left")
    tab = (j.pivot_table(index="borough", columns="grid_state", values="h3",
                         aggfunc="size", fill_value=0))
    tab["cells"] = tab.sum(axis=1)
    print("grid state by borough:")
    print(tab.to_string())
