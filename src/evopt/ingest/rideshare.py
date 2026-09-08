"""Stage 8b - current-era for-hire activity -> zone-level correction of the surface.

The problem this fixes
---------------------
`ingest/mobility.py` measures demand from 2015-01 and 2016-01..03 yellow taxi,
the only NYC releases carrying raw pickup/dropoff lat-lon. That buys a
continuous sub-zone surface, at two costs the README lists as limitations:
yellow taxi is concentrated in Manhattan, and the window predates both the
pandemic and the TLC electrification mandate.

Since 2019 the city publishes high-volume for-hire vehicle trips -- 1.65 billion
of them against 333 million yellow taxi over the same span. That is now the
larger and far less Manhattan-skewed half of the fleet, and the surface has
never seen it.

Why this is a re-weighting and not a new surface
-----------------------------------------------
HVFHV records carry **no coordinates**. Verified against the April 2026 file
(20,995,953 rows): the schema has `PULocationID` / `DOLocationID` only -- 263
taxi zones, roughly 33x coarser than the H3 res-9 cells the model runs on. The
2015-16 taxi vintages remain the only source of within-zone structure.

So each source is used for what it can actually support:

    2015-16 taxi (lat/lon)      within-zone SHAPE, at res-9
    2019-2026 HVFHV/yellow/green  cross-zone MAGNITUDE, at zone level

**Residual limitation, stated rather than buried:** within-zone shape is still
2015-16. Where a zone's activity moved from one corner to another after 2016,
this cannot see it.

Shares, not levels
------------------
The correction is a redistribution, not a rescaling. For zone z

    r_z = (current share of z) / (2015-16 share of z)

and the rescaled surface is renormalised so its **total is exactly preserved**.
Two reasons. The two sources are not on a common absolute scale -- one is
bbox-and-distance-filtered trip records, the other is a published trip count --
so only within-source shares are comparable. And the surface's absolute level is
already a modelled scenario (`surface.EV_SHARE`), whereas its spatial
distribution is the measurement. Conserving the total and correcting the
distribution keeps each part doing its own job.

Season is matched, not assumed
------------------------------
The taxi baseline is Q1-heavy (Jan 2015 + Jan-Mar 2016), so the current window
is Q1 2026. The three raw months downloaded are exactly the months the
aggregate window covers, which makes the count-vs-VMT cross-check compare the
same trips rather than two different periods.

Sources
-------
`data.cityofnewyork.us/resource/c5iv-bn4s` monthly pickups and drop-offs by
taxi zone and industry, 2019-01..2026-07. It carries trip **counts** only, and
the surface runs on VMT, so per-zone mean trip miles come from three raw
monthly files. Zone geometry is TLC's own `taxi_zones.zip`.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..config import (RAW, INTERIM, PROCESSED, REPORTS,
                      TLC_CURRENT_MONTHS, TLC_INDUSTRIES)
from ..geo import h3_centroids

AGG_URL = "https://data.cityofnewyork.us/resource/c5iv-bn4s.json"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{kind}_tripdata_{m}.parquet"

CACHE_DIR = RAW / "mobility" / "tlc_current"
CACHE_AGG = CACHE_DIR / "zone_monthly.parquet"
CACHE_ZONES = CACHE_DIR / "taxi_zones.zip"
CACHE_VMT = CACHE_DIR / "zone_vmt_q1.parquet"

PAGE = 50_000

# Aggregate-table industry label -> raw-file prefix. Green is ~1 MB/month and
# 0.7% of trips, but excluding it would bias outer-borough zones where it runs.
INDUSTRY_FILE = {"FHV - High Volume": "fhvhv",
                 "Yellow Taxi": "yellow",
                 "Green Cab": "green"}

# Per-industry column names for the raw files.
_TRIP_COLS = {"fhvhv": ("DOLocationID", "PULocationID", "trip_miles"),
              "yellow": ("DOLocationID", "PULocationID", "trip_distance"),
              "green": ("DOLocationID", "PULocationID", "trip_distance")}


# ── fetch ───────────────────────────────────────────────────────────

def fetch_aggregate(force: bool = False) -> pd.DataFrame:
    """Monthly trip counts by taxi zone, industry and pickup/drop-off."""
    if CACHE_AGG.exists() and not force:
        return pd.read_parquet(CACHE_AGG)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    off = 0
    while True:
        url = (f"{AGG_URL}?$limit={PAGE}&$offset={off}"
               "&$select=metric_month,industry,pickup_dropoff,locationid,"
               "borough,zone,trip_count")
        with urllib.request.urlopen(url, timeout=180) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        rows.extend(batch)
        print(f"  aggregate: {len(rows):,} rows", flush=True)
        if len(batch) < PAGE:
            break
        off += PAGE
    df = pd.DataFrame(rows)
    df["locationid"] = pd.to_numeric(df.locationid, errors="coerce").astype("Int64")
    df["trip_count"] = pd.to_numeric(df.trip_count, errors="coerce").fillna(0.0)
    df["metric_month"] = pd.to_datetime(df.metric_month).dt.tz_localize(None)
    df = df.dropna(subset=["locationid"])
    df.to_parquet(CACHE_AGG, index=False)
    return df


def fetch_zones(force: bool = False):
    """TLC taxi-zone polygons, reprojected to WGS84."""
    import geopandas as gpd
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_ZONES.exists() or force:
        with urllib.request.urlopen(ZONES_URL, timeout=300) as r:
            CACHE_ZONES.write_bytes(r.read())
    # The archive nests the shapefile one directory down, so the zip root has
    # nothing readable in it -- the member path has to be given explicitly.
    with zipfile.ZipFile(CACHE_ZONES) as zf:
        shp = next(n for n in zf.namelist() if n.endswith(".shp"))
    z = gpd.read_file(f"zip://{CACHE_ZONES}!{shp}")
    z = z.to_crs(4326)
    z["locationid"] = pd.to_numeric(z["LocationID"], errors="coerce").astype("Int64")
    return z[["locationid", "zone", "borough", "geometry"]].dropna(subset=["locationid"])


def _download(kind: str, month: str):
    """Fetch one monthly trip file, atomically.

    These are 0.5 GB each. Writing straight to the final name would leave a
    truncated file looking like a complete cache after any interruption, and
    the next run would silently aggregate a partial month.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{kind}_{month}.parquet"
    if p.exists():
        return p
    tmp = p.with_suffix(".parquet.part")
    url = TRIP_URL.format(kind=kind, m=month)
    print(f"  downloading {p.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=1800) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    tmp.replace(p)
    return p


def fetch_zone_vmt(force: bool = False) -> pd.DataFrame:
    """Per-zone trip count and VMT for the raw window, by industry.

    The aggregate table has counts but no miles, and `kwh_day` is built from
    terminating VMT -- so mean trip miles have to come from raw records.
    """
    if CACHE_VMT.exists() and not force:
        return pd.read_parquet(CACHE_VMT)

    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='3GB'")

    out = []
    for kind in TLC_INDUSTRIES:
        do, pu, miles = _TRIP_COLS[kind]
        paths = [str(_download(kind, m)) for m in TLC_CURRENT_MONTHS]
        sql = f"""
        WITH src AS (
            SELECT "{do}" AS do_id, "{pu}" AS pu_id, "{miles}" AS miles
            FROM read_parquet({paths!r})
            WHERE "{miles}" > 0 AND "{miles}" < 100
        ),
        d AS (SELECT do_id AS locationid, count(*) AS drops,
                     sum(miles) AS vmt_terminating
              FROM src GROUP BY 1),
        p AS (SELECT pu_id AS locationid, count(*) AS picks
              FROM src GROUP BY 1)
        SELECT coalesce(d.locationid, p.locationid) AS locationid,
               coalesce(drops, 0) AS drops, coalesce(picks, 0) AS picks,
               coalesce(vmt_terminating, 0) AS vmt_terminating
        FROM d FULL OUTER JOIN p USING (locationid)
        """
        df = con.execute(sql).df()
        df["kind"] = kind
        out.append(df)
        print(f"  {kind}: {df.drops.sum():,.0f} drop-offs, "
              f"{df.vmt_terminating.sum():,.0f} mi", flush=True)
    con.close()

    v = pd.concat(out, ignore_index=True)
    v["locationid"] = pd.to_numeric(v.locationid, errors="coerce").astype("Int64")
    v = v.dropna(subset=["locationid"])
    v["mean_miles"] = v.vmt_terminating / v.drops.replace(0, np.nan)
    v.to_parquet(CACHE_VMT, index=False)
    return v


# ── build ───────────────────────────────────────────────────────────

def cells_to_zones(cells: pd.DataFrame) -> pd.DataFrame:
    """Assign every H3 cell to a taxi zone by centroid point-in-polygon.

    Centroids come from H3 itself rather than `features_h3.parquet`, so this
    stage can run straight after `mobility` without waiting for `features`.
    """
    import geopandas as gpd

    zones = fetch_zones()
    g = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells.lon, cells.lat), crs=4326)
    j = gpd.sjoin(g, zones[["locationid", "zone", "geometry"]],
                  how="left", predicate="within")
    j = j.drop(columns=["geometry", "index_right"])
    return pd.DataFrame(j)[["h3", "locationid", "zone"]]


def _shares(x: pd.Series) -> pd.Series:
    t = float(x.sum())
    return x / t if t > 0 else x * 0.0


def build() -> pd.DataFrame:
    agg = fetch_aggregate()
    vmt = fetch_zone_vmt()
    demand = pd.read_parquet(INTERIM / "demand_h3.parquet")
    cz = cells_to_zones(h3_centroids(demand.h3.tolist()))

    months = pd.to_datetime(pd.Series(TLC_CURRENT_MONTHS) + "-01")
    win = agg[agg.metric_month.isin(months)]

    # ── current magnitude, per zone ────────────────────────────────
    # counts come from the aggregate (authoritative, all industries);
    # mean trip miles come from the raw window (same months, same trips).
    mm = (vmt.assign(industry=vmt.kind.map(
              {v: k for k, v in INDUSTRY_FILE.items()}))
             [["locationid", "industry", "mean_miles"]])
    drops = (win[win.pickup_dropoff.str.startswith("Drop")]
             .groupby(["locationid", "industry"], as_index=False)
             .trip_count.sum())
    picks = (win[win.pickup_dropoff.str.startswith("Pick")]
             .groupby(["locationid", "industry"], as_index=False)
             .trip_count.sum())

    d = drops.merge(mm, on=["locationid", "industry"], how="left")
    # A zone with no raw miles falls back to that industry's city-wide mean.
    ind_mean = mm.groupby("industry").mean_miles.mean()
    d["mean_miles"] = d.mean_miles.fillna(d.industry.map(ind_mean))
    d["vmt"] = d.trip_count * d.mean_miles

    # TLC codes drop-offs it cannot place -- outside NYC, or unrecorded -- as
    # zones 264/265, which have no polygon. They are ~13% of current VMT and
    # are not geography, so they are removed from the share denominator rather
    # than counted as activity stranded somewhere real.
    real_zones = set(fetch_zones().locationid.dropna().astype(int))
    unloc = float(d.loc[~d.locationid.isin(real_zones), "vmt"].sum())
    d = d[d.locationid.isin(real_zones)]
    picks = picks[picks.locationid.isin(real_zones)]

    cur = (d.groupby("locationid", as_index=False)
             .agg(cur_vmt=("vmt", "sum"), cur_drops=("trip_count", "sum")))
    cur = cur.merge(picks.groupby("locationid", as_index=False)
                        .trip_count.sum().rename(columns={"trip_count": "cur_picks"}),
                    on="locationid", how="outer").fillna(0.0)
    hv = (d[d.industry == "FHV - High Volume"].groupby("locationid")
            .trip_count.sum())
    cur["hvfhv_share"] = (cur.locationid.map(hv).fillna(0.0)
                          / cur.cur_drops.replace(0, np.nan)).fillna(0.0)

    # ── 2015-16 baseline, same zones, from the surface's own source ─
    base = (demand.merge(cz, on="h3", how="inner")
                  .groupby("locationid", as_index=False)
                  .agg(base_vmt=("vmt_terminating", "sum"),
                       base_drops=("dropoffs", "sum"),
                       base_picks=("pickups", "sum")))

    z = base.merge(cur, on="locationid", how="outer").fillna(0.0)
    for a, b, name in (("cur_vmt", "base_vmt", "r_vmt"),
                       ("cur_drops", "base_drops", "r_drops"),
                       ("cur_picks", "base_picks", "r_picks")):
        sc, sb = _shares(z[a]), _shares(z[b])
        z[name] = np.where(sb > 0, sc / sb.replace(0, np.nan), np.nan)
    z["zone_growth"] = z.r_vmt

    # ── apply to the res-9 surface, conserving each total exactly ──
    out = demand.merge(cz, on="h3", how="left").merge(
        z[["locationid", "r_vmt", "r_drops", "r_picks", "zone_growth",
           "hvfhv_share"]], on="locationid", how="left")

    stranded = float(z.loc[z.base_vmt <= 0, "cur_vmt"].sum())
    for src, fac, dst in (("kwh_day", "r_vmt", "kwh_day_now"),
                          ("vmt_terminating", "r_vmt", "vmt_terminating_now"),
                          ("dropoffs", "r_drops", "dropoffs_now"),
                          ("pickups", "r_picks", "pickups_now")):
        # A zone with no 2015-16 taxi activity has no within-zone shape to
        # redistribute along, so its cells keep their own value (factor 1) and
        # the unplaceable current activity is reported as stranded rather than
        # smeared across ground the taxi feed never observed.
        f = out[fac].fillna(1.0).to_numpy(float)
        raw = out[src].to_numpy(float) * f
        tot_src, tot_raw = out[src].sum(), raw.sum()
        out[dst] = raw * (tot_src / tot_raw) if tot_raw > 0 else raw

    out["dwell_surplus_now"] = (out.dropoffs_now - out.pickups_now).clip(lower=0)

    keep = ["h3", "locationid", "zone", "zone_growth", "hvfhv_share",
            "r_vmt", "r_drops", "r_picks", "kwh_day_now",
            "vmt_terminating_now", "dropoffs_now", "pickups_now",
            "dwell_surplus_now"]
    res = out[keep].copy()

    # ── the count-vs-VMT cross-check the raw months were for ───────
    m = z[(z.base_vmt > 0) & (z.cur_vmt > 0)].copy()
    r_count = np.where(_shares(m.base_drops) > 0,
                       _shares(m.cur_drops) / _shares(m.base_drops), np.nan)
    ok = np.isfinite(r_count) & np.isfinite(m.r_vmt.to_numpy())
    pear = float(np.corrcoef(np.log(r_count[ok]),
                             np.log(m.r_vmt.to_numpy()[ok]))[0, 1])
    spear = float(pd.Series(r_count[ok]).rank()
                  .corr(pd.Series(m.r_vmt.to_numpy()[ok]).rank()))
    m = m.assign(r_count=r_count,
                 ratio_gap=np.abs(np.log(r_count / m.r_vmt.to_numpy())))
    worst = (m.nlargest(6, "ratio_gap")
              [["locationid", "r_count", "r_vmt", "cur_drops"]]
              .to_dict(orient="records"))

    # Zone polygons stop at the shoreline, so cells whose centroid sits in a
    # river or a park edge get no zone and keep factor 1.0. That is a quarter
    # of cells but ~1% of demand -- the same water-dilution pattern the grid
    # coverage shows, so it is reported weighted by demand, not by cell count.
    in_zone = out.locationid.notna()
    rep = {
        "window_current": TLC_CURRENT_MONTHS,
        "window_baseline": "2015-01, 2016-01..03 (yellow taxi, Q1)",
        "cells_total": int(len(out)),
        "cells_in_zone": int(in_zone.sum()),
        "demand_share_in_zone": round(
            float(out.loc[in_zone, "kwh_day"].sum()
                  / max(float(out.kwh_day.sum()), 1e-9)), 4),
        "zones_matched": int(m.shape[0]),
        "zones_no_baseline": int((z.base_vmt <= 0).sum()),
        "stranded_current_vmt_share": round(
            stranded / max(float(z.cur_vmt.sum()), 1e-9), 4),
        # Drop-offs TLC could not place to a zone at all (codes 264/265).
        "unlocatable_vmt_share": round(
            unloc / max(unloc + float(z.cur_vmt.sum()), 1e-9), 4),
        "count_vs_vmt_pearson_log": round(pear, 4),
        "count_vs_vmt_spearman": round(spear, 4),
        "worst_disagreements": worst,
        "hvfhv_share_of_current_dropoffs": round(
            float(cur.locationid.map(hv).fillna(0).sum()
                  / max(cur.cur_drops.sum(), 1e-9)), 4),
        "conservation_err": {
            dst: float(abs(res[dst].sum() - demand[src].sum())
                       / max(demand[src].sum(), 1e-9))
            for src, dst in (("kwh_day", "kwh_day_now"),
                             ("dropoffs", "dropoffs_now"),
                             ("pickups", "pickups_now"))},
    }
    (REPORTS / "rideshare.json").write_text(json.dumps(rep, indent=2))

    res.to_parquet(INTERIM / "zone_scale.parquet", index=False)
    z.to_parquet(INTERIM / "zone_activity.parquet", index=False)
    res.attrs.update(rep)
    return res


if __name__ == "__main__":
    r = build()
    a = r.attrs
    print(f"current window   : {', '.join(a['window_current'])}")
    print(f"baseline window  : {a['window_baseline']}")
    print(f"HVFHV share of current drop-offs : "
          f"{100*a['hvfhv_share_of_current_dropoffs']:.1f}%")
    print()
    print("conservation (rescaled total vs original total)")
    for k, v in a["conservation_err"].items():
        print(f"  {k:<22} relative error {v:.2e}")
        assert v < 1e-9, f"{k} does not conserve its total"
    print()
    print(f"cells inside a taxi zone : {a['cells_in_zone']:,} of "
          f"{a['cells_total']:,} ({100*a['cells_in_zone']/a['cells_total']:.1f}%), "
          f"holding {100*a['demand_share_in_zone']:.2f}% of taxi demand")
    print(f"zones matched            : {a['zones_matched']}")
    print(f"zones with no 2015-16 baseline : {a['zones_no_baseline']} "
          f"(holding {100*a['stranded_current_vmt_share']:.2f}% of current VMT, "
          f"left unredistributed)")
    print(f"drop-offs TLC could not place to a zone (codes 264/265) : "
          f"{100*a['unlocatable_vmt_share']:.2f}% of current VMT, excluded")
    print()
    print("count-based factor vs VMT-corrected factor")
    print(f"  Pearson (log)  : {a['count_vs_vmt_pearson_log']:.4f}")
    print(f"  Spearman       : {a['count_vs_vmt_spearman']:.4f}")
    print("  largest disagreements:")
    for w in a["worst_disagreements"]:
        print(f"    zone {int(w['locationid']):>3}  r_count {w['r_count']:>7.2f}  "
              f"r_vmt {w['r_vmt']:>7.2f}  ({w['cur_drops']:,.0f} drop-offs)")
    print()
    fp = PROCESSED / "features_h3.parquet"
    if not fp.exists():
        print("(features_h3 not built yet -- skipping the borough shift table)")
        raise SystemExit(0)
    boro_map = pd.read_parquet(fp, columns=["h3", "borough"])
    b = r.merge(boro_map, on="h3", how="inner")
    old = (pd.read_parquet(INTERIM / "demand_h3.parquet")
             .merge(boro_map, on="h3", how="inner")
             .groupby("borough").kwh_day.sum())
    new = b.groupby("borough").kwh_day_now.sum()
    print("borough share of demand, 2015-16 taxi -> Q1 2026 for-hire")
    for boro in sorted(set(old.index) & set(new.index)):
        o, n = old[boro] / old.sum(), new[boro] / new.sum()
        print(f"  {boro:<14} {100*o:>5.1f}%  ->  {100*n:>5.1f}%   "
              f"({'+' if n >= o else ''}{100*(n-o):.1f} pt)")
