"""NYC Automated Traffic Volume Counts -> H3 throughput / congestion features.

3.3 GB, 14,952 identified road segments, hourly directional counts 2000-2020.
This supplies two distinct things the taxi data cannot:
  * total-fleet throughput (taxis are a biased ~1-2% sample of vehicles), and
  * the congestion term -- siting a DC fast charger on an already-saturated
    corridor makes queueing worse, so corridor load enters as a penalty.

Geometry is published in EPSG:2263 (NY State Plane Long Island, US ft) and
must be reprojected before any H3 work.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ..config import TRAFFIC, INTERIM, H3_RES
from ..geo import statesplane_to_wgs84, to_h3, in_bbox

# Counts are 15-minute bins; sum to segment-hour first, then average, so a
# segment counted for many days is not double-weighted against a short study.
_SQL = """
WITH hourly AS (
    SELECT SegmentID AS seg, Boro AS boro, Yr AS yr, HH AS hh,
           any_value(WktGeom)   AS wkt,
           any_value(street)    AS street,
           any_value(fromSt)    AS fromst,
           any_value(toSt)      AS tost,
           any_value(Direction) AS dir,
           sum(Vol)             AS vol
    FROM read_csv_auto($path, ignore_errors=true)
    WHERE Vol IS NOT NULL AND Yr >= 2010
    GROUP BY SegmentID, Boro, Yr, M, D, HH
),
per_seg AS (
    SELECT seg,
           any_value(boro)   AS boro,
           any_value(wkt)    AS wkt,
           any_value(street) AS street,
           any_value(fromst) AS fromst,
           any_value(tost)   AS tost,
           count(DISTINCT dir) AS n_dirs,
           max(yr)           AS last_year,
           avg(vol)          AS mean_hourly_vol,
           quantile_cont(vol, 0.95) AS p95_hourly_vol,
           avg(CASE WHEN hh BETWEEN 7 AND 9 OR hh BETWEEN 16 AND 18
                    THEN vol END) AS peak_hourly_vol,
           avg(CASE WHEN hh BETWEEN 0 AND 5 THEN vol END) AS night_hourly_vol,
           count(*) AS n_obs
    FROM hourly GROUP BY seg
)
SELECT * FROM per_seg WHERE mean_hourly_vol > 0
"""


def build(threads: int = 4) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA memory_limit='3GB'")
    seg = con.execute(_SQL, {"path": str(TRAFFIC)}).df()
    con.close()

    xy = seg.wkt.str.extract(r"POINT \((?P<x>-?[\d.]+) (?P<y>-?[\d.]+)\)")
    x = pd.to_numeric(xy.x, errors="coerce")
    y = pd.to_numeric(xy.y, errors="coerce")
    ok = x.notna() & y.notna()
    seg = seg[ok].copy()
    lon, lat = statesplane_to_wgs84(x[ok].values, y[ok].values)
    seg["lon"], seg["lat"] = lon, lat
    seg = seg[in_bbox(seg.lat, seg.lon)].copy()

    seg["aadt_proxy"] = seg.mean_hourly_vol * 24.0
    # Peak-to-mean ratio is a congestion signature: a corridor whose peak is
    # far above its mean is already strained at exactly the hours drivers
    # would want to charge.
    seg["peak_ratio"] = seg.peak_hourly_vol / seg.mean_hourly_vol.replace(0, np.nan)
    seg["h3"] = to_h3(seg.lat.values, seg.lon.values, H3_RES)

    # ── street-network morphology ────────────────────────────────────
    # Road hierarchy from the name suffix. Arterials carry through-traffic and
    # suit en-route fast charging; local streets carry destination trips and
    # suit slower destination charging. The distinction is not in the volume
    # column, but it is in the name.
    up = seg.street.fillna("").str.upper()
    arterial = up.str.contains(
        r"BOULEVARD|BLVD|EXPRESSWAY|EXPY|PARKWAY|PKWY|HIGHWAY|TURNPIKE|BRIDGE"
        r"|TUNNEL|AVENUE OF|BROADWAY", regex=True)
    local = up.str.contains(r"PLACE|COURT|LANE|TERRACE|WALK|ALLEY", regex=True)
    seg["is_arterial"] = arterial.astype(float)
    seg["is_local"] = local.astype(float)
    # Intersection identity: an unordered cross-street pair per segment.
    seg["xing"] = [
        "|".join(sorted([str(a).upper().strip(), str(b).upper().strip()]))
        for a, b in zip(seg.fromst.fillna(""), seg.tost.fillna(""))
    ]

    agg = (seg.groupby("h3", as_index=False)
           .agg(aadt_proxy=("aadt_proxy", "sum"),
                mean_hourly_vol=("mean_hourly_vol", "mean"),
                peak_hourly_vol=("peak_hourly_vol", "mean"),
                night_hourly_vol=("night_hourly_vol", "mean"),
                p95_hourly_vol=("p95_hourly_vol", "max"),
                peak_ratio=("peak_ratio", "mean"),
                n_segments=("seg", "nunique"),
                n_intersections=("xing", "nunique"),
                n_streets=("street", "nunique"),
                arterial_share=("is_arterial", "mean"),
                local_share=("is_local", "mean"),
                dir_diversity=("n_dirs", "mean"),
                last_year=("last_year", "max")))
    agg.to_parquet(INTERIM / "traffic_h3.parquet", index=False)
    seg.to_parquet(INTERIM / "traffic_segments.parquet", index=False)
    return agg


if __name__ == "__main__":
    a = build()
    print(f"H3 cells with traffic counts : {len(a):,}")
    print(f"segments retained            : {a.n_segments.sum():,}")
    print(f"total AADT proxy             : {a.aadt_proxy.sum():,.0f} veh/day")
    print(f"median peak/mean ratio       : {a.peak_ratio.median():.2f}")
    print(f"intersections resolved       : {a.n_intersections.sum():,}")
    print(f"mean arterial share          : {a.arterial_share.mean():.3f}")
    print(a.nlargest(5, "aadt_proxy")[["h3", "aadt_proxy", "peak_ratio",
                                       "n_segments"]].to_string(index=False))
