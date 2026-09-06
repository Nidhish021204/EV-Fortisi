"""NYC taxi -> H3 demand surface.

These four vintages (2015-01, 2016-01..03) are the only NYC releases that
carry raw pickup/dropoff lat-lon; from 2016-07 the city publishes anonymised
zone IDs only. That is what makes a continuous sub-zone demand surface
possible at all.

~47.4M trips / 7.4 GB is streamed by DuckDB out-of-core: we aggregate to a
~55 m rounded lattice in SQL, then lift that lattice to H3 in pandas. The
rounding is finer than an H3 res-9 edge (174 m), so the lift is lossless
at the resolution we actually model on.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from ..config import TAXI_GLOB, INTERIM, NYC_BBOX, H3_RES, KWH_PER_MILE
from ..geo import to_h3

GRID = 0.0005  # degrees, ~55 m in latitude

_SQL = """
WITH src AS (
    SELECT
        pickup_latitude   AS plat, pickup_longitude  AS plon,
        dropoff_latitude  AS dlat, dropoff_longitude AS dlon,
        trip_distance     AS miles,
        hour(tpep_pickup_datetime) AS hr
    FROM read_csv_auto($glob, union_by_name=true, ignore_errors=true)
    WHERE trip_distance > 0 AND trip_distance < 100
),
pk AS (
    SELECT round(plat/{g})*{g} AS lat, round(plon/{g})*{g} AS lon,
           count(*) AS pickups,
           sum(CASE WHEN hr BETWEEN 7 AND 10 OR hr BETWEEN 16 AND 19
                    THEN 1 ELSE 0 END) AS pickups_peak
    FROM src
    WHERE plat BETWEEN {la0} AND {la1} AND plon BETWEEN {lo0} AND {lo1}
    GROUP BY 1,2
),
dp AS (
    SELECT round(dlat/{g})*{g} AS lat, round(dlon/{g})*{g} AS lon,
           count(*) AS dropoffs,
           sum(miles) AS vmt_terminating,
           avg(miles) AS mean_trip_miles
    FROM src
    WHERE dlat BETWEEN {la0} AND {la1} AND dlon BETWEEN {lo0} AND {lo1}
    GROUP BY 1,2
)
SELECT coalesce(pk.lat, dp.lat) AS lat,
       coalesce(pk.lon, dp.lon) AS lon,
       coalesce(pickups, 0)         AS pickups,
       coalesce(pickups_peak, 0)    AS pickups_peak,
       coalesce(dropoffs, 0)        AS dropoffs,
       coalesce(vmt_terminating, 0) AS vmt_terminating,
       mean_trip_miles
FROM pk FULL OUTER JOIN dp USING (lat, lon)
"""


def build(threads: int = 4) -> pd.DataFrame:
    b = NYC_BBOX
    sql = _SQL.format(g=GRID, la0=b["lat_min"], la1=b["lat_max"],
                      lo0=b["lon_min"], lo1=b["lon_max"])
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA memory_limit='3GB'")
    lattice = con.execute(sql, {"glob": TAXI_GLOB}).df()
    con.close()

    lattice["h3"] = to_h3(lattice.lat.values, lattice.lon.values, H3_RES)
    agg = (lattice.groupby("h3", as_index=False)
           .agg(pickups=("pickups", "sum"),
                pickups_peak=("pickups_peak", "sum"),
                dropoffs=("dropoffs", "sum"),
                vmt_terminating=("vmt_terminating", "sum"),
                mean_trip_miles=("mean_trip_miles", "mean")))

    # Dwell proxy: vehicles that arrive and do not immediately leave. This is
    # the signal that distinguishes "a place cars pass through" from "a place
    # cars rest", and resting is when charging can happen.
    agg["dwell_surplus"] = (agg.dropoffs - agg.pickups).clip(lower=0)
    agg["flow_imbalance"] = ((agg.dropoffs - agg.pickups)
                             / (agg.dropoffs + agg.pickups).replace(0, 1))
    # Physically-interpretable target: daily kWh implied by terminating VMT.
    agg["kwh_day"] = agg.vmt_terminating * KWH_PER_MILE / 120.0  # 4 months ~ 120 d
    agg.to_parquet(INTERIM / "demand_h3.parquet", index=False)
    return agg


if __name__ == "__main__":
    d = build()
    print(f"H3 res-9 cells with taxi activity : {len(d):,}")
    print(f"total pickups   : {d.pickups.sum():,.0f}")
    print(f"total dropoffs  : {d.dropoffs.sum():,.0f}")
    print(f"terminating VMT : {d.vmt_terminating.sum():,.0f} mi")
    print(f"implied demand  : {d.kwh_day.sum():,.0f} kWh/day fleet-wide")
    print(d.nlargest(5, "kwh_day")[["h3", "dropoffs", "kwh_day", "dwell_surplus"]]
          .to_string(index=False))
