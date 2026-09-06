"""Assemble the H3 feature matrix: demand x traffic x supply x spatial lag.

Design notes
------------
Land mask.  No external boundary shapefile is required: a cell that recorded
zero taxi activity and zero traffic counts across four months is water, park
interior, or otherwise unreachable. The observed data *is* the land mask.

Borough assignment.  The traffic table carries an authoritative `Boro` label
per segment, and segments exist only inside the five boroughs. Assigning each
hex the borough of its nearest labelled segment (within a 2 km cut-off) both
labels the cell and drops the New Jersey collar that the bounding box admits.

Supply is not binary.  A station 300 m away serves a cell; one 3 km away does
not. Accessible supply is accumulated through a Gaussian gravity kernel with
a separate bandwidth per charger class (see config).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (INTERIM, PROCESSED, H3_RES, L2_KERNEL_M, DCFC_KERNEL_M,
                     TRAIN_CUTOFF)
from .geo import h3_centroids, haversine_m, gravity_kernel, kring_aggregate

DEMAND_COLS = ["pickups", "pickups_peak", "dropoffs", "vmt_terminating",
               "kwh_day", "dwell_surplus", "flow_imbalance", "mean_trip_miles"]
TRAFFIC_COLS = ["aadt_proxy", "mean_hourly_vol", "peak_hourly_vol",
                "night_hourly_vol", "p95_hourly_vol", "peak_ratio", "n_segments",
                "n_intersections", "n_streets", "arterial_share", "local_share",
                "dir_diversity"]
# Built environment from PLUTO tax lots. Absent for water, park interiors and
# other lot-free cells, where zero is the correct value rather than a gap.
LANDUSE_COLS = [
    "lots_n", "lot_area", "bldg_area", "res_area", "com_area", "office_area",
    "retail_area", "garage_area", "factory_area", "storage_area",
    "units_res", "units_total", "floors_mean", "far_mean", "year_built_med",
    "assess_total", "lu_entropy", "far_built", "jobs_area", "parking_share",
    "assess_per_sqft", "res_density",
] + [f"lu_{k}_sh" for k in
     ["res_low", "res_mid", "res_high", "mixed", "commercial", "industrial",
      "transport", "institutional", "openspace", "parking", "vacant"]]
# ACS vehicle availability, allocated tract -> cell by residential units.
# `no_home_charge` is the derived hypothesis: cars whose owners rent and have
# no off-street space, i.e. the ones that must charge in public.
ACS_COLS = ['hh_total', 'zero_veh_hh', 'veh_total', 'renter_veh_hh', 'owner_hh', 'renter_hh', 'veh_per_hh', 'zero_veh_share', 'renter_share', 'home_charge_cov', 'no_home_charge', 'no_home_charge_density']


def _supply_at(cells: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Gravity-weighted accessible supply + distance-to-nearest, per hex.

    Two variants are produced. The plain columns include every station. The
    `_ex` columns exclude stations sited in the cell being scored, and those
    are the ones the suitability model may use.

    Why: a cell that holds a station has dist_station_m ~ 0 and non-zero
    access by construction, so a classifier trained on the plain columns can
    recover the label from its own footprint. That is target leakage -- it
    drives Elkan-Noto c to ~1.0 and makes the model an incumbency detector
    rather than a siting model. The `_ex` columns ask the honest question:
    what does the *neighbourhood* supply look like?
    """
    if stations.empty:
        z = np.zeros(len(cells))
        far = np.full(len(cells), 50_000.0)
        return pd.DataFrame({"access_l2": z, "access_dcfc": z,
                             "dist_station_m": far, "dist_dcfc_m": far,
                             "access_l2_ex": z, "access_dcfc_ex": z,
                             "dist_station_ex_m": far, "dist_dcfc_ex_m": far},
                            index=cells.index)

    clat = cells.lat.to_numpy()[:, None]
    clon = cells.lon.to_numpy()[:, None]
    slat = stations.lat.to_numpy()[None, :]
    slon = stations.lon.to_numpy()[None, :]
    d = haversine_m(clat, clon, slat, slon)          # (n_cells, n_stations)

    l2_ports = stations.l2.to_numpy()[None, :]
    dc_ports = stations.dcfc.to_numpy()[None, :]

    access_l2 = (l2_ports * gravity_kernel(d, L2_KERNEL_M)).sum(axis=1)
    access_dc = (dc_ports * gravity_kernel(d, DCFC_KERNEL_M)).sum(axis=1)

    dist_any = d.min(axis=1)
    has_dc = stations.dcfc.to_numpy() > 0
    far = np.full(len(cells), 50_000.0)
    dist_dc = d[:, has_dc].min(axis=1) if has_dc.any() else far

    # Leave-one-cell-out variant: blank any station sharing the cell's H3 id.
    same = (cells.h3.to_numpy()[:, None] == stations.h3.to_numpy()[None, :])
    d_ex = np.where(same, np.inf, d)
    k_l2_ex = np.where(same, 0.0, gravity_kernel(d, L2_KERNEL_M))
    k_dc_ex = np.where(same, 0.0, gravity_kernel(d, DCFC_KERNEL_M))
    access_l2_ex = (l2_ports * k_l2_ex).sum(axis=1)
    access_dc_ex = (dc_ports * k_dc_ex).sum(axis=1)
    dist_any_ex = np.minimum(d_ex.min(axis=1), 50_000.0)
    dist_dc_ex = (np.minimum(d_ex[:, has_dc].min(axis=1), 50_000.0)
                  if has_dc.any() else far)

    return pd.DataFrame({"access_l2": access_l2, "access_dcfc": access_dc,
                         "dist_station_m": dist_any, "dist_dcfc_m": dist_dc,
                         "access_l2_ex": access_l2_ex,
                         "access_dcfc_ex": access_dc_ex,
                         "dist_station_ex_m": dist_any_ex,
                         "dist_dcfc_ex_m": dist_dc_ex},
                        index=cells.index)


def build(cutoff: str = TRAIN_CUTOFF) -> pd.DataFrame:
    demand = pd.read_parquet(INTERIM / "demand_h3.parquet")
    traffic = pd.read_parquet(INTERIM / "traffic_h3.parquet")
    landuse = pd.read_parquet(INTERIM / "landuse_h3.parquet")
    acs = pd.read_parquet(INTERIM / "acs_h3.parquet")
    seg = pd.read_parquet(INTERIM / "traffic_segments.parquet")
    st = pd.read_parquet(INTERIM / "stations_nyc.parquet")

    # ── universe: any cell with observed activity ───────────────────
    cells = pd.Index(sorted(set(demand.h3) | set(traffic.h3) | set(st.h3)
                            | set(landuse.h3) | set(acs.h3)), name="h3")
    df = h3_centroids(cells)
    df = (df.merge(demand, on="h3", how="left")
            .merge(traffic, on="h3", how="left")
            .merge(landuse, on="h3", how="left")
            .merge(acs, on="h3", how="left"))
    for c in DEMAND_COLS + TRAFFIC_COLS + LANDUSE_COLS + ACS_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    df["last_year"] = df["last_year"].fillna(0)

    # ── borough via nearest labelled traffic segment (also drops NJ) ─
    d_seg = haversine_m(df.lat.to_numpy()[:, None], df.lon.to_numpy()[:, None],
                        seg.lat.to_numpy()[None, :], seg.lon.to_numpy()[None, :])
    nearest = d_seg.argmin(axis=1)
    df["borough"] = seg.boro.to_numpy()[nearest]
    df["dist_to_road_m"] = d_seg.min(axis=1)
    df = df[df.dist_to_road_m <= 2000].copy()

    # ── supply as of the cutoff (temporal integrity) ────────────────
    st_t = st[st.open_date <= pd.Timestamp(cutoff)]
    df = df.join(_supply_at(df, st_t))

    by_cell = st_t.groupby("h3").agg(n_stations=("name", "size"),
                                     ports=("ports", "sum"),
                                     kw=("kw", "sum"),
                                     dcfc=("dcfc", "sum"),
                                     l2=("l2", "sum"))
    df = df.merge(by_cell.reset_index(), on="h3", how="left").fillna(
        {"n_stations": 0, "ports": 0, "kw": 0, "dcfc": 0, "l2": 0})

    # ── spatial lag: what surrounds a site drives its value ─────────
    df = df.reset_index(drop=True)
    lag_cols = ["kwh_day", "dropoffs", "aadt_proxy", "dwell_surplus", "ports",
                "peak_hourly_vol", "night_hourly_vol", "n_segments",
                "peak_ratio", "mean_hourly_vol",
                # Built environment spills across cell edges: the shop you
                # drive to is rarely in the same 174 m hexagon you park in.
                "units_res", "jobs_area", "garage_area", "far_built",
                "lu_entropy", "n_intersections", "arterial_share",
                "no_home_charge", "veh_per_hh"]
    for k in (1, 2):
        df = pd.concat([df, kring_aggregate(df, lag_cols, k=k)], axis=1)

    # ── labels ──────────────────────────────────────────────────────
    df["y_now"] = (df.n_stations > 0).astype(int)          # positives at cutoff
    future = st[st.open_date.between(*map(pd.Timestamp,
                                          ("2019-01-01", "2021-12-31")))]
    df["y_future"] = df.h3.isin(set(future.h3)).astype(int)  # forward-test target

    df.to_parquet(PROCESSED / "features_h3.parquet", index=False)
    return df


if __name__ == "__main__":
    f = build()
    print(f"feature matrix : {f.shape[0]:,} hexes x {f.shape[1]} cols")
    print(f"cells with tax lots : {(f.lots_n > 0).sum():,} "
          f"({100*(f.lots_n > 0).mean():.1f}%)   "
          f"units_res {f.units_res.sum():,.0f}")
    print(f"positives @cutoff (y_now)    : {f.y_now.sum():,} "
          f"({100*f.y_now.mean():.2f}%)")
    print(f"forward targets  (y_future)  : {f.y_future.sum():,} "
          f"({100*f.y_future.mean():.2f}%)")
    print("\nby borough:")
    print(f.groupby("borough").agg(hexes=("h3", "size"),
                                   kwh_day=("kwh_day", "sum"),
                                   ports=("ports", "sum"),
                                   built_2019_21=("y_future", "sum")).to_string())
