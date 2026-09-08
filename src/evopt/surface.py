"""Composite demand surface consumed by the optimiser.

Why this is a measurement, not a prediction
-------------------------------------------
Spatial-block validation of the learned demand model (see
reports/demand_metrics.json) showed it capturing 20.0% of true energy in its
top decile against 48.5% for a trivial AADT ranking and an 89.3% oracle -- i.e.
it does not beat sorting by traffic volume once spatial leakage is removed.
Predicting a quantity we have already measured across 47.4M trips would be
laundering a measurement through a weaker estimator, so for NYC the surface is
built from observation:

    taxi term      direct kWh from terminating VMT -- energy that vehicles
                   arriving at this cell must replenish (measured)
    fleet term     AADT x *cell traversal length* x kWh/mi -- energy the
                   non-taxi fleet consumes crossing the cell
    dwell weight   uplift where vehicles rest rather than pass through

The traversal length matters: a segment AADT counts vehicles passing a point,
so multiplying it by whole-trip length would charge every one of the ~20 cells
a trip crosses for that trip's entire energy. Energy consumed *within* a cell
is throughput x the distance across the cell.

The learned model is retained for transfer to cities with no taxi feed, and is
reported with its honest performance rather than its flattering one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (PROCESSED, INTERIM, KWH_PER_MILE, L2_KERNEL_M,
                     DCFC_KERNEL_M, TRAIN_CUTOFF)
from .geo import haversine_m, gravity_kernel

# Yellow taxis are a small, non-uniform slice of the vehicle fleet. The fleet
# term is therefore anchored on traffic counts, with EV share held explicit so
# a planner can vary the adoption assumption directly.
EV_SHARE = 0.08          # modelled fleet EV penetration
DWELL_UPLIFT = 0.35      # weight on resting vehicles vs through traffic
# H3 res-9 mean area is 0.105 km2; a straight crossing is ~sqrt(area).
CELL_TRAVERSE_MI = 0.201


def _served_capacity(cells: pd.DataFrame, demand: np.ndarray,
                     cutoff: str = TRAIN_CUTOFF) -> tuple[np.ndarray, float]:
    """Deliverable kWh/day reaching each cell. Capacity is CONSERVED and
    routed to DEMAND.

    Two properties matter and each fixes a distinct artefact.

    Conservation. A station's energy cannot be spent twice, so weights are
    normalised per station: each distributes exactly its own daily capacity.
    Without this, city-wide coverage inflates by an order of magnitude because
    one charger is counted once per cell it can reach.

    Demand weighting. Reachability alone routes energy into empty cells, which
    produced coverage ratios of 17,000% for cells with essentially no demand
    sitting near a charger. Energy flows to where the demand is, so the weight
    is kernel x demand. A station with no demand in range now allocates
    nothing, and that unallocated energy is reported as stranded capacity
    rather than silently credited to empty ground.
    """
    st = pd.read_parquet(INTERIM / "stations_nyc.parquet")
    st = st[st.open_date <= pd.Timestamp(cutoff)]
    if st.empty:
        return np.zeros(len(cells)), 0.0

    d = haversine_m(cells.lat.to_numpy()[:, None], cells.lon.to_numpy()[:, None],
                    st.lat.to_numpy()[None, :], st.lon.to_numpy()[None, :])

    # Daily deliverable energy per station, by port class and duty cycle.
    cap_l2 = (st.l1.to_numpy() * 1.4 + st.l2.to_numpy() * 7.2) * 8.0
    cap_dc = st.dcfc.to_numpy() * 50.0 * 4.0

    dem = np.maximum(np.asarray(demand, float), 0.0)[:, None]
    out = np.zeros(len(cells))
    stranded = 0.0
    for kernel_m, cap in ((L2_KERNEL_M, cap_l2), (DCFC_KERNEL_M, cap_dc)):
        w = gravity_kernel(d, kernel_m) * dem        # reachable AND wanted
        colsum = w.sum(axis=0)
        reachable = colsum > 0
        # A station with no demand in range delivers nothing to anyone.
        stranded += float(cap[~reachable].sum())
        safe = np.where(reachable, colsum, 1.0)
        out += (w / safe) @ np.where(reachable, cap, 0.0)
    return out, stranded


def build(ev_share: float = EV_SHARE, current: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "features_h3.parquet")

    # Stage 8b: re-weight the measured terms to the current for-hire era.
    # HVFHV records carry no coordinates, so this changes how much of the
    # city's activity each taxi *zone* holds, not where activity sits inside
    # a zone -- that shape is still 2015-16 and stays so. The fleet term is
    # left alone: it comes from traffic counts, a different source on its own
    # vintage, and rescaling it by a for-hire factor would be a category error.
    zs = INTERIM / "zone_scale.parquet"
    if current and zs.exists():
        z = pd.read_parquet(zs)[["h3", "kwh_day_now", "dwell_surplus_now",
                                 "zone_growth"]]
        df = df.merge(z, on="h3", how="left")
        df.attrs["demand_era"] = "current"
    else:
        df["zone_growth"] = np.nan
        df.attrs["demand_era"] = "2015-16"
    # Cells outside the taxi feed have no zone factor and no taxi demand to
    # rescale; falling back to their own value is a no-op for them.
    for now, was in (("kwh_day_now", "kwh_day"),
                     ("dwell_surplus_now", "dwell_surplus")):
        df[now] = df.get(now, pd.Series(np.nan, index=df.index)).fillna(df[was])

    taxi_kwh = df.kwh_day_now.to_numpy(float)

    # Local mean trip length, falling back to the borough median where the
    # taxi sample in a cell is too thin to trust. Used only for the dwell
    # term, where a whole arriving trip's energy is genuinely at stake.
    mtm = df.mean_trip_miles.replace(0, np.nan)
    mtm = mtm.fillna(df.groupby("borough").mean_trip_miles
                     .transform(lambda s: s[s > 0].median()))
    mtm = mtm.fillna(mtm.median()).to_numpy(float)

    fleet_kwh = (df.aadt_proxy.to_numpy(float)
                 * CELL_TRAVERSE_MI * KWH_PER_MILE * ev_share)

    dwell = df.dwell_surplus_now.to_numpy(float)
    dwell_kwh = dwell * KWH_PER_MILE * mtm * DWELL_UPLIFT / 120.0

    df["demand_kwh"] = taxi_kwh + fleet_kwh + dwell_kwh
    df["demand_taxi_kwh"] = taxi_kwh
    df["demand_fleet_kwh"] = fleet_kwh

    # The 2015-16 surface, kept alongside the current one.
    #
    # This is a temporal-integrity requirement, not a convenience. The
    # suitability model fits on the network as of TRAIN_CUTOFF and is tested on
    # what got built in 2019-21, and it takes `demand_kwh` and `unmet_kwh` as
    # features. The re-weighting above is derived from Q1 2026 activity, so
    # feeding the current surface into that model would put post-cutoff
    # information into a pre-cutoff fit. The optimiser plans against the
    # current surface; the forward test only ever sees the historical one.
    hist_kwh = (df.kwh_day.to_numpy(float) + fleet_kwh
                + df.dwell_surplus.to_numpy(float)
                * KWH_PER_MILE * mtm * DWELL_UPLIFT / 120.0)
    df["demand_kwh_hist"] = hist_kwh
    df["demand_taxi_kwh_hist"] = df.kwh_day.to_numpy(float)

    # ── served capacity vs demand -> the gap ────────────────────────
    served_arr, stranded = _served_capacity(df, df.demand_kwh.to_numpy(float))
    df["served_kwh"] = served_arr
    df.attrs["stranded_kwh"] = stranded
    served = df.served_kwh.to_numpy(float)
    df["unmet_kwh"] = np.clip(df.demand_kwh - served, 0, None)

    # Capacity is routed to demand, so the historical gap needs its own
    # routing rather than reusing the current one.
    served_hist, _ = _served_capacity(df, hist_kwh)
    df["served_kwh_hist"] = served_hist
    df["unmet_kwh_hist"] = np.clip(hist_kwh - served_hist, 0, None)
    # Guard the ratio: below ~1 kWh/day a cell's demand is numerical dust and
    # the ratio is meaningless, so it is reported as undefined rather than as
    # a spectacular percentage.
    ratio = np.where(df.demand_kwh > 1.0, served / np.maximum(df.demand_kwh, 1e-9),
                     np.nan)
    df["service_ratio"] = np.clip(ratio, 0, 10)

    df.to_parquet(PROCESSED / "surface_h3.parquet", index=False)
    return df


if __name__ == "__main__":
    d = build()
    tot = d.demand_kwh.sum()
    print(f"demand era       : {d.attrs.get('demand_era', '2015-16')}"
          + ("  (for-hire terms re-weighted to Q1 2026 zone shares)"
             if d.attrs.get("demand_era") == "current" else ""))
    print(f"composite demand : {tot:,.0f} kWh/day "
          f"(taxi {d.demand_taxi_kwh.sum()/tot:.0%}, "
          f"fleet {d.demand_fleet_kwh.sum()/tot:.0%})")
    print(f"served capacity  : {d.served_kwh.sum():,.0f} kWh/day")
    print(f"unmet demand     : {d.unmet_kwh.sum():,.0f} kWh/day "
          f"({d.unmet_kwh.sum()/tot:.0%} of demand)")
    print(f"stranded capacity: {d.attrs.get('stranded_kwh', 0):,.0f} kWh/day "
          f"(stations with no demand in range)")
    print(f"cells fully unserved (ratio<0.1) : {(d.service_ratio < 0.1).sum():,}")
    print()
    g = d.groupby("borough").agg(demand=("demand_kwh", "sum"),
                                 served=("served_kwh", "sum"),
                                 unmet=("unmet_kwh", "sum"))
    g["cover_%"] = 100 * g.served / g.demand
    print(g.round(0).to_string())
