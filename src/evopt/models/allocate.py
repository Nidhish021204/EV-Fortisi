"""Stage 4 - allocation: capacitated maximal covering with congestion.

Ranking cells is not planning. The top-k scoring cells sit on top of each other
and cover the same demand three times over, so choosing a *set* is a
combinatorial problem, not a sort.

Coverage under a radius is monotone submodular, so lazy-forward greedy carries
a (1 - 1/e) ~ 63% worst-case guarantee and runs in milliseconds -- that is what
drives the dashboard's live budget slider.

Two constraints do the real work:

  Capacity, not just radius.  A site captures min(demand in radius, what its
  ports can physically deliver). Without this the optimiser hands one cell
  37 MWh/day and "solves" Manhattan with a single station that would need
  76 ports -- an artefact, not a plan. Real hubs are 4-20 ports, so capture is
  bounded by MAX_PORTS x PORT_KW x ACTIVE_HOURS x RHO_MAX and the surplus
  stays on the table for the next site.

  Congestion, modelled.  Each site is an M/M/c queue sized so utilisation
  stays under RHO_MAX, which by construction keeps rho < 1 and the queue
  stable. A plan therefore states an expected wait instead of hoping for one.

An equity floor guarantees every borough a minimum allocation, so the optimiser
cannot answer "where is demand densest" with "all of it in Manhattan".

Grid capacity binds too, and at the feeder.  Con Edison publishes a per-feeder
limit on new charging load (`STORAGE_MAX_CHARGE`), and that -- not the
substation bank rating -- is what a site actually draws against. City-wide the
feeder limits sum to 533 MW against 1,686 MW of bank headroom, so planning
against the bank was 3.2x too permissive; a fifth of feeders publish zero.

Both limits are enforced, because both are real: a site consumes its feeder's
charging allowance *and* its bank's headroom, and whichever runs out first
stops it. Rather than ban sites outright, the grid *shrinks* them -- a hub gets
as many ports as its feeder can still carry, which is what happens when a
utility reviews an interconnection. Anything with under one port of room left
is excluded outright, which is also how feeders on Con Edison's explicit
"no charging capacity" layer drop out: their limit is a published zero.
"""
from __future__ import annotations

import json
from math import ceil, factorial

import numpy as np
import pandas as pd

from ..config import PROCESSED, REPORTS, INTERIM, MEAN_SESSION_KWH
from ..geo import haversine_m

COVER_RADIUS_M = 800.0     # walk-to-destination service radius
MIN_SPACING_M = 400.0      # anti-clustering: no two new sites closer than this
RHO_MAX = 0.70             # utilisation ceiling -> the congestion constraint
ACTIVE_HOURS = 14.0        # hours/day over which sessions arrive
PORT_KW = 50.0             # DC fast port assumed for new build
MAX_PORTS = 12             # largest hub we will site in one cell
CAPEX_PER_PORT = 45_000.0  # USD, installed

# Most energy one site can deliver per day while staying under RHO_MAX.
SITE_CAPACITY_KWH = MAX_PORTS * PORT_KW * ACTIVE_HOURS * RHO_MAX


def erlang_c(c: int, a: float) -> float:
    """Probability an arrival waits, for M/M/c with offered load a = lambda/mu."""
    if a >= c:
        return 1.0
    s = sum(a ** k / factorial(k) for k in range(c))
    last = a ** c / (factorial(c) * (1 - a / c))
    return last / (s + last)


def size_ports(kwh_day: float) -> dict:
    """Port count holding utilisation <= RHO_MAX, with the resulting wait.

    Capture is capped at SITE_CAPACITY_KWH upstream, so rho <= RHO_MAX < 1 here
    and the queue is always stable -- no infinite-wait branch can arise.
    """
    mu = PORT_KW / MEAN_SESSION_KWH                  # sessions/hour per port
    lam = kwh_day / MEAN_SESSION_KWH / ACTIVE_HOURS  # sessions/hour
    ports = max(1, min(MAX_PORTS, ceil(lam / (mu * RHO_MAX))))
    a = lam / mu
    rho = a / ports
    pw = erlang_c(ports, a)
    wq_hours = pw / max(ports * mu - lam, 1e-9)
    return {"ports": int(ports), "rho": round(float(rho), 3),
            "p_wait": round(float(pw), 4),
            "wait_min": round(float(wq_hours * 60.0), 2),
            "sessions_day": round(float(lam * ACTIVE_HOURS), 1)}


def load_grid() -> pd.DataFrame:
    """h3 -> serving feeder and bank, with the capacity each still has in kW."""
    f = INTERIM / "feeders_h3.parquet"
    cols = ["h3", "feeder_id", "substation", "headroom_kw",
            "bank_headroom_kw", "grid_state", "charge_excluded",
            "dist_to_feeder_m"]
    if not f.exists():
        return pd.DataFrame(columns=cols)
    g = pd.read_parquet(f)
    return g[[c for c in cols if c in g.columns]]


def greedy_mclp(cells: pd.DataFrame, k: int, radius_m: float = COVER_RADIUS_M,
                min_spacing_m: float = MIN_SPACING_M,
                min_per_borough: dict | None = None,
                use_grid: bool = True) -> list[dict]:
    """Lazy-forward greedy over marginal covered demand, capacity-limited.

    Weight is unmet demand scaled by modelled viability: the optimiser chases
    demand the current network fails to serve *and* that a station could
    plausibly be built to serve.
    """
    lat, lon = cells.lat.to_numpy(), cells.lon.to_numpy()
    boro = cells.borough.to_numpy()
    w = cells.unmet_kwh.to_numpy() * cells.p_viable.to_numpy()

    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    covers = d <= radius_m

    remaining = w.copy()
    chosen: list[int] = []
    out: list[dict] = []
    per_boro: dict[str, int] = {}
    min_per_borough = min_per_borough or {}

    # Grid state: two shared ledgers, because a site draws on its feeder's
    # published charging allowance *and* on its substation bank's headroom.
    # The feeder is the tighter of the two almost everywhere, but the bank is
    # still real, so whichever exhausts first is what stops the site.
    def _keys(col: str) -> np.ndarray:
        if use_grid and col in cells:
            return cells[col].where(cells[col].notna(), None).to_numpy()
        return np.array([None] * len(cells), dtype=object)

    def _caps(col: str) -> np.ndarray:
        if use_grid and col in cells:
            return cells[col].to_numpy(float)
        return np.full(len(cells), np.inf)

    fdr, sub = _keys("feeder_id"), _keys("substation")
    head_f, head_b = _caps("headroom_kw"), _caps("bank_headroom_kw")
    used_f: dict[object, float] = {}
    used_b: dict[object, float] = {}

    def _left(key, head, used, i: int) -> float:
        # A missing key or a NaN cap means Con Edison publishes nothing here,
        # which is not the same as "no capacity here" -- read as unconstrained.
        if key[i] is None or not np.isfinite(head[i]):
            return float("inf")
        return head[i] - used.get(key[i], 0.0)

    def head_left(i: int) -> float:
        if not use_grid:
            return float("inf")
        return min(_left(fdr, head_f, used_f, i),
                   _left(sub, head_b, used_b, i))

    # Lazy greedy: gains are upper bounds that only shrink, so a candidate
    # still on top after refresh is exact.
    gains = covers.astype(float) @ remaining
    fresh = np.zeros(len(cells), dtype=bool)

    while len(chosen) < k:
        slots_left = k - len(chosen)
        deficit = {b: n - per_boro.get(b, 0) for b, n in min_per_borough.items()
                   if n - per_boro.get(b, 0) > 0}
        # When remaining slots exactly cover the outstanding floor, restrict
        # the pool to boroughs still short -- that is what makes it a floor.
        forced = sum(deficit.values()) >= slots_left and bool(deficit)

        eligible = []
        for i in range(len(cells)):
            if i in chosen:
                continue
            if chosen and d[i, chosen].min() < min_spacing_m:
                continue
            if forced and boro[i] not in deficit:
                continue
            # A bank without room for even one port cannot host anything.
            if head_left(i) < PORT_KW:
                continue
            eligible.append(i)
        if not eligible:
            break

        fresh[:] = False
        best, best_gain = None, -1.0
        for i in sorted(eligible, key=lambda j: -gains[j]):
            if not fresh[i]:
                gains[i] = covers[i] @ remaining
                fresh[i] = True
            if gains[i] > best_gain:
                best, best_gain = i, gains[i]
            # Everything after this point has a smaller upper bound already.
            if best is not None and best_gain >= gains[i]:
                nxt = [j for j in eligible if not fresh[j]]
                if not nxt or best_gain >= max(gains[j] for j in nxt):
                    break
        if best is None or best_gain <= 0:
            break

        available = covers[best] * remaining
        total_avail = float(available.sum())

        # The grid caps how many ports this bank can still carry, which caps
        # the energy the site can deliver. Unconstrained capacity is the
        # 12-port hub; the binding limit is whichever is smaller.
        hl = head_left(best)
        grid_ports = MAX_PORTS if np.isinf(hl) else int(hl // PORT_KW)
        grid_ports = max(0, min(MAX_PORTS, grid_ports))
        cap_here = grid_ports * PORT_KW * ACTIVE_HOURS * RHO_MAX
        captured = min(total_avail, SITE_CAPACITY_KWH, cap_here)
        if total_avail > 0 and captured > 0:
            remaining = remaining - available * (captured / total_avail)

        chosen.append(best)
        per_boro[boro[best]] = per_boro.get(boro[best], 0) + 1

        row = cells.iloc[best]
        sizing = size_ports(captured)
        draw = sizing["ports"] * PORT_KW
        if fdr[best] is not None:
            used_f[fdr[best]] = used_f.get(fdr[best], 0.0) + draw
        if sub[best] is not None:
            used_b[sub[best]] = used_b.get(sub[best], 0.0) + draw
        out.append({
            "rank": len(chosen), "h3": row.h3,
            "lat": float(row.lat), "lon": float(row.lon),
            "borough": str(row.borough),
            "captured_kwh_day": round(captured, 1),
            "demand_kwh_day": round(float(row.demand_kwh), 1),
            "p_viable": round(float(row.p_viable), 4),
            "cells_covered": int(covers[best].sum()),
            "saturated": bool(total_avail > SITE_CAPACITY_KWH),
            "feeder_id": (None if fdr[best] is None else str(fdr[best])),
            "substation": (None if sub[best] is None else str(sub[best])),
            "headroom_kw_before": (None if np.isinf(hl) else round(float(hl), 1)),
            "grid_limited": bool(not np.isinf(hl) and cap_here < min(
                total_avail, SITE_CAPACITY_KWH)),
            "capex_usd": sizing["ports"] * CAPEX_PER_PORT,
            **sizing,
        })
    return out


def grid_sweep(cand: pd.DataFrame, min_per_borough: dict,
               k_max: int = 600) -> dict:
    """At what budget does the grid become the binding limit?

    "The constraint does not bind at k=120" is true but useless on its own. A
    planner needs to know how far the programme can grow before Con Edison,
    not demand, is what stops it.

    Each budget gets its own pair of runs. Greedy is nested only while every
    pick is unconstrained: the equity floor's forcing rule triggers on
    `sum(deficit) >= slots_left`, which depends on the budget, so the first
    120 picks of a k=600 run are *not* the greedy solution for k=120. Taking
    cumulative sums over one long run was cheaper but reported a different
    number at k=120 than `run()` did for the same instance.
    """
    ks = (120, 200, 300, 400, 500, 600)
    pts, con_max, free_max = [], None, None
    for k in ks:
        con = pd.DataFrame(greedy_mclp(cand, k=k,
                                       min_per_borough=min_per_borough))
        free = pd.DataFrame(greedy_mclp(cand, k=k,
                                        min_per_borough=min_per_borough,
                                        use_grid=False))
        if k == ks[-1]:
            con_max, free_max = con, free
        c, f = float(con.captured_kwh_day.sum()), float(free.captured_kwh_day.sum())
        pts.append({"k": k,
                    "constrained_kwh_day": round(c, 1),
                    "unconstrained_kwh_day": round(f, 1),
                    "cost_pct": round(100 * (f - c) / max(f, 1e-9), 2),
                    "sites_available_constrained": int(len(con)),
                    "port_limited": int(con.grid_limited.sum())
                    if "grid_limited" in con else 0})
    return {"k_max": k_max,
            "sites_before_grid_exhausts": int(len(con_max)),
            "sites_without_grid_limit": int(len(free_max)),
            "curve": pts}


def run(k: int = 120, min_per_borough: dict | None = None) -> dict:
    df = pd.read_parquet(PROCESSED / "surface_h3.parquet")
    su = pd.read_parquet(PROCESSED / "suitability.parquet")
    df = df.merge(su, on="h3", how="left")
    df["p_viable"] = df.p_viable.fillna(0.0)

    # Accessibility mandate: no borough left below a usable minimum.
    if min_per_borough is None:
        min_per_borough = {"Bronx": 18, "Staten Island": 12, "Brooklyn": 22,
                           "Queens": 22}

    grid = load_grid()
    if len(grid):
        df = df.merge(grid, on="h3", how="left")
        # NaN headroom means "Con Edison publishes nothing here", which is not
        # the same as "no capacity here". Leave it NaN so head_left() reads it
        # as unconstrained rather than excluding the cell outright.

    cand = df[(df.unmet_kwh > 0) & (df.dist_to_road_m < 800)].copy()
    n_pre = len(cand)
    # Con Edison's layer 2 lists feeders with no charging capacity at all.
    # Those cells are excluded outright rather than carried with a zero cap,
    # so the greedy and the MILP certificate see the same candidate set.
    if "charge_excluded" in cand:
        cand = cand[~cand.charge_excluded.fillna(False)]
    n_excluded = n_pre - len(cand)
    cand = cand.nlargest(1500, "unmet_kwh").reset_index(drop=True)

    plan = greedy_mclp(cand, k=k, min_per_borough=min_per_borough)
    pdf = pd.DataFrame(plan)

    # What the grid limit costs: the same instance solved without it.
    # Same counterfactual discipline as explain.equity_forced_sites().
    free = greedy_mclp(cand, k=k, min_per_borough=min_per_borough,
                       use_grid=False)
    fdf = pd.DataFrame(free)
    grid_impact = {
        "constrained_kwh_day": round(float(pdf.captured_kwh_day.sum()), 1),
        "unconstrained_kwh_day": round(float(fdf.captured_kwh_day.sum()), 1),
        "cost_kwh_day": round(float(fdf.captured_kwh_day.sum()
                                    - pdf.captured_kwh_day.sum()), 1),
        "constrained_ports": int(pdf.ports.sum()),
        "unconstrained_ports": int(fdf.ports.sum()),
        "sites_displaced": int(len(set(fdf.h3) - set(pdf.h3))),
        "sites_grid_limited": int(pdf.get("grid_limited", pd.Series(dtype=bool)).sum()),
        "substations_used": int(pdf.substation.nunique()) if "substation" in pdf else 0,
        "feeders_used": int(pdf.feeder_id.nunique()) if "feeder_id" in pdf else 0,
        "candidates_excluded_no_charge": int(n_excluded),
    }
    grid_impact["cost_pct"] = round(
        100 * grid_impact["cost_kwh_day"]
        / max(grid_impact["unconstrained_kwh_day"], 1e-9), 2)
    grid_impact["sweep"] = grid_sweep(cand, min_per_borough)
    (REPORTS / "feeder_impact.json").write_text(json.dumps(grid_impact, indent=2))

    total_unmet = float(df.unmet_kwh.sum())
    captured = float(pdf.captured_kwh_day.sum())
    res = {
        "k": k,
        "candidate_pool": int(len(cand)),
        "sites_selected": len(plan),
        "captured_kwh_day": round(captured, 1),
        "total_unmet_kwh_day": round(total_unmet, 1),
        "coverage_gain_pct": round(100 * captured / max(total_unmet, 1e-9), 2),
        "total_ports": int(pdf.ports.sum()),
        "total_capex_usd": float(pdf.capex_usd.sum()),
        "mean_wait_min": round(float(pdf.wait_min.mean()), 2),
        "p95_wait_min": round(float(pdf.wait_min.quantile(0.95)), 2),
        "max_utilisation": round(float(pdf.rho.max()), 3),
        "sites_at_capacity": int(pdf.saturated.sum()),
        "equity_floor": min_per_borough,
        "grid_impact": grid_impact,
        "by_borough": pdf.groupby("borough").size().to_dict(),
        "sites": plan,
    }
    (REPORTS / "allocation_plan.json").write_text(json.dumps(res, indent=2))
    pdf.to_parquet(PROCESSED / "plan_sites.parquet", index=False)
    return res


if __name__ == "__main__":
    r = run()
    print("Stage 4 - allocation (capacity-limited MCLP + M/M/c sizing)")
    print("  candidate pool      : %d cells" % r["candidate_pool"])
    print("  sites selected      : %d   ports %d   capex $%.1fM"
          % (r["sites_selected"], r["total_ports"], r["total_capex_usd"] / 1e6))
    print("  unmet demand served : %.0f of %.0f kWh/day  (%.1f%%)"
          % (r["captured_kwh_day"], r["total_unmet_kwh_day"],
             r["coverage_gain_pct"]))
    print("  queue wait          : mean %.1f min, p95 %.1f min"
          % (r["mean_wait_min"], r["p95_wait_min"]))
    print("  max utilisation     : %.2f  (ceiling %.2f)"
          % (r["max_utilisation"], RHO_MAX))
    print("  sites at capacity   : %d" % r["sites_at_capacity"])
    print("  by borough          : %s" % r["by_borough"])
    print("  equity floor        : %s" % r["equity_floor"])

    g = r["grid_impact"]
    print()
    print("  GRID (Con Edison per-feeder charging limit, bank headroom also enforced)")
    print("    with constraint   : %s kWh/day, %d ports"
          % (f'{g["constrained_kwh_day"]:,.0f}', g["constrained_ports"]))
    print("    without           : %s kWh/day, %d ports"
          % (f'{g["unconstrained_kwh_day"]:,.0f}', g["unconstrained_ports"]))
    print("    cost of the grid  : %s kWh/day (%.2f%%), %d sites displaced, "
          "%d port-limited"
          % (f'{g["cost_kwh_day"]:,.0f}', g["cost_pct"], g["sites_displaced"],
             g["sites_grid_limited"]))
    print("    feeders / banks used : %d / %d"
          % (g["feeders_used"], g["substations_used"]))
    print("    candidates dropped on no-charge feeders : %d"
          % g["candidates_excluded_no_charge"])
    # The constraint can only bind where Con Edison publishes capacity, so a
    # borough with no published feeder data competes unconstrained against
    # boroughs that are constrained. Manhattan is entirely in that position.
    print("    NOTE: cells with no published feeder capacity are unconstrained,")
    print("          so the constraint applies unevenly across boroughs.")
    sw = g.get("sweep", {})
    if sw:
        print()
        print("    budget sweep -- when does the grid start to bind?")
        print("      %6s %14s %14s %9s %10s"
              % ("k", "constrained", "unconstrained", "cost", "port-ltd"))
        for pt in sw["curve"]:
            print("      %6d %14s %14s %8.2f%% %10d"
                  % (pt["k"], f'{pt["constrained_kwh_day"]:,.0f}',
                     f'{pt["unconstrained_kwh_day"]:,.0f}', pt["cost_pct"],
                     pt["port_limited"]))
        print("      grid exhausts the pool at %d sites (vs %d unconstrained)"
              % (sw["sites_before_grid_exhausts"],
                 sw["sites_without_grid_limit"]))
    print()
    print(pd.DataFrame(r["sites"]).head(8)[
        ["rank", "borough", "captured_kwh_day", "ports", "rho", "wait_min",
         "sessions_day"]].to_string(index=False))
