"""Stage 7d - ACS vehicle availability -> H3 household and vehicle features.

Source: ACS 2022 5-year table B25044 (Tenure by Vehicles Available), taken from
the Census **bulk summary file**, not the API. The API now demands a key; the
bulk file does not, and it carries the same estimates.

    https://www2.census.gov/programs-surveys/acs/summary_file/2022/
        table-based-SF/data/5YRData/acsdt5y2022-b25044.dat

Column layout, verified against a real row rather than assumed:
    E001 total           E002 owner occupied     E003 owner, no vehicle
    E004..E008 owner with 1,2,3,4,5+            E009 renter occupied
    E010 renter, no vehicle                      E011..E015 renter with 1..5+
Checks that hold on tract 36061000202: E002 == sum(E003..E008) == 887, and
E001 == E002 + E009 == 3293.

Tract -> cell allocation
------------------------
A census tract is far larger than an H3 res-9 cell, so tract values are
*allocated*, not broadcast. PLUTO carries `bct2020` on 99.9% of lots, so each
tract's households are split across the cells it touches in proportion to the
residential units actually standing in each cell. Households follow homes.
This conserves totals -- `build()` asserts it -- rather than counting a tract's
households once per cell it overlaps, which is the same double-count that had
to be fixed in the supply kernel in Stage 5.

The derived feature
-------------------
Vehicle ownership cuts both ways for public charging. High ownership usually
means private off-street parking and therefore home charging; Manhattan has few
cars, but the ones it has often have nowhere to plug in overnight. What matters
is not cars, but *cars with no home charging*:

    garage_spaces   = garage_area / SQFT_PER_SPACE
    home_charge_cov = min(1, garage_spaces / units_res)
    no_home_charge  = renter_veh_hh * (1 - home_charge_cov)

That is a hypothesis, so it ships next to its own raw components and the SHAP
panel is free to refute it.
"""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from ..config import RAW, INTERIM

URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/2022/"
       "table-based-SF/data/5YRData/acsdt5y2022-b25044.dat")
CACHE = RAW / "acs" / "b25044_nyc.parquet"

# state 36 (NY) + the five borough counties
NYC_COUNTIES = {"005": "Bronx", "047": "Brooklyn", "061": "Manhattan",
                "081": "Queens", "085": "Staten Island"}
# PLUTO's bct2020 leads with a borough digit; map it back to a county code.
BORO_DIGIT_TO_COUNTY = {"1": "061", "2": "005", "3": "047", "4": "081",
                        "5": "085"}

SQFT_PER_SPACE = 325.0   # structured parking space incl. circulation

_E = lambda i: f"B25044_E{i:03d}"  # noqa: E731


def fetch(force: bool = False) -> pd.DataFrame:
    """Download once, keep only NYC tracts, cache as parquet."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = ["GEO_ID"] + [_E(i) for i in range(1, 16)]
    rows = []
    with urllib.request.urlopen(URL, timeout=300) as r:
        header = r.readline().decode().rstrip("\n").split("|")
        idx = {c: header.index(c) for c in keep_cols}
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if not line.startswith("1400000US36"):
                continue
            parts = line.split("|")
            geo = parts[idx["GEO_ID"]]
            if geo[len("1400000US36"):][:3] not in NYC_COUNTIES:
                continue
            rows.append([geo] + [parts[idx[_E(i)]] for i in range(1, 16)])

    df = pd.DataFrame(rows, columns=keep_cols)
    for c in keep_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df.to_parquet(CACHE, index=False)
    return df


def tract_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tract household and vehicle counts."""
    out = pd.DataFrame({"geo_id": df.GEO_ID})
    out["county"] = df.GEO_ID.str[len("1400000US36"):].str[:3]
    out["tract"] = df.GEO_ID.str[-6:]

    owner_k = [df[_E(i)] for i in range(4, 9)]      # owner with 1..5+
    renter_k = [df[_E(i)] for i in range(11, 16)]   # renter with 1..5+

    out["hh_total"] = df[_E(1)]
    out["zero_veh_hh"] = df[_E(3)] + df[_E(10)]
    # 5+ is open-ended; 5 is the conservative reading of it.
    out["veh_total"] = sum((k + 1) * (o + r)
                           for k, (o, r) in enumerate(zip(owner_k, renter_k)))
    out["renter_veh_hh"] = df[_E(9)] - df[_E(10)]
    out["owner_hh"] = df[_E(2)]
    out["renter_hh"] = df[_E(9)]
    return out


def build() -> pd.DataFrame:
    acs = tract_measures(fetch())
    acs["key"] = acs.county + acs.tract

    lots = pd.read_parquet(RAW / "landuse" / "pluto.parquet",
                           columns=["bct2020", "unitsres", "latitude",
                                    "longitude", "garagearea"])
    lots = lots.dropna(subset=["bct2020", "latitude", "longitude"]).copy()
    bct = lots.bct2020.astype(str).str.zfill(7)
    lots["key"] = bct.str[0].map(BORO_DIGIT_TO_COUNTY) + bct.str[1:7]
    lots = lots.dropna(subset=["key"])

    from ..geo import to_h3, in_bbox
    from ..config import H3_RES
    lots = lots[in_bbox(lots.latitude, lots.longitude)].copy()
    lots["h3"] = to_h3(lots.latitude.values, lots.longitude.values, H3_RES)
    lots["unitsres"] = lots.unitsres.fillna(0.0)

    # Weight of each (tract, cell) pair = its share of the tract's homes.
    # Tracts with no recorded units fall back to an equal split over their
    # cells, so their households are neither lost nor concentrated.
    pair = lots.groupby(["key", "h3"], as_index=False).agg(
        units=("unitsres", "sum"), lots_n=("unitsres", "size"))
    tot = pair.groupby("key").units.transform("sum")
    cnt = pair.groupby("key").units.transform("size")
    pair["w"] = np.where(tot > 0, pair.units / tot.replace(0, np.nan), 1.0 / cnt)
    pair["w"] = pair.w.fillna(0.0)

    m = pair.merge(acs, on="key", how="inner")
    for c in ("hh_total", "zero_veh_hh", "veh_total", "renter_veh_hh",
              "owner_hh", "renter_hh"):
        m[c] = m[c] * m.w

    cell = m.groupby("h3", as_index=False)[
        ["hh_total", "zero_veh_hh", "veh_total", "renter_veh_hh",
         "owner_hh", "renter_hh"]].sum()

    cell["veh_per_hh"] = cell.veh_total / cell.hh_total.replace(0, np.nan)
    cell["zero_veh_share"] = cell.zero_veh_hh / cell.hh_total.replace(0, np.nan)
    cell["renter_share"] = cell.renter_hh / cell.hh_total.replace(0, np.nan)

    # ── derived: cars with nowhere to charge at home ────────────────
    lu = pd.read_parquet(INTERIM / "landuse_h3.parquet",
                         columns=["h3", "garage_area", "units_res"])
    cell = cell.merge(lu, on="h3", how="left")
    cell[["garage_area", "units_res"]] = cell[["garage_area", "units_res"]].fillna(0.0)

    spaces = cell.garage_area / SQFT_PER_SPACE
    cov = np.minimum(1.0, spaces / cell.units_res.replace(0, np.nan))
    cell["home_charge_cov"] = cov.fillna(0.0)
    cell["no_home_charge"] = cell.renter_veh_hh * (1.0 - cell.home_charge_cov)
    cell["no_home_charge_density"] = (
        cell.no_home_charge / cell.hh_total.replace(0, np.nan))

    cell = cell.drop(columns=["garage_area", "units_res"])
    cell = cell.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Conservation: allocation must not create or destroy households.
    src = acs[acs.key.isin(set(pair.key))].hh_total.sum()
    got = cell.hh_total.sum()
    cell.attrs["hh_source"] = float(src)
    cell.attrs["hh_allocated"] = float(got)
    cell.attrs["hh_error_pct"] = float(100 * abs(got - src) / max(src, 1))

    cell.to_parquet(INTERIM / "acs_h3.parquet", index=False)
    return cell


if __name__ == "__main__":
    c = build()
    print(f"H3 cells with households : {len(c):,}")
    print(f"households allocated     : {c.hh_total.sum():,.0f} "
          f"(source {c.attrs['hh_source']:,.0f}, "
          f"error {c.attrs['hh_error_pct']:.4f}%)")
    print(f"vehicles                 : {c.veh_total.sum():,.0f}")
    print(f"zero-vehicle households  : {c.zero_veh_hh.sum():,.0f} "
          f"({100*c.zero_veh_hh.sum()/max(c.hh_total.sum(),1):.1f}%)")
    print(f"cars with no home charge : {c.no_home_charge.sum():,.0f}")
    print(f"median veh/household     : {c.veh_per_hh.median():.3f}")
    print()
    print(c.nlargest(5, "no_home_charge")[
        ["h3", "hh_total", "veh_per_hh", "zero_veh_share", "home_charge_cov",
         "no_home_charge"]].round(3).to_string(index=False))
