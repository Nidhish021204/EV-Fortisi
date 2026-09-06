"""NYC MapPLUTO tax-lot land use -> H3 built-environment features.

Source: NYC Open Data resource 64uk-42ks (858,284 tax lots), fetched once over
the Socrata API and cached to data/raw/landuse/. Re-running is a no-op unless
the cache is deleted.

Why PLUTO rather than census tracts. Tract-level ACS is coarse (a Manhattan
tract dwarfs an H3 res-9 cell) and reports people where they sleep. PLUTO is
per-lot, so it resolves below the modelling unit, and it carries *what the
floor area is used for* -- residential, office, retail, factory, garage --
which is what actually generates and absorbs vehicle trips.

The single most directly relevant column is `garagearea`. Charging hardware
goes into parking structures; a cell with no parking floor area is a cell where
siting is physically hard regardless of how much demand passes through it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from ..config import RAW, INTERIM, H3_RES
from ..geo import to_h3, in_bbox

RESOURCE = "https://data.cityofnewyork.us/resource/64uk-42ks.json"
CACHE = RAW / "landuse" / "pluto.parquet"
PAGE = 50_000

FIELDS = [
    "latitude", "longitude", "landuse", "lotarea", "bldgarea", "comarea",
    "resarea", "officearea", "retailarea", "garagearea", "factryarea",
    "strgearea", "otherarea", "unitsres", "unitstotal", "numfloors",
    "builtfar", "yearbuilt", "assesstot", "borough", "bct2020",
]

# PLUTO land use codes, collapsed to the classes that generate distinct trip
# and dwell behaviour. 1-2 family and walkup differ in density but not in kind.
LU_GROUPS = {
    "lu_res_low": ["01"], "lu_res_mid": ["02"], "lu_res_high": ["03"],
    "lu_mixed": ["04"], "lu_commercial": ["05"], "lu_industrial": ["06"],
    "lu_transport": ["07"], "lu_institutional": ["08"], "lu_openspace": ["09"],
    "lu_parking": ["10"], "lu_vacant": ["11"],
}

NUMERIC = ["latitude", "longitude", "lotarea", "bldgarea", "comarea", "resarea",
           "officearea", "retailarea", "garagearea", "factryarea", "strgearea",
           "otherarea", "unitsres", "unitstotal", "numfloors", "builtfar",
           "yearbuilt", "assesstot"]


def fetch(force: bool = False) -> pd.DataFrame:
    """Page the whole resource into a local parquet cache."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    sel = ",".join(FIELDS)
    frames, offset = [], 0
    while True:
        url = (f"{RESOURCE}?$select={sel}&$limit={PAGE}&$offset={offset}"
               "&$order=bbl")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    rows = json.loads(r.read().decode())
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        got = sum(len(f) for f in frames)
        print(f"  fetched {got:,} lots", flush=True)
        if len(rows) < PAGE:
            break
        offset += PAGE

    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df.to_parquet(CACHE, index=False)
    return df


def build() -> pd.DataFrame:
    lots = fetch()
    lots = lots.dropna(subset=["latitude", "longitude"])
    lots = lots[in_bbox(lots.latitude, lots.longitude)].copy()
    lots["h3"] = to_h3(lots.latitude.values, lots.longitude.values, H3_RES)

    lu = lots.landuse.astype(str).str.zfill(2)
    for name, codes in LU_GROUPS.items():
        lots[name] = np.where(lu.isin(codes), lots.lotarea.fillna(0), 0.0)

    area_cols = list(LU_GROUPS)
    agg = lots.groupby("h3", as_index=False).agg(
        lots_n=("lotarea", "size"),
        lot_area=("lotarea", "sum"),
        bldg_area=("bldgarea", "sum"),
        res_area=("resarea", "sum"),
        com_area=("comarea", "sum"),
        office_area=("officearea", "sum"),
        retail_area=("retailarea", "sum"),
        garage_area=("garagearea", "sum"),
        factory_area=("factryarea", "sum"),
        storage_area=("strgearea", "sum"),
        units_res=("unitsres", "sum"),
        units_total=("unitstotal", "sum"),
        floors_mean=("numfloors", "mean"),
        far_mean=("builtfar", "mean"),
        year_built_med=("yearbuilt", "median"),
        assess_total=("assesstot", "sum"),
        **{c: (c, "sum") for c in area_cols},
    )

    # Shares rather than raw areas: a cell's *composition* transfers across
    # cities, its absolute square footage does not.
    tot = agg[area_cols].sum(axis=1).replace(0, np.nan)
    for c in area_cols:
        agg[c + "_sh"] = (agg[c] / tot).fillna(0.0)

    # Shannon entropy over the land-use mix. The standard mixed-use index:
    # high entropy means trips start and end in the same cell, which is
    # exactly where a charger gets used across the whole day rather than in
    # one commuting spike.
    p = agg[[c + "_sh" for c in area_cols]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    # +0.0 normalises the -0.0 a single-land-use cell produces (p=1 -> -1*0).
    agg["lu_entropy"] = np.clip(ent / np.log(len(area_cols)), 0.0, 1.0) + 0.0

    # Derived intensities.
    agg["far_built"] = agg.bldg_area / agg.lot_area.replace(0, np.nan)
    agg["jobs_area"] = agg.office_area + agg.retail_area + agg.com_area
    agg["parking_share"] = (agg.garage_area
                            / agg.bldg_area.replace(0, np.nan)).fillna(0.0)
    agg["assess_per_sqft"] = (agg.assess_total
                              / agg.lot_area.replace(0, np.nan))
    agg["res_density"] = agg.units_res / agg.lot_area.replace(0, np.nan)
    agg = agg.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # PLUTO carries a small number of lots whose component areas exceed their
    # own building area (median ratio is exactly 1.000, mean 1.026). Left
    # alone these produce absurd intensity ratios, so the derived ratios are
    # winsorised at the 99.5th percentile. Raw areas are kept untouched.
    for c in ("far_built", "assess_per_sqft", "res_density", "parking_share"):
        agg[c] = agg[c].clip(upper=float(agg[c].quantile(0.995)))

    agg.to_parquet(INTERIM / "landuse_h3.parquet", index=False)
    return agg


if __name__ == "__main__":
    a = build()
    print(f"H3 cells with tax lots : {len(a):,}")
    print(f"lots assigned          : {a.lots_n.sum():,.0f}")
    print(f"residential units      : {a.units_res.sum():,.0f}")
    print(f"garage floor area      : {a.garage_area.sum()/1e6:,.1f}M sq ft")
    print(f"office+retail area     : {a.jobs_area.sum()/1e6:,.1f}M sq ft")
    print(f"median land-use entropy: {a.lu_entropy.median():.3f}")
    print()
    print(a.nlargest(5, "units_res")[
        ["h3", "lots_n", "units_res", "far_built", "lu_entropy",
         "garage_area"]].to_string(index=False))
