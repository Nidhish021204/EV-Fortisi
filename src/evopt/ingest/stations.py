"""Station ingest: AFDC national (labels), Connecticut (OOD probe), Germany (prior)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (AFDC_NATIONAL, STATIONS_CT, STATIONS_DE, INTERIM,
                      H3_RES, H3_RES_COARSE)
from ..geo import in_bbox, to_h3

_NUM = ["EV Level1 EVSE Num", "EV Level2 EVSE Num", "EV DC Fast Count"]


def _ports(df: pd.DataFrame) -> pd.DataFrame:
    for c in _NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df.rename(columns={"EV Level1 EVSE Num": "l1",
                            "EV Level2 EVSE Num": "l2",
                            "EV DC Fast Count": "dcfc"})
    df["ports"] = df.l1 + df.l2 + df.dcfc
    # Nameplate power: L1 1.4 kW, L2 7.2 kW, DCFC 50 kW.
    df["kw"] = 1.4 * df.l1 + 7.2 * df.l2 + 50.0 * df.dcfc
    return df


def load_afdc_nyc() -> pd.DataFrame:
    """AFDC electric stations inside the NYC analysis window.

    This is the label source. `Open Date` (99.86% populated) drives the
    temporal forward test, so rows without it are kept but flagged.
    """
    usecols = ["Fuel Type Code", "Station Name", "City", "State", "ZIP",
               "Latitude", "Longitude", "Open Date", "EV Network",
               "Access Code", "Facility Type", "EV Connector Types"] + _NUM
    df = pd.read_csv(AFDC_NATIONAL, usecols=usecols, low_memory=False)
    df = df[df["Fuel Type Code"] == "ELEC"].copy()

    df["Latitude"] = pd.to_numeric(df.Latitude, errors="coerce")
    df["Longitude"] = pd.to_numeric(df.Longitude, errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df = df[in_bbox(df.Latitude, df.Longitude)].copy()

    df = _ports(df)
    df["open_date"] = pd.to_datetime(df["Open Date"], errors="coerce")
    df["open_year"] = df.open_date.dt.year
    df["h3"] = to_h3(df.Latitude.values, df.Longitude.values, H3_RES)
    df["h3_8"] = to_h3(df.Latitude.values, df.Longitude.values, H3_RES_COARSE)
    df = df.rename(columns={"Latitude": "lat", "Longitude": "lon",
                            "Station Name": "name", "City": "city",
                            "State": "state", "EV Network": "network",
                            "Access Code": "access",
                            "Facility Type": "facility"})
    keep = ["name", "city", "state", "lat", "lon", "l1", "l2", "dcfc",
            "ports", "kw", "open_date", "open_year", "network", "access",
            "facility", "h3", "h3_8"]
    return df[keep].reset_index(drop=True)


def load_germany() -> pd.DataFrame:
    """Bundesnetzagentur register: a converged network used as transfer prior.

    `inbetriebnahmedatum` gives a 2014-2023 commissioning panel, which is
    what parameterises the adoption-saturation curve.
    """
    df = pd.read_csv(STATIONS_DE, index_col=0)
    df["lat"] = pd.to_numeric(df.breitengrad, errors="coerce")
    df["lon"] = pd.to_numeric(df.laengengrad, errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df["ports"] = pd.to_numeric(df.anzahl_ladepunkte, errors="coerce").fillna(0)
    df["kw"] = pd.to_numeric(df.anschlussleistung, errors="coerce").fillna(0)
    df["is_fast"] = (df.art_der_ladeeinrichung == "Schnellladeeinrichtung").astype(int)
    df["open_date"] = pd.to_datetime(df.inbetriebnahmedatum, errors="coerce")
    df["open_year"] = df.open_date.dt.year
    df["h3"] = to_h3(df.lat.values, df.lon.values, H3_RES)
    # `betreiber` is the operator. Kept because operator diversity in a
    # neighbourhood is one of the few network-structure features that exists
    # identically in the German register and the US AFDC file, so it can
    # cross domains in the transfer study.
    df["operator"] = df.betreiber.fillna("unknown")
    return df[["lat", "lon", "ports", "kw", "is_fast", "open_date",
               "open_year", "ort", "kreis_kreisfreie_stadt", "operator", "h3"]]


def load_connecticut() -> pd.DataFrame:
    """CT open-data extract. Out-of-domain generalisation probe only.

    Geometry arrives as WKT in `New Georeferenced Column`; address fields
    contain unescaped commas, so this needs a real CSV parser.
    """
    df = pd.read_csv(STATIONS_CT)
    wkt = df["New Georeferenced Column"].astype(str)
    ext = wkt.str.extract(r"POINT \((?P<lon>-?\d+\.?\d*) (?P<lat>-?\d+\.?\d*)\)")
    df["lon"] = pd.to_numeric(ext.lon, errors="coerce")
    df["lat"] = pd.to_numeric(ext.lat, errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    df = _ports(df)
    df["h3"] = to_h3(df.lat.values, df.lon.values, H3_RES)
    return df[["Station Name", "City", "lat", "lon", "l1", "l2", "dcfc",
               "ports", "kw", "h3"]].rename(columns={"Station Name": "name",
                                                     "City": "city"})


def build() -> dict[str, pd.DataFrame]:
    out = {"nyc": load_afdc_nyc(), "de": load_germany(), "ct": load_connecticut()}
    for k, v in out.items():
        v.to_parquet(INTERIM / f"stations_{k}.parquet", index=False)
    return out


if __name__ == "__main__":
    d = build()
    n = d["nyc"]
    print(f"NYC-window stations : {len(n):,}   ports {n.ports.sum():,.0f}   "
          f"{n.kw.sum()/1000:,.1f} MW")
    print(f"  with open_date    : {n.open_date.notna().sum():,}")
    print(f"  <= 2018 (train)   : {(n.open_year <= 2018).sum():,}")
    print(f"  2019-2021 (test)  : {n.open_year.between(2019, 2021).sum():,}")
    print(f"  distinct h3 res9  : {n.h3.nunique():,}")
    print(f"  DCFC sites        : {(n.dcfc > 0).sum():,}")
    print(f"Germany  : {len(d['de']):,} stations, {d['de'].ports.sum():,.0f} ports")
    print(f"CT probe : {len(d['ct']):,} stations")
