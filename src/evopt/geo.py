"""Geospatial primitives: H3 binning, reprojection, distance kernels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import h3
from pyproj import Transformer

from .config import H3_RES, NYC_BBOX

# NY State Plane Long Island (US survey feet) -> WGS84.
# The NYC traffic-count WktGeom column is published in EPSG:2263.
_T2263 = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)

EARTH_R = 6_371_000.0


def statesplane_to_wgs84(x, y):
    """EPSG:2263 (ft) -> (lon, lat). Vectorised."""
    lon, lat = _T2263.transform(np.asarray(x, float), np.asarray(y, float))
    return lon, lat


def in_bbox(lat, lon) -> np.ndarray:
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    b = NYC_BBOX
    return (
        (lat >= b["lat_min"]) & (lat <= b["lat_max"])
        & (lon >= b["lon_min"]) & (lon <= b["lon_max"])
    )


def to_h3(lat, lon, res: int = H3_RES) -> np.ndarray:
    """Vectorised lat/lon -> H3 cell id (string)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    return np.array(
        [h3.latlng_to_cell(a, o, res) for a, o in zip(lat, lon)], dtype=object
    )


def h3_centroids(cells) -> pd.DataFrame:
    ll = [h3.cell_to_latlng(c) for c in cells]
    return pd.DataFrame({"h3": list(cells),
                         "lat": [p[0] for p in ll],
                         "lon": [p[1] for p in ll]})


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between broadcastable arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gravity_kernel(dist_m, sigma_m):
    """Gaussian distance decay used for accessible-supply accumulation."""
    d = np.asarray(dist_m, float)
    return np.exp(-(d ** 2) / (2.0 * sigma_m ** 2))


def kring_aggregate(df: pd.DataFrame, cols, k: int = 1, prefix: str | None = None):
    """Mean of `cols` over each cell's k-ring neighbourhood (excluding self).

    Spatial-lag features: the single most useful class of feature for a
    problem where a site's value depends on what surrounds it.
    """
    prefix = prefix or f"k{k}_"
    idx = {c: i for i, c in enumerate(df["h3"].values)}
    vals = df[cols].to_numpy(float)
    out = np.zeros((len(df), len(cols)), float)
    for i, cell in enumerate(df["h3"].values):
        ring = h3.grid_ring(cell, k)
        rows = [idx[r] for r in ring if r in idx]
        out[i] = vals[rows].mean(axis=0) if rows else 0.0
    return pd.DataFrame(out, columns=[f"{prefix}{c}" for c in cols], index=df.index)
