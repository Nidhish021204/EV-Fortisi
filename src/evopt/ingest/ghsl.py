"""Stage 7f - GHSL built form and population, sampled to H3 cells.

Why this source and no other
----------------------------
The transfer study needs a built-environment feature space that exists
*identically* in New York, Connecticut and Germany. Taxi, traffic and PLUTO
stop at the NYC line. Overpass is reachable but rate-limited to two concurrent
queries, which is hopeless against Germany's 241,490 frontier cells. The JRC
Global Human Settlement Layer is open, global, and produced by one method
everywhere, so a German cell and a Queens cell are measured the same way:

    GHS_BUILT_S_E2020_..._4326_3ss   built-up surface, m2 per pixel
    GHS_POP_E2020_..._4326_3ss       residents per pixel

Both are WGS84 at 3 arc-seconds (~90 m), so no reprojection is needed and a
pixel is finer than the ~380 m across an H3 res-9 cell.

What this does and does not test
--------------------------------
GHSL carries built *form* and *population*, not land-use *composition*. NYC's
own strongest signals -- office/retail floor area, garage area, land-use
entropy -- have no global equivalent. So this supports the claim "built form
and population do / do not transfer", which is strictly weaker than "land use
transfers". The distinction is stated in the results rather than buried.

Tiling, calibrated from a real tile rather than assumed
-------------------------------------------------------
R4_C12 reads as bounds (-70.008, 49.1) to (-60.008, 59.1) at 12000x12000, so
tiles are 10 degrees square on an origin of 180.008W / 89.1N:

    col = floor((lon + 180.008) / 10) + 1
    row = floor((89.1 - lat)  / 10) + 1

NYC and Connecticut both fall in R5_C11; Germany spans four tiles.

Sampling
--------
Cells are sampled at two scales: a 5x5 window (~450 m), a close cover for the
380 m hexagon itself, and a 15x15 window (~1.3 km) for its surroundings. The
second scale matters because k-ring spatial lags are consistently the strongest
features in the NYC model -- a site's value is set largely by what surrounds it
-- and a wider raster window is the raster-native way to say the same thing
without running grid_disk over 241k German cells.

Pixel indices are computed arithmetically for every cell at once, so there is
no per-cell file I/O: one windowed read per tile, then vectorised gathers.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..config import RAW, INTERIM

BASE = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
        "GHS_{p}_GLOBE_R2023A/GHS_{p}_E2020_GLOBE_R2023A_4326_3ss/V1-0/tiles/"
        "GHS_{p}_E2020_GLOBE_R2023A_4326_3ss_V1_0_R{r}_C{c}.zip")
CACHE = RAW / "ghsl"
PRODUCTS = ("BUILT_S", "POP")

LON0, LAT0 = -180.008, 89.1     # tile grid origin
TILE_DEG = 10.0
PIX = 12000                     # pixels per tile side
RES = TILE_DEG / PIX            # ~0.000833 deg
HALF = 2                        # 5x5 window ~ 450 m, the cell itself
HALF_CTX = 7                    # 15x15 ~ 1.3 km, the surrounding context


def tile_of(lat: float, lon: float) -> tuple[int, int]:
    return (int((LAT0 - lat) // TILE_DEG) + 1,
            int((lon - LON0) // TILE_DEG) + 1)


def tiles_for(lats, lons) -> set[tuple[int, int]]:
    rc = {tile_of(a, o) for a, o in zip(np.asarray(lats), np.asarray(lons))}
    return rc


M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


def pixel_area_m2(lat) -> np.ndarray:
    """Ground area of one 3 arc-second pixel at a given latitude."""
    lat = np.asarray(lat, float)
    h = RES * M_PER_DEG_LAT
    w = RES * M_PER_DEG_LON_EQ * np.cos(np.radians(lat))
    return h * w


def _tif_path(product: str, r: int, c: int):
    return CACHE / f"GHS_{product}_E2020_GLOBE_R2023A_4326_3ss_V1_0_R{r}_C{c}.tif"


def fetch_tile(product: str, r: int, c: int):
    """Download and unzip one tile; a no-op once cached."""
    out = _tif_path(product, r, c)
    if out.exists():
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    url = BASE.format(p=product, r=r, c=c)
    with urllib.request.urlopen(url, timeout=600) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".tif")][0]
        out.write_bytes(z.read(name))
    print(f"  fetched {product} R{r}_C{c} ({len(blob)/1e6:.1f} MB)", flush=True)
    return out


def sample(cells: pd.DataFrame) -> pd.DataFrame:
    """Attach GHSL built form and population to `cells` (h3, lat, lon)."""
    import rasterio

    lat = cells.lat.to_numpy(float)
    lon = cells.lon.to_numpy(float)
    scales = {"": HALF, "_ctx": HALF_CTX}
    out = {(p, sfx): np.full(len(cells), np.nan)
           for p in PRODUCTS for sfx in scales}

    for (r, c) in sorted(tiles_for(lat, lon)):
        # Cells belonging to this tile, by the same arithmetic used to pick it.
        sel = np.array([tile_of(a, o) == (r, c) for a, o in zip(lat, lon)])
        if not sel.any():
            continue
        west = LON0 + (c - 1) * TILE_DEG
        north = LAT0 - (r - 1) * TILE_DEG
        px = ((lon[sel] - west) / RES).astype(int)
        py = ((north - lat[sel]) / RES).astype(int)

        for product in PRODUCTS:
            try:
                path = fetch_tile(product, r, c)
            except Exception as e:                      # ocean / absent tile
                print(f"  ! {product} R{r}_C{c} unavailable: {e}", flush=True)
                continue
            pad = max(scales.values())
            with rasterio.open(path) as src:
                x0 = max(0, int(px.min()) - pad)
                x1 = min(PIX, int(px.max()) + pad + 1)
                y0 = max(0, int(py.min()) - pad)
                y1 = min(PIX, int(py.max()) + pad + 1)
                win = rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0)
                arr = src.read(1, window=win).astype(np.float32)
                nod = src.nodata
            if nod is not None:
                arr[arr == nod] = 0.0

            # Window mean about each centroid, vectorised over all cells.
            for sfx, half in scales.items():
                gx = np.clip(px - x0, half, arr.shape[1] - half - 1)
                gy = np.clip(py - y0, half, arr.shape[0] - half - 1)
                acc = np.zeros(gx.shape[0], np.float64)
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        acc += arr[gy + dy, gx + dx]
                out[(product, sfx)][sel] = acc / ((2 * half + 1) ** 2)

    # Pixel area is NOT constant: a 3 arc-second pixel keeps its height but
    # narrows as cos(latitude). Ignoring that would inflate built_share and
    # pop_density at high latitude relative to low -- and since this feature
    # space exists to compare Germany (~52N, cos 0.62) against New York
    # (~40.7N, cos 0.76), a latitude-blind denominator would bake a ~19%
    # systematic bias straight into the transfer result it is meant to measure.
    px_area = pixel_area_m2(lat)

    df = pd.DataFrame({"h3": cells.h3.values})
    for sfx in ("", "_ctx"):
        df["built_share" + sfx] = np.clip(out[("BUILT_S", sfx)] / px_area, 0, 1)
        df["pop_density" + sfx] = out[("POP", sfx)] / (px_area / 1e6)  # per km2
    df["built_surface"] = out[("BUILT_S", "")]
    df["pop"] = out[("POP", "")]
    df["px_area_m2"] = px_area
    # Contrast between the cell and its surroundings: a locally dense cell in a
    # sparse area behaves differently from one inside a dense district, and the
    # ratio expresses that in a way that carries across countries.
    df["built_contrast"] = df.built_share / df.built_share_ctx.replace(0, np.nan)
    df["pop_contrast"] = df.pop_density / df.pop_density_ctx.replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


GHSL_FEATURES = ["built_share", "built_share_ctx", "pop_density",
                 "pop_density_ctx", "built_contrast", "pop_contrast"]


def build(cells: pd.DataFrame | None = None, tag: str = "nyc") -> pd.DataFrame:
    """Sample GHSL for a cell table and cache it under `tag`."""
    if cells is None:
        from ..config import PROCESSED
        cells = pd.read_parquet(PROCESSED / "features_h3.parquet",
                                columns=["h3", "lat", "lon"])
    df = sample(cells)
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERIM / f"ghsl_{tag}.parquet", index=False)
    return df


if __name__ == "__main__":
    g = build()
    # An H3 res-9 cell is ~105,000 m2; `pop` is residents per pixel, so the
    # cell total is pop x (cell area / pixel area). Multiplying by the 25
    # sampled pixels instead would double-count, because neighbouring cells'
    # 450 m windows overlap.
    H3_RES9_M2 = 105_000.0
    per_cell = g["pop"] * (H3_RES9_M2 / g.px_area_m2)
    print(f"cells sampled     : {len(g):,}")
    print(f"population implied: {per_cell.sum():,.0f}  (NYC actual ~8.3M)")
    print(f"built share       : min {g.built_share.min():.3f} "
          f"max {g.built_share.max():.3f} median {g.built_share.median():.3f}")
    print(f"pop density       : median {g.pop_density.median():,.0f}/km2  "
          f"max {g.pop_density.max():,.0f}/km2")
    print(f"cells with zero   : built {int((g.built_share <= 0).sum()):,}  "
          f"pop {int((g['pop'] <= 0).sum()):,}")
