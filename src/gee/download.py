"""Tiled GeoTIFF download of an ee.Image onto a fixed UTM grid (no Drive/Cloud Storage needed).

Why tiled: `getDownloadURL` is limited (~32 MB / 10k px per side). The island is ~20 x 17 km, i.e.
~2000 x 1700 px at 10 m x up to 13 bands, so we request <= tile_px x tile_px chunks and stitch them.
Grid: bounds are snapped OUTWARD to multiples of `snap` (30 m by default) so that 10 m
and 30 m rasters from different calls nest exactly and can be stacked without resampling.
Pure helpers (`analysis_bounds`, `tile_grid`, `assemble`) are unit-tested offline.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

log = logging.getLogger(__name__)
logging.getLogger("rasterio._env").setLevel(logging.ERROR)  # EE GeoTIFF extra-sample notice is harmless
NODATA = -9999.0


def analysis_bounds(bbox_wgs84, crs: str, snap: float = 30.0):
    """WGS84 bbox -> (xmin, ymin, xmax, ymax) in `crs`, snapped outward to multiples of snap."""
    from pyproj import Transformer
    minx, miny, maxx, maxy = bbox_wgs84
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x in (minx, maxx) for y in (miny, maxy)])
    return (math.floor(min(xs) / snap) * snap, math.floor(min(ys) / snap) * snap,
            math.ceil(max(xs) / snap) * snap, math.ceil(max(ys) / snap) * snap)


def tile_grid(bounds, scale: float, tile_px: int):
    """Split bounds into tiles of at most tile_px x tile_px pixels. Returns list of
    (x0, y0, x1, y1, col_off, row_off, w, h) with row 0 at the TOP (ymax)."""
    xmin, ymin, xmax, ymax = bounds
    W = int(round((xmax - xmin) / scale)); H = int(round((ymax - ymin) / scale))
    tiles = []
    for r0 in range(0, H, tile_px):
        for c0 in range(0, W, tile_px):
            w, h = min(tile_px, W - c0), min(tile_px, H - r0)
            x0 = xmin + c0 * scale; x1 = x0 + w * scale
            y1 = ymax - r0 * scale; y0 = y1 - h * scale
            tiles.append((x0, y0, x1, y1, c0, r0, w, h))
    return tiles, W, H


def assemble(tile_arrays, tiles, W, H, n_bands):
    out = np.full((n_bands, H, W), np.nan, dtype="float32")
    for arr, (_, _, _, _, c0, r0, w, h) in zip(tile_arrays, tiles):
        a = arr.astype("float32")
        a[a == NODATA] = np.nan
        out[:, r0:r0 + h, c0:c0 + w] = a[:, :h, :w]
    return out


def _fetch_tile_bytes(image, tile, crs, scale, retries=3) -> bytes:
    import ee
    import requests
    x0, y0, x1, y1 = tile[:4]
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], proj=crs, geodesic=False)
    last = None
    for k in range(retries):
        try:
            url = image.getDownloadURL({"region": region, "crs": crs,
                                        "crs_transform": [scale, 0, x0, 0, -scale, y1],
                                        "format": "GEO_TIFF"})
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.content
        except Exception as e:  # transient EE / network errors
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"Tile download failed after {retries} tries: {last}")


def download_image(image, bounds, crs: str, scale: float, out_path, band_names=None,
                   tile_px: int = 400, fetch=_fetch_tile_bytes) -> Path:
    """Download `image` (unmasked with NODATA) to a GeoTIFF on the snapped UTM grid."""
    img = image.unmask(NODATA)
    tiles, W, H = tile_grid(bounds, scale, tile_px)
    arrays = []
    for i, t in enumerate(tiles):
        log.info("tile %d/%d", i + 1, len(tiles))
        data = fetch(img, t, crs, scale)
        with MemoryFile(data) as mf, mf.open() as src:
            arrays.append(src.read())
    n_bands = arrays[0].shape[0]
    out = assemble(arrays, tiles, W, H, n_bands)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=H, width=W, count=n_bands, dtype="float32", crs=crs,
                transform=from_origin(bounds[0], bounds[3], scale, scale), nodata=np.nan,
                compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out)
        if band_names:
            for i, n in enumerate(band_names, 1):
                dst.set_band_description(i, n)
    return out_path
