"""DEM acquisition and preparation (Phase 1).

Rationale
---------
Terrain and hydrology need a metric-CRS DEM on a regular grid. We keep acquisition
(`fetch_*`) separate from preparation (`prepare_dem`) so any DEM — Copernicus GLO-30,
ALOS AW3D30, or the better-suited GSI 5/10 m DEM — can be dropped in via `local_file`.

Assumptions / caveats (also in docs/ARCHITECTURE.md)
----------------------------------------------------
* Copernicus GLO-30 is a *surface* model (DSM): canopy height is baked in, which biases
  small forested channels. The 30 m grid cannot resolve incised gullies. GSI DEM5A/5B/10B
  (airborne LiDAR / photogrammetry derived, ground surface) is preferred when available.
* Sea is masked from the DEM (`<= sea_level_m`), so the island mask is *derived*, not assumed.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds
from scipy import ndimage

log = logging.getLogger(__name__)

COP_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")


def _tile_names(bbox):
    minx, miny, maxx, maxy = bbox
    for lat in range(math.floor(miny), math.floor(maxy) + 1):
        for lon in range(math.floor(minx), math.floor(maxx) + 1):
            yield COP_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                 ew="E" if lon >= 0 else "W", lon=abs(lon))


def fetch_copernicus_dem(bbox, out_path: Path) -> Path:
    """Windowed read of public Copernicus GLO-30 COG tiles (no auth) -> GeoTIFF (EPSG:4326).

    Needs outbound HTTPS to copernicus-dem-30m.s3.amazonaws.com.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    for url in _tile_names(bbox):
        log.info("Reading %s", url)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
            with rasterio.open("/vsicurl/" + url) as src:
                win = from_bounds(*bbox, transform=src.transform)
                data = src.read(1, window=win, boundless=False)
                tr = src.window_transform(win)
                pieces.append((data, tr, src.crs, src.nodata))
    if not pieces:
        raise RuntimeError("No Copernicus tiles intersect bbox")
    if len(pieces) == 1:
        data, tr, crs, nd = pieces[0]
        prof = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                    dtype="float32", crs=crs, transform=tr, nodata=nd, compress="deflate")
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data.astype("float32"), 1)
    else:  # multi-tile case: write each, then merge
        tmp = []
        for i, (data, tr, crs, nd) in enumerate(pieces):
            p = out_path.with_suffix(f".part{i}.tif")
            prof = dict(driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
                        dtype="float32", crs=crs, transform=tr, nodata=nd)
            with rasterio.open(p, "w", **prof) as dst:
                dst.write(data.astype("float32"), 1)
            tmp.append(p)
        srcs = [rasterio.open(p) for p in tmp]
        mosaic, tr = merge(srcs)
        prof = dict(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2], count=1,
                    dtype="float32", crs=srcs[0].crs, transform=tr, nodata=srcs[0].nodata)
        for s in srcs:
            s.close()
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(mosaic[0], 1)
        for p in tmp:
            p.unlink()
    return out_path


def prepare_dem(raw_path: Path, out_path: Path, dst_crs: str, resolution_m: float,
                resampling: str = "bilinear", sea_level_m: float = 0.5,
                min_island_cells: int = 200, bbox_wgs84=None):
    """Reproject to the metric analysis CRS, resample, and build the land mask.

    Returns (dem float32 with NaN on sea, transform, crs).
    Writes a GeoTIFF with NaN as nodata.
    """
    with rasterio.open(raw_path) as src:
        src_nodata = src.nodata
        left, bottom, right, top = src.bounds
        tr, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height,
                                               left, bottom, right, top,
                                               resolution=resolution_m)
        dem = np.full((h, w), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dem, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr, dst_crs=dst_crs, resampling=getattr(Resampling, resampling),
                  src_nodata=src_nodata, dst_nodata=np.nan)
    land = np.isfinite(dem) & (dem > sea_level_m)
    # Remove tiny land specks (rocks/noise) and fill tiny holes inside land
    lab, n = ndimage.label(land, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(land, lab, index=np.arange(1, n + 1))
        keep = np.isin(lab, 1 + np.flatnonzero(sizes >= min_island_cells))
        land = keep
    dem = np.where(land, dem, np.nan).astype("float32")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=dst_crs,
                transform=tr, nodata=np.nan, compress="deflate")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(dem, 1)
    log.info("DEM prepared: %dx%d @ %.1f m, land cells=%d", w, h, resolution_m, int(land.sum()))
    return dem, tr, rasterio.crs.CRS.from_user_input(dst_crs)


def load_prepared_dem(path: Path):
    with rasterio.open(path) as src:
        return src.read(1).astype("float32"), src.transform, src.crs


def dem_mosaic(cfg, ee):
    """Earth Engine DEM mosaic with its native projection restored (needed for resampling and
    terrain ops, because the DEM is an ImageCollection of tiles)."""
    d = cfg["dem"]
    aoi = ee.Geometry.Rectangle(cfg["study_area"]["bbox"])
    col = ee.ImageCollection(d["gee_asset"]).filterBounds(aoi).select(d["gee_band"])
    proj = col.first().select(0).projection()
    return col.mosaic().setDefaultProjection(proj)


def fetch_gee_dem(cfg, out_path: Path) -> Path:
    """Export the configured GEE DEM on the 30 m snapped UTM analysis grid.

    Cells with no DEM tile (open ocean) come back as NODATA and are treated as sea.
    """
    from ..gee import download, init
    ee = init.initialize(cfg)
    crs = cfg["study_area"]["analysis_crs"]
    res = float(cfg["dem"]["target_resolution_m"])
    bounds = download.analysis_bounds(cfg["study_area"]["bbox"], crs, snap=30.0)
    img = dem_mosaic(cfg, ee).resample("bilinear").rename("DEM")
    p = download.download_image(img, bounds, crs, res, out_path, ["DEM"], cfg["gee"]["tile_px"])
    # nodata handling: prepare_dem reads NaN as sea
    return p
