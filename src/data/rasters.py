"""Small raster helpers shared by Phase 2+ (no GEE dependency)."""
from __future__ import annotations

import numpy as np
import rasterio


def read_stack(path):
    """Return ({band_name: 2-D float32 array (NaN=nodata)}, transform, crs)."""
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = [d or f"band{i + 1}" for i, d in enumerate(src.descriptions)]
        return dict(zip(names, arr)), src.transform, src.crs


def to_grid(path, shape, transform, crs, band=1, resampling="nearest"):
    """Resample a single-band raster onto a reference grid (e.g. 30 m DEM products -> 10 m grid)."""
    import rasterio.warp as rw
    out = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        rw.reproject(rasterio.band(src, band), out, src_transform=src.transform, src_crs=src.crs,
                     dst_transform=transform, dst_crs=crs, src_nodata=src.nodata, dst_nodata=np.nan,
                     resampling=getattr(rw.Resampling, resampling))
    return out


def write_raster(path, arr, transform, crs, nodata=np.nan, dtype=None, descriptions=None):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None]
    dtype = dtype or ("float32" if arr.dtype.kind == "f" else str(arr.dtype))
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype,
                crs=crs, transform=transform, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype))
        for i, d in enumerate(descriptions or [], 1):
            dst.set_band_description(i, d)
