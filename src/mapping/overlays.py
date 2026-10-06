"""Georeferenced raster overlays (WGS84 RGBA PNG + bounds) so the four final maps can also be shown as layers on the
interactive web map. Pure numpy/rasterio/PIL: no Streamlit import. Layers mirror what the static maps draw."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio.warp as rw
from matplotlib import cm, colors as mcolors
from PIL import Image
from rasterio.transform import array_bounds

from .final_maps import CHANGE_COL


def to_rgba(rgb=None, alpha=None, rgba=None):
    """uint8 RGBA from float RGB (0-1) + alpha (0-1), or pass a float RGBA directly."""
    if rgba is None:
        rgba = np.dstack([rgb, alpha])
    return (np.clip(np.nan_to_num(rgba), 0, 1) * 255).astype("uint8")


def warp_rgba(rgba, transform, crs, resampling="nearest"):
    """Reproject an (H, W, 4) uint8 image to EPSG:4326. Returns (image, (west, south, east, north))."""
    h, w, _ = rgba.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dt, dw, dh = rw.calculate_default_transform(crs, "EPSG:4326", w, h, left, bottom, right, top)
    out = np.zeros((dh, dw, 4), "uint8")
    for b in range(4):
        rw.reproject(np.ascontiguousarray(rgba[..., b]), out[..., b], src_transform=transform, src_crs=crs,
                     dst_transform=dt, dst_crs="EPSG:4326", dst_nodata=0, resampling=getattr(rw.Resampling, resampling))
    west, south, east, north = array_bounds(dh, dw, dt)
    return out, (west, south, east, north)


def class_rgba(cls, colors, alpha=0.9):
    out = np.zeros(cls.shape + (4,))
    for k, c in colors.items():
        out[cls == k] = mcolors.to_rgba(c, alpha)
    return to_rgba(rgba=out)


def build_layers(hs, cls_shown, s2_rgb=None, forest=None, source_pot=None):
    """Dict name -> (RGBA uint8 in the analysis grid, resampling). Only layers whose input exists are returned."""
    L = {"hillshade": (to_rgba(np.dstack([np.nan_to_num(hs)] * 3), np.isfinite(hs).astype(float)), "bilinear"),
         "change": (class_rgba(cls_shown, CHANGE_COL), "nearest")}
    if s2_rgb is not None:
        L["baseline_rgb"] = (to_rgba(rgba=s2_rgb), "bilinear")             # RGBA, sea transparent
    if forest is not None:
        L["forest"] = (class_rgba(np.where(forest, 1, 0), {1: "#1a9850"}, 0.35), "nearest")
    if source_pot is not None:
        p = np.where(np.isfinite(source_pot), np.clip(source_pot, 0, 1), np.nan)
        rgba = cm.Purples(np.nan_to_num(p))
        rgba[..., 3] = np.where(np.isfinite(p), 0.25 + 0.55 * np.nan_to_num(p), 0)
        L["wood_source"] = (to_rgba(rgba=rgba), "nearest")
    return L


def write_overlays(out_dir, layers, transform, crs):
    """Write each layer as <name>.png and an overlays.json index of WGS84 bounds. Returns the index dict."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, (img, rs) in layers.items():
        warped, b = warp_rgba(img, transform, crs, rs)
        Image.fromarray(warped, "RGBA").save(out_dir / f"{name}.png", optimize=True)
        index[name] = dict(file=f"{name}.png", bounds=[float(v) for v in b], size=list(warped.shape[1::-1]))
    (out_dir / "overlays.json").write_text(json.dumps(index, indent=2))
    return index
