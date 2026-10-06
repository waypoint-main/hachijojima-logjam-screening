"""MERIT Hydro (GEE: MERIT/Hydro/v1_0_1) as a COARSE reference for the derived river network.

MERIT Hydro is ~93 m, built from a 90 m SRTM-based DEM. Most Hachijojima streams are narrower and
shorter than its pixel size, so low agreement is expected and is NOT evidence that the derived
network is wrong. Use it for (i) a sanity check of network position and (ii) a scale check of
contributing area (MERIT `upa` km² vs. derived upstream_area_m2).
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.spatial import cKDTree

from . import download, init

BANDS = ["upa", "wth", "hnd", "elv", "dir"]


def fetch_merit(cfg, out_path):
    ee = init.initialize(cfg)
    crs = cfg["study_area"]["analysis_crs"]
    asset = cfg["hydrology"]["reference_hydrography"]["merit"]["asset"]
    bounds = download.analysis_bounds(cfg["study_area"]["bbox"], crs, snap=90.0)
    img = ee.Image(asset).select(BANDS).toFloat()
    return download.download_image(img, bounds, crs, 90.0, out_path, BANDS, cfg["gee"]["tile_px"])


def river_cell_xy(merit_path, upa_min_km2):
    with rasterio.open(merit_path) as src:
        upa = src.read(1)
        t = src.transform
    rr, cc = np.where(np.isfinite(upa) & (upa >= upa_min_km2))
    xy = np.column_stack([t.c + (cc + 0.5) * t.a, t.f + (rr + 0.5) * t.e])
    return xy, upa, t


def _densify(geoms, step):
    pts = []
    for g in geoms:
        n = max(2, int(g.length // step) + 1)
        pts.extend(g.interpolate(i / (n - 1), normalized=True).coords[0] for i in range(n))
    return np.asarray(pts)


def compare_with_merit(streams, reaches, merit_path, upa_min_km2: float, tol_m: float) -> dict:
    """Agreement of derived streams with MERIT river cells + scale check of upstream area."""
    xy, upa, t = river_cell_xy(merit_path, upa_min_km2)
    out = dict(merit_river_cells=int(len(xy)), tolerance_m=tol_m, upa_min_km2=upa_min_km2)
    if len(xy) == 0 or len(streams) == 0:
        out.update(derived_within_tol=float("nan"), merit_within_tol=float("nan"),
                   note="No MERIT river cells at this threshold — expected on very small catchments.")
        return out
    dpts = _densify(streams.geometry, 15.0)
    d_to_m, _ = cKDTree(xy).query(dpts)
    m_to_d, _ = cKDTree(dpts).query(xy)
    out.update(derived_within_tol=float((d_to_m <= tol_m).mean()),
               merit_within_tol=float((m_to_d <= tol_m).mean()))
    # scale check: max MERIT upa in 3x3 around each reach outlet vs derived upstream area
    ratios = []
    for g, a in zip(reaches.geometry, reaches["upstream_area_m2"]):
        x, y = g.coords[-1][:2]
        c, r = int((x - t.c) // t.a), int((y - t.f) // t.e)
        win = upa[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
        if win.size and np.isfinite(win).any() and np.nanmax(win) >= upa_min_km2:
            ratios.append(np.log10(max(a, 1.0) / (np.nanmax(win) * 1e6)))
    if ratios:
        out.update(n_reach_outlets_compared=len(ratios), median_log10_area_ratio=float(np.median(ratios)),
                   note="log10(derived/MERIT) upstream area; 0 = agreement, ±0.3 = factor 2")
    return out
