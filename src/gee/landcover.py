"""ESA WorldCover (GEE: ESA/WorldCover/v200) tree-cover mask on the analysis grid.

Rationale: Phase 3 used baseline NDVI >= 0.6 as a stand-in for 'forest'. NDVI saturates and also
catches tall grass / crops; a land-cover product is a more defensible definition of woody vegetation.

Limitations: WorldCover v200 describes 2021 (not the 2025 baseline); it cannot see clearing or
regrowth since then. 10 m class boundaries are generalised. It is an input, not ground truth.
"""
from __future__ import annotations

from . import download, init


def fetch_worldcover(cfg, out_path):
    ee = init.initialize(cfg)
    lc = cfg["wood_model"]["landcover"]
    crs = cfg["study_area"]["analysis_crs"]
    bounds = download.analysis_bounds(cfg["study_area"]["bbox"], crs, snap=30.0)
    img = ee.ImageCollection(lc["asset"]).mosaic().select([lc["band"]]).toFloat()
    return download.download_image(img, bounds, crs, 10.0, out_path, [lc["band"]], cfg["gee"]["tile_px"])
