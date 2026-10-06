"""Per-scene island statistics from Earth Engine for exposed vs sheltered forest slopes.

Masks (all within baseline forest, NDVI>=forest_ndvi_min of the baseline composite, slope >= min_slope):
  exposed   : aspect 0-135 deg  (N-E facing; the first run showed far more change here)
  sheltered : aspect 180-270 deg (S-W facing)
Exposure sectors are CONFIG-FREE defaults here because they are a diagnostic, not a model input; edit
EXPOSED/SHELTERED below if the event direction turns out different.

Output rows: date, <mask>_NDVI, <mask>_NBR, <mask>_n (valid pixels) for Sentinel-2;
             date, <mask>_VH, <mask>_VV, <mask>_n for Sentinel-1 (single orbit geometry).
"""
from __future__ import annotations

import logging

import pandas as pd

from . import sentinel1, sentinel2

log = logging.getLogger(__name__)
EXPOSED = (0, 135)
SHELTERED = (180, 270)


def build_masks(ee, cfg, aoi, min_slope=15.0):
    slope, aspect, land = sentinel1._terrain(ee, cfg)
    p = cfg["periods"]
    base = sentinel2.composite(ee, sentinel2.build_collection(ee, cfg, aoi, str(p["baseline_start"]), str(p["baseline_end"])))
    forest = base.select("NDVI").gte(cfg["wood_model"]["forest_ndvi_min"]).And(land).And(slope.gte(min_slope))
    inr = lambda lo, hi: aspect.gte(lo).And(aspect.lt(hi))
    return {"exposed": forest.And(inr(*EXPOSED)), "sheltered": forest.And(inr(*SHELTERED))}


def _stats_fn(ee, masks, bands, region, scale):
    red = ee.Reducer.mean().combine(ee.Reducer.count(), "", True)

    def f(img):
        d = ee.Dictionary({"date": img.date().format("YYYY-MM-dd")})
        for name, m in masks.items():
            r = img.select(bands).updateMask(m).reduceRegion(red, region, scale, maxPixels=1e9, tileScale=4)
            d = d.combine(r.rename(r.keys(), r.keys().map(lambda k: ee.String(name).cat("_").cat(ee.String(k)))))
        return ee.Feature(None, d)
    return f


def _chunks(start, end, days):
    ts = pd.date_range(start, end, freq=f"{days}D")
    edges = list(ts) + [pd.Timestamp(end)]
    return [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def _tidy(df):
    """reduceRegion names are '<mask>_<band>_mean' / '_count'; keep '<mask>_<band>' for the means."""
    df.columns = [c[:-5] if c.endswith("_mean") else c for c in df.columns]
    return df


def _fetch(ee, fc, tries=5):
    """getInfo with back-off on Earth Engine's 'Too many concurrent aggregations' / 429 quota errors."""
    import time
    for k in range(tries):
        try:
            return [f["properties"] for f in fc.getInfo()["features"]]
        except ee.EEException as e:
            if "concurrent" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                wait = 15 * (k + 1)
                log.warning("EE quota/concurrency limit; waiting %ds (try %d/%d)", wait, k + 1, tries)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Earth Engine kept refusing (concurrency limit). Re-run later or use a smaller chunk.")


def s2_series(ee, cfg, aoi, masks, start, end, scale=30, chunk_days=10):
    rows = []
    for a, b in _chunks(start, end, chunk_days):
        col = sentinel2.build_collection(ee, cfg, aoi, a, b)
        fc = col.map(_stats_fn(ee, masks, ["NDVI", "NBR"], aoi, scale))
        rows += _fetch(ee, fc)
        log.info("S2 %s..%s: %d scenes", a, b, len(rows))
    return _tidy(pd.DataFrame(rows))


def s1_series(ee, cfg, aoi, masks, start, end, orbit_pass, rel_orbit, scale=30, chunk_days=36):
    rows = []
    for a, b in _chunks(start, end, chunk_days):
        col = sentinel1.build_collection(ee, cfg, aoi, a, b, orbit_pass, rel_orbit)
        fc = col.map(_stats_fn(ee, masks, ["VV", "VH"], aoi, scale))
        rows += _fetch(ee, fc)
        log.info("S1 %s..%s: %d scenes", a, b, len(rows))
    return _tidy(pd.DataFrame(rows))
