"""Woody-debris SOURCE areas (Phase 4). Pure numpy/scipy; no Earth Engine, no Streamlit.

SCIENTIFIC RATIONALE
  Large woody debris (LWD) in a river comes from (i) trees toppled or stripped near/into the channel
  (windthrow, bank erosion) and (ii) trees carried down by landslides / debris flows. Satellite
  change can only show WHERE canopy was lost, not how many stems fell or whether they moved.
  We therefore produce a transparent *source potential index* (0-1, NOT a wood volume):

      source_potential = disturbance_strength x terrain_mobility

  disturbance_strength : Phase 3 optical anomaly score (0-1) on pixels that Phase 3 classed as forest
                         disturbance or landslide candidate, in forest, above a confidence floor.
  terrain_mobility     : linear ramp of slope between wood_model.slope_factor_deg.low and .high
                         (steeper => felled wood more likely to move downslope).

ASSUMPTIONS (all [UNCALIBRATED])
  A1 Canopy-loss pixels in forest are potential wood sources; spectral change alone cannot separate
     windthrow from defoliation/salt burn/harvest.
  A2 Mobility rises linearly with slope between the configured bounds.
  A3 Euclidean distance to the derived stream network is only a *proximity descriptor* here; actual
     downslope delivery along flow paths is Phase 5.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

OCTANTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def forest_mask(worldcover: np.ndarray | None, baseline_ndvi: np.ndarray, land: np.ndarray, wm: dict):
    """Return (forest_mask, info). `worldcover` may be None (falls back to NDVI with a warning in info)."""
    ndvi_f = land & (baseline_ndvi >= wm["forest_ndvi_min"])
    mode = wm.get("forest_definition", "both")
    info = {"definition_requested": mode, "ndvi_forest_km2": float(ndvi_f.sum() * 100 / 1e6)}
    if worldcover is None or mode == "ndvi":
        if mode != "ndvi":
            info["warning"] = "WorldCover not available; used NDVI-only forest"
        info["definition_used"] = "ndvi"
        return ndvi_f, info
    wc_f = land & (worldcover == wm["landcover"]["tree_class"])
    inter = (wc_f & ndvi_f).sum()
    info.update(worldcover_forest_km2=float(wc_f.sum() * 100 / 1e6),
                worldcover_ndvi_agreement_iou=float(inter / max((wc_f | ndvi_f).sum(), 1)),
                worldcover_forest_not_ndvi_pct=float(100 * (wc_f & ~ndvi_f).sum() / max(wc_f.sum(), 1)))
    used = wc_f if mode == "worldcover" else (wc_f & ndvi_f)
    info["definition_used"] = mode
    info["forest_km2"] = float(used.sum() * 100 / 1e6)
    return used, info


def terrain_mobility(slope_deg: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((slope_deg - low) / (high - low), 0, 1)


def source_potential(cls: np.ndarray, conf: np.ndarray, optical_score: np.ndarray, forest: np.ndarray,
                     slope_deg: np.ndarray, wm: dict):
    """Per-pixel source mask and potential index (NaN outside sources)."""
    sc = wm["slope_factor_deg"]
    mob = terrain_mobility(slope_deg, sc["low"], sc["high"])
    in_class = np.isin(cls, wm["source_change_classes"])
    src = forest & in_class & (np.nan_to_num(conf, nan=0) >= wm["source_min_confidence"])
    strength = np.clip(np.nan_to_num(optical_score, nan=0.0), 0, 1)
    pot = np.where(src, strength * mob, np.nan).astype("float32")
    return src, pot, mob


def distance_to_streams(stream_mask: np.ndarray, cell_m: float) -> np.ndarray:
    if not stream_mask.any():
        return np.full(stream_mask.shape, np.nan, dtype="float32")
    return (ndimage.distance_transform_edt(~stream_mask) * cell_m).astype("float32")


def aspect_octant(aspect_deg: np.ndarray) -> np.ndarray:
    """0..7 = N, NE, ... NW; -1 where aspect is NaN."""
    a = np.where(np.isfinite(aspect_deg), aspect_deg, 0.0)
    o = (np.floor(((a + 22.5) % 360) / 45)).astype(int)
    return np.where(np.isfinite(aspect_deg), o, -1)


def exposure_table(forest: np.ndarray, src: np.ndarray, octant: np.ndarray) -> dict:
    """Share of forest area in each aspect octant that is a source pixel (a wind-exposure diagnostic)."""
    out = {}
    for i, n in enumerate(OCTANTS):
        f = (forest & (octant == i)).sum()
        out[n] = dict(forest_km2=float(f * 100 / 1e6),
                      source_pct_of_forest=float(100 * (src & (octant == i)).sum() / f) if f else None)
    return out


def label_patches(src: np.ndarray, pot: np.ndarray, conf: np.ndarray, slope: np.ndarray, cls: np.ndarray,
                  dist: np.ndarray, cell_m: float, min_area_m2: float):
    """Connected components (8-neighbour) of source pixels -> list of attribute dicts + label raster."""
    lab, n = ndimage.label(src, structure=np.ones((3, 3)))
    if n == 0:
        return lab, []
    idx = np.arange(1, n + 1)
    px = ndimage.sum(np.ones_like(lab), lab, idx)
    keep = px * cell_m * cell_m >= min_area_m2
    f = lambda a: ndimage.mean(np.nan_to_num(a), lab, idx)
    mn_d = ndimage.minimum(np.where(np.isfinite(dist), dist, 1e9), lab, idx)
    land_frac = ndimage.mean((cls == 3).astype(float), lab, idx)
    sum_pot = ndimage.sum(np.nan_to_num(pot), lab, idx)
    rows = []
    for i in range(n):
        if not keep[i]:
            continue
        rows.append(dict(label=i + 1, area_m2=float(px[i] * cell_m * cell_m),
                         mean_potential=float(f(pot)[i]), source_index_ha=float(sum_pot[i] * cell_m * cell_m / 1e4),
                         mean_slope_deg=float(f(slope)[i]), mean_confidence=float(f(conf)[i]),
                         min_dist_to_stream_m=float(mn_d[i]), landslide_fraction=float(land_frac[i])))
    lab = np.where(np.isin(lab, [r["label"] for r in rows]), lab, 0)
    return lab, rows
