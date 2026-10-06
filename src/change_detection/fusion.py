"""Fuse SAR and optical change into classes + a confidence index (Phase 3).

OBSERVED satellite change only. Hazard/susceptibility inference is NOT done here.

Class codes (uint8; 0 = no meaningful change, 255 = no data):
  1 forest_disturbance (possible windthrow / canopy loss)
  2 vegetation_loss (non-forest vegetation)
  3 landslide_candidate (exposed soil on steep slope)
  4 exposed_soil_or_sediment (exposed soil on gentler slope)
  5 water_extent_change
  6 sar_only_change (UNCERTAIN: backscatter change without optical support)
  7 weak_optical_change (UNCERTAIN: sub-threshold optical anomaly, no SAR support)
  [8 river_corridor_disturbance, 9 possible_debris_accumulation: assigned in Phase 7 using reaches]

Priority when flags overlap: 3 > 1 > 4 > 2 > 5 (most geomorphically specific first).

confidence (0-1) is a transparent heuristic index (weights in config.change.pixel_confidence):
  magnitude (max of SAR/optical scores) + sensor agreement (SAR flag within `agreement_dilate_px` of
  an optical flag) + Sentinel-1 and Sentinel-2 observation counts. It is NOT a calibrated probability.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .common import remove_small_patches

CLASS_NAMES = {0: "no_meaningful_change", 1: "forest_disturbance", 2: "vegetation_loss",
               3: "landslide_candidate", 4: "exposed_soil_or_sediment", 5: "water_extent_change",
               6: "sar_only_change_uncertain", 7: "weak_optical_change_uncertain",
               8: "river_corridor_disturbance", 9: "possible_debris_accumulation"}


def fuse(sar: dict, opt: dict, s1b: dict, s1a: dict, s2b: dict, s2a: dict, cfg: dict, min_px: int):
    valid = sar["valid"] | opt["valid"]
    cls = np.zeros(valid.shape, dtype="uint8")
    cls[opt["water_change"]] = 5
    cls[opt["veg_loss"]] = 2
    cls[opt["exposed_soil"]] = 4
    cls[opt["forest_disturbance"]] = 1
    cls[opt["landslide_candidate"]] = 3
    # SAR-only: SAR flag with no optical flag nearby
    near_opt = ndimage.binary_dilation(opt["optical_flag"], iterations=cfg["agreement_dilate_px"] + 1)
    sar_only = sar["sar_flag"] & ~near_opt & (cls == 0)
    cls[sar_only] = 6
    # weak optical: optical score >= 0.5 but not flagged and no SAR flag
    weak = remove_small_patches((opt["optical_score"] >= 0.5) & (cls == 0) & ~sar["sar_flag"], min_px)
    cls[weak] = 7
    cls[~valid] = 255

    near_sar = ndimage.binary_dilation(sar["sar_flag"], iterations=cfg["agreement_dilate_px"])
    agree = near_sar & opt["optical_flag"]
    one = (sar["sar_flag"] | opt["optical_flag"]) & ~agree
    agreement = np.where(agree, 1.0, np.where(one, 0.4, 0.0))
    mag = np.fmax(np.nan_to_num(sar["sar_score"]), np.nan_to_num(opt["optical_score"]))
    n1 = np.clip(np.minimum(s1b["n_obs"], s1a["n_obs"]) / cfg["s1_obs_full"], 0, 1)
    n2 = np.clip(np.minimum(s2b["n_clear"], s2a["n_clear"]) / cfg["s2_obs_full"], 0, 1)
    w = cfg["pixel_confidence"]
    conf = (w["magnitude"] * mag + w["sensor_agreement"] * agreement +
            w["s1_observations"] * np.nan_to_num(n1) + w["s2_observations"] * np.nan_to_num(n2))
    conf = np.where((cls > 0) & (cls != 255), np.clip(conf, 0, 1), np.nan).astype("float32")
    info = {"class_area_km2": {CLASS_NAMES[k]: float((cls == k).sum() * 100 / 1e6) for k in range(1, 8)},
            "sensor_agreement_px": int(agree.sum()),
            "optical_flag_with_sar_support_pct": float(100 * agree.sum() / max(opt["optical_flag"].sum(), 1)),
            "sar_flag_with_optical_support_pct": float(100 * (sar["sar_flag"] & near_opt).sum() / max(sar["sar_flag"].sum(), 1)),
            "median_confidence_changed": float(np.nanmedian(conf)) if np.isfinite(conf).any() else None}
    return cls, conf, info
