"""Observed river-corridor change (Phase 7). Pure numpy/pandas; no Earth Engine, no Streamlit.

WHAT IT MEASURES
  How much satellite-observed change (Phase 3) lies in the corridor around each reach (default 50 m),
  and whether the change inside the channel/bank core (15 m) looks like debris or an obstruction.
  This is OBSERVATION of change, kept separate from hazard susceptibility (Phase 6).

SCORE
  observed_change_score = sum_i w_i * v_i / sum_i w_i over the AVAILABLE metrics (each 0-1 by a linear ramp):
    sar_change (fraction of valid corridor px with a SAR flag), sar_persistence, optical_vegetation_loss,
    optical_exposed_material, water_change (coastal zone excluded), spatial_concentration.
  channel_morphology is NOT measurable at 10 m (streams are narrower than a pixel) and is excluded;
  the remaining weights are renormalised.

DEBRIS / OBSTRUCTION EVIDENCE (hypotheses, never confirmation)
  Probable Debris Accumulation : in the core zone, a PERSISTENT SAR backscatter INCREASE agrees with optical
                                 disturbance (vegetation loss / exposed material within 1 px), compact patch >= N px.
  Possible Logjam              : change concentrated in the core zone relative to the surrounding corridor ring
                                 (channel-specific, not general slope damage), compact, on a reach with susceptibility >= threshold.
  Rationale: stranded wood and debris are rough, bright scatterers (backscatter up) and strip vegetation or expose
  bare material at banks; but canopy damage, wet soil, roads and surf can produce the same signals. Sentinel-1/2
  at 10 m can only see jams/accumulations that are tens of metres across.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

from .logjam_susceptibility import classify, ramp

METRICS = ["sar_change", "sar_persistence", "optical_vegetation_loss", "optical_exposed_material", "water_change",
           "spatial_concentration"]


def _count(rid, mask, n):
    return np.bincount(rid[mask], minlength=n + 1)[1:].astype(float)


def largest_patch_px(rid, mask, n, structure=None):
    """Per reach: size (px) of the largest 8-connected patch of `mask` inside that reach's cells, and total px."""
    structure = np.ones((3, 3)) if structure is None else structure
    lab, L = ndimage.label(mask, structure=structure)
    out_max, out_tot = np.zeros(n), np.zeros(n)
    m = (lab > 0) & (rid > 0)
    if not m.any():
        return out_max, out_tot
    key = rid[m].astype("int64") * (L + 1) + lab[m]
    u, c = np.unique(key, return_counts=True)
    df = pd.DataFrame(dict(r=u // (L + 1), c=c))
    g = df.groupby("r")["c"].agg(["max", "sum"])
    out_max[g.index.to_numpy() - 1] = g["max"].to_numpy()
    out_tot[g.index.to_numpy() - 1] = g["sum"].to_numpy()
    return out_max, out_tot


def corridor_metrics(rid, dist_m, valid, cls, sar_flag, sar_dir, sar_persist, coast_m, n, oc):
    """Raw per-reach fractions and counts. All inputs are 10 m grids (`rid`: nearest reach id 1..n)."""
    cor = valid & (dist_m <= oc["corridor_buffer_m"]) & (rid > 0)
    core = valid & (dist_m <= oc["core_buffer_m"]) & (rid > 0)
    nv = _count(rid, cor, n)
    veg = np.isin(cls, [1, 2]); exp = np.isin(cls, [3, 4])
    water = (cls == 5) & (coast_m >= oc["coastal_exclusion_m"])
    anyc = cor & (veg | exp | water | sar_flag | (cls == 6))
    mx, tot = largest_patch_px(rid, anyc, n)
    d = pd.DataFrame(dict(n_valid_corridor_px=nv))
    frac = lambda m: np.where(nv > 0, _count(rid, cor & m, n) / np.maximum(nv, 1), np.nan)
    d["sar_change"] = frac(sar_flag)
    d["sar_persistence"] = frac(sar_persist.astype(bool))
    d["optical_vegetation_loss"] = frac(veg)
    d["optical_exposed_material"] = frac(exp)
    d["water_change"] = frac(water)
    d["any_change_frac"] = np.where(nv > 0, tot / np.maximum(nv, 1), np.nan)
    share = np.where(tot > 0, mx / np.maximum(tot, 1), 0.0)
    d["largest_patch_share"] = share
    d["spatial_concentration"] = np.where(d["any_change_frac"] >= oc["ramps"]["spatial_concentration_min_frac"], share, 0.0)
    d["n_core_px"] = _count(rid, core, n)
    return d, cor, core


def score_observed(metrics: pd.DataFrame, oc: dict):
    """-> (score, class, used_weight_fraction, driver table). Reaches with too few valid px get NaN."""
    r, w = oc["ramps"], oc["weights"]
    v = pd.DataFrame(index=metrics.index)
    for k in METRICS:
        if k == "spatial_concentration":
            v[k] = metrics[k].fillna(0).clip(0, 1)
        else:
            v[k] = ramp(metrics[k].to_numpy(), r[k]["lo"], r[k]["hi"])
    ww = np.array([w[k] for k in METRICS], float)
    used = ww.sum() / sum(w.values())
    score = (v[METRICS].to_numpy() * ww).sum(axis=1) / ww.sum()
    score = np.where(metrics["n_valid_corridor_px"] >= oc["min_valid_corridor_px"], score, np.nan)
    labels = ["Low", "Moderate", "High", "Very High"]
    klass = np.where(np.isfinite(score), classify(np.nan_to_num(score), oc["class_breaks"], labels), "Insufficient data")
    return score, klass, float(used), v


def debris_evidence(rid, core, ring, chg, cls, sar_dir, sar_persist, n, dc, px_m2, susceptibility):
    """Per-reach debris/obstruction flags. Returns (DataFrame, probable_pixel_mask, core_change_mask).

    core  : channel + immediate bank pixels;  ring : the rest of the corridor (hillslope side)
    chg   : any observed change pixel (optical classes 1-4 or a SAR flag)

    Probable Debris Accumulation: persistent SAR backscatter increase AND optical disturbance in the core, compact.
    Possible Logjam: change concentrated in the core relative to the surrounding ring (channel-specific, not
        general slope damage), compact, on a reach with susceptibility >= threshold.
    NOTE: an earlier version flagged any compact core anomaly; 94% of those were ordinary riparian canopy loss in
    storm-damaged catchments, so it did not discriminate. The core-vs-ring contrast is what makes the flag meaningful.
    """
    inc_pers = (sar_dir == 1) & sar_persist.astype(bool)
    sar_px = core & inc_pers
    opt_raw = core & np.isin(cls, [1, 2, 3, 4])
    opt_dil = ndimage.binary_dilation(opt_raw, iterations=dc["optical_dilate_px"]) & core
    probable_px = sar_px & ndimage.binary_dilation(opt_raw, iterations=dc["optical_dilate_px"])
    core_chg = core & chg
    pm, _ = largest_patch_px(rid, probable_px, n)
    om, ot = largest_patch_px(rid, core_chg, n)
    ncore, nring = _count(rid, core, n), _count(rid, ring, n)
    cf = np.where(ncore > 0, _count(rid, core_chg, n) / np.maximum(ncore, 1), 0.0)
    rf_ = np.where(nring > 0, _count(rid, ring & chg, n) / np.maximum(nring, 1), 0.0)
    ratio = cf / np.maximum(rf_, dc["ring_floor_frac"])
    d = pd.DataFrame(dict(debris_sar_px=_count(rid, sar_px, n), debris_optical_px=_count(rid, opt_dil, n),
                          debris_agreement_px=_count(rid, probable_px, n), debris_patch_px=pm, one_sensor_patch_px=om,
                          core_change_frac=cf, ring_change_frac=rf_, channel_contrast_ratio=ratio))
    d["debris_area_m2"] = d["debris_agreement_px"] * px_m2
    share = np.where(ot > 0, om / np.maximum(ot, 1), 0.0)
    probable = (d["debris_agreement_px"] * px_m2 >= dc["probable_min_area_m2"]) & (pm >= dc["probable_min_patch_px"])
    possible = ((om * px_m2 >= dc["possible_min_area_m2"]) & (om >= dc["possible_min_patch_px"]) &
                (share >= dc["possible_min_patch_share"]) & (cf >= dc["possible_min_core_change_frac"]) &
                (ratio >= dc["possible_min_contrast_ratio"]) & (np.asarray(susceptibility) >= dc["possible_min_susceptibility"]))
    d["debris_flag"] = np.where(probable, "Probable Debris Accumulation", np.where(possible, "Possible Logjam", ""))
    return d, probable_px, core_chg
