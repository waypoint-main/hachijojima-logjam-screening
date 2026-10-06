"""Optical (Sentinel-2) change detection — Phase 3.

Differences of median composites (after - baseline) for NDVI, NBR, NDMI, BSI, MNDWI, each minus its
island-wide median shift (reported). Flags (all cleaned to >= minimum patch size):

* vegetation_loss     : baseline NDVI >= veg_min AND anomaly dNDVI <= veg_loss_dNDVI AND dNBR <= veg_loss_dNBR
* forest_disturbance  : vegetation_loss within baseline FOREST (NDVI >= wood_model.forest_ndvi_min).
                        'Possible windthrow / canopy loss' — spectral change alone cannot separate
                        windthrow from defoliation, salt burn, harvest or landslide.
* exposed_soil        : anomaly dBSI >= threshold AND dNDVI <= threshold
* landslide_candidate : near-total vegetation removal (dNDVI <= landslide_dNDVI) with strong soil signal
                        (dBSI >= landslide_dBSI) on slope >= landslide_min_slope_deg. Still only a CANDIDATE:
                        a landslide scar, a debris-flow track and severe canopy stripping look alike.
* water_change        : |anomaly dMNDWI| >= threshold (rare at 10 m for narrow streams)

optical_score = clip(max(-aNDVI/s_ndvi, -aNBR/s_nbr, aBSI/s_bsi), 0, 1).

Caveat: S2 composites are medians, so a pixel with few clear observations (summit cloud) is less
reliable — n_clear is carried into the confidence score. NDVI saturates in dense forest; NBR/NDMI
are more sensitive to canopy loss and are required jointly with NDVI.
"""
from __future__ import annotations

import numpy as np

from .common import remove_small_patches, robust_shift


def optical_change(b: dict, a: dict, land: np.ndarray, slope10: np.ndarray | None, cfg: dict,
                   forest_ndvi_min: float, min_px: int, shift_method: str = "mode") -> dict:
    oc = cfg["optical"]
    valid = land & (b["n_clear"] >= oc["min_clear_obs"]) & (a["n_clear"] >= oc["min_clear_obs"])
    out, info, an = {"valid": valid}, {}, {}
    for n in ("NDVI", "NBR", "NDMI", "BSI", "MNDWI"):
        d = a[n] - b[n]
        shift = robust_shift(d, valid & (b["NDVI"] >= oc["veg_min_baseline_ndvi"]), shift_method)
        info[f"{n}_shift_median"] = robust_shift(d, valid & (b["NDVI"] >= oc["veg_min_baseline_ndvi"]), "median")
        an[n] = np.where(valid, d - shift, np.nan)
        out[f"anom{n}"] = an[n]
        info[f"{n}_regional_shift"] = shift
    veg = valid & (b["NDVI"] >= oc["veg_min_baseline_ndvi"])
    forest = valid & (b["NDVI"] >= forest_ndvi_min)
    veg_loss = veg & (an["NDVI"] <= oc["veg_loss_dNDVI"]) & (an["NBR"] <= oc["veg_loss_dNBR"])
    exposed = veg & (an["BSI"] >= oc["exposed_soil_dBSI"]) & (an["NDVI"] <= oc["exposed_soil_dNDVI"])
    # near-total vegetation removal + strong soil signal (stricter than 'exposed soil')
    scar = veg & (an["BSI"] >= oc["landslide_dBSI"]) & (an["NDVI"] <= oc["landslide_dNDVI"])
    water = valid & (np.abs(an["MNDWI"]) >= oc["water_dMNDWI"])
    clean = lambda m: remove_small_patches(m, min_px)
    veg_loss, exposed, water, scar = clean(veg_loss), clean(exposed), clean(water), clean(scar)
    forest_dist = veg_loss & forest
    slope_ok = (slope10 >= oc["landslide_min_slope_deg"]) if slope10 is not None else np.zeros_like(exposed)
    s = oc["score_saturation"]
    score = np.fmax.reduce([-an["NDVI"] / s["dNDVI"], -an["NBR"] / s["dNBR"], an["BSI"] / s["dBSI"]])
    out.update(optical_score=np.where(valid, np.clip(score, 0, 1), np.nan),
               veg_loss=veg_loss, forest_disturbance=forest_dist, exposed_soil=exposed,
               landslide_candidate=scar & slope_ok, water_change=water,
               optical_flag=veg_loss | exposed | water, baseline_forest=forest)
    px = 100.0  # 10 m pixels
    info.update(forest_area_km2=float(forest.sum() * px / 1e6),
                forest_disturbance_km2=float(forest_dist.sum() * px / 1e6),
                veg_loss_km2=float(veg_loss.sum() * px / 1e6),
                exposed_soil_km2=float(exposed.sum() * px / 1e6),
                landslide_candidate_km2=float(out["landslide_candidate"].sum() * px / 1e6),
                water_change_km2=float(water.sum() * px / 1e6),
                forest_disturbance_pct_of_forest=float(100 * forest_dist.sum() / max(forest.sum(), 1)))
    out["info"] = info
    return out
