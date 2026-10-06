"""SAR (Sentinel-1) change detection — Phase 3.

Method
------
For each polarisation (VV, VH):
  d      = median_dB(after) - median_dB(baseline)          (a log-ratio; same orbit geometry)
  shift  = mode(d) over valid land (median also reported)    (island-wide offset)
  anom   = d - shift                                        (local anomaly)
  SE     = 1.2533 * sqrt(sd_b^2/n_b + sd_a^2/n_a)           (sampling error of a difference of
                                                              medians; floored at se_floor_db)
  z      = anom / SE
A pixel is flagged when |z| >= z_threshold AND |anom| >= min_abs_anomaly_db, then cleaned
(1-px closing + minimum patch size). sar_score = clip(max|z| / z_saturation, 0, 1).

Rationale: using the per-pixel temporal spread makes the threshold adapt to naturally noisy
surfaces (wet slopes, surf zone) instead of one global dB cut-off; requiring a minimum dB size
avoids flagging statistically-significant-but-physically-trivial differences.

Limitations
-----------
* Detects a change in backscatter, not its cause: canopy loss, wet soil, new bare ground, debris
  and surf can all do it. Direction (+/-) is returned; interpretation happens in fusion.
* Composites hide persistence; `sar_persistent` is a proxy (every post scene beyond baseline by k std devs, via the min/max composites), not a per-scene count.
* C-band is weakly sensitive to under-canopy changes in dense forest.
"""
from __future__ import annotations

import numpy as np

from .common import remove_small_patches, robust_shift


def _se(sd_b, sd_a, n_b, n_a, floor):
    with np.errstate(divide="ignore", invalid="ignore"):
        se = 1.2533 * np.sqrt(sd_b**2 / n_b + sd_a**2 / n_a)
    return np.maximum(se, floor)


def sar_change(b: dict, a: dict, land: np.ndarray, cfg: dict, min_px: int,
               shift_method: str = "mode") -> dict:
    sc = cfg["sar"]
    valid = (land & (b["n_obs"] >= sc["min_obs"]) & (a["n_obs"] >= sc["min_obs"])
             & np.isfinite(b["VV_median"]) & np.isfinite(a["VV_median"])
             & np.isfinite(b["VH_median"]) & np.isfinite(a["VH_median"]))
    out, info = {"valid": valid}, {}
    flags = {}
    for pol in ("VV", "VH"):
        d = a[f"{pol}_median"] - b[f"{pol}_median"]
        shift = robust_shift(d, valid, shift_method)
        info[f"{pol}_shift_median_db"] = robust_shift(d, valid, "median")
        anom = d - shift
        se = _se(b[f"{pol}_stdDev"], a[f"{pol}_stdDev"], b["n_obs"], a["n_obs"], sc["se_floor_db"])
        z = anom / se
        f = valid & (np.abs(z) >= sc["z_threshold"]) & (np.abs(anom) >= sc["min_abs_anomaly_db"])
        flags[pol] = remove_small_patches(f, min_px, close=True)
        out.update({f"d{pol}_db": np.where(valid, d, np.nan), f"anom{pol}_db": np.where(valid, anom, np.nan),
                    f"z{pol}": np.where(valid, z, np.nan)})
        info[f"{pol}_regional_shift_db"] = shift
        info[f"{pol}_median_se_db"] = float(np.nanmedian(se[valid]))
        info[f"{pol}_flag_frac_before_cleanup"] = float(f[valid].mean())
    zmax = np.fmax(np.abs(out["zVV"]), np.abs(out["zVH"]))
    out["sar_score"] = np.where(valid, np.clip(zmax / sc["z_saturation"], 0, 1), np.nan)
    out["sar_flag"] = flags["VV"] | flags["VH"]
    out["sar_flag_both"] = flags["VV"] & flags["VH"]
    # direction from VH where flagged, else VV (+1 backscatter increase, -1 decrease)
    sign = np.where(flags["VH"], np.sign(out["anomVH_db"]), np.sign(out["anomVV_db"]))
    out["sar_dir"] = np.where(out["sar_flag"], np.nan_to_num(sign), 0).astype("int8")
    # persistence proxy (composites only): the flag is "persistent" if EVEN the most extreme post-event scene
    # (VH min for increases, VH max for decreases) is beyond the baseline median by >= k baseline std devs.
    k = sc.get("persistence_sd_k", 1.0)
    sd_b = np.maximum(b["VH_stdDev"], sc["se_floor_db"])
    if "VH_min" in a and "VH_max" in a:
        with np.errstate(invalid="ignore"):
            pers_inc = (out["sar_dir"] > 0) & ((a["VH_min"] - b["VH_median"]) >= k * sd_b)
            pers_dec = (out["sar_dir"] < 0) & ((b["VH_median"] - a["VH_max"]) >= k * sd_b)
        out["sar_persistent"] = pers_inc | pers_dec
    else:                                   # composites without min/max bands: persistence unknown -> False
        out["sar_persistent"] = np.zeros_like(out["sar_flag"])
    info["sar_persistent_frac_of_flagged"] = float(out["sar_persistent"].sum() / max(out["sar_flag"].sum(), 1))
    info["sar_flag_frac_clean"] = float(out["sar_flag"][valid].mean())
    info["sar_flag_both_frac_clean"] = float(out["sar_flag_both"][valid].mean())
    info["valid_land_px"] = int(valid.sum())
    out["info"] = info
    return out
