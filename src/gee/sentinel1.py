"""Sentinel-1 composites on Earth Engine (Phase 2) — built-in implementation.

(You can still plug in your own preprocessing: set `gee.sentinel1.mode: user_module` and see
`user_sentinel1_TEMPLATE.py`.)

Scientific design
-----------------
* Source: COPERNICUS/S1_GRD, IW, VV+VH. GEE provides thermal-noise removal, calibration and
  orthorectification in dB. It is NOT guaranteed to be radiometrically slope-flattened, and
  Hachijojima's volcanic cones are steep, so we do not rely on absolute backscatter:
    1. ONE orbit geometry (pass + relative orbit) for baseline AND after. Auto-selected as the
       geometry with the most observations in both windows (`pick_orbit`).
    2. Change is the *difference of dB composites of the same geometry* (a log-ratio), so the
       static terrain/geometry term largely cancels.
    3. An approximate DEM-based layover/shadow mask removes pixels where SAR cannot see the surface.
* Border noise / no-data / swath edges masked by dB and incidence-angle limits; sea masked by DEM.
* Composites are temporal median/min/max/stdDev of dB values (median/min/max are invariant to the
  dB transform; stdDev in dB is a variability index, not a physical quantity). Median stacks suppress
  speckle; an optional small focal median is applied afterwards.
* Output bands per window: {VV,VH,VVVH}_{median,min,max,stdDev} + n_obs.
  VVVH = VV − VH in dB (= VV/VH ratio). stdDev of the BASELINE is later used for robust z-scores.

Known limitations (documented, not hidden): C-band saturates under dense canopy so windthrow and
under-canopy debris are weakly visible; the layover/shadow mask uses nominal satellite headings;
wet-soil/rain and surf can change backscatter without any geomorphic change.
"""
from __future__ import annotations

import importlib
import logging
import math

log = logging.getLogger(__name__)

STATS = ["median", "min", "max", "stdDev"]
BASE_BANDS = ["VV", "VH", "VVVH"]
OUT_BANDS = [f"{b}_{s}" for b in BASE_BANDS for s in STATS] + ["n_obs"]


def pick_orbit(counts_base: dict, counts_after: dict, min_obs: int = 5):
    """Choose the geometry key 'PASS_relorbit' with the best worst-case count over both windows.

    Pure function (unit-tested). Returns (pass, relative_orbit:int, info) or raises if none qualify.
    """
    keys = set(counts_base) & set(counts_after)
    scored = sorted(((min(counts_base[k], counts_after[k]), counts_base[k] + counts_after[k], k)
                     for k in keys), reverse=True)
    if not scored or scored[0][0] < min_obs:
        raise RuntimeError(f"No single Sentinel-1 geometry has >= {min_obs} obs in both windows. "
                           f"baseline={counts_base} after={counts_after}")
    _, _, key = scored[0]
    p, r = key.rsplit("_", 1)
    return p, int(r), {"selected": key, "counts_base": counts_base, "counts_after": counts_after}


def _month_filter(ee, months):
    if not months:
        return None
    return ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months])


def base_collection(ee, cfg, aoi, start, end, months=None):
    s1 = cfg["gee"]["sentinel1"]
    col = (ee.ImageCollection(s1["collection"]).filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", s1["instrument_mode"]))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    mf = _month_filter(ee, months)
    return col.filter(mf) if mf is not None else col


def orbit_inventory(ee, cfg, aoi, start, end, months=None) -> dict:
    col = base_collection(ee, cfg, aoi, start, end, months).map(
        lambda i: i.set("geom", ee.String(i.get("orbitProperties_pass")).cat("_").cat(
            ee.Number(i.get("relativeOrbitNumber_start")).format("%d"))))
    return col.aggregate_histogram("geom").getInfo() or {}


def _terrain(ee, cfg):
    """Slope/aspect (deg) computed at the DEM's native projection, then resampled."""
    from ..data.dem import dem_mosaic
    dem = dem_mosaic(cfg, ee)
    proj = dem.projection()
    slope = ee.Terrain.slope(dem).reproject(proj).resample("bilinear")
    aspect = ee.Terrain.aspect(dem).reproject(proj).resample("bilinear")
    land = dem.gt(cfg["gee"]["land_mask_dem_min_m"])
    return slope, aspect, land


def _geometry_mask(ee, img, slope, aspect, look_az_deg, margin):
    """Approximate layover/shadow mask (cf. Vollrath et al. 2020).

    alpha = terrain tilt toward the sensor in the range direction.
    layover if alpha > theta_i (incidence); shadow if alpha < -(90 - theta_i). `margin` widens both.
    """
    rad = math.pi / 180.0
    theta = img.select("angle")
    asp, sl = aspect.multiply(rad), slope.multiply(rad)
    alpha = sl.tan().multiply(asp.subtract(look_az_deg * rad).cos().multiply(-1)).atan().divide(rad)
    layover = alpha.gt(theta.subtract(margin))
    shadow = alpha.lt(theta.subtract(90).add(margin))
    return layover.Or(shadow).Not()


def build_collection(ee, cfg, aoi, start, end, orbit_pass, relative_orbit, months=None):
    s1 = cfg["gee"]["sentinel1"]
    if s1["mode"] == "user_module":
        fn = getattr(importlib.import_module(s1["user_module"]), s1["user_function"])
        col = fn(aoi, start, end, orbit_pass=orbit_pass, relative_orbit=relative_orbit)
        return col.map(lambda i: i.addBands(i.select("VV").subtract(i.select("VH")).rename("VVVH")))

    col = (base_collection(ee, cfg, aoi, start, end, months)
           .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
           .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
           .select(["VV", "VH", "angle"]))
    slope, aspect, land = _terrain(ee, cfg)
    look = s1["heading_deg"][orbit_pass] + 90.0

    def prep(img):
        vv, vh, ang = img.select("VV"), img.select("VH"), img.select("angle")
        m = (vv.gt(s1["vv_min_db"]).And(vh.gt(s1["vh_min_db"]))
             .And(ang.gte(s1["incidence_min_deg"])).And(ang.lte(s1["incidence_max_deg"])).And(land))
        if s1["mask_layover_shadow"]:
            m = m.And(_geometry_mask(ee, img, slope, aspect, look, s1["layover_shadow_margin_deg"]))
        out = img.updateMask(m)
        return out.addBands(out.select("VV").subtract(out.select("VH")).rename("VVVH"))

    return col.map(prep)


def composite(ee, col, cfg):
    s1 = cfg["gee"]["sentinel1"]
    red = (ee.Reducer.median().combine(ee.Reducer.min(), "", True)
           .combine(ee.Reducer.max(), "", True).combine(ee.Reducer.stdDev(), "", True))
    comp = col.select(BASE_BANDS).reduce(red)
    r = s1.get("speckle_focal_radius_px", 0)
    if r and r > 0:
        comp = comp.focalMedian(r, "circle", "pixels")
    n = col.select("VV").count().rename("n_obs")
    return comp.addBands(n).select(OUT_BANDS).toFloat()


def build_windows(ee, cfg, aoi):
    """Return ({'baseline': image, 'after': image}, meta) using ONE orbit geometry."""
    p = cfg["periods"]
    months = cfg["periods"].get("season_months")
    s1 = cfg["gee"]["sentinel1"]
    win = {"baseline": (str(p["baseline_start"]), str(p["baseline_end"])),
           "after": (str(p["after_start"]), str(p["after_end"]))}
    meta = {"windows": win, "season_months": months}
    if s1["mode"] == "user_module" or (s1["orbit_pass"] and s1["relative_orbit"]):
        op, ro = s1["orbit_pass"], s1["relative_orbit"]
        meta["orbit_selection"] = "from config / user module"
    else:
        cb = orbit_inventory(ee, cfg, aoi, *win["baseline"], months)
        ca = orbit_inventory(ee, cfg, aoi, *win["after"], months)
        op, ro, info = pick_orbit(cb, ca, s1["min_obs_per_period"])
        meta["orbit_selection"] = info
    meta.update(orbit_pass=op, relative_orbit=ro)
    out = {}
    for k, (a, b) in win.items():
        col = build_collection(ee, cfg, aoi, a, b, op, ro, months)
        meta[f"n_scenes_{k}"] = col.size().getInfo()
        out[k] = composite(ee, col, cfg)
    log.info("S1 meta: %s", meta)
    return out, meta
