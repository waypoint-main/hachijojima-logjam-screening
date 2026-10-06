"""Sentinel-2 composites on Earth Engine (Phase 2).

* Surface reflectance: COPERNICUS/S2_SR_HARMONIZED (harmonised across the 2022 processing
  baseline change, so 2025 and 2026 are consistent).
* Cloud/shadow masking: Google Cloud Score+ (`cs_cdf` >= threshold, catalog suggests 0.50-0.65)
  linked by system:index. Chosen over s2cloudless because it handles thin cloud/shadow better in
  humid, hazy scenes, which is Hachijojima's normal condition. A scene-level cloud % pre-filter is
  coarse only.
* Composite = per-pixel MEDIAN of clear observations in each window (robust to residual cloud).
  `n_clear` (count of clear obs) is exported and feeds the confidence score.
* Bands exported: B2 B3 B4 B8 B11 B12 (reflectance 0-1), NDVI, NDMI, NBR, MNDWI, BSI, n_clear.
    NDVI  = (B8-B4)/(B8+B4)          vegetation vigour
    NDMI  = (B8-B11)/(B8+B11)        canopy moisture
    NBR   = (B8-B12)/(B8+B12)        burn / disturbance; dNBR also flags canopy loss
    MNDWI = (B3-B11)/(B3+B11)        open water
    BSI   = ((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))   bare soil
Limitations: a median over a whole year mixes seasons — use `periods.season_months` to compare like
with like. Sub-pixel debris is invisible; only canopy-scale disturbance and exposed material show.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

REFL = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDICES = ["NDVI", "NDMI", "NBR", "MNDWI", "BSI"]
OUT_BANDS = REFL + INDICES + ["n_clear"]


def build_collection(ee, cfg, aoi, start, end, months=None):
    s2 = cfg["gee"]["sentinel2"]
    col = (ee.ImageCollection(s2["collection"]).filterBounds(aoi).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s2["max_scene_cloud_pct"])))
    if months:
        col = col.filter(ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month") for m in months]))
    cs = ee.ImageCollection(s2["cloud_score_collection"])
    col = col.linkCollection(cs, [s2["cs_band"]])
    thr, csb = s2["cs_threshold"], s2["cs_band"]

    def prep(img):
        clear = img.select(csb).gte(thr)
        r = img.select(REFL).divide(10000).updateMask(clear)
        nd = lambda a, b, n: r.normalizedDifference([a, b]).rename(n)
        bsi = (r.select("B11").add(r.select("B4")).subtract(r.select("B8").add(r.select("B2")))
               .divide(r.select("B11").add(r.select("B4")).add(r.select("B8").add(r.select("B2"))))
               .rename("BSI"))
        out = r.addBands([nd("B8", "B4", "NDVI"), nd("B8", "B11", "NDMI"),
                          nd("B8", "B12", "NBR"), nd("B3", "B11", "MNDWI"), bsi])
        # arithmetic drops image properties; restore the ones needed for dates/ids
        return out.copyProperties(img, ["system:time_start", "system:index"])

    return col.map(prep)


def composite(ee, col):
    med = col.select(REFL + INDICES).median()
    n = col.select("NDVI").count().rename("n_clear")
    return med.addBands(n).select(OUT_BANDS).toFloat()


def build_windows(ee, cfg, aoi):
    p = cfg["periods"]
    months = p.get("season_months")
    win = {"baseline": (str(p["baseline_start"]), str(p["baseline_end"])),
           "after": (str(p["after_start"]), str(p["after_end"]))}
    out, meta = {}, {"windows": win, "season_months": months}
    for k, (a, b) in win.items():
        col = build_collection(ee, cfg, aoi, a, b, months)
        meta[f"n_scenes_{k}"] = col.size().getInfo()
        out[k] = composite(ee, col)
    log.info("S2 meta: %s", meta)
    return out, meta
