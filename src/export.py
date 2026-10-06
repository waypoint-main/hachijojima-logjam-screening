"""Final river_reaches export (GeoPackage + GeoJSON) with the agreed schema. No Streamlit dependency."""
from __future__ import annotations

import numpy as np

REQUIRED = ["reach_id", "susceptibility_score", "susceptibility_class", "observed_change_score", "observed_change_class",
            "priority_score", "priority_class", "confidence_score", "confidence_class", "upstream_disturbed_area",
            "wood_delivery_score", "sar_change", "optical_change", "local_slope", "flow_accumulation", "bridge_or_crossing", "notes"]
EXTRA = ["debris_flag", "inspection_labels", "stream_order", "length_m", "upstream_area_m2", "small_catchment", "upstream_disturbed_pct",
         "upstream_landslide_area_m2", "riparian_disturbed_frac", "crossing_type", "channel_slope_m_per_m", "confluence",
         "channel_contrast_ratio", "core_change_frac", "ring_change_frac", "susceptibility_drivers"]


def standard_reaches(reaches):
    """Add the agreed column names (units: areas in m2, slopes in m/m, accumulation in cells)."""
    g = reaches.copy()
    for new, old in (("upstream_disturbed_area", "upstream_disturbed_area_m2"), ("local_slope", "local_slope_window_m_per_m"),
                     ("flow_accumulation", "flow_accumulation_cells")):
        g[new] = g[old] if old in g else np.nan           # never invent a value when the source column is absent
    if "bridge_or_crossing" not in g:                       # older Phase 1 outputs: rebuild from the component flags
        parts = [g[c].astype(bool) for c in ("road_crossing", "bridge", "culvert") if c in g]
        g["bridge_or_crossing"] = np.logical_or.reduce(parts) if parts else False
    g["bridge_or_crossing"] = g["bridge_or_crossing"].astype(bool)
    for c in ("susceptibility_class", "observed_change_class", "priority_class", "confidence_class", "debris_flag", "notes", "inspection_labels"):
        if c in g:
            g[c] = g[c].fillna("")
    cols = REQUIRED + [c for c in EXTRA if c in g.columns] + ["geometry"]
    return g[cols]


def export_reaches(reaches, out_gpkg, out_geojson):
    g = standard_reaches(reaches)
    g.to_file(out_gpkg, layer="river_reaches", driver="GPKG")
    gw = g.to_crs(4326)
    for c in gw.columns:
        if gw[c].dtype.kind == "f":
            gw[c] = gw[c].round(5).astype(float).where(np.isfinite(gw[c]), None)
    gw.to_file(out_geojson, driver="GeoJSON")
    return g
