"""OpenStreetMap helpers (Overpass API): reference waterways, roads and bridges.

OSM coverage on small islands is variable; treat as a *comparison* dataset, not truth.
Requires outbound HTTPS to an Overpass endpoint. Alternatively supply local files
(e.g. MLIT/GSI river centrelines, road network) via config `local_path`.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import requests
from shapely.geometry import LineString

log = logging.getLogger(__name__)
OVERPASS_ENDPOINTS = ["https://overpass-api.de/api/interpreter",
                      "https://overpass.private.coffee/api/interpreter",
                      "https://overpass.kumi.systems/api/interpreter"]
# Overpass returns HTTP 406 to requests without a descriptive User-Agent.
HEADERS = {"User-Agent": "waypoint-hachijojima-lwd-poc/0.1 (research prototype)",
           "Accept": "application/json"}


def _overpass(query: str, timeout: int = 120, attempts: int = 3) -> dict:
    """POST to each Overpass endpoint in turn, with retries/backoff for 429/5xx. All failures are
    collected so the final error says exactly what happened on each server."""
    import time
    errors = []
    for url in OVERPASS_ENDPOINTS:
        for k in range(attempts):
            try:
                r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=timeout)
                if r.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.reason}")
                r.raise_for_status()
                return r.json()
            except Exception as e:
                errors.append(f"{url} try{k + 1}: {e}")
                log.warning("Overpass %s try %d failed: %s", url, k + 1, e)
                time.sleep(3 * (k + 1))
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors[-6:]))


def _ways_to_gdf(elements, tag_keys) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({"osm_id": el["id"], **{k: tags.get(k) for k in tag_keys},
                     "geometry": LineString(coords)})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def fetch_waterways(bbox) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    q = f'[out:json][timeout:90];way["waterway"~"river|stream|canal|drain|ditch"]' \
        f'({miny},{minx},{maxy},{maxx});out geom;'
    return _ways_to_gdf(_overpass(q)["elements"], ["waterway", "name", "intermittent", "tunnel"])


def fetch_roads(bbox) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    q = f'[out:json][timeout:90];way["highway"]({miny},{minx},{maxy},{maxx});out geom;'
    return _ways_to_gdf(_overpass(q)["elements"],
                        ["highway", "name", "bridge", "tunnel", "culvert", "layer"])
