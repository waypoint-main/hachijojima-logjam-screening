"""Road/bridge–stream crossings (Phase 1 inputs to Map A and Phase 6 scoring).

Crossings are where road centrelines intersect (buffered) derived streams. `bridge` is True
if the road way carries a bridge tag; `culvert` if tagged culvert/tunnel. Roads with no tag at
a crossing are treated as "road crossing, structure unknown" — NOT as bridges.

Caveat: derived streams are ±1-2 cells off the true channel, hence `snap_tolerance_m`.
Missing OSM tags are common; absence of a bridge flag is not evidence of absence.
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

_TRUE = lambda v: v not in (None, "", "no") and not (isinstance(v, float) and v != v)


def find_crossings(roads: gpd.GeoDataFrame, streams: gpd.GeoDataFrame,
                   snap_tolerance_m: float = 15.0, major=None, minor=None) -> gpd.GeoDataFrame:
    """Crossings of vehicle roads with derived streams. `road_class` is 'major' or 'minor'.
    If major/minor are None every road is kept as 'major' (backward compatible). Roads of any other
    class (footways, paths, ...) are dropped: foot access cannot form a debris-trapping crossing."""
    cols = ["crossing_id", "highway", "road_class", "bridge", "culvert", "geometry"]
    if roads is not None and len(roads) and (major is not None or minor is not None):
        keep = set(major or []) | set(minor or [])
        roads = roads[roads["highway"].isin(keep)]
    if roads is None or len(roads) == 0 or len(streams) == 0:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=streams.crs)
    roads = roads.to_crs(streams.crs)
    zone = unary_union(streams.geometry).buffer(snap_tolerance_m)
    rows = []
    for _, r in roads.iterrows():
        inter = r.geometry.intersection(zone)
        if inter.is_empty:
            continue
        parts = getattr(inter, "geoms", [inter])
        for p in parts:
            rows.append(dict(
                highway=r.get("highway"),
                road_class="minor" if (minor and r.get("highway") in minor) else "major",
                bridge=_TRUE(r.get("bridge")),
                culvert=_TRUE(r.get("culvert")) or _TRUE(r.get("tunnel")),
                geometry=p.centroid))
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=streams.crs)
    if len(g):
        g = g.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
        g.insert(0, "crossing_id", [f"X{i:04d}" for i in range(len(g))])
    return g


def flag_reaches(reaches: gpd.GeoDataFrame, crossings: gpd.GeoDataFrame, tol_m: float = 30.0,
                 min_upstream_area_m2: float = 0.0) -> gpd.GeoDataFrame:
    """Flag reaches near a crossing. Reaches draining less than min_upstream_area_m2 are never flagged
    (gullies too small to trap large wood). `crossing_type` = 'major' | 'minor' | '' (major wins)."""
    out = reaches.copy()
    out["crossing_type"] = ""
    out["road_crossing"] = False
    out["bridge"] = False
    out["culvert"] = False
    if crossings is None or len(crossings) == 0:
        return out
    buf = gpd.GeoDataFrame(geometry=out.geometry.buffer(tol_m), index=out.index, crs=out.crs)
    j = gpd.sjoin(crossings, buf, how="inner", predicate="within")
    for idx, grp in j.groupby("index_right"):
        if out.loc[idx, "upstream_area_m2"] < min_upstream_area_m2:
            continue
        out.loc[idx, "road_crossing"] = True
        out.loc[idx, "crossing_type"] = "major" if (grp["road_class"] == "major").any() else "minor"
        out.loc[idx, "bridge"] = bool(grp["bridge"].any())
        out.loc[idx, "culvert"] = bool(grp["culvert"].any())
    out["bridge_or_crossing"] = out["road_crossing"]
    return out
