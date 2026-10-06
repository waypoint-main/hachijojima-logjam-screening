"""Split the derived stream network into fixed-length river reaches with terrain attributes.

Reaches are the unit of analysis for susceptibility, observed change and priority (Phases 6-8).

Method
------
Each stream link (head/confluence -> next confluence/outlet) is cut into
n = max(1, round(L_link / reach_length)) equal parts, ordered upstream -> downstream.
Attributes are sampled from the conditioned (filled) DEM and the accumulation grid:

* z_up, z_down, mean_elev_m   : elevation at reach ends / mean
* channel_slope_m_per_m       : (z_up - z_down)/length over the reach (≥0 by construction)
* local_slope_window          : same, but over a longer window centred on the reach
                                (less noisy at 30 m resolution)
* upstream_area_m2            : contributing area at the reach's downstream end
* stream_order                : Strahler
* confluence                  : reach starts or ends at a confluence
* sinuosity_window            : path length / chord over a longer window (relative bend metric)
* hillslope slope (DEM slope) : sampled at reach midpoint (corridor slope proxy)

Caveats
-------
* Derived-channel geometry on a 30 m DSM is approximate; at 100 m reach length a reach covers
  only ~3 cells. Channel *width* / constriction cannot be derived from a 30 m DEM and is left
  to Phase 2+ (SAR/optical water width, VHR imagery) — it is NOT estimated here.
* Sinuosity from D8 paths is inflated by grid artefacts; use only as a relative indicator.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring


def _sample(arr, transform, x, y):
    col = int(np.floor((x - transform.c) / transform.a))
    row = int(np.floor((y - transform.f) / transform.e))
    if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
        return arr[row, col]
    return np.nan


def _window_metrics(line: LineString, s_mid: float, window: float, filled, transform):
    L = line.length
    s0, s1 = max(0.0, s_mid - window / 2), min(L, s_mid + window / 2)
    p0, p1 = line.interpolate(s0), line.interpolate(s1)
    z0 = _sample(filled, transform, p0.x, p0.y)
    z1 = _sample(filled, transform, p1.x, p1.y)
    path = s1 - s0
    chord = p0.distance(p1)
    slope = (z0 - z1) / path if path > 0 else np.nan
    sinu = path / chord if chord > 0 else np.nan
    return slope, sinu


def build_reaches(net, filled, acc_cells, slope_deg, transform, cell, crs,
                  reach_length_m=100.0, slope_window_m=100.0, curvature_window_m=300.0
                  ) -> gpd.GeoDataFrame:
    rows = []
    rid = 0
    for link in net.links:
        line = LineString(link["coords"])
        Ltot = line.length
        if Ltot <= 0:
            continue
        n = max(1, int(round(Ltot / reach_length_m)))
        edges = np.linspace(0, Ltot, n + 1)
        for k in range(n):
            a, b = edges[k], edges[k + 1]
            seg = substring(line, a, b)
            if seg.is_empty or seg.length == 0:
                continue
            pu, pd_, pm = line.interpolate(a), line.interpolate(b), line.interpolate((a + b) / 2)
            z_up = _sample(filled, transform, pu.x, pu.y)
            z_dn = _sample(filled, transform, pd_.x, pd_.y)
            sl_w, _ = _window_metrics(line, (a + b) / 2, slope_window_m, filled, transform)
            _, sin_w = _window_metrics(line, (a + b) / 2, curvature_window_m, filled, transform)
            rows.append(dict(
                reach_id=f"R{rid:05d}", link_id=link["link_id"], reach_idx=k, n_in_link=n,
                stream_order=link["strahler"], length_m=seg.length,
                z_up_m=float(z_up), z_down_m=float(z_dn),
                mean_elev_m=float(np.nanmean([z_up, z_dn])),
                channel_slope_m_per_m=float((z_up - z_dn) / seg.length),
                local_slope_window_m_per_m=float(sl_w),
                upstream_area_m2=float(_sample(acc_cells, transform, pd_.x, pd_.y) * cell * cell),
                flow_accumulation_cells=float(_sample(acc_cells, transform, pd_.x, pd_.y)),
                hillslope_slope_deg=float(_sample(slope_deg, transform, pm.x, pm.y)),
                sinuosity_window=float(sin_w),
                confluence=bool((k == 0 and link["start_is_confluence"]) or
                                (k == n - 1 and link["end_is_confluence"])),
                drains_to_sea=bool(link["drains_to_sea"] and k == n - 1),
                geometry=seg,
            ))
            rid += 1
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gdf


def corridor(reaches: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """River-corridor polygons (per reach, flat caps so neighbours don't overlap much)."""
    out = reaches[["reach_id"]].copy()
    out["geometry"] = reaches.geometry.buffer(buffer_m, cap_style="flat")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=reaches.crs)
