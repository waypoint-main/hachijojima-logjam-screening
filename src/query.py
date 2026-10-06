"""Pure filtering / ranking / detail helpers used by the Streamlit app (no Streamlit import; fully testable).

Everything here works on the standard `river_reaches` table written by Phase 9 (see src/export.py).
"""
from __future__ import annotations

import re

import numpy as np

PRIORITY_ORDER = ["Priority 1", "Priority 2", "Priority 3", "Baseline", "Not assessed"]
SUSCEPTIBILITY_ORDER = ["Low", "Moderate", "High", "Very High"]
CONFIDENCE_ORDER = ["Low", "Medium", "High"]
# Wording allowed on screen (policy: never "Confirmed Logjam" without validation data).
ALLOWED_LABELS = ["Possible Logjam", "Probable Debris Accumulation", "High Logjam Susceptibility", "Priority Inspection Location"]


def load_reaches(path, layer="river_reaches"):
    import geopandas as gpd
    return gpd.read_file(path, layer=layer) if str(path).endswith(".gpkg") else gpd.read_file(path)


def filter_reaches(df, priority=None, susceptibility=None, confidence=None, min_susceptibility=0.0, min_observed=0.0,
                   debris_only=False, crossing_only=False, min_stream_order=1, text="", min_upstream_km2=0.0):
    """Return the rows matching every given criterion (None / empty = no restriction on that field)."""
    m = np.ones(len(df), bool)
    for col, vals in (("priority_class", priority), ("susceptibility_class", susceptibility), ("confidence_class", confidence)):
        if vals:
            m &= df[col].isin(list(vals)).to_numpy()
    m &= (df["susceptibility_score"] >= min_susceptibility).to_numpy()
    if min_observed > 0:                                      # reaches without a measurable change score only pass when no minimum is set
        m &= (df["observed_change_score"] >= min_observed).to_numpy()
    if min_upstream_km2 > 0 and "upstream_area_m2" in df:
        m &= (df["upstream_area_m2"] >= min_upstream_km2 * 1e6).to_numpy()
    if "stream_order" in df:
        m &= (df["stream_order"] >= min_stream_order).to_numpy()
    if debris_only:
        m &= (df["debris_flag"].fillna("") != "").to_numpy()
    if crossing_only:
        m &= df["bridge_or_crossing"].astype(bool).to_numpy()
    if text:
        m &= df["reach_id"].astype(str).str.contains(text.strip(), case=False, regex=False).to_numpy()
    return df[m]


def ranked(df):
    """Inspection order: Priority 1 and 2 TOGETHER by priority_score (descending), then Priority 3, Baseline, Not assessed."""
    from .models.priority import rank_order
    return df.iloc[rank_order(df["priority_class"].to_numpy(), df["priority_score"].to_numpy())]


def summary(df):
    """Headline counts for the current selection."""
    pc = df["priority_class"].value_counts()
    return dict(n_reaches=int(len(df)),
                **{k.lower().replace(" ", "_"): int(pc.get(k, 0)) for k in PRIORITY_ORDER},
                n_possible_logjam=int((df["debris_flag"] == "Possible Logjam").sum()),
                n_probable_debris=int((df["debris_flag"] == "Probable Debris Accumulation").sum()),
                n_high_susceptibility=int(df["susceptibility_class"].isin(["High", "Very High"]).sum()),
                n_low_confidence=int((df["confidence_class"] == "Low").sum()),
                n_small_catchment_in_list=int(((df["priority_class"].isin(["Priority 1", "Priority 2"])) & df.get("small_catchment", False)).sum())
                if "small_catchment" in df else 0)


_DRV = re.compile(r"([A-Za-z_]+) \((\d+(?:\.\d+)?)\)")


def parse_drivers(text):
    """'wood_delivery_score (0.21), upstream_slope (0.08)' -> [(name, 0.21), ...] (descending)."""
    return sorted(((n, float(v)) for n, v in _DRV.findall(text or "")), key=lambda t: -t[1])


def reach_detail(df, reach_id):
    """Everything the 'Site details' tab shows for one reach, as plain python types."""
    r = df.loc[df["reach_id"] == reach_id]
    if r.empty:
        raise KeyError(reach_id)
    r = r.iloc[0]
    clean = lambda v: None if (isinstance(v, float) and not np.isfinite(v)) else (v.item() if hasattr(v, "item") else v)
    scalars = {c: clean(r[c]) for c in df.columns if c != "geometry"}
    mid = r.geometry.interpolate(0.5, normalized=True)
    lon, lat = _to_wgs84(mid, df.crs)
    labels = [s.strip() for s in str(scalars.get("inspection_labels") or "").split(";") if s.strip()]
    return dict(scalars=scalars, lon=lon, lat=lat, labels=labels, drivers=parse_drivers(scalars.get("susceptibility_drivers")),
                reasons=str(scalars.get("notes") or ""))


def _to_wgs84(pt, crs):
    if crs is None or crs.to_epsg() == 4326:
        return float(pt.x), float(pt.y)
    from pyproj import Transformer
    x, y = Transformer.from_crs(crs, 4326, always_xy=True).transform(pt.x, pt.y)
    return float(x), float(y)


def compare_scenarios(a, b, classes=("Priority 1", "Priority 2")):
    """Agreement between two scenario tables (Jaccard on the chosen priority classes) + reaches present in both / only one."""
    sa = set(a.loc[a["priority_class"].isin(classes), "reach_id"]); sb = set(b.loc[b["priority_class"].isin(classes), "reach_id"])
    return dict(jaccard=len(sa & sb) / max(len(sa | sb), 1), n_a=len(sa), n_b=len(sb), both=sorted(sa & sb), only_a=sorted(sa - sb), only_b=sorted(sb - sa))


def to_geojson_wgs84(df, columns):
    """GeoJSON dict (WGS84) with a small, JSON-safe property set for map layers."""
    import json
    g = df[list(columns) + ["geometry"]].to_crs(4326)
    for c in columns:
        if g[c].dtype.kind == "f":
            g[c] = g[c].round(3)
    return json.loads(g.to_json())


def midpoints(df):
    """DataFrame (reach_id, lon, lat, debris_flag if present) of each reach's midpoint in WGS84."""
    import pandas as pd
    if len(df) == 0:
        return pd.DataFrame(columns=["reach_id", "lon", "lat", "debris_flag"])
    mid = df.geometry.interpolate(0.5, normalized=True)
    import geopandas as gpd
    ll = gpd.GeoSeries(mid, crs=df.crs).to_crs(4326)
    out = pd.DataFrame({"reach_id": df["reach_id"].to_numpy(), "lon": ll.x.to_numpy(), "lat": ll.y.to_numpy()})
    out["debris_flag"] = df["debris_flag"].fillna("").to_numpy() if "debris_flag" in df else ""
    return out
