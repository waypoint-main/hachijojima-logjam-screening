"""Compare the DEM-derived stream network with an existing hydrography dataset.

Reports, within a tolerance (m): the fraction of derived length lying near reference lines
(precision-like) and the fraction of reference length lying near derived lines (recall-like).
Low recall is expected where reference lines are under-canopy or intermittent; low precision
where the area threshold is too permissive. Neither implies the DEM streams are "wrong" — both
datasets are imperfect — it only quantifies agreement for documentation.
"""
from __future__ import annotations

import geopandas as gpd
from shapely.ops import unary_union


def compare_networks(derived: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, tol_m: float) -> dict:
    ref = reference.to_crs(derived.crs)
    d_len = float(derived.length.sum())
    r_len = float(ref.length.sum())
    if d_len == 0 or r_len == 0:
        return dict(derived_km=d_len / 1e3, reference_km=r_len / 1e3,
                    derived_within_tol=float("nan"), reference_within_tol=float("nan"),
                    tolerance_m=tol_m)
    ref_zone = unary_union(ref.geometry).buffer(tol_m)
    der_zone = unary_union(derived.geometry).buffer(tol_m)
    d_in = float(derived.geometry.intersection(ref_zone).length.sum())
    r_in = float(ref.geometry.intersection(der_zone).length.sum())
    return dict(derived_km=d_len / 1e3, reference_km=r_len / 1e3,
                derived_within_tol=d_in / d_len, reference_within_tol=r_in / r_len,
                tolerance_m=tol_m)
