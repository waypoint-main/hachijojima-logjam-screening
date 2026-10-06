"""Terrain derivatives from a metric-CRS DEM (Phase 1).

All functions take a float array (NaN = no data / sea) and the cell size in metres, and
return arrays of the same shape with NaN where the input is NaN.

Methods
-------
* slope: Horn (1981) 3x3 finite differences, returned in degrees.
* hillshade: standard Lambertian illumination (visual aid only, not analysis input).
* curvature: Zevenbergen & Thorne (1987) profile and plan curvature (1/m); positive =
  concave. Used later as a *relative* indicator, not an absolute physical quantity.

Assumption: the DEM is a regular grid in a projected CRS, so dx = dy = cell size.
Edge / coastline cells use nearest-valid-neighbour padding so the sea mask does not
create artificial cliffs.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _fill_nan_nearest(a: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(a)
    if not mask.any():
        return a.astype("float64")
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)].astype("float64")


def _neighbours(z):
    p = np.pad(z, 1, mode="edge")
    n = lambda dr, dc: p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
    return (n(-1, -1), n(-1, 0), n(-1, 1), n(0, -1), n(0, 0), n(0, 1), n(1, -1), n(1, 0), n(1, 1))


def slope_degrees(dem: np.ndarray, cell: float) -> np.ndarray:
    z = _fill_nan_nearest(dem)
    z1, z2, z3, z4, _, z6, z7, z8, z9 = _neighbours(z)
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell)   # rows increase southward
    s = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return np.where(np.isfinite(dem), s, np.nan)


def aspect_degrees(dem: np.ndarray, cell: float) -> np.ndarray:
    """Downslope direction, degrees clockwise from north."""
    z = _fill_nan_nearest(dem)
    z1, z2, z3, z4, _, z6, z7, z8, z9 = _neighbours(z)
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell)
    # downslope vector = (-dzdx east, +dzdy north) since rows increase to the south
    a = (np.degrees(np.arctan2(-dzdx, dzdy))) % 360
    return np.where(np.isfinite(dem), a, np.nan)


def hillshade(dem: np.ndarray, cell: float, azimuth: float = 315, altitude: float = 45,
              z_factor: float = 1.0) -> np.ndarray:
    z = _fill_nan_nearest(dem) * z_factor
    z1, z2, z3, z4, _, z6, z7, z8, z9 = _neighbours(z)
    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8 * cell)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8 * cell)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    asp = np.arctan2(dzdy, -dzdx)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - asp)
    hs = np.clip(hs, 0, 1)
    return np.where(np.isfinite(dem), hs, np.nan)


def curvature(dem: np.ndarray, cell: float):
    """Zevenbergen–Thorne (profile, plan) curvature in 1/m.

    Returns (profile, plan). Convention: positive = concave (profile: slope steepens... i.e.
    bowl-like; plan: laterally convergent, valley-like), negative = convex / divergent
    (ridge-like). The profile sign is flipped from the raw Zevenbergen–Thorne formula so
    that "positive = concave" holds for both outputs.
    """
    z = _fill_nan_nearest(dem)
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _neighbours(z)
    L = cell
    D = ((z4 + z6) / 2 - z5) / L**2
    E = ((z2 + z8) / 2 - z5) / L**2
    F = (-z1 + z3 + z7 - z9) / (4 * L**2)
    G = (-z4 + z6) / (2 * L)
    H = (z2 - z8) / (2 * L)   # z2 north, z8 south: H = dz/dy with y northward
    den = G**2 + H**2
    with np.errstate(divide="ignore", invalid="ignore"):
        profile = 2 * (D * G**2 + E * H**2 + F * G * H) / den
        plan = 2 * (D * H**2 + E * G**2 - F * G * H) / den
    profile = np.where(den > 1e-12, profile, 0.0)
    plan = np.where(den > 1e-12, plan, 0.0)
    ok = np.isfinite(dem)
    return np.where(ok, profile, np.nan), np.where(ok, plan, np.nan)
