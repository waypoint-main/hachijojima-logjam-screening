"""Raster context around reaches on the 30 m DEM grid (valley confinement, corridor change). No EE, no Streamlit."""
from __future__ import annotations

import numpy as np
import rasterio.features as rf
from scipy import ndimage


def reach_id_rasters(reaches, shape, transform, stream: np.ndarray):
    """(ids, rid_near, dist_to_stream_cells, nearest_stream_flat_index).

    rid_near: id (1..N) of the reach nearest to every cell (all_touched rasterisation, then nearest-cell fill).
    """
    ids = np.arange(1, len(reaches) + 1)
    rid = rf.rasterize(zip(reaches.geometry, ids), out_shape=shape, transform=transform, all_touched=True,
                       dtype="int32", fill=0)
    _, idx = ndimage.distance_transform_edt(rid == 0, return_indices=True)
    rid_near = rid[idx[0], idx[1]]
    d, (ir, ic) = ndimage.distance_transform_edt(~stream, return_indices=True)
    return ids, rid_near, d, ir * shape[1] + ic


def valley_confinement(filled, rid_near, dist_cells, nearest_flat, ids, cell, band_m=(30.0, 90.0)):
    """Mean rise per metre of lateral distance in a band beside the channel (m/m), per reach id. NaN if no cells.

    Proxy for valley confinement (steep, narrow valley => high). It is NOT channel width.
    """
    dm = dist_cells * cell
    m = (dm >= band_m[0]) & (dm <= band_m[1]) & np.isfinite(filled)
    z_ch = filled.ravel()[nearest_flat]
    rise = np.where(m, (filled - z_ch) / np.maximum(dm, cell), np.nan)
    lab = np.where(m, rid_near, 0)
    cnt = ndimage.sum(m, lab, ids)
    mean = ndimage.sum(np.nan_to_num(rise), lab, ids) / np.maximum(cnt, 1)
    return np.where(cnt > 0, mean, np.nan)


def corridor_mean(values, rid_near, dist_cells, ids, cell, band_m):
    """Mean of a 30 m raster (e.g. fraction of change) inside the corridor band around each reach."""
    m = (dist_cells * cell <= band_m) & np.isfinite(values)
    lab = np.where(m, rid_near, 0)
    cnt = ndimage.sum(m, lab, ids)
    return np.where(cnt > 0, ndimage.sum(np.nan_to_num(values), lab, ids) / np.maximum(cnt, 1), np.nan)


def corridor_labels(reaches, shape, transform, cell: float):
    """(rid_near, dist_m): id (1..N) of the nearest reach for every cell and the distance (m) to it."""
    ids = np.arange(1, len(reaches) + 1)
    rid = rf.rasterize(zip(reaches.geometry, ids), out_shape=shape, transform=transform, all_touched=True,
                       dtype="int32", fill=0)
    d, idx = ndimage.distance_transform_edt(rid == 0, return_indices=True)
    return rid[idx[0], idx[1]], (d * cell).astype("float32")
