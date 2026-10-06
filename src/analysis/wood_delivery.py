"""Wood delivery (Phase 5): from hillslope source areas to river reaches. Pure numpy/scipy.

SCIENTIFIC RATIONALE
  Wood only becomes a logjam hazard if it reaches a channel. Hillslope wood is delivered mostly from
  nearby slopes (the chance of reaching a stream falls with flow-path distance), and wood already in
  a channel is carried downstream, with attrition/deposition along the way. We route each source
  cell along its D8 flow path to the first stream cell and weight by distance, then carry the
  delivered quantity down the channel network with an exponential attenuation.

  Quantity unit: "effective source area" (m2) = sum over cells of (source potential x cell area).
  It is an index of how much disturbed, mobile forest feeds a reach — NOT a wood volume or mass.

ASSUMPTIONS (UNCALIBRATED)
  A20 Delivery probability = exp(-d / distance_decay_scale_m) for flow-path distance d <= delivery_max_distance_m, else 0.
  A21 Wood follows the single D8 flow direction (no lateral spreading, no hillslope storage or benches).
  A22 In-channel supply decays as exp(-L / channel_transport_decay_m) over channel distance L.
  A23 Source quantity on a cell is independent of cell position along the slope.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage



def _step_len(recv: np.ndarray, shape, cell: float) -> np.ndarray:
    """Distance (m) from each cell to its D8 receiver (cell * 1 or sqrt2); 0 where no receiver."""
    h, w = shape
    idx = np.arange(h * w)
    r, c = idx // w, idx % w
    rr, rc = recv // w, recv % w
    dr, dc = np.abs(rr - r), np.abs(rc - c)
    d = np.where(recv >= 0, np.where((dr == 1) & (dc == 1), 2 ** 0.5, 1.0) * cell, 0.0)
    return d


def route_to_streams(filled: np.ndarray, recv: np.ndarray, stream: np.ndarray, cell: float):
    """For each cell: flat index of the first stream cell reached along D8 flow, and the flow-path distance (m).

    Cells that never reach a stream get entry = -1, distance = inf. Stream cells map to themselves (distance 0).
    """
    shape = filled.shape
    flat = filled.ravel()
    valid = np.isfinite(flat)
    st = stream.ravel() & valid
    step = _step_len(recv, shape, cell)
    entry = np.full(flat.size, -1, dtype=np.int64)
    dist = np.full(flat.size, np.inf)
    order = np.argsort(np.where(valid, flat, np.inf), kind="stable")      # low -> high: receivers first
    for i in order[: int(valid.sum())]:
        if st[i]:
            entry[i], dist[i] = i, 0.0
        else:
            j = recv[i]
            if j >= 0 and entry[j] >= 0:
                entry[i], dist[i] = entry[j], dist[j] + step[i]
    return entry.reshape(shape), dist.reshape(shape)


def delivery_factor(dist_m: np.ndarray, max_dist_m: float, scale_m: float) -> np.ndarray:
    f = np.exp(-np.where(np.isfinite(dist_m), dist_m, np.inf) / scale_m)
    return np.where(np.isfinite(dist_m) & (dist_m <= max_dist_m), f, 0.0)


def hillslope_delivery(eff_area: np.ndarray, entry: np.ndarray, dist: np.ndarray, wm: dict):
    """Return (delivered_per_cell, direct_input_at_stream_cells). Conservation: sums are equal."""
    f = delivery_factor(dist, wm["delivery_max_distance_m"], wm["distance_decay_scale_m"])
    delivered = np.nan_to_num(eff_area) * f
    direct = np.zeros(delivered.size)
    m = (entry.ravel() >= 0) & (delivered.ravel() > 0)
    np.add.at(direct, entry.ravel()[m], delivered.ravel()[m])
    return delivered, direct.reshape(delivered.shape)


def channel_supply(filled: np.ndarray, recv: np.ndarray, stream: np.ndarray, direct: np.ndarray,
                   cell: float, decay_m: float) -> np.ndarray:
    """Accumulate direct inputs down the stream cells, attenuating by exp(-step/decay_m) per cell step."""
    shape = filled.shape
    flat = filled.ravel()
    valid = np.isfinite(flat)
    st = stream.ravel() & valid
    step = _step_len(recv, shape, cell)
    supply = direct.ravel().astype(float).copy()
    order = np.argsort(-np.where(valid, flat, -np.inf), kind="stable")     # high -> low
    for i in order[: int(valid.sum())]:
        if st[i] and supply[i] != 0:
            j = recv[i]
            if j >= 0 and st[j]:
                supply[j] += supply[i] * np.exp(-step[i] / decay_m)
    return supply.reshape(shape)


def nearest_stream_cell(stream: np.ndarray):
    """(distance_cells, flat index of nearest stream cell) for every cell."""
    d, (ir, ic) = ndimage.distance_transform_edt(~stream, return_indices=True)
    return d, (ir * stream.shape[1] + ic)


def normalise_score(raw: np.ndarray, percentile: float = 95.0) -> np.ndarray:
    """0-1 score: log1p(raw)/log1p(P_percentile of positive values), clipped. Data-dependent by design (documented)."""
    raw = np.nan_to_num(np.asarray(raw, float), nan=0.0)
    pos = raw[raw > 0]
    if pos.size == 0:
        return np.zeros_like(raw)
    ref = np.percentile(pos, percentile)
    return np.clip(np.log1p(raw) / np.log1p(ref), 0, 1)
