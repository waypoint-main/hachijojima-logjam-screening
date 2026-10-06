"""D8 hydrology on a DEM: depression filling, flow direction, accumulation, stream network.

Why a bespoke implementation?
-----------------------------
pysheds/whitebox are fine tools, but pysheds currently breaks on numpy>=2 and WhiteboxTools is
a binary dependency. The algorithms needed here are short, standard and — important for a
PoC that must be auditable — easy to unit-test:

1. Priority-flood depression filling with epsilon increments (Barnes, Lehman & Mulla 2014).
   Seeds are valid cells on the coast/grid edge, i.e. the sea is the outlet.
2. D8 steepest-descent flow direction (O'Callaghan & Mark 1984) on the filled DEM, with
   diagonal distance weighting.
3. Flow accumulation by processing cells from high to low (valid because the epsilon-filled
   DEM has a strictly descending path from every cell to an outlet).
4. Stream cells = accumulation area >= threshold. Links are traced between junctions/heads;
   Strahler order is computed on cells.

Assumptions
-----------
* Filling modifies the DEM (a DSM with canopy artefacts will have many spurious pits).
  Filling is a *conditioning step for routing*, not a claim about real topography.
* Single-flow-direction D8 produces parallel/straight artefacts on planar slopes. Acceptable
  for a reach-level screening PoC; revisit (D-infinity / burned DEM) if needed.
* Streams are *derived*, not surveyed. Compare against reference hydrography (reference.py).
"""
from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

log = logging.getLogger(__name__)

# D8 neighbour offsets (drow, dcol), row index increases southward.
D8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
SQRT2 = 2 ** 0.5
D8_DIST = [1, SQRT2, 1, SQRT2, 1, SQRT2, 1, SQRT2]


def coastal_mask(valid: np.ndarray) -> np.ndarray:
    """Valid cells touching the grid edge or an invalid (sea) cell = drainage outlets."""
    h, w = valid.shape
    pad = np.pad(valid, 1, constant_values=False)
    t = np.zeros_like(valid)
    for dr, dc in D8:
        t |= ~pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return valid & t


def fill_depressions(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Priority-flood + epsilon (Barnes et al. 2014). NaN cells are outside the domain/sea."""
    z = dem.astype("float64")
    valid = np.isfinite(z)
    h, w = z.shape
    filled = np.where(valid, z, np.nan)
    closed = ~valid.copy()                       # invalid cells are never visited
    heap: list = []
    # seeds: valid cells touching the grid edge or an invalid (sea) cell
    pad = np.pad(valid, 1, constant_values=False)
    touching = np.zeros_like(valid)
    for dr, dc in D8:
        touching |= ~pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    seeds = np.argwhere(valid & touching)
    for r, c in seeds:
        heapq.heappush(heap, (filled[r, c], int(r), int(c)))
        closed[r, c] = True
    zf = filled
    cl = closed
    while heap:
        zc, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not cl[rr, cc]:
                cl[rr, cc] = True
                if zf[rr, cc] <= zc:
                    zf[rr, cc] = zc + epsilon
                heapq.heappush(heap, (zf[rr, cc], rr, cc))
    return zf


def d8_receivers(filled: np.ndarray, cell: float) -> np.ndarray:
    """1-D array (row-major flat) of each cell's D8 receiver flat index; -1 = drains off-domain (to sea) or no descent.

    Sea/outside cells are treated as lower than every land cell, so coastal cells drain to sea.
    """
    h, w = filled.shape
    valid = np.isfinite(filled)
    sea = (np.nanmin(filled) - 1.0)
    z = np.where(valid, filled, sea)
    zp = np.pad(z, 1, constant_values=sea)
    best = np.zeros((h, w))
    best_dir = np.full((h, w), -1, dtype=np.int8)
    for k, (dr, dc) in enumerate(D8):
        nb = zp[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (z - nb) / (cell * D8_DIST[k])
        better = slope > best
        best = np.where(better, slope, best)
        best_dir = np.where(better, k, best_dir)
    rows, cols = np.indices((h, w))
    recv = np.full((h, w), -1, dtype=np.int64)
    for k, (dr, dc) in enumerate(D8):
        m = (best_dir == k)
        rr, cc = rows + dr, cols + dc
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        tgt = np.where(inside, rr * w + cc, -1)
        tgt_valid = np.zeros_like(inside)
        tgt_valid[inside] = valid[rr[inside], cc[inside]]
        recv = np.where(m & tgt_valid, tgt, recv)
    recv[~valid] = -1
    return recv.ravel()


def flow_accumulation(filled: np.ndarray, recv: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Number of upstream cells including self (float64), computed high -> low.

    With `weights` (same shape; NaN treated as 0) returns the weighted sum over the upstream area instead
    (e.g. weights = disturbed area per cell -> upstream disturbed area).
    """
    h, w = filled.shape
    flat = filled.ravel()
    valid = np.isfinite(flat)
    order = np.argsort(-np.where(valid, flat, -np.inf), kind="stable")
    n_valid = int(valid.sum())
    acc = np.where(valid, 1.0, 0.0) if weights is None else np.where(valid, np.nan_to_num(weights.ravel()), 0.0)
    r = recv
    for i in order[:n_valid]:
        j = r[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc.reshape(h, w)


@dataclass
class StreamNetwork:
    links: list          # list[dict(link_id, coords(list[(x,y)]), cells(list[int]), ...)]
    stream_mask: np.ndarray
    strahler: np.ndarray
    donors: np.ndarray   # number of stream donors per cell


def extract_streams(filled, recv, acc_cells, transform, cell, threshold_area_m2,
                    min_isolated_len_m: float = 0.0) -> StreamNetwork:
    h, w = filled.shape
    n = h * w
    valid = np.isfinite(filled).ravel()
    area = acc_cells.ravel() * cell * cell
    stream = valid & (area >= threshold_area_m2)
    donors = np.zeros(n, dtype=np.int32)
    idx = np.flatnonzero(stream)
    r = recv
    for i in idx:
        j = r[i]
        if j >= 0 and stream[j]:
            donors[j] += 1

    # Strahler order, high -> low
    order_idx = idx[np.argsort(-filled.ravel()[idx], kind="stable")]
    strahler = np.zeros(n, dtype=np.int16)
    max_in = np.zeros(n, dtype=np.int16)
    cnt_max = np.zeros(n, dtype=np.int16)
    for i in order_idx:
        if donors[i] == 0:
            o = 1
        else:
            o = max_in[i] + (1 if cnt_max[i] >= 2 else 0)
        strahler[i] = o
        j = r[i]
        if j >= 0 and stream[j]:
            if o > max_in[j]:
                max_in[j], cnt_max[j] = o, 1
            elif o == max_in[j]:
                cnt_max[j] += 1

    a, _, c0, _, e, f0 = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

    def xy(i):
        rr, cc = divmod(int(i), w)
        return (c0 + (cc + 0.5) * a, f0 + (rr + 0.5) * e)

    nodes = idx[donors[idx] != 1]
    links = []
    for s in nodes:
        path = [int(s)]
        cur = int(r[s])
        while cur >= 0 and stream[cur]:
            path.append(cur)
            if donors[cur] != 1:
                break
            cur = int(r[cur])
        outlet = not (cur >= 0 and stream[cur])
        if len(path) < 2:
            continue
        coords = [xy(i) for i in path]
        length = float(sum(np.hypot(coords[k + 1][0] - coords[k][0], coords[k + 1][1] - coords[k][1])
                           for k in range(len(coords) - 1)))
        head = donors[path[0]] == 0
        if head and outlet and length < min_isolated_len_m:
            continue
        links.append(dict(
            link_id=len(links), cells=path, coords=coords, length_m=length,
            strahler=int(strahler[path[len(path) // 2]]),
            start_is_confluence=bool(donors[path[0]] >= 2),
            end_is_confluence=bool((not outlet) and donors[path[-1]] >= 2),
            drains_to_sea=bool(outlet),
        ))
    log.info("Streams: %d stream cells, %d links, max Strahler %d",
             int(stream.sum()), len(links), int(strahler.max()) if strahler.size else 0)
    return StreamNetwork(links, stream.reshape(h, w), strahler.reshape(h, w), donors.reshape(h, w))


def links_to_lines(net: StreamNetwork):
    return [LineString(l["coords"]) for l in net.links]
