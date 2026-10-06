"""Logjam susceptibility (Phase 6): rule-based, reach-level, interpretable. No ML, no Streamlit.

WHAT IT IS
  A weighted sum of 0-1 drivers describing how readily woody debris would (a) arrive at, and (b) be
  trapped in, a river reach. It describes HAZARD POTENTIAL from terrain, network and wood supply.
  It does NOT use any logjam observation, so it cannot be called validated; the weights and ramps are
  expert-judgement starting values (config: susceptibility.*, all [UNCALIBRATED]).

      score = sum_i w_i * v_i / sum_i w_i          (v_i in [0, 1])

DRIVERS AND RATIONALE
  wood_delivery_score              wood must arrive (Phase 5).
  upstream_disturbance_pct         share of upstream forest that is a source.
  disturbance_to_stream_proximity  disturbed forest within the riparian band feeds the channel directly.
  upstream_landslide               landslide/debris-flow sources deliver wood in pulses.
  slope_break                      a gradient decrease downstream is a deposition zone.
  low_local_slope                  wood is trapped more readily on gentle reaches.
  upstream_area                    larger flows can carry more/longer wood (log scale).
  channel_constriction             confined valleys (proxy, see below) reduce room for wood to pass.
  curvature_bend                   bends snag long pieces.
  confluence                       flow changes and sediment/wood deposition at junctions.
  road_crossing / bridge           structures with limited opening trap wood (culverts/bridges).
  observed_river_change            small term only; observed change is scored separately in Phase 7.

LIMITATIONS
  * No channel width/depth data: 'channel_constriction' is a valley-confinement proxy from a 30 m DSM
    (a DSM includes canopy; confinement is likely overestimated under forest).
  * Wood piece length, channel width and structure openings (the real controls on jamming) are unknown.
  * Weights/ramps are not validated; see `weight_sensitivity` for how stable the ranking is to them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

DRIVERS = ["wood_delivery_score", "upstream_disturbance_pct", "disturbance_to_stream_proximity", "upstream_landslide",
           "slope_break", "low_local_slope", "upstream_area", "channel_constriction", "curvature_bend",
           "confluence", "road_crossing", "bridge", "observed_river_change"]


def ramp(x, lo: float, hi: float, log: bool = False, invert: bool = False):
    x = np.asarray(x, dtype=float)
    if log:
        x, lo, hi = np.log10(np.maximum(x, 1e-12)), np.log10(max(lo, 1e-12)), np.log10(hi)
    v = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    v = np.where(np.isfinite(x), v, 0.0)
    return 1.0 - v if invert else v


def upstream_channel_slope(reaches, steps: int = 3) -> np.ndarray:
    """Mean channel slope of up to `steps` reaches upstream along the largest-area path (m/m).

    Upstream neighbour of reach r = reach whose downstream end coincides with r's upstream start
    (rounded to 1 m). Headwater reaches (no upstream neighbour) get their own slope (slope_break = 0).
    """
    ends = defaultdict(list)
    for i, g in enumerate(reaches.geometry):
        ends[tuple(np.round(g.coords[-1], 0))].append(i)
    area = reaches["upstream_area_m2"].to_numpy()
    slope = np.clip(reaches["channel_slope_m_per_m"].to_numpy(), 0, None)
    out = np.empty(len(reaches))
    for i, g in enumerate(reaches.geometry):
        vals, cur = [], i
        for _ in range(steps):
            cand = ends.get(tuple(np.round(reaches.geometry.iloc[cur].coords[0], 0)), [])
            cand = [c for c in cand if c != cur]
            if not cand:
                break
            cur = max(cand, key=lambda c: area[c])
            vals.append(slope[cur])
        out[i] = np.mean(vals) if vals else slope[i]
    return out


def build_drivers(reaches: pd.DataFrame, extras: dict, sc: dict) -> pd.DataFrame:
    """0-1 driver table. `extras`: upstream_channel_slope, channel_constriction_raw, corridor_change_frac (arrays)."""
    r = sc["ramps"]
    rp = lambda k, x: ramp(x, r[k]["lo"], r[k]["hi"], r[k].get("log", False), r[k].get("invert", False))
    local = np.clip(reaches["local_slope_window_m_per_m"].to_numpy(), 0, None)
    cv = sc["crossing_values"]
    d = pd.DataFrame(index=reaches.index)
    d["wood_delivery_score"] = np.clip(reaches["wood_delivery_score"].to_numpy(), 0, 1)
    d["upstream_disturbance_pct"] = rp("upstream_disturbance_pct", reaches["upstream_disturbed_pct"])
    d["disturbance_to_stream_proximity"] = rp("disturbance_to_stream_proximity", reaches["riparian_disturbed_frac"])
    d["upstream_landslide"] = rp("upstream_landslide", reaches["upstream_landslide_area_m2"])
    d["slope_break"] = rp("slope_break", extras["upstream_channel_slope"] - local)
    d["low_local_slope"] = rp("low_local_slope", local)
    d["upstream_area"] = rp("upstream_area", reaches["upstream_area_m2"])
    d["channel_constriction"] = rp("channel_constriction", extras["channel_constriction_raw"])
    d["curvature_bend"] = rp("curvature_bend", reaches["sinuosity_window"])
    d["confluence"] = reaches["confluence"].astype(float).to_numpy()
    d["road_crossing"] = reaches["crossing_type"].map(cv).fillna(0.0).to_numpy()
    structure = reaches["bridge"].astype(bool) | (reaches["culvert"].astype(bool) if "culvert" in reaches else False)
    d["bridge"] = np.where(structure, sc["bridge_or_culvert_value"], 0.0)
    d["observed_river_change"] = rp("observed_river_change", extras["corridor_change_frac"])
    return d[DRIVERS]


def classify(score: np.ndarray, breaks, labels) -> np.ndarray:
    return np.array(labels, dtype=object)[np.digitize(score, breaks)]


def score_reaches(drivers: pd.DataFrame, weights: dict, breaks, labels):
    w = np.array([weights[k] for k in DRIVERS], dtype=float)
    w = w / w.sum()
    contrib = drivers[DRIVERS].to_numpy() * w
    s = contrib.sum(axis=1)
    return s, classify(s, breaks, labels), pd.DataFrame(contrib, columns=DRIVERS, index=drivers.index)


def top_drivers(contrib: pd.DataFrame, n: int = 3) -> list[str]:
    arr, cols = contrib.to_numpy(), np.array(contrib.columns)
    out = []
    for row in arr:
        idx = np.argsort(-row)[:n]
        out.append(", ".join(f"{cols[j]} ({row[j]:.2f})" for j in idx if row[j] > 0.005))
    return out


def weight_sensitivity(drivers: pd.DataFrame, weights: dict, n_draws=300, rel=0.5, seed=42, top_frac=0.10) -> dict:
    """Perturb each weight by a random factor in [1-rel, 1+rel]; report rank stability of the susceptibility score."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    base = np.array([weights[k] for k in DRIVERS], float)
    v = drivers[DRIVERS].to_numpy()
    s0 = v @ (base / base.sum())
    k = max(1, int(len(s0) * top_frac))
    top0 = set(np.argsort(-s0)[:k])
    rho, ov = [], []
    for _ in range(n_draws):
        w = base * rng.uniform(1 - rel, 1 + rel, size=base.size)
        s = v @ (w / w.sum())
        rho.append(spearmanr(s0, s).correlation)
        ov.append(len(top0 & set(np.argsort(-s)[:k])) / k)
    return dict(n_draws=n_draws, relative_perturbation=rel, median_spearman=float(np.median(rho)),
                min_spearman=float(np.min(rho)), median_top_decile_overlap=float(np.median(ov)),
                min_top_decile_overlap=float(np.min(ov)))
