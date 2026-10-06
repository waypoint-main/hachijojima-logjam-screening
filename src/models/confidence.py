"""Per-reach confidence in the OBSERVATION basis of a priority assignment (Phase 8).

confidence = sum_k w_k * c_k with components in [0, 1] (weights: config `confidence.weights`):
  s1_availability       mean over corridor pixels of min(S1 scenes before, after) / s1_obs_full   (capped at 1)
  s2_valid_fraction     mean over corridor pixels of min(clear S2 obs before, after) / s2_obs_full (capped at 1)
  temporal_coverage     island-wide: how many post-event scenes the window holds relative to the full-credit counts
  sensor_agreement      Jaccard of SAR-flag and (1 px dilated) optical-flag pixels in the corridor; 1.0 if neither flags anything
  change_magnitude_persistence   1.0 if nothing flagged; else 0.5 * (persistent share of SAR flags) + 0.5 * observed-change score
It is a transparent heuristic, NOT a probability of being correct, and it says nothing about whether the
susceptibility weights are right.
"""
from __future__ import annotations

import numpy as np


def temporal_coverage(n_s1_after: float, n_s2_after: float, s1_full: float, s2_full: float) -> float:
    return float(0.5 * min(1.0, n_s1_after / s1_full) + 0.5 * min(1.0, n_s2_after / s2_full))


def combine(components: dict, weights: dict) -> np.ndarray:
    w = {k: weights[k] for k in components}
    tot = sum(w.values())
    return sum(w[k] * np.clip(np.nan_to_num(np.asarray(v, float), nan=0.0), 0, 1) for k, v in components.items()) / tot


def magnitude_persistence(flagged_any, persist_share, O) -> np.ndarray:
    f = np.asarray(flagged_any, bool)
    return np.where(f, 0.5 * np.nan_to_num(persist_share) + 0.5 * np.clip(np.nan_to_num(O), 0, 1), 1.0)
