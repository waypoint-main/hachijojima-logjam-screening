"""Shared helpers for change detection."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

S8 = np.ones((3, 3), dtype=bool)


def robust_shift(diff: np.ndarray, valid: np.ndarray, method: str = "mode") -> float:
    """Island-wide reference level of a difference image (the 'no local change' offset).

    method:
      'mode'   : peak of a smoothed histogram (default). When a large fraction of pixels really
                 changed (e.g. a storm-exposed flank), the MEDIAN is dragged toward the damage; the
                 mode tracks the typical unaffected pixel. Use the median-vs-mode gap as a sanity
                 check (reported in phase3_qa.json).
      'median' : plain median.
      'none'   : 0 (no regional correction).
    """
    if method == "none":
        return 0.0
    x = diff[valid & np.isfinite(diff)]
    if x.size == 0:
        return 0.0
    if method == "median":
        return float(np.median(x))
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return float(np.median(x))
    bins = np.linspace(lo, hi, 201)
    h, e = np.histogram(x, bins=bins)
    h = ndimage.gaussian_filter1d(h.astype(float), 2)
    i = int(h.argmax())
    return float((e[i] + e[i + 1]) / 2)


def remove_small_patches(mask: np.ndarray, min_px: int, close: bool = False) -> np.ndarray:
    """Keep 8-connected patches of >= min_px pixels (optionally after a 1-px closing)."""
    m = ndimage.binary_closing(mask, structure=S8) if close else mask
    lab, n = ndimage.label(m, structure=S8)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= min_px))
