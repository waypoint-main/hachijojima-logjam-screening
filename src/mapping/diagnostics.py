"""Phase 2 QA figure: baseline vs after composites and observation counts.

Purpose is to CHECK the inputs (coverage, orbit consistency, cloud gaps) before any change
detection. Differences shown here are raw dB / index differences, not classified change.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..data.rasters import read_stack


def plot_phase2_diagnostic(paths, min_valid_obs, out_png, dpi=150):
    s1b, _, _ = read_stack(paths["s1_baseline"]); s1a, _, _ = read_stack(paths["s1_after"])
    s2b, _, _ = read_stack(paths["s2_baseline"]); s2a, _, _ = read_stack(paths["s2_after"])
    fig, ax = plt.subplots(3, 4, figsize=(18, 12))
    pan = lambda a, arr, t, cmap, lim=None, lab="": (
        lambda im: (a.set_title(t, fontsize=9), a.set_xticks([]), a.set_yticks([]),
                    plt.colorbar(im, ax=a, shrink=0.7, label=lab)))(a.imshow(arr, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None))
    pan(ax[0, 0], s1b["VV_median"], "S1 VV median — baseline", "gray", (-20, 0), "dB")
    pan(ax[0, 1], s1a["VV_median"], "S1 VV median — after", "gray", (-20, 0), "dB")
    pan(ax[0, 2], s1a["VV_median"] - s1b["VV_median"], "dVV (after − baseline)", "RdBu", (-4, 4), "dB")
    pan(ax[0, 3], s1a["VH_median"] - s1b["VH_median"], "dVH (after − baseline)", "RdBu", (-4, 4), "dB")
    pan(ax[1, 0], s1b["VV_stdDev"], "S1 VV temporal σ — baseline", "viridis", (0, 3), "dB")
    pan(ax[1, 1], s1b["n_obs"], "S1 n_obs — baseline", "magma", None, "scenes")
    pan(ax[1, 2], s1a["n_obs"], "S1 n_obs — after", "magma", None, "scenes")
    pan(ax[1, 3], (s1a["VVVH_median"] - s1b["VVVH_median"]), "d(VV−VH) ratio (dB)", "RdBu", (-4, 4), "dB")
    pan(ax[2, 0], s2b["NDVI"], "S2 NDVI — baseline", "YlGn", (0, 1), "")
    pan(ax[2, 1], s2a["NDVI"], "S2 NDVI — after", "YlGn", (0, 1), "")
    pan(ax[2, 2], s2a["NDVI"] - s2b["NDVI"], "dNDVI (after − baseline)", "RdBu", (-0.3, 0.3), "")
    frac_low = float(np.nanmean(s2a["n_clear"] < min_valid_obs))
    pan(ax[2, 3], s2a["n_clear"], f"S2 clear obs — after ({frac_low:.0%} of pixels < {min_valid_obs} → low confidence)", "magma", None, "obs")
    fig.suptitle("Phase 2 composites — input QA (not classified change)", fontsize=13)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
