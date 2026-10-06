"""MAP B support — Phase 3 diagnostic of OBSERVED change (not hazard/susceptibility).

The final report-quality Map B (with river corridors and debris flags) is assembled in Phase 9.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..change_detection.fusion import CLASS_NAMES

COLORS = {1: "#d95f02", 2: "#e7298a", 3: "#3b0f70", 4: "#ffd400", 5: "#00bcd4", 6: "#3d3d3d", 7: "#e0e0e0"}  # keep in sync with final_maps.CHANGE_COL


def plot_phase3_diagnostic(cls, conf, sar, opt, hillshade, out_png, dpi=150, note="", label="2025→2026, season-matched"):
    fig, ax = plt.subplots(2, 3, figsize=(19, 12.5))
    show = lambda a, arr, t, **kw: (a.imshow(arr, **kw), a.set_title(t, fontsize=10), a.set_xticks([]), a.set_yticks([]))
    # class map
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    rgba = np.zeros(cls.shape + (4,))
    for k, c in COLORS.items():
        m = cls == k
        rgba[m] = matplotlib.colors.to_rgba(c)
    a.imshow(rgba, interpolation="nearest")
    a.set_title(f"OBSERVED satellite change classes ({label})", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    a.legend(handles=[Patch(color=COLORS[k], label=CLASS_NAMES[k]) for k in COLORS], loc="lower left", fontsize=7)
    im = ax[0, 1].imshow(conf, cmap="viridis", vmin=0, vmax=1)
    ax[0, 1].set_title("Confidence index (changed pixels; heuristic, not a probability)", fontsize=10)
    ax[0, 1].set_xticks([]); ax[0, 1].set_yticks([]); plt.colorbar(im, ax=ax[0, 1], shrink=0.7)
    im = ax[0, 2].imshow(sar["zVH"], cmap="RdBu", vmin=-6, vmax=6)
    ax[0, 2].set_title("SAR VH anomaly z-score (after regional-shift removal)", fontsize=10)
    ax[0, 2].set_xticks([]); ax[0, 2].set_yticks([]); plt.colorbar(im, ax=ax[0, 2], shrink=0.7)
    im = ax[1, 0].imshow(opt["anomNDVI"], cmap="RdBu", vmin=-0.3, vmax=0.3)
    ax[1, 0].set_title("dNDVI anomaly (after regional-shift removal)", fontsize=10)
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([]); plt.colorbar(im, ax=ax[1, 0], shrink=0.7)
    # agreement
    agree = np.zeros(cls.shape, dtype=int)
    agree[opt["optical_flag"]] = 1
    agree[sar["sar_flag"]] = 2
    agree[opt["optical_flag"] & sar["sar_flag"]] = 3
    cm = ListedColormap(["#f5f5f5", "#d95f02", "#1f78b4", "#000000"])
    ax[1, 1].imshow(agree, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    ax[1, 1].set_title("Sensor agreement", fontsize=10); ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    ax[1, 1].legend(handles=[Patch(color="#d95f02", label="optical only"), Patch(color="#1f78b4", label="SAR only"),
                             Patch(color="#000", label="both")], loc="lower left", fontsize=8)
    # areas
    names = [CLASS_NAMES[k] for k in COLORS]
    areas = [(cls == k).sum() * 100 / 1e6 for k in COLORS]
    ax[1, 2].barh(names[::-1], areas[::-1], color=[COLORS[k] for k in list(COLORS)[::-1]])
    ax[1, 2].set_xlabel("km²"); ax[1, 2].set_title("Area by class", fontsize=10)
    fig.suptitle("Phase 3 — observed change diagnostic" + (f"\n{note}" if note else ""), fontsize=13)
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
