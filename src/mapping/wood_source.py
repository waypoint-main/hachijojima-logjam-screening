"""Phase 4 diagnostic: forest definition, woody-debris source potential, exposure and proximity."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from ..analysis.wood_source import OCTANTS


def plot_phase4(hillshade, forest, ndvi_forest, wc_forest, pot, streams_mask, exposure, slope, src, dist,
                patch_rows, out_png, scenario="default", dpi=150):
    fig, ax = plt.subplots(2, 3, figsize=(19, 12.5))
    for a in ax.flat[:2]:
        a.set_xticks([]); a.set_yticks([])
    # (0,0) forest definitions
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    code = np.zeros(forest.shape, int)
    if wc_forest is not None:
        code[wc_forest] = 1
    code[ndvi_forest & (code == 0)] = 2
    code[ndvi_forest & (wc_forest if wc_forest is not None else False)] = 3
    cm = ListedColormap([(0, 0, 0, 0), "#1b9e77", "#d95f02", "#006d2c"])
    a.imshow(code, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    a.set_title("Forest definition: WorldCover tree only (light green) | NDVI only (orange) | both (dark green)", fontsize=9)
    a.legend(handles=[Patch(color="#1b9e77", label="WorldCover only"), Patch(color="#d95f02", label="NDVI only"),
                      Patch(color="#006d2c", label="both")], loc="lower left", fontsize=7)
    # (0,1) source potential
    a = ax[0, 1]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    im = a.imshow(pot, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    sm = np.where(streams_mask, 1.0, np.nan)
    a.imshow(sm, cmap=ListedColormap(["#1f78b4"]), interpolation="nearest")
    a.set_title(f"Woody-debris SOURCE potential index (0-1; not a volume) — scenario: {scenario}", fontsize=9)
    plt.colorbar(im, ax=a, shrink=0.7)
    # (0,2) exposure by aspect
    a = ax[0, 2]
    v = [exposure[o]["source_pct_of_forest"] or 0 for o in OCTANTS]
    a.bar(OCTANTS, v, color="#d95f02")
    a.set_ylabel("% of forest area that is a source pixel")
    a.set_title("Source share by aspect octant (wind-exposure diagnostic)", fontsize=10)
    # (1,0) slope
    a = ax[1, 0]
    bins = np.arange(0, 61, 5)
    fh, _ = np.histogram(slope[forest & np.isfinite(slope)], bins)
    sh, _ = np.histogram(slope[src & np.isfinite(slope)], bins)
    a.bar(bins[:-1] + 2.5, 100 * sh / np.maximum(fh, 1), width=4.5, color="#7b3294")
    a.set_xlabel("slope (deg)"); a.set_ylabel("% of forest in bin that is source")
    a.set_title("Source share by slope", fontsize=10)
    # (1,1) distance to stream
    a = ax[1, 1]
    d = dist[src & np.isfinite(dist)]
    if d.size:
        a.hist(d, bins=np.arange(0, 1001, 50), color="#1f78b4")
    a.set_xlabel("Euclidean distance from source pixel to nearest derived stream (m)")
    a.set_title("Source-to-stream proximity (descriptor only; delivery is Phase 5)", fontsize=9)
    # (1,2) patch sizes
    a = ax[1, 2]
    areas = [r["area_m2"] / 1e4 for r in patch_rows]
    if areas:
        a.hist(np.log10(areas), bins=25, color="#636363")
    a.set_xlabel("log10 patch area (ha)"); a.set_title(f"Source patches (n={len(areas)})", fontsize=10)
    fig.suptitle("Phase 4 — forest definition and woody-debris source areas (inference, not observation)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
