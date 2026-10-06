"""Phase 6 diagnostic for reach-level logjam susceptibility (screening index; not validated)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from ..models.logjam_susceptibility import DRIVERS

COLORS = {"Low": "#ffffb2", "Moderate": "#fecc5c", "High": "#fd8d3c", "Very High": "#bd0026"}


def plot_phase6(hillshade, reaches, contrib, transform, breaks, labels, sens, out_png, scenario="default", dpi=150):
    inv = ~transform
    segs = [np.array([(inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f) for x, y in g.coords])
            for g in reaches.geometry]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12.5))
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    a.add_collection(LineCollection(segs, colors=[COLORS[c] for c in reaches["susceptibility_class"]],
                                    linewidths=np.where(reaches["susceptibility_class"].isin(["High", "Very High"]), 2.0, 0.9)))
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"Logjam SUSCEPTIBILITY class by reach (rule-based screening) — {scenario}", fontsize=10)
    a.legend(handles=[plt.Line2D([0], [0], color=COLORS[k], lw=3, label=k) for k in labels], loc="lower left", fontsize=8)
    a = ax[0, 1]
    a.hist(reaches["susceptibility_score"], bins=40, color="#636363")
    for b in breaks:
        a.axvline(b, color="#d95f02", ls="--")
    a.set_xlabel("susceptibility score (0-1)"); a.set_ylabel("reaches"); a.set_title("Score distribution and class breaks", fontsize=10)
    a = ax[1, 0]
    cm = plt.get_cmap("tab20")
    bottoms = np.zeros(len(labels))
    for j, d in enumerate(DRIVERS):
        vals = np.array([contrib.loc[reaches["susceptibility_class"] == l, d].mean() if (reaches["susceptibility_class"] == l).any() else 0
                         for l in labels])
        a.bar(labels, vals, bottom=bottoms, color=cm(j % 20), label=d)
        bottoms += vals
    a.set_ylabel("mean weighted contribution"); a.set_title("What drives each class (mean contribution of each driver)", fontsize=10)
    a.legend(fontsize=6, ncol=2, loc="upper left")
    a = ax[1, 1]
    sc = a.scatter(reaches["upstream_area_m2"] / 1e6, reaches["susceptibility_score"], c=reaches["stream_order"], s=6, cmap="viridis")
    a.set_xscale("log"); a.set_xlabel("upstream area (km²)"); a.set_ylabel("score")
    plt.colorbar(sc, ax=a, label="Strahler order")
    a.set_title(f"Score vs size | weight-perturbation stability: median Spearman {sens['median_spearman']:.2f}, "
                f"top-10% overlap {sens['median_top_decile_overlap']:.2f}", fontsize=9)
    fig.suptitle("Phase 6 — logjam susceptibility (UNCALIBRATED weights; not a prediction of jams)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
