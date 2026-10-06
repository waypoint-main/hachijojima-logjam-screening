"""Phase 7 diagnostic: observed change in river corridors and debris/obstruction evidence flags."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

COLORS = {"Low": "#ffffcc", "Moderate": "#a1dab4", "High": "#41b6c4", "Very High": "#225ea8", "Insufficient data": "#d9d9d9"}
FLAG_COL = {"Probable Debris Accumulation": "#e31a1c", "Possible Logjam": "#ff7f00"}


def plot_phase7(hillshade, reaches, transform, out_png, scenario="default", dpi=150):
    inv = ~transform
    tp = lambda x, y: (inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f)
    segs = [np.array([tp(x, y) for x, y in g.coords]) for g in reaches.geometry]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12.5))
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    a.add_collection(LineCollection(segs, colors=[COLORS[c] for c in reaches["observed_change_class"]], linewidths=1.4))
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"OBSERVED change in river corridors (satellite; not hazard) — {scenario}", fontsize=10)
    a.legend(handles=[plt.Line2D([0], [0], color=c, lw=3, label=k) for k, c in COLORS.items()], loc="lower left", fontsize=7)
    a = ax[0, 1]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    for k, c in FLAG_COL.items():
        sub = reaches[reaches["debris_flag"] == k]
        if len(sub):
            mid = np.array([tp(g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y) for g in sub.geometry])
            a.scatter(mid[:, 0], mid[:, 1], s=22, c=c, edgecolors="k", linewidths=0.3, label=f"{k} (n={len(sub)})")
    a.legend(fontsize=8, loc="lower left"); a.set_xticks([]); a.set_yticks([])
    a.set_title("Satellite evidence of obstruction/debris (HYPOTHESES — not confirmed)", fontsize=10)
    a = ax[1, 0]
    ok = reaches["observed_change_score"].notna()
    a.scatter(reaches.loc[ok, "susceptibility_score"], reaches.loc[ok, "observed_change_score"], s=5, c="#636363", alpha=0.5)
    for k, c in FLAG_COL.items():
        sub = reaches[ok & (reaches["debris_flag"] == k)]
        a.scatter(sub["susceptibility_score"], sub["observed_change_score"], s=26, c=c, edgecolors="k", linewidths=0.3, label=k)
    a.set_xlabel("susceptibility score (Phase 6)"); a.set_ylabel("observed-change score (Phase 7)")
    a.set_title("Hazard potential vs. observed change (the priority matrix lives in the upper right)", fontsize=9)
    a.legend(fontsize=8)
    a = ax[1, 1]
    cnt = reaches["observed_change_class"].value_counts().reindex(list(COLORS)).fillna(0)
    a.bar(cnt.index, cnt.values, color=[COLORS[k] for k in cnt.index], edgecolor="k")
    a.set_title("Reaches by observed-change class", fontsize=10); a.tick_params(axis="x", labelsize=8)
    fig.suptitle("Phase 7 — observed river-corridor change and debris/obstruction evidence", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
