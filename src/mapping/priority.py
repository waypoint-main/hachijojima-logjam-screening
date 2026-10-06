"""Phase 8 diagnostic for priority classes (the final Map D is assembled in Phase 9)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

COL = {"Priority 1": "#d7191c", "Priority 2": "#fd8d3c", "Priority 3": "#fed976", "Baseline": "#c7c7c7", "Not assessed": "#6a51a3"}


def plot_phase8(hillshade, reaches, transform, pc, out_png, scenario="default", dpi=150):
    inv = ~transform
    tp = lambda x, y: (inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f)
    fig, ax = plt.subplots(2, 2, figsize=(16, 12.5))
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    for k in ("Baseline", "Not assessed", "Priority 3", "Priority 2", "Priority 1"):
        sub = reaches[reaches["priority_class"] == k]
        segs = [np.array([tp(x, y) for x, y in g.coords]) for g in sub.geometry]
        a.add_collection(LineCollection(segs, colors=COL[k], linewidths={"Baseline": 0.5, "Not assessed": 1.0, "Priority 3": 1.3, "Priority 2": 2.0, "Priority 1": 3.0}[k]))
    a.set_xticks([]); a.set_yticks([])
    a.set_title(f"Priority classes by reach — {scenario} (screening; nothing here is a confirmed logjam)", fontsize=10)
    a.legend(handles=[plt.Line2D([0], [0], color=c, lw=3, label=f"{k} (n={(reaches['priority_class'] == k).sum()})") for k, c in COL.items()],
             loc="lower left", fontsize=8)
    a = ax[0, 1]
    for k in ("Baseline", "Not assessed", "Priority 3", "Priority 2", "Priority 1"):
        sub = reaches[reaches["priority_class"] == k]
        a.scatter(sub["susceptibility_score"], sub["observed_change_score"].fillna(0), s=6 if k in ("Baseline", "Not assessed") else 20, c=COL[k],
                  edgecolors="none" if k == "Baseline" else "k", linewidths=0.3, label=k)
    a.axvline(pc["high_susceptibility_min"], color="k", ls="--", lw=0.8); a.axhline(pc["strong_change_min"], color="k", ls="--", lw=0.8)
    a.axhline(pc["moderate_change_min"], color="k", ls=":", lw=0.8)
    a.set_xlabel("susceptibility S"); a.set_ylabel("observed change O"); a.legend(fontsize=8)
    a.set_title("Rule matrix (dashed = strong/high thresholds, dotted = moderate)", fontsize=10)
    a = ax[1, 0]
    for k in ("Priority 1", "Priority 2", "Priority 3", "Baseline"):
        v = reaches.loc[reaches["priority_class"] == k, "confidence_score"]
        if len(v):
            a.hist(v, bins=np.linspace(0, 1, 26), alpha=0.55, color=COL[k], label=f"{k} (median {v.median():.2f})", density=True)
    a.set_xlabel("confidence score (heuristic)"); a.legend(fontsize=8); a.set_title("Confidence by priority class", fontsize=10)
    a = ax[1, 1]
    top = reaches[reaches["priority_class"].isin(["Priority 1", "Priority 2"])].nlargest(20, "priority_score")
    a.barh(top["reach_id"][::-1], top["priority_score"][::-1], color=[COL[k] for k in top["priority_class"][::-1]], edgecolor="k")
    for i, (cs, fl) in enumerate(zip(top["confidence_score"][::-1], top["debris_flag"][::-1])):
        a.text(0.01, i, f"conf {cs:.2f}  {fl[:8]}", va="center", fontsize=7)
    a.set_title("Top 20 Priority 1-2 reaches by priority score", fontsize=10)
    fig.suptitle("Phase 8 — priority inspection classification (UNCALIBRATED rules)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
