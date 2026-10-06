"""Phase 5 diagnostic: wood delivery by reach (inference; not observation)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def plot_phase5(hillshade, reaches, transform, entry_dist, eff_area, wm, out_png, scenario="default", dpi=150):
    inv = ~transform
    segs = [np.array([(inv.a * x + inv.b * y + inv.c, inv.d * x + inv.e * y + inv.f) for x, y in g.coords])
            for g in reaches.geometry]
    fig, ax = plt.subplots(2, 2, figsize=(16, 12.5))
    a = ax[0, 0]
    a.imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    lc = LineCollection(segs, array=reaches["wood_delivery_score"].to_numpy(), cmap="magma_r", linewidths=1.6)
    lc.set_clim(0, 1)
    a.add_collection(lc)
    plt.colorbar(lc, ax=a, shrink=0.7, label="wood_delivery_score (0-1, relative index)")
    a.set_title(f"Reach wood-delivery score — scenario: {scenario}", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    # distance distribution + assumed decay
    a = ax[0, 1]
    m = np.isfinite(entry_dist) & (eff_area > 0)
    a.hist(entry_dist[m], bins=np.arange(0, 1001, 30), weights=eff_area[m], color="#1f78b4", alpha=0.8)
    a2 = a.twinx()
    x = np.arange(0, 1001, 10)
    f = np.where(x <= wm["delivery_max_distance_m"], np.exp(-x / wm["distance_decay_scale_m"]), 0)
    a2.plot(x, f, color="#d95f02"); a2.set_ylabel("assumed delivery factor", color="#d95f02")
    a.set_xlabel("D8 flow-path distance from source cell to first stream cell (m)")
    a.set_ylabel("effective source area (m²)")
    a.set_title("Where source area sits relative to channels (bars) and the assumed decay (line)", fontsize=9)
    # supply vs area
    a = ax[1, 0]
    ok = reaches["upstream_wood_supply_m2"] > 0
    sc = a.scatter(reaches.loc[ok, "upstream_area_m2"] / 1e6, reaches.loc[ok, "upstream_wood_supply_m2"],
                   c=reaches.loc[ok, "stream_order"], cmap="viridis", s=10)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("upstream area (km²)"); a.set_ylabel("upstream wood supply (effective m²)")
    plt.colorbar(sc, ax=a, label="Strahler order")
    a.set_title("Supply vs. catchment size (supply should rise with area; scatter shows disturbance pattern)", fontsize=9)
    # top reaches
    a = ax[1, 1]
    top = reaches.nlargest(15, "wood_delivery_score")
    a.barh(top["reach_id"][::-1], top["wood_delivery_score"][::-1], color="#7b3294")
    a.set_title("Top 15 reaches by wood-delivery score", fontsize=10)
    fig.suptitle("Phase 5 — wood delivery to river reaches (UNCALIBRATED relative index)", fontsize=13)
    fig.tight_layout(); fig.savefig(out_png, dpi=dpi); plt.close(fig)
    return out_png
