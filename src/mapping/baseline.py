"""MAP A — baseline map. Phase 1 renders the terrain/hydrology layers only.

Panels that depend on later phases (Sentinel-1/2 baseline, forest cover, existing landslides)
are drawn as explicit "pending" placeholders. Nothing is faked: a layer appears only if its
data exist.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def _extent(transform, shape):
    h, w = shape
    return (transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f)


def _pending(ax, title, need):
    ax.set_facecolor("#f1f1f1")
    ax.text(0.5, 0.5, f"{title}\n\nPENDING\n{need}", ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="#555")
    ax.set_xticks([]); ax.set_yticks([])


def plot_phase1_baseline(dem, hs, slope, acc_cells, cell, transform, reaches, crossings,
                         out_png: Path, title="Baseline (Phase 1: terrain & hydrology)",
                         reference=None, merit_xy=None, extras=None, dpi=200, data_note=""):
    ext = _extent(transform, dem.shape)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5))
    ax = axes[0, 0]
    ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1)
    im = ax.imshow(dem, extent=ext, cmap="terrain", alpha=0.55)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Elevation (m)")
    ax.set_title("Elevation + hillshade")

    ax = axes[0, 1]
    im = ax.imshow(slope, extent=ext, cmap="magma_r", vmin=0, vmax=45)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Slope (deg)")
    ax.set_title("Slope (Horn)")

    ax = axes[0, 2]
    a = np.where(np.isfinite(dem), acc_cells * cell * cell / 1e6, np.nan)
    im = ax.imshow(a, extent=ext, cmap="Blues", norm=LogNorm(vmin=max(np.nanmin(a), 1e-4), vmax=np.nanmax(a)))
    plt.colorbar(im, ax=ax, shrink=0.7, label="Contributing area (km², log)")
    ax.set_title("Flow accumulation")

    ax = axes[1, 0]
    ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.8)
    if len(reaches):
        lw = 0.5 + 0.6 * reaches["stream_order"].to_numpy()
        reaches.plot(ax=ax, column="stream_order", cmap="winter_r", linewidth=lw, legend=True,
                     legend_kwds=dict(label="Strahler order", shrink=0.7), categorical=False)
        conf = reaches[reaches["confluence"]]
        if len(conf):
            ax.scatter([g.coords[-1][0] for g in conf.geometry], [g.coords[-1][1] for g in conf.geometry],
                       s=3, c="yellow", edgecolors="none", label="Confluence reach", zorder=5)
    if reference is not None and len(reference):
        reference.plot(ax=ax, color="red", linewidth=0.6, alpha=0.7, label="Reference hydrography")
    if merit_xy is not None and len(merit_xy):
        ax.scatter(merit_xy[:, 0], merit_xy[:, 1], marker="x", s=16, c="red", linewidths=0.8, label="MERIT Hydro river cells (93 m)", zorder=4)
    if crossings is not None and len(crossings):
        br = crossings[crossings["bridge"]]
        major = crossings[(~crossings["bridge"]) & (crossings.get("road_class", "major") == "major")]
        minor = crossings[(~crossings["bridge"]) & (crossings.get("road_class", "major") == "minor")]
        ax.scatter(minor.geometry.x, minor.geometry.y, marker="s", s=3, c="#9e9e9e", label="Minor (track/service) crossing", zorder=5)
        ax.scatter(major.geometry.x, major.geometry.y, marker="s", s=5, c="k", label="Road crossing (vehicle road)", zorder=6)
        ax.scatter(br.geometry.x, br.geometry.y, marker="D", s=14, c="magenta", label="Bridge (OSM tag)", zorder=7)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("Derived river network, confluences, crossings")

    extras = extras or {}
    def _extra(ax, key, title, cmap, lim, label):
        if key not in extras:
            _pending(ax, title, "run Phase 2 first")
            return
        arr, tr = extras[key]
        im = ax.imshow(arr, extent=_extent(tr, arr.shape), cmap=cmap, vmin=lim[0], vmax=lim[1])
        plt.colorbar(im, ax=ax, shrink=0.7, label=label)
        ax.set_title(title)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    _extra(axes[1, 1], "s1_vv", "Sentinel-1 VV median (baseline, one orbit geometry)", "gray", (-20, 0), "dB")
    _extra(axes[1, 2], "s2_ndvi", "Sentinel-2 NDVI median (baseline)", "YlGn", (0, 1), "NDVI")

    for ax in list(axes[0]) + [axes[1, 0]]:
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(title + (f"\n{data_note}" if data_note else ""), fontsize=13)
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
