"""The four final maps (A baseline, B 2025-2026 change, C susceptibility, D priority inspection).

Each function takes plain arrays / GeoDataFrames (UTM 54N) and writes a PNG. They never invent data: a layer is
drawn only when it is passed in, and every map states what is observed and what is inferred.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import cartography as cg

# One hue per class, well separated (min pairwise CIELAB distance ~43; the old palette had 26). Greys = the two "uncertain" classes.
CHANGE_COL = {1: "#d95f02", 2: "#e7298a", 3: "#3b0f70", 4: "#ffd400", 5: "#00bcd4", 6: "#3d3d3d", 7: "#e0e0e0"}
CHANGE_NAME = {1: "Forest disturbance (possible windthrow / canopy loss)", 2: "Vegetation loss (non-forest)",
               3: "Landslide candidate", 4: "Exposed soil / sediment", 5: "Water-extent change (inland)",
               6: "SAR-only change (uncertain)", 7: "Weak optical change (uncertain)"}
SUS_COL = {"Low": "#ffffb2", "Moderate": "#fecc5c", "High": "#fd8d3c", "Very High": "#bd0026"}
PRI_COL = {"Priority 1": "#d7191c", "Priority 2": "#fd8d3c", "Priority 3": "#fed976", "Baseline": "#b0b0b0", "Not assessed": "#6a51a3"}
EVID_MARK = {"Probable Debris Accumulation": ("D", "#e31a1c"), "Possible Logjam": ("o", "#ff7f00")}


def _rgb(b4, b3, b2, lo=2, hi=98, gamma=0.8, land=None):
    """True-colour RGBA. The contrast stretch uses LAND pixels only (open water would otherwise dominate); sea is transparent."""
    out = []
    for b in (b4, b3, b2):
        v = np.where(np.isfinite(b), b, np.nan)
        ref = v[land] if land is not None and np.isfinite(v[land]).any() else v
        p0, p1 = np.nanpercentile(ref, [lo, hi])
        out.append(np.clip((v - p0) / max(p1 - p0, 1e-9), 0, 1) ** gamma)
    rgb = np.dstack(out)
    ok = np.isfinite(rgb).all(axis=2) & (land if land is not None else True)
    return np.dstack([np.where(np.isfinite(rgb), rgb, 0.0), ok.astype(float)])


def _mid_xy(g):
    return np.array([(p.x, p.y) for p in g.geometry.interpolate(0.5, normalized=True)]) if len(g) else np.empty((0, 2))


def map_a_baseline(out_png, ext, hs, s2_rgb, forest, reaches, crossings, windows, dpi=200, view=None):
    fig, ax = cg.new_map(view or ext, "Map A — Baseline: terrain, drainage and forest",
                         f"Hachijojima, Tokyo | baseline imagery {windows['baseline'][0]} to {windows['baseline'][1]} (Sentinel-2 median composite)")
    if s2_rgb is not None:
        ax.imshow(s2_rgb, extent=ext, zorder=1, interpolation="bilinear")
        ax.imshow(hs, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.35, zorder=2, interpolation="bilinear")
    else:
        cg.hillshade(ax, hs, ext)
    if forest is not None:
        ax.imshow(np.where(forest, 1.0, np.nan), extent=ext, cmap=ListedColormap(["#1a9850"]), alpha=0.18, zorder=3, interpolation="nearest")
    o = reaches["stream_order"].to_numpy()
    cg.lines(ax, reaches, ["#2b83ba" if k <= 2 else "#08519c" for k in o], 0.5 + 0.7 * (o - 1), zorder=6)
    handles = [Line2D([0], [0], color="#2b83ba", lw=1, label="Derived stream, order 1-2"),
               Line2D([0], [0], color="#08519c", lw=2, label="Derived stream, order 3+"),
               Patch(fc="#1a9850", alpha=0.35, label="Forest (WorldCover tree cover AND baseline NDVI)")]
    if crossings is not None and len(crossings):
        for cls_, mk, lab in (("major", "s", "Road crossing, major (on reaches draining >= 0.2 km2)"), ("minor", "^", "Road crossing, minor (on reaches draining >= 0.2 km2)")):
            sub = crossings[crossings.get("road_class", "") == cls_] if "road_class" in crossings else crossings.iloc[0:0]
            if len(sub):
                ax.scatter([p.x for p in sub.geometry], [p.y for p in sub.geometry], marker=mk, s=7, c="k", alpha=0.7, linewidths=0, zorder=8)
                handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor="k", markersize=6, label=lab))
    cg.legend(ax, handles, "Legend")
    cg.north_arrow(ax); cg.scale_bar(ax)
    return cg.finish(fig, out_png, dpi)


def map_b_change(out_png, ext, hs, cls, coast_m, reaches, windows, scenario_label, coastal_exclusion_m=100, dpi=200, view=None):
    fig, ax = cg.new_map(view or ext, "Map B — Observed change, " + scenario_label,
                         f"Baseline {windows['baseline'][0]}..{windows['baseline'][1]}  →  after {windows['after'][0]}..{windows['after'][1]} | "
                         "OBSERVED satellite change only (hazard inference is on Maps C and D)")
    cg.hillshade(ax, hs, ext)
    show = np.where((cls >= 1) & (cls <= 7), cls, 0).astype(float)
    show[(cls == 5) & (coast_m < coastal_exclusion_m)] = 0            # coastal MNDWI shifts are treated as artefacts
    rgba = np.zeros(cls.shape + (4,))
    for k, c in CHANGE_COL.items():
        rgba[show == k] = matplotlib.colors.to_rgba(c, 0.9)
    ax.imshow(rgba, extent=ext, zorder=3, interpolation="nearest")
    cg.lines(ax, reaches, "#08519c", 0.5, zorder=6)
    handles = [Patch(fc=c, label=CHANGE_NAME[k]) for k, c in CHANGE_COL.items()]
    handles.append(Line2D([0], [0], color="#08519c", lw=1, label="Derived river reaches"))
    ev = reaches[reaches["debris_flag"].fillna("") != ""] if "debris_flag" in reaches else reaches.iloc[0:0]
    for k, (mk, c) in EVID_MARK.items():
        sub = ev[ev["debris_flag"] == k]
        if len(sub):
            xy = _mid_xy(sub)
            ax.scatter(xy[:, 0], xy[:, 1], marker=mk, s=60, facecolor="none", edgecolor=c, linewidths=1.6, zorder=9)
            handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor="none", markeredgecolor=c, markersize=8,
                                  label=f"{k} (satellite evidence; hypothesis)"))
    cg.legend(ax, handles, "Observed change", fontsize=7.5)
    ax.text(0.01, 0.01, f"Inland water-extent flags only; flags within {coastal_exclusion_m} m of the sea suppressed.", transform=ax.transAxes,
            fontsize=6.5, color="#444", zorder=20)
    cg.north_arrow(ax); cg.scale_bar(ax)
    return cg.finish(fig, out_png, dpi)


def map_c_susceptibility(out_png, ext, hs, source_pot, reaches, scenario_label, dpi=200, view=None):
    n = {k: int((reaches["susceptibility_class"] == k).sum()) for k in SUS_COL}
    fig, ax = cg.new_map(view or ext, "Map C — Logjam susceptibility by river reach",
                         f"Rule-based, reach-level index (0-1) | {scenario_label} | normalised, NOT validated against observed logjams")
    cg.hillshade(ax, hs, ext)
    if source_pot is not None:
        ax.imshow(np.where(np.isfinite(source_pot), source_pot, np.nan), extent=ext, cmap="Purples", vmin=0, vmax=1, alpha=0.45, zorder=3,
                  interpolation="nearest")
    order = ["Low", "Moderate", "High", "Very High"]
    for k in order:
        sub = reaches[reaches["susceptibility_class"] == k]
        if len(sub):
            cg.lines(ax, sub, SUS_COL[k], {"Low": 0.7, "Moderate": 1.2, "High": 2.2, "Very High": 3.2}[k], zorder=6 + order.index(k))
    handles = [Line2D([0], [0], color=SUS_COL[k], lw=3, label=f"{k if k != 'High' else 'High Logjam Susceptibility'} (n={n[k]})") for k in order]
    handles.append(Patch(fc="#9e9ac8", alpha=0.6, label="Inferred wood-source areas (disturbed forest, mobility-weighted)"))
    cg.legend(ax, handles, "Susceptibility class", fontsize=7.5)
    cg.north_arrow(ax); cg.scale_bar(ax)
    return cg.finish(fig, out_png, dpi)


def map_d_priority(out_png, ext, hs, reaches, scenario_label, windows, n_label=10, dpi=200, view=None):
    cnt = {k: int((reaches["priority_class"] == k).sum()) for k in PRI_COL}
    fig, ax = cg.new_map(view or ext, "Map D — Priority inspection locations",
                         f"{scenario_label} | susceptibility × observed change | line style = confidence | "
                         "a screening list, not a list of confirmed logjams")
    cg.hillshade(ax, hs, ext, alpha=0.9)
    ls = {"High": "solid", "Medium": (0, (5, 2)), "Low": (0, (1, 1.5))}
    wd = {"Baseline": 0.5, "Not assessed": 1.0, "Priority 3": 1.4, "Priority 2": 2.4, "Priority 1": 3.6}
    for k in ("Baseline", "Not assessed", "Priority 3", "Priority 2", "Priority 1"):
        for cc in ("High", "Medium", "Low"):
            sub = reaches[(reaches["priority_class"] == k) & (reaches["confidence_class"] == cc)]
            if len(sub):
                cg.lines(ax, sub, PRI_COL[k], wd[k], zorder=5 + list(wd).index(k), linestyles=ls[cc])
    top = reaches[reaches["priority_class"].isin(["Priority 1", "Priority 2"])].sort_values("priority_score", ascending=False).head(n_label)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        pt = r.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(i), (pt.x, pt.y), xytext=(0, 0), textcoords="offset points", ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=12,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=PRI_COL[r["priority_class"]] if r["priority_class"] == "Priority 1" else "#e6550d", ec="k", lw=0.6))
    handles = [Line2D([0], [0], color=PRI_COL[k], lw=wd[k] + 1, label=f"{k}{' (Priority Inspection Location)' if k in ('Priority 1', 'Priority 2') else ''} (n={cnt[k]})")
               for k in ("Priority 1", "Priority 2", "Priority 3", "Baseline", "Not assessed")]
    handles += [Line2D([0], [0], color="k", lw=1.6, ls=ls["High"], label="Confidence: High"),
                Line2D([0], [0], color="k", lw=1.6, ls=ls["Medium"], label="Confidence: Medium"),
                Line2D([0], [0], color="k", lw=1.6, ls=ls["Low"], label="Confidence: Low")]
    cg.legend(ax, handles, "Priority class", fontsize=7.5)
    if len(top):
        rows = ["#  reach   class       score  flag                      data conf."]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            rows.append(f"{i:<2} {r['reach_id']}  {r['priority_class']:<10}  {r['priority_score']:.2f}  {str(r['debris_flag'])[:24]:<24}  {r['confidence_score']:.2f}")
        ax.text(0.015, 0.985, "Visit order: top Priority 1+2 reaches by score\n" + "\n".join(rows), transform=ax.transAxes, va="top", ha="left", fontsize=6.3,
                family="monospace", bbox=dict(fc="white", ec="#888", alpha=0.93), zorder=25)
    cg.north_arrow(ax); cg.scale_bar(ax)
    return cg.finish(fig, out_png, dpi)
