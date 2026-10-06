"""Shared cartographic furniture for the four final maps (matplotlib; no Streamlit, no Earth Engine)."""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

FOOTER = ("Screening product from satellite data (Sentinel-1/2, Copernicus DEM). Weights and thresholds are UNCALIBRATED and unvalidated. "
          "CRS: WGS 84 / UTM 54N (EPSG:32654).\n"
          "A logjam is a pile of wood wedged in a channel; a debris accumulation is any build-up of material in or beside it. "
          "'Possible Logjam' = change concentrated along the channel. 'Probable Debris Accumulation' = radar and optical images both show "
          "channel change. Both are hints from satellite change, not confirmed logjams.")


def extent_of(transform, shape):
    h, w = shape
    return (transform.c, transform.c + w * transform.a, transform.f + h * transform.e, transform.f)


def new_map(extent, title, subtitle="", figsize=(11.5, 13.5), margin=0.03):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.07, 0.10, 0.88, 0.80])
    x0, x1, y0, y1 = extent
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    ax.set_xlim(x0 - mx, x1 + mx); ax.set_ylim(y0 - my, y1 + my)
    ax.set_aspect("equal")
    ax.set_facecolor("#dfe8ee")                               # sea
    fig.text(0.07, 0.955, title, fontsize=16, fontweight="bold", ha="left")
    if subtitle:
        fig.text(0.07, 0.928, subtitle, fontsize=10, color="#444", ha="left")
    dec = 0 if (x1 - x0) >= 5000 else 1
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.{dec}f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.{dec}f}")
    ax.tick_params(labelsize=7); ax.set_xlabel("Easting (km, UTM 54N)", fontsize=8); ax.set_ylabel("Northing (km)", fontsize=8)
    ax.grid(color="white", lw=0.4, alpha=0.7)
    fig.text(0.07, 0.015, FOOTER, fontsize=6.3, color="#444", ha="left", va="bottom", wrap=True)
    fig.text(0.95, 0.04, f"Generated {dt.date.today().isoformat()}", fontsize=6.5, color="#666", ha="right")
    return fig, ax


def hillshade(ax, hs, extent, alpha=1.0):
    ax.imshow(hs, extent=extent, cmap="gray", vmin=0, vmax=1, alpha=alpha, zorder=1, interpolation="bilinear")


def north_arrow(ax, x=0.93, y=0.93, size=0.06):
    ax.annotate("", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6), zorder=20)
    ax.text(x, y + 0.008, "N", transform=ax.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=20)


def scale_bar(ax, length_m=2000, x=0.06, y=0.05):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    sx, sy = x0 + (x1 - x0) * x, y0 + (y1 - y0) * y
    while length_m > 0.4 * (x1 - x0):                         # keep the bar proportionate on small extents
        length_m = {2000: 1000, 1000: 500, 500: 200, 200: 100}.get(length_m, length_m / 2)
    h = (y1 - y0) * 0.006
    ax.add_patch(Rectangle((sx, sy), length_m / 2, h, fc="k", ec="k", zorder=20))
    ax.add_patch(Rectangle((sx + length_m / 2, sy), length_m / 2, h, fc="w", ec="k", zorder=20))
    ax.text(sx, sy + h * 2, "0", fontsize=7, ha="center", zorder=20)
    ax.text(sx + length_m, sy + h * 2, (f"{length_m / 1000:g} km" if length_m >= 1000 else f"{length_m:g} m"), fontsize=7, ha="center", zorder=20)


def lines(ax, gdf, colors, widths, zorder=5, linestyles="solid"):
    segs = [np.asarray(g.coords)[:, :2] for g in gdf.geometry]
    lc = LineCollection(segs, colors=colors, linewidths=widths, zorder=zorder, linestyles=linestyles, capstyle="round")
    ax.add_collection(lc)
    return lc


def legend(ax, handles, title=None, loc="lower right", fontsize=8):
    leg = ax.legend(handles=handles, title=title, loc=loc, fontsize=fontsize, title_fontsize=fontsize + 1,
                    framealpha=0.92, facecolor="white", edgecolor="#888")
    leg.set_zorder(30)
    return leg


def finish(fig, out_png, dpi=200):
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
