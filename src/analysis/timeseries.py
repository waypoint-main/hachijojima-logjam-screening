"""Pure-pandas analysis of island time series (no Earth Engine needed): season-matched comparison and
empirical event timing.

Idea: storm damage is usually directional (wind-exposed slopes). The DIFFERENCE between storm-facing and
sheltered forest slopes removes the seasonal/phenology cycle that both share, so a STEP in that
difference marks when exposure-related damage occurred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def exposure_difference(df: pd.DataFrame, var: str, exposed: str = "exposed", sheltered: str = "sheltered"):
    """df: columns date, <exposed>_<var>, <sheltered>_<var>. Returns Series (exposed - sheltered) by date."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    return (d[f"{exposed}_{var}"] - d[f"{sheltered}_{var}"]).dropna()


def detect_step(series: pd.Series, min_seg: int = 4, ci_boot: int = 0, seed: int = 0) -> dict:
    """Single change-point by minimum total squared error of a two-level step model.

    Returns {'date', 'before_mean', 'after_mean', 'step', 'sse_ratio'}; sse_ratio = SSE(step)/SSE(flat)
    (lower = stronger evidence; ~1 means no step). NOT a significance test: use with the plot.
    """
    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 2 * min_seg:
        raise ValueError("series too short for step detection")
    sse_flat = float(((y - y.mean()) ** 2).sum())
    best = None
    for k in range(min_seg, n - min_seg + 1):
        a, b = y[:k], y[k:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k)
    sse, k = best
    return dict(date=series.index[k], last_before=series.index[k - 1], before_mean=float(y[:k].mean()),
                after_mean=float(y[k:].mean()), step=float(y[k:].mean() - y[:k].mean()),
                sse_ratio=sse / sse_flat if sse_flat > 0 else 1.0)


def doy_overlay(series: pd.Series) -> dict:
    """{year: Series indexed by day-of-year} for season-matched overlays."""
    out = {}
    for yr, g in series.groupby(series.index.year):
        out[int(yr)] = pd.Series(g.to_numpy(), index=g.index.dayofyear)
    return out


def make_figure(s2: pd.DataFrame, s1: pd.DataFrame, out_png, min_cover: float = 0.3):
    """Exposed-vs-sheltered time series figure + best single step per variable.

    s2: date, exposed_/sheltered_{NDVI,NBR} and exposed_NDVI_count.   s1: same for VH/VV (+ exposed_VH_count).
    Scenes with fewer than `min_cover` x the maximum valid-pixel count are dropped (cloud/layover gaps).
    Returns {var: step dict or {'error': ...}}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s2 = s2[s2["exposed_NDVI_count"] >= min_cover * s2["exposed_NDVI_count"].max()].copy()
    s1 = s1[s1["exposed_VH_count"] >= min_cover * s1["exposed_VH_count"].max()].copy()
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))
    res = {}
    for row, (df, var, lab) in enumerate([(s2, "NDVI", "S2 NDVI"), (s2, "NBR", "S2 NBR"), (s1, "VH", "S1 VH (dB)")]):
        d = df.copy(); d["date"] = pd.to_datetime(d["date"]); d = d.sort_values("date")
        ax[row, 0].plot(d["date"], d[f"exposed_{var}"], ".-", label="exposed (N-E facing)", color="#d95f02")
        ax[row, 0].plot(d["date"], d[f"sheltered_{var}"], ".-", label="sheltered (S-W facing)", color="#1b9e77")
        ax[row, 0].set_title(f"{lab} - forest slopes >=15 deg"); ax[row, 0].legend(fontsize=8)
        diff = exposure_difference(df, var)
        for yr, ser in doy_overlay(diff).items():
            ax[row, 1].plot(ser.index, ser.values, ".-", label=str(yr))
        ax[row, 1].axhline(0, color="k", lw=0.5)
        ax[row, 1].set_xlabel("day of year")
        ax[row, 1].set_title(f"{lab}: exposed - sheltered, by day of year"); ax[row, 1].legend()
        try:
            r = detect_step(diff); res[var] = {k: str(v) for k, v in r.items()}
            ax[row, 0].axvline(r["date"], color="k", ls="--", lw=1)
        except ValueError as e:
            res[var] = {"error": str(e)}
    fig.suptitle("Island time series - exposed vs sheltered forest slopes (dashed = best single step in the difference)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return res
