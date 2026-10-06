"""Hachijojima logjam-susceptibility screening — Streamlit front end (Phase 10).

Run from the project root:   streamlit run app/streamlit_app.py
All analytics live in src/ and are Streamlit-free; this file only reads the Phase 8/9 outputs and displays them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import pydeck as pdk                     # noqa: E402
import streamlit as st                   # noqa: E402

from src import glossary as gl           # noqa: E402
from src import query as q               # noqa: E402
from src.config import load_config       # noqa: E402

st.set_page_config(page_title="Hachijojima debris & logjam screening", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

PRIORITY_RGB = {"Priority 1": [215, 25, 28], "Priority 2": [253, 141, 60], "Priority 3": [254, 217, 118], "Baseline": [176, 176, 176], "Not assessed": [106, 81, 163]}
SUS_RGB = {"Low": [255, 255, 178], "Moderate": [254, 204, 92], "High": [253, 141, 60], "Very High": [189, 0, 38]}
CONF_RGB = {"Low": [189, 189, 189], "Medium": [116, 169, 207], "High": [5, 112, 176]}
# label -> (file tag, config scenario name). The post-event window (pivot: early October 2025) is the default.
SCENARIOS = {"After the October 2025 event (recommended)": ("_post_event", "post_event"),
             "Whole-year comparison, 2025 vs 2026": ("", None)}
SHORT_NAME = {"_post_event": "post-event window", "": "whole-year window"}
FOREST, RIVER, SOIL = "#1b6b5f", "#1f6f9f", "#8a6d3b"

CSS = """
<style>
.block-container{padding-top:3.6rem;max-width:1500px}
.hero{background:linear-gradient(120deg,#0f3d3e 0%,#1b6b5f 55%,#1f6f9f 100%);color:#fff;border-radius:14px;padding:22px 28px 18px;margin-bottom:10px}
.hero h1{color:#fff;font-size:1.75rem;line-height:1.2;margin:0 0 4px;padding:0;font-weight:700}
.hero p{color:#e3f1ec;margin:0;font-size:.97rem}
.hero .pill{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.3);border-radius:999px;padding:3px 12px;margin:10px 8px 0 0;font-size:.8rem;color:#fff}
.note{background:#fff8e6;border-left:4px solid #d9a21b;border-radius:6px;padding:7px 12px;font-size:.84rem;color:#4a3b10;margin-bottom:6px}
.step{min-height:158px;background:#fff;border:1px solid #d5dfd6;border-top:4px solid #1b6b5f;border-radius:10px;padding:12px 14px;height:100%}
.step b{color:#1b6b5f}
.step span{font-size:.88rem;color:#3a4a43}
div[data-testid="stMetric"]{background:#fff;border:1px solid #d5dfd6;border-left:5px solid #1b6b5f;border-radius:10px;padding:10px 14px}
div[data-testid="stMetricValue"]{color:#12433b}
div[data-testid="stMetricDelta"] svg{display:none}
div[data-testid="stMetricLabel"] p{white-space:normal;overflow:visible}
.stTabs [data-baseweb="tab-list"]{gap:2px;border-bottom:2px solid #d5dfd6}
.stTabs [data-baseweb="tab"]{padding:8px 14px;font-weight:600;color:#44564e}
.stTabs [aria-selected="true"]{color:#1b6b5f}
section[data-testid="stSidebar"]{border-right:1px solid #d5dfd6}
section[data-testid="stSidebar"] h1{font-size:1.15rem;color:#12433b}
.legend{font-size:.8rem;line-height:2.1;background:#fff;border:1px solid #d5dfd6;border-radius:8px;padding:4px 12px;margin-top:6px}
.legend i{display:inline-block;width:13px;height:13px;border-radius:3px;margin:0 5px -2px 0;border:1px solid rgba(0,0,0,.35)}
.legend span{margin-right:14px;white-space:nowrap}
.status{font-size:.88rem;color:#2c4a42;background:#e6ede6;border-radius:8px;padding:6px 12px;margin:2px 0 8px}
.foot{font-size:.78rem;color:#5b6b63;border-top:1px solid #d5dfd6;margin-top:28px;padding-top:8px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
DISCLAIMER = ("**Screening, not a finding.** Rule-based scores from satellite data, **not validated** against real logjams. "
              "\"Possible Logjam\" and \"Probable Debris Accumulation\" are hypotheses to check on the ground. Click ⓘ or ? for plain-language explanations.")


def info(*keys, label="ⓘ"):
    """A small ⓘ button that opens a plain-language explanation of one or more glossary terms."""
    body = "\n\n---\n\n".join(gl.text(k) for k in keys)
    if hasattr(st, "popover"):
        with st.popover(label):
            st.markdown(body)
    else:                                                    # very old Streamlit: fall back to an expander
        with st.expander(label):
            st.markdown(body)


def head(text, *keys, level=3):
    """Heading with an ⓘ explanation beside it."""
    c1, c2 = st.columns([8, 1])
    c1.markdown(f"{'#' * level} {text}")
    with c2:
        info(*keys)


def tip(key):
    return gl.text(key)


def num(v, fmt="{:.2f}"):
    return "not measured" if v is None or (isinstance(v, float) and v != v) else fmt.format(v)


@st.cache_resource
def _cfg():
    return load_config(os.environ.get("HACHIJOJIMA_CONFIG") or None)   # env override is used by the tests


@st.cache_data(show_spinner=False)
def _load(tag: str):
    out = _cfg().path("outputs")
    p = out / f"river_reaches{tag}.gpkg"
    return q.load_reaches(p) if p.exists() else None


VIEWS = {"A — Baseline": "A", "B — Observed change": "B", "C — Logjam susceptibility": "C", "D — Priority inspection": "D",
         "Reaches only (colour by priority)": "R"}
GREY = [150, 150, 150, 110]


@st.cache_data(show_spinner=False)
def _overlays(tag: str):
    """{layer: (URL, [w, s, e, n])}. Phase 9 writes the PNGs to data/outputs/overlays{tag}; they are mirrored into
    app/static/ so Streamlit can serve them (deck.gl cannot take data: URIs through pydeck's JSON converter)."""
    import shutil
    src = _cfg().path("outputs", f"overlays{tag}")
    idx = src / "overlays.json"
    if not idx.exists():
        return {}
    dst = Path(__file__).resolve().parent / "static" / f"overlays{tag}"
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, m in json.loads(idx.read_text()).items():
        f = src / m["file"]
        if not f.exists():
            continue
        # version in the file name (no '?' / ':' - deck.gl's JSON converter parses strings as expressions)
        t = dst / f"{f.stem}_{int(f.stat().st_mtime)}{f.suffix}"
        if not t.exists():
            for old in dst.glob(f"{f.stem}_*{f.suffix}"):
                old.unlink()
            shutil.copy2(f, t)
        out[name] = (f"app/static/overlays{tag}/{t.name}", m["bounds"])
    return out


PHOTO_DIR = ROOT / "field_photos"


@st.cache_data(show_spinner=False)
def _photos():
    return q.load_field_photos(PHOTO_DIR / "photos.csv")


def _bitmap(ov, name, opacity):
    if name not in ov:
        return None
    img, b = ov[name]
    return pdk.Layer("BitmapLayer", image=img, bounds=b, opacity=opacity, pickable=False)


def _legend(items):
    chips = "".join(f"<span><i style='background:{c}'></i>{t}</span>" for t, c in items)
    st.markdown(f"<div class='legend'>{chips}</div>", unsafe_allow_html=True)


def _rgb_hex(c):
    return "#%02x%02x%02x" % tuple(c[:3])


def _fit_zoom(bounds, height_px, width_px=1000, fill=0.92):
    """Web-Mercator zoom at which the (west, south, east, north) box fills `fill` of the map area."""
    import math
    w, s_, e, n = bounds
    merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * 180 / math.pi        # degrees of mercator-y
    dx, dy = max(e - w, 1e-6), max(merc(n) - merc(s_), 1e-6)
    return round(min(math.log2(width_px * fill * 360 / (512 * dx)), math.log2(height_px * fill * 360 / (512 * dy))), 2)


def _lines(df, cols, px, pickable, rounded):
    """One GeoJsonLayer of reaches drawn `px` pixels wide at every zoom."""
    kw = dict(line_joint_rounded=True, line_cap_rounded=True) if rounded else {}
    return pdk.Layer("GeoJsonLayer", q.to_geojson_wgs84(df, cols), get_line_color="properties.color", get_line_width=1,
                     line_width_min_pixels=px, line_width_max_pixels=px, pickable=pickable, auto_highlight=pickable, **kw)


def _map(data, sel, view, tag, opacity=0.85, show_all=True, selected=None, height=620, zoom=None, center=None):
    from src.mapping import final_maps as fm
    ov = _overlays(tag)
    d = data[data["reach_id"].isin(sel["reach_id"])] if len(sel) else data.iloc[0:0]
    if view == "A":                                                    # the baseline map is context: show every reach, ignore the filters
        d = data
    rest = data[~data["reach_id"].isin(d["reach_id"])] if show_all else data.iloc[0:0]
    d = d.copy()
    layers, legend = [], []
    if view == "A":
        layers += [_bitmap(ov, "baseline_rgb", 1.0), _bitmap(ov, "hillshade", 0.30), _bitmap(ov, "forest", min(opacity, 0.35))]
        d["color"] = [[43, 131, 186, 255] if o <= 2 else [8, 81, 156, 255] for o in d["stream_order"]]
        d["grp"] = np.where(d["stream_order"] <= 2, "order 1-2", "order 3+")
        px = {"order 1-2": 1.2, "order 3+": 2.6}
        legend = [("Stream order 1-2", "#2b83ba"), ("Stream order 3+", "#08519c"), ("Forest (WorldCover AND baseline NDVI)", "#1a9850")]
    elif view == "B":
        layers += [_bitmap(ov, "hillshade", 1.0), _bitmap(ov, "change", opacity)]
        d["color"] = [[8, 81, 156, 255]] * len(d); d["grp"] = "all"; px = {"all": 1.2}
        legend = [(fm.CHANGE_NAME[k], c) for k, c in fm.CHANGE_COL.items()]
    elif view == "C":
        layers += [_bitmap(ov, "hillshade", 1.0), _bitmap(ov, "wood_source", opacity)]
        d["color"] = _rgb_col(d["susceptibility_class"], SUS_RGB)
        d["grp"] = d["susceptibility_class"]
        px = {"Low": 1.0, "Moderate": 2.0, "High": 4.0, "Very High": 6.0}
        legend = [("High Logjam Susceptibility" if k == "High" else k, _rgb_hex(v)) for k, v in SUS_RGB.items()] + [("Inferred wood-source area", "#9e9ac8")]
    else:
        layers += [_bitmap(ov, "hillshade", 0.9 if view == "D" else 1.0)]
        d["color"] = _rgb_col(d["priority_class"], PRIORITY_RGB, d["confidence_class"])
        d["grp"] = d["priority_class"]
        px = {"Priority 1": 6.0, "Priority 2": 4.0, "Priority 3": 2.5, "Baseline": 1.2, "Not assessed": 2.0}
        legend = [(k + (" (Priority Inspection Location)" if k in ("Priority 1", "Priority 2") else ""), _rgb_hex(v)) for k, v in PRIORITY_RGB.items()]
    cols = ["reach_id", "priority_class", "priority_score", "susceptibility_class", "susceptibility_score", "observed_change_score",
            "confidence_class", "debris_flag", "color"]
    # NOTE: line widths use deck.gl's min/max-pixel clamps (constant on-screen width). `line_width_units="pixels"` is avoided: it renders
    # as a screen-filling block in some WebGL set-ups.
    if len(rest):
        r = rest.copy(); r["color"] = [GREY] * len(r)
        layers.append(_lines(r, cols, 0.8, pickable=False, rounded=False))
    for g_, sub in d.groupby("grp"):
        layers.append(_lines(sub, cols, px.get(g_, 1.2), pickable=True, rounded=True))
    if view in ("B", "C") and len(d):                                   # satellite-evidence hypotheses (as on the static Map B)
        ev = q.midpoints(d[d["debris_flag"].fillna("") != ""])
        if len(ev):
            ev["rgb"] = [[227, 26, 28] if f == "Probable Debris Accumulation" else [255, 127, 0] for f in ev["debris_flag"]]
            layers.append(pdk.Layer("ScatterplotLayer", ev, get_position=["lon", "lat"], get_fill_color=[255, 255, 255, 0], get_line_color="rgb",
                                    stroked=True, filled=False, get_radius=14, radius_min_pixels=7, line_width_min_pixels=2, pickable=False))
            legend += [("Possible Logjam (hypothesis)", "#ff7f00"), ("Probable Debris Accumulation (hypothesis)", "#e31a1c")]
    if view == "D" and len(d):
        top = q.ranked(d[d["priority_class"].isin(["Priority 1", "Priority 2"])]).head(15)
        p1 = q.midpoints(top).reset_index(drop=True)
        if len(p1):
            p1["n"] = [str(i + 1) for i in range(len(p1))]
            layers.append(pdk.Layer("TextLayer", p1, get_position=["lon", "lat"], get_text="n", get_size=15, get_color=[255, 255, 255],
                                    get_background_color=[215, 25, 28], background=True, get_text_anchor="'middle'", pickable=False))
    ph = _photos()
    if len(ph) and selected is None:                                    # field photos: ring + label on every map view
        layers.append(pdk.Layer("ScatterplotLayer", ph, get_position=["lon", "lat"], get_fill_color=[255, 255, 255, 60], get_line_color=[20, 20, 20],
                                stroked=True, filled=True, get_radius=18, radius_min_pixels=9, line_width_min_pixels=3, pickable=False))
        ph = ph.assign(label=["Photo " + str(i) for i in ph["photo_id"]])
        layers.append(pdk.Layer("TextLayer", ph, get_position=["lon", "lat"], get_text="label", get_size=13, get_color=[20, 20, 20],
                                get_background_color=[255, 255, 255, 235], background=True, get_pixel_offset=[0, -22], pickable=False))
    if selected is not None:
        sl = data[data["reach_id"] == selected].copy(); sl["color"] = [[0, 0, 0, 255]] * len(sl)
        layers.append(_lines(sl, ["reach_id", "color"], 9.0, pickable=False, rounded=True))
    if center is None:
        b = data.to_crs(4326).total_bounds
        center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
        zoom = zoom or _fit_zoom(b, height)
    tip = {"html": "<b>{reach_id}</b><br/>{priority_class} ({priority_score})<br/>Susceptibility: {susceptibility_class} ({susceptibility_score})"
                   "<br/>Observed change: {observed_change_score}<br/>Confidence: {confidence_class}<br/>{debris_flag}"}
    st.pydeck_chart(pdk.Deck(layers=[l for l in layers if l is not None], map_style="light", tooltip=tip,
                             initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1], zoom=zoom or 12)), height=height)
    _legend(legend)
    if view in ("A", "B", "C", "D") and not ov:
        st.caption("Raster layers not found — re-run `python -m src.pipeline --phase 9` to add them. Showing reaches only.")
    elif ov and not st.get_option("server.enableStaticServing"):
        st.warning("Raster layers need static file serving. Start the app from the project folder (it contains `.streamlit/config.toml`) "
                   "or add `--server.enableStaticServing true`.")


def _bars(counts, palette, order):
    """Horizontal bar chart coloured with the same palette as the maps."""
    import altair as alt
    df = pd.DataFrame({"class": list(order), "reaches": [int(counts.get(k, 0)) for k in order]})
    base = alt.Chart(df).encode(y=alt.Y("class:N", sort=list(order), title=None))
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("reaches:Q", title=None, axis=None),
        color=alt.Color("class:N", scale=alt.Scale(domain=list(order), range=[palette[k] for k in order]), legend=None),
        tooltip=["class", "reaches"])
    labels = base.mark_text(align="left", dx=4, fontSize=12, color="#1c2a24").encode(x="reaches:Q", text="reaches:Q")
    ch = (bars + labels).properties(height=34 * len(order) + 10).configure_view(strokeWidth=0)
    st.altair_chart(ch, use_container_width=True)


def _rgb_col(classes, table, conf=None):
    alpha = {"High": 255, "Medium": 200, "Low": 120}
    out = []
    for i, v in enumerate(classes):
        c = table.get(v, [120, 120, 120])
        out.append(c + [alpha.get(conf.iloc[i], 255) if conf is not None else 255])
    return out


# ---------------------------------------------------------------- sidebar
VIEW_TERMS = {"A": ["reach", "stream_order", "crossing", "hillshade", "forest_disturbance"],
              "B": ["observed_change", "forest_disturbance", "landslide_candidate", "sar", "optical", "possible_logjam", "probable_debris"],
              "C": ["susceptibility", "high_susceptibility", "wood_delivery", "possible_logjam", "probable_debris"],
              "D": ["priority", "priority_inspection", "priority_score", "confidence", "small_catchment"],
              "R": ["priority", "reach", "confidence"]}
MAP_TERMS = {"Map A — Baseline": "A", "Map B — Observed change": "B", "Map C — Logjam susceptibility": "C", "Map D — Priority inspection": "D"}


def terms_box(keys, title="What do these terms mean?"):
    with st.expander(title):
        for k in keys:
            st.markdown(gl.text(k))


def _window_text(sc_name):
    c = _cfg().for_scenario(sc_name)
    return (f"Before: {c.get('periods.baseline_start')} to {c.get('periods.baseline_end')}  ·  "
            f"After: {c.get('periods.after_start')} to {c.get('periods.after_end')}")


avail = {lab: v for lab, v in SCENARIOS.items() if _load(v[0]) is not None}
if not avail:
    st.error("No `river_reaches*.gpkg` found in data/outputs. Run `python -m src.pipeline --phase 9 --scenario post_event` first.")
    st.stop()

st.sidebar.title("Controls")
scen_label = st.sidebar.radio("Comparison window", list(avail), help=tip("windows"))
tag, scen_name = avail[scen_label]
other_tag = "" if tag else "_post_event"
data = _load(tag)

PRESETS = {"Inspect first: Priority 1 + 2": ["Priority 1", "Priority 2"],
           "Inspect first + watch list: Priority 1-3": ["Priority 1", "Priority 2", "Priority 3"],
           "All reaches": None,
           "Custom filters": "custom"}
preset = st.sidebar.selectbox("Which reaches to show", list(PRESETS), help="Start with the first option: it lists the places to check first. "
                              "'Custom filters' lets you pick classes yourself.")
with st.sidebar.expander("More filters"):
    pri = PRESETS[preset]
    if pri == "custom":
        pri = st.multiselect("Priority class", q.PRIORITY_ORDER, default=["Priority 1", "Priority 2"], help=tip("priority"))
    sus = st.multiselect("Susceptibility class", q.SUSCEPTIBILITY_ORDER, default=[], help=tip("susceptibility"))
    conf = st.multiselect("Data confidence", q.CONFIDENCE_ORDER, default=[], help=tip("confidence"))
    min_s = st.slider("Min susceptibility score", 0.0, 1.0, 0.0, 0.05, help=tip("susceptibility"))
    min_o = st.slider("Min observed-change score", 0.0, 1.0, 0.0, 0.05, help=tip("observed_change"))
    max_ord = int(data["stream_order"].max()) if "stream_order" in data else 1
    min_ord = st.slider("Min stream order", 1, max(max_ord, 2), 1, help=tip("stream_order"))
    min_area = st.slider("Min upstream catchment (km²)", 0.0, 1.0, 0.0, 0.05,
                         help="Hides reaches that drain less land than this. " + gl.short("small_catchment") + " Try 0.2 to drop likely dry gullies.")
    debris = st.checkbox("Only reaches with a satellite debris flag", help=gl.text("possible_logjam") + "\n\n" + gl.text("probable_debris"))
    xing = st.checkbox("Only reaches at a road crossing / bridge", help=tip("crossing"))
sel = q.filter_reaches(data, pri or None, sus or None, conf or None, min_s, min_o, debris, xing, min_ord, min_upstream_km2=min_area)
rk = q.ranked(sel)

st.markdown(f"""<div class='hero'><h1>Hachijojima typhoon debris &amp; logjam screening</h1>
<p>Satellite screening of forest damage and river corridors on Hachijojima, Tokyo: where wood may have come from, and which river reaches to check first.</p>
<span class='pill'>{scen_label}</span><span class='pill'>{_window_text(scen_name)}</span><span class='pill'>{len(data)} river reaches</span>
<span class='pill'>Not validated</span></div>""", unsafe_allow_html=True)
dc1, dc2 = st.columns([14, 1])
dc1.markdown(f"<div class='note'>{DISCLAIMER.replace('**', '')}</div>", unsafe_allow_html=True)
with dc2:
    info("logjam_vs_debris", "logjam", "debris_accumulation", label="ⓘ")
if len(avail) < 2:
    st.caption("Only one comparison window has been processed. Run Phase 9 for the other window to enable the switch and the comparison tab.")
TAB_NAMES = ["Overview", "Map", "Map gallery", "Inspection list", "Reach details", "Window comparison", "Glossary", "Method & limits"]
T = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ---------------------------------------------------------------- overview
with T["Overview"]:
    s_all, s_sel = q.summary(data), q.summary(sel)
    st.markdown("#### How to use this app")
    h = st.columns(3)
    h[0].markdown("<div class='step'><b>1 · Look</b><br><span>Open the <b>Map</b> tab. Red and orange lines are the river reaches to check first. "
                  "Switch between the four maps to see the baseline, the change, the susceptibility and the priority.</span></div>", unsafe_allow_html=True)
    h[1].markdown("<div class='step'><b>2 · Decide</b><br><span>The <b>Inspection list</b> gives the visit order. Open <b>Reach details</b> to see "
                  "why a reach is listed and how sure the data is.</span></div>", unsafe_allow_html=True)
    h[2].markdown("<div class='step'><b>3 · Check</b><br><span>Every flag is a hypothesis. Confirm on the ground or with high-resolution "
                  "images. Unsure of a word? Click any ⓘ or ? icon, or open the <b>Glossary</b>.</span></div>", unsafe_allow_html=True)
    st.markdown("")
    c = st.columns(5)
    c[0].metric("River reaches (all)", s_all["n_reaches"], help=tip("reach"))
    c[1].metric("Priority 1", s_all["priority_1"], help=gl.text("priority"))
    c[2].metric("Priority 2", s_all["priority_2"], help=gl.text("priority"))
    c[3].metric("Possible Logjam flags", s_all["n_possible_logjam"], help=gl.text("possible_logjam"))
    c[4].metric("Probable Debris Accumulation", s_all["n_probable_debris"], help=gl.text("probable_debris"))
    a_, b_ = st.columns(2)
    with a_:
        head("Reaches by priority class", "priority", level=4)
        _bars(data["priority_class"].value_counts().reindex(q.PRIORITY_ORDER).fillna(0), {k: _rgb_hex(v) for k, v in PRIORITY_RGB.items()}, q.PRIORITY_ORDER)
    with b_:
        head("Reaches by susceptibility class", "susceptibility", "high_susceptibility", level=4)
        _bars(data["susceptibility_class"].value_counts().reindex(q.SUSCEPTIBILITY_ORDER).fillna(0), {k: _rgb_hex(v) for k, v in SUS_RGB.items()}, q.SUSCEPTIBILITY_ORDER)
    pl = data[data["priority_class"].isin(["Priority 1", "Priority 2"])]
    n_small = int(pl["small_catchment"].sum()) if "small_catchment" in pl else 0
    st.info(f"**Places to check first:** the {len(pl)} Priority 1 + Priority 2 reaches, in the order on the Inspection list "
            f"(highest priority score first). {n_small} of them drain under 0.2 km² and may be dry gullies rather than permanent streams. "
            "Nothing here is a confirmed logjam.")
    if s_all["not_assessed"]:
        st.caption(f"{s_all['not_assessed']} reaches are 'Not assessed': too few clear satellite pixels to measure change. "
                   "That is different from 'no change'.")

# ---------------------------------------------------------------- interactive map
VIEW_BLURB = {"A": "Where the rivers, forest and terrain are, before the event. Thicker blue = bigger stream.",
              "B": "What changed between the two windows (observed only). Rings mark reaches with a satellite debris hypothesis.",
              "C": "Where logjams are more likely to form, from terrain and wood supply. Line thickness = susceptibility class.",
              "D": "Where to look first. Thicker, redder lines = higher priority. Numbers give the visit order of the top reaches.",
              "R": "Just the river reaches, coloured by priority class."}
with T["Map"]:
    view_label = st.radio("Map view", list(VIEWS), horizontal=True, key="map_view",
                          help="The four final maps as live layers. Static PNG versions are in the 'Map gallery' tab.")
    view = VIEWS[view_label]
    st.markdown(f"<div class='status'><b>{view_label}.</b> {VIEW_BLURB[view]}"
                + ("" if view == "A" else f" &nbsp;·&nbsp; Showing <b>{len(sel)}</b> of {len(data)} reaches (sidebar filters).") + "</div>", unsafe_allow_html=True)
    with st.expander("Map display options"):
        oc1, oc2 = st.columns(2)
        opacity = oc1.slider("Raster layer opacity", 0.1, 1.0, 0.85, 0.05, help=tip("opacity"))
        show_all = oc2.checkbox("Draw filtered-out reaches in grey", value=True)
    _map(data, sel, view, tag, opacity=opacity, show_all=show_all)
    st.caption("Hover a reach for its values. Scroll to zoom, drag to pan.")
    terms_box(VIEW_TERMS[view])
    photos = _photos()
    if len(photos):
        st.markdown("#### Field photos (visual checks)")
        pid = st.radio("Photo", list(photos["photo_id"]), horizontal=True, label_visibility="collapsed") if len(photos) > 1 else photos["photo_id"].iloc[0]
        p = photos[photos["photo_id"] == pid].iloc[0]
        pc1, pc2 = st.columns([1, 2])
        f = PHOTO_DIR / str(p["file"])
        if f.exists():
            pc1.image(str(f), use_container_width=True)
        nr = q.nearest_reach(data, float(p["lat"]), float(p["lon"]))
        with pc2:
            st.markdown(f"**{p['photo_id']}{' · ' + p['title'] if isinstance(p['title'], str) and p['title'].strip() else ''}**  \n{p['lat']:.5f}° N, {p['lon']:.5f}° E" + (f"  ·  {p['date']}" if isinstance(p["date"], str) else ""))
            if isinstance(p["what_it_shows"], str) and p["what_it_shows"].strip():
                st.write(p["what_it_shows"])
            if nr:
                r = nr["row"]
                st.markdown(f"**What the satellite screening says here.** Nearest river reach: **{nr['reach_id']}**, {nr['distance_m']:.0f} m from the photo point. "
                            f"Priority class: **{r['priority_class']}**; observed change: **{num(r.get('observed_change_score'))}**"
                            f" ({r.get('observed_change_class') or 'no class'}); satellite debris flag: **{r.get('debris_flag') or 'none'}**.")
                if nr["distance_m"] > 100:
                    st.caption("This reach is more than 100 m away, so it may not be the stream the photo relates to.")
            ov = _overlays(tag)
            if "change" in ov:
                import json as _json
                png = _cfg().path("outputs", f"overlays{tag}") / _json.loads((_cfg().path("outputs", f"overlays{tag}") / "overlays.json").read_text())["change"]["file"]
                cl = q.change_classes_near(png, ov["change"][1], float(p["lat"]), float(p["lon"]), 30.0)
                if cl:
                    top = ", ".join(f"{k.split(' (')[0].lower()} {100 * v:.0f}%" for k, v in sorted(cl.items(), key=lambda kv: -kv[1]))
                    st.markdown(f"**Observed change within 30 m of the photo point** (share of 10 m pixels): {top}. "
                                "Compare this with what the photo shows.")
                else:
                    st.markdown("**Observed change within 30 m of the photo point:** none detected.")

# ---------------------------------------------------------------- static maps
with T["Map gallery"]:
    mdir = _cfg().path("outputs", f"maps{tag}")
    names = [("Map A — Baseline", "baseline_map.png"), ("Map B — Observed change", "change_map_2025_2026.png"),
             ("Map C — Logjam susceptibility", "logjam_susceptibility_map.png"), ("Map D — Priority inspection", "priority_inspection_map.png")]
    choice = st.radio("Map", [n for n, _ in names], horizontal=True)
    f = mdir / dict(names)[choice]
    if f.exists():
        st.image(str(f), use_container_width=True)
        st.download_button("Download PNG", f.read_bytes(), file_name=f.name, mime="image/png")
    else:
        st.warning(f"{f.name} not found — run Phase 9.")
    terms_box(VIEW_TERMS[MAP_TERMS[choice]], "What do the terms on this map mean?")

# ---------------------------------------------------------------- inspection list
with T["Inspection list"]:
    st.caption("Order of visits: Priority 1 and 2 together, highest priority score first, then Priority 3. "
               "The class says why a reach is listed; the score decides the order.")
    rk_show = rk.copy()
    rk_show.insert(0, "visit_order", range(1, len(rk_show) + 1))
    if "upstream_area_m2" in rk_show:
        rk_show["upstream_km2"] = (rk_show["upstream_area_m2"] / 1e6).round(3)
    show = ["visit_order", "reach_id", "priority_class", "priority_score", "susceptibility_class", "susceptibility_score",
            "observed_change_score", "debris_flag", "confidence_class", "confidence_score", "bridge_or_crossing", "stream_order",
            "upstream_km2", "small_catchment", "inspection_labels"]
    show = [c for c in show if c in rk_show.columns]
    cc = st.column_config
    cfgcols = {"visit_order": cc.NumberColumn("Visit order", help="1 = check first. Priority 1 and 2 are ordered together by priority score."),
               "reach_id": cc.TextColumn("Reach", help=gl.short("reach")),
               "priority_class": cc.TextColumn("Priority", help=gl.text("priority")),
               "priority_score": cc.NumberColumn("Priority score", format="%.2f", help=gl.text("priority_score")),
               "susceptibility_class": cc.TextColumn("Susceptibility", help=gl.text("susceptibility")),
               "susceptibility_score": cc.NumberColumn("Susceptibility score", format="%.2f", help=gl.text("susceptibility")),
               "observed_change_score": cc.NumberColumn("Observed change", format="%.2f", help=gl.text("observed_change")),
               "debris_flag": cc.TextColumn("Satellite debris flag", help=gl.text("possible_logjam") + "\n\n" + gl.text("probable_debris")),
               "confidence_class": cc.TextColumn("Data confidence", help=gl.text("confidence")),
               "confidence_score": cc.NumberColumn("Confidence score", format="%.2f", help=gl.text("confidence")),
               "bridge_or_crossing": cc.CheckboxColumn("Road crossing", help=gl.text("crossing")),
               "stream_order": cc.NumberColumn("Stream order", help=gl.text("stream_order")),
               "upstream_km2": cc.NumberColumn("Catchment (km²)", help="Land area draining to this reach."),
               "small_catchment": cc.CheckboxColumn("Small catchment", help=gl.text("small_catchment")),
               "inspection_labels": cc.TextColumn("Labels", help="Plain-language labels given to this reach.")}
    st.dataframe(pd.DataFrame(rk_show[show]), use_container_width=True, hide_index=True, column_config={k: v for k, v in cfgcols.items() if k in show})
    st.download_button("Download selection (CSV)", pd.DataFrame(rk_show.drop(columns="geometry")).to_csv(index=False).encode(),
                       file_name=f"inspection_selection{tag}.csv", mime="text/csv")
    gp = _cfg().path("outputs") / f"river_reaches{tag}.geojson"
    if gp.exists():
        st.download_button("Download all reaches (GeoJSON)", gp.read_bytes(), file_name=gp.name, mime="application/geo+json")

# ---------------------------------------------------------------- site details
with T["Reach details"]:
    pool = rk if len(rk) else q.ranked(data)
    score_txt = lambda r: "no score" if r != r else f"{r:.2f}"
    lookup = dict(zip(pool["reach_id"], zip(pool["priority_class"], pool["priority_score"])))
    rid = st.selectbox("Reach", list(pool["reach_id"]), format_func=lambda i: f"{i} — {lookup[i][0]} (score {score_txt(lookup[i][1])})",
                       help="Ordered by visit order. Change the sidebar filters to widen the list.")
    d = q.reach_detail(data, rid)
    s = d["scalars"]
    left, right = st.columns([3, 2])
    with left:
        head(f"{rid} — {s['priority_class']}", "priority", level=3)
        if s.get("small_catchment"):
            st.warning("Small catchment: this reach drains under 0.2 km², so it may be a dry gully rather than a permanent stream. "
                       + gl.short("small_catchment"))
        label_keys = {"Possible Logjam": "possible_logjam", "Probable Debris Accumulation": "probable_debris",
                      "High Logjam Susceptibility": "high_susceptibility", "Priority Inspection Location": "priority_inspection"}
        for lab in d["labels"]:
            a, b = st.columns([8, 1])
            a.markdown(f"- **{lab}**")
            with b:
                info(label_keys[lab], *(["logjam_vs_debris"] if lab in ("Possible Logjam", "Probable Debris Accumulation") else []))
        c = st.columns(4)
        c[0].metric("Susceptibility", num(s["susceptibility_score"]), s["susceptibility_class"], delta_color="off", help=gl.text("susceptibility"))
        c[1].metric("Observed change", num(s["observed_change_score"]), s.get("observed_change_class") or "", delta_color="off", help=gl.text("observed_change"))
        c[2].metric("Priority score", num(s["priority_score"]), help=gl.text("priority_score"))
        c[3].metric("Data confidence", num(s["confidence_score"]), s["confidence_class"], delta_color="off", help=gl.text("confidence"))
        rows = [("Satellite debris flag", s.get("debris_flag") or "none", "possible_logjam"),
                ("Radar change (share of channel strip)", s.get("sar_change"), "sar_change"),
                ("Optical change (share of channel strip)", s.get("optical_change"), "optical_change"),
                ("Wood delivery score", s.get("wood_delivery_score"), "wood_delivery"),
                ("Upstream disturbed area (m²)", s.get("upstream_disturbed_area"), "upstream_disturbed_area"),
                ("Local channel slope (m/m)", s.get("local_slope"), "local_slope"),
                ("Flow accumulation (cells)", s.get("flow_accumulation"), "flow_accumulation"),
                ("Road crossing / bridge", "yes" if s.get("bridge_or_crossing") else "no", "crossing"),
                ("Stream order", s.get("stream_order"), "stream_order"),
                ("Length (m)", s.get("length_m"), "reach")]
        fmt = lambda b: "—" if b is None else (f"{b:,.0f}" if isinstance(b, float) and abs(b) >= 1000 else f"{b:.3g}" if isinstance(b, float) else str(b))
        tbl = pd.DataFrame([(a, fmt(b), gl.short(k)) for a, b, k in rows], columns=["Attribute", "Value", "What it means"])
        st.table(tbl.set_index("Attribute"))
        st.markdown("**Why this reach is listed**")
        st.write(d["reasons"] or "—")
        if d["drivers"]:
            head("Largest susceptibility contributions", "susceptibility", level=5)
            st.bar_chart(pd.DataFrame(d["drivers"], columns=["driver", "contribution"]).set_index("driver"))
        st.caption(f"Midpoint: {d['lat']:.5f}, {d['lon']:.5f}")
    with right:
        _map(data, data, "D", tag, selected=rid, height=480, zoom=14, center=(d["lat"], d["lon"]))
        st.caption("Black outline = selected reach. 10 m satellite pixels cannot resolve small jams; verify in the field or on high-resolution imagery.")

# ---------------------------------------------------------------- scenario comparison
with T["Window comparison"]:
    head("How much do the two comparison windows agree?", "windows", "jaccard", level=4)
    other = _load(other_tag)
    if other is None:
        st.info("Run Phase 9 for the other comparison window to enable this tab.")
    else:
        a, b = (data, other) if tag == "" else (other, data)
        res12 = q.compare_scenarios(a, b)
        res1 = q.compare_scenarios(a, b, ("Priority 1",))
        c = st.columns(3)
        c[0].metric("Priority 1+2 agreement", f"{res12['jaccard']:.2f}", help=gl.text("jaccard"))
        c[1].metric("Priority 1 only agreement", f"{res1['jaccard']:.2f}", help=gl.text("jaccard") + "\n\nPriority 1 alone is sensitive to the cut-off values, so it agrees less.")
        c[2].metric("Priority 1+2 in both windows", len(res12["both"]))
        st.write("Reaches that are Priority 1 or 2 in **both** windows are the most robust candidates to inspect.")
        st.dataframe(pd.DataFrame(dict(reach_id=res12["both"])), hide_index=True)
        st.caption("Whole-year window only: " + (", ".join(res12["only_a"][:40]) or "none") + "  |  Post-event window only: " + (", ".join(res12["only_b"][:40]) or "none"))

# ---------------------------------------------------------------- glossary
with T["Glossary"]:
    st.markdown("Plain-language meanings of every term used in the maps, tables and this app.")
    qtxt = st.text_input("Search the glossary", "")
    n_shown = 0
    for k in gl.ORDER_FOR_GLOSSARY:
        nm, sh, lg = gl.TERMS[k]
        if qtxt.strip() and qtxt.strip().lower() not in (nm + sh + lg).lower():
            continue
        n_shown += 1
        with st.expander(nm, expanded=k in ("logjam_vs_debris",) and not qtxt):
            st.markdown(sh + (f"\n\n{lg}" if lg else ""))
    if not n_shown:
        st.caption("No matching terms.")

# ---------------------------------------------------------------- method
with T["Method & limits"]:
    ms = ROOT / "docs" / "METHOD_SUMMARY.md"
    st.markdown(ms.read_text() if ms.exists() else "See `docs/ARCHITECTURE.md` and `docs/VALIDATION_STRATEGY.md`.")
    flags = ROOT / "docs" / "VALIDATION_FLAGS.md"
    if flags.exists():
        with st.expander("Validation flags (inferred dates and signals to check)"):
            st.markdown(flags.read_text())
    vs = ROOT / "docs" / "VALIDATION_STRATEGY.md"
    if vs.exists():
        with st.expander("Validation strategy"):
            st.markdown(vs.read_text())

st.markdown("<div class='foot'>Hachijojima debris &amp; logjam screening · proof of concept · satellite-derived, rule-based and not validated. "
            "Sources: Sentinel-1 and Sentinel-2 (Copernicus), Copernicus GLO-30 DEM, ESA WorldCover.</div>", unsafe_allow_html=True)
