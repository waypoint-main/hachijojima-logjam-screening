from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely.geometry import LineString

from src import query as q



def _df():
    g = gpd.GeoDataFrame({
        "reach_id": ["R1", "R2", "R3", "R4", "R5"],
        "priority_class": ["Priority 1", "Priority 2", "Priority 3", "Baseline", "Not assessed"],
        "priority_score": [0.5, 0.7, 0.4, 0.1, np.nan],
        "upstream_area_m2": [3e5, 1e5, 5e5, 1e6, 4e5],
        "small_catchment": [False, True, False, False, False],
        "susceptibility_class": ["High", "High", "Moderate", "Low", "High"],
        "susceptibility_score": [0.7, 0.55, 0.4, 0.1, 0.6],
        "observed_change_score": [0.6, 0.3, 0.2, 0.0, np.nan],
        "confidence_class": ["High", "Low", "Medium", "High", "Low"],
        "debris_flag": ["Possible Logjam", "", None, "", ""],
        "bridge_or_crossing": [True, False, False, False, False],
        "stream_order": [3, 2, 1, 1, 1],
        "inspection_labels": ["Possible Logjam; High Logjam Susceptibility; Priority Inspection Location", "", "", "", ""],
        "susceptibility_drivers": ["wood_delivery_score (0.21), upstream_slope (0.08)", "", None, "", ""],
        "notes": ["reason", "", "", "", ""],
    }, geometry=[LineString([(500000, 3650000 - 100 * i), (500000, 3650000 - 100 * i - 50)]) for i in range(5)], crs="EPSG:32654")
    return g


def test_filter_rank_summary():
    d = _df()
    assert list(q.filter_reaches(d, priority=["Priority 1", "Priority 2"])["reach_id"]) == ["R1", "R2"]
    assert "R5" in list(q.filter_reaches(d)["reach_id"])                                  # unmeasured change passes when no minimum is set
    assert "R5" not in list(q.filter_reaches(d, min_observed=0.1)["reach_id"])
    assert list(q.filter_reaches(d, min_upstream_km2=0.2)["reach_id"]) == ["R1", "R3", "R4", "R5"]
    assert list(q.filter_reaches(d, debris_only=True)["reach_id"]) == ["R1"]
    assert list(q.filter_reaches(d, crossing_only=True)["reach_id"]) == ["R1"]
    assert list(q.filter_reaches(d, min_susceptibility=0.5, confidence=["Low"])["reach_id"]) == ["R2", "R5"]
    assert list(q.filter_reaches(d, min_stream_order=2)["reach_id"]) == ["R1", "R2"]
    assert list(q.filter_reaches(d, text="r3")["reach_id"]) == ["R3"]
    assert list(q.ranked(d.sample(frac=1, random_state=1))["reach_id"]) == ["R2", "R1", "R3", "R4", "R5"]   # P2 (0.70) before P1 (0.50)
    s = q.summary(d)
    assert s["n_reaches"] == 5 and s["priority_1"] == 1 and s["not_assessed"] == 1 and s["n_small_catchment_in_list"] == 1 and s["n_possible_logjam"] == 1 and s["n_low_confidence"] == 2
    assert s["n_high_susceptibility"] == 3


def test_detail_and_drivers():
    d = _df()
    out = q.reach_detail(d, "R1")
    assert out["drivers"][0] == ("wood_delivery_score", 0.21)
    assert out["labels"] == ["Possible Logjam", "High Logjam Susceptibility", "Priority Inspection Location"]
    assert set(out["labels"]) <= set(q.ALLOWED_LABELS)
    assert len(q.midpoints(d[d["debris_flag"] == "Possible Logjam"])) == 1 and len(q.midpoints(d.iloc[0:0])) == 0
    assert 125 < out["lon"] < 145 and 25 < out["lat"] < 40
    assert q.reach_detail(d, "R3")["drivers"] == []
    with pytest.raises(KeyError):
        q.reach_detail(d, "nope")
    assert "Confirmed Logjam" not in " ".join(q.ALLOWED_LABELS)


def test_compare_and_geojson():
    a, b = _df(), _df()
    b.loc[1, "priority_class"] = "Baseline"
    c = q.compare_scenarios(a, b)
    assert c["both"] == ["R1"] and c["only_a"] == ["R2"] and abs(c["jaccard"] - 0.5) < 1e-9
    gj = q.to_geojson_wgs84(a, ["reach_id", "priority_score"])
    assert len(gj["features"]) == 5 and gj["features"][0]["properties"]["priority_score"] == 0.5


def test_app_runs_headless(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src import pipeline
    from .test_phase5 import build_inputs
    # reuse the phase-9 fixture pipeline by calling it with a fresh dir, then point the app at that data dir
    cfg = build_inputs(tmp_path)
    import json
    import rasterio
    cd = cfg.path("processed", "change")
    with rasterio.open(cd / "change_class.tif") as d:
        cls, prof = d.read(1), d.profile
    cls[65:80, 60:63] = 3
    with rasterio.open(cd / "change_class.tif", "w", **prof) as d:
        d.write(cls, 1)
    st_ = np.zeros((2,) + cls.shape, "uint8"); st_[0, 65:80, 60:63] = 1; st_[1, 65:80, 60:63] = 1
    with rasterio.open(cd / "sar_state.tif", "w", **{**prof, "count": 2}) as d:
        d.write(st_)
    gd = cfg.path("interim", "gee")
    fp = {**prof, "dtype": "float32", "nodata": np.nan}
    for nm, band, val in (("s1_baseline", "n_obs", 16.0), ("s1_after", "n_obs", 9.0), ("s2_baseline", "n_clear", 40.0), ("s2_after", "n_clear", 30.0)):
        with rasterio.open(gd / f"{nm}.tif", "w", **fp) as d:
            d.write(np.full((1,) + cls.shape, val, "float32")); d.set_band_description(1, band)
    (cfg.path("processed") / "phase2_meta.json").write_text(json.dumps({"s1": {"n_scenes_after": 9}, "s2": {"n_scenes_after": 32}}))
    for n in (5, 6, 7, 8, 9):
        pipeline.run_phase(n, cfg)
    cfgfile = tmp_path / "cfg.yaml"
    cfgfile.write_text(yaml.safe_dump(cfg.data))
    monkeypatch.setenv("HACHIJOJIMA_CONFIG", str(cfgfile))
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    assert any("not validated" in m.value for m in at.markdown)
    views = [o for o in at.tabs[1].radio[0].options]
    assert len(views) == 5
    for v in views:                                   # every map view renders without error
        at.tabs[1].radio[0].set_value(v).run()
        assert not at.exception, (v, [e.value for e in at.exception])


def test_glossary_covers_app_and_is_plain():
    import re
    from src import glossary as gl
    app = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text()
    keys = set(re.findall(r'(?:gl\.(?:text|short|long|name)|tip)\("(\w+)"\)', app))
    for call in re.findall(r'\binfo\(([^)]*)\)', app) + re.findall(r'\bhead\(([^)]*)\)', app):
        keys |= {k for k in re.findall(r'"(\w+)"', call) if k not in ("level",)}
    block = app[app.index("VIEW_TERMS = {"):app.index("MAP_TERMS")]
    keys |= set(re.findall(r'"(\w+)"', block)) - {"A", "B", "C", "D", "R"}
    keys = {k for k in keys if k in gl.TERMS or "_" in k or k in ("logjam", "sar", "optical", "priority", "reach", "confidence", "susceptibility", "hillshade", "windows")}
    assert keys and not [k for k in keys if k not in gl.TERMS], [k for k in keys if k not in gl.TERMS]
    for k, (name, short, long_) in gl.TERMS.items():
        assert name and short.endswith(".") and len(short) < 260, k
    both = gl.text("logjam_vs_debris").lower()
    assert "logjam" in both and "debris accumulation" in both and "wood" in both
    assert set(gl.ORDER_FOR_GLOSSARY) <= set(gl.TERMS)
    md = gl.to_markdown()
    for lab in ("Possible Logjam", "Probable Debris Accumulation", "High Logjam Susceptibility", "Priority Inspection Location"):
        assert lab in md
