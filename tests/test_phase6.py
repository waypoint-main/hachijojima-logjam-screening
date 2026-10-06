import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.config import load_config
from src.models import logjam_susceptibility as ls
from .test_phase5 import build_inputs


def test_ramp_behaviour():
    assert ls.ramp([0, 20, 40, 80], 0, 40).tolist() == [0.0, 0.5, 1.0, 1.0]
    assert np.allclose(ls.ramp([0.05, 0.125, 0.3], 0.05, 0.20, invert=True), [1.0, 0.5, 0.0])
    assert np.isclose(ls.ramp([1e5], 1e4, 1e6, log=True)[0], 0.5)
    assert ls.ramp([np.nan], 0, 1)[0] == 0.0


def test_upstream_channel_slope_walks_upstream():
    # three reaches in a row, flowing south (y decreasing): A -> B -> C
    segs = [LineString([(0, 300), (0, 200)]), LineString([(0, 200), (0, 100)]), LineString([(0, 100), (0, 0)])]
    g = gpd.GeoDataFrame({"upstream_area_m2": [1e5, 2e5, 3e5], "channel_slope_m_per_m": [0.20, 0.10, 0.02]}, geometry=segs)
    s = ls.upstream_channel_slope(g, steps=3)
    assert np.isclose(s[0], 0.20)                       # headwater: own slope (no break)
    assert np.isclose(s[1], 0.20)
    assert np.isclose(s[2], np.mean([0.10, 0.20]))      # C looks upstream through B and A


def test_scoring_range_classes_and_sensitivity():
    sc = load_config()["susceptibility"]
    rng = np.random.default_rng(1)
    d = pd.DataFrame(rng.random((300, len(ls.DRIVERS))), columns=ls.DRIVERS)
    s, k, c = ls.score_reaches(d, sc["weights"], sc["class_breaks"], sc["class_labels"])
    assert s.min() >= 0 and s.max() <= 1 and np.allclose(c.sum(axis=1), s)
    assert set(k) <= set(sc["class_labels"])
    hi = d.copy(); hi[:] = 1.0
    lo = d.copy(); lo[:] = 0.0
    assert np.isclose(ls.score_reaches(hi, sc["weights"], sc["class_breaks"], sc["class_labels"])[0][0], 1.0)
    assert ls.score_reaches(lo, sc["weights"], sc["class_breaks"], sc["class_labels"])[1][0] == "Low"
    sens = ls.weight_sensitivity(d, sc["weights"], n_draws=50, rel=0.5, seed=0)
    assert 0.5 < sens["median_spearman"] <= 1.0 and 0 < sens["median_top_decile_overlap"] <= 1.0
    zero = ls.weight_sensitivity(d, sc["weights"], n_draws=5, rel=0.0, seed=0)
    assert np.isclose(zero["median_spearman"], 1.0)     # no perturbation => identical ranking


def test_class_boundaries():
    assert ls.classify(np.array([0.0, 0.249, 0.25, 0.5, 0.75, 1.0]), [0.25, 0.5, 0.75],
                       ["Low", "Moderate", "High", "Very High"]).tolist() == ["Low", "Low", "Moderate", "High", "Very High", "Very High"]


def test_phase6_end_to_end(tmp_path):
    from src import pipeline
    cfg = build_inputs(tmp_path)
    pipeline.run_phase5(cfg)
    out = pipeline.run_phase6(cfg)
    r = out["reaches"]
    assert {"susceptibility_score", "susceptibility_class", "notes"} <= set(r.columns)
    assert r["susceptibility_score"].between(0, 1).all()
    qa = json.loads((cfg.path("processed") / "phase6_qa.json").read_text())
    assert sum(qa["class_counts"].values()) == 2 and out["diagnostic_png"].exists()
    # the lower reach has a crossing, a bridge, a confluence, lower slope and a bend => scores higher than the upper one
    assert r.loc[1, "susceptibility_score"] > r.loc[0, "susceptibility_score"]
