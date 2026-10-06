import json

import numpy as np
import rasterio

from src.config import load_config
from src.models import confidence as cf
from src.models import priority as pr
from .test_phase5 import build_inputs


def _pc():
    return load_config()["priority"]


def test_priority_matrix_rules():
    pc = _pc()
    S = np.array([0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.2, 0.6, 0.6])
    O = np.array([0.7, 0.7, 0.3, 0.1, 0.7, 0.3, 0.9, 0.0, 0.6])
    F = np.array([True, False, False, True, False, False, False, False, False])
    k = pr.priority_class(S, O, F, pc).tolist()
    assert k[0] == "Priority 1"            # high S, strong O, flagged
    assert k[1] == "Priority 2"            # high S, strong O, no obstruction flag -> not P1 (obstruction consistency)
    assert k[2] == "Priority 3"            # high S, moderate change
    assert k[3] == "Priority 2"            # flagged on a high-S reach even with weak change
    assert k[4] == "Priority 3"            # moderate S, strong change = watch
    assert k[5] == "Baseline" and k[6] == "Baseline" and k[7] == "Baseline"
    pc2 = dict(pc, obstruction_consistency_required=False)
    assert pr.priority_class(S, O, F, pc2)[1] == "Priority 1"


def test_not_assessed_and_visit_order():
    pc = _pc()
    S = np.array([0.8, 0.8, 0.8, 0.6])
    O = np.array([np.nan, 0.6, 0.9, 0.3])
    F = np.array([False, True, False, False])
    k = pr.priority_class(S, O, F, pc)
    assert k[0] == "Not assessed"                      # unmeasured change is never silently 'Baseline'
    sc = pr.priority_score(S, O, pc)
    assert np.isnan(sc[0]) and not np.isnan(sc[1:]).any()
    # P1+P2 are ordered together by score: the P2 with the higher score is visited before the P1 with the lower one
    klass = np.array(["Priority 1", "Priority 2", "Priority 3", "Baseline", "Not assessed", "Priority 2"])
    score = np.array([0.46, 0.72, 0.5, 0.2, np.nan, 0.55])
    order = pr.rank_order(klass, score).tolist()
    assert order == [1, 5, 0, 2, 3, 4]


def test_priority_score_and_label_policy():
    pc = _pc()
    s = pr.priority_score([1.0, 0.0], [1.0, 0.0], pc)
    assert np.isclose(s[0], pc["w_susceptibility"] + pc["w_observed"] + pc["w_interaction"]) and s[1] == 0
    allowed = set(load_config()["outputs"]["allowed_labels"])
    labs = pr.labels([0.9, 0.2], ["Possible Logjam", ""], ["Priority 1", "Baseline"], 0.5)
    for text in labs:
        for part in filter(None, text.split("; ")):
            assert part in allowed
    assert "confirmed" not in " ".join(labs).lower()


def test_confidence_components():
    assert np.isclose(cf.temporal_coverage(9, 32, 15, 20), 0.5 * 0.6 + 0.5)
    w = load_config()["confidence"]["weights"]
    comps = {k: np.array([1.0, 0.0]) for k in w}
    assert np.allclose(cf.combine(comps, w), [1.0, 0.0])
    m = cf.magnitude_persistence([False, True], [0.0, 1.0], [0.0, 1.0])
    assert m.tolist() == [1.0, 1.0]
    assert cf.magnitude_persistence([True], [0.0], [0.0])[0] == 0.0


def test_phase8_end_to_end(tmp_path):
    from src import pipeline
    cfg = build_inputs(tmp_path)
    cd = cfg.path("processed", "change")
    with rasterio.open(cd / "change_class.tif") as d:
        cls, prof = d.read(1), d.profile
    cls[65:80, 60:63] = 3
    with rasterio.open(cd / "change_class.tif", "w", **prof) as d:
        d.write(cls, 1)
    st = np.zeros((2,) + cls.shape, "uint8"); st[0, 65:80, 60:63] = 1; st[1, 65:80, 60:63] = 1
    with rasterio.open(cd / "sar_state.tif", "w", **{**prof, "count": 2}) as d:
        d.write(st)
    gd = cfg.path("interim", "gee")
    fp = {**prof, "dtype": "float32", "nodata": np.nan}
    for nm, band, val in (("s1_baseline", "n_obs", 16.0), ("s1_after", "n_obs", 9.0), ("s2_baseline", "n_clear", 40.0), ("s2_after", "n_clear", 30.0)):
        with rasterio.open(gd / f"{nm}.tif", "w", **fp) as d:
            d.write(np.full((1,) + cls.shape, val, "float32")); d.set_band_description(1, band)
    (cfg.path("processed") / "phase2_meta.json").write_text(json.dumps({"s1": {"n_scenes_after": 9}, "s2": {"n_scenes_after": 32}}))
    pipeline.run_phase5(cfg); pipeline.run_phase6(cfg); pipeline.run_phase7(cfg)
    out = pipeline.run_phase8(cfg)
    r = out["reaches"]
    assert set(r["priority_class"]) <= set(pr.CLASSES)
    assert r["confidence_score"].between(0, 1).all()
    assert (cfg.path("outputs", "tables") / "priority_inspection_locations.csv").exists()
    assert out["diagnostic_png"].exists()
    qa = json.loads((cfg.path("processed") / "phase8_qa.json").read_text())
    assert sum(qa["priority_counts"].values()) == len(r)
