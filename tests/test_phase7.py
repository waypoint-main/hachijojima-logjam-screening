import json

import numpy as np
import pytest
import rasterio

from src.config import load_config
from src.models import observed_change as oc
from .test_phase5 import build_inputs


def _oc():
    return load_config()["observed_change"]


def test_largest_patch_and_metrics():
    n, H = 2, 30
    rid = np.zeros((H, H), int); rid[:, :15] = 1; rid[:, 15:] = 2
    mask = np.zeros((H, H), bool); mask[2:6, 2:6] = True; mask[20, 3] = True; mask[10:12, 20:22] = True
    mx, tot = oc.largest_patch_px(rid, mask, n)
    assert mx.tolist() == [16, 4] and tot.tolist() == [17, 4]


def test_score_excludes_morphology_and_handles_sparse_reaches():
    cfg = _oc()
    m = oc.pd.DataFrame(dict(n_valid_corridor_px=[500, 5], sar_change=[0.3, 0.3], sar_persistence=[0.15, 0.15],
                             optical_vegetation_loss=[0.5, 0.5], optical_exposed_material=[0.15, 0.15], water_change=[0.1, 0.1],
                             spatial_concentration=[1.0, 1.0]))
    s, k, used, v = oc.score_observed(m, cfg)
    assert np.isclose(s[0], 1.0) and np.isnan(s[1]) and k[1] == "Insufficient data" and k[0] == "Very High"
    assert np.isclose(used, 1 - cfg["weights"]["channel_morphology"] / sum(cfg["weights"].values()))


def test_debris_rules_probable_vs_possible():
    cfg = _oc()["debris"]
    n, H = 3, 60
    rid = np.zeros((H, H), int); rid[:, :20] = 1; rid[:, 20:40] = 2; rid[:, 40:] = 3
    dist = np.full((H, H), 40.0)
    for c0 in (0, 20, 40):                       # core = middle 4 columns of each reach, ring = the rest
        dist[:, c0 + 8:c0 + 12] = 5.0
    core, ring = dist <= 15, (dist > 15)
    cls = np.zeros((H, H), "uint8"); sar_dir = np.zeros((H, H), "uint8"); pers = np.zeros((H, H), bool)
    cls[5:15, 8:12] = 3; sar_dir[5:15, 8:12] = 1; pers[5:15, 8:12] = True       # reach 1: both sensors agree in the core
    cls[5:25, 28:32] = 3                                                          # reach 2: change ONLY in the core
    cls[:, 40:48] = 1; cls[:, 52:60] = 1; cls[5:15, 48:52] = 1                    # reach 3: diffuse damage in core AND ring
    chg = np.isin(cls, [1, 2, 3, 4]) | (sar_dir > 0)
    S = [0.2, 0.8, 0.8]
    d, probable, _ = oc.debris_evidence(rid, core, ring, chg, cls, sar_dir, pers, n, cfg, 100.0, S)
    assert d["debris_flag"].tolist() == ["Probable Debris Accumulation", "Possible Logjam", ""]
    assert d.loc[1, "channel_contrast_ratio"] > 1.5 > d.loc[2, "channel_contrast_ratio"]
    d2, _, _ = oc.debris_evidence(rid, core, ring, chg, cls, sar_dir, pers, n, cfg, 100.0, [0.2, 0.2, 0.2])   # low susceptibility
    assert d2["debris_flag"].tolist() == ["Probable Debris Accumulation", "", ""]
    # a SAR DEcrease must never count as debris
    dec = np.where(sar_dir > 0, 2, 0).astype("uint8")
    d3, _, _ = oc.debris_evidence(rid, core, ring, np.isin(np.zeros_like(cls), [1]), np.zeros_like(cls), dec, pers, n, cfg, 100.0, [0.9] * 3)
    assert (d3["debris_flag"] == "").all()


def test_sar_persistence_proxy():
    from src.change_detection import sar_change
    H = 40
    rng = np.random.default_rng(0)

    def s1(plant_min=None):
        d = {}
        for p in ("VV", "VH", "VVVH"):
            d[f"{p}_median"] = -12 + rng.normal(0, 0.3, (H, H)); d[f"{p}_stdDev"] = np.full((H, H), 1.0)
        d["n_obs"] = np.full((H, H), 20.0)
        return d
    b, a = s1(), s1()
    sl = (slice(10, 20), slice(10, 20))
    for p in ("VV", "VH"):
        a[f"{p}_median"][sl] += 6.0
    a["VH_min"] = a["VH_median"] - 1.0; a["VH_max"] = a["VH_median"] + 1.0     # every post scene well above baseline
    cfg = load_config()["change"]
    out = sar_change.sar_change(b, a, np.ones((H, H), bool), cfg, 5, "median")
    assert out["sar_persistent"][sl].mean() > 0.9 and out["sar_persistent"].sum() <= out["sar_flag"].sum()
    a["VH_min"] = np.full((H, H), -30.0)                                          # one very dark post scene => not persistent
    assert sar_change.sar_change(b, a, np.ones((H, H), bool), cfg, 5, "median")["sar_persistent"].sum() == 0


def test_phase7_end_to_end(tmp_path):
    from src import pipeline
    cfg = build_inputs(tmp_path)
    pipeline.run_phase5(cfg); pipeline.run_phase6(cfg)
    cd = cfg.path("processed", "change")
    with rasterio.open(cd / "change_class.tif") as d:
        cls, prof = d.read(1), d.profile
    cls[65:80, 60:63] = 3                                              # exposed material along the stream (10 m grid)
    with rasterio.open(cd / "change_class.tif", "w", **prof) as d:
        d.write(cls, 1)
    st = np.zeros((2,) + cls.shape, "uint8"); st[0, 65:80, 60:63] = 1; st[1, 65:80, 60:63] = 1
    with rasterio.open(cd / "sar_state.tif", "w", **{**prof, "count": 2}) as d:
        d.write(st)
    out = pipeline.run_phase7(cfg)
    r = out["reaches"]
    assert "Probable Debris Accumulation" in set(r["debris_flag"])
    assert {"observed_change_score", "observed_change_class", "sar_change", "optical_change"} <= set(r.columns)
    qa = json.loads((cfg.path("processed") / "phase7_qa.json").read_text())
    assert qa["probable_debris_area_m2"] > 0 and out["diagnostic_png"].exists()
    with rasterio.open(cd / "change_class_corridor.tif") as d:
        assert (d.read(1) == 9).any()
    with pytest.raises(FileNotFoundError):
        (cd / "sar_state.tif").unlink(); pipeline.run_phase7(cfg)
