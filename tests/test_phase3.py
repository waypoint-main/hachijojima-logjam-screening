"""Phase 3 tests on SYNTHETIC arrays with planted changes (not satellite data)."""
import numpy as np
import pytest

from src.change_detection import fusion, optical_change, sar_change
from src.change_detection.common import remove_small_patches, robust_shift
from src.config import load_config

H = W = 120


def _s1(rng, plant=None, shift=0.0):
    d = {}
    for p in ("VV", "VH", "VVVH"):
        d[f"{p}_median"] = -12 + rng.normal(0, 0.4, (H, W)) + shift
        d[f"{p}_stdDev"] = np.full((H, W), 1.8)
    d["n_obs"] = np.full((H, W), 20.0)
    if plant is not None:
        for p in ("VV", "VH"):
            d[f"{p}_median"][plant] += 5.0
    return d


def _s2(rng, plant=None, dndvi=0.0, dnbr=0.0, dbsi=0.0, shift=0.0):
    d = {n: np.zeros((H, W)) for n in ("NDVI", "NBR", "NDMI", "BSI", "MNDWI")}
    d["NDVI"] += 0.8; d["NBR"] += 0.6; d["NDMI"] += 0.5; d["BSI"] += -0.1; d["MNDWI"] += -0.5
    for k in d:
        d[k] = d[k] + rng.normal(0, 0.01, (H, W))
    d["n_clear"] = np.full((H, W), 30.0)
    d["NDVI"] += shift
    if plant is not None:
        d["NDVI"][plant] += dndvi; d["NBR"][plant] += dnbr; d["BSI"][plant] += dbsi
    return d


def test_mode_beats_median_when_many_pixels_changed():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.01, 4000), rng.normal(-0.2, 0.05, 6000)])
    v = np.ones_like(x, dtype=bool)
    assert abs(robust_shift(x, v, "mode")) < 0.02
    assert robust_shift(x, v, "median") < -0.03
    assert robust_shift(x, v, "none") == 0


def test_remove_small_patches():
    m = np.zeros((50, 50), bool); m[5:8, 5:8] = True; m[20:30, 20:30] = True
    out = remove_small_patches(m, 20)
    assert not out[6, 6] and out[25, 25]


def test_sar_detects_planted_patch_and_ignores_global_shift():
    rng = np.random.default_rng(1)
    land = np.ones((H, W), bool)
    plant = np.zeros((H, W), bool); plant[40:55, 40:55] = True
    cfg = load_config()["change"]
    b, a = _s1(rng), _s1(rng, plant, shift=0.7)            # +0.7 dB island-wide AND +5 dB patch
    out = sar_change.sar_change(b, a, land, cfg, min_px=20, shift_method="mode")
    assert out["sar_flag"][plant].mean() > 0.9             # patch found
    assert out["sar_flag"][~plant].mean() < 0.005          # island-wide shift not flagged
    assert abs(out["info"]["VH_regional_shift_db"] - 0.7) < 0.2
    assert (out["sar_dir"][plant] == 1).mean() > 0.9


def test_optical_and_fusion_classes():
    rng = np.random.default_rng(2)
    land = np.ones((H, W), bool)
    forest = np.zeros((H, W), bool); forest[20:40, 20:40] = True     # moderate canopy loss, gentle slope
    scar = np.zeros((H, W), bool); scar[70:85, 70:85] = True         # near-total removal, steep
    cfg = load_config()["change"]
    slope = np.full((H, W), 10.0); slope[scar] = 35.0
    b = _s2(rng)
    a = _s2(rng, forest, -0.15, -0.2, 0.05, shift=-0.03)
    a2 = _s2(rng, scar, -0.45, -0.5, 0.35)
    for k in a:
        a[k] = np.where(scar, a2[k], a[k])
    opt = optical_change.optical_change(b, a, land, slope, cfg, 0.6, 20, "mode")
    assert opt["forest_disturbance"][forest].mean() > 0.9
    assert opt["landslide_candidate"][scar].mean() > 0.9
    assert not opt["landslide_candidate"][forest].any()           # moderate loss is NOT a landslide
    assert opt["forest_disturbance"][~(forest | scar)].mean() < 0.005
    s1b, s1a = _s1(rng), _s1(rng, scar)
    sar = sar_change.sar_change(s1b, s1a, land, cfg, 20, "mode")
    cls, conf, info = fusion.fuse(sar, opt, s1b, s1a, b, a, cfg, 20)
    assert (cls[scar] == 3).mean() > 0.9 and (cls[forest] == 1).mean() > 0.9
    assert (cls[:10, :10] == 0).all()
    # SAR-corroborated scar gets higher confidence than optical-only forest disturbance
    assert np.nanmean(conf[scar]) > np.nanmean(conf[forest])
    assert np.isnan(conf[cls == 0]).all()


def test_timeseries_step_detection_and_doy_overlay():
    import pandas as pd
    from src.analysis.timeseries import detect_step, doy_overlay, exposure_difference
    rng = np.random.default_rng(3)
    dates = pd.date_range("2025-01-01", "2026-10-01", freq="6D")
    season = 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365)                  # shared seasonal cycle
    event = pd.Timestamp("2026-08-20")
    drop = np.where(dates >= event, -0.10, 0.0)                                # damage only on exposed slopes
    df = pd.DataFrame({"date": dates, "exposed_NDVI": 0.8 + season + drop + rng.normal(0, 0.01, len(dates)),
                       "sheltered_NDVI": 0.8 + season + rng.normal(0, 0.01, len(dates))})
    diff = exposure_difference(df, "NDVI")
    r = detect_step(diff)
    assert abs((r["date"] - event).days) <= 6 and r["step"] < -0.07 and r["sse_ratio"] < 0.5
    assert set(doy_overlay(diff)) == {2025, 2026}


class _FakeEE:
    class EEException(Exception):
        pass


class _FakeFC:
    def __init__(self, rows, fail_times=0):
        self.rows, self.fail = rows, fail_times

    def getInfo(self):
        if self.fail > 0:
            self.fail -= 1
            raise _FakeEE.EEException("Too many concurrent aggregations.")
        return {"features": [{"properties": r} for r in self.rows]}


class _FakeCol:
    def __init__(self, a, b, kind, fail_times=0):
        self.a, self.b, self.kind, self.fail = a, b, kind, fail_times

    def map(self, fn):
        import pandas as pd
        dates = pd.date_range(self.a, self.b, freq="4D", inclusive="left")
        rows = []
        for d in dates:
            drop = -0.1 if d >= pd.Timestamp("2026-08-20") else 0.0
            var = ("NDVI", "NBR") if self.kind == "s2" else ("VH", "VV")
            r = {"date": d.strftime("%Y-%m-%d")}
            for m, off in (("exposed", drop), ("sheltered", 0.0)):
                for v in var:
                    r[f"{m}_{v}_mean"] = 0.8 + off
                    r[f"{m}_{v}_count"] = 1000
            rows.append(r)
        return _FakeFC(rows, self.fail)


def test_timeseries_pipeline_end_to_end_with_fake_earth_engine(tmp_path, monkeypatch):
    """Exercises s2_series/s1_series -> tidy -> make_figure with a fake `ee` (catches name/column bugs offline)."""
    from src.analysis.timeseries import make_figure
    from src.gee import sentinel1, sentinel2, timeseries
    monkeypatch.setattr(sentinel2, "build_collection", lambda ee, cfg, aoi, a, b, months=None: _FakeCol(a, b, "s2"))
    monkeypatch.setattr(sentinel1, "build_collection", lambda ee, cfg, aoi, a, b, op, ro, months=None: _FakeCol(a, b, "s1"))
    monkeypatch.setattr(timeseries, "_stats_fn", lambda *a, **k: None)
    monkeypatch.setattr("time.sleep", lambda s: None)
    s2 = timeseries.s2_series(_FakeEE, {}, None, {}, "2025-01-01", "2026-10-05")
    s1 = timeseries.s1_series(_FakeEE, {}, None, {}, "2025-01-01", "2026-10-05", "ASCENDING", 39)
    assert {"date", "exposed_NDVI", "sheltered_NBR", "exposed_NDVI_count"} <= set(s2.columns)
    assert {"exposed_VH", "sheltered_VH", "exposed_VH_count"} <= set(s1.columns)
    res = make_figure(s2, s1, tmp_path / "ts.png")
    assert (tmp_path / "ts.png").exists()
    assert res["NDVI"]["date"].startswith("2026-08") and float(res["NDVI"]["step"]) < -0.05


def test_fetch_retries_on_concurrency_error(monkeypatch):
    from src.gee import timeseries
    monkeypatch.setattr("time.sleep", lambda s: None)
    rows = timeseries._fetch(_FakeEE, _FakeFC([{"date": "x"}], fail_times=2))
    assert rows == [{"date": "x"}]
    with pytest.raises(RuntimeError):
        timeseries._fetch(_FakeEE, _FakeFC([], fail_times=99), tries=3)


def test_scenario_windows_and_tag():
    from src.config import load_config, scen_tag
    base = load_config()
    assert scen_tag(base) == ""
    ev = base.for_scenario("post_event")
    assert scen_tag(ev) == "_post_event"
    assert str(ev["periods"]["after_start"]) == "2025-10-06"
    assert str(base["periods"]["after_start"]) == "2026-01-01"      # original untouched
    import pytest
    with pytest.raises(KeyError):
        base.for_scenario("nope")
