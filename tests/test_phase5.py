import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.analysis import wood_delivery as wd
from src.config import load_config
from src.hydrology import drainage as dr


def _valley(n=40, cell=30.0):
    """V-shaped valley: stream along column n//2 draining toward row n-1 (south)."""
    yy, xx = np.mgrid[:n, :n]
    z = 200 - 1.0 * yy + 3.0 * np.abs(xx - n // 2)
    return z.astype("float32")


def test_routing_distance_decay_and_mass_conservation():
    n, cell = 40, 30.0
    z = _valley(n)
    filled = dr.fill_depressions(z.astype(float), 1e-5)
    recv = dr.d8_receivers(filled, cell)
    acc = dr.flow_accumulation(filled, recv)
    stream = np.zeros((n, n), bool); stream[:, n // 2] = True
    entry, dist = wd.route_to_streams(filled, recv, stream, cell)
    assert dist[10, n // 2] == 0 and entry[10, n // 2] == 10 * n + n // 2
    assert np.isclose(dist[10, n // 2 - 5], 5 * cell)                       # 5 cells straight across the valley
    wm = dict(delivery_max_distance_m=300, distance_decay_scale_m=120)
    assert wd.delivery_factor(np.array([0.0, 120.0, 301.0]), 300, 120).tolist() == [1.0, np.exp(-1), 0.0]
    eff = np.zeros((n, n)); eff[10, 5] = 900.0; eff[10, n // 2 - 3] = 900.0; eff[20, 0] = 900.0   # last is > max distance
    delivered, direct = wd.hillslope_delivery(eff, entry, dist, wm)
    assert np.isclose(direct.sum(), delivered.sum())                          # mass conservation
    assert direct[10, n // 2] > 0 and delivered[20, 0] == 0
    assert np.isclose(delivered[10, n // 2 - 3], 900 * np.exp(-90 / 120))


def test_channel_supply_attenuation_downstream():
    n, cell = 40, 30.0
    z = _valley(n)
    filled = dr.fill_depressions(z.astype(float), 1e-5)
    recv = dr.d8_receivers(filled, cell)
    stream = np.zeros((n, n), bool); stream[:, n // 2] = True
    direct = np.zeros((n, n)); direct[5, n // 2] = 100.0
    sup = wd.channel_supply(filled, recv, stream, direct, cell, 600.0)
    assert np.isclose(sup[5, n // 2], 100.0)
    assert np.isclose(sup[15, n // 2], 100.0 * np.exp(-10 * cell / 600.0))    # 10 cells downstream
    assert sup[4, n // 2] == 0                                                 # nothing upstream of the input


def test_normalise_score_properties():
    s = wd.normalise_score(np.array([0, 1, 10, 100, 1000, np.nan]), 95)
    assert s[0] == 0 and s[-1] == 0 and np.all(np.diff(s[:5]) >= 0) and s.max() == 1.0


def test_weighted_flow_accumulation():
    z = _valley(20); filled = dr.fill_depressions(z.astype(float), 1e-5)
    recv = dr.d8_receivers(filled, 30.0)
    w = np.zeros((20, 20)); w[3, 3] = 7.0
    a = dr.flow_accumulation(filled, recv, w)
    assert a.max() == 7.0 and a[3, 3] == 7.0 and a.sum() > 7.0               # carried downstream


def build_inputs(tmp_path):
    cfg = load_config(overrides={"project": {"data_dir": str(tmp_path / "data")},
                                 "hydrology": {"stream_threshold_area_m2": 30 * 30 * 60}})
    n, cell, crs = 40, 30.0, "EPSG:32654"
    t30 = from_origin(500000, 3650000, cell, cell)
    z = _valley(n); filled = dr.fill_depressions(z.astype(float), 1e-5)
    recv = dr.d8_receivers(filled, cell); acc = dr.flow_accumulation(filled, recv)
    pd_ = cfg.path("processed")

    def w(path, arr, tr, dtype="float32", nodata=np.nan):
        arr = np.asarray(arr)
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                           crs=crs, transform=tr, nodata=nodata) as d:
            d.write(arr.astype(dtype), 1)
    w(pd_ / "dem.tif", z, t30); w(pd_ / "filled_dem.tif", filled, t30); w(pd_ / "flow_accumulation_cells.tif", acc, t30)
    w(pd_ / "hillshade.tif", np.full((n, n), 0.5), t30)
    x = 500000 + (n // 2 + 0.5) * cell
    y0, y1, y2 = 3650000 - 20 * cell, 3650000 - 30 * cell, 3650000 - (n - 0.5) * cell
    line1, line2 = LineString([(x, y0), (x, y1)]), LineString([(x, y1), (x, y2)])
    gpd.GeoDataFrame({"reach_id": ["R00000", "R00001"], "link_id": [0, 0], "reach_idx": [0, 1], "n_in_link": [2, 2], "stream_order": [2, 2],
                      "upstream_area_m2": [1e6, 2e6], "drains_to_sea": [False, True], "channel_slope_m_per_m": [0.10, 0.02],
                      "local_slope_window_m_per_m": [0.10, 0.02], "sinuosity_window": [1.0, 1.2], "confluence": [False, True],
                      "crossing_type": ["", "major"], "road_crossing": [False, True], "bridge": [False, True], "culvert": [False, False]},
                     geometry=[line1, line2], crs=crs).to_file(pd_ / "river_reaches_phase1.gpkg", layer="reaches", driver="GPKG")
    # 10 m source rasters on the same extent: a disturbed block near the stream
    t10 = from_origin(500000, 3650000, 10.0, 10.0); m = n * 3
    pot = np.full((m, m), np.nan, "float32"); pot[60:90, 40:60] = 0.8       # rows 60-90 (10 m) = cells 20-30; cols 40-60 near stream col 60
    forest = np.ones((m, m), "uint8"); cls = np.zeros((m, m), "uint8"); cls[60:90, 40:60] = 1
    wd_dir = cfg.path("processed", "wood_source")
    w(wd_dir / "wood_source_potential.tif", pot, t10); w(wd_dir / "forest_mask.tif", forest, t10, "uint8", 255)
    w(cfg.path("processed", "change") / "change_class.tif", cls, t10, "uint8", 255)
    return cfg


def test_phase5_end_to_end(tmp_path):
    from src import pipeline
    cfg = build_inputs(tmp_path)
    out = pipeline.run_phase5(cfg)
    q = out["qa"]
    assert q["mass_balance_error"] < 1e-6 and q["effective_source_area_m2"] > 0
    assert 0 < q["delivered_fraction"] <= 1
    r = out["reaches"].iloc[0]
    assert r["wood_delivery_score"] > 0 and r["upstream_wood_supply_m2"] > 0
    assert r["upstream_disturbed_area_m2"] > 0 and 0 < r["upstream_disturbed_pct"] <= 100
    assert out["diagnostic_png"].exists()
    assert len(out["reaches"]) == 2
