import numpy as np
import pytest

from src.config import load_config
from src.data.dem import prepare_dem
from src.hydrology import drainage
from src.preprocessing import terrain
from .conftest import make_island


def test_config_valid_and_temporal_params():
    cfg = load_config()
    p = cfg.periods
    assert p["baseline_end"] < p["after_start"]
    with pytest.raises(ValueError):
        load_config(overrides={"periods": {"baseline_end": "2026-06-01"}})


def test_slope_of_plane():
    y, x = np.mgrid[:50, :50]
    dem = (x * 10.0 * np.tan(np.radians(20))).astype("float64")   # 20 deg, cell=10
    s = terrain.slope_degrees(dem, 10.0)
    assert np.allclose(s[5:-5, 5:-5], 20, atol=1e-6)


def test_curvature_signs():
    y, x = np.mgrid[-25:25, -25:25].astype("float64")
    bowl = 0.01 * (x**2 + y**2)
    prof, plan = terrain.curvature(bowl, 1.0)
    assert plan[25, 30] > 0 and prof[25, 30] > 0          # concave = positive
    prof2, plan2 = terrain.curvature(-bowl, 1.0)
    assert plan2[25, 30] < 0 and prof2[25, 30] < 0


def test_fill_removes_pits_and_flows_to_sea():
    z, tr = make_island()
    dem = np.where(z > 0.5, z, np.nan)
    filled = drainage.fill_depressions(dem, 1e-5)
    assert np.nanmax(filled - dem) > 5                     # the artificial pit was filled
    assert np.all(filled[np.isfinite(dem)] >= dem[np.isfinite(dem)] - 1e-9)
    recv = drainage.d8_receivers(filled, 30.0)
    land = np.isfinite(dem).ravel()
    # every land cell either has a receiver or is a coastal outlet (receiver -1 only at coast)
    no_recv = np.flatnonzero(land & (recv < 0))
    h, w = dem.shape
    for i in no_recv:
        r, c = divmod(i, w)
        nbrs = [(r + a, c + b) for a, b in drainage.D8]
        assert any((not (0 <= rr < h and 0 <= cc < w)) or not np.isfinite(dem[rr, cc]) for rr, cc in nbrs)


def test_accumulation_conservation():
    z, tr = make_island()
    dem = np.where(z > 0.5, z, np.nan)
    filled = drainage.fill_depressions(dem)
    recv = drainage.d8_receivers(filled, 30.0)
    acc = drainage.flow_accumulation(filled, recv)
    outlets = np.flatnonzero(np.isfinite(dem).ravel() & (recv < 0))
    assert acc.ravel()[outlets].sum() == pytest.approx(np.isfinite(dem).sum())   # all water reaches outlets
    assert np.nanmin(acc[np.isfinite(dem)]) >= 1


def test_pipeline_phase1_end_to_end(island_tif, tmp_path):
    from src.pipeline import run_phase1
    cfg = load_config(overrides={"project": {"data_dir": str(tmp_path / "data")},
                                 "dem": {"sea_level_m": 0.5, "target_resolution_m": 30, "min_island_cells": 50},
                                 "hydrology": {"stream_threshold_area_m2": 40000,
                                               "reference_hydrography": {"source": "none"}},
                                 "crossings": {"source": "none"},
                                 "outputs": {"figure_dpi": 60}})
    cfg.data["project"]["data_dir"] = str(tmp_path / "data")
    out = run_phase1(cfg, dem_override=str(island_tif), offline_vectors={},
                     data_note="SYNTHETIC TEST ISLAND — not Hachijojima")
    r = out["reaches"]
    assert len(r) > 20 and r["stream_order"].max() >= 2
    assert out["qa"]["unresolved_inland_pits"] == 0
    # reach lengths within the documented envelope of the equal-split rule
    assert r["length_m"].between(30, 200).mean() > 0.9
    assert (r["channel_slope_m_per_m"] >= -1e-9).all()      # filled DEM never rises downstream
    # upstream area increases downstream within each link
    for _, g in r.groupby("link_id"):
        a = g.sort_values("reach_idx")["upstream_area_m2"].to_numpy()
        assert np.all(np.diff(a) >= -1e-6)
    assert r["confluence"].any()
    assert out["diagnostic_png"].exists()


def test_cache_signature_invalidates_on_change(tmp_path):
    from src.cache import sig_ok, write_sig
    f = tmp_path / "x.tif"; f.write_bytes(b"1")
    assert not sig_ok(f, {"bbox": [1, 2]})            # no signature yet -> stale
    write_sig(f, {"bbox": [1, 2]})
    assert sig_ok(f, {"bbox": [1, 2]})
    assert not sig_ok(f, {"bbox": [1, 3]})            # input changed -> refetch


def test_crossings_filter_classes_and_small_gullies():
    import geopandas as gpd
    from shapely.geometry import LineString
    from src.hydrology import crossings as xc
    streams = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1000)])], crs="EPSG:32654")
    roads = gpd.GeoDataFrame({"highway": ["secondary", "footway", "track"], "bridge": ["yes", None, None],
                              "tunnel": [None] * 3, "culvert": [None] * 3},
                             geometry=[LineString([(-50, 200), (50, 200)]), LineString([(-50, 400), (50, 400)]),
                                       LineString([(-50, 800), (50, 800)])], crs="EPSG:32654")
    cr = xc.find_crossings(roads, streams, 15, ["secondary"], ["track"])
    assert sorted(cr["highway"]) == ["secondary", "track"]                      # footway dropped
    assert cr.set_index("highway").loc["secondary", "bridge"] and not cr.set_index("highway").loc["track", "bridge"]
    reaches = gpd.GeoDataFrame({"upstream_area_m2": [5e5, 5e4, 5e5]},
                               geometry=[LineString([(0, 150), (0, 250)]), LineString([(0, 750), (0, 850)]),
                                         LineString([(0, 380), (0, 420)])], crs="EPSG:32654")
    out = xc.flag_reaches(reaches, cr, 30, min_upstream_area_m2=2e5)
    assert out["crossing_type"].tolist() == ["major", "", ""]                  # tiny gully skipped; footway reach unflagged
