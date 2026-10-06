import json

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

from src.analysis import wood_source as ws
from src.config import load_config


def _wm():
    return load_config()["wood_model"]


def test_forest_mask_definitions():
    wm = dict(_wm())
    land = np.ones((10, 10), bool)
    ndvi = np.full((10, 10), 0.8, "float32"); ndvi[:, 5:] = 0.3          # NDVI forest = left half
    wc = np.full((10, 10), 30.0, "float32"); wc[2:, :] = 10.0             # WC tree = rows 2..9
    wm["forest_definition"] = "both"
    f, info = ws.forest_mask(wc, ndvi, land, wm)
    assert f.sum() == 8 * 5 and info["definition_used"] == "both"
    wm["forest_definition"] = "worldcover"
    assert ws.forest_mask(wc, ndvi, land, wm)[0].sum() == 80
    f, info = ws.forest_mask(None, ndvi, land, wm)                          # fallback, flagged
    assert f.sum() == 50 and "warning" in info


def test_source_potential_slope_and_confidence_gates():
    wm = _wm()
    n = 6
    cls = np.ones((n, n), "uint8"); conf = np.full((n, n), 0.9, "float32"); conf[0] = 0.1
    opt = np.full((n, n), 0.8, "float32")
    slope = np.zeros((n, n), "float32"); slope[:, 3:] = 40                  # flat vs steep
    forest = np.ones((n, n), bool); forest[:, 5] = False
    src, pot, mob = ws.source_potential(cls, conf, opt, forest, slope, wm)
    assert not src[0].any()                       # low confidence row excluded
    assert not src[:, 5].any()                    # non-forest excluded
    assert np.nanmax(pot[:, :3]) == 0.0           # flat => no mobility
    assert np.isclose(np.nanmax(pot), 0.8)        # steep => full mobility x strength
    assert np.isnan(pot[~src]).all()


def test_exposure_table_and_patches():
    asp = np.array([[0.0, 90.0], [180.0, 270.0]])
    octs = ws.aspect_octant(asp)
    assert list(octs.ravel()) == [0, 2, 4, 6]
    forest = np.ones((2, 2), bool); src = np.array([[True, False], [False, False]])
    t = ws.exposure_table(forest, src, octs)
    assert t["N"]["source_pct_of_forest"] == 100.0 and t["E"]["source_pct_of_forest"] == 0.0
    s = np.zeros((20, 20), bool); s[2:8, 2:8] = True; s[15, 15] = True     # 36 px blob + 1 px speck
    pot = np.where(s, 0.5, np.nan).astype("float32")
    dist = np.full(s.shape, 50.0, "float32")
    lab, rows = ws.label_patches(s, pot, np.full(s.shape, .7, "float32"), np.full(s.shape, 30., "float32"),
                                 np.ones(s.shape, "uint8"), dist, 10.0, 2000)
    assert len(rows) == 1 and rows[0]["area_m2"] == 3600 and (lab > 0).sum() == 36


def test_phase4_end_to_end_without_earth_engine(tmp_path, monkeypatch):
    from src import pipeline
    cfg = load_config(overrides={"project": {"data_dir": str(tmp_path / "data")}})
    n, tr = 60, from_origin(500000, 3650000, 10.0, 10.0)
    crs = "EPSG:32654"
    yy, xx = np.mgrid[:n, :n]
    dem = (100 + 3 * (n - yy)).astype("float32")                             # slopes down toward row n
    prof = dict(driver="GTiff", height=n, width=n, count=1, dtype="float32", crs=crs, transform=tr, nodata=np.nan)

    def w(path, arr, names=None):
        arr = np.atleast_3d(arr).transpose(2, 0, 1) if np.ndim(arr) == 3 else np.asarray(arr)[None]
        with rasterio.open(path, "w", **{**prof, "count": arr.shape[0]}) as d:
            d.write(arr.astype("float32"))
            for i, nm in enumerate(names or [], 1):
                d.set_band_description(i, nm)
    pd_, gd = cfg.path("processed"), cfg.path("interim", "gee")
    w(pd_ / "dem.tif", dem); w(pd_ / "slope_deg.tif", np.full((n, n), 30.0)); w(pd_ / "hillshade.tif", np.full((n, n), .5))
    w(gd / "s2_baseline.tif", np.stack([np.full((n, n), .8), np.full((n, n), 20.)]), ["NDVI", "n_clear"])
    cls = np.zeros((n, n), "uint8"); cls[10:30, 10:30] = 1
    conf = np.where(cls > 0, .8, np.nan)
    cd = cfg.path("processed", "change")
    with rasterio.open(cd / "change_class.tif", "w", **{**prof, "dtype": "uint8", "nodata": 255}) as d:
        d.write(cls[None])
    w(cd / "change_confidence.tif", conf)
    w(cd / "change_scores.tif", np.stack([np.zeros((n, n)), np.full((n, n), .9)]), ["sar_score", "optical_score"])
    gpd.GeoDataFrame({"link_id": [1]}, geometry=[LineString([(500005, 3649500), (500595, 3649500)])], crs=crs) \
        .to_file(pd_ / "river_reaches_phase1.gpkg", layer="stream_links", driver="GPKG")

    import src.gee.landcover as lc
    monkeypatch.setattr(lc, "fetch_worldcover", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no EE in tests")))
    out = pipeline.run_phase4(cfg)
    qa = json.loads((pd_ / "phase4_qa.json").read_text())
    assert qa["forest"]["definition_used"] == "ndvi" and "warning" in qa["forest"]
    assert abs(qa["source_km2"] - 400 * 100 / 1e6) < 1e-9                    # 20x20 px block
    assert qa["n_patches"] == 1 and out["diagnostic_png"].exists()
    assert (pd_ / "wood_source_patches.gpkg").exists()
