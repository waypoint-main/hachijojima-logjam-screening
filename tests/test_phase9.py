import json

import geopandas as gpd
import numpy as np
import rasterio

from .test_phase5 import build_inputs


def test_phase9_end_to_end(tmp_path):
    from src import export, pipeline
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
    # s2_baseline with true-colour bands for Map A
    rng = np.random.default_rng(0)
    with rasterio.open(gd / "s2_baseline.tif", "w", **{**fp, "count": 4}) as d:
        d.write(np.concatenate([rng.uniform(0.02, 0.2, (3,) + cls.shape), np.full((1,) + cls.shape, 40.0)]).astype("float32"))
        for i, b in enumerate(("B4", "B3", "B2", "n_clear"), 1):
            d.set_band_description(i, b)
    (cfg.path("processed") / "phase2_meta.json").write_text(json.dumps({"s1": {"n_scenes_after": 9}, "s2": {"n_scenes_after": 32}}))
    for n in (5, 6, 7, 8):
        getattr(pipeline, f"run_phase{n}")(cfg)
    out = pipeline.run_phase(9, cfg)
    for p in out["maps"].values():
        assert p.exists() and p.stat().st_size > 10_000
    o = cfg.path("outputs")
    g = gpd.read_file(o / "river_reaches.gpkg", layer="river_reaches")
    for c in export.REQUIRED:
        assert c in g.columns
    gj = gpd.read_file(o / "river_reaches.geojson")
    assert len(gj) == len(g) and gj.crs.to_epsg() == 4326
    assert g["susceptibility_score"].between(0, 1).all()
    import json as _j
    from PIL import Image
    idx = _j.loads((o / "overlays" / "overlays.json").read_text())
    assert {"hillshade", "change", "baseline_rgb", "forest", "wood_source"} <= set(idx)
    for name, m in idx.items():
        w_, s_, e_, n_ = m["bounds"]
        assert 130 < w_ < e_ < 145 and 25 < s_ < n_ < 40
        im = Image.open(o / "overlays" / m["file"])
        assert im.mode == "RGBA" and list(im.size) == m["size"]
    assert np.asarray(Image.open(o / "overlays" / "change.png"))[..., 3].max() > 0       # the change patch survived reprojection
