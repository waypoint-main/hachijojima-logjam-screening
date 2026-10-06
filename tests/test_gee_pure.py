"""Offline tests of the pure-Python parts of the GEE layer (no Earth Engine access needed)."""
import io
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from src.gee import download, merit, sentinel1


def test_analysis_bounds_snapped_and_contains_bbox():
    b = download.analysis_bounds([139.72, 33.05, 139.90, 33.20], "EPSG:32654", 30)
    assert all(v % 30 == 0 for v in b)
    assert 15_000 < b[2] - b[0] < 20_000 and 14_000 < b[3] - b[1] < 18_000


def test_tile_grid_covers_without_gaps():
    tiles, W, H = download.tile_grid((0, 0, 1000, 700), 10, 40)
    cover = np.zeros((H, W), int)
    for *_, c0, r0, w, h in tiles:
        cover[r0:r0 + h, c0:c0 + w] += 1
    assert (cover == 1).all() and (W, H) == (100, 70)


def test_download_assembly_with_fake_fetch(tmp_path):
    bounds, scale = (500000, 3650000, 500300, 3650200), 10   # 30 x 20 px
    truth = np.arange(2 * 20 * 30, dtype="float32").reshape(2, 20, 30)

    def fake_fetch(img, tile, crs, sc):
        x0, y0, x1, y1, c0, r0, w, h = tile
        mf = MemoryFile()
        with mf.open(driver="GTiff", height=h, width=w, count=2, dtype="float32", crs=crs,
                     transform=from_origin(x0, y1, sc, sc)) as d:
            d.write(truth[:, r0:r0 + h, c0:c0 + w])
        return mf.read()

    class Img:
        def unmask(self, v): return self

    out = download.download_image(Img(), bounds, "EPSG:32654", scale, tmp_path / "x.tif",
                                  ["a", "b"], tile_px=16, fetch=fake_fetch)
    with rasterio.open(out) as src:
        assert np.array_equal(src.read(), truth)
        assert src.transform.c == 500000 and src.transform.f == 3650200
        assert src.descriptions == ("a", "b")


def test_pick_orbit_prefers_balanced_geometry():
    p, r, info = sentinel1.pick_orbit({"ASCENDING_46": 30, "DESCENDING_153": 60},
                                      {"ASCENDING_46": 25, "DESCENDING_153": 4}, min_obs=5)
    assert (p, r) == ("ASCENDING", 46)
    with pytest.raises(RuntimeError):
        sentinel1.pick_orbit({"A_1": 10}, {"B_2": 10}, 5)      # no shared geometry
    with pytest.raises(RuntimeError):
        sentinel1.pick_orbit({"A_1": 10}, {"A_1": 2}, 5)       # too few after


def test_merit_comparison_on_synthetic(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    tr = from_origin(500000, 3650000, 90, 90)
    upa = np.full((50, 50), np.nan, "float32")
    upa[:, 25] = np.linspace(0.6, 5, 50)                  # a N-S MERIT river at x=502295
    p = tmp_path / "m.tif"
    with rasterio.open(p, "w", driver="GTiff", height=50, width=50, count=5, dtype="float32",
                       crs="EPSG:32654", transform=tr, nodata=np.nan) as d:
        d.write(np.stack([upa] + [np.zeros_like(upa)] * 4))
    line = LineString([(502295, 3650000 - 20), (502295, 3650000 - 4400)])
    streams = gpd.GeoDataFrame(geometry=[line], crs="EPSG:32654")
    reaches = gpd.GeoDataFrame({"upstream_area_m2": [3e6]}, geometry=[LineString([(502295, 3648000), (502295, 3647000)])], crs="EPSG:32654")
    out = merit.compare_with_merit(streams, reaches, p, 0.5, 140)
    assert out["derived_within_tol"] > 0.95 and out["merit_within_tol"] > 0.95
    assert abs(out["median_log10_area_ratio"]) < 0.5


def test_phase2_diagnostic_renders_from_synthetic_rasters(tmp_path):
    """Plumbing test only: random arrays, NOT satellite data."""
    from src.gee import sentinel2
    from src.mapping.diagnostics import plot_phase2_diagnostic
    rng = np.random.default_rng(1)
    paths = {}
    for sensor, bands in (("s1", sentinel1.OUT_BANDS), ("s2", sentinel2.OUT_BANDS)):
        for k in ("baseline", "after"):
            p = tmp_path / f"{sensor}_{k}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=30, width=40, count=len(bands),
                               dtype="float32", crs="EPSG:32654", transform=from_origin(0, 300, 10, 10)) as d:
                d.write(rng.random((len(bands), 30, 40)).astype("float32"))
                for i, n in enumerate(bands, 1):
                    d.set_band_description(i, n)
            paths[f"{sensor}_{k}"] = p
    assert plot_phase2_diagnostic(paths, 3, tmp_path / "d.png", dpi=40).exists()
