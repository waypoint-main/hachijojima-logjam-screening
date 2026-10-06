"""Shared TEST FIXTURES ONLY. The synthetic island below is NOT Hachijojima data and must never
be used to produce deliverables; it exists to verify algorithm correctness."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def make_island(n=160, cell=30.0, seed=0, pit=True):
    """Conical island with radial gullies, a coastline, and one artificial pit."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n]
    cx = cy = n / 2
    r = np.hypot(x - cx, y - cy)
    R = n * 0.4
    ang = np.arctan2(y - cy, x - cx)
    z = np.where(r < R, 600 * np.clip(1 - r / R, 0, None) ** 1.2 + 6 * np.cos(10 * ang) * (r / R) * 5 + rng.normal(0, 0.3, (n, n)), -5.0)
    if pit:
        z[cy.__int__() + 20, cx.__int__() + 5] -= 25   # depression to be filled
    z = z.astype("float32")
    return z, from_origin(500000, 3650000, cell, cell)


@pytest.fixture(scope="session")
def island_tif(tmp_path_factory):
    z, tr = make_island()
    p = tmp_path_factory.mktemp("dem") / "island.tif"
    with rasterio.open(p, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1,
                       dtype="float32", crs="EPSG:32654", transform=tr, nodata=-9999) as dst:
        dst.write(z, 1)
    return p
