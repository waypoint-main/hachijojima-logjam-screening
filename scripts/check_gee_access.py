"""Run this FIRST on your machine:  python scripts/check_gee_access.py

Checks authentication, project access, asset availability and scene counts for the configured
windows, and prints the auto-selected Sentinel-1 orbit geometry. No data are downloaded.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.gee import init, sentinel1, sentinel2  # noqa: E402

logging.basicConfig(level=logging.INFO)
cfg = load_config()
ee = init.initialize(cfg)
aoi = ee.Geometry.Rectangle(cfg["study_area"]["bbox"])
p = cfg["periods"]
print("project:", cfg["gee"]["project"])
for name, aid in [("DEM", cfg["dem"]["gee_asset"]), ("MERIT", cfg["hydrology"]["reference_hydrography"]["merit"]["asset"])]:
    try:
        n = ee.ImageCollection(aid).filterBounds(aoi).size().getInfo() if name == "DEM" else len(ee.Image(aid).bandNames().getInfo())
        print(f"{name} {aid}: OK ({n})")
    except Exception as e:
        print(f"{name} {aid}: FAILED {e}")
win = {"baseline": (str(p["baseline_start"]), str(p["baseline_end"])), "after": (str(p["after_start"]), str(p["after_end"]))}
for k, (a, b) in win.items():
    s1c = sentinel1.base_collection(ee, cfg, aoi, a, b).size().getInfo()
    s2c = sentinel2.build_collection(ee, cfg, aoi, a, b).size().getInfo()
    print(f"{k} {a}..{b}: S1 IW VV+VH scenes={s1c}  S2 scenes={s2c}")
cb = sentinel1.orbit_inventory(ee, cfg, aoi, *win["baseline"])
ca = sentinel1.orbit_inventory(ee, cfg, aoi, *win["after"])
print("S1 geometries baseline:", cb); print("S1 geometries after   :", ca)
print("selected:", json.dumps(sentinel1.pick_orbit(cb, ca, cfg["gee"]["sentinel1"]["min_obs_per_period"])[:2]))
