"""TEMPLATE — copy to `src/gee/user_sentinel1.py` and paste your existing GEE Sentinel-1 code.

Requirements for the function below:
  * returns an ee.ImageCollection
  * bands 'VV' and 'VH' present (state whether linear or dB in UNITS)
  * filtering by `start`/`end` (ISO dates) and `aoi` happens inside
  * keep orbit properties (orbitProperties_pass, relativeOrbitNumber_start) on the images
"""
import ee

UNITS = "dB"   # or "linear"  <-- set to match your code's output


def get_sentinel1_collection(aoi, start, end, orbit_pass=None, relative_orbit=None,
                             instrument_mode="IW", polarizations=("VV", "VH")):
    # >>> PASTE / CALL YOUR EXISTING CODE HERE <<<
    raise NotImplementedError("Paste your existing Sentinel-1 extraction code here.")
