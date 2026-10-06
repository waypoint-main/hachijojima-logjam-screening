"""Find WHEN the 2026 change happened:  python scripts/island_timeseries.py

Pulls per-scene mean NDVI/NBR (Sentinel-2) and VH/VV (Sentinel-1, the Phase 2 orbit) over forested slopes that
face the likely storm direction (N-E) vs sheltered slopes (S-W). Their DIFFERENCE removes the shared seasonal
cycle; a step in it dates the damage. Writes CSVs to data/processed and a figure to data/outputs/diagnostics.
Takes several minutes (many small Earth Engine requests). No raster downloads.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.timeseries import make_figure  # noqa: E402
from src.config import load_config  # noqa: E402
from src.gee import init, timeseries  # noqa: E402

logging.basicConfig(level=logging.INFO)
cfg = load_config()
ee = init.initialize(cfg)
aoi = ee.Geometry.Rectangle(cfg["study_area"]["bbox"])
start, end = "2025-01-01", str(cfg["periods"]["after_end"])
masks = timeseries.build_masks(ee, cfg, aoi)

out = cfg.path("processed")
meta = json.loads((out / "phase2_meta.json").read_text())
op, ro = meta["s1"]["orbit_pass"], meta["s1"]["relative_orbit"]

s2 = timeseries.s2_series(ee, cfg, aoi, masks, start, end)
s2.to_csv(out / "island_timeseries_s2.csv", index=False)      # saved immediately: no need to redo S2 if S1 fails
s1 = timeseries.s1_series(ee, cfg, aoi, masks, start, end, op, ro)
s1.to_csv(out / "island_timeseries_s1.csv", index=False)

png = cfg.path("outputs", "diagnostics", "island_timeseries.png")
res = make_figure(s2, s1, png)
(out / "island_timeseries_steps.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2)); print("figure:", png)
