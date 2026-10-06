# Hachijojima typhoon-change & logjam-susceptibility PoC

Satellite-based screening for (1) 2025→2026 forest/river-corridor change, (2) likely woody-debris source
areas, (3) logjam-susceptible river reaches, (4) priority inspection locations. **Screening, not detection
of confirmed logjams.** See `docs/ARCHITECTURE.md` for datasets, limitations, assumptions and validation.

## Status
Phase 1 (DEM + river system + baseline) and Phase 2 (Sentinel-1/2 composites via GEE) are implemented.
Offline logic is unit-tested on synthetic fixtures. **The Earth Engine calls have NOT been executed yet** (the
build sandbox cannot reach Earth Engine) — run `scripts/check_gee_access.py` first. Phase 3 (observed change) runs on the real composites. Phases 4–10 are stubs.

## Quick start
```bash
pip install -r requirements.txt
pytest -q                                   # algorithm tests (synthetic fixtures only)
python scripts/check_gee_access.py          # 1st: sign-in (browser), project ee-alexdeclaro, assets, scene counts, S1 orbit
python -m src.pipeline --phase 1            # GEE DEM (GLO30_2024_1) + MERIT Hydro + OSM -> terrain, streams, reaches
python -m src.pipeline --phase 2            # S1/S2 2025 & 2026 composites -> data/interim/gee/*.tif + QA figure
python -m src.pipeline --phase 1            # re-run AFTER phase 2 to fill the baseline map's S1/S2 panels (+ OSM vectors)
python -m src.pipeline --phase 3            # observed change: classes, confidence, patches -> data/processed/change/
python scripts/island_timeseries.py         # WHEN did it change? exposed vs sheltered slopes time series (dates the event)
python -m src.pipeline --phase 1 --dem path/to/gsi_dem.tif   # optional: GSI ground-surface DEM (better under forest)
```
Outputs: `data/processed/` (rasters, `river_reaches_phase1.gpkg`, `phase1_qa.json`) and
`data/outputs/diagnostics/phase1_baseline_diagnostic.png`.

## Sentinel-1
A built-in implementation is used (`gee.sentinel1.mode: builtin`): single orbit geometry for both periods,
border/sea/layover-shadow masking, dB composites. To use your own code instead, set `mode: user_module` and
see `src/gee/user_sentinel1_TEMPLATE.py`.

## Configure
Everything is in `config/config.yaml` (periods, DEM source, thresholds, reach length, weights). Weights marked
`[UNCALIBRATED]` are placeholders, not validated values.
