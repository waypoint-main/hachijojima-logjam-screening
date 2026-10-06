# Validation flags — dates and signals to check later

Items below were inferred **from satellite imagery only**. None is matched to a named storm, a news report, or
ground observations. Treat them as hypotheses to be checked, not as facts.

## VF-1  Candidate disturbance event, early October 2025  [UNVALIDATED]

- **Evidence:** `scripts/island_timeseries.py` → `data/outputs/diagnostics/island_timeseries.png`,
  `data/processed/island_timeseries_steps.json` (run 2026-10-06).
  - Exposed (aspect 0–135°) minus sheltered (180–270°) forest-slope NDVI and NBR are ≈0 through 2025, then drop
    abruptly: NDVI step ≈ −0.083, NBR step ≈ −0.108, step located at the first Sentinel-2 image on 2025-10-05
    (last pre-step image 2025-10-02).
  - The gap persists through 2026 but narrows toward ~0 around day-of-year 210 (late July 2026), then falls again
    in August 2026. Optical coverage is sparse between ~DOY 137 and ~DOY 210 in 2026, so regrowth vs. cloud
    artefact vs. a second event cannot be separated yet.
  - Sentinel-1 VH exposed–sheltered difference is ≈ −6 dB all year (look-angle geometry); only a small step
    (+0.49 dB, ~2025-10-16) is detected. SAR is therefore weak evidence for dating this event.
- **To validate:** match 2025-10-02…05 to the actual storm record (name, track, wind) from JMA / Tokyo
  Metropolitan Government; compare against any damage reports or photos for Hachijojima; check that the inferred
  date lies after the storm's closest approach.
- **Also check:** whether a second event occurred around August 2026 (DOY ≈ 210–220 drop).
- **Used in:** scenario `post_event` in `config/config.yaml` (baseline 2025-01-01..2025-10-01, after
  2025-10-06..2026-02-01). Not used to label any output as a storm-caused change.

## VF-2  Exposed vs. sheltered divergence interpreted as wind damage  [UNVALIDATED]

- Aspect dependence is consistent with wind-driven forest damage but could also be (a) a sensor/processing
  artefact that differs by illumination geometry, (b) salt-spray or other non-structural browning. Needs VHR
  imagery or field/photo evidence.

## VF-3  Season mismatch in the `post_event` scenario

- Baseline (Jan–Sep) and after (Oct–Jan) cover different seasons. The island-wide regional-shift correction
  absorbs a uniform offset, but any spatially varying phenology remains in the change maps.
