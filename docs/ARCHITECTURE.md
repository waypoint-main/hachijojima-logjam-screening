# Architecture, datasets, limitations and assumptions

Status: **Phase 1 implemented and unit-tested on a synthetic test island. Not yet run on real
Hachijojima data** (the build environment had no access to DEM/OSM/GEE hosts).

## 1. Review of the spec / existing code
* The connected project folder was empty → nothing to reuse, nothing rewritten.
* The user delegated the Sentinel-1/2 GEE code to us (project `ee-alexdeclaro`); `src/gee/sentinel1.py` is a built-in implementation.
  A `user_module` hook remains if existing code is preferred later.

## 2. Final architecture (changes vs. the proposed tree, with reasons)
```
config/config.yaml        all tunables, weights flagged [UNCALIBRATED]
src/config.py             load/validate/override (UI passes overrides; no Streamlit import)
src/data/                 dem.py (Copernicus/local), osm.py (waterways, roads)   <- added: acquisition ≠ processing
src/gee/                  sentinel1.py (user-code plug-in), sentinel2.py
src/preprocessing/        terrain.py (done), sar.py, optical.py
src/hydrology/            drainage.py, river_segments.py, crossings.py, reference.py (done)
src/change_detection/     sar_change, optical_change, fusion          (Phase 3)
src/models/               wood_source, wood_delivery, logjam_susceptibility, observed_change, priority
src/mapping/              baseline (Phase 1 version done), change, susceptibility, priority
src/pipeline.py           run_phase(n, cfg) — the only entry point a UI needs
app/                      Streamlit (renamed from streamlit/ so it can't shadow the package)
tests/                    pytest; synthetic fixtures are test-only
```
Dependency rule: `src/` never imports Streamlit, folium or geemap. Earth Engine is imported lazily.

**Unit of analysis:** the river reach (default 100 m, equal-split of each stream link).
**Two independent scores** (susceptibility, observed change) feed a third (priority) — never mixed.

## 3. Datasets
| Need | Preferred | Fallback | Notes |
|---|---|---|---|
| DEM | **GSI DEM5A/5B/10B** (ground surface, `local_file`) | **GEE `COPERNICUS/DEM/GLO30_2024_1`** (default), ALOS AW3D30 (`JAXA/ALOS/AW3D30/V3_2`, band DSM) | GLO-30 is a DSM (canopy included) at 30 m. FABDEM (canopy-removed) is in the GEE community catalog but CC BY-NC-SA → not default |
| Coarse river reference | **GEE `MERIT/Hydro/v1_0_1`** (93 m: upa, wth, hnd) | — | sanity check only; cannot resolve most Hachijojima streams |
| Reference hydrography | MLIT/GSI river centrelines or National Land Numerical Info | OSM waterways | OSM coverage on small islands is patchy |
| Roads / bridges | GSI vector basemap / OSM | OSM | bridge tags often missing |
| SAR | GEE `COPERNICUS/S1_GRD` (IW, VV+VH, dB) | — | one orbit geometry for both periods; not assumed radiometrically slope-flattened |
| Optical | GEE `COPERNICUS/S2_SR_HARMONIZED` + `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` (cs_cdf ≥ 0.60) | — | heavy cloud expected |
| Landslides / disaster imagery | Tokyo Metropolitan Govt / GSI disaster photos, JAXA/Planet/Maxar VHR | — | **validation**, not model input |
| Typhoon dates/tracks | JMA | — | not provided: after-period = all of 2026 to date (`events: []`). A whole-year composite dilutes event-specific signal; add event windows when dates are known |

## 4. Hachijojima-specific technical limitations
1. **Scale vs. resolution.** Streams are short, steep, often <10 m wide; a 100 m reach is ~3 DEM cells at 30 m and ~10 Sentinel pixels long. Channel width/constriction is *not derivable* from 30 m DEM or 10 m S1/S2 — Phase 1 does not estimate it.
2. **DSM bias.** GLO-30 includes canopy; under dense subtropical forest, channel position and pits are unreliable. Prefer GSI ground DEM.
3. **Porous volcanic ground.** Many channels are ephemeral/dry; DEM-derived "streams" ≠ flowing water. SAR/optical water-extent signals will be weak.
4. **SAR geometry.** Steep volcanic cones (Nishiyama ≈ 850 m) cause layover/shadow and strong orbit-dependent backscatter. Baseline/after composites must share orbit; layover/shadow masking is required. C-band saturates in dense forest and is weakly sensitive to windthrow and under-canopy debris.
5. **Optical cloud.** Humid oceanic climate and typhoon season (Aug–Oct) → sparse clear observations; valid-observation counts feed `confidence_score`.
6. **Confounders to check against local knowledge:** bare scoria/volcanic ground, agriculture and greenhouses, pasture, airfield, quarries/forestry operations, coastal surf/sea clutter in SAR.
7. **Temporal.** 2026 is incomplete (today 2026-10-05); typhoon season ongoing. Phenology/seasonal mismatch can mimic change → season-matched windows supported (`periods.season_months`).
8. **Wood sources not seen from space.** Bank erosion, legacy wood and under-canopy debris dominate real LWD supply; the model sees only canopy-scale disturbance.
9. **Network.** No outbound access to DEM/OSM/GEE in the build sandbox → real-data run still pending.

## 5. Assumption log (Phase 1)
| # | Assumption | Where | Consequence |
|---|---|---|---|
| A1 | Sea = DEM ≤ 0.5 m; island mask derived from DEM | `dem.sea_level_m` | coastal/low-lying land may be clipped |
| A2 | Analysis CRS UTM 54N (EPSG:32654) | config | distortion negligible at island scale |
| A3 | Priority-flood + ε fill to condition DEM | `drainage.py` | modifies terrain for routing only |
| A4 | D8 single-flow-direction | `drainage.py` | planar-slope parallel flow artefacts |
| A5 | Stream = contributing area ≥ 0.05 km² **[UNCALIBRATED]** | `hydrology.stream_threshold_area_m2` | controls network density; tune vs. reference |
| A6 | Reach = equal split of link, ≈100 m | `reaches.length_m` | reach length 0.67–1.5×nominal |
| A7 | Local slope from filled DEM drop/length | `river_segments.py` | smoothed by filling; window variant provided |
| A8 | Sinuosity from D8 path is a *relative* indicator only | `river_segments.py` | grid artefacts inflate values |
| A9 | Road crossing = road ∩ stream buffer (15 m); "bridge" only if tagged | `crossings.py` | untagged bridges missed |
| A10 | Reference-hydrography agreement is reported, not enforced | `reference.py` | documentation, not a gate |
All weights in `susceptibility`, `observed_change`, `priority`, `confidence` are expert-judgement placeholders and **not validated**.

## 6. Priority method (decided now, implemented Phase 8)
Interpretable two-stage rule: (i) class matrix on susceptibility class × observed-change class gives
Priority 1/2/3/Baseline, with Priority 1 additionally requiring change *consistent with obstruction/debris*
(positive SAR persistence in the corridor + vegetation/exposed-material signal near the channel);
(ii) continuous `priority_score = w_s·S + w_o·O + w_i·S·O` ranks within class. A pure product S·O is avoided
because it zeroes out strong evidence on one axis (Priority 2 and 3 would vanish). Weights in config.

## 7. Terminology policy
Labels allowed: *Possible Logjam, Probable Debris Accumulation, High Logjam Susceptibility, Priority Inspection Location*.
"Confirmed logjam" is never emitted without validation data (`outputs.allowed_labels`).

## 8. Validation strategy
1. Build a reference set: VHR/aerial/drone/government disaster photos interpreted for debris/logjam presence, landslide polygons, field reports (`validation.reference_points_path`).
2. Evaluate separately: (a) disturbance/landslide maps vs. mapped polygons (precision/recall/IoU), (b) observed-change score at reaches vs. interpreted obstruction (ROC/PR), (c) susceptibility vs. observed accumulations (rank-based, e.g. AUC; reaches spatially blocked CV).
3. Calibrate weights only after (2); report uncertainty; keep a held-out island region/event.
4. Until then every output is labelled *uncalibrated screening*.

## 9. Roadmap
P1 ✔ · P2 ✔ (run on EE, Oct 2026) · P3 ✔ (run on real composites) ·  P4 wood source · P5 delivery · P6 susceptibility · P7 observed change · P8 priority · P9 Maps A–D · P10 Streamlit.


## 10. Phase 3 — observed change (implemented; real-data first pass)
**Method.** SAR: dB-composite difference (same orbit) minus island-wide reference, divided by sampling error from the
per-pixel temporal spread; flag |z|≥3 and ≥2 dB, then 1-px closing + ≥2000 m² patches. Optical: index differences (NDVI, NBR,
NDMI, BSI, MNDWI) minus reference; forest disturbance = vegetation loss within baseline forest; landslide *candidate* =
near-total vegetation removal + strong soil signal on slope ≥20°. Fusion assigns classes (observed only) and a heuristic
confidence index (magnitude, SAR/optical agreement, observation counts). Outputs: `data/processed/change/*.tif`,
`change_patches.gpkg`, `phase3_qa.json`, `outputs/diagnostics/phase3_change_diagnostic.png`.

**First real-data findings (uncalibrated; NOT validated):**
* Change is spatially organised: strongest on the south-east volcano's N-E facing flanks and valley/ridge streaks.
  Steep forest slopes facing 0–135° show mean dNDVI ≈ −0.07 (22–30 % of pixels < −0.10) vs ≈ 0 to +0.03 on 180–270° slopes.
  Sentinel-1 VH shows the same sign (+0.3 dB N-E vs −0.2 dB S-W) and SAR is sun-independent, so a pure illumination artefact
  is unlikely — consistent with wind-exposure damage, but the cause (windthrow, defoliation/salt burn, landslide) is NOT identified.
* Island-wide offsets exist (VH ≈ +0.7 dB; optical median NDVI −0.047, mode −0.027). SAR median ≈ mode, so the SAR offset is
  a genuine whole-island difference (instrument/wetness/seasonal/storm), not an artefact of a damaged subset. For optical the
  median is dragged by damaged pixels, hence **mode is the default reference**. Disturbed forest area is sensitive to this
  choice (≈ 5.9 km² with median vs ≈ 7.9 km² with mode before fusion) — report as a range, not a point value.
* S1 baseline has 17 scenes vs 25 after (unequal noise; handled via per-pixel standard error). S2 has ≈8–15 clear scenes only
  over the two summits (cloud cap).
* Only ~16 % of optical-flagged pixels have SAR support but ~78 % of SAR flags have optical support: C-band is insensitive
  to much canopy loss, so absence of SAR change does NOT mean no change.

**Added assumptions:** A11 SAR sampling error = 1.2533·sqrt(sd_b²/n_b + sd_a²/n_a), floor 0.3 dB; A12 regional reference =
histogram mode; A13 min patch 2000 m²; A14 landslide candidate thresholds (dNDVI ≤ −0.25, dBSI ≥ 0.20, slope ≥ 20°);
A15 confidence weights (config) are heuristic, not probabilities; A16 forest = baseline NDVI ≥ 0.6 (89 % of land — NDVI saturates,
so this is a weak forest mask; replace with a land-cover product in Phase 4).
**Not yet done:** per-scene persistence, river-corridor disturbance / debris classes (need reaches → Phase 7), validation.


## Phase 3 scenario comparison (run 2026-10-06)

| | default (2026 season-matched) | post_event (2025-10-06..2026-02-01 vs 2025-01-01..10-01) |
|---|---|---|
| forest disturbance (class 1) | 7.3 km² (12.7% of NDVI forest) | 9.8 km² (16.8%) |
| landslide candidates | 0.56 km² | 0.66 km² |
| water_extent_change | 0.10 km² | **1.61 km²** — mostly coastal; likely tide/glint/seasonal MNDWI shift, treated as an ARTEFACT, ignored by Phases 4-8 except within river corridors |
| optical flags with SAR support | 16% | 8.5% (only 9 post-event S1 scenes; SE ~0.9 dB) |
| median pixel confidence | 0.76 | 0.64 |

Interpretation: both windows put damage in the same places (NE/E slopes, Mihara-side flanks); the post-event window shows more,
consistent with partial recovery by 2026 or with the season mismatch (see docs/VALIDATION_FLAGS.md VF-1, VF-3). Not separable without validation data.

## Phase 4 — forest definition and woody-debris source areas

Module: `src/analysis/wood_source.py` (+ `src/gee/landcover.py`, `src/mapping/wood_source.py`). Run: `python -m src.pipeline --phase 4 [--scenario post_event]`.

* Forest = ESA WorldCover (2021) tree class AND baseline NDVI >= 0.6 (`wood_model.forest_definition: both`). WorldCover is a 2021 product and cannot see later change.
* Source pixels = Phase 3 classes {forest_disturbance, landslide_candidate} in forest with confidence >= 0.40.
* `source_potential` = optical disturbance score x slope mobility ramp (5-35 deg). UNCALIBRATED 0-1 index; NOT a wood volume.
* Descriptors: aspect-octant exposure table, Euclidean distance to derived streams (100/300 m bands), per-patch source_index_ha.

Assumptions: A17 canopy loss in forest = potential wood source (cannot separate windthrow from salt burn/defoliation);
A18 mobility is linear in slope; A19 Euclidean proximity is a descriptor only — flow-path delivery is Phase 5.


## Phase 4 result (post_event; run 2026-10-06)

Forest: WorldCover tree vs baseline NDVI>=0.6 agree strongly (IoU 0.95), so the choice barely matters; combined forest 61.4 km².
Source area 10.4 km² (16.9% of forest; 7.8 km² / 12.8% for the 2026 window); 397 patches, most < 1 ha. Landslide-candidate sources 0.66 km².
Aspect: 35-38% of N/NE/E-facing forest is a source pixel vs 1.5-3% on S/SW/W. Source share rises with slope (5% at <5 deg to ~43% at 55-60 deg).
**Caveat (VF-2):** the post_event after-window is low-sun season; terrain illumination can darken N/NE-facing slopes in winter and mimic a
north-east-biased NDVI/NBR drop. Evidence for a real effect: in the island time series, Jan-Mar 2025 shows ~0 exposed-minus-sheltered difference while
Jan-Mar 2026 (same sun geometry) shows -0.1. Sentinel-2 SR here is NOT topographically corrected. The season-matched 2026 scenario is the cleaner test.
Proximity: 60% of source area lies within 100 m (Euclidean) of a derived stream and 98% within 300 m — streams are dense relative to hillslope length,
so Euclidean proximity does not discriminate between sources; Phase 5 uses D8 flow-path distance.

## Phase 5 — wood delivery

Module `src/analysis/wood_delivery.py`; run `python -m src.pipeline --phase 5 [--scenario post_event]`.
Method: source potential (10 m) -> effective source area per 30 m cell (m2); D8 route to first stream cell; delivery factor exp(-d/120 m), 0 beyond 300 m;
direct input per stream cell; in-channel supply accumulated downstream with exp(-L/1500 m) attenuation; reach attributes:
direct_wood_delivery_m2, upstream_wood_supply_m2, wood_delivery_score (log-normalised to the P95 of positive reaches; relative, data-dependent),
upstream_disturbed_area_m2 / _pct, upstream_landslide_area_m2, riparian_disturbed_frac (forest within 60 m of stream that is a source).
Assumptions A20-A23 are listed in the module docstring. Mass balance (delivered == sum of direct inputs) is asserted in tests and reported in phase5_qa.json.
Limitation: D8 single-direction routing, no hillslope storage, benches or roads intercepting wood; coefficients are uncalibrated.


## Phase 5 result (run 2026-10-06) and fix

27% of effective source area reaches a stream (31% in the 2026 window); ~19% has no path to any derived stream (coastal slopes/gullies below the
50,000 m2 stream threshold) and a further ~20% lies beyond the 300 m delivery limit. Mass balance exact. Highest delivery: Mihara-side and
north-east Nishiyama catchments. FIX: the P95 normalisation saturated the top 5% of reaches at exactly 1.0; now scaled to the maximum (percentile 100).
Open question: sources with no stream path may reflect the stream threshold rather than real coastal drainage — test with a lower threshold.

## Phase 6 — logjam susceptibility

`src/models/logjam_susceptibility.py`; run `python -m src.pipeline --phase 6 [--scenario post_event]`. Score = weighted mean of 13 drivers on absolute
0-1 ramps (config susceptibility.ramps; all UNCALIBRATED). Includes upstream-vs-local slope break, valley-confinement proxy (30 m DSM, not channel width),
crossings/bridges, confluence, bend, riparian disturbance and a small observed-change term. Reports per-class driver contributions and a weight-perturbation
rank-stability test (300 draws, +-50%). Assumptions A24-A27: absolute ramps; confinement proxy; upstream slope from 3 reaches along the largest-area path;
missing confinement counts as unconfined (0).


## Phase 6 result (run 2026-10-06)

post_event: Low 1041 / Moderate 1142 / High 319 / Very High 3 (score P95 0.57, max 0.83). 2026 window: 1238 / 987 / 277 / 3.
Weight perturbation (+-50%, 300 draws): median Spearman 0.994 (min 0.97); top-10% overlap median 0.90 (min 0.77) — the ranking is insensitive to weights,
mostly because wood_delivery_score, upstream disturbance and riparian disturbance are correlated and dominate the upper classes.
Score vs upstream area Spearman 0.36 (not just a size proxy). Bridge driver active on only 29 reaches (1.2%); no culverts are tagged in OSM.
Notes: (1) "Very High" is almost empty because an additive 13-driver mean rarely exceeds 0.75; breaks were NOT tuned to the data. (2) Upper classes inherit
the optical-change pattern through wood supply, so susceptibility (S) and observed change (O) in Phase 8 are not independent evidence.

## Phase 7 — observed corridor change and debris/obstruction evidence

`src/models/observed_change.py`; run `python -m src.pipeline --phase 7 [--scenario post_event]` (requires Phase 3 re-run to create `sar_state.tif`).
Per reach, within a 50 m corridor: SAR-flag fraction, persistent-SAR fraction, vegetation-loss, exposed-material, water change (coast excluded within 100 m),
spatial concentration (largest patch share). channel_morphology is unavailable at 10 m and excluded (weights renormalised).
Debris/obstruction evidence in the 15 m core zone: **Probable Debris Accumulation** = persistent SAR backscatter increase agreeing with optical disturbance,
compact patch; **Possible Logjam** = compact one-sensor anomaly on a reach with susceptibility >= 0.5. A backscatter DEcrease never counts. Hypotheses only.
Persistence is a composite-based proxy (every post-event scene beyond the baseline median by k std devs, via VH min/max). Classes 8/9 are written to
`change_class_corridor.tif`. Limitations: 10 m pixels cannot resolve narrow streams or small jams; canopy damage, wet soil, roads and surf also change backscatter.


## Phase 7 first-run result and rule fix (run 2026-10-06)

First rule set (post_event): 3 Probable Debris Accumulation + 204 Possible Logjam (176 in the 2026 window, 0 probable). Diagnosis on the real outputs:
flagged reaches are NOT concentrated on the coast (19% within 300 m vs 16% of all reaches); 94% of flags had no persistent-SAR evidence, and their mean riparian
vegetation-loss fraction was 0.46 vs 0.09 for unflagged reaches. The flag was effectively "storm-damaged riparian forest on a susceptible reach" — not obstruction-specific.
Fix: "Possible Logjam" now also requires change concentrated in the 15 m core relative to the 15-50 m ring (core change >= 25% and >= 1.5x the ring fraction).
Prototype on the post_event data: 21 reaches (vs 154 for the old logic at the same thresholds); requiring exposed material or persistent SAR as well leaves 3.
The Probable rule (persistent SAR increase agreeing with optical disturbance in the core) is unchanged: 3 reaches post_event, 0 for the 2026 window.
Observed-change scores: post_event Low 2042 / Moderate 355 / High 90 / Very High 11; Spearman with susceptibility 0.61 (the two share the optical signal).
Still unvalidated: no reference data exist to say whether ANY flagged reach holds a jam.


## Phase 7 re-run (rule fix applied)

post_event: 19 Possible Logjam + 3 Probable Debris Accumulation (areas 400-500 m2 = 4-5 px, headwater orders 1-2, susceptibility 0.47-0.50 — weak evidence);
2026 window: 15 Possible, 0 Probable. 9 reaches are flagged in BOTH windows (R01921, R01941, R01757, R01175, R02201, R00988, R02091, R01063, R01453).
Caveat: 10 m pixels cannot resolve order-1/2 streams; a 4-5 px SAR+optical change beside a headwater stream is as likely a gully, slide scar or road as a jam.

## Phase 8 — priority classes and confidence

`src/models/priority.py`, `src/models/confidence.py`; run `python -m src.pipeline --phase 8 [--scenario post_event]`.
Priority 1: S>=0.5 AND O>=0.5 AND a Probable/Possible flag (obstruction consistency). Priority 2: S>=0.5 AND (O>=0.5 OR flag). Priority 3 (watch):
S>=0.5 AND O>=0.25, or S>=0.35 AND O>=0.5. Baseline otherwise. Ranking inside classes by 0.45 S + 0.35 O + 0.20 S*O. Confidence = weighted S1 availability,
S2 availability, temporal coverage of the window, SAR-optical agreement and change persistence (heuristic, not a probability). Outputs: river_reaches_priority*.gpkg,
data/outputs/tables/priority_inspection_locations*.csv (lon/lat of reach midpoints), phase8_qa*.json with cross-scenario agreement of the Priority 1-2 sets.
Assumptions A28-A30: thresholds are expert judgement; S and O are not independent (shared optical signal); confidence says nothing about weight correctness.

## Phase 9 — Final maps and exports

`python -m src.pipeline --phase 9 [--scenario post_event]` reads the Phase 8 table (`river_reaches_priority{tag}.gpkg`) and the Phase 2–4 rasters and writes:

- `data/outputs/maps{tag}/baseline_map.png` (Map A), `change_map_2025_2026.png` (Map B), `logjam_susceptibility_map.png` (Map C), `priority_inspection_map.png` (Map D)
- `data/outputs/river_reaches{tag}.gpkg` (layer `river_reaches`) and `.geojson` (WGS84), standard schema from `src/export.py`
- `data/processed/phase9_qa{tag}.json`

Design notes: Map B shows observed change only (flags within the coastal exclusion distance of the sea are hidden); inferred hazard appears only on Maps C/D. Map A degrades gracefully to hillshade if S2 bands or crossings are missing. The export never invents values: missing source columns become NaN. All maps carry the "uncalibrated / not confirmed logjams" footer.

## Phase 10 — Streamlit app

`app/streamlit_app.py` (UI only) + `src/query.py` (pure filter/rank/summary/detail/compare helpers). Sidebar: comparison window, priority/susceptibility/confidence filters, score sliders, stream order, debris and crossing toggles, colour-by. Tabs: Overview, Interactive map (pydeck), Final maps (PNG + download), Inspection list (CSV/GeoJSON download), Site details, Scenario comparison, Method & caveats. Docs shown in-app: `METHOD_SUMMARY.md`, `VALIDATION_FLAGS.md`, `VALIDATION_STRATEGY.md`. Tests: `tests/test_phase10.py` (including a headless `AppTest` run).

### Phase 10 addendum — maps as live layers

Phase 9 also writes WGS84 RGBA overlays + `overlays.json` (bounds) to `data/outputs/overlays{tag}/` (`src/mapping/overlays.py`): hillshade, true-colour baseline, forest, change classes (coastal class-5 suppressed as on Map B), wood-source potential. The app's "Map view" selector (A baseline / B change / C susceptibility / D priority / reaches only) stacks the matching overlays and reach styling, with legend, opacity slider, debris-hypothesis markers (B, C) and numbered Priority 1 labels (D). The static PNG maps are unchanged and remain in the "Final maps" tab.

Note (v18): pydeck/deck.gl parses string props as expressions, so `data:` URIs and URLs containing `?` or `:` break BitmapLayer ("Unexpected ':'"). The app therefore copies the overlay PNGs into `app/static/overlays{tag}/` under versioned file names (no query string) and loads them from `app/static/...`; `.streamlit/config.toml` enables Streamlit static serving. Verified with headless Chromium.

## v19 addendum
- New class **Not assessed** (observed change not measurable: too few clear pixels). Priority score is NaN for these reaches; they rank last.
- Ranking: Priority 1 and 2 are ranked together by priority score (`priority.rank_order`), then Priority 3, Baseline, Not assessed. Consequence: the top of the visit order can be Priority 2 reaches with strong change but no debris flag.
- `small_catchment` flag: upstream area < 0.2 km2 (`priority.small_catchment_m2`); possible dry gully. App has a minimum-upstream-catchment slider.
- Sensitivity (real data): crossing weight x0 gives P1+P2 Jaccard about 0.87-0.88, x0.5 about 0.91-0.92; upstream filters of 0.2 / 0.5 km2 remove 16% / 56% of the default P1+P2 list.
- Known limits, documented not changed: susceptibility and observed-change correlation (~0.61), empty Very High class, heuristic "Data confidence".
- Plain-language glossary (`src/glossary.py`, `docs/GLOSSARY.md`) drives the info icons in the app.
- Maps: land-only stretch for Map A (sea transparent), cropped framing, smaller crossing markers. Interactive map: use per-class layers with `line_width_min_pixels == line_width_max_pixels` (`line_width_units="pixels"` renders as blocks in headless WebGL).
- Overlays are served from `app/static/` (needs `enableStaticServing = true`).

## v20 addendum
- Map view selector, raster opacity and the grey-reaches toggle moved from the sidebar into the Interactive map tab (the sidebar keeps only data filters).
- Observed-change palette (`final_maps.CHANGE_COL`, mirrored in `mapping/change.py`) redesigned so every class has its own hue: orange, magenta, dark indigo, yellow, cyan, dark grey, light grey. Minimum pairwise CIELAB distance rose from 26 to 43.

## v21 addendum
- The app now defaults to the post-event window (pivot: early October 2025; before = to 2025-10-01, after = 2025-10-06 onward). The whole-year 2025-vs-2026 window is the alternative. If only one window has been processed, the app uses that one. Pipeline defaults and file names are unchanged (`--scenario post_event` still writes the `_post_event` files).
- UI redesign: theme in `.streamlit/config.toml` (forest green / river blue), hero header with the window and data status, shortened disclaimer, "How to use" cards, palette-matched bar charts, simplified sidebar (window, a "which reaches to show" preset, the rest under "More filters"), tabs renamed (Overview, Map, Inspection list, Reach details, Window comparison, Map gallery, Glossary, Method & limits), styled map legend and a filter status line.

## v22 addendum
- Field photos: `field_photos/photos.csv` + image files. Each photo is drawn on the interactive map (ring + label) and shown under the map with the nearest river reach (`query.nearest_reach`) and the share of observed-change classes within 30 m (`query.change_classes_near`, read from the exported change overlay PNG). Visual checks only: they never change a screening label.
- Tab order: Overview, Map, Map gallery, Inspection list, Reach details, Window comparison, Glossary, Method & limits.
