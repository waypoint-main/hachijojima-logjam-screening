"""Analytical pipeline. NO Streamlit imports here — a UI only calls these functions.

Phase 1 (implemented): study area + DEM + river system + baseline terrain/hydrology.
Phases 2-9: interfaces exist elsewhere in src/; they raise NotImplementedError until built.

CLI:  python -m src.pipeline --phase 1 [--config config/config.yaml] [--dem path.tif]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

from .cache import sig_ok, write_sig
from .config import scen_tag, Config, load_config
from .data import dem as dem_mod
from .data import osm
from .hydrology import crossings as xing
from .hydrology import drainage, reference, river_segments
from .mapping.baseline import plot_phase1_baseline
from .preprocessing import terrain

log = logging.getLogger("pipeline")


def _write_raster(path, arr, transform, crs, nodata=np.nan):
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32",
                crs=crs, transform=transform, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype("float32"), 1)


def _cached_vector(cfg, name, bbox, fetch):
    """Fetch a vector layer once and reuse it (keyed on bbox) so reruns don't depend on Overpass."""
    p = cfg.path("raw", name)
    sig = dict(bbox=bbox)
    if sig_ok(p, sig):
        return gpd.read_file(p)
    g = fetch()
    if len(g):
        g.to_file(p, driver="GPKG")
        write_sig(p, sig)
    return g


def run_phase1(cfg: Config, dem_override: str | None = None, offline_vectors: dict | None = None,
               data_note: str = "") -> dict:
    """Study area + DEM + river system + baseline.

    `offline_vectors` ({'waterways': gdf, 'roads': gdf}) lets tests/offline runs inject vectors.
    Returns a dict of paths + QA numbers + in-memory arrays (for diagnostics / UI).
    """
    sa, dcfg, hcfg = cfg["study_area"], cfg["dem"], cfg["hydrology"]
    crs = sa["analysis_crs"]
    res = float(dcfg["target_resolution_m"])

    # 1. DEM ---------------------------------------------------------------
    raw = cfg.path("raw", dcfg["raw_filename"])
    src = dem_override or (dcfg["local_path"] if dcfg["source"] == "local_file" else None)
    if src:
        raw = Path(src)
    elif not sig_ok(raw, dem_sig := dict(bbox=sa["bbox"], crs=crs, res=res, src=dcfg["source"],
                                          asset=dcfg.get("gee_asset"), band=dcfg.get("gee_band"))):
        if dcfg["source"] == "gee":
            dem_mod.fetch_gee_dem(cfg, raw)
            write_sig(raw, dem_sig)
        elif dcfg["source"] == "copernicus_aws":
            dem_mod.fetch_copernicus_dem(sa["bbox"], raw)
            write_sig(raw, dem_sig)
        else:
            raise NotImplementedError(f"DEM source '{dcfg['source']}' needs an input file: set dem.local_path")
    dem, transform, rcrs = dem_mod.prepare_dem(
        raw, cfg.path("interim", "dem_utm.tif"), crs, res, dcfg["resampling"],
        dcfg["sea_level_m"], dcfg["min_island_cells"])

    # 2. Terrain -------------------------------------------------------------
    slope = terrain.slope_degrees(dem, res)
    hs = terrain.hillshade(dem, res, cfg["terrain"]["hillshade_azimuth"], cfg["terrain"]["hillshade_altitude"])
    prof_c, plan_c = terrain.curvature(dem, res)

    # 3. Hydrology -----------------------------------------------------------
    filled = drainage.fill_depressions(dem, hcfg["fill_epsilon_m"])
    recv = drainage.d8_receivers(filled, res)
    acc = drainage.flow_accumulation(filled, recv)
    net = drainage.extract_streams(filled, recv, acc, transform, res, hcfg["stream_threshold_area_m2"],
                                   hcfg["min_isolated_stream_length_m"])

    # 4. Reaches ---------------------------------------------------------------
    rc = cfg["reaches"]
    reaches = river_segments.build_reaches(net, filled, acc, slope, transform, res, rcrs,
                                           rc["length_m"], rc["slope_window_m"], rc["curvature_window_m"])
    streams = gpd.GeoDataFrame({"link_id": [l["link_id"] for l in net.links],
                                "strahler": [l["strahler"] for l in net.links],
                                "length_m": [l["length_m"] for l in net.links]},
                               geometry=drainage.links_to_lines(net), crs=rcrs)

    # 5. Reference hydrography + crossings (network-dependent; degrade gracefully) -------
    bbox = sa["bbox"]
    ref_gdf, roads, qa = None, None, {}
    rcfg = hcfg["reference_hydrography"]
    try:
        if offline_vectors is not None:
            ref_gdf, roads = offline_vectors.get("waterways"), offline_vectors.get("roads")
        else:
            if rcfg["source"] == "osm":
                ref_gdf = _cached_vector(cfg, "osm_waterways.gpkg", bbox, lambda: osm.fetch_waterways(bbox))
            elif rcfg["source"] == "local_file" and rcfg["local_path"]:
                ref_gdf = gpd.read_file(rcfg["local_path"])
            ccfg = cfg["crossings"]
            if ccfg["source"] == "osm":
                roads = _cached_vector(cfg, "osm_roads.gpkg", bbox, lambda: osm.fetch_roads(bbox))
            elif ccfg["source"] == "local_file" and ccfg["local_path"]:
                roads = gpd.read_file(ccfg["local_path"])
    except Exception as e:  # network blocked etc. — report, don't fabricate
        log.warning("Vector reference data unavailable (%s). Continuing without them.", e)
        qa["vector_data_error"] = str(e)
    if ref_gdf is not None and len(ref_gdf):
        qa["hydrography_comparison"] = reference.compare_networks(streams, ref_gdf, rcfg["match_tolerance_m"])
        ref_gdf = ref_gdf.to_crs(rcrs)
    merit_xy = None
    mcfg = rcfg.get("merit", {})
    if mcfg.get("enabled") and offline_vectors is None:
        try:
            from .gee import merit
            mp = cfg.path("raw", "merit_hydro_utm.tif")
            msig = dict(bbox=sa["bbox"], crs=crs, asset=mcfg["asset"])
            if not sig_ok(mp, msig):
                merit.fetch_merit(cfg, mp)
                write_sig(mp, msig)
            qa["merit_comparison"] = merit.compare_with_merit(
                streams, reaches, mp, mcfg["river_upa_min_km2"], mcfg["match_tolerance_m"])
            merit_xy = merit.river_cell_xy(mp, mcfg["river_upa_min_km2"])[0]
        except Exception as e:
            log.warning("MERIT Hydro comparison skipped (%s)", e)
            qa["merit_error"] = str(e)
    cx = cfg["crossings"]
    cr = xing.find_crossings(roads, streams, cx["snap_tolerance_m"], cx["major_highways"], cx["minor_highways"]) \
        if roads is not None else gpd.GeoDataFrame(columns=["crossing_id", "bridge", "road_class", "geometry"],
                                                   geometry="geometry", crs=rcrs)
    reaches = xing.flag_reaches(reaches, cr, cx["reach_match_tolerance_m"], cx["min_upstream_area_m2"]) \
        if len(reaches) else reaches

    # 6. Outputs --------------------------------------------------------------
    pdir = lambda n: cfg.path("processed", n)
    for name, arr in dict(dem=dem, filled_dem=np.where(np.isfinite(dem), filled, np.nan), slope_deg=slope,
                          hillshade=hs, curvature_profile=prof_c, curvature_plan=plan_c,
                          flow_accumulation_cells=np.where(np.isfinite(dem), acc, np.nan)).items():
        _write_raster(pdir(f"{name}.tif"), arr, transform, rcrs)
    reaches.to_file(pdir("river_reaches_phase1.gpkg"), layer="reaches", driver="GPKG")
    streams.to_file(pdir("river_reaches_phase1.gpkg"), layer="stream_links", driver="GPKG")
    if len(cr):
        cr.to_file(pdir("river_reaches_phase1.gpkg"), layer="crossings", driver="GPKG")
    extras = {}
    gdir = cfg.path("interim", "gee")
    try:
        from .data.rasters import read_stack
        if (gdir / "s1_baseline.tif").exists():
            b, t, _ = read_stack(gdir / "s1_baseline.tif"); extras["s1_vv"] = (b["VV_median"], t)
        if (gdir / "s2_baseline.tif").exists():
            b, t, _ = read_stack(gdir / "s2_baseline.tif"); extras["s2_ndvi"] = (b["NDVI"], t)
    except Exception as e:
        log.warning("Phase 2 composites not usable for the baseline map yet (%s)", e)
    png = plot_phase1_baseline(dem, hs, slope, acc, res, transform, reaches, cr,
                               cfg.path("outputs", "diagnostics", "phase1_baseline_diagnostic.png"),
                               reference=ref_gdf, merit_xy=merit_xy, extras=extras, dpi=cfg["outputs"]["figure_dpi"], data_note=data_note)

    land = np.isfinite(dem)
    qa.update(
        land_area_km2=float(land.sum() * res * res / 1e6),
        dem_min_max_m=[float(np.nanmin(dem)), float(np.nanmax(dem))],
        pits_filled_cells=int(np.sum(np.isfinite(dem) & (filled - dem > 1e-3))),
        max_fill_depth_m=float(np.nanmax(filled - dem)),
        n_stream_links=len(net.links), n_reaches=int(len(reaches)),
        total_stream_length_km=float(streams.length.sum() / 1e3),
        max_strahler=int(net.strahler.max()) if net.strahler.size else 0,
        n_crossings=int(len(cr)), n_bridges=int(cr["bridge"].sum()) if len(cr) else 0,
        n_reaches_flagged_major_crossing=int((reaches["crossing_type"] == "major").sum()) if "crossing_type" in reaches else 0,
        n_reaches_flagged_minor_crossing=int((reaches["crossing_type"] == "minor").sum()) if "crossing_type" in reaches else 0,
        coastal_outlet_cells=int(drainage.coastal_mask(land).sum()),
        unresolved_inland_pits=int(np.sum(land & ~drainage.coastal_mask(land) & (recv.reshape(dem.shape) < 0))),
    )
    (cfg.path("processed") / "phase1_qa.json").write_text(json.dumps(qa, indent=2, default=float))
    log.info("Phase 1 QA: %s", json.dumps(qa, default=float))
    return dict(qa=qa, diagnostic_png=png, reaches=reaches, streams=streams, crossings=cr,
                dem=dem, filled=filled, slope=slope, acc=acc, transform=transform, crs=rcrs)


def run_phase2(cfg: Config, force: bool = False) -> dict:
    """Download 2025 baseline / 2026 after Sentinel-1 and Sentinel-2 composites from GEE.

    Writes data/interim/gee/{s1,s2}_{baseline,after}.tif on the 10 m UTM grid (snapped to 30 m),
    phase2_meta.json (scene counts, orbit geometry, thresholds) and a QA figure.
    """
    from .gee import download, init, sentinel1, sentinel2
    from .mapping.diagnostics import plot_phase2_diagnostic
    ee = init.initialize(cfg)
    crs, g = cfg["study_area"]["analysis_crs"], cfg["gee"]
    scale = float(g["export_scale_m"])
    bounds = download.analysis_bounds(cfg["study_area"]["bbox"], crs, snap=30.0)
    aoi = ee.Geometry.Rectangle(cfg["study_area"]["bbox"])
    tag = scen_tag(cfg)
    d = cfg.path("interim", f"gee{tag}")
    meta = {"bounds_utm": bounds, "crs": crs, "scale_m": scale, "scenario": cfg.get("project.scenario", "default")}
    paths = {}
    for sensor, mod, bands in (("s1", sentinel1, sentinel1.OUT_BANDS), ("s2", sentinel2, sentinel2.OUT_BANDS)):
        imgs, m = mod.build_windows(ee, cfg, aoi)
        meta[sensor] = m
        for k, img in imgs.items():
            out = d / f"{sensor}_{k}.tif"
            csig = dict(bounds=bounds, scale=scale, crs=crs, window=m["windows"][k],
                        months=m["season_months"], params=g["sentinel1" if sensor == "s1" else "sentinel2"],
                        orbit=[m.get("orbit_pass"), m.get("relative_orbit")], dem=cfg["dem"]["gee_asset"])
            if force or not sig_ok(out, csig):
                download.download_image(img, bounds, crs, scale, out, bands, g["tile_px"])
                write_sig(out, csig)
            paths[f"{sensor}_{k}"] = out
    (cfg.path("processed") / f"phase2_meta{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    png = plot_phase2_diagnostic(paths, cfg["gee"]["sentinel2"]["min_valid_obs"],
                                 cfg.path("outputs", "diagnostics", f"phase2_composites_diagnostic{tag}.png"),
                                 dpi=cfg["outputs"]["figure_dpi"])
    return dict(paths=paths, meta=meta, diagnostic_png=png)


def run_phase3(cfg: Config) -> dict:
    """Observed 2025->2026 change from the Phase 2 composites (SAR + optical + fusion).

    Needs: data/interim/gee/*.tif (Phase 2) and data/processed/{dem,slope_deg,hillshade}.tif (Phase 1).
    Writes data/processed/change/*.tif, change_patches.gpkg, phase3_qa.json and a diagnostic figure.
    """
    import geopandas as gpd
    import rasterio.features as rf
    from scipy import ndimage
    from shapely.geometry import shape as to_shape
    from .change_detection import fusion, optical_change, sar_change
    from .data.rasters import read_stack, to_grid, write_raster
    from .mapping.change import plot_phase3_diagnostic

    tag = scen_tag(cfg)
    gdir = cfg.path("interim", f"gee{tag}")
    s1b, tr, crs = read_stack(gdir / "s1_baseline.tif"); s1a, _, _ = read_stack(gdir / "s1_after.tif")
    s2b, _, _ = read_stack(gdir / "s2_baseline.tif"); s2a, _, _ = read_stack(gdir / "s2_after.tif")
    shape = s1b["VV_median"].shape
    pdir = cfg.path("processed")
    land = np.isfinite(to_grid(pdir / "dem.tif", shape, tr, crs))
    slope10 = to_grid(pdir / "slope_deg.tif", shape, tr, crs)
    hs10 = to_grid(pdir / "hillshade.tif", shape, tr, crs, resampling="bilinear")
    ch = cfg["change"]
    min_px = int(round(ch["min_patch_area_m2"] / (abs(tr.a) * abs(tr.e))))
    rs = ch["regional_shift"]
    sar = sar_change.sar_change(s1b, s1a, land, ch, min_px, rs)
    opt = optical_change.optical_change(s2b, s2a, land, slope10, ch, cfg["wood_model"]["forest_ndvi_min"], min_px, rs)
    cls, conf, finfo = fusion.fuse(sar, opt, s1b, s1a, s2b, s2a, ch, min_px)

    od = cfg.path("processed", f"change{tag}")
    write_raster(od / "change_class.tif", cls, tr, crs, nodata=255, dtype="uint8")
    write_raster(od / "change_confidence.tif", conf, tr, crs)
    write_raster(od / "change_scores.tif", np.stack([sar["sar_score"], opt["optical_score"], sar["zVV"], sar["zVH"],
                 opt["anomNDVI"], opt["anomNBR"], opt["anomBSI"]]), tr, crs,
                 descriptions=["sar_score", "optical_score", "zVV", "zVH", "anomNDVI", "anomNBR", "anomBSI"])

    state = np.stack([np.where(sar["sar_dir"] > 0, 1, np.where(sar["sar_dir"] < 0, 2, 0)),
                      sar["sar_persistent"].astype("uint8")]).astype("uint8")
    state[:, ~sar["valid"]] = 255
    write_raster(od / "sar_state.tif", state, tr, crs, nodata=255, dtype="uint8", descriptions=["sar_direction_1inc_2dec", "sar_persistent"])

    # patch polygons
    rows, geoms = [], []
    for k in range(1, 8):
        lab, n = ndimage.label(cls == k, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum(np.ones_like(lab), lab, idx) * abs(tr.a * tr.e)
        mc = ndimage.mean(np.nan_to_num(conf), lab, idx)
        ms = ndimage.mean(np.nan_to_num(sar["sar_score"]), lab, idx)
        mo = ndimage.mean(np.nan_to_num(opt["optical_score"]), lab, idx)
        msl = ndimage.mean(np.nan_to_num(slope10), lab, idx)
        for geom, v in rf.shapes(lab.astype("int32"), mask=lab > 0, transform=tr, connectivity=8):
            i = int(v) - 1
            geoms.append(to_shape(geom))
            rows.append(dict(class_code=k, class_name=fusion.CLASS_NAMES[k], area_m2=float(area[i]),
                             mean_confidence=float(mc[i]), mean_sar_score=float(ms[i]),
                             mean_optical_score=float(mo[i]), mean_slope_deg=float(msl[i])))
    patches = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)
    patches.insert(0, "patch_id", [f"C{i:05d}" for i in range(len(patches))])
    patches.to_file(od.parent / f"change_patches{tag}.gpkg", layer="change_patches", driver="GPKG")

    qa = dict(sar=sar["info"], optical=opt["info"], fusion=finfo, min_patch_px=min_px,
              n_patches=int(len(patches)), thresholds=ch)
    (od.parent / f"phase3_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    png = plot_phase3_diagnostic(cls, conf, sar, opt, hs10,
                                 cfg.path("outputs", "diagnostics", f"phase3_change_diagnostic{tag}.png"),
                                 dpi=cfg["outputs"]["figure_dpi"],
                                 label=f"{cfg['periods']['baseline_start']}..{cfg['periods']['baseline_end']} → "
                                       f"{cfg['periods']['after_start']}..{cfg['periods']['after_end']}")
    log.info("Phase 3 QA: %s", json.dumps({k: qa[k] for k in ("fusion", "n_patches")}, default=float))
    return dict(qa=qa, diagnostic_png=png, class_raster=od / "change_class.tif", patches=patches)


def run_phase4(cfg: Config, force: bool = False) -> dict:
    """Forest definition (ESA WorldCover + baseline NDVI) and woody-debris SOURCE areas.

    Needs Phase 1 (dem/slope/streams), Phase 2 (s2_baseline) and Phase 3 (change_class, confidence, scores)
    for the chosen scenario. Downloads WorldCover once (data/raw/worldcover_utm.tif).
    Writes data/processed/wood_source{tag}/*.tif, wood_source_patches{tag}.gpkg, phase4_qa{tag}.json, diagnostic PNG.
    """
    import geopandas as gpd
    import rasterio.features as rf
    from shapely.geometry import shape as to_shape
    from .analysis import wood_source as ws
    from .data.rasters import read_stack, to_grid, write_raster
    from .mapping.wood_source import plot_phase4

    tag, wm = scen_tag(cfg), cfg["wood_model"]
    gdir, pdir = cfg.path("interim", f"gee{tag}"), cfg.path("processed")
    s2b, tr, crs = read_stack(gdir / "s2_baseline.tif")
    shape = s2b["NDVI"].shape
    cell = abs(tr.a)
    cls = rasterio.open(pdir / f"change{tag}" / "change_class.tif").read(1)
    conf = rasterio.open(pdir / f"change{tag}" / "change_confidence.tif").read(1)
    scores, _, _ = read_stack(pdir / f"change{tag}" / "change_scores.tif")
    land = np.isfinite(to_grid(pdir / "dem.tif", shape, tr, crs))
    slope = to_grid(pdir / "slope_deg.tif", shape, tr, crs)
    hs = to_grid(pdir / "hillshade.tif", shape, tr, crs, resampling="bilinear")
    # aspect (30 m DEM -> 10 m grid)
    with rasterio.open(pdir / "dem.tif") as d:
        dem30, t30, c30 = d.read(1).astype("float32"), d.transform, d.crs
    asp_path = pdir / "aspect_deg.tif"
    if not asp_path.exists():
        _write_raster(asp_path, terrain.aspect_degrees(dem30, abs(t30.a)), t30, c30)
    aspect = to_grid(asp_path, shape, tr, crs)

    # land cover
    wcp = cfg.path("raw", "worldcover_utm.tif")
    wc = None
    lsig = dict(bbox=cfg["study_area"]["bbox"], crs=cfg["study_area"]["analysis_crs"], asset=wm["landcover"]["asset"])
    try:
        if force or not sig_ok(wcp, lsig):
            from .gee import landcover
            landcover.fetch_worldcover(cfg, wcp)
            write_sig(wcp, lsig)
        wc = read_stack(wcp)[0][wm["landcover"]["band"]]
    except Exception as e:
        log.warning("WorldCover unavailable (%s) -> NDVI-only forest", e)
    forest, finfo = ws.forest_mask(wc, s2b["NDVI"], land, wm)
    ndvi_f = land & (s2b["NDVI"] >= wm["forest_ndvi_min"])
    wc_f = (land & (wc == wm["landcover"]["tree_class"])) if wc is not None else None

    src, pot, mob = ws.source_potential(cls, conf, scores["optical_score"], forest, slope, wm)
    # derived stream network -> 10 m mask -> Euclidean distance
    streams = gpd.read_file(pdir / "river_reaches_phase1.gpkg", layer="stream_links").to_crs(crs)
    smask = rf.rasterize(((g, 1) for g in streams.geometry), out_shape=shape, transform=tr, all_touched=True,
                         dtype="uint8").astype(bool)
    dist = ws.distance_to_streams(smask, cell)
    octant = ws.aspect_octant(aspect)
    expo = ws.exposure_table(forest, src, octant)
    lab, rows = ws.label_patches(src, pot, conf, slope, cls, dist, cell, wm["source_min_patch_area_m2"])

    od = cfg.path("processed", f"wood_source{tag}")
    write_raster(od / "wood_source_potential.tif", pot, tr, crs)
    write_raster(od / "forest_mask.tif", np.where(land, forest, 255).astype("uint8"), tr, crs, nodata=255, dtype="uint8")
    write_raster(od / "dist_to_stream_m.tif", dist, tr, crs)
    geoms = {}
    for geom, v in rf.shapes(lab.astype("int32"), mask=lab > 0, transform=tr, connectivity=8):
        geoms.setdefault(int(v), []).append(to_shape(geom))
    from shapely.ops import unary_union
    recs = [dict(r, geometry=unary_union(geoms[r["label"]])) for r in rows if r["label"] in geoms]
    patches = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs).drop(columns="label") if recs else \
        gpd.GeoDataFrame(columns=["area_m2", "geometry"], geometry="geometry", crs=crs)
    patches.insert(0, "source_id", [f"W{i:05d}" for i in range(len(patches))])
    patches.to_file(pdir / f"wood_source_patches{tag}.gpkg", layer="wood_source_patches", driver="GPKG")

    px_km2 = cell * cell / 1e6
    near = {f"source_km2_within_{int(b)}m": float((src & (dist <= b)).sum() * px_km2) for b in wm["near_stream_m"]}
    qa = dict(scenario=cfg.get("project.scenario", "default"), forest=finfo,
              source_km2=float(src.sum() * px_km2), source_pct_of_forest=float(100 * src.sum() / max(forest.sum(), 1)),
              landslide_source_km2=float((src & (cls == 3)).sum() * px_km2),
              mean_potential=float(np.nanmean(pot)) if src.any() else None,
              n_patches=int(len(patches)), proximity=near, exposure_by_aspect=expo,
              note="source_potential is an UNCALIBRATED 0-1 index, not wood volume.")
    (pdir / f"phase4_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    png = plot_phase4(hs, forest, ndvi_f, wc_f, pot, smask, expo, slope, src, dist, rows,
                      cfg.path("outputs", "diagnostics", f"phase4_wood_source_diagnostic{tag}.png"),
                      scenario=qa["scenario"], dpi=cfg["outputs"]["figure_dpi"])
    log.info("Phase 4 QA: %s", json.dumps({k: qa[k] for k in ("source_km2", "source_pct_of_forest", "n_patches", "proximity")}, default=float))
    return dict(qa=qa, diagnostic_png=png, patches=patches)


def run_phase5(cfg: Config) -> dict:
    """Wood delivery: route Phase 4 source areas down D8 flow paths to streams, carry supply down the channel network,
    and summarise per reach. Needs Phases 1, 3, 4 for the chosen scenario.
    Writes data/processed/wood_delivery{tag}/*.tif, wood_delivery_reaches{tag}.gpkg, phase5_qa{tag}.json, diagnostic PNG.
    """
    import geopandas as gpd
    import rasterio.features as rf
    import rasterio.warp as rw
    from scipy import ndimage
    from .analysis import wood_delivery as wd
    from .data.rasters import write_raster
    from .hydrology import drainage as dr
    from .mapping.wood_delivery import plot_phase5

    tag, wm, hc = scen_tag(cfg), cfg["wood_model"], cfg["hydrology"]
    pdir = cfg.path("processed")
    with rasterio.open(pdir / "filled_dem.tif") as d:
        filled, t30, crs = d.read(1).astype("float32"), d.transform, d.crs
    cell = abs(t30.a)
    with rasterio.open(pdir / "flow_accumulation_cells.tif") as d:
        acc = d.read(1).astype("float64")
    shape = filled.shape
    stream = np.isfinite(acc) & (acc * cell * cell >= hc["stream_threshold_area_m2"])
    recv = dr.d8_receivers(filled, cell)

    # 10 m Phase 3/4 products -> 30 m cell quantities (m2 of each thing per cell)
    wdir = pdir / f"wood_source{tag}"
    def _read(p):
        with rasterio.open(p) as d:
            return d.read(1), d.transform, d.crs
    pot, t10, c10 = _read(wdir / "wood_source_potential.tif")
    forest, _, _ = _read(wdir / "forest_mask.tif")
    cls, _, _ = _read(pdir / f"change{tag}" / "change_class.tif")

    def to30(a):
        out = np.zeros(shape, dtype="float32")
        rw.reproject(a.astype("float32"), out, src_transform=t10, src_crs=c10, dst_transform=t30, dst_crs=crs,
                     resampling=rw.Resampling.average, src_nodata=None, dst_nodata=None)
        return out * cell * cell
    src10 = np.isfinite(pot)
    eff_area = to30(np.nan_to_num(pot))
    dist_area = to30(src10)
    forest_area = to30(forest == 1)
    ls_area = to30(src10 & (cls == 3))

    entry, edist = wd.route_to_streams(filled, recv, stream, cell)
    delivered, direct = wd.hillslope_delivery(eff_area, entry, edist, wm)
    supply = wd.channel_supply(filled, recv, stream, direct, cell, wm["channel_transport_decay_m"])
    up_dist = dr.flow_accumulation(filled, recv, dist_area)
    up_forest = dr.flow_accumulation(filled, recv, forest_area)
    up_ls = dr.flow_accumulation(filled, recv, ls_area)

    reaches = gpd.read_file(pdir / "river_reaches_phase1.gpkg", layer="reaches").to_crs(crs)
    ids = np.arange(1, len(reaches) + 1)
    rid = rf.rasterize(zip(reaches.geometry, ids), out_shape=shape, transform=t30, all_touched=True,
                       dtype="int32", fill=0)
    # nearest reach id for every cell (stream cells missed by rasterisation, and riparian cells)
    _, idx = ndimage.distance_transform_edt(rid == 0, return_indices=True)
    rid_near = rid[idx[0], idx[1]]
    st_lab = np.where(stream, rid_near, 0)
    dstream, _ = wd.nearest_stream_cell(stream)
    band = (dstream * cell <= wm["riparian_band_m"]) & np.isfinite(filled)

    def at_downstream(a):   # max over a reach's stream cells (accumulated quantities grow downstream)
        return ndimage.maximum(np.nan_to_num(a), st_lab, ids)
    reaches["direct_wood_delivery_m2"] = ndimage.sum(direct, st_lab, ids)
    reaches["upstream_wood_supply_m2"] = at_downstream(supply)
    reaches["wood_delivery_score"] = wd.normalise_score(reaches["upstream_wood_supply_m2"].to_numpy(),
                                                        wm["delivery_normalization_percentile"])
    reaches["upstream_disturbed_area_m2"] = at_downstream(up_dist)
    reaches["upstream_forest_area_m2"] = at_downstream(up_forest)
    reaches["upstream_disturbed_pct"] = 100 * reaches["upstream_disturbed_area_m2"] / reaches["upstream_forest_area_m2"].clip(lower=1)
    reaches["upstream_landslide_area_m2"] = at_downstream(up_ls)
    rf_area = ndimage.sum(np.where(band, forest_area, 0), np.where(band, rid_near, 0), ids)
    rs_area = ndimage.sum(np.where(band, dist_area, 0), np.where(band, rid_near, 0), ids)
    reaches["riparian_disturbed_frac"] = np.where(rf_area > 0, rs_area / np.maximum(rf_area, 1), 0.0)

    od = cfg.path("processed", f"wood_delivery{tag}")
    write_raster(od / "entry_distance_m.tif", np.where(np.isfinite(edist), edist, np.nan), t30, crs)
    write_raster(od / "delivered_to_stream_m2.tif", direct, t30, crs)
    write_raster(od / "channel_wood_supply_m2.tif", np.where(stream, supply, np.nan), t30, crs)
    reaches.to_file(pdir / f"wood_delivery_reaches{tag}.gpkg", layer="reaches", driver="GPKG")

    tot, dlv = float(eff_area.sum()), float(delivered.sum())
    reach_out = float(reaches.loc[reaches["drains_to_sea"], "upstream_wood_supply_m2"].sum()) if "drains_to_sea" in reaches else None
    qa = dict(scenario=cfg.get("project.scenario", "default"),
              effective_source_area_m2=tot, delivered_to_streams_m2=dlv,
              delivered_fraction=dlv / tot if tot else None,
              source_with_no_stream_path_pct=float(100 * eff_area[~np.isfinite(edist)].sum() / tot) if tot else None,
              source_beyond_max_distance_pct=float(100 * eff_area[np.isfinite(edist) & (edist > wm["delivery_max_distance_m"])].sum() / tot) if tot else None,
              mass_balance_error=float(abs(direct.sum() - dlv)),
              n_reaches=int(len(reaches)), n_reaches_with_supply=int((reaches["upstream_wood_supply_m2"] > 0).sum()),
              supply_at_sea_outlets_m2=reach_out,
              top_reaches=reaches.nlargest(10, "wood_delivery_score")[["reach_id", "wood_delivery_score", "upstream_wood_supply_m2", "stream_order"]].to_dict("records"),
              note="Index units (effective source m2). UNCALIBRATED relative score; not wood volume.")
    (pdir / f"phase5_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    with rasterio.open(pdir / "hillshade.tif") as d:
        hs = d.read(1)
    png = plot_phase5(hs, reaches, t30, edist, eff_area, wm,
                      cfg.path("outputs", "diagnostics", f"phase5_wood_delivery_diagnostic{tag}.png"),
                      scenario=qa["scenario"], dpi=cfg["outputs"]["figure_dpi"])
    log.info("Phase 5 QA: %s", json.dumps({k: qa[k] for k in ("delivered_fraction", "source_with_no_stream_path_pct", "n_reaches_with_supply")}, default=float))
    return dict(qa=qa, diagnostic_png=png, reaches=reaches)


def run_phase6(cfg: Config) -> dict:
    """Reach-level logjam susceptibility from Phase 1 (network/crossings) + Phase 5 (wood supply) + terrain.

    Writes data/processed/river_reaches_susceptibility{tag}.gpkg, susceptibility_drivers{tag}.csv, phase6_qa{tag}.json, diagnostic PNG.
    """
    import geopandas as gpd
    import pandas as pd
    from .analysis import reach_context as rcx
    from .mapping.susceptibility import plot_phase6
    from .models import logjam_susceptibility as ls

    tag, sc, wm, hc = scen_tag(cfg), cfg["susceptibility"], cfg["wood_model"], cfg["hydrology"]
    pdir = cfg.path("processed")
    reaches = gpd.read_file(pdir / f"wood_delivery_reaches{tag}.gpkg", layer="reaches")
    with rasterio.open(pdir / "filled_dem.tif") as d:
        filled, t30, crs = d.read(1).astype("float32"), d.transform, d.crs
    with rasterio.open(pdir / "flow_accumulation_cells.tif") as d:
        acc = d.read(1).astype("float64")
    cell = abs(t30.a)
    stream = np.isfinite(acc) & (acc * cell * cell >= hc["stream_threshold_area_m2"])
    ids, rid_near, dcells, nearest = rcx.reach_id_rasters(reaches, filled.shape, t30, stream)
    conf_raw = rcx.valley_confinement(filled, rid_near, dcells, nearest, ids, cell)
    # Phase 3 observed change in the corridor (small-weight driver; observed change is scored properly in Phase 7)
    with rasterio.open(pdir / f"change{tag}" / "change_class.tif") as d:
        cls, t10, c10 = d.read(1), d.transform, d.crs
    import rasterio.warp as rw
    chg = np.zeros(filled.shape, dtype="float32")
    rw.reproject(np.isin(cls, [1, 2, 3, 4, 6]).astype("float32"), chg, src_transform=t10, src_crs=c10,
                 dst_transform=t30, dst_crs=crs, resampling=rw.Resampling.average)
    corr = rcx.corridor_mean(np.where(np.isfinite(filled), chg, np.nan), rid_near, dcells, ids, cell, wm["riparian_band_m"])
    extras = dict(upstream_channel_slope=ls.upstream_channel_slope(reaches, sc["upstream_slope_steps"]),
                  channel_constriction_raw=conf_raw, corridor_change_frac=corr)
    drivers = ls.build_drivers(reaches, extras, sc)
    score, klass, contrib = ls.score_reaches(drivers, sc["weights"], sc["class_breaks"], sc["class_labels"])
    se = sc["sensitivity"]
    sens = ls.weight_sensitivity(drivers, sc["weights"], n_draws=se["n_draws"], rel=se["relative_perturbation"], seed=se["seed"])
    reaches["susceptibility_score"] = score
    reaches["susceptibility_class"] = klass
    reaches["upstream_channel_slope"] = extras["upstream_channel_slope"]
    reaches["channel_constriction_raw"] = conf_raw
    reaches["corridor_change_frac"] = corr
    reaches["notes"] = ls.top_drivers(contrib)
    reaches.to_file(pdir / f"river_reaches_susceptibility{tag}.gpkg", layer="reaches", driver="GPKG")
    pd.concat([reaches[["reach_id"]], drivers.add_prefix("v_"), contrib.add_prefix("c_")], axis=1).to_csv(
        pdir / f"susceptibility_drivers{tag}.csv", index=False)
    from scipy.stats import spearmanr
    qa = dict(scenario=cfg.get("project.scenario", "default"), n_reaches=int(len(reaches)),
              class_counts=reaches["susceptibility_class"].value_counts().reindex(sc["class_labels"]).fillna(0).astype(int).to_dict(),
              score_quantiles=np.percentile(score, [5, 25, 50, 75, 95, 100]).round(3).tolist(),
              driver_mean=drivers.mean().round(3).to_dict(),
              driver_share_of_nonzero=(drivers > 0).mean().round(3).to_dict(),
              spearman_score_vs_upstream_area=float(spearmanr(score, reaches["upstream_area_m2"]).correlation),
              weight_sensitivity=sens, nan_confinement_reaches=int(np.isnan(conf_raw).sum()),
              note="Rule-based screening index with UNCALIBRATED weights; not validated against observed logjams.")
    (pdir / f"phase6_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    with rasterio.open(pdir / "hillshade.tif") as d:
        hs = d.read(1)
    png = plot_phase6(hs, reaches, contrib, t30, sc["class_breaks"], sc["class_labels"], sens,
                      cfg.path("outputs", "diagnostics", f"phase6_susceptibility_diagnostic{tag}.png"),
                      scenario=qa["scenario"], dpi=cfg["outputs"]["figure_dpi"])
    log.info("Phase 6 QA: %s", json.dumps({k: qa[k] for k in ("class_counts", "weight_sensitivity")}, default=float))
    return dict(qa=qa, diagnostic_png=png, reaches=reaches, drivers=drivers)


def run_phase7(cfg: Config) -> dict:
    """Observed change in river corridors + debris/obstruction evidence per reach. Needs Phases 1, 3 (rerun for sar_state.tif), 6.

    Writes data/processed/river_reaches_observed{tag}.gpkg, change{tag}/change_class_corridor.tif (adds classes 8 and 9),
    phase7_qa{tag}.json and a diagnostic PNG.
    """
    import geopandas as gpd
    from scipy import ndimage
    from scipy.stats import spearmanr
    from .analysis import reach_context as rcx
    from .data.rasters import to_grid, write_raster
    from .mapping.observed import plot_phase7
    from .models import observed_change as oc_mod

    tag, oc = scen_tag(cfg), cfg["observed_change"]
    pdir = cfg.path("processed")
    cdir = pdir / f"change{tag}"
    if not (cdir / "sar_state.tif").exists():
        raise FileNotFoundError("sar_state.tif missing: re-run Phase 3 (python -m src.pipeline --phase 3"
                                + (f" --scenario {cfg.get('project.scenario')}" if tag else "") + ")")
    with rasterio.open(cdir / "change_class.tif") as d:
        cls, tr, crs = d.read(1), d.transform, d.crs
    with rasterio.open(cdir / "sar_state.tif") as d:
        st = d.read()
    sar_dir, sar_pers = np.where(st[0] == 255, 0, st[0]), (st[1] == 1)
    sar_flag = sar_dir > 0
    valid = cls != 255
    shape, cell = cls.shape, abs(tr.a)
    reaches = gpd.read_file(pdir / f"river_reaches_susceptibility{tag}.gpkg", layer="reaches").to_crs(crs)
    land = np.isfinite(to_grid(pdir / "dem.tif", shape, tr, crs))
    coast_m = ndimage.distance_transform_edt(land) * cell
    rid, dist_m = rcx.corridor_labels(reaches, shape, tr, cell)
    n = len(reaches)
    m, cor, core = oc_mod.corridor_metrics(rid, dist_m, valid, cls, sar_flag, sar_dir, sar_pers, coast_m, n, oc)
    score, klass, used_w, drv = oc_mod.score_observed(m, oc)
    ring = valid & (dist_m > oc["core_buffer_m"]) & (dist_m <= oc["corridor_buffer_m"]) & (rid > 0)
    chg = np.isin(cls, [1, 2, 3, 4]) | sar_flag
    dev, probable_px, _ = oc_mod.debris_evidence(rid, core, ring, chg, cls, sar_dir, sar_pers, n, oc["debris"], cell * cell,
                                                 reaches["susceptibility_score"].to_numpy())
    for k in m.columns:
        reaches[k] = m[k].to_numpy()
    reaches["optical_change"] = (m["optical_vegetation_loss"] + m["optical_exposed_material"] + m["water_change"]).clip(upper=1).to_numpy()
    reaches["observed_change_score"], reaches["observed_change_class"] = score, klass
    for k in dev.columns:
        reaches[k] = dev[k].to_numpy()
    reaches.to_file(pdir / f"river_reaches_observed{tag}.gpkg", layer="reaches", driver="GPKG")

    out = cls.copy()
    out[cor & np.isin(cls, [1, 2, 3, 4, 6])] = 8                    # river_corridor_disturbance
    out[probable_px] = 9                                              # possible_debris_accumulation (evidence, not confirmation)
    write_raster(cdir / "change_class_corridor.tif", out, tr, crs, nodata=255, dtype="uint8")

    ok = np.isfinite(score)
    rho = float(spearmanr(reaches.loc[ok, "susceptibility_score"], score[ok]).correlation) if ok.sum() > 3 else None
    flagged = reaches[reaches["debris_flag"] != ""].sort_values(["debris_flag", "susceptibility_score"], ascending=[False, False])
    qa = dict(scenario=cfg.get("project.scenario", "default"), n_reaches=int(n),
              class_counts=reaches["observed_change_class"].value_counts().to_dict(),
              score_quantiles=np.nanpercentile(score, [5, 25, 50, 75, 95, 100]).round(3).tolist(),
              weight_fraction_available=used_w,
              unavailable_metrics=["channel_morphology (not measurable at 10 m)"],
              metric_means=m[oc_mod.METRICS].mean().round(4).to_dict(),
              debris_flag_counts=reaches["debris_flag"].replace("", "none").value_counts().to_dict(),
              probable_debris_area_m2=float(probable_px.sum() * cell * cell),
              spearman_susceptibility_vs_observed=rho,
              corridor_valid_px_median=float(np.median(m["n_valid_corridor_px"])),
              flagged_reaches=flagged[["reach_id", "debris_flag", "susceptibility_score", "observed_change_score",
                                       "debris_area_m2", "stream_order"]].head(25).to_dict("records"),
              note="Debris flags are hypotheses from satellite change; never confirmed without validation data.")
    (pdir / f"phase7_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    with rasterio.open(pdir / "hillshade.tif") as d:
        hs = d.read(1)
    with rasterio.open(pdir / "dem.tif") as d:
        t30 = d.transform
    png = plot_phase7(hs, reaches, t30, cfg.path("outputs", "diagnostics", f"phase7_observed_change_diagnostic{tag}.png"),
                      scenario=qa["scenario"], dpi=cfg["outputs"]["figure_dpi"])
    log.info("Phase 7 QA: %s", json.dumps({k: qa[k] for k in ("class_counts", "debris_flag_counts", "spearman_susceptibility_vs_observed")}, default=float))
    return dict(qa=qa, diagnostic_png=png, reaches=reaches)


def run_phase8(cfg: Config) -> dict:
    """Priority classes + per-reach confidence + inspection list. Needs Phase 7 (and Phase 2/3 rasters for the confidence)."""
    import geopandas as gpd
    from scipy import ndimage
    from .analysis import reach_context as rcx
    from .data.rasters import read_stack
    from .mapping.priority import plot_phase8
    from .models import confidence as cf
    from .models import priority as pr

    tag, pc, oc = scen_tag(cfg), cfg["priority"], cfg["observed_change"]
    pdir, gdir, cdir = cfg.path("processed"), cfg.path("interim", f"gee{tag}"), cfg.path("processed", f"change{tag}")
    reaches = gpd.read_file(pdir / f"river_reaches_observed{tag}.gpkg", layer="reaches")
    with rasterio.open(cdir / "change_class.tif") as d:
        cls, tr, crs = d.read(1), d.transform, d.crs
    with rasterio.open(cdir / "sar_state.tif") as d:
        st = d.read()
    sar_flag, sar_pers = (st[0] != 255) & (st[0] > 0), (st[1] == 1)
    s1b, _, _ = read_stack(gdir / "s1_baseline.tif"); s1a, _, _ = read_stack(gdir / "s1_after.tif")
    s2b, _, _ = read_stack(gdir / "s2_baseline.tif"); s2a, _, _ = read_stack(gdir / "s2_after.tif")
    n = len(reaches)
    rid, dist = rcx.corridor_labels(reaches.to_crs(crs), cls.shape, tr, abs(tr.a))
    cor = (cls != 255) & (dist <= oc["corridor_buffer_m"]) & (rid > 0)
    cnt = lambda m: np.bincount(rid[m], minlength=n + 1)[1:].astype(float)
    ncor = np.maximum(cnt(cor), 1)
    ch = cfg["change"]
    full1, full2 = ch["s1_obs_full"], ch["s2_obs_full"]
    s1_av = np.bincount(rid[cor], weights=np.clip(np.nan_to_num(np.minimum(s1b["n_obs"], s1a["n_obs"]))[cor] / full1, 0, 1), minlength=n + 1)[1:] / ncor
    s2_av = np.bincount(rid[cor], weights=np.clip(np.nan_to_num(np.minimum(s2b["n_clear"], s2a["n_clear"]))[cor] / full2, 0, 1), minlength=n + 1)[1:] / ncor
    opt_dil = ndimage.binary_dilation(np.isin(cls, [1, 2, 3, 4, 5]), iterations=1)
    inter, union = cnt(cor & sar_flag & opt_dil), cnt(cor & (sar_flag | opt_dil))
    agree = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    n_sar = cnt(cor & sar_flag)
    pers_share = np.where(n_sar > 0, cnt(cor & sar_flag & sar_pers) / np.maximum(n_sar, 1), 0.0)
    O = reaches["observed_change_score"].to_numpy()
    flagged_any = (n_sar > 0) | (cnt(cor & np.isin(cls, [1, 2, 3, 4, 5])) > 0)
    meta = json.loads((pdir / f"phase2_meta{tag}.json").read_text())
    tcov = cf.temporal_coverage(meta["s1"]["n_scenes_after"], meta["s2"]["n_scenes_after"], full1, full2)
    comps = dict(s1_availability=s1_av, s2_valid_fraction=s2_av, temporal_coverage=np.full(n, tcov), sensor_agreement=agree,
                 change_magnitude_persistence=cf.magnitude_persistence(flagged_any, pers_share, O))
    conf = cf.combine(comps, cfg["confidence"]["weights"])
    cb = pc["confidence_class_breaks"]
    reaches["confidence_score"] = conf
    reaches["confidence_class"] = np.where(conf < cb[0], "Low", np.where(conf < cb[1], "Medium", "High"))
    for k, v in comps.items():
        reaches[f"conf_{k}"] = v

    S = reaches["susceptibility_score"].to_numpy()
    flag = reaches["debris_flag"].fillna("").to_numpy()
    reaches["priority_score"] = pr.priority_score(S, O, pc)
    reaches["priority_class"] = pr.priority_class(S, O, flag != "", pc)
    reaches["small_catchment"] = reaches["upstream_area_m2"].to_numpy() < pc["small_catchment_m2"]
    reaches["inspection_labels"] = pr.labels(S, flag, reaches["priority_class"], pc["high_susceptibility_min"])
    reaches["susceptibility_drivers"] = reaches["notes"]
    reaches["notes"] = [r + (f" | drivers: {d}" if d else "") + (" | small catchment: may be a dry gully" if sm else "")
                        for r, d, sm in zip(pr.reasons(S, O, flag, pc["high_susceptibility_min"], pc["strong_change_min"]),
                                            reaches["susceptibility_drivers"], reaches["small_catchment"])]
    reaches.to_file(pdir / f"river_reaches_priority{tag}.gpkg", layer="reaches", driver="GPKG")

    mid = reaches.geometry.interpolate(0.5, normalized=True)
    ll = gpd.GeoSeries(mid, crs=reaches.crs).to_crs(4326)
    tab = reaches.assign(lon=ll.x.round(6), lat=ll.y.round(6), upstream_area_km2=(reaches["upstream_area_m2"] / 1e6).round(3))
    tab = tab[tab["priority_class"].isin(["Priority 1", "Priority 2", "Priority 3"])].copy()
    tab = tab.iloc[pr.rank_order(tab["priority_class"].to_numpy(), tab["priority_score"].to_numpy())]
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    tab["inspection_group"] = np.where(tab["priority_class"].isin(["Priority 1", "Priority 2"]), "Priority Inspection Location", "Watch list")
    cols = ["rank", "reach_id", "priority_class", "priority_score", "susceptibility_score", "susceptibility_class",
            "observed_change_score", "observed_change_class", "debris_flag", "confidence_score", "confidence_class",
            "inspection_group", "inspection_labels", "lon", "lat", "stream_order", "upstream_area_km2", "small_catchment", "crossing_type", "notes"]
    tdir = cfg.path("outputs", "tables")
    tab[cols].to_csv(tdir / f"priority_inspection_locations{tag}.csv", index=False)

    other = "_post_event" if tag == "" else ""
    agree_qa = None
    op = pdir / f"river_reaches_priority{other}.gpkg"
    if op.exists():
        o = gpd.read_file(op, layer="reaches")
        jac = lambda a, b: float(len(a & b) / max(len(a | b), 1))
        sets = lambda df, ks: set(df.loc[df["priority_class"].isin(ks), "reach_id"])
        agree_qa = dict(other_scenario=other.strip("_") or "default",
                        jaccard_priority_1=jac(sets(reaches, ["Priority 1"]), sets(o, ["Priority 1"])),
                        jaccard_priority_1_2=jac(sets(reaches, ["Priority 1", "Priority 2"]), sets(o, ["Priority 1", "Priority 2"])),
                        n_priority_1_2_in_both=len(sets(reaches, ["Priority 1", "Priority 2"]) & sets(o, ["Priority 1", "Priority 2"])))
    qa = dict(scenario=cfg.get("project.scenario", "default"), n_reaches=int(n),
              priority_counts=reaches["priority_class"].value_counts().reindex(pr.CLASSES).fillna(0).astype(int).to_dict(),
              confidence_counts=reaches["confidence_class"].value_counts().to_dict(),
              confidence_median_by_priority=reaches.groupby("priority_class")["confidence_score"].median().round(3).to_dict(),
              low_confidence_in_priority_1_2=int(((reaches["priority_class"].isin(["Priority 1", "Priority 2"])) & (reaches["confidence_class"] == "Low")).sum()),
              temporal_coverage=tcov, scenario_agreement=agree_qa,
              priority_1=tab[tab["priority_class"] == "Priority 1"][["reach_id", "priority_score", "susceptibility_score", "observed_change_score", "debris_flag", "confidence_score", "lon", "lat"]].to_dict("records"),
              note="Rule-based screening list. No reach is a confirmed logjam; validate with VHR imagery/field visits.")
    (pdir / f"phase8_qa{tag}.json").write_text(json.dumps(qa, indent=2, default=float))
    with rasterio.open(pdir / "hillshade.tif") as d:
        hs = d.read(1)
    with rasterio.open(pdir / "dem.tif") as d:
        t30 = d.transform
    png = plot_phase8(hs, reaches, t30, pc, cfg.path("outputs", "diagnostics", f"phase8_priority_diagnostic{tag}.png"),
                      scenario=qa["scenario"], dpi=cfg["outputs"]["figure_dpi"])
    log.info("Phase 8 QA: %s", json.dumps({k: qa[k] for k in ("priority_counts", "confidence_counts", "scenario_agreement")}, default=float))
    return dict(qa=qa, diagnostic_png=png, reaches=reaches, table=tab[cols])


def run_phase9(cfg: Config) -> dict:
    """Final deliverables: four maps (PNG) + the standard-schema river_reaches GPKG/GeoJSON. Needs Phases 1-8."""
    import geopandas as gpd
    from scipy import ndimage
    from . import export
    from .data.rasters import read_stack, to_grid
    from .mapping import cartography as cg
    from .mapping import final_maps as fm

    tag, pdir = scen_tag(cfg), cfg.path("processed")
    gdir, cdir, wdir = cfg.path("interim", f"gee{tag}"), cfg.path("processed", f"change{tag}"), cfg.path("processed", f"wood_source{tag}")
    reaches = gpd.read_file(pdir / f"river_reaches_priority{tag}.gpkg", layer="reaches")
    with rasterio.open(cdir / "change_class.tif") as d:
        cls, tr, crs = d.read(1), d.transform, d.crs
    shape, ext, cell = cls.shape, cg.extent_of(tr, cls.shape), abs(tr.a)
    reaches = reaches.to_crs(crs)
    hs = to_grid(pdir / "hillshade.tif", shape, tr, crs, resampling="bilinear")           # NaN offshore -> drawn as sea colour
    land = np.isfinite(to_grid(pdir / "dem.tif", shape, tr, crs))
    hs = np.where(land, hs, np.nan)
    rr, cc = np.where(land)                                                             # frame the maps on the island, not on the grid
    view = (tr.c + cc.min() * cell, tr.c + (cc.max() + 1) * cell, tr.f - (rr.max() + 1) * cell, tr.f - rr.min() * cell)
    coast_m = ndimage.distance_transform_edt(land) * cell
    p = cfg["periods"]
    windows = dict(baseline=(p["baseline_start"], p["baseline_end"]), after=(p["after_start"], p["after_end"]))
    label = cfg.get("project.scenario", "default")
    label = "2025 vs 2026 (season-matched)" if label in (None, "default") else f"{label} window"

    s2_rgb = None
    s2p = gdir / "s2_baseline.tif"
    if s2p.exists():
        s2b, _, _ = read_stack(s2p)
        if all(b in s2b for b in ("B4", "B3", "B2")):
            s2_rgb = fm._rgb(s2b["B4"], s2b["B3"], s2b["B2"], land=ndimage.binary_erosion(land, iterations=2))     # trim mixed land/water edge pixels
        else:
            log.warning("s2_baseline.tif lacks B4/B3/B2 -> Map A uses hillshade only")
    forest = None
    if (wdir / "forest_mask.tif").exists():
        with rasterio.open(wdir / "forest_mask.tif") as d:
            forest = d.read(1) == 1
    pot = None
    if (wdir / "wood_source_potential.tif").exists():
        with rasterio.open(wdir / "wood_source_potential.tif") as d:
            pot = d.read(1)
    crossings = None
    try:
        crossings = gpd.read_file(pdir / "river_reaches_phase1.gpkg", layer="crossings").to_crs(crs)
        if "bridge_or_crossing" in reaches and reaches["bridge_or_crossing"].any() and len(crossings):
            zone = reaches[reaches["bridge_or_crossing"].astype(bool)].geometry.buffer(cfg["crossings"]["reach_match_tolerance_m"]).unary_union
            crossings = crossings[crossings.within(zone)]                              # only crossings that actually flag a reach
    except Exception:
        log.info("No crossings layer found; Map A drawn without crossing markers")

    mdir = cfg.path("outputs", f"maps{tag}")
    mdir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["outputs"]["figure_dpi"]
    pngs = dict(
        baseline=fm.map_a_baseline(mdir / "baseline_map.png", ext, hs, s2_rgb, forest, reaches, crossings, windows, dpi, view),
        change=fm.map_b_change(mdir / "change_map_2025_2026.png", ext, hs, cls, coast_m, reaches, windows, label,
                               cfg["observed_change"]["coastal_exclusion_m"], dpi, view),
        susceptibility=fm.map_c_susceptibility(mdir / "logjam_susceptibility_map.png", ext, hs, pot, reaches, label, dpi, view),
        priority=fm.map_d_priority(mdir / "priority_inspection_map.png", ext, hs, reaches, label, windows, dpi=dpi, view=view))
    from .mapping import overlays as ov
    shown = np.where((cls >= 1) & (cls <= 7), cls, 0)
    shown[(cls == 5) & (coast_m < cfg["observed_change"]["coastal_exclusion_m"])] = 0     # same suppression as Map B
    ov_index = ov.write_overlays(cfg.path("outputs", f"overlays{tag}"), ov.build_layers(hs, shown, s2_rgb, forest, pot), tr, crs)
    odir = cfg.path("outputs")
    g = export.export_reaches(reaches, odir / f"river_reaches{tag}.gpkg", odir / f"river_reaches{tag}.geojson")
    qa = dict(scenario=cfg.get("project.scenario", "default"), n_reaches=int(len(g)), maps={k: str(v) for k, v in pngs.items()}, overlays=sorted(ov_index),
              columns=list(g.columns),
              note="Maps show rule-based screening results; nothing is a confirmed logjam.")
    (pdir / f"phase9_qa{tag}.json").write_text(json.dumps(qa, indent=2))
    log.info("Phase 9: wrote %d maps and river_reaches%s.gpkg/.geojson to %s", len(pngs), tag, odir)
    return dict(qa=qa, maps=pngs, reaches=g)


def run_phase(n: int, cfg: Config, **kw):
    if n == 1:
        return run_phase1(cfg, **kw)
    if n == 2:
        return run_phase2(cfg, **kw)
    if n == 3:
        return run_phase3(cfg)
    if n == 4:
        return run_phase4(cfg, **kw)
    if n == 5:
        return run_phase5(cfg)
    if n == 6:
        return run_phase6(cfg)
    if n == 7:
        return run_phase7(cfg)
    if n == 8:
        return run_phase8(cfg)
    if n == 9:
        return run_phase9(cfg)
    raise NotImplementedError(f"Phase {n} is not implemented yet (see docs/ARCHITECTURE.md roadmap).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--dem", default=None, help="Use this DEM GeoTIFF instead of downloading")
    ap.add_argument("--scenario", default=None, help="named window set from periods.scenarios (e.g. post_event)")
    ap.add_argument("--force", action="store_true", help="Phase 2: re-download existing composites")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(a.config).for_scenario(a.scenario)
    kw = {"dem_override": a.dem} if (a.dem and a.phase == 1) else {}
    if a.phase in (2, 4):
        kw["force"] = a.force
    run_phase(a.phase, cfg, **kw)


if __name__ == "__main__":
    main()
