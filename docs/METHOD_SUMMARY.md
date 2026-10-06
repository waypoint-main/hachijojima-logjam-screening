## How this screening works

**Inputs.** Sentinel-1 radar and Sentinel-2 optical composites (2025 baseline vs. 2026), Copernicus 30 m terrain, ESA WorldCover, OpenStreetMap roads. Everything is derived from public satellite data; no field data were used.

**1. Observed change.** Radar z-score change and optical vegetation/exposed-material change are fused into a change class per 10 m pixel. This is *observation only*.

**2. Wood source and delivery.** Disturbed forest on steep slopes is treated as potential woody-debris source. Debris is routed downslope (D8) to the first stream cell, with distance-decay on the hillslope and in the channel.

**3. Logjam susceptibility (Map C).** Each river reach gets a 0-1 weighted index of 13 interpretable drivers: wood delivery, upstream disturbance, slope break/low gradient, confluences, road crossings and bridges, channel size and others. Classes: Low / Moderate / High / Very High. Weights are uncalibrated.

**4. Debris evidence.** In a narrow channel corridor, persistent radar increase that agrees with optical disturbance gives *Probable Debris Accumulation*; change concentrated in the channel relative to its surroundings gives *Possible Logjam*. Both are hypotheses.

**5. Priority (Map D).** Susceptibility and observed change are combined in a matrix into Priority 1 / 2 / 3 / Baseline, with a heuristic confidence score from data availability, cloud-free coverage, temporal coverage, sensor agreement and change persistence.

**Allowed labels:** Possible Logjam, Probable Debris Accumulation, High Logjam Susceptibility, Priority Inspection Location. Nothing is called a confirmed logjam without validation data.

**Limits.** 10 m pixels cannot resolve small streams or small jams; the post-event window date is inferred from imagery only; wind-exposure asymmetry may partly reflect illumination effects; no reference data exist yet, so accuracy is unknown.
