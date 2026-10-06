# Validation strategy

Nothing in this system has been validated. This document says what evidence would be needed, how to use it, and what each result would change.

## 1. Reference data to collect (in order of cost-effectiveness)

1. **Very-high-resolution imagery** (drone, or commercial/aerial 0.1-0.5 m) acquired after the event over the Priority 1-2 reaches and a matched sample of Priority 3 and Baseline reaches. Interpreters digitise logjams / debris accumulations / bank failures blind to the model output.
2. **Field or helicopter/photo records**: municipal, road-authority or river-management damage reports, culvert/bridge blockage logs, with date and location.
3. **Pre-event VHR or street-level imagery** to confirm pre-existing wood or obstructions (avoids counting old jams as new).
4. **Storm record**: JMA best-track/wind data for the inferred early-October 2025 event and any later event (see `VALIDATION_FLAGS.md` VF-1).

## 2. Sampling design

- Stratify by priority class (P1, P2, P3, Baseline) and by confidence class. Sample all Priority 1 reaches, a random half of Priority 2, and at least 30 reaches each from P3 and Baseline. Baseline reaches are essential: without them precision can be measured but recall cannot.
- Keep a spatial hold-out (for example one or two catchments) never used to tune weights.
- Record for each reach: jam present (yes/no/uncertain), type, approximate size, whether it obstructs a crossing, and acquisition date.

## 3. Metrics

- **Precision at the inspection list**: share of Priority 1+2 reaches with a confirmed accumulation; also reported for the Possible Logjam / Probable Debris flags separately.
- **Recall / lift**: share of all confirmed accumulations falling in Priority 1+2 and in High/Very High susceptibility; lift versus the island-wide base rate.
- **Rank quality**: ROC-AUC and precision-recall of `susceptibility_score`, `observed_change_score` and `priority_score` against confirmed reaches.
- **Calibration of confidence**: does the hit rate rise with `confidence_score`?
- **Crossing obstruction**: hit rate at road crossings versus elsewhere.

## 4. Calibration (only after metrics are known)

- Adjust susceptibility weights and class breaks using the training catchments; check on the hold-out. Keep rules interpretable (no black-box models).
- Revisit thresholds already flagged as sensitive: priority-1 susceptibility cut (about 0.50), Possible Logjam contrast ratio, persistence k.
- Re-run the weight-sensitivity and cross-scenario agreement diagnostics after any change; report both.

## 5. Specific questions the data should answer

- Is the aspect asymmetry (N/NE/E more affected) real wind damage or illumination bias (VF-2)?
- How much of the after-window change is seasonality or recovery rather than event damage (VF-3)?
- Do reaches without a stream path from disturbed areas (~19% of source area) hide real sources, i.e. is the stream threshold too coarse?
- Is 10 m resolution adequate for order 1-2 streams, or should those be excluded from the inspection list?

## 6. Reporting rules until validated

Use only the allowed labels; show confidence beside every priority; state that results are screening hypotheses; never describe a reach as a confirmed logjam without a reference observation.
