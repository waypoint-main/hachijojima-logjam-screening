"""Priority inspection classes (Phase 8): rule-based matrix + continuous ranking. No ML, no Streamlit.

Priority is the intersection of HAZARD POTENTIAL (susceptibility S, Phase 6) and OBSERVED CHANGE (O, Phase 7).
Both partly derive from the same optical change signal (through wood supply), so they are not independent
evidence; the matrix therefore also demands an obstruction-consistent flag for Priority 1.

  Priority 1: S >= high_susceptibility_min AND O >= strong_change_min AND debris/obstruction flag
  Priority 2: S >= high_susceptibility_min AND (O >= strong_change_min OR debris flag)           [not P1]
  Priority 3: (S >= high AND O >= moderate_change_min) OR (S >= watch_susceptibility_min AND O >= strong)   [not P1/P2]
  Baseline  : otherwise
  Not assessed: observed change could not be measured (too few clear pixels) - never silently treated as "no change".

RANKING: the inspection list orders Priority 1 and Priority 2 TOGETHER by priority_score (a Priority 2 reach with a higher
score is visited before a Priority 1 reach with a lower one); then Priority 3, Baseline, Not assessed. The class explains
WHY a reach is listed; the score decides the visiting order.

Terminology policy: only 'Possible Logjam', 'Probable Debris Accumulation', 'High Logjam Susceptibility' and
'Priority Inspection Location' are produced. Nothing here is a confirmed logjam.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["Priority 1", "Priority 2", "Priority 3", "Baseline", "Not assessed"]
RANK_GROUP = {"Priority 1": 0, "Priority 2": 0, "Priority 3": 1, "Baseline": 2, "Not assessed": 3}


def priority_score(S, O, w) -> np.ndarray:
    S, O0 = np.nan_to_num(np.asarray(S, float)), np.asarray(O, float)
    O = np.nan_to_num(O0)
    out = w["w_susceptibility"] * S + w["w_observed"] * O + w["w_interaction"] * S * O
    return np.where(np.isnan(O0), np.nan, out)          # unmeasured change -> no score (not 'zero change')


def priority_class(S, O, flagged, pc) -> np.ndarray:
    O0 = np.asarray(O, float)
    S, O, f = np.nan_to_num(np.asarray(S, float)), np.nan_to_num(O0), np.asarray(flagged, bool)
    hi, strong, mod, watch = pc["high_susceptibility_min"], pc["strong_change_min"], pc["moderate_change_min"], pc["watch_susceptibility_min"]
    consistent = f if pc.get("obstruction_consistency_required", True) else np.ones_like(f)
    p1 = (S >= hi) & (O >= strong) & consistent
    p2 = (S >= hi) & ((O >= strong) | f) & ~p1
    p3 = (((S >= hi) & (O >= mod)) | ((S >= watch) & (O >= strong))) & ~p1 & ~p2
    out = np.where(p1, "Priority 1", np.where(p2, "Priority 2", np.where(p3, "Priority 3", "Baseline")))
    return np.where(np.isnan(O0), "Not assessed", out)


def rank_order(klass, score) -> np.ndarray:
    """Positions that sort reaches for inspection: P1+P2 together by score, then P3, Baseline, Not assessed (score descending)."""
    g = np.array([RANK_GROUP.get(k, 9) for k in klass])
    sc = -np.nan_to_num(np.asarray(score, float), nan=-1.0)
    return np.lexsort((np.arange(len(g)), sc, g))


def labels(S, flag, klass, hi_min) -> list[str]:
    """Policy-compliant label text per reach ('' when none apply)."""
    out = []
    for s, f, k in zip(S, flag, klass):
        parts = []
        if f:
            parts.append(f)
        if s >= hi_min:
            parts.append("High Logjam Susceptibility")
        if k in ("Priority 1", "Priority 2"):
            parts.append("Priority Inspection Location")
        out.append("; ".join(parts))
    return out


def reasons(S, O, flag, hi, strong) -> list[str]:
    out = []
    for s, o, f in zip(S, O, flag):
        r = [f"susceptibility {s:.2f}" + (" (high)" if s >= hi else ""),
             "observed change not measured (too few clear pixels)" if np.isnan(o) else f"observed change {o:.2f}" + (" (strong)" if o >= strong else "")]
        if f:
            r.append(f"satellite evidence: {f}")
        out.append("; ".join(r))
    return out
