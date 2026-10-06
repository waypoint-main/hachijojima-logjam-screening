"""Plain-language definitions for every technical term shown to users (app, maps, docs).

Assume the reader has never heard of any of these terms. Each entry: key -> (display name, one-sentence answer, longer explanation).
No Streamlit import; docs/GLOSSARY.md is generated from this file (see `to_markdown`).
"""
from __future__ import annotations

TERMS = {
    # ---- what we are looking for -------------------------------------------------------------------------------------------------
    "logjam": (
        "Logjam",
        "A pile of tree trunks, branches and other woody material wedged in or across a river channel.",
        "A logjam can partly or fully block the flow, make water back up behind it, and may break loose in a sudden surge during the next "
        "heavy rain. Logjams are normally confirmed by looking at the river directly or on a very sharp photo; 10 m satellite pixels "
        "cannot see them, so this project only shows where one is *possible*."),
    "debris_accumulation": (
        "Debris accumulation",
        "Any build-up of material (wood, sediment, rocks, litter) in or beside a channel, or against a bridge or culvert.",
        "It is the broader term. A debris accumulation does not have to span the channel or block the water. Every logjam is a debris "
        "accumulation, but not every debris accumulation is a logjam: a heap of branches on a gravel bar, or silt and rubble piled "
        "at the mouth of a culvert, is debris without being a logjam."),
    "logjam_vs_debris": (
        "Logjam vs. debris accumulation",
        "A logjam is a specific kind of debris accumulation: woody material that wedges across or along the channel and can obstruct flow.",
        "Debris accumulation = any pile-up of material (the general idea). Logjam = a pile-up made mostly of wood that sits in the channel "
        "and can block it. In this project the two labels describe different satellite evidence, and neither means a jam was seen: "
        "'Probable Debris Accumulation' = two sensors agree that something has changed in the channel strip; "
        "'Possible Logjam' = change is concentrated in the channel strip in a reach where wood is likely to arrive and get stuck."),
    "possible_logjam": (
        "Possible Logjam",
        "A satellite hint: change is packed into the narrow strip along the channel, much more than in the land around it.",
        "It is flagged only on reaches that are already highly susceptible (see 'High Logjam Susceptibility'). It is the *weaker* of the two "
        "hints, because the same pattern can come from bank erosion, fallen streamside trees or image noise. Only a site visit or a "
        "high-resolution photo can tell. It is a hypothesis, not a finding."),
    "probable_debris": (
        "Probable Debris Accumulation",
        "A stronger satellite hint: radar and optical images agree that the channel strip has changed and the radar change has persisted.",
        "Rough new material such as piled wood or rubble can make the radar return brighter, and optical images show the matching "
        "vegetation or ground disturbance. 'Probable' means two different sensors agree, not that a jam has been confirmed."),
    "high_susceptibility": (
        "High Logjam Susceptibility",
        "This reach has conditions that make it easy for wood to arrive and get stuck. It says what could happen, not what has happened.",
        "Examples of such conditions: lots of damaged forest upstream, wood that can slide or wash into the stream, a gentle slope or "
        "narrow channel where the flow slows, a road crossing. The score is a weighted checklist of 13 such conditions, scaled 0-1."),
    "priority_inspection": (
        "Priority Inspection Location",
        "A reach where high susceptibility meets signs of change, so it is a sensible place to check first.",
        "These are Priority 1 and Priority 2 reaches. The list is a screening tool for deciding where to look first. It is not a list of confirmed problems."),
    # ---- scores and classes ------------------------------------------------------------------------------------------------------
    "reach": (
        "River reach",
        "A short piece of river, about 100 m long, that is scored on its own.",
        "The stream network is split into ~100 m pieces so each can be given its own scores and shown on a map."),
    "susceptibility": (
        "Susceptibility score (0-1)",
        "How easily this reach could collect wood. Higher = more conditions that favour a logjam.",
        "Rule-based, not machine learning. Classes: Low (under 0.25), Moderate (0.25-0.5), High (0.5-0.75), Very High (0.75+). "
        "The weights are judgement calls and have not been checked against real logjams."),
    "observed_change": (
        "Observed change score (0-1)",
        "How much the satellite images changed along this reach between the two periods.",
        "It looks at a 50 m strip along the channel and combines radar change, loss of vegetation, exposed soil and water-extent change. "
        "It measures change in the images, not necessarily damage; seasons, clouds and shadows can also change an image."),
    "priority": (
        "Priority class",
        "A combined ranking of susceptibility and observed change: Priority 1, 2, 3, Baseline or Not assessed.",
        "Priority 1: high susceptibility AND strong observed change AND a satellite debris flag. Priority 2: high susceptibility AND "
        "(strong change OR a debris flag). Priority 3: a watch list (high susceptibility with moderate change, or strong change on a "
        "moderately susceptible reach). Baseline: none of these. Not assessed: too few clear satellite pixels to measure change."),
    "priority_score": (
        "Priority score",
        "A number used only to order reaches: 0.45 x susceptibility + 0.35 x observed change + 0.20 x (both multiplied).",
        "The inspection list is ordered by this score across Priority 1 and 2 together, so the highest-scoring reach is first even if "
        "it is Priority 2. The class says why a reach is listed; the score says how to order the visits."),
    "confidence": (
        "Data confidence",
        "How good the satellite data were for this reach. It is not the probability that a logjam exists.",
        "Built from five parts: how many radar scenes and clear optical pixels were available, how well the 'after' period is covered, "
        "whether radar and optical agree, and whether the change persisted. High confidence means solid data, not a confirmed finding."),
    "small_catchment": (
        "Small catchment (headwater)",
        "The reach drains less than 0.2 km² of land, so it may be a dry gully rather than a permanent stream.",
        "The stream network is computed from a 30 m elevation model, which produces many tiny channels that may not exist on the ground. "
        "Reaches in small catchments are less certain to be real channels."),
    # ---- satellite and map terms -------------------------------------------------------------------------------------------------
    "sar": (
        "Radar (SAR, Sentinel-1)",
        "A satellite that sends its own radar pulses and records what bounces back. It sees through cloud and at night.",
        "Rough surfaces such as piled wood or rubble tend to return more signal; smooth water returns little. Radar is noisy and affected "
        "by slope direction, so it is used as supporting evidence."),
    "optical": (
        "Optical imagery (Sentinel-2)",
        "Ordinary satellite photos in visible and infrared light, 10 m per pixel. Clouds block them.",
        "Used to see loss of vegetation and exposed soil. A 10 m pixel is larger than most jams."),
    "sar_change": (
        "Radar change",
        "Share of the channel strip where the radar signal changed more than normal between the two periods.",
        "Values 0-1. Shown only as supporting evidence."),
    "optical_change": (
        "Optical change",
        "Share of the channel strip where vegetation loss, exposed soil or water change appeared in optical images.",
        "Values 0-1."),
    "wood_delivery": (
        "Wood delivery score (0-1)",
        "How much damaged-forest wood could plausibly be washed or slide into this reach.",
        "Disturbed forest on steep ground is treated as a wood source. The score follows the water downhill to the stream, so wood far "
        "from any stream counts for less. Relative index, not a volume."),
    "upstream_disturbed_area": (
        "Upstream disturbed area (m²)",
        "Area of changed forest above this reach that is treated as a wood source.",
        "Square metres, summed over everything that drains to this reach."),
    "flow_accumulation": (
        "Flow accumulation (cells)",
        "How many 30 m map cells drain through this point. A bigger number means a bigger river.",
        "One cell is about 900 m². 1,000 cells is about 0.9 km² of catchment."),
    "stream_order": (
        "Stream order",
        "1 = a small headwater stream; each time two streams of the same order meet, the order goes up by one. Higher = bigger river.",
        "Hachijojima's rivers reach order 4. Order 1 streams are smaller than the satellite can resolve well."),
    "local_slope": (
        "Local channel slope (m/m)",
        "How steeply the channel drops, metres of drop per metre of length. 0.10 = a 10% grade.",
        "Wood tends to get stuck where a steep channel suddenly flattens."),
    "crossing": (
        "Road crossing / bridge",
        "A road, bridge or culvert crosses this reach.",
        "Structures narrow the channel and can catch debris. Only vehicle roads count; bridges are known only where OpenStreetMap tags them."),
    "forest_disturbance": (
        "Forest disturbance",
        "Forest where the canopy shows less green than before. It may be wind damage, but the images cannot say what caused it.",
        "Possible causes include typhoon winds, salt spray and landslides."),
    "landslide_candidate": (
        "Landslide candidate",
        "A steep patch where vegetation was lost and bare ground appeared.",
        "A hint from the images, not a surveyed landslide."),
    "hillshade": (
        "Hillshade",
        "A grey shaded-relief picture of the terrain, as if lit from the side.", ""),
    "jaccard": (
        "Agreement (Jaccard)",
        "Share of reaches listed in both comparison windows. 1 = identical lists, 0 = nothing in common.",
        "Computed as (reaches in both) / (reaches in either)."),
    "windows": (
        "Comparison windows",
        "Which two time periods are compared.",
        "After the event (default): before = January to 1 October 2025, after = 6 October 2025 to 1 February 2026. The cut is at an abrupt "
        "change visible in the images around 2-5 October 2025. That date is inferred from images alone and is not matched to a named storm. "
        "Whole-year comparison: January-October 2025 (before) vs January-October 2026 (after), matched by season; it also picks up "
        "ordinary seasonal and year-to-year differences."),
    "unvalidated": (
        "Not validated",
        "Nothing here has been checked against real logjams. There is no reference data yet.",
        "Weights and thresholds are untested judgement calls. See the validation strategy."),
    "opacity": (
        "Raster layer opacity",
        "How see-through the coloured map layers are. Lower = more of the terrain shows through.", ""),
}

ORDER_FOR_GLOSSARY = ["logjam", "debris_accumulation", "logjam_vs_debris", "possible_logjam", "probable_debris", "high_susceptibility",
                      "priority_inspection", "priority", "priority_score", "susceptibility", "observed_change", "confidence", "reach",
                      "small_catchment", "stream_order", "crossing", "wood_delivery", "upstream_disturbed_area", "flow_accumulation",
                      "local_slope", "sar", "optical", "sar_change", "optical_change", "forest_disturbance", "landslide_candidate",
                      "hillshade", "windows", "jaccard", "unvalidated"]


def name(key): return TERMS[key][0]
def short(key): return TERMS[key][1]
def long(key): return TERMS[key][2]


def text(key):
    """Short + long explanation as one block (what the info icons show)."""
    n, s, l = TERMS[key]
    return f"**{n}** — {s}" + (f"\n\n{l}" if l else "")


def to_markdown():
    out = ["# Glossary", "", "Plain-language definitions of every term used in the maps, tables and app. Generated from `src/glossary.py`.", ""]
    for k in ORDER_FOR_GLOSSARY:
        n, s, l = TERMS[k]
        out += [f"## {n}", "", s, ""] + ([l, ""] if l else [])
    return "\n".join(out)
