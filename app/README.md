# Streamlit app (Phase 10)

    streamlit run app/streamlit_app.py

Needs Phase 9 outputs (`data/outputs/river_reaches.gpkg`, `maps/*.png`; `_post_event` versions for the second window).
The app only reads those files. All filtering/ranking logic is in `src/query.py` (no Streamlit import, unit-tested).
Re-run Phase 9 once after upgrading so the raster overlays exist. The interactive map has a Map view selector (A-D) that layers the four final maps live.
Tabs: Overview, Map, Inspection list, Reach details, Window comparison, Map gallery, Glossary, Method & limits.
The default comparison window is the post-event one (before = to 1 Oct 2025, after = from 6 Oct 2025); the whole-year 2025-vs-2026 window is the alternative. Colours come from `.streamlit/config.toml` (theme) and the CSS block at the top of `streamlit_app.py`.
Set `HACHIJOJIMA_CONFIG=/path/to/config.yaml` to use another config.
