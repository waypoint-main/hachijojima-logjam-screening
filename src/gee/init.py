"""Earth Engine initialisation. First run on a new machine opens a browser for sign-in."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)
_DONE = False


def initialize(cfg):
    """Initialise EE with the project from config (gee.project, default ee-alexdeclaro)."""
    global _DONE
    import ee  # lazy: analytics/tests never require Earth Engine
    if _DONE:
        return ee
    project = cfg["gee"]["project"]
    try:
        ee.Initialize(project=project)
    except Exception as e:  # not authenticated yet
        log.info("EE init failed (%s); running ee.Authenticate()", e)
        ee.Authenticate()
        ee.Initialize(project=project)
    _DONE = True
    log.info("Earth Engine initialised (project=%s)", project)
    return ee
