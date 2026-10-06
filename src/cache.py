"""Tiny cache-signature helper so downloaded rasters are reused ONLY if the inputs that produced
them (bbox, dates, scale, thresholds, ...) are unchanged. Prevents silently mixing stale and new data."""
from __future__ import annotations

import json
from pathlib import Path


def _norm(sig):
    return json.loads(json.dumps(sig, default=str, sort_keys=True))


def _sigfile(path: Path) -> Path:
    return path.with_name(path.name + ".sig.json")


def sig_ok(path, sig) -> bool:
    path = Path(path)
    sf = _sigfile(path)
    return path.exists() and sf.exists() and json.loads(sf.read_text()) == _norm(sig)


def write_sig(path, sig) -> None:
    _sigfile(Path(path)).write_text(json.dumps(_norm(sig), indent=1, sort_keys=True))
