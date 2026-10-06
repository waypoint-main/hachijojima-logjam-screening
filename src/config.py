"""Configuration loading. No Streamlit dependency.

`load_config()` returns a plain nested dict (so a UI can pass overrides in), wrapped in a
thin `Config` helper that resolves paths and validates the temporal strategy.
"""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Config:
    def __init__(self, data: dict, root: Path = PROJECT_ROOT):
        self.data = data
        self.root = Path(root)
        self.validate()

    # dict-style access: cfg["dem"]["source"]; dotted: cfg.get("dem.source")
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def with_overrides(self, overrides: dict) -> "Config":
        """Return a new Config with nested overrides (used by the Streamlit sidebar)."""
        return Config(_deep_update(copy.deepcopy(self.data), overrides), self.root)

    def for_scenario(self, name: str | None) -> "Config":
        """Config whose baseline/after windows come from `periods.scenarios[name]` (None/'default' => unchanged).

        The scenario name is stored in `project.scenario`; `scen_tag(cfg)` turns it into a file-name suffix so
        scenario outputs never overwrite the default ones.
        """
        if not name or name == "default":
            return self
        sc = (self.data["periods"].get("scenarios") or {}).get(name)
        if sc is None:
            raise KeyError(f"unknown scenario '{name}'; defined: {list((self.data['periods'].get('scenarios') or {}))}")
        return self.with_overrides({"periods": dict(sc), "project": {"scenario": name}})

    # ---- paths -----------------------------------------------------------
    def path(self, stage: str, *parts: str, create: bool = True) -> Path:
        assert stage in {"raw", "interim", "processed", "outputs"}
        p = self.root / self.data["project"]["data_dir"] / stage
        p = p.joinpath(*parts)
        if create:
            (p if not p.suffix else p.parent).mkdir(parents=True, exist_ok=True)
        return p

    # ---- validation ------------------------------------------------------
    def validate(self) -> None:
        p = self.data["periods"]
        d = {k: dt.date.fromisoformat(str(p[k])) for k in
             ("baseline_start", "baseline_end", "after_start", "after_end")}
        if d["baseline_start"] >= d["baseline_end"] or d["after_start"] >= d["after_end"]:
            raise ValueError("periods: each window must have start < end")
        if d["baseline_end"] >= d["after_start"]:
            raise ValueError("periods: baseline must end before the after period starts")
        minx, miny, maxx, maxy = self.data["study_area"]["bbox"]
        if not (minx < maxx and miny < maxy):
            raise ValueError("study_area.bbox must be [min_lon, min_lat, max_lon, max_lat]")
        w = self.data["susceptibility"]["weights"]
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError("susceptibility.weights must be non-negative with positive sum")

    @property
    def periods(self) -> dict:
        p = self.data["periods"]
        return {k: dt.date.fromisoformat(str(p[k])) for k in
                ("baseline_start", "baseline_end", "after_start", "after_end")}


def scen_tag(cfg: "Config") -> str:
    """'' for the default scenario, '_<name>' otherwise (appended to interim/processed/output names)."""
    n = cfg.get("project.scenario")
    return f"_{n}" if n and n != "default" else ""


def month_list(cfg: "Config") -> list[int] | None:
    m = cfg.data["periods"].get("season_months")
    return sorted(int(x) for x in m) if m else None


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        data = yaml.safe_load(f)
    if overrides:
        _deep_update(data, overrides)
    return Config(data, root=PROJECT_ROOT)
