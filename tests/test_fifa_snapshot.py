"""teams.csv base strength must match the bundled official FIFA ranking snapshot.

data/fifa_ranking_<YYYYMMDD>.json files are raw responses from the official
FIFA ranking API (see fetch_fifa_points.OFFICIAL_API); the NEWEST one is the
release teams.csv must match. An external review (CR4) flagged that fifa_points
had no verifiable provenance; this test pins every team's rating to the
official release so any silent drift fails CI.

Refresh flow (e.g. when FIFA publishes a new release): `python hermes.py fifa
--write` writes the exact official values AND saves the new snapshot JSON, so
this test stays green — commit both files together.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
# Newest snapshot by date-in-filename (YYYYMMDD sorts lexicographically).
SNAPSHOT = sorted(glob.glob(os.path.join(DATA_DIR, "fifa_ranking_*.json")))[-1]


def _official_points() -> dict[str, float]:
    with open(SNAPSHOT, encoding="utf-8") as f:
        d = json.load(f)
    return {r["rankingItem"]["countryCode"]: float(r["rankingItem"]["totalPoints"])
            for r in d["rankings"]}


def test_snapshot_has_full_ranking():
    pts = _official_points()
    assert len(pts) >= 200  # the full men's ranking, not a truncated page


def test_every_team_present_in_official_ranking():
    pts = _official_points()
    ds = DataStore.load(DATA_DIR)
    missing = [t for t in ds.teams.team_id if t not in pts]
    assert missing == [], f"team_ids absent from FIFA ranking snapshot: {missing}"


def test_fifa_points_match_official_release():
    """Every team's fifa_points equals the official 2026-04-01 release."""
    pts = _official_points()
    ds = DataStore.load(DATA_DIR)
    bad = []
    for t in ds.teams.team_id:
        ours = float(ds.team_rating(t))
        official = pts[t]
        if abs(ours - official) > 0.01:
            bad.append((t, ours, official))
    assert bad == [], f"fifa_points drifted from the official release: {bad}"


def test_power_rating_is_minmax_of_fifa_points():
    """power_rating must be the 0-100 min-max rescale of fifa_points."""
    ds = DataStore.load(DATA_DIR)
    fp = ds.teams["fifa_points"].astype(float)
    lo, hi = fp.min(), fp.max()
    expect = ((fp - lo) / (hi - lo) * 100.0).round(2)
    diff = (ds.teams["power_rating"].astype(float) - expect).abs()
    assert float(diff.max()) <= 0.01, "power_rating out of sync with fifa_points"
