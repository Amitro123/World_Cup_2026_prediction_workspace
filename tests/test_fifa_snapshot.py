"""teams.csv base strength must match the bundled official FIFA ranking snapshot.

data/fifa_ranking_20260401.json is the raw API response from
  https://inside.fifa.com/api/ranking-overview?locale=en&dateId=id15065&rankingType=football
— the FIFA/Coca-Cola Men's World Ranking released 1 April 2026, the last
official release before the 2026 World Cup kicks off (next update 11 June 2026,
after the group stage starts). An external review (CR4) flagged that
fifa_points had no verifiable provenance; this test pins every team's rating
to that official release so any silent drift fails CI.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SNAPSHOT = os.path.join(DATA_DIR, "fifa_ranking_20260401.json")


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
