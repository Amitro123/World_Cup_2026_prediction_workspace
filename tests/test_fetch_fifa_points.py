"""Tests for fetch_fifa_points: snippet parsing + propose/write logic (no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetch_fifa_points as ffp  # noqa: E402


def test_parse_points_picks_value_near_cue():
    text = "Mexico are ranked with 1681.0 FIFA points as of June 2026."
    assert ffp._parse_points(text, "MEX") == 1681.0


def test_parse_points_rejects_years_and_junk():
    # 2026 is a year (outside the FIFA window); 99 is too small — expect None.
    text = "Preview of the 2026 tournament. Group A. 99 days to go."
    assert ffp._parse_points(text, "MEX") is None


def test_parse_points_out_of_range_ignored():
    text = "Random table values 4500 and 300 appear here, no ranking shown."
    assert ffp._parse_points(text, "MEX") is None


def test_parse_points_prefers_cue_over_stray_number():
    # A stray in-range number with no cue, plus the real one next to 'points'.
    text = "Stadium capacity 1200 seats. Brazil hold 1776.03 ranking points."
    assert ffp._parse_points(text, "BRA") == 1776.03


class _FakeDS:
    """Minimal DataStore stand-in: holds ratings, records writes."""

    def __init__(self, ratings):
        self._r = dict(ratings)
        self.writes = []

    def team_rating(self, t):
        return self._r[t]

    def set_team_rating(self, t, v):
        self._r[t] = v
        self.writes.append((t, v))
        return {"team": t, "new": v}


def test_run_proposes_only_significant_changes(monkeypatch):
    canned = {"MEX": 1690.0, "BRA": 1776.0, "ARG": 1850.4}
    monkeypatch.setattr(
        ffp, "fetch_team",
        lambda t, retries=3: {"team": t, "ok": True, "points": canned[t]},
    )
    monkeypatch.setattr(ffp.time, "sleep", lambda *_: None)
    # MEX moves +9 (keep), BRA moves +0.0 (skip), ARG moves +0.4 (skip at delta 1)
    ds = _FakeDS({"MEX": 1681.0, "BRA": 1776.0, "ARG": 1850.0})
    out = ffp.run(ds, ["MEX", "BRA", "ARG"], write=False, min_delta=1.0)
    assert out["n_proposals"] == 1
    assert out["proposals"][0]["team"] == "MEX"
    assert out["proposals"][0]["delta"] == 9.0
    assert ds.writes == []  # dry-run never writes


def test_run_write_applies_proposals(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ffp, "fetch_team",
        lambda t, retries=3: {"team": t, "ok": True, "points": 1720.0},
    )
    monkeypatch.setattr(ffp.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ffp, "DATA", str(tmp_path))  # stamp() writes here
    ds = _FakeDS({"MEX": 1681.0})
    out = ffp.run(ds, ["MEX"], write=True, min_delta=1.0)
    assert out["written"] == 1
    assert ds.writes == [("MEX", 1720.0)]


def test_run_missing_points_keeps_old(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ffp, "fetch_team",
        lambda t, retries=3: {"team": t, "ok": False, "points": None},
    )
    monkeypatch.setattr(ffp.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ffp, "DATA", str(tmp_path))  # never touch the real data/ dir
    ds = _FakeDS({"MEX": 1681.0})
    out = ffp.run(ds, ["MEX"], write=True, min_delta=1.0)
    assert out["n_proposals"] == 0
    assert ds.writes == []  # a scrape miss never corrupts a rating


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
