"""Tests for fetch_odds pure parser (no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetch_odds as fo  # noqa: E402


def _event(home, away, dh, dd, da, book="pinnacle"):
    return {
        "home_team": home, "away_team": away,
        "bookmakers": [{
            "key": book,
            "markets": [{
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": dh},
                    {"name": away, "price": da},
                    {"name": "Draw", "price": dd},
                ],
            }],
        }],
    }


def test_parse_event_basic():
    p = fo.parse_event(_event("Brazil", "Morocco", 1.8, 3.6, 4.5))
    assert p["home_team"] == "Brazil" and p["away_team"] == "Morocco"
    assert p["dec_home"] == 1.8 and p["dec_draw"] == 3.6 and p["dec_away"] == 4.5
    assert p["bookmaker"] == "pinnacle"


def test_parse_event_prefers_named_book():
    ev = _event("A", "B", 2.0, 3.0, 4.0, book="bet365")
    ev["bookmakers"].insert(0, {
        "key": "williamhill",
        "markets": [{"key": "h2h", "outcomes": [
            {"name": "A", "price": 2.5}, {"name": "B", "price": 3.5},
            {"name": "Draw", "price": 3.1}]}],
    })
    p = fo.parse_event(ev, prefer_book="bet365")
    assert p["bookmaker"] == "bet365"
    assert p["dec_home"] == 2.0


def test_parse_event_none_when_two_way():
    ev = {"home_team": "A", "away_team": "B", "bookmakers": [{
        "key": "x", "markets": [{"key": "h2h", "outcomes": [
            {"name": "A", "price": 1.5}, {"name": "B", "price": 2.5}]}]}]}
    assert fo.parse_event(ev) is None


def test_parse_event_none_when_no_h2h():
    ev = {"home_team": "A", "away_team": "B", "bookmakers": [{
        "key": "x", "markets": [{"key": "totals", "outcomes": []}]}]}
    assert fo.parse_event(ev) is None


def test_rows_from_payload_matches_index():
    index = {("brazil", "morocco"): "C1"}
    payload = [_event("Brazil", "Morocco", 1.8, 3.6, 4.5)]
    rows = fo.rows_from_payload(payload, index, captured_at="2026-06-01")
    assert len(rows) == 1
    r = rows[0]
    assert r["match_id"] == "C1"
    assert r["dec_home"] == 1.8
    assert r["captured_at"] == "2026-06-01"


def test_rows_from_payload_swaps_reversed_fixture():
    # our schedule has Brazil home; the book lists Morocco home -> swap decimals
    index = {("brazil", "morocco"): "C1"}
    payload = [_event("Morocco", "Brazil", 4.5, 3.6, 1.8)]
    rows = fo.rows_from_payload(payload, index)
    assert len(rows) == 1
    r = rows[0]
    assert r["match_id"] == "C1"
    # oriented to OUR home (Brazil) -> dec_home should be Brazil's price (1.8)
    assert r["dec_home"] == 1.8 and r["dec_away"] == 4.5


def test_rows_from_payload_skips_unresolved():
    index = {("brazil", "morocco"): "C1"}
    payload = [_event("France", "Argentina", 2.0, 3.4, 3.5)]
    assert fo.rows_from_payload(payload, index) == []


def test_build_match_index(tmp_path):
    teams = tmp_path / "teams.csv"
    teams.write_text("team_id,name_he,name_en\nBRA,ברזיל,Brazil\nMAR,מרוקו,Morocco\n",
                     encoding="utf-8")
    matches = tmp_path / "matches.csv"
    matches.write_text("match_id,home_id,away_id\nC1,BRA,MAR\n", encoding="utf-8")
    index = fo.build_match_index(str(teams), str(matches))
    assert index == {("brazil", "morocco"): "C1"}


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            import inspect
            if "tmp_path" in inspect.signature(fn).parameters:
                import pathlib
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
