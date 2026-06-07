"""Tests for fetch_player_props pure parser (no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetch_player_props as fpp  # noqa: E402


def _event():
    return {"bookmakers": [{
        "key": "bet365",
        "markets": [
            {"key": "player_goal_scorer_anytime", "outcomes": [
                {"description": "Kylian Mbappe", "price": 2.1},
                {"description": "Ousmane Dembele", "price": 3.4}]},
            {"key": "player_assists", "outcomes": [
                {"description": "Kylian Mbappe", "price": 4.0}]},
        ],
    }]}


def test_parse_event_collects_player_columns():
    players = fpp.parse_event(_event())
    mb = players["kylian mbappe"]
    assert mb["score_odds"] == 2.1
    assert mb["assist_odds"] == 4.0
    assert mb["bookmaker"] == "bet365"


def test_parse_event_ignores_unknown_markets():
    ev = {"bookmakers": [{"key": "x", "markets": [
        {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9}]}]}]}
    assert fpp.parse_event(ev) == {}


def test_parse_event_first_price_wins_with_prefer_book():
    ev = {"bookmakers": [
        {"key": "willhill", "markets": [{"key": "player_assists", "outcomes": [
            {"description": "X", "price": 5.0}]}]},
        {"key": "bet365", "markets": [{"key": "player_assists", "outcomes": [
            {"description": "X", "price": 4.0}]}]},
    ]}
    p = fpp.parse_event(ev, prefer_book="bet365")
    assert p["x"]["assist_odds"] == 4.0  # preferred book seen first


def test_rows_from_payload_resolves_known_players():
    index = {
        "kylian mbappe": {"team_id": "FRA", "name_en": "Kylian Mbappe", "name_he": "אמבפה"},
        "ousmane dembele": {"team_id": "FRA", "name_en": "Ousmane Dembele", "name_he": "דמבלה"},
    }
    rows = fpp.rows_from_payload([_event()], "F1", index, captured_at="2026-06-01")
    by_name = {r["name_en"]: r for r in rows}
    assert by_name["Kylian Mbappe"]["score_odds"] == 2.1
    assert by_name["Kylian Mbappe"]["assist_odds"] == 4.0
    assert by_name["Kylian Mbappe"]["team_id"] == "FRA"
    assert by_name["Kylian Mbappe"]["match_id"] == "F1"
    assert by_name["Ousmane Dembele"]["assist_odds"] == ""  # no assist price


def test_rows_from_payload_skips_unknown_players():
    index = {"kylian mbappe": {"team_id": "FRA", "name_en": "Kylian Mbappe", "name_he": ""}}
    rows = fpp.rows_from_payload([_event()], "F1", index)
    assert {r["name_en"] for r in rows} == {"Kylian Mbappe"}  # Dembele dropped


def test_build_player_index(tmp_path):
    p = tmp_path / "players.csv"
    p.write_text("team_id,name_he,name_en,role,goal_share,assist_share\n"
                 "FRA,אמבפה,Kylian Mbappe,FW,0.34,0.16\n", encoding="utf-8")
    index = fpp.build_player_index(str(p))
    assert index["kylian mbappe"]["team_id"] == "FRA"
    assert index["kylian mbappe"]["name_he"] == "אמבפה"


def test_write_players_market_appends(tmp_path):
    path = tmp_path / "players_market.csv"
    r = {"match_id": "F1", "team_id": "FRA", "name_en": "X", "name_he": "",
         "score_odds": 2.0, "assist_odds": "", "score_or_assist_odds": "",
         "bookmaker": "b", "captured_at": "2026-06-01"}
    fpp.write_players_market([r], path=str(path), append=False)
    fpp.write_players_market([{**r, "match_id": "F2"}], path=str(path), append=True)
    text = path.read_text(encoding="utf-8")
    assert text.count("F1") == 1 and text.count("F2") == 1
    assert text.count("match_id") == 1  # single header


if __name__ == "__main__":
    import traceback
    import inspect
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
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
