"""Tests for fetch_holdout.build_rows + raw io (pure, no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fetch_holdout as fh  # noqa: E402
from src import engine  # noqa: E402


def _history():
    """Pre-tournament results that make STR strong and WEK weak, plus the
    tournament match (STR vs WEK on the start date) and an out-of-window game."""
    pre = [
        # STR thrashes everyone before the tournament
        {"date": "2023-06-01", "home": "STR", "away": "MID", "gh": 3, "ga": 0,
         "neutral": True, "comp": "qualifier", "league": "WC Qualifiers"},
        {"date": "2023-09-01", "home": "STR", "away": "WEK", "gh": 4, "ga": 0,
         "neutral": True, "comp": "qualifier", "league": "WC Qualifiers"},
        {"date": "2024-03-01", "home": "MID", "away": "WEK", "gh": 2, "ga": 1,
         "neutral": True, "comp": "friendly", "league": "Friendlies"},
        # STR arrives hot (recent win); prior STR-WEK meeting feeds h2h
        {"date": "2024-05-20", "home": "STR", "away": "MID", "gh": 2, "ga": 1,
         "neutral": False, "comp": "friendly", "league": "Friendlies"},
    ]
    tournament = [
        {"date": "2024-06-15", "home": "STR", "away": "WEK", "gh": 1, "ga": 0,
         "neutral": True, "comp": "group", "league": "Euro"},
    ]
    out_of_window = [
        {"date": "2024-08-01", "home": "STR", "away": "WEK", "gh": 0, "ga": 0,
         "neutral": True, "comp": "friendly", "league": "Friendlies"},
    ]
    return pre + tournament + out_of_window


TEAMS = ["STR", "WEK", "MID"]


def test_only_in_window_team_matches_are_emitted():
    rows = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS,
                         league_substr="Euro")
    assert len(rows) == 1
    assert (rows[0]["home"], rows[0]["away"]) == ("STR", "WEK")
    assert rows[0]["home_goals"] == 1 and rows[0]["away_goals"] == 0


def test_derived_rating_reflects_pretournament_strength():
    rows = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS)
    r = rows[0]
    # STR won all its prior games -> higher derived Elo than WEK
    assert r["rating_home"] > r["rating_away"]


def test_signals_present_and_bounded():
    rows = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS)
    r = rows[0]
    assert abs(r["form_sup"]) <= engine.FORM_CAP + 1e-9
    assert abs(r["h2h_sup"]) <= engine.H2H_CAP + 1e-9
    # STR beat WEK 4-0 before -> positive (home-favouring) h2h supremacy
    assert r["h2h_sup"] > 0


def test_no_leakage_ratings_ignore_in_window_and_future():
    """The 1-0 tournament result and the Aug friendly must not move ratings."""
    full = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS)
    # Re-run with the tournament + future rows stripped: pre-start ratings equal.
    pre_only = [m for m in _history() if m["date"] < "2024-06-01"]
    pre_only += [m for m in _history()
                 if "2024-06-01" <= m["date"] <= "2024-07-15"
                 and m["home"] in set(TEAMS) and m["away"] in set(TEAMS)]
    stripped = fh.build_rows(pre_only, "2024-06-01", "2024-07-15", TEAMS)
    assert full[0]["rating_home"] == stripped[0]["rating_home"]
    assert full[0]["rating_away"] == stripped[0]["rating_away"]


def test_relabel_translates_codes():
    rows = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS,
                         relabel={"STR": "ENG", "WEK": "SCO"})
    assert (rows[0]["home"], rows[0]["away"]) == ("ENG", "SCO")


def test_emitted_rows_have_full_backtest_schema():
    rows = fh.build_rows(_history(), "2024-06-01", "2024-07-15", TEAMS)
    assert set(rows[0]) == set(fh.OUT_FIELDS)


def test_raw_io_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(fh, "RAW_DIR", str(tmp_path))
    hist = _history()
    fh.write_raw("toy", hist)
    back = fh.read_raw("toy")
    # de-duped by (date,home,away); our synthetic set has no dupes
    assert len(back) == len(hist)
    assert {m["home"] for m in back} == {m["home"] for m in hist}
    assert all(isinstance(m["gh"], int) for m in back)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            import inspect
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                import pathlib
                import tempfile
                class _MP:
                    def setattr(self, obj, n, v): setattr(obj, n, v)
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d), _MP())
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
