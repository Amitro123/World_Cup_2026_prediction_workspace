"""Tests for the API-Football provider adapter (no network — _get is stubbed)."""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import providers  # noqa: E402
from src.providers import APIFootball, RateLimitError  # noqa: E402


def _client(tmp):
    return APIFootball(key="TEST", data_dir=tmp, polite=0.0)


# --- comp classification -----------------------------------------------------

def test_classify_maps_stages():
    assert providers._classify("Friendlies", "") == "friendly"
    assert providers._classify("World Cup - Qualification", "Round 5") == "qualifier"
    assert providers._classify("World Cup", "Semi-finals") == "semifinal"
    assert providers._classify("World Cup", "Round of 16") == "knockout"
    assert providers._classify("World Cup", "Final") == "final"
    assert providers._classify("World Cup", "Group A") == "group"
    assert providers._classify("UEFA Nations League", "") == "competitive"


# --- recent_form orientation -------------------------------------------------

def test_recent_form_orients_goals_to_team():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        c._map = {"BRA": 6}
        c._save_map()

        def fake_get(path, params):
            assert path == "/fixtures"
            return [
                {  # BRA at home, won 2-1
                    "fixture": {"date": "2026-03-25T20:00:00+00:00", "status": {"short": "FT"}},
                    "league": {"name": "Friendlies", "round": ""},
                    "teams": {"home": {"id": 6}, "away": {"id": 10}},
                    "goals": {"home": 2, "away": 1},
                },
                {  # BRA away, lost 0-3
                    "fixture": {"date": "2026-03-20T20:00:00+00:00", "status": {"short": "FT"}},
                    "league": {"name": "World Cup - Qualification", "round": "R7"},
                    "teams": {"home": {"id": 11}, "away": {"id": 6}},
                    "goals": {"home": 3, "away": 0},
                },
                {  # not finished -> skipped
                    "fixture": {"date": "2026-06-12T20:00:00+00:00", "status": {"short": "NS"}},
                    "league": {"name": "Friendlies", "round": ""},
                    "teams": {"home": {"id": 6}, "away": {"id": 12}},
                    "goals": {"home": None, "away": None},
                },
            ]

        c._get = fake_get
        rows = c.recent_form("BRA", "Brazil")
        assert len(rows) == 2
        home, away = rows
        assert (home["gf"], home["ga"], home["comp"]) == (2, 1, "friendly")
        assert (away["gf"], away["ga"], away["comp"]) == (0, 3, "qualifier")
        assert all(r["team_id"] == "BRA" for r in rows)
        assert home["date"] == "2026-03-25"


# --- head_to_head orientation + cutoff ---------------------------------------

def test_head_to_head_orients_and_filters_cutoff():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        c._map = {"ENG": 10, "CRO": 3}
        c._save_map()

        def fake_get(path, params):
            assert path == "/fixtures/headtohead"
            return [
                {  # ENG home, 4-2, 2021 -> oriented a=ENG
                    "fixture": {"date": "2021-06-13T16:00:00+00:00", "status": {"short": "FT"}},
                    "league": {"name": "Euro", "round": "Group D"},
                    "teams": {"home": {"id": 10}, "away": {"id": 3}},
                    "goals": {"home": 4, "away": 2},
                },
                {  # CRO home, 2-1, 2018 semifinal -> oriented to ENG => 1-2
                    "fixture": {"date": "2018-07-11T18:00:00+00:00", "status": {"short": "AET"}},
                    "league": {"name": "World Cup", "round": "Semi-finals"},
                    "teams": {"home": {"id": 3}, "away": {"id": 10}},
                    "goals": {"home": 2, "away": 1},
                },
                {  # 2012 -> before cutoff, dropped
                    "fixture": {"date": "2012-09-07T18:00:00+00:00", "status": {"short": "FT"}},
                    "league": {"name": "Friendlies", "round": ""},
                    "teams": {"home": {"id": 10}, "away": {"id": 3}},
                    "goals": {"home": 1, "away": 1},
                },
            ]

        c._get = fake_get
        rows = c.head_to_head("ENG", "CRO", "England", "Croatia", cutoff=2018)
        assert len(rows) == 2
        assert rows[0] == {"team_a": "ENG", "team_b": "CRO", "a_goals": 4,
                           "b_goals": 2, "comp": "group", "year": 2021}
        # second oriented to ENG (was CRO home 2-1)
        assert rows[1]["a_goals"] == 1 and rows[1]["b_goals"] == 2
        assert rows[1]["comp"] == "semifinal" and rows[1]["year"] == 2018


# --- id resolution caches ----------------------------------------------------

def test_resolve_team_id_caches_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        calls = {"n": 0}

        def fake_get(path, params):
            calls["n"] += 1
            return [{"team": {"id": 6, "name": "Brazil", "national": True}}]

        c._get = fake_get
        assert c.resolve_team_id("BRA", "Brazil") == 6
        assert c.resolve_team_id("BRA", "Brazil") == 6  # cached, no 2nd call
        assert calls["n"] == 1
        # persisted for a fresh client
        c2 = _client(tmp)
        assert c2._map.get("BRA") == 6


# --- env gating --------------------------------------------------------------

def test_provider_from_env_none_without_key(monkeypatch=None):
    saved = os.environ.pop("API_FOOTBALL_KEY", None)
    # Stub _load_dotenv so a real .env at the repo root (a developer's actual key)
    # cannot leak back in and make this "no key configured" assertion flaky.
    orig_loader = providers._load_dotenv
    providers._load_dotenv = lambda *a, **k: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            assert providers.provider_from_env(tmp) is None
    finally:
        providers._load_dotenv = orig_loader
        if saved is not None:
            os.environ["API_FOOTBALL_KEY"] = saved


def test_rate_limit_error_is_raised():
    with tempfile.TemporaryDirectory() as tmp:
        c = _client(tmp)
        c._map = {"BRA": 6}

        def fake_get(path, params):
            raise RateLimitError("quota")

        c._get = fake_get
        try:
            c.recent_form("BRA", "Brazil")
            raised = False
        except RateLimitError:
            raised = True
        assert raised


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
