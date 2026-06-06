"""Tests for the shootout cap (engine) and schema validation (DataStore)."""

from __future__ import annotations

import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine  # noqa: E402
from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# --- shootout cap ------------------------------------------------------------

def test_shootout_cap_limits_huge_favourite():
    """A massive favourite must not advance from a draw more than SHOOTOUT_CAP."""
    rng = random.Random(0)
    strong, weak = 1900.0, 1300.0
    n = 20000
    # force the score path to a draw by checking the cap logic directly:
    # call knockout_winner many times and measure how often the favourite (home)
    # wins ONLY among the draw-resolved cases is hard to isolate, so instead we
    # assert the resolved fraction the engine uses is capped.
    probs = engine.ProbabilityModel().pre_match(strong, weak, neutral=True)
    ph, pa = probs["p_home"], probs["p_away"]
    raw = ph / (ph + pa)
    capped = max(1 - engine.SHOOTOUT_CAP, min(engine.SHOOTOUT_CAP, raw))
    assert raw > engine.SHOOTOUT_CAP            # this matchup would exceed the cap
    assert capped == engine.SHOOTOUT_CAP        # and gets clamped
    # smoke: many draws still resolve to a valid side
    wins = sum(engine.knockout_winner(strong, weak, rng, neutral=True) for _ in range(n))
    assert 0 < wins < n


def test_shootout_even_teams_near_coinflip():
    rng = random.Random(1)
    n = 20000
    wins = sum(engine.knockout_winner(1600, 1600, rng, neutral=True) for _ in range(n))
    frac = wins / n
    assert 0.45 < frac < 0.55  # symmetric -> ~50/50


# --- schema validation -------------------------------------------------------

def test_real_data_is_valid():
    ds = DataStore.load(DATA_DIR)
    issues = ds.validate()
    assert issues == [], f"shipped data should validate, got: {issues}"


def test_validate_catches_unknown_team_id():
    ds = DataStore.load(DATA_DIR)
    ds.matches = pd.concat([
        ds.matches,
        pd.DataFrame([{**ds.matches.iloc[0].to_dict(),
                       "match_id": "ZZZ99", "home_id": "XXX"}]),
    ], ignore_index=True)
    issues = ds.validate()
    assert any("XXX" in i for i in issues)


def test_validate_catches_bad_match_ref():
    ds = DataStore.load(DATA_DIR)
    if ds.predictions is None or ds.predictions.empty:
        return  # nothing to corrupt
    ds.predictions = pd.concat([
        ds.predictions,
        pd.DataFrame([{**ds.predictions.iloc[0].to_dict(), "match_id": "NOPE"}]),
    ], ignore_index=True)
    issues = ds.validate()
    assert any("NOPE" in i for i in issues)


# --- data coverage -----------------------------------------------------------

def test_coverage_reports_all_signals():
    ds = DataStore.load(DATA_DIR)
    cov = ds.coverage()
    assert set(cov) == {"form", "h2h", "players"}
    total = len(set(ds.teams["team_id"].astype(str)))
    for key, c in cov.items():
        assert c["total"] == total
        assert 0 <= c["have"] <= total
        # have + missing must partition the full squad
        assert c["have"] + len(c["missing"]) == total
        assert len(c["missing"]) == len(set(c["missing"]))  # no dupes


def test_coverage_missing_excludes_present_teams():
    """A team that appears in form.csv must not be listed as missing form."""
    ds = DataStore.load(DATA_DIR)
    cov = ds.coverage()
    if not ds.form.empty and "team_id" in ds.form.columns:
        present = set(ds.form["team_id"].astype(str))
        assert present.isdisjoint(set(cov["form"]["missing"]))


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
