"""Tests for src/playerprops.py — per-match player goal/assist props."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import playerprops as pp  # noqa: E402


# --- p_at_least_one ----------------------------------------------------------

def test_p_at_least_one_zero_rate():
    assert pp.p_at_least_one(0.0) == 0.0


def test_p_at_least_one_matches_poisson():
    lam = 0.7
    assert abs(pp.p_at_least_one(lam) - (1 - math.exp(-lam))) < 1e-12


def test_p_at_least_one_monotonic_and_bounded():
    a = pp.p_at_least_one(0.2)
    b = pp.p_at_least_one(1.5)
    assert 0 <= a < b < 1


def test_p_at_least_one_clamps_negative():
    assert pp.p_at_least_one(-3.0) == 0.0


# --- player_match_props ------------------------------------------------------

def test_player_match_props_basic():
    r = pp.player_match_props(goal_share=0.34, assist_share=0.16, team_lambda=1.8)
    assert abs(r["exp_goals"] - 0.34 * 1.8) < 1e-12
    assert abs(r["exp_assists"] - 0.16 * 1.8) < 1e-12
    assert abs(r["p_score"] - (1 - math.exp(-0.34 * 1.8))) < 1e-12


def test_player_match_props_score_or_assist_combines_rates():
    r = pp.player_match_props(0.30, 0.20, 2.0)
    expected = 1 - math.exp(-(0.30 * 2.0 + 0.20 * 2.0))
    assert abs(r["p_score_or_assist"] - expected) < 1e-12
    # the combined prob exceeds either single prob
    assert r["p_score_or_assist"] > r["p_score"]
    assert r["p_score_or_assist"] > r["p_assist"]


def test_player_match_props_zero_lambda():
    r = pp.player_match_props(0.5, 0.5, 0.0)
    assert r["p_score"] == 0.0 and r["p_assist"] == 0.0
    assert r["p_score_or_assist"] == 0.0


def test_player_match_props_higher_share_higher_prob():
    low = pp.player_match_props(0.1, 0.1, 1.5)
    high = pp.player_match_props(0.4, 0.1, 1.5)
    assert high["p_score"] > low["p_score"]


# --- market_props_from_row ---------------------------------------------------

def test_market_props_from_row_both_odds():
    row = {"score_odds": 2.5, "assist_odds": 4.0}
    m = pp.market_props_from_row(row)
    assert abs(m["p_score"] - 1 / 2.5) < 1e-12
    assert abs(m["p_assist"] - 1 / 4.0) < 1e-12
    assert m["p_score_or_assist"] is None  # no explicit price given


def test_market_props_from_row_margin_reduces():
    base = pp.market_props_from_row({"score_odds": 2.0})
    vigged = pp.market_props_from_row({"score_odds": 2.0}, margin=0.10)
    assert vigged["p_score"] < base["p_score"]


def test_market_props_from_row_partial():
    m = pp.market_props_from_row({"score_odds": 3.0})
    assert m["p_score"] is not None
    assert m["p_assist"] is None


def test_market_props_from_row_explicit_combo():
    m = pp.market_props_from_row({"score_or_assist_odds": 1.8})
    assert abs(m["p_score_or_assist"] - 1 / 1.8) < 1e-12


def test_market_props_from_row_empty():
    m = pp.market_props_from_row({})
    assert m == {"p_score": None, "p_assist": None, "p_score_or_assist": None}


def test_market_props_from_row_handles_nan_and_bad():
    m = pp.market_props_from_row({"score_odds": float("nan"), "assist_odds": 0})
    assert m["p_score"] is None and m["p_assist"] is None


# --- compare_props -----------------------------------------------------------

def test_compare_props_flags_large_gap():
    model = {"p_score": 0.55, "p_assist": 0.20, "p_score_or_assist": 0.62}
    market = {"p_score": 0.35, "p_assist": 0.18, "p_score_or_assist": 0.45}
    c = pp.compare_props(model, market, flag_threshold=0.12)
    assert c["p_score"]["flag"] is True
    assert c["p_assist"]["flag"] is False
    assert c["p_score"]["gap"] > 0


def test_compare_props_skips_missing_selections():
    model = {"p_score": 0.5, "p_assist": 0.2, "p_score_or_assist": 0.6}
    market = {"p_score": 0.45, "p_assist": None, "p_score_or_assist": None}
    c = pp.compare_props(model, market)
    assert "p_score" in c
    assert "p_assist" not in c
    assert "p_score_or_assist" not in c


def test_compare_props_gap_sign():
    model = {"p_score": 0.30}
    market = {"p_score": 0.50}
    c = pp.compare_props(model, market)
    assert c["p_score"]["gap"] < 0  # model below market


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
