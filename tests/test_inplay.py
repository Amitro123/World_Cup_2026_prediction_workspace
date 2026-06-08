"""Tests for in-play red-card support (engine.in_play + red_card_multipliers)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine  # noqa: E402


def test_multipliers_neutral_when_no_cards():
    assert engine.red_card_multipliers(0, 0) == (1.0, 1.0)


def test_home_red_helps_away():
    mh, ma = engine.red_card_multipliers(red_home=1)
    assert mh == engine.RED_CARD_OWN
    assert ma == engine.RED_CARD_OPP
    assert mh < 1.0 < ma


def test_symmetry_home_vs_away_card():
    mh1, ma1 = engine.red_card_multipliers(red_home=1)
    mh2, ma2 = engine.red_card_multipliers(red_away=1)
    assert (mh1, ma1) == (ma2, mh2)  # mirror image


def test_multiple_cards_compose():
    mh, ma = engine.red_card_multipliers(red_home=2)
    assert abs(mh - engine.RED_CARD_OWN ** 2) < 1e-12
    assert abs(ma - engine.RED_CARD_OPP ** 2) < 1e-12


def test_negative_cards_clamped():
    assert engine.red_card_multipliers(-3, 0) == (1.0, 1.0)


def test_in_play_red_shifts_probabilities():
    m = engine.ProbabilityModel()
    base = m.in_play(1600, 1600, 60, 1, 1)
    home_red = m.in_play(1600, 1600, 60, 1, 1, red_home=1)
    # a home dismissal must lower home's win prob and raise the opponent's
    assert home_red["p_home"] < base["p_home"]
    assert home_red["p_away"] > base["p_away"]
    # base strength (lambda) is unchanged; only the remaining-time rate is scaled
    assert home_red["lambda_home"] == base["lambda_home"]
    assert home_red["red_mult_home"] == engine.RED_CARD_OWN
    assert home_red["red_mult_away"] == engine.RED_CARD_OPP


def test_in_play_red_noop_at_full_time():
    """With no time remaining, a red card cannot change a settled result."""
    m = engine.ProbabilityModel()
    base = m.in_play(1600, 1600, 90, 2, 0)
    red = m.in_play(1600, 1600, 90, 2, 0, red_home=1)
    for k in ("p_home", "p_draw", "p_away"):
        assert abs(base[k] - red[k]) < 1e-12


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
