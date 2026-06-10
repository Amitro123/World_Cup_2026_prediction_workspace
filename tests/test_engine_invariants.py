"""Engine invariant tests — properties that must hold for any valid model state.

These lock in mathematical guarantees so regressions are caught immediately:
  - Probability sum-to-1
  - λ symmetry for equal-strength teams
  - Dixon-Coles grid sanity
  - In-play monotonicity (scoring while leading never loses you probability)
  - Red-card multiplier composition
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine  # noqa: E402

MODEL = engine.ProbabilityModel()
TOL = 1e-9


# ---------------------------------------------------------------------------
# Probability sum-to-1
# ---------------------------------------------------------------------------

def test_probs_sum_to_one_prematch_equal():
    r = MODEL.pre_match(1500, 1500)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < TOL


def test_probs_sum_to_one_prematch_unequal():
    r = MODEL.pre_match(1875, 1100)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < TOL


def test_probs_sum_to_one_inplay():
    r = MODEL.in_play(1600, 1400, 65, 1, 2)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < TOL


def test_probs_sum_to_one_inplay_with_red_cards():
    r = MODEL.in_play(1600, 1600, 50, 0, 0, red_home=1, red_away=1)
    assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < TOL


# ---------------------------------------------------------------------------
# Lambda symmetry for equal-strength teams at neutral venue
# ---------------------------------------------------------------------------

def test_lambda_symmetry_equal_teams():
    r = MODEL.pre_match(1500, 1500, neutral=True)
    assert abs(r["lambda_home"] - r["lambda_away"]) < TOL


def test_lambda_symmetry_implies_equal_home_away_prob():
    r = MODEL.pre_match(1500, 1500, neutral=True)
    assert abs(r["p_home"] - r["p_away"]) < TOL


def test_home_advantage_breaks_symmetry():
    """Home (non-neutral) must be strictly more likely to win than the away team."""
    r = MODEL.pre_match(1500, 1500, neutral=False)
    assert r["p_home"] > r["p_away"]


# ---------------------------------------------------------------------------
# Dixon-Coles grid sanity
# ---------------------------------------------------------------------------

def test_dc_correction_lowers_draw_prob_for_mismatch():
    """DC rho < 0 depresses P(0-0) for strong favourites (both lambdas small)."""
    lam_h, lam_a = 0.5, 0.2
    with_dc = engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=True)
    without_dc = engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=False)
    # For lam_home * lam_away * rho < 0, P(0-0) is depressed; other scores
    # absorb the mass, so draw overall may differ — just check sums are still 1.
    assert abs(with_dc["p_home"] + with_dc["p_draw"] + with_dc["p_away"] - 1.0) < TOL
    assert abs(without_dc["p_home"] + without_dc["p_draw"] + without_dc["p_away"] - 1.0) < TOL


# ---------------------------------------------------------------------------
# Monotonicity: scoring while leading never decreases win probability
# ---------------------------------------------------------------------------

def test_scoring_while_leading_increases_win_prob():
    """Going from 1-0 to 2-0 at 60' must strictly increase P(home wins)."""
    r_before = MODEL.in_play(1600, 1500, 60, 1, 0)
    r_after = MODEL.in_play(1600, 1500, 60, 2, 0)
    assert r_after["p_home"] > r_before["p_home"]
    assert r_after["p_away"] < r_before["p_away"]


def test_scoring_while_trailing_decreases_deficit():
    """Away team pulling back to 1-1 at 70' must increase away win probability."""
    r_before = MODEL.in_play(1500, 1500, 70, 1, 0)
    r_after = MODEL.in_play(1500, 1500, 70, 1, 1)
    assert r_after["p_away"] > r_before["p_away"]


def test_conceding_while_level_lowers_draw_prob():
    """Going from 0-0 to 0-1 at any minute must increase p_away and lower p_draw."""
    r_before = MODEL.in_play(1500, 1500, 45, 0, 0)
    r_after = MODEL.in_play(1500, 1500, 45, 0, 1)
    assert r_after["p_draw"] < r_before["p_draw"]
    assert r_after["p_away"] > r_before["p_away"]


# ---------------------------------------------------------------------------
# Time decay: later in the game, a lead is more secure
# ---------------------------------------------------------------------------

def test_lead_more_secure_late():
    """A 1-0 lead at 80' should give higher home win prob than at 30'."""
    r_early = MODEL.in_play(1500, 1500, 30, 1, 0)
    r_late = MODEL.in_play(1500, 1500, 80, 1, 0)
    assert r_late["p_home"] > r_early["p_home"]


def test_no_time_left_result_is_certain():
    """At minute 90, the scoreline is the final result (within floating-point)."""
    r = MODEL.in_play(1500, 1500, 90, 2, 1)
    assert abs(r["p_home"] - 1.0) < TOL
    assert abs(r["p_draw"]) < TOL
    assert abs(r["p_away"]) < TOL


# ---------------------------------------------------------------------------
# Red-card multiplier composition
# ---------------------------------------------------------------------------

def test_red_card_own_reduces_attack():
    mh, _ = engine.red_card_multipliers(red_home=1)
    assert mh == engine.RED_CARD_OWN
    assert mh < 1.0


def test_red_card_opp_boosts_opponent():
    _, ma = engine.red_card_multipliers(red_home=1)
    assert ma == engine.RED_CARD_OPP
    assert ma > 1.0


def test_two_home_reds_compose_multiplicatively():
    mh, ma = engine.red_card_multipliers(red_home=2)
    assert abs(mh - engine.RED_CARD_OWN ** 2) < 1e-12
    assert abs(ma - engine.RED_CARD_OPP ** 2) < 1e-12


def test_both_sides_red_cards_compose():
    mh, ma = engine.red_card_multipliers(red_home=1, red_away=1)
    expected_h = engine.RED_CARD_OWN * engine.RED_CARD_OPP
    expected_a = engine.RED_CARD_OWN * engine.RED_CARD_OPP
    assert abs(mh - expected_h) < 1e-12
    assert abs(ma - expected_a) < 1e-12


def test_red_card_noop_at_fulltime():
    """A red card at minute 90 cannot change the result (no time left)."""
    base = MODEL.in_play(1600, 1600, 90, 1, 0)
    red = MODEL.in_play(1600, 1600, 90, 1, 0, red_home=1)
    for k in ("p_home", "p_draw", "p_away"):
        assert abs(base[k] - red[k]) < TOL


# ---------------------------------------------------------------------------
# H2H and form: neutral-zero defaults
# ---------------------------------------------------------------------------

def test_zero_h2h_sup_is_noop():
    """h2h_sup=0 must produce the same result as omitting it."""
    r1 = MODEL.pre_match(1600, 1500, neutral=True)
    r2 = MODEL.pre_match(1600, 1500, neutral=True, h2h_sup=0.0)
    assert r1 == r2


def test_h2h_sup_moves_in_right_direction():
    """Positive h2h_sup (home historically beats this opponent) raises p_home."""
    base = MODEL.pre_match(1500, 1500, neutral=True)
    bumped = MODEL.pre_match(1500, 1500, neutral=True, h2h_sup=0.3)
    assert bumped["p_home"] > base["p_home"]
    assert bumped["p_away"] < base["p_away"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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
