"""Tests for the knockout resolution: regulation -> extra time -> penalties."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine, knockout  # noqa: E402


def _resolve_many(rh, ra, n=20000, seed=0):
    """Tally how a tie between two ratings is decided over many sims."""
    rng = random.Random(seed)
    reg = et = pens = 0
    fav_et = fav_pen = 0
    for _ in range(n):
        wi, info = engine.resolve_knockout(rh, ra, rng)
        assert wi in (0, 1)
        assert info["reg"][0] >= 0 and info["reg"][1] >= 0
        if info["et"] is None:
            reg += 1
            assert info["reg"][0] != info["reg"][1]  # decisive in 90'
            assert not info["pens"]
        elif not info["pens"]:
            et += 1
            assert info["reg"][0] == info["reg"][1]   # 90' was level
            assert info["et"][0] != info["et"][1]     # ET decisive
            fav_et += (wi == 0)
        else:
            pens += 1
            assert info["reg"][0] == info["reg"][1]
            assert info["et"][0] == info["et"][1]      # ET also level
            fav_pen += (wi == 0)
    return reg, et, pens, fav_et, fav_pen


def test_resolve_knockout_always_decides():
    reg, et, pens, _, _ = _resolve_many(1700, 1600, n=5000)
    assert reg + et + pens == 5000
    assert reg > 0 and et > 0 and pens > 0  # all three regimes occur


def test_favourite_keeps_full_edge_in_et_but_not_pens():
    """A clear favourite should win ET far above 50%, but penalties near 50%."""
    reg, et, pens, fav_et, fav_pen = _resolve_many(1850, 1500, n=40000)
    et_rate = fav_et / et
    pen_rate = fav_pen / pens
    assert et_rate > 0.65          # ET rewards the stronger side
    # shootout is capped near a coin flip by SHOOTOUT_CAP
    assert pen_rate <= engine.SHOOTOUT_CAP + 0.03
    assert 0.45 <= pen_rate <= engine.SHOOTOUT_CAP + 0.03


def test_even_teams_reach_et_in_plausible_share():
    """Evenly matched: a realistic ~20-35% of ties go beyond 90'."""
    reg, et, pens, _, _ = _resolve_many(1600, 1600, n=40000)
    beyond_90 = (et + pens) / (reg + et + pens)
    assert 0.18 <= beyond_90 <= 0.38


def test_et_lower_scoring_than_regulation():
    """ET goal expectation is scaled down (ET_LAMBDA_SCALE < 1)."""
    assert 0.0 < engine.ET_LAMBDA_SCALE < 1.0


def test_knockout_winner_matches_resolve():
    """The thin wrapper returns the same index resolve_knockout would."""
    a = random.Random(123)
    b = random.Random(123)
    for _ in range(500):
        wi_wrap = engine.knockout_winner(1720, 1580, a)
        wi_full, _ = engine.resolve_knockout(1720, 1580, b)
        assert wi_wrap == wi_full


def test_play_detail_notes_reflect_stage():
    """_play_detail tags ET and penalty outcomes; aggregate score >= reg."""
    rng = random.Random(2)
    saw_et = saw_pen = saw_reg = False
    for _ in range(3000):
        wi, hg, ag, note = knockout._play_detail(1600, 1600, rng)
        assert wi in (0, 1)
        if note == " (פנדלים)":
            saw_pen = True
            assert hg == ag           # level after ET -> shootout
        elif note == " (הארכה)":
            saw_et = True
            assert hg != ag
        else:
            saw_reg = True
            assert note == ""
    assert saw_reg and saw_et and saw_pen


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
