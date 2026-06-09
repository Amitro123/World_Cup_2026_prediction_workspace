"""Tests for src/oddslib.py — pure odds math (de-vig, divergence, compare)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import oddslib  # noqa: E402

# --- implied_1x2 -------------------------------------------------------------

def test_implied_1x2_sums_to_one():
    p = oddslib.implied_1x2(2.0, 3.5, 4.0)
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert set(p) == set(oddslib.OUTCOMES)


def test_implied_1x2_removes_vig_lowers_each_prob():
    # raw 1/odds before de-vig
    raw = {"p_home": 1 / 2.0, "p_draw": 1 / 3.5, "p_away": 1 / 4.0}
    p = oddslib.implied_1x2(2.0, 3.5, 4.0)
    # overround > 0, so every de-vigged prob is below its raw implied prob
    for k in oddslib.OUTCOMES:
        assert p[k] < raw[k]


def test_implied_1x2_fair_book_is_identity():
    # a fair (no-vig) book: 1/odds already sums to 1
    p = oddslib.implied_1x2(3.0, 3.0, 3.0)
    for k in oddslib.OUTCOMES:
        assert abs(p[k] - 1 / 3) < 1e-12


def test_implied_1x2_ordering_preserved():
    p = oddslib.implied_1x2(1.5, 4.0, 7.0)
    assert p["p_home"] > p["p_draw"] > p["p_away"]


def test_implied_1x2_rejects_bad_odds():
    for bad in [(0, 3.0, 4.0), (2.0, -1.0, 4.0), (2.0, 3.0, None)]:
        try:
            oddslib.implied_1x2(*bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


# --- overround ---------------------------------------------------------------

def test_overround_positive_for_real_book():
    assert oddslib.overround(2.0, 3.5, 4.0) > 0


def test_overround_zero_for_fair_book():
    assert abs(oddslib.overround(3.0, 3.0, 3.0)) < 1e-12


# --- implied_one -------------------------------------------------------------

def test_implied_one_basic():
    assert abs(oddslib.implied_one(4.0) - 0.25) < 1e-12


def test_implied_one_margin_reduces_prob():
    p0 = oddslib.implied_one(4.0, margin=0.0)
    p1 = oddslib.implied_one(4.0, margin=0.10)
    assert p1 < p0
    assert abs(p1 - (0.25 / 1.10)) < 1e-12


def test_implied_one_clamped():
    # odds < 1 would imply prob > 1 -> clamp to 1
    assert oddslib.implied_one(0.5) == 1.0


def test_implied_one_rejects_bad_odds():
    for bad in [0, -2.0, None]:
        try:
            oddslib.implied_one(bad)
            assert False
        except ValueError:
            pass


# --- kl ----------------------------------------------------------------------

def test_kl_identical_is_zero():
    p = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    assert abs(oddslib.kl(p, p)) < 1e-12


def test_kl_positive_when_different():
    p = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    q = {"p_home": 0.3, "p_draw": 0.3, "p_away": 0.4}
    assert oddslib.kl(p, q) > 0


def test_kl_handles_zero_without_blowup():
    p = {"p_home": 1.0, "p_draw": 0.0, "p_away": 0.0}
    q = {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}
    val = oddslib.kl(p, q)
    assert math.isfinite(val) and val > 0


# --- compare -----------------------------------------------------------------

def test_compare_agree_no_flag_when_close():
    model = {"p_home": 0.50, "p_draw": 0.30, "p_away": 0.20}
    market = {"p_home": 0.52, "p_draw": 0.29, "p_away": 0.19}
    r = oddslib.compare(model, market, flag_threshold=0.10)
    assert r["agree"] is True
    assert r["flag"] is False
    assert r["pick_model"] == "p_home" == r["pick_market"]
    assert abs(r["max_gap"]) < 0.10


def test_compare_flags_large_disagreement():
    model = {"p_home": 0.20, "p_draw": 0.30, "p_away": 0.50}
    market = {"p_home": 0.55, "p_draw": 0.25, "p_away": 0.20}
    r = oddslib.compare(model, market, flag_threshold=0.10)
    assert r["flag"] is True
    assert r["agree"] is False
    assert r["pick_model"] == "p_away"
    assert r["pick_market"] == "p_home"


def test_compare_max_gap_outcome_and_sign():
    model = {"p_home": 0.60, "p_draw": 0.25, "p_away": 0.15}
    market = {"p_home": 0.40, "p_draw": 0.30, "p_away": 0.30}
    r = oddslib.compare(model, market)
    assert r["max_gap_outcome"] == "p_home"
    assert r["max_gap"] > 0  # model > market on home


# --- market_from_row ---------------------------------------------------------

def test_market_from_row_decimal_columns():
    row = {"dec_home": 2.0, "dec_draw": 3.5, "dec_away": 4.0}
    p = oddslib.market_from_row(row)
    assert p is not None and abs(sum(p.values()) - 1.0) < 1e-12


def test_market_from_row_implied_columns_normalised():
    row = {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.4}  # sums to 1.2
    p = oddslib.market_from_row(row)
    assert abs(sum(p.values()) - 1.0) < 1e-12
    assert abs(p["p_home"] - 0.5 / 1.2) < 1e-12


def test_market_from_row_prefers_decimal_over_implied():
    row = {"dec_home": 3.0, "dec_draw": 3.0, "dec_away": 3.0,
           "p_home": 0.8, "p_draw": 0.1, "p_away": 0.1}
    p = oddslib.market_from_row(row)
    assert abs(p["p_home"] - 1 / 3) < 1e-12  # used decimal, not implied


def test_market_from_row_none_when_empty():
    assert oddslib.market_from_row({}) is None
    assert oddslib.market_from_row({"dec_home": None, "dec_draw": None,
                                    "dec_away": None}) is None


def test_market_from_row_handles_nan():
    row = {"dec_home": float("nan"), "dec_draw": 3.0, "dec_away": 4.0}
    # one missing decimal -> falls through; no implied -> None
    assert oddslib.market_from_row(row) is None


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
