"""Tests for the rating-gap -> supremacy mapping (engine.SUP_MODE).

The default is "linear" (the shipped, backtested model); "logratio" is a
validated opt-in that compresses blowout supremacies. These tests lock in:
  - the default is linear and unchanged;
  - the two modes agree at the mean (SUP_ALPHA = FIFA_MEAN/K slope-match);
  - logratio compresses the tails (a huge gap yields LESS supremacy);
  - switching modes via setattr is honoured at call time (so the backtest
    override mechanism works);
  - non-positive ratings fall back to linear instead of blowing up ln().
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine  # noqa: E402


def test_default_mode_is_linear():
    assert engine.SUP_MODE == "linear"


def test_linear_mapping_matches_formula():
    sup = engine._rating_supremacy(1700, 1500)
    assert abs(sup - (1700 - 1500) / engine.K) < 1e-12


def _with_mode(mode, alpha=None):
    """Context-ish helper: set mode/alpha, return a restore callable."""
    saved = (engine.SUP_MODE, engine.SUP_ALPHA)
    engine.SUP_MODE = mode
    if alpha is not None:
        engine.SUP_ALPHA = alpha
    return lambda: setattr_both(*saved)


def setattr_both(mode, alpha):
    engine.SUP_MODE = mode
    engine.SUP_ALPHA = alpha


def test_logratio_matches_linear_at_the_mean():
    """With SUP_ALPHA = FIFA_MEAN/K the two modes agree for small gaps."""
    restore = _with_mode("logratio", engine.FIFA_MEAN / engine.K)
    try:
        # small gap around the mean: linear slope ~ log slope
        lin = (1520 - 1500) / engine.K
        log = engine._rating_supremacy(1520, 1500)
        assert abs(lin - log) < 5e-3  # close near the anchor, not identical
    finally:
        restore()


def test_logratio_compresses_vs_mean_anchor():
    """A favourite measured against a mean-rated opponent gets LESS supremacy
    under slope-matched logratio — the guaranteed compression (ln(u) < u-1)."""
    rh, ra = 1850.0, engine.FIFA_MEAN  # opponent exactly at the anchor
    lin = (rh - ra) / engine.K
    restore = _with_mode("logratio", engine.FIFA_MEAN / engine.K)  # slope-match
    try:
        log = engine._rating_supremacy(rh, ra)
        assert log < lin  # ln(rh/mean) < (rh-mean)/mean -> compression
    finally:
        restore()


def test_mode_switch_is_honoured_by_expected_goals():
    """setattr on the module constant must change expected_goals at call time."""
    rh, ra = 1850.0, engine.FIFA_MEAN
    lam_lin = engine.expected_goals(rh, ra, neutral=True)
    restore = _with_mode("logratio", engine.FIFA_MEAN / engine.K)  # slope-match
    try:
        lam_log = engine.expected_goals(rh, ra, neutral=True)
    finally:
        restore()
    # the favourite's lambda should be lower under compressing logratio
    assert lam_log[0] < lam_lin[0]
    assert lam_log != lam_lin  # the switch actually did something


def test_nonpositive_rating_falls_back_to_linear():
    """A placeholder team with rating 0 must not raise (no ln(0))."""
    restore = _with_mode("logratio", 7.0)
    try:
        sup = engine._rating_supremacy(0.0, 1500.0)  # would be ln(0) if not guarded
        assert sup == (0.0 - 1500.0) / engine.K
    finally:
        restore()


def test_probabilities_still_sum_to_one_in_logratio():
    restore = _with_mode("logratio", 7.0)
    try:
        r = engine.ProbabilityModel().pre_match(1700, 1300)
        assert abs(r["p_home"] + r["p_draw"] + r["p_away"] - 1.0) < 1e-9
    finally:
        restore()


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
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
