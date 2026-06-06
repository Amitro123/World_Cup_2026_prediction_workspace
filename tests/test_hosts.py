"""Tests for host-only home advantage (World Cup 2026 is on neutral soil).

Group games carry a home-crowd advantage ONLY when the home team is a 2026 host
nation (USA / MEX / CAN). Every other group game — and all knockout games — is
played at a neutral venue, so HOME_SUP must not be applied to a non-host home_id.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import engine  # noqa: E402
from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _recompute(ds, mid, m, *, neutral):
    """Replicate pre_match_probs for a match with an explicit neutral flag."""
    r_home, r_away, mult_h, mult_a, _ = ds._adjusted_inputs(mid, apply_news=False)
    expert = ds.expert_for(mid)
    h2h_sup = ds.h2h_supremacy_for(m.home_id, m.away_id)
    form_sup = ds.form_supremacy_for(m.home_id, m.away_id)
    lam_h, lam_a = engine.expected_goals(
        r_home, r_away, neutral=neutral, expert=expert, h2h_sup=h2h_sup, form_sup=form_sup
    )
    return engine.probs_from_lambdas(lam_h * mult_h, lam_a * mult_a, dixon_coles=True)


def test_is_host_identifies_the_three_co_hosts():
    ds = DataStore.load(DATA_DIR)
    assert ds.is_host("USA")
    assert ds.is_host("MEX")
    assert ds.is_host("CAN")
    # a non-host team in the field
    assert not ds.is_host("KOR")
    assert not ds.is_host("CZE")


def test_engine_hosts_constant():
    assert engine.HOSTS == frozenset({"USA", "MEX", "CAN"})


def test_non_host_group_game_is_neutral():
    """A non-host home team gets NO home bump: pre_match == neutral computation."""
    ds = DataStore.load(DATA_DIR)
    target = next(
        (mid, m)
        for mid, m in ((r.match_id, r) for _, r in ds.matches.iterrows())
        if not ds.is_host(m.home_id)
    )
    mid, m = target
    got = ds.pre_match_probs(mid)
    expected = _recompute(ds, mid, m, neutral=True)
    assert abs(got["p_home"] - expected["p_home"]) < 1e-9
    assert abs(got["p_away"] - expected["p_away"]) < 1e-9


def test_host_home_game_keeps_crowd_advantage():
    """A host home team DOES get the bump: pre_match == non-neutral, and the host
    is favoured more at home than it would be on neutral soil."""
    ds = DataStore.load(DATA_DIR)
    target = next(
        ((mid, m) for mid, m in ((r.match_id, r) for _, r in ds.matches.iterrows())
         if ds.is_host(m.home_id)),
        None,
    )
    if target is None:
        return  # no host plays at home in the fixtures shipped
    mid, m = target
    got = ds.pre_match_probs(mid)
    with_home = _recompute(ds, mid, m, neutral=False)
    neutral = _recompute(ds, mid, m, neutral=True)
    assert abs(got["p_home"] - with_home["p_home"]) < 1e-9
    assert got["p_home"] > neutral["p_home"]  # crowd advantage is real


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
