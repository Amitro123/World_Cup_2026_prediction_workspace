"""
בדיקת מומנטום — proof that the recent-form (momentum) signal works AND that the
agent-facing path (pre_match_probs / match_briefing) actually uses it.

Run directly:
    python tests/test_form.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import engine
from src.models import DataStore

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _ok(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


def test_engine_unit():
    print("[1] engine.form_score / form_supremacy — pure logic")

    # direction: a win is positive momentum, a loss negative, symmetric
    won = engine.form_score([{"gf": 2, "ga": 0, "comp": "friendly", "date": "2026-05-01"}],
                            ref_date="2026-06-06")
    lost = engine.form_score([{"gf": 0, "ga": 2, "comp": "friendly", "date": "2026-05-01"}],
                             ref_date="2026-06-06")
    _ok("a win scores positive, a loss negative, symmetric", won > 0 > lost and abs(won + lost) < 1e-9)

    # bigger margin = more momentum (but capped)
    small = engine.form_score([{"gf": 1, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                              ref_date="2026-06-06")
    big = engine.form_score([{"gf": 4, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                            ref_date="2026-06-06")
    _ok("a bigger win is stronger momentum", big > small > 0)

    # margin is capped: a 7-0 is not meaningfully more than a 3-0
    blowout = engine.form_score([{"gf": 7, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                                ref_date="2026-06-06")
    capped = engine.form_score([{"gf": 3, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                               ref_date="2026-06-06")
    _ok("goal margin is capped (7-0 ~= 3-0)", blowout == capped)

    # competitive form counts a bit more than a friendly
    def one(comp):
        return engine.form_score([{"gf": 2, "ga": 0, "comp": comp, "date": "2026-05-01"}],
                                 ref_date="2026-06-06")
    _ok("a competitive win moves more than a friendly", one("friendly") < one("group"))

    # recency: a recent result counts more than an old one
    recent = engine.form_score([{"gf": 2, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                               ref_date="2026-06-06")
    old = engine.form_score([{"gf": 2, "ga": 0, "comp": "group", "date": "2024-05-01"}],
                            ref_date="2026-06-06")
    _ok("older results decay", 0 < old < recent)

    # shrinkage: a streak of wins moves more than a single one
    one_win = engine.form_score([{"gf": 2, "ga": 0, "comp": "group", "date": "2026-05-01"}],
                                ref_date="2026-06-06")
    streak = engine.form_score([{"gf": 2, "ga": 0, "comp": "group", "date": "2026-05-01"}] * 5,
                               ref_date="2026-06-06")
    _ok("a streak is stronger than one game (sample-size shrinkage)", one_win < streak)

    # no data -> exactly zero momentum (a team with no record is unaffected)
    _ok("empty form = zero", engine.form_score([]) == 0.0)

    # supremacy: hotter team gets the nudge; equal/absent form cancels to ~0
    sup = engine.form_supremacy(0.8, -0.4)
    _ok("the hotter team gets positive supremacy", sup > 0)
    _ok("identical form cancels to 0", engine.form_supremacy(0.5, 0.5) == 0.0)
    _ok("no form on either side = 0", engine.form_supremacy(0.0, 0.0) == 0.0)

    # cap: an extreme momentum gap is still bounded
    huge = engine.form_supremacy(100.0, -100.0)
    _ok("supremacy is capped at FORM_CAP", huge <= engine.FORM_CAP + 1e-9)

    # never-played pair leaves expected goals unchanged
    rh = ra = 1700.0
    untouched = engine.expected_goals(rh, ra, form_sup=engine.form_supremacy(0.0, 0.0))
    baseline = engine.expected_goals(rh, ra)
    _ok("no-momentum pair leaves expected goals unchanged", untouched == baseline)


def test_probs_shift():
    print("[2] win probability actually moves with momentum")
    rh, ra = 1800.0, 1800.0  # identical strength, neutral -> isolate the form effect
    base = engine.ProbabilityModel().pre_match(rh, ra, neutral=True)
    hot = engine.ProbabilityModel().pre_match(rh, ra, neutral=True, form_sup=0.3)
    print(f"     even teams, no momentum : p_home={base['p_home']:.3f}")
    print(f"     even teams, +momentum   : p_home={hot['p_home']:.3f}")
    _ok("positive momentum raises the home win probability", hot["p_home"] > base["p_home"])


def test_agent_path():
    print("[3] the AGENT path (match_briefing) reflects momentum")
    ds = DataStore.load(DATA)

    # find a real fixture where at least one side has recorded form
    target = None
    for m in ds.matches.itertuples():
        if ds.recent_form(m.home_id) or ds.recent_form(m.away_id):
            target = m
            break

    if target is None:
        # No seeded fixture had form -> inject one so the test still proves the path.
        m0 = ds.matches.iloc[0]
        ds.form = pd.DataFrame(
            [{"team_id": m0.home_id, "gf": 4, "ga": 0, "comp": "group", "date": "2026-05-20"}]
        )
        target = m0
        print(f"     (no seeded fixture had form; injected a {m0.home_id} 4-0 for the test)")

    mid = target.match_id
    with_form = ds.match_briefing(mid)

    # now wipe form and recompute -> the briefing must change
    ds.form = ds.form.iloc[0:0]
    without = ds.match_briefing(mid)

    print(f"     match {mid}: with momentum base={with_form['base']}")
    print(f"     match {mid}: no   momentum base={without['base']}")
    _ok("match_briefing output changes when momentum is present vs absent",
        with_form["base"] != without["base"])


if __name__ == "__main__":
    test_engine_unit()
    test_probs_shift()
    test_agent_path()
    print("\nALL FORM CHECKS PASSED")
