"""
בדיקת מפגשי עבר — proof that the head-to-head signal works AND that the
agent-facing path (pre_match_probs / match_briefing) actually uses it.

Run directly:
    python tests/test_h2h.py
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
    print("[1] engine.h2h_supremacy — pure logic")

    # direction: home won the past meetings -> positive supremacy
    won = engine.h2h_supremacy([{"gd": 2, "comp": "competitive", "year": 2022}], ref_year=2026)
    lost = engine.h2h_supremacy([{"gd": -2, "comp": "competitive", "year": 2022}], ref_year=2026)
    _ok("winner gets positive, loser negative, symmetric", won > 0 > lost and abs(won + lost) < 1e-9)

    # friendlies count less than competitive
    comp = engine.h2h_supremacy([{"gd": 2, "comp": "competitive", "year": 2024}], ref_year=2026)
    frnd = engine.h2h_supremacy([{"gd": 2, "comp": "friendly", "year": 2024}], ref_year=2026)
    _ok("a friendly moves the line less than a competitive game", 0 < frnd < comp)

    # stage gradation: the same win means more the bigger the occasion
    def one(stage):
        return engine.h2h_supremacy([{"gd": 2, "comp": stage, "year": 2024}], ref_year=2026)
    fr, gp, ko, sf, fn = one("friendly"), one("group"), one("knockout"), one("semifinal"), one("final")
    _ok("status is graded: friendly < group < knockout < semifinal < final",
        fr < gp < ko < sf < fn)

    # free-text stage labels (from the web scraper) map by keyword
    _ok("'World Cup semi-final' is read as a semifinal",
        one("World Cup semi-final") == one("semifinal"))
    _ok("'Euro qualifier' is read as a qualifier",
        one("Euro qualifier") == one("qualifier"))

    # recency: an old win counts less than a recent one
    recent = engine.h2h_supremacy([{"gd": 2, "comp": "competitive", "year": 2024}], ref_year=2026)
    old = engine.h2h_supremacy([{"gd": 2, "comp": "competitive", "year": 2006}], ref_year=2026)
    _ok("older meetings decay", 0 < old < recent)

    # shrinkage: 5 identical wins move more than 1 (small samples shrink to 0)
    one = engine.h2h_supremacy([{"gd": 2, "comp": "competitive", "year": 2024}], ref_year=2026)
    five = engine.h2h_supremacy(
        [{"gd": 2, "comp": "competitive", "year": 2024}] * 5, ref_year=2026
    )
    _ok("more games = stronger signal (sample-size shrinkage)", one < five)

    # cap: a 10-0 demolition is still bounded
    huge = engine.h2h_supremacy([{"gd": 10, "comp": "competitive", "year": 2025}] * 10, ref_year=2026)
    _ok("contribution is capped at H2H_CAP", huge <= engine.H2H_CAP + 1e-9)

    # no data -> no effect (teams that never met must not move the model at all)
    _ok("empty history = zero", engine.h2h_supremacy([]) == 0.0)
    rh = ra = 1700.0
    untouched = engine.expected_goals(rh, ra, h2h_sup=engine.h2h_supremacy([]))
    baseline = engine.expected_goals(rh, ra)
    _ok("never-met pair leaves expected goals unchanged", untouched == baseline)


def test_probs_shift():
    print("[2] win probability actually moves")
    rh, ra = 1800.0, 1800.0  # identical strength, neutral -> isolate the H2H effect
    base = engine.ProbabilityModel().pre_match(rh, ra, neutral=True)
    favoured = engine.ProbabilityModel().pre_match(rh, ra, neutral=True, h2h_sup=0.3)
    print(f"     even teams, no H2H : p_home={base['p_home']:.3f}")
    print(f"     even teams, +H2H   : p_home={favoured['p_home']:.3f}")
    _ok("a positive H2H raises the home win probability", favoured["p_home"] > base["p_home"])


def test_agent_path():
    print("[3] the AGENT path (match_briefing) reflects H2H")
    ds = DataStore.load(DATA)

    # find a real fixture whose H2H actually moves the model (a balanced record,
    # e.g. a single draw, legitimately nets to zero supremacy and leaves the
    # briefing unchanged — so require a non-zero net effect, not merely a meeting)
    target = None
    for m in ds.matches.itertuples():
        if (ds.h2h_meetings(m.home_id, m.away_id)
                and abs(ds.h2h_supremacy_for(m.home_id, m.away_id)) > 1e-9):
            target = m
            break

    if target is None:
        # No group fixture has history yet -> simulate one so the test still proves the path.
        m0 = ds.matches.iloc[0]
        h, a = m0.home_id, m0.away_id
        ds.h2h = pd.DataFrame(
            [{"team_a": h, "team_b": a, "a_goals": 4, "b_goals": 0,
              "comp": "competitive", "year": 2025}]
        )
        target = m0
        print(f"     (no seeded fixture had history; injected {h} 4-0 {a} for the test)")

    mid = target.match_id
    with_h2h = ds.match_briefing(mid)

    # now wipe history and recompute -> the briefing must change
    ds.h2h = ds.h2h.iloc[0:0]
    without = ds.match_briefing(mid)

    print(f"     match {mid}: with H2H base={with_h2h['base']}")
    print(f"     match {mid}: no   H2H base={without['base']}")
    _ok("match_briefing output changes when H2H is present vs absent",
        with_h2h["base"] != without["base"])


if __name__ == "__main__":
    test_engine_unit()
    test_probs_shift()
    test_agent_path()
    print("\nALL H2H CHECKS PASSED")
