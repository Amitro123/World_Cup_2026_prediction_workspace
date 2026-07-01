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


# --- bracket-geometry helpers (deterministic, no simulation) -----------------
def test_winner_r32_covers_all_12_groups():
    assert set(knockout._WINNER_R32) == set("ABCDEFGHIJKL")


def test_path_to_final_ends_at_root():
    for m in range(73, 89):
        path = knockout._path_to_final(m)
        assert path[0] == m
        assert path[-1] == 104          # every R32 path climbs to the final
        assert len(set(path)) == len(path)  # no repeats


def test_quarter_half_partitions_the_draw():
    """Each group winner sits in exactly one of 4 quarters / 2 halves, and the
    quarters nest correctly inside the halves."""
    seen = {}
    for g, m in knockout._WINNER_R32.items():
        q, half = knockout._quarter_half(m)
        assert q in (1, 2, 3, 4)
        assert half in ("עליון", "תחתון")
        assert (half == "עליון") == (q in (1, 2))  # Q1/Q2 top, Q3/Q4 bottom
        seen[g] = (q, half)
    # all four quarters are actually used by some group winner
    assert {q for q, _ in seen.values()} == {1, 2, 3, 4}


def test_meet_stage_is_symmetric_and_known():
    a, b = knockout._WINNER_R32["I"], knockout._WINNER_R32["H"]
    assert knockout._meet_stage(a, b) == knockout._meet_stage(b, a)
    # France(I) and Spain(H) are both top-half group winners -> meet in the SF
    assert knockout._meet_stage(a, b) == "חצי גמר"
    # France(I) and Germany(E) feed the same R16 match -> meet in the 1/8
    assert knockout._meet_stage(a, knockout._WINNER_R32["E"]) == "1/8"


def test_meet_stage_same_match_never_final():
    """Any two distinct group winners must collide at or before the final."""
    valid = {"1/16", "1/8", "רבע גמר", "חצי גמר", "גמר"}
    ms = list(knockout._WINNER_R32.values())
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            assert knockout._meet_stage(ms[i], ms[j]) in valid


def test_draw_difficulty_shape():
    from src.models import DataStore

    ds = DataStore.load(os.path.join(os.path.dirname(__file__), "..", "data"))
    d = knockout.draw_difficulty(ds, n=400, seed=1)
    assert len(d["groups"]) == 12                 # one row per group
    assert d["groups"] == sorted(                 # sorted by group title equity
        d["groups"], key=lambda r: r["group_title"], reverse=True
    )
    for row in d["groups"]:
        assert 0.0 <= row["avg_qualify"] <= 100.0
        assert row["quarter"] in (1, 2, 3, 4)
    # collisions only flag pre-final meetings among the strongest teams
    assert all(c["stage"] != "גמר" for c in d["collisions"])
    names = {r["top_team"] for r in d["groups"]}
    for c in d["collisions"]:
        assert c["team_a"] in names and c["team_b"] in names


# --- knockout news_adjustments interface (stable match_id, rating_delta) ----

def _inject_news_row(ds, match_id, team_id, kind, value):
    """Add a news row directly in memory, bypassing add_news_adjustment's disk
    write so the test never touches the real data/news_adjustments.csv."""
    import pandas as pd

    from src.models import NEWS_COLUMNS

    row = pd.DataFrame([{
        "adj_id": "test", "match_id": match_id, "team_id": team_id,
        "kind": kind, "value": value, "note_he": "test", "source": "unit-test",
        "created_at": "", "active": 1,
    }], columns=NEWS_COLUMNS)
    ds.news = row if ds.news.empty else pd.concat([ds.news, row], ignore_index=True)


def test_match_id_for_covers_every_simulated_match_no():
    assert knockout.match_id_for(73) == "M73"
    assert knockout.match_id_for(104) == "M104"
    # every R32/TREE match number gets a ("M"-prefixed) stable id; M103 (the
    # third-place play-off) is deliberately not simulated and so not included.
    assert 103 not in knockout.ALL_MATCH_NOS
    assert set(knockout.R32) <= set(knockout.ALL_MATCH_NOS)
    assert set(knockout.TREE) <= set(knockout.ALL_MATCH_NOS)


def _cleared_news(ds):
    """ds with news wiped in memory -- shipped news_adjustments.csv now carries
    real knockout entries (e.g. M82, from live pre-match briefings), so these
    tests must not assume it's empty."""
    from src.models import NEWS_COLUMNS
    import pandas as pd

    ds.news = pd.DataFrame(columns=NEWS_COLUMNS)
    return ds


def test_build_knockout_news_empty_by_default():
    from src.models import DataStore

    ds = _cleared_news(DataStore.load(os.path.join(os.path.dirname(__file__), "..", "data")))
    assert knockout.build_knockout_news(ds) == {}


def test_build_knockout_news_reads_rating_delta_by_match_id():
    from src.models import DataStore

    ds = _cleared_news(DataStore.load(os.path.join(os.path.dirname(__file__), "..", "data")))
    _inject_news_row(ds, "M78", "BEL", "rating_delta", -40.0)
    _inject_news_row(ds, "M78", "BEL", "rating_delta", -10.0)  # sums per team
    _inject_news_row(ds, "M78", "SEN", "lambda_mult", 0.8)     # not wired -> ignored
    kn = knockout.build_knockout_news(ds)
    assert kn == {78: {"BEL": -50.0}}


def test_knockout_rating_delta_shifts_run():
    """A rating_delta news adjustment on a real R32 match_id must change the
    affected team's reach-probability in knockout.run() — the concrete gap
    reported in the 2026-07 session: knockout ties had no stable match_id for
    news_adjustments.csv to attach to."""
    from src.models import DataStore

    ds = DataStore.load(os.path.join(os.path.dirname(__file__), "..", "data"))
    ctx = knockout._prepare(ds)
    pos, third_assign, _ = knockout._group_phase(ctx, random.Random(7))
    r32 = knockout._resolve_r32(pos, third_assign)
    match_no, (team_a, _team_b) = next(iter(r32.items()))
    match_id = knockout.match_id_for(match_no)

    baseline = knockout.run(ds, n=6000, seed=7).set_index("team_id")["r16_%"]

    _inject_news_row(ds, match_id, team_a, "rating_delta", -300.0)
    adjusted = knockout.run(ds, n=6000, seed=7).set_index("team_id")["r16_%"]

    # A crippling delta on one side of an R32 tie must tank that team's odds
    # of reaching the R16 by a wide margin -- well outside Monte-Carlo noise.
    assert adjusted[team_a] < baseline[team_a] - 15.0


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
