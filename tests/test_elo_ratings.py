"""Tests for src/elo.py — derived World Football Elo (pure, no network)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import elo  # noqa: E402

# --- expected score ----------------------------------------------------------

def test_equal_ratings_neutral_is_coin_flip():
    assert abs(elo.expected_home(1500, 1500, neutral=True) - 0.5) < 1e-9


def test_home_advantage_lifts_expectation():
    assert elo.expected_home(1500, 1500, neutral=False) > 0.5


def test_stronger_team_expected_higher():
    assert elo.expected_home(1800, 1500, neutral=True) > 0.5


# --- goal-difference multiplier ----------------------------------------------

def test_gd_multiplier_steps():
    assert elo._gd_multiplier(0) == 1.0
    assert elo._gd_multiplier(1) == 1.0
    assert elo._gd_multiplier(2) == 1.5
    assert elo._gd_multiplier(3) == (11 + 3) / 8
    assert elo._gd_multiplier(-4) == (11 + 4) / 8  # uses absolute value


# --- single update is zero-sum and direction-correct -------------------------

def test_update_is_zero_sum_and_winner_gains():
    rh, ra = elo.update_pair(1500, 1500, 2, 0, weight=40, neutral=True)
    assert rh > 1500 and ra < 1500
    assert abs((rh - 1500) + (ra - 1500)) < 1e-9  # points conserved


def test_bigger_win_moves_more():
    small = elo.update_pair(1500, 1500, 1, 0, weight=40, neutral=True)[0]
    big = elo.update_pair(1500, 1500, 4, 0, weight=40, neutral=True)[0]
    assert big > small


def test_draw_against_stronger_team_gains_points():
    # underdog (1400) draws favourite (1700) -> underdog should rise
    rh, ra = elo.update_pair(1400, 1700, 1, 1, weight=40, neutral=True)
    assert rh > 1400 and ra < 1700


# --- run() chronology + cutoff -----------------------------------------------

def _matches():
    return [
        {"date": "2023-06-01", "home": "A", "away": "B", "gh": 3, "ga": 0,
         "weight": 40, "neutral": True},
        {"date": "2023-09-01", "home": "B", "away": "C", "gh": 1, "ga": 1,
         "weight": 40, "neutral": True},
        {"date": "2024-06-01", "home": "A", "away": "C", "gh": 0, "ga": 2,
         "weight": 60, "neutral": True},  # tournament match — must be excluded
    ]


def test_run_processes_all_and_seeds_new_teams():
    r = elo.run(_matches())
    assert set(r) == {"A", "B", "C"}
    # A won big early, lost later; net direction not asserted, just presence
    assert all(isinstance(v, float) for v in r.values())


def test_until_excludes_on_or_after_date():
    before = elo.snapshot_before(_matches(), "2024-06-01")
    # the 2024-06-01 game must not have been consumed: A only played the 3-0 win
    # (gained) and nothing on/after the cutoff.
    assert before["A"] > elo.START
    # C had only the draw vs B (vs a team that had lost) -> exists, near start
    assert "C" in before


def test_unordered_input_is_sorted_by_date():
    ordered = elo.run(_matches())
    shuffled = elo.run(list(reversed(_matches())))
    assert ordered == shuffled


# --- recenter preserves gaps -------------------------------------------------

def test_recenter_sets_mean_and_preserves_gaps():
    r = {"A": 1900, "B": 1700, "C": 1500}
    out = elo.recenter(r, mean=1500)
    assert abs(sum(out.values()) / 3 - 1500) < 1e-9
    # every pairwise gap is identical after the pure shift
    assert abs((out["A"] - out["B"]) - (r["A"] - r["B"])) < 1e-9
    assert abs((out["A"] - out["C"]) - (r["A"] - r["C"])) < 1e-9


def test_recenter_over_subset_of_teams():
    r = {"A": 2000, "B": 1000, "X": 1500}
    out = elo.recenter(r, teams=["A", "B"], mean=1500)
    # mean of A,B becomes 1500; X shifted by the same amount
    assert abs((out["A"] + out["B"]) / 2 - 1500) < 1e-9


# --- weight inference --------------------------------------------------------

def test_weight_for_distinguishes_tournaments():
    assert elo.weight_for(league="FIFA World Cup") == 60
    assert elo.weight_for(comp="group", league="UEFA Euro") == 50
    assert elo.weight_for(league="Friendlies") == 20
    assert elo.weight_for(league="World Cup - Qualification") == 40
    assert elo.weight_for(comp="qualifier") == 40


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
