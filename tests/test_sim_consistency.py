"""Global conservation laws of the tournament Monte-Carlo.

The unit tests check the engine match-by-match; these check the *whole
simulation* obeys the tournament's structure. They would catch bracket-wiring
bugs (wrong slot feeding a match), double-counting, or a team advancing twice —
classes of error no single-match test can see.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import knockout  # noqa: E402
from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

N = 800  # small but enough for the conservation sums' tolerances below


def _run():
    ds = DataStore.load(DATA_DIR)
    return knockout.run(ds, n=N, seed=42)


def test_exactly_one_champion():
    df = _run()
    assert abs(df["title_%"].sum() - 100.0) < 2.0


def test_round_sizes_conserved():
    """48->32->16->8->4->2 teams: each round's reach-% sums to its size*100."""
    df = _run()
    for col, expect in [("qualify_%", 3200), ("r16_%", 1600), ("qf_%", 800),
                        ("sf_%", 400), ("final_%", 200)]:
        assert abs(df[col].sum() - expect) < expect * 0.03, col


def test_per_team_round_monotonicity():
    """No team is more likely to reach a later round than an earlier one."""
    df = _run()
    cols = ["qualify_%", "r16_%", "qf_%", "sf_%", "final_%", "title_%"]
    for _, r in df.iterrows():
        vals = [r[c] for c in cols]
        assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)), \
            f"{r.team_id}: {vals}"


def test_same_seed_reproduces():
    ds = DataStore.load(DATA_DIR)
    a = knockout.run(ds, n=300, seed=7)
    b = knockout.run(ds, n=300, seed=7)
    assert a.equals(b)


def test_every_team_appears_once():
    df = _run()
    assert len(df) == 48
    assert df["team_id"].is_unique
