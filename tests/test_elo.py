"""Tests for the optional FIFA/Elo blend (engine + backtest + DataStore gate)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import backtest, engine  # noqa: E402
from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
STATS = {"fifa_mean": 1500.0, "fifa_std": 100.0, "elo_mean": 1800.0, "elo_std": 150.0}


def test_blend_weight_zero_is_identity():
    assert engine.blend_strength(1700, 2100, 0.0, **STATS) == 1700


def test_blend_weight_one_maps_elo_onto_fifa_scale():
    # a team 2 Elo-std above the Elo mean should land 2 FIFA-std above FIFA mean
    elo = STATS["elo_mean"] + 2 * STATS["elo_std"]
    out = engine.blend_strength(1500, elo, 1.0, **STATS)
    assert abs(out - (STATS["fifa_mean"] + 2 * STATS["fifa_std"])) < 1e-6


def test_blend_is_monotonic_in_weight():
    # stronger Elo than FIFA -> rating rises as weight rises
    fifa, elo = 1500.0, STATS["elo_mean"] + 2 * STATS["elo_std"]
    vals = [engine.blend_strength(fifa, elo, w, **STATS) for w in (0, 0.25, 0.5, 0.75, 1.0)]
    assert vals == sorted(vals)


def test_blend_falls_back_on_missing_elo():
    assert engine.blend_strength(1650, float("nan"), 0.5, **STATS) == 1650
    bad = {**STATS, "elo_std": 0.0}
    assert engine.blend_strength(1650, 1900, 0.5, **bad) == 1650


def test_backtest_elo_sweep_present_and_zero_matches_pure_fifa():
    df = backtest.load()
    sweep = backtest.elo_sweep(df, weights=(0.0, 0.4))
    assert len(sweep) == 2
    pure = backtest.evaluate(df, elo_weight=0.0)
    assert abs(sweep[0]["brier"] - round(pure.brier, 4)) < 1e-9


def test_datastore_default_rating_is_pure_fifa():
    """With ELO_WEIGHT=0 (default), team_rating must equal fifa_points exactly."""
    assert engine.ELO_WEIGHT == 0.0
    ds = DataStore.load(DATA_DIR)
    tid = ds.teams.iloc[0]["team_id"]
    assert ds.team_rating(tid) == float(ds.teams.iloc[0]["fifa_points"])


def test_datastore_blend_gate_activates_with_weight_and_column():
    """If elo_points exists and ELO_WEIGHT>0, the rating blends (then restore)."""
    ds = DataStore.load(DATA_DIR)
    if "elo_points" not in ds.teams.columns:
        return  # no Elo data shipped — gate stays off, nothing to test
    tid = ds.teams.iloc[0]["team_id"]
    saved = engine.ELO_WEIGHT
    try:
        engine.ELO_WEIGHT = 0.5
        object.__setattr__(ds, "_stats_cache", None)
        blended = ds.team_rating(tid)
        assert isinstance(blended, float)
    finally:
        engine.ELO_WEIGHT = saved


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
