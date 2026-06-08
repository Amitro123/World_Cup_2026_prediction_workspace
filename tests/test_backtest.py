"""Tests for the backtest / calibration module (src/backtest.py)."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import backtest, engine  # noqa: E402


def _toy_df():
    """Two lopsided games + one even game, neutral venue."""
    return pd.DataFrame([
        # strong home, big win -> model should be confident & right
        {"rating_home": 1850, "rating_away": 1400, "home_goals": 3, "away_goals": 0,
         "neutral": 1, "stage": "group"},
        # strong away, away win
        {"rating_home": 1400, "rating_away": 1850, "home_goals": 0, "away_goals": 2,
         "neutral": 1, "stage": "group"},
        # even teams, draw
        {"rating_home": 1600, "rating_away": 1600, "home_goals": 1, "away_goals": 1,
         "neutral": 1, "stage": "group"},
    ])


def test_predict_row_sums_to_one():
    row = _toy_df().iloc[0]
    p = backtest.predict_row(row, engine.ProbabilityModel())
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_metrics_in_valid_ranges():
    m = backtest.evaluate(_toy_df())
    assert m.n == 3
    assert 0.0 <= m.brier <= 2.0            # 3-class Brier upper bound
    assert m.log_loss >= 0.0
    assert 0.0 <= m.accuracy <= 1.0
    # by-class contributions sum to the total Brier
    assert abs(sum(m.by_class_brier.values()) - m.brier) < 1e-9


def test_model_beats_uniform_on_toy():
    df = _toy_df()
    m = backtest.evaluate(df)
    base = backtest.baselines(df)
    # an informed model should not be worse than blind 1/3 guessing here
    assert m.brier <= base["uniform"].brier


def test_overrides_are_restored():
    before = engine.K
    backtest.evaluate(_toy_df(), overrides={"K": 999})
    assert engine.K == before  # constant restored after the call


def test_override_must_be_tunable():
    try:
        backtest.evaluate(_toy_df(), overrides={"NOT_A_CONST": 1})
    except KeyError:
        return
    raise AssertionError("expected KeyError for non-tunable constant")


def test_sweep_shape():
    rows = backtest.sweep(_toy_df(), "K", [180, 200, 220])
    assert len(rows) == 3
    assert all("brier" in r and r["K"] in (180, 200, 220) for r in rows)


def test_calibration_bins_well_formed():
    table = backtest.calibration_table(_toy_df(), bins=10)
    for b in table:
        assert b["n"] > 0
        assert 0.0 <= b["mean_pred"] <= 1.0
        assert 0.0 <= b["observed"] <= 1.0


def test_per_match_brier_matches_evaluate():
    df = _toy_df()
    mb, ub = backtest.per_match_brier(df)
    assert len(mb) == len(ub) == len(df)
    # mean of per-match model Brier == evaluate().brier
    assert abs(sum(mb) / len(mb) - backtest.evaluate(df).brier) < 1e-9
    # uniform per-match Brier is constant 3*(1/3 - y)^2 summed = 2/3
    assert all(abs(u - 2 / 3) < 1e-9 for u in ub)


def test_skill_ci_brackets_point_and_is_ordered():
    ci = backtest.skill_ci(_toy_df(), n_boot=200, seed=1)
    assert ci["n"] == 3
    assert ci["ci95_lo"] <= ci["skill"] <= ci["ci95_hi"]


def test_skill_ci_deterministic_with_seed():
    a = backtest.skill_ci(_toy_df(), n_boot=200, seed=7)
    b = backtest.skill_ci(_toy_df(), n_boot=200, seed=7)
    assert a == b


def test_grid_fit_finds_a_minimum_and_restores():
    before_k, before_bt = engine.K, engine.BASE_TOTAL
    grid = {"K": [180.0, 240.0, 300.0], "BASE_TOTAL": [2.4, 2.6, 2.8]}
    rep = backtest.grid_fit(_toy_df(), grid)
    assert rep["best_params"]["K"] in grid["K"]
    assert rep["best_params"]["BASE_TOTAL"] in grid["BASE_TOTAL"]
    # the best score is no worse than the default score by construction
    assert rep["best_score"] <= rep["default_score"] + 1e-9
    # constants restored after fitting
    assert engine.K == before_k and engine.BASE_TOTAL == before_bt


def test_cv_fit_pooled_keys_present():
    frames = {"a": _toy_df(), "b": _toy_df()}
    grid = {"K": [200.0, 240.0], "BASE_TOTAL": [2.5, 2.6]}
    rep = backtest.cv_fit(frames, grid)
    assert len(rep["folds"]) == 2
    assert rep["n"] == 6
    for key in ("pooled_fitted", "pooled_default", "pooled_improvement"):
        assert key in rep


def test_fit_report_real_data_keeps_constants_unfitted_expert():
    """The CV fit must run on real holdouts and flag EXPERT_W as unfittable."""
    rep = backtest.fit_report(grid={"K": [200.0, 240.0], "BASE_TOTAL": [2.6, 2.9]})
    assert "cross_validation" in rep and "final_fit_all_data" in rep
    assert rep["current"]["K"] == engine.K
    assert "EXPERT_W" in rep["expert_w_note"]


def test_real_2022_report_beats_baselines():
    """The shipped 2022 dataset: the model must show positive skill."""
    rep = backtest.run()
    assert rep["n"] == 64
    assert rep["skill_vs_uniform"] > 0       # better than blind guessing
    assert rep["skill_vs_base_rate"] > 0     # better than knowing only base rates
    assert rep["model"]["brier"] < rep["baselines"]["uniform"]["brier"]


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
