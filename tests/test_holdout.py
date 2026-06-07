"""Tests for the multi-tournament holdout harness (src/backtest.py).

These use tiny synthetic frames so they are fast, deterministic, and do not
depend on any shipped CSV. They lock in three behaviours the CR cares about:

  * config_compare always tests fifa_only and *only* tests a signal config when
    its column(s) are actually present (no silent duplication of the baseline);
  * the per-match signal columns really flow into the prediction (a positive
    h2h_sup must push probability toward the home team);
  * holdout pools several tournaments and reports per-tournament + pooled.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import backtest, engine  # noqa: E402


def _frame(with_signals: bool = False, with_elo: bool = True) -> pd.DataFrame:
    rows = [
        # strong home vs weak away, home won
        {"date": "2024-01-01", "home": "AAA", "away": "BBB",
         "rating_home": 1800, "rating_away": 1400,
         "home_goals": 2, "away_goals": 0, "neutral": 1, "stage": "group"},
        # even teams, draw
        {"date": "2024-01-02", "home": "CCC", "away": "DDD",
         "rating_home": 1550, "rating_away": 1545,
         "home_goals": 1, "away_goals": 1, "neutral": 1, "stage": "group"},
        # weak home vs strong away, away won
        {"date": "2024-01-03", "home": "EEE", "away": "FFF",
         "rating_home": 1450, "rating_away": 1750,
         "home_goals": 0, "away_goals": 3, "neutral": 1, "stage": "group"},
    ]
    df = pd.DataFrame(rows)
    if with_elo:
        df["elo_home"] = df["rating_home"] + 200
        df["elo_away"] = df["rating_away"] + 200
    if with_signals:
        df["h2h_sup"] = [0.3, 0.0, -0.3]
        df["form_sup"] = [0.2, 0.0, -0.2]
    return df


# --- predict_row honors per-match signals ------------------------------------

def test_h2h_signal_shifts_probability_toward_home():
    df = _frame(with_signals=True)
    model = engine.ProbabilityModel()
    row = df.iloc[0]
    base = backtest.predict_row(row, model, config={})
    boosted = backtest.predict_row(row, model, config={"use_h2h": True})
    # positive h2h_sup (0.3, home POV) must raise P(home win)
    assert boosted["H"] > base["H"]
    assert abs(sum(boosted.values()) - 1.0) < 1e-9


def test_signals_off_by_default_reproduces_pure_fifa():
    df = _frame(with_signals=True)
    model = engine.ProbabilityModel()
    row = df.iloc[0]
    # config=None must ignore the h2h_sup/form_sup columns entirely
    assert backtest.predict_row(row, model, config=None) == \
        backtest.predict_row(row, model, config={})


def test_row_get_treats_nan_as_absent():
    df = pd.DataFrame([{"a": 1, "b": float("nan")}])
    row = df.iloc[0]
    assert backtest._row_get(row, "a") == 1
    assert backtest._row_get(row, "b", "fallback") == "fallback"
    assert backtest._row_get(row, "missing", 7) == 7


# --- config_compare gates on column presence ---------------------------------

def test_config_compare_skips_absent_signal_columns():
    df = _frame(with_signals=False, with_elo=False)  # only FIFA
    rows = backtest.config_compare(df)
    names = {r["config"] for r in rows}
    assert names == {"fifa_only"}  # nothing else is testable here


def test_config_compare_includes_signals_when_present():
    df = _frame(with_signals=True, with_elo=True)
    rows = backtest.config_compare(df)
    names = {r["config"] for r in rows}
    # h2h/form/elo all have columns; expert never does -> excluded
    assert {"fifa_only", "+h2h", "+form", "+elo", "all"} <= names
    assert "+expert" not in names
    # sorted ascending by Brier
    briers = [r["brier"] for r in rows]
    assert briers == sorted(briers)


# --- holdout pools tournaments -----------------------------------------------

def test_holdout_pools_multiple_tournaments(tmp_path):
    a = tmp_path / "backtest_alpha.csv"
    b = tmp_path / "backtest_beta.csv"
    _frame().to_csv(a, index=False)
    _frame().to_csv(b, index=False)
    rep = backtest.holdout({"alpha": str(a), "beta": str(b)})
    assert set(rep["tournaments"]) == {"alpha", "beta"}
    assert rep["pooled"]["n"] == 6  # 3 + 3
    assert rep["tournaments"]["alpha"]["n"] == 3
    assert "configs" in rep["pooled"]
    assert "calibration" in rep["pooled"]


def test_holdout_reports_error_when_no_csvs():
    rep = backtest.holdout({"nope": "/does/not/exist.csv"})
    assert "error" in rep


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            import inspect
            if "tmp_path" in inspect.signature(fn).parameters:
                import tempfile
                with tempfile.TemporaryDirectory() as d:
                    import pathlib
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
