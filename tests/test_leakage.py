"""Tests for the backtest leakage guard (src.backtest.leakage_check).

A holdout is only honest if every rating/signal in its CSV was knowable BEFORE
the tournament started. These tests assert (a) the shipped manifest passes the
check, and (b) the check actually catches the three failure modes — a future
as-of date (real leakage), a missing manifest entry, and a manifest whose
tournament_start disagrees with the CSV.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import backtest  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


# ---------------------------------------------------------------------------
# The shipped repo must be clean.
# ---------------------------------------------------------------------------

def test_shipped_holdouts_are_leakage_free():
    rep = backtest.leakage_check()
    assert rep["ok"], f"leakage violations in shipped data: {rep['violations']}"
    assert rep["violations"] == []


def test_every_discovered_csv_has_a_manifest_entry():
    sources = backtest._discover_sources()
    rep = backtest.leakage_check()
    for label in sources:
        assert label in rep["tournaments"], f"{label} missing from leakage report"
        assert rep["tournaments"][label]["ok"]


def test_asof_is_on_or_before_tournament_start_for_all():
    rep = backtest.leakage_check()
    for label, t in rep["tournaments"].items():
        assert t["ratings_asof"] <= t["tournament_start"], label
        # And the manifest must describe the actual data it ships with.
        assert t["tournament_start"] == t["data_start"], label


# ---------------------------------------------------------------------------
# The guard must catch each failure mode (use a tmp manifest, real CSVs).
# ---------------------------------------------------------------------------

def _write_meta(tmp_path, payload) -> str:
    p = os.path.join(tmp_path, "backtest_meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return p


def test_detects_future_asof_as_leakage(tmp_path):
    sources = {"2022": os.path.join(DATA, "backtest_2022.csv")}
    # ratings dated AFTER the first match = leakage.
    meta = _write_meta(tmp_path, {
        "2022": {"tournament_start": "2022-11-20", "ratings_asof": "2022-12-01"}
    })
    rep = backtest.leakage_check(sources=sources, meta_path=meta)
    assert not rep["ok"]
    assert any("LEAKAGE" in v for v in rep["violations"])


def test_detects_missing_manifest_entry(tmp_path):
    sources = {"2022": os.path.join(DATA, "backtest_2022.csv")}
    meta = _write_meta(tmp_path, {})  # no entry for 2022
    rep = backtest.leakage_check(sources=sources, meta_path=meta)
    assert not rep["ok"]
    assert any("no entry" in v for v in rep["violations"])


def test_detects_manifest_start_mismatch(tmp_path):
    sources = {"2022": os.path.join(DATA, "backtest_2022.csv")}
    # asof <= start (no leakage) but start disagrees with the CSV's first match.
    meta = _write_meta(tmp_path, {
        "2022": {"tournament_start": "2022-11-01", "ratings_asof": "2022-10-06"}
    })
    rep = backtest.leakage_check(sources=sources, meta_path=meta)
    assert not rep["ok"]
    assert any("!=" in v for v in rep["violations"])


def test_equal_asof_and_start_is_allowed(tmp_path):
    sources = {"2022": os.path.join(DATA, "backtest_2022.csv")}
    meta = _write_meta(tmp_path, {
        "2022": {"tournament_start": "2022-11-20", "ratings_asof": "2022-11-20"}
    })
    rep = backtest.leakage_check(sources=sources, meta_path=meta)
    assert rep["ok"], rep["violations"]


# ---------------------------------------------------------------------------
# The holdout report embeds the leakage block.
# ---------------------------------------------------------------------------

def test_holdout_report_includes_leakage_block():
    rep = backtest.holdout()
    assert "leakage" in rep
    assert rep["leakage"]["ok"]


if __name__ == "__main__":
    import tempfile
    import traceback

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    fn(td)
            else:
                fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
