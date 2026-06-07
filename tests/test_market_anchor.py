"""Integration tests for DataStore market anchor + player props (synthetic data)."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import DataStore  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def _store():
    return DataStore.load(DATA)


def test_market_anchor_dormant_without_data():
    ds = _store()
    ds.market_odds = pd.DataFrame()  # ensure empty
    mid = str(ds.matches.iloc[0]["match_id"])
    assert ds.market_for(mid) is None
    assert ds.market_anchor(mid) is None
    assert ds.market_anchors() == []


def test_market_anchor_with_decimal_odds():
    ds = _store()
    mid = str(ds.matches.iloc[0]["match_id"])
    ds.market_odds = pd.DataFrame([{
        "match_id": mid, "dec_home": 2.1, "dec_draw": 3.4, "dec_away": 3.6,
    }])
    a = ds.market_anchor(mid)
    assert a is not None
    assert abs(sum(a["market"].values()) - 1.0) < 1e-3  # values rounded to 4dp
    assert set(a["model"]) == {"p_home", "p_draw", "p_away"}
    assert isinstance(a["flag"], bool)
    assert a["pick_market"] == "p_home"  # shortest odds -> market favourite


def test_market_anchors_sorted_flagged_first():
    ds = _store()
    ids = [str(x) for x in ds.matches["match_id"].head(2)]
    ds.market_odds = pd.DataFrame([
        # near-agreement (no flag, small gap)
        {"match_id": ids[0], "dec_home": 3.0, "dec_draw": 3.0, "dec_away": 3.0},
        # lopsided (likely flagged)
        {"match_id": ids[1], "dec_home": 1.2, "dec_draw": 7.0, "dec_away": 12.0},
    ])
    anchors = ds.market_anchors()
    assert len(anchors) == 2
    # flagged anchors sort first
    assert anchors[0]["flag"] or not anchors[1]["flag"]


def test_player_props_model_only():
    ds = _store()
    teams_with = set(ds.players.team_id)
    row = next(r for _, r in ds.matches.iterrows()
               if r.home_id in teams_with and r.away_id in teams_with)
    props = ds.player_props(str(row.match_id))
    assert props, "expected player props for a match with squad data"
    p = props[0]
    assert 0 <= p["model"]["p_score"] <= 1
    assert 0 <= p["model"]["p_score_or_assist"] <= 1
    # sorted descending by score-or-assist
    vals = [x["model"]["p_score_or_assist"] for x in props]
    assert vals == sorted(vals, reverse=True)
    assert "market" not in p  # no market file row injected


def test_player_props_merges_market():
    ds = _store()
    teams_with = set(ds.players.team_id)
    row = next(r for _, r in ds.matches.iterrows()
               if r.home_id in teams_with and r.away_id in teams_with)
    mid = str(row.match_id)
    star = ds.player_props(mid)[0]
    ds.players_market = pd.DataFrame([{
        "match_id": mid, "team_id": star["team_id"],
        "name_en": star["name_en"], "name_he": "",
        "score_odds": 2.5, "assist_odds": 4.0, "score_or_assist_odds": 1.8,
    }])
    merged = next(p for p in ds.player_props(mid)
                  if p["name_en"] == star["name_en"])
    assert "market" in merged
    assert merged["market"]["p_score"] is not None
    assert "compare" in merged
    assert "p_score" in merged["compare"]


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
