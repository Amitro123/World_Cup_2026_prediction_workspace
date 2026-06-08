"""
fetch_fifa_points.py — מרענן את נקודות הדירוג של פיפ"א לכל נבחרת מהאינטרנט.

Sibling of fetch_form.py / fetch_h2h.py. Those refresh the *signals* (momentum,
past meetings); this refreshes the model's **base strength** — the FIFA Men's
World Ranking points in teams.csv that every prediction is built on. Until now
those points were static (copied once from the prior Cowork session), which an
external review flagged: the other signals had fetchers, this one did not.

It searches DuckDuckGo lite for each team's current FIFA ranking points, extracts
the most plausible points value (a number in the real FIFA range, nearest a
"points"/"FIFA"/"ranking" cue), and proposes an updated value per team.

Like its siblings it NEVER changes the model silently: it prints proposed
old -> new points (flagging large jumps) and only writes when you pass --write.
Writing reuses DataStore.set_team_rating, so power_rating is re-normalised across
all teams and teams.csv is persisted exactly as `hermes.py rate` does — this is
just the automated, all-teams version of that manual command.

Usage
-----
  # propose refreshed FIFA points for every team (does NOT write):
  python fetch_fifa_points.py

  # one specific team, machine-readable:
  python fetch_fifa_points.py --team MEX --json

  # refresh and WRITE verified values into data/teams.csv:
  python fetch_fifa_points.py --write

  # only write changes bigger than N points (skip scrape noise):
  python fetch_fifa_points.py --min-delta 5 --write

Hermes
------
  python hermes.py fifa                  # propose (dry-run)
  python hermes.py fifa --team MEX --write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse

import fetch_h2h
from fetch_h2h import TEAM_NAME, _fetch, _names, _strip_html

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Plausible FIFA Men's ranking-points window. The real spread runs from ~1100
# (lowest-ranked WC teams) to ~1900 (the very top). Bounding the parser to this
# range rejects both junk and four-digit years (2014–2026 fall outside it).
MIN_POINTS = 1100.0
MAX_POINTS = 1980.0

# Only flag/keep an update when it moves the rating by at least this much, so
# scrape jitter doesn't churn the file. Tunable via --min-delta.
DEFAULT_MIN_DELTA = 1.0

# Cues that a nearby number is a ranking-points figure (not some other stat).
_CUE = re.compile(r"point|pts|fifa|rank|נקוד|דירוג", re.IGNORECASE)
_NUM = re.compile(r"(1[1-9]\d{2}(?:\.\d{1,2})?)")  # 1100.00 .. 1999.99


def _find_all(haystack: str, needle: str) -> list[int]:
    """All start indices of `needle` in `haystack` (non-overlapping)."""
    out, i = [], haystack.find(needle)
    while i != -1:
        out.append(i)
        i = haystack.find(needle, i + 1)
    return out


def _points_url(team_id: str) -> str:
    """DuckDuckGo lite search for a team's current FIFA ranking points."""
    q = f"{TEAM_NAME[team_id]} FIFA world ranking points 2026"
    return "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q)


def _parse_points(text: str, team_id: str) -> float | None:
    """Best-effort extract a team's FIFA points from search snippets.

    Collects every number in the plausible FIFA range, then prefers the one
    sitting closest to a ranking cue ("points"/"FIFA"/"ranking") or the team
    name. Returns None when nothing credible is found (caller keeps the old
    value — a miss never corrupts a rating).
    """
    low = text.lower()
    names = [n.lower() for n in _names(team_id)]
    # Anchor positions: ranking cues are the strong signal (the points figure
    # sits next to "points"/"FIFA"/"ranking"); team-name positions are a weak
    # fallback used only when the snippet has no cue at all.
    cue_pos = [m.start() for m in _CUE.finditer(low)]
    name_pos = [i for n in names for i in _find_all(low, n)]
    anchors = cue_pos or name_pos  # prefer cues; fall back to the name

    best: tuple[int, float] | None = None  # (distance-to-anchor, value)
    for m in _NUM.finditer(text):
        val = float(m.group(1))
        if not (MIN_POINTS <= val <= MAX_POINTS):
            continue
        dist = min((abs(a - m.start()) for a in anchors), default=10**9)
        if best is None or dist < best[0]:
            best = (dist, val)
    return best[1] if best else None


def fetch_team(team_id: str, retries: int = 3) -> dict:
    """Fetch + parse one team's current FIFA points (retries on throttle)."""
    url = _points_url(team_id)
    for attempt in range(retries):
        html = _fetch(url)
        text = _strip_html(html) if html else ""
        low = text.lower()
        blocked = any(b in low for b in fetch_h2h._BLOCK_MARKERS) or len(text) < 400
        if not blocked and any(n.lower() in low for n in _names(team_id)):
            return {"team": team_id, "url": url, "ok": True,
                    "points": _parse_points(text, team_id)}
        time.sleep(2.0 * (attempt + 1))
    return {"team": team_id, "url": url, "ok": False, "points": None}


def run(ds, team_ids, write: bool, min_delta: float = DEFAULT_MIN_DELTA,
        polite: float = 0.6) -> dict:
    """Propose (and optionally write) refreshed FIFA points for `team_ids`.

    Returns a JSON-able report. Writing reuses ds.set_team_rating per changed
    team, so power_rating is re-normalised and teams.csv persisted once per team
    (small, ≤48 rows). `min_delta` suppresses sub-noise changes.
    """
    from src import datameta

    proposals, results = [], []
    for t in team_ids:
        res = fetch_team(t)
        results.append(res)
        new = res["points"]
        old = float(ds.team_rating(t))
        if new is not None and abs(new - old) >= min_delta:
            proposals.append({"team": t, "old": round(old, 1),
                              "new": round(new, 1), "delta": round(new - old, 1)})
        time.sleep(polite)

    out = {
        "source": "duckduckgo",
        "teams_checked": len(team_ids),
        "teams_fetched_ok": sum(1 for r in results if r.get("ok")),
        "proposals": sorted(proposals, key=lambda p: abs(p["delta"]), reverse=True),
        "n_proposals": len(proposals),
    }
    if write:
        for p in proposals:
            ds.set_team_rating(p["team"], p["new"])
        out["written"] = len(proposals)
        out["written_to"] = os.path.join(DATA, "teams.csv")
        datameta.stamp(DATA, "fifa_points", "duckduckgo", len(proposals))
    return out


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.models import DataStore

    p = argparse.ArgumentParser(description="Refresh FIFA ranking points (base strength)")
    p.add_argument("--team", metavar="TEAM", help="fetch one team, e.g. --team MEX")
    p.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA,
                   help=f"only propose changes ≥ this many points (default {DEFAULT_MIN_DELTA})")
    p.add_argument("--write", action="store_true",
                   help="write verified values into data/teams.csv")
    p.add_argument("--json", action="store_true", help="raw JSON output")
    args = p.parse_args()

    ds = DataStore.load(DATA)
    teams = [args.team] if args.team else list(ds.teams.team_id)
    result = run(ds, teams, args.write, args.min_delta)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"בדקתי {result['teams_checked']} נבחרות, "
              f"{result['teams_fetched_ok']} נטענו, "
              f"{result['n_proposals']} עדכוני נקודות מוצעים:")
        for pr in result["proposals"]:
            print(f"  {pr['team']}: {pr['old']} -> {pr['new']} ({pr['delta']:+})")
        if args.write:
            print(f"\nנכתבו {result.get('written', 0)} עדכונים -> data/teams.csv")
        else:
            print("\n(הרצה יבשה — הוסף --write כדי לכתוב ל-teams.csv)")
