"""
fetch_form.py — מאחזר מומנטום (recent form) לכל נבחרת מהאינטרנט עבור World Cup 2026.

Sibling of fetch_h2h.py. Where fetch_h2h scrapes the history *between* two teams,
this scrapes a single team's *last handful of matches* — the momentum it carries
into the tournament (a team on a winning streak arrives sharper than one limping
in on losses). It queries DuckDuckGo lite for each team's recent results, reads
the snippets, extracts "<Team> g1-g2 <Opponent>" lines (either ordering),
orients goals to the team, classifies friendly/competitive, recency-filters, and
proposes rows for data/form.csv.

Like fetch_h2h / scout it never changes the model silently — it prints proposed
rows and only writes when you pass --write (Amit / Hermes decides).

The engine consumes form.csv as a small, bounded momentum signal
(see src/engine.form_score / form_supremacy + README §"מומנטום"). Teams with no
recent rows contribute exactly 0 — momentum only ever nudges the line toward
whoever is genuinely hotter coming in.

Usage
-----
  # propose refreshed form for every team (does NOT write):
  python fetch_form.py

  # one specific team, machine-readable:
  python fetch_form.py --team MEX --json

  # refresh and MERGE verified rows into data/form.csv:
  python fetch_form.py --write

  # widen/narrow the recency window (default: matches from 2025 on):
  python fetch_form.py --cutoff 2024 --write

Hermes
------
  python hermes.py form                 # propose (dry-run)
  python hermes.py form --team MEX --write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse

# Reuse the scraper plumbing from fetch_h2h so this script stays DRY.
import fetch_h2h
from fetch_h2h import TEAM_NAME, _fetch, _names, _strip_html

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FORM_CSV = os.path.join(DATA, "form.csv")

# Keep matches at least this recent — momentum is about the run-in, not history.
DEFAULT_CUTOFF = 2025

# Cap how many recent matches we keep per team (most recent first).
MAX_PER_TEAM = 6

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _form_url(team_id: str) -> str:
    """DuckDuckGo lite search for a team's recent results."""
    q = f"{TEAM_NAME[team_id]} national team recent results 2025 2026 fixtures"
    return "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q)


def _extract_date(window: str, cutoff: int) -> str | None:
    """Pull a best-effort YYYY-MM-DD (or YYYY) from the text around a score."""
    low = window.lower()
    # "13 Jun 2026" / "Jun 13, 2026" / "June 2026"
    m = re.search(r"(\d{1,2})\s+([a-z]{3,9})\.?\s+(20\d{2})", low)
    if m and m.group(2)[:3] in _MONTHS:
        d, mon, y = int(m.group(1)), _MONTHS[m.group(2)[:3]], int(m.group(3))
        if cutoff <= y <= 2026:
            return f"{y:04d}-{mon:02d}-{d:02d}"
    m = re.search(r"([a-z]{3,9})\.?\s+(\d{1,2})?,?\s*(20\d{2})", low)
    if m and m.group(1)[:3] in _MONTHS:
        mon, y = _MONTHS[m.group(1)[:3]], int(m.group(3))
        if cutoff <= y <= 2026:
            d = int(m.group(2)) if m.group(2) else 1
            return f"{y:04d}-{mon:02d}-{d:02d}"
    m = re.search(r"(20\d{2})", low)
    if m:
        y = int(m.group(1))
        if cutoff <= y <= 2026:
            return f"{y:04d}"
    return None


def _parse_form(text: str, team_id: str, cutoff: int) -> list[dict]:
    """Extract a team's recent matches from search snippets.

    Matches "<Team> g1-g2 <Opp>" and "<Opp> g1-g2 <Team>" (goals oriented to the
    team), reads a nearby date, recency-filters, and classifies the stage.
    Returns rows in the form.csv shape, newest first, capped at MAX_PER_TEAM.
    """
    names = "|".join(re.escape(n) for n in _names(team_id))
    team_set = {n.lower() for n in _names(team_id)}
    # team listed first (gf-ga) or second (ga-gf, swapped to team POV)
    pat = re.compile(
        rf"(?P<left>{names})\s+(?P<g1>\d{{1,2}})\s*[-:–]\s*(?P<g2>\d{{1,2}})"
        rf"|(?P<g3>\d{{1,2}})\s*[-:–]\s*(?P<g4>\d{{1,2}})\s+(?P<right>{names})",
        re.IGNORECASE,
    )
    rows: list[dict] = []
    seen: set[tuple] = set()
    for m in pat.finditer(text):
        if m.group("left"):
            gf, ga = int(m.group("g1")), int(m.group("g2"))
        else:
            # team on the right -> "<opp> ga-gf <team>"
            ga, gf = int(m.group("g3")), int(m.group("g4"))
        if gf > 20 or ga > 20:
            continue
        window = text[max(0, m.start() - 60): m.end() + 80]
        date = _extract_date(window, cutoff)
        if not date:
            continue
        comp = fetch_h2h._classify(window)
        key = (date, gf, ga, comp)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "team_id": team_id, "gf": gf, "ga": ga, "comp": comp, "date": date,
        })
    # newest first; keep the most recent MAX_PER_TEAM
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows[:MAX_PER_TEAM]


def fetch_team(team_id: str, cutoff: int = DEFAULT_CUTOFF, retries: int = 3) -> dict:
    """Fetch + parse one team's recent form (retries on throttle)."""
    url = _form_url(team_id)
    for attempt in range(retries):
        html = _fetch(url)
        text = _strip_html(html) if html else ""
        # reuse fetch_h2h's block/short-body detection
        low = text.lower()
        blocked = any(b in low for b in fetch_h2h._BLOCK_MARKERS) or len(text) < 400
        if not blocked and any(n.lower() in low for n in _names(team_id)):
            return {"team": team_id, "url": url, "ok": True,
                    "rows": _parse_form(text, team_id, cutoff)}
        time.sleep(2.0 * (attempt + 1))
    return {"team": team_id, "url": url, "ok": False, "rows": []}


def _load_existing() -> list[dict]:
    import csv
    if not os.path.exists(FORM_CSV):
        return []
    with open(FORM_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _merge_write(new_rows: list[dict]) -> int:
    """Merge fetched rows into form.csv, de-duping on (team, score, comp, date)."""
    import csv
    existing = _load_existing()

    def norm(r):
        return (r["team_id"], int(r["gf"]), int(r["ga"]),
                str(r["comp"]).lower(), str(r["date"]))

    merged = {norm(r): r for r in existing}
    added = 0
    for r in new_rows:
        k = norm(r)
        if k not in merged:
            merged[k] = r
            added += 1

    fields = ["team_id", "gf", "ga", "comp", "date"]
    ordered = sorted(merged.values(), key=lambda r: (str(r["team_id"]), str(r["date"])))
    with open(FORM_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r[k] for k in fields})
    return added


def _provider_form(provider, team_id: str, cutoff: int) -> list[dict]:
    """Fetch one team's form via the structured API, recency-filtered to cutoff."""
    rows = provider.recent_form(team_id, TEAM_NAME.get(team_id, team_id),
                                last=MAX_PER_TEAM)
    out = []
    for r in rows:
        y = str(r.get("date", ""))[:4]
        if y.isdigit() and int(y) >= cutoff:
            out.append(r)
    return out


def run(team_ids, cutoff: int, write: bool, polite: float = 0.6) -> dict:
    # Prefer the structured API-Football source when a key is configured; fall
    # back to the DuckDuckGo scrape per-team when it is missing or returns nothing.
    from src import datameta
    from src.providers import RateLimitError, provider_from_env
    provider = provider_from_env(DATA)
    source = "api-football" if provider else "duckduckgo"

    results, all_rows = [], []
    for t in team_ids:
        rows, ok = [], False
        if provider:
            try:
                rows = _provider_form(provider, t, cutoff)
                ok = bool(rows)
            except RateLimitError as e:
                results.append({"team": t, "ok": False, "error": str(e)})
                break
            except Exception:
                rows, ok = [], False
        if not rows:  # API miss -> scrape fallback
            res = fetch_team(t, cutoff)
            rows, ok = res["rows"], res["ok"]
            time.sleep(polite)
        results.append({"team": t, "ok": ok, "rows": rows})
        all_rows.extend(rows)
    out = {
        "source": source,
        "cutoff": cutoff,
        "teams_checked": len(team_ids),
        "teams_fetched_ok": sum(1 for r in results if r.get("ok")),
        "rows_found": len(all_rows),
        "rows": all_rows,
    }
    if write:
        out["rows_added"] = _merge_write(all_rows)
        out["written_to"] = FORM_CSV
        datameta.stamp(DATA, "form", source, out["rows_added"])
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.models import DataStore

    p = argparse.ArgumentParser(description="Fetch recent team form (momentum)")
    p.add_argument("--team", metavar="TEAM", help="fetch one team, e.g. --team MEX")
    p.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF,
                   help=f"earliest year to keep (default {DEFAULT_CUTOFF})")
    p.add_argument("--write", action="store_true",
                   help="merge verified rows into data/form.csv")
    p.add_argument("--json", action="store_true", help="raw JSON output")
    args = p.parse_args()

    ds = DataStore.load(DATA)
    teams = [args.team] if args.team else list(ds.teams.team_id)
    result = run(teams, args.cutoff, args.write)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"בדקתי {result['teams_checked']} נבחרות, "
              f"{result['teams_fetched_ok']} נטענו, "
              f"{result['rows_found']} משחקים מ-{result['cutoff']} ואילך:")
        for r in result["rows"]:
            print(f"  {r['team_id']} {r['gf']}-{r['ga']} ({r['comp']}, {r['date']})")
        if args.write:
            print(f"\nנוספו {result.get('rows_added', 0)} שורות חדשות -> {FORM_CSV}")
        else:
            print("\n(הרצה יבשה — הוסף --write כדי למזג ל-form.csv)")
