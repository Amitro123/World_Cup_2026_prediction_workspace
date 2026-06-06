"""
fetch_h2h.py — מאחזר מפגשי-עבר (head-to-head) מהאינטרנט עבור World Cup 2026.

Mirrors scout.py: a browser-headed scraper. It queries an open web-search
endpoint (DuckDuckGo) for each pair, reads the result snippets, extracts recent
meetings between two national teams (e.g. "England 1-0 Croatia (13 Jun, 2021)"),
classifies friendly vs competitive, filters to a recency window, and proposes
rows for data/h2h.csv. Like scout.py it never changes the model silently — it
prints proposed rows and only writes when you pass --write (Amit / Hermes decides).

The engine consumes h2h.csv as a small, bounded supremacy signal
(see src/engine.h2h_supremacy + README §"מפגשי עבר"), so refreshing this file is
all that's needed for the agent to "take past meetings into account".

Usage
-----
  # propose refreshed rows for every group-stage pair (does NOT write):
  python fetch_h2h.py

  # one specific pair, machine-readable:
  python fetch_h2h.py --pair ENG CRO --json

  # refresh all group pairs and MERGE verified rows into data/h2h.csv:
  python fetch_h2h.py --write

  # widen/narrow the recency window (default: meetings from 2018 on):
  python fetch_h2h.py --cutoff 2019 --write

Hermes
------
  python hermes.py h2h            # propose (dry-run)
  python hermes.py h2h --write    # fetch + merge into data/h2h.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Optional

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
H2H_CSV = os.path.join(DATA, "h2h.csv")

# Only keep meetings this recent. The user asked for "recent meetings only"
# (≈5–8 years); 2018 is the start of the last full World-Cup cycle.
DEFAULT_CUTOFF = 2018

# Primary English display name per team_id (used to build URLs + match rows).
# Mirrors scout.TEAM_ALIASES[id][0] but kept local so this script stands alone.
TEAM_NAME: dict[str, str] = {
    "MEX": "Mexico", "RSA": "South Africa", "KOR": "South Korea", "CZE": "Czech Republic",
    "CAN": "Canada", "BIH": "Bosnia and Herzegovina", "QAT": "Qatar", "SUI": "Switzerland",
    "BRA": "Brazil", "MAR": "Morocco", "HAI": "Haiti", "SCO": "Scotland",
    "USA": "USA", "PAR": "Paraguay", "AUS": "Australia", "TUR": "Turkey",
    "GER": "Germany", "CUW": "Curacao", "CIV": "Ivory Coast", "ECU": "Ecuador",
    "NED": "Netherlands", "JPN": "Japan", "SWE": "Sweden", "TUN": "Tunisia",
    "BEL": "Belgium", "EGY": "Egypt", "IRN": "Iran", "NZL": "New Zealand",
    "ESP": "Spain", "CPV": "Cape Verde", "KSA": "Saudi Arabia", "URU": "Uruguay",
    "FRA": "France", "SEN": "Senegal", "NOR": "Norway", "IRQ": "Iraq",
    "ARG": "Argentina", "ALG": "Algeria", "AUT": "Austria", "JOR": "Jordan",
    "POR": "Portugal", "COD": "DR Congo", "UZB": "Uzbekistan", "COL": "Colombia",
    "ENG": "England", "CRO": "Croatia", "GHA": "Ghana", "PAN": "Panama",
}

# Extra name variants used when matching a team inside a search snippet
# (a result may say "United States" or "Korea Republic" instead of our primary).
NAME_VARIANTS: dict[str, list[str]] = {
    "USA": ["USA", "United States", "US"],
    "KOR": ["South Korea", "Korea Republic", "Korea"],
    "TUR": ["Turkey", "Turkiye", "Türkiye"],
    "IRN": ["Iran", "IR Iran"],
    "CIV": ["Ivory Coast", "Cote d'Ivoire"],
    "CZE": ["Czech Republic", "Czechia"],
    "COD": ["DR Congo", "Congo DR", "Congo"],
    "BIH": ["Bosnia and Herzegovina", "Bosnia"],
    "CPV": ["Cape Verde", "Cabo Verde"],
    "RSA": ["South Africa"],
}

FRIENDLY_WORDS = ("friendly", "friendlies", "kirin", "club friendlies", "int. friendly")

# Keyword -> stage label written to comp (engine grades these; see
# engine.H2H_COMP_WEIGHTS). Checked in order; more specific stages first.
STAGE_KEYWORDS = (
    (("semi-final", "semifinal", "semi final"), "semifinal"),
    (("quarter", "round of 16", "round of 32", "last 16", "last 8",
      "knockout", "play-off", "playoff"), "knockout"),
    (("final",), "final"),
    (FRIENDLY_WORDS, "friendly"),
    (("qualif",), "qualifier"),
    (("group",), "group"),
)


def _fetch(url: str, timeout: int = 12) -> Optional[str]:
    """GET a page with a real browser UA (same approach as scout._fetch)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _names(team_id: str) -> list[str]:
    """All name spellings to look for in a snippet, longest first."""
    variants = NAME_VARIANTS.get(team_id, [TEAM_NAME[team_id]])
    if TEAM_NAME[team_id] not in variants:
        variants = [TEAM_NAME[team_id]] + variants
    return sorted(set(variants), key=len, reverse=True)


def _h2h_url(home_id: str, away_id: str) -> str:
    """DuckDuckGo lite search for the pair's results (server-rendered snippets)."""
    # listing recent years steers results toward a head-to-head/results page
    # (a bare "A vs B" query returns the upcoming 2026 fixture instead).
    q = f"{TEAM_NAME[home_id]} {TEAM_NAME[away_id]} 2018 2021 2022 2023 2024 results history"
    return "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(q)


def _classify(window: str) -> str:
    """Infer the match status/stage from the snippet around the score."""
    w = window.lower()
    for words, label in STAGE_KEYWORDS:
        if any(k in w for k in words):
            return label
    return "competitive"


def _parse_meetings(text: str, home_id: str, away_id: str, cutoff: int) -> list[dict]:
    """
    Extract meetings from search snippets such as
    "England 1-0 Croatia (13 Jun, 2021)" or "Morocco 2 - 1 Brazil ... 2023".

    For every "<TeamX> g1[-:]g2 <TeamY>" pattern we locate a 4-digit year in the
    surrounding window, recency-filter it, and normalise the score to the home
    point of view. Returns rows in the h2h.csv shape.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()
    names_h = "|".join(re.escape(n) for n in _names(home_id))
    names_a = "|".join(re.escape(n) for n in _names(away_id))
    # both orderings: home-first and away-first
    pat = re.compile(
        rf"(?P<left>{names_h}|{names_a})\s+(?P<g1>\d{{1,2}})\s*[-:–]\s*"
        rf"(?P<g2>\d{{1,2}})\s+(?P<right>{names_h}|{names_a})",
        re.IGNORECASE,
    )
    home_set = {n.lower() for n in _names(home_id)}
    year_re = re.compile(r"(20\d{2})")

    for m in pat.finditer(text):
        left, right = m.group("left").lower(), m.group("right").lower()
        left_is_home = left in home_set
        right_is_home = right in home_set
        if left_is_home == right_is_home:
            continue  # need exactly one of each team
        g1, g2 = int(m.group("g1")), int(m.group("g2"))
        if g1 > 20 or g2 > 20:
            continue
        if not left_is_home:  # away listed first -> swap to home POV
            g1, g2 = g2, g1
        window = text[max(0, m.start() - 40): m.end() + 120]
        years = [int(y) for y in year_re.findall(window) if cutoff <= int(y) <= 2026]
        if not years:
            continue
        year = max(years)
        comp = _classify(window)
        key = (year, g1, g2, comp)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "team_a": home_id, "team_b": away_id,
            "a_goals": g1, "b_goals": g2, "comp": comp, "year": year,
        })
    return rows


# Under rapid load DuckDuckGo serves a bot-challenge page instead of results.
# We detect that (and short/empty bodies) and retry with backoff. For a full
# 72-pair sweep some pairs may still get blocked — that's an inherent limit of
# free scraping; the typical use is a per-pair refresh before a fixture, which
# is a single request and does not trip the limiter.
_BLOCK_MARKERS = ("bots use duckduckgo", "complete the following challenge",
                  "confirm this search was made by a human")


def _usable_snippets(home_id: str, away_id: str, text: str) -> bool:
    low = text.lower()
    if any(m in low for m in _BLOCK_MARKERS) or len(text) < 400:
        return False
    # at least the away team should be mentioned in the result snippets
    return any(n.lower() in low for n in _names(away_id))


def fetch_pair(home_id: str, away_id: str, cutoff: int = DEFAULT_CUTOFF,
               retries: int = 3) -> dict:
    """Fetch + parse recent meetings for one ordered pair (retries on throttle)."""
    url = _h2h_url(home_id, away_id)
    for attempt in range(retries):
        html = _fetch(url)
        text = _strip_html(html) if html else ""
        if _usable_snippets(home_id, away_id, text):
            rows = _parse_meetings(text, home_id, away_id, cutoff)
            return {"pair": [home_id, away_id], "url": url, "ok": True, "rows": rows}
        time.sleep(2.0 * (attempt + 1))  # back off, let the limiter clear
    return {"pair": [home_id, away_id], "url": url, "ok": False, "rows": []}


def group_pairs(ds) -> list[tuple[str, str]]:
    """Unique unordered team pairs that actually meet in the group stage."""
    seen: set[frozenset] = set()
    pairs: list[tuple[str, str]] = []
    for _, m in ds.matches.iterrows():
        if str(m.stage) != "group":
            continue
        key = frozenset((m.home_id, m.away_id))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((m.home_id, m.away_id))
    return pairs


def _load_existing() -> list[dict]:
    import csv
    if not os.path.exists(H2H_CSV):
        return []
    with open(H2H_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _merge_write(new_rows: list[dict]) -> int:
    """Merge fetched rows into h2h.csv, de-duping on (pair, score, comp, year)."""
    import csv
    existing = _load_existing()

    def norm(r):
        a, b = r["team_a"], r["team_b"]
        ga, gb = int(r["a_goals"]), int(r["b_goals"])
        if a > b:  # canonical orientation for de-dup
            a, b, ga, gb = b, a, gb, ga
        return (a, b, ga, gb, str(r["comp"]).lower(), int(r["year"]))

    merged = {norm(r): r for r in existing}
    added = 0
    for r in new_rows:
        k = norm(r)
        if k not in merged:
            merged[k] = r
            added += 1

    fields = ["team_a", "team_b", "a_goals", "b_goals", "comp", "year"]
    ordered = sorted(merged.values(), key=lambda r: (str(r["comp"]), int(r["year"])))
    with open(H2H_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r[k] for k in fields})
    return added


def run(pairs, cutoff: int, write: bool, polite: float = 0.6) -> dict:
    results, all_rows = [], []
    for h, a in pairs:
        res = fetch_pair(h, a, cutoff)
        results.append(res)
        all_rows.extend(res["rows"])
        time.sleep(polite)  # be a good citizen
    out = {
        "cutoff": cutoff,
        "pairs_checked": len(pairs),
        "pairs_fetched_ok": sum(1 for r in results if r["ok"]),
        "rows_found": len(all_rows),
        "rows": all_rows,
    }
    if write:
        out["rows_added"] = _merge_write(all_rows)
        out["written_to"] = H2H_CSV
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.models import DataStore

    p = argparse.ArgumentParser(description="Fetch recent head-to-head meetings")
    p.add_argument("--pair", nargs=2, metavar=("HOME", "AWAY"),
                   help="fetch one pair, e.g. --pair ENG CRO")
    p.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF,
                   help=f"earliest year to keep (default {DEFAULT_CUTOFF})")
    p.add_argument("--write", action="store_true",
                   help="merge verified rows into data/h2h.csv")
    p.add_argument("--json", action="store_true", help="raw JSON output")
    args = p.parse_args()

    ds = DataStore.load(DATA)
    pairs = [tuple(args.pair)] if args.pair else group_pairs(ds)
    result = run(pairs, args.cutoff, args.write)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"בדקתי {result['pairs_checked']} זוגות, "
              f"{result['pairs_fetched_ok']} נטענו, "
              f"{result['rows_found']} מפגשים מ-{result['cutoff']} ואילך:")
        for r in result["rows"]:
            print(f"  {r['team_a']} {r['a_goals']}-{r['b_goals']} {r['team_b']} "
                  f"({r['comp']}, {r['year']})")
        if args.write:
            print(f"\nנוספו {result.get('rows_added', 0)} שורות חדשות -> {H2H_CSV}")
        else:
            print("\n(הרצה יבשה — הוסף --write כדי למזג ל-h2h.csv)")
