"""
fetch_odds.py — pull 1X2 closing/pre-match odds into data/market_odds.csv.

Why
---
The code review's biggest unbuilt recommendation was a **bookmaker anchor**:
closing 1X2 odds are the gold standard for football calibration. src/oddslib.py
does the math (de-vig + model-vs-market divergence); this script fills the data.

Source
------
The Odds API (the-odds-api.com) — free tier ~500 requests/month, returns
decimal 1X2 (`h2h`) prices for many bookmakers. The API key is read from the
``ODDS_API_KEY`` env var (or a local ``.env``) and is NEVER committed. With no
key the script prints how to get one and exits cleanly, so the anchor simply
stays dormant — nothing else in the project depends on it.

The **parser is pure** (`parse_event`, `rows_from_payload`) so it is unit-tested
without any network. Mapping bookmaker team *names* back to our FIFA codes is the
only fiddly part; we match against teams.csv `name_en` (case-insensitive), and
skip events we cannot resolve rather than guessing.

Usage
-----
    python fetch_odds.py --sport soccer_fifa_world_cup            # live
    python fetch_odds.py --from-json sample_odds.json             # offline parse
"""

from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT_FIELDS = ["match_id", "dec_home", "dec_draw", "dec_away",
              "p_home", "p_draw", "p_away", "bookmaker", "captured_at"]


# --- pure parser (no network) ------------------------------------------------

def parse_event(event: dict, prefer_book: str | None = None) -> dict | None:
    """One The-Odds-API event -> {home_team, away_team, dec_home/draw/away, book}.

    Reads the `h2h` (moneyline = 1X2) market. Picks `prefer_book` if present,
    else the first bookmaker that lists all three prices. Returns None if no
    usable three-way price exists (e.g. a two-way book).
    """
    home = event.get("home_team")
    away = event.get("away_team")
    if not home or not away:
        return None
    books = event.get("bookmakers") or []
    if prefer_book:
        books = sorted(books, key=lambda b: b.get("key") != prefer_book)
    for book in books:
        for market in book.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            prices = {o.get("name"): o.get("price") for o in market.get("outcomes") or []}
            dh, dd, da = prices.get(home), prices.get("Draw"), prices.get(away)
            if dh and dd and da:
                return {
                    "home_team": home, "away_team": away,
                    "dec_home": float(dh), "dec_draw": float(dd),
                    "dec_away": float(da), "bookmaker": book.get("key", ""),
                }
    return None


def rows_from_payload(payload, match_index: dict, prefer_book: str | None = None,
                      captured_at: str = "") -> list[dict]:
    """Parse a full The-Odds-API response into market_odds.csv rows.

    match_index: {(home_name_lower, away_name_lower): match_id} built from
        teams.csv + matches.csv by the caller. Only events whose BOTH teams
        resolve to a scheduled match_id are emitted (orientation-aware: if the
        book lists the fixture the other way round, home/away decimals are
        swapped to match our match_id's orientation). Unresolved events are
        skipped, not guessed.
    """
    rows = []
    for event in payload or []:
        parsed = parse_event(event, prefer_book=prefer_book)
        if parsed is None:
            continue
        h = str(parsed["home_team"]).lower()
        a = str(parsed["away_team"]).lower()
        if (h, a) in match_index:
            mid = match_index[(h, a)]
            dh, dd, da = parsed["dec_home"], parsed["dec_draw"], parsed["dec_away"]
        elif (a, h) in match_index:  # book lists it reversed -> swap
            mid = match_index[(a, h)]
            dh, dd, da = parsed["dec_away"], parsed["dec_draw"], parsed["dec_home"]
        else:
            continue
        rows.append({
            "match_id": mid, "dec_home": dh, "dec_draw": dd, "dec_away": da,
            "p_home": "", "p_draw": "", "p_away": "",
            "bookmaker": parsed["bookmaker"], "captured_at": captured_at,
        })
    return rows


def build_match_index(teams_csv: str, matches_csv: str) -> dict:
    """{(home_name_lower, away_name_lower): match_id} from teams + matches CSVs."""
    code_to_name = {}
    with open(teams_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code_to_name[r["team_id"]] = str(r.get("name_en", "")).lower()
    index = {}
    with open(matches_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            h = code_to_name.get(r.get("home_id", ""))
            a = code_to_name.get(r.get("away_id", ""))
            if h and a:
                index[(h, a)] = r["match_id"]
    return index


# --- io ----------------------------------------------------------------------

def write_market_odds(rows: list[dict], path: str | None = None) -> str:
    path = path or os.path.join(DATA, "market_odds.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


# --- network (thin; needs a key) ---------------------------------------------

def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def fetch_payload(sport: str, api_key: str, regions: str = "uk,eu",
                  timeout: int = 15) -> list:
    """GET the live h2h odds for a sport. Raises on transport/HTTP error."""
    params = urllib.parse.urlencode({
        "apiKey": api_key, "regions": regions, "markets": "h2h",
        "oddsFormat": "decimal",
    })
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?{params}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"The Odds API HTTP {e.code}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"The Odds API transport error: {e}") from e


def main(argv=None) -> None:
    import argparse
    import datetime as dt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sport", default="soccer_fifa_world_cup",
                    help="The Odds API sport key (default: soccer_fifa_world_cup)")
    ap.add_argument("--regions", default="uk,eu", help="bookmaker regions")
    ap.add_argument("--book", default=None, help="preferred bookmaker key")
    ap.add_argument("--from-json", default=None,
                    help="parse a saved JSON payload instead of calling the API")
    args = ap.parse_args(argv)

    index = build_match_index(os.path.join(DATA, "teams.csv"),
                              os.path.join(DATA, "matches.csv"))
    stamp = dt.date.today().isoformat()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        _load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        key = os.environ.get("ODDS_API_KEY", "").strip()
        if not key:
            print("[fetch_odds] no ODDS_API_KEY set. Get a free key at "
                  "https://the-odds-api.com (≈500 req/month), add it to .env as "
                  "ODDS_API_KEY=..., then re-run. The anchor stays dormant until "
                  "then — nothing else breaks.")
            return
        payload = fetch_payload(args.sport, key, regions=args.regions)

    rows = rows_from_payload(payload, index, prefer_book=args.book, captured_at=stamp)
    out = write_market_odds(rows)
    print(f"[fetch_odds] wrote {len(rows)} matched fixtures -> {out}")
    if not rows:
        print("  (no events resolved to scheduled matches — check team names / "
              "sport key, or the tournament may not be listed yet.)")
    else:
        print("  view:  streamlit run app.py  ->  'מול בוקמייקרים'")


if __name__ == "__main__":
    main()
