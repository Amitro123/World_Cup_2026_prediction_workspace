"""
fetch_player_props.py — (dormant) pull anytime scorer/assist odds into
data/players_market.csv.

Status: DORMANT by design
-------------------------
There is no *free* feed for player props. The Odds API lists them under premium
market keys (`player_goal_scorer_anytime`, `player_assists`, etc.) that the free
tier does not return, and bet365's on-screen "Player to Score or Assist" market
(the screenshot that motivated this) has no free public API. So this script does
NOT scrape anything and ships dormant: src/playerprops.py already computes the
model props, and the dashboard shows them with or without market odds.

What IS built and tested
------------------------
The **parser is pure** (`parse_event`, `rows_from_payload`) and unit-tested, so
the day you add a paid key — or paste a saved JSON payload — the data lands in
the right shape with no further code. It maps The-Odds-API player-prop markets to
our players_market.csv columns and resolves player names case-insensitively
against players.csv, skipping anyone it cannot match rather than guessing.

players_market.csv columns
---------------------------
    match_id, team_id, name_en, name_he, score_odds, assist_odds,
    score_or_assist_odds, bookmaker, captured_at

Usage (once a feed exists)
--------------------------
    python fetch_player_props.py --from-json saved_props.json --match-id C1
"""

from __future__ import annotations

import csv
import json
import os

DATA = os.path.join(os.path.dirname(__file__), "data")
OUT_FIELDS = ["match_id", "team_id", "name_en", "name_he", "score_odds",
              "assist_odds", "score_or_assist_odds", "bookmaker", "captured_at"]

# The-Odds-API premium market keys -> our column.
MARKET_KEYS = {
    "player_goal_scorer_anytime": "score_odds",
    "player_anytime_scorer": "score_odds",
    "player_assists": "assist_odds",
    "player_goal_scorer_or_assist": "score_or_assist_odds",
    "player_to_score_or_assist": "score_or_assist_odds",
}


# --- pure parser (no network) ------------------------------------------------

def parse_event(event: dict, prefer_book: str | None = None) -> dict:
    """One props event -> {player_name_lower: {score_odds/assist_odds/...}}.

    Walks every bookmaker's player-prop markets. For each market we recognise
    (MARKET_KEYS), each outcome's `description` (or `name`) is the player and
    `price` the decimal odds. `prefer_book` is tried first; the first price seen
    per (player, column) wins so a preferred book is not overwritten.
    """
    books = event.get("bookmakers") or []
    if prefer_book:
        books = sorted(books, key=lambda b: b.get("key") != prefer_book)
    players: dict[str, dict] = {}
    for book in books:
        bkey = book.get("key", "")
        for market in book.get("markets") or []:
            col = MARKET_KEYS.get(market.get("key"))
            if col is None:
                continue
            for o in market.get("outcomes") or []:
                name = o.get("description") or o.get("name")
                price = o.get("price")
                if not name or not price:
                    continue
                rec = players.setdefault(str(name).lower(),
                                         {"name": name, "bookmaker": bkey})
                rec.setdefault(col, float(price))  # first price wins
    return players


def rows_from_payload(payload, match_id: str, player_index: dict,
                      captured_at: str = "") -> list[dict]:
    """Parse a props payload into players_market.csv rows for one match.

    player_index: {name_en_lower: {"team_id","name_en","name_he"}} from
        players.csv. Only players we recognise are emitted; unknown names are
        skipped (no guessing). A player with no recognised odds is dropped.
    """
    rows = []
    for event in payload or []:
        for key, rec in parse_event(event).items():
            who = player_index.get(key)
            if who is None:
                continue
            if not any(c in rec for c in ("score_odds", "assist_odds",
                                          "score_or_assist_odds")):
                continue
            rows.append({
                "match_id": match_id,
                "team_id": who["team_id"],
                "name_en": who["name_en"],
                "name_he": who.get("name_he", ""),
                "score_odds": rec.get("score_odds", ""),
                "assist_odds": rec.get("assist_odds", ""),
                "score_or_assist_odds": rec.get("score_or_assist_odds", ""),
                "bookmaker": rec.get("bookmaker", ""),
                "captured_at": captured_at,
            })
    return rows


def build_player_index(players_csv: str) -> dict:
    """{name_en_lower: {team_id, name_en, name_he}} from players.csv."""
    index = {}
    with open(players_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ne = str(r.get("name_en", "")).strip()
            if ne:
                index[ne.lower()] = {
                    "team_id": r.get("team_id", ""),
                    "name_en": ne,
                    "name_he": r.get("name_he", ""),
                }
    return index


def write_players_market(rows: list[dict], path: str | None = None,
                         append: bool = True) -> str:
    """Write/append rows to players_market.csv (props are per-match, so append)."""
    path = path or os.path.join(DATA, "players_market.csv")
    exists = os.path.exists(path)
    mode = "a" if (append and exists) else "w"
    with open(path, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None) -> None:
    import argparse
    import datetime as dt
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", default=None,
                    help="parse a saved props JSON payload (no free live source)")
    ap.add_argument("--match-id", default=None,
                    help="the match_id these props belong to (required with --from-json)")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite players_market.csv instead of appending")
    args = ap.parse_args(argv)

    if not args.from_json:
        print("[fetch_player_props] DORMANT: no free player-props feed. This "
              "script parses a saved JSON payload only. Provide one with "
              "--from-json FILE --match-id C1 once you have a paid feed. The "
              "model props already show in the dashboard without it.")
        return
    if not args.match_id:
        ap.error("--match-id is required with --from-json")

    with open(args.from_json, encoding="utf-8") as f:
        payload = json.load(f)
    index = build_player_index(os.path.join(DATA, "players.csv"))
    rows = rows_from_payload(payload, args.match_id, index,
                             captured_at=dt.date.today().isoformat())
    out = write_players_market(rows, append=not args.overwrite)
    print(f"[fetch_player_props] wrote {len(rows)} player rows for "
          f"{args.match_id} -> {out}")


if __name__ == "__main__":
    main()
