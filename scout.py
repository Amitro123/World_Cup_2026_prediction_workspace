"""
scout.py — Pre-game intelligence scraper for World Cup 2026.

Scrapes BBC Sport, ESPN, Goal.com and Transfermarkt for:
  - Injury / suspension news
  - Expected lineups
  - Betting line movements (where accessible)

Returns a structured dict with findings + RECOMMENDED model adjustments.
Amit makes the final call — this module never writes to models directly.

Usage:
  python scout.py --match A1
  python scout.py --match H1 --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Team name aliases for search
# ---------------------------------------------------------------------------
TEAM_ALIASES: dict[str, list[str]] = {
    "MEX": ["Mexico", "Mexican"],
    "RSA": ["South Africa"],
    "KOR": ["South Korea", "Korea"],
    "CZE": ["Czech", "Czechia"],
    "CAN": ["Canada", "Canadian"],
    "BIH": ["Bosnia", "Herzegovina"],
    "QAT": ["Qatar"],
    "SUI": ["Switzerland", "Swiss"],
    "BRA": ["Brazil", "Brazilian"],
    "MAR": ["Morocco", "Moroccan"],
    "HAI": ["Haiti"],
    "SCO": ["Scotland", "Scottish"],
    "USA": ["USA", "United States", "USMNT"],
    "PAR": ["Paraguay"],
    "AUS": ["Australia", "Socceroos"],
    "TUR": ["Turkey", "Turkiye"],
    "GER": ["Germany", "German"],
    "CUW": ["Curacao"],
    "CIV": ["Ivory Coast", "Côte d'Ivoire", "Cote d'Ivoire"],
    "ECU": ["Ecuador"],
    "NED": ["Netherlands", "Holland", "Dutch"],
    "JPN": ["Japan", "Japanese"],
    "SWE": ["Sweden", "Swedish"],
    "TUN": ["Tunisia"],
    "BEL": ["Belgium", "Belgian"],
    "EGY": ["Egypt", "Egyptian"],
    "IRN": ["Iran", "Iranian"],
    "NZL": ["New Zealand"],
    "ESP": ["Spain", "Spanish"],
    "CPV": ["Cape Verde"],
    "KSA": ["Saudi Arabia", "Saudi"],
    "URU": ["Uruguay"],
    "FRA": ["France", "French"],
    "SEN": ["Senegal"],
    "NOR": ["Norway", "Norwegian"],
    "IRQ": ["Iraq", "Iraqi"],
    "ARG": ["Argentina", "Argentine"],
    "ALG": ["Algeria", "Algerian"],
    "AUT": ["Austria", "Austrian"],
    "JOR": ["Jordan"],
    "POR": ["Portugal", "Portuguese"],
    "COD": ["DR Congo", "Congo"],
    "UZB": ["Uzbekistan"],
    "COL": ["Colombia", "Colombian"],
    "ENG": ["England", "English"],
    "CRO": ["Croatia", "Croatian"],
    "GHA": ["Ghana"],
    "PAN": ["Panama"],
}

# Key players per team (for injury detection)
KEY_PLAYERS: dict[str, list[str]] = {
    "FRA": ["Mbappe", "Mbappé", "Dembele", "Griezmann"],
    "ARG": ["Messi", "Martinez", "Alvarez"],
    "ESP": ["Yamal", "Williams", "Pedri", "Morata"],
    "ENG": ["Bellingham", "Salah", "Saka", "Kane"],
    "POR": ["Ronaldo", "Bruno Fernandes", "Bernardo"],
    "BRA": ["Vinicius", "Rodrygo", "Endrick"],
    "GER": ["Wirtz", "Musiala", "Havertz"],
    "NED": ["van Dijk", "Gakpo", "Dumfries"],
    "BEL": ["De Bruyne", "Lukaku", "Tielemans"],
    "MAR": ["Hakimi", "Ziyech", "En-Nesyri"],
    "MEX": ["Lozano", "Jimenez", "Alvarez"],
    "USA": ["Pulisic", "Reyna", "McKennie"],
    "URU": ["Valverde", "Nunez", "Suarez"],
    "COL": ["James", "Luiz Diaz", "Falcao"],
}

# Impact of a key player injury on FIFA points
INJURY_IMPACT = {
    "star":    -80,   # unambiguous starter + team talisman
    "key":     -50,   # regular starter
    "rotation": -25,  # squad player
}

# Suspension impact — always certain (100%), so no severity multiplier
SUSPENSION_IMPACT = {
    "star":     -80,
    "key":      -55,
    "rotation": -25,
}

# Words that indicate a suspension (card-related absence)
SUSPENSION_WORDS = [
    "suspended", "suspension", "ban", "banned",
    "red card", "second yellow", "two yellows",
    "accumulated", "bookings", "will miss",
    "misses the next", "serves a ban",
]


def _fetch(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch URL content, return None on failure."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
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


def _find_injury_mentions(text: str, team_id: str) -> list[dict]:
    """Scan text for injury/doubt/suspension mentions for a team's players."""
    found = []
    players = KEY_PLAYERS.get(team_id, [])
    injury_words = [
        "injur", "doubt", "doubtful", "suspended", "suspension",
        "ruled out", "miss", "fitness", "concern", "knock", "strain",
        "hamstring", "ankle", "thigh", "calf", "muscle", "ligament"
    ]
    text_lower = text.lower()
    for player in players:
        pl_lower = player.lower()
        idx = text_lower.find(pl_lower)
        while idx != -1:
            window = text_lower[max(0, idx - 120):idx + 200]
            for word in injury_words:
                if word in window:
                    snippet = text[max(0, idx - 80):idx + 160].strip()
                    if not any(p["player"] == player for p in found):
                        found.append({
                            "player": player,
                            "team_id": team_id,
                            "snippet": snippet[:250],
                            "severity": _guess_severity(window),
                            "type": "injury",
                        })
                    break
            idx = text_lower.find(pl_lower, idx + 1)
    return found


def _find_suspension_mentions(text: str, team_id: str) -> list[dict]:
    """Scan text for card/suspension mentions for a team's players."""
    found = []
    players = KEY_PLAYERS.get(team_id, [])
    text_lower = text.lower()
    for player in players:
        pl_lower = player.lower()
        idx = text_lower.find(pl_lower)
        while idx != -1:
            window = text_lower[max(0, idx - 150):idx + 250]
            for word in SUSPENSION_WORDS:
                if word in window:
                    snippet = text[max(0, idx - 80):idx + 180].strip()
                    if not any(p["player"] == player for p in found):
                        # Determine card type from context
                        if "red card" in window or "straight red" in window:
                            card_type = "red_card"
                        elif "second yellow" in window or "two yellow" in window or "accumulated" in window:
                            card_type = "yellow_accumulation"
                        else:
                            card_type = "suspension"
                        found.append({
                            "player": player,
                            "team_id": team_id,
                            "snippet": snippet[:250],
                            "severity": "out",  # suspensions are always certain
                            "type": "suspension",
                            "card_type": card_type,
                        })
                    break
            idx = text_lower.find(pl_lower, idx + 1)
    return found


def _guess_severity(window: str) -> str:
    """Guess injury severity from surrounding text."""
    if any(w in window for w in ["ruled out", "will not play", "misses", "out for"]):
        return "out"
    if any(w in window for w in ["doubtful", "doubt", "50-50", "uncertain"]):
        return "doubt"
    if any(w in window for w in ["fitness test", "late fitness", "concern", "knock"]):
        return "concern"
    return "concern"


def _find_odds_movement(text: str, team_id: str) -> Optional[dict]:
    """Look for betting line movement mentions."""
    aliases = TEAM_ALIASES.get(team_id, [])
    movement_words = ["odds", "favourite", "favorite", "backed", "drifted",
                      "shortened", "price", "betting", "wagering"]
    text_lower = text.lower()
    for alias in aliases:
        idx = text_lower.find(alias.lower())
        if idx != -1:
            window = text_lower[max(0, idx - 100):idx + 200]
            for word in movement_words:
                if word in window:
                    snippet = text[max(0, idx - 60):idx + 160].strip()
                    return {"team_id": team_id, "snippet": snippet[:200]}
    return None


def scrape_bbc(team_ids: list[str]) -> dict:
    """Scrape BBC Sport World Cup section for injury/suspension news."""
    results = {"source": "BBC Sport", "injuries": [], "suspensions": [], "odds": []}
    urls = [
        "https://www.bbc.com/sport/football/world-cup",
        "https://www.bbc.com/sport/football/world-cup/teams",
    ]
    for url in urls:
        html = _fetch(url)
        if not html:
            continue
        text = _strip_html(html)
        for tid in team_ids:
            results["injuries"].extend(_find_injury_mentions(text, tid))
            results["suspensions"].extend(_find_suspension_mentions(text, tid))
        time.sleep(0.5)
    return results


def scrape_goal(team_ids: list[str]) -> dict:
    """Scrape Goal.com for World Cup team news."""
    results = {"source": "Goal.com", "injuries": [], "suspensions": [], "odds": []}
    html = _fetch("https://www.goal.com/en/world-cup")
    if html:
        text = _strip_html(html)
        for tid in team_ids:
            results["injuries"].extend(_find_injury_mentions(text, tid))
            results["suspensions"].extend(_find_suspension_mentions(text, tid))
    return results


def scrape_espn(team_ids: list[str]) -> dict:
    """Scrape ESPN for World Cup injury/suspension/news."""
    results = {"source": "ESPN", "injuries": [], "suspensions": [], "odds": []}
    urls = [
        "https://www.espn.com/soccer/story/_/id/world-cup-2026",
        "https://www.espn.com/soccer/world-cup/",
    ]
    for url in urls:
        html = _fetch(url)
        if not html:
            continue
        text = _strip_html(html)
        for tid in team_ids:
            results["injuries"].extend(_find_injury_mentions(text, tid))
            results["suspensions"].extend(_find_suspension_mentions(text, tid))
            odds = _find_odds_movement(text, tid)
            if odds:
                results["odds"].append(odds)
        time.sleep(0.5)
    return results


def scrape_transfermarkt(team_ids: list[str]) -> dict:
    """Scrape Transfermarkt for team news and lineup hints."""
    results = {"source": "Transfermarkt", "injuries": [], "suspensions": [], "lineups": []}
    tm_slugs = {
        "ESP": "spanien", "FRA": "frankreich", "ARG": "argentinien",
        "ENG": "england", "BRA": "brasilien", "GER": "deutschland",
        "POR": "portugal", "NED": "niederlande", "BEL": "belgien",
        "MAR": "marokko", "MEX": "mexiko", "USA": "vereinigte-staaten",
    }
    for tid in team_ids:
        slug = tm_slugs.get(tid)
        if not slug:
            continue
        url = f"https://www.transfermarkt.com/{slug}/kader/verein/0"
        html = _fetch(url)
        if not html:
            continue
        text = _strip_html(html)
        results["injuries"].extend(_find_injury_mentions(text, tid))
        results["suspensions"].extend(_find_suspension_mentions(text, tid))
        time.sleep(0.8)
    return results


def build_recommendations(all_injuries: list[dict], all_suspensions: list[dict]) -> list[dict]:
    """
    Translate injury + suspension findings into recommended model adjustments.
    Returns a list of recommended hermes.py update calls.
    Amit decides whether to apply them.

    Key difference:
      - Injuries have severity multiplier (out=1.0, doubt=0.6, concern=0.3)
      - Suspensions are always certain → full impact, no multiplier
    """
    recs = []
    seen = set()

    stars = {
        "Mbappe", "Mbappé", "Messi", "Ronaldo", "Yamal",
        "Vinicius", "Bellingham", "De Bruyne", "Wirtz"
    }
    keys = {
        "Pedri", "Williams", "Morata", "Griezmann", "Salah", "Kane",
        "Pulisic", "Hakimi", "Bruno Fernandes", "Gakpo", "Musiala"
    }

    def get_tier(player):
        if player in stars:
            return "star"
        if player in keys:
            return "key"
        return "rotation"

    # --- Injuries (with severity multiplier) ---
    for inj in all_injuries:
        key = ("injury", inj["player"], inj["team_id"])
        if key in seen:
            continue
        seen.add(key)

        severity = inj["severity"]
        mult = {"out": 1.0, "doubt": 0.6, "concern": 0.3}.get(severity, 0.3)
        delta = round(INJURY_IMPACT[get_tier(inj["player"])] * mult)
        if delta == 0:
            continue

        recs.append({
            "match_relevant_team": inj["team_id"],
            "player": inj["player"],
            "event_type": "injury",
            "severity": severity,
            "kind": "rating_delta",
            "value": delta,
            "note_he": f"פציעה: {inj['player']} ({severity})",
            "source": inj.get("source", "web"),
            "snippet": inj.get("snippet", ""),
        })

    # --- Suspensions (always certain — full impact) ---
    for sus in all_suspensions:
        key = ("suspension", sus["player"], sus["team_id"])
        if key in seen:
            continue
        seen.add(key)

        card_type = sus.get("card_type", "suspension")
        delta = SUSPENSION_IMPACT[get_tier(sus["player"])]

        card_emoji = {"red_card": "🟥", "yellow_accumulation": "🟨🟨", "suspension": "🚫"}.get(card_type, "🚫")
        card_label = {
            "red_card": "כרטיס אדום",
            "yellow_accumulation": "צהובים מצטברים",
            "suspension": "השעיה",
        }.get(card_type, "השעיה")

        recs.append({
            "match_relevant_team": sus["team_id"],
            "player": sus["player"],
            "event_type": "suspension",
            "card_type": card_type,
            "card_emoji": card_emoji,
            "severity": "out",  # always certain
            "kind": "rating_delta",
            "value": delta,
            "note_he": f"{card_label}: {sus['player']} — לא ישחק",
            "source": sus.get("source", "web"),
            "snippet": sus.get("snippet", ""),
        })

    return recs


def scout_match(match_id: str, home_id: str, away_id: str) -> dict:
    """
    Full pre-game scout for a match.
    Returns findings + recommendations (NOT applied — Amit decides).
    """
    team_ids = [home_id, away_id]
    all_injuries = []
    all_suspensions = []
    sources_results = []

    for scraper, label in [
        (lambda t: scrape_bbc(t), "BBC Sport"),
        (lambda t: scrape_goal(t), "Goal.com"),
        (lambda t: scrape_espn(t), "ESPN"),
        (lambda t: scrape_transfermarkt(t), "Transfermarkt"),
    ]:
        try:
            r = scraper(team_ids)
            for inj in r.get("injuries", []):
                inj["source"] = label
            for sus in r.get("suspensions", []):
                sus["source"] = label
            all_injuries.extend(r.get("injuries", []))
            all_suspensions.extend(r.get("suspensions", []))
            sources_results.append(r)
        except Exception:
            pass

    recs = build_recommendations(all_injuries, all_suspensions)

    return {
        "match_id": match_id,
        "home_id": home_id,
        "away_id": away_id,
        "injuries_found": all_injuries,
        "suspensions_found": all_suspensions,
        "recommendations": recs,
        "sources_scraped": [s["source"] for s in sources_results],
    }


def format_for_human(result: dict, ds=None) -> str:
    """Format scout results as a readable Hebrew Telegram message."""
    lines = []
    mid = result["match_id"]

    def tname(tid):
        if ds:
            return ds.team_name(tid, "he")
        return tid

    home = tname(result["home_id"])
    away = tname(result["away_id"])

    lines.append(f"🔍 **סריקת מודיעין — {home} נגד {away} ({mid})**")
    lines.append(f"📡 מקורות: {', '.join(result['sources_scraped'])}")
    lines.append("")

    # --- Injuries ---
    injuries = result.get("injuries_found", [])
    suspensions = result.get("suspensions_found", [])

    if injuries:
        lines.append("🚑 **פציעות / ספקות:**")
        seen = set()
        for inj in injuries:
            if inj["player"] in seen:
                continue
            seen.add(inj["player"])
            sev_emoji = {"out": "🔴", "doubt": "🟡", "concern": "🟠"}.get(inj["severity"], "⚪")
            lines.append(f"  {sev_emoji} **{inj['player']}** ({tname(inj['team_id'])}) — {inj['severity']}")
            if inj.get("snippet"):
                lines.append(f"     _{inj['snippet'][:120]}_")

    # --- Suspensions ---
    if suspensions:
        lines.append("🚫 **השעיות / כרטיסים:**")
        seen = set()
        for sus in suspensions:
            if sus["player"] in seen:
                continue
            seen.add(sus["player"])
            card_emoji = sus.get("card_emoji", "🚫")
            card_label = {"red_card": "כרטיס אדום", "yellow_accumulation": "צהובים מצטברים",
                         "suspension": "השעיה"}.get(sus.get("card_type", ""), "השעיה")
            lines.append(f"  {card_emoji} **{sus['player']}** ({tname(sus['team_id'])}) — {card_label} | **לא ישחק**")
            if sus.get("snippet"):
                lines.append(f"     _{sus['snippet'][:120]}_")

    if not injuries and not suspensions:
        lines.append("✅ לא נמצאו פציעות, השעיות או ספקות מרכזיים")

    lines.append("")

    # --- Recommendations ---
    recs = result.get("recommendations", [])
    if recs:
        lines.append("💡 **המלצות עדכון מודל (ממתינות לאישורך):**")
        for i, r in enumerate(recs, 1):
            delta = r["value"]
            sign = "+" if delta > 0 else ""
            event_emoji = "🚫" if r.get("event_type") == "suspension" else "🚑"
            card_em = r.get("card_emoji", "") if r.get("event_type") == "suspension" else ""
            lines.append(
                f"  **[{i}]** {event_emoji}{card_em} {tname(r['match_relevant_team'])}: "
                f"`rating_delta {sign}{delta}` ← {r['player']} ({r.get('card_type', r.get('severity', ''))})"
            )
        lines.append("")
        lines.append("↩️ השב עם מספרי ההמלצות לאישור (למשל: `1 2`) או `דלג`")
    else:
        lines.append("📊 אין המלצות — המודל עובד על נתוני הבסיס")

    return "\n".join(lines)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-game scout for a World Cup match")
    parser.add_argument("--match", required=True, help="Match ID, e.g. A1")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from src.models import DataStore
    ds = DataStore.load(os.path.join(os.path.dirname(__file__), "data"))

    row = ds.matches[ds.matches.match_id == args.match]
    if row.empty:
        print(json.dumps({"error": f"match {args.match} not found"}))
        sys.exit(1)

    row = row.iloc[0]
    result = scout_match(args.match, row.home_id, row.away_id)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_for_human(result, ds))
