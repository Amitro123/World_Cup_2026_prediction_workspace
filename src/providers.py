"""
src/providers.py — structured football-data source for the fetch_* ingestion.

The original `fetch_form.py` / `fetch_h2h.py` scraped DuckDuckGo HTML with regex.
That is fragile (bot challenges, missed snippets) and is why coverage stayed low.
This module adds a **structured** source — API-Football (api-sports.io) — that
returns clean fixtures with dates, scores and competition labels, so the same
fetch → merge → CSV pipeline can fill all 48 teams reliably.

Design
------
- Dependency-free (urllib + json), like the existing scrapers.
- The API key is read from the ``API_FOOTBALL_KEY`` env var (or a local ``.env``)
  and is NEVER committed. If no key is present, ``provider_from_env`` returns
  ``None`` and the callers fall back to their DuckDuckGo scrape, so the project
  still runs with zero setup.
- API-Football uses its own numeric team ids, so we resolve each FIFA 3-letter
  code once via ``/teams?search=`` and cache the mapping in
  ``data/team_id_map.json`` to spend the daily request budget only once.
- Free plan = 100 requests/day. We read the remaining-quota response header and
  raise a clear ``RateLimitError`` before the API starts rejecting calls, so a
  half-finished backfill is obvious rather than silent.

Row shapes returned (identical to what fetch_form/fetch_h2h already merge):
    form:  {"team_id", "gf", "ga", "comp", "date"}            # date = YYYY-MM-DD
    h2h:   {"team_a", "team_b", "a_goals", "b_goals", "comp", "year"}
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://v3.football.api-sports.io"
FINISHED = {"FT", "AET", "PEN"}  # only completed matches carry signal


class RateLimitError(RuntimeError):
    """Raised when the API-Football daily quota is exhausted."""


class APIError(RuntimeError):
    """Raised on an API-Football error payload or transport failure."""


def _load_dotenv(path: str) -> None:
    """Populate os.environ from a local .env (KEY=VALUE lines). No dependency."""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def _classify(league_name: str, round_name: str) -> str:
    """Map an API-Football league + round to our engine comp labels.

    Mirrors engine.H2H_COMP_WEIGHTS / FORM_COMP_WEIGHTS keys: friendly,
    qualifier, group, knockout, semifinal, final, competitive. Specific stages
    are checked before the generic 'final' so 'Semi-finals' is not read as final.
    """
    ln = str(league_name or "").lower()
    rn = str(round_name or "").lower()
    text = f"{ln} {rn}"
    if "friendl" in text:
        return "friendly"
    if "qualif" in text:
        return "qualifier"
    if "semi" in text:
        return "semifinal"
    if any(k in text for k in ("quarter", "round of", "last 16", "last 8",
                               "knockout", "play-off", "playoff", "round of 32",
                               "round of 16")):
        return "knockout"
    if "final" in text:
        return "final"
    if "group" in text:
        return "group"
    return "competitive"


class APIFootball:
    """Thin API-Football client returning form/h2h rows in the CSV shapes."""

    def __init__(self, key: str, data_dir: str, polite: float = 1.0,
                 timeout: int = 15):
        self.key = key
        self.data_dir = data_dir
        self.polite = polite
        self.timeout = timeout
        self.map_path = os.path.join(data_dir, "team_id_map.json")
        self._map = self._load_map()
        self.remaining: int | None = None  # last-seen daily quota remaining

    @property
    def available(self) -> bool:
        return bool(self.key)

    # --- low-level -----------------------------------------------------------
    def _load_map(self) -> dict[str, int]:
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, encoding="utf-8") as f:
                    return {k: int(v) for k, v in json.load(f).items() if v is not None}
            except (OSError, ValueError):
                return {}
        return {}

    def _save_map(self) -> None:
        try:
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(self._map, f, ensure_ascii=False, indent=2, sort_keys=True)
        except OSError:
            pass

    def _get(self, path: str, params: dict) -> list:
        """GET {BASE}{path}?params, return the 'response' list or raise."""
        url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"x-apisports-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rem = r.headers.get("x-ratelimit-requests-remaining")
                if rem is not None:
                    try:
                        self.remaining = int(rem)
                    except ValueError:
                        pass
                payload = json.loads(r.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RateLimitError("API-Football daily quota exhausted (HTTP 429)")
            raise APIError(f"HTTP {e.code} for {path}") from e
        except (urllib.error.URLError, ValueError) as e:
            raise APIError(f"transport error for {path}: {e}") from e

        errors = payload.get("errors")
        if errors:
            # API-Football reports quota/plan errors here as a dict or list.
            text = json.dumps(errors, ensure_ascii=False).lower()
            if "rate" in text or "limit" in text or "requests" in text:
                raise RateLimitError(f"API-Football limit: {errors}")
            raise APIError(f"API-Football error: {errors}")
        time.sleep(self.polite)  # be a good citizen; protect the daily budget
        return payload.get("response", []) or []

    # --- public -------------------------------------------------------------
    def resolve_team_id(self, fifa_code: str, name: str) -> int | None:
        """FIFA code -> API-Football numeric team id (cached, national teams)."""
        if fifa_code in self._map:
            return self._map[fifa_code]
        resp = self._get("/teams", {"search": name})
        chosen = None
        for item in resp:
            team = item.get("team", {})
            if team.get("national") and team.get("id"):
                chosen = int(team["id"])
                break
        if chosen is None and resp:  # fall back to first hit
            team = resp[0].get("team", {})
            chosen = int(team["id"]) if team.get("id") else None
        if chosen is not None:
            self._map[fifa_code] = chosen
            self._save_map()
        return chosen

    def recent_form(self, fifa_code: str, name: str, last: int = 6) -> list[dict]:
        """A team's last finished matches as form.csv rows (oriented to team)."""
        tid = self.resolve_team_id(fifa_code, name)
        if tid is None:
            return []
        resp = self._get("/fixtures", {"team": tid, "last": last})
        rows: list[dict] = []
        for fx in resp:
            if fx.get("fixture", {}).get("status", {}).get("short") not in FINISHED:
                continue
            goals = fx.get("goals", {})
            teams = fx.get("teams", {})
            gh, ga = goals.get("home"), goals.get("away")
            if gh is None or ga is None:
                continue
            is_home = teams.get("home", {}).get("id") == tid
            gf, gc = (gh, ga) if is_home else (ga, gh)
            league = fx.get("league", {})
            rows.append({
                "team_id": fifa_code,
                "gf": int(gf), "ga": int(gc),
                "comp": _classify(league.get("name"), league.get("round")),
                "date": str(fx.get("fixture", {}).get("date", ""))[:10],
            })
        return rows

    def head_to_head(self, home_id: str, away_id: str, home_name: str,
                     away_name: str, last: int = 10, cutoff: int = 2018) -> list[dict]:
        """Recent meetings as h2h.csv rows, oriented to home_id (team_a)."""
        a = self.resolve_team_id(home_id, home_name)
        b = self.resolve_team_id(away_id, away_name)
        if a is None or b is None:
            return []
        resp = self._get("/fixtures/headtohead", {"h2h": f"{a}-{b}", "last": last})
        rows: list[dict] = []
        for fx in resp:
            if fx.get("fixture", {}).get("status", {}).get("short") not in FINISHED:
                continue
            goals = fx.get("goals", {})
            teams = fx.get("teams", {})
            gh, ga = goals.get("home"), goals.get("away")
            if gh is None or ga is None:
                continue
            date = str(fx.get("fixture", {}).get("date", ""))
            year = int(date[:4]) if date[:4].isdigit() else 0
            if year < cutoff:
                continue
            # orient to our home_id (team_a)
            a_is_home = teams.get("home", {}).get("id") == a
            a_goals, b_goals = (gh, ga) if a_is_home else (ga, gh)
            league = fx.get("league", {})
            rows.append({
                "team_a": home_id, "team_b": away_id,
                "a_goals": int(a_goals), "b_goals": int(b_goals),
                "comp": _classify(league.get("name"), league.get("round")),
                "year": year,
            })
        return rows


def provider_from_env(data_dir: str) -> APIFootball | None:
    """Build an APIFootball client if a key is configured, else None.

    Looks for API_FOOTBALL_KEY in the environment or a local .env at the repo
    root. Returns None when unset so callers fall back to the DDG scraper.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _load_dotenv(os.path.join(root, ".env"))
    key = os.environ.get("API_FOOTBALL_KEY", "").strip()
    if not key:
        return None
    return APIFootball(key=key, data_dir=data_dir)
