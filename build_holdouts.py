"""build_holdouts.py — build holdout backtest CSVs from a public results dataset.

The free API-Football plan only covers seasons 2022-2024 and forbids the bulk
`last` parameter, so it cannot supply the multi-year history needed to derive
pre-tournament Elo/form/h2h for older tournaments. Instead we use the open
`martj42/international_results` dataset (every international match since 1872),
cached at data/holdout_raw/_intl_results.csv.

For each named tournament we:
  1. read the whole dataset as match history (date,home,away,gh,ga,neutral,
     comp,league) — comp mapped to the engine's weight labels, qualifiers
     renamed so the finals filter can't accidentally swallow them;
  2. auto-detect the participants (teams that played a finals match in the
     tournament window);
  3. call fetch_holdout.build_rows — which derives Elo/form/h2h strictly from
     matches BEFORE the start date (no leakage) — and write data/backtest_<name>.csv.

Run:  python build_holdouts.py
Then: python -m src.backtest --holdout
"""
from __future__ import annotations

import csv
import os

from fetch_holdout import build_rows, write_backtest

DATA = os.path.join(os.path.dirname(__file__), "data")
INTL = os.path.join(DATA, "holdout_raw", "_intl_results.csv")

# name -> (exact finals tournament label, start, end) inclusive YYYY-MM-DD
TOURNAMENTS = {
    "wc2014":   ("FIFA World Cup", "2014-06-12", "2014-07-13"),
    "wc2018":   ("FIFA World Cup", "2018-06-14", "2018-07-15"),
    "euro2020": ("UEFA Euro",      "2021-06-11", "2021-07-11"),  # played 2021
    "euro2024": ("UEFA Euro",      "2024-06-14", "2024-07-14"),
}


def _comp_label(tournament: str) -> str:
    """Map a dataset tournament string to an engine comp-weight label."""
    t = tournament.lower()
    if "qualif" in t:
        return "qualifier"
    if "friendly" in t:
        return "friendly"
    # finals of major comps; we lack round granularity -> baseline competitive
    return "competitive"


def _league_tag(tournament: str) -> str:
    """League field used only by build_rows' tournament-window filter.

    Qualifiers are renamed so a finals substring (e.g. 'FIFA World Cup') can't
    match 'FIFA World Cup qualification' and leak qualifiers into the holdout.
    """
    return "Qualifiers" if "qualif" in tournament.lower() else tournament


def load_history() -> list[dict]:
    out: list[dict] = []
    with open(INTL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                gh = int(r["home_score"]); ga = int(r["away_score"])
            except (KeyError, ValueError):
                continue  # unplayed / malformed
            tour = r.get("tournament", "")
            out.append({
                "date": r["date"],
                "home": r["home_team"], "away": r["away_team"],
                "gh": gh, "ga": ga,
                "neutral": str(r.get("neutral", "")).strip().lower() == "true",
                "comp": _comp_label(tour),
                "league": _league_tag(tour),
            })
    return out


def participants(history, finals_label, start, end) -> list[str]:
    teams = set()
    for m in history:
        if m["league"] == finals_label and start <= m["date"] <= end:
            teams.add(m["home"]); teams.add(m["away"])
    return sorted(teams)


def main() -> None:
    history = load_history()
    print(f"loaded {len(history)} historical matches from {os.path.basename(INTL)}\n")
    for name, (label, start, end) in TOURNAMENTS.items():
        teams = participants(history, label, start, end)
        rows = build_rows(history, start, end, teams, league_substr=label)
        path = write_backtest(name, rows)
        print(f"{name:9} {len(teams):>2} teams, {len(rows):>2} matches "
              f"-> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
