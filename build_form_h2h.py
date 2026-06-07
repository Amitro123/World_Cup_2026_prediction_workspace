"""build_form_h2h.py — regenerate data/form.csv and data/h2h.csv for all 48
World Cup 2026 teams from the open martj42/international_results snapshot
(data/holdout_raw/_intl_results.csv), strictly from matches BEFORE the cutoff.

Why: the hand-curated form.csv (12 rows, mostly MEX/RSA) and h2h.csv (30 rows)
were too sparse to move most predictions — exactly the gap the project review
flagged. The dataset is real, complete and fresh, so we can fill both signals
for every participant with no network and no leakage (cutoff = today).

form.csv  rows: team_id, gf, ga, comp, date          (one per recent match)
h2h.csv   rows: team_a, team_b, a_goals, b_goals, comp, year   (as played)

Run:  python build_form_h2h.py
"""
from __future__ import annotations

import csv
import os
import unicodedata
from collections import defaultdict

DATA = os.path.join(os.path.dirname(__file__), "data")
INTL = os.path.join(DATA, "holdout_raw", "_intl_results.csv")
TEAMS = os.path.join(DATA, "teams.csv")

CUTOFF = "2026-06-07"          # today; use only matches strictly before this
FORM_PER_TEAM = 12            # most recent N matches kept per team (form)
H2H_FROM_YEAR = 2006          # include meetings from this year on (older decay out)

# martj42 names that don't normalise to our name_en. normalise() handles accents
# and punctuation; these cover genuine spelling differences.
ALIASES = {
    "czech republic": "CZE",
    "turkiye": "TUR",
    "korea republic": "KOR",
    "congo dr": "COD",
    "democratic republic of the congo": "COD",
    "cape verde islands": "CPV",
    "united states of america": "USA",
    "ivory coast": "CIV",
}


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


def _comp_label(tournament: str) -> str:
    t = tournament.lower()
    if "qualif" in t:
        return "qualifier"
    if "friendly" in t:
        return "friendly"
    return "competitive"  # finals of major comps (no round granularity in dataset)


def load_name2code() -> dict[str, str]:
    m: dict[str, str] = {}
    with open(TEAMS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[normalise(r["name_en"])] = r["team_id"]
    for alias, code in ALIASES.items():
        m[normalise(alias)] = code
    return m


def load_matches(name2code) -> list[dict]:
    out = []
    with open(INTL, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["date"]
            if d >= CUTOFF:
                continue
            h = name2code.get(normalise(r["home_team"]))
            a = name2code.get(normalise(r["away_team"]))
            if not h or not a or h == a:
                continue
            try:
                gh = int(r["home_score"]); ga = int(r["away_score"])
            except (KeyError, ValueError):
                continue
            out.append({"date": d, "home": h, "away": a, "gh": gh, "ga": ga,
                        "comp": _comp_label(r.get("tournament", ""))})
    out.sort(key=lambda m: m["date"])
    return out


def build_form(matches, codes) -> list[dict]:
    by_team: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        by_team[m["home"]].append({"team_id": m["home"], "gf": m["gh"],
                                   "ga": m["ga"], "comp": m["comp"], "date": m["date"]})
        by_team[m["away"]].append({"team_id": m["away"], "gf": m["ga"],
                                   "ga": m["gh"], "comp": m["comp"], "date": m["date"]})
    rows = []
    for code in sorted(codes):
        recent = sorted(by_team.get(code, []), key=lambda r: r["date"])[-FORM_PER_TEAM:]
        rows.extend(recent)
    return rows


def build_h2h(matches, codes) -> list[dict]:
    rows = []
    for m in matches:
        if int(m["date"][:4]) < H2H_FROM_YEAR:
            continue
        if m["home"] in codes and m["away"] in codes:
            rows.append({"team_a": m["home"], "team_b": m["away"],
                         "a_goals": m["gh"], "b_goals": m["ga"],
                         "comp": m["comp"], "year": m["date"][:4]})
    return rows


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    name2code = load_name2code()
    codes = set(name2code.values())
    matches = load_matches(name2code)
    print(f"loaded {len(matches)} relevant matches before {CUTOFF} "
          f"covering {len({c for m in matches for c in (m['home'], m['away'])})} teams")

    form = build_form(matches, codes)
    h2h = build_h2h(matches, codes)
    write_csv(os.path.join(DATA, "form.csv"),
              ["team_id", "gf", "ga", "comp", "date"], form)
    write_csv(os.path.join(DATA, "h2h.csv"),
              ["team_a", "team_b", "a_goals", "b_goals", "comp", "year"], h2h)

    # coverage report
    form_teams = {r["team_id"] for r in form}
    missing = sorted(codes - form_teams)
    print(f"form.csv: {len(form)} rows, {len(form_teams)}/48 teams")
    print(f"h2h.csv:  {len(h2h)} rows since {H2H_FROM_YEAR}")
    if missing:
        print(f"WARNING no form for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
