"""
fetch_holdout.py — build a holdout backtest CSV for a past tournament, with
*derived* pre-tournament ratings (no hand-keyed snapshots).

What it produces
----------------
`data/backtest_<name>.csv` in the exact schema the holdout harness reads:

    date, home, away, rating_home, rating_away, home_goals, away_goals,
    neutral, stage, form_sup, h2h_sup

`rating_home/away` are **derived World Football Elo** (src/elo.py) computed
*strictly from results before the tournament started*, then recentred to the
FIFA-points scale (mean 1500) so the engine's supremacy and total-goals terms
stay calibrated. `form_sup` / `h2h_sup` are the same bounded signals the
production model uses (engine.form_supremacy / engine.h2h_supremacy), also
computed as-of the tournament start — so adding the tournament can never see its
own outcomes (no leakage), and the holdout's config comparison
(`python -m src.backtest --holdout`) can finally measure whether H2H/form earn
their place across more than one tournament.

Two ways to run
---------------
1. Offline (no key, fully reproducible): drop a results file at
   `data/holdout_raw/<name>.csv` (columns: date,home,away,gh,ga,neutral,comp,
   league) and run `python fetch_holdout.py --name <name> --start YYYY-MM-DD
   --end YYYY-MM-DD --teams CODE,CODE,...`.
2. Online: add your API-Football key (.env) and pass `--fetch`; one call per
   team (`--last` matches each) fills the raw file and is cached to disk, so you
   can backfill a few teams a day — no need to pull everything at once.

The build step is a pure function (`build_rows`) so it is unit-tested without
any network.
"""

from __future__ import annotations

import csv
import os

from src import elo, engine

DATA = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA, "holdout_raw")

OUT_FIELDS = ["date", "home", "away", "rating_home", "rating_away",
              "home_goals", "away_goals", "neutral", "stage",
              "form_sup", "h2h_sup"]


# --- pure core (no network) --------------------------------------------------

def _oriented_form(pre: list[dict], team: str) -> list[dict]:
    """A team's matches from `pre`, oriented to it as {gf,ga,comp,date}."""
    out = []
    for m in pre:
        if m["home"] == team:
            out.append({"gf": m["gh"], "ga": m["ga"],
                        "comp": m.get("comp", ""), "date": m.get("date", "")})
        elif m["away"] == team:
            out.append({"gf": m["ga"], "ga": m["gh"],
                        "comp": m.get("comp", ""), "date": m.get("date", "")})
    return out


def _meetings(pre: list[dict], a: str, b: str) -> list[dict]:
    """Past A-vs-B meetings from `pre`, oriented to A as {gd,comp,year}."""
    out = []
    for m in pre:
        if {m["home"], m["away"]} != {a, b}:
            continue
        gd = (m["gh"] - m["ga"]) if m["home"] == a else (m["ga"] - m["gh"])
        year = int(str(m.get("date", "0"))[:4] or 0) if str(m.get("date", ""))[:4].isdigit() else None
        out.append({"gd": gd, "comp": m.get("comp", ""), "year": year})
    return out


def build_rows(
    history: list[dict],
    start: str,
    end: str,
    teams,
    league_substr: str | None = None,
    relabel: dict | None = None,
) -> list[dict]:
    """Turn a results history into holdout rows for one tournament.

    history: all known results, each {date,home,away,gh,ga,neutral,comp,league}.
    start/end: 'YYYY-MM-DD' bounds of the tournament (inclusive) — matches in
        this window between two `teams`, optionally filtered to `league_substr`,
        are the ones predicted.
    teams: the set of competitor ids (same id space as history).
    relabel: optional {id -> output code} applied to emitted home/away (e.g.
        API ids -> FIFA codes).

    Ratings/signals are computed from matches strictly BEFORE `start`, so the
    tournament never informs its own predictions.
    """
    teams = set(teams)
    pre = [m for m in history if str(m.get("date", "")) < start]
    tour = [
        m for m in history
        if start <= str(m.get("date", "")) <= end
        and m["home"] in teams and m["away"] in teams
        and (league_substr is None
             or league_substr.lower() in str(m.get("league", "")).lower())
    ]
    tour.sort(key=lambda m: str(m.get("date", "")))

    # derived pre-tournament Elo, recentred onto the FIFA-points scale
    elo_raw = elo.snapshot_before(pre, start)
    elo_c = elo.recenter(elo_raw, teams=teams, mean=engine.FIFA_MEAN)

    # one momentum score per competitor (as-of start)
    fscore = {t: engine.form_score(_oriented_form(pre, t), ref_date=start) for t in teams}

    ref_year = int(start[:4]) if start[:4].isdigit() else None
    rl = relabel or {}
    rows = []
    for m in tour:
        h, a = m["home"], m["away"]
        form_sup = engine.form_supremacy(fscore.get(h, 0.0), fscore.get(a, 0.0))
        h2h_sup = engine.h2h_supremacy(_meetings(pre, h, a), ref_year=ref_year)
        rows.append({
            "date": m.get("date", ""),
            "home": rl.get(h, h), "away": rl.get(a, a),
            "rating_home": round(elo_c.get(h, engine.FIFA_MEAN), 1),
            "rating_away": round(elo_c.get(a, engine.FIFA_MEAN), 1),
            "home_goals": m["gh"], "away_goals": m["ga"],
            "neutral": 1 if m.get("neutral", True) else 0,
            "stage": m.get("comp", "group"),
            "form_sup": round(form_sup, 4),
            "h2h_sup": round(h2h_sup, 4),
        })
    return rows


# --- io ----------------------------------------------------------------------

def read_raw(name: str) -> list[dict]:
    """Load data/holdout_raw/<name>.csv into history dicts (typed)."""
    path = os.path.join(RAW_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                out.append({
                    "date": r["date"], "home": r["home"], "away": r["away"],
                    "gh": int(r["gh"]), "ga": int(r["ga"]),
                    "neutral": str(r.get("neutral", "1")).strip() in ("1", "True", "true"),
                    "comp": r.get("comp", ""), "league": r.get("league", ""),
                })
            except (KeyError, ValueError):
                continue
    return out


def write_raw(name: str, history: list[dict]) -> str:
    """Persist/cache fetched history, de-duplicated by (date,home,away)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    seen = {}
    for m in history:
        seen[(m.get("date"), m.get("home"), m.get("away"))] = m
    rows = sorted(seen.values(), key=lambda m: str(m.get("date", "")))
    path = os.path.join(RAW_DIR, f"{name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "home", "away", "gh", "ga",
                                          "neutral", "comp", "league"])
        w.writeheader()
        for m in rows:
            w.writerow({k: (1 if k == "neutral" and m.get(k) else
                            (0 if k == "neutral" else m.get(k, ""))) for k in
                        ["date", "home", "away", "gh", "ga", "neutral", "comp", "league"]})
    return path


def write_backtest(name: str, rows: list[dict]) -> str:
    path = os.path.join(DATA, f"backtest_{name}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


# --- network gather (thin; needs a key) --------------------------------------

def gather_history(name: str, team_names: dict, last: int = 60) -> list[dict]:
    """Fetch each team's recent matches via the provider and cache them.

    team_names: {FIFA_CODE: English name}. One API call per team. Merges with any
    already-cached raw file so you can backfill incrementally (a few teams a day).
    Returns the merged history (also written to data/holdout_raw/<name>.csv).
    """
    from src.providers import RateLimitError, provider_from_env
    provider = provider_from_env(DATA)
    if provider is None:
        raise RuntimeError("no API_FOOTBALL_KEY configured (see .env.example); "
                           "or build offline from data/holdout_raw/<name>.csv")
    history = read_raw(name)
    for code, nm in team_names.items():
        try:
            history.extend(provider.team_matches(code, nm, last=last))
        except RateLimitError:
            print(f"[rate-limit] stopped after partial backfill at {code}; "
                  f"cached so far — resume tomorrow.")
            break
    write_raw(name, history)
    return read_raw(name)


def run(name: str, start: str, end: str, teams, league_substr: str | None = None,
        fetch: bool = False, team_names: dict | None = None, last: int = 60) -> dict:
    """Build (and optionally fetch) the holdout CSV for one tournament."""
    if fetch:
        if not team_names:
            raise ValueError("--fetch needs team names (CODE=Name,...)")
        gather_history(name, team_names, last=last)
    history = read_raw(name)
    if not history:
        return {"error": f"no history at data/holdout_raw/{name}.csv "
                         f"(use --fetch with a key, or drop the file in)"}
    rows = build_rows(history, start, end, teams, league_substr=league_substr)
    out = write_backtest(name, rows)
    return {"name": name, "matches": len(rows), "csv": out,
            "teams": len(set(teams))}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="tournament label, e.g. euro2024")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD tournament start")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD tournament end")
    ap.add_argument("--teams", required=True,
                    help="comma-separated competitor codes (FIFA codes)")
    ap.add_argument("--league", default=None,
                    help="substring filter on league name (e.g. 'Euro')")
    ap.add_argument("--fetch", action="store_true",
                    help="fetch history via API-Football first (needs a key)")
    ap.add_argument("--names", default=None,
                    help="for --fetch: CODE=Name,CODE=Name,... (English names)")
    ap.add_argument("--last", type=int, default=60,
                    help="matches to pull per team when fetching (default 60)")
    args = ap.parse_args(argv)

    teams = [t.strip() for t in args.teams.split(",") if t.strip()]
    team_names = None
    if args.names:
        team_names = dict(p.split("=", 1) for p in args.names.split(",") if "=" in p)
    rep = run(args.name, args.start, args.end, teams, league_substr=args.league,
              fetch=args.fetch, team_names=team_names, last=args.last)
    if "error" in rep:
        print(f"[fetch_holdout] {rep['error']}")
        return
    print(f"[fetch_holdout] wrote {rep['matches']} matches for {rep['teams']} "
          f"teams -> {rep['csv']}")
    print("  run:  python -m src.backtest --holdout")


if __name__ == "__main__":
    main()
