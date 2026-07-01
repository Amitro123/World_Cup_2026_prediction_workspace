"""Pre-match briefing for one fixture — group-stage OR knockout tie.

Pulls the model's current numbers for a match (FIFA ratings, h2h/form
supremacy, xG, 1X2, advance-probability), leaves a clearly marked TODO for the
live news/injury/lineup check this script cannot perform on its own, and
prints any news findings you supply as PROPOSED news_adjustments rows — it
never calls `add_news_adjustment` itself. Review the table, then either paste
the printed call into a Python shell or re-run with `--apply` to commit it.

Usage
-----
    python scripts/pre_match_briefing.py --match BEL SEN
    python scripts/pre_match_briefing.py --match-id C4          # group-stage row
    python scripts/pre_match_briefing.py --match BEL SEN --news-file findings.json
    python scripts/pre_match_briefing.py --match BEL SEN --news-file findings.json --apply

findings.json is a list of objects shaped like `add_news_adjustment`'s args::

    [{"team_id": "BEL", "kind": "rating_delta", "value": -40,
      "note_he": "קפטן נפצע באימון", "source": "https://..."}]

Why a knockout tie needs `--match TEAM TEAM` instead of a plain match_id: R32
ties get a stable id once the group stage is final (`knockout.match_id_for`,
e.g. "M78"), but that id alone doesn't tell this script which two teams are
playing — R16-onward ties aren't even fixed until earlier rounds are decided.
Team names are the only input that always resolves.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

from src import engine, knockout  # noqa: E402
from src.models import DataStore  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Same GOLAZO name aliasing as predict_value.py (app_odds.csv uses FIFA's own
# display names, which don't always match teams.csv name_en).
APP_ODDS_ALIASES = {
    "Korea Republic": "South Korea", "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Curaçao": "Curacao",
}


# --- team / match resolution -------------------------------------------------

def resolve_team(ds: DataStore, s: str) -> str:
    """team_id, exact name_en, or unique substring of name_en -> team_id."""
    s = s.strip()
    ids = set(ds.teams.team_id.astype(str))
    if s.upper() in ids:
        return s.upper()
    low = s.lower()
    names = ds.teams.name_en.astype(str)
    exact = ds.teams[names.str.lower() == low]
    if not exact.empty:
        return exact.iloc[0].team_id
    partial = ds.teams[names.str.lower().str.contains(low, regex=False)]
    if len(partial) == 1:
        return partial.iloc[0].team_id
    if len(partial) > 1:
        raise SystemExit(
            f"ambiguous team '{s}': matches {partial.name_en.tolist()} "
            "-- use the team_id instead"
        )
    raise SystemExit(f"could not resolve team '{s}' (try a team_id like BEL)")


def group_match_row(ds: DataStore, a: str, b: str):
    """The matches.csv row for a and b if they played each other in the group
    stage, else None."""
    m = ds.matches
    pair = m[((m.home_id == a) & (m.away_id == b)) | ((m.home_id == b) & (m.away_id == a))]
    return None if pair.empty else pair.iloc[0]


def r32_match_no(ds: DataStore, a: str, b: str, seed: int = knockout.DEFAULT_SEED):
    """The official R32 match number a-vs-b are drawn into, or None.

    Only resolvable once every group game is `finished` (no simulation left to
    do). With a full group stage this is deterministic except for genuine
    exact-tiebreak coincidences, which `_group_phase`'s last-resort
    `rng.random()` breaks arbitrarily -- rare, and worth a manual double-check
    if the printed pairing looks off.
    """
    ctx = knockout._prepare(ds)
    pos, third_assign, _ = knockout._group_phase(ctx, random.Random(seed))
    r32 = knockout._resolve_r32(pos, third_assign)
    for m, (x, y) in r32.items():
        if {x, y} == {a, b}:
            return m
    return None


class Fixture:
    def __init__(self, ds, home_id, away_id, match_id, is_group, resolvable):
        self.ds, self.home_id, self.away_id = ds, home_id, away_id
        self.match_id, self.is_group, self.resolvable = match_id, is_group, resolvable


def resolve_fixture(ds: DataStore, args) -> Fixture:
    if args.match_id and not args.match:
        row = ds.matches.loc[ds.matches.match_id == args.match_id]
        if row.empty:
            raise SystemExit(
                f"--match-id {args.match_id} is not a matches.csv row and no "
                "--match TEAM TEAM was given to resolve a knockout tie"
            )
        r = row.iloc[0]
        return Fixture(ds, r.home_id, r.away_id, args.match_id, True, True)

    if not args.match:
        raise SystemExit("supply --match TEAM1 TEAM2 or --match-id GROUP_MATCH_ID")

    home_id = resolve_team(ds, args.match[0])
    away_id = resolve_team(ds, args.match[1])

    if args.match_id:
        return Fixture(ds, home_id, away_id, args.match_id, False, True)

    grow = group_match_row(ds, home_id, away_id)
    if grow is not None:
        return Fixture(ds, grow.home_id, grow.away_id, grow.match_id, True, True)

    m = r32_match_no(ds, home_id, away_id)
    if m is not None:
        return Fixture(ds, home_id, away_id, knockout.match_id_for(m), False, True)

    placeholder = f"KO-{home_id}-{away_id}"
    return Fixture(ds, home_id, away_id, placeholder, False, False)


# --- model numbers -------------------------------------------------------

def advance_probability(rh: float, ra: float, sup: float) -> float:
    """P(rh-side advances a neutral-venue tie) through 90' -> ET -> capped pens.

    Mirrors predict_bracket.py's `ko_prob` so headline numbers agree between
    the two tools.
    """
    p = engine.ProbabilityModel().pre_match(rh, ra, neutral=True, h2h_sup=sup)
    ph, pd_, pa = p["p_home"], p["p_draw"], p["p_away"]
    frac = max(1 - engine.SHOOTOUT_CAP, min(engine.SHOOTOUT_CAP, ph / (ph + pa + 1e-9)))
    lh, la = engine.expected_goals(rh, ra, neutral=True, h2h_sup=sup)
    et = engine.probs_from_lambdas(
        lh * engine.ET_LAMBDA_SCALE, la * engine.ET_LAMBDA_SCALE, dixon_coles=False
    )
    return ph + pd_ * (et["p_home"] + et["p_draw"] * frac)


def model_numbers(fx: Fixture, extra_delta: dict[str, float] | None = None) -> dict:
    ds = fx.ds
    extra_delta = extra_delta or {}
    r_home = ds.team_rating(fx.home_id) + extra_delta.get(fx.home_id, 0.0)
    r_away = ds.team_rating(fx.away_id) + extra_delta.get(fx.away_id, 0.0)
    h2h_sup = ds.h2h_supremacy_for(fx.home_id, fx.away_id)
    form_sup = ds.form_supremacy_for(fx.home_id, fx.away_id)

    if fx.is_group:
        neutral = not ds.is_host(fx.home_id)
        lam_h, lam_a = engine.expected_goals(
            r_home, r_away, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
        )
        probs = engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=True)
        advance = None  # a single group game doesn't decide qualification alone
    else:
        host_adj = 0.0
        if ds.is_host(fx.home_id):
            host_adj += knockout.KNOCKOUT_HOST_ADV
        if ds.is_host(fx.away_id):
            host_adj -= knockout.KNOCKOUT_HOST_ADV
        sup_total = h2h_sup + form_sup + host_adj
        lam_h, lam_a = engine.expected_goals(r_home, r_away, neutral=True, h2h_sup=sup_total)
        probs = engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=True)
        advance = advance_probability(r_home, r_away, sup_total)

    return {
        "r_home": r_home, "r_away": r_away,
        "h2h_sup": h2h_sup, "form_sup": form_sup,
        "lambda_home": lam_h, "lambda_away": lam_a,
        "p_home": probs["p_home"], "p_draw": probs["p_draw"], "p_away": probs["p_away"],
        "advance_home": advance,
    }


# --- news findings (proposed, never auto-saved unless --apply) -----------

def load_findings(path: str | None) -> list[dict]:
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_news_todo(fx: Fixture) -> None:
    print("\n=== NEWS / INJURIES / LINEUPS / CARDS / MOMENTUM (last 72h) ===")
    print("TODO: this script has no live web access. Check, for each team:")
    print(f"  - FIFA.com match centre / team pages for {fx.home_id} and {fx.away_id}")
    print("  - the official federation's injury/squad announcements")
    print("  - established sports desks (not aggregators), last 72h")
    print("Save findings as JSON and re-run with --news-file:")
    print(
        '  [{"team_id": "...", "kind": "rating_delta"|"lambda_mult"|"info", '
        '"value": <float>, "note_he": "...", "source": "https://..."}]'
    )


def print_proposed_adjustments(fx: Fixture, findings: list[dict], applied: bool) -> None:
    if not findings:
        return
    print("\n=== PROPOSED ADJUSTMENTS" + (" (applied)" if applied else " (NOT saved — review then apply)") + " ===")
    header = f"{'team_id':8s} {'kind':13s} {'value':>8s}  note_he / source"
    print(header)
    print("-" * len(header))
    for f in findings:
        val = f.get("value", "")
        print(f"{f['team_id']:8s} {f['kind']:13s} {str(val):>8s}  {f.get('note_he','')} [{f.get('source','')}]")
    if not applied:
        print("\nTo commit one of these, call (per finding):")
        for f in findings:
            print(
                f"  ds.add_news_adjustment({fx.match_id!r}, {f['team_id']!r}, "
                f"{f['kind']!r}, {f.get('value', 0)!r}, {f.get('note_he','')!r}, "
                f"{f.get('source','')!r})"
            )
        print("...or re-run this script with --apply to do it automatically.")


def deltas_from_findings(findings: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in findings:
        if f.get("kind") != "rating_delta":
            continue
        out[f["team_id"]] = out.get(f["team_id"], 0.0) + float(f.get("value", 0.0))
    return out


# --- market / EV ----------------------------------------------------------

def print_market(fx: Fixture, base: dict) -> None:
    print("\n=== MARKET / EV ===")
    ds = fx.ds
    printed = False

    if fx.is_group:
        anchor = ds.market_anchor(fx.match_id)
        if anchor:
            printed = True
            print(f"bookmaker anchor ({fx.match_id}): market={anchor['market']}  "
                  f"model={anchor['model']}  flag={anchor['flag']}")
            for key, label in (("p_home", fx.home_id), ("p_draw", "DRAW"), ("p_away", fx.away_id)):
                mkt_p = anchor["market"][key]
                if mkt_p > 0:
                    dec_odds = 1.0 / mkt_p
                    ev = base[key] * dec_odds
                    print(f"  {label:6s} de-vigged market p={mkt_p:.3f} (~{dec_odds:.2f} dec)  "
                          f"model p={base[key]:.3f}  EV(1u stake)={ev:.2f}")

    app = None
    names = dict(zip(ds.teams.team_id, ds.teams.name_en))
    app_odds_path = os.path.join(DATA_DIR, "app_odds.csv")
    if os.path.exists(app_odds_path):
        import pandas as pd
        app_odds = pd.read_csv(app_odds_path)
        h_name, a_name = names.get(fx.home_id), names.get(fx.away_id)
        for _, r in app_odds.iterrows():
            rh = APP_ODDS_ALIASES.get(r.home_team, r.home_team)
            ra = APP_ODDS_ALIASES.get(r.away_team, r.away_team)
            if {rh, ra} == {h_name, a_name}:
                app = r
                break

    if app is not None:
        printed = True
        ph, pd_, pa = base["p_home"], base["p_draw"], base["p_away"]
        # app_odds.csv orders each fixture independently of our home/away
        # convention -- re-orient its odds to fx.home_id/away_id before using
        # them (see predict_value.py's identical `flipped` handling).
        app_home_name = APP_ODDS_ALIASES.get(app.home_team, app.home_team)
        flipped = app_home_name != names.get(fx.home_id)
        odds_home, odds_away = (
            (float(app.odds_away), float(app.odds_home)) if flipped
            else (float(app.odds_home), float(app.odds_away))
        )
        odds_draw = float(app.odds_draw)
        print(f"GOLAZO app odds (oriented to {fx.home_id}/{fx.away_id}): "
              f"home={odds_home}  draw={odds_draw}  away={odds_away}")
        print(f"  EV home={ph*odds_home:.2f}  draw={pd_*odds_draw:.2f}  "
              f"away={pa*odds_away:.2f}  (model P x decimal odds)")

    if not printed:
        print("no market odds available for this fixture.")


# --- main -------------------------------------------------------------------

def print_header(fx: Fixture, base: dict) -> None:
    ds = fx.ds
    names = dict(zip(ds.teams.team_id, ds.teams.name_en))
    kind = "group-stage" if fx.is_group else "knockout"
    print(f"=== PRE-MATCH BRIEFING: {names.get(fx.home_id, fx.home_id)} vs "
          f"{names.get(fx.away_id, fx.away_id)}  [{kind}, match_id={fx.match_id}] ===")
    if not fx.resolvable:
        print("!! WARNING: no stable match_id could be resolved for this pairing "
              "(not a played group game and not a currently-known R32 tie). Any "
              "rating_delta you commit below cannot be attached to a real "
              "news_adjustments match_id yet -- re-run once the tie is official.")
    print(f"FIFA rating: {fx.home_id}={ds.team_rating(fx.home_id):.0f}  "
          f"{fx.away_id}={ds.team_rating(fx.away_id):.0f}")
    print(f"h2h_sup={base['h2h_sup']:+.2f}  form_sup={base['form_sup']:+.2f}")


def print_numbers(label: str, n: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"xG: {n['lambda_home']:.2f} - {n['lambda_away']:.2f}")
    print(f"1X2: home={n['p_home']:.3f}  draw={n['p_draw']:.3f}  away={n['p_away']:.3f}")
    if n["advance_home"] is not None:
        print(f"advance-probability (home side advances tie): {n['advance_home']:.3f}")
    else:
        print("advance-probability: n/a for a single group-stage game "
              "(run knockout.run(ds) for group qualification odds)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match", nargs=2, metavar=("TEAM1", "TEAM2"))
    p.add_argument("--match-id")
    p.add_argument("--news-file", help="JSON findings file (see module docstring)")
    p.add_argument("--apply", action="store_true",
                    help="actually call add_news_adjustment for each finding (default: print only)")
    args = p.parse_args()

    ds = DataStore.load(DATA_DIR)
    fx = resolve_fixture(ds, args)

    base = model_numbers(fx)
    print_header(fx, base)
    print_numbers("MODEL (pre-news)", base)

    findings = load_findings(args.news_file)
    if not findings:
        print_news_todo(fx)
    else:
        if args.apply:
            if not fx.resolvable:
                raise SystemExit(
                    "refusing --apply: this fixture has no resolvable match_id "
                    "(see the WARNING above) -- the adjustment would be attached "
                    "to a match_id the engine can never look up"
                )
            for f in findings:
                ds.add_news_adjustment(
                    fx.match_id, f["team_id"], f["kind"], float(f.get("value", 0.0)),
                    f.get("note_he", ""), f.get("source", ""),
                )
        adjusted = model_numbers(fx, extra_delta=deltas_from_findings(findings))
        print_numbers("MODEL (news-adjusted)", adjusted)
        print_proposed_adjustments(fx, findings, applied=args.apply)

    print_market(fx, base)


if __name__ == "__main__":
    main()
