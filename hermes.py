"""
ממשק Hermes — CLI bridge for the external Telegram agent.

Hermes scrapes the network (betting sites, sports news, FIFA news) and pushes
pre-game adjustments INTO this workspace, then pulls a briefing to decide whether
to alert the user. This file is the language-agnostic contract: Hermes shells out
and reads JSON from stdout.

Examples
--------
# Brazil loses a key player before match C1 — knock 60 FIFA points off Brazil:
python hermes.py update --match C1 --team BRA --kind rating_delta --value -60 \
    --note "ברזיל מאבדת את ויניסיוס - פגיעה בהתקפה" --source "espn.com"

# A betting line moved Morocco's implied win prob up ~10% — bump its goals 12%:
python hermes.py update --match C1 --team MAR --kind lambda_mult --value 1.12 \
    --note "הימור על מרוקו זז משמעותית" --source "pinnacle"

# Pre-tournament rating refresh (permanent, all models) — e.g. new FIFA ranking:
python hermes.py rate --team BRA --value 1730      # set absolute FIFA points
python hermes.py rate --team ARG --delta -120      # or shift relative

# Pull the current briefing (base vs adjusted probs + Hebrew recommendation):
python hermes.py briefing --match C1

# List / clear adjustments:
python hermes.py list --match C1
python hermes.py clear --id <adj_id>

All commands print JSON to stdout (ensure_ascii=False, so Hebrew is readable).
"""

from __future__ import annotations

import argparse
import json
import os

from src.models import DataStore

DATA = os.path.join(os.path.dirname(__file__), "data")


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_update(args) -> None:
    ds = DataStore.load(DATA)
    adj_id = ds.add_news_adjustment(
        match_id=args.match,
        team_id=args.team,
        kind=args.kind,
        value=args.value,
        note_he=args.note,
        source=args.source,
    )
    _print({"ok": True, "adj_id": adj_id, "briefing": ds.match_briefing(args.match)})


def cmd_briefing(args) -> None:
    ds = DataStore.load(DATA)
    _print(ds.match_briefing(args.match))


def cmd_list(args) -> None:
    ds = DataStore.load(DATA)
    df = ds.news if args.match is None else ds.news[ds.news.match_id == args.match]
    _print({"adjustments": df.to_dict("records")})


def cmd_clear(args) -> None:
    ds = DataStore.load(DATA)
    ok = ds.deactivate_adjustment(args.id)
    _print({"ok": ok, "adj_id": args.id})


def cmd_rate(args) -> None:
    """Pre-tournament rating refresh: permanently set a team's FIFA points.

    Use this (not `update`) when Hermes learns of a lasting strength change
    before the tournament — a new FIFA ranking release, a long-term injury, a
    squad-list shock. It rewrites teams.csv so every downstream model (groups,
    knockout, bonus) uses the new strength. `update` stays for single-match,
    live news that should not change the team's base rating.
    """
    ds = DataStore.load(DATA)
    if args.delta is not None:
        new_val = ds.team_rating(args.team) + args.delta
    elif args.value is not None:
        new_val = args.value
    else:
        raise SystemExit("rate: provide --value or --delta")
    _print({"ok": True, "result": ds.set_team_rating(args.team, new_val)})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hermes <-> WorldCup2026 bridge")
    sub = p.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("update", help="add a pre-game news adjustment")
    u.add_argument("--match", required=True)
    u.add_argument("--team", required=True)
    u.add_argument("--kind", required=True, choices=["rating_delta", "lambda_mult", "info"])
    u.add_argument("--value", type=float, default=0.0)
    u.add_argument("--note", required=True)
    u.add_argument("--source", default="")
    u.set_defaults(func=cmd_update)

    b = sub.add_parser("briefing", help="base vs adjusted probs + recommendation")
    b.add_argument("--match", required=True)
    b.set_defaults(func=cmd_briefing)

    l = sub.add_parser("list", help="list adjustments")
    l.add_argument("--match", default=None)
    l.set_defaults(func=cmd_list)

    c = sub.add_parser("clear", help="deactivate an adjustment by id")
    c.add_argument("--id", required=True)
    c.set_defaults(func=cmd_clear)

    r = sub.add_parser("rate", help="pre-tournament: set/adjust a team's FIFA points")
    r.add_argument("--team", required=True)
    r.add_argument("--value", type=float, default=None, help="absolute new FIFA points")
    r.add_argument("--delta", type=float, default=None, help="add to current FIFA points")
    r.set_defaults(func=cmd_rate)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
