"""
src/playerprops.py — per-match player goal/assist props (model + optional market).

The engine is team-level: it predicts a team's expected goals (lambda) in a match.
`src/bonus.py` already turns that into *tournament-aggregate* player goals via
``exp_goals = goal_share * team_goals_over_the_run``. This module does the same at
the **single-match** level — the shape a bookmaker prices (see the bet365 "Player
to Score or Assist" market) — and, when market odds are supplied, de-vigs them so
the model and the book can sit side by side.

Model
-----
Given a player's ``goal_share`` / ``assist_share`` (fraction of the team's goals
they finish / assist) and the team's expected goals in this match ``team_lambda``:

    exp_goals   = goal_share   * team_lambda
    exp_assists = assist_share * team_lambda

Goals are Poisson, so the probability of *at least one*:

    P(score)  = 1 - exp(-exp_goals)
    P(assist) = 1 - exp(-exp_assists)

A player cannot assist their own goal, so "score **or** assist" events fall on
different goals and are treated as independent Poisson streams with the combined
rate (a documented approximation):

    P(score or assist) = 1 - exp(-(exp_goals + exp_assists))

Market
------
Anytime-scorer / assist markets are NOT mutually exclusive, so they can only be
de-vigged with a flat per-selection margin (same limitation as oddslib.implied_one).
`market_props_from_row` reads decimal ``score_odds`` / ``assist_odds`` and returns
de-vigged implied probabilities, so the dashboard can show model% vs market%.

Pure module: no IO, no network — fully unit-tested.
"""

from __future__ import annotations

import math

from . import oddslib


def p_at_least_one(exp_events: float) -> float:
    """Poisson P(N>=1) for a rate of `exp_events`. Clamped to [0, 1]."""
    lam = max(0.0, float(exp_events))
    return 1.0 - math.exp(-lam)


def player_match_props(goal_share: float, assist_share: float,
                       team_lambda: float) -> dict:
    """Model props for one player in one match.

    goal_share / assist_share: fraction of the team's goals this player finishes
        / assists (from players.csv).
    team_lambda: the team's expected goals in THIS match (engine output).

    Returns exp_goals, exp_assists and the three at-least-one probabilities.
    """
    gl = max(0.0, float(goal_share))
    al = max(0.0, float(assist_share))
    lam = max(0.0, float(team_lambda))
    exp_goals = gl * lam
    exp_assists = al * lam
    return {
        "exp_goals": exp_goals,
        "exp_assists": exp_assists,
        "p_score": p_at_least_one(exp_goals),
        "p_assist": p_at_least_one(exp_assists),
        # different goals -> independent streams -> combined rate
        "p_score_or_assist": p_at_least_one(exp_goals + exp_assists),
    }


def _f(row, key):
    """NaN/typed-safe float read from a dict or pandas row; None if unusable."""
    v = row.get(key) if hasattr(row, "get") else (row[key] if key in row else None)
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def market_props_from_row(row, margin: float = 0.0) -> dict:
    """Read one players_market.csv row -> de-vigged market props.

    Accepts decimal ``score_odds`` / ``assist_odds`` (each optional). `margin`
    removes a flat per-selection vig (these markets are not exhaustive, so a
    proportional de-vig is impossible without the "no" price). Missing odds yield
    a None for that field rather than raising, so a half-filled sheet degrades
    gracefully.
    """
    so, ao = _f(row, "score_odds"), _f(row, "assist_odds")
    out: dict[str, float | None] = {"p_score": None, "p_assist": None,
                                    "p_score_or_assist": None}
    if so is not None and so > 0:
        out["p_score"] = oddslib.implied_one(so, margin=margin)
    if ao is not None and ao > 0:
        out["p_assist"] = oddslib.implied_one(ao, margin=margin)
    # an explicit "score or assist" price, if the book lists it
    soa = _f(row, "score_or_assist_odds")
    if soa is not None and soa > 0:
        out["p_score_or_assist"] = oddslib.implied_one(soa, margin=margin)
    return out


def compare_props(model: dict, market: dict, flag_threshold: float = 0.12) -> dict:
    """Model-vs-market gaps for a player's props, per available selection.

    For each of p_score / p_assist / p_score_or_assist present in BOTH dicts,
    reports the gap (model - market) and flags it when |gap| >= flag_threshold.
    Selections missing from either side are skipped.
    """
    keys = ("p_score", "p_assist", "p_score_or_assist")
    out: dict[str, dict] = {}
    for k in keys:
        mv, bv = model.get(k), market.get(k)
        if mv is None or bv is None:
            continue
        gap = float(mv) - float(bv)
        out[k] = {
            "model": float(mv),
            "market": float(bv),
            "gap": gap,
            "flag": abs(gap) >= flag_threshold,
        }
    return out
