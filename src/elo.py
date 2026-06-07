"""
src/elo.py — World Football Elo, derived from match results (no external snapshot).

Why this exists
---------------
The backtest needs a *strength rating as it stood before* each historical
tournament. FIFA points for past dates are not cleanly available, and the review
warned us off hand-keying stale numbers. So instead of sourcing ratings we
*derive* them: run the standard World Football Elo algorithm forward over real
results, stopping at any cut-off date. This is:

  * reproducible — same results in, same ratings out;
  * leakage-free — `snapshot_before(date)` only consumes matches strictly before
    the date, so a tournament can never "see" its own outcomes; and
  * dual-use — the same ratings give production the Elo column the review asked
    for, not just the holdout.

Algorithm (eloratings.net conventions)
--------------------------------------
    dr   = (Rh + home_adv) - Ra              # home_adv = 0 at neutral venues
    We   = 1 / (10**(-dr/400) + 1)           # home expected score
    W    = 1.0 win / 0.5 draw / 0.0 loss     # home actual score
    G    = goal-difference multiplier        # 1, 1.5, or (11+gd)/8 for gd>=3
    K    = weight * G                        # weight = match importance
    Rh' += K * (W - We);  Ra' -= K * (W - We)

A match dict is::

    {"date": "YYYY-MM-DD", "home": id, "away": id, "gh": int, "ga": int,
     "neutral": bool, "weight": float|None, "comp": str|None}

`weight` may be given directly; otherwise it is inferred from `comp`/league text
via `weight_for`. Teams default to START (1500) on first appearance.
"""

from __future__ import annotations

START = 1500.0          # debut rating for an unseen team
HOME_ADV = 100.0        # Elo points added to the home side at a non-neutral venue

# Match-importance weights (eloratings.net index). Higher = a result moves the
# rating more. Keyed by our engine comp labels plus a couple of league hints.
WEIGHTS = {
    "friendly":    20.0,
    "qualifier":   40.0,
    "competitive": 40.0,   # Nations League & similar
    "group":       50.0,   # continental-championship group stage
    "knockout":    50.0,
    "semifinal":   50.0,
    "final":       50.0,
    "worldcup":    60.0,   # any World Cup finals match
}


def weight_for(comp: str | None = None, league: str | None = None) -> float:
    """Importance weight for a match from its comp label and/or league name.

    World Cup *finals* matches are 60 regardless of stage; continental-final
    stages are 50; qualifiers/Nations-League 40; friendlies 20. The league name
    is checked first because it distinguishes a World Cup group game (60) from a
    Euro group game (50), which the stage label alone cannot.
    """
    lt = str(league or "").lower()
    if "world cup" in lt and "qualif" not in lt:
        return WEIGHTS["worldcup"]
    if "friendl" in lt:
        return WEIGHTS["friendly"]
    if "qualif" in lt:
        return WEIGHTS["qualifier"]
    c = str(comp or "").strip().lower()
    if c in WEIGHTS:
        return WEIGHTS[c]
    if "friendl" in c:
        return WEIGHTS["friendly"]
    if "qualif" in c:
        return WEIGHTS["qualifier"]
    if "semi" in c or "final" in c or "knock" in c:
        return WEIGHTS["knockout"]
    if "group" in c:
        return WEIGHTS["group"]
    return WEIGHTS["competitive"]


def _gd_multiplier(gd: int) -> float:
    """Goal-difference multiplier G (eloratings.net): bigger wins move more."""
    gd = abs(int(gd))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected_home(rh: float, ra: float, neutral: bool = True) -> float:
    """Home team's expected score (0..1) given the two ratings."""
    dr = (rh - ra) + (0.0 if neutral else HOME_ADV)
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def update_pair(
    rh: float, ra: float, gh: int, ga: int,
    weight: float, neutral: bool = True,
) -> tuple[float, float]:
    """Return the post-match (home, away) ratings for one result."""
    we = expected_home(rh, ra, neutral=neutral)
    w = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
    k = weight * _gd_multiplier(gh - ga)
    delta = k * (w - we)
    return rh + delta, ra - delta


def _match_date(m) -> str:
    return str(m.get("date") or "")


def run(matches, start: float = START, until: str | None = None) -> dict[str, float]:
    """Ratings after processing `matches` chronologically.

    until: optional 'YYYY-MM-DD'; only matches with date < until are consumed
    (strict — a tournament starting on `until` never sees its own games).
    Unseen teams enter at `start`. Matches missing a score are skipped.
    """
    ratings: dict[str, float] = {}
    ordered = sorted(matches, key=_match_date)
    for m in ordered:
        d = _match_date(m)
        if until is not None and d and d >= until:
            continue
        gh, ga = m.get("gh"), m.get("ga")
        if gh is None or ga is None:
            continue
        home, away = str(m["home"]), str(m["away"])
        rh = ratings.get(home, start)
        ra = ratings.get(away, start)
        wt = m.get("weight")
        if wt is None:
            wt = weight_for(m.get("comp"), m.get("league"))
        neutral = bool(m.get("neutral", True))
        ratings[home], ratings[away] = update_pair(rh, ra, int(gh), int(ga),
                                                    float(wt), neutral=neutral)
    return ratings


def snapshot_before(matches, date: str, start: float = START) -> dict[str, float]:
    """Pre-`date` ratings — convenience wrapper over `run(..., until=date)`."""
    return run(matches, start=start, until=date)


def recenter(ratings: dict[str, float], teams=None, mean: float = 1500.0) -> dict[str, float]:
    """Shift ratings so the mean over `teams` (or all) equals `mean`.

    A pure additive shift: every pairwise gap (and therefore every supremacy the
    engine derives via (rating_home-rating_away)/K) is preserved exactly, while
    the absolute level is moved onto the FIFA-points scale the engine's
    total-goals term expects (deviation from FIFA_MEAN=1500). Use this before
    feeding derived Elo into engine.expected_goals as a `rating`.
    """
    keys = list(teams) if teams is not None else list(ratings)
    vals = [ratings[k] for k in keys if k in ratings]
    if not vals:
        return dict(ratings)
    shift = mean - sum(vals) / len(vals)
    return {k: v + shift for k, v in ratings.items()}
