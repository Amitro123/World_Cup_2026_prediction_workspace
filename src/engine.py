"""
מנוע ההסתברויות — Probability & simulation engine for World Cup 2026.

The engine is intentionally simple and explainable, and exposes a single
`ProbabilityModel` interface so a more advanced model can be plugged in later
without touching the dashboard or the data layer.

Model summary
-------------
1. Power ratings: bookmaker group-winner moneylines -> implied probability
   (American-odds conversion). The raw implied probability is used as a global
   strength proxy on a 0-100 scale (stronger favourite -> higher rating).
2. Pre-match: the rating gap between the two teams maps to expected goals
   (lambda) for each side via an exponential link, plus a fixed home multiplier.
   Two independent Poisson distributions give the win/draw/loss probabilities.
3. In-play: only the *remaining* share of each lambda still applies
   (scaled by remaining minutes). Future goals are added to the current score
   and the win/draw/loss grid is recomputed.

Assumptions (documented so you can challenge / replace them):
- Goals follow independent Poisson processes (no explicit correlation / red
  cards / momentum). Good enough for an explainable baseline.
- 90 minutes of regular time; stoppage time is ignored.
- Home advantage is a flat multiplier; host nations get no extra boost beyond
  being listed as home where applicable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Tunable model constants -------------------------------------------------
BASE_GOALS = 1.30   # league-average expected goals for an evenly matched side
ALPHA = 1.00        # how strongly a rating gap swings expected goals
HOME_MULT = 1.15    # flat home-side expected-goals multiplier
MAX_GOALS = 10      # truncation for the Poisson scoreline grid

# Status thresholds (probability that my pick is still the final outcome)
ON_TRACK_MIN = 0.55
AT_RISK_MIN = 0.25


def american_to_prob(odds: float) -> float:
    """Convert American moneyline odds to an implied probability (with vig)."""
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def odds_to_power_rating(odds: float) -> float:
    """Group-winner moneyline -> 0-100 global strength proxy."""
    return round(100.0 * american_to_prob(odds), 2)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def expected_goals(
    rating_home: float, rating_away: float, neutral: bool = False
) -> tuple[float, float]:
    """Map two power ratings to (lambda_home, lambda_away).

    neutral=True drops the home multiplier (knockout games at neutral venues).
    """
    d = (rating_home - rating_away) / 100.0
    home_mult = 1.0 if neutral else HOME_MULT
    lam_home = BASE_GOALS * math.exp(ALPHA * d) * home_mult
    lam_away = BASE_GOALS * math.exp(-ALPHA * d)
    return lam_home, lam_away


def _grid_probs(
    lam_home: float,
    lam_away: float,
    base_home: int = 0,
    base_away: int = 0,
) -> dict[str, float]:
    """Win/draw/loss from two independent Poisson tallies added to a base score."""
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        ph = _poisson_pmf(i, lam_home)
        for j in range(MAX_GOALS + 1):
            pa = _poisson_pmf(j, lam_away)
            prob = ph * pa
            fh, fa = base_home + i, base_away + j
            if fh > fa:
                p_home += prob
            elif fh == fa:
                p_draw += prob
            else:
                p_away += prob
    total = p_home + p_draw + p_away
    return {
        "p_home": p_home / total,
        "p_draw": p_draw / total,
        "p_away": p_away / total,
    }


@dataclass
class ProbabilityModel:
    """Baseline Poisson model. Swap this class to upgrade the engine."""

    base_goals: float = BASE_GOALS
    alpha: float = ALPHA
    home_mult: float = HOME_MULT

    def pre_match(
        self, rating_home: float, rating_away: float, neutral: bool = False
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(rating_home, rating_away, neutral=neutral)
        out = _grid_probs(lam_h, lam_a)
        out["lambda_home"] = lam_h
        out["lambda_away"] = lam_a
        return out

    def in_play(
        self,
        rating_home: float,
        rating_away: float,
        minute: int,
        home_goals: int,
        away_goals: int,
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(rating_home, rating_away)
        remaining = max(0.0, (90 - minute) / 90.0)
        out = _grid_probs(
            lam_h * remaining,
            lam_a * remaining,
            base_home=home_goals,
            base_away=away_goals,
        )
        out["lambda_home"] = lam_h
        out["lambda_away"] = lam_a
        out["remaining_fraction"] = remaining
        return out


# --- My-prediction evaluation ------------------------------------------------

def pick_probability(probs: dict[str, float], pick: str) -> float:
    """Probability that a H/D/A pick matches the (eventual) outcome."""
    return {"H": probs["p_home"], "D": probs["p_draw"], "A": probs["p_away"]}[pick]


def prediction_status(prob: float) -> str:
    if prob >= ON_TRACK_MIN:
        return "ON_TRACK"
    if prob >= AT_RISK_MIN:
        return "AT_RISK"
    return "ALMOST_DEAD"


def outcome_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"


# --- sampling helpers (for Monte-Carlo tournament simulation) ----------------

def sample_poisson(lam: float, rng) -> int:
    """Knuth's algorithm — Poisson sample without numpy."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def sample_score(
    rating_home: float, rating_away: float, rng, neutral: bool = False
) -> tuple[int, int]:
    lam_h, lam_a = expected_goals(rating_home, rating_away, neutral=neutral)
    return sample_poisson(lam_h, rng), sample_poisson(lam_a, rng)


def knockout_winner(
    rating_home: float, rating_away: float, rng, neutral: bool = True
) -> int:
    """Return 0 if home advances, 1 if away. Draws resolve (ET/penalties) by
    splitting proportionally to each side's win strength."""
    hg, ag = sample_score(rating_home, rating_away, rng, neutral=neutral)
    if hg > ag:
        return 0
    if ag > hg:
        return 1
    probs = ProbabilityModel().pre_match(rating_home, rating_away, neutral=neutral)
    ph, pa = probs["p_home"], probs["p_away"]
    return 0 if rng.random() < ph / (ph + pa) else 1
