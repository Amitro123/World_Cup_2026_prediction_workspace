"""
מנוע ההסתברויות — Probability & simulation engine for World Cup 2026.

The engine exposes a single `ProbabilityModel` interface so the dashboard, the
data layer and the knockout simulation all share one model. As of the FIFA-points
upgrade the model is a **Dixon-Coles correlated Poisson** built on FIFA ranking
points, optionally blended with expert scorelines.

Model summary
-------------
1. Team strength = FIFA ranking points (≈1400-1900 scale, mean ≈1500). This is a
   neutral measure of strength, unlike group-winner odds which conflate strength
   with how easy a team's group is. (That is why Brazil — short group-winner odds
   in an easy group — sits ~6th here, while Spain stays top-2, matching the docs.)
2. Pre-match goals (Cowork-tuned formula):
       sup   = (rating_home - rating_away) / K          (+ home advantage)
       total = BASE_TOTAL + |r_h + r_a - 2*FIFA_MEAN| / 4000   (stronger ties score more)
       λ_home = max(MIN_LAMBDA, (total + sup) / 2)
       λ_away = max(MIN_LAMBDA, (total - sup) / 2)
   The win/draw/loss grid uses a Dixon-Coles low-score correction (DC_RHO) so
   0-0 / 1-0 / 0-1 / 1-1 are dependent rather than independent.
3. Optional expert blend: λ is pulled toward an expert scoreline target with
   weight (1 - EXPERT_W) on the expert, EXPERT_W on the model.
4. In-play: only the *remaining* share of each λ still applies (scaled by
   remaining minutes), added to the current score; recomputed as independent
   Poisson (the Dixon-Coles correction is a full-match calibration).

Assumptions (documented so you can challenge / replace them):
- 90 minutes of regular time; stoppage time ignored.
- Home advantage is a flat supremacy bump (HOME_SUP goals); neutral=True drops it
  for knockout games at neutral venues.
- Swap the `ProbabilityModel` class to upgrade the engine; callers use only
  `pre_match`, `in_play`, and the sampling helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Tunable model constants (FIFA-points Dixon-Coles) -----------------------
K = 200.0            # FIFA points per 1 goal of supremacy
BASE_TOTAL = 2.6     # neutral expected total goals for an even tie
FIFA_MEAN = 1500.0   # reference FIFA rating (strength scaling anchor)
DC_RHO = -0.06       # Dixon-Coles low-score dependence
HOME_SUP = 0.35      # home advantage, in goals of supremacy (added to `sup`)
EXPERT_W = 0.55      # weight on the model vs an expert scoreline target
MIN_LAMBDA = 0.18    # floor on any expected-goals value
MAX_GOALS = 8        # truncation for the Poisson scoreline grid

# --- Head-to-head (past meetings) signal ------------------------------------
# FIFA points already capture most of a team's strength, so H2H is a small,
# bounded nudge: a team that has historically beaten this specific opponent gets
# a little extra supremacy. Friendlies count less; older meetings decay; small
# samples shrink toward zero so a single game barely moves the line.
H2H_WEIGHT = 0.18         # goals of supremacy per goal of weighted-avg H2H margin
H2H_CAP = 0.50            # max |supremacy| H2H may contribute (goals)
H2H_SHRINK = 3.0          # pseudo-count: small samples shrink toward zero
H2H_FRIENDLY_W = 0.4      # a friendly counts less than a competitive game
H2H_HALFLIFE_YEARS = 6.0  # recency half-life for down-weighting old meetings

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
    """Group-winner moneyline -> 0-100 strength proxy (legacy / display only)."""
    return round(100.0 * american_to_prob(odds), 2)


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _dc_tau(i: int, j: int, lam_home: float, lam_away: float) -> float:
    """Dixon-Coles low-score dependence factor for scoreline (i, j)."""
    if i == 0 and j == 0:
        return 1.0 - lam_home * lam_away * DC_RHO
    if i == 0 and j == 1:
        return 1.0 + lam_home * DC_RHO
    if i == 1 and j == 0:
        return 1.0 + lam_away * DC_RHO
    if i == 1 and j == 1:
        return 1.0 - DC_RHO
    return 1.0


def h2h_supremacy(meetings, ref_year: int | None = None) -> float:
    """Weighted head-to-head supremacy (goals), from the HOME team's perspective.

    meetings: iterable of dicts, each oriented to the home team:
        gd    int  home_goals - away_goals in that past meeting
        comp  str  'friendly' (down-weighted) or anything else (competitive)
        year  int  optional, for recency weighting against ref_year

    Returns a bounded supremacy delta in goals (positive favours the home team).
    Friendlies count less; older games decay; small samples shrink toward zero.
    """
    num = den = 0.0
    for m in meetings:
        comp = str(m.get("comp", "")).strip().lower()
        w = H2H_FRIENDLY_W if comp.startswith("f") else 1.0
        year = m.get("year")
        if ref_year and year:
            try:
                age = max(0, int(ref_year) - int(year))
                w *= 0.5 ** (age / H2H_HALFLIFE_YEARS)
            except (TypeError, ValueError):
                pass
        num += w * float(m["gd"])
        den += w
    if den <= 0:
        return 0.0
    avg = num / den
    shrunk = avg * (den / (den + H2H_SHRINK))   # shrink small samples toward 0
    bump = H2H_WEIGHT * shrunk
    return max(-H2H_CAP, min(H2H_CAP, bump))


def expected_goals(
    rating_home: float,
    rating_away: float,
    neutral: bool = False,
    expert: tuple[float, float] | None = None,
    expert_w: float = EXPERT_W,
    h2h_sup: float = 0.0,
) -> tuple[float, float]:
    """Map two FIFA ratings to (lambda_home, lambda_away).

    neutral=True drops the home advantage (knockout games at neutral venues).
    expert=(home_goals, away_goals) blends the model toward an expert scoreline.
    h2h_sup: extra supremacy (goals) from past meetings; see `h2h_supremacy`.
        Applied regardless of venue — history travels with the matchup.
    """
    sup = (rating_home - rating_away) / K
    if not neutral:
        sup += HOME_SUP
    sup += h2h_sup
    total = BASE_TOTAL + abs(rating_home + rating_away - 2.0 * FIFA_MEAN) / 4000.0
    lam_home = max(MIN_LAMBDA, (total + sup) / 2.0)
    lam_away = max(MIN_LAMBDA, (total - sup) / 2.0)
    if expert is not None:
        eh, ea = expert
        lam_home = max(MIN_LAMBDA, expert_w * lam_home + (1.0 - expert_w) * eh)
        lam_away = max(MIN_LAMBDA, expert_w * lam_away + (1.0 - expert_w) * ea)
    return lam_home, lam_away


def _grid_probs(
    lam_home: float,
    lam_away: float,
    base_home: int = 0,
    base_away: int = 0,
    dixon_coles: bool = False,
) -> dict[str, float]:
    """Win/draw/loss from two Poisson tallies added to a base score.

    dixon_coles=True applies the low-score dependence correction (pre-match only;
    not meaningful once a base score is in play).
    """
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        ph = _poisson_pmf(i, lam_home)
        for j in range(MAX_GOALS + 1):
            pa = _poisson_pmf(j, lam_away)
            tau = _dc_tau(i, j, lam_home, lam_away) if dixon_coles else 1.0
            prob = ph * pa * tau
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


def probs_from_lambdas(
    lam_home: float, lam_away: float, dixon_coles: bool = True
) -> dict[str, float]:
    """Public helper: win/draw/loss directly from a pair of expected-goals."""
    out = _grid_probs(lam_home, lam_away, dixon_coles=dixon_coles)
    out["lambda_home"] = lam_home
    out["lambda_away"] = lam_away
    return out


@dataclass
class ProbabilityModel:
    """FIFA-points Dixon-Coles model. Swap this class to upgrade the engine."""

    base_total: float = BASE_TOTAL
    k: float = K
    home_sup: float = HOME_SUP

    def pre_match(
        self,
        rating_home: float,
        rating_away: float,
        neutral: bool = False,
        expert: tuple[float, float] | None = None,
        h2h_sup: float = 0.0,
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(
            rating_home, rating_away, neutral=neutral, expert=expert, h2h_sup=h2h_sup
        )
        out = _grid_probs(lam_h, lam_a, dixon_coles=True)
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
        expert: tuple[float, float] | None = None,
        h2h_sup: float = 0.0,
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(rating_home, rating_away, expert=expert, h2h_sup=h2h_sup)
        remaining = max(0.0, (90 - minute) / 90.0)
        out = _grid_probs(
            lam_h * remaining,
            lam_a * remaining,
            base_home=home_goals,
            base_away=away_goals,
            dixon_coles=False,
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
    rating_home: float,
    rating_away: float,
    rng,
    neutral: bool = False,
    expert: tuple[float, float] | None = None,
    h2h_sup: float = 0.0,
) -> tuple[int, int]:
    lam_h, lam_a = expected_goals(
        rating_home, rating_away, neutral=neutral, expert=expert, h2h_sup=h2h_sup
    )
    return sample_poisson(lam_h, rng), sample_poisson(lam_a, rng)


def knockout_winner(
    rating_home: float, rating_away: float, rng, neutral: bool = True, h2h_sup: float = 0.0
) -> int:
    """Return 0 if home advances, 1 if away. Draws resolve (ET/penalties) by
    splitting proportionally to each side's win strength."""
    hg, ag = sample_score(rating_home, rating_away, rng, neutral=neutral, h2h_sup=h2h_sup)
    if hg > ag:
        return 0
    if ag > hg:
        return 1
    probs = ProbabilityModel().pre_match(
        rating_home, rating_away, neutral=neutral, h2h_sup=h2h_sup
    )
    ph, pa = probs["p_home"], probs["p_away"]
    return 0 if rng.random() < ph / (ph + pa) else 1
