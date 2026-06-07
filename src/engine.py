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

import datetime as _dt
import math
from dataclasses import dataclass

# --- Tunable model constants (FIFA-points Dixon-Coles) -----------------------
K = 240.0            # FIFA points per 1 goal of supremacy (calibrated to the
                     # bookmaker market 2026-06: K=200 was overconfident on
                     # favourites; K=240 is the knee of the market-KL/holdout-
                     # Brier trade-off — most of the alignment gain, ~0.4% Brier cost)
BASE_TOTAL = 2.6     # neutral expected total goals for an even tie
FIFA_MEAN = 1500.0   # reference FIFA rating (strength scaling anchor)
DC_RHO = -0.06       # Dixon-Coles low-score dependence
HOME_SUP = 0.35      # home advantage, in goals of supremacy (added to `sup`)
EXPERT_W = 0.85      # weight on the model vs an expert scoreline target; 0.55
                     # over-weighted the expert and pulled the model away from
                     # the market — 0.85 (15% expert) best matches market 1X2

# --- Host nations (World Cup 2026 co-hosts) ----------------------------------
# 2026 is played across the USA, Mexico and Canada, so *every* match is at a
# neutral venue for the visiting teams EXCEPT when a host nation plays at home —
# there a real crowd advantage applies. Group games are therefore treated as
# neutral unless the home_id is one of these three; knockout games stay neutral.
HOSTS = frozenset({"USA", "MEX", "CAN"})
MIN_LAMBDA = 0.18    # floor on any expected-goals value
MAX_GOALS = 8        # truncation for the Poisson scoreline grid

# --- Optional Elo blend ------------------------------------------------------
# FIFA points are the default strength input. A second source (World Football Elo)
# can be blended in: ELO_WEIGHT is the share given to Elo, 1-ELO_WEIGHT to FIFA.
# Elo and FIFA live on different scales, so we blend in z-score space and map the
# result back onto the FIFA scale — that way the engine's K / FIFA_MEAN constants
# stay valid and ELO_WEIGHT=0.0 reproduces the pure-FIFA model exactly. The value
# is 0.0 by default and only raised if a backtest shows it lowers the Brier score.
ELO_WEIGHT = 0.0


def blend_strength(
    fifa: float,
    elo: float,
    weight: float,
    fifa_mean: float,
    fifa_std: float,
    elo_mean: float,
    elo_std: float,
) -> float:
    """Blend a FIFA rating with an Elo rating, expressed back in FIFA units.

    weight=0 -> pure FIFA (unchanged); weight=1 -> Elo mapped onto FIFA's
    mean/spread. Both are converted to z-scores against their own population so
    the blend is scale-free, then rescaled to the FIFA distribution so downstream
    constants (K, FIFA_MEAN) keep their meaning. Falls back to FIFA if Elo is
    missing or a std is degenerate.
    """
    if weight <= 0 or elo is None or elo != elo:  # NaN-safe
        return fifa
    if fifa_std <= 0 or elo_std <= 0:
        return fifa
    zf = (fifa - fifa_mean) / fifa_std
    ze = (elo - elo_mean) / elo_std
    z = (1.0 - weight) * zf + weight * ze
    return fifa_mean + fifa_std * z


# Knockout draws go to extra time + penalties. The stronger team keeps a real but
# LIMITED edge there: shootouts in particular are close to a coin flip regardless
# of the skill gap. We therefore resolve a knockout draw proportionally to win
# strength but cap the favourite's advance probability at SHOOTOUT_CAP (so even a
# huge favourite is at most ~58% to survive ET/pens, not 80%+).
SHOOTOUT_CAP = 0.58

# --- Head-to-head (past meetings) signal ------------------------------------
# FIFA points already capture most of a team's strength, so H2H is a small,
# bounded nudge: a team that has historically beaten this specific opponent gets
# a little extra supremacy. The status of each past meeting is graded (a World Cup
# final beating means more than a friendly); older meetings decay; small samples
# shrink toward zero so a single game barely moves the line; and two teams that
# never met contribute exactly zero.
H2H_WEIGHT = 0.18         # goals of supremacy per goal of weighted-avg H2H margin
H2H_CAP = 0.50            # max |supremacy| H2H may contribute (goals)
H2H_SHRINK = 3.0          # pseudo-count: small samples shrink toward zero
H2H_HALFLIFE_YEARS = 6.0  # recency half-life for down-weighting old meetings

# Weight per match status — a higher-stakes meeting carries more signal than a
# friendly. Free-text `comp` values are mapped by keyword in `_comp_weight`.
H2H_COMP_WEIGHTS = {
    "friendly":    0.40,   # least meaningful — experimental line-ups, low stakes
    "qualifier":   0.85,   # qualifiers / minor competitive
    "group":       1.00,   # tournament group stage = baseline competitive
    "competitive": 1.00,   # generic competitive (unknown stage)
    "knockout":    1.25,   # round of 32/16, quarter-final — elimination pressure
    "semifinal":   1.40,   # semi-final
    "final":       1.50,   # final — the highest-stakes meeting
}
H2H_FRIENDLY_W = H2H_COMP_WEIGHTS["friendly"]  # kept for back-compat / readability


def _comp_weight(comp) -> float:
    """Map a meeting's `comp`/stage label to its head-to-head weight.

    Accepts both the canonical keys above and free-text values (e.g. from the
    web scraper) like 'World Cup semi-final', 'Euro qualifier', 'friendlies'.
    Order matters: 'semifinal'/'quarterfinal' contain 'final', so check the more
    specific stage words first. Unknown competitive values fall back to 1.0.
    """
    c = str(comp or "").strip().lower()
    if not c:
        return H2H_COMP_WEIGHTS["competitive"]
    if c in H2H_COMP_WEIGHTS:
        return H2H_COMP_WEIGHTS[c]
    if c.startswith("f"):                       # friendly / friendlies
        return H2H_COMP_WEIGHTS["friendly"]
    if "semi" in c:
        return H2H_COMP_WEIGHTS["semifinal"]
    if any(k in c for k in ("quarter", "knockout", "round of", "last 16",
                            "last 8", "r16", "r32", "play-off", "playoff")):
        return H2H_COMP_WEIGHTS["knockout"]
    if "final" in c:                            # plain final (after semi/quarter)
        return H2H_COMP_WEIGHTS["final"]
    if "qualif" in c:
        return H2H_COMP_WEIGHTS["qualifier"]
    if "group" in c:
        return H2H_COMP_WEIGHTS["group"]
    return H2H_COMP_WEIGHTS["competitive"]

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
        comp  str  match status/stage — 'friendly', 'group', 'knockout',
                   'semifinal', 'final', ... (graded by `_comp_weight`)
        year  int  optional, for recency weighting against ref_year

    Returns a bounded supremacy delta in goals (positive favours the home team).
    Higher-stakes meetings count more; older games decay; small samples shrink
    toward zero; no meetings -> exactly 0.0 (never-met teams are unaffected).
    """
    num = den = 0.0
    for m in meetings:
        w = _comp_weight(m.get("comp", ""))
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


# --- Momentum / recent-form signal ------------------------------------------
# How a team is ARRIVING at the tournament — its last handful of matches. FIFA
# points are a slow-moving baseline; momentum captures the recent swing (a team
# on a winning streak arrives sharper than one limping in on losses). Like H2H it
# is a small, bounded supremacy nudge: each side gets a form score, and the
# DIFFERENCE between the two scores nudges supremacy. No recent matches -> the
# team's form score is exactly 0, so a fixture with no form data is unaffected.
FORM_WEIGHT = 0.30        # goals of supremacy per unit of form-score difference
FORM_CAP = 0.35           # max |supremacy| momentum may contribute (goals)
FORM_SHRINK = 2.5         # pseudo-count: few recent games shrink the score to 0
FORM_HALFLIFE_DAYS = 180  # recency half-life (~6 months) for down-weighting
FORM_GD_COEF = 0.25       # how much each goal of margin adds beyond the W/D/L point
FORM_GD_CAP = 3           # clamp a single result's margin (a 7-0 ≈ a 3-0 for form)

# Weight per match status for FORM. Milder than H2H's friendly penalty: warm-up
# friendlies are the NORM before a World Cup, so they still carry real signal,
# while competitive form (qualifiers, AFCON, Nations League) counts a bit more.
FORM_COMP_WEIGHTS = {
    "friendly":    0.60,
    "qualifier":   1.00,
    "group":       1.00,
    "competitive": 1.00,
    "knockout":    1.15,
    "semifinal":   1.20,
    "final":       1.25,
}


def _form_comp_weight(comp) -> float:
    """Map a match's `comp`/stage label to its momentum weight (see _comp_weight)."""
    c = str(comp or "").strip().lower()
    if not c:
        return FORM_COMP_WEIGHTS["competitive"]
    if c in FORM_COMP_WEIGHTS:
        return FORM_COMP_WEIGHTS[c]
    if c.startswith("f"):
        return FORM_COMP_WEIGHTS["friendly"]
    if "semi" in c:
        return FORM_COMP_WEIGHTS["semifinal"]
    if any(k in c for k in ("quarter", "knockout", "round of", "last 16",
                            "last 8", "r16", "r32", "play-off", "playoff")):
        return FORM_COMP_WEIGHTS["knockout"]
    if "final" in c:
        return FORM_COMP_WEIGHTS["final"]
    if "qualif" in c:
        return FORM_COMP_WEIGHTS["qualifier"]
    if "group" in c:
        return FORM_COMP_WEIGHTS["group"]
    return FORM_COMP_WEIGHTS["competitive"]


def _parse_date(value):
    """Best-effort parse of a YYYY-MM-DD (or YYYY) date string to a date."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def form_score(matches, ref_date=None) -> float:
    """A team's recent-form (momentum) scalar, from its OWN perspective.

    matches: iterable of dicts describing the team's recent games, each oriented
    to this team:
        gf    int  goals the team scored
        ga    int  goals the team conceded
        comp  str  match status/stage (graded by `_form_comp_weight`)
        date  str  optional 'YYYY-MM-DD', for recency weighting against ref_date

    Each match contributes a result point (+1 win / 0 draw / -1 loss) plus a
    capped goal-margin term, recency- and stage-weighted. The weighted average is
    shrunk toward 0 for small samples. Returns ~[-1.75, 1.75]; no matches -> 0.0
    (a team with no recent record contributes no momentum, by design).
    """
    ref = _parse_date(ref_date) if ref_date else _dt.date.today()
    num = den = 0.0
    for m in matches:
        gf, ga = int(m["gf"]), int(m["ga"])
        result = 1.0 if gf > ga else (-1.0 if gf < ga else 0.0)
        margin = max(-FORM_GD_CAP, min(FORM_GD_CAP, gf - ga))
        value = result + FORM_GD_COEF * margin
        w = _form_comp_weight(m.get("comp", ""))
        d = _parse_date(m.get("date"))
        if ref and d:
            age_days = max(0, (ref - d).days)
            w *= 0.5 ** (age_days / FORM_HALFLIFE_DAYS)
        num += w * value
        den += w
    if den <= 0:
        return 0.0
    avg = num / den
    return avg * (den / (den + FORM_SHRINK))   # shrink small samples toward 0


def form_supremacy(form_home: float, form_away: float) -> float:
    """Bounded supremacy (goals) from the momentum gap between two teams.

    Positive favours the home team (it arrives in better form). Two teams with
    identical (or absent) form cancel to ~0, so momentum only ever nudges the
    line toward whoever is genuinely hotter coming in.
    """
    bump = FORM_WEIGHT * (form_home - form_away)
    return max(-FORM_CAP, min(FORM_CAP, bump))


def expected_goals(
    rating_home: float,
    rating_away: float,
    neutral: bool = False,
    expert: tuple[float, float] | None = None,
    expert_w: float = EXPERT_W,
    h2h_sup: float = 0.0,
    form_sup: float = 0.0,
) -> tuple[float, float]:
    """Map two FIFA ratings to (lambda_home, lambda_away).

    neutral=True drops the home advantage (knockout games at neutral venues).
    expert=(home_goals, away_goals) blends the model toward an expert scoreline.
    h2h_sup: extra supremacy (goals) from past meetings; see `h2h_supremacy`.
        Applied regardless of venue — history travels with the matchup.
    form_sup: extra supremacy (goals) from recent momentum; see `form_supremacy`.
        The hotter team coming into the tournament gets a small nudge.
    """
    sup = (rating_home - rating_away) / K
    if not neutral:
        sup += HOME_SUP
    sup += h2h_sup
    sup += form_sup
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
    # Precompute each side's Poisson column ONCE (the away PMF is independent of
    # i, so recomputing it inside the inner loop was N² _poisson_pmf calls).
    pmf_home = [_poisson_pmf(i, lam_home) for i in range(MAX_GOALS + 1)]
    pmf_away = [_poisson_pmf(j, lam_away) for j in range(MAX_GOALS + 1)]
    p_home = p_draw = p_away = 0.0
    for i in range(MAX_GOALS + 1):
        ph = pmf_home[i]
        fh = base_home + i
        for j in range(MAX_GOALS + 1):
            tau = _dc_tau(i, j, lam_home, lam_away) if dixon_coles else 1.0
            prob = ph * pmf_away[j] * tau
            fa = base_away + j
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
        form_sup: float = 0.0,
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(
            rating_home, rating_away, neutral=neutral, expert=expert,
            h2h_sup=h2h_sup, form_sup=form_sup,
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
        neutral: bool = False,
        expert: tuple[float, float] | None = None,
        h2h_sup: float = 0.0,
        form_sup: float = 0.0,
    ) -> dict[str, float]:
        lam_h, lam_a = expected_goals(
            rating_home, rating_away, neutral=neutral, expert=expert,
            h2h_sup=h2h_sup, form_sup=form_sup,
        )
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
    form_sup: float = 0.0,
) -> tuple[int, int]:
    lam_h, lam_a = expected_goals(
        rating_home, rating_away, neutral=neutral, expert=expert,
        h2h_sup=h2h_sup, form_sup=form_sup,
    )
    return sample_poisson(lam_h, rng), sample_poisson(lam_a, rng)


def knockout_winner(
    rating_home: float, rating_away: float, rng, neutral: bool = True,
    h2h_sup: float = 0.0, form_sup: float = 0.0,
) -> int:
    """Return 0 if home advances, 1 if away. Draws resolve (ET/penalties) by
    splitting proportionally to each side's win strength."""
    hg, ag = sample_score(
        rating_home, rating_away, rng, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
    )
    if hg > ag:
        return 0
    if ag > hg:
        return 1
    probs = ProbabilityModel().pre_match(
        rating_home, rating_away, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
    )
    ph, pa = probs["p_home"], probs["p_away"]
    frac = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    # Cap the favourite's ET/penalties edge — a shootout is near a coin flip.
    frac = max(1.0 - SHOOTOUT_CAP, min(SHOOTOUT_CAP, frac))
    return 0 if rng.random() < frac else 1
