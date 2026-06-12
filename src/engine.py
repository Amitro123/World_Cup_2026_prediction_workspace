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
       total = BASE_TOTAL + |r_h + r_a - 2*FIFA_MEAN| * TOTAL_STRENGTH  (flat by default)
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
- 90 minutes of regular time plus STOPPAGE_MIN expected stoppage: the in-play
  clock treats (90 + STOPPAGE_MIN) as the effective full-time mark, so a
  trailing team keeps a nonzero chance at minute 90.
- Home advantage is a flat supremacy bump (HOME_SUP goals); neutral=True drops it
  for knockout games at neutral venues.
- Swap the `ProbabilityModel` class to upgrade the engine; callers use only
  `pre_match`, `in_play`, and the sampling helpers.
"""

from __future__ import annotations

import datetime as _dt
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict


# --- Public output shapes (TypedDict so the dashboard / Excel mirror / knockout
# sim can't silently drift from what the engine actually returns) ---------------
class MatchProbs(TypedDict):
    """The 1X2 probability vector — sums to 1.0."""
    p_home: float
    p_draw: float
    p_away: float


class PreMatchResult(MatchProbs):
    """`pre_match` / `probs_from_lambdas` output: 1X2 plus the underlying λs."""
    lambda_home: float
    lambda_away: float


class InPlayResult(PreMatchResult):
    """`in_play` output: pre-match fields plus the live-state extras."""
    remaining_fraction: float
    red_mult_home: float
    red_mult_away: float

# --- Tunable model constants (FIFA-points Dixon-Coles) -----------------------
K = 240.0            # FIFA points per 1 goal of supremacy (calibrated to the
                     # bookmaker market 2026-06: K=200 was overconfident on
                     # favourites; K=240 is the knee of the market-KL/holdout-
                     # Brier trade-off — most of the alignment gain, ~0.4% Brier cost)
BASE_TOTAL = 2.6     # neutral expected total goals for an even tie
TOTAL_STRENGTH = 0.0 # goals added to the total per FIFA point of combined-strength
                     # deviation from 2*FIFA_MEAN. Was 1/4000 (=0.00025), which made
                     # stronger pairs score more — but across 294 holdout matches the
                     # correlation between combined strength and total goals is ~0
                     # (-0.014), so the term pointed at a signal that isn't there.
                     # Flattened to 0.0 (constant BASE_TOTAL); re-enable to tune.
                     # CR4 asked whether deep-knockout games (cagey, low-scoring)
                     # mask a positive group-stage effect: tested by stage split —
                     # group corr is -0.21 (n=48), knockout +0.03 (n=246), i.e.
                     # the hypothesis is refuted, not just null. Stays 0.0.
FIFA_MEAN = 1500.0   # reference FIFA rating (strength scaling anchor)

# Rating-gap -> goal-supremacy mapping (CR §3A: the linear /K mapping is
# "floor-bound on weak opponents" — a 1875-vs-1100 gap yields ~3.2 goals of
# supremacy, pinning the minnow's lambda at MIN_LAMBDA so every minnow looks
# identical). "logratio" uses SUP_ALPHA * ln(r_home / r_away), which compresses
# the tails while matching the linear slope at the mean (SUP_ALPHA = FIFA_MEAN/K
# makes the two modes locally identical at r_home≈r_away≈FIFA_MEAN, so switching
# is a pure tail-compression change).
#
# VERDICT (measured 2026-06, fit on the 294-match holdout — see _rating_supremacy):
# logratio at the fitted alpha≈7 lowers POOLED Brier 0.5769->0.5744 and trims the
# favourites ~1pp (France/Spain title odds 16->15%), exactly as intended. BUT in
# leave-one-tournament-out CV the out-of-sample gain collapses to -0.0002 log-loss
# (it helps 4 tournaments, hurts Euro-2024), i.e. a wash — same call we made on
# Elo. So the structural fix is implemented and validated, but the DEFAULT stays
# "linear" because the data doesn't justify flipping it. Re-run the fit if more
# holdout tournaments are added; flip only if the CV gain becomes robust.
SUP_MODE = "linear"      # "linear" | "logratio"  (default linear — see verdict above)
SUP_ALPHA = FIFA_MEAN / K  # 1500/240 = 6.25 (slope-match); fitted optimum ≈7.0

# Dixon-Coles low-score dependence. Club-football fits put rho near -0.13
# (Dixon & Coles 1997; dashee87.github.io / opisthokonta.net replications), but
# those samples are league seasons. International tournament football has fewer
# 0-0/1-0 grinds than club leagues (different incentives, no relegation chess),
# so we deliberately run HALF the literature value rather than a fitted one —
# rho is barely identifiable on our holdout: the pooled-294-match sweep
# (backtest.sweep(df, "DC_RHO", [0, -.03, -.06, -.09, -.13])) gives Brier
# 0.5765 / 0.5766 / 0.5769 / 0.5773 / 0.5781 — a 0.0016 spread, an order of
# magnitude below the bootstrap SE (~0.03). -0.06 keeps the qualitative
# draw-inflation correction at half the club value without measurable cost.
# Re-fit only when more holdout tournaments are added.
DC_RHO = -0.06       # Dixon-Coles low-score dependence (see note above)
HOME_SUP = 0.35      # home advantage, in goals of supremacy (added to `sup`).
                     # Empirically validated (2026-06, CR6 §2): pooled over the
                     # 54 host-nation matches of WC 1990-2022 with opponent
                     # strength controlled via leakage-free Elo snapshots
                     # (elo.snapshot_before each tournament), hosts overperform
                     # their neutral-venue expectation by +0.133 expected-score
                     # pts/match (8 of 10 host campaigns positive; Brazil 2014
                     # and Qatar 2022 the exceptions) ~= +0.48 goals of
                     # supremacy, SE ~0.06 pts. 0.35 sits inside the 95% CI —
                     # slightly conservative, which is right for n=54. NB the
                     # naive version of this study (host supremacy vs their own
                     # nearby competitive matches, no opponent control) gives
                     # -0.19 goals — qualifier opposition inflates the baseline;
                     # don't "re-derive" this constant without Elo control.
EXPERT_W = 0.85      # weight on the model vs an expert scoreline target; 0.55
                     # over-weighted the expert and pulled the model away from
                     # the market — 0.85 (15% expert) best matches market 1X2

# --- Host nations (World Cup 2026 co-hosts) ----------------------------------
# 2026 is played across the USA, Mexico and Canada, so *every* match is at a
# neutral venue for the visiting teams EXCEPT when a host nation plays at home —
# there a real crowd advantage applies. Group games are therefore treated as
# neutral unless the home_id is one of these three; knockout games stay neutral.
HOSTS = frozenset({"USA", "MEX", "CAN"})
# Floor on any expected-goals value. In extreme 2026 mismatches the raw formula
# goes below it (Spain~1876 vs Curacao~1295: sup=2.42 -> lam_away=0.09) and the
# floor binds, so the minnow keeps a realistic ~1-in-5 chance of a goal per
# match — WC history says even hopeless underdogs average ~0.2+ goals/game vs
# top sides, not 0.09. The 294-match holdout cannot arbitrate (no pairing there
# is lopsided enough for the floor to bind: Brier identical for 0.05..0.25), so
# 0.18 is an empirical-prior guard, not a fitted value. CR4 suggested trying
# 0.10; tested — no measurable holdout effect, kept at 0.18.
MIN_LAMBDA = 0.18
MAX_GOALS = 10       # truncation for the Poisson scoreline grid.
                     # Raised from 8: for the strongest realistic matchup in this
                     # model (France 1877 vs a minnow, λ_home ≈ 2.4) the
                     # P(goals > 8) tail is ~0.09% — enough to bias the 1X2
                     # normalisation. MAX_GOALS=10 reduces the tail to <5e-5,
                     # well under 1e-4 for all achievable λ without expert blending
                     # (~2.5 max). The extra two loop iterations are negligible.

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


# Stoppage time — FIFA data for the 2022 WC shows average group-stage stoppage of
# ~5.2 minutes per half. At minute 90:00 (the "clock stop") the model previously
# returned remaining=0, giving the trailing team 0% win probability even though
# 4-7 minutes of stoppage still follow. STOPPAGE_MIN is the expected added time
# treated as a buffer: the live-probability engine uses (90 + STOPPAGE_MIN) as the
# effective full-time mark, so a trailing team retains meaningful win probability
# right up to the final whistle. Set to 0 to reproduce the old behaviour.
STOPPAGE_MIN = 5

# Knockout draws go to 30' of extra time and then, if still level, penalties.
# We model these as two distinct regimes (see `resolve_knockout`):
#   1. EXTRA TIME — a real mini-match where the stronger side keeps its full
#      edge. ET is 30 min ≈ 1/3 of regulation, and historically lower-scoring
#      per minute (cagey, fatigue), so we scale each side's expected goals by
#      ET_LAMBDA_SCALE. Matches that reach ET average ~0.8 total goals in it vs
#      ~2.5 in regulation, so ~0.33 is empirically reasonable.
#   2. PENALTY SHOOTOUT — near a coin flip regardless of the skill gap. Only the
#      still-level-after-ET ties reach it; we split proportionally to win
#      strength but cap the favourite at SHOOTOUT_CAP. 0.53 matches the empirical
#      long-run favourite win rate in shootouts (~52-55%); 0.58 was generous.
ET_LAMBDA_SCALE = 0.33
SHOOTOUT_CAP = 0.53

# In-play red cards. A team reduced to 10 men creates fewer chances and concedes
# more space: empirically its own scoring rate falls to ~0.7-0.75x while the
# opponent's rises to ~1.3-1.4x. These multipliers apply to the REMAINING-time
# expected goals only, and compose multiplicatively for multiple / both-side
# dismissals (see `red_card_multipliers`).
RED_CARD_OWN = 0.74   # attack multiplier for the side that is a man down
RED_CARD_OPP = 1.40   # attack multiplier for the side with the man advantage

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


def _comp_weight(comp: object) -> float:
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


def h2h_supremacy(meetings: Iterable[Mapping[str, Any]], ref_year: int | None = None) -> float:
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


def _form_comp_weight(comp: object) -> float:
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


def _parse_date(value: object) -> _dt.date | None:
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


def form_score(matches: Iterable[Mapping[str, Any]], ref_date: object = None) -> float:
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


def _rating_supremacy(rating_home: float, rating_away: float) -> float:
    """Convert a rating gap to goal supremacy under the active SUP_MODE.

    "linear" (default): (r_home - r_away) / K — the original mapping.
    "logratio": SUP_ALPHA * ln(r_home / r_away) — compresses blowout gaps so a
    minnow's lambda is no longer pinned at the floor. Falls back to linear if a
    rating is non-positive (a placeholder team) so ln() never blows up.
    """
    if SUP_MODE == "logratio" and rating_home > 0 and rating_away > 0:
        return SUP_ALPHA * math.log(rating_home / rating_away)
    return (rating_home - rating_away) / K


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
    sup = _rating_supremacy(rating_home, rating_away)
    if not neutral:
        sup += HOME_SUP
    sup += h2h_sup
    sup += form_sup
    total = BASE_TOTAL + abs(rating_home + rating_away - 2.0 * FIFA_MEAN) * TOTAL_STRENGTH
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
) -> PreMatchResult:
    """Public helper: win/draw/loss directly from a pair of expected-goals."""
    out = _grid_probs(lam_home, lam_away, dixon_coles=dixon_coles)
    return {
        "p_home": out["p_home"],
        "p_draw": out["p_draw"],
        "p_away": out["p_away"],
        "lambda_home": lam_home,
        "lambda_away": lam_away,
    }


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
    ) -> PreMatchResult:
        lam_h, lam_a = expected_goals(
            rating_home, rating_away, neutral=neutral, expert=expert,
            h2h_sup=h2h_sup, form_sup=form_sup,
        )
        out = _grid_probs(lam_h, lam_a, dixon_coles=True)
        return {
            "p_home": out["p_home"],
            "p_draw": out["p_draw"],
            "p_away": out["p_away"],
            "lambda_home": lam_h,
            "lambda_away": lam_a,
        }

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
        red_home: int = 0,
        red_away: int = 0,
    ) -> InPlayResult:
        lam_h, lam_a = expected_goals(
            rating_home, rating_away, neutral=neutral, expert=expert,
            h2h_sup=h2h_sup, form_sup=form_sup,
        )
        remaining = max(0.0, (90 + STOPPAGE_MIN - minute) / 90.0)
        red_h, red_a = red_card_multipliers(red_home, red_away)
        out = _grid_probs(
            lam_h * remaining * red_h,
            lam_a * remaining * red_a,
            base_home=home_goals,
            base_away=away_goals,
            dixon_coles=False,
        )
        return {
            "p_home": out["p_home"],
            "p_draw": out["p_draw"],
            "p_away": out["p_away"],
            "lambda_home": lam_h,
            "lambda_away": lam_a,
            "remaining_fraction": remaining,
            "red_mult_home": red_h,
            "red_mult_away": red_a,
        }


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

def red_card_multipliers(red_home: int = 0, red_away: int = 0) -> tuple[float, float]:
    """Expected-goals multipliers (home, away) given each side's red cards.

    Each home dismissal scales home's scoring rate by RED_CARD_OWN and away's by
    RED_CARD_OPP (a man up); away dismissals do the mirror. Effects compose, so
    two home reds apply RED_CARD_OWN twice. 11-vs-11 returns (1.0, 1.0).
    """
    red_home = max(0, int(red_home))
    red_away = max(0, int(red_away))
    mh = (RED_CARD_OWN ** red_home) * (RED_CARD_OPP ** red_away)
    ma = (RED_CARD_OWN ** red_away) * (RED_CARD_OPP ** red_home)
    return mh, ma


def sample_poisson(lam: float, rng: random.Random) -> int:
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
    rng: random.Random,
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


def resolve_knockout(
    rating_home: float, rating_away: float, rng: random.Random, neutral: bool = True,
    h2h_sup: float = 0.0, form_sup: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """Play a knockout tie to a single winner via regulation -> ET -> penalties.

    Returns ``(winner_idx, info)`` where ``winner_idx`` is 0 (home) or 1 (away)
    and ``info`` describes how it was decided::

        {"reg": (hg, ag),            # 90' score
         "et":  (eh, ea) | None,     # extra-time-only goals, None if decided in 90'
         "pens": bool}               # True if it went to a shootout

    Regulation and ET share the same expected-goal rates (ET scaled by
    ET_LAMBDA_SCALE); only a tie after both reaches the capped shootout.
    """
    lam_h, lam_a = expected_goals(
        rating_home, rating_away, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
    )
    hg, ag = sample_poisson(lam_h, rng), sample_poisson(lam_a, rng)
    if hg != ag:
        return (0 if hg > ag else 1), {"reg": (hg, ag), "et": None, "pens": False}

    # Level after 90' -> 30' of extra time, lower-scoring (ET_LAMBDA_SCALE).
    eh = sample_poisson(lam_h * ET_LAMBDA_SCALE, rng)
    ea = sample_poisson(lam_a * ET_LAMBDA_SCALE, rng)
    if eh != ea:
        return (0 if eh > ea else 1), {"reg": (hg, ag), "et": (eh, ea), "pens": False}

    # Still level -> penalty shootout: near a coin flip, capped by SHOOTOUT_CAP.
    probs = ProbabilityModel().pre_match(
        rating_home, rating_away, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
    )
    ph, pa = probs["p_home"], probs["p_away"]
    frac = ph / (ph + pa) if (ph + pa) > 0 else 0.5
    frac = max(1.0 - SHOOTOUT_CAP, min(SHOOTOUT_CAP, frac))
    wi = 0 if rng.random() < frac else 1
    return wi, {"reg": (hg, ag), "et": (eh, ea), "pens": True}


def knockout_winner(
    rating_home: float, rating_away: float, rng: random.Random, neutral: bool = True,
    h2h_sup: float = 0.0, form_sup: float = 0.0,
) -> int:
    """Return 0 if home advances, 1 if away (regulation -> ET -> penalties).

    Thin wrapper over `resolve_knockout` for the Monte-Carlo hot loop, which only
    needs the winner index.
    """
    wi, _ = resolve_knockout(
        rating_home, rating_away, rng, neutral=neutral,
        h2h_sup=h2h_sup, form_sup=form_sup,
    )
    return wi
