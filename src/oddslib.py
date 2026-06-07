"""
src/oddslib.py — bookmaker-odds math: de-vig + model-vs-market divergence.

The review's single biggest unbuilt recommendation was a **bookmaker anchor**:
closing 1X2 odds are the gold standard for football calibration, so comparing
our model to the market catches miscalibration we cannot see from the inside.

This module is pure (no IO, no network) so it is fully unit-tested:

  * `implied_1x2`  — decimal odds -> de-vigged, normalised P(home/draw/away).
  * `implied_one` — a single yes/no selection (e.g. anytime scorer) -> P.
  * `kl`, `compare` — how far the model sits from the market, with a flag for
    disagreements worth a human look.

De-vig method
-------------
For the mutually-exclusive, exhaustive 1X2 market we use the *proportional*
(a.k.a. multiplicative) method: take each raw probability 1/odds and divide by
their sum (the "overround"). It is the standard baseline; it assumes the bookable
margin is spread proportionally across outcomes. Single selections (scorer/assist)
are NOT exhaustive, so they can only be de-vigged with a flat per-selection
margin, which the caller supplies.
"""

from __future__ import annotations

import math

OUTCOMES = ("p_home", "p_draw", "p_away")


def implied_1x2(dec_home: float, dec_draw: float, dec_away: float) -> dict[str, float]:
    """Decimal 1X2 odds -> de-vigged probabilities that sum to 1.0.

    Raises ValueError on non-positive odds (a decimal price is always > 1.0).
    """
    for o in (dec_home, dec_draw, dec_away):
        if o is None or float(o) <= 0:
            raise ValueError(f"decimal odds must be > 0, got {o}")
    raw = [1.0 / float(dec_home), 1.0 / float(dec_draw), 1.0 / float(dec_away)]
    s = sum(raw)
    return {k: r / s for k, r in zip(OUTCOMES, raw)}


def overround(dec_home: float, dec_draw: float, dec_away: float) -> float:
    """Bookmaker margin: sum(1/odds) - 1 (e.g. 0.05 = a 5% book)."""
    return sum(1.0 / float(o) for o in (dec_home, dec_draw, dec_away)) - 1.0


def implied_one(dec_odds: float, margin: float = 0.0) -> float:
    """A single yes/no market (anytime scorer/assist) -> implied probability.

    margin>0 removes a flat per-selection vig: p = (1/odds) / (1 + margin).
    Scorer markets are not mutually exclusive, so this is the only honest de-vig
    without the complementary "no" price. Result is clamped to [0, 1].
    """
    if dec_odds is None or float(dec_odds) <= 0:
        raise ValueError(f"decimal odds must be > 0, got {dec_odds}")
    p = (1.0 / float(dec_odds)) / (1.0 + max(0.0, margin))
    return max(0.0, min(1.0, p))


def _clip(p: float, eps: float = 1e-12) -> float:
    return min(1.0 - eps, max(eps, p))


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the 1X2 outcomes, in nats. 0 = identical.

    Read as "information lost using q (model) to approximate p (market)". Robust
    to zeros via clipping.
    """
    return sum(_clip(p[k]) * math.log(_clip(p[k]) / _clip(q[k])) for k in OUTCOMES)


def _argmax(p: dict[str, float]) -> str:
    return max(OUTCOMES, key=lambda k: p[k])


def compare(
    model: dict[str, float],
    market: dict[str, float],
    flag_threshold: float = 0.10,
) -> dict:
    """Model-vs-market diagnostic for one match.

    model/market: dicts with p_home/p_draw/p_away (model from the engine, market
    de-vigged from odds). Returns the KL divergence (market || model), the
    largest per-outcome gap, whether the two agree on the favourite, and a `flag`
    that trips when |gap| on any outcome exceeds `flag_threshold` — i.e. the model
    and the market disagree enough that a human should look.
    """
    diffs = {k: model[k] - market[k] for k in OUTCOMES}
    max_k = max(OUTCOMES, key=lambda k: abs(diffs[k]))
    return {
        "kl": kl(market, model),
        "max_gap": diffs[max_k],
        "max_gap_outcome": max_k,
        "pick_model": _argmax(model),
        "pick_market": _argmax(market),
        "agree": _argmax(model) == _argmax(market),
        "flag": abs(diffs[max_k]) >= flag_threshold,
    }


def market_from_row(row) -> dict[str, float] | None:
    """Read one market_odds.csv row into 1X2 probabilities.

    Accepts either decimal columns (dec_home/dec_draw/dec_away) or pre-computed
    implied columns (p_home/p_draw/p_away). Returns None if neither is usable, so
    a half-filled sheet degrades gracefully instead of raising.
    """
    def _f(key):
        v = row.get(key) if hasattr(row, "get") else (row[key] if key in row else None)
        try:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    dh, dd, da = _f("dec_home"), _f("dec_draw"), _f("dec_away")
    if dh and dd and da:
        try:
            return implied_1x2(dh, dd, da)
        except ValueError:
            return None
    ph, pd_, pa = _f("p_home"), _f("p_draw"), _f("p_away")
    if ph is not None and pd_ is not None and pa is not None:
        s = ph + pd_ + pa
        if s > 0:
            return {"p_home": ph / s, "p_draw": pd_ / s, "p_away": pa / s}
    return None
