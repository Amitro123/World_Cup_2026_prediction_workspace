"""
בקטסט וכיול — retrospective validation of the probability engine.

The dashboard predicts the 2026 World Cup, but until the model is scored against
*known* results we cannot say whether it is well-calibrated. This module runs the
exact same `ProbabilityModel` over a historical tournament (the 2022 World Cup,
shipped in data/backtest_2022.csv) and reports standard forecasting metrics:

    * Brier score  — mean squared error of the 3-way (H/D/A) probability vector.
                     Lower is better. 0 = perfect, 0.667 = always-uniform guess.
    * Log loss     — penalises confident wrong calls harshly. Lower is better.
    * Calibration  — when the model says "60%", does it happen ~60% of the time?
                     Pooled over all three outcomes into reliability bins.

It also compares against two naive baselines (uniform 1/3-1/3-1/3, and the
dataset's empirical base rate) so a number like "Brier 0.61" has context, and it
can sweep an engine constant (e.g. K) to show how calibration responds — turning
the CR's "lower the expert weight / tune the model" debate from opinion into a
measured curve.

Data schema (data/backtest_2022.csv), one row per match:
    date, home, away, rating_home, rating_away, home_goals, away_goals,
    neutral (0/1), stage

The whole 2022 tournament was at neutral venues (Qatar), so rows are neutral=1;
this isolates pure strength prediction with no home-advantage confound. Knockout
games decided on penalties are recorded as their 90'/120' draw (the model
predicts regulation outcome, not the shootout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from . import engine

OUTCOMES = ("H", "D", "A")
_PROB_KEY = {"H": "p_home", "D": "p_draw", "A": "p_away"}

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_2022.csv")

# Engine constants this module can temporarily override for a sweep.
_TUNABLE = ("K", "BASE_TOTAL", "HOME_SUP", "DC_RHO", "FIFA_MEAN")


@dataclass
class Metrics:
    n: int
    brier: float
    log_loss: float
    accuracy: float          # share where the argmax pick matched the result
    by_class_brier: dict     # H/D/A contribution, for diagnostics

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "brier": round(self.brier, 4),
            "log_loss": round(self.log_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "by_class_brier": {k: round(v, 4) for k, v in self.by_class_brier.items()},
        }


def _clip(p: float, eps: float = 1e-12) -> float:
    return min(1.0 - eps, max(eps, p))


def predict_row(row, model: engine.ProbabilityModel) -> dict[str, float]:
    """Pre-match H/D/A probabilities for one historical match row."""
    neutral = bool(int(row.get("neutral", 1)))
    probs = model.pre_match(
        float(row["rating_home"]), float(row["rating_away"]), neutral=neutral
    )
    return {o: probs[_PROB_KEY[o]] for o in OUTCOMES}


def evaluate(
    df: pd.DataFrame,
    model: engine.ProbabilityModel | None = None,
    overrides: dict | None = None,
) -> Metrics:
    """Score the model over a historical-match frame.

    overrides: temporarily patch engine constants (e.g. {"K": 240}) for the
    duration of this call, then restore them — used by `sweep`.
    """
    model = model or engine.ProbabilityModel()
    saved = {}
    if overrides:
        for name, val in overrides.items():
            if name not in _TUNABLE:
                raise KeyError(f"non-tunable engine constant: {name}")
            saved[name] = getattr(engine, name)
            setattr(engine, name, val)
    try:
        n = 0
        brier_sum = 0.0
        ll_sum = 0.0
        hits = 0
        cls_sum = {o: 0.0 for o in OUTCOMES}
        for _, row in df.iterrows():
            actual = engine.outcome_from_score(int(row["home_goals"]), int(row["away_goals"]))
            p = predict_row(row, model)
            for o in OUTCOMES:
                y = 1.0 if o == actual else 0.0
                d2 = (p[o] - y) ** 2
                brier_sum += d2
                cls_sum[o] += d2
            ll_sum += -engine.math.log(_clip(p[actual]))
            if max(OUTCOMES, key=lambda o: p[o]) == actual:
                hits += 1
            n += 1
        if n == 0:
            return Metrics(0, 0.0, 0.0, 0.0, {o: 0.0 for o in OUTCOMES})
        return Metrics(
            n=n,
            brier=brier_sum / n,
            log_loss=ll_sum / n,
            accuracy=hits / n,
            by_class_brier={o: cls_sum[o] / n for o in OUTCOMES},
        )
    finally:
        for name, val in saved.items():
            setattr(engine, name, val)


def baselines(df: pd.DataFrame) -> dict[str, Metrics]:
    """Reference scores so the model's numbers have context.

    uniform   — always 1/3, 1/3, 1/3 (knows nothing).
    base_rate — always the dataset's empirical H/D/A frequencies (knows the
                outcome distribution but nothing about the specific teams).
    """
    actuals = [
        engine.outcome_from_score(int(r["home_goals"]), int(r["away_goals"]))
        for _, r in df.iterrows()
    ]
    n = len(actuals)
    out: dict[str, Metrics] = {}

    def _score(prob_fn) -> Metrics:
        brier_sum = ll_sum = 0.0
        hits = 0
        cls_sum = {o: 0.0 for o in OUTCOMES}
        for actual in actuals:
            p = prob_fn()
            for o in OUTCOMES:
                y = 1.0 if o == actual else 0.0
                d2 = (p[o] - y) ** 2
                brier_sum += d2
                cls_sum[o] += d2
            ll_sum += -engine.math.log(_clip(p[actual]))
            if max(OUTCOMES, key=lambda o: p[o]) == actual:
                hits += 1
        return Metrics(n, brier_sum / n, ll_sum / n, hits / n,
                       {o: cls_sum[o] / n for o in OUTCOMES})

    out["uniform"] = _score(lambda: {o: 1 / 3 for o in OUTCOMES})

    rate = {o: actuals.count(o) / n for o in OUTCOMES}
    out["base_rate"] = _score(lambda: rate)
    return out


def calibration_table(
    df: pd.DataFrame,
    model: engine.ProbabilityModel | None = None,
    bins: int = 10,
) -> list[dict]:
    """Reliability bins pooled over all three outcomes.

    For every match we emit three (predicted_prob, hit) points — one per outcome.
    Points are bucketed by predicted probability; a well-calibrated model has
    mean predicted ≈ observed frequency in each bucket.
    """
    model = model or engine.ProbabilityModel()
    edges = [i / bins for i in range(bins + 1)]
    buckets = [{"lo": edges[i], "hi": edges[i + 1], "sum_p": 0.0, "hits": 0, "n": 0}
               for i in range(bins)]
    for _, row in df.iterrows():
        actual = engine.outcome_from_score(int(row["home_goals"]), int(row["away_goals"]))
        p = predict_row(row, model)
        for o in OUTCOMES:
            prob = p[o]
            idx = min(bins - 1, int(prob * bins))
            b = buckets[idx]
            b["sum_p"] += prob
            b["hits"] += 1 if o == actual else 0
            b["n"] += 1
    table = []
    for b in buckets:
        if b["n"] == 0:
            continue
        table.append({
            "range": f"{b['lo']:.1f}-{b['hi']:.1f}",
            "n": b["n"],
            "mean_pred": round(b["sum_p"] / b["n"], 3),
            "observed": round(b["hits"] / b["n"], 3),
            "gap": round(b["hits"] / b["n"] - b["sum_p"] / b["n"], 3),
        })
    return table


def sweep(
    df: pd.DataFrame,
    param: str,
    values,
    model: engine.ProbabilityModel | None = None,
) -> list[dict]:
    """Re-score the model across values of one engine constant.

    Returns rows sorted by the values given, each with brier/log_loss/accuracy,
    so you can see which setting best calibrates on the historical data instead
    of arguing about it. e.g. sweep(df, "K", [160, 180, 200, 220, 240]).
    """
    rows = []
    for v in values:
        m = evaluate(df, model=model, overrides={param: v})
        rows.append({param: v, "brier": round(m.brier, 4),
                     "log_loss": round(m.log_loss, 4),
                     "accuracy": round(m.accuracy, 4)})
    return rows


def load(csv_path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(csv_path or DEFAULT_CSV)


def run(csv_path: str | None = None) -> dict:
    """Full backtest report as a structured dict (used by CLI + dashboard)."""
    df = load(csv_path)
    model = engine.ProbabilityModel()
    m = evaluate(df, model)
    base = baselines(df)
    return {
        "n": int(len(df)),
        "model": m.as_dict(),
        "baselines": {k: v.as_dict() for k, v in base.items()},
        "skill_vs_uniform": round(1 - m.brier / base["uniform"].brier, 4),
        "skill_vs_base_rate": round(1 - m.brier / base["base_rate"].brier, 4),
        "calibration": calibration_table(df, model),
        "k_sweep": sweep(df, "K", [160, 180, 200, 220, 240, 260]),
    }


def _print_report(rep: dict) -> None:
    print(f"\n=== Backtest: {rep['n']} historical matches ===\n")
    m = rep["model"]
    print(f"MODEL      Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  "
          f"Acc={m['accuracy']:.3f}")
    for name, b in rep["baselines"].items():
        print(f"  {name:9} Brier={b['brier']:.4f}  LogLoss={b['log_loss']:.4f}  "
              f"Acc={b['accuracy']:.3f}")
    print(f"\nSkill score (Brier vs uniform):    {rep['skill_vs_uniform']:+.1%}")
    print(f"Skill score (Brier vs base rate):  {rep['skill_vs_base_rate']:+.1%}")
    print("  (positive = better than the baseline)\n")

    print("Calibration (predicted vs observed, pooled over H/D/A):")
    print(f"  {'range':>9} {'n':>4} {'mean_pred':>10} {'observed':>9} {'gap':>7}")
    for r in rep["calibration"]:
        print(f"  {r['range']:>9} {r['n']:>4} {r['mean_pred']:>10.3f} "
              f"{r['observed']:>9.3f} {r['gap']:>+7.3f}")

    print("\nK sweep (FIFA points per goal of supremacy):")
    best = min(rep["k_sweep"], key=lambda r: r["brier"])
    for r in rep["k_sweep"]:
        flag = "  <- best Brier" if r is best else ""
        print(f"  K={r['K']:>4}  Brier={r['brier']:.4f}  "
              f"LogLoss={r['log_loss']:.4f}  Acc={r['accuracy']:.3f}{flag}")
    print()


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Backtest the probability engine.")
    ap.add_argument("--csv", default=None, help="historical matches CSV "
                    "(default: data/backtest_2022.csv)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    rep = run(args.csv)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_report(rep)


if __name__ == "__main__":
    main()
