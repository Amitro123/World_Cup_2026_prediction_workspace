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


def _mean_std(vals) -> tuple[float, float]:
    vals = [float(v) for v in vals]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return mu, var ** 0.5


def team_stats(df: pd.DataFrame) -> dict | None:
    """Population mean/std of FIFA and Elo across the distinct teams in df.

    Returns None if the frame has no `elo_home`/`elo_away` columns (so the blend
    is simply unavailable and callers fall back to pure FIFA).
    """
    if "elo_home" not in df.columns or "elo_away" not in df.columns:
        return None
    fifa: dict[str, float] = {}
    elo: dict[str, float] = {}
    for _, r in df.iterrows():
        fifa[r["home"]] = float(r["rating_home"]); elo[r["home"]] = float(r["elo_home"])
        fifa[r["away"]] = float(r["rating_away"]); elo[r["away"]] = float(r["elo_away"])
    fmu, fsd = _mean_std(fifa.values())
    emu, esd = _mean_std(elo.values())
    return {"fifa_mean": fmu, "fifa_std": fsd, "elo_mean": emu, "elo_std": esd}


def _row_get(row, key, default=None):
    """Safe column access that treats NaN as absent (pandas rows)."""
    if key not in row:
        return default
    val = row[key]
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def predict_row(
    row,
    model: engine.ProbabilityModel,
    elo_weight: float = 0.0,
    stats: dict | None = None,
    config: dict | None = None,
) -> dict[str, float]:
    """Pre-match H/D/A probabilities for one historical match row.

    elo_weight>0 blends the row's Elo into the FIFA rating (needs `stats` from
    `team_stats`); 0 uses pure FIFA, reproducing the production model.

    config (optional) turns the per-match nudges on/off so the holdout can
    measure whether they help, instead of assuming it:
        {"use_h2h": bool, "use_form": bool, "use_expert": bool}
    Each is only applied when the matching column(s) exist on the row:
        use_h2h    -> column `h2h_sup`     (signed supremacy, home POV)
        use_form   -> column `form_sup`    (signed supremacy, home POV)
        use_expert -> columns `exp_home`,`exp_away` (an expert scoreline)
    Absent columns are silently neutral (0 / None), matching production's
    "missing data = neutral zero" contract.
    """
    config = config or {}
    neutral = bool(int(row.get("neutral", 1)))
    rh = float(row["rating_home"])
    ra = float(row["rating_away"])
    if elo_weight > 0 and stats is not None and "elo_home" in row:
        rh = engine.blend_strength(rh, float(row["elo_home"]), elo_weight, **stats)
        ra = engine.blend_strength(ra, float(row["elo_away"]), elo_weight, **stats)

    h2h_sup = float(_row_get(row, "h2h_sup", 0.0) or 0.0) if config.get("use_h2h") else 0.0
    form_sup = float(_row_get(row, "form_sup", 0.0) or 0.0) if config.get("use_form") else 0.0
    expert = None
    if config.get("use_expert"):
        eh = _row_get(row, "exp_home")
        ea = _row_get(row, "exp_away")
        if eh is not None and ea is not None:
            expert = (float(eh), float(ea))

    probs = model.pre_match(
        rh, ra, neutral=neutral,
        expert=expert, h2h_sup=h2h_sup, form_sup=form_sup,
    )
    return {o: probs[_PROB_KEY[o]] for o in OUTCOMES}


def evaluate(
    df: pd.DataFrame,
    model: engine.ProbabilityModel | None = None,
    overrides: dict | None = None,
    elo_weight: float = 0.0,
    config: dict | None = None,
) -> Metrics:
    """Score the model over a historical-match frame.

    overrides: temporarily patch engine constants (e.g. {"K": 240}) for the
    duration of this call, then restore them — used by `sweep`.
    elo_weight: share given to Elo in a FIFA/Elo blend (0 = pure FIFA).
    config: per-match signal switches (use_h2h/use_form/use_expert) — see
    `predict_row`. None reproduces the pure pre-match FIFA model.
    """
    model = model or engine.ProbabilityModel()
    stats = team_stats(df) if elo_weight > 0 else None
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
            p = predict_row(row, model, elo_weight=elo_weight, stats=stats, config=config)
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


def elo_sweep(
    df: pd.DataFrame,
    weights=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
    model: engine.ProbabilityModel | None = None,
) -> list[dict]:
    """Re-score across FIFA/Elo blend weights to test whether Elo helps.

    weight=0 is the production (pure-FIFA) model; weight=1 is pure Elo. The CR
    claims Elo is more predictive — this measures it instead of assuming it. If
    no weight beats 0.0 on Brier, FIFA alone wins and we keep it.
    """
    if team_stats(df) is None:
        return []  # dataset has no Elo columns
    rows = []
    for w in weights:
        m = evaluate(df, model=model, elo_weight=w)
        rows.append({"elo_weight": w, "brier": round(m.brier, 4),
                     "log_loss": round(m.log_loss, 4),
                     "accuracy": round(m.accuracy, 4)})
    return rows


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


# --- multi-tournament holdout ------------------------------------------------

# The named configurations the holdout compares. Each is a recipe for
# predict_row, so we can measure whether each signal *earns its place* instead
# of assuming it (the CR's core ask). elo_weight=None means "pick the best
# weight from elo_sweep on this data"; a float pins it.
CONFIGS: dict[str, dict] = {
    "fifa_only": {},
    "+h2h":      {"use_h2h": True},
    "+form":     {"use_form": True},
    "+expert":   {"use_expert": True},
    "+elo":      {"elo_weight": None},
    "all":       {"use_h2h": True, "use_form": True, "use_expert": True,
                  "elo_weight": None},
}


def _best_elo_weight(df: pd.DataFrame) -> float:
    """Lowest-Brier Elo blend weight on df, or 0.0 if Elo unavailable."""
    rows = elo_sweep(df)
    if not rows:
        return 0.0
    return min(rows, key=lambda r: r["brier"])["elo_weight"]


def _signal_cols_present(df: pd.DataFrame) -> dict[str, bool]:
    return {
        "h2h": "h2h_sup" in df.columns,
        "form": "form_sup" in df.columns,
        "expert": "exp_home" in df.columns and "exp_away" in df.columns,
        "elo": "elo_home" in df.columns and "elo_away" in df.columns,
    }


def config_compare(
    df: pd.DataFrame,
    model: engine.ProbabilityModel | None = None,
) -> list[dict]:
    """Score every named CONFIG on df so each signal is judged on the data.

    Configs whose required columns are absent are skipped (not silently scored
    as fifa_only, which would dishonestly duplicate the baseline). The result is
    sorted by Brier so the winner is obvious.
    """
    model = model or engine.ProbabilityModel()
    have = _signal_cols_present(df)
    rows: list[dict] = []
    for name, cfg in CONFIGS.items():
        needs_elo = "elo_weight" in cfg
        if name == "all":
            # "all" = every signal that is actually available; strip the rest so
            # it is never skipped just because one source is missing.
            cfg = {k: v for k, v in cfg.items()
                   if not (k == "use_h2h" and not have["h2h"])
                   and not (k == "use_form" and not have["form"])
                   and not (k == "use_expert" and not have["expert"])
                   and not (k == "elo_weight" and not have["elo"])}
            needs_elo = "elo_weight" in cfg
            if not cfg:  # no signals available at all -> identical to fifa_only
                continue
        else:
            if cfg.get("use_h2h") and not have["h2h"]:
                continue
            if cfg.get("use_form") and not have["form"]:
                continue
            if cfg.get("use_expert") and not have["expert"]:
                continue
            if needs_elo and not have["elo"]:
                continue
        w = cfg.get("elo_weight", 0.0)
        if w is None:
            w = _best_elo_weight(df)
        m = evaluate(df, model=model, elo_weight=w or 0.0, config=cfg)
        rows.append({
            "config": name,
            "elo_weight": round(w or 0.0, 2),
            "brier": round(m.brier, 4),
            "log_loss": round(m.log_loss, 4),
            "accuracy": round(m.accuracy, 4),
            "n": m.n,
        })
    rows.sort(key=lambda r: r["brier"])
    return rows


def holdout(sources: dict[str, str] | None = None) -> dict:
    """Multi-tournament out-of-sample report.

    sources: {label: csv_path}. Defaults to every data/backtest_*.csv found.
    For each tournament and for the pooled set it reports the model vs baselines
    and the full config comparison, so the verdict ("does H2H/form/expert/Elo
    help?") is measured across several tournaments rather than overfit to one.
    """
    sources = sources or _discover_sources()
    frames: dict[str, pd.DataFrame] = {}
    for label, path in sources.items():
        if os.path.exists(path):
            frames[label] = pd.read_csv(path)
    if not frames:
        return {"error": "no holdout CSVs found", "looked_for": sources}

    def _block(df: pd.DataFrame) -> dict:
        model = engine.ProbabilityModel()
        m = evaluate(df, model)
        base = baselines(df)
        return {
            "n": int(len(df)),
            "model": m.as_dict(),
            "baselines": {k: v.as_dict() for k, v in base.items()},
            "skill_vs_uniform": round(1 - m.brier / base["uniform"].brier, 4),
            "configs": config_compare(df, model),
        }

    per = {label: _block(df) for label, df in frames.items()}
    pooled_df = pd.concat(list(frames.values()), ignore_index=True)
    pooled = _block(pooled_df)
    pooled["calibration"] = calibration_table(pooled_df)
    return {"tournaments": per, "pooled": pooled}


def _discover_sources() -> dict[str, str]:
    """Map every data/backtest_*.csv to a label (the bit after backtest_)."""
    import glob
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "backtest_*.csv"))):
        label = os.path.basename(path)[len("backtest_"):-len(".csv")]
        out[label] = path
    return out


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
        "elo_sweep": elo_sweep(df),
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

    if rep.get("elo_sweep"):
        print("\nElo blend sweep (0.0 = pure FIFA, 1.0 = pure Elo):")
        best_e = min(rep["elo_sweep"], key=lambda r: r["brier"])
        for r in rep["elo_sweep"]:
            flag = "  <- best Brier" if r is best_e else ""
            print(f"  w={r['elo_weight']:.1f}  Brier={r['brier']:.4f}  "
                  f"LogLoss={r['log_loss']:.4f}  Acc={r['accuracy']:.3f}{flag}")
        verdict = ("Elo blend helps" if best_e["elo_weight"] > 0
                   else "pure FIFA wins — keep ELO_WEIGHT=0")
        print(f"  => {verdict}")
    print()


def _print_configs(configs: list[dict]) -> None:
    if not configs:
        print("  (no signal columns present — only fifa_only is testable here)")
        return
    best = configs[0]  # already sorted by Brier
    print(f"  {'config':>10} {'elo_w':>6} {'brier':>7} {'logloss':>8} "
          f"{'acc':>6} {'n':>4}")
    for r in configs:
        flag = "  <- best" if r is best else ""
        print(f"  {r['config']:>10} {r['elo_weight']:>6.2f} {r['brier']:>7.4f} "
              f"{r['log_loss']:>8.4f} {r['accuracy']:>6.3f} {r['n']:>4}{flag}")


def _print_holdout(rep: dict) -> None:
    if "error" in rep:
        print(f"\n[holdout] {rep['error']}: {rep.get('looked_for')}\n")
        return
    print("\n=== Multi-tournament holdout ===")
    for label, blk in rep["tournaments"].items():
        m = blk["model"]
        print(f"\n--- {label}  ({blk['n']} matches) ---")
        print(f"  MODEL  Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  "
              f"Acc={m['accuracy']:.3f}   skill vs uniform: "
              f"{blk['skill_vs_uniform']:+.1%}")
        _print_configs(blk["configs"])

    pl = rep["pooled"]
    m = pl["model"]
    print(f"\n=== POOLED  ({pl['n']} matches) ===")
    print(f"  MODEL  Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}  "
          f"Acc={m['accuracy']:.3f}   skill vs uniform: "
          f"{pl['skill_vs_uniform']:+.1%}")
    for name, b in pl["baselines"].items():
        print(f"  {name:9} Brier={b['brier']:.4f}")
    print("\nConfig comparison (pooled — does each signal earn its place?):")
    _print_configs(pl["configs"])
    if pl["configs"]:
        best = pl["configs"][0]
        verdict = ("fifa_only is unbeaten — extra signals add noise on this data"
                   if best["config"] == "fifa_only"
                   else f"'{best['config']}' wins — it beats fifa_only out of sample")
        print(f"  => {verdict}")
    print()


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Backtest the probability engine.")
    ap.add_argument("--csv", default=None, help="historical matches CSV "
                    "(default: data/backtest_2022.csv)")
    ap.add_argument("--holdout", action="store_true",
                    help="multi-tournament holdout across all data/backtest_*.csv")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    if args.holdout:
        rep = holdout()
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            _print_holdout(rep)
        return

    rep = run(args.csv)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        _print_report(rep)


if __name__ == "__main__":
    main()
