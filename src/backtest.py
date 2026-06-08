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
# EXPERT_W is tunable too, but only bites on rows that carry expert columns
# (exp_home/exp_away); our historical holdouts don't, so fitting it needs new
# data — see `fit_report`'s note.
_TUNABLE = ("K", "BASE_TOTAL", "HOME_SUP", "DC_RHO", "FIFA_MEAN", "EXPERT_W")


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


def per_match_brier(
    df: pd.DataFrame,
    model: engine.ProbabilityModel | None = None,
    config: dict | None = None,
    elo_weight: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Per-match Brier for the model and the uniform baseline (parallel lists).

    Returned element-by-element so a bootstrap can resample matches; summed and
    divided by n they reproduce `evaluate(...).brier` and the uniform baseline.
    """
    model = model or engine.ProbabilityModel()
    stats = team_stats(df) if elo_weight > 0 else None
    model_b: list[float] = []
    unif_b: list[float] = []
    for _, row in df.iterrows():
        actual = engine.outcome_from_score(int(row["home_goals"]), int(row["away_goals"]))
        p = predict_row(row, model, elo_weight=elo_weight, stats=stats, config=config)
        mb = ub = 0.0
        for o in OUTCOMES:
            y = 1.0 if o == actual else 0.0
            mb += (p[o] - y) ** 2
            ub += (1 / 3 - y) ** 2
        model_b.append(mb)
        unif_b.append(ub)
    return model_b, unif_b


def skill_ci(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 12345,
    model: engine.ProbabilityModel | None = None,
    config: dict | None = None,
    elo_weight: float = 0.0,
) -> dict:
    """Bootstrap 95% CI on the skill-vs-uniform score.

    Resamples matches with replacement `n_boot` times; each replicate's skill is
    ``1 - mean(model_brier) / mean(uniform_brier)``. Reports the point estimate
    with a percentile interval, so a headline like "+12% over random" carries its
    uncertainty instead of standing as a bare number (the CR's #9 honesty ask).

    Note on "Brier vs market" (CR #10): a true market skill score needs the
    historical *closing odds* for these holdout tournaments, which we do not have
    (market_odds.csv holds only forward 2026 lines, with no results yet). Until
    those are sourced, this CI on the vs-uniform skill is the honest substitute —
    it quantifies how solid the "better than random" claim is.
    """
    import random

    mb, ub = per_match_brier(df, model=model, config=config, elo_weight=elo_weight)
    n = len(mb)
    if n == 0:
        return {"skill": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0, "n_boot": n_boot, "n": 0}
    point = 1 - (sum(mb) / n) / (sum(ub) / n)
    rng = random.Random(seed)
    skills: list[float] = []
    for _ in range(n_boot):
        sm = su = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sm += mb[j]
            su += ub[j]
        skills.append(1 - sm / su if su else 0.0)
    skills.sort()
    lo = skills[int(0.025 * n_boot)]
    hi = skills[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "skill": round(point, 4),
        "ci95_lo": round(lo, 4),
        "ci95_hi": round(hi, 4),
        "n_boot": n_boot,
        "n": n,
    }


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


# --- parameter fitting -------------------------------------------------------

# The default grid `fit_report` searches. Centred on the shipped constants so
# the fit can confirm or move them. K = FIFA points per goal of supremacy;
# BASE_TOTAL = baseline expected goals per match.
DEFAULT_GRID: dict[str, list[float]] = {
    "K": [float(k) for k in range(150, 401, 10)],          # 150..400 step 10
    "BASE_TOTAL": [round(2.0 + 0.1 * i, 1) for i in range(15)],  # 2.0..3.4
}

_METRICS = ("log_loss", "brier")


def _metric_sum(df: pd.DataFrame, overrides: dict, metric: str) -> tuple[float, int]:
    """Total (not mean) of `metric` over df under engine overrides, plus n.

    Returned as a sum so several frames can be pooled before averaging — needed
    for leave-one-tournament-out CV where each fold contributes its own matches.
    """
    m = evaluate(df, overrides=overrides)
    return getattr(m, metric) * m.n, m.n


def _product(grid: dict[str, list[float]]):
    """Yield every combination of the grid as an {param: value} dict."""
    import itertools
    names = list(grid)
    for combo in itertools.product(*(grid[n] for n in names)):
        yield dict(zip(names, combo))


def grid_fit(
    df: pd.DataFrame,
    grid: dict[str, list[float]] | None = None,
    metric: str = "log_loss",
) -> dict:
    """Exhaustive grid search for the constants that minimise `metric` on df.

    Returns the best combination, its score, and the score at the *current*
    shipped constants so the improvement (if any) is explicit.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    grid = grid or DEFAULT_GRID
    best: dict | None = None
    for combo in _product(grid):
        s, n = _metric_sum(df, combo, metric)
        score = s / n if n else float("inf")
        if best is None or score < best["score"]:
            best = {"params": combo, "score": score}
    # score at the shipped defaults (empty override = use current engine values)
    base_s, base_n = _metric_sum(df, {}, metric)
    return {
        "metric": metric,
        "best_params": best["params"],
        "best_score": round(best["score"], 4),
        "default_score": round(base_s / base_n, 4),
        "improvement": round(base_s / base_n - best["score"], 4),
        "n": base_n,
    }


def cv_fit(
    frames: dict[str, pd.DataFrame],
    grid: dict[str, list[float]] | None = None,
    metric: str = "log_loss",
) -> dict:
    """Leave-one-tournament-out cross-validated fit.

    For each tournament: fit the grid on the *other* tournaments, then score the
    fitted params — and, separately, the shipped defaults — on the held-out
    tournament the fit never saw. Pooling the held-out scores gives an honest
    out-of-sample comparison (fitted vs default) that cannot overfit, directly
    answering the CR's "don't pick K by intuition" while avoiding the in-sample
    trap of fitting and reporting on the same matches.
    """
    grid = grid or DEFAULT_GRID
    labels = list(frames)
    folds: list[dict] = []
    fit_sum = def_sum = 0.0
    total_n = 0
    for held in labels:
        train = pd.concat([frames[l] for l in labels if l != held], ignore_index=True)
        test = frames[held]
        fitted = grid_fit(train, grid, metric)["best_params"]
        f_s, f_n = _metric_sum(test, fitted, metric)
        d_s, d_n = _metric_sum(test, {}, metric)
        folds.append({
            "held_out": held,
            "n": f_n,
            "fitted_params": fitted,
            "fitted_score": round(f_s / f_n, 4),
            "default_score": round(d_s / d_n, 4),
            "delta": round(d_s / d_n - f_s / f_n, 4),  # +ve = fit helped
        })
        fit_sum += f_s; def_sum += d_s; total_n += f_n
    return {
        "metric": metric,
        "folds": folds,
        "pooled_fitted": round(fit_sum / total_n, 4),
        "pooled_default": round(def_sum / total_n, 4),
        "pooled_improvement": round((def_sum - fit_sum) / total_n, 4),
        "n": total_n,
    }


def fit_report(
    sources: dict[str, str] | None = None,
    grid: dict[str, list[float]] | None = None,
    metric: str = "log_loss",
) -> dict:
    """Full fitting report: cross-validated verdict + a final all-data fit.

    1. `cv_fit` measures, out of sample, whether *re-fitting* K/BASE_TOTAL beats
       the shipped defaults — the honest test of "should we tune these?".
    2. `grid_fit` on all tournaments pooled gives the single best constants to
       actually ship, if (1) says fitting helps.

    EXPERT_W is intentionally absent: none of the historical holdouts carry
    expert scorelines, so a sweep of EXPERT_W would be a silent no-op. Fitting it
    honestly requires expert columns on the backtest data, which we do not have.
    """
    sources = sources or _discover_sources()
    frames = {l: pd.read_csv(p) for l, p in sources.items() if os.path.exists(p)}
    if not frames:
        return {"error": "no holdout CSVs found", "looked_for": sources}
    grid = grid or DEFAULT_GRID
    cv = cv_fit(frames, grid, metric)
    pooled_df = pd.concat(list(frames.values()), ignore_index=True)
    final = grid_fit(pooled_df, grid, metric)
    return {
        "grid": {k: [v[0], v[-1], len(v)] for k, v in grid.items()},  # lo, hi, count
        "cross_validation": cv,
        "final_fit_all_data": final,
        "current": {"K": engine.K, "BASE_TOTAL": engine.BASE_TOTAL,
                    "EXPERT_W": engine.EXPERT_W},
        "expert_w_note": ("EXPERT_W not fitted: no holdout tournament carries "
                          "exp_home/exp_away, so it cannot be measured here."),
    }


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
    pooled["skill_ci"] = skill_ci(pooled_df)
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
    ci = pl.get("skill_ci")
    if ci:
        print(f"  skill 95% CI (bootstrap, {ci['n_boot']} resamples): "
              f"[{ci['ci95_lo']:+.1%}, {ci['ci95_hi']:+.1%}]  "
              f"(point {ci['skill']:+.1%}, n={ci['n']})")
        print("  note: vs-market Brier needs historical closing odds we don't "
              "have for these tournaments; this CI is the honest substitute.")
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


def _print_fit(rep: dict) -> None:
    if "error" in rep:
        print(f"\n[fit] {rep['error']}: {rep.get('looked_for')}\n")
        return
    g = rep["grid"]
    print("\n=== Parameter fit (minimise log-loss) ===")
    print("Grid: " + ", ".join(f"{k} {v[0]:g}..{v[1]:g} ({v[2]})" for k, v in g.items()))

    cv = rep["cross_validation"]
    print(f"\nLeave-one-tournament-out CV ({cv['metric']}, {cv['n']} matches):")
    print(f"  {'held-out':>10} {'n':>4} {'fitted':>8} {'default':>8} {'delta':>7}  fitted params")
    for f in cv["folds"]:
        pstr = ", ".join(f"{k}={v:g}" for k, v in f["fitted_params"].items())
        print(f"  {f['held_out']:>10} {f['n']:>4} {f['fitted_score']:>8.4f} "
              f"{f['default_score']:>8.4f} {f['delta']:>+7.4f}  {pstr}")
    print(f"  {'POOLED':>10} {cv['n']:>4} {cv['pooled_fitted']:>8.4f} "
          f"{cv['pooled_default']:>8.4f} {cv['pooled_improvement']:>+7.4f}")
    verdict = ("re-fitting helps out of sample — adopt the all-data fit"
               if cv["pooled_improvement"] > 0
               else "shipped constants already match/beat any re-fit — keep them")
    print(f"  => {verdict}")

    fin = rep["final_fit_all_data"]
    cur = rep["current"]
    print(f"\nBest fit on ALL data ({fin['metric']}): "
          + ", ".join(f"{k}={v:g}" for k, v in fin["best_params"].items()))
    print(f"  in-sample {fin['metric']}: {fin['best_score']:.4f} "
          f"vs default {fin['default_score']:.4f} ({fin['improvement']:+.4f})")
    print(f"  current shipped: K={cur['K']:g}, BASE_TOTAL={cur['BASE_TOTAL']:g}, "
          f"EXPERT_W={cur['EXPERT_W']:g}")
    print(f"  {rep['expert_w_note']}\n")


def main(argv=None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Backtest the probability engine.")
    ap.add_argument("--csv", default=None, help="historical matches CSV "
                    "(default: data/backtest_2022.csv)")
    ap.add_argument("--holdout", action="store_true",
                    help="multi-tournament holdout across all data/backtest_*.csv")
    ap.add_argument("--fit", action="store_true",
                    help="cross-validated fit of K/BASE_TOTAL across all "
                    "data/backtest_*.csv (minimise log-loss)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    if args.fit:
        rep = fit_report()
        if args.json:
            print(json.dumps(rep, ensure_ascii=False, indent=2))
        else:
            _print_fit(rep)
        return

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
