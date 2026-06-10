# Backtest Results

> **These numbers are produced at runtime, not hand-keyed.** Regenerate them with
> the commands below and paste the output here whenever the model changes. The
> README links to this file instead of quoting a headline number that can drift.
>
> Last regenerated: **2026-06-09** · engine constants at that commit:
> `K=240`, `BASE_TOTAL=2.6`, `EXPERT_W=0.85`, `DC_RHO=-0.06`, `ELO_WEIGHT=0.0`.

## How to regenerate

```bash
python -m src.backtest                 # single-tournament (2022 WC) report
python -m src.backtest --holdout       # multi-tournament holdout (294 matches)
python -m src.backtest --holdout --json  # machine-readable
python -m src.backtest --fit           # re-fit K / BASE_TOTAL via CV
```

The harness re-runs the **current** `ProbabilityModel` against historical CSVs in
`data/backtest_*.csv`, so the metrics always reflect the engine as it stands —
there are no cached scores. A lower Brier / log-loss is better.

---

## Headline (2022 World Cup, 64 matches)

| Metric | Model | Uniform (1/3) | Base-rate |
|--------|------:|--------------:|----------:|
| Brier | **0.5895** | 0.6667 | 0.6421 |
| Log-loss | **1.0037** | 1.0986 | 1.0622 |
| Accuracy | **56.3%** | 45.3% | 45.3% |

Skill vs uniform: **+11.6%** · vs base-rate: **+8.2%**

---

## Leakage ledger (provable, not just claimed)

Each holdout is only honest if its ratings/signals were knowable **before** the
tournament started. That promise is recorded per tournament in
[`data/backtest_meta.json`](data/backtest_meta.json) and **enforced** by
`src.backtest.leakage_check()` — which asserts `ratings_asof <= tournament_start`
*and* that `tournament_start` matches the first match date in the CSV (so a stale
manifest can't pass). CI runs it on every push (`python -m src.backtest --leakage`).

| Tournament | ratings as-of | tournament start | source |
|------------|---------------|------------------|--------|
| WC 2022    | 2022-10-06 | 2022-11-20 | Official FIFA ranking, last pre-WC release |
| Euro 2020  | 2021-06-10 | 2021-06-11 | Derived Elo from results before kickoff |
| Euro 2024  | 2024-06-13 | 2024-06-14 | Derived Elo from results before kickoff |
| WC 2014    | 2014-06-11 | 2014-06-12 | Derived Elo from results before kickoff |
| WC 2018    | 2018-06-13 | 2018-06-14 | Derived Elo from results before kickoff |

```bash
python -m src.backtest --leakage        # [OK] / exits non-zero on any violation
```

## Multi-tournament holdout (294 matches, no leakage)

Each tournament uses **pre-tournament** ratings only (FIFA points as-of the event
for 2022; derived World-Football Elo for the rest — see `fetch_holdout.py`), so a
team's later success never feeds back into its own prediction. The as-of dates
above are checked automatically — see the leakage ledger.

| Tournament | n | Brier | Log-loss | Accuracy | Skill vs uniform |
|------------|--:|------:|---------:|---------:|-----------------:|
| WC 2022    | 64 | 0.5895 | 1.0037 | 56.3% | +11.6% |
| Euro 2020  | 51 | 0.5360 | 0.9098 | 62.7% | +19.6% |
| Euro 2024  | 51 | 0.6185 | 1.0298 | 47.1% | +7.2% |
| WC 2014    | 64 | 0.5586 | 0.9402 | 60.9% | +16.2% |
| WC 2018    | 64 | 0.5819 | 0.9804 | 51.6% | +12.7% |
| **POOLED** | **294** | **0.5769** | **0.9730** | **55.8%** | **+13.5%** |

**Pooled skill 95% CI** (bootstrap, 2000 resamples): **[+8.4%, +18.8%]**
(point +13.5%, n=294).

> A proper vs-market Brier would need historical closing odds, which we don't have
> for these tournaments. The bootstrap CI on skill-vs-uniform is the honest
> substitute — it shows the edge is real and not a single lucky tournament.

---

## Does each signal earn its place? (pooled, 294 matches)

Out-of-sample config comparison — does adding a signal lower the pooled Brier?

| Config | elo_w | Brier | Log-loss | Accuracy |
|--------|------:|------:|---------:|---------:|
| **all** | 0.00 | **0.5745** | **0.9696** | 56.8% |
| +form  | 0.00 | 0.5749 | 0.9705 | 57.5% |
| +h2h   | 0.00 | 0.5761 | 0.9716 | 55.8% |
| fifa_only | 0.00 | 0.5769 | 0.9730 | 55.8% |
| +elo   | 0.00 | 0.5769 | 0.9730 | 55.8% |

**`all` wins** — the full signal blend beats `fifa_only` out of sample, so H2H and
form each pull their weight. Elo at `ELO_WEIGHT=0.0` is a no-op on the pooled set
(it helped only on 2022), which is why it stays disabled by default.

---

## Supremacy mapping: linear vs log-ratio (CR §3A, measured)

The review flagged that the linear `sup = (r_home − r_away)/K` mapping is
"floor-bound on weak opponents" and proposed `sup = α·ln(r_home/r_away)`. Rather
than adopt it on intuition, it's implemented as `engine.SUP_MODE = "logratio"`
(default `"linear"`) and **fit on the holdout**:

| Mapping | pooled Brier | pooled LogLoss |
|---------|-------------:|---------------:|
| linear (K=240, shipped) | 0.5769 | 0.9730 |
| logratio (fitted α≈7.0)  | **0.5744** | **0.9702** |

In-sample, log-ratio wins (−0.0025 Brier) and trims the favourites ~1pp
(France/Spain title odds 16%→15%, the field tightens) — exactly the intended
effect. **But leave-one-tournament-out CV is the deciding test, and there the
gain collapses to −0.0002 log-loss** (helps 4 tournaments, hurts Euro-2024) — a
wash, the same verdict reached for the Elo blend.

**Decision: keep `SUP_MODE = "linear"`.** The structural fix is implemented,
validated, and toggle-able (`engine.SUP_MODE`), but the default is not flipped
because the out-of-sample evidence doesn't justify it. Re-run the fit if more
holdout tournaments are added.
