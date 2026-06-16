"""Live calibration diagnostic — run after each matchday to catch real model
drift (vs variance) early.

For every finished group game in matches.csv it recomputes the model's pre-match
1X2 (same inputs as the sim: host-aware venue, expert blend, h2h/form supremacy)
and reports:
  - direction accuracy OVERALL and on DECISIVE games only (draws are never an
    argmax pick, so overall accuracy is depressed purely by the draw rate);
  - expected vs actual draws, with a z-score (is the draw count variance or signal?);
  - live Brier / log-loss vs the historical backtest baseline (~0.59 / ~1.00).

A bug/regression shows up as bad numbers on DECISIVE games or a Brier far above
the historical baseline that is NOT explained by a draw spike. A pure draw spike
(high z on draws, but decisive-game accuracy and per-match probs still sane) is a
calibration/variance story, not a code bug — do NOT refit constants to it.
"""
import math
import sys

from src import engine
from src.models import DataStore

sys.stdout.reconfigure(encoding='utf-8')

# Historical backtest baselines (python -m src.backtest), for context.
HIST_BRIER = 0.59
HIST_LOGLOSS = 1.00


def main() -> None:
    ds = DataStore.load('data')
    fin = ds.matches[ds.matches.status == 'finished']

    exp_draw = brier = logloss = 0.0
    correct = dec_correct = dec_n = 0
    rows = []
    for _, m in fin.iterrows():
        h, a = m.home_id, m.away_id
        rh = float(ds.teams[ds.teams.team_id == h].fifa_points.iloc[0])
        ra = float(ds.teams[ds.teams.team_id == a].fifa_points.iloc[0])
        lh, la = engine.expected_goals(rh, ra, neutral=not ds.is_host(h),
                                       expert=ds.expert_for(m.match_id),
                                       h2h_sup=ds.h2h_supremacy_for(h, a),
                                       form_sup=ds.form_supremacy_for(h, a))
        p = engine.probs_from_lambdas(lh, la)
        hg, ag = int(m.home_goals), int(m.away_goals)
        actual = 'p_home' if hg > ag else ('p_draw' if hg == ag else 'p_away')
        pred = max(('p_home', 'p_draw', 'p_away'), key=lambda k: p[k])
        hit = pred == actual
        correct += hit
        if actual != 'p_draw':
            dec_n += 1
            dec_correct += hit
        for k in ('p_home', 'p_draw', 'p_away'):
            brier += (p[k] - (1.0 if k == actual else 0.0)) ** 2
        logloss += -math.log(max(p[actual], 1e-12))
        exp_draw += p['p_draw']
        rows.append((m.match_id, f"{h}-{a}", f"{hg}-{ag}", actual == 'p_draw',
                     p['p_home'], p['p_draw'], p['p_away'], hit))

    n = len(rows)
    if n == 0:
        print("no finished games yet")
        return
    actual_draws = sum(1 for r in rows if r[3])
    pbar = exp_draw / n
    mean = n * pbar
    sd = math.sqrt(n * pbar * (1 - pbar)) or 1e-9
    z = (actual_draws - mean) / sd

    print(f"=== LIVE DIAGNOSTIC ({n} finished group games) ===")
    print(f"direction accuracy : {correct}/{n} = {correct/n*100:.0f}% overall")
    print(f"  on DECISIVE games: {dec_correct}/{dec_n} = "
          f"{dec_correct/dec_n*100:.0f}%  (draws excluded — unpickable by argmax)")
    print(f"Brier (per match)  : {brier/n:.3f}   (historical baseline {HIST_BRIER})")
    print(f"log-loss (per match): {logloss/n:.3f}   (historical baseline {HIST_LOGLOSS})")
    print(f"draws: expected {mean:.1f}±{sd:.1f}, actual {actual_draws}  ->  z={z:.2f}")
    if abs(z) >= 2:
        print(f"  ⚠️  draw count is {z:+.1f}σ from model — watch, but z alone is not a bug")
    print(f"\n{'mt':3} {'match':10} {'res':5} {'D?':3} {'pH/pD/pA':14} hit?")
    for mid, mt, res, isd, ph, pd_, pa, hit in rows:
        print(f"{mid:3} {mt:10} {res:5} {'Y' if isd else '.':3} "
              f"{ph*100:.0f}/{pd_*100:.0f}/{pa*100:<6.0f} {'OK' if hit else 'miss'}")


if __name__ == '__main__':
    main()
