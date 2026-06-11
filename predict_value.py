"""Value metric: model+market probabilities vs the GOLAZO app's frozen odds.

GOLAZO scoring: correct 1X2 pick pays the app's odds for that outcome (wrong = 0),
plus an exact-score bonus by scoreline rarity. The app odds are frozen at
submission while the real market moves, so EV per pick is:

    EV(scoreline s) = P_blend(outcome(s)) * app_odds(outcome(s)) + P_model(s) * bonus(s)

P_blend = mean of injury-adjusted model prob and de-vigged bookmaker prob
(market_odds.csv) when available. The joker (x3) is worth +2 extra match-EVs,
so the best joker spot is simply the highest-EV match.

Usage: python predict_value.py [--days N]   (default: all priced matches)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

from src.models import DataStore
from src import engine
from src.engine import _poisson_pmf, _dc_tau

NEWS = {
    'NED': -80, 'BRA': -100, 'MAR': -55, 'JPN': -45, 'GER': -20,
    'SCO': -15, 'CAN': -25, 'ESP': -20, 'ARG': -25, 'ENG': -10, 'USA': -15,
}

ALIASES = {
    'Korea Republic': 'South Korea', 'Türkiye': 'Turkey',
    "Côte d'Ivoire": 'Ivory Coast', 'IR Iran': 'Iran', 'Curaçao': 'Curacao',
}

# Exact-score bonus tiers (unordered goal pair -> bonus points)
BONUS_TIERS = [
    ({(0, 0), (0, 1), (1, 1), (0, 2), (1, 2)}, 2.0),
    ({(2, 2), (0, 3), (1, 3)}, 2.7),
    ({(2, 3), (0, 4), (3, 3)}, 3.6),
    ({(1, 4), (2, 4), (0, 5), (4, 4)}, 4.8),
]
BONUS_OTHER = 6.4


def bonus(i: int, j: int) -> float:
    key = (min(i, j), max(i, j))
    for tier, pts in BONUS_TIERS:
        if key in tier:
            return pts
    return BONUS_OTHER


def main() -> None:
    ds = DataStore.load('data')
    names = dict(zip(ds.teams.team_id, ds.teams.name_en))
    name_to_id = {v: k for k, v in names.items()}
    rt = {t: r + NEWS.get(t, 0)
          for t, r in zip(ds.teams.team_id, ds.teams.fifa_points)}

    # match lookup by unordered pair (group stage)
    pair_to_match = {frozenset((m.home_id, m.away_id)): m
                     for _, m in ds.matches.iterrows()}

    # de-vigged market probs by unordered pair, oriented to our matches.csv home
    market = {}
    for _, m in ds.matches.iterrows():
        a = ds.market_anchor(m.match_id)
        if a:
            market[m.match_id] = a['market']

    app = pd.read_csv('data/app_odds.csv')
    rows = []
    for _, r in app.iterrows():
        h_name = ALIASES.get(r.home_team, r.home_team)
        a_name = ALIASES.get(r.away_team, r.away_team)
        h, a = name_to_id.get(h_name), name_to_id.get(a_name)
        if h is None or a is None:
            print(f"!! unmapped team in app row: {r.home_team} / {r.away_team}")
            continue
        m = pair_to_match.get(frozenset((h, a)))
        if m is None:
            print(f"!! no fixture for {h}-{a}")
            continue
        # orient everything to the APP's home/away
        flipped = (m.home_id != h)
        lh, la = engine.expected_goals(
            rt[m.home_id], rt[m.away_id], neutral=not ds.is_host(m.home_id),
            expert=ds.expert_for(m.match_id),
            h2h_sup=ds.h2h_supremacy_for(m.home_id, m.away_id),
            form_sup=ds.form_supremacy_for(m.home_id, m.away_id),
        )
        if flipped:
            lh, la = la, lh
        p_model = engine.probs_from_lambdas(lh, la)
        mk = market.get(m.match_id)
        if mk and flipped:
            mk = {'p_home': mk['p_away'], 'p_draw': mk['p_draw'], 'p_away': mk['p_home']}
        blend = {k: (p_model[k] + mk[k]) / 2 if mk else p_model[k]
                 for k in ('p_home', 'p_draw', 'p_away')}
        odds = {'p_home': float(r.odds_home), 'p_draw': float(r.odds_draw),
                'p_away': float(r.odds_away)}

        # best scoreline: outcome EV + exact bonus EV over the DC grid
        grid = {}
        for i in range(7):
            for j in range(7):
                grid[(i, j)] = _poisson_pmf(i, lh) * _poisson_pmf(j, la) * _dc_tau(i, j, lh, la)
        tot = sum(grid.values())
        best_s, best_ev, best_out_ev = None, -1.0, 0.0
        for (i, j), pr in grid.items():
            out = 'p_home' if i > j else ('p_draw' if i == j else 'p_away')
            ev = blend[out] * odds[out] + (pr / tot) * bonus(i, j)
            if ev > best_ev:
                best_s, best_ev, best_out_ev = (i, j), ev, blend[out] * odds[out]
        out_key = ('p_home' if best_s[0] > best_s[1]
                   else 'p_draw' if best_s[0] == best_s[1] else 'p_away')
        pick_team = (r.home_team if out_key == 'p_home'
                     else r.away_team if out_key == 'p_away' else 'DRAW')
        rows.append({
            'date': r.kickoff_date, 'match': f"{r.home_team}-{r.away_team}",
            'pick': pick_team, 'score': f"{best_s[0]}-{best_s[1]}",
            'P_blend': blend[out_key], 'app_odds': odds[out_key],
            'EV': best_ev, 'outcome_EV': best_out_ev,
            'edge': blend[out_key] * odds[out_key] - 1.0,
        })

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print("ALL PRICED MATCHES — ranked by total EV (best joker spots on top)")
    print(df.sort_values('EV', ascending=False).to_string(index=False,
          formatters={'P_blend': '{:.0%}'.format, 'EV': '{:.2f}'.format,
                      'outcome_EV': '{:.2f}'.format, 'edge': '{:+.0%}'.format}))
    print("\nedge = P_blend x app_odds - 1  (positive = the app underprices this pick)")


if __name__ == '__main__':
    main()
