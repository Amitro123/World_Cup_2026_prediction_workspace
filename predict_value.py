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

import pandas as pd

from src import engine
from src.engine import _dc_tau, _poisson_pmf
from src.models import DataStore

sys.stdout.reconfigure(encoding="utf-8")

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


def outcome_of(i: int, j: int) -> str:
    return 'p_home' if i > j else ('p_draw' if i == j else 'p_away')


def team_of(out_key: str, row) -> str:
    return (row.home_team if out_key == 'p_home'
            else row.away_team if out_key == 'p_away' else 'DRAW')


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

        # DC scoreline grid
        grid = {}
        for i in range(7):
            for j in range(7):
                grid[(i, j)] = _poisson_pmf(i, lh) * _poisson_pmf(j, la) * _dc_tau(i, j, lh, la)
        tot = sum(grid.values())

        # --- field-share proxy + leverage -----------------------------------
        # In a large pool ~everyone backs the favourite, so the points from a
        # chalk pick give little RANK advantage. crowd(outcome) ~ the de-vigged
        # implied prob from the app's own odds (short odds => crowded). The
        # leverage score down-weights crowded outcomes: it rewards a contrarian
        # pick that, when it lands, leapfrogs the field. This is the same logic
        # we apply to jokers, now applied to the pick itself (CR: the Spain 0-0
        # miss — a +EV, high-leverage draw we under-weighted by chasing raw EV).
        inv = {k: 1.0 / odds[k] for k in odds}
        s_inv = sum(inv.values())
        crowd = {k: inv[k] / s_inv for k in inv}
        dir_ev = {k: blend[k] * odds[k] for k in odds}
        leverage = {k: dir_ev[k] * (1.0 - crowd[k]) for k in odds}

        def modal_score(target):
            return max((s for s in grid if outcome_of(*s) == target), key=grid.get)

        # EV-optimal scoreline (direction odds + exact bonus)
        best_s, best_ev = None, -1.0
        for (i, j), pr in grid.items():
            out = outcome_of(i, j)
            ev = blend[out] * odds[out] + (pr / tot) * bonus(i, j)
            if ev > best_ev:
                best_s, best_ev = (i, j), ev
        ev_out = outcome_of(*best_s)

        # leverage-optimal direction (rank play), with its modal scoreline
        lev_out = max(leverage, key=leverage.get)
        lev_s = modal_score(lev_out)
        fav_out = min(odds, key=odds.get)
        rows.append({
            'date': r.kickoff_date, 'match': f"{r.home_team}-{r.away_team}",
            'ev_pick': team_of(ev_out, r), 'ev_score': f"{best_s[0]}-{best_s[1]}",
            'EV': best_ev,
            'lev_pick': team_of(lev_out, r), 'lev_score': f"{lev_s[0]}-{lev_s[1]}",
            'lev': leverage[lev_out],
            'edge': blend[ev_out] * odds[ev_out] - 1.0,
            # chalk trap: heavy favourite (odds<=1.20) where EV says take the
            # crowd; the leverage pick diverging is the flag to consider it.
            'trap': odds[fav_out] <= 1.20 and ev_out == fav_out and lev_out != ev_out,
        })

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    fmt = {'EV': '{:.2f}'.format, 'lev': '{:.2f}'.format, 'edge': '{:+.0%}'.format}
    print("=== EV VIEW — single-entry optimal (best joker spots on top) ===")
    print(df.sort_values('EV', ascending=False).to_string(index=False, formatters=fmt))
    print("\n=== LEVERAGE VIEW — rank-optimal contrarian picks in a big field ===")
    lv = df[['date', 'match', 'lev_pick', 'lev_score', 'lev', 'ev_pick', 'trap']]
    print(lv.sort_values('lev', ascending=False).to_string(index=False, formatters=fmt))
    traps = df[df['trap']]
    if not traps.empty:
        print("\n⚠️  CHALK TRAPS (heavy favourite, low rank value — weigh the leverage pick):")
        for _, t in traps.iterrows():
            print(f"   {t['match']}: EV says {t['ev_pick']} {t['ev_score']}, "
                  f"but leverage says {t['lev_pick']} {t['lev_score']} (lev {t['lev']:.2f})")
    print("\nedge = P_blend x app_odds - 1  |  lev = dir_EV x (1 - crowd_share)")


if __name__ == '__main__':
    main()
