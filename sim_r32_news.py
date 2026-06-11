"""News-adjusted R32 bracket simulation."""
import sys, random
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from src import engine

teams = pd.read_csv('data/teams.csv')
base = dict(zip(teams.team_id, teams.fifa_points))
names = dict(zip(teams.team_id, teams.name_en))
nm = lambda t: names.get(t, t)

# News-based injury deltas — Source: ESPN/AP/BBC/MARCA June 2026
news = {
    'NED': -80,   # Timber OUT, Xavi Simons OUT, de Ligt OUT, Schouten OUT
    'BRA': -100,  # Wesley OUT, Rodrygo OUT, Militao OUT, Estevao OUT + Neymar doubtful
    'MAR': -55,   # Aguerd OUT (key CB), Ezzalzouli OUT (key winger)
    'JPN': -45,   # Mitoma OUT, Minamino OUT
    'GER': -20,   # Gnabry OUT, Karl OUT
    'SCO': -15,   # Gilmour OUT -> teenager Fletcher
    'CAN': -25,   # Bombito OUT (best CB), Flores OUT
    'ESP': -20,   # Fermin Lopez OUT, Merino doubtful
    'ARG': -25,   # Romero/Molina not 100%, Balerdi out
    'ENG': -10,   # Livramento OUT
    'USA': -15,   # Cardoso OUT, Richards ankle (expected to play)
}

def rt(t):
    return base.get(t, 1500.0) + news.get(t, 0)

def ko_prob(a, b):
    ra, rb = rt(a), rt(b)
    probs = engine.ProbabilityModel().pre_match(ra, rb, neutral=True)
    ph, pd_, pa = probs['p_home'], probs['p_draw'], probs['p_away']
    frac = max(1 - engine.SHOOTOUT_CAP, min(engine.SHOOTOUT_CAP, ph / (ph + pa + 1e-9)))
    lh, la = engine.expected_goals(ra, rb, neutral=True)
    et = engine.probs_from_lambdas(lh * engine.ET_LAMBDA_SCALE, la * engine.ET_LAMBDA_SCALE, dixon_coles=False)
    return ph + pd_ * (et['p_home'] + et['p_draw'] * frac)

def ko_prob_base(a, b):
    ra, rb = base.get(a, 1500.0), base.get(b, 1500.0)
    probs = engine.ProbabilityModel().pre_match(ra, rb, neutral=True)
    ph, pd_, pa = probs['p_home'], probs['p_draw'], probs['p_away']
    frac = max(1 - engine.SHOOTOUT_CAP, min(engine.SHOOTOUT_CAP, ph / (ph + pa + 1e-9)))
    lh, la = engine.expected_goals(ra, rb, neutral=True)
    et = engine.probs_from_lambdas(lh * engine.ET_LAMBDA_SCALE, la * engine.ET_LAMBDA_SCALE, dixon_coles=False)
    return ph + pd_ * (et['p_home'] + et['p_draw'] * frac)

def sim_ko(a, b, rng):
    wi, _ = engine.resolve_knockout(rt(a), rt(b), rng, neutral=True)
    return a if wi == 0 else b

N = 10000
rng = random.Random(2026)

R32 = [
    ('GER', 'SCO'), ('FRA', 'EGY'), ('KOR', 'CAN'), ('NED', 'BRA'),
    ('COL', 'CRO'), ('ESP', 'AUT'), ('USA', 'QAT'), ('BEL', 'KSA'),
    ('MAR', 'JPN'), ('ECU', 'SEN'), ('MEX', 'CIV'), ('ENG', 'NOR'),
    ('ARG', 'URU'), ('TUR', 'IRN'), ('SUI', 'ALG'), ('POR', 'PAN'),
]

title = {}
for _ in range(N):
    w = [sim_ko(a, b, rng) for a, b in R32]
    w = [sim_ko(w[i], w[i+1], rng) for i in range(0, 16, 2)]
    w = [sim_ko(w[i], w[i+1], rng) for i in range(0, 8, 2)]
    w = [sim_ko(w[i], w[i+1], rng) for i in range(0, 4, 2)]
    champ = sim_ko(w[0], w[1], rng)
    title[champ] = title.get(champ, 0) + 1

injury_notes = {
    'GER': 'Gnabry/Karl OUT — Musiala+Wirtz+Havertz still there',
    'SCO': 'Gilmour OUT -> teenager Fletcher',
    'FRA': 'Ekitike OUT (depth only) — Mbappe/Dembele/Olise full strength',
    'EGY': 'No significant injuries reported',
    'KOR': 'Minor withdrawals only',
    'CAN': 'Bombito OUT (best CB), Flores OUT — Alphonso Davies fit',
    'NED': 'Timber/Simons/de Ligt/Schouten OUT — major depth loss',
    'BRA': 'Wesley/Rodrygo/Militao OUT + Neymar doubtful — squad severely depleted',
    'COL': 'No major injuries reported',
    'CRO': 'Modric confirmed fit',
    'ESP': 'Fermin Lopez/Merino OUT — Yamal+Williams fit, still very strong',
    'AUT': 'No major injuries reported',
    'USA': 'Cardoso OUT, Richards ankle (expected to play)',
    'QAT': 'No major injuries reported',
    'BEL': 'No major injuries reported',
    'KSA': 'No major injuries reported',
    'MAR': 'Aguerd OUT (key CB) + Ezzalzouli OUT (key winger) — double blow',
    'JPN': 'Mitoma OUT + Minamino OUT — lost two best attackers',
    'ECU': 'No major injuries reported',
    'SEN': 'No major injuries reported',
    'MEX': 'No major injuries reported',
    'CIV': 'Minor withdrawals reported',
    'ENG': 'Livramento OUT, Reece James fit',
    'NOR': 'No major injuries reported',
    'ARG': 'Romero/Molina not fully fit, Messi expected available — minor concerns',
    'URU': 'Gimenez OUT (key CB)',
    'TUR': 'Arda Guler confirmed fit',
    'IRN': 'No major injuries reported',
    'SUI': 'No major injuries reported',
    'ALG': 'No major injuries reported',
    'POR': 'Full strength — Ronaldo, Dias, Bruno, Leao all available',
    'PAN': 'No major injuries reported',
}

flip_note = {
    'NED': '<<< FLIPPED: NED now favoured (was BRA 50.4%)',
    'MAR': '<<< FLIP RISK: MAR edge reduced (key injuries)',
}

print()
print('ROUND OF 32  -  News + Model combined  (injury-adjusted, June 11 2026)')
print('=' * 70)
favs = []
for a, b in R32:
    p = ko_prob(a, b)
    p_raw = ko_prob_base(a, b)
    da = news.get(a, 0)
    db = news.get(b, 0)
    fav = a if p >= 0.5 else b
    favs.append(fav)
    diff = (p - p_raw) * 100

    verdict = nm(a) if p >= 0.5 else nm(b)
    conf = 'STRONG' if abs(p - 0.5) > 0.20 else ('CLEAR' if abs(p - 0.5) > 0.10 else ('SLIGHT' if abs(p - 0.5) > 0.04 else 'TOSS-UP'))

    chg_str = ''
    if abs(diff) > 1.0:
        chg_str = f'  [was {p_raw*100:.1f}% raw -> {p*100:.1f}% adjusted]'

    print(f'  {nm(a):22s}  {p*100:5.1f}%  vs  {(1-p)*100:5.1f}%  {nm(b):22s}')
    print(f'    => {conf}: {verdict} wins{chg_str}')
    # News
    na = injury_notes.get(a, '')
    nb = injury_notes.get(b, '')
    if da != 0 or db != 0:
        if da != 0: print(f'    [!] {nm(a)}: {na}')
        if db != 0: print(f'    [!] {nm(b)}: {nb}')
    print()

print('=' * 70)
print(f'TITLE ODDS  ({N:,} MC simulations, news-adjusted)')
print('=' * 70)
for tid, cnt in sorted(title.items(), key=lambda x: -x[1]):
    if cnt == 0:
        continue
    pct = cnt / N * 100
    bar = '#' * int(pct * 2)
    inj = ' (injuries!)' if tid in news and news[tid] <= -50 else ''
    print(f'  {nm(tid):22s}  {pct:5.1f}%  {bar}{inj}')
