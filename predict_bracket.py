"""User-bracket knockout prediction (stage 03 of the prediction game).

Takes the R32 pairings exactly as they appear in the user's game bracket
(winners fold sequentially: 1v2, 3v4, ...), applies the June-2026 injury deltas,
h2h + form supremacy, and the half home edge for hosts (knockout.KNOCKOUT_HOST_ADV).
Outputs: per-tie win% (analytic, through ET + capped pens), chalk picks per round,
and Monte-Carlo reach probabilities for every team.
"""
import random
import sys

from src import engine, knockout
from src.models import DataStore

sys.stdout.reconfigure(encoding="utf-8")

NEWS = {
    'NED': -80, 'BRA': -100, 'MAR': -55, 'JPN': -45, 'GER': -20,
    'SCO': -15, 'CAN': -25, 'ESP': -20, 'ARG': -25, 'ENG': -10, 'USA': -15,
}

R32 = [
    ('GER', 'AUS'), ('FRA', 'SWE'), ('KOR', 'CAN'), ('NED', 'BRA'),
    ('COL', 'CRO'), ('ESP', 'AUT'), ('USA', 'ALG'), ('BEL', 'CZE'),
    ('MAR', 'JPN'), ('ECU', 'SEN'), ('MEX', 'SCO'), ('ENG', 'NOR'),
    ('ARG', 'URU'), ('TUR', 'IRN'), ('SUI', 'EGY'), ('POR', 'CIV'),
]

N = 20000
ds = DataStore.load("data")
names = dict(zip(ds.teams.team_id, ds.teams.name_en))
base = dict(zip(ds.teams.team_id, ds.teams.fifa_points))
rt = {t: base[t] + NEWS.get(t, 0) for t in base}

h2h = knockout.build_h2h(ds)
form = knockout.build_form(ds)


def sups(a, b):
    """Combined non-rating supremacy for a (home-slot) vs b: h2h + form + host."""
    s = h2h.get((a, b), 0.0) + knockout._form_sup(form, a, b)
    if a in engine.HOSTS:
        s += knockout.KNOCKOUT_HOST_ADV
    if b in engine.HOSTS:
        s -= knockout.KNOCKOUT_HOST_ADV
    return s


def ko_prob(a, b):
    """P(a beats b) through 90' + ET + capped shootout (analytic)."""
    s = sups(a, b)
    p = engine.ProbabilityModel().pre_match(rt[a], rt[b], neutral=True, h2h_sup=s)
    ph, pd_, pa = p['p_home'], p['p_draw'], p['p_away']
    frac = max(1 - engine.SHOOTOUT_CAP, min(engine.SHOOTOUT_CAP, ph / (ph + pa + 1e-9)))
    lh, la = engine.expected_goals(rt[a], rt[b], neutral=True, h2h_sup=s)
    et = engine.probs_from_lambdas(lh * engine.ET_LAMBDA_SCALE,
                                   la * engine.ET_LAMBDA_SCALE, dixon_coles=False)
    return ph + pd_ * (et['p_home'] + et['p_draw'] * frac)


def sim_ko(a, b, rng):
    wi, _ = engine.resolve_knockout(rt[a], rt[b], rng, neutral=True, h2h_sup=sups(a, b))
    return a if wi == 0 else b


# --- Monte-Carlo reach probabilities -----------------------------------------
ROUNDS = ["R16", "QF", "SF", "FINAL", "CHAMP"]
reach = {t: {r: 0 for r in ROUNDS} for pair in R32 for t in pair}
rng = random.Random(2026)
for _ in range(N):
    w = [sim_ko(a, b, rng) for a, b in R32]
    for t in w:
        reach[t]["R16"] += 1
    for rnd, size in (("QF", 16), ("SF", 8), ("FINAL", 4)):
        w = [sim_ko(w[i], w[i + 1], rng) for i in range(0, size, 2)]
        for t in w:
            reach[t][rnd] += 1
    champ = sim_ko(w[0], w[1], rng)
    reach[champ]["CHAMP"] += 1

# --- Chalk bracket (pick the favourite at every node) -------------------------
print("R32 PICKS")
print("=" * 64)
cur = []
for a, b in R32:
    p = ko_prob(a, b)
    fav = a if p >= 0.5 else b
    cur.append(fav)
    conf = ("STRONG" if abs(p - 0.5) > 0.20 else
            "CLEAR" if abs(p - 0.5) > 0.10 else
            "SLIGHT" if abs(p - 0.5) > 0.04 else "TOSS-UP")
    print(f"  {names[a]:16s} {p*100:5.1f}%  vs  {(1-p)*100:5.1f}%  {names[b]:16s}"
          f"  -> {names[fav]:14s} [{conf}]")

for rnd in ("R16", "QF", "SF", "FINAL"):
    print(f"\n{rnd} PICKS (chalk)")
    print("=" * 64)
    nxt = []
    for i in range(0, len(cur), 2):
        a, b = cur[i], cur[i + 1]
        p = ko_prob(a, b)
        fav = a if p >= 0.5 else b
        nxt.append(fav)
        print(f"  {names[a]:16s} {p*100:5.1f}%  vs  {(1-p)*100:5.1f}%  {names[b]:16s}"
              f"  -> {names[fav]}")
    cur = nxt

print(f"\nCHAMPION PICK: {names[cur[0]]}")

print("\nMONTE-CARLO REACH % (top 12 by title)")
print("=" * 64)
top = sorted(reach, key=lambda t: -reach[t]["CHAMP"])[:12]
print(f"  {'team':16s} {'R16':>6s} {'QF':>6s} {'SF':>6s} {'FINAL':>6s} {'TITLE':>6s}")
for t in top:
    r = reach[t]
    print(f"  {names[t]:16s} " + " ".join(f"{r[k]/N*100:5.1f}%" for k in ROUNDS))
