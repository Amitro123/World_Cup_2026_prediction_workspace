"""Golden Boot prediction on the user's bracket (stage 04 of the game).

Joint Monte-Carlo: group stage (all 72 games, same machinery as knockout.py)
+ the user's R32 bracket (reg + ET goals; shootout goals don't count).
Each team's sampled goals are allocated to its players multinomially by
goal_share (residual mass -> unnamed squad players). Injured players excluded.
"""
import sys
import random
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

from src.models import DataStore
from src import engine, knockout

NEWS = {
    'NED': -80, 'BRA': -100, 'MAR': -55, 'JPN': -45, 'GER': -20,
    'SCO': -15, 'CAN': -25, 'ESP': -20, 'ARG': -25, 'ENG': -10, 'USA': -15,
}

INJURED = {  # OUT for the tournament (June 2026 news) — cannot win the boot
    'Rodrygo', 'Wesley', 'Estevao', 'Eder Militao',           # BRA
    'Xavi Simons', 'Jurrien Timber', 'Matthijs de Ligt',      # NED
    'Kaoru Mitoma', 'Takumi Minamino',                        # JPN
    'Serge Gnabry',                                           # GER
    'Fermin Lopez',                                           # ESP
    'Hakim Ziyech',  # not injured but verify squad — keep? (see note in output)
}
INJURED.discard('Hakim Ziyech')  # no report of him out; keep him in

R32 = [
    ('GER', 'AUS'), ('FRA', 'SWE'), ('KOR', 'CAN'), ('NED', 'BRA'),
    ('COL', 'CRO'), ('ESP', 'AUT'), ('USA', 'ALG'), ('BEL', 'CZE'),
    ('MAR', 'JPN'), ('ECU', 'SEN'), ('MEX', 'SCO'), ('ENG', 'NOR'),
    ('ARG', 'URU'), ('TUR', 'IRN'), ('SUI', 'EGY'), ('POR', 'CIV'),
]

N = 10000
ds = DataStore.load("data")
names = dict(zip(ds.teams.team_id, ds.teams.name_en))

ctx = knockout._prepare(ds)
ctx["ratings"] = {t: r + NEWS.get(t, 0) for t, r in ctx["ratings"].items()}
rt, h2h, form = ctx["ratings"], ctx["h2h"], ctx["form"]

# Player rosters: team -> list of (name_en, share), excluding the injured.
rosters = defaultdict(list)
for _, p in ds.players.iterrows():
    if p["name_en"] in INJURED:
        continue
    rosters[p["team_id"]].append((p["name_en"], float(p["goal_share"])))


def sups(a, b):
    s = h2h.get((a, b), 0.0) + knockout._form_sup(form, a, b)
    if a in engine.HOSTS:
        s += knockout.KNOCKOUT_HOST_ADV
    if b in engine.HOSTS:
        s -= knockout.KNOCKOUT_HOST_ADV
    return s


def alloc(team, goals, rng, tally):
    """Distribute a team's goals among its players by share (rest -> squad)."""
    for _ in range(goals):
        u = rng.random()
        acc = 0.0
        for nm, sh in rosters.get(team, ()):
            acc += sh
            if u < acc:
                tally[(team, nm)] += 1
                break
        # else: unnamed squad player scored — ignored for the boot race


win_boot = defaultdict(int)
exp_goals = defaultdict(float)
rng = random.Random(2026)

for _ in range(N):
    tally = defaultdict(int)
    # group stage: 12 groups, 6 games each — rec[t]['gf'] = team's group goals
    for g in ctx["groups"]:
        _ranked, rec = knockout._simulate_group_fast(ctx, g, rng)
        for t, r in rec.items():
            alloc(t, r["gf"], rng, tally)
    # knockout: user's bracket, reg + ET goals count
    cur = list(R32)
    while cur:
        winners = []
        for a, b in cur:
            wi, info = engine.resolve_knockout(rt[a], rt[b], rng, neutral=True,
                                               h2h_sup=sups(a, b))
            hg, ag = info["reg"]
            if info["et"]:
                hg += info["et"][0]; ag += info["et"][1]
            alloc(a, hg, rng, tally)
            alloc(b, ag, rng, tally)
            winners.append(a if wi == 0 else b)
        cur = ([(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]
               if len(winners) > 1 else [])
    if tally:
        best = max(tally.values())
        tied = [k for k, v in tally.items() if v == best]
        for k in tied:
            win_boot[k] += 1 / len(tied)
    for k, v in tally.items():
        exp_goals[k] += v / N

print(f"GOLDEN BOOT  ({N:,} joint sims: groups + user bracket, injury-adjusted)")
print("=" * 66)
print(f"  {'player':22s} {'team':14s} {'P(top scorer)':>13s} {'exp. goals':>11s}")
for (t, nm), w in sorted(win_boot.items(), key=lambda x: -x[1])[:15]:
    print(f"  {nm:22s} {names.get(t, t):14s} {w/N*100:12.1f}% {exp_goals[(t, nm)]:10.2f}")
