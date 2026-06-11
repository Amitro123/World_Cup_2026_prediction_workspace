"""Best-thirds prediction: P(each group's 3rd qualifies among the best 8).

Reuses knockout._group_phase (FIFA third-ranking: pts -> GD -> GF), injury-adjusted
in-memory like predict_groups.py. Reports per group letter:
  - P(this group's third qualifies)            [letter-level pick]
  - P(the user's named team is the qualifying third of this group)
"""
import sys
import random

sys.stdout.reconfigure(encoding="utf-8")

from src.models import DataStore
from src import knockout

NEWS = {
    'NED': -80, 'BRA': -100, 'MAR': -55, 'JPN': -45, 'GER': -20,
    'SCO': -15, 'CAN': -25, 'ESP': -20, 'ARG': -25, 'ENG': -10, 'USA': -15,
}

# The user's stage-01 third-place picks (shown in the Best Thirds screen)
USER_THIRDS = {
    'A': 'CZE', 'B': 'BIH', 'C': 'SCO', 'D': 'AUS', 'E': 'CIV', 'F': 'SWE',
    'G': 'EGY', 'H': 'KSA', 'I': 'NOR', 'J': 'ALG', 'K': 'COD', 'L': 'PAN',
}

N = 20000
ds = DataStore.load("data")
names = dict(zip(ds.teams.team_id, ds.teams.name_en))

ctx = knockout._prepare(ds)
ctx["ratings"] = {t: r + NEWS.get(t, 0) for t, r in ctx["ratings"].items()}

rng = random.Random(2026)
letter_qual = {g: 0 for g in ctx["groups"]}
team_third_qual = {g: 0 for g in ctx["groups"]}   # named team is 3rd AND qualifies
team_is_third = {g: 0 for g in ctx["groups"]}     # named team is 3rd

for _ in range(N):
    pos, third_assign, _standings = knockout._group_phase(ctx, rng)
    qual_groups = set(third_assign.values())
    for g in ctx["groups"]:
        third = pos[(g, 3)]
        if g in qual_groups:
            letter_qual[g] += 1
        if third == USER_THIRDS[g]:
            team_is_third[g] += 1
            if g in qual_groups:
                team_third_qual[g] += 1

print(f"{'grp':3s} {'your 3rd pick':16s} {'P(grp 3rd qual.)':>17s} "
      f"{'P(pick is 3rd)':>15s} {'P(pick 3rd & qual.)':>20s}")
rows = []
for g in ctx["groups"]:
    rows.append((g, letter_qual[g] / N, team_is_third[g] / N, team_third_qual[g] / N))
for g, lq, t3, tq in sorted(rows, key=lambda r: -r[1]):
    print(f"{g:3s} {names.get(USER_THIRDS[g], USER_THIRDS[g]):16s} {lq*100:16.1f}% "
          f"{t3*100:14.1f}% {tq*100:19.1f}%")
