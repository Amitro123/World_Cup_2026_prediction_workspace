"""Group-stage standings prediction: rank distribution per group, injury-adjusted.

Runs the exact group machinery from src/knockout.py (same tiebreakers: pts -> GD -> GF),
with the June-2026 injury deltas from sim_r32_news.py applied in-memory (the
news_adjustments pipeline is empty, so this is the only way they reach the sim).
Outputs, per group: P(rank 1..4) per team, the modal full ordering, and the
per-slot best assignment (max expected correct slots over all 24 permutations).
"""
import random
import sys
from itertools import permutations

from src import knockout
from src.models import DataStore

sys.stdout.reconfigure(encoding="utf-8")

# June 2026 injury deltas (ESPN/AP/BBC/MARCA) — same values as sim_r32_news.py
NEWS = {
    'NED': -80, 'BRA': -100, 'MAR': -55, 'JPN': -45, 'GER': -20,
    'SCO': -15, 'CAN': -25, 'ESP': -20, 'ARG': -25, 'ENG': -10, 'USA': -15,
}

N = 20000
ds = DataStore.load("data")
names = dict(zip(ds.teams.team_id, ds.teams.name_en))


def run(adjusted: bool):
    ctx = knockout._prepare(ds)
    if adjusted:
        ctx["ratings"] = {t: r + NEWS.get(t, 0) for t, r in ctx["ratings"].items()}
    out = {}
    for g in ctx["groups"]:
        teams = ctx["group_teams"][g]
        rank_counts = {t: [0, 0, 0, 0] for t in teams}
        order_counts = {}
        rng = random.Random(2026)
        for _ in range(N):
            ranked, _rec = knockout._simulate_group_fast(ctx, g, rng)
            for pos, t in enumerate(ranked):
                rank_counts[t][pos] += 1
            key = tuple(ranked)
            order_counts[key] = order_counts.get(key, 0) + 1
        out[g] = (rank_counts, order_counts)
    return out


def best_assignment(rank_counts):
    """Permutation maximizing expected number of correct slots."""
    teams = list(rank_counts)
    best, best_ev = None, -1.0
    for perm in permutations(teams):
        ev = sum(rank_counts[t][i] for i, t in enumerate(perm)) / N
        if ev > best_ev:
            best, best_ev = perm, ev
    return best, best_ev


base = run(adjusted=False)
adj = run(adjusted=True)

for g in sorted(adj):
    rank_counts, order_counts = adj[g]
    base_counts, _ = base[g]
    print(f"\n=== GROUP {g} ===")
    by_p1 = sorted(rank_counts, key=lambda t: -rank_counts[t][0])
    for t in by_p1:
        c = rank_counts[t]
        b = base_counts[t]
        d1 = (c[0] - b[0]) / N * 100
        flag = f"  [inj: P1 {d1:+.1f}pp]" if t in NEWS else ""
        print(f"  {names.get(t, t):20s} 1st {c[0]/N*100:5.1f}%  2nd {c[1]/N*100:5.1f}%  "
              f"3rd {c[2]/N*100:5.1f}%  4th {c[3]/N*100:5.1f}%{flag}")
    modal = max(order_counts, key=order_counts.get)
    bestp, ev = best_assignment(rank_counts)
    print(f"  modal order : {' > '.join(modal)}  ({order_counts[modal]/N*100:.1f}% of sims)")
    if bestp != modal:
        print(f"  best-EV slot: {' > '.join(bestp)}  (exp. correct slots {ev:.2f}/4)")
    else:
        print(f"  best-EV slot: same  (exp. correct slots {ev:.2f}/4)")
    # close calls: any pair where P(rank1) within 8pp, or 2nd/3rd within 8pp
    p2 = sorted(rank_counts, key=lambda t: -(rank_counts[t][0] + rank_counts[t][1]))
    q_gap = (rank_counts[p2[1]][0] + rank_counts[p2[1]][1]
             - rank_counts[p2[2]][0] - rank_counts[p2[2]][1]) / N * 100
    if q_gap < 10:
        print(f"  CLOSE qualify battle: {names[p2[1]]} vs {names[p2[2]]} (gap {q_gap:.1f}pp)")
