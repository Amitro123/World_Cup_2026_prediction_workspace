"""
סימולציית נוקאאוט — Monte-Carlo knockout-bracket simulation.

Simulates the whole tournament many times to estimate, for each team, the
probability of qualifying from the group and reaching each knockout round up to
winning the cup.

How it works
------------
1. Group stage: each of the 72 group games is either taken from a *finished*
   result in matches.csv (so the sim stays consistent with real results your
   live agent writes back) or sampled from the Poisson engine. Standings rank by
   points, goal difference, goals for.
2. Qualifiers: 12 group winners + 12 runners-up + the 8 best third-placed teams.
3. Bracket: the **official FIFA 2026 bracket** (matches M73–M104). The Round of
   32 slots are fixed; the eight third-place slots each carry the official
   Annex-C candidate-group list, and the 8 qualifying thirds are assigned to
   those slots by constrained bipartite matching (so a third always lands in a
   slot its group is eligible for — no same-group rematch in R32).
4. Knockout games are played at a neutral venue; draws resolve via the
   strength-weighted tiebreak in engine.knockout_winner (ET / penalties proxy).

The R16/QF/SF tree is wired by explicit match dependencies (TREE), so the
left/right halves of the draw match the real bracket rather than a naive
consecutive fold.
"""

from __future__ import annotations

import random

import pandas as pd

from . import engine

# --- Official FIFA 2026 Round of 32 (matches M73–M88) ---------------------
# A slot is one of:
#   ("W", group)  group winner
#   ("R", group)  runner-up
#   ("3", frozenset(candidate_groups))  best-third placed into this slot
R32 = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", frozenset("ABCDF"))),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", frozenset("CDFGH"))),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", frozenset("CEFHI"))),
    80: (("W", "L"), ("3", frozenset("EHIJK"))),
    81: (("W", "D"), ("3", frozenset("BEFIJ"))),
    82: (("W", "G"), ("3", frozenset("AEHIJ"))),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", frozenset("EFGIJ"))),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", frozenset("DEIJL"))),
    88: (("R", "D"), ("R", "G")),
}

# Later rounds: match_no -> (feeder_match_a, feeder_match_b). Winners fold here.
TREE = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
    101: (97, 98), 102: (99, 100),
    104: (101, 102),  # M103 is the third-place play-off, not simulated
}

# Third-place slots: match_no -> candidate groups (Annex C).
THIRD_SLOTS = {m: set(slot[1]) for m, (a, b) in R32.items() for slot in (a, b) if slot[0] == "3"}

# Which "reached the next round" counter a win in each match increments.
WIN_COUNTER = {}
for _m in range(73, 89):
    WIN_COUNTER[_m] = "r16"
for _m in range(89, 97):
    WIN_COUNTER[_m] = "qf"
for _m in range(97, 101):
    WIN_COUNTER[_m] = "sf"
for _m in (101, 102):
    WIN_COUNTER[_m] = "final"
WIN_COUNTER[104] = "title"

STAGE_NAMES = ["r16", "qf", "sf", "final", "title"]

# Display grouping for the single detailed bracket (Hebrew labels).
DISPLAY_ROUNDS = [
    ("1/16", list(range(73, 89))),
    ("1/8", list(range(89, 97))),
    ("רבע גמר", [97, 98, 99, 100]),
    ("חצי גמר", [101, 102]),
    ("גמר", [104]),
]


def build_h2h(ds) -> dict[tuple, float]:
    """Pairwise head-to-head supremacy lookup: (home_id, away_id) -> goals.

    Precomputed once per run so the Monte-Carlo loop stays cheap. Pairings with
    no recorded history are simply absent (treated as 0).
    """
    out: dict[tuple, float] = {}
    if getattr(ds, "h2h", None) is None or ds.h2h.empty:
        return out
    seen = set()
    for r in ds.h2h.itertuples():
        fs = frozenset((r.team_a, r.team_b))
        if len(fs) != 2 or fs in seen:
            continue
        seen.add(fs)
        a, b = r.team_a, r.team_b
        out[(a, b)] = ds.h2h_supremacy_for(a, b)
        out[(b, a)] = ds.h2h_supremacy_for(b, a)
    return out


def simulate_group(ds, group_id, ratings, rng, h2h=None):
    """Return (ranked_team_ids, record_dict) for one group."""
    h2h = h2h or {}
    teams = list(ds.teams.loc[ds.teams.group_id == group_id, "team_id"])
    rec = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    fixtures = ds.matches.loc[ds.matches.group_id == group_id]
    for _, m in fixtures.iterrows():
        h, a = m.home_id, m.away_id
        if str(m.status) == "finished" and pd.notna(m.home_goals):
            hg, ag = int(m.home_goals), int(m.away_goals)
        else:
            hg, ag = engine.sample_score(
                ratings[h], ratings[a], rng,
                expert=ds.expert_for(m.match_id), h2h_sup=h2h.get((h, a), 0.0),
            )
        rec[h]["gf"] += hg; rec[h]["ga"] += ag
        rec[a]["gf"] += ag; rec[a]["ga"] += hg
        if hg > ag:
            rec[h]["pts"] += 3
        elif ag > hg:
            rec[a]["pts"] += 3
        else:
            rec[h]["pts"] += 1; rec[a]["pts"] += 1
    ranked = sorted(
        teams,
        key=lambda t: (rec[t]["pts"], rec[t]["gf"] - rec[t]["ga"], rec[t]["gf"], rng.random()),
        reverse=True,
    )
    return ranked, rec


def _match_thirds(qualifying_groups: set[str]) -> dict[int, str]:
    """Assign the 8 qualifying third-place groups to the 8 R32 third-slots.

    Constrained bipartite matching: each slot may only take a group from its
    Annex-C candidate list. FIFA's table guarantees a perfect matching exists for
    any 8-of-12 combination. Returns match_no -> group_letter.
    """
    # Most-constrained slots first keeps the backtracking shallow.
    slots = sorted(THIRD_SLOTS, key=lambda m: len(THIRD_SLOTS[m] & qualifying_groups))
    assign: dict[int, str] = {}
    used: set[str] = set()

    def bt(i: int) -> bool:
        if i == len(slots):
            return True
        m = slots[i]
        for g in THIRD_SLOTS[m]:
            if g in qualifying_groups and g not in used:
                assign[m] = g
                used.add(g)
                if bt(i + 1):
                    return True
                used.discard(g)
                del assign[m]
        return False

    if not bt(0):
        # Should not happen with the official table; fill any leftovers safely.
        for m in slots:
            if m not in assign:
                for g in qualifying_groups:
                    if g not in used:
                        assign[m] = g
                        used.add(g)
                        break
    return assign


def _slot_team(slot, pos, third_assign, match_no):
    kind, key = slot
    if kind == "W":
        return pos[(key, 1)]
    if kind == "R":
        return pos[(key, 2)]
    return pos[(third_assign[match_no], 3)]  # ("3", ...)


def _resolve_r32(pos, third_assign) -> dict[int, tuple]:
    """match_no -> (team_a, team_b) for all 16 R32 ties."""
    out = {}
    for m, (sa, sb) in R32.items():
        out[m] = (_slot_team(sa, pos, third_assign, m), _slot_team(sb, pos, third_assign, m))
    return out


def _group_phase(ds, ratings, rng, h2h=None):
    """Run all 12 groups; return (pos, third_assign, standings)."""
    h2h = h2h or {}
    pos, group_thirds, standings = {}, [], {}
    for g in ds.groups.group_id:
        ranked, rec = simulate_group(ds, g, ratings, rng, h2h)
        pos[(g, 1)], pos[(g, 2)], pos[(g, 3)], pos[(g, 4)] = ranked
        standings[g] = [(t, rec[t]) for t in ranked]
        group_thirds.append((g, rec[ranked[2]]))

    thirds_sorted = sorted(
        group_thirds,
        key=lambda x: (x[1]["pts"], x[1]["gf"] - x[1]["ga"], x[1]["gf"], rng.random()),
        reverse=True,
    )
    qual_groups = {g for (g, _) in thirds_sorted[:8]}
    third_assign = _match_thirds(qual_groups)
    return pos, third_assign, standings


def simulate_once(ds, ratings, rng, counts, h2h=None):
    h2h = h2h or {}
    pos, third_assign, _ = _group_phase(ds, ratings, rng, h2h)

    # tally qualifiers
    for g in ds.groups.group_id:
        counts[pos[(g, 1)]]["knockout"] += 1
        counts[pos[(g, 2)]]["knockout"] += 1
    for g in set(third_assign.values()):
        counts[pos[(g, 3)]]["knockout"] += 1

    r32 = _resolve_r32(pos, third_assign)
    winners = {}
    for m in range(73, 89):
        h, a = r32[m]
        w = h if engine.knockout_winner(ratings[h], ratings[a], rng, h2h_sup=h2h.get((h, a), 0.0)) == 0 else a
        winners[m] = w
        counts[w][WIN_COUNTER[m]] += 1
    for m in sorted(TREE):  # ascending: every feeder is resolved before its match
        fa, fb = TREE[m]
        h, a = winners[fa], winners[fb]
        w = h if engine.knockout_winner(ratings[h], ratings[a], rng, h2h_sup=h2h.get((h, a), 0.0)) == 0 else a
        winners[m] = w
        counts[w][WIN_COUNTER[m]] += 1


def _play_detail(rh, ra, rng, neutral=True, h2h_sup=0.0):
    """Play one knockout tie, return (winner_idx, home_goals, away_goals, note)."""
    hg, ag = engine.sample_score(rh, ra, rng, neutral=neutral, h2h_sup=h2h_sup)
    if hg != ag:
        return (0 if hg > ag else 1, hg, ag, "")
    probs = engine.ProbabilityModel().pre_match(rh, ra, neutral=neutral, h2h_sup=h2h_sup)
    ph, pa = probs["p_home"], probs["p_away"]
    wi = 0 if rng.random() < ph / (ph + pa) else 1
    return (wi, hg, ag, " (פנדלים)")


def simulate_detail(ds, seed: int | None = None) -> dict:
    """Simulate the tournament ONCE and return the full bracket with scores."""
    rng = random.Random(seed)
    ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))
    h2h = build_h2h(ds)

    pos, third_assign, _ = _group_phase(ds, ratings, rng, h2h)
    r32 = _resolve_r32(pos, third_assign)

    winners, rounds = {}, []
    for label, mlist in DISPLAY_ROUNDS:
        ties = []
        for m in mlist:
            if m in r32:
                h, a = r32[m]
            else:
                fa, fb = TREE[m]
                h, a = winners[fa], winners[fb]
            wi, hg, ag, note = _play_detail(ratings[h], ratings[a], rng, h2h_sup=h2h.get((h, a), 0.0))
            w = h if wi == 0 else a
            winners[m] = w
            ties.append(
                {
                    "home": ds.team_name(h, "he"),
                    "away": ds.team_name(a, "he"),
                    "score": f"{hg}-{ag}{note}",
                    "winner": ds.team_name(w, "he"),
                }
            )
        rounds.append({"label": label, "ties": ties})

    champion = ds.team_name(winners[104], "he")
    qualifiers = {
        g: {"1": ds.team_name(pos[(g, 1)], "he"), "2": ds.team_name(pos[(g, 2)], "he")}
        for g in ds.groups.group_id
    }
    best_thirds = [ds.team_name(pos[(g, 3)], "he") for g in third_assign.values()]
    return {
        "champion": champion,
        "rounds": rounds,
        "qualifiers": qualifiers,
        "best_thirds": best_thirds,
    }


def run(ds, n: int = 2000, seed: int | None = None) -> pd.DataFrame:
    """Run the Monte-Carlo and return a probability table sorted by title odds."""
    rng = random.Random(seed)
    ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))
    h2h = build_h2h(ds)
    counts = {
        t: {"knockout": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "title": 0}
        for t in ds.teams.team_id
    }
    for _ in range(n):
        simulate_once(ds, ratings, rng, counts, h2h)

    rows = []
    for t, c in counts.items():
        rows.append(
            {
                "team_id": t,
                "name_he": ds.team_name(t, "he"),
                "group": ds.teams.loc[ds.teams.team_id == t, "group_id"].iloc[0],
                "qualify_%": round(100 * c["knockout"] / n, 1),
                "r16_%": round(100 * c["r16"] / n, 1),
                "qf_%": round(100 * c["qf"] / n, 1),
                "sf_%": round(100 * c["sf"] / n, 1),
                "final_%": round(100 * c["final"] / n, 1),
                "title_%": round(100 * c["title"] / n, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("title_%", ascending=False).reset_index(drop=True)
