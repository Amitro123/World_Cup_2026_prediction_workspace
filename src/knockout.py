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
3. Bracket: a fixed position-based R32 template (BRACKET_R32, editable below),
   with a light de-confliction pass so a third-placed team does not meet a side
   from its own group in the Round of 32.
4. Knockout games are played at a neutral venue; draws resolve via the
   strength-weighted tiebreak in engine.knockout_winner (ET / penalties proxy).

NOTE: BRACKET_R32 is a representative, self-consistent bracket, not FIFA's exact
official slot table. Edit it to match the official bracket if you want exact
paths; the simulation logic is unchanged.
"""

from __future__ import annotations

import random

import pandas as pd

from . import engine

# Each R32 tie is (slot_a, slot_b). A slot is:
#   ("W", group) group winner | ("R", group) runner-up | ("3", idx) idx-th best third
BRACKET_R32 = [
    (("W", "A"), ("3", 0)),
    (("R", "C"), ("R", "E")),
    (("W", "F"), ("3", 1)),
    (("W", "C"), ("3", 2)),
    (("W", "I"), ("3", 3)),
    (("R", "A"), ("R", "B")),
    (("W", "E"), ("3", 4)),
    (("W", "B"), ("R", "F")),
    (("W", "K"), ("3", 5)),
    (("R", "I"), ("R", "J")),
    (("W", "H"), ("3", 6)),
    (("W", "J"), ("R", "H")),
    (("W", "L"), ("3", 7)),
    (("R", "D"), ("R", "G")),
    (("W", "D"), ("R", "K")),
    (("W", "G"), ("R", "L")),
]

STAGE_NAMES = ["r16", "qf", "sf", "final", "title"]


def _partner_group_per_third() -> dict[int, str]:
    """For each third-slot index, the group of the team it faces in R32."""
    out = {}
    for a, b in BRACKET_R32:
        if a[0] == "3" and b[0] in ("W", "R"):
            out[a[1]] = b[1]
        elif b[0] == "3" and a[0] in ("W", "R"):
            out[b[1]] = a[1]
        elif a[0] == "3" and b[0] == "3":
            out.setdefault(a[1], None)
            out.setdefault(b[1], None)
    return out


PARTNER_GROUP = _partner_group_per_third()


def simulate_group(ds, group_id, ratings, rng):
    """Return (ranked_team_ids, record_dict) for one group."""
    teams = list(ds.teams.loc[ds.teams.group_id == group_id, "team_id"])
    rec = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    fixtures = ds.matches.loc[ds.matches.group_id == group_id]
    for _, m in fixtures.iterrows():
        h, a = m.home_id, m.away_id
        if str(m.status) == "finished" and pd.notna(m.home_goals):
            hg, ag = int(m.home_goals), int(m.away_goals)
        else:
            hg, ag = engine.sample_score(
                ratings[h], ratings[a], rng, expert=ds.expert_for(m.match_id)
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


def _deconflict_thirds(third_ids, third_group):
    for i in range(len(third_ids)):
        partner = PARTNER_GROUP.get(i)
        if partner and third_group[third_ids[i]] == partner:
            for j in range(len(third_ids)):
                if i == j:
                    continue
                pj = PARTNER_GROUP.get(j)
                if (third_group[third_ids[j]] != partner
                        and third_group[third_ids[i]] != pj):
                    third_ids[i], third_ids[j] = third_ids[j], third_ids[i]
                    break
    return third_ids


def simulate_once(ds, ratings, rng, counts):
    pos, group_thirds = {}, []
    for g in ds.groups.group_id:
        ranked, rec = simulate_group(ds, g, ratings, rng)
        pos[(g, 1)], pos[(g, 2)], pos[(g, 3)], pos[(g, 4)] = ranked
        t3 = ranked[2]
        group_thirds.append((g, t3, rec[t3]))

    thirds_sorted = sorted(
        group_thirds,
        key=lambda x: (x[2]["pts"], x[2]["gf"] - x[2]["ga"], x[2]["gf"], rng.random()),
        reverse=True,
    )
    third_ids = [t for (_, t, _) in thirds_sorted[:8]]
    third_group = {t: g for (g, t, _) in thirds_sorted[:8]}
    third_ids = _deconflict_thirds(third_ids, third_group)

    # tally qualifiers
    for g in ds.groups.group_id:
        counts[pos[(g, 1)]]["knockout"] += 1
        counts[pos[(g, 2)]]["knockout"] += 1
    for t in third_ids:
        counts[t]["knockout"] += 1

    def resolve(slot):
        kind, key = slot
        if kind == "W":
            return pos[(key, 1)]
        if kind == "R":
            return pos[(key, 2)]
        return third_ids[key]

    cur = [(resolve(a), resolve(b)) for a, b in BRACKET_R32]
    for stage in STAGE_NAMES:
        winners = []
        for h, a in cur:
            w = h if engine.knockout_winner(ratings[h], ratings[a], rng) == 0 else a
            counts[w][stage] += 1
            winners.append(w)
        if len(winners) <= 1:
            break
        cur = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]


def _play_detail(rh, ra, rng, neutral=True):
    """Play one knockout tie, return (winner_idx, home_goals, away_goals, note)."""
    hg, ag = engine.sample_score(rh, ra, rng, neutral=neutral)
    if hg != ag:
        return (0 if hg > ag else 1, hg, ag, "")
    probs = engine.ProbabilityModel().pre_match(rh, ra, neutral=neutral)
    ph, pa = probs["p_home"], probs["p_away"]
    wi = 0 if rng.random() < ph / (ph + pa) else 1
    return (wi, hg, ag, " (פנדלים)")


def simulate_detail(ds, seed: int | None = None) -> dict:
    """Simulate the tournament ONCE and return the full bracket with scores."""
    rng = random.Random(seed)
    ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))

    pos, group_thirds, standings = {}, [], {}
    for g in ds.groups.group_id:
        ranked, rec = simulate_group(ds, g, ratings, rng)
        pos[(g, 1)], pos[(g, 2)], pos[(g, 3)], pos[(g, 4)] = ranked
        standings[g] = [(t, rec[t]) for t in ranked]
        group_thirds.append((g, ranked[2], rec[ranked[2]]))

    thirds_sorted = sorted(
        group_thirds,
        key=lambda x: (x[2]["pts"], x[2]["gf"] - x[2]["ga"], x[2]["gf"], rng.random()),
        reverse=True,
    )
    third_ids = [t for (_, t, _) in thirds_sorted[:8]]
    third_group = {t: g for (g, t, _) in thirds_sorted[:8]}
    third_ids = _deconflict_thirds(third_ids, third_group)

    def resolve(slot):
        kind, key = slot
        if kind == "W":
            return pos[(key, 1)]
        if kind == "R":
            return pos[(key, 2)]
        return third_ids[key]

    cur = [(resolve(a), resolve(b)) for a, b in BRACKET_R32]
    round_labels = ["1/16", "1/8", "רבע גמר", "חצי גמר", "גמר"]
    rounds, champion = [], None
    for label in round_labels:
        ties, winners = [], []
        for h, a in cur:
            wi, hg, ag, note = _play_detail(ratings[h], ratings[a], rng)
            w = h if wi == 0 else a
            ties.append(
                {
                    "home": ds.team_name(h, "he"),
                    "away": ds.team_name(a, "he"),
                    "score": f"{hg}-{ag}{note}",
                    "winner": ds.team_name(w, "he"),
                }
            )
            winners.append(w)
        rounds.append({"label": label, "ties": ties})
        if len(winners) <= 1:
            champion = ds.team_name(winners[0], "he")
            break
        cur = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

    qualifiers = {
        g: {"1": ds.team_name(pos[(g, 1)], "he"), "2": ds.team_name(pos[(g, 2)], "he")}
        for g in ds.groups.group_id
    }
    best_thirds = [ds.team_name(t, "he") for t in third_ids]
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
    counts = {
        t: {"knockout": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "title": 0}
        for t in ds.teams.team_id
    }
    for _ in range(n):
        simulate_once(ds, ratings, rng, counts)

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
