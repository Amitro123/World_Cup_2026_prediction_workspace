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
4. Knockout games are played at a neutral venue. A tie after 90' goes to 30' of
   (lower-scoring) extra time where the stronger side keeps its full edge, and
   only a still-level ET reaches a near-coin-flip penalty shootout — see
   engine.resolve_knockout.

The R16/QF/SF tree is wired by explicit match dependencies (TREE), so the
left/right halves of the draw match the real bracket rather than a naive
consecutive fold.
"""

from __future__ import annotations

import math
import random

import pandas as pd

from . import engine

# Partial home advantage for host nations in the knockout phase.
# Full group-stage edge (HOME_SUP = 0.35 goals) is reduced by half in
# high-stakes elimination games where both sides arrive at peak preparation.
KNOCKOUT_HOST_ADV = engine.HOME_SUP * 0.5

# --- Monte-Carlo defaults (CR4 P1-8) -----------------------------------------
# N_SIMS is the pinned default sample size; DEFAULT_SEED makes any headline
# number quoted from the dashboard/README reproducible. Sample-size sensitivity:
# every run() row carries a `title_ci` column (±pp, 95%) so the noise is shown,
# not hidden. At n=2000 a 12% title favourite reads ±1.4pp and long-tail teams
# (<0.5%) read ±0.3pp — fine for the interactive UI. For publication-grade
# long-tail odds (48-team title table) use n>=100_000, where the favourite
# tightens to ±0.2pp; run() is O(n) and ~100k sims take a few minutes.
N_SIMS = 2000
DEFAULT_SEED = 2026

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

# --- Stable match_ids for knockout ties (news_adjustments interface) --------
# Every simulated knockout tie gets a stable, predictable match_id — "M" + the
# official FIFA match number already used as the R32/TREE dict keys above — so
# a news_adjustments.csv row can target a specific knockout tie the same way it
# targets a group-stage match_id. R32 ties (M73-M88) are known once the group
# stage resolves; R16-onward ties are only known once earlier rounds are
# simulated, so a rating_delta attached to e.g. "M97" applies to whichever team
# ends up playing that match number in each simulated run — it is not pinned to
# a specific pair of teams the way a group match_id is.
ALL_MATCH_NOS = sorted(set(R32) | set(TREE))


def match_id_for(match_no: int) -> str:
    return f"M{match_no}"


def build_knockout_news(ds) -> dict[int, dict[str, float]]:
    """Active knockout rating_delta adjustments: match_no -> {team_id: delta}.

    Precomputed once per run so the Monte-Carlo loop stays cheap (same pattern
    as build_h2h/build_form). Absent unless `ds.add_news_adjustment` was called
    with one of the match_ids from `match_id_for`.
    """
    out: dict[int, dict[str, float]] = {}
    if getattr(ds, "news", None) is None or ds.news.empty:
        return out
    for m in ALL_MATCH_NOS:
        deltas = ds.knockout_rating_deltas(match_id_for(m))
        if deltas:
            out[m] = deltas
    return out


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


def build_form(ds) -> dict[str, float]:
    """Per-team momentum scalar lookup: team_id -> form score.

    Precomputed once per run so the Monte-Carlo loop stays cheap. A pairing's
    form supremacy is engine.form_supremacy(form[home], form[away]); teams with
    no recent record are simply absent (treated as 0).
    """
    out: dict[str, float] = {}
    if getattr(ds, "form", None) is None or ds.form.empty:
        return out
    for t in ds.teams.team_id:
        s = ds.team_form(t)
        if s:
            out[t] = s
    return out


def _form_sup(form, home, away) -> float:
    """Form supremacy for a pairing from a precomputed per-team form lookup."""
    if not form:
        return 0.0
    return engine.form_supremacy(form.get(home, 0.0), form.get(away, 0.0))


# Ascending feeder order, computed once (every feeder resolves before its match).
TREE_ORDER = sorted(TREE)


def _prepare(ds) -> dict:
    """Pre-compute everything the Monte-Carlo loop needs as plain Python.

    The hot loop must never touch pandas: a single `DataFrame.loc` or `iterrows`
    per game, multiplied by 72 games × tens of thousands of sims, dominated the
    runtime (≈28 ms/sim). We resolve all of it ONCE here — ratings, per-group
    fixture tuples (with the finished/expert/neutral flags baked in), group
    rosters, and the h2h/form lookups — so `simulate_once` runs on dicts and
    tuples only. Returns a context dict consumed by the `*_fast` helpers.
    """
    ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))
    groups = list(ds.groups.group_id)
    group_teams = {
        g: list(ds.teams.loc[ds.teams.group_id == g, "team_id"]) for g in groups
    }
    group_fixtures: dict[str, list[tuple]] = {g: [] for g in groups}
    for g in groups:
        for _, m in ds.matches.loc[ds.matches.group_id == g].iterrows():
            h, a = m.home_id, m.away_id
            finished = str(m.status) == "finished" and pd.notna(m.home_goals)
            hg = int(m.home_goals) if finished else 0
            ag = int(m.away_goals) if finished else 0
            # A group game carries a home-crowd edge only when home is a host.
            neutral = not ds.is_host(h)
            group_fixtures[g].append(
                (h, a, finished, hg, ag, ds.expert_for(m.match_id), neutral)
            )
    return {
        "ratings": ratings,
        "groups": groups,
        "group_teams": group_teams,
        "group_fixtures": group_fixtures,
        "h2h": build_h2h(ds),
        "form": build_form(ds),
        "knockout_news": build_knockout_news(ds),
    }


def _ko_rating(ctx, match_no, team_id) -> float:
    """A team's rating for knockout match `match_no`, plus any active rating_delta."""
    delta = ctx["knockout_news"].get(match_no, {}).get(team_id, 0.0)
    return ctx["ratings"][team_id] + delta


def simulate_group(ds, group_id, ratings, rng, h2h=None, form=None):
    """Compatibility shim: build a context on the fly and delegate to the fast
    path. Prefer `_simulate_group_fast` inside the Monte-Carlo loop."""
    ctx = {
        "ratings": ratings,
        "group_teams": {group_id: list(
            ds.teams.loc[ds.teams.group_id == group_id, "team_id"])},
        "group_fixtures": {group_id: [
            (m.home_id, m.away_id,
             str(m.status) == "finished" and pd.notna(m.home_goals),
             int(m.home_goals) if (str(m.status) == "finished"
                                   and pd.notna(m.home_goals)) else 0,
             int(m.away_goals) if (str(m.status) == "finished"
                                   and pd.notna(m.home_goals)) else 0,
             ds.expert_for(m.match_id), not ds.is_host(m.home_id))
            for _, m in ds.matches.loc[ds.matches.group_id == group_id].iterrows()]},
        "h2h": h2h or {},
        "form": form or {},
    }
    return _simulate_group_fast(ctx, group_id, rng)


def _simulate_group_fast(ctx, group_id, rng):
    """Return (ranked_team_ids, record_dict) for one group — no pandas."""
    teams = ctx["group_teams"][group_id]
    ratings, h2h, form = ctx["ratings"], ctx["h2h"], ctx["form"]
    rec = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}
    for h, a, finished, hg, ag, expert, neutral in ctx["group_fixtures"][group_id]:
        if not finished:
            hg, ag = engine.sample_score(
                ratings[h], ratings[a], rng, neutral=neutral, expert=expert,
                h2h_sup=h2h.get((h, a), 0.0), form_sup=_form_sup(form, h, a),
            )
        rh, ra = rec[h], rec[a]
        rh["gf"] += hg; rh["ga"] += ag
        ra["gf"] += ag; ra["ga"] += hg
        if hg > ag:
            rh["pts"] += 3
        elif ag > hg:
            ra["pts"] += 3
        else:
            rh["pts"] += 1; ra["pts"] += 1
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


def _group_phase(ctx, rng):
    """Run all 12 groups; return (pos, third_assign, standings)."""
    pos, group_thirds, standings = {}, [], {}
    for g in ctx["groups"]:
        ranked, rec = _simulate_group_fast(ctx, g, rng)
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


def simulate_once(ctx, rng, counts):
    """One full tournament; tally each team's round-reached counters in `counts`."""
    ratings, h2h, form = ctx["ratings"], ctx["h2h"], ctx["form"]
    pos, third_assign, _ = _group_phase(ctx, rng)

    # tally qualifiers
    for g in ctx["groups"]:
        counts[pos[(g, 1)]]["knockout"] += 1
        counts[pos[(g, 2)]]["knockout"] += 1
    for g in set(third_assign.values()):
        counts[pos[(g, 3)]]["knockout"] += 1

    r32 = _resolve_r32(pos, third_assign)
    winners = {}
    for m in range(73, 89):
        h, a = r32[m]
        host_sup = KNOCKOUT_HOST_ADV if h in engine.HOSTS else 0.0
        w = h if engine.knockout_winner(
            _ko_rating(ctx, m, h), _ko_rating(ctx, m, a), rng,
            h2h_sup=h2h.get((h, a), 0.0) + host_sup,
            form_sup=_form_sup(form, h, a),
        ) == 0 else a
        winners[m] = w
        counts[w][WIN_COUNTER[m]] += 1
    for m in TREE_ORDER:  # ascending: every feeder is resolved before its match
        fa, fb = TREE[m]
        h, a = winners[fa], winners[fb]
        host_sup = KNOCKOUT_HOST_ADV if h in engine.HOSTS else 0.0
        w = h if engine.knockout_winner(
            _ko_rating(ctx, m, h), _ko_rating(ctx, m, a), rng,
            h2h_sup=h2h.get((h, a), 0.0) + host_sup,
            form_sup=_form_sup(form, h, a),
        ) == 0 else a
        winners[m] = w
        counts[w][WIN_COUNTER[m]] += 1


def _play_detail(rh, ra, rng, neutral=True, h2h_sup=0.0, form_sup=0.0):
    """Play one knockout tie, return (winner_idx, home_goals, away_goals, note).

    Goals are the aggregate after extra time; the note marks how it was decided
    (ET = "(הארכה)", shootout = "(פנדלים)")."""
    wi, info = engine.resolve_knockout(
        rh, ra, rng, neutral=neutral, h2h_sup=h2h_sup, form_sup=form_sup
    )
    hg, ag = info["reg"]
    if info["et"] is None:
        return (wi, hg, ag, "")
    eh, ea = info["et"]
    note = " (פנדלים)" if info["pens"] else " (הארכה)"
    return (wi, hg + eh, ag + ea, note)


def simulate_detail(ds, seed: int | None = None) -> dict:
    """Simulate the tournament ONCE and return the full bracket with scores."""
    rng = random.Random(seed)
    ctx = _prepare(ds)
    ratings, h2h, form = ctx["ratings"], ctx["h2h"], ctx["form"]

    pos, third_assign, _ = _group_phase(ctx, rng)
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
            wi, hg, ag, note = _play_detail(
                _ko_rating(ctx, m, h), _ko_rating(ctx, m, a), rng,
                h2h_sup=h2h.get((h, a), 0.0), form_sup=_form_sup(form, h, a),
            )
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


def _ci95_pct(count: int, n: int) -> float:
    """95% half-width CI in percentage points for a proportion (normal approx)."""
    p = count / n
    return round(100.0 * 1.96 * math.sqrt(p * (1.0 - p) / n), 1)


def run(ds, n: int = N_SIMS, seed: int | None = DEFAULT_SEED) -> pd.DataFrame:
    """Run the Monte-Carlo and return a probability table sorted by title odds.

    Columns include ``title_%`` (point estimate) and ``title_ci`` (±pp at 95%
    confidence) so callers can display e.g. "12.4 ± 0.6" rather than implying
    false precision between close candidates.
    """
    rng = random.Random(seed)
    ctx = _prepare(ds)
    counts = {
        t: {"knockout": 0, "r16": 0, "qf": 0, "sf": 0, "final": 0, "title": 0}
        for t in ds.teams.team_id
    }
    for _ in range(n):
        simulate_once(ctx, rng, counts)

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
                "title_ci": _ci95_pct(c["title"], n),
            }
        )
    return pd.DataFrame(rows).sort_values("title_%", ascending=False).reset_index(drop=True)


# --- Bracket-region helpers (deterministic "chalk" structure) ----------------
# These derive the draw geometry straight from R32/TREE — no simulation needed.
# "Chalk" = the bracket you get if every group winner advances; it tells you
# which favorites are bunched into the same half/quarter and therefore must
# knock each other out before the final.
_PARENT = {child: parent for parent, feeders in TREE.items() for child in feeders}

# group letter -> R32 match its WINNER enters (from the ("W", group) slots).
_WINNER_R32 = {
    slot[1]: m for m, (a, b) in R32.items() for slot in (a, b) if slot[0] == "W"
}


def _path_to_final(m: int) -> list[int]:
    """Match numbers from R32 match `m` up the tree to the final (M104)."""
    path, cur = [m], m
    while cur in _PARENT:
        cur = _PARENT[cur]
        path.append(cur)
    return path


def _stage_of(m: int) -> str:
    """Hebrew round label for the round contested in match number `m`."""
    if m <= 88:
        return "1/16"
    if m <= 96:
        return "1/8"
    if m <= 100:
        return "רבע גמר"
    if m <= 102:
        return "חצי גמר"
    return "גמר"


def _quarter_half(m: int) -> tuple[int, str]:
    """(quarter 1-4, half) for the team entering at R32 match `m`."""
    qf = next(x for x in _path_to_final(m) if 97 <= x <= 100)
    half = "עליון" if qf in (97, 98) else "תחתון"
    return qf - 96, half


def _meet_stage(m1: int, m2: int) -> str:
    """Round at which the winners of R32 matches m1 and m2 first collide.

    The lowest common ancestor of the two paths in the bracket tree.
    """
    p2 = set(_path_to_final(m2))
    lca = next(x for x in _path_to_final(m1) if x in p2)
    return _stage_of(lca)


_STAGE_ORDER = {"1/16": 0, "1/8": 1, "רבע גמר": 2, "חצי גמר": 3, "גמר": 4}


def draw_difficulty(ds, n: int = 8000, seed: int | None = None) -> dict:
    """Group-by-group draw difficulty + deterministic chalk-bracket collisions.

    Returns {"groups": [...], "collisions": [...], "n": n}:
      groups     – per group: its strongest team, that team's title%, the group's
                   total title equity, average qualify%, and the half/quarter the
                   group WINNER lands in. Sorted by total title equity (the
                   "group of death" first).
      collisions – pairs among the eight strongest groups' winners that would meet
                   BEFORE the final in the chalk bracket, with the round — i.e.
                   which favorites are on a collision course and suppress each
                   other's headline title odds.
    """
    df = run(ds, n=n, seed=seed)  # already sorted by title_% descending
    groups = []
    for g, sub in df.groupby("group"):
        top = sub.iloc[0]  # strongest team in the group (global title sort)
        quarter, half = _quarter_half(_WINNER_R32[g])
        groups.append(
            {
                "group": g,
                "top_team": top["name_he"],
                "top_title": round(float(top["title_%"]), 1),
                "group_title": round(float(sub["title_%"].sum()), 1),
                "avg_qualify": round(float(sub["qualify_%"].mean()), 1),
                "half": half,
                "quarter": quarter,
            }
        )
    groups.sort(key=lambda r: r["group_title"], reverse=True)

    # Collision contenders are the eight strongest *teams* (by their own title
    # odds), not the eight strongest groups — otherwise a heavyweight in a weak
    # group (e.g. a lone favorite) is dropped and its early collisions hidden.
    top8 = sorted(groups, key=lambda r: r["top_title"], reverse=True)[:8]
    collisions = []
    for i in range(len(top8)):
        for j in range(i + 1, len(top8)):
            a, b = top8[i], top8[j]
            stage = _meet_stage(_WINNER_R32[a["group"]], _WINNER_R32[b["group"]])
            if stage != "גמר":  # meeting in the final isn't a "collision" worth flagging
                collisions.append(
                    {"team_a": a["top_team"], "team_b": b["top_team"], "stage": stage}
                )
    collisions.sort(key=lambda c: _STAGE_ORDER.get(c["stage"], 9))
    return {"groups": groups, "collisions": collisions, "n": n}
