"""
שאלות בונוס — derive answers to the tournament bonus questions from the model.

All answers are recomputed from the same engine + knockout simulation, so they
stay consistent with real results as Hermes/your live agent writes them back into
matches.csv (finished games are locked; the rest is sampled).

Team-level questions are full model output. Two questions are *player-level*
(top assists; Mbappé vs Vinícius goal count) — the model is team-level, so those
are returned as informed proxies and flagged with `player_level=True`.
"""

from __future__ import annotations

import random
from collections import defaultdict

from . import engine, knockout

# Player -> team, for the head-to-head questions.
PLAYER_TEAM = {
    "מסי": "ARG",
    "רונאלדו": "POR",
    "אמבפה": "FRA",
    "ויניסיוס": "BRA",
}

# Tournament opener (known schedule): Mexico vs South Africa at Estadio Azteca.
OPENER_TEAMS = ("MEX", "RSA")


def _group_goal_sim(ds, n: int, rng) -> tuple[dict, dict, dict]:
    """Monte-Carlo the 72 group games -> expected goals for/against per team.

    Fixtures (status, expert, h2h/form supremacy) are resolved to plain tuples
    ONCE; the n-loop then touches no pandas. Iterating `DataFrame.iterrows()` and
    re-querying `expert_for` per game inside the loop was the dominant cost.
    """
    ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))
    h2h = knockout.build_h2h(ds)
    form = knockout.build_form(ds)
    # Pre-resolve every fixture: (home, away, finished, hg, ag, expert, h2h_sup, form_sup)
    fixtures: list[tuple] = []
    for _, m in ds.matches.iterrows():
        h, a = m.home_id, m.away_id
        finished = str(m.status) == "finished" and _notna(m.home_goals)
        hg = int(m.home_goals) if finished else 0
        ag = int(m.away_goals) if finished else 0
        fixtures.append((h, a, finished, hg, ag, ds.expert_for(m.match_id),
                         h2h.get((h, a), 0.0), knockout._form_sup(form, h, a)))

    gf = defaultdict(float)
    ga = defaultdict(float)
    for _ in range(n):
        for h, a, finished, hg, ag, expert, hsup, fsup in fixtures:
            if not finished:
                hg, ag = engine.sample_score(
                    ratings[h], ratings[a], rng,
                    expert=expert, h2h_sup=hsup, form_sup=fsup,
                )
            gf[h] += hg; ga[h] += ag
            gf[a] += ag; ga[a] += hg
    gf = {t: v / n for t, v in gf.items()}
    ga = {t: v / n for t, v in ga.items()}
    return gf, ga, ratings


def _notna(x) -> bool:
    return x == x and x is not None  # NaN != NaN


def _find_opener(ds):
    pair = frozenset(OPENER_TEAMS)
    for _, m in ds.matches.iterrows():
        if frozenset((m.home_id, m.away_id)) == pair:
            return m
    return ds.matches.iloc[0]


def _expected_ko_games(row) -> float:
    """Expected knockout games a team plays = sum of P(reach each KO round)."""
    return (
        row["qualify_%"] + row["r16_%"] + row["qf_%"] + row["sf_%"] + row["final_%"]
    ) / 100.0


def compute(ds, n_ko: int = 4000, n_group: int = 3000, seed: int = 2026) -> dict:
    """Return a structured dict of bonus-question answers."""
    name = lambda t: ds.team_name(t, "he")
    df = knockout.run(ds, n=n_ko, seed=seed).set_index("team_id")
    rng = random.Random(seed)
    gf, ga, ratings = _group_goal_sim(ds, n_group, rng)

    # --- 1) runner-up: reached the final but did not win it ---
    runnerup = (df["final_%"] - df["title_%"]).sort_values(ascending=False)
    champion = df["title_%"].idxmax()
    # most likely runner-up that is NOT your champion pick
    runnerup_pick = next(t for t in runnerup.index if t != champion)
    top_finalists = [
        {"team": name(t), "final_%": round(df.loc[t, "final_%"], 1),
         "title_%": round(df.loc[t, "title_%"], 1),
         "runnerup_%": round(runnerup[t], 1)}
        for t in runnerup.head(6).index
    ]

    # --- 3) first goal of the tournament (opener favourite by expected goals) ---
    opener = _find_opener(ds)
    lam_h, lam_a = engine.expected_goals(
        ratings[opener.home_id], ratings[opener.away_id],
        expert=ds.expert_for(opener.match_id),
        h2h_sup=ds.h2h_supremacy_for(opener.home_id, opener.away_id),
        form_sup=ds.form_supremacy_for(opener.home_id, opener.away_id),
    )
    opener_fav = opener.home_id if lam_h >= lam_a else opener.away_id

    # --- 4) punching bag / 5) most group-stage goals ---
    most_conceded = sorted(ga.items(), key=lambda x: -x[1])[:5]
    most_scored = sorted(gf.items(), key=lambda x: -x[1])[:5]

    # --- 6) Messi vs Ronaldo: who goes further ---
    def depth(tid):
        r = df.loc[tid]
        return (r["qualify_%"] + r["r16_%"] + r["qf_%"] + r["sf_%"]
                + r["final_%"] + r["title_%"])
    arg, por = PLAYER_TEAM["מסי"], PLAYER_TEAM["רונאלדו"]
    further = "מסי" if depth(arg) >= depth(por) else "רונאלדו"

    # --- 7) Mbappé vs Vinícius: who scores more ---
    fra, bra = PLAYER_TEAM["אמבפה"], PLAYER_TEAM["ויניסיוס"]
    # expected total goals = team scoring rate/game * expected total games
    def exp_team_goals(tid):
        rate = gf[tid] / 3.0  # per-game scoring rate from group stage
        games = 3.0 + _expected_ko_games(df.loc[tid])
        return rate * games

    # --- player model: expected goals/assists per player ---
    # player goals  = goal_share  * team expected goals over the run
    # player assists = assist_share * team expected goals over the run
    prows = []
    if ds.players is not None and not ds.players.empty:
        for _, p in ds.players.iterrows():
            tid = p["team_id"]
            if tid not in df.index:
                continue
            teamg = exp_team_goals(tid)
            prows.append({
                "name": p["name_he"],
                "team": name(tid),
                "team_id": tid,
                "exp_goals": float(p["goal_share"]) * teamg,
                "exp_assists": float(p["assist_share"]) * teamg,
            })

    def _find_player(name_key, team_id):
        for r in prows:
            if r["name"] == name_key and r["team_id"] == team_id:
                return r
        return None

    mb, vi = _find_player("אמבפה", fra), _find_player("ויניסיוס", bra)
    if mb and vi:
        more_goals = "אמבפה" if mb["exp_goals"] >= vi["exp_goals"] else "ויניסיוס"
        mbv_note = (f"ממודל השחקנים: אמבפה ~{mb['exp_goals']:.1f} שערים צפויים, "
                    f"ויניסיוס ~{vi['exp_goals']:.1f}.")
    else:
        fra_g, bra_g = exp_team_goals(fra), exp_team_goals(bra)
        more_goals = "אמבפה" if fra_g >= bra_g else "ויניסיוס"
        mbv_note = (f"פרוקסי לפי עומק בטורניר: צרפת ~{fra_g:.1f} שערים צפויים, "
                    f"ברזיל ~{bra_g:.1f}.")

    def stage_row(tid):
        r = df.loc[tid]
        return {
            "team": name(tid),
            "qf_%": round(r["qf_%"], 1), "sf_%": round(r["sf_%"], 1),
            "final_%": round(r["final_%"], 1), "title_%": round(r["title_%"], 1),
        }

    if prows:
        assist_sorted = sorted(prows, key=lambda r: -r["exp_assists"])
        ak = assist_sorted[0]
        top_assists = {
            "answer": f"{ak['name']} ({ak['team']})",
            "player_level": True,
            "note": "ממודל השחקנים: בישולים צפויים = נתח בישול × תוחלת שערי הנבחרת לאורך הטורניר.",
            "table": [{"player": f"{r['name']} ({r['team']})",
                       "exp_assists": round(r["exp_assists"], 2)}
                      for r in assist_sorted[:6]],
        }
    else:
        top_assists = {
            "answer": "למין ימאל (ספרד)",
            "player_level": True,
            "note": "שאלת שחקן — אין players.csv; בורר מרכזי בנבחרת שצולחת עמוק.",
        }

    return {
        "runner_up": {
            "answer": name(runnerup_pick),
            "note": f"בהנחה ש{name(champion)} אלופה; הסגנית הסבירה הבאה.",
            "table": top_finalists,
        },
        "top_assists": top_assists,
        "first_goal": {
            "answer": name(opener_fav),
            "note": f"משחק פתיחה: {name(opener.home_id)} נגד {name(opener.away_id)}; "
                    f"תוחלת שערים {lam_h:.2f}-{lam_a:.2f}.",
        },
        "punching_bag": {
            "answer": name(most_conceded[0][0]),
            "table": [{"team": name(t), "goals_against": round(v, 2)} for t, v in most_conceded],
        },
        "most_group_goals": {
            "answer": name(most_scored[0][0]),
            "table": [{"team": name(t), "goals_for": round(v, 2)} for t, v in most_scored],
        },
        "messi_vs_ronaldo": {
            "answer": further,
            "table": [stage_row(arg), stage_row(por)],
        },
        "mbappe_vs_vinicius": {
            "answer": more_goals,
            "player_level": True,
            "note": mbv_note,
        },
    }
