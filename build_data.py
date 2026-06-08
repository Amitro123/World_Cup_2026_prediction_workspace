"""
בניית קבצי הנתונים הנגזרים — generate derived data files.

Run once after editing teams.csv / matches.csv:
    python build_data.py

It:
1. Fills teams.csv `fifa_points` (real strength) from the Cowork model
   (data/cowork_model.json) and a 0-100 `power_rating` rescale for display.
2. Writes expert_scores.csv (per-match expert scoreline targets) from
   data/cowork_expert.json, mapped to this workspace's match_ids.
3. Generates model_probs.csv (pre-match win/draw/loss per match) with the
   FIFA-points Dixon-Coles engine, blended with the expert scorelines. (Named
   model_probs, NOT odds — these are the model's own probabilities; bookmaker
   odds live in market_odds.csv.)
4. Seeds my_predictions.csv from the research-doc scorelines (doc_pred_*),
   only for matches you have not already predicted.

Why FIFA points (not group-winner odds): group-winner odds conflate team
strength with how easy the group is. Brazil's short group-winner odds (easy
group) inflated its rating; on neutral FIFA strength it sits ~6th, while Spain
stays top-2 — matching the research docs.
"""

import json
import os

import pandas as pd

from src import engine

DATA = os.path.join(os.path.dirname(__file__), "data")

# Cowork FIFA-key (English) -> this workspace's name_en, where they differ.
NAME_FIXES = {
    "Czech Republic": "Czechia",
    "Korea Republic": "South Korea",
    "Curaçao": "Curacao",
}


def _load_cowork():
    with open(os.path.join(DATA, "cowork_model.json"), encoding="utf-8") as f:
        model = json.load(f)
    with open(os.path.join(DATA, "cowork_expert.json"), encoding="utf-8") as f:
        expert = json.load(f)
    return model, expert


def _name_to_id(teams: pd.DataFrame) -> dict[str, str]:
    """Map a Cowork English team name to this workspace's team_id."""
    by_en = dict(zip(teams.name_en, teams.team_id))
    out = {}
    for fifa_name in by_en:
        out[fifa_name] = by_en[fifa_name]
    # add fixes (cowork spelling -> our team_id via our spelling)
    for cowork_name, our_name in NAME_FIXES.items():
        if our_name in by_en:
            out[cowork_name] = by_en[our_name]
    return out


def build_fifa_ratings(teams: pd.DataFrame, model: dict) -> pd.DataFrame:
    teams = teams.copy()
    name_id = _name_to_id(teams)
    fifa = {}
    for cowork_name, pts in model["fifa"].items():
        tid = name_id.get(cowork_name)
        if tid:
            fifa[tid] = float(pts)
    missing = [t for t in teams.team_id if t not in fifa]
    if missing:
        raise ValueError(f"No FIFA points mapped for: {missing}")
    teams["fifa_points"] = teams.team_id.map(fifa).round(1)
    lo, hi = teams.fifa_points.min(), teams.fifa_points.max()
    teams["power_rating"] = ((teams.fifa_points - lo) / (hi - lo) * 100.0).round(2)
    return teams


def build_expert_scores(matches: pd.DataFrame, teams: pd.DataFrame, model, expert) -> pd.DataFrame:
    """Map Cowork expert scorelines (keyed by match num) onto our match_ids."""
    name_id = _name_to_id(teams)
    # our (frozenset of the two team_ids) -> match row
    pair_to_match = {}
    for _, m in matches.iterrows():
        pair_to_match[frozenset((m.home_id, m.away_id))] = m
    rows = []
    for mm in model["matches"]:
        num = str(mm["num"])
        ex = expert.get(num)
        if not ex or ex.get("hs") is None:
            continue
        h_id = name_id.get(mm["home"])
        a_id = name_id.get(mm["away"])
        if not h_id or not a_id:
            continue
        match_row = pair_to_match.get(frozenset((h_id, a_id)))
        if match_row is None:
            continue
        # orient expert scoreline to OUR home/away order
        if match_row.home_id == h_id:
            hs, as_ = ex["hs"], ex["as_"]
        else:
            hs, as_ = ex["as_"], ex["hs"]
        rows.append({"match_id": match_row.match_id, "expert_home": hs, "expert_away": as_})
    return pd.DataFrame(rows)


def build_odds(matches, teams, expert_df) -> pd.DataFrame:
    model = engine.ProbabilityModel()
    rating = dict(zip(teams.team_id, teams.fifa_points))
    expert = {
        r.match_id: (float(r.expert_home), float(r.expert_away))
        for r in expert_df.itertuples()
    }
    rows = []
    for _, m in matches.iterrows():
        probs = model.pre_match(
            rating[m.home_id], rating[m.away_id], expert=expert.get(m.match_id)
        )
        rows.append(
            {
                "match_id": m.match_id,
                "p_home": round(probs["p_home"], 4),
                "p_draw": round(probs["p_draw"], 4),
                "p_away": round(probs["p_away"], 4),
            }
        )
    return pd.DataFrame(rows)


def seed_predictions(matches: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    have = set(existing.match_id) if not existing.empty else set()
    rows = list(existing.to_dict("records")) if not existing.empty else []
    for _, m in matches.iterrows():
        if m.match_id in have:
            continue
        hg, ag = int(m.doc_pred_home), int(m.doc_pred_away)
        pick = engine.outcome_from_score(hg, ag)
        rows.append(
            {
                "match_id": m.match_id,
                "pick": pick,
                "pred_home": hg,
                "pred_away": ag,
                "confidence": 3,
                "stake": 1,
            }
        )
    return pd.DataFrame(rows)


def main():
    teams = pd.read_csv(os.path.join(DATA, "teams.csv"))
    matches = pd.read_csv(os.path.join(DATA, "matches.csv"))
    model, expert = _load_cowork()

    teams = build_fifa_ratings(teams, model)
    teams.to_csv(os.path.join(DATA, "teams.csv"), index=False)
    top = teams.sort_values("fifa_points", ascending=False).head(6)
    print(f"teams.csv: FIFA points + power_rating for {len(teams)} teams")
    print("  top 6:", ", ".join(f"{r.team_id} {r.fifa_points:.0f}" for r in top.itertuples()))

    expert_df = build_expert_scores(matches, teams, model, expert)
    expert_df.to_csv(os.path.join(DATA, "expert_scores.csv"), index=False)
    print(f"expert_scores.csv: {len(expert_df)} expert scorelines mapped")

    odds = build_odds(matches, teams, expert_df)
    odds.to_csv(os.path.join(DATA, "model_probs.csv"), index=False)
    print(f"model_probs.csv: pre-match 1X2 for {len(odds)} matches")

    pred_path = os.path.join(DATA, "my_predictions.csv")
    existing = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()
    preds = seed_predictions(matches, existing)
    preds.to_csv(pred_path, index=False)
    print(f"my_predictions.csv: {len(preds)} predictions ({len(preds) - len(existing)} new)")


if __name__ == "__main__":
    main()
