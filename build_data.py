"""
בניית קבצי הנתונים הנגזרים — generate derived data files.

Run once after editing teams.csv / matches.csv:
    python build_data.py

It:
1. Fills teams.csv `power_rating` from the group-winner moneylines.
2. Generates odds.csv (pre-match win/draw/loss per match).
3. Seeds my_predictions.csv from the research-doc scorelines (doc_pred_*),
   only for matches you have not already predicted.
"""

import os

import pandas as pd

from src import engine

DATA = os.path.join(os.path.dirname(__file__), "data")


def build_power_ratings(teams: pd.DataFrame) -> pd.DataFrame:
    teams = teams.copy()
    teams["power_rating"] = teams["group_winner_odds"].apply(engine.odds_to_power_rating)
    return teams


def build_odds(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    model = engine.ProbabilityModel()
    rating = dict(zip(teams.team_id, teams.power_rating))
    rows = []
    for _, m in matches.iterrows():
        probs = model.pre_match(rating[m.home_id], rating[m.away_id])
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

    teams = build_power_ratings(teams)
    teams.to_csv(os.path.join(DATA, "teams.csv"), index=False)
    print(f"teams.csv: power ratings for {len(teams)} teams")

    odds = build_odds(matches, teams)
    odds.to_csv(os.path.join(DATA, "odds.csv"), index=False)
    print(f"odds.csv: pre-match 1X2 for {len(odds)} matches")

    pred_path = os.path.join(DATA, "my_predictions.csv")
    existing = pd.read_csv(pred_path) if os.path.exists(pred_path) else pd.DataFrame()
    preds = seed_predictions(matches, existing)
    preds.to_csv(pred_path, index=False)
    print(f"my_predictions.csv: {len(preds)} predictions seeded from research docs")


if __name__ == "__main__":
    main()
