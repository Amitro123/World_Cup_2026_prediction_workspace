"""
שכבת הנתונים — data model + store for the World Cup 2026 workspace.

Loads the CSV files into pandas frames, wires them to the probability engine,
and exposes the high-level operations the dashboard and Excel mirror need.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from . import engine

DATA_FILES = {
    "groups": "groups.csv",
    "teams": "teams.csv",
    "matches": "matches.csv",
    "odds": "odds.csv",
    "my_predictions": "my_predictions.csv",
}


@dataclass
class DataStore:
    data_dir: str
    groups: pd.DataFrame
    teams: pd.DataFrame
    matches: pd.DataFrame
    odds: pd.DataFrame
    predictions: pd.DataFrame
    model: engine.ProbabilityModel

    # --- loading / saving ---------------------------------------------------
    @classmethod
    def load(cls, data_dir: str, model: engine.ProbabilityModel | None = None) -> "DataStore":
        def _read(name, required=True):
            path = os.path.join(data_dir, DATA_FILES[name])
            if os.path.exists(path):
                return pd.read_csv(path)
            if required:
                raise FileNotFoundError(path)
            return pd.DataFrame()

        return cls(
            data_dir=data_dir,
            groups=_read("groups"),
            teams=_read("teams"),
            matches=_read("matches"),
            odds=_read("odds", required=False),
            predictions=_read("my_predictions", required=False),
            model=model or engine.ProbabilityModel(),
        )

    def save_matches(self) -> None:
        self.matches.to_csv(os.path.join(self.data_dir, DATA_FILES["matches"]), index=False)

    def save_predictions(self) -> None:
        self.predictions.to_csv(
            os.path.join(self.data_dir, DATA_FILES["my_predictions"]), index=False
        )

    # --- lookups ------------------------------------------------------------
    def team_name(self, team_id: str, lang: str = "he") -> str:
        row = self.teams.loc[self.teams.team_id == team_id]
        if row.empty:
            return team_id
        return row.iloc[0]["name_he" if lang == "he" else "name_en"]

    def team_rating(self, team_id: str) -> float:
        row = self.teams.loc[self.teams.team_id == team_id]
        return float(row.iloc[0]["power_rating"]) if not row.empty else 50.0

    def match(self, match_id: str) -> pd.Series:
        return self.matches.loc[self.matches.match_id == match_id].iloc[0]

    def prediction(self, match_id: str) -> pd.Series | None:
        if self.predictions.empty:
            return None
        rows = self.predictions.loc[self.predictions.match_id == match_id]
        return None if rows.empty else rows.iloc[0]

    # --- core operation -----------------------------------------------------
    def update_match_state(
        self,
        match_id: str,
        minute: int,
        home_goals: int,
        away_goals: int,
        latest_odds: dict | None = None,
    ) -> dict:
        """Persist a live state for a match and return all derived probabilities.

        latest_odds (optional): {"home": <american>, "away": <american>} to
        override the stored power ratings for this single computation.
        """
        m = self.match(match_id)
        if latest_odds:
            r_home = engine.odds_to_power_rating(latest_odds["home"])
            r_away = engine.odds_to_power_rating(latest_odds["away"])
        else:
            r_home = self.team_rating(m.home_id)
            r_away = self.team_rating(m.away_id)

        finished = minute >= 90
        probs = self.model.in_play(r_home, r_away, minute, home_goals, away_goals)

        # persist live state back to the matches frame
        idx = self.matches.index[self.matches.match_id == match_id][0]
        self.matches.at[idx, "minute"] = minute
        self.matches.at[idx, "home_goals"] = home_goals
        self.matches.at[idx, "away_goals"] = away_goals
        self.matches.at[idx, "status"] = "finished" if finished else "live"

        result = {
            "match_id": match_id,
            "home_id": m.home_id,
            "away_id": m.away_id,
            "minute": minute,
            "score": {"home": home_goals, "away": away_goals},
            "status": "finished" if finished else "live",
            "probabilities": {
                "home": round(probs["p_home"], 4),
                "draw": round(probs["p_draw"], 4),
                "away": round(probs["p_away"], 4),
            },
            "lambda": {
                "home": round(probs["lambda_home"], 3),
                "away": round(probs["lambda_away"], 3),
            },
        }

        pred = self.prediction(match_id)
        if pred is not None:
            pick = str(pred["pick"])
            p_pick = engine.pick_probability(
                {
                    "p_home": probs["p_home"],
                    "p_draw": probs["p_draw"],
                    "p_away": probs["p_away"],
                },
                pick,
            )
            result["my_prediction"] = {
                "pick": pick,
                "pred_score": {"home": int(pred["pred_home"]), "away": int(pred["pred_away"])},
                "prob_correct": round(p_pick, 4),
                "status": engine.prediction_status(p_pick),
                "stake": float(pred.get("stake", 1) or 1),
            }
        return result

    def recompute_match(self, match_id: str) -> dict:
        """Recompute using whatever live state is currently stored on the match."""
        m = self.match(match_id)
        minute = int(m.minute) if pd.notna(m.minute) else 0
        hg = int(m.home_goals) if pd.notna(m.home_goals) else 0
        ag = int(m.away_goals) if pd.notna(m.away_goals) else 0
        return self.update_match_state(match_id, minute, hg, ag)

    def pre_match_probs(self, match_id: str) -> dict:
        m = self.match(match_id)
        return self.model.pre_match(self.team_rating(m.home_id), self.team_rating(m.away_id))

    # --- aggregate views ----------------------------------------------------
    def my_summary(self) -> dict:
        """Expected points and live status counts across all predictions."""
        if self.predictions.empty:
            return {"expected_points": 0.0, "counts": {}, "n": 0}
        counts = {"ON_TRACK": 0, "AT_RISK": 0, "ALMOST_DEAD": 0, "CORRECT": 0, "WRONG": 0}
        exp_points = 0.0
        for _, pred in self.predictions.iterrows():
            mid = pred["match_id"]
            m = self.match(mid)
            state = self.recompute_match(mid)
            p = state.get("my_prediction", {}).get("prob_correct", 0.0)
            exp_points += p  # 1 point per correct outcome, expectation = prob
            if str(m.status) == "finished":
                actual = engine.outcome_from_score(
                    int(m.home_goals), int(m.away_goals)
                )
                counts["CORRECT" if actual == str(pred["pick"]) else "WRONG"] += 1
            else:
                counts[engine.prediction_status(p)] += 1
        return {"expected_points": round(exp_points, 2), "counts": counts, "n": len(self.predictions)}
