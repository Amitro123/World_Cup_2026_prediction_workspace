"""
שכבת הנתונים — data model + store for the World Cup 2026 workspace.

Loads the CSV files into pandas frames, wires them to the probability engine,
and exposes the high-level operations the dashboard, the Excel mirror and the
external Hermes agent need.

Team strength is FIFA ranking points (teams.csv `fifa_points`); per-match goals
are blended with expert scorelines (expert_scores.csv). The Hermes agent feeds
pre-game news (injuries, line moves) via news_adjustments.csv — see
`add_news_adjustment` / `match_briefing`.
"""

from __future__ import annotations

import datetime as _dt
import os
import uuid
from dataclasses import dataclass, field

import pandas as pd

from . import engine

DATA_FILES = {
    "groups": "groups.csv",
    "teams": "teams.csv",
    "matches": "matches.csv",
    "odds": "odds.csv",
    "my_predictions": "my_predictions.csv",
    "expert": "expert_scores.csv",
    "news": "news_adjustments.csv",
}

NEWS_COLUMNS = [
    "adj_id", "match_id", "team_id", "kind", "value",
    "note_he", "source", "created_at", "active",
]

# A news adjustment moves the favoured pick's probability by at least this much
# (percentage points) before it is surfaced as a recommendation.
RECOMMEND_THRESHOLD = 0.05


@dataclass
class DataStore:
    data_dir: str
    groups: pd.DataFrame
    teams: pd.DataFrame
    matches: pd.DataFrame
    odds: pd.DataFrame
    predictions: pd.DataFrame
    expert: pd.DataFrame
    news: pd.DataFrame
    model: engine.ProbabilityModel = field(default_factory=engine.ProbabilityModel)

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

        news = _read("news", required=False)
        if news.empty:
            news = pd.DataFrame(columns=NEWS_COLUMNS)

        return cls(
            data_dir=data_dir,
            groups=_read("groups"),
            teams=_read("teams"),
            matches=_read("matches"),
            odds=_read("odds", required=False),
            predictions=_read("my_predictions", required=False),
            expert=_read("expert", required=False),
            news=news,
            model=model or engine.ProbabilityModel(),
        )

    def save_matches(self) -> None:
        self.matches.to_csv(os.path.join(self.data_dir, DATA_FILES["matches"]), index=False)

    def save_predictions(self) -> None:
        self.predictions.to_csv(
            os.path.join(self.data_dir, DATA_FILES["my_predictions"]), index=False
        )

    def save_news(self) -> None:
        self.news.to_csv(os.path.join(self.data_dir, DATA_FILES["news"]), index=False)

    # --- lookups ------------------------------------------------------------
    def team_name(self, team_id: str, lang: str = "he") -> str:
        row = self.teams.loc[self.teams.team_id == team_id]
        if row.empty:
            return team_id
        return row.iloc[0]["name_he" if lang == "he" else "name_en"]

    def team_rating(self, team_id: str) -> float:
        """FIFA ranking points (the engine's strength input)."""
        row = self.teams.loc[self.teams.team_id == team_id]
        return float(row.iloc[0]["fifa_points"]) if not row.empty else engine.FIFA_MEAN

    def match(self, match_id: str) -> pd.Series:
        return self.matches.loc[self.matches.match_id == match_id].iloc[0]

    def prediction(self, match_id: str) -> pd.Series | None:
        if self.predictions.empty:
            return None
        rows = self.predictions.loc[self.predictions.match_id == match_id]
        return None if rows.empty else rows.iloc[0]

    def expert_for(self, match_id: str) -> tuple[float, float] | None:
        if self.expert.empty:
            return None
        rows = self.expert.loc[self.expert.match_id == match_id]
        if rows.empty:
            return None
        r = rows.iloc[0]
        return float(r["expert_home"]), float(r["expert_away"])

    # --- core operation -----------------------------------------------------
    def update_match_state(
        self,
        match_id: str,
        minute: int,
        home_goals: int,
        away_goals: int,
        rating_override: dict | None = None,
        use_news: bool = True,
    ) -> dict:
        """Persist a live state for a match and return all derived probabilities.

        rating_override (optional): {"home": <fifa>, "away": <fifa>} to override
        the stored FIFA ratings for this single computation.
        use_news: also apply active news adjustments for this match.
        """
        m = self.match(match_id)
        if rating_override:
            r_home = float(rating_override["home"])
            r_away = float(rating_override["away"])
            mult_h = mult_a = 1.0
        else:
            r_home, r_away, mult_h, mult_a, _ = self._adjusted_inputs(
                match_id, apply_news=use_news
            )

        finished = minute >= 90
        expert = self.expert_for(match_id)
        probs = self.model.in_play(r_home, r_away, minute, home_goals, away_goals, expert=expert)
        if mult_h != 1.0 or mult_a != 1.0:
            remaining = probs.get("remaining_fraction", 0.0)
            lam_h = probs["lambda_home"] * remaining * mult_h
            lam_a = probs["lambda_away"] * remaining * mult_a
            regrid = engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=False)
            probs.update({k: regrid[k] for k in ("p_home", "p_draw", "p_away")})

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
                {"p_home": probs["p_home"], "p_draw": probs["p_draw"], "p_away": probs["p_away"]},
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
        m = self.match(match_id)
        minute = int(m.minute) if pd.notna(m.minute) else 0
        hg = int(m.home_goals) if pd.notna(m.home_goals) else 0
        ag = int(m.away_goals) if pd.notna(m.away_goals) else 0
        return self.update_match_state(match_id, minute, hg, ag)

    def pre_match_probs(self, match_id: str, apply_news: bool = False) -> dict:
        m = self.match(match_id)
        r_home, r_away, mult_h, mult_a, _ = self._adjusted_inputs(match_id, apply_news=apply_news)
        expert = self.expert_for(match_id)
        lam_h, lam_a = engine.expected_goals(r_home, r_away, expert=expert)
        lam_h *= mult_h
        lam_a *= mult_a
        return engine.probs_from_lambdas(lam_h, lam_a, dixon_coles=True)

    # --- Hermes news interface ---------------------------------------------
    def active_adjustments(self, match_id: str) -> pd.DataFrame:
        if self.news.empty:
            return self.news
        n = self.news
        return n[(n.match_id == match_id) & (n.active.astype(str).isin(["1", "True", "true"]))]

    def _adjusted_inputs(self, match_id: str, apply_news: bool = True):
        """Return (rating_home, rating_away, lambda_mult_home, lambda_mult_away, notes)
        after applying active news adjustments."""
        m = self.match(match_id)
        r_home = self.team_rating(m.home_id)
        r_away = self.team_rating(m.away_id)
        mult_h = mult_a = 1.0
        notes: list[dict] = []
        if not apply_news:
            return r_home, r_away, mult_h, mult_a, notes
        for a in self.active_adjustments(match_id).itertuples():
            kind = str(a.kind)
            val = float(a.value) if pd.notna(a.value) and a.value != "" else 0.0
            side = "home" if a.team_id == m.home_id else ("away" if a.team_id == m.away_id else None)
            if kind == "rating_delta" and side == "home":
                r_home += val
            elif kind == "rating_delta" and side == "away":
                r_away += val
            elif kind == "lambda_mult" and side == "home":
                mult_h *= val
            elif kind == "lambda_mult" and side == "away":
                mult_a *= val
            notes.append(
                {
                    "team_id": a.team_id,
                    "kind": kind,
                    "value": val,
                    "note_he": str(a.note_he),
                    "source": str(a.source),
                }
            )
        return r_home, r_away, mult_h, mult_a, notes

    def add_news_adjustment(
        self,
        match_id: str,
        team_id: str,
        kind: str,
        value: float,
        note_he: str,
        source: str = "",
    ) -> str:
        """Append a pre-game news adjustment (called by Hermes) and persist it.

        kind: 'rating_delta' (FIFA-point delta on team_id),
              'lambda_mult'  (multiplier on that team's expected goals),
              'info'         (note only, no numeric effect).
        Returns the new adjustment id.
        """
        adj_id = uuid.uuid4().hex[:8]
        row = {
            "adj_id": adj_id,
            "match_id": match_id,
            "team_id": team_id,
            "kind": kind,
            "value": value,
            "note_he": note_he,
            "source": source,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "active": 1,
        }
        new_row = pd.DataFrame([row], columns=NEWS_COLUMNS)
        self.news = new_row if self.news.empty else pd.concat(
            [self.news, new_row], ignore_index=True
        )
        self.save_news()
        return adj_id

    def deactivate_adjustment(self, adj_id: str) -> bool:
        if self.news.empty:
            return False
        mask = self.news.adj_id == adj_id
        if not mask.any():
            return False
        self.news.loc[mask, "active"] = 0
        self.save_news()
        return True

    def match_briefing(self, match_id: str) -> dict:
        """Base vs news-adjusted pre-match probabilities + a Hebrew recommendation.

        This is the payload Hermes/the dashboard consume before kickoff.
        """
        m = self.match(match_id)
        base = self.pre_match_probs(match_id, apply_news=False)
        adj = self.pre_match_probs(match_id, apply_news=True)
        _, _, _, _, notes = self._adjusted_inputs(match_id, apply_news=True)

        labels = {"H": "ניצחון ביתית", "D": "תיקו", "A": "ניצחון אורחת"}
        key = {"H": "p_home", "D": "p_draw", "A": "p_away"}
        base_pick = max(("H", "D", "A"), key=lambda k: base[key[k]])
        adj_pick = max(("H", "D", "A"), key=lambda k: adj[key[k]])

        # Track the shift on MY prediction's pick if I have one, else the favourite.
        pred = self.prediction(match_id)
        focus = str(pred["pick"]) if pred is not None else base_pick
        delta = adj[key[focus]] - base[key[focus]]

        rec = ""
        if notes:
            mine = pred is not None
            who = "הניחוש שלך" if mine else "הפייבוריט"
            if adj_pick != base_pick:
                rec = (
                    f"⚠️ שינוי המלצה: לפי החדשות הפייבוריט עובר מ-{labels[base_pick]} "
                    f"ל-{labels[adj_pick]}. שקול לעדכן את הניחוש."
                )
            elif abs(delta) >= RECOMMEND_THRESHOLD:
                direction = "ירד" if delta < 0 else "עלה"
                rec = (
                    f"שים לב: הסיכוי של {who} ({labels[focus]}) {direction} ב-"
                    f"{abs(delta)*100:.0f} נק' אחוז בעקבות החדשות."
                )
            else:
                rec = "החדשות נלקחו בחשבון; השפעה זניחה על ההמלצה."

        return {
            "match_id": match_id,
            "home_id": m.home_id,
            "away_id": m.away_id,
            "base": {k: round(base[v], 4) for k, v in key.items()},
            "adjusted": {k: round(adj[v], 4) for k, v in key.items()},
            "base_pick": base_pick,
            "adjusted_pick": adj_pick,
            "focus_pick": focus,
            "shift": round(delta, 4),
            "notes": notes,
            "recommendation": rec,
        }

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
            exp_points += p
            if str(m.status) == "finished":
                actual = engine.outcome_from_score(int(m.home_goals), int(m.away_goals))
                counts["CORRECT" if actual == str(pred["pick"]) else "WRONG"] += 1
            else:
                counts[engine.prediction_status(p)] += 1
        return {"expected_points": round(exp_points, 2), "counts": counts, "n": len(self.predictions)}
