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

from . import engine, oddslib, playerprops

DATA_FILES = {
    "groups": "groups.csv",
    "teams": "teams.csv",
    "matches": "matches.csv",
    "model_probs": "model_probs.csv",  # the model's OWN 1X2 probs (NOT market odds)
    "my_predictions": "my_predictions.csv",
    "expert": "expert_scores.csv",
    "news": "news_adjustments.csv",
    "players": "players.csv",
    "h2h": "h2h.csv",
    "form": "form.csv",
    # bookmaker anchors (both optional; absent files leave the anchors dormant)
    "market_odds": "market_odds.csv",        # 1X2 closing odds per match
    "players_market": "players_market.csv",  # scorer/assist props per match
}

# Flat per-selection vig assumed when de-vigging single scorer/assist markets
# (they are not mutually exclusive, so a proportional de-vig is impossible).
PLAYER_MARKET_MARGIN = 0.06

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
    model_probs: pd.DataFrame
    predictions: pd.DataFrame
    expert: pd.DataFrame
    news: pd.DataFrame
    players: pd.DataFrame
    h2h: pd.DataFrame
    form: pd.DataFrame
    market_odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    players_market: pd.DataFrame = field(default_factory=pd.DataFrame)
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
            model_probs=_read("model_probs", required=False),
            predictions=_read("my_predictions", required=False),
            expert=_read("expert", required=False),
            news=news,
            players=_read("players", required=False),
            h2h=_read("h2h", required=False),
            form=_read("form", required=False),
            market_odds=_read("market_odds", required=False),
            players_market=_read("players_market", required=False),
            model=model or engine.ProbabilityModel(),
        )

    def validate(self) -> list[str]:
        """Sanity-check the loaded data and return a list of human-readable issues.

        Catches the silent failure modes a typo introduces: a team_id in
        matches/h2h/form/predictions that does not exist in teams.csv, or a
        missing required column. Returns [] when everything lines up. Cheap to
        run on load; surfaced in the dashboard sidebar so a bad edit is obvious
        instead of quietly skewing predictions.
        """
        issues: list[str] = []

        required = {
            "teams": ["team_id", "name_he", "fifa_points"],
            "matches": ["match_id", "home_id", "away_id"],
        }
        for name, cols in required.items():
            df = getattr(self, "predictions" if name == "my_predictions" else name)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                issues.append(f"{name}.csv חסר עמודות: {', '.join(missing)}")

        if "team_id" not in self.teams.columns:
            return issues  # cannot cross-check ids without the master list
        known = set(self.teams["team_id"].astype(str))

        def _check_ids(df, cols, label):
            if df is None or df.empty:
                return
            for col in cols:
                if col not in df.columns:
                    continue
                bad = sorted(set(df[col].astype(str)) - known - {"nan", ""})
                if bad:
                    issues.append(f"{label}: team_id לא מוכר ({', '.join(bad[:8])})")

        _check_ids(self.matches, ["home_id", "away_id"], "matches.csv")
        _check_ids(self.h2h, ["team_a", "team_b"], "h2h.csv")
        _check_ids(self.form, ["team_id"], "form.csv")
        _check_ids(self.players, ["team_id"], "players.csv")

        # cross-file match_id references
        if "match_id" in self.matches.columns:
            match_ids = set(self.matches["match_id"].astype(str))
            for df, label in [(self.predictions, "my_predictions.csv"),
                              (self.expert, "expert_scores.csv")]:
                if df is not None and not df.empty and "match_id" in df.columns:
                    bad = sorted(set(df["match_id"].astype(str)) - match_ids - {"nan", ""})
                    if bad:
                        issues.append(f"{label}: match_id לא מוכר ({', '.join(bad[:8])})")

        # duplicate team ids
        if self.teams["team_id"].duplicated().any():
            dups = sorted(self.teams.loc[self.teams["team_id"].duplicated(), "team_id"].astype(str))
            issues.append(f"teams.csv: team_id כפול ({', '.join(dups[:8])})")

        return issues

    def coverage(self) -> dict[str, dict]:
        """Report how many of the 48 teams have each optional signal populated.

        The model treats missing form/H2H/player data as a neutral zero (no
        nudge), so sparse data never *corrupts* a prediction — it just leaves a
        signal dormant. This surfaces exactly where real data would add signal,
        without ever fabricating numbers. Returns, per signal, the set size and a
        sorted list of team_ids that are still missing it.

        Keys: 'form', 'h2h', 'players'. Each value is
        {'have': int, 'total': int, 'missing': list[str]}.
        """
        all_teams = set(self.teams["team_id"].astype(str)) if "team_id" in self.teams.columns else set()
        total = len(all_teams)

        def _have(df, cols) -> set[str]:
            if df is None or df.empty:
                return set()
            present: set[str] = set()
            for col in cols:
                if col in df.columns:
                    present |= set(df[col].astype(str))
            return present & all_teams

        out: dict[str, dict] = {}
        for key, (df, cols) in {
            "form": (self.form, ["team_id"]),
            "h2h": (self.h2h, ["team_a", "team_b"]),
            "players": (self.players, ["team_id"]),
        }.items():
            have = _have(df, cols)
            out[key] = {
                "have": len(have),
                "total": total,
                "missing": sorted(all_teams - have),
            }
        return out

    def save_matches(self) -> None:
        self.matches.to_csv(os.path.join(self.data_dir, DATA_FILES["matches"]), index=False)

    def save_predictions(self) -> None:
        self.predictions.to_csv(
            os.path.join(self.data_dir, DATA_FILES["my_predictions"]), index=False
        )

    def save_news(self) -> None:
        self.news.to_csv(os.path.join(self.data_dir, DATA_FILES["news"]), index=False)

    def save_teams(self) -> None:
        self.teams.to_csv(os.path.join(self.data_dir, DATA_FILES["teams"]), index=False)

    def set_team_rating(self, team_id: str, fifa_points: float) -> dict:
        """Permanently set a team's FIFA points (pre-tournament refresh).

        Recomputes the 0–100 power_rating for ALL teams from the new spread and
        persists teams.csv. Unlike a Hermes news adjustment (single match, live),
        this changes the team's base strength everywhere — group sim, knockout,
        bonus questions. Returns {team, old, new, power_rating}.
        """
        mask = self.teams.team_id == team_id
        if not mask.any():
            raise KeyError(f"unknown team_id: {team_id}")
        old = float(self.teams.loc[mask, "fifa_points"].iloc[0])
        self.teams.loc[mask, "fifa_points"] = float(fifa_points)
        if "power_rating" in self.teams.columns:
            fp = self.teams["fifa_points"].astype(float)
            lo, hi = fp.min(), fp.max()
            span = hi - lo if hi > lo else 1.0
            self.teams["power_rating"] = ((fp - lo) / span * 100.0).round(2)
        self.save_teams()
        new_pr = float(self.teams.loc[mask, "power_rating"].iloc[0]) if "power_rating" in self.teams.columns else None
        return {"team": team_id, "old": old, "new": float(fifa_points), "power_rating": new_pr}

    # --- lookups ------------------------------------------------------------
    def team_name(self, team_id: str, lang: str = "he") -> str:
        row = self.teams.loc[self.teams.team_id == team_id]
        if row.empty:
            return team_id
        return row.iloc[0]["name_he" if lang == "he" else "name_en"]

    def _strength_stats(self) -> dict | None:
        """Population mean/std of FIFA and Elo across all teams, for the blend.

        Returns None unless teams.csv has an `elo_points` column, so the blend is
        purely opt-in (add the column + raise engine.ELO_WEIGHT to enable it).
        Cached on first use.
        """
        if "elo_points" not in self.teams.columns:
            return None
        cached = getattr(self, "_stats_cache", None)
        if cached is not None:
            return cached
        fifa = self.teams["fifa_points"].astype(float)
        elo = self.teams["elo_points"].astype(float)
        stats = {
            "fifa_mean": float(fifa.mean()), "fifa_std": float(fifa.std(ddof=0)),
            "elo_mean": float(elo.mean()), "elo_std": float(elo.std(ddof=0)),
        }
        object.__setattr__(self, "_stats_cache", stats)
        return stats

    def team_rating(self, team_id: str) -> float:
        """The engine's strength input — FIFA points, optionally blended with Elo.

        Pure FIFA by default. If teams.csv has an `elo_points` column AND
        engine.ELO_WEIGHT > 0, the rating is a z-score blend of FIFA and Elo
        (see engine.blend_strength). Backtest evidence on 2022 showed only a
        marginal gain, so ELO_WEIGHT ships at 0.0; this wiring lets you enable it
        without code changes once you have verified Elo data.
        """
        row = self.teams.loc[self.teams.team_id == team_id]
        if row.empty:
            return engine.FIFA_MEAN
        fifa = float(row.iloc[0]["fifa_points"])
        if engine.ELO_WEIGHT > 0:
            stats = self._strength_stats()
            if stats is not None and "elo_points" in row.columns:
                elo = float(row.iloc[0]["elo_points"])
                return engine.blend_strength(fifa, elo, engine.ELO_WEIGHT, **stats)
        return fifa

    def is_host(self, team_id: str) -> bool:
        """True if a team is a 2026 host nation (USA / MEX / CAN).

        Host nations keep a home-crowd advantage when they play at home; every
        other group game is at a neutral venue for the visiting side. Data-driven
        via an optional `host` column in teams.csv, falling back to engine.HOSTS.
        """
        if "host" in self.teams.columns:
            row = self.teams.loc[self.teams.team_id == team_id]
            if not row.empty:
                try:
                    return bool(int(row.iloc[0]["host"]))
                except (TypeError, ValueError):
                    pass
        return str(team_id) in engine.HOSTS

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

    def h2h_meetings(self, home_id: str, away_id: str) -> list[dict]:
        """Past meetings between two teams, each oriented to `home_id`.

        Reads h2h.csv (columns: team_a, team_b, a_goals, b_goals, comp, year).
        Order in the file does not matter; goal difference is flipped so it is
        always from the home team's perspective. Returns [] if no data.
        """
        if self.h2h is None or self.h2h.empty:
            return []
        h = self.h2h
        pair = h[((h.team_a == home_id) & (h.team_b == away_id))
                 | ((h.team_a == away_id) & (h.team_b == home_id))]
        out = []
        for r in pair.itertuples():
            ag = int(r.a_goals)
            bg = int(r.b_goals)
            gd = (ag - bg) if r.team_a == home_id else (bg - ag)
            year = int(r.year) if "year" in pair.columns and pd.notna(r.year) else None
            comp = str(r.comp) if "comp" in pair.columns and pd.notna(r.comp) else ""
            out.append({"gd": gd, "comp": comp, "year": year})
        return out

    def h2h_supremacy_for(self, home_id: str, away_id: str, ref_year: int = 2026) -> float:
        """Bounded supremacy (goals) from past meetings, home perspective."""
        return engine.h2h_supremacy(self.h2h_meetings(home_id, away_id), ref_year=ref_year)

    def recent_form(self, team_id: str) -> list[dict]:
        """A team's recent matches (momentum input), each oriented to the team.

        Reads form.csv (columns: team_id, gf, ga, comp, date). gf/ga are goals
        for/against from this team's perspective. Returns [] if no data.
        """
        if self.form is None or self.form.empty:
            return []
        f = self.form
        rows = f[f.team_id == team_id]
        out = []
        for r in rows.itertuples():
            comp = str(r.comp) if "comp" in rows.columns and pd.notna(r.comp) else ""
            date = str(r.date) if "date" in rows.columns and pd.notna(r.date) else None
            out.append({"gf": int(r.gf), "ga": int(r.ga), "comp": comp, "date": date})
        return out

    def team_form(self, team_id: str, ref_date=None) -> float:
        """A team's momentum scalar from its recent matches (0.0 if none)."""
        return engine.form_score(self.recent_form(team_id), ref_date=ref_date)

    def form_supremacy_for(self, home_id: str, away_id: str, ref_date=None) -> float:
        """Bounded supremacy (goals) from the momentum gap, home perspective."""
        return engine.form_supremacy(
            self.team_form(home_id, ref_date), self.team_form(away_id, ref_date)
        )

    # --- bookmaker anchors (optional) --------------------------------------
    def market_for(self, match_id: str) -> dict | None:
        """De-vigged 1X2 market probabilities for a match, or None.

        Reads market_odds.csv (match_id + either dec_home/dec_draw/dec_away or
        pre-computed p_home/p_draw/p_away). Returns None when the file is absent
        or the row is unusable, so the anchor stays dormant until you add odds.
        """
        if self.market_odds is None or self.market_odds.empty:
            return None
        if "match_id" not in self.market_odds.columns:
            return None
        rows = self.market_odds.loc[self.market_odds.match_id == match_id]
        if rows.empty:
            return None
        return oddslib.market_from_row(rows.iloc[0])

    def market_anchor(self, match_id: str, apply_news: bool = False) -> dict | None:
        """Model vs market 1X2 diagnostic for one match (None if no market row).

        Compares the engine's pre-match probabilities to the de-vigged closing
        odds: KL divergence, the largest per-outcome gap, whether they agree on
        the favourite, and a `flag` when they disagree enough to warrant a look.
        """
        market = self.market_for(match_id)
        if market is None:
            return None
        mp = self.pre_match_probs(match_id, apply_news=apply_news)
        model = {k: mp[k] for k in oddslib.OUTCOMES}
        diag = oddslib.compare(model, market)
        return {
            "match_id": match_id,
            "model": {k: round(model[k], 4) for k in oddslib.OUTCOMES},
            "market": {k: round(market[k], 4) for k in oddslib.OUTCOMES},
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in diag.items()},
        }

    def market_anchors(self, apply_news: bool = False) -> list[dict]:
        """market_anchor for every match that has a market row, flagged first."""
        out = []
        for mid in self.matches["match_id"].astype(str):
            a = self.market_anchor(mid, apply_news=apply_news)
            if a is not None:
                out.append(a)
        out.sort(key=lambda a: (not a["flag"], -abs(a["max_gap"])))
        return out

    def player_props(self, match_id: str, apply_news: bool = False) -> list[dict]:
        """Per-player score/assist props for a match (model, + market if present).

        Uses the match's expected goals (engine output) and each squad player's
        goal_share / assist_share to compute P(score), P(assist) and
        P(score or assist). When players_market.csv carries decimal odds for the
        same (match_id, player), they are de-vigged and compared. Returns [] if
        there is no player data for the two teams. Sorted by model P(score|assist).
        """
        if self.players is None or self.players.empty:
            return []
        m = self.match(match_id)
        mp = self.pre_match_probs(match_id, apply_news=apply_news)
        lam = {m.home_id: mp.get("lambda_home", 0.0),
               m.away_id: mp.get("lambda_away", 0.0)}
        squad = self.players[self.players.team_id.isin([m.home_id, m.away_id])]
        market_rows = None
        if (self.players_market is not None and not self.players_market.empty
                and "match_id" in self.players_market.columns):
            market_rows = self.players_market.loc[
                self.players_market.match_id == match_id]

        out = []
        for p in squad.itertuples():
            tid = p.team_id
            model = playerprops.player_match_props(
                float(p.goal_share), float(p.assist_share), lam.get(tid, 0.0))
            entry = {
                "match_id": match_id,
                "team_id": tid,
                "name_he": getattr(p, "name_he", ""),
                "name_en": getattr(p, "name_en", ""),
                "model": {k: round(model[k], 4) for k in
                          ("p_score", "p_assist", "p_score_or_assist")},
                "exp_goals": round(model["exp_goals"], 3),
                "exp_assists": round(model["exp_assists"], 3),
            }
            if market_rows is not None and not market_rows.empty:
                name_en = getattr(p, "name_en", "")
                mr = market_rows.loc[
                    market_rows.get("name_en", "").astype(str) == str(name_en)] \
                    if "name_en" in market_rows.columns else market_rows.iloc[0:0]
                if not mr.empty:
                    market = playerprops.market_props_from_row(
                        mr.iloc[0], margin=PLAYER_MARKET_MARGIN)
                    entry["market"] = {k: (round(v, 4) if v is not None else None)
                                       for k, v in market.items()}
                    entry["compare"] = playerprops.compare_props(model, market)
            out.append(entry)
        out.sort(key=lambda e: -e["model"]["p_score_or_assist"])
        return out

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
        h2h_sup = 0.0 if rating_override else self.h2h_supremacy_for(m.home_id, m.away_id)
        form_sup = 0.0 if rating_override else self.form_supremacy_for(m.home_id, m.away_id)
        neutral = not self.is_host(m.home_id)  # crowd edge only when a host is home
        probs = self.model.in_play(
            r_home, r_away, minute, home_goals, away_goals,
            neutral=neutral, expert=expert, h2h_sup=h2h_sup, form_sup=form_sup,
        )
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
        h2h_sup = self.h2h_supremacy_for(m.home_id, m.away_id)
        form_sup = self.form_supremacy_for(m.home_id, m.away_id)
        neutral = not self.is_host(m.home_id)  # crowd edge only when a host is home
        lam_h, lam_a = engine.expected_goals(
            r_home, r_away, neutral=neutral, expert=expert, h2h_sup=h2h_sup, form_sup=form_sup
        )
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
            # Expected goals (λ) — used for dynamic score prediction
            "base_lambda_home": round(base.get("lambda_home", 0), 2),
            "base_lambda_away": round(base.get("lambda_away", 0), 2),
            "adj_lambda_home": round(adj.get("lambda_home", 0), 2),
            "adj_lambda_away": round(adj.get("lambda_away", 0), 2),
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
