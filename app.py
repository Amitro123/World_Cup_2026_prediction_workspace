"""
לוח בקרה — World Cup 2026 prediction dashboard (Hebrew / RTL).

Run:  streamlit run app.py
"""

import os
import random
import subprocess

import altair as alt
import pandas as pd
import streamlit as st

from src import backtest, bonus, engine, i18n, knockout
from src.models import DataStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(page_title="World Cup 2026 — Dashboard", layout="wide")

# Language is chosen in the sidebar but the direction CSS must be emitted up here,
# before any widgets render, so read it from session_state (set on the previous
# run by the language selector). Defaults to Hebrew on first load.
lang = st.session_state.get("lang", i18n.DEFAULT_LANG)
_dir = "rtl" if i18n.is_rtl(lang) else "ltr"
_align = "right" if i18n.is_rtl(lang) else "left"
st.markdown(
    f"""
    <style>
      .stApp, .block-container {{ direction: {_dir}; text-align: {_align}; }}
      [data-testid="stSidebar"] {{ direction: {_dir}; text-align: {_align}; }}
      table {{ direction: {_dir}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def _pooled_calibration() -> tuple[list[dict], int]:
    """Calibration bins + match count pooled over every data/backtest_*.csv.

    A larger sample (≈294 matches) fills the reliability bins far better than the
    single 64-match 2022 set, so the diagram is less jumpy.
    """
    import glob

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "backtest_*.csv")))
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return [], 0
    df = pd.concat(frames, ignore_index=True)
    return backtest.calibration_table(df), int(len(df))


def _reliability_chart(cal: pd.DataFrame):
    """A true reliability diagram: predicted (x) vs observed (y) with the y=x
    perfect-calibration reference. Points on the diagonal = well calibrated;
    above = model under-confident, below = over-confident. Point size = bin n."""
    diag = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    ideal = (
        alt.Chart(diag)
        .mark_line(strokeDash=[6, 4], color="#888")
        .encode(x=alt.X("x", title="הסתברות חזויה"),
                y=alt.Y("y", title="תדירות בפועל"))
    )
    pts = (
        alt.Chart(cal)
        .mark_circle(opacity=0.85, color="#1f77b4")
        .encode(
            x=alt.X("ממוצע חזוי", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("תדירות בפועל", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("n", title="מס' נקודות", scale=alt.Scale(range=[30, 400])),
            tooltip=["טווח חזוי", "n", "ממוצע חזוי", "תדירות בפועל", "פער"],
        )
    )
    line = (
        alt.Chart(cal)
        .mark_line(color="#1f77b4", opacity=0.5)
        .encode(x="ממוצע חזוי", y="תדירות בפועל")
    )
    return (ideal + line + pts).properties(height=360).interactive()


PICK_HE = {"H": "ניצחון ביתית", "D": "תיקו", "A": "ניצחון אורחת"}
STATUS_HE = {
    "ON_TRACK": "🟢 במסלול",
    "AT_RISK": "🟡 בסיכון",
    "ALMOST_DEAD": "🔴 כמעט אבוד",
    "CORRECT": "✅ צדק",
    "WRONG": "❌ טעה",
}


@st.cache_resource
def get_store() -> DataStore:
    return DataStore.load(DATA_DIR)


def pull_latest() -> tuple[bool, str]:
    """git pull the latest data Hermes pushed, then clear the cache.

    Returns (ok, message). The dashboard re-reads CSVs on the next run.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        return False, "git לא מותקן / לא נמצא ב-PATH"
    except subprocess.TimeoutExpired:
        return False, "git pull חרג מהזמן (60 שניות)"
    out = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        return False, out or "git pull נכשל"
    get_store.clear()  # drop cached DataStore so fresh CSVs are read
    return True, out or "עדכון הושלם"


ds = get_store()


def team(tid: str) -> str:
    return ds.team_name(tid, "he")


# --- sidebar navigation ------------------------------------------------------
st.sidebar.title(i18n.t("app_title", lang))

# Language selector. Writing the choice to session_state and rerunning lets the
# direction CSS (emitted at the top of the script) pick it up on the next pass.
_lang_choice = st.sidebar.radio(
    i18n.t("language", lang),
    list(i18n.LANGUAGES),
    index=list(i18n.LANGUAGES).index(lang),
    format_func=lambda c: i18n.LANGUAGES[c],
    horizontal=True,
)
if _lang_choice != lang:
    st.session_state["lang"] = _lang_choice
    st.rerun()

# Navigation returns a stable view KEY; only the label is translated, so the
# dispatch below never depends on the active language.
view = st.sidebar.radio(
    i18n.t("nav", lang),
    i18n.VIEW_KEYS,
    format_func=lambda k: i18n.view_label(k, lang),
)
st.sidebar.caption(i18n.t("model_tag", lang))

st.sidebar.divider()
if st.sidebar.button(i18n.t("refresh", lang), use_container_width=True,
                     help=i18n.t("refresh_help", lang)):
    with st.spinner("מושך עדכונים מ-GitHub…"):
        ok, msg = pull_latest()
    if ok:
        st.sidebar.success("הנתונים עודכנו")
        st.rerun()
    else:
        st.sidebar.error(f"רענון נכשל: {msg}")

# Data integrity indicator — flags typos (unknown team_id, bad references).
# Guarded with hasattr so a stale cached DataStore (from before validate() was
# added) degrades gracefully instead of crashing — click "🔄 רענן נתונים".
if hasattr(ds, "validate"):
    _issues = ds.validate()
    if _issues:
        st.sidebar.warning("⚠️ בעיות בנתונים:\n\n" + "\n".join(f"- {i}" for i in _issues))
    else:
        st.sidebar.caption(i18n.t("data_ok", lang))

# Data-coverage indicator — how many of the 48 teams have each optional signal.
# Missing data is a neutral zero (never corrupts a prediction), so this is an
# honest "where would real data add signal?" gauge, not an error.
if hasattr(ds, "coverage"):
    from src import datameta
    _cov = ds.coverage()
    _meta = datameta.read(ds.data_dir)
    _labels = {"form": "כושר עדכני", "h2h": "מפגשים היסטוריים", "players": "שחקנים"}
    _lines = []
    for _k in ("form", "h2h", "players"):
        _c = _cov[_k]
        _mark = "✓" if _c["have"] == _c["total"] else "•"
        _line = f"{_mark} {_labels[_k]}: {_c['have']}/{_c['total']}"
        _stamp = _meta.get(_k)
        if _stamp and _stamp.get("updated"):
            _line += f" — עודכן {_stamp['updated']} ({_stamp.get('source', '?')})"
        _lines.append(_line)
    with st.sidebar.expander("📊 כיסוי נתונים", expanded=False):
        st.caption("\n\n".join(_lines))
        st.caption(
            "נתונים חסרים נספרים כאפס (ללא השפעה) — אינם פוגמים בתחזית, "
            "רק מצביעים היכן מידע אמיתי יוסיף אות."
        )


# --- view: fixtures ----------------------------------------------------------
if view == "fixtures":
    st.header(i18n.header("fixtures", lang))
    groups = ["הכל"] + list(ds.groups.group_id)
    sel = st.selectbox("סינון לפי בית", groups)

    rows = []
    for _, m in ds.matches.iterrows():
        if sel != "הכל" and m.group_id != sel:
            continue
        pr = ds.pre_match_probs(m.match_id)
        pred = ds.prediction(m.match_id)
        pick = str(pred["pick"]) if pred is not None else ""
        rows.append(
            {
                "בית": m.group_id,
                "ביתית": team(m.home_id),
                "אורחת": team(m.away_id),
                "1 (ביתית)": f"{pr['p_home']*100:.0f}%",
                "X (תיקו)": f"{pr['p_draw']*100:.0f}%",
                "2 (אורחת)": f"{pr['p_away']*100:.0f}%",
                "התחזית שלי": PICK_HE.get(pick, ""),
                "תוצאה צפויה (מהמחקר)": f"{int(m.doc_pred_home)}-{int(m.doc_pred_away)}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --- view: live match --------------------------------------------------------
elif view == "live":
    st.header(i18n.header("live", lang))
    labels = {
        f"{m.group_id} | {team(m.home_id)} - {team(m.away_id)}": m.match_id
        for _, m in ds.matches.iterrows()
    }
    chosen = st.selectbox("בחר משחק", list(labels.keys()))
    mid = labels[chosen]
    m = ds.match(mid)

    c1, c2, c3 = st.columns(3)
    minute = c1.number_input("דקה", 0, 120, int(m.minute) if pd.notna(m.minute) else 0)
    hg = c2.number_input(f"שערי {team(m.home_id)}", 0, 20,
                         int(m.home_goals) if pd.notna(m.home_goals) else 0)
    ag = c3.number_input(f"שערי {team(m.away_id)}", 0, 20,
                         int(m.away_goals) if pd.notna(m.away_goals) else 0)

    r1, r2 = st.columns(2)
    _red_h0 = int(m.red_home) if ("red_home" in m.index and pd.notna(m.red_home)) else 0
    _red_a0 = int(m.red_away) if ("red_away" in m.index and pd.notna(m.red_away)) else 0
    red_h = r1.number_input(f"🟥 כרטיסים אדומים {team(m.home_id)}", 0, 5, _red_h0,
                            help="נבחרת בחסר שחקן יוצרת פחות ומקבלת יותר — המודל "
                                 "מתאים את קצב השערים לזמן שנותר.")
    red_a = r2.number_input(f"🟥 כרטיסים אדומים {team(m.away_id)}", 0, 5, _red_a0)

    state = ds.update_match_state(mid, int(minute), int(hg), int(ag),
                                  red_home=int(red_h), red_away=int(red_a))
    p = state["probabilities"]

    if red_h != red_a:
        down = team(m.home_id) if red_h > red_a else team(m.away_id)
        up = team(m.away_id) if red_h > red_a else team(m.home_id)
        st.caption(f"⚠️ נחיתות מספרית: {down} בחסר שחקן — קצב הסיכויים הוטה לטובת {up} "
                   f"לזמן שנותר.")

    st.subheader(f"{team(m.home_id)} {hg} - {ag} {team(m.away_id)}  ·  דקה {minute}")
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric(f"ניצחון {team(m.home_id)}", f"{p['home']*100:.1f}%")
    pc2.metric("תיקו", f"{p['draw']*100:.1f}%")
    pc3.metric(f"ניצחון {team(m.away_id)}", f"{p['away']*100:.1f}%")
    st.bar_chart(pd.DataFrame({"הסתברות": p}).T)

    mp = state.get("my_prediction")
    if mp:
        st.divider()
        st.markdown(
            f"**התחזית שלי:** {PICK_HE[mp['pick']]} "
            f"(ניחוש תוצאה {mp['pred_score']['home']}-{mp['pred_score']['away']})  \n"
            f"**סיכוי שהתחזית תתממש:** {mp['prob_correct']*100:.1f}%  \n"
            f"**סטטוס:** {STATUS_HE[mp['status']]}"
        )

    if st.button("💾 שמור מצב משחק לקובץ"):
        ds.save_matches()
        st.success("נשמר ל-matches.csv")


# --- view: Hermes news adjustments ------------------------------------------
elif view == "hermes":
    st.header(i18n.header("hermes", lang))
    st.caption(
        "סוכן Hermes (רץ ב-Telegram) סורק אתרי הימורים, חדשות ספורט ופיפ\"א "
        "ומזין כאן עדכונים לפני המשחק (פציעות, שינויי הרכב, תזוזות בקו ההימורים). "
        "המודל מחשב מחדש את ההסתברויות ומפיק המלצה אם הניחוש שלך מושפע."
    )

    labels = {
        f"{m.group_id} | {team(m.home_id)} - {team(m.away_id)}": m.match_id
        for _, m in ds.matches.iterrows()
    }
    chosen = st.selectbox("בחר משחק", list(labels.keys()))
    mid = labels[chosen]
    m = ds.match(mid)

    b = ds.match_briefing(mid)
    st.subheader(f"{team(m.home_id)} מול {team(m.away_id)}")

    cols = st.columns(3)
    keys = [("H", f"ניצחון {team(m.home_id)}"), ("D", "תיקו"), ("A", f"ניצחון {team(m.away_id)}")]
    for col, (k, lab) in zip(cols, keys):
        base_p = b["base"][k] * 100
        adj_p = b["adjusted"][k] * 100
        col.metric(lab, f"{adj_p:.1f}%", f"{adj_p - base_p:+.1f}% מהבסיס")

    if b["recommendation"]:
        if "⚠️" in b["recommendation"]:
            st.warning(b["recommendation"])
        else:
            st.info(b["recommendation"])

    if b["notes"]:
        st.markdown("**עדכונים פעילים למשחק זה:**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "נבחרת": team(n["team_id"]),
                        "סוג": n["kind"],
                        "ערך": n["value"],
                        "הערה": n["note_he"],
                        "מקור": n["source"],
                    }
                    for n in b["notes"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("אין עדכוני חדשות פעילים למשחק זה.")

    st.divider()
    with st.expander("➕ הוספת עדכון ידני (בדיקה / שימוש ללא Hermes)"):
        cc = st.columns(2)
        side = cc[0].selectbox(
            "נבחרת מושפעת", [m.home_id, m.away_id],
            format_func=lambda t: team(t),
        )
        kind = cc[1].selectbox(
            "סוג עדכון",
            ["rating_delta", "lambda_mult", "info"],
            format_func={
                "rating_delta": "שינוי דירוג (נק' פיפ\"א)",
                "lambda_mult": "מכפיל תוחלת שערים",
                "info": "מידע בלבד",
            }.get,
        )
        default_val = -60.0 if kind == "rating_delta" else (1.0 if kind == "lambda_mult" else 0.0)
        value = st.number_input("ערך", value=default_val, step=1.0 if kind == "rating_delta" else 0.05)
        note = st.text_input("הערה (עברית)", "")
        source = st.text_input("מקור", "ידני")
        if st.button("שמור עדכון"):
            ds.add_news_adjustment(mid, side, kind, float(value), note, source)
            st.success("נשמר ל-news_adjustments.csv — רענן את הדף לראות את ההשפעה.")

    active = ds.active_adjustments(mid)
    if not active.empty:
        st.divider()
        st.markdown("**ניהול עדכונים פעילים:**")
        for a in active.itertuples():
            c = st.columns([5, 1])
            c[0].write(f"`{a.adj_id}` · {team(a.team_id)} · {a.kind}={a.value} · {a.note_he}")
            if c[1].button("בטל", key=f"del_{a.adj_id}"):
                ds.deactivate_adjustment(a.adj_id)
                st.rerun()


# --- view: tournament overview ----------------------------------------------
elif view == "overview":
    st.header(i18n.header("overview", lang))
    s = ds.my_summary()
    m1, m2, m3 = st.columns(3)
    m1.metric("מספר תחזיות", s["n"])
    m2.metric("נקודות צפויות (תוחלת)", s["expected_points"])
    decided = s["counts"].get("CORRECT", 0) + s["counts"].get("WRONG", 0)
    m3.metric("משחקים שהוכרעו", decided)

    st.subheader("פילוח סטטוס התחזיות")
    counts = {STATUS_HE.get(k, k): v for k, v in s["counts"].items() if v}
    if counts:
        st.bar_chart(pd.DataFrame({"כמות": counts}))

    st.subheader("תרחישי המשך (סימולציית מונטה קרלו)")
    n_sim = st.slider("מספר סימולציות", 200, 5000, 1000, step=200)
    if st.button("הרץ סימולציה"):
        scores = []
        # pre-compute per-match outcome probabilities + my pick
        precomp = []
        for _, pred in ds.predictions.iterrows():
            mid = pred["match_id"]
            mt = ds.match(mid)
            if str(mt.status) == "finished":
                actual = engine.outcome_from_score(int(mt.home_goals), int(mt.away_goals))
                precomp.append(("decided", 1 if actual == str(pred["pick"]) else 0))
            else:
                pr = ds.pre_match_probs(mid)
                precomp.append(("open", (pr["p_home"], pr["p_draw"], pr["p_away"], str(pred["pick"]))))
        pick_idx = {"H": 0, "D": 1, "A": 2}
        for _ in range(n_sim):
            pts = 0
            for kind, payload in precomp:
                if kind == "decided":
                    pts += payload
                else:
                    ph, pdr, pa, pick = payload
                    r = random.random()
                    drawn = "H" if r < ph else ("D" if r < ph + pdr else "A")
                    if drawn == pick:
                        pts += 1
            scores.append(pts)
        ser = pd.Series(scores)
        st.write(
            f"ממוצע: **{ser.mean():.1f}** · חציון: **{ser.median():.0f}** · "
            f"טווח 90%: **{ser.quantile(0.05):.0f}–{ser.quantile(0.95):.0f}** "
            f"מתוך {len(precomp)} משחקים"
        )
        hist = ser.value_counts().sort_index()
        st.bar_chart(pd.DataFrame({"שכיחות": hist}))


# --- view: knockout simulation ----------------------------------------------
elif view == "knockout":
    st.header(i18n.header("knockout", lang))
    st.caption(
        "סימולציית מונטה קרלו של כל הטורניר: שלב הבתים → 1/16 → 1/8 → רבע → "
        "חצי → גמר. תוצאות שכבר הוזנו (status=finished) ננעלות; השאר נדגם מהמודל."
    )
    c1, c2 = st.columns(2)
    n_sim = c1.slider("מספר סימולציות", 500, 10000, 2000, step=500)
    seed = c2.number_input("seed (לשחזור)", 0, 9999, 42)

    if st.button("הרץ סימולציית טורניר"):
        with st.spinner("מריץ סימולציות..."):
            df = knockout.run(ds, n=int(n_sim), seed=int(seed))
        st.session_state["ko_df"] = df

    if "ko_df" in st.session_state:
        df = st.session_state["ko_df"]
        show = df.rename(
            columns={
                "name_he": "נבחרת",
                "group": "בית",
                "qualify_%": "העפלה %",
                "r16_%": "1/8 %",
                "qf_%": "רבע %",
                "sf_%": "חצי %",
                "final_%": "גמר %",
                "title_%": "אליפות %",
            }
        )[["נבחרת", "בית", "העפלה %", "1/8 %", "רבע %", "חצי %", "גמר %", "אליפות %"]]
        st.dataframe(show, use_container_width=True, hide_index=True, height=560)
        st.subheader("מועמדות לאליפות (10 המובילות)")
        top = df.head(10).set_index("name_he")["title_%"]
        st.bar_chart(pd.DataFrame({"אליפות %": top}))


# --- view: one simulated bracket --------------------------------------------
elif view == "bracket":
    st.header(i18n.header("bracket", lang))
    st.caption(
        "הרצה אחת של כל הטורניר (לא הסתברויות) — מי ניצח את מי בכל שלב עד "
        "האלופה. שנה את ה-seed כדי לראות תרחיש אחר."
    )
    seed = st.number_input("seed", 0, 99999, 7)
    if st.button("הגרל בראקט"):
        st.session_state["bracket"] = knockout.simulate_detail(ds, seed=int(seed))

    if "bracket" in st.session_state:
        b = st.session_state["bracket"]
        st.success(f"🏆 האלופה: {b['champion']}")
        for rnd in b["rounds"]:
            st.subheader(rnd["label"])
            rows = [
                {
                    "ביתית": t["home"],
                    "תוצאה": t["score"],
                    "אורחת": t["away"],
                    "עולה": t["winner"],
                }
                for t in rnd["ties"]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# --- view: draw difficulty / bracket geometry --------------------------------
elif view == "draw":
    st.header(i18n.header("draw", lang))
    st.caption(
        "סיכויי האליפות אינם רק עניין של חוזק — הם תלויים גם בהגרלה. כאן רואים "
        "אילו בתים מרכזים את רוב סיכויי האליפות, ואילו פייבוריטיות נמצאות באותה "
        "חצי/רבע של הבראקט ולכן חייבות להדיח זו את זו לפני הגמר."
    )
    c1, c2 = st.columns(2)
    n_sim = c1.slider("מספר סימולציות", 2000, 20000, 8000, step=1000)
    seed = c2.number_input("seed (לשחזור)", 0, 9999, 2026)

    if st.button("חשב קושי הגרלה"):
        with st.spinner("מריץ סימולציות..."):
            st.session_state["draw"] = knockout.draw_difficulty(
                ds, n=int(n_sim), seed=int(seed)
            )

    if "draw" in st.session_state:
        d = st.session_state["draw"]

        st.subheader("אליפות לפי בית")
        st.caption(
            "כל בית ממוין לפי סך סיכויי האליפות שלו (סכום על כל ארבע הנבחרות). "
            "ערך גבוה = 'בית מוות' שמחזיק פייבוריטית חזקה; 'אליפות (בית)' גבוה עם "
            "נבחרת מובילה אחת = מסלול נקי לאותה נבחרת."
        )
        gdf = pd.DataFrame(d["groups"]).rename(
            columns={
                "group": "בית",
                "top_team": "הנבחרת החזקה",
                "top_title": "אליפות (נבחרת) %",
                "group_title": "אליפות (בית) %",
                "avg_qualify": "העפלה ממוצעת %",
                "half": "חצי",
                "quarter": "רבע",
            }
        )[
            ["בית", "הנבחרת החזקה", "אליפות (נבחרת) %", "אליפות (בית) %",
             "העפלה ממוצעת %", "חצי", "רבע"]
        ]
        st.dataframe(gdf, use_container_width=True, hide_index=True)

        st.subheader("פייבוריטיות על מסלול התנגשות (בראקט 'צ'אלק')")
        st.caption(
            "בהנחה שכל בית מסתיים לפי הדירוג, באיזה שלב כל זוג פייבוריטיות (מבין "
            "שמונת הבתים החזקים) ייפגש לראשונה. פגישה מוקדמת = שתי נבחרות חזקות "
            "מדיחות זו את זו לפני הגמר, מה שמדכא את סיכויי האליפות המוצגים שלהן."
        )
        if d["collisions"]:
            cdf = pd.DataFrame(d["collisions"]).rename(
                columns={"team_a": "נבחרת א'", "team_b": "נבחרת ב'", "stage": "שלב הפגישה"}
            )
            st.dataframe(cdf, use_container_width=True, hide_index=True)
            st.caption(
                "ככל שזוג נפגש מוקדם יותר — כך הסימטריה ב'סיכויי אליפות' שלהן היא "
                "תוצר ההגרלה לא פחות מתוצר החוזק. נבחרת בחצי הנגדי תפגוש את "
                "השתיים רק בגמר — המסלול הנקי ביותר."
            )
        else:
            st.info("אף זוג פייבוריטיות אינו על מסלול התנגשות לפני הגמר.")


# --- view: bonus questions --------------------------------------------------
elif view == "bonus":
    st.header(i18n.header("bonus", lang))
    st.caption(
        "כל התשובות מחושבות מאותו מנוע + סימולציית נוקאאוט, ולכן מתעדכנות ככל "
        "ש-Hermes כותב תוצאות אמת ל-matches.csv. שתי שאלות (מלך בישולים, "
        "אמבפה/ויניסיוס) הן ברמת השחקן — מסומנות כהערכה ⚠️."
    )
    c1, c2 = st.columns(2)
    n_ko = c1.slider("סימולציות נוקאאוט", 1000, 10000, 4000, step=1000)
    seed = c2.number_input("seed (לשחזור)", 0, 9999, 2026)

    if st.button("חשב תשובות בונוס"):
        with st.spinner("מחשב..."):
            st.session_state["bonus"] = bonus.compute(
                ds, n_ko=int(n_ko), n_group=int(n_ko), seed=int(seed)
            )

    if "bonus" in st.session_state:
        b = st.session_state["bonus"]

        def flag(d):
            return " ⚠️ (הערכה ברמת שחקן)" if d.get("player_level") else ""

        def tie_flag(d):
            return " 🔀 (צמד צמוד — תיקו בתוך רעש הסימולציה)" if d.get("tie") else ""

        st.subheader("📋 תשובות מומלצות")
        st.dataframe(
            pd.DataFrame(
                [
                    {"שאלה": "סגנית האלופה" + tie_flag(b["runner_up"]), "תשובה": b["runner_up"]["answer"]},
                    {"שאלה": "מלך הבישולים" + flag(b["top_assists"]), "תשובה": b["top_assists"]["answer"]},
                    {"שאלה": "שער ראשון בטורניר", "תשובה": b["first_goal"]["answer"]},
                    {"שאלה": "שק החבטות (הכי הרבה ספיגות)", "תשובה": b["punching_bag"]["answer"]},
                    {"שאלה": "הכי הרבה שערים בשלב הבתים", "תשובה": b["most_group_goals"]["answer"]},
                    {"שאלה": "מסי vs רונאלדו (מי רחוק יותר)", "תשובה": b["messi_vs_ronaldo"]["answer"]},
                    {"שאלה": "אמבפה vs ויניסיוס (מי כובש יותר)" + flag(b["mbappe_vs_vinicius"]), "תשובה": b["mbappe_vs_vinicius"]["answer"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        st.markdown(f"**סגנית האלופה: {b['runner_up']['answer']}** — {b['runner_up']['note']}")
        if b["runner_up"].get("tie"):
            st.info("🔀 הפסגה צפופה: המועמדות המובילות נמצאות בתוך רעש הסימולציה, "
                    "ולכן הסגנית מדווחת כצמד ולא כנבחרת יחידה.")
        st.dataframe(pd.DataFrame(b["runner_up"]["table"]), use_container_width=True, hide_index=True)

        st.markdown(f"**שער ראשון: {b['first_goal']['answer']}** — {b['first_goal']['note']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**שק החבטות: {b['punching_bag']['answer']}**")
            st.dataframe(pd.DataFrame(b["punching_bag"]["table"]), use_container_width=True, hide_index=True)
        with col_b:
            st.markdown(f"**הכי הרבה שערים: {b['most_group_goals']['answer']}**")
            st.dataframe(pd.DataFrame(b["most_group_goals"]["table"]), use_container_width=True, hide_index=True)

        st.markdown(f"**מסי vs רונאלדו → {b['messi_vs_ronaldo']['answer']}**")
        st.dataframe(pd.DataFrame(b["messi_vs_ronaldo"]["table"]), use_container_width=True, hide_index=True)

        st.markdown(f"**מלך הבישולים: {b['top_assists']['answer']}** ⚠️ — {b['top_assists']['note']}")
        if b["top_assists"].get("table"):
            st.dataframe(pd.DataFrame(b["top_assists"]["table"]), use_container_width=True, hide_index=True)
        st.markdown(f"**אמבפה vs ויניסיוס → {b['mbappe_vs_vinicius']['answer']}** ⚠️ — {b['mbappe_vs_vinicius']['note']}")


# --- view: backtest / model reliability --------------------------------------
elif view == "reliability":
    st.header(i18n.header("reliability", lang))
    st.caption(
        "אותו מנוע הסתברויות מורץ רטרוספקטיבית על 64 משחקי מונדיאל 2022 (תוצאות "
        "אמת + נקודות FIFA מאוקטובר 2022, מגרש ניטרלי). כך בודקים אם המודל מכויל: "
        "ככל שה-Brier וה-Log-loss נמוכים יותר — טוב יותר. הכרעות בפנדלים נרשמות "
        "כתיקו ב-90 דק' (המודל חוזה זמן חוקי, לא דו-קרב פנדלים)."
    )

    rep = backtest.run()
    m = rep["model"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brier (מודל)", f"{m['brier']:.3f}",
              help="שגיאה ריבועית של וקטור ההסתברות. נמוך = טוב. 0.667 = ניחוש אחיד.")
    c2.metric("Log-loss", f"{m['log_loss']:.3f}",
              help="קונס בחומרה ניבוי בטוח ושגוי. נמוך = טוב.")
    c3.metric("דיוק (פגיעה בפייבוריט)", f"{m['accuracy']:.1%}")
    c4.metric("יתרון על ניחוש אחיד", f"{rep['skill_vs_uniform']:+.1%}",
              help="כמה ה-Brier טוב יותר מ-1/3-1/3-1/3. חיובי = המודל לומד.")

    st.subheader("השוואה ל-baselines")
    base_rows = [{"שיטה": "המודל שלנו", **m}]
    for name_he, key in [("ניחוש אחיד (1/3)", "uniform"), ("שכיחות בסיס", "base_rate")]:
        b = rep["baselines"][key]
        base_rows.append({"שיטה": name_he, "n": b["n"], "brier": b["brier"],
                          "log_loss": b["log_loss"], "accuracy": b["accuracy"]})
    bt = pd.DataFrame(base_rows)[["שיטה", "n", "brier", "log_loss", "accuracy"]]
    bt.columns = ["שיטה", "מס' משחקים", "Brier", "Log-loss", "דיוק"]
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.markdown(
        f"המודל טוב ב-**{rep['skill_vs_uniform']:+.1%}** מניחוש אחיד ו-"
        f"**{rep['skill_vs_base_rate']:+.1%}** משכיחות הבסיס (חיובי = עדיף)."
    )

    st.subheader("כיול (Calibration)")
    st.caption("כשהמודל אומר '60%' — האם זה קורה בערך ב-60% מהמקרים? "
               "פער חיובי = המודל זהיר מדי; שלילי = בטוח מדי. מקובץ על H/D/A.")

    pooled_cal, pooled_n = _pooled_calibration()
    src = "מונדיאל 2022 (64 משחקים)"
    if pooled_n > rep["n"]:
        src = st.radio(
            "מקור הנתונים לדיאגרמת הכיול:",
            [f"מונדיאל 2022 ({rep['n']} משחקים)",
             f"כל הטורנירים ({pooled_n} משחקים)"],
            index=1, horizontal=True,
        )
    cal_rows = pooled_cal if src.startswith("כל") else rep["calibration"]
    cal = pd.DataFrame(cal_rows)
    cal.columns = ["טווח חזוי", "n", "ממוצע חזוי", "תדירות בפועל", "פער"]

    st.caption("דיאגרמת אמינות: כל נקודה = פלח הסתברות. נקודה על הקו המקווקו "
               "(y=x) = כיול מושלם; מעליו = המודל זהיר מדי, מתחתיו = בטוח מדי. "
               "גודל הנקודה = כמות התחזיות בפלח.")
    try:
        st.altair_chart(_reliability_chart(cal), use_container_width=True)
    except Exception:
        chart = cal.set_index("טווח חזוי")[["ממוצע חזוי", "תדירות בפועל"]]
        st.line_chart(chart)
    st.dataframe(cal, use_container_width=True, hide_index=True)

    st.subheader("כיול פרמטר K (נקודות FIFA לכל שער עליונות)")
    st.caption("מריצים את הבקטסט על ערכי K שונים — מי שממזער את ה-Brier מכויל "
               "הכי טוב לנתונים ההיסטוריים. כך מחליטים לפי מדידה ולא לפי דעה.")
    ks = pd.DataFrame(rep["k_sweep"])
    best_k = min(rep["k_sweep"], key=lambda r: r["brier"])["K"]
    ks.columns = ["K", "Brier", "Log-loss", "דיוק"]
    st.dataframe(ks, use_container_width=True, hide_index=True)
    st.markdown(f"ה-K הממזער את ה-Brier: **{best_k}** "
                f"(הערך הנוכחי במנוע: **{engine.K:.0f}**).")

    if rep.get("elo_sweep"):
        st.subheader("שילוב Elo מול FIFA (המלצת ה-CR)")
        st.caption("בדיקה אובייקטיבית האם דירוג Elo משפר על נקודות FIFA: "
                   "w=0.0 = FIFA טהור (המודל הנוכחי), w=1.0 = Elo טהור. "
                   "מאמצים רק אם ה-Brier יורד.")
        es = pd.DataFrame(rep["elo_sweep"])
        best_e = min(rep["elo_sweep"], key=lambda r: r["brier"])
        es.columns = ["משקל Elo", "Brier", "Log-loss", "דיוק"]
        st.dataframe(es, use_container_width=True, hide_index=True)
        if best_e["elo_weight"] > 0:
            base_b = next(r["brier"] for r in rep["elo_sweep"] if r["elo_weight"] == 0.0)
            gain = (base_b - best_e["brier"]) / base_b
            st.markdown(
                f"ה-Brier הטוב ביותר ב-**w={best_e['elo_weight']:.1f}** "
                f"(שיפור של **{gain:+.1%}** מ-FIFA טהור) — שיפור **זעיר** על טורניר "
                f"בודד. לכן `ELO_WEIGHT` נשאר **{engine.ELO_WEIGHT:.1f}** (כבוי) עד "
                f"שתהיה דאטת Elo מאומתת מכמה טורנירים. ה-plumbing מוכן: הוסף עמודת "
                f"`elo_points` ל-teams.csv והעלה את `engine.ELO_WEIGHT`."
            )
        else:
            st.markdown("FIFA טהור מנצח — `ELO_WEIGHT` נשאר 0.")


# --- view: vs bookmakers -----------------------------------------------------
elif view == "market":
    st.header(i18n.header("market", lang))
    st.caption(
        "השוואת הסתברויות המודל מול הסתברויות השוק (יחסי הימור 1X2 לאחר הסרת "
        "מרווח הסוכן). יחס הימור עשרוני → הסתברות = 1/יחס, מנורמל לסכום 1. "
        "פער גדול = הזדמנות לבדיקה ידנית, לא בהכרח טעות."
    )

    anchors = ds.market_anchors() if hasattr(ds, "market_anchors") else []
    if not anchors:
        st.info(
            "אין עדיין נתוני שוק. הוסף שורות ל-`data/market_odds.csv` "
            "(match_id + dec_home/dec_draw/dec_away או p_home/p_draw/p_away) "
            "כדי להפעיל את העוגן. הקובץ קיים עם כותרות בלבד — דורמנטי עד שתמלא אותו."
        )
    else:
        n_flag = sum(1 for a in anchors if a["flag"])
        st.markdown(f"**{len(anchors)}** משחקים עם יחסי שוק · "
                    f"**{n_flag}** מסומנים לפער מהותי (≥10 נק' אחוז).")
        rows = []
        for a in anchors:
            mo = a["model"]; mk = a["market"]
            rows.append({
                "משחק": f"{team(ds.match(a['match_id']).home_id)} - "
                        f"{team(ds.match(a['match_id']).away_id)}",
                "מודל 1": f"{mo['p_home']*100:.0f}%",
                "שוק 1": f"{mk['p_home']*100:.0f}%",
                "מודל X": f"{mo['p_draw']*100:.0f}%",
                "שוק X": f"{mk['p_draw']*100:.0f}%",
                "מודל 2": f"{mo['p_away']*100:.0f}%",
                "שוק 2": f"{mk['p_away']*100:.0f}%",
                "פער מרבי": f"{a['max_gap']*100:+.0f} נק'",
                "מסכים על פייבוריט": "✓" if a["agree"] else "✗",
                "דגל": "🚩" if a["flag"] else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("KL גבוה = המודל רחוק מהשוק. ✗ ב'מסכים על פייבוריט' = המודל "
                   "והשוק חלוקים על מי הפייבוריט.")

    # --- per-match player props (anytime scorer / assist) ---
    st.divider()
    st.subheader("שחקנים — לקלוע או לבשל (פר משחק)")
    st.caption(
        "הסתברות שחקן לקלוע/לבשל במשחק בודד, לפי נתח השערים/בישולים שלו "
        "ותוחלת השערים של הנבחרת (פואסון: P(לפחות 1)=1-e^(-λ)). אם קיימים "
        "יחסי הימור ב-`data/players_market.csv`, הם מוצגים לצד המודל."
    )
    labels = {
        f"{m.group_id} | {team(m.home_id)} - {team(m.away_id)}": m.match_id
        for _, m in ds.matches.iterrows()
    }
    chosen = st.selectbox("בחר משחק", list(labels.keys()), key="props_match")
    props = ds.player_props(labels[chosen]) if hasattr(ds, "player_props") else []
    if not props:
        st.info("אין נתוני שחקנים לשתי הנבחרות במשחק זה (players.csv).")
    else:
        has_market = any("market" in p for p in props)
        rows = []
        for p in props:
            mo = p["model"]
            row = {
                "שחקן": p.get("name_he") or p.get("name_en"),
                "נבחרת": team(p["team_id"]),
                "מודל קלע": f"{mo['p_score']*100:.0f}%",
                "מודל בישל": f"{mo['p_assist']*100:.0f}%",
                "מודל קלע/בישל": f"{mo['p_score_or_assist']*100:.0f}%",
            }
            if has_market:
                mk = p.get("market", {})
                def _pct(v):
                    return f"{v*100:.0f}%" if v is not None else "—"
                row["שוק קלע"] = _pct(mk.get("p_score"))
                row["שוק קלע/בישל"] = _pct(mk.get("p_score_or_assist"))
                cmp = p.get("compare", {})
                flagged = any(c.get("flag") for c in cmp.values())
                row["דגל"] = "🚩" if flagged else ""
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if not has_market:
            st.caption(
                "להוספת יחסי שוק: מלא `data/players_market.csv` "
                "(match_id, team_id, name_en, score_odds, assist_odds, "
                "score_or_assist_odds). דורמנטי עד שתמלא אותו."
            )
