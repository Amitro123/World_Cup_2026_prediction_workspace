"""
לוח בקרה — World Cup 2026 prediction dashboard (Hebrew / RTL).

Run:  streamlit run app.py
"""

import os
import random
import subprocess

import pandas as pd
import streamlit as st

from src import backtest, bonus, engine, knockout
from src.models import DataStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

st.set_page_config(page_title="מונדיאל 2026 — לוח תחזיות", layout="wide")

# Right-to-left layout
st.markdown(
    """
    <style>
      .stApp, .block-container { direction: rtl; text-align: right; }
      [data-testid="stSidebar"] { direction: rtl; text-align: right; }
      table { direction: rtl; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
st.sidebar.title("מונדיאל 2026 ⚽")
view = st.sidebar.radio(
    "תצוגה",
    ["משחקים", "משחק חי", "עדכוני Hermes", "סקירת טורניר", "סימולציית נוקאאוט",
     "בראקט מסומלץ", "שאלות בונוס", "אמינות המודל"],
)
st.sidebar.caption("מודל: דיקסון-קולס על נקודות דירוג פיפ\"א, בשילוב תחזיות מומחה.")

st.sidebar.divider()
if st.sidebar.button("🔄 רענן נתונים", use_container_width=True,
                     help="מושך מ-GitHub את העדכונים האחרונים של Hermes ומרענן את הלוח"):
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
        st.sidebar.caption("✓ נתונים תקינים")


# --- view: fixtures ----------------------------------------------------------
if view == "משחקים":
    st.header("טבלת משחקי שלב הבתים")
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
elif view == "משחק חי":
    st.header("מעקב משחק חי")
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

    state = ds.update_match_state(mid, int(minute), int(hg), int(ag))
    p = state["probabilities"]

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
elif view == "עדכוני Hermes":
    st.header("עדכוני Hermes — חדשות לפני משחק")
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
elif view == "סקירת טורניר":
    st.header("סקירת טורניר — המצב שלי")
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
elif view == "סימולציית נוקאאוט":
    st.header("סימולציית נוקאאוט — סיכויי כל נבחרת")
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
elif view == "בראקט מסומלץ":
    st.header("בראקט מסומלץ — ריצה בודדת")
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


# --- view: bonus questions --------------------------------------------------
elif view == "שאלות בונוס":
    st.header("שאלות בונוס — תשובות מהמודל")
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

        st.subheader("📋 תשובות מומלצות")
        st.dataframe(
            pd.DataFrame(
                [
                    {"שאלה": "סגנית האלופה", "תשובה": b["runner_up"]["answer"]},
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
elif view == "אמינות המודל":
    st.header("אמינות המודל — בקטסט מונדיאל 2022")
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
    cal = pd.DataFrame(rep["calibration"])
    cal.columns = ["טווח חזוי", "n", "ממוצע חזוי", "תדירות בפועל", "פער"]
    st.dataframe(cal, use_container_width=True, hide_index=True)
    try:
        chart = cal.set_index("טווח חזוי")[["ממוצע חזוי", "תדירות בפועל"]]
        st.line_chart(chart)
    except Exception:
        pass

    st.subheader("כיול פרמטר K (נקודות FIFA לכל שער עליונות)")
    st.caption("מריצים את הבקטסט על ערכי K שונים — מי שממזער את ה-Brier מכויל "
               "הכי טוב לנתונים ההיסטוריים. כך מחליטים לפי מדידה ולא לפי דעה.")
    ks = pd.DataFrame(rep["k_sweep"])
    best_k = min(rep["k_sweep"], key=lambda r: r["brier"])["K"]
    ks.columns = ["K", "Brier", "Log-loss", "דיוק"]
    st.dataframe(ks, use_container_width=True, hide_index=True)
    st.markdown(f"ה-K הממזער את ה-Brier: **{best_k}** "
                f"(הערך הנוכחי במנוע: **{engine.K:.0f}**).")
