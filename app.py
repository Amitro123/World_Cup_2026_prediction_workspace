"""
לוח בקרה — World Cup 2026 prediction dashboard (Hebrew / RTL).

Run:  streamlit run app.py
"""

import os
import random

import pandas as pd
import streamlit as st

from src import engine, knockout
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


ds = get_store()


def team(tid: str) -> str:
    return ds.team_name(tid, "he")


# --- sidebar navigation ------------------------------------------------------
st.sidebar.title("מונדיאל 2026 ⚽")
view = st.sidebar.radio(
    "תצוגה",
    ["משחקים", "משחק חי", "עדכוני Hermes", "סקירת טורניר", "סימולציית נוקאאוט", "בראקט מסומלץ"],
)
st.sidebar.caption("מודל: דיקסון-קולס על נקודות דירוג פיפ\"א, בשילוב תחזיות מומחה.")


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
