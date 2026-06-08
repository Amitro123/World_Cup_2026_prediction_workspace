"""
src/i18n.py — lightweight UI localization for the dashboard (Hebrew default, English option).

The dashboard ships Hebrew-first (RTL). An external review asked for an English
(LTR) option for a broader audience. Streamlit has no built-in i18n, so this is a
tiny string registry: each UI string has a stable KEY and a per-language value.

Design note — view identity is decoupled from its label. The navigation radio
returns a stable key ("fixtures", "live", …); only the *displayed* label is
translated. So the view dispatch in app.py never depends on the active language,
which keeps adding a language a data-only change here (no control-flow edits).

Coverage: this phase localizes the app chrome — navigation, page headers,
sidebar, and the language selector. Deep per-view body text (captions, table
column names) is still Hebrew and is the next localization phase; `t()` falls
back to Hebrew for any missing key so nothing ever renders blank.
"""

from __future__ import annotations

DEFAULT_LANG = "he"

# code -> native display name (shown in the language selector itself).
LANGUAGES = {"he": "עברית", "en": "English"}


def is_rtl(lang: str) -> bool:
    """Whether the layout should render right-to-left for this language."""
    return lang == "he"


# Navigation: (stable_key, {lang: label}). Order = sidebar order.
VIEWS: list[tuple[str, dict[str, str]]] = [
    ("fixtures",    {"he": "משחקים",            "en": "Fixtures"}),
    ("live",        {"he": "משחק חי",           "en": "Live Match"}),
    ("hermes",      {"he": "עדכוני Hermes",      "en": "Hermes Updates"}),
    ("overview",    {"he": "סקירת טורניר",       "en": "Tournament Overview"}),
    ("knockout",    {"he": "סימולציית נוקאאוט",  "en": "Knockout Simulation"}),
    ("bracket",     {"he": "בראקט מסומלץ",       "en": "Simulated Bracket"}),
    ("draw",        {"he": "קושי ההגרלה",        "en": "Draw Difficulty"}),
    ("bonus",       {"he": "שאלות בונוס",        "en": "Bonus Questions"}),
    ("market",      {"he": "מול בוקמייקרים",     "en": "vs Bookmakers"}),
    ("reliability", {"he": "אמינות המודל",       "en": "Model Reliability"}),
]

VIEW_KEYS: list[str] = [k for k, _ in VIEWS]
_VIEW_LABELS: dict[str, dict[str, str]] = {k: v for k, v in VIEWS}

# General chrome + per-view H1 headers.
STR: dict[str, dict[str, str]] = {
    # --- sidebar / app chrome ---
    "app_title":   {"he": "מונדיאל 2026 ⚽", "en": "World Cup 2026 ⚽"},
    "nav":         {"he": "תצוגה", "en": "View"},
    "language":    {"he": "שפה", "en": "Language"},
    "model_tag":   {"he": "מודל: דיקסון-קולס על נקודות דירוג פיפ\"א, בשילוב תחזיות מומחה.",
                    "en": "Model: Dixon-Coles on FIFA ranking points, blended with expert scorelines."},
    "refresh":     {"he": "🔄 רענן נתונים", "en": "🔄 Refresh data"},
    "refresh_help": {"he": "מושך מ-GitHub את העדכונים האחרונים של Hermes ומרענן את הלוח",
                     "en": "Pulls Hermes's latest updates from GitHub and refreshes the dashboard"},
    "data_ok":     {"he": "✓ נתונים תקינים", "en": "✓ Data valid"},
    # --- per-view headers (keyed by view key) ---
    "hdr_fixtures":    {"he": "טבלת משחקי שלב הבתים", "en": "Group-Stage Fixtures"},
    "hdr_live":        {"he": "מעקב משחק חי", "en": "Live Match Tracker"},
    "hdr_hermes":      {"he": "עדכוני Hermes — חדשות לפני משחק",
                        "en": "Hermes Updates — Pre-Match News"},
    "hdr_overview":    {"he": "סקירת טורניר — המצב שלי",
                        "en": "Tournament Overview — My Standing"},
    "hdr_knockout":    {"he": "סימולציית נוקאאוט — סיכויי כל נבחרת",
                        "en": "Knockout Simulation — Each Team's Odds"},
    "hdr_bracket":     {"he": "בראקט מסומלץ — ריצה בודדת",
                        "en": "Simulated Bracket — Single Run"},
    "hdr_draw":        {"he": "קושי ההגרלה — מי קיבל מסלול קל",
                        "en": "Draw Difficulty — Who Got the Easy Path"},
    "hdr_bonus":       {"he": "שאלות בונוס — תשובות מהמודל",
                        "en": "Bonus Questions — Model Answers"},
    "hdr_market":      {"he": "מול בוקמייקרים — עוגן שוק",
                        "en": "vs Bookmakers — Market Anchor"},
    "hdr_reliability": {"he": "אמינות המודל — בקטסט מונדיאל 2022",
                        "en": "Model Reliability — 2022 World Cup Backtest"},
}


def t(key: str, lang: str = DEFAULT_LANG) -> str:
    """Translate a chrome/header string key, falling back to Hebrew then the key."""
    d = STR.get(key, {})
    return d.get(lang) or d.get(DEFAULT_LANG) or key


def view_label(view_key: str, lang: str = DEFAULT_LANG) -> str:
    """Display label for a navigation view key in the given language."""
    d = _VIEW_LABELS.get(view_key, {})
    return d.get(lang) or d.get(DEFAULT_LANG) or view_key


def header(view_key: str, lang: str = DEFAULT_LANG) -> str:
    """The H1 header string for a view key."""
    return t(f"hdr_{view_key}", lang)
