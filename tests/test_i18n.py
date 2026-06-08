"""Tests for the UI localization registry (src/i18n.py)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import i18n  # noqa: E402


def test_default_language_is_hebrew_rtl():
    assert i18n.DEFAULT_LANG == "he"
    assert i18n.is_rtl("he") is True
    assert i18n.is_rtl("en") is False


def test_every_view_has_a_label_in_each_language():
    for key, labels in i18n.VIEWS:
        for lang in i18n.LANGUAGES:
            assert labels.get(lang), f"view {key} missing {lang} label"


def test_every_view_has_a_header_in_each_language():
    for key in i18n.VIEW_KEYS:
        for lang in i18n.LANGUAGES:
            h = i18n.header(key, lang)
            assert h and h != f"hdr_{key}", f"view {key} missing {lang} header"


def test_view_keys_match_views_order():
    assert i18n.VIEW_KEYS == [k for k, _ in i18n.VIEWS]
    assert len(set(i18n.VIEW_KEYS)) == len(i18n.VIEW_KEYS)  # no duplicate keys


def test_t_falls_back_to_hebrew_then_key():
    # a key that exists only conceptually: unknown -> returns the key itself
    assert i18n.t("definitely_missing_key", "en") == "definitely_missing_key"
    # English missing for a key would fall back to Hebrew; all real keys have both,
    # so verify the fallback mechanism directly on a synthetic entry.
    i18n.STR["_tmp_test"] = {"he": "עברית בלבד"}
    try:
        assert i18n.t("_tmp_test", "en") == "עברית בלבד"
    finally:
        del i18n.STR["_tmp_test"]


def test_translation_actually_differs_by_language():
    assert i18n.view_label("draw", "en") == "Draw Difficulty"
    assert i18n.view_label("draw", "he") != i18n.view_label("draw", "en")
    assert i18n.t("app_title", "en").startswith("World Cup")


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
