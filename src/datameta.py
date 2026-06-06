"""
src/datameta.py — tiny freshness ledger for the ingested CSVs.

Records when each data file was last refreshed (and by which source) so the
dashboard can show "form: updated 2026-06-05 (api-football)" instead of leaving
the user guessing whether the numbers are current. Plain JSON at
``data/.data_meta.json``; absent keys simply read back as None.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

_FILE = ".data_meta.json"


def _path(data_dir: str) -> str:
    return os.path.join(data_dir, _FILE)


def read(data_dir: str) -> dict:
    """Return the freshness ledger ({} if none yet)."""
    p = _path(data_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def stamp(data_dir: str, key: str, source: str, added: int | None = None) -> None:
    """Record that `key` (e.g. 'form') was refreshed now from `source`."""
    meta = read(data_dir)
    meta[key] = {
        "updated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
    }
    if added is not None:
        meta[key]["rows_added"] = int(added)
    try:
        with open(_path(data_dir), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    except OSError:
        pass
