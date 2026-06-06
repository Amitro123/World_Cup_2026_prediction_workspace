#!/usr/bin/env bash
#
# hermes_push.sh — run on the Hostinger VPS, AFTER Hermes finishes updating data.
#
# Hermes writes its updates into data/*.csv. This script commits those changes
# and pushes them to GitHub, which is the shared sync layer. The local Streamlit
# dashboard then pulls them via the "🔄 רענן נתונים" button.
#
# --- cron setup (example: every 30 min) -------------------------------------
#   crontab -e
#   */30 * * * * cd /path/to/WorldCup2026 && ./hermes_push.sh >> /tmp/hermes_push.log 2>&1
#
# Prereqs on the VPS (one-time):
#   - git installed and the repo cloned
#   - push auth configured (deploy key or a PAT in the remote URL / credential store)
#   - git identity set:
#       git config user.name  "Hermes Bot"
#       git config user.email "hermes@bot.local"
# ----------------------------------------------------------------------------

set -euo pipefail

# Move to the repo root (directory this script lives in).
cd "$(dirname "$0")"

# Make sure we are current before committing, to avoid push rejections.
git pull --ff-only origin main || git pull --ff-only origin master || true

# Stage only the data files Hermes touches — never the whole tree.
git add data/*.csv data/*.json 2>/dev/null || true

# Nothing changed? Exit quietly so cron logs stay clean.
if git diff --cached --quiet; then
  echo "$(date -Is) no data changes, nothing to push"
  exit 0
fi

STAMP="$(date -Is)"
git commit -m "data: Hermes auto-update ${STAMP}"
git push

echo "${STAMP} pushed Hermes data update"
