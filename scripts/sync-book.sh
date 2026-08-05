#!/usr/bin/env bash
# One-way sync of ../book/ → src/content/book/ via the monorepo helper.
# In the full monorepo checkout that script exists; in a standalone site/
# checkout (e.g. Cursor Cloud) it does not — the committed mirror under
# src/content/book/ is already current, so we no-op instead of failing.
set -euo pipefail

SCRIPT="../tools/sync-book-to-site.sh"

if [[ -f "$SCRIPT" ]]; then
  exec bash "$SCRIPT"
fi

echo "[sync:book] $SCRIPT not found; using committed src/content/book/ mirror"
exit 0
