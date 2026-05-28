#!/usr/bin/env bash
# Drive the audio narration pipeline using uv.
#
# uv reads pyproject.toml + uv.lock + .python-version and manages the venv
# automatically. First run downloads Python 3.12 (if needed), creates a venv
# at scripts/audio/.venv, and installs the locked deps. Subsequent runs are fast.
#
# Invoked by:
#   npm run build:audio              → full batch
#   npm run build:audio:force        → --force (regenerate everything)
#   npm run build:audio:section 1.1  → --section 1.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure uv is installed.
if ! command -v uv >/dev/null 2>&1; then
  echo "audio: uv not found on PATH." >&2
  echo "audio:   install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "audio:   or:           brew install uv" >&2
  exit 1
fi

# Ensure ffmpeg is available (required by pydub for MP3 encoding).
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "audio: ffmpeg not found on PATH. Install with: brew install ffmpeg" >&2
  exit 1
fi

# Sync the book mirror so generate.py reads the latest content.
# Soft-skip if the workspace-root tools/ dir isn't reachable (e.g. on CI / GPU box
# where only site/ was cloned — in that case the committed mirror at
# site/src/content/book/ is already the source of truth).
SYNC_SCRIPT="$SITE_DIR/../tools/sync-book-to-site.sh"
if [ -f "$SYNC_SCRIPT" ]; then
  bash "$SYNC_SCRIPT" >&2
else
  echo "audio: sync script not present (expected outside the authoring machine) — skipping." >&2
fi

# Run via uv. From scripts/audio/ uv resolves pyproject.toml + uv.lock and
# executes inside the project venv. No need to activate manually.
cd "$SCRIPT_DIR"
exec uv run python generate.py "$@"
