#!/usr/bin/env bash
# Drive the audio narration pipeline.
#
# Creates site/scripts/audio/.venv on first run, installs requirements if
# the venv is missing or stale, makes sure the book mirror is in sync, then
# runs generate.py with whatever flags were passed.
#
# Invoked by:
#   npm run build:audio              → full batch
#   npm run build:audio:force        → --force (regenerate everything)
#   npm run build:audio:section 1.1  → --section 1.1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQS="$SCRIPT_DIR/requirements.txt"
REQS_STAMP="$VENV_DIR/.requirements.sha"

# Kokoro 0.9.4's published wheel requires Python >=3.10,<3.13 (note: github
# pyproject.toml says <3.14, but the pypi wheel disagrees — and pypi wins).
# Pick the newest interpreter in that range. Order: explicit env var → 3.12
# from uv → 3.12 from Homebrew → 3.11 → 3.10 → fail.
if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
  : # caller already set it
else
  CANDIDATES=(
    "/opt/homebrew/bin/python3.12"
    "/usr/local/bin/python3.12"
    "/opt/homebrew/bin/python3.11"
    "/usr/local/bin/python3.11"
    "/opt/homebrew/bin/python3.10"
    "/usr/local/bin/python3.10"
  )
  # Also probe uv's managed Python installs.
  if [ -d "$HOME/.local/share/uv/python" ]; then
    for d in "$HOME/.local/share/uv/python"/cpython-3.12*/bin/python3.12 \
             "$HOME/.local/share/uv/python"/cpython-3.11*/bin/python3.11 \
             "$HOME/.local/share/uv/python"/cpython-3.10*/bin/python3.10; do
      if [ -x "$d" ]; then CANDIDATES+=("$d"); fi
    done
  fi
  PYTHON_BIN=""
  for c in "${CANDIDATES[@]}"; do
    if [ -x "$c" ]; then PYTHON_BIN="$c"; break; fi
  done
  if [ -z "$PYTHON_BIN" ]; then
    echo "audio: no Python 3.10–3.12 interpreter found on this machine." >&2
    echo "audio: install one (e.g. \`brew install python@3.12\` or \`uv python install 3.12\`)" >&2
    echo "audio: or set PYTHON_BIN to a compatible interpreter and re-run." >&2
    exit 1
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "audio: creating venv at $VENV_DIR using $PYTHON_BIN" >&2
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools >&2
fi

# Reinstall requirements when requirements.txt changes.
REQS_HASH="$(shasum -a 256 "$REQS" | awk '{print $1}')"
if [ ! -f "$REQS_STAMP" ] || [ "$(cat "$REQS_STAMP")" != "$REQS_HASH" ]; then
  echo "audio: installing requirements" >&2
  "$VENV_DIR/bin/pip" install -r "$REQS" >&2
  echo "$REQS_HASH" > "$REQS_STAMP"
fi

# Make sure the book mirror is in sync — generate.py reads from it.
if command -v bash >/dev/null 2>&1; then
  bash "$SITE_DIR/../tools/sync-book-to-site.sh" >&2
fi

# ffmpeg is required by pydub. Fail loudly if it's missing.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "audio: ffmpeg not found on PATH. Install with: brew install ffmpeg" >&2
  exit 1
fi

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/generate.py" "$@"
