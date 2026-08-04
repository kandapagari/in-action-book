#!/usr/bin/env bash
# TTS shootout driver: render §1.1 with every engine, then emit the
# comparison page. Idempotent — engines whose samples.json entry is already
# "ok" skip themselves, so re-running resumes where a failed run stopped.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

UV="${UV:-$HOME/.local/bin/uv}"
OUT="scratch/tts-samples"
SECTIONS=("$@")
if [ ${#SECTIONS[@]} -eq 0 ]; then SECTIONS=("1.1"); fi

mkdir -p "$OUT/logs"

SECTION_ARGS=()
for s in "${SECTIONS[@]}"; do SECTION_ARGS+=(--section "$s"); done

echo "==> ref clip (kokoro --emit-ref)"
if [ ! -f "$OUT/ref.wav" ]; then
  "$UV" run scripts/audio/samples/kokoro.py --emit-ref --out "$OUT" \
    || { echo "FATAL: ref clip generation failed"; exit 1; }
else
  echo "    ref.wav exists, skip"
fi

echo "==> vendored CosyVoice"
if [ ! -d scripts/audio/samples/vendor/CosyVoice/third_party/Matcha-TTS/matcha ]; then
  rm -rf scripts/audio/samples/vendor/CosyVoice
  git clone --depth 1 --shallow-submodules --recursive \
    https://github.com/FunAudioLLM/CosyVoice scripts/audio/samples/vendor/CosyVoice \
    || echo "WARN: CosyVoice clone failed (cosyvoice engine will fail)"
else
  echo "    vendor/CosyVoice exists, skip"
fi

ENGINES=(kokoro vibevoice chatterbox f5tts dia orpheus cosyvoice)
FAILED=()
for e in "${ENGINES[@]}"; do
  log="$OUT/logs/$e.log"
  echo "==> $e (log: $log)"
  if "$UV" run "scripts/audio/samples/$e.py" "${SECTION_ARGS[@]}" --out "$OUT" >"$log" 2>&1; then
    tail -1 "$log"
  else
    echo "    FAILED — see $log"
    FAILED+=("$e")
  fi
done

echo "==> comparison page"
"$UV" run scripts/audio/samples/emit_page.py --out "$OUT"

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "DONE with failures: ${FAILED[*]}"
  exit 1
fi
echo "DONE — open $OUT/index.html"
