"""Shared plumbing for the TTS engine shootout sample scripts.

Every engine script (kokoro.py, vibevoice.py, …) is a standalone PEP 723
script with its own isolated uv env. They all share this module for:

  * locating + preprocessing book sections (same spoken text for every
    model = fair comparison) — reuses preprocess.py from the parent dir
  * the Kokoro reference clip used as the zero-shot prompt (ref.wav/REF_TEXT)
  * WAV→MP3 encoding, chunk concatenation, samples.json bookkeeping
  * the common CLI: --section / --out / --max-chars / --selftest / --force

ponytail: the frontmatter/discovery helpers are copied from generate.py
rather than imported — generate.py configures espeak at import time and
hard-fails without a system espeak-ng (absent on this machine, no sudo).
If generate.py's discovery logic changes, re-copy. Upgrade path: extract
the shared half of generate.py into an import-safe module.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

AUDIO_DIR = Path(__file__).resolve().parent.parent  # scripts/audio
REPO_ROOT = AUDIO_DIR.parent.parent
CONTENT_DIR = REPO_ROOT / "src" / "content" / "book"
DEFAULT_OUT_DIR = REPO_ROOT / "scratch" / "tts-samples"

sys.path.insert(0, str(AUDIO_DIR))
from preprocess import preprocess_markdown, split_into_chunks  # noqa: E402

# Zero-shot prompt for F5-TTS and CosyVoice 2: Kokoro af_heart reading the
# opening of §1.1 — our own text, so no transcription step. ~10 s of speech.
REF_WAV = DEFAULT_OUT_DIR / "ref.wav"
REF_TEXT = "Start with an admission: action is the part of robotics that has resisted us the longest."

SELFTEST_TEXT = (
    "Start with an admission: action is the part of robotics that has resisted us the longest. "
    "In the 1980s the bottleneck was perception, and getting a robot to reliably segment a coffee cup "
    "from a table counted as a research result on its own."
)

def unshadow_script_dir() -> None:
    """Drop samples/ from sys.path so an engine script named like its package
    (kokoro.py, dia.py, cosyvoice.py, …) doesn't shadow the third-party import.
    Call right after importing common — `common` itself is already in
    sys.modules by then, and preprocess lives in the parent dir."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter + section discovery (copied from generate.py — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SectionFile:
    path: Path
    chapter: int
    section: str
    title: str
    status: str
    body: str


_FREE_TEXT_FIELDS = ("title", "prereqs")


def _normalize_frontmatter_block(block: str) -> str:
    out_lines: list[str] = []
    for line in block.splitlines():
        matched = False
        for field in _FREE_TEXT_FIELDS:
            prefix = f"{field}:"
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                already_quoted = (
                    (value.startswith('"') and value.endswith('"'))
                    or (value.startswith("'") and value.endswith("'"))
                )
                if not already_quoted and ": " in value:
                    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                    out_lines.append(f'{field}: "{escaped}"')
                else:
                    out_lines.append(line)
                matched = True
                break
        if not matched:
            out_lines.append(line)
    return "\n".join(out_lines)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        raise ValueError("missing frontmatter")
    end = raw.find("\n---", 3)
    if end == -1:
        raise ValueError("unterminated frontmatter")
    fm_block = raw[3:end].strip()
    body = raw[end + 4 :].lstrip("\n")
    normalized = _normalize_frontmatter_block(fm_block)
    data = yaml.safe_load(normalized) or {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter is not a mapping")
    return data, body


def get_section(section_id: str) -> SectionFile:
    """Find one section (e.g. '1.1') in the content mirror and parse it."""
    for md_path in sorted(CONTENT_DIR.glob("chapter_*/section_*.md")):
        raw = md_path.read_text(encoding="utf-8")
        try:
            data, body = _split_frontmatter(raw)
        except ValueError:
            continue
        if str(data.get("section")) == section_id:
            return SectionFile(
                path=md_path,
                chapter=int(data["chapter"]),
                section=section_id,
                title=str(data.get("title", "")),
                status=str(data.get("status", "")),
                body=body,
            )
    raise FileNotFoundError(f"section {section_id} not found under {CONTENT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────


def encode_mp3(samples: np.ndarray, sample_rate: int, out_path: Path, bitrate: str = "64k") -> None:
    """Mono float32 numpy array → MP3 on disk (pydub shells out to ffmpeg)."""
    from pydub import AudioSegment

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    seg = AudioSegment.from_file(buf, format="wav").set_channels(1)
    seg.export(out_path, format="mp3", bitrate=bitrate, parameters=["-ac", "1"])


def concat_chunks(chunks: list[np.ndarray], sample_rate: int, pause_ms: int = 100) -> np.ndarray:
    """Join per-chunk audio with a short silent pad (100 ms reads as a breath)."""
    if len(chunks) == 1:
        return chunks[0]
    pause = np.zeros(int(pause_ms * sample_rate / 1000), dtype=np.float32)
    out: list[np.ndarray] = [chunks[0]]
    for c in chunks[1:]:
        out.extend((pause, c))
    return np.concatenate(out)


def section_output_relpath(engine: str, section: str) -> str:
    return f"{engine}/section-{section.replace('.', '_')}.mp3"


# ─────────────────────────────────────────────────────────────────────────────
# samples.json bookkeeping
# ─────────────────────────────────────────────────────────────────────────────


def _samples_path(out_dir: Path) -> Path:
    return out_dir / "samples.json"


def load_samples(out_dir: Path) -> dict:
    p = _samples_path(out_dir)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"sections": {}, "engines": {}}


def _save_samples(out_dir: Path, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _samples_path(out_dir).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def already_done(out_dir: Path, engine: str, section: str) -> bool:
    data = load_samples(out_dir)
    entry = data["engines"].get(engine, {}).get("sections", {}).get(section, {})
    return entry.get("status") == "ok" and (out_dir / entry.get("file", "")).exists()


def record_result(out_dir: Path, engine: str, section: str, meta: dict, entry: dict) -> None:
    data = load_samples(out_dir)
    eng = data["engines"].setdefault(engine, {"meta": {}, "sections": {}})
    eng["meta"].update(meta)
    entry = dict(entry)
    entry["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    eng["sections"][section] = entry
    _save_samples(out_dir, data)


def record_section_info(out_dir: Path, sec: SectionFile, word_count: int) -> None:
    data = load_samples(out_dir)
    data["sections"][sec.section] = {"title": sec.title, "chapter": sec.chapter, "word_count": word_count}
    _save_samples(out_dir, data)


# ─────────────────────────────────────────────────────────────────────────────
# Engine driver
# ─────────────────────────────────────────────────────────────────────────────


def run_engine(
    engine: str,
    load_model,
    synthesize,
    *,
    chunk_chars: int,
    license: str,
    voice: str,
    notes: str = "",
) -> int:
    """Common CLI + driver for one engine.

    load_model() is called lazily (once, only when there is work to do) and its
    return value is passed to synthesize(model, text) -> (float32 mono pcm, sr).
    """
    p = argparse.ArgumentParser(description=f"{engine} sample generator (TTS shootout)")
    p.add_argument("--section", action="append", default=None, help="Section id (e.g. 1.1). Repeatable. Default: 1.1")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output dir (default: scratch/tts-samples)")
    p.add_argument("--max-chars", type=int, default=None, help=f"Chars per synth chunk (default: {chunk_chars})")
    p.add_argument("--selftest", action="store_true", help="Synthesize two sentences to a wav and exit")
    p.add_argument("--emit-ref", action="store_true", help=f"Write the zero-shot reference clip to {REF_WAV} and exit")
    p.add_argument("--force", action="store_true", help="Regenerate even if samples.json says done")
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"license": license, "voice": voice, "notes": notes}

    if args.selftest:
        t0 = time.time()
        model = load_model()
        audio, sr = synthesize(model, SELFTEST_TEXT)
        dur = len(audio) / sr
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        assert dur > 2.0, f"selftest audio too short: {dur:.2f}s"
        assert rms > 1e-4, f"selftest audio is silent (rms={rms})"
        path = out_dir / "selftest" / f"{engine}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, sr)
        print(f"SELFTEST OK {engine}: {dur:.1f}s audio, rms {rms:.4f}, {time.time() - t0:.1f}s wall -> {path}")
        return 0

    if args.emit_ref:
        model = load_model()
        audio, sr = synthesize(model, REF_TEXT)
        dur = len(audio) / sr
        REF_WAV.parent.mkdir(parents=True, exist_ok=True)
        sf.write(REF_WAV, audio, sr)
        print(f"REF OK {engine}: {dur:.1f}s -> {REF_WAV}")
        return 0

    sections = args.section or ["1.1"]
    max_chars = args.max_chars or chunk_chars
    model = None
    failures: list[str] = []

    for section_id in sections:
        sec = get_section(section_id)
        prep = preprocess_markdown(sec.body)
        record_section_info(out_dir, sec, prep.word_count)

        rel = section_output_relpath(engine, section_id)
        out_path = out_dir / rel
        if not args.force and already_done(out_dir, engine, section_id):
            print(f"[{engine}] §{section_id} already done, skip")
            continue

        print(f"[{engine}] §{section_id} \"{sec.title}\" — {prep.word_count} words, max {max_chars} chars/chunk", flush=True)
        if model is None:
            model = load_model()

        t0 = time.time()
        try:
            chunks = split_into_chunks(prep.text, max_chars=max_chars)
            parts: list[np.ndarray] = []
            sr = None
            for ci, chunk in enumerate(chunks, 1):
                audio, sr = synthesize(model, chunk)
                audio = np.asarray(audio, dtype=np.float32).squeeze()
                if audio.ndim != 1 or audio.size == 0:
                    raise RuntimeError(f"chunk {ci}: engine returned empty/non-mono audio")
                parts.append(audio)
                if len(chunks) > 1:
                    print(f"    chunk {ci}/{len(chunks)} ({len(chunk)} chars, {len(audio) / sr:.1f}s) done", flush=True)
            samples = concat_chunks(parts, sr)
            duration = float(len(samples) / sr)
            encode_mp3(samples, sr, out_path)
        except Exception as exc:
            record_result(out_dir, engine, section_id, meta, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{engine}] §{section_id} FAILED: {exc!r}", file=sys.stderr, flush=True)
            failures.append(section_id)
            continue

        wall = time.time() - t0
        record_result(
            out_dir,
            engine,
            section_id,
            meta,
            {
                "status": "ok",
                "file": rel,
                "duration_seconds": round(duration, 2),
                "file_size_bytes": out_path.stat().st_size,
                "wall_seconds": round(wall, 1),
            },
        )
        print(f"[{engine}] §{section_id} done in {wall:.1f}s — {duration:.1f}s audio, {out_path.stat().st_size / 1024:.0f} KB", flush=True)

    return 1 if failures else 0
