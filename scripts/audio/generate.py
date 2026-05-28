#!/usr/bin/env python3
"""Generate Kokoro audio for every drafted section in the book.

Reads section markdown from ``site/src/content/book/chapter_NN/section_*.md``
(the Astro mirror of ``book/`` at the workspace root — kept in sync by
``tools/sync-book-to-site.sh``, run automatically by ``npm run dev`` and
``npm run build``).

Writes MP3s to ``site/public/audio/chapter-{N}/section-{N_N}.mp3`` and two
manifests:

  * ``site/scripts/audio/.audio-manifest.json``  (build cache)
  * ``site/public/audio/manifest.json``           (consumed by the Astro UI)

Idempotent: a section whose preprocessed text SHA-256 matches the cached
hash is skipped. Pass ``--force`` to regenerate everything, or
``--section 1.1`` to regenerate one.

Voice: defaults to ``af_heart`` (American female, warm, well-paced for
long-form). Override with ``--voice am_michael`` etc. See the Kokoro voices
list: https://github.com/hexgrad/kokoro

Device: Apple Silicon → MPS by default; everything else → CPU. Override with
``--device cpu``.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# espeak-ng wiring
# ─────────────────────────────────────────────────────────────────────────────
#
# Kokoro → misaki[en] → phonemizer-fork → espeak-ng. The bundled
# `espeakng-loader` wheel ships the dylib + data, but the dylib has the CI
# build path (/Users/runner/...) baked in for the data files, and macOS
# dyld does NOT honor phonemizer's runtime override.
#
# Workaround: prefer a system-installed espeak-ng (`brew install espeak-ng`)
# whose data files live at /opt/homebrew/share/espeak-ng-data/, where the
# brew-built dylib expects them. Fall back to the bundled loader only if
# brew is unavailable.

def _configure_espeak() -> None:
    """Point misaki/phonemizer at a working espeak-ng install.

    ``misaki/espeak.py`` unconditionally calls
    ``EspeakWrapper.set_library(espeakng_loader.get_library_path())`` on
    import. The bundled ``espeakng-loader`` dylib has the CI build path
    (``/Users/runner/...``) baked in for the data files and macOS dyld
    does NOT honor the runtime override. So we *force* misaki to import
    first (running its broken setup), then overwrite both paths to a
    system-installed espeak-ng.
    """
    candidates = [
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
    ]
    data_candidates = [
        "/opt/homebrew/share/espeak-ng-data",
        "/usr/local/share/espeak-ng-data",
        "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
        "/usr/share/espeak-ng-data",
    ]
    lib_path = next((p for p in candidates if Path(p).exists()), None)
    data_path = next((p for p in data_candidates if Path(p).exists()), None)
    if not lib_path:
        raise RuntimeError(
            "audio: cannot find a system-installed espeak-ng library.\n"
            "audio: install one with `brew install espeak-ng` (macOS) or "
            "`sudo apt install espeak-ng libespeak-ng-dev` (Linux), then "
            "re-run."
        )
    # Step 1: import misaki.espeak so its own (broken) setup runs first.
    import misaki.espeak  # noqa: F401
    # Step 2: stomp the wrapper state with the system paths. This is what
    # actually fixes the loader.
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(lib_path)
    if data_path:
        EspeakWrapper.set_data_path(data_path)


_configure_espeak()


from manifest import (
    AudioEntry,
    Manifest,
    load_manifest,
    save_manifest,
    section_disk_path,
    section_relative_url,
    sha256_of_text,
    upsert_entry,
    public_manifest_path,
)
from preprocess import preprocess_markdown, split_into_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# This script lives at site/scripts/audio/generate.py.
SCRIPT_DIR = Path(__file__).resolve().parent
SITE_DIR = SCRIPT_DIR.parent.parent              # .../site
CONTENT_DIR = SITE_DIR / "src" / "content" / "book"
AUDIO_DIR = SITE_DIR / "public" / "audio"
BUILD_MANIFEST_PATH = SCRIPT_DIR / ".audio-manifest.json"


DEFAULT_VOICE = "af_heart"
# Statuses we generate audio for. PROGRESS.md uses [x] (drafted) / [r] (revised);
# the section frontmatter uses status: draft | revised | in-progress.
ELIGIBLE_STATUSES = {"draft", "revised"}

KOKORO_SAMPLE_RATE = 24_000  # Kokoro outputs 24 kHz mono float32.


# ─────────────────────────────────────────────────────────────────────────────
# Frontmatter + section discovery
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
    """Quote unquoted free-text values whose body contains a colon.

    Mirrors the normalization in ``site/src/lib/book-loader.ts`` so the
    Python and Astro halves of the toolchain see the same frontmatter.

    Authoring rule: titles like
        title: Three loss families: supervised, RL, self-supervised
    are unquoted on disk for readability, but YAML treats the second
    colon as a nested-mapping delimiter. We wrap the value in double
    quotes here.
    """
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


def discover_sections(content_dir: Path) -> list[SectionFile]:
    sections: list[SectionFile] = []
    if not content_dir.exists():
        raise FileNotFoundError(
            f"audio: content mirror missing at {content_dir}.\n"
            f"audio: run `npm run sync:book` from site/ first."
        )
    for md_path in sorted(content_dir.glob("chapter_*/section_*.md")):
        raw = md_path.read_text(encoding="utf-8")
        try:
            data, body = _split_frontmatter(raw)
        except ValueError as exc:
            print(f"  skip {md_path.name}: {exc}", file=sys.stderr)
            continue
        chapter = data.get("chapter")
        section = data.get("section")
        title = data.get("title", "")
        status = data.get("status", "")
        if chapter is None or section is None:
            print(f"  skip {md_path.name}: missing chapter/section in frontmatter", file=sys.stderr)
            continue
        sections.append(
            SectionFile(
                path=md_path,
                chapter=int(chapter),
                section=str(section),
                title=str(title),
                status=str(status),
                body=body,
            )
        )
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Kokoro pipeline
# ─────────────────────────────────────────────────────────────────────────────

def select_device(requested: Optional[str]) -> str:
    import torch

    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_PIPELINE = None
_PIPELINE_DEVICE: Optional[str] = None


def get_pipeline(device: str, lang_code: str = "a"):
    """Load Kokoro's KPipeline once and cache it.

    ``lang_code='a'`` is American English. Kokoro voices are language-locked
    by their first letter ('a' = American, 'b' = British, etc.).
    """
    global _PIPELINE, _PIPELINE_DEVICE
    if _PIPELINE is not None and _PIPELINE_DEVICE == device:
        return _PIPELINE
    from kokoro import KPipeline
    import torch

    print(f"  loading Kokoro KPipeline (device={device}, lang={lang_code}) …")
    pipeline = KPipeline(lang_code=lang_code, device=torch.device(device))
    _PIPELINE = pipeline
    _PIPELINE_DEVICE = device
    return pipeline


def synthesize(text: str, voice: str, pipeline) -> np.ndarray:
    """Run Kokoro on a single text chunk; return float32 mono PCM at 24 kHz."""
    audio_chunks: list[np.ndarray] = []
    # KPipeline returns a generator of (graphemes, phonemes, audio) tuples
    # — one per internal sentence chunk.
    for _, _, audio in pipeline(text, voice=voice):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        audio_chunks.append(np.asarray(audio, dtype=np.float32))
    if not audio_chunks:
        raise RuntimeError("kokoro returned no audio chunks")
    return np.concatenate(audio_chunks)


def encode_mp3(samples: np.ndarray, sample_rate: int, out_path: Path, bitrate: str = "64k") -> None:
    """Encode a mono float32 numpy array to MP3 at the given bitrate.

    Path: write WAV to memory → pydub → MP3 on disk (pydub shells out to
    ffmpeg, which is required to be on PATH).
    """
    from pydub import AudioSegment

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    seg = AudioSegment.from_file(buf, format="wav").set_channels(1)
    seg.export(out_path, format="mp3", bitrate=bitrate, parameters=["-ac", "1"])


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"Kokoro voice ID (default: {DEFAULT_VOICE}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate audio even if the content hash matches the cache.",
    )
    p.add_argument(
        "--section",
        action="append",
        default=None,
        help="Regenerate only this section (e.g. 1.1). May be passed multiple times.",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Torch device (default: auto — MPS on Apple Silicon, CUDA on Linux/NVIDIA, else CPU).",
    )
    p.add_argument(
        "--bitrate",
        default="64k",
        help="MP3 bitrate for pydub/ffmpeg (default: 64k).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover sections and show what would be generated, but don't load Kokoro or write audio.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    sections = discover_sections(CONTENT_DIR)
    eligible = [s for s in sections if s.status in ELIGIBLE_STATUSES]
    if args.section:
        wanted = set(args.section)
        eligible = [s for s in eligible if s.section in wanted]
        missing = wanted - {s.section for s in eligible}
        if missing:
            print(
                f"audio: requested section(s) not found or not drafted: {sorted(missing)}",
                file=sys.stderr,
            )
            return 1

    print(f"audio: discovered {len(sections)} section files; {len(eligible)} eligible to render")

    manifest = load_manifest(BUILD_MANIFEST_PATH, default_voice=args.voice)
    # If the manifest's default voice differs from the requested voice we keep
    # both — but record what the latest run was using. Per-entry voice still
    # reflects what the saved MP3 was actually generated with.
    manifest.voice_default = args.voice

    device = None
    pipeline = None

    total = len(eligible)
    generated = 0
    skipped = 0
    failures: list[tuple[str, str]] = []
    start_wall = time.time()

    for idx, sec in enumerate(eligible, 1):
        prep = preprocess_markdown(sec.body)
        content_hash = sha256_of_text(prep.text)
        out_path = section_disk_path(AUDIO_DIR, sec.chapter, sec.section)

        existing = manifest.entries.get(sec.section)
        up_to_date = (
            existing is not None
            and existing.content_hash == content_hash
            and existing.voice == args.voice
            and out_path.exists()
            and not args.force
        )
        if up_to_date:
            print(f"  [{idx}/{total}] §{sec.section} up-to-date, skip")
            skipped += 1
            continue

        approx_minutes = max(1, prep.word_count // 150)  # ~150 wpm for spoken English
        print(
            f"  [{idx}/{total}] §{sec.section} \"{sec.title}\""
            f" — {prep.word_count} words (~{approx_minutes} min audio) …",
            flush=True,
        )

        if args.dry_run:
            generated += 1
            continue

        if pipeline is None:
            device = select_device(args.device)
            pipeline = get_pipeline(device)

        chunk_start = time.time()
        try:
            chunks = split_into_chunks(prep.text, max_chars=1500)
            audio_chunks: list[np.ndarray] = []
            for ci, chunk in enumerate(chunks, 1):
                audio_chunks.append(synthesize(chunk, args.voice, pipeline))
                if len(chunks) > 1:
                    print(f"      chunk {ci}/{len(chunks)} ({len(chunk)} chars) done", flush=True)
            samples = np.concatenate(audio_chunks)
            duration = float(len(samples) / KOKORO_SAMPLE_RATE)
            encode_mp3(samples, KOKORO_SAMPLE_RATE, out_path, bitrate=args.bitrate)
        except Exception as exc:  # noqa: BLE001 — surface and continue
            print(f"      FAILED: {exc!r}", file=sys.stderr)
            failures.append((sec.section, str(exc)))
            continue

        file_size = out_path.stat().st_size
        wall = time.time() - chunk_start
        print(
            f"      done in {wall:0.1f}s — {duration:0.1f}s audio, "
            f"{file_size / 1024:0.0f} KB MP3"
        )

        entry = AudioEntry(
            section=sec.section,
            chapter=sec.chapter,
            title=sec.title,
            file=section_relative_url(sec.chapter, sec.section),
            path=str(out_path),
            content_hash=content_hash,
            duration_seconds=round(duration, 2),
            file_size_bytes=file_size,
            voice=args.voice,
            generated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
            char_count=prep.char_count,
            word_count=prep.word_count,
        )
        upsert_entry(manifest, entry)
        # Persist manifest after every section so a Ctrl-C mid-batch doesn't
        # lose progress.
        save_manifest(manifest, BUILD_MANIFEST_PATH)
        save_manifest(manifest, public_manifest_path(AUDIO_DIR))
        generated += 1

    # Always re-write both manifests at the end (covers the dry-run case and
    # the no-change case).
    save_manifest(manifest, BUILD_MANIFEST_PATH)
    save_manifest(manifest, public_manifest_path(AUDIO_DIR))

    wall_total = time.time() - start_wall
    print()
    print(f"audio: {generated} generated, {skipped} skipped, {len(failures)} failed in {wall_total:0.1f}s total")
    if failures:
        print("audio: failures:", file=sys.stderr)
        for sec, reason in failures:
            print(f"  §{sec}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
