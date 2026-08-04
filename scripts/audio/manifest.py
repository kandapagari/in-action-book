"""Cache manifest for the audio generation pipeline.

Two manifests live in the repo:

  1. ``site/scripts/audio/.audio-manifest.json``
     The build-cache manifest. Maps section ID → per-engine renditions
     (content hash, output URL, duration, voice). Used by ``generate.py``
     to decide whether to skip a (section, engine, voice) whose
     preprocessed text hasn't changed.

  2. ``site/public/audio/manifest.json``
     The runtime manifest. Same data. The Astro build imports this at
     build time so the player knows which sections have audio and which
     engine renditions are available.

Both manifests share the same schema (one entry per section, one rendition
per engine). We write the build manifest first, then mirror it into the
public manifest.

Schema v2: renditions are keyed by engine so several TTS models can serve
the same section. v1 manifests (single Kokoro rendition per entry) are
migrated on load — see ``load_manifest``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Schema version. Bump when the entry shape changes in a backwards-
# incompatible way so we can invalidate stale caches.
SCHEMA_VERSION = 2

# The engine readers get unless they pick another one in the player.
DEFAULT_ENGINE = "orpheus"

# Kokoro is special on disk: its 636 MB of MP3s are already committed at
# /audio/chapter-N/… (no engine prefix) and moving them is not worth a
# history rewrite. Every other engine — including the default — writes
# /audio/<engine>/chapter-N/….
LEGACY_ENGINE = "kokoro"

# Display metadata stamped onto renditions migrated from v1 manifests (the
# v1 pipeline was Kokoro-only, so every migrated rendition is Kokoro).
KOKORO_LABEL = "Kokoro"
KOKORO_HOMEPAGE = "https://github.com/hexgrad/kokoro"


@dataclass
class Rendition:
    """One engine's rendering of a section."""

    file: str              # Relative URL served by Astro, e.g. "/audio/chapter-1/section-1_1.mp3".
    duration_seconds: float
    file_size_bytes: int
    voice: str             # Engine voice ID, e.g. "af_heart".
    generated_at: str      # ISO-8601 UTC timestamp.
    content_hash: str      # SHA-256 of the preprocessed spoken text.
    label: Optional[str] = None     # Display name for the reader picker, e.g. "Kokoro".
    homepage: Optional[str] = None  # Engine project URL for the disclosure link.


@dataclass
class AudioEntry:
    """One row in the manifest: a section plus every engine's rendition."""

    section: str           # "1.1", "4.x", etc. — matches the page route segment.
    chapter: int           # Numeric chapter (1, 2, …).
    title: str             # Section title (frontmatter).
    char_count: int
    word_count: int
    renditions: dict[str, Rendition] = field(default_factory=dict)  # keyed by engine name


@dataclass
class Manifest:
    schema_version: int
    engine_default: str
    voice_default: str
    entries: dict[str, AudioEntry]   # keyed by section ID

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "engine_default": self.engine_default,
            "voice_default": self.voice_default,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": {
                sec: {
                    **{k: v for k, v in asdict(e).items() if k != "renditions"},
                    "renditions": {
                        eng: {k: v for k, v in asdict(r).items() if v is not None}
                        for eng, r in e.renditions.items()
                    },
                }
                for sec, e in self.entries.items()
            },
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _migrate_v1(raw: dict, audio_dir: Optional[Path]) -> dict[str, AudioEntry]:
    """Re-register v1 Kokoro entries as v2 renditions without re-synthesis.

    v1 content hashes were computed over the same preprocessed text, so a
    migrated rendition whose MP3 still exists on disk hashes equal and
    ``generate.py`` skips it. Entries whose MP3 is missing are dropped —
    the next run regenerates them.
    """
    entries: dict[str, AudioEntry] = {}
    dropped = 0
    for sec, data in raw.get("entries", {}).items():
        try:
            chapter = int(data["chapter"])
            disk_path = section_disk_path(audio_dir, chapter, sec) if audio_dir else None
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if disk_path is not None and not disk_path.exists():
            dropped += 1
            continue
        try:
            rendition = Rendition(
                file=str(data["file"]),
                duration_seconds=float(data["duration_seconds"]),
                file_size_bytes=int(data["file_size_bytes"]),
                voice=str(data["voice"]),
                generated_at=str(data["generated_at"]),
                content_hash=str(data["content_hash"]),
                label=KOKORO_LABEL,
                homepage=KOKORO_HOMEPAGE,
            )
            entries[sec] = AudioEntry(
                section=sec,
                chapter=chapter,
                title=str(data.get("title", "")),
                char_count=int(data.get("char_count", 0)),
                word_count=int(data.get("word_count", 0)),
                renditions={LEGACY_ENGINE: rendition},
            )
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
    print(
        f"audio: migrated {len(entries)} kokoro rendition(s) from v1 manifest"
        + (f" ({dropped} dropped, MP3 missing or stale)" if dropped else "")
    )
    return entries


def load_manifest(
    path: Path,
    default_voice: str,
    audio_dir: Optional[Path] = None,
) -> Manifest:
    """Load a manifest from disk, or return a fresh empty one.

    Raises only on JSON parse errors. A v1 manifest is migrated in place
    (see ``_migrate_v1``); ``audio_dir`` is used to verify the referenced
    MP3s still exist. Any other schema mismatch discards the file — the
    next ``save`` call overwrites it.
    """
    if not path.exists():
        return Manifest(SCHEMA_VERSION, DEFAULT_ENGINE, default_voice, {})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"audio: cannot parse manifest at {path}: {exc}.\n"
            f"audio: delete the file by hand to start over."
        ) from exc
    version = raw.get("schema_version")
    if version == 1:
        return Manifest(
            SCHEMA_VERSION,
            DEFAULT_ENGINE,
            raw.get("voice_default", default_voice),
            _migrate_v1(raw, audio_dir),
        )
    if version != SCHEMA_VERSION:
        return Manifest(SCHEMA_VERSION, DEFAULT_ENGINE, default_voice, {})
    entries: dict[str, AudioEntry] = {}
    for sec, data in raw.get("entries", {}).items():
        try:
            renditions = {
                eng: Rendition(**r) for eng, r in data.pop("renditions", {}).items()
            }
            entries[sec] = AudioEntry(**data, renditions=renditions)
        except TypeError:
            # Stale entry shape — drop it and force a regen for that section.
            continue
    return Manifest(
        SCHEMA_VERSION,
        raw.get("engine_default", DEFAULT_ENGINE),
        raw.get("voice_default", default_voice),
        entries,
    )


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def public_manifest_path(audio_dir: Path) -> Path:
    return audio_dir / "manifest.json"


def section_relative_url(chapter: int, section: str, engine: str = LEGACY_ENGINE) -> str:
    """Map (chapter=1, section='1.1') → '/audio/chapter-1/section-1_1.mp3'.

    Only kokoro (the pre-multi-engine pipeline) keeps the unprefixed legacy
    paths; every other engine gets its own subtree '/audio/<engine>/chapter-1/…'.
    """
    prefix = "" if engine == LEGACY_ENGINE else f"{engine}/"
    return f"/audio/{prefix}chapter-{chapter}/section-{section.replace('.', '_')}.mp3"


def section_disk_path(audio_dir: Path, chapter: int, section: str, engine: str = LEGACY_ENGINE) -> Path:
    """Map (chapter=1, section='1.1') → audio_dir/chapter-1/section-1_1.mp3."""
    base = audio_dir if engine == LEGACY_ENGINE else audio_dir / engine
    return base / f"chapter-{chapter}" / f"section-{section.replace('.', '_')}.mp3"


def get_entry(manifest: Manifest, section: str) -> Optional[AudioEntry]:
    return manifest.entries.get(section)


def upsert_entry(manifest: Manifest, entry: AudioEntry) -> None:
    manifest.entries[entry.section] = entry
