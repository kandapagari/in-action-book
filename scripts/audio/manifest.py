"""Cache manifest for the audio generation pipeline.

Two manifests live in the repo:

  1. ``site/scripts/audio/.audio-manifest.json``
     The build-cache manifest. Maps section ID → (content hash, output path,
     duration, voice). Used by ``generate.py`` to decide whether to skip a
     section whose preprocessed text hasn't changed.

  2. ``site/public/audio/manifest.json``
     The runtime manifest. Same data plus a relative URL each MP3 is served
     from. The Astro build imports this at build time so the player knows
     which sections have audio.

Both manifests share the same schema (one entry per section). We write the
build manifest first, then mirror it into the public manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Schema version. Bump when the entry shape changes in a backwards-
# incompatible way so we can invalidate stale caches.
SCHEMA_VERSION = 1


@dataclass
class AudioEntry:
    """One row in the manifest."""

    section: str           # "1.1", "4.x", etc. — matches the page route segment.
    chapter: int           # Numeric chapter (1, 2, …).
    title: str             # Section title (frontmatter).
    file: str              # Relative URL served by Astro, e.g. "/audio/chapter-1/section-1_1.mp3".
    path: str              # Absolute disk path on the build machine (debug only; not used by the site).
    content_hash: str      # SHA-256 of the preprocessed spoken text.
    duration_seconds: float
    file_size_bytes: int
    voice: str             # Kokoro voice ID, e.g. "af_heart".
    generated_at: str      # ISO-8601 UTC timestamp.
    char_count: int
    word_count: int


@dataclass
class Manifest:
    schema_version: int
    voice_default: str
    entries: dict[str, AudioEntry]   # keyed by section ID

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "voice_default": self.voice_default,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entries": {sec: asdict(e) for sec, e in self.entries.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path, default_voice: str) -> Manifest:
    """Load a manifest from disk, or return a fresh empty one.

    Raises only on JSON parse errors. A schema mismatch causes us to discard
    the file silently — the next ``save`` call will overwrite it.
    """
    if not path.exists():
        return Manifest(SCHEMA_VERSION, default_voice, {})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"audio: cannot parse manifest at {path}: {exc}.\n"
            f"audio: delete the file by hand to start over."
        ) from exc
    if raw.get("schema_version") != SCHEMA_VERSION:
        return Manifest(SCHEMA_VERSION, default_voice, {})
    entries: dict[str, AudioEntry] = {}
    for sec, data in raw.get("entries", {}).items():
        try:
            entries[sec] = AudioEntry(**data)
        except TypeError:
            # Stale entry shape — drop it and force a regen for that section.
            continue
    return Manifest(SCHEMA_VERSION, raw.get("voice_default", default_voice), entries)


def save_manifest(manifest: Manifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json() + "\n", encoding="utf-8")


def public_manifest_path(audio_dir: Path) -> Path:
    return audio_dir / "manifest.json"


def section_relative_url(chapter: int, section: str) -> str:
    """Map (chapter=1, section='1.1') → '/audio/chapter-1/section-1_1.mp3'."""
    return f"/audio/chapter-{chapter}/section-{section.replace('.', '_')}.mp3"


def section_disk_path(audio_dir: Path, chapter: int, section: str) -> Path:
    """Map (chapter=1, section='1.1') → audio_dir/chapter-1/section-1_1.mp3."""
    return audio_dir / f"chapter-{chapter}" / f"section-{section.replace('.', '_')}.mp3"


def get_entry(manifest: Manifest, section: str) -> Optional[AudioEntry]:
    return manifest.entries.get(section)


def upsert_entry(manifest: Manifest, entry: AudioEntry) -> None:
    manifest.entries[entry.section] = entry
