# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "kokoro>=0.9.4,<1.0",
#     "misaki[en]>=0.9.4",
#     "torch>=2.8,<3.0",
#     "numpy>=2.0",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
# ]
# ///
"""Kokoro sample generator — the production baseline engine.

ponytail: the synth internals (trim/join/synthesize) are copied from
generate.py; importing generate.py is impossible here because it hard-requires
a system espeak-ng at import time. This sample-side script instead uses the
bundled espeakng-loader wheel — its baked-path breakage is macOS-specific and
on Linux it is the happy path (no system espeak-ng on this machine, no sudo).
If the bundled loader turns out broken on Linux too, the selftest fails loudly
— that is the intended signal, not a case for another fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_engine, unshadow_script_dir  # noqa: E402

unshadow_script_dir()

KOKORO_SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"


def _configure_espeak() -> None:
    """Point misaki/phonemizer at the bundled espeak-ng (lib + data)."""
    import espeakng_loader
    import misaki.espeak  # noqa: F401 — runs misaki's own loader setup first
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def _trim_silence(samples: np.ndarray, threshold_db: float = -45.0, keep_ms: int = 20) -> np.ndarray:
    if samples.size == 0:
        return samples
    threshold = 10 ** (threshold_db / 20)
    above = np.abs(samples) > threshold
    if not above.any():
        return samples[:0]
    first = int(np.argmax(above))
    last = len(samples) - int(np.argmax(above[::-1]))
    keep = int(keep_ms * KOKORO_SAMPLE_RATE / 1000)
    return samples[max(0, first - keep) : min(len(samples), last + keep)]


def _join_with_pause(chunks: list[np.ndarray], pause_ms: int = 80) -> np.ndarray:
    if len(chunks) == 1:
        return chunks[0]
    pause = np.zeros(int(pause_ms * KOKORO_SAMPLE_RATE / 1000), dtype=np.float32)
    out: list[np.ndarray] = [chunks[0]]
    for c in chunks[1:]:
        out.extend((pause, c))
    return np.concatenate(out)


def load_model():
    import torch
    from kokoro import KPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[kokoro] loading KPipeline (device={device}, torch {torch.__version__}) …", flush=True)
    if device == "cuda":
        print(f"[kokoro] cuda capability {torch.cuda.get_device_capability()}", flush=True)
    return KPipeline(lang_code="a", device=torch.device(device))


def synthesize(pipeline, text: str):
    audio_chunks: list[np.ndarray] = []
    for _, _, audio in pipeline(text, voice=DEFAULT_VOICE):
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        arr = _trim_silence(np.asarray(audio, dtype=np.float32))
        if arr.size:
            audio_chunks.append(arr)
    if not audio_chunks:
        raise RuntimeError("kokoro returned no audio chunks")
    return _join_with_pause(audio_chunks), KOKORO_SAMPLE_RATE


if __name__ == "__main__":
    _configure_espeak()
    sys.exit(
        run_engine(
            "kokoro",
            load_model,
            synthesize,
            chunk_chars=4000,
            license="Apache-2.0",
            voice=DEFAULT_VOICE,
            notes="Production baseline (same pipeline settings as generate.py).",
        )
    )
