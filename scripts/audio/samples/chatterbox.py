# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "chatterbox-tts==0.1.7",
#     "torch>=2.8,<3.0",
#     "torchaudio>=2.8",
#     "numpy<2",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
# ]
#
# [tool.uv]
# # chatterbox-tts pins torch==2.6.0 (cu124 — no sm_120 kernels for the
# # RTX 5090). Override to a Blackwell-capable torch.
# override-dependencies = ["torch>=2.8,<3.0", "torchaudio>=2.8,<3.0"]
# ///
"""Chatterbox sample generator — Resemble's 0.5B Llama-style TTS (MIT).

Default built-in voice, no cloning, so the comparison is prompt-free like
Kokoro/VibeVoice-with-demo-voice/Dia/Orpheus. Chunked small (~800 chars):
Chatterbox drifts on long inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_engine, unshadow_script_dir  # noqa: E402

unshadow_script_dir()


def load_model():
    import perth
    import torch
    from chatterbox.tts import ChatterboxTTS

    # resemble-perth gates PerthImplicitWatermarker to specific torch builds;
    # on torch 2.13 it's None and chatterbox 0.1.7 calls it unconditionally.
    # The watermark is an inaudible provenance marker — irrelevant to a
    # quality shootout — so stub it with a passthrough rather than
    # downgrading torch.
    if perth.PerthImplicitWatermarker is None:
        class _NoWatermark:
            def apply_watermark(self, wav, sample_rate=None):
                return wav

        perth.PerthImplicitWatermarker = _NoWatermark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[chatterbox] loading ChatterboxTTS (device={device}) …", flush=True)
    return ChatterboxTTS.from_pretrained(device=device)


def synthesize(model, text: str):
    wav = model.generate(text)
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    return np.asarray(wav, dtype=np.float32).squeeze(), model.sr


if __name__ == "__main__":
    sys.exit(
        run_engine(
            "chatterbox",
            load_model,
            synthesize,
            chunk_chars=800,
            license="MIT",
            voice="default built-in voice",
            notes="Resemble 0.5B (LLaMA backbone + vocoder). ~800-char chunks; longer inputs drift in pace/pitch.",
        )
    )
