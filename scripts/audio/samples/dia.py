# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "nari-tts @ git+https://github.com/nari-labs/dia",
#     "numba>=0.59",
#     "torch>=2.8,<3.0",
#     "torchaudio>=2.8",
#     "numpy>=2.0",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
# ]
#
# [tool.uv]
# # nari-tts pins torch==2.6.0 on linux (cu124 — no sm_120 kernels for the
# # RTX 5090). Override to a Blackwell-capable torch; Dia README issue #26
# # says 5000-series needs torch>=2.8 anyway.
# override-dependencies = ["torch>=2.8,<3.0", "torchaudio>=2.8,<3.0", "triton"]
# ///
"""Dia sample generator — Nari Labs' 1.6B dialogue TTS (Apache-2.0).

Single-speaker: every chunk prefixed [S1]. Dia is not fine-tuned on a fixed
voice, so the seed is re-fixed before each chunk for cross-chunk voice
consistency (Nari's own recommendation for consistency without an audio
prompt). Chunks are small (~380 chars ≈ 25 s): per Nari's guidelines, inputs
producing >20 s of speech come out unnaturally fast. torch.compile is off —
not worth the warmup for a one-shot sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_engine, unshadow_script_dir  # noqa: E402

unshadow_script_dir()

SAMPLE_RATE = 44_100


def load_model():
    import torch
    from dia.model import Dia

    print("[dia] loading nari-labs/Dia-1.6B-0626 (bfloat16) …", flush=True)
    return Dia.from_pretrained(
        "nari-labs/Dia-1.6B-0626",
        compute_dtype="bfloat16",
        device=torch.device("cuda"),
    )


def synthesize(model, text: str):
    import torch

    torch.manual_seed(42)  # ponytail: per-chunk seed = same voice across chunks
    audio = model.generate(
        "[S1] " + text,
        max_tokens=3072,
        cfg_scale=3.0,
        temperature=1.2,
        top_p=0.95,
        cfg_filter_top_k=45,
        use_torch_compile=False,
    )
    return np.asarray(audio, dtype=np.float32).squeeze(), SAMPLE_RATE


if __name__ == "__main__":
    sys.exit(
        run_engine(
            "dia",
            load_model,
            synthesize,
            chunk_chars=380,
            license="Apache-2.0",
            voice="no fixed voice (seed-pinned for consistency)",
            notes="Nari 1.6B, dialogue-oriented; may inject spontaneous non-verbals. ~25s chunks per Nari's guidance (longer → rushed speech). Seed 42 per chunk.",
        )
    )
