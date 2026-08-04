# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "vibevoice @ git+https://github.com/vibevoice-community/VibeVoice",
#     "torch>=2.8,<3.0",
#     "transformers>=4.46",
#     "librosa>=0.10",
#     "numpy>=2.0",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
# ]
# ///
"""VibeVoice-1.5B sample generator — Microsoft's long-form TTS model.

Code: Microsoft removed the TTS inference code from microsoft/VibeVoice in
Sept 2025 ("disabled due to widespread misuse"); the MIT-licensed community
fork vibevoice-community/VibeVoice preserves it, so the dep points there.
Weights: tries microsoft/VibeVoice-1.5B first, falls back to the ungated
mirror vibevoice/VibeVoice-1.5B (no HF token on this machine).

Single-speaker narration: one "Speaker 1: <text>" script per chunk, voice
cloning (is_prefill) from the bundled en-Maya_woman demo voice (Alice is
avoided — Microsoft's FAQ notes she triggers random BGM).

ponytail: attn_implementation="sdpa" instead of flash_attention_2 — flash-attn
has no sm_120 wheel and a source build isn't worth it for a shootout sample.
Microsoft notes only FA2 is fully tested; if the sample sounds off, building
flash-attn is the upgrade path.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_OUT_DIR, run_engine, unshadow_script_dir  # noqa: E402

unshadow_script_dir()

SAMPLE_RATE = 24_000
MODEL_CANDIDATES = ["microsoft/VibeVoice-1.5B", "vibevoice/VibeVoice-1.5B"]
VOICE_URL = "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/en-Maya_woman.wav"
VOICE_PATH = DEFAULT_OUT_DIR / "voices" / "en-Maya_woman.wav"


def _ensure_voice() -> Path:
    if not VOICE_PATH.exists():
        VOICE_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[vibevoice] downloading demo voice -> {VOICE_PATH}", flush=True)
        urllib.request.urlretrieve(VOICE_URL, VOICE_PATH)
    if VOICE_PATH.stat().st_size < 100_000:
        raise RuntimeError(f"voice prompt looks truncated: {VOICE_PATH}")
    return VOICE_PATH


def load_model():
    import torch
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    voice = _ensure_voice()
    last_exc: Exception | None = None
    for model_id in MODEL_CANDIDATES:
        try:
            print(f"[vibevoice] loading {model_id} (bf16, sdpa) …", flush=True)
            processor = VibeVoiceProcessor.from_pretrained(model_id)
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            model.eval()
            model.set_ddpm_inference_steps(num_steps=10)
            print(f"[vibevoice] loaded {model_id}", flush=True)
            return processor, model, voice
        except Exception as exc:  # gated repo / download failure → try mirror
            print(f"[vibevoice] {model_id} unavailable: {type(exc).__name__}: {exc}", flush=True)
            last_exc = exc
    raise RuntimeError(f"no VibeVoice weights loadable (tried {MODEL_CANDIDATES})") from last_exc


def _multi_turn_script(text: str, max_turn_chars: int = 600) -> str:
    """Split a chunk into several "Speaker 1: …" turns.

    A single multi-thousand-char turn is outside VibeVoice's per-turn
    training distribution: the model emits EOS after ~300-400 audio frames
    and the chunk comes out truncated (observed: 3.8k chars → 55 s). Their
    own FAQ recommends chunking long text into multiple turns with the same
    speaker label. ~600 chars ≈ 45 s of speech per turn.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    turns: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + len(sent) + 1 > max_turn_chars:
            turns.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        turns.append(buf)
    return "\n".join(f"Speaker 1: {t}" for t in turns)


def synthesize(ctx, text: str):
    import torch

    processor, model, voice = ctx
    script = _multi_turn_script(text.replace("’", "'"))
    inputs = processor(
        text=[script],
        voice_samples=[[str(voice)]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
        is_prefill=True,
    )
    speech = outputs.speech_outputs[0] if outputs.speech_outputs else None
    if speech is None:
        raise RuntimeError("vibevoice returned no audio for chunk")
    return np.asarray(speech.detach().float().cpu()).squeeze(), SAMPLE_RATE


if __name__ == "__main__":
    sys.exit(
        run_engine(
            "vibevoice",
            load_model,
            synthesize,
            chunk_chars=4000,
            license="MIT (code); weights research-only per Microsoft",
            voice="en-Maya_woman (voice-cloned, Speaker 1)",
            notes="Long-form model (Qwen2.5-1.5B + diffusion head, 7.5 Hz tokens). Community fork of Microsoft's removed TTS code; sdpa attention. Spontaneous BGM is a documented quirk.",
        )
    )
