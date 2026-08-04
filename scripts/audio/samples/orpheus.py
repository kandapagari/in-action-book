# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "orpheus-speech",
#     "vllm",
#     "snac",
#     "transformers",
#     "numpy>=2.0",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
# ]
# ///
"""Orpheus sample generator — canopylabs 3B Llama TTS via vLLM (Apache-2.0).

Voice: tara. No torch pin here on purpose — vLLM pins the torch it was built
against, and recent PyPI torch builds are cu130 (sm_120-capable). Chunks are
small (~350 chars) so a chunk's audio tokens stay well under max_tokens.

Uses vLLM's SYNC LLM API directly instead of orpheus-speech's OrpheusModel:
its AsyncLLMEngine wrapper dies silently after the first request on this
vllm/sm_120 stack (EngineDeadError at the second add_request, no python
traceback). The sync path survives sequential requests (verified with a
3-request probe). Prompt format, sampling params and the SNAC decode are
replicated exactly from orpheus_tts.engine_class / orpheus_tts.decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_engine, unshadow_script_dir  # noqa: E402

unshadow_script_dir()

SAMPLE_RATE = 24_000
MODEL_ID = "unsloth/orpheus-3b-0.1-ft"  # ungated mirror of canopylabs/orpheus-3b-0.1-ft
VOICE = "tara"


def load_model():
    import os

    # ponytail: vLLM uses flashinfer kernels on Blackwell, but flashinfer-JIT
    # needs nvcc (absent) and dies with a misleading "requires sm75" error.
    # Force vLLM's vendored FlashAttention + native torch sampler instead.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"[orpheus] loading {MODEL_ID} via sync vLLM …", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, dtype=torch.bfloat16)
    from orpheus_tts.decoder import tokens_decoder_sync  # SNAC on cuda, loads at import

    return tok, llm, tokens_decoder_sync


def synthesize(ctx, text: str):
    import torch
    from vllm import SamplingParams

    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        from vllm import TokensPrompt

    tok, llm, tokens_decoder_sync = ctx

    # exact prompt construction from orpheus_tts.engine_class._format_prompt
    prompt_ids = tok(f"{VOICE}: {text}", return_tensors="pt").input_ids
    start = torch.tensor([[128259]], dtype=torch.int64)
    end = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    full = torch.cat([start, prompt_ids, end], dim=1)[0].tolist()

    sp = SamplingParams(
        temperature=0.6,
        top_p=0.8,
        max_tokens=2600,  # engine default 1200 ≈ 14 s of audio; 350-char chunks need ~2x
        stop_token_ids=[49158],
        repetition_penalty=1.1,
    )
    out = llm.generate([TokensPrompt(prompt_token_ids=full)], sp)
    gen_ids = list(out[0].outputs[0].token_ids)

    # feed the SNAC decoder one token-string at a time, like the async stream
    token_strings = tok.convert_ids_to_tokens(gen_ids)
    pcm = np.frombuffer(b"".join(tokens_decoder_sync(iter(token_strings))), dtype=np.int16)
    if pcm.size == 0:
        raise RuntimeError("orpheus returned no audio for chunk")
    return (pcm.astype(np.float32) / 32768.0), SAMPLE_RATE


if __name__ == "__main__":
    sys.exit(
        run_engine(
            "orpheus",
            load_model,
            synthesize,
            chunk_chars=350,
            license="Apache-2.0",
            voice="tara",
            notes="Canopy 3B Llama + SNAC decoder on vLLM (sync path; the async wrapper crashes on sm_120). Weights: unsloth mirror — canopylabs repos are gated. ~350-char chunks.",
        )
    )
