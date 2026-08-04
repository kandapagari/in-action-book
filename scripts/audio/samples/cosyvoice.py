# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "torch>=2.8,<3.0",
#     "torchaudio>=2.8",
#     "torchcodec",
#     "modelscope",
#     "HyperPyYAML",
#     "conformer==0.3.2",
#     "diffusers",
#     "einops",
#     "inflect",
#     "wetext",
#     "regex",
#     "openai-whisper",
#     "tqdm",
#     "onnxruntime",
#     "omegaconf",
#     "hydra-core",
#     "lightning",
#     "x-transformers",
#     "gdown",
#     "matplotlib",
#     "rich",
#     "rootutils",
#     "wget",
#     "scipy",
#     "pyarrow",
#     "tensorboard",
#     "networkx",
#     "pandas",
#     "transformers",
#     "librosa>=0.10",
#     "numpy>=2.0",
#     "soundfile>=0.13",
#     "pydub>=0.25.1",
#     "pyyaml>=6.0",
#     "huggingface_hub",
# ]
# ///
"""CosyVoice 2 sample generator — Alibaba's 0.5B zero-shot TTS (Apache-2.0).

Not pip-clean (Matcha-TTS is a git submodule), so the repo is vendor-cloned to
scripts/audio/samples/vendor/CosyVoice (gitignored) by run-samples.sh and
imported via sys.path. Zero-shot voice cloning from the Kokoro ref clip, same
as F5-TTS. Weights: FunAudioLLM/CosyVoice2-0.5B from the HF hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "CosyVoice"
sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(VENDOR_DIR / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import REF_TEXT, REF_WAV, run_engine, unshadow_script_dir  # noqa: E402

# vendor paths stay: they were inserted as absolute paths above
unshadow_script_dir()

MODEL_REPO = "FunAudioLLM/CosyVoice2-0.5B"


def load_model():
    if not VENDOR_DIR.exists():
        raise RuntimeError(f"vendored CosyVoice missing: {VENDOR_DIR} — run run-samples.sh (it clones it)")
    if not REF_WAV.exists():
        raise RuntimeError(f"reference clip missing: {REF_WAV} — run `uv run scripts/audio/samples/kokoro.py --emit-ref` first")

    # ponytail: cosyvoice.yaml references cosyvoice.dataset.processor, so
    # HyperPyYAML imports it at config load — and it top-level imports
    # pyworld, a training-only C-extension with no py3.12 wheels. Inference
    # never calls it, so satisfy the import with an inert stub instead of
    # building pyworld from source.
    import types

    try:
        import pyworld  # noqa: F401
    except ImportError:
        sys.modules.setdefault("pyworld", types.ModuleType("pyworld"))

    from cosyvoice.cli.cosyvoice import CosyVoice2
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(MODEL_REPO)
    print(f"[cosyvoice] loading CosyVoice2 from {model_dir} …", flush=True)
    # fp16=True is the supported path: checkpoints are bf16 and without
    # autocast the LLM dies on a Float-vs-BFloat16 matmul (surfaces later as
    # a 2-frame hifigan conv error). text_frontend=False per the repo's note
    # for reproducing CosyVoice2 results.
    model = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=True)
    return model


def synthesize(model, text: str):
    from cosyvoice.utils.common import set_all_random_seed

    set_all_random_seed(42)  # ponytail: per-chunk seed = reproducible voice
    # this CosyVoice version re-loads the prompt internally (24k + 16k), so
    # it takes the wav PATH, not a preloaded tensor
    speech = None
    for out in model.inference_zero_shot(text, REF_TEXT, str(REF_WAV), stream=False, text_frontend=False):
        speech = out["tts_speech"]
    if speech is None:
        raise RuntimeError("cosyvoice returned no audio for chunk")
    return np.asarray(speech.detach().float().cpu()).squeeze(), model.sample_rate


if __name__ == "__main__":
    sys.exit(
        run_engine(
            "cosyvoice",
            load_model,
            synthesize,
            chunk_chars=800,
            license="Apache-2.0",
            voice="zero-shot clone of Kokoro af_heart ref clip",
            notes="Alibaba CosyVoice2-0.5B (Qwen2.5-0.5B LLM + flow matching + HiFiGAN). Vendored repo, HF weights. Voice-cloned from the Kokoro reference, not model-native.",
        )
    )
