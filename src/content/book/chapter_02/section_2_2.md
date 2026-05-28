---
chapter: 2
section: 2.2
title: Setting up the environment (OpenVLA weights, LIBERO simulator)
target_words: 2000
status: draft
prereqs: §2.1 (what the running model looks like end-to-end); Python, pip, comfort reading tracebacks; an NVIDIA GPU
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
---

# 2.2  Setting up the environment (OpenVLA weights, LIBERO simulator)

Most "the VLA does not work" bug reports are setup bugs. The model loads, the
simulator renders, and then somewhere between them a driver mismatch or a
half-downloaded weight file turns the whole pipeline into a generator of
plausible-looking nonsense. The discipline this section asks of you is to
separate those layers and verify each one in isolation before you put them
in series. By the end of §2.2 you will have, in this order: a working
Python environment with a CUDA-matched PyTorch, a verified OpenVLA-7B
checkpoint on disk, and a LIBERO smoke test that rolls out a scripted
episode without the VLA touching anything. Only after all three of those
green lights does §2.3 wire them together.

## Hardware floor

OpenVLA-7B (Kim et al., 2024, arXiv:2406.09246) has 7.3 billion parameters.
In bfloat16, the weights alone are about 14 GB on disk and just under 15 GB
on the GPU; with activations, KV cache, and the dual vision encoders, a
single inference step needs roughly 18 GB of memory in steady state. The
practical hardware story comes out as follows.

A 24 GB-class consumer card (RTX 3090, 4090, A5000) is the comfortable
floor for bfloat16 inference. You will see no out-of-memory errors during
ordinary use and you can keep a few extra images in flight. A 16 GB card
(RTX 4080, A4000, T4) works with 8-bit quantization via `bitsandbytes`,
but quantization changes the numerics enough that some of the failure modes
you are going to study in §2.4 will look different from the canonical
ones — useful to know, frustrating to debug. A 12 GB card will not run
OpenVLA-7B end-to-end without aggressive offloading; this chapter assumes
you have at least 16 GB and prefers 24.

CUDA is a separate axis. `nvidia-smi` reports the highest CUDA toolkit
your installed driver supports — anything above that, and PyTorch will
either refuse to import or, worse, import and raise a cryptic `libcudart`
error on the first kernel launch. Match the PyTorch wheel index to your
driver, not to "whatever pip suggests." The example below pins CUDA 12.4;
adjust to your `nvidia-smi` output before running.

Disk and network are mundane but easy to get wrong. The OpenVLA weights
are ~14 GB; LIBERO's task assets, scene meshes, and demonstration data
add a few more. Stage them on the same physical disk as the project
working tree so the simulator does not silently fall back to a slow
NFS path mid-rollout. The first weight download is the largest single
network event in this chapter; if you are on a metered link, mirror the
checkpoint once with a pinned revision and set `HF_HUB_OFFLINE=1` on
every subsequent invocation.

## Python environment

Use a virtual environment. Not because conda is wrong — conda is fine —
but because a fresh `venv` is the smallest reproducible unit, and small
reproducible units are the only kind of dependency story that survives
six months. The illustrative commands below are not pinned versions;
they are the shape of the install. Your companion repository
(`code/chapter_02/`) holds the pinned `requirements.txt` and the exact
revision hashes used to produce this chapter's figures.

```bash
# CUDA 12.4 example — substitute your driver's CUDA major.minor.
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# Torch first: the wheel index must match nvidia-smi.
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Hugging Face stack for OpenVLA inference.
pip install "transformers>=4.40" accelerate sentencepiece \
    timm einops huggingface_hub
```

Three things to verify before installing anything else. `python -c "import
torch; print(torch.cuda.is_available(), torch.version.cuda)"` should print
`True` and a CUDA version that is less than or equal to whatever
`nvidia-smi` reports. `python -c "import torch; x =
torch.randn(1024, 1024, device='cuda'); print((x @ x).sum().item())"`
should run a CUDA kernel without raising. If either step fails, stop and
fix the driver/wheel mismatch — nothing downstream will work, and the
errors will get progressively less helpful the further you go.

## Downloading the OpenVLA weights

The canonical home for OpenVLA-7B is the Hugging Face Hub repository
`openvla/openvla-7b`. The library will resolve this name automatically the
first time you instantiate the model, but doing so leaves you at the
mercy of whatever revision happens to be at `main` on download day. Pin a
revision instead. The `snapshot_download` call below pulls the entire
repo to a local cache, verifies SHA-256 hashes, and writes a file you can
inspect.

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="openvla/openvla-7b",
    revision="main",          # replace with the commit hash for reproducibility
    local_dir="weights/openvla-7b",
    local_dir_use_symlinks=False,
)
print("Weights at:", local_dir)
```

After the download finishes, a healthy local copy contains the model
shards (`model-00001-of-00003.safetensors` through
`model-00003-of-00003.safetensors`), a `config.json`, a `processor_config.json`,
and a `dataset_statistics.json` file that holds the per-embodiment
normalization statistics. The last file is the most often overlooked and
the most consequential — it is the table the detokenizer reads to convert
the seven discrete action tokens back into a continuous 7-vector. Open it
once and look at the keys; you will see one entry per embodiment in Open
X-Embodiment (O'Neill et al., 2023, arXiv:2310.08864). The `unnorm_key`
parameter that appears in §2.3's `predict_action` call selects which of
those entries to use, and getting it wrong is one of the failure modes
covered in §2.4.

A quick sanity check before LIBERO is installed at all:

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained(
    "weights/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "weights/openvla-7b", torch_dtype=torch.bfloat16,
    trust_remote_code=True).to("cuda").eval()

print("Params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
```

This should print roughly `7.34 B` and use about 15 GB of GPU memory. If it
hangs on the first call, the cause is almost always a partial download —
delete the local directory and re-run `snapshot_download` with
`resume_download=True`. If it loads but produces NaNs on a dummy forward
pass, the same fix applies; checksums are not magic, and a process killed
mid-download can leave a file whose shape is correct but whose bytes are
not.

## Installing LIBERO

LIBERO (Liu et al., 2023, arXiv:2306.03310) is a benchmark suite for
language-conditioned manipulation built on top of RoboSuite, MuJoCo, and a
collection of curated task families: `libero_object` (pick a named object
out of distractors), `libero_spatial` (place at a named spatial relation),
`libero_goal` (achieve a goal state described in language), and
`libero_10` and `libero_90` (longer-horizon compositions). The suite ships
~6,500 teleoperated demonstrations; this chapter uses none of them, but
the same task definitions drive the evaluation in §2.4.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO
pip install -e LIBERO
python -m libero.libero.benchmark.download_assets   # one-time, ~3 GB
```

MuJoCo wants three system libraries on a Linux machine and they are not
always present on a fresh cloud image: `libGL`, `libEGL`, and
`libosmesa6`. On Ubuntu, `sudo apt-get install libgl1 libegl1
libosmesa6 patchelf` covers the common cases. On a headless server,
export `MUJOCO_GL=egl` before any LIBERO import; with a display attached,
`MUJOCO_GL=glfw` is fine. The symptom of getting this wrong is not an
import error but an all-black observation array — the simulator runs,
returns zeros for pixels, and the VLA's first action prediction looks
plausible until you realize the image it was conditioned on was blank.

## A smoke test that does not touch the VLA

Before any OpenVLA code talks to LIBERO, run a scripted episode with a
random action. The point is to verify that the simulator produces images,
that those images are not black, and that the action interface matches
what the VLA will eventually emit. The script below does all three in
about forty lines.

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark("libero_object")()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=task.bddl_file,
    camera_heights=256, camera_widths=256,
)
obs = env.reset()
print("image shape:", obs["agentview_image"].shape,
      "non-zero pixels:", int(np.count_nonzero(obs["agentview_image"])))

# 7-dimensional action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
rng = np.random.default_rng(0)
for step in range(40):
    action = rng.uniform(-0.02, 0.02, size=7)
    action[-1] = 1.0
    obs, reward, done, info = env.step(action)
env.close()
print("episode finished, reward at last step:", reward)
```

Three things have to be true at the end of that script. The image shape
is `(256, 256, 3)`. The non-zero pixel count is large — tens of thousands,
not zero. The episode runs for 40 steps without raising. If any of those
fails, you do not yet have a working LIBERO install, and adding the VLA
on top will only make the next traceback harder to read.

## Three setup failures that account for almost all of them

After watching dozens of readers stand this stack up, three failure
modes dominate.

The first is the *CUDA wheel mismatch*. PyTorch imports successfully, the
first `torch.cuda.is_available()` even returns `True`, and then the first
real kernel call raises something like `CUDA error: invalid argument` or
`libcudart.so.12: cannot open shared object file`. The fix is always to
reinstall the PyTorch wheel from the index that matches your driver's
maximum CUDA version. There is no other fix; the error message is
deliberately not helpful because the failure is below PyTorch's
abstraction layer.

The second is the *headless rendering trap*. LIBERO imports, the
simulator runs, the rollout finishes — and every image is black. This is
the symptom for a missing `libEGL` or an unset `MUJOCO_GL`. The smoke
test above catches it because the non-zero pixel count comes out as zero;
without that check, you find out hours later, after wondering why a VLA
that was correct on a development laptop is hallucinating actions on a
cloud instance.

The third is the *half-downloaded weight file*. `model.from_pretrained`
loads without complaint, the parameter count is right, and the first
forward pass returns NaNs. The cause is almost always a network drop
during a `snapshot_download` that did not raise — a single corrupted
shard inside an otherwise-valid directory tree. The fix is to delete the
local cache and re-download with `resume_download=True`; running the
hash check on every shard is overkill for an interactive workflow but
the right move for a CI job.

None of these failures is unique to OpenVLA or LIBERO. They are the
generic failure modes of GPU-plus-simulator stacks in 2026, and they
are the cost of admission for running any large model on real
hardware. Skipping the verification steps does not save time; it
relocates the time to a worse spot, after you have spent an hour
suspecting the model.

## A short reproducibility note

Save the exact versions you used. `pip freeze > pinned.txt` after the
install succeeds, and record the OpenVLA revision hash printed by
`snapshot_download`. When you come back to this chapter in six months —
or when a colleague reports a result that you cannot reproduce — those
two files are what separate a debugging session from an archaeological
dig. The book's companion repository keeps a `chapter_02/pinned.txt`
checked in; the rest of the book assumes you can do the same.

With a verified GPU, a pinned OpenVLA-7B checkpoint, and a LIBERO smoke
test that produces non-black images, the stack is ready. §2.3 opens the
inference loop and walks it one line at a time.
