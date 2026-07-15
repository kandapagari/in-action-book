---
chapter: 2
section: 2.3
title: Walking through the inference loop one line at a time
target_words: 2000
status: draft
prereqs: §2.1 (the end-to-end picture), §2.2 (working GPU, pinned weights, LIBERO smoke test); comfort with PyTorch tensors and Hugging Face Auto* APIs
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
---

# 2.3  Walking through the inference loop one line at a time

By the end of §2.2 you have a verified GPU, pinned OpenVLA-7B weights, and a LIBERO environment producing non-black images under a scripted policy. This section glues the two together. The result is a closed loop: image and language in, seven discrete tokens out, a 7-vector commanded to MuJoCo, a new image one timestep later. We'll walk that loop line by line, name each intermediate object, and stop at exactly the places where things can go quietly wrong without raising an exception. The model in question is OpenVLA-7B (Kim et al., 2024, arXiv:2406.09246), running against LIBERO's `libero_object` suite (Liu et al., 2023, arXiv:2306.03310).

## The whole loop, in one listing

Read the following script before the prose below it. It's the entire chapter in twenty-five lines; everything after is annotation.

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

processor = AutoProcessor.from_pretrained("weights/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "weights/openvla-7b", torch_dtype=torch.bfloat16,
    trust_remote_code=True).to("cuda").eval()

suite = benchmark.get_benchmark("libero_object")()
task = suite.get_task(0)
env = OffScreenRenderEnv(bddl_file_name=task.bddl_file,
                        camera_heights=256, camera_widths=256)
obs = env.reset()
instruction = task.language                         # "pick up the alphabet soup and place it in the basket"

for step in range(400):
    image = Image.fromarray(obs["agentview_image"][::-1])          # flip; LIBERO returns upside-down
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key="libero_object", do_sample=False)
    obs, reward, done, info = env.step(action.tolist())
    if done: break
env.close()
```

That's the loop: a model instantiation, an environment, and a `for` step where an image becomes a prompt, the prompt becomes seven action tokens, the tokens become a 7-vector, and the simulator advances. Each of those transitions earns its own paragraph below.

## Loading the processor and the model

`AutoProcessor.from_pretrained` returns a bundle: a SigLIP image processor, a DINOv2 image processor, and a Llama-2 tokenizer, all driven by a `__call__` that takes `(prompt, image)` and returns the dict of tensors the model expects. `trust_remote_code=True` is required because OpenVLA's processor class lives in the model repository rather than in the upstream `transformers` release; the flag tells `transformers` to import that class from the downloaded snapshot. It isn't optional, and skipping it produces a `KeyError` on the model's architecture string instead of a helpful message.

`AutoModelForVision2Seq.from_pretrained(..., torch_dtype=torch.bfloat16)` allocates the 7.3-billion-parameter model in bfloat16 directly, skipping the fp32-to-bf16 cast that would otherwise spike memory at load time. `.to("cuda")` moves the weights to the GPU; `.eval()` disables dropout and switches LayerNorm and GroupNorm into inference mode. After this line completes, `nvidia-smi` should report roughly 15 GB of allocated memory. See 28 GB instead, and the dtype got silently coerced to float32, usually because of an older `transformers` version that doesn't honor `torch_dtype` for vision-language models. Pin `transformers>=4.40` and the issue disappears.

## Reading an observation out of LIBERO

`env.reset()` returns a dict with several arrays. The one that matters here is `obs["agentview_image"]`, a `(256, 256, 3)` uint8 array from the third-person "agent view" camera. Two non-obvious things about it. First, LIBERO returns images vertically flipped relative to what a human, or OpenVLA's vision encoder, expects. This is a MuJoCo rendering convention, not a bug, and `obs["agentview_image"][::-1]` corrects it. Skip the flip and the model sees the world upside down; the loop keeps running, the actions stay syntactically valid, and the success rate collapses without raising anything. It's the cleanest example of a silent failure in the whole loop, and the one §2.4 returns to first.

Second, `obs` also contains `obs["robot0_eye_in_hand_image"]`, a wrist camera view. OpenVLA-7B was trained primarily on third-person observations, so passing the wrist view as `image` produces coherent but task-irrelevant actions. Different VLAs expect different camera conventions: RT-2 (Brohan et al., 2023, arXiv:2307.15818) takes a single base-mounted view, π0 takes multiple, RDT-1B (arXiv:2410.07864) takes wrist plus base. Chapter 12 catalogs these choices. For OpenVLA in `libero_object`, `agentview_image` is the correct one.

`task.language` is the natural-language instruction baked into the LIBERO task definition. For `libero_object` task 0 it reads `"pick up the alphabet soup and place it in the basket"`. The string stays fixed per task; LIBERO doesn't paraphrase, by design, so the benchmark measures execution rather than language robustness. Robustness to paraphrasing gets studied separately in Chapter 15.

## Formatting the prompt

OpenVLA was fine-tuned with a specific instruction template, and inference has to use it verbatim:

```
In: What action should the robot take to {instruction}?
Out:
```

Mismatch the template and the model still produces seven action tokens (the output layer can't do otherwise), but the tokens come from a distribution the model was never trained on. The symptom is action sequences that hover near zero or saturate at the gripper extremes regardless of the scene. This is the second of the three silent failures in §2.4. The fix is mechanical: copy the template from the OpenVLA model card and don't edit it. Casing matters less than spacing; the colon and newline matter most.

## The processor call

`processor(prompt, image)` does three things. It tokenizes the prompt with the Llama-2 tokenizer into a `(1, T)` long tensor of token IDs, with `T` typically 20 to 25. It runs the image through both SigLIP and DINOv2 image processors (resize to 224×224, normalize with encoder-specific means and standard deviations, stack into a `(1, 2, 3, 224, 224)` tensor of pixel values). And it packages everything into a dict whose keys match the model's `forward` signature: `input_ids`, `attention_mask`, `pixel_values`. The `.to("cuda", dtype=torch.bfloat16)` call moves the tensors to the GPU and casts the floating-point ones; long tensors (token IDs) get silently left alone, which is exactly what you want.

A useful debugging habit at this point: print `inputs.pixel_values.mean()` and `inputs.input_ids.shape` once per episode. If the mean comes out exactly zero, the image was black (the headless-rendering failure from §2.2). If the input shape is unexpectedly short, the instruction string was empty, which happens when `task.language` gets read before `env.reset()` on some LIBERO versions.

## The forward pass: `predict_action`

`model.predict_action(**inputs, unnorm_key="libero_object", do_sample=False)` is a thin wrapper around `model.generate` followed by an action detokenizer. Inside, three things happen in sequence.

The model generates exactly seven new tokens by greedy decoding. With `do_sample=False`, each token is the argmax of the next-token logits over the full Llama-2 vocabulary of 32,064 entries. Training pushes those argmaxes into the bottom 256 entries, the action bins, but nothing in the architecture forces that choice. If training has drifted, the model can emit a non-action token, and `predict_action` will raise. On the pinned checkpoint, in practice, this doesn't happen.

Those seven token IDs get mapped to bin indices in `[0, 255]` by subtracting the vocabulary base. Bin indices then map to continuous values using the per-embodiment statistics in `dataset_statistics.json`, keyed by `unnorm_key`. The statistics for `"libero_object"` give a `q01` (1st percentile) and `q99` (99th percentile) per axis from the training-data distribution; bin `k` maps to `q01 + (k / 255) * (q99 - q01)`. The output is a `(7,)` numpy array of floats: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.

The gripper bit is the seventh axis and is conceptually binary, but the detokenizer returns a continuous value, since the same code path serves embodiments with proportional grippers. LIBERO's Panda accepts the continuous value and thresholds internally: values above roughly 0.5 close the gripper, values below open it. Quantize at the boundary or leave it continuous, both work fine, though rounding to 0.5 makes debugging logs more readable.

The third silent failure lives here, in `unnorm_key`. Pass `"libero_object"` and your actions get scaled to the LIBERO Panda's training-time motion envelope. Pass `"bridge_orig"` instead (a Bridge V2 embodiment, present in Open X-Embodiment, O'Neill et al., 2023, arXiv:2310.08864), and your actions get scaled to a WidowX 250, typically 2 to 5 times too large for LIBERO's Panda controller. The model is doing the right thing in token space; the rescaling is wrong, and the robot flails. Always print `unnorm_key` once at the top of the loop, and again into the wandb run name if you're logging.

## Stepping the simulator

`env.step(action.tolist())` advances LIBERO by one control cycle. The default control rate is 20 Hz, but RoboSuite frame-skips internally, so a single `env.step` actually advances MuJoCo by several physics substeps (typically five at 100 Hz physics). The returned `obs` is the next agent-view image. `reward` is sparse, typically zero until task success, when it becomes 1.0. `done` is `True` when the task succeeds or the episode exceeds the suite's horizon (300 steps for `libero_object`, 600 for `libero_10`). `info` carries task-specific diagnostics.

One subtlety: LIBERO's reward function is a completion check, not a shaped signal. The agent gets no partial credit for being close. This matters for evaluation in §2.4 and for training in Chapter 16, where the standard metric is success rate over many seeds, not mean reward. Sparse rewards also mean the only way to know an episode is going wrong before timeout is to watch the rendered video or instrument the loop with object-pose probes.

## Throughput and the latency budget

On a 4090, the loop above runs at roughly 6 to 8 Hz: about 120 ms per `predict_action` call, plus about 30 ms for the LIBERO step. That's fast enough for tabletop manipulation in simulation, but slower than a teleoperator's natural rate, often 20 to 30 Hz. The dominant cost is the 7-billion-parameter transformer; the simulator is essentially free by comparison. Chapter 13 returns to this. π0 (Black et al., 2024, arXiv:2410.24164) and other flow-matching VLAs achieve higher effective control rates by predicting action chunks rather than single steps, and by running smaller policy networks behind a larger language backbone. For OpenVLA on a consumer GPU, the rate above is simply the rate.

## A 5-line debug print to add right now

Before §2.4 makes you watch the loop fail, instrument it. The lines below cost nothing and make every failure mode in §2.4 obvious:

```python
print(f"step={step:03d} "
      f"img_mean={obs['agentview_image'].mean():.1f} "
      f"action={np.round(action, 4).tolist()} "
      f"reward={reward:.1f}")
```

If `img_mean` is ever near zero, the simulator broke. If `action` saturates at the bin edges every step, the prompt template is wrong. If `action` looks plausible but the robot drifts away from the scene, `unnorm_key` is wrong. And if everything looks plausible while `reward` is still zero at step 300, the model is doing exactly what it was trained to do, it's just that the training data never included this task.

With the inference loop running and instrumented, the next step is to make it fail on purpose, three times, and read the failures as diagnostic signal. §2.4 shows you how.
