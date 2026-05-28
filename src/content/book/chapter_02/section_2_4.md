---
chapter: 2
section: 2.4
title: When it works and when it does not
target_words: 2000
status: draft
prereqs: §2.3 (the inference loop with `predict_action`, `unnorm_key`, and the `agentview_image` flip); willingness to break the loop in three specific ways and read the resulting behavior; LIBERO suite vocabulary (`libero_object`, `libero_spatial`, `libero_goal`, `libero_10`)
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
---

# 2.4  When it works and when it does not

The loop in §2.3 runs. That is not the same as working. A loop that runs is a
loop that completes 400 steps without raising; a loop that works is a loop
that places the alphabet soup in the basket. The gap between the two is
where almost everything interesting in deploying a VLA happens, and it is
worth spending a section inside that gap before we leave Chapter 2. We will
do this in four passes. First, the success case — what a clean run looks
like, on the task the model was demonstrably trained on. Second, the three
silent failures that §2.3 trailed: upside-down images, wrong prompt
template, wrong `unnorm_key`. Third, the broader failure taxonomy you will
encounter the moment you stray from `libero_object` task 0. Fourth, a brief
treatment of why a single rollout tells you almost nothing and what to
measure instead.

## What success looks like

Run the §2.3 script unchanged on `libero_object`, task index 0, with a fixed
seed. On a single GeForce 4090 the episode takes about 50 to 70 seconds of
wall clock. The instruction is `"pick up the alphabet soup and place it in
the basket"`. The Panda arm moves down and slightly forward, the fingers
close on the soup can after roughly 60 to 90 control steps, the arm lifts,
translates above the basket, and releases. `reward` jumps from `0.0` to
`1.0` at around step 180 to 220, `done` becomes `True`, the loop breaks.
This sequence happens on roughly nine out of ten seeds — Kim et al. (2024,
arXiv:2406.09246) report 88 to 90% success on `libero_object` for the
released OpenVLA-7B checkpoint after their LIBERO-specific fine-tune, and
that number reproduces within a couple of percentage points on a clean
install.

There are two things to notice about that success run, both of which look
unremarkable until they stop happening. The first is that the gripper
closes once. It does not chatter — it does not open and close every other
step. A jittery gripper means the seventh-axis bin is sitting right at the
0.5 threshold and noise in the policy is flipping it; that is a sign of
distribution mismatch, not of a working policy. The second is that the
arm's motion is smooth across timesteps even though each step is decoded
independently. OpenVLA does not predict an action chunk; it predicts one
7-vector per inference call. The visual continuity of the resulting
trajectory is an emergent consequence of the model having seen smooth
demonstration trajectories during pretraining, not of any explicit
smoothness loss. When a VLA's trajectory looks jagged, that is the second
distribution-mismatch tell.

## The three silent failures

A failure is *silent* when the loop continues to run, the tensors continue
to have the right shapes, and no exception is raised — but the robot is no
longer doing the task. The script in §2.3 has exactly three of these built
in, and every one of them is a one-character or one-keyword change.

**The image flip.** Remove the `[::-1]` from
`obs["agentview_image"][::-1]`. The MuJoCo OpenGL convention is that pixel
row 0 is the bottom of the image; OpenVLA's vision encoders (SigLIP and
DINOv2) were pretrained on web images where row 0 is the top. Without the
flip, the model sees a world where the table is on the ceiling. The actions
it produces are still syntactically valid 7-vectors, the Panda still
executes them, but the arm drifts away from the can and either swipes
through it or stops at a height that has nothing to do with the geometry.
Success rate on `libero_object` task 0 drops from ~90% to single digits.
The diagnostic, if you instrumented the loop with the five-line print from
§2.3, is that `img_mean` is fine (the rendering works), but the rendered
trajectory makes no sense. If you also dump the agent-view PNG once per
episode, the flip is obvious to the eye.

**The prompt template.** Change the prompt from `"In: What action should
the robot take to {instruction.lower()}?\nOut:"` to anything else — even a
seemingly equivalent rephrasing like `"What action should the robot take?
{instruction}"`. The model still emits seven action tokens; the output
layer is unconditional on prompt format. But the model was fine-tuned with
that exact `In:`/`Out:` scaffold (see the OpenVLA repo's `prompt_builder`
and the discussion in arXiv:2406.09246 §3), and the language tower's
attention pattern over the visual tokens depends on it. Without the
scaffold, the seven generated tokens drift toward the action-bin centers
(`128` is the bin that maps to roughly the middle of every per-axis
range), and the resulting commanded actions hover near zero. The arm
twitches and never reaches the object. The diagnostic is that
`np.round(action, 2)` returns a row of values clustered near zero or
clustered at the per-axis extremes, every step, regardless of the scene.

**The `unnorm_key`.** Change `unnorm_key="libero_object"` to
`unnorm_key="bridge_orig"`. Both keys exist in OpenVLA's
`dataset_statistics.json` — `bridge_orig` is the Bridge V2 embodiment from
Open X-Embodiment (O'Neill et al., 2023, arXiv:2310.08864), which used a
WidowX 250 arm with a much larger workspace and faster commanded
velocities than LIBERO's Panda. The model is doing the right thing in
*token space* — it still picks bins that correspond to "move down and
forward" — but the bin-to-meters mapping is wrong by a factor of 3 to 5
on translation axes. The Panda's controller saturates, the arm slams
toward joint limits, and LIBERO either terminates the episode with an
error flag in `info` or the simulation goes unstable. The diagnostic is
that the magnitudes in your action print are obviously too large
(`|dx|` > 0.1 m per step for an embodiment that should be commanding
2–5 mm steps). This failure mode is also a useful object lesson: action
detokenization is *embodiment-specific*, even when the policy backbone is
shared, and the choice of `unnorm_key` is the cleanest example of a
non-policy parameter that determines whether a VLA works. Chapter 11
returns to action tokenization in more depth; for now, treat
`unnorm_key` as a required argument with no safe default.

These three failures are not exhaustive but they are the failures you will
hit first, and each one corresponds to one of the three transformation
boundaries inside the loop: pixels-into-encoder, instruction-into-tokens,
tokens-into-meters. A useful rule for debugging any VLA wrapper is to check
each boundary in turn before suspecting the policy itself.

## When it does not work even with the script right

Now make the script correct again and change the *task*. Switch from
`libero_object` to `libero_10`, the long-horizon suite. Pick task 2, which
involves picking a plate from one fixture, moving it to a stove, and then
returning to retrieve a separate item. Run the same loop. On the released
OpenVLA-7B, success drops from ~90% to roughly 50 to 55% (Kim et al.,
2024). What is going wrong is not the loop; the loop is identical. The
policy is simply less good at sequencing two subtasks than at executing
one. You will watch episodes where the plate is grasped correctly, placed
correctly, and then the arm never returns for the second item — it instead
re-approaches the empty plate location and stalls. This is a *behavioral*
failure of the model itself, not an integration bug, and there is nothing
to fix on the integration side. The fix lives in Chapter 14, where
dual-system architectures (Helix, GR00T N1; arXiv:2503.14734) introduce a
slower high-level planner alongside the fast policy.

Distribution shift is the second category. LIBERO randomizes object
positions on each reset but does not randomize lighting, camera angle, or
distractor objects. The moment you deviate from LIBERO's defaults — add a
second can of soup to the scene, dim the lighting, rotate the camera by
15 degrees — success on the *same* `libero_object` task 0 falls
sharply. OpenVLA's pretraining mix (Open X-Embodiment plus LIBERO
fine-tuning data) does cover a great deal of visual variation, but the
specific combination of LIBERO's renderer, camera, and lighting is what
the LIBERO checkpoint was tuned on. Visual robustness, and how to measure
it, is a Chapter 15 problem.

Language brittleness is the third category. `task.language` for
`libero_object` task 0 is exactly `"pick up the alphabet soup and place it
in the basket"`. Rewrite the instruction in the prompt to `"put the soup
can in the bin"` — semantically identical, lexically different — and
success drops by 10 to 30 percentage points depending on the seed. This is
not a bug in the model; it is a measurement of how much of OpenVLA's
language conditioning is verbatim memorization of training prompts versus
semantic understanding. The numbers are public (Kim et al., 2024, Table 4)
and the gap is large. RT-2 (Brohan et al., 2023, arXiv:2307.15818) and π0
(Black et al., 2024, arXiv:2410.24164) close it partially by relying on
larger language backbones; neither closes it fully.

The fourth category is the one with no satisfying fix at this layer of the
stack: the model can *fail at a task that is in its training distribution*
simply because the per-episode physical configuration is unfavorable. The
soup can spawns leaning against the basket lip; the gripper-can contact
angle is bad; the friction parameters MuJoCo sampled this rollout cause
the can to slip mid-lift. These failures are stochastic at the level of
the simulator. They contribute to the 10% of episodes that fail even on
the easiest LIBERO suite. They are not a sign that anything is wrong.

## Why one rollout does not tell you anything

The corollary of stochastic failure is that a single rollout has almost no
information content. If you run task 0 once and it succeeds, you cannot
distinguish "the model is at 90% success rate" from "the model is at 10%
success rate and you got lucky". The standard evaluation protocol for
LIBERO (and for VLA work generally) is to run 50 episodes per task across
all 10 tasks in a suite, with seeded initial conditions, and report mean
success rate plus standard error. The OpenVLA paper uses 500 episodes per
suite; reproducing within ±2% requires at least 100. On a single 4090,
500 episodes at ~60 seconds each is roughly 8 hours of wall clock per
suite — a useful planning number for the rest of the book and for §2.5,
which is about deciding what to chase and what to skip.

The instrumentation from §2.3 — `step`, `img_mean`, `action`, `reward` —
becomes especially useful when you are doing a 50-episode sweep. Dump it
to a `.jsonl` file per episode; the file is small (a few hundred KB per
episode at 20 Hz), human-readable, and post-hoc analyzable. A useful
descriptive statistic is the *step at first nonzero gripper command*:
across successful episodes it is tightly distributed (gripper closes
between step 60 and 100); across failed episodes it is bimodal (either
never closes, or closes at step 5). That single statistic catches roughly
80% of the failure modes above without watching a single video.

With OpenVLA working on `libero_object`, three named failure modes
debugged, and an honest read on where the model breaks, the remaining
question is what the rest of the book is going to do with this baseline.
§2.5 lays that out.
