---
chapter: 3
section: 3.5
title: Debugging a model that will not train
target_words: 2000
status: draft
prereqs: §3.1 (gradients, chain rule, vanishing/exploding), §3.2 (cross-entropy, KL divergence), §3.3 (PyTorch training loop), §3.4 (three loss families and their diagnostic signatures)
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 3.5  Debugging a model that will not train

"It's not converging" is the most common thing a researcher says after spending a
week setting up a training run. It covers at least four distinct situations:
the loss is decreasing but too slowly; the loss is stuck at a plateau; the loss
is oscillating or spiking; or the loss is NaN. Each has a different cause and a
different fix, and lumping them together wastes days on the wrong hypothesis.
This section gives a structured way to tell them apart, organized around the loss
curve as a primary diagnostic instrument, with secondary checks on gradients,
data, and loss-family-specific failure modes.

The SmallPolicy from §3.3 and the three loss families from §3.4 are the running
examples. Where behavior differs between a 50-line MLP and a 7B-parameter VLA,
that is noted explicitly.

## Read the loss curve first

Before touching hyperparameters, look at the loss curve over training steps.
Log it to TensorBoard, Weights & Biases, or even a CSV — but log it, every N
steps, not just at epoch end. The shape of the curve is the fastest way to
eliminate hypotheses.

**Loss decreasing steadily, but not reaching a useful value.** This is usually
a capacity problem or a data problem, not a training problem. The optimizer is
doing its job; there is simply not enough signal in the data or not enough model
capacity to reach a useful optimum. Check: is your model actually large enough
for the task complexity? Are your demonstrations consistent, or do different
demonstrators do the same task in incompatible ways? Inconsistent demonstrations
can prevent any model from fitting below a certain loss floor — there is no single
answer to learn.

**Loss plateauing early, then flat.** The optimizer found a local minimum early
and stopped moving. Common causes: learning rate too small (the gradient steps
are so tiny the optimization never escapes the initialization basin), weight
decay too large (the regularizer overpowers the loss gradient), or gradient
vanishing (covered below). Try increasing the learning rate by one order of
magnitude, re-run for a few hundred steps, and watch whether the loss breaks
out of the plateau. If it does, you have a learning rate problem, not a capacity
problem.

**Loss oscillating without converging.** The loss goes down, then up, then down
again, cycling. Almost always: learning rate too large. The optimizer is
overshooting the minimum on every step. Halve the learning rate and check again.
If the oscillation persists at a lower learning rate, check whether your loss
function has a discontinuity or a very narrow valley — this can happen with
binned cross-entropy if the action-bin boundaries are poorly chosen relative to
the action distribution.

**Loss spiking intermittently.** A training curve that is mostly well-behaved but
has periodic large spikes is a gradient explosion signature. A single bad batch —
one with a large action outlier, or a numerical instability introduced by a very
deep network — produces a huge gradient that shifts the weights far from the
current minimum, destroying progress. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before the optimizer step.
OpenVLA (arXiv:2406.09246) uses a max-norm clip of 1.0 as a default for exactly
this reason. If spikes persist after clipping, look at the data: an action
outlier that is orders of magnitude larger than the median action will produce
a proportionally large MSE gradient. Outlier filtering or robust normalization
(see below) is the fix.

**Loss is NaN immediately or within a few steps.** Something has blown up
before training has meaningfully started. Check these in order: (1) Is there a
`log(0)` or `0/0` in the loss? Cross-entropy on a one-hot target that the
model assigns exactly zero probability produces `-log(0) = ∞`. This should be
caught by PyTorch's internal `log_softmax`, but numerical precision failures in
mixed-precision (fp16 or bf16) training can produce genuine zeros in the softmax
output. Switch to full fp32 to diagnose, then add epsilon to probabilities if
needed. (2) Are your input features normalized? NaN on step 1 is almost always a
normalization failure: the network receives inputs of magnitude 10³ or 10⁴, the
first linear layer produces similarly large pre-activations, and the sigmoid or
softmax saturates to exactly 0 or 1 in float16. Normalize all inputs to
approximately zero mean and unit variance before the first layer.

## Gradient inspection

Once you have characterized the loss shape, the next diagnostic is the gradient
magnitude. Add this to your training loop:

```python
for name, param in policy.named_parameters():
    if param.grad is not None:
        writer.add_scalar(
            f"grad_norm/{name}",
            param.grad.norm().item(),
            global_step
        )
```

A well-behaved run shows gradient norms in the range `[1e-4, 1e0]` across most
layers, roughly uniform depth-to-depth. Two pathologies look very different.

**Vanishing gradients:** gradient norms near the input end of the network are
orders of magnitude smaller than those near the output. The chain-rule product
(§3.1) is being multiplied to near-zero by near-zero Jacobians deep in the
network. Solutions: switch from sigmoid or tanh activations to ReLU or GELU;
add residual connections (which provide a direct gradient path that bypasses
deep multiplicative chains); use layer normalization before each attention or
MLP block. Transformer-based VLAs like RT-1 and OpenVLA are architecturally
resistant to vanishing gradients because they have residual connections
everywhere, but an MLP policy without residuals can suffer severely in networks
deeper than four or five layers.

**Exploding gradients:** gradient norms spike to `1e3` or `1e6` in certain
layers. Gradient clipping controls the symptom, but the root cause is usually
one of: a very large learning rate, a badly initialized weight matrix (random
initialization producing singular values much larger than 1), or a loss function
that is not Lipschitz near the current weights. For flow-matching losses
(π0, arXiv:2410.24164), check that the time variable `t` is being sampled from
`[0, 1]` with appropriate clamping — a `t` very close to 0 causes the velocity
target `(a* - ε)` to be evaluated at nearly pure noise, which can produce extreme
gradient magnitudes.

## Data diagnostics

The training loop, gradients, and loss function are the plumbing. The
demonstrations are the signal. Getting the signal wrong can mimic every failure
mode above, and it is frequently the actual cause.

**Check action statistics before anything else.** Print or plot the mean,
standard deviation, minimum, and maximum of each action dimension across the
full training set. If a joint range is `[-3.14, 3.14]` radians but a few
teleoperator slips produced actions of `±50`, those outliers will dominate the
MSE gradient and make everything else irrelevant. If two action dimensions have
standard deviations of `0.01` and `2.0`, an MSE loss will effectively ignore
the first dimension — it contributes negligibly to the scalar loss. The standard
fix is per-dimension normalization: subtract the per-dimension mean and divide by
the per-dimension standard deviation across the training set. Store these
statistics; you will need them at inference to denormalize the output.

**For cross-entropy losses: check bin coverage.** If you are using discretized
action bins (as in RT-1, OpenVLA), check how many of the 256 bins actually
receive any training examples in each action dimension. If 90 % of demonstrated
actions fall in bins 100–130, the model will quickly learn to assign zero
probability to bins 0–99 and 131–255 and never revisit them. This is fine for
accuracy but produces overconfident predictions and poor generalization. Using
the action statistics to set bin boundaries so that each bin covers an equal
fraction of the training distribution (i.e., computing the percentile boundaries
rather than linear spacing) substantially helps; this is what OpenVLA's
preprocessing pipeline does.

**For supervised loss: check for label noise.** A common form of label noise in
robot demonstrations is late labeling — the teleoperator's hand was already moving
when the timestamp was assigned, so the recorded action corresponds to a slightly
later observation frame. Add a constant time-shift correction and verify that the
action timestamps align with the observation timestamps within the latency of your
recording hardware. A 50 ms misalignment at 10 Hz means 50 % of your labels are
wrong.

## Loss-family-specific diagnostics

§3.4 argued that the loss family determines which failure mode to suspect first.
Here are the failure-mode-specific questions.

**Supervised loss not converging:** is the training set large enough? Behavior
cloning is sample-hungry because it learns a direct mapping from observation to
action without any mechanism to generalize. As a rule of thumb, you need at
least a few hundred demonstrations to learn a single-task policy in a clean
simulator, and orders of magnitude more for cross-task generalization. If you
have under 100 demonstrations, expect a supervised model to overfit rather than
generalize. Also ask: are the demonstrations consistent? Two demonstrators with
different preferred grasp orientations will produce a bimodal action distribution
that an MSE-trained model cannot represent. Adding a diffusion or flow-matching
head (Chapter 10) is the correct architectural fix; reducing the demonstrator
pool is the pragmatic data fix.

**RL loss not converging:** is the reward dense enough? A sparse reward that
fires only at episode completion gives the policy no gradient signal for the
vast majority of its experience. Shaped rewards — intermediate rewards for
partial progress, such as reducing end-effector distance to the target — are
almost always necessary for any manipulation task longer than a few steps. Also
check reward scale: a reward of `1.0` for success combined with an episode length
of 500 steps and discount `γ = 0.99` means the discounted value at step 0 is
only `0.99^500 ≈ 0.007`. The policy sees no useful gradient through the discount
unless the learning rate is tuned to compensate. Chapter 5 covers RL diagnostics
in depth.

**Self-supervised (diffusion/flow) loss not converging:** is the noise schedule
appropriate for your action distribution? A diffusion model trained with a noise
schedule calibrated for images (which are bounded in `[0, 1]`) will behave poorly
on robot actions with a standard deviation of `2.0` — the noise magnitudes are
miscalibrated. The noise schedule should be set relative to the action statistics,
not copied from an image diffusion paper. For flow matching specifically, check
that the learned velocity field produces trajectories that actually arrive at the
target distribution: sample 10 trajectories from the model at inference and plot
the final-step distribution against the training action distribution. If they do
not match, the velocity is pointing in the right direction but the step size is
wrong, or the ODE integrator is accumulating too much error.

## A checklist

The sequence of checks in practice:

1. **Print action statistics.** Verify range, mean, standard deviation per
   dimension. Fix outliers and normalize.
2. **Verify timestamp alignment.** Confirm observation-action pairs are correctly
   synchronized.
3. **Run for 10 steps with a tiny batch of 4.** Loss should strictly decrease.
   If it does not, there is a fundamental problem with the forward pass or loss
   computation.
4. **Check gradient norms per layer.** They should be within three orders of
   magnitude of each other. Add clipping if norms exceed `1.0`.
5. **Check for NaN.** Add `assert not torch.isnan(loss)` and
   `assert not torch.isinf(loss)` after the loss computation.
6. **Read the loss curve shape.** Plateau → learning rate. Oscillation → learning
   rate too high. Spike → exploding gradients or data outlier. Steadily
   decreasing but wrong floor → capacity or data consistency.
7. **Apply the loss-family diagnostic.** Supervised → data size and label noise.
   RL → reward density and scale. Self-supervised → noise schedule calibration.

None of this is specific to robotics. It is standard deep learning practice.
What makes robot learning unusual is that the feedback loop between training and
evaluation is long and expensive — running 100 real-robot trials to evaluate
whether a training change helped takes days. That cost makes it worth investing
heavily in the simulator and logging infrastructure described in Chapter 2 and
Chapter 15, so that you can evaluate cheaply before committing hardware time.

A model that refuses to train is telling you something specific. Read the loss
curve, check the gradients, inspect the data, apply the loss-family checklist in
that order — and you will usually identify the cause before reaching for a more
exotic hypothesis.

With the prerequisite tools in place — the calculus, the probabilistic framing,
the training loop, the loss families, and the debugging practice — we are ready
to close Part 1 and summarize what the first three chapters have established.
