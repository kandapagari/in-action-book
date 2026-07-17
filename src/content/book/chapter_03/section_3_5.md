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

"It's not converging" is the most common thing a researcher says after spending a week setting up a training run, and it covers at least four distinct situations: the loss decreasing but too slowly, the loss stuck at a plateau, the loss oscillating or spiking, or the loss going NaN outright. Each has a different cause and a different fix, and lumping them together wastes days chasing the wrong hypothesis. This section gives a structured way to tell them apart, organized around the loss curve as the primary diagnostic instrument, with secondary checks on gradients, data, and loss-family-specific failure modes.

The SmallPolicy from §3.3 and the three loss families from §3.4 serve as running examples throughout. Where behavior differs between a 50-line MLP and a 7B-parameter VLA, that gets noted explicitly.

## Read the loss curve first

Before touching hyperparameters, look at the loss curve over training steps. Log it to TensorBoard, Weights & Biases, or even a CSV, but log it every N steps, not just at epoch end. The shape of the curve is the fastest way to eliminate hypotheses.

Loss decreasing steadily but not reaching a useful value is usually a capacity problem or a data problem, not a training problem. The optimizer is doing its job; there just isn't enough signal in the data or enough model capacity to reach a useful optimum. Check whether your model is actually large enough for the task's complexity, and whether your demonstrations are consistent or whether different demonstrators do the same task in incompatible ways. Inconsistent demonstrations can hold any model below a certain loss floor, since there's no single answer for it to learn.

Loss plateauing early, then flat, means the optimizer found a local minimum and stopped moving. Common causes: learning rate too small (gradient steps are so tiny the optimization never escapes the initialization basin), weight decay too large (the regularizer overpowers the loss gradient), or gradient vanishing (covered below). Try increasing the learning rate by one order of magnitude, re-run for a few hundred steps, and watch whether the loss breaks out of the plateau. If it does, you have a learning rate problem, not a capacity problem.

Loss oscillating without converging, going down, then up, then down again in a cycle, is almost always a learning rate that's too large. The optimizer overshoots the minimum on every step. Halve the learning rate and check again. If oscillation persists at a lower rate, check whether your loss function has a discontinuity or a very narrow valley; this can happen with binned cross-entropy if the action-bin boundaries sit poorly relative to the actual action distribution.

Loss spiking intermittently, in a training curve that's otherwise well-behaved, is a gradient explosion signature. A single bad batch, one with a large action outlier or a numerical instability from a very deep network, produces a huge gradient that shifts the weights far from the current minimum and destroys progress. Add gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before the optimizer step. OpenVLA (arXiv:2406.09246) uses a max-norm clip of 1.0 as a default for exactly this reason. If spikes persist after clipping, look at the data: an action outlier orders of magnitude larger than the median action will produce a proportionally large MSE gradient. Outlier filtering or robust normalization (covered below) is the fix.

Loss going NaN immediately or within a few steps means something blew up before training meaningfully started. Check these in order. Is there a `log(0)` or `0/0` in the loss? Cross-entropy on a one-hot target that the model assigns exactly zero probability produces `-log(0) = ∞`. PyTorch's internal `log_softmax` should catch this, but numerical precision failures in mixed-precision (fp16 or bf16) training can produce genuine zeros in the softmax output; switch to full fp32 to diagnose, then add epsilon to probabilities if needed. Are your input features normalized? NaN on step 1 is almost always a normalization failure: the network receives inputs of magnitude 10³ or 10⁴, the first linear layer produces similarly large pre-activations, and the sigmoid or softmax saturates to exactly 0 or 1 in float16. Normalize all inputs to approximately zero mean and unit variance before the first layer.

## Gradient inspection

Once you've characterized the loss shape, the next diagnostic is gradient magnitude. Add this to your training loop:

```python
for name, param in policy.named_parameters():
    if param.grad is not None:
        writer.add_scalar(
            f"grad_norm/{name}",
            param.grad.norm().item(),
            global_step
        )
```

A well-behaved run shows gradient norms in the range `[1e-4, 1e0]` across most layers, roughly uniform depth-to-depth. Two pathologies look very different from that baseline.

Vanishing gradients: gradient norms near the input end of the network run orders of magnitude smaller than those near the output. The chain-rule product (§3.1) is being multiplied down to near-zero by near-zero Jacobians deep in the network. Solutions: switch from sigmoid or tanh activations to ReLU or GELU, add residual connections (which provide a direct gradient path bypassing deep multiplicative chains), and use layer normalization before each attention or MLP block. Transformer-based VLAs like RT-1 and OpenVLA resist vanishing gradients architecturally, since they have residual connections everywhere, but an MLP policy without residuals can suffer badly in networks deeper than four or five layers.

Exploding gradients: gradient norms spike to `1e3` or `1e6` in certain layers. Gradient clipping controls the symptom, but the root cause is usually one of a very large learning rate, a badly initialized weight matrix (random initialization producing singular values much larger than 1), or a loss function that isn't Lipschitz near the current weights. For flow-matching losses (π0, arXiv:2410.24164), check that the time variable `t` is sampled from `[0, 1]` with appropriate clamping. A `t` very close to 0 causes the velocity target `(a* - ε)` to get evaluated at nearly pure noise, which can produce extreme gradient magnitudes.

## Data diagnostics

The training loop, gradients, and loss function are the plumbing. The demonstrations are the signal, and getting the signal wrong can mimic every failure mode above; it's frequently the actual cause.

Check action statistics before anything else. Print or plot the mean, standard deviation, minimum, and maximum of each action dimension across the full training set. A joint range of `[-3.14, 3.14]` radians with a few teleoperator slips producing actions of `±50` means those outliers will dominate the MSE gradient and make everything else irrelevant. Two action dimensions with standard deviations of `0.01` and `2.0` means an MSE loss effectively ignores the first, since it contributes negligibly to the scalar loss. The standard fix is per-dimension normalization: subtract the per-dimension mean and divide by the per-dimension standard deviation across the training set. Store these statistics; you'll need them at inference to denormalize the output.

For cross-entropy losses, check bin coverage. If you're using discretized action bins (as in RT-1, OpenVLA), check how many of the 256 bins actually receive training examples in each action dimension. If 90% of demonstrated actions fall in bins 100 through 130, the model quickly learns to assign zero probability to bins 0 through 99 and 131 through 255 and never revisits them. This is fine for accuracy but produces overconfident predictions and poor generalization. Setting bin boundaries from the action statistics so each bin covers an equal fraction of the training distribution (computing percentile boundaries rather than linear spacing) helps substantially; this is what OpenVLA's preprocessing pipeline does.

For supervised loss, check for label noise. A common form in robot demonstrations is late labeling, where the teleoperator's hand was already moving when the timestamp got assigned, so the recorded action corresponds to a slightly later observation frame. Add a constant time-shift correction and verify action timestamps align with observation timestamps within the latency of your recording hardware. A 50 ms misalignment at 10 Hz means half your labels are wrong.

## Loss-family-specific diagnostics

§3.4 argued that the loss family determines which failure mode to suspect first. Here are the failure-mode-specific questions for each.

Supervised loss not converging: is the training set large enough? Behavior cloning is sample-hungry, since it learns a direct mapping from observation to action with no mechanism to generalize. As a rule of thumb, you need at least a few hundred demonstrations to learn a single-task policy in a clean simulator, and orders of magnitude more for cross-task generalization. Fewer than 100 demonstrations, and expect a supervised model to overfit rather than generalize. Also ask whether the demonstrations are consistent: two demonstrators with different preferred grasp orientations produce a bimodal action distribution that an MSE-trained model can't represent. Adding a diffusion or flow-matching head (Chapter 10) is the correct architectural fix; reducing the demonstrator pool is the pragmatic data fix.

RL loss not converging: is the reward dense enough? A sparse reward firing only at episode completion gives the policy no gradient signal for the vast majority of its experience. Shaped rewards, intermediate rewards for partial progress such as reducing end-effector distance to the target, are almost always necessary for manipulation tasks longer than a few steps. Also check reward scale: a reward of `1.0` for success combined with an episode length of 500 steps and discount `γ = 0.99` means the discounted value at step 0 is only `0.99^500 ≈ 0.007`. The policy sees no useful gradient through the discount unless the learning rate is tuned to compensate. Chapter 5 covers RL diagnostics in depth.

Self-supervised (diffusion/flow) loss not converging: is the noise schedule appropriate for your action distribution? A diffusion model trained with a noise schedule calibrated for images (bounded in `[0, 1]`) behaves poorly on robot actions with a standard deviation of `2.0`, since the noise magnitudes are miscalibrated. Set the noise schedule relative to the action statistics, not copied from an image diffusion paper. For flow matching specifically, check that the learned velocity field produces trajectories arriving at the target distribution: sample 10 trajectories from the model at inference and plot the final-step distribution against the training action distribution. A mismatch means the velocity points in the right direction but the step size is wrong, or the ODE integrator is accumulating too much error.

## A checklist

The sequence of checks, in practice:

1. Print action statistics. Verify range, mean, standard deviation per dimension. Fix outliers and normalize.
2. Verify timestamp alignment. Confirm observation-action pairs are correctly synchronized.
3. Run for 10 steps with a tiny batch of 4. Loss should strictly decrease. If it doesn't, there's a fundamental problem with the forward pass or loss computation.
4. Check gradient norms per layer. They should stay within three orders of magnitude of each other. Add clipping if norms exceed `1.0`.
5. Check for NaN. Add `assert not torch.isnan(loss)` and `assert not torch.isinf(loss)` after the loss computation.
6. Read the loss curve shape. Plateau points to learning rate. Oscillation points to a learning rate that's too high. Spike points to exploding gradients or a data outlier. Steadily decreasing but stuck at the wrong floor points to capacity or data consistency.
7. Apply the loss-family diagnostic. Supervised points to data size and label noise. RL points to reward density and scale. Self-supervised points to noise schedule calibration.

None of this is specific to robotics; it's standard deep learning practice. What makes robot learning unusual is that the feedback loop between training and evaluation runs long and expensive. Running 100 real-robot trials to evaluate whether a training change helped takes days. That cost makes it worth investing heavily in the simulator and logging infrastructure described in Chapter 2 and Chapter 15, so you can evaluate cheaply before committing hardware time.

A model that refuses to train is telling you something specific. Read the loss curve, check the gradients, inspect the data, apply the loss-family checklist in that order, and you'll usually identify the cause before reaching for a more exotic hypothesis.

With the prerequisite tools in place, the calculus, the probabilistic framing, the training loop, the loss families, and the debugging practice, we're ready to close Part 1 and summarize what the first three chapters have established.
