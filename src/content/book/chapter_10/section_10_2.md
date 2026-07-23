---
chapter: 10
section: 10.2
title: "Diffusion Policy and ACT"
target_words: 2000
status: draft
prereqs: §10.1 (diffusion mechanism, conditioning), §6.2–§6.3 (behavior cloning, compounding error and DAgger), §8.1 (transformer for control). Helpful, §3.2 for the KL term that reappears in ACT's CVAE.
key_refs:
  - Chi, C. et al. (2023). Diffusion Policy — Visuomotor Policy Learning via Action Diffusion. RSS 2023.
  - Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT). RSS 2023.
---

# 10.2  Diffusion Policy and ACT

At the end of §10.1 the sample $x_0$ was still an abstraction, some vector the denoiser learns to produce. This section fixes what that vector is. In both systems below, $x_0$ is a short stretch of future robot actions: a sequence of end-effector poses or joint targets, maybe sixteen of them, stacked into one object the model generates in a single shot. That one design decision, predict a chunk of the future, not the next step, is what these two papers share, and it turns out to matter more than the generative machinery bolted on top of it.

Diffusion Policy (Chi et al., 2023) and ACT (Zhao et al., 2023) landed within weeks of each other in 2023 and became the two reference points for modern imitation learning on real hardware. They reach similar places by different roads: one diffuses the action chunk, the other draws it from a conditional VAE. Reading them side by side is the fastest way to see which parts of the recipe are essential and which are interchangeable.

## Diffusion Policy: the head from §10.1, pointed at actions

Take the conditional denoiser from the end of §10.1 and make three concrete commitments. First, the thing being generated is not one action but a horizon of them, Diffusion Policy predicts a block of $T_p$ future actions at once, typically around sixteen. Second, the condition $o$ is the recent observation history: the last two camera frames, encoded by a vision backbone, plus proprioception. Third, the denoiser is a network sized for sequences, the paper offers both a 1-D convolutional U-Net over the time axis and a transformer variant, and reports the U-Net as the more forgiving default.

Training is exactly the loss from §10.1, no additions. Sample a chunk of expert actions from the demonstration set, noise it to a random level $t$, and regress the noise given the observation:

$$
\mathcal{L} = \mathbb{E}_{\,\mathbf{a}_0,\,t,\,\epsilon}
\left[\ \big\|\ \epsilon - \epsilon_\theta(\mathbf{a}_t,\ t,\ o)\ \big\|^2\ \right],
$$

where $\mathbf{a}_0$ is now the clean action chunk and $\mathbf{a}_t$ its noised version. Nothing here knows or cares that the target is a trajectory rather than a picture. That is the point of §10.1's second observation paying off: the recipe is indifferent to what $x_0$ means.

Two smaller choices in the paper are worth flagging because they travel with the method. The observation conditions the denoiser through FiLM, feature-wise linear modulation, where the encoded observation produces per-channel scale and shift parameters that reweight the denoiser's activations, rather than being concatenated to the input. And the action space matters: Diffusion Policy reports that predicting end-effector *positions* works better than predicting velocities, because position targets keep the generated chunk anchored to where the gripper should be rather than accumulating drift through integration. Neither point is exotic, but both are the kind of detail that decides whether a reimplementation trains at all.

The interesting engineering is at deployment, and it goes by **receding-horizon control**. The policy predicts $T_p$ actions but you do not execute all of them. You run the first $T_a$, say the first eight of sixteen, then throw the rest away, take a fresh observation, and generate a new chunk. Predicting a long horizon keeps the motion committed and smooth; executing only a prefix keeps the robot reactive to whatever changed while it was moving. The gap between $T_p$ and $T_a$ is a knob: predict far, act briefly, replan often.

### Why the chunk fixes what §10.1 promised

Two failure modes from earlier chapters dissolve here, and it is worth naming both.

The first is multimodality, the mug-and-laptop problem from §10.1. On the PushT benchmark, a 2-D task where the robot must nudge a T-shaped block into a target pose using a circular pusher, there are often two equally good ways to approach the block, from one side or the other. A network that regresses the mean action steers straight down the middle and stalls against the block's flat edge. Diffusion Policy, because it samples from a distribution with mass on both approaches, commits to one side and completes the push. PushT is deliberately small and deliberately bimodal; it exists to make this exact difference visible, and Diffusion Policy's jump in success rate on it over mean-regression baselines is the headline result of the paper.

The second is compounding error, the reason §6.3 had to introduce DAgger. A single-step policy that drifts slightly off the demonstrated states keeps drifting, because each small error moves it toward states it never trained on. Predicting a coherent chunk of actions at once does not eliminate the problem, but it dampens it: the model reasons over a window rather than reacting frame by frame, so short-term noise averages out inside the chunk instead of accumulating across steps. Diffusion Policy buys some of DAgger's robustness without DAgger's online data collection, which is a large part of why practitioners reached for it.

The cost is the one §10.1 flagged. Each action chunk requires running the denoiser many times, and the original DDPM sampler wanted a hundred-plus steps. Diffusion Policy leans on DDIM-style samplers to cut that to roughly ten network calls per chunk, and receding-horizon execution means you only pay that cost once every $T_a$ steps rather than every step. Even so, the latency question does not go away, and §10.4 is where we account for it honestly.

## ACT: same chunk, a different generator

ACT, Action Chunking with Transformers, arrives at the chunk from the other direction. Zhao et al. (2023) built it as the learning half of ALOHA, a low-cost bimanual teleoperation rig assembled from two pairs of leader-follower arms. The hardware is the point of the paper as much as the algorithm: it made fine-grained two-handed tasks, threading a zip tie, slotting a battery into a slot with millimeters of clearance, opening a translucent condiment cup, collectable by an untrained human operator and learnable from a few tens of demonstrations.

ACT keeps the action chunk but replaces the diffusion head with a **conditional variational autoencoder** wrapped around a transformer. The structure is an encoder-decoder transformer, the same shape as §8.1, that maps the current observation to a chunk of $k$ future actions in one forward pass, no iterative denoising. The "variational" part is a small latent variable $z$: at training time an encoder compresses the demonstrated action chunk into $z$, and the decoder reconstructs the chunk from $z$ plus the observation. The training loss is the standard CVAE objective, a reconstruction term plus the KL penalty on $z$ you met in §3.2:

$$
\mathcal{L}_{\text{ACT}} = \underbrace{\|\mathbf{a} - \hat{\mathbf{a}}\|_1}_{\text{reconstruction}}
\;+\; \beta \underbrace{D_{\mathrm{KL}}\!\left(q(z\mid \mathbf{a}, o)\ \|\ \mathcal{N}(0,I)\right)}_{\text{latent regularizer}}.
$$

At deployment the encoder is discarded. You sample $z$ from the prior, in practice ACT often just sets $z=0$, its mean, feed it with the observation to the decoder, and read off a chunk of actions in a single pass. Where Diffusion Policy handles multimodality by sampling a noisy denoising trajectory, ACT parks it in $z$: different draws of the latent correspond to different coherent ways of doing the task. In practice ACT uses the latent lightly and gets most of its mileage from chunking itself.

ACT's counterpart to receding-horizon control is **temporal ensembling**. Rather than execute a prefix and discard the rest, ACT queries the policy at every timestep, producing overlapping chunks, and averages the multiple predictions that now exist for each moment in time (weighted so recent predictions count more). The overlapping votes smooth the executed trajectory and suppress the jitter a single chunk boundary can introduce. It is a different answer to the same question Diffusion Policy answers with receding horizons: how do you turn a stream of overlapping chunk predictions into one smooth stream of commands.

## Reading them side by side

Strip away the branding and the two systems line up cleanly.

Both predict a chunk of future actions rather than a single next action, and both treat that as the load-bearing idea. The chunk is what gives them smooth motion and partial immunity to compounding error, and it is the design choice that survived into every VLA action head that came after; you will see it again in Octo's diffusion head (§12.3) and in π0's flow-matching head (Chapter 13).

Where they differ is the generator strapped to the chunk. Diffusion Policy denoises, paying for expressiveness with a handful of sequential network calls. ACT decodes a CVAE in one pass, paying for that speed with a generator that captures multimodality less sharply, a Gaussian latent is a blunter instrument than a full denoising chain. Neither choice is universally right, which is precisely the trade-off §10.4 lays out: how much multimodality does your task actually need, and how many milliseconds can your control loop spare to get it.

A concrete way to feel the difference: on a task with one clearly correct motion and tight timing, inserting a peg where there is only one sane approach angle, ACT's single-pass speed is close to free and its weaker multimodality costs nothing, because there was only one mode to capture. On PushT, where two approaches genuinely compete, Diffusion Policy's sharper distribution earns its extra compute. Matching the head to the task is the skill; the two papers are the anchor examples at opposite ends of the spectrum.

One historical note worth carrying forward. ACT is not, strictly, a diffusion model, it belongs in this chapter because it solves the same problem with the same chunking insight, and because the field consistently benchmarks the two against each other. When someone says "diffusion versus ACT," they mean "iterative generative head versus single-pass generative head," and the real subject of the comparison is the chunk they have in common.

Both systems still generate their chunk from scratch each time, whether by ten denoising steps or one CVAE pass. The next section asks whether the ten steps can be collapsed toward one without giving up the distribution, which is the question flow matching was built to answer.
