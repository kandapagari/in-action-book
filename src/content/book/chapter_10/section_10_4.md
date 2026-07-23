---
chapter: 10
section: 10.4
title: "Trade-offs: latency, multimodality, smoothness"
target_words: 2000
status: draft
prereqs: §10.1 (denoising ladder, multimodality argument), §10.2 (action chunks, receding horizon, temporal ensembling, ACT vs Diffusion Policy), §10.3 (flow matching, reflow, NFE as a dial). Helpful, §6.3 for compounding error and §2.4 for control-loop timing.
key_refs:
  - Chi, C. et al. (2023). Diffusion Policy — Visuomotor Policy Learning via Action Diffusion. RSS 2023.
  - Zhao, T. Z. et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT). RSS 2023.
  - Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 10.4  Trade-offs: latency, multimodality, smoothness

The last three sections handed you three action heads and, each time, punted the hard question to this one. ACT decodes a chunk in one pass but represents multimodality with a blunt Gaussian latent. Diffusion Policy samples a sharp distribution but pays for it in denoising steps. Flow matching straightens the sampling path to buy back most of those steps but still cannot conjure a mode the training data never showed it. Every one of those sentences is a trade. This section names the three axes the trades live on, latency, multimodality, smoothness, shows why you cannot maximize all three at once, and turns that into a way to pick a head for a task instead of picking one by habit.

The reason this deserves its own section rather than a table is that the three axes are coupled. Turn the latency knob and multimodality moves. Ask for more multimodality and smoothness gets harder to guarantee. Nobody ships the corner of the cube where all three are perfect, because that corner does not exist. What you ship is a defensible point in the interior, chosen for the task in front of you.

## Latency: count the network calls

Start with the axis a robot feels most directly. A control loop has a budget, the interval between when a fresh observation arrives and when an action must be on the wire. On the LIBERO-style setups from Chapter 2 that budget is tens of milliseconds; on the 50 Hz dexterous tasks π0 targets it is 20 milliseconds, full stop. Blow the budget and the robot stutters, or worse, acts on stale sensing.

The dominant term in a generative head's latency is the number of function evaluations, how many times you run the network to produce one output. The field abbreviates it NFE, and it is the honest currency for comparing these heads:

- **ACT: 1 NFE.** One forward pass through the CVAE decoder yields the whole chunk. This is the floor; you cannot do fewer than one.
- **Diffusion Policy: roughly 10 NFEs.** DDPM wanted a hundred-plus steps; DDIM-style samplers (§10.2) cut it to about ten without much quality loss. Each step is a full denoiser call.
- **Flow matching: 1 to 10 NFEs.** Naive flow matching lands near diffusion because its marginal field is curved (§10.3). Rectified flow pushes toward a handful of steps, and a well-reflowed model can approach one, π0 sits at a small number rather than one because the last step of straightening costs quality on hard action distributions.

Two multipliers turn NFE into wall-clock time, and both cut in your favor. The first is per-call cost: a call to π0's billion-parameter backbone is not remotely the same price as a call to Diffusion Policy's compact U-Net, so NFE only compares heads of similar size, across sizes you multiply by the per-call latency. The second is the one §10.2 already banked: **receding-horizon execution amortizes the generation cost over a whole chunk.** If you generate sixteen actions and execute eight before replanning, you pay the head's latency once every eight control ticks, not once per tick. A ten-NFE head that would be hopeless at 50 Hz per step becomes affordable when its cost is spread across an eight-step prefix. Chunking is not only a smoothness trick; it is the reason a multi-step sampler can meet a real-time budget at all.

## Multimodality: how sharply can the head disagree with itself

The second axis is expressiveness, how faithfully the head represents the fact that a task often has several correct actions. §10.1 made the case with the mug-and-laptop image and §10.2 made it measurable on PushT, the bimodal block-pushing benchmark where a mean-regression policy stalls against the block's flat edge because the average of "go left" and "go right" is "drive straight into it." Here we rank the heads by how sharply they hold competing modes apart.

At the bottom is plain regression, an L2 or L1 head that predicts one action. It has no multimodality at all; it collapses every mode to their mean, and on a bimodal task that mean is often the one action that is wrong. This is the baseline the whole chapter exists to beat, and it is worth remembering it is still the right choice when the task genuinely has one mode.

In the middle sits ACT's CVAE. A Gaussian latent $z$ can carry more than one mode, but a unimodal prior smears nearby modes together, and, as §10.2 noted, ACT often runs with $z$ near its mean and leans on chunking for most of its performance. Call it multimodality on paper that is used lightly in practice.

At the top are diffusion and flow matching. Both represent a full distribution over action chunks and sample a single committed mode from it, so both clear PushT cleanly. The catch is the coupling to the first axis: **the sharpness of that distribution is what you spend NFEs to recover.** Collapse a diffusion sampler to one or two steps, or push rectified flow toward genuine one-step generation, and the sampled distribution starts to blur back toward its mean, you are trading multimodality for latency directly, and past a task-dependent point the head stops committing to a mode and starts averaging again. This is why "just use one step" is a claim to check against your task's mode structure, not a free upgrade.

## Smoothness: the axis that lives between chunks

The third axis is the one demos undersell and hardware punishes: whether the executed trajectory is smooth or jerks. Jerk is not a cosmetic complaint. It shakes cameras, overshoots contact-rich targets, and on a real arm it is the difference between seating a battery and crushing it.

Chunking is the main lever, and §10.2 already explained the mechanism: predicting a coherent block of future actions keeps the motion committed over a window, so short-term noise averages out inside the chunk instead of accumulating across single-step reactions. That is also the smoothness half of chunking's answer to compounding error (§6.3). But chunking creates its own seam, the boundary where one chunk ends and the next begins, which is exactly where a discontinuity can appear if the new chunk disagrees with where the old one left the arm. The two systems handle the seam differently, and the difference is instructive:

- **Receding horizon (Diffusion Policy)** executes a prefix and replans, accepting a potential discontinuity at each replan boundary and keeping it small by predicting far and acting briefly.
- **Temporal ensembling (ACT)** queries every timestep, producing overlapping chunks, and averages the several predictions that now exist for each moment. The overlap smooths the seam by construction, at the cost of running the head every tick, which, since ACT is 1 NFE, it can afford. A ten-NFE head cannot run every tick, which is why Diffusion Policy reaches for receding horizon instead. The smoothness strategy and the latency budget are, again, the same decision.

One more smoothness lever is upstream of the generator entirely: the action space. §10.2 noted Diffusion Policy predicts end-effector positions rather than velocities, because absolute position targets stay anchored while integrated velocities drift. Continuous heads help here too, π0's flow head emits continuous action chunks with no quantization, whereas a discrete action-token head (RT-2, reached in Chapter 12) must round every action to a bin, and bin edges are small discontinuities baked into the representation before the controller ever sees them. Smoothness is partly won or lost before you choose a sampler.

## Choosing a head: match the corner to the task

Put the three axes together and the choice stops being about which paper is newest. Ask three questions about the task, in order:

**Is it multimodal?** If there is one sane way to do the motion, insert this peg at the one approach angle the geometry allows, you need no multimodality, and a regression head or ACT with a near-mean latent is not just adequate, it is the right call: fewer moving parts, one NFE, nothing to tune. Spending a ten-step diffusion sampler on a unimodal task is paying for a distribution with one point in it.

**How tight is the control loop?** If the loop is slow enough that ten NFEs fit inside the budget after receding-horizon amortization, Diffusion Policy's sharp distribution is free to use. If the loop is tight and the model is large, the 50 Hz, billion-parameter regime of dexterous manipulation, you cannot afford ten calls to a big backbone, and flow matching's few-step sampling is what makes the frequency reachable at all. This is precisely why π0 chose a flow head over a diffusion head (§10.3): not fashion, arithmetic.

**How much does smoothness matter, and where are the seams?** Contact-rich and high-frequency tasks want the tightest seam handling, temporal ensembling if you can afford per-tick inference, a continuous action space regardless. Coarser pick-and-place tolerates receding-horizon seams fine.

Two worked cases make the framework concrete. Peg insertion with a fixed approach: unimodal, moderately tight, moderate smoothness, ACT is close to ideal, and its single pass costs nothing you needed. PushT: genuinely bimodal, loop timing forgiving, Diffusion Policy earns its extra compute because the multimodality is the whole task. Folding laundry at 50 Hz on a foundation-scale model: multimodal, brutally tight loop, smoothness critical, flow matching on a continuous action space, which is the π0 corner. Same three axes, three different points in the cube, each defensible for its task and indefensible for the others.

The honest summary is that there is no dominant head, only heads matched well or badly to tasks. The three axes are coupled tightly enough that improving one usually spends another, so "best" is a property of the pairing, not of the algorithm. What travels across all of them is the chunk, every head here predicts a block of future actions, and that one shared decision is doing more work than the generator bolted on top of it. Section 10.5 takes this framework to the models that actually ship and asks which corner each one chose: how RT-2, OpenVLA, π0, and Helix each answered these three questions, and what their answers reveal about where foundation action models are heading.
