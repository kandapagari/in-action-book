---
chapter: 10
section: 10.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §10.1–§10.6; Python with PyTorch installed; a working grasp of the denoising process, the action chunk and receding-horizon execution, the difference between Diffusion Policy and ACT, flow matching and rectified flow, and the latency/multimodality/smoothness trade-off; the PushT benchmark (ships with the Diffusion Policy and LeRobot codebases); about three hours, most of it training two small policies on the same data
key_refs:
  - Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
  - Chi, C. et al. (2023). Diffusion Policy — Visuomotor Policy Learning via Action Diffusion. RSS 2023.
  - Zhao, T. Z. et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT). RSS 2023.
  - Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 10.x  Hands-on exercise + chapter references

Chapter 10 spent five sections insisting that the generator strapped to an action chunk is a design choice, not a law of nature, that diffusion and flow matching produce the same kind of output and differ mainly in how many network calls they cost you to get it. That claim is easy to accept on paper and easy to forget the moment you have a policy that works. The way to make it stick is to build one policy, swap the head, and hold the two side by side on identical data. The headline exercise does exactly that, and it is the drill the TOC names: train a Diffusion Policy on PushT, replace the diffusion head with a flow-matching head, and compare what changes and what does not. The supporting drills probe the edges the chapter argued about, whether the multimodality of §10.1 is real or a story, what the sampling-step count actually buys you, and how the receding-horizon window of §10.2 trades smoothness against reactivity.

Budget about three hours. Most of it is two training runs on a small task that you can start and walk away from; the code you write on top is short. Start the first run early and read while it trains.

```
pip install diffusion_policy   # or: pip install lerobot, which ships both heads
```

PushT is the task from §10.2: a circular pusher must nudge a T-shaped block into a target pose in a 2-D plane. It exists precisely to make bimodal behavior visible, the block can be approached from either side, and it trains in minutes, not hours, on a single GPU. Everything below runs offline against the demonstration set that ships with the benchmark.

## Exercise 10.x.1 — Train Diffusion Policy, then swap in a flow-matching head

Start from a working Diffusion Policy on PushT. Use the reference implementation rather than writing the denoiser from scratch; the point of this exercise is the comparison, not a reimplementation of §10.1. Train it on the PushT demonstrations with the defaults, a 1-D convolutional U-Net over the time axis, an action chunk of sixteen steps, the last two observations as the condition, DDIM sampling at roughly ten steps, and confirm you can reproduce the benchmark's headline: a success rate well above what a plain mean-regression policy reaches on the same data. That gap is §10.1's multimodality argument cashed out, and you want it in hand before you touch anything.

Now the swap the TOC asks for. Keep every part of the system fixed, the vision encoder, the U-Net backbone, the chunk size, the receding-horizon execution, and change only the training objective and the sampler. Where diffusion trains $\epsilon_\theta$ to predict the noise added at a random level $t$, the flow-matching head of §10.3 trains a velocity field $v_\theta(\mathbf{a}_t, t, o)$ to point from a noise sample straight toward the clean action chunk along the linear interpolation between them:

```python
# a0: clean action chunk;  a1 ~ N(0, I): noise
# t ~ Uniform(0, 1)
a_t   = (1 - t) * a0 + t * a1          # linear path from data to noise
target = a1 - a0                        # constant velocity along that path
loss  = ((v_theta(a_t, t, obs) - target) ** 2).mean()
```

Sampling is now integrating an ODE from noise back to data, a handful of Euler steps, sometimes as few as one or two after the path has been straightened, rather than the ten-rung denoising ladder. Retrain on the same PushT demonstrations and evaluate the same way.

The result you are looking for is deliberately anticlimactic. Success rate should land close to the diffusion policy's, within noise, if you tuned both fairly, because both heads represent the same multimodal distribution over the same chunk. What moves is inference cost: the flow head reaches comparable quality at far fewer function evaluations. Record both numbers, success rate and network calls per chunk, for both heads, and write one sentence: the generator was interchangeable, the compute was not. That sentence is the whole thesis of §10.4, and you now have it as two rows in a table you produced.

Wall clock: about ninety minutes, most of it the two training runs.

## Exercise 10.x.2 — Sweep the sampling steps and find where quality saturates

This drill turns "the compute was not interchangeable" into a curve. Take both trained heads from 10.x.1 and vary the number of function evaluations (NFE) used at sampling time, for the diffusion head, the number of DDIM steps; for the flow head, the number of Euler integration steps. Sweep each across a range, say 1, 2, 4, 8, 16, 32, and record PushT success rate at every setting. Plot NFE against success for both heads on one axis.

The two curves tell the §10.4 story in one picture. The diffusion head's success collapses at very low step counts and climbs as you add steps, plateauing somewhere around eight to ten, below that, the denoiser has too few rungs to land on the data manifold and the actions come out garbled. The flow head's curve is shifted left: it reaches its plateau at a handful of steps, and if you applied reflow (§10.3) it may hold up even at one or two. Mark the NFE where each curve flattens. That knee is the head's honest latency cost, and the gap between the two knees is exactly why π0 (Black et al., 2024, arXiv:2410.24164) chose a flow-matching head for real-time control rather than a diffusion one. For the written part, multiply each knee by your measured per-call latency and state the control frequency each head can actually sustain. One of them may miss a 10 Hz budget; that is the point.

Wall clock: about forty minutes, reusing both trained models.

## Exercise 10.x.3 — Kill the multimodality and watch PushT stall

§10.1 claimed that averaging two good actions produces a bad one, and PushT was built to expose it. Make the failure appear on purpose. Train a third policy on the same data with the same backbone and chunk, but replace the generative head with a plain regression head, one that outputs the action chunk directly under an L2 loss, no noise, no latent, no velocity field. This is the mean-regression baseline §10.1 warned about, and PushT will punish it.

Run it and watch where it fails. On approaches where the demonstrations split between going around the block's left and right, the regressed action is the average of the two, which drives the pusher straight into the flat edge of the T and stalls. Log the states where the episode stalls and confirm they cluster at exactly these decision points, not at random. Then compare success rates across all three heads, regression, diffusion, flow, on the identical dataset. The regression head should sit well below the other two, and the two generative heads should sit together. That ordering is the chapter's central claim rendered as three numbers: the generator has to be able to represent more than one mode, but which multimodal generator you pick is close to a wash on quality.

Wall clock: about thirty minutes; the regression head trains fastest of the three.

## Exercise 10.x.4 — Tune the receding-horizon window

The last drill isolates the deployment knob from §10.2 that has nothing to do with the generative head at all. Diffusion Policy predicts a chunk of $T_p$ actions and executes only the first $T_a$ before replanning. Take your flow head from 10.x.1 (it is the cheapest to sample, so this sweep is fast) and vary $T_a$ while holding the predicted horizon $T_p$ fixed at sixteen. Try executing the full chunk ($T_a = 16$, replan rarely), a prefix ($T_a = 8$), and nearly every step ($T_a = 1$, replan constantly). Record success rate and, if your harness exposes it, a smoothness measure such as mean squared jerk of the executed trajectory.

The trade-off should separate cleanly. Executing the whole chunk gives the smoothest motion, the policy commits to a plan and follows it, but the robot is blind to anything that changes mid-chunk, so success dips on any perturbation. Replanning every step keeps it maximally reactive but reintroduces the jitter at chunk boundaries that §10.2 said temporal ensembling exists to suppress. Somewhere in between is the setting that keeps PushT both smooth and reactive. Name it, and note that you found it without retraining anything: this is a pure inference-time knob, orthogonal to the head choice you spent the first three exercises studying. Keeping those two axes separate, what generator, and how you schedule its output, is the mental model to carry into Part 4.

Wall clock: about twenty-five minutes, sweeping one trained model.

## Chapter 10 reading list

The works below are cited in §10.1–§10.6, grouped by purpose. Full bibliographic entries for everything cited in the book live in Appendix E.2; this is the chapter-local subset.

### The diffusion mechanism

- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*. The corrupt-then-denoise recipe §10.1 builds from and the DDPM sampler Exercise 10.x.2 sweeps.
- Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*. The score-matching, SDE view §10.1 names as the second dialect for the same object.

### Diffusion and CVAE action heads

- Chi, C., et al. (2023). "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *RSS 2023*. The system Exercises 10.x.1–10.x.4 train; the action chunk, receding-horizon execution, and the PushT result at the center of this chapter.
- Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (ACT). *RSS 2023*. §10.2's single-pass CVAE alternative and the temporal-ensembling counterpart to receding horizons.

### Flow matching and rectified flow

- Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." *ICLR 2023*. The velocity-field objective §10.3 derives and Exercise 10.x.1 swaps in.
- Liu, X., Gong, C., & Liu, Q. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR 2023*. The reflow procedure §10.3 uses to straighten the sampling path down to one or two steps, the effect Exercise 10.x.2 measures.

### Action heads in shipping VLAs

- Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818. §10.5's discrete action-token classifier, the pole opposite continuous diffusion.
- Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. §10.5's open discrete-token policy; the baseline Part 4 returns to repeatedly.
- Octo Model Team (2024). "Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213. §10.5's generalist policy with a diffusion action head, developed in Chapter 12.
- Black, K., et al. (2024). "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164. The flow-matching head §10.3 and §10.5 build toward and Exercise 10.x.2 explains the latency motivation for; Chapter 13 in full.
- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for Vision-Language-Action Models." arXiv:2501.09747. §10.5's frequency-domain action tokenizer, the discrete-token camp's answer to the smoothness problem.

## Chapter summary

Chapter 10 answered the question every earlier chapter deferred: once a policy has decided what to do, how does it emit the numbers, and how does it emit them when more than one answer is right? You can now explain the diffusion mechanism without hand-waving, corrupt a sample along a fixed noise schedule, train a network to predict the noise, and generate by denoising from pure noise, and say why that machinery represents a full multimodal distribution where mean-regression collapses to a single averaged, often colliding, action. You can build the two reference action heads of 2023, Diffusion Policy and ACT, and explain the one idea they share, the action chunk, and the two knobs they use to turn overlapping chunks into smooth commands: receding-horizon execution and temporal ensembling, the latter of which Exercise 10.x.4 had you tune by hand. You can describe flow matching in plain terms, learn a velocity field that carries noise to data along a straight path, and say precisely where it beats diffusion for control: not on quality, which Exercise 10.x.1 showed is a wash, but on sampling steps, the axis Exercise 10.x.2 measured and the reason π0 runs in real time. And you can decide which head a task actually needs by weighing the three quantities this chapter kept returning to, latency, multimodality, and smoothness, against what the robot can afford. That last skill is the one Part 4 leans on. Starting in Chapter 11, we stop studying action heads in isolation and read the modern VLA literature paper by paper, beginning where the recipe began: CLIP, language conditioning, and the discrete action tokens of RT-1.
