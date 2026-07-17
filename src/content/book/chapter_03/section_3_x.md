---
chapter: 3
section: 3.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §3.1–§3.6; the SmallPolicy training loop from §3.3; the debugging checklist from §3.5; PyTorch installed; about two hours of laptop CPU time (no GPU required)
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Chi et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. arXiv:2303.04137.
  - Zhao et al. (2023). Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. (ACT.) arXiv:2304.13705.
  - Collaboration et al. (2024). Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
---

# 3.x  Hands-on exercise + chapter references

Chapter 3 was the math chapter; the exercise is where that math shows up on a stopwatch. The four drills below take a combined two hours on a laptop CPU. No GPU required, no robot required, no internet required after the initial dependency install. The point is to leave the chapter with the SmallPolicy from §3.3 sitting in a directory on your disk, a debugging notebook beside it with the seven §3.5 checks scripted into reusable functions, and a one-page side-by-side comparison of the three §3.4 loss families on the same toy task. That artifact is what Chapters 6, 7, and 10 will assume you can stand up from scratch in under twenty minutes.

## Exercise 3.x.1 — Implement SmallPolicy and reproduce the baseline curve

Open a new file called `small_policy.py` and re-implement the SmallPolicy from §3.3 based on the chapter's description. Don't copy the listing verbatim. Writing it from the description forces you to notice the six lines of actual optimization work against the surrounding plumbing, which is the central claim of §3.6. The architecture is a three-layer MLP: input is a 14-dimensional observation (7 joint angles, 7 joint velocities), output is a 7-dimensional continuous action, hidden size 256, ReLU activations, no dropout. The dataset is synthetic, a toy inverse-dynamics map where the action is a fixed linear function of the observation plus Gaussian noise. Generate 10,000 training pairs and 2,000 validation pairs once, cache them to `.npz`, and reuse the cache across all four exercises so the data side stays identical throughout.

Train with MSE loss, Adam at `lr=1e-3`, batch size 64, for 50 epochs. Log training and validation loss to a CSV every 100 steps. The deliverable is one plot, training loss and validation loss on the same y-axis, step on the x-axis. You should see a clean monotone decrease for both, validation sitting slightly above training, both flattening around step 3,000. If your curves look qualitatively different, oscillation, NaN, flat from step 0, no gap between train and validation, stop and run the §3.5 checklist before continuing. The whole point of this exercise is that the baseline curve has a known shape, and any deviation is diagnostic. Save the plot as `baseline_mse.png`.

Wall clock on a 2023-era laptop CPU: about ten minutes.

## Exercise 3.x.2 — Break the loop on purpose, one fault at a time

Copy `small_policy.py` to `broken_drills.py`. Introduce, one at a time and never simultaneously, the following five faults, each in its own branch controlled by a `fault` argument. Run a 50-epoch training for each fault, and save the loss curve. Don't look at the §3.5 prediction for a fault until after you've run the drill and stared at the curve yourself.

1. `fault="big_lr"`: change `lr` from `1e-3` to `1e0`. Expected
   signature from §3.5: oscillation, possibly NaN.
2. `fault="tiny_lr"`: change `lr` to `1e-8`. Expected signature:
   training loss decreases by less than a percent in 50 epochs; flat-
   looking curve.
3. `fault="unnormalized_inputs"`: skip the per-dimension normalization
   step on the observations, so they are passed in raw with a standard
   deviation around 50. Expected signature: NaN within the first few
   steps, or extremely large initial loss that decays anomalously.
4. `fault="action_outlier"`: inject one demonstration per batch with an
   action value of `1e3` in dimension 3, leaving the other 99% of data
   intact. Expected signature: periodic loss spikes whenever a poisoned
   batch is drawn; gradient clipping makes them disappear.
5. `fault="bimodal_labels"`: synthesize the dataset with two
   demonstrators whose preferred actions differ by a fixed offset in
   dimension 0, so the target distribution at each observation is
   bimodal. Expected signature: training loss plateaus at a value
   roughly equal to half the squared inter-mode distance, never lower.

For each fault, write one sentence, "training loss did X, gradient norm did Y, the diagnosis from §3.5 is Z," and save all six plots (baseline plus five faults) to `drills/curves/`. The collection is the deliverable. Most students get four of the five diagnoses right on the first read of the curves; the bimodal one is hardest, since the plateau looks similar to a capacity ceiling. The Chapter 10 discussion of Diffusion Policy (arXiv:2303.04137) and ACT (arXiv:2304.13705) returns to this exact failure as the motivating example for distributional action heads.

Wall clock: about thirty minutes for all five drills combined.

## Exercise 3.x.3 — The three-loss-family bake-off

This is the loss-family exercise from §3.4 made concrete. Take the same SmallPolicy architecture and the same synthetic dataset from Exercise 3.x.1, and train three versions:

- *Supervised cross-entropy*: discretize each action dimension into 64
  uniform bins (taken from the per-dimension min and max of the training
  set). The model outputs `7 × 64` logits per observation; loss is
  per-dimension cross-entropy summed across dimensions. This is a
  smaller version of the RT-1 / OpenVLA recipe (arXiv:2212.06817,
  arXiv:2406.09246), bins reduced from 256 to 64 to fit a CPU budget.
- *Supervised MSE*: the baseline from 3.x.1. Continuous output, MSE
  loss. This is the ACT-style recipe (arXiv:2304.13705), simplified.
- *Self-supervised denoising MSE*: add Gaussian noise of variance `σ²`
  to the target actions (with `σ` sampled per-batch from a uniform
  schedule on `[0.01, 1.0]`), pass observation plus the noisy action
  plus `σ` to the model, and have it predict the noise. This is a
  minimal Diffusion-Policy-shaped objective (arXiv:2303.04137).

Train each for 50 epochs with the same Adam settings, log the same fields, and produce one comparison plot: three loss curves on the same axes, normalized by their respective theoretical minimum so the y-axes stay comparable. Then, at inference, do something interesting: for each of 20 held-out observations, sample 10 predictions from each model (deterministic argmax for cross-entropy, deterministic forward pass for MSE, 10 reverse-process samples with 20 integration steps each for the denoising model). Plot the spread.

The plot should make three things visible. The cross-entropy model is deterministic at argmax and produces zero spread; switching to a sampled prediction makes it multimodal, but only at bin resolution. The MSE model collapses to a single mean prediction at every observation regardless of bimodality in the data. The denoising model produces a smooth distribution that, on the bimodal dataset from Exercise 3.x.2 fault 5, recovers both modes. That single plot is the visual answer to "why does π0 use flow matching" (arXiv:2410.24164) and "why does Octo use a diffusion head" (arXiv:2405.12213). Chapter 10 spends most of its length unpacking what you see here in 30 minutes of CPU time.

Wall clock: about forty-five minutes including the inference sweep.

## Exercise 3.x.4 — Read one PyTorch paper appendix

Open the OpenVLA paper (Kim et al., 2024, arXiv:2406.09246) to the training-details appendix (typically Appendix A or B; the exact section heading differs between arXiv versions). Read only that appendix. Mark, with a pencil or in a text file, every hyperparameter specifically called out: optimizer, learning rate, schedule, weight decay, batch size, gradient-clip threshold, mixed-precision dtype, warmup steps, total steps. Then open `small_policy.py` and write a short comment block above your training loop recording the difference between each of those values and what you used. For example: "OpenVLA: AdamW, `lr=5e-5`, cosine schedule with 1000 warmup steps, weight decay `0.01`, grad clip `1.0`, bf16 mixed precision. Me: Adam, `lr=1e-3`, no schedule, no weight decay, no clip, fp32." There's no scoring rubric here. The point is making the OpenVLA training appendix stop feeling like a list of inscrutable magic numbers, since every one of those numbers is a knob you've now turned at least once in the wrong direction yourself.

When you reach Chapter 16 and need to fine-tune OpenVLA for your own data, you'll return to this comment block. The shape of what you have to tune away from defaults matters far more than the specific values.

Wall clock: about thirty minutes for the read plus annotation.

## Chapter 3 reading list

The works below are the ones cited across §3.1 through §3.6, grouped by purpose. Full bibliographic entries for everything cited in the whole book live in Appendix E.2; this list is just the chapter-local subset.

### Deep learning foundations (linear algebra, calculus, training)

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.*
  MIT Press. Chapters 2–4 (linear algebra, probability, numerical
  computation) cover what §3.1 and §3.2 sketched in 30 pages.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An
  Introduction.* MIT Press. The probability and information-theory
  chapters are the canonical reference for §3.2.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.*
  Springer. The older reference, still the cleanest exposition of the
  KL-vs-cross-entropy equivalence used in §3.2.
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic
  Optimization." arXiv:1412.6980. The optimizer that shows up in §3.3
  and in every chapter from 6 onward.
- Loshchilov, I., & Hutter, H. (2019). "Decoupled Weight Decay
  Regularization." (AdamW.) arXiv:1711.05101. The OpenVLA training
  appendix uses AdamW; this is the paper that explains why.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). "Delving Deep into
  Rectifiers." arXiv:1502.01852. The initialization scheme that prevents
  the vanishing-gradient pathology §3.5 names.

### Loss families: supervised, RL, self-supervised

- Pomerleau, D. A. (1988). "ALVINN." *NeurIPS 1988*. The original
  supervised behavior-cloning paper; the entire §3.4 supervised-family
  diagnostic table grows out of the failure modes ALVINN exhibited.
- Ross, S., Gordon, G., & Bagnell, D. (2011). "A Reduction of Imitation
  Learning and Structured Prediction to No-Regret Online Learning."
  (DAgger.) *AISTATS 2011*. Cited in Chapter 6; the canonical
  compounding-error response §3.4 forward-references.
- Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). "Proximal
  Policy Optimization Algorithms." arXiv:1707.06347. The reference
  RL loss for the §3.4 RL-family discussion.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft
  Actor-Critic." *ICML 2018*. The off-policy comparison point in §3.4.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion
  Probabilistic Models." arXiv:2006.11239. The diffusion training
  objective Exercise 3.x.3 minimally re-implements.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., et al. (2023). "Flow
  Matching for Generative Modeling." arXiv:2210.02747. The flow-matching
  objective π0 (arXiv:2410.24164) inherits, mentioned in §3.4 and
  §3.5.

### Action-model instances used as worked examples

- Brohan, A., Brown, N., Carbajal, J., et al. (2022). "RT-1: Robotics
  Transformer for Real-World Control at Scale." arXiv:2212.06817. The
  cross-entropy-on-bins recipe from §3.2 and §3.4.
- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). "OpenVLA:
  An Open-Source Vision-Language-Action Model." arXiv:2406.09246. The
  scaled-up version of the same recipe, and the source of the
  gradient-clip-1.0 default §3.5 cites.
- Chi, C., Feng, S., Du, Y., et al. (2023). "Diffusion Policy:
  Visuomotor Policy Learning via Action Diffusion." arXiv:2303.04137.
  The self-supervised-on-actions worked example in §3.4 and the
  motivating model behind Exercise 3.x.3.
- Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). "Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware." (ACT.)
  arXiv:2304.13705. The MSE-on-continuous-actions instance compared in
  Exercise 3.x.3.
- Octo Model Team (2024). "Octo: An Open-Source Generalist Robot
  Policy." arXiv:2405.12213. The diffusion-head VLA referenced in §3.4.
- Black, K., Brown, N., Driess, D., et al. (2024). "π0: A
  Vision-Language-Action Flow Model for General Robot Control."
  arXiv:2410.24164. The flow-matching example in §3.5; the chapter that
  unpacks it is Chapter 13.

### Background you may want nearby

- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics,
  Planning, and Control.* Cambridge University Press. Chapter 5
  derives the manipulator Jacobian used in §3.1.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010).
  *Robotics: Modelling, Planning and Control.* Springer. The
  alternative reference for the same Jacobian derivation.
- Paszke, A., Gross, S., Massa, F., et al. (2019). "PyTorch: An
  Imperative Style, High-Performance Deep Learning Library."
  arXiv:1912.01703. The PyTorch reference; useful when the §3.3
  training loop has to be ported to a more involved data pipeline.

## Chapter summary

Chapter 3 was the math, code, and debugging chapter, and it closes Part 1. You can now write the gradient of a scalar loss with respect to a parameter vector and follow it through a transformer. You can convert between cross-entropy and KL divergence without thinking about it. You can stand up a 50-line PyTorch training loop for any of the three loss families in under twenty minutes, and you have a seven-step debugging checklist that diagnoses the most common reasons a training run refuses to converge. With those four capabilities in hand, Part 2 begins. Chapter 4 covers the classical-actions family, STRIPS, PDDL, inverse kinematics, motion planning, computed-torque control, both the oldest layer in the action-model story and the one still running underneath every modern VLA stack you'll encounter in the rest of the book.
