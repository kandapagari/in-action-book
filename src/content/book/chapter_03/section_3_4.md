---
chapter: 3
section: 3.4
title: Three loss families: supervised, RL, self-supervised
target_words: 2000
status: draft
prereqs: §3.1 (gradients, chain rule), §3.2 (cross-entropy, KL divergence), §3.3 (PyTorch training loop)
key_refs:
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Collaboration et al. (2024). Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
---

# 3.4  Three loss families: supervised, RL, self-supervised

The training loop in §3.3 assumed a specific loss: cross-entropy on discretized
action bins, computed against human-demonstrated actions. That assumption holds
for RT-1 (arXiv:2212.06817) and OpenVLA (arXiv:2406.09246), but it is not the
only game in town. Every action model in this book trains by minimizing some
scalar loss function, and the properties of the model are shaped not just by
its architecture but by which loss family that function belongs to.

Three families cover essentially everything in the current literature. The first
is **supervised loss**: you have labeled data, you know the right answer, and you
penalize prediction errors directly. The second is **reinforcement learning loss**:
you do not have labeled actions, but you have a reward signal, and you update the
policy to increase expected future reward. The third is **self-supervised loss**:
there are no human labels at all — the training signal is manufactured from the
structure of the data itself, typically by corrupting something and asking the
model to reconstruct it. Understanding these three families, their trade-offs,
and the conditions that favor each one is a prerequisite for reading any modern
VLA paper.

## Supervised loss

A supervised loss measures the discrepancy between a model's prediction and a
target that was provided by a human or a trusted oracle. In robot learning, the
oracle is typically a teleoperator who demonstrated the task.

**Discrete targets, cross-entropy.** When actions are discretized into bins —
as in RT-1, OpenVLA, and the SmallPolicy in §3.3 — the loss is cross-entropy
between the model's predicted probability distribution and the one-hot target:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log q_\theta(a_i^* \mid o_i)
$$

where $a_i^*$ is the demonstrated action bin and $q_\theta$ is the model's
predicted distribution. This is identical to the formula in §3.2; we are
minimizing the average negative log-probability of the demonstrated actions
under the model's distribution. The discrete-bin vocabulary makes this a standard
classification problem: one class per bin per action dimension, with cross-entropy
aggregated across dimensions.

**Continuous targets, regression loss.** Some models output raw continuous
vectors rather than bin indices. The simplest loss then is mean-squared error:

$$
\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \| f_\theta(o_i) - a_i^* \|^2
$$

MSE is differentiable everywhere, easy to implement, and corresponds to maximum
likelihood under a Gaussian likelihood on the actions. Its limitation for robot
learning is that it is unimodal: if a demonstrated action is sometimes "grasp
from the left" and sometimes "grasp from the right" depending on subtle visual
cues, MSE will average the two and predict a point between them — a prediction
that may be physically invalid. Discrete-bin cross-entropy is also unimodal (the
cross-entropy gradient pushes all probability mass toward the single demonstrated
bin), but the binning can at least represent high-entropy predictions by spreading
mass across adjacent bins. The problem of multimodal action distributions is a
recurring theme in this book; the diffusion and flow-matching losses in Chapter 10
are specifically designed to address it.

**When to use supervised loss.** The prerequisite is demonstrated action data —
teleoperation, kinesthetic teaching, motion capture, or any other method that
produces $(o, a)$ pairs where the action is known to be correct. Modern large
datasets like Open X-Embodiment (arXiv:2310.08864) are overwhelmingly composed
of this kind of data. If you have a few thousand demonstrations and no simulator,
supervised loss is almost always the right choice: it is sample-efficient,
numerically stable, and produces gradients that directly increase the
log-probability of behaviors you want.

The weakness is dependency on demonstration quality. A supervised model cannot
exceed its training data; a teleoperator who uses a suboptimal strategy will
teach the model that strategy. Supervised loss also cannot benefit from
interactions the policy generates itself — every gradient step uses human-provided
targets, not the model's own exploratory experience.

## Reinforcement learning loss

A reinforcement learning (RL) loss does not require demonstrated actions. Instead,
the policy generates trajectories by interacting with an environment, and a reward
function evaluates each trajectory. The goal is to find parameters that maximize
expected cumulative reward:

$$
\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

The negative sign is a convention: minimization of the loss corresponds to
maximization of reward. The discount factor $\gamma \in [0, 1)$ down-weights
rewards that occur far in the future; setting $\gamma$ close to 1 makes the
policy care about long-horizon outcomes.

The challenge is computing the gradient of this expectation with respect to $\theta$.
Unlike supervised loss, where $a_i^*$ is fixed and the gradient of $\log
q_\theta(a_i^*|o_i)$ is straightforward, the distribution over trajectories
itself depends on $\theta$. The standard solution is the policy gradient theorem,
which rewrites the gradient as:

$$
\nabla_\theta \mathcal{L}_{\text{RL}} = -\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\right]
$$

where $G_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$ is the return from step $t$.
The gradient is an expectation over sampled trajectories, estimated in practice
by rolling out the policy, collecting rewards, and treating each step as a
weighted log-probability maximization. High-return steps get strong positive
gradient signal; low-return steps get weak or negative signal.

**The variance problem.** $G_t$ is a sum of random rewards, and its variance
can be enormous — especially in long-horizon tasks where a single episode
involves hundreds of steps. High-variance gradient estimates require many
samples to converge. Practical algorithms like PPO (Chapter 7) reduce variance
by subtracting a baseline, clipping the probability ratio between the old and
new policy, and using advantage estimates rather than raw returns. The mechanics
are detailed in Chapter 7; the key point here is that RL losses are inherently
noisier than supervised losses, which means RL training typically requires one
to two orders of magnitude more compute and far more environment interactions.

**When to use RL loss.** RL requires a reward signal, which in turn requires
either a real environment (expensive and slow), a simulator (faster but subject
to sim-to-real gap), or a learned reward model. In the current VLA landscape,
pure RL from scratch for manipulation is rare: the exploration problem in
high-dimensional continuous spaces is severe, and most tasks do not yield a
natural dense reward. RL appears most prominently as a fine-tuning signal on
top of supervised pretraining — the reinforcement-learning-from-human-feedback
(RLHF) paradigm imported from language models, applied to robotics actions.
Chapter 5 and Chapter 7 treat RL in depth; Chapter 18 discusses its emerging
role in reasoning-augmented VLAs.

## Self-supervised loss

A self-supervised loss generates training signal from the structure of the data
itself, without external labels. The key move is to define an auxiliary task that
can be constructed automatically from unlabeled observations: corrupt the input
in some way, and train the model to reconstruct what was removed.

**Reconstruction and denoising.** The classic form is denoising: add noise to
a clean sample, then penalize the model's error in recovering the original.
Formally, let $x$ be a clean data point, $\tilde{x} = x + \epsilon$ a noisy
version, and $f_\theta$ the model. The loss is:

$$
\mathcal{L}_{\text{denoise}}(\theta) = \mathbb{E}_{x, \epsilon}\left[\| f_\theta(\tilde{x}) - x \|^2\right]
$$

This looks like an MSE loss, but the target $x$ is not a human-provided label —
it is a corrupted version of the model's own input. This is the loss that trains
diffusion models: with a specific noise schedule and parameterization, the denoising
objective is equivalent to learning the score function of the data distribution
(the gradient of the log-density), which can then be used to generate new samples
by iterative denoising. Octo (arXiv:2405.12213) uses a diffusion head on top of a
transformer backbone; the head is trained with a variant of this loss, learning
to generate smooth action trajectories from a noise-corrupted starting point.

**Flow matching.** A related but distinct self-supervised objective is flow
matching. Instead of learning to denoise, the model learns a vector field that
transports samples from a simple noise distribution $p_0$ (e.g., Gaussian) to
the target action distribution $p_1$ (the distribution of demonstrated actions).
Given a linearly interpolated path $x_t = (1-t) \epsilon + t a^*$ between a noise
sample $\epsilon \sim \mathcal{N}(0, I)$ and a demonstrated action $a^*$, the
target velocity is $(a^* - \epsilon)$ and the loss is:

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \epsilon, a^*}\left[\| v_\theta(x_t, t) - (a^* - \epsilon) \|^2\right]
$$

The model $v_\theta$ learns to predict the velocity that points from the noise
toward the target. At inference, you start from Gaussian noise and integrate
the learned velocity field, arriving at a sample from the action distribution
after a fixed number of steps. This is exactly the action head in π0
(arXiv:2410.24164): the flow-matching loss lets the model represent multimodal
action distributions without committing to a single mode, which is critical for
dexterous manipulation where the same task can be accomplished in qualitatively
different ways. We return to flow matching in Chapter 10.

**Contrastive loss.** A second major branch of self-supervised learning is
contrastive: rather than predicting corrupted inputs, the model learns an
embedding space where semantically similar pairs are pulled together and
dissimilar pairs are pushed apart. The InfoNCE loss, which underlies CLIP, is:

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top g(y^+) / \tau)}{\sum_k \exp(f(x)^\top g(y_k) / \tau)}\right]
$$

where $(x, y^+)$ is a positive pair (e.g., an image and its caption),
$\{y_k\}$ includes the positive and negatives, and $\tau$ is a temperature
parameter. CLIP (which underlies the visual encoder in OpenVLA) was trained
with this loss on 400 million image-text pairs. The result is a visual encoder
whose representations are aligned with natural-language descriptions — which is
precisely the property that makes zero-shot language conditioning possible in VLAs.
Chapter 11 covers CLIP's role in the VLA recipe in detail.

**When to use self-supervised loss.** Self-supervised losses are the dominant
pretraining signal when labeled data is scarce or nonexistent. The internet
contains billions of images with captions, hours of robot video without action
labels, and structured language data of every variety. Self-supervised losses
can exploit all of it. The cost is indirection: a denoising or contrastive loss
trains the model on a proxy task, and the quality of the downstream behavior
depends on how well that proxy predicts the real task. Getting this right requires
careful choices about architecture, noise schedule, and what pairs count as
"positive" — choices that are largely empirical and chapter-specific.

## Comparing the three families

The table below summarizes the practical decision factors:

| Factor | Supervised | RL | Self-supervised |
|---|---|---|---|
| Requires labeled actions | Yes | No | No |
| Requires reward function | No | Yes | No |
| Requires environment interaction | No | Yes (extensively) | No |
| Handles multimodal actions | Poorly (MSE) / Partially (CE) | Yes | Yes (diffusion, flow) |
| Sample efficiency | High | Low | Medium |
| Training stability | High | Low | Medium |
| Common use in VLAs | Fine-tuning, action heads | Post-training RLHF | Pretraining, diffusion/flow heads |

In practice, the largest action models combine all three. A typical pipeline looks
like this: (1) a vision-language backbone is pretrained on internet-scale data
using contrastive or masked-prediction self-supervised losses; (2) an action head
is added and the combined model is supervised on robot demonstrations; (3) the
model is optionally fine-tuned with RL or RLHF using a reward model trained from
human preferences. RT-2 (arXiv:2307.15818) is an example of step 1 feeding into
step 2. Post-training with RL remains an active area of research; Chapter 18
discusses the current state of reasoning-augmented VLAs that use RL to extend
beyond the behaviors seen in demonstrations.

## A diagnostic question

Before writing any loss function, ask: *where does the training signal come from?*
If a human selected the target, it is supervised. If the environment evaluated a
trajectory, it is RL. If the target was derived from the input data itself without
human selection, it is self-supervised. This question is sometimes subtle: the
denoising target $x$ in a diffusion loss was originally a human-demonstrated
action — but the noisy version and the loss construction are automatic, so the
loss is self-supervised in character. What matters operationally is whether the
gradient update is pointing toward human-provided targets (supervised),
toward high-reward behavior (RL), or toward internal data consistency
(self-supervised).

Getting this right matters for debugging. If a supervised model fails, the
first suspect is data quality: are the demonstrated actions actually correct?
If an RL model fails, the first suspect is reward design: is the reward function
specifying what you think it is? If a self-supervised model fails, the first
suspect is the proxy task: is denoising/contrastive alignment actually predictive
of downstream performance? Chapter 3.5 gives a checklist for diagnosing each of
these failure modes in a training run that will not converge.
