---
chapter: 3
section: 3.2
title: Random variables, expectations, KL divergence
target_words: 2000
status: draft
prereqs: §3.1 (vectors, gradients, chain rule); familiarity with summation notation and basic set theory
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Ghosh et al. (2024). Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
---

# 3.2  Random variables, expectations, KL divergence

The previous section described a robot policy as a function that turns inputs into
outputs — a vector in, a vector out. That picture is incomplete. The action
OpenVLA emitted in Chapter 2 was not a deterministic response to the image; it
was the mode of a distribution over 256 discrete bins. The policy learned not
"given this image, move the gripper 4.2 mm right" but "given this image, the
probability distribution over move-right distances peaks near bin 143." That
shift — from functions to distributions — is the subject of this section. If
§3.1 introduced the spatial language of robot learning, this section introduces
its probabilistic language.

## Random variables and distributions

A *random variable* $X$ is a quantity whose value is not fixed in advance; you
can only describe how likely different values are. For a discrete random variable
— one whose outcomes come from a finite or countable set $\mathcal{X}$ — the
distribution is a *probability mass function* $p(x) = P(X = x)$, satisfying
$p(x) \geq 0$ for all $x$ and $\sum_{x \in \mathcal{X}} p(x) = 1$.

For a continuous random variable — one that can take any value in an interval or
in $\mathbb{R}^{n}$ — the distribution is a *probability density function*
$p(x)$ satisfying $p(x) \geq 0$ and $\int p(x)\, dx = 1$. A density does not
assign probability to a single point; rather, $\int_{A} p(x)\, dx$ is the
probability that $X$ lands in the set $A$.

The distinction is load-bearing. RT-1 (Brohan et al., 2022, arXiv:2212.06817)
and OpenVLA (Kim et al., 2024, arXiv:2406.09246) discretize each action
dimension into 256 bins, so the predicted action is a product of 7 discrete
distributions — one PMF per joint. The loss is a sum of 7 cross-entropies,
each comparing a PMF to a one-hot target. π0 (Black et al., 2024,
arXiv:2410.24164), by contrast, treats the action as a continuous random
variable in $\mathbb{R}^{7}$ and learns a flow from Gaussian noise to the
target action distribution; its loss never involves a PMF at all. The choice of
discrete versus continuous is not aesthetic. It determines which math you write
and which pathologies you fight, a tension we return to in Chapter 10.

A *Gaussian* (normal) distribution over $\mathbb{R}^{n}$ with mean $\mu$ and
covariance $\Sigma$ is:

$$
\mathcal{N}(x;\, \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right).
$$

The exponent is negative and large when $x$ is far from $\mu$ in the metric
defined by $\Sigma^{-1}$, so high probability mass sits close to the mean.
Gaussians are the default prior distribution for continuous action spaces
because they are closed under affine transformations, they have closed-form
entropy, and minimizing the negative log-likelihood of a Gaussian target with
a fixed diagonal covariance is exactly minimizing mean-squared error — the loss
you would write on intuition for a regression problem. When Octo (Ghosh et al.,
2024, arXiv:2405.12213) conditions a diffusion head on task tokens, the
diffusion starts from $\mathcal{N}(0, I)$ and learns to transport that noise to
the target action region; every step of that transport is a perturbation of a
Gaussian. Know the Gaussian and you understand the starting point of most
continuous generative action models.

## Expectation

The *expectation* of a function $f$ under a distribution $p$ is the
probability-weighted average of $f$'s values:

$$
\mathbb{E}_{x \sim p}[f(x)] =
\begin{cases}
\displaystyle\sum_{x \in \mathcal{X}} p(x)\, f(x) & \text{discrete} \\[6pt]
\displaystyle\int p(x)\, f(x)\, dx & \text{continuous.}
\end{cases}
$$

For the discrete case, if $p$ is the distribution over action bins that OpenVLA
predicts and $f(x)$ is the cost of taking action $x$, then $\mathbb{E}[f(X)]$
is the expected cost of sampling from the policy's output. For the continuous
case, if $p$ is the stationary distribution induced by some policy interacting
with the environment and $f(x) = r(x)$ is a reward function, then
$\mathbb{E}_{x \sim p}[r(x)]$ is the average reward — exactly the quantity
that reinforcement learning maximizes (Chapter 5 formalizes this with Markov
decision processes).

Three properties of expectation are worth memorizing because they are used
constantly in derivations.

*Linearity.* $\mathbb{E}[af(X) + bg(X)] = a\,\mathbb{E}[f(X)] + b\,\mathbb{E}[g(X)]$.
This seems obvious but it means you can move constant factors in and out of
expectation freely, which lets you rearrange loss expressions at will.

*Law of total expectation.* $\mathbb{E}_x[f(x)] = \mathbb{E}_y[\mathbb{E}_{x \mid y}[f(x)]]$.
Condition on some intermediate variable $y$, compute the inner expectation over
$x$ given $y$, then average out the $y$. In robotics, $y$ might be the latent
state in a world model (Chapter 9), and this identity is how you justify
marginalizing over it.

*Monte Carlo estimation.* $\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)$ for
$x_i \overset{\text{iid}}{\sim} p$. The law of large numbers guarantees this
converges. In practice, $N = 1$ during training — you sample a single
trajectory and use its reward as an unbiased estimator of the expected reward.
The high variance of that estimate is the main reason policy-gradient training
is noisy, and the variance-reduction tricks in Chapter 7 (baselines, GAE,
advantage normalization) are all interventions on this Monte Carlo estimator.

## Entropy and cross-entropy

The *entropy* of a distribution $p$ measures how spread out it is:

$$
H(p) = -\mathbb{E}_{x \sim p}[\log p(x)] = -\sum_{x} p(x) \log p(x).
$$

A one-hot distribution (all mass at one outcome) has $H = 0$; a uniform
distribution over $k$ outcomes has $H = \log k$. In the context of action
distributions, low entropy means the policy is confident (one action dominates);
high entropy means the policy is uncertain or deliberately exploratory.

*Cross-entropy* between a target distribution $p$ and a predicted distribution $q$ is:

$$
H(p, q) = -\mathbb{E}_{x \sim p}[\log q(x)] = -\sum_{x} p(x) \log q(x).
$$

When $p$ is the one-hot distribution over the correct action token, as it is in
every discrete-action VLA trained on demonstrations, $H(p, q)$ reduces to
$-\log q(a^{\star})$: the negative log-probability assigned to the correct
action. That is the cross-entropy loss you saw applied to each of the 7 action
dimensions in §2.3. Minimizing it pushes $q$ to assign high probability to the
demonstrated action. The cross-entropy loss dominates the supervised phase of
nearly every language-conditioned imitation model from RT-1 (arXiv:2212.06817)
through RT-2 (arXiv:2307.15818) through OpenVLA (arXiv:2406.09246).

One subtlety worth noting: cross-entropy is not symmetric. $H(p, q) \neq H(q, p)$
in general. The convention matters when you move beyond supervised
imitation — for instance, when a VLA must be calibrated to cover all high-
probability actions of a multimodal demonstration distribution, not merely
fit one mode. We revisit that asymmetry in Chapter 6, when discussing why
behavior cloning struggles with multimodal data.

## KL divergence

The *Kullback-Leibler divergence* from $q$ to $p$ (read: "how different $q$
is from $p$") is:

$$
D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
= H(p, q) - H(p).
$$

The last equality comes from expanding the expectation: $\mathbb{E}_{p}[\log p - \log q] =
-H(p) + H(p, q)$, rearranged. Because $D_{\mathrm{KL}} \geq 0$ always (by
Jensen's inequality, with equality iff $p = q$), minimizing $H(p, q)$
over $q$ is the same as minimizing $D_{\mathrm{KL}}(p \,\|\, q)$ when the
target $p$ does not depend on $q$ — since $H(p)$ is a constant in that case.
The cross-entropy loss *is* the KL divergence, shifted by the entropy of the
target. Knowing this is more than trivia: it tells you what you are actually
doing when you train with cross-entropy. You are driving the predicted
distribution as close as possible to the demonstrated distribution, in the
specific asymmetric sense that $D_{\mathrm{KL}}(p \,\|\, q)$ penalizes placing
zero probability mass where the target $p$ has nonzero mass. If the teacher
only ever demonstrates one action variant, $p$ has low entropy, and the
penalty structure of $D_{\mathrm{KL}}(p \,\|\, q)$ will happily ignore the
modes of the true (unknown) distribution that the teacher never exhibited.
Compounding error in behavior cloning, discussed in Chapter 6, is a consequence
of this.

The reverse KL, $D_{\mathrm{KL}}(q \,\|\, p)$, penalizes placing mass where
the target $p$ is zero — which produces *mode-seeking* behavior. Variational
inference and some policy-optimization methods use the reverse KL; supervised
imitation typically uses the forward KL. The direction of the divergence is an
architectural choice that shapes the failure modes of the resulting policy.

### KL divergence in world models and latent variable models

The KL divergence appears explicitly in the training objective of any model with
a stochastic latent variable. The standard variational bound (ELBO) for a latent
variable model with observations $o$, latent state $z$, and parameters $\theta$
is:

$$
\log p_\theta(o) \geq \underbrace{\mathbb{E}_{z \sim q_\phi(z \mid o)}\!\left[\log p_\theta(o \mid z)\right]}_{\text{reconstruction}}
- \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z \mid o) \,\|\, p_\theta(z)\right)}_{\text{regularization}}.
$$

The first term rewards the model for explaining the observation using the
learned latent. The second term regularizes the posterior $q_\phi(z \mid o)$
toward the prior $p_\theta(z)$, preventing the latent from memorizing and
ignoring structure. World models such as RSSM (used in DreamerV3, which we
cover in Chapter 9) minimize a variant of this bound; the KL term is what keeps
the learned dynamics smooth enough to plan in. If you set the KL coefficient
to zero, the latent space collapses into a lookup table and the world model
loses its generalization ability. If you set it too high, the posterior is
dragged too close to a unit Gaussian and the reconstruction degrades. Tuning
that balance — the $\beta$ in $\beta$-VAE variants — is a core engineering
concern in latent world models.

## A concrete worked example: the OpenVLA action head

Putting it together concretely. OpenVLA (arXiv:2406.09246) outputs 7 consecutive
token positions, each decoded from a 32,000-token vocabulary. In practice, only
256 of those tokens are used as action bins; the rest are masked. The model's
output at position $t$ is a vector of 32,000 logits, which a softmax converts
to a PMF $q(\cdot)$ over the vocabulary. The training target is the
demonstration action $a^{\star}_t$, encoded as a bin index, which is a one-hot
$p$. The loss for that position is:

$$
\ell_t = -\log q(a^{\star}_t).
$$

This is $H(p, q)$ for a one-hot $p$, which is $D_{\mathrm{KL}}(p \,\|\, q) + H(p) = D_{\mathrm{KL}}(p \,\|\, q)$
(since $H(\text{one-hot}) = 0$). The total action loss is $\frac{1}{7}\sum_{t=1}^{7} \ell_t$,
averaged over the 7 dimensions and over the minibatch.

Now consider what happens when the demonstrator is ambiguous — say, the task
"place the cup on the tray" could be completed by moving left first or right
first. The demonstration dataset will contain both trajectories. The one-hot
targets for those trajectories will land in different bins on the first step.
The model, minimizing forward KL, will try to assign nonzero probability to
*both* bins. For a unimodal softmax output, the result is a compromise
distribution spread across two modes — the classic mode-averaging failure.
The observation that this is a consequence of the KL's direction, not a bug in
the architecture, motivates much of Chapter 10's focus on diffusion and flow
heads, which can represent multimodal action distributions explicitly.

## What you need to carry forward

Three facts do most of the work in the remaining chapters.

First, the cross-entropy loss equals the KL divergence when the target is
fixed. Every supervised VLA trains by minimizing a KL from the demonstration
distribution to the model's output distribution, whether or not the paper
frames it that way.

Second, expectations appear everywhere: the expected reward in RL, the expected
loss over a minibatch, the expected reconstruction in the ELBO. The Monte Carlo
estimator ($N = 1$) is almost always what is used in practice, and its variance
is almost always the bottleneck.

Third, the direction of KL divergence is a design decision with downstream
consequences: forward KL (used in supervised imitation) is mode-averaging,
reverse KL (used in some RL variants) is mode-seeking. Choosing between them
is choosing between different failure modes.

The next section puts these tools to work. We will write a complete 50-line
training loop in PyTorch, step through the cross-entropy and MSE losses in
code, and watch the gradient flow backward through a small policy network
from loss to weight update.
