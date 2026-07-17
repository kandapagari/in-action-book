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

The previous section described a robot policy as a function turning inputs into outputs, a vector in, a vector out. That picture is incomplete. The action OpenVLA emitted in Chapter 2 wasn't a deterministic response to the image; it was the mode of a distribution over 256 discrete bins. The policy learned not "given this image, move the gripper 4.2 mm right" but "given this image, the probability distribution over move-right distances peaks near bin 143." That shift, from functions to distributions, is what this section covers. If §3.1 introduced the spatial language of robot learning, this one introduces its probabilistic language.

## Random variables and distributions

A random variable $X$ is a quantity whose value isn't fixed in advance; you can only describe how likely different values are. For a discrete random variable, one whose outcomes come from a finite or countable set $\mathcal{X}$, the distribution is a probability mass function $p(x) = P(X = x)$, satisfying $p(x) \geq 0$ for all $x$ and $\sum_{x \in \mathcal{X}} p(x) = 1$.

For a continuous random variable, one that can take any value in an interval or in $\mathbb{R}^{n}$, the distribution is a probability density function $p(x)$ satisfying $p(x) \geq 0$ and $\int p(x)\, dx = 1$. A density doesn't assign probability to a single point. Instead, $\int_{A} p(x)\, dx$ gives the probability that $X$ lands somewhere in the set $A$.

The distinction is load-bearing here. RT-1 (Brohan et al., 2022, arXiv:2212.06817) and OpenVLA (Kim et al., 2024, arXiv:2406.09246) discretize each action dimension into 256 bins, so the predicted action is a product of 7 discrete distributions, one PMF per joint, and the loss becomes a sum of 7 cross-entropies, each comparing a PMF to a one-hot target. π0 (Black et al., 2024, arXiv:2410.24164), by contrast, treats the action as a continuous random variable in $\mathbb{R}^{7}$ and learns a flow from Gaussian noise to the target action distribution; its loss never touches a PMF at all. Discrete versus continuous isn't an aesthetic choice. It determines which math you write and which pathologies you end up fighting, a tension we return to in Chapter 10.

A Gaussian (normal) distribution over $\mathbb{R}^{n}$ with mean $\mu$ and covariance $\Sigma$ is:

$$
\mathcal{N}(x;\, \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}}
\exp\!\left(-\tfrac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right).
$$

The exponent runs negative and large when $x$ sits far from $\mu$ in the metric defined by $\Sigma^{-1}$, so high probability mass clusters near the mean. Gaussians are the default prior for continuous action spaces because they're closed under affine transformations, have closed-form entropy, and because minimizing the negative log-likelihood of a Gaussian target with fixed diagonal covariance turns out to be exactly minimizing mean-squared error, the loss you'd write on intuition alone for a regression problem. When Octo (Ghosh et al., 2024, arXiv:2405.12213) conditions a diffusion head on task tokens, the diffusion starts from $\mathcal{N}(0, I)$ and learns to transport that noise toward the target action region; every step of that transport is a perturbation of a Gaussian. Know the Gaussian, and you understand the starting point of most continuous generative action models.

## Expectation

The expectation of a function $f$ under a distribution $p$ is the probability-weighted average of $f$'s values:

$$
\mathbb{E}_{x \sim p}[f(x)] =
\begin{cases}
\displaystyle\sum_{x \in \mathcal{X}} p(x)\, f(x) & \text{discrete} \\[6pt]
\displaystyle\int p(x)\, f(x)\, dx & \text{continuous.}
\end{cases}
$$

In the discrete case, if $p$ is the distribution over action bins OpenVLA predicts and $f(x)$ is the cost of taking action $x$, then $\mathbb{E}[f(X)]$ is the expected cost of sampling from the policy's output. In the continuous case, if $p$ is the stationary distribution induced by some policy interacting with the environment and $f(x) = r(x)$ is a reward function, then $\mathbb{E}_{x \sim p}[r(x)]$ is the average reward, exactly the quantity reinforcement learning maximizes (Chapter 5 formalizes this with Markov decision processes).

Three properties of expectation are worth memorizing, since derivations lean on them constantly.

Linearity: $\mathbb{E}[af(X) + bg(X)] = a\,\mathbb{E}[f(X)] + b\,\mathbb{E}[g(X)]$. This looks obvious, but it means constant factors move in and out of expectation freely, which lets you rearrange loss expressions however you need.

Law of total expectation: $\mathbb{E}_x[f(x)] = \mathbb{E}_y[\mathbb{E}_{x \mid y}[f(x)]]$. Condition on some intermediate variable $y$, compute the inner expectation over $x$ given $y$, then average out $y$. In robotics, $y$ might be the latent state in a world model (Chapter 9), and this identity is how you justify marginalizing over it.

Monte Carlo estimation: $\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x_i)$ for $x_i \overset{\text{iid}}{\sim} p$. The law of large numbers guarantees convergence. In practice, $N = 1$ during training; you sample a single trajectory and use its reward as an unbiased estimator of the expected reward. The high variance of that estimate is the main reason policy-gradient training feels noisy, and the variance-reduction tricks in Chapter 7 (baselines, GAE, advantage normalization) are all interventions on this exact Monte Carlo estimator.

## Entropy and cross-entropy

The entropy of a distribution $p$ measures how spread out it is:

$$
H(p) = -\mathbb{E}_{x \sim p}[\log p(x)] = -\sum_{x} p(x) \log p(x).
$$

A one-hot distribution (all mass at one outcome) has $H = 0$. A uniform distribution over $k$ outcomes has $H = \log k$. In action distributions, low entropy means the policy is confident (one action dominates); high entropy means the policy is uncertain, or deliberately exploratory.

Cross-entropy between a target distribution $p$ and a predicted distribution $q$ is:

$$
H(p, q) = -\mathbb{E}_{x \sim p}[\log q(x)] = -\sum_{x} p(x) \log q(x).
$$

When $p$ is the one-hot distribution over the correct action token, as it is in every discrete-action VLA trained on demonstrations, $H(p, q)$ reduces to $-\log q(a^{\star})$: the negative log-probability assigned to the correct action. That's the cross-entropy loss you saw applied to each of the 7 action dimensions in §2.3. Minimizing it pushes $q$ to assign high probability to the demonstrated action, and this loss dominates the supervised phase of nearly every language-conditioned imitation model from RT-1 (arXiv:2212.06817) through RT-2 (arXiv:2307.15818) through OpenVLA (arXiv:2406.09246).

One subtlety worth flagging: cross-entropy isn't symmetric. $H(p, q) \neq H(q, p)$ in general. The convention starts to matter once you move beyond supervised imitation, for instance when a VLA needs to cover all high-probability actions of a multimodal demonstration distribution rather than just fit one mode. Chapter 6 revisits that asymmetry when discussing why behavior cloning struggles with multimodal data.

## KL divergence

The Kullback-Leibler divergence from $q$ to $p$ (read: how different $q$ is from $p$) is:

$$
D_{\mathrm{KL}}(p \,\|\, q) = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
= H(p, q) - H(p).
$$

That last equality comes from expanding the expectation: $\mathbb{E}_{p}[\log p - \log q] = -H(p) + H(p, q)$, rearranged. Because $D_{\mathrm{KL}} \geq 0$ always (by Jensen's inequality, with equality iff $p = q$), minimizing $H(p, q)$ over $q$ is the same as minimizing $D_{\mathrm{KL}}(p \,\|\, q)$ whenever the target $p$ doesn't depend on $q$, since $H(p)$ is then just a constant. The cross-entropy loss is the KL divergence, shifted by the entropy of the target. Knowing this is more than trivia. It tells you what you're actually doing when you train with cross-entropy: driving the predicted distribution as close as possible to the demonstrated distribution, in the specific asymmetric sense that $D_{\mathrm{KL}}(p \,\|\, q)$ penalizes placing zero probability mass where the target $p$ has nonzero mass. If the teacher only ever demonstrates one action variant, $p$ carries low entropy, and the penalty structure of $D_{\mathrm{KL}}(p \,\|\, q)$ will happily ignore modes of the true (unknown) distribution the teacher never exhibited. Compounding error in behavior cloning, covered in Chapter 6, follows directly from this.

The reverse KL, $D_{\mathrm{KL}}(q \,\|\, p)$, penalizes placing mass where the target $p$ is zero, which produces mode-seeking behavior instead. Variational inference and some policy-optimization methods use the reverse KL; supervised imitation typically uses the forward KL. Which direction you pick is an architectural choice that shapes the failure modes of the resulting policy.

### KL divergence in world models and latent variable models

KL divergence appears explicitly in the training objective of any model with a stochastic latent variable. The standard variational bound (ELBO) for a latent variable model with observations $o$, latent state $z$, and parameters $\theta$ is:

$$
\log p_\theta(o) \geq \underbrace{\mathbb{E}_{z \sim q_\phi(z \mid o)}\!\left[\log p_\theta(o \mid z)\right]}_{\text{reconstruction}}
- \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(z \mid o) \,\|\, p_\theta(z)\right)}_{\text{regularization}}.
$$

The first term rewards the model for explaining the observation using the learned latent. The second regularizes the posterior $q_\phi(z \mid o)$ toward the prior $p_\theta(z)$, keeping the latent from memorizing and ignoring structure. World models such as RSSM (used in DreamerV3, covered in Chapter 9) minimize a variant of this bound; the KL term is what keeps the learned dynamics smooth enough to plan in. Set the KL coefficient to zero and the latent space collapses into a lookup table, losing its generalization ability. Set it too high, and the posterior gets dragged too close to a unit Gaussian, degrading reconstruction. Tuning that balance, the $\beta$ in $\beta$-VAE variants, is a core engineering concern in latent world models.

## A concrete worked example: the OpenVLA action head

Putting this together concretely: OpenVLA (arXiv:2406.09246) outputs 7 consecutive token positions, each decoded from a 32,000-token vocabulary. Only 256 of those tokens actually get used as action bins in practice; the rest stay masked. The model's output at position $t$ is a vector of 32,000 logits, which a softmax converts into a PMF $q(\cdot)$ over the vocabulary. The training target is the demonstration action $a^{\star}_t$, encoded as a bin index, which is a one-hot $p$. The loss for that position is:

$$
\ell_t = -\log q(a^{\star}_t).
$$

This is $H(p, q)$ for a one-hot $p$, which equals $D_{\mathrm{KL}}(p \,\|\, q) + H(p) = D_{\mathrm{KL}}(p \,\|\, q)$ (since $H(\text{one-hot}) = 0$). The total action loss is $\frac{1}{7}\sum_{t=1}^{7} \ell_t$, averaged over the 7 dimensions and over the minibatch.

Now consider what happens when the demonstrator is ambiguous, say the task "place the cup on the tray" could be completed by moving left first or right first. The demonstration dataset ends up with both trajectories. The one-hot targets for those trajectories land in different bins on the first step. The model, minimizing forward KL, tries to assign nonzero probability to both bins at once. For a unimodal softmax output, the result is a compromise distribution spread across two modes, the classic mode-averaging failure. This is a consequence of the KL's direction, not a bug in the architecture, and that observation motivates much of Chapter 10's focus on diffusion and flow heads, which can represent multimodal action distributions explicitly instead of averaging over them.

## What you need to carry forward

Three facts do most of the work in the remaining chapters.

First, cross-entropy loss equals KL divergence when the target is fixed. Every supervised VLA trains by minimizing a KL from the demonstration distribution to the model's output distribution, whether or not the paper frames it that way.

Second, expectations appear everywhere: expected reward in RL, expected loss over a minibatch, expected reconstruction in the ELBO. The Monte Carlo estimator ($N = 1$) is almost always what gets used in practice, and its variance is almost always the bottleneck.

Third, the direction of KL divergence is a design decision with downstream consequences. Forward KL (used in supervised imitation) is mode-averaging. Reverse KL (used in some RL variants) is mode-seeking. Choosing between them means choosing between different failure modes.

The next section puts these tools to work. We'll write a complete 50-line training loop in PyTorch, step through the cross-entropy and MSE losses in code, and watch the gradient flow backward through a small policy network from loss to weight update.
