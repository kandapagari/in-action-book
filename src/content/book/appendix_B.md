---
appendix: B
title: "Probability and information theory"
target_words: 3600
status: draft
prereqs: Appendix A; one semester of calculus; comfort with the word "expectation"
key_refs:
  - Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
  - Cover, T. M., & Thomas, J. A. (2006). Elements of Information Theory (2nd ed.). Wiley.
  - Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. arXiv:1312.6114.
---

# Appendix B.  Probability and information theory

Action models are statistical objects. The policy $\pi(a \mid s)$ is a
conditional distribution; the loss in §3.4 is an expected negative
log-likelihood; the KL divergence appears in PPO's clipping objective,
in the ELBO of a VAE-style world model, and in the contrastive loss
that trained CLIP. The body of the book invokes these objects without
re-deriving them. This appendix is the bench manual: every concept is
stated precisely once, illustrated with one robotics example, and
linked back to the section where it does its real work.

## B.1  Probability spaces, random variables, and distributions

A *probability space* is a triple $(\Omega, \mathcal{F}, P)$: a sample
space $\Omega$, a $\sigma$-algebra $\mathcal{F}$ of subsets of
$\Omega$ called events, and a probability measure $P$ that assigns
each event a number in $[0, 1]$ with $P(\Omega) = 1$. For the body of
this book the measure-theoretic scaffolding can stay implicit; what
matters is that there is a procedure (sometimes physical, sometimes
simulated) that produces outcomes, and a function that assigns
probabilities to sets of those outcomes.

A *random variable* $X : \Omega \to \mathcal{X}$ is a function from
the sample space to a value space. When $\mathcal{X}$ is a finite or
countable set, $X$ is *discrete*; when $\mathcal{X} = \mathbb{R}^{n}$,
it is *continuous*. A discrete random variable is fully described by
its probability mass function $p(x) = P(X = x)$; a continuous one by
its probability density function $p(x)$, where $P(X \in B) = \int_{B}
p(x)\, dx$ for measurable sets $B$. The notation $X \sim p$ means
"$X$ is distributed according to $p$" and is the line you will see at
the top of every algorithm box from §3.3 onward.

Three distributions appear in this book often enough to be worth
naming. The *Bernoulli* $\mathrm{Bern}(\theta)$ describes a coin flip
with bias $\theta$ and is the distribution behind every binary
classification head. The *categorical* $\mathrm{Cat}(\theta_1, \ldots,
\theta_K)$ describes a single draw from $K$ options with probabilities
$\theta_i$; the action-token distribution of RT-1 (arXiv:2212.06817)
and OpenVLA (arXiv:2406.09246) is a categorical over 256 bins per
action dimension. The *multivariate Gaussian* $\mathcal{N}(\mu,
\Sigma)$ with mean $\mu \in \mathbb{R}^{n}$ and positive-definite
covariance $\Sigma \in \mathbb{R}^{n \times n}$ has density

$$
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\!\Big({-\tfrac{1}{2}}
(x - \mu)^\top \Sigma^{-1} (x - \mu)\Big),
$$

and is the workhorse model for sensor noise, dynamics noise, and
the action distribution of every Gaussian policy in §7.

## B.2  Expectation, variance, and the law of large numbers

The *expectation* of a random variable is its probability-weighted
average:

$$
\mathbb{E}[X] = \begin{cases}
\sum_{x} x\, p(x) & \text{discrete}, \\
\int x\, p(x)\, dx & \text{continuous}.
\end{cases}
$$

Expectation is linear: $\mathbb{E}[\alpha X + \beta Y] = \alpha\,
\mathbb{E}[X] + \beta\, \mathbb{E}[Y]$, with no independence
required. This single property is why so much of the math in
reinforcement learning works: the expected return decomposes into a
sum of expected per-step rewards regardless of how the random states
and actions are correlated through the policy. The *variance*
$\mathrm{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^{2}]$ measures
spread. For a vector-valued random variable, the analogue is the
covariance matrix $\Sigma = \mathbb{E}[(X - \mu)(X - \mu)^\top]$,
symmetric and positive semi-definite as Appendix A.6 promised.

The *law of large numbers* says that the empirical mean of $N$
independent samples from $p$ converges to $\mathbb{E}[X]$ as $N \to
\infty$. This is the mathematical license for Monte Carlo estimation:
$\mathbb{E}_{p}[f(X)] \approx \frac{1}{N} \sum_{i=1}^{N} f(X_i)$ with
$X_i$ drawn from $p$. The *central limit theorem* sharpens the law
into a rate: the standard error of the empirical mean scales as
$1/\sqrt{N}$, which is why doubling your training data only reduces
the policy gradient estimator's noise by about 30%. Every "we trained
for 5 seeds and report mean ± std" line in a robot-learning paper is
a finite-sample version of these two theorems.

## B.3  Conditional probability, independence, and Bayes' rule

Two events $A$ and $B$ are *independent* if $P(A \cap B) = P(A) P(B)$.
The conditional probability of $A$ given $B$ is

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)},
$$

defined whenever $P(B) > 0$. *Bayes' rule* is the same identity
rearranged: $P(A \mid B) = P(B \mid A) P(A) / P(B)$. In the body of
the book, Bayes' rule shows up in the IRL discussion of §6.4 (recover
the reward $R$ given demonstrations $\tau$ by computing $P(R \mid
\tau)$), in the diffusion-model derivation of §10.1 (the reverse
process is the conditional density of the cleaner sample given the
noisier one), and implicitly in every Kalman-filter-style state
estimator a real robot runs under its perception stack.

The closely related concept is *conditional independence*: $X$ and $Y$
are conditionally independent given $Z$ if $P(X, Y \mid Z) = P(X \mid
Z) P(Y \mid Z)$. The *Markov property* in §5.1 is precisely the claim
that the future is conditionally independent of the past given the
present state, and it is the single assumption that makes the MDP
formalism tractable.

## B.4  KL divergence and the Kullback-Leibler family

The Kullback-Leibler divergence between two distributions $p$ and $q$
on the same space is

$$
\mathrm{KL}(p \,\|\, q) = \mathbb{E}_{p}\!\left[\log \frac{p(X)}{q(X)}\right]
= \int p(x) \log \frac{p(x)}{q(x)}\, dx.
$$

KL is non-negative and zero only when $p = q$ almost everywhere. It
is not symmetric — $\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$ in
general — and it is not a metric, but it is the *correct* divergence
for almost every information-theoretic argument in machine learning.
Three concrete uses to keep in mind. PPO (Schulman et al., 2017,
arXiv:1707.06347) penalizes $\mathrm{KL}(\pi_{\text{new}} \|
\pi_{\text{old}})$ to prevent destructive policy updates and the size
of that penalty is the single most-tuned hyperparameter of
the algorithm. The VAE objective of §3.4 includes a $\mathrm{KL}(q
\| p)$ term that pushes the encoder distribution toward a Gaussian
prior; Kingma and Welling (2014, arXiv:1312.6114) is the canonical
reference. Diffusion training loss is, after rearrangement, a sum of
per-timestep KL divergences between the true and learned reverse
transitions; the noise-prediction objective most VLAs use is the
"reparameterization" of that KL sum.

A practical note: KL is finite only when $q$ assigns positive
probability everywhere $p$ does. When $q(x) = 0$ but $p(x) > 0$, the
KL is infinite. Numerical implementations therefore add a small
$\epsilon$ inside the log or smooth the distributions; if a training
run's loss suddenly spikes to NaN, "a distribution went to zero in
the support of the other" is the second culprit to check, after
"the learning rate is too high."

## B.5  Entropy and cross-entropy

The *entropy* of a discrete distribution $p$ is

$$
H(p) = -\sum_{x} p(x) \log p(x) = \mathbb{E}_{p}[-\log p(X)].
$$

Entropy is non-negative and is maximized by the uniform distribution
on a given support. It measures the average number of nats (or bits,
if base 2) needed to describe a sample from $p$. In control, high
policy entropy means the policy is exploring; low policy entropy
means it has committed. SAC (Haarnoja et al., 2018, ICML 2018) makes
entropy maximization an explicit term in its objective, which is the
mechanism that keeps it exploring while learning.

The *cross-entropy* between $p$ and $q$ is

$$
H(p, q) = -\sum_{x} p(x) \log q(x) = H(p) + \mathrm{KL}(p \| q).
$$

If $p$ is fixed (as it is when $p$ is the empirical distribution of
training data), minimizing the cross-entropy in $q$ is equivalent to
minimizing $\mathrm{KL}(p \| q)$. This is the equivalence behind every
"cross-entropy loss = log-likelihood loss" claim in §3.4: the
empirical-data distribution plays the role of $p$, the model
distribution plays the role of $q$, and Bayes' optimal predictor is
the $q$ that minimizes $H(p, q)$. Behavior cloning (§6.2) is exactly
cross-entropy minimization between the demonstrator's action
distribution and the policy's action distribution.

## B.6  Likelihood, log-likelihood, and maximum likelihood

A *likelihood* is a probability density read as a function of the
parameters $\theta$ that produced it, with the data held fixed:
$\mathcal{L}(\theta; D) = p_{\theta}(D)$. The *log-likelihood* is its
logarithm, $\ell(\theta; D) = \log p_{\theta}(D)$, and is the
quantity actually optimized in practice — products turn into sums,
underflow disappears, and the derivative is easier. The *maximum
likelihood estimator* (MLE) is $\hat\theta = \arg\max_{\theta}
\ell(\theta; D)$.

Three uses, all already in the book by Chapter 12. The supervised
loss in §3.4 is a negative log-likelihood. The reward of an MDP is
sometimes inferred via maximum likelihood from demonstrations
(maximum-entropy IRL is the classic example, treated lightly in §6.4).
The training objective of every autoregressive language model that
gets repurposed as a VLA backbone — Llama-2 for OpenVLA, PaLI-X for
RT-2 — is the next-token log-likelihood of the training corpus.

Two known failure modes of MLE worth knowing about. First, MLE is
*consistent* (converges to the truth as $N \to \infty$) but not
necessarily efficient at finite $N$, and on small datasets a Bayesian
or regularized estimator may dominate it. Second, MLE is
*model-misspecification-sensitive*: if the chosen family $p_\theta$
does not include the data-generating distribution, MLE finds the
$\theta$ that minimizes $\mathrm{KL}(p_{\text{data}} \| p_\theta)$,
not the truth, and the asymmetry of KL means the result is the
*support-covering* mode rather than the most likely one. This
asymmetry is why mode collapse in GAN-style generators is a different
failure than mode covering in MLE-trained density models.

## B.7  The reparameterization trick and gradients of expectations

A recurring problem in ML is to compute gradients of an expectation
$\mathbb{E}_{p_\theta}[f(X)]$ with respect to $\theta$, the
parameters of the distribution. Two strategies dominate. The *score-
function estimator* (used in REINFORCE and policy gradient, §7.2)
rewrites the gradient as

$$
\nabla_\theta \mathbb{E}_{p_\theta}[f(X)] = \mathbb{E}_{p_\theta}\!\left[
f(X)\, \nabla_\theta \log p_\theta(X)\right].
$$

This is correct in expectation, but has high variance and is the
reason policy gradient methods need millions of environment steps.

The *reparameterization trick* applies when $X$ can be written as a
deterministic function of $\theta$ and an auxiliary noise variable
$\epsilon \sim p(\epsilon)$: $X = g_\theta(\epsilon)$. Then

$$
\nabla_\theta \mathbb{E}_{p_\theta}[f(X)] = \mathbb{E}_{p(\epsilon)}\!\left[
\nabla_\theta f(g_\theta(\epsilon))\right].
$$

The gradient slides inside the expectation because $\epsilon$ does
not depend on $\theta$. For a Gaussian with parameters $\mu, \sigma$,
the reparameterization is $X = \mu + \sigma\, \epsilon$ with $\epsilon
\sim \mathcal{N}(0, 1)$. This trick is what makes the VAE trainable
in §3.4 and what makes the noise-injection in Diffusion Policy
(§10.2) differentiable end-to-end. When you can reparameterize,
you should; the variance reduction over the score-function
estimator is dramatic.

## B.8  The ELBO and variational inference

For a latent-variable model $p_\theta(x, z) = p_\theta(x \mid z)
p(z)$, the log-likelihood of an observation $x$ is

$$
\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z)\, dz,
$$

an integral that is intractable in essentially every interesting case.
The variational trick introduces an *approximate posterior*
$q_\phi(z \mid x)$ and decomposes the log-likelihood as

$$
\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)]
- \mathrm{KL}(q_\phi(z \mid x) \| p(z))}_{\text{ELBO}(\theta, \phi; x)}
+ \mathrm{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x)).
$$

The second KL on the right is non-negative, so the ELBO (evidence
lower bound) is a *lower bound* on the log-likelihood. Maximizing the
ELBO simultaneously fits the model ($\theta$) and tightens the
variational approximation ($\phi$). This is the entire machinery of
the VAE (Kingma and Welling, 2014, arXiv:1312.6114) and the latent
dynamics model in DreamerV3 (Hafner et al., 2023, arXiv:2301.04104)
that Chapter 9 treats in detail.

## B.9  Information-theoretic asides used in the book

A handful of additional information-theoretic notions appear in the
body of the book and are worth a one-line gloss.

*Mutual information* $I(X; Y) = \mathrm{KL}(p(x, y) \| p(x) p(y))$
measures how much knowing one variable tells you about the other.
Contrastive losses (CLIP, InfoNCE) are lower-bound estimators of
mutual information.

*Differential entropy* of a continuous random variable extends the
discrete entropy formula with an integral and a sign convention. A
Gaussian with covariance $\Sigma$ has differential entropy $\frac{1}{2}
\log((2\pi e)^{n} |\Sigma|)$, which is why "maximum-entropy Gaussian
prior" is the default well-behaved prior.

*Jensen's inequality* says that for any convex function $\phi$ and
random variable $X$, $\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]$
(reversed for concave $\phi$). This is the inequality behind the ELBO
derivation, behind every "lower-bound argument" in variational
inference, and behind the proof that KL is non-negative.

## B.10  A practitioner's checklist

Three reminders for the kind of bug that probability theory creates in
practice. First, never let a probability go to zero in code; clamp
densities away from zero by a tiny $\epsilon$, or work in log-space
throughout. Second, never use a `softmax(logits)` followed by `log`;
use the numerically stable `log_softmax` your framework provides.
Third, when you sample from a Gaussian for any reason — VAE, diffusion,
policy noise — verify the covariance you are using is the one you
mean: a 7-vector "with variance 0.1" might mean covariance $0.1 \cdot
I$ or covariance $0.1^{2} \cdot I = 0.01 \cdot I$, and the field uses
both conventions inconsistently. Read the data loader; do not trust
the paper.

That, plus the linear-algebra background of Appendix A and the
PyTorch glue of Appendix C, is the math the rest of the book leans on.
