---
chapter: 7
section: 7.2
title: "Policy gradients and the variance problem"
target_words: 2000
status: draft
prereqs: §7.1 (the argmax-over-actions obstacle that value-only methods hit in continuous spaces); §5.1 (policies, returns, the objective J(π)); §3.1 (gradients, the chain rule); §3.2 (expectations, sampling, why an estimator can be unbiased but high-variance); §3.3 (a PyTorch training loop)
key_refs:
  - Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229-256.
  - Sutton, McAllester, Singh & Mansour (2000). Policy gradient methods for reinforcement learning with function approximation. NeurIPS.
  - Konda & Tsitsiklis (2000). Actor-critic algorithms. NeurIPS.
  - Schulman, Moritz, Levine, Jordan & Abbeel (2016). High-dimensional continuous control using generalized advantage estimation. ICLR. arXiv:1506.02438.
  - Mnih et al. (2016). Asynchronous methods for deep reinforcement learning. ICML. arXiv:1602.01783.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 13. MIT Press.
---

# 7.2  Policy gradients and the variance problem

Section 7.1 ended on an obstacle. A Q-network acts by computing
$\arg\max_a Q_\theta(s,a)$, and for a robot whose action is a
seven-dimensional vector of joint velocities, that argmax is an
optimization problem solved afresh at every control step. Enumerating
actions is hopeless when there are infinitely many of them. The fix in
this section is to stop learning values and then reading off a policy,
and instead parameterize the policy directly: a network
$\pi_\theta(a \mid s)$ that takes a state and emits an action — or, more
usefully, a distribution over actions — with no inner maximization
anywhere. For a continuous action space the network outputs the
parameters of a distribution, typically the mean and standard deviation
of a Gaussian, $\pi_\theta(a\mid s) = \mathcal{N}\bigl(\mu_\theta(s),
\sigma_\theta(s)\bigr)$; sampling from it is one cheap forward pass, and
acting requires no argmax at all.

The question this raises is the one the rest of the section answers. If
the policy is just a network with weights $\theta$, and we want to
maximize expected return, can we get a gradient of the return with
respect to $\theta$ and do ordinary gradient ascent? The answer is yes,
the gradient has a clean closed form, and the catch — the reason policy
gradients are delicate where supervised learning is not — is that the
obvious unbiased estimator of that gradient has enormous variance. Most
of the engineering in modern policy-gradient methods, including the PPO
of §7.3, is variance reduction.

## The objective and the log-derivative trick

Write a trajectory $\tau = (s_0, a_0, s_1, a_1, \dots)$ and its return
$R(\tau) = \sum_t \gamma^t r_t$. The policy induces a distribution over
trajectories, $p_\theta(\tau)$, and the objective is the expected
return,

$$
J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\bigl[R(\tau)\bigr].
$$

We want $\nabla_\theta J$. The difficulty is that $\theta$ sits inside
the distribution we are averaging over, not inside the thing being
averaged — moving $\theta$ changes *which trajectories appear*, not
their returns, and you cannot differentiate a sum of fixed numbers with
respect to a parameter that only reweights them, at least not directly.
The way through is an identity sometimes called the log-derivative
trick or the score-function estimator. For any distribution,
$\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log
p_\theta(\tau)$, simply because $\nabla \log p = (\nabla p)/p$. Substitute
it into $\nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau)\,
d\tau$ and the integral folds back into an expectation:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\bigl[\nabla_\theta \log p_\theta(\tau)\, R(\tau)\bigr].
$$

This is already useful, because $\log p_\theta(\tau)$ decomposes. The
trajectory probability is the product of the (unknown) environment
dynamics and the (known) policy: $p_\theta(\tau) = p(s_0)\prod_t
p(s_{t+1}\mid s_t, a_t)\,\pi_\theta(a_t \mid s_t)$. Taking the log turns
the product into a sum, and every term that does not depend on $\theta$
— the initial-state distribution, every dynamics factor — has zero
gradient and drops out. What survives is only the policy:

$$
\nabla_\theta J(\theta) \;=\; \mathbb{E}_{\tau \sim p_\theta}\Bigl[\Bigl(\textstyle\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\Bigr)\, R(\tau)\Bigr].
$$

This is the policy gradient theorem (Sutton et al. 2000), and the
remarkable thing about it is the absence: nowhere does
$\nabla_\theta p(s_{t+1}\mid s_t, a_t)$ appear. We never need to know,
or differentiate, the dynamics of the robot or the simulator. The
gradient is an expectation over trajectories the current policy
generates, of a quantity we can compute entirely from the policy's own
output. We estimate it by running the policy, collecting trajectories,
and averaging — the same Monte Carlo idea from §3.2.

## REINFORCE, and why it is a starting point and not an answer

The estimator that drops out of the theorem, with the return-to-go
refinement that an action cannot influence rewards that already
happened, is Williams' (1992) REINFORCE:

$$
\hat{g} \;=\; \frac{1}{N}\sum_{i=1}^{N} \sum_t \nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)\, G_t^i,
\qquad G_t = \sum_{k \ge t} \gamma^{k-t} r_k.
$$

Read the term $\nabla_\theta \log \pi_\theta(a_t\mid s_t)\, G_t$ as a
weighted likelihood update and the intuition becomes plain. The
gradient $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ points in the
direction that makes the action actually taken more probable. Multiply
it by the return that followed: if the outcome was good ($G_t$ large
and positive), take a step that makes that action more likely; if it
was bad, step the other way. The policy is nudged toward actions that
preceded high return and away from those that preceded low return. No
value function, no argmax, no model — just "do more of what worked."

In a PyTorch loop this is three lines once trajectories are collected,
and the structure is worth seeing because it makes the connection to
ordinary supervised training concrete:

```python
# logp: log pi_theta(a_t | s_t) for each step, with grad
# returns: G_t (return-to-go), detached, shape [T]
loss = -(logp * returns).mean()      # ascent on J => descent on -J
opt.zero_grad(); loss.backward(); opt.step()
```

It is a weighted negative-log-likelihood: the same loss as supervised
classification, except the "labels" are the agent's own actions and
each example is weighted by how well things went. That framing — policy
gradient as supervised imitation of your own lucky rollouts — is the
thread that ties this chapter to the behavior cloning of Chapter 6 and,
much later, to how VLAs are trained.

REINFORCE is unbiased: in expectation $\hat{g}$ equals the true
gradient. But unbiased is not the same as usable. The estimator's
variance is large enough that on most control tasks plain REINFORCE
either learns glacially or not at all, and understanding why is the
real content of this section.

## The variance problem

Three things make $\hat{g}$ noisy, and they compound. First, $G_t$ is a
sum of many random rewards over a long horizon, so its own variance
grows with episode length — a single rollout that happened to end well
or badly swings the whole estimate. Second, the credit assignment is
indiscriminate: the weight on $\nabla \log \pi_\theta(a_t\mid s_t)$ is
the *entire* return-to-go $G_t$, which lumps together the consequences
of action $a_t$ with a great deal of reward that $a_t$ had nothing to
do with. The signal about any one action is buried in noise from all
the others. Third — and this is the deep one — the scale of $G_t$ is
arbitrary in a way that corrupts the direction of the update.

The third point deserves a concrete example, because it is the cleanest
motivation for everything that follows. Suppose every trajectory in a
task earns a return between $+100$ and $+110$. The good actions and the
bad actions are now both multiplied by a large positive number, so
REINFORCE makes *all* of them more probable, merely by different
amounts. The gradient still points the right way on average, but the
useful signal is the $\pm 5$ spread riding on top of a $+105$ pedestal,
and the pedestal contributes pure variance. Shift every reward down by
$105$ and the task is unchanged — the optimal policy is identical — yet
the estimator behaves completely differently. An estimator whose
quality depends on an additive constant that does not change the problem
is one we should not trust as written.

## Baselines and the advantage

The repair follows directly from the example: subtract off the
pedestal. For any function $b(s)$ that depends on the state but not the
action, one can show that

$$
\mathbb{E}_{a \sim \pi_\theta}\bigl[\nabla_\theta \log \pi_\theta(a\mid s)\, b(s)\bigr] \;=\; 0,
$$

because $b(s)$ pulls out of the action-expectation and what remains,
$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a\mid s)] = \nabla_\theta
\sum_a \pi_\theta(a\mid s) = \nabla_\theta 1 = 0$. So we may subtract any
such *baseline* from the return without changing the gradient in
expectation — it stays unbiased — while changing the variance, often
dramatically. The best simple choice of $b(s)$ is the state-value
function $V(s)$, the expected return from $s$ under the current policy.
Subtracting it replaces $G_t$ with $G_t - V(s_t)$, which estimates the
*advantage*

$$
A(s_t, a_t) \;=\; Q(s_t, a_t) - V(s_t),
$$

the amount by which taking $a_t$ beat the average action from $s_t$.
Advantage is exactly the quantity the example was crying out for:
positive when an action was better than the state's baseline
expectation, negative when worse, and invariant to the $+105$ pedestal.
Now "do more of what worked" means "do more of what worked *better than
expected*," which is the signal we actually wanted.

## Actor-critic: learn the baseline

We do not know $V(s)$, so we learn it — with exactly the
value-regression machinery of §7.1. This is the *actor-critic*
architecture (Konda & Tsitsiklis 2000): an *actor* $\pi_\theta$ that is
updated by the policy gradient, and a *critic* $V_\phi$ that is trained
by TD regression to predict returns and supplies the baseline the actor
needs. The two learn together, the critic chasing the actor's current
return distribution while the actor leans on the critic to tell good
actions from merely good situations. Note the two halves descend from
the two branches of this chapter: the critic is the value learning of
§7.1, the actor is the policy gradient of this section. Every method in
§7.4 is a variation on how to wire them together.

Using $V_\phi$ also lets us trade variance for a little bias. Instead of
the full Monte Carlo $G_t$ (unbiased, high variance) we can use a
short-horizon, bootstrapped advantage such as the one-step TD residual
$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ (lower variance,
biased because $V_\phi$ is imperfect). Generalized advantage estimation
(GAE; Schulman et al. 2016, arXiv:1506.02438) makes this a dial: an
exponentially weighted average of $n$-step residuals with a parameter
$\lambda \in [0,1]$ that interpolates from the high-bias one-step
estimate at $\lambda = 0$ to the high-variance Monte Carlo estimate at
$\lambda = 1$. In practice $\lambda \approx 0.95$ is a near-universal
default, and GAE is the advantage estimator inside virtually every
modern on-policy method, including the PPO of the next section. The
asynchronous actor-critic A3C (Mnih et al. 2016, arXiv:1602.01783)
added a second variance-reducing trick still in use — running many
parallel actors so each gradient is averaged over decorrelated
experience, the on-policy analogue of §7.1's replay buffer.

None of this changes the fundamental shape of the update. We still
ascend $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,
\hat{A}_t]$; we have only replaced the raw, noisy $G_t$ with a
centered, lower-variance advantage estimate $\hat{A}_t$. That single
substitution — return-to-go becomes advantage — is the difference
between an estimator that is technically correct and one that trains a
robot.

One sharp edge remains, and §7.3 exists to file it down. The policy
gradient is an *on-policy* expectation: it is only valid for
trajectories drawn from the current $\pi_\theta$. The moment we take a
gradient step, the data we collected is stale, so each batch of
rollouts buys us exactly one update before it must be thrown away. Push
that one step too hard and the policy can collapse; too softly and
learning crawls. Managing the size of that step, so we extract as much
improvement as possible from each precious batch of on-policy data
without falling off a cliff, is precisely the problem PPO solves, and it
is where we turn next.
