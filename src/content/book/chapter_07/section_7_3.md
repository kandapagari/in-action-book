---
chapter: 7
section: 7.3
title: "PPO in 100 lines"
target_words: 2000
status: draft
prereqs: §7.2 (policy gradients, the log-derivative trick, advantage estimation and GAE, the actor-critic split, and the on-policy staleness that makes each batch good for one update); §5.1 (returns, the objective J(π)); §3.3 (a minibatch PyTorch training loop); §3.2 (KL divergence, expectations over samples)
key_refs:
  - Schulman, Levine, Moritz, Jordan & Abbeel (2015). Trust region policy optimization. ICML.
  - Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
  - Schulman, Moritz, Levine, Jordan & Abbeel (2016). High-dimensional continuous control using generalized advantage estimation. ICLR. arXiv:1506.02438.
---

# 7.3  PPO in 100 lines

Section 7.2 left us with a single sharp problem. The policy gradient is an on-policy expectation, valid only for trajectories drawn from the current $\pi_\theta$. Collect a batch of rollouts, take one gradient step, and the data is stale; in principle every batch buys exactly one update before it must be discarded. That is ruinously sample-inefficient when each rollout costs a simulator episode, and the obvious fix, just take a bigger step, or several steps, on the same batch, is exactly what destabilizes training. Push the policy too far on one batch and it can land in a region where the new action distribution bears no resemblance to the one that generated the data, the advantage estimates become meaningless, and the run collapses into a flat or thrashing return curve that never recovers. Proximal Policy Optimization (PPO; Schulman et al. 2017, arXiv:1707.06347) is the method that lets you squeeze many updates out of each batch *without* falling off that cliff, and it does so with so little machinery that a complete, working implementation fits in about a hundred lines. It is, for that reason, the default first thing to reach for in on-policy deep RL, and the algorithm most likely to be running inside a robotics RL paper you pick up today.

## The step-size problem, stated precisely

To control the step we need to measure it. The natural yardstick is not the change in the weights $\theta$, two networks with very different weights can encode nearly the same policy, and vice versa, but the change in the *policy's output distribution*. PPO's predecessor, Trust Region Policy Optimization (TRPO; Schulman et al. 2015, ICML), made this exact, optimizing the policy improvement subject to a hard constraint that the KL divergence between the new and old action distributions stay below a small threshold. TRPO works well and comes with a monotonic improvement guarantee, but the constraint is enforced with a conjugate-gradient solve and a line search that are awkward to implement and do not slot cleanly into a standard autograd training loop. PPO's contribution is to get most of TRPO's stability from a plain first-order objective that any optimizer can descend, with no second derivatives and no constrained-optimization solver.

The object both methods center on is the *probability ratio* between the new and old policy for an action that was actually taken:

$$
r_t(\theta) \;=\; \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}.
$$

Here $\theta_{\text{old}}$ are the weights that collected the batch, held fixed; $\theta$ is the policy we are currently optimizing. At the start of a batch the two are identical and every $r_t = 1$. As we take update steps on the batch, $r_t$ drifts away from $1$: above $1$ for actions the new policy now favors more than the old one, below for those it favors less. The ratio is precisely how far we have moved on the data we already collected, measured per action, and it is the quantity PPO watches.

## The clipped surrogate objective

The importance-sampling form of the policy gradient says we can estimate the new policy's expected advantage using old data by weighting each sample's advantage $\hat{A}_t$ by the ratio: maximize $\mathbb{E}_t[r_t(\theta)\,\hat{A}_t]$. Left unconstrained this is exactly the trap; the objective rewards driving $r_t$ arbitrarily large whenever $\hat{A}_t > 0$, and the advantage estimate is only trustworthy near $r_t = 1$. PPO's fix is almost crude in its simplicity. It clips the ratio so that, once it leaves a small interval $[1-\epsilon, 1+\epsilon]$, moving it further stops helping the objective:

$$
L^{\text{CLIP}}(\theta) \;=\; \mathbb{E}_t\Bigl[\min\bigl(r_t(\theta)\,\hat{A}_t,\; \operatorname{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\bigr)\Bigr],
$$

with $\epsilon$ typically $0.2$. Read the two cases. When $\hat{A}_t > 0$, the action beat the baseline, the term grows as $r_t$ rises, but the clip caps it at $(1+\epsilon)\hat{A}_t$; past that point the gradient is zero, so the optimizer has no incentive to push the action's probability any higher on this batch. When $\hat{A}_t < 0$, the same logic floors the ratio at $1-\epsilon$, removing the incentive to drive a bad action's probability to zero in one batch. The outer $\min$ is what makes the clip a genuine pessimistic bound rather than a cosmetic one: it ensures that when the ratio has moved in the *unhelpful* direction (for example $r_t < 1$ while $\hat{A}_t > 0$, meaning we accidentally made a good action less likely), the unclipped term is selected and the full corrective gradient flows. The effect is a flat, no-reward plateau in the objective for any action whose probability has already moved "enough," which removes the optimizer's reason to walk off the cliff while still letting it correct mistakes. No KL constraint is solved; the clip approximates the trust region with one line of arithmetic.

In practice many implementations also add an explicit KL term as a diagnostic and an early-stopping trigger: if the measured KL between old and new policy over the batch exceeds a target, stop updating on this batch. But the clip alone is what carries the method.

## The full objective and the loop

PPO trains the actor and critic together, so the loss minimized is the clipped policy objective plus a value-function regression and an entropy bonus:

$$
L(\theta) \;=\; -\,L^{\text{CLIP}}(\theta) \;+\; c_v\, \mathbb{E}_t\bigl[(V_\phi(s_t) - \hat{R}_t)^2\bigr] \;-\; c_e\,\mathbb{E}_t\bigl[\mathcal{H}[\pi_\theta(\cdot \mid s_t)]\bigr].
$$

The first term is the policy objective (negated, since optimizers descend). The second is the critic loss from §7.2, a mean-squared error between the value head's prediction and the empirical return target $\hat{R}_t = \hat{A}_t + V_\phi(s_t)$, weighted by $c_v \approx 0.5$. The third is an entropy bonus, weighted by a small $c_e$ (often $0.0$ to $0.01$): rewarding high entropy keeps the action distribution from collapsing to a near-deterministic policy too early, preserving the exploration that on-policy methods otherwise lose quickly. These three terms, clipped policy gain, value regression, entropy, are the whole of what PPO optimizes.

The surrounding loop is the part worth committing to memory, because its shape is the same in every implementation:

```python
for iteration in range(num_iters):
    # 1. COLLECT: run current policy for T steps across N parallel envs
    obs, acts, logp_old, rews, dones, vals = rollout(policy, envs, T)

    # 2. ESTIMATE: GAE advantages + value targets, then normalize A
    adv, ret = gae(rews, vals, dones, gamma=0.99, lam=0.95)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # 3. OPTIMIZE: K epochs of minibatch SGD on the SAME batch
    for epoch in range(K):                 # K ~ 4-10
        for mb in minibatches(batch, size=64):
            logp, entropy, value = policy.evaluate(mb.obs, mb.acts)
            ratio = (logp - mb.logp_old).exp()
            unclipped = ratio * mb.adv
            clipped   = ratio.clamp(0.8, 1.2) * mb.adv   # eps = 0.2
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss  = 0.5 * (value - mb.ret).pow(2).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
```

That is the algorithm. Step 1 collects a fixed budget of on-policy experience, typically by running $N$ environments in parallel for $T$ steps each (the A3C-style decorrelation trick from §7.2) and recording, critically, the log-probability $\log\pi_{\theta_{\text{old}}}(a_t \mid s_t)$ at collection time; this is the denominator of the ratio, frozen. Step 2 turns rewards and value predictions into GAE advantages (Schulman et al. 2016, arXiv:1506.02438) with the usual $\gamma = 0.99$, $\lambda = 0.95$, and normalizes the advantages to zero mean and unit variance across the batch, a small step that stabilizes the gradient scale enormously. Step 3 is where PPO earns its sample efficiency: it sweeps the *same* batch $K$ times in shuffled minibatches, taking many gradient steps where REINFORCE would take one. The clip is the only thing that makes those repeated steps safe; on the first inner step every ratio is $1$ and PPO is an ordinary advantage policy gradient, but by the last, the clip is actively zeroing the gradient for actions that have moved far enough.

## Why the details are not optional

PPO has a reputation for being finicky, and the reputation is earned but misdirected: the core objective is robust, while a handful of "engineering" choices in the loop above are what actually separate a run that hits expert performance on a MuJoCo control task from one that plateaus. Advantage normalization, observation normalization (tracking a running mean and variance of the inputs and standardizing them), global gradient-norm clipping to $0.5$, orthogonal weight initialization with a small gain on the policy's output layer, and learning-rate annealing are the usual suspects. None of them appear in the objective; all of them matter in practice, sometimes more than the clip parameter itself. Studies that reproduced PPO carefully found that much of its measured advantage over plain policy gradients came from this surrounding scaffolding rather than the clipped objective alone, a useful reminder that in deep RL the algorithm on the page and the algorithm that works are separated by a layer of unglamorous normalization.

A concrete target makes this real. On HalfCheetah-v4, a continuous MuJoCo control task with a 17-dimensional observation and a 6-dimensional torque action, a textbook PPO with the loop above, $N=8$ to $16$ parallel environments, $T \approx 2048$ steps per rollout, $K \approx 10$ epochs, and the normalization tricks in place will climb from a random-flailing return near zero to a competent running gait over a few million environment steps. Strip out advantage normalization or let the gradient norm run free and the same code will often stall. That gap, same objective, different scaffolding, is the single most important practical lesson of this section, and it is why the chapter's hands-on exercise asks you to build PPO on exactly this task.

PPO's strength is also its boundary. Because it is on-policy, it must throw away every batch after its $K$ epochs and collect fresh experience; it cannot reuse the millions of transitions sitting in a replay buffer the way a value-based method can. On a fast simulator, where samples are nearly free and parallel environments are cheap, that trade is fine and PPO's stability wins. On a real robot, where every sample is a slow and possibly destructive interaction, throwing data away is a luxury you cannot afford, and that is precisely the pressure that motivates the off-policy actor-critic methods, which keep and reuse old experience, that we turn to next.
