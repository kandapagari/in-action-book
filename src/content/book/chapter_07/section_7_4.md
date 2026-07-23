---
chapter: 7
section: 7.4
title: "Off-policy actor-critic: DDPG, TD3, SAC"
target_words: 2000
status: draft
prereqs: §7.3 (PPO's on-policy constraint — every batch is discarded after a few epochs — and why that is unaffordable when samples are expensive); §7.2 (the actor-critic split, policy gradients, the deterministic vs. stochastic policy distinction); §7.1 (the argmax-over-continuous-actions obstacle, the replay buffer and target network from DQN); §5.3 (Q-learning, the Bellman target, exploration); §3.4 (loss families)
key_refs:
  - Lillicrap, Hunt, Pritzel, Heess, Erez, Tassa, Silver & Wierstra (2016). Continuous control with deep reinforcement learning (DDPG). ICLR. arXiv:1509.02971.
  - Silver, Lever, Heess, Degris, Wierstra & Riedmiller (2014). Deterministic policy gradient algorithms (DPG). ICML.
  - Fujimoto, van Hoof & Meger (2018). Addressing function approximation error in actor-critic methods (TD3). ICML. arXiv:1802.09477.
  - Haarnoja, Zhou, Abbeel & Levine (2018). Soft actor-critic: off-policy maximum entropy deep RL with a stochastic actor (SAC). ICML. arXiv:1801.01290.
---

# 7.4  Off-policy actor-critic: DDPG, TD3, SAC

Section 7.3 ended on a complaint disguised as a fact. PPO is on-policy: after it sweeps a batch for its handful of epochs, the data is stale and must be thrown away. On a fast simulator that is fine, because samples are nearly free. On a real robot it is close to fatal; every transition is a slow, wear-inducing, possibly destructive physical interaction, and collecting the millions of them PPO wants is not an option. What we need is a method that keeps every transition it has ever seen and squeezes many gradient updates out of each one. That is the promise of *off-policy* learning, and the family that delivers it for continuous control is the off-policy actor-critic: DDPG, its hardened successor TD3, and the maximum-entropy variant SAC that is now the default reach-for algorithm when sample efficiency matters.

## The idea: marry Q-learning to a policy network

Recall the two obstacles. Q-learning (§5.3, §7.1) is beautifully off-policy; it learns from a replay buffer of transitions collected by any policy, because its Bellman target $r + \gamma \max_{a'} Q(s', a')$ depends only on the transition, not on who chose the action. But that $\max_{a'}$ is an optimization over the action space, and in a continuous space, a seven-joint arm with a six-dimensional torque vector, you cannot enumerate actions to maximize over them. Policy gradients (§7.2) sidestep the argmax by parameterizing the policy directly, but the vanilla version is on-policy and sample-hungry.

Deterministic Policy Gradient (DPG; Silver et al. 2014) and its deep incarnation DDPG (Lillicrap et al. 2016, arXiv:1509.02971) fuse the two. The trick is to replace the intractable $\max_{a'} Q(s', a')$ with a *learned* maximizer: train a deterministic actor network $\mu_\phi(s)$ whose job is to output the action that maximizes the critic. Then the Bellman target becomes

$$
y \;=\; r \;+\; \gamma\, Q_\theta\bigl(s',\, \mu_\phi(s')\bigr),
$$

with no maximization to solve; the actor *is* the approximate argmax. The critic $Q_\theta$ is trained by ordinary temporal-difference regression toward $y$ (the same squared-error Bellman loss as DQN). The actor is trained to push its output in whatever direction makes the critic's estimate larger, which is just gradient ascent on the critic through the action input:

$$
\nabla_\phi J \;=\; \mathbb{E}_{s \sim \mathcal{D}}\Bigl[\nabla_a Q_\theta(s, a)\big|_{a=\mu_\phi(s)} \,\nabla_\phi \mu_\phi(s)\Bigr].
$$

Read that chain rule carefully, because it is the whole method. The gradient of the critic with respect to the action, $\nabla_a Q$, says "to raise the Q-value, nudge the action this way"; the actor Jacobian $\nabla_\phi \mu$ then translates that desired action-nudge into a weight update. Because both terms are computed on states $s$ sampled from the replay buffer $\mathcal{D}$, states collected by *old* policies, the whole scheme is off-policy. DDPG inherits DQN's two stabilizers wholesale: a replay buffer that decorrelates samples and lets each transition be reused many times, and slowly-updated *target networks* $Q_{\theta'}$ and $\mu_{\phi'}$ that supply the bootstrap target $y$ so the regression is not chasing a target that moves in lockstep with the weights being trained. DDPG's target networks use a soft update, $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ with $\tau \approx 0.005$, rather than DQN's periodic hard copy.

Exploration needs a separate mechanism, because a deterministic actor left alone always emits the same action in a given state and never tries anything new. DDPG handles this by adding noise to the actor's output at collection time, $a = \mu_\phi(s) + \mathcal{N}$, and storing the noisy action in the buffer. The original paper used temporally correlated Ornstein-Uhlenbeck noise; later work found plain Gaussian noise works as well or better, which is one less moving part.

## TD3: DDPG that does not lie to itself

DDPG works, when it works, but it earned a reputation for being fragile: wildly sensitive to hyperparameters, prone to a return curve that climbs promisingly and then collapses. The diagnosis, made precise by Fujimoto et al. (2018, arXiv:1802.09477), is *overestimation bias*. The actor is trained to maximize the critic, so it actively seeks out whatever actions the critic happens to overvalue; those errors then feed straight into the next Bellman target through $Q(s', \mu(s'))$, get bootstrapped forward, and compound. The critic, in effect, believes its own most optimistic mistakes, and the actor chases them off a cliff.

TD3, Twin Delayed DDPG, is DDPG plus three targeted fixes, and it is worth knowing them individually because each addresses a distinct failure and each shows up again in SAC.

The first is *clipped double-Q learning*. Train two independent critics $Q_{\theta_1}, Q_{\theta_2}$ and form the Bellman target using the *minimum* of the two:

$$
y \;=\; r \;+\; \gamma \min_{i=1,2} Q_{\theta_i'}\bigl(s', \tilde{a}\bigr).
$$

Taking the min is a deliberate pessimism. Two critics will not overestimate the same action in the same way, so the smaller of the two is a conservative estimate that systematically counteracts the upward bias. It can underestimate instead, but underestimation does not get chased and amplified the way overestimation does, so the trade is heavily favorable.

The second is *target policy smoothing*. The action $\tilde{a}$ in the target above is not $\mu_{\phi'}(s')$ exactly but that action plus a small clipped noise: $\tilde{a} = \mu_{\phi'}(s') + \operatorname{clip}(\epsilon, -c, c)$. The reasoning is that a good action should have a similar value to its near neighbors; without smoothing, a critic can develop a sharp, spurious spike at one action that the deterministic actor then exploits. Averaging the target over a little noise around the action regularizes those spikes away. This is a regularizer on the *value target*, not on exploration, a subtle but important distinction.

The third is *delayed policy updates*. Update the actor (and the target networks) less often than the critics, typically once every two critic updates. The point is to let the critic settle toward an accurate estimate before the actor takes a step that depends on it; chasing a noisy, half-trained critic is what produces the thrashing. The three fixes together turn DDPG from a method you coax into working into one that trains reliably on standard MuJoCo benchmarks, and TD3 remains a strong, simple baseline.

## SAC: reward entropy, not just return

Soft Actor-Critic (Haarnoja et al. 2018, arXiv:1801.01290) keeps the off-policy, replay-buffer, twin-critic skeleton but changes the objective itself. Where every method so far maximizes expected return, SAC maximizes return *plus* the entropy of the policy:

$$
J(\pi) \;=\; \sum_t \mathbb{E}\bigl[\, r_t \;+\; \alpha\, \mathcal{H}\bigl(\pi(\cdot \mid s_t)\bigr)\bigr].
$$

The temperature $\alpha$ sets the exchange rate between reward and randomness. The reframing has real consequences. The policy is now *stochastic*, it outputs a distribution, typically a squashed Gaussian, rather than a single action, and it is rewarded for staying as random as it can while still collecting return. This bakes exploration directly into the objective instead of bolting it on as injected noise the way DDPG and TD3 do: the agent explores because spreading probability mass is literally part of what it is maximizing. It also tends to learn more robust policies, because a high-entropy policy that still succeeds cannot be relying on one brittle sequence of actions.

Mechanically, SAC borrows TD3's clipped double-Q (it had a version of the idea concurrently) and adds the entropy bonus to the critic's target:

$$
y \;=\; r \;+\; \gamma\Bigl(\min_{i} Q_{\theta_i'}(s', a') \;-\; \alpha \log \pi_\phi(a' \mid s')\Bigr), \qquad a' \sim \pi_\phi(\cdot \mid s').
$$

The actor is trained with the reparameterization trick, sample $a = \tanh(\mu_\phi(s) + \sigma_\phi(s)\odot\xi)$ with $\xi \sim \mathcal{N}(0,I)$, so the gradient of the expected (Q minus log-probability) objective flows back through the sampled action into the network, exactly the pathwise gradient used in §3.2's discussion of reparameterization. The single most consequential practical refinement came shortly after: rather than tune $\alpha$ by hand, make it *automatic* by treating it as a dual variable and adjusting it to hold the policy's entropy near a target value (a reasonable default target is $-\dim(a)$, the negative of the action dimension). With automatic temperature, SAC has remarkably few knobs to turn, which is much of why it became the default.

The following sketch shows the off-policy update loop the three methods share; the bracketed comments mark where they diverge.

```python
for step in range(total_steps):
    a = actor.act(s, explore=True)        # DDPG/TD3: mu(s)+noise; SAC: sample
    s2, r, done = env.step(a)
    buffer.add(s, a, r, s2, done)         # off-policy: keep everything
    s = env.reset() if done else s2

    batch = buffer.sample(256)            # reuse old transitions
    with torch.no_grad():
        a2 = target_actor(batch.s2)       # SAC: sample + entropy term
        a2 = a2 + clip_noise(a2)          # TD3 only: target smoothing
        q_tgt = torch.min(qt1(batch.s2, a2), qt2(batch.s2, a2))   # twin min
        y = batch.r + gamma * (1 - batch.done) * q_tgt
    update_critics(batch, y)              # TD-regression for Q1, Q2

    if step % policy_delay == 0:          # TD3: delayed; SAC/DDPG: every step
        update_actor(batch)               # ascend Q through the action
        soft_update(targets, tau=0.005)
```

## How to choose, and what comes next

The pattern across the three is a steady accretion of pessimism and self-regularization on top of one core trick, a learned actor standing in for the continuous argmax. DDPG is the foundational idea but fragile in practice. TD3 is DDPG made trustworthy with three small, well-motivated fixes, and it is a fine choice when you want a deterministic policy and minimal conceptual overhead. SAC is the one to default to: its maximum-entropy objective gives principled exploration and robustness, its automatic temperature removes the most painful hyperparameter, and on continuous-control benchmarks it is reliably as sample-efficient as or better than TD3. All three crush PPO on sample efficiency, often by an order of magnitude in environment steps to reach a given performance, which is exactly the property you want when those steps are expensive.

That sample efficiency is why off-policy actor-critic, and SAC in particular, is the workhorse behind much of the learning-on-real-hardware literature, and why its replay-buffer logic reappears inside the offline-RL and Q-learning-flavored components of some later action-generation methods. It also sets up the central tension of robot RL: even SAC's "order of magnitude fewer samples" is still tens of thousands of real interactions, which is tens of thousands too many for most robots. The usual escape, train in a simulator where samples are cheap, then transfer to hardware, brings its own problem, the reality gap, which §7.5 takes up under the heading of domain randomization.
