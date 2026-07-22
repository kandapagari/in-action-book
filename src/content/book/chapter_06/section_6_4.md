---
chapter: 6
section: 6.4
title: "A glance at IRL and adversarial imitation"
target_words: 2000
status: draft
prereqs: §6.2 (BC objective, mode-averaging); §6.3 (compounding error, covariate shift, DAgger); §5.1 (rewards, policies, returns); §5.4 (reward design)
key_refs:
  - Ng & Russell (2000). Algorithms for Inverse Reinforcement Learning. ICML.
  - Abbeel & Ng (2004). Apprenticeship Learning via Inverse Reinforcement Learning. ICML.
  - Ziebart et al. (2008). Maximum Entropy Inverse Reinforcement Learning. AAAI.
  - Ho & Ermon (2016). Generative Adversarial Imitation Learning. NeurIPS.
  - Fu, Luo & Levine (2018). Learning Robust Rewards with Adversarial Inverse Reinforcement Learning (AIRL). ICLR.
---

# A glance at IRL and adversarial imitation

Behavior cloning, and the DAgger machinery built on top of it, share a blind spot: they copy *what* the expert does and never ask *why*. A cloned policy reproduces the expert's actions on states it has seen and flails everywhere else, because it has learned a mapping from observations to actions, not the objective that mapping was serving. Inverse reinforcement learning (IRL) flips the problem around. Instead of fitting the policy directly, it asks what reward function would make this expert's behavior optimal. Recover that reward, and you can hand it to any of the RL machinery from Chapter 5 to produce a policy, one that, in principle, generalizes to states the expert never visited, because it is pursuing the expert's *goal* rather than mimicking the expert's *reflexes*.

This section is a glance, not a manual. IRL and its modern adversarial descendants form a large subfield, and most production VLAs do not use them. But the ideas explain a real limitation of cloning, they connect imitation back to the reward-centric view of Chapter 5, and adversarial imitation in particular keeps resurfacing in robot-learning research. You should know what the words mean and when the approach earns its considerable extra cost.

## Why recover a reward at all

Consider a concrete contrast. An expert demonstrates parking a car: it approaches the spot, reverses, straightens, stops. A BC policy trained on these trajectories learns "when the view looks like *this*, turn the wheel like *that*." Move the spot two meters, change the approach angle, and the policy is off-distribution, §6.3's failure mode. Now suppose instead you could recover the reward the driver was optimizing: something like "end up inside the lines, parallel to the curb, without hitting anything, with minimal maneuvering." That reward is *portable*. It scores any state in any parking lot. Plan or learn a policy against it and you get sensible behavior in configurations no demonstration covered, because the objective, not the trajectory, transfers.

That is the promise. The reward is a vastly more compact and generalizable description of a task than a pile of trajectories, in the same way that "minimize travel time subject to traffic laws" is a more useful description of good driving than ten thousand recorded drives. IRL tries to extract that compact description from the drives.

The catch, and it is a deep one, is that the problem is badly ill-posed. Ng and Russell (2000), the paper that named the field, opened by observing that infinitely many reward functions make any given policy optimal, including the degenerate reward that is zero everywhere, under which *every* policy, the expert's included, is trivially optimal. Demonstrations underdetermine the reward. Every IRL algorithm is, at heart, a different answer to the question "which of the many consistent rewards should we pick?", and the quality of that answer is what separates the methods.

## From feature matching to maximum entropy

The first practical answers came from Abbeel and Ng (2004) under the banner of *apprenticeship learning*. Assume the reward is linear in some features of the state; for driving, features might be "distance from lane center," "speed," "proximity to other cars." A policy's value under such a reward depends only on its *expected feature counts*: how much lane-deviation it accumulates, how fast it tends to go, and so on, in expectation over trajectories. The algorithm then searches for a policy whose expected feature counts match the expert's. If your features capture what matters, a policy that drives with the same average lane-deviation and speed profile as the expert is, for practical purposes, driving like the expert, without anyone ever writing down the weights that trade those features off.

Feature matching left one nagging slack: many policies and many reward weightings match a given set of feature counts, so which do you commit to? Ziebart et al. (2008) supplied the answer that became the field's workhorse: *maximum entropy* IRL. Among all trajectory distributions consistent with the expert's feature counts, pick the one that is otherwise as random, as high-entropy, as possible. This is the same principle that picks the least-committal probability distribution subject to known constraints, imported into trajectory space. It has a clean probabilistic reading: trajectories are assumed to occur with probability proportional to $\exp(\text{reward})$, so high-reward behavior is exponentially more likely but nothing is ever assigned probability zero. MaxEnt IRL handles the suboptimality and noise in real human demonstrations gracefully (the expert is allowed to be imperfect), and it removed the arbitrary tie-breaking that plagued earlier methods. For roughly a decade it was the default, and it still underlies much of what follows.

The practical wall MaxEnt IRL hit is computational. Recovering the reward requires, in its inner loop, solving the *forward* RL problem, finding the optimal policy for the current reward estimate, and doing so repeatedly as the reward is refined. In a small discrete world this is a tabular dynamic-programming sweep (Chapter 5). On a robot with continuous states and unknown dynamics, each inner solve is itself a full, expensive RL run. IRL was, for years, an algorithm that contained reinforcement learning as a subroutine, which made it roughly as hard as RL times the number of reward updates.

## Adversarial imitation: skip the reward, match the distribution

The breakthrough that made imitation-via-objectives practical at scale came from reframing it as a *distribution-matching* problem and borrowing the machinery of generative adversarial networks. Generative Adversarial Imitation Learning, GAIL, Ho and Ermon (2016), is the pivot, and the idea is elegant enough to state in one breath: train a *discriminator* to tell expert state-action pairs apart from the learner's, and train the *policy* to fool it.

The two play the familiar adversarial game. The discriminator $D$ is a classifier outputting the probability that a given $(s, a)$ came from the expert rather than the policy. The policy is rewarded for producing state-action pairs the discriminator mistakes for expert data; concretely, the policy's reward at each step is something like $-\log(1 - D(s,a))$, high when $D$ is fooled. As the policy improves, the discriminator is retrained to find the remaining tells; as the discriminator sharpens, the policy is pushed to close the remaining gaps. At equilibrium the learner's state-action distribution is indistinguishable from the expert's, which is exactly the goal: matching the occupancy of the demonstrations, not echoing individual actions.

```text
initialize policy π, discriminator D
for each iteration:
    roll out π to collect trajectories
    update D to classify expert (s,a) vs. policy (s,a)
    set per-step reward r(s,a) = -log(1 - D(s,a))
    update π with an RL step (e.g., PPO) to maximize that reward
```

Two things make this matter for robotics. First, GAIL never explicitly recovers a reward function; the discriminator *is* the reward, learned and updated on the fly, which sidesteps the ill-posedness that haunted classical IRL. Second, and decisively, the policy update is just an ordinary RL step (PPO, from Chapter 7, is the standard choice). So instead of solving a full RL problem inside every reward update, GAIL interleaves *one* RL step with *one* discriminator step. It is dramatically more sample-efficient in expert demonstrations than behavior cloning; a handful of trajectories can suffice, at the cost of needing many environment interactions for the RL inner loop, which is why GAIL lives mostly in simulation.

The structural payoff against §6.3 is worth naming. BC only ever sees expert states; GAIL's policy is rolled out in the environment, so the discriminator scores the learner's *own* visited states, including the off-distribution ones, and pushes the policy back toward expert-like behavior there. The mechanism that makes covariate shift catastrophic for BC is, for GAIL, simply part of the training loop. It buys robustness to distribution shift in exchange for online interaction.

If you want the recovered reward back, because a discriminator that classifies $(s,a)$ pairs entangles the task objective with the particular dynamics it was trained under, and so transfers poorly to a new robot or a changed environment, Adversarial Inverse Reinforcement Learning (AIRL, Fu, Luo and Levine 2018) restructures the discriminator so that a genuine, dynamics-disentangled reward falls out of it. That reward is portable in the way the parking example wanted: train in one setting, recover the reward, re-optimize in another.

## Where this sits relative to VLAs

Now the honest accounting. Open the training recipe of RT-1 (arXiv:2212.06817), OpenVLA (arXiv:2406.09246), or Octo (arXiv:2405.12213) and you will find no IRL, no discriminator, no adversarial game. They are behavior cloning at scale, for reasons §6.1 laid out. Adversarial imitation requires online environment interaction and a stable two-player optimization, and both are liabilities at foundation-model scale. The GAN-style minimax is notoriously finicky to tune, and "collect a million teleop episodes and fit them offline" is operationally far simpler than "stand up a simulator the policy can safely explore in for millions of steps while two networks chase each other." When demonstrations are abundant, cloning their actions is cheaper and more stable than inferring their intent.

So why spend a section on it? Three reasons. First, IRL is the cleanest statement of what cloning gives up, the objective behind the behavior, and naming that gap sharpens your judgment about when cloning will generalize and when it will not. Second, adversarial distribution matching is the right tool precisely when demonstrations are scarce but a simulator is cheap, the inverse of the foundation-model regime; it recurs in sim-heavy locomotion and dexterous-manipulation research for exactly that reason. Third, the distribution-matching framing, making the learner's occupancy look like the expert's rather than copying actions pointwise, is the conceptual seed of ideas you will meet again when offline RL re-enters the picture, and it is the cleanest bridge from "imitate the expert" to "optimize a reward," which is where the next section's decision lives.

This section deliberately stayed at the level of ideas; the algorithms here each deserve their own chapter and will not get one, because the book's spine runs through cloning, not reward inference. With behavior cloning (§6.2), its compounding-error failure mode and DAgger fix (§6.3), and now reward-inference and adversarial alternatives in view, you are equipped for the practical question that closes the chapter: given a particular dataset and a particular robot, which of behavior cloning, inverse reinforcement learning, and ordinary reinforcement learning should you actually reach for first?
