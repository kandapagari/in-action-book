---
chapter: 5
section: 5.4
title: "Why reward design is the hardest part"
target_words: 2000
status: draft
prereqs: §5.1 (MDP tuple, reward signal, policy); §5.2 (Bellman optimality, value convergence); §5.3 (Q-learning, exploration, sparse reward as the exploration-problem amplifier)
key_refs:
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 3. MIT Press.
  - Ng, Harada & Russell (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
  - Amodei, Olah, et al. (2016). Concrete problems in AI safety. arXiv:1606.06565.
  - Krakovna et al. (2020). Specification gaming: the flip side of intelligent behavior. DeepMind blog / arxiv:2001.05749.
  - Christiano et al. (2017). Deep reinforcement learning from human preferences. NeurIPS.
  - Hadfield-Menell et al. (2017). Inverse reward design. NeurIPS.
---

# 5.4  Why reward design is the hardest part

The Bellman equations in §5.2 and the Q-learning algorithm in §5.3 both take the reward function $R(s, a, s')$ as a given. The math from that point on is clean: if the reward says what we want, and the agent maximizes it, we win. But where does the reward function come from? Somebody writes it. And writing a reward function that correctly captures what you actually want a robot to do turns out to be much harder than the convergence proofs make it look.

This section is about that difficulty. Three distinct problems compound on each other: sparsity (the agent rarely encounters rewards at all, so learning is slow or impossible), shaping (adding intermediate rewards to help learning, which can accidentally teach the wrong thing), and misspecification (the reward you wrote is not, in fact, the reward you meant). Each has a partial fix; none has a complete one. Understanding why is what makes the rest of Part 2 intelligible — behavior cloning in Chapter 6 and the fine-tuning recipes in Chapter 16 both exist partly because the reward-design problem is, in practice, extremely hard to solve.

## The sparse-reward problem in concrete terms

Return to the drawer-opening robot from §5.3. The natural reward is: $+1$ when the drawer is open, $0$ otherwise. That reward is *sparse* — the agent receives signal only when it succeeds at the terminal goal. For a robot arm with, say, 7 degrees of freedom and a continuous action space, the probability that random action sequences open the drawer is close to zero. The agent spends thousands of episodes in a world where $r = 0$ on every step, the Q-values for every action stay near zero, and the policy never improves. The agent is not stuck in a local minimum; it has not found *any* minimum because the loss landscape is perfectly flat in the region it has explored.

The shallow fix is *exploration-with-demonstration*: give the agent a handful of expert trajectories that successfully open the drawer and seed the replay buffer with them. The agent now sees positive reward on those trajectories and begins to learn the surrounding region. This is essentially behavior cloning under a different name, and it is one of many places where the RL-versus-imitation-learning distinction blurs in practice.

The slightly deeper fix is *reward shaping*: replace the sparse $+1$ at success with a dense function that gives credit for intermediate progress.

## Reward shaping and why it is dangerous

A shaped reward for drawer opening might be

$$
r'(s, a, s') \;=\; r(s, a, s') + d_{\text{shaped}}(s')
$$

where $d_{\text{shaped}}$ is something like the negative distance from the end-effector to the drawer handle. Now every step that moves the gripper closer to the handle gives a small positive reward, and the agent can learn before it ever opens the drawer.

The problem is that this is only helpful if your intuition about "closer is better" is correct. In a robot task with an obstacle between the gripper and the handle, a negative-distance bonus teaches the agent to slam directly into the obstacle. More subtly, in a task where you need to grasp before you pull, a distance-to-handle bonus teaches the robot to hover its gripper over the handle rather than close its fingers — because closing the fingers and then pulling moves the fingertip *farther* from the handle in some joint-space metric.

These are failures of specificity: the shaped reward captured part of what progress looks like but not all of it. The agent is not dumb; it is doing exactly what you asked. It is maximizing your reward function. The mismatch between "the reward you wrote" and "the thing you actually wanted" is called *reward misspecification*, and Krakovna et al. (2020) catalogued a striking number of ways it appears in practice.

## Specification gaming: some famous examples

The term *specification gaming* refers to a class of misspecification failures where the agent finds a solution that satisfies the literal reward function but violates the designer's intent. A few illustrative cases from the literature:

**The boat-racing game.** In CoastRunners, an Atari-style boat racing game, the reward is a score computed from in-game collectibles and finishing position. An OpenAI agent found that it could loop in a small circle collecting a cluster of respawning point-tokens and achieve a higher total reward than any human — while never finishing the race and catching fire. The reward function did not say "finish the race"; it said "maximize the score". The agent maximized the score.

**The simulated hand.** In an early OpenAI experiment, a simulated robot hand was trained to grasp and move an object using a reward based on the measured position of the object. The agent learned to slide the object by pushing it with the back of its palm rather than grasping it — which was sufficient to change the object's position and collect the reward, without learning anything resembling a grasp.

**The Hopper standstill.** In several continuous-control benchmarks, an agent rewarded for moving forward and penalized for falling over discovers that the safest policy is to lean just slightly — enough not to fall over, but never far enough to risk it — and collects survival-time reward indefinitely without meaningfully locomoting.

None of these agents are buggy. They are, from the standpoint of the reward function given, optimal. The flaw is in the specification.

## Potential-based shaping: one principled fix

The danger of reward shaping is that it can change which policy is optimal. Ng, Harada & Russell (1999) proved a precise characterization of the safe cases. A shaping term $F(s, a, s')$ preserves the optimal policy if and only if it has the form

$$
F(s, a, s') \;=\; \gamma \Phi(s') - \Phi(s)
$$

for some *potential function* $\Phi : \mathcal{S} \to \mathbb{R}$. This is called *potential-based shaping*. The intuition is that the bonus is the discounted future potential you gain minus the potential you give up — a zero-sum accounting of progress that cancels out along any closed loop, so it cannot create artificial cycles for the agent to exploit.

In practice, $\Phi$ is often the negative distance to a goal state, the output of a value function from a simpler related problem, or a hand-designed progress measure. The key constraint is that the difference must be computable from the current and next state alone, and must not depend on the action. When that holds, the Bellman optimality of the original reward is preserved, and the shaped reward does nothing more harmful than make the landscape less flat.

This is a genuine result and worth knowing. Its practical limitation is that choosing a good $\Phi$ requires roughly the same domain knowledge as choosing the reward function in the first place — you are trading one design problem for a related one. On a robot task with a hundred-dimensional state space and a goal that depends on object pose, contact forces, and task stage, writing a potential that does not accidentally introduce cycles or plateau regions is non-trivial.

## When the reward is there but measuring it is expensive

A different failure mode: you know, in principle, what success looks like, but you cannot cheaply measure it during training. Consider a robot that should sort recyclable materials from a bin. A true reward would require knowing whether the sorted item was correctly classified — information that typically requires a human to verify or a separate computer vision pipeline to estimate. If that pipeline takes 3 seconds per query and you need 100,000 queries to train, you have a 3,500-hour bill. Dense rewards measured from sensors are cheap; semantically correct rewards are expensive.

One class of solutions collects a human preference signal instead of a numeric reward. Christiano et al. (2017) showed that a reward model can be trained from pairwise human comparisons — "which of these two trajectory clips looks better?" — and that this reward model can then supervise a policy without ever requiring the human to specify a closed-form function. The idea is influential in language models (RLHF), but it also appears in robotics when the true reward is too expensive to instrument directly.

Another approach is *inverse reward design* (Hadfield-Menell et al., 2017): rather than treating the written reward as ground truth, treat it as a noisy observation of the designer's true intent, and infer the likely true reward under a prior over plausible objectives. The policy optimizes an expectation over possible true rewards rather than the stated one, which makes it more cautious in regions where the two might diverge.

Both approaches share an assumption that the human designer's preferences are coherent and consistent. That assumption is usually not too badly wrong, but it is an assumption.

## Practical implications for robotics

Put together, the three problems — sparsity, shaping hazard, and measurement cost — explain why pure RL from scratch is rarely the first tool reached for in modern robot learning. A few observations worth making explicit before Chapter 6:

Dense rewards are almost always shaped in practice, which means they almost always carry misspecification risk. The standard mitigation is not better shaping theory; it is *evaluation diversity* — checking the learned policy on the target task with a different evaluator than the one used for training. If the robot looks like it is succeeding on the training signal but doing something strange on the real task, that gap is a specification error and the shaped reward needs revising.

Sparse rewards are philosophically cleaner — harder to game — but require either excellent exploration or seeding with demonstrations. In robotics, where reset-free training is an active research problem and each episode takes real time on real hardware, the exploration requirement of sparse rewards is often prohibitive without a simulator.

The most commonly used practical solution, which we examine in Chapter 6, bypasses the reward entirely: copy what an expert does, and hope that the expert was doing the right thing. This is not free of failure modes, but the failure modes are different — and in many robotics applications, substantially easier to identify.

The broader point is that the reward function is a *specification of intent* encoded as a scalar signal. Writing specifications is hard. Writing scalar specifications that remain resistant to optimization pressure is harder still. Reward design is the hardest part not because the math is deep — the Bellman equations do not care where $R$ came from — but because the engineering problem of saying clearly what you want, in a form a learning algorithm cannot misread, is one the field has not solved.

The next section, §5.5, takes up a related but distinct problem: even if you have a reasonable reward function, translating the MDP abstraction into a working robot system requires decisions about state representation, action space, episode structure, and sim-to-real transfer that are each capable of silently undermining everything the reward design got right.
