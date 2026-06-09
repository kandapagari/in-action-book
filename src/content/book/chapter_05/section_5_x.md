---
chapter: 5
section: 5.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §5.1–§5.6; Python with gymnasium installed; a working understanding of the MDP five-tuple, tabular Q-learning, reward shaping, and the MDP-to-robot translation problem; about two hours of laptop CPU time (no GPU required)
key_refs:
  - Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3-4):279-292.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.). MIT Press.
  - Ng, Harada & Russell (1999). Policy invariance under reward transformations. ICML 1999.
  - Brockman et al. (2016). OpenAI Gym. arXiv:1606.01540.
  - Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature 518:529-533.
  - Schulman et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
  - Haarnoja et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. ICML 2018.
---

# 5.x  Hands-on exercise + chapter references

Chapter 5 was the reward chapter; the exercise is the one where you write the reward and watch what an agent does with it. The four drills below take a combined two hours on a laptop CPU. No GPU is required, no robot is required, and no internet is required after the initial dependency install. The point is to leave Chapter 5 with a working Q-learning implementation that you trust, a first-hand example of reward gaming you designed yourself, and a practiced eye for the four MDP-to-robot design decisions from §5.5 as they appear in a real paper. Chapter 7 will build on all three.

Install `gymnasium` (the maintained fork of OpenAI Gym, Brockman et al., 2016, arXiv:1606.01540) if you do not have it:

```
pip install gymnasium
```

FrozenLake-v1 with `is_slippery=False` is the test environment for Exercises 5.x.1 and 5.x.2; the slippery variant appears in 5.x.2. No other dependencies are needed for the first three exercises.

## Exercise 5.x.1 — Implement tabular Q-learning and verify against value iteration

Write two Python files: `value_iteration.py` and `q_learning.py`. Both operate on the 4×4 FrozenLake-v1 grid with `is_slippery=False`, so the transition tensor $P$ is deterministic.

**Value iteration** (`value_iteration.py`): gymnasium's `FrozenLakeEnv` exposes the full transition tensor as `env.P[state][action]`, a list of `(probability, next_state, reward, done)` tuples. Use that to run the Bellman optimality update

$$
V^{(k+1)}(s) = \max_{a}\,\sum_{s'} P(s' \mid s, a)\,\bigl[R(s, a, s') + \gamma\,V^{(k)}(s')\bigr]
$$

with $\gamma = 0.99$ until $\max_s |V^{(k+1)}(s) - V^{(k)}(s)| < 10^{-8}$. Extract the greedy policy $\pi^\star(s) = \arg\max_a Q^\star(s, a)$, where you derive $Q^\star$ from $V^\star$ in the usual one-step lookahead. Save $V^\star$ and $\pi^\star$ as reference objects.

**Q-learning** (`q_learning.py`): implement the update from §5.3 —

```python
Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
```

with $\alpha = 0.1$, $\gamma = 0.99$, $\varepsilon$-greedy exploration starting at $\varepsilon = 1.0$ and decaying by a factor of 0.999 per episode to a floor of 0.01. Run for 5 000 episodes. Record the episode-return rolling average (window 100 episodes) and save it as `q_learning_curve.png`. After training, extract the greedy policy $\hat\pi(s) = \arg\max_a Q(s, a)$.

The diagnostic is a side-by-side comparison: for each of the 16 states, print $\pi^\star(s)$ from value iteration beside $\hat\pi(s)$ from Q-learning. On the deterministic grid they should agree on every non-terminal state. If they do not, the most common causes are: (1) the Q-table is indexed $(s, a)$ but the `done` flag is not handled — if you bootstrap on a terminal state you get a corrupt target — and (2) the episode return is not resetting between episodes. Fix these before Exercise 5.x.2.

Wall clock: about twenty-five minutes.

## Exercise 5.x.2 — Watch Q-learning fail with a shaped reward and then succeed with a principled one

This exercise is in two parts and demonstrates §5.4's central argument in miniature.

**Part A — a shape that backfires.** Make a copy of `q_learning.py` called `q_learning_bad_shape.py`. Add a shaping term to the reward:

```python
# Manhattan distance to goal (state 15), lower is better
def manhattan_to_goal(s):
    row, col = s // 4, s % 4
    return abs(row - 3) + abs(col - 3)

r_shaped = r - 0.1 * manhattan_to_goal(s_next)
```

Train for 5 000 episodes on the slippery variant (`is_slippery=True`, so transitions are stochastic: the intended action fires with probability 1/3, and each orthogonal direction fires with probability 1/3). Plot the episode-return curve.

You will observe that the agent learns to hover near the goal rather than stepping onto it: stepping onto the goal gives $r = +1$ but also terminates the episode, ending the stream of small negative shaping bonuses for being close. The agent has found a policy that scores higher under your shaped reward than the intended optimal policy does. This is the §5.4 specification-gaming failure. Describe it in one sentence: "the agent did X because Y metric was higher that way than the correct way."

**Part B — potential-based shaping that does not corrupt the optimal policy.** Make another copy, `q_learning_good_shape.py`. Replace the raw shaping term with the potential-based version from §5.4 (Ng, Harada & Russell, 1999):

```python
# Potential: negative Manhattan distance, so Phi(s) is higher near goal
def phi(s):
    row, col = s // 4, s % 4
    return -0.1 * (abs(row - 3) + abs(col - 3))

r_shaped = r + gamma * phi(s_next) - phi(s)
```

Train for 5 000 episodes on the slippery variant. Plot the curve on the same axes as Part A. The potential-based agent should converge to the correct policy — the one that steps onto the goal rather than camping adjacent to it — while still converging faster than the un-shaped baseline from 5.x.1.

Save the comparison plot as `shaping_comparison.png`. The three curves — no shaping (from 5.x.1), bad shaping, and potential-based shaping — on the same axes are the visual summary of why the Ng et al. result matters. The curves for no-shaping and good-shaping should end near the same asymptotic performance; the bad-shaping curve should end lower.

Wall clock: about forty minutes including the two training runs and the plot.

## Exercise 5.x.3 — Exploration ablation: ε-greedy vs. constant ε vs. no exploration

This exercise puts a number on the §5.3 claim that exploration scheduling matters. Use the slippery FrozenLake-v1 from Exercise 5.x.2 with potential-based shaping. Train five agents:

1. $\varepsilon = 1.0$ (pure random — never greedy; the agent collects experience but cannot converge).
2. $\varepsilon = 0.0$ (no exploration — greedy from a zero-initialized Q-table; never leaves the first local path it finds).
3. $\varepsilon = 0.1$ fixed (constant low exploration for the full 5 000 episodes).
4. $\varepsilon$ decays from $1.0$ to $0.01$ over 5 000 episodes (the schedule from 5.x.1).
5. $\varepsilon$ decays from $1.0$ to $0.01$ over 500 episodes, then fixed at $0.01$ (fast decay).

Plot all five episode-return rolling averages (window 100) on the same axes. Save as `exploration_ablation.png`. Write one sentence about each curve: what did the agent learn, and why does the curve look the way it does? Then answer one question in plain prose: which of the five would you pick for a real robot, and why?

There is a correct answer for the gridworld — agent 4 or 5 depending on how patient you are — but the reasoning is more important than the number. The §5.5 argument that episode length and reset cost interact with exploration is the argument you are making in miniature here: on a slippery grid, agent 2 (no exploration) gets stuck in the top-left corner and never discovers the goal, exactly as a real robot with a deterministic "always move right" policy never discovers that the grasp has to be approached from above.

Wall clock: about twenty-five minutes for five training runs and the plot.

## Exercise 5.x.4 — Read a robotics RL paper through the MDP-to-robot lens

Open one of the following: Haarnoja et al. (2018), "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Learned Reward Function," which includes experiments on a simulated half-cheetah and ant; or any deep-RL-for-manipulation paper from the last three years you have available. Read the experiment section and the appendix. In a text file `paper_audit.txt`, answer the following five questions, one paragraph each:

1. **State representation.** What is $\mathcal{S}$? Is it Markov for the task? If the paper uses a raw image, what temporal information is discarded and how does the system compensate?

2. **Action space.** What is $\mathcal{A}$? What control frequency is used? Is this joint torques, joint positions, Cartesian end-effector poses, or something else? Is the choice motivated in the text?

3. **Reward function.** What is $R$? Is it sparse, dense, or shaped? If shaped, is the shaping potential-based? Can you identify at least one behavior the reward might accidentally incentivize that is different from the intended task?

4. **Episode structure.** How are episodes reset? What does the reset cost? Does the reset cost match real-world deployment? If trained in simulation, is the sim-to-real gap discussed?

5. **Overall judgment.** Of the four design decisions — state, action, reward, episode — which is the most load-bearing and which is the most underspecified? What one change would most improve the validity of the result?

There is no scoring rubric. The point is that the five-question template from §5.5 stops being abstract once you have applied it to a paper that matters to you. Every design-review conversation you will have in the rest of your career about a robotics RL system is some version of these five questions.

Wall clock: about thirty minutes for the read plus the written audit.

## Chapter 5 reading list

The works below are cited in §5.1–§5.6. They are grouped by purpose. Full bibliographic entries for everything cited in the book live in Appendix E.2; this list is the chapter-local subset.

### MDP theory: the formal backbone

- Bellman, R. (1957). *Dynamic Programming.* Princeton University Press. The source of the Bellman equations §5.2 derives. Worth reading the first chapter even in 2026.
- Howard, R. A. (1960). *Dynamic Programming and Markov Processes.* MIT Press. The first systematic treatment of policy iteration; §5.2 follows Howard's presentation more closely than Bellman's.
- Puterman, M. L. (1994). *Markov Decision Processes — Discrete Stochastic Dynamic Programming.* Wiley. The authoritative technical reference; Chapter 4 has the contraction proof that §5.2 sketches.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning — An Introduction*, 2nd ed. MIT Press. Free online. The indispensable textbook; Chapters 3–6 cover the material in §5.1–§5.3.
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). "Planning and Acting in Partially Observable Stochastic Domains." *Artificial Intelligence* 101(1–2). The POMDP reference §5.5 cites when it explains why most robotics problems are technically POMDPs and why practitioners treat them as MDPs anyway.

### Q-learning: derivation and convergence

- Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards.* PhD thesis, University of Cambridge. The original derivation.
- Watkins, C. J. C. H., & Dayan, P. (1992). "Q-learning." *Machine Learning* 8(3–4):279–292. The published version; four pages long, entirely worth reading.
- Lai, T. L., & Robbins, H. (1985). "Asymptotically Efficient Adaptive Allocation Rules." *Advances in Applied Mathematics* 6(1):4–22. The bandit theory §5.3 draws on when discussing the exploration-exploitation tradeoff; UCB is a direct descendant.

### Reward design and shaping

- Ng, A. Y., Harada, D., & Russell, S. (1999). "Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping." *ICML 1999*. The result Exercise 5.x.2 Part B depends on: only potential-based shaping preserves the optimal policy.
- Amodei, D., Olah, C., Steinhardt, J., et al. (2016). "Concrete Problems in AI Safety." The specification-gaming section of §5.4 draws on the framework in this paper.
- Krakovna, V., et al. (2020). "Specification Gaming: The Flip Side of Intelligent Behavior." DeepMind blog. A catalogue of real-world reward-gaming failures; every example in §5.4 is a mild version of something on this list.
- Christiano, P., Ziegler, J., Stiennon, N., et al. (2017). "Deep Reinforcement Learning from Human Preferences." *NeurIPS 2017*. The RLHF paper; the reward-learning framing §5.4 describes as an alternative to hand-specified reward is this paper's contribution.

### Deep RL: the bridge to Chapter 7

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-Level Control through Deep Reinforcement Learning." *Nature* 518:529–533. The DQN paper; Chapter 7 starts here. §5.3's discussion of the replay buffer is a preview of this paper's two key engineering contributions.
- Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347. The PPO reference for Chapter 7's policy-gradient section.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Learned Reward Function." *ICML 2018*. The SAC reference for Chapter 7's off-policy actor-critic section. Exercise 5.x.4 uses this paper as one of two options.

### The MDP-to-robot translation problem

- Brockman, G., Cheung, V., Pettersson, L., et al. (2016). "OpenAI Gym." arXiv:1606.01540. The standard benchmark environment library, and the conceptual predecessor to the gymnasium package used in this chapter's exercises.
- Todorov, E., Erez, T., & Tassa, Y. (2012). "MuJoCo: A Physics Engine for Model-Based Control." *IROS 2012*. The simulation environment underlying most of the deep-RL manipulation results Chapter 7 cites.
- Tobin, J., Fong, R., Ray, A., et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS 2017*. The sim-to-real strategy §5.5 names as the pragmatic alternative to system identification.
- Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA 2018*. The companion result that shows dynamics randomization, not just visual randomization, is necessary for real-world transfer.

## Chapter summary

Chapter 5 gave you the theoretical substrate for reward-based learning, and the four exercises made it concrete. You can now implement tabular Q-learning from scratch, verify it against value iteration on a known environment, and trust that the implementation is correct before touching a neural network. You have seen, with your own hands, a reward-gaming failure and understood exactly why it happened and how potential-based shaping avoids it. You can run an exploration ablation and read its results rather than taking on faith that ε-decay schedules matter. And you have a reusable five-question template for auditing the MDP design choices in any robotics RL paper you encounter. Chapter 6 takes the question §5.4 raised — what do you do when writing a reward function is too expensive or too error-prone? — and answers it with demonstrations instead of rewards.
