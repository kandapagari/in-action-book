---
chapter: 5
section: 5.3
title: "Q-learning and the role of exploration"
target_words: 2000
status: draft
prereqs: §5.1 (MDP tuple, $Q^\pi$, $Q^\star$); §5.2 (Bellman optimality operator, value iteration as a contraction); §3.4 (the supervised vs. RL loss split); comfort with stochastic-approximation intuition (running averages converge if step sizes shrink the right way)
key_refs:
  - Watkins (1989). Learning from Delayed Rewards. PhD thesis, Cambridge University.
  - Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3-4):279-292.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 6. MIT Press.
  - Lai & Robbins (1985). Asymptotically efficient adaptive allocation rules. Advances in Applied Mathematics, 6(1):4-22.
  - Auer, Cesa-Bianchi & Fischer (2002). Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2-3):235-256.
  - Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518:529-533.
  - Pathak et al. (2017). Curiosity-driven exploration by self-supervised prediction. ICML.
  - Burda et al. (2018). Exploration by random network distillation. arXiv:1810.12894.
---

# 5.3  Q-learning and the role of exploration

Value iteration in §5.2 needed the full transition tensor $P$ and reward tensor $R$. A robot doesn't have those. What it has is a stream of experience: at each time step it sees a state, picks an action, lands somewhere, and collects a reward. Q-learning, due to Watkins (1989), is the algorithm that drops the model assumption and keeps almost everything else. It's the single most important tabular RL algorithm in the book, not because anyone runs tabular Q-learning on a real robot, but because every deep RL method we touch in Chapter 7, and the critic inside every actor-critic VLA paper, is some approximation of it.

This section does two things. The first half derives the Q-learning update from the Bellman optimality equation and states the conditions under which it converges. The second half covers the part of the algorithm those conditions hide: how the agent decides which actions to try in the first place. That problem, exploration, is the part of RL that becomes the rate-limiting step once you leave the gridworld.

## From Bellman to a sampled update

Recall the Bellman optimality equation for $Q^\star$:

$$
Q^\star(s, a) \;=\; \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\bigl[\,R(s, a, s') + \gamma \max_{a'} Q^\star(s', a')\,\bigr].
$$

Value iteration computes the right-hand side by enumeration, summing over all $s'$. Q-learning computes it by sampling: take one transition $(s, a, r, s')$ from the world, treat $r + \gamma \max_{a'} Q(s', a')$ as a one-sample estimate of the expectation, and nudge $Q(s, a)$ toward it. The update is

$$
Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \,\bigl[\,r + \gamma \max_{a'} Q(s', a') - Q(s, a)\,\bigr].
$$

The bracketed quantity is the temporal-difference error, or TD error: the gap between the current estimate $Q(s, a)$ and what one new piece of evidence says it should be. The step size $\alpha \in (0, 1]$ controls how much the estimate moves on each update. With $\alpha = 1$ the new value just overwrites the old one, high-variance and unstable. With $\alpha$ small, the estimate becomes a slow running average of TD targets, which is what we want.

A few features of this update repay attention. It doesn't depend on $P$, since the transition is drawn from the world, not from a model. It doesn't depend on the policy the agent is following, either; the $\max_{a'}$ inside the target runs over the agent's current $Q$, not over the action it actually took next. That last property is what makes Q-learning off-policy: the learned $Q$ converges to the optimum even if the data-collecting policy is bad, as long as the data covers every state-action pair sufficiently. The behaviour policy and the target policy are decoupled, and that decoupling is what lets DQN later (Mnih et al. 2015) train from a replay buffer of stale transitions.

## Tabular Q-learning, end to end

Here's the algorithm. Initialize $Q$ as a table, usually zeros, or small random values to break ties:

```python
Q = np.zeros((n_states, n_actions))
alpha, gamma, eps = 0.1, 0.95, 0.1

for episode in range(n_episodes):
    s = env.reset()
    done = False
    while not done:
        # behaviour policy: epsilon-greedy
        a = (np.random.randint(n_actions)
             if np.random.rand() < eps
             else int(Q[s].argmax()))
        s_next, r, done = env.step(a)
        # off-policy TD update
        td_target = r + (0 if done else gamma * Q[s_next].max())
        Q[s, a] += alpha * (td_target - Q[s, a])
        s = s_next
```

Twenty lines, two of which do the actual learning. Run this on the §5.1 4×4 gridworld for a few thousand episodes and $Q$ converges to within rounding of the $Q^\star$ that value iteration produced from the known $P$ in §5.2, but with one consequential difference: you never wrote $P$ down. The convergence guarantee, due to Watkins & Dayan (1992), requires three things. Every $(s, a)$ must be visited infinitely often. The step sizes must satisfy the standard stochastic-approximation conditions $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, large enough to keep moving, small enough to settle. And the rewards must be bounded. Under those conditions $Q_t \to Q^\star$ with probability 1. The proof is a stochastic-contraction argument generalizing the deterministic contraction of §5.2; the intuition stays unchanged.

The first of those three conditions looks innocent on the page and is the source of every hard problem in the rest of this chapter.

## Why "visit every (s, a) infinitely often" is the hard part

The convergence theorem assumes a stream of transitions covering the whole state-action space. The agent's job is generating that stream. If the policy it follows during learning never tries a particular action in a particular state, the $Q$-value for that pair never updates, and the algorithm can't find out whether that pair was secretly the best one.

This is the exploration problem, and it has two faces. The shallow one is the multi-armed bandit: with $k$ arms and unknown rewards, how do you balance pulling the arm that looks best so far (exploitation) against pulling under-tried arms whose true mean might be higher (exploration)? Lai & Robbins (1985) proved that the regret of any reasonable algorithm grows at least logarithmically in the number of pulls; Auer, Cesa-Bianchi & Fischer (2002) showed that the upper confidence bound (UCB) algorithm matches that lower bound up to constants. UCB picks the arm $a$ that maximizes $\hat{\mu}_a + \sqrt{2 \log t / n_a}$, where $\hat{\mu}_a$ is the empirical mean and $n_a$ is the visit count: optimistic in the face of uncertainty, with the optimism shrinking as you sample more.

The deep face of the problem is that an MDP isn't a bandit. To discover that an action is good you may have to take it, then take five more downstream actions correctly, then notice a reward at the end. The credit-assignment delay turns the visit-everywhere requirement from "try every action a few times" into "produce trajectories that reach every relevant region of state space." On a gridworld this is easy. On a robot arm trying to open a drawer, an agent that has never closed its hand on the handle never sees the positive reward and can't learn to want to do so. Reward sparsity amplifies the exploration problem until it dominates everything else, and the next section, §5.4, is largely about that.

## Four ways to explore

The simplest exploration strategy is $\varepsilon$-greedy: take the greedy action with probability $1 - \varepsilon$, a uniformly random action with probability $\varepsilon$. It satisfies the visit-everywhere condition asymptotically, it's one line of code, and it's what the listing above uses. Its weakness is that the random fraction gets spent equally on actions you already know are bad; the agent that has discovered the goal is in the top-right corner of the gridworld still spends 10% of its actions moving west.

Boltzmann (softmax) exploration fixes that by making the randomness graded:

$$
\pi(a \mid s) \;=\; \frac{\exp(Q(s, a) / \tau)}{\sum_{a'} \exp(Q(s, a') / \tau)}.
$$

A high temperature $\tau$ makes the policy nearly uniform; a low one makes it nearly greedy. The agent now wastes less effort on actions it already knows are losses, but it can still get stuck if a single suboptimal action holds a slight Q-value lead.

Optimism in the face of uncertainty generalizes UCB to MDPs. Add a bonus to the Q-target that grows with how rarely $(s, a)$ has been visited. A common form is $r + \gamma \max_{a'} Q(s', a') + \beta / \sqrt{n(s, a)}$. The bonus pulls the agent toward state-action pairs it hasn't seen, and shrinks to zero in the limit, so the asymptotic Q-values are still $Q^\star$. The challenge, and the reason this stops being trivial in deep RL, is counting visits in a state space where you never see the same state twice.

Intrinsic motivation drops counts and uses a learned signal instead. The agent maintains a prediction model of the environment and rewards itself for transitions its model finds surprising. Pathak et al. (2017) train a forward-dynamics model on visited states and use its prediction error as an intrinsic reward, the intrinsic curiosity module, ICM. Burda et al. (2018) instead distill a fixed random network and use the distillation error as a surprise signal, random network distillation, RND, in arXiv:1810.12894, which avoids ICM's pathology of being surprised by stochastic TVs. Both methods let an agent make progress in environments where the extrinsic reward is sparse enough that $\varepsilon$-greedy never sees it, and both come back in Chapter 7 when we discuss exploration in the deep RL setting and again in Chapter 18 when we ask how to explore on real robots without breaking them.

## What changes with function approximation

Two pieces of the story above are tabular. The convergence proof assumes $Q$ is stored as a table; the visit-count exploration bonuses assume you can count visits to individual states. Replace the table with a neural network, DQN (Mnih et al. 2015) was the moment this became routine, on Atari from raw pixels, and both guarantees soften. The TD update is now a gradient step on a regression loss

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\bigl[\,(r + \gamma \max_{a'} Q_{\bar{\theta}}(s', a') - Q_\theta(s, a))^2\,\bigr],
$$

with a separate target network $Q_{\bar{\theta}}$ to keep the moving target from making training oscillate, and a replay buffer $\mathcal{D}$ to break temporal correlations. The Watkins-Dayan contraction argument no longer applies: function approximation can in principle cause Q-learning to diverge, and a fair fraction of the deep RL literature is engineering around the gap between the tabular guarantee and the practical reality. Chapter 7 takes this up in detail.

The exploration side gets harder too. Counting visits in pixel space is meaningless; two frames an agent has clearly seen before differ in JPEG noise. Pseudo-count methods, density models, and the distillation tricks of RND all exist because the elementary state-action counter tabular optimism relies on has no straightforward analogue in a continuous, high-dimensional state space. The robotics literature, when it talks about exploration at all, has mostly given up on intrinsic-bonus methods and replaced exploration with demonstrations: behavior cloning (Chapter 6) and fine-tuning a pretrained VLA (Chapter 16) sidestep the problem by copying somebody else's reachable trajectories. That's one of the reasons modern foundation action models look less like Q-learning than the historical lineage of §1.3 might suggest.

## What Q-learning leaves on the table

Two limitations are worth naming before we move on. Q-learning is value-only: it never represents the policy explicitly. To act, you compute $\arg\max_a Q(s, a)$, which in a continuous action space turns into an inner optimization problem at every step. The policy-gradient methods of Chapter 7, and the actor-critic methods built on top of them, give up the off-policy guarantee in exchange for a parametric policy you can evaluate in constant time. And Q-learning assumes the reward signal exists. That assumption is doing a lot of work, and §5.4 takes apart what happens when it doesn't, or when it does, but at a cost in design effort exceeding whatever it would have cost to just teach the robot by demonstration.

For now, the takeaway is the algorithm. Sampled TD updates replace the enumerated Bellman sweep of §5.2, and the off-policy property allows training from any reasonable data stream. The asymptotic guarantees rest on the agent visiting every relevant state-action pair, and engineering that coverage is the practical bulk of using Q-learning on anything more complicated than a gridworld.

The next section, §5.4, makes that practical bulk concrete: it's about why reward design, choosing what the agent gets paid for, and shaping the signal so it actually drives learning, is the part of RL that consumes the most engineering time and produces the most surprising failures.
