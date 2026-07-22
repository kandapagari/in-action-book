---
chapter: 5
section: 5.2
title: "Value iteration and policy iteration"
target_words: 2000
status: draft
prereqs: §5.1 (MDP tuple, the Markov property, returns, $V^\pi$ and $Q^\pi$); §3.1 (vector and matrix norms, fixed points); §4.1 for the contrast with symbolic search; comfort reading a `for` loop as a fixed-point iteration
key_refs:
  - Bellman (1957). Dynamic Programming. Princeton University Press.
  - Howard (1960). Dynamic Programming and Markov Processes. MIT Press.
  - Puterman (1994). Markov Decision Processes — Discrete Stochastic Dynamic Programming, Chapter 6. Wiley.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 4. MIT Press.
  - Bertsekas (2017). Dynamic Programming and Optimal Control, Vol. I, 4th ed. Athena Scientific.
---

# 5.2  Value iteration and policy iteration

§5.1 introduced the value function $V^\pi$ and the action-value $Q^\pi$ but didn't say how to compute them. This section closes that gap for the case the rest of the chapter doesn't assume: the case where the transition $P$ and the reward $R$ are known. That assumption sounds restrictive, and from a robotics standpoint it is; you almost never have a clean tabular $P$ for a real arm. But the two algorithms that fall out of it, value iteration and policy iteration, are the algorithmic foundation every later method in this book inherits its structure from, from Q-learning in §5.3 to the PPO update of §7.3 to the critic inside SAC. Read this section as the place where the shape of all subsequent RL algorithms gets set, with the inconvenient assumption stripped away later.

Both algorithms rest on a single equation. We spend the first half of the section understanding that equation, then turn it into code.

## The Bellman equations

Fix a policy $\pi$ and a state $s$. The value $V^\pi(s)$ is the expected return from $s$ under $\pi$. The defining recursive identity is the Bellman expectation equation:

$$
V^\pi(s) \;=\; \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \,\bigl[\,R(s, a, s') + \gamma V^\pi(s')\,\bigr].
$$

In words: the value of $s$ under $\pi$ is the average, over the actions $\pi$ would pick and the transitions those actions would induce, of the immediate reward plus the discounted value of where you land. The equation is a statement about consistency, not optimality. It's true for any policy. The corresponding identity for $Q^\pi$ is

$$
Q^\pi(s, a) \;=\; \sum_{s'} P(s' \mid s, a)\,\bigl[\,R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\,\bigr],
$$

which differs only in committing to action $a$ at the current step before falling back on $\pi$.

The Bellman optimality equation is the same identity, written for the optimal value $V^\star(s) = \max_\pi V^\pi(s)$:

$$
V^\star(s) \;=\; \max_{a}\, \sum_{s'} P(s' \mid s, a)\,\bigl[\,R(s, a, s') + \gamma V^\star(s')\,\bigr].
$$

The $\sum_a \pi(a \mid s)$ has become a $\max_a$. The optimal value of a state is the best you can do in one step plus the optimal value of where that step lands you. Bellman (1957) proved that this equation has a unique fixed point in $V$ for $\gamma < 1$ and bounded rewards. Both algorithms in this section are ways to find that fixed point.

## Value iteration

Define the Bellman optimality operator $T^\star$ acting on a value function $V$:

$$
(T^\star V)(s) \;=\; \max_{a}\, \sum_{s'} P(s' \mid s, a)\,\bigl[\,R(s, a, s') + \gamma V(s')\,\bigr].
$$

A value function is optimal exactly when $V = T^\star V$. The algorithm called value iteration picks an initial $V_0$, usually the all-zeros vector indexed by $\mathcal{S}$, and applies $T^\star$ repeatedly:

$$
V_{k+1}(s) \;=\; (T^\star V_k)(s).
$$

The key analytical fact is that $T^\star$ is a $\gamma$-contraction in the $\ell_\infty$ norm: for any two value functions $V$ and $V'$, $\|T^\star V - T^\star V'\|_\infty \leq \gamma \|V - V'\|_\infty$. Banach's fixed-point theorem then guarantees $V_k \to V^\star$ at a geometric rate, and after $k$ steps the worst-case error is at most $\gamma^k \|V_0 - V^\star\|_\infty$. With $\gamma = 0.95$, an error tolerance of $10^{-3}$ on rewards bounded by $1$ takes roughly 150 sweeps, small enough to run on a laptop for tabular problems and small enough to fit in a footnote in every RL textbook.

Once $V^\star$ is in hand, the optimal policy is the greedy policy with respect to it:

$$
\pi^\star(s) \;=\; \arg\max_{a}\, \sum_{s'} P(s' \mid s, a)\,\bigl[\,R(s, a, s') + \gamma V^\star(s')\,\bigr].
$$

Here's value iteration as a 12-line Python loop on the §5.1 gridworld:

```python
import numpy as np

n_states, n_actions = 16, 4   # 4x4 grid; N, S, E, W
P = build_transition_tensor()  # shape (n_states, n_actions, n_states)
R = build_reward_tensor()      # shape (n_states, n_actions, n_states)
gamma, tol = 0.95, 1e-6

V = np.zeros(n_states)
while True:
    Q = (P * (R + gamma * V[None, None, :])).sum(axis=2)  # (S, A)
    V_new = Q.max(axis=1)
    if np.max(np.abs(V_new - V)) < tol:
        break
    V = V_new
pi_star = Q.argmax(axis=1)
```

Two lines do all the work. `Q = (P * (R + gamma * V[None, None, :])).sum(axis=2)` is the Bellman operator written as a tensor contraction; `V_new = Q.max(axis=1)` is the $\max_a$. On the deterministic gridworld of §5.1, this loop converges in well under a hundred sweeps and recovers the optimal policy of "move toward the goal," with $V^\star$ taking values close to $-1/(1-\gamma) \approx -20$ in the corner cells and 0 at the absorbing goal.

## Policy iteration

Howard (1960) proposed an alternative scheme that splits the fixed-point problem into two simpler ones. Start with an arbitrary policy $\pi_0$, then repeat two steps.

Policy evaluation computes $V^{\pi_k}$, the value of the current policy, by solving the Bellman expectation equation. Because $\pi_k$ is fixed, this is a linear system in $|\mathcal{S}|$ unknowns:

$$
V^{\pi_k}(s) \;=\; \sum_{s'} P_{\pi_k}(s' \mid s)\,\bigl[\,R_{\pi_k}(s, s') + \gamma V^{\pi_k}(s')\,\bigr],
$$

where $P_{\pi_k}$ and $R_{\pi_k}$ marginalize over the policy. You can either solve this linear system directly with a matrix inverse, fine for a 16-state gridworld, painful for a 1000-state model, or run a small inner loop of Bellman expectation updates to convergence.

Policy improvement replaces $\pi_k$ with the greedy policy with respect to $V^{\pi_k}$:

$$
\pi_{k+1}(s) \;=\; \arg\max_{a}\, \sum_{s'} P(s' \mid s, a)\,\bigl[\,R(s, a, s') + \gamma V^{\pi_k}(s')\,\bigr].
$$

The policy improvement theorem (Howard 1960; Sutton & Barto 2018 §4.2) guarantees $\pi_{k+1}$ is at least as good as $\pi_k$ everywhere, with strict improvement in at least one state unless $\pi_k$ is already optimal. Since there are only finitely many deterministic policies, the iteration terminates in finitely many steps, usually a handful, even for moderately sized MDPs.

A compact Python sketch:

```python
pi = np.zeros(n_states, dtype=int)    # start with "always N"
while True:
    # 1. Evaluation: solve linear system for V^pi
    P_pi = P[np.arange(n_states), pi, :]            # (S, S)
    R_pi = (P[np.arange(n_states), pi, :]
            * R[np.arange(n_states), pi, :]).sum(axis=1)  # (S,)
    V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)
    # 2. Improvement: greedy w.r.t. V
    Q = (P * (R + gamma * V[None, None, :])).sum(axis=2)
    pi_new = Q.argmax(axis=1)
    if np.array_equal(pi_new, pi):
        break
    pi = pi_new
```

On the gridworld of §5.1, policy iteration converges in three or four outer iterations: an initial "always go north" policy improves to something sensible in the first sweep, refines to the optimal policy in the second, and the third sweep confirms there's nothing left to improve. Each outer step solves a small linear system and does much more work per step than a value-iteration sweep, but there are far fewer of them.

## Two algorithms, one family

Value iteration and policy iteration look different but interpolate into each other. Modified policy iteration replaces the exact linear solve in policy evaluation with $m$ Bellman-expectation sweeps; at $m = 1$ this is approximately value iteration, at $m = \infty$ it's exact policy iteration. Asynchronous variants (Bertsekas 2017, §1.3) update individual states rather than the whole vector at once, and converge under the same contraction guarantee as long as every state is visited infinitely often. Sutton & Barto (2018) §4.5 introduces these under the name generalized policy iteration and makes the point this section will end on: every learning algorithm in the rest of the book is some form of GPI. There's an evaluation step that pushes a value estimate toward consistency with the current policy, and an improvement step that pushes the policy toward greediness with respect to the current value. Q-learning, SAC, PPO, and the critic-and-actor objectives of every VLA paper involving RL are all this same pattern with different approximations and different truncations.

The choice between value and policy iteration in practice is a matter of constant factors. Value iteration is simpler to implement, has no inner loop, and is easy to vectorize. Policy iteration converges in many fewer outer steps but each step is expensive, especially with an exact linear solve. For small tabular problems the difference is invisible; for the function-approximation versions of Chapter 7, the two algorithms morph into the value-target and policy-target sides of an actor-critic loop, and choosing which to lean on is what gives TD3, SAC, and PPO their distinct flavors.

## What dynamic programming gives, and what it cannot

Both algorithms share three properties worth pulling out. They're off-line: they need the full $P$ and $R$ in advance, not sampled experience. They're exact under their assumptions: at convergence, the policy is provably optimal, not approximately so. And they're exponential in state-space dimension: a 4×4 gridworld has 16 states, a $20 \times 20 \times 20$ block-stacking task on a discretized table has $8{,}000$, and a robot arm with even a moderate discretization of joint angles has more states than there are atoms in something dramatic. Bellman (1957) coined the phrase "curse of dimensionality" for exactly this: every classical dynamic-programming algorithm scales polynomially in $|\mathcal{S}|$, and $|\mathcal{S}|$ scales exponentially in the number of state variables.

Two consequences shape the rest of the book. Sampling replaces enumeration. Q-learning in §5.3 and the deep actor-critic methods of Chapter 7 estimate $Q^\star$ from sampled transitions, never writing $P$ down. And function approximation replaces tabular storage: a neural network with a few hundred thousand parameters represents a $V$ or $Q$ defined over millions of states without ever materializing the table. The elegance is lost; the scaling isn't. Every deep RL method we discuss from Chapter 7 onward runs approximate value or policy iteration with neural networks standing in for the table, and the analytical guarantees falling out of the contraction argument here get weakened or lost in the process. That's the price of leaving the tabular world.

Section 5.3 takes the first step in that direction: it drops the assumption that $P$ is known, keeps the tabular setting, and shows that a single sampled transition is enough to do approximate value iteration in the form known as Q-learning.
