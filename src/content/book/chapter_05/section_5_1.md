---
chapter: 5
section: 5.1
title: "States, actions, rewards, and policies"
target_words: 2000
status: draft
prereqs: §1.2 (the three-slot anatomy), §1.4 (the four families), §3.2 (random variables, expectations); §4.1 for the contrast with symbolic actions; high-school probability
key_refs:
  - Bellman (1957). Dynamic Programming. Princeton University Press.
  - Howard (1960). Dynamic Programming and Markov Processes. MIT Press.
  - Puterman (1994). Markov Decision Processes — Discrete Stochastic Dynamic Programming. Wiley.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 3. MIT Press.
  - Kaelbling, Littman, Cassandra (1998). Planning and Acting in Partially Observable Stochastic Domains. Artificial Intelligence 101(1–2).
---

# 5.1  States, actions, rewards, and policies

Chapter 4 ended with a complaint and a promise. The complaint was that classical action models leave the training-signal slot of the §1.2 anatomy empty: a STRIPS planner knows what its actions do, but only because somebody wrote the effects by hand, and it has no way to learn that one sequence of actions was better or worse than another. The promise was that the next three chapters of Part 2 would each fill that empty slot with a different signal. Chapter 5's signal is reward.

The mathematical object that lets you talk about reward properly is the Markov decision process, or MDP. Almost every reinforcement-learning paper of the last forty years opens by writing down an MDP tuple in its second sentence, and this section spends ten pages on what those five letters mean, because the rest of the chapter, most of Chapter 7, and the offline-RL background you'll need in Chapter 12 stay unreadable until you can read the tuple without translating in your head. The section also makes the case that the MDP is a modeling choice, not a fact about the world, and that the gap between a clean MDP on paper and a messy robot in a lab is the source of more wasted GPU-hours than any other modeling decision in the field.

## The five-tuple

An MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$. Five letters, each of which has been the subject of its own subfield. Take them one at a time.

$\mathcal{S}$ is the state space, the set of all configurations the world can be in. For the canonical 4×4 gridworld appearing in every RL textbook from Bellman (1957) onward, $\mathcal{S}$ is the set of sixteen cells, with one cell marked as the goal and one or two marked as obstacles. For a tabletop manipulation task with a 7-DOF Franka arm and one block, the state might be the 14-vector of joint positions and velocities concatenated with the block's 6-DOF pose, so $\mathcal{S} \subseteq \mathbb{R}^{20}$. For an end-to-end pixels-in policy like OpenVLA (arXiv:2406.09246), the "state" the policy sees is a 224×224×3 image, and whether that image is really a Markov state is a question we return to in §5.5.

$\mathcal{A}$ is the action space. In a discrete-action MDP the action space is a finite set: `{up, down, left, right}` for the gridworld, `{open-gripper, close-gripper, move-to-pose}` for a high-level skill MDP. In a continuous-action MDP it's a subset of $\mathbb{R}^n$: the 7-vector OpenVLA emits, the 8-DOF (joint torques plus gripper) command an SAC controller produces, the chunk of 16 poses Diffusion Policy outputs at once. The same physical robot can be described as either a discrete-action or a continuous-action MDP depending on what the engineer chooses to expose as $\mathcal{A}$, and that choice is a major architectural commitment. Chapter 7 lives mostly in continuous action spaces; this chapter, for clarity, stays discrete.

$P$ is the transition function. Given that the agent is in state $s$ and takes action $a$, $P(s' \mid s, a)$ is the probability of landing in state $s'$. For a deterministic gridworld, $P$ collapses onto a single $s'$ per $(s, a)$ pair. For a noisy gridworld, one with "slippery" cells where `up` moves the agent up with probability 0.8 and sideways with probability 0.1 each, $P$ becomes a non-trivial distribution. For a real robot, $P$ encodes contact dynamics, motor backlash, sensor noise, and the friction of the table the block sits on. It's unknown in almost every interesting case, and the central problem statement of reinforcement learning is "learn a good policy without ever writing $P$ down."

$R$ is the reward function, a scalar signal $R(s, a, s')$ the environment hands the agent on each transition. By convention $R$ is real-valued, often dense in a sim benchmark (one unit of reward per small move toward the goal) and almost always sparse on a real robot (one unit at task completion, zero otherwise). §5.4 spends its entire budget on the consequences of that single design choice.

$\gamma \in [0, 1)$ is the discount factor, the multiplicative weight applied to future rewards: a reward $r_t$ received at time $t$ contributes $\gamma^{t} r_t$ to the agent's total. $\gamma = 0$ makes the agent myopic, caring only about the immediate reward, while $\gamma$ near 1 makes it far-sighted; common values run $0.95$ to $0.99$ for episodic tasks, and the precise value affects both the solution and the convergence rate of the algorithms in §5.2. The mathematical reason $\gamma < 1$ gets enforced, rather than left at 1, is that an infinite sum of bounded rewards otherwise diverges. The practical reason is that a finite $\gamma$ acts as a kind of horizon truncation keeping the value function bounded.

Five letters. The full Sutton & Barto (2018) treatment elaborates each one for a chapter; this section's goal is working fluency, not completeness.

## The Markov property

The "Markov" in "Markov decision process" is doing real work, and it's the single assumption the rest of the formalism rests on. A process is Markov if the next state depends only on the current state and action, not on the history that produced the current state:

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t).
$$

For the gridworld, the property holds trivially: the cell you're in plus the action you take determines (stochastically) the cell you land in, and how you got to the current cell is irrelevant. For the Franka arm with a full joint-position-and-velocity state vector, the property holds approximately, since given $q$ and $\dot q$, the dynamics equation $M(q)\ddot q + C(q, \dot q)\dot q + g(q) = \tau$ from §4.3 says the next state is determined by the current one plus the torque command, with no explicit dependence on the past.

For OpenVLA staring at a single 224×224 image with no temporal context, the Markov assumption is false. A still image of a block on a table doesn't tell you whether the gripper is moving toward the block or away from it; a stack of two or three frames is a strictly larger Markov state than one frame. RT-1 (arXiv:2212.06817) and most modern VLAs include a short history of frames precisely because a single frame is sub-Markov for most manipulation tasks. The general treatment of "the state the agent observes is not the true Markov state" is the partially observable MDP (POMDP) of Kaelbling, Littman, and Cassandra (1998). We won't develop the POMDP machinery in this book, but the name is worth knowing, because every time a VLA paper talks about "history conditioning," it's effectively reaching for a finite-history approximation of the underlying POMDP.

## Policies, returns, and value

A policy $\pi$ is a mapping from states to actions. A deterministic policy $\pi : \mathcal{S} \to \mathcal{A}$ commits to a single action in each state. A stochastic policy $\pi(a \mid s)$ gives a probability distribution over actions in each state, and this is the form almost every learned policy in this book takes. RT-1 and OpenVLA both output a distribution over discretized action tokens; Diffusion Policy samples from a learned conditional distribution; π0 (arXiv:2410.24164) samples from a flow-matching distribution. The notation $a \sim \pi(\cdot \mid s)$, which you'll see in almost every algorithm box from here on, means "sample an action from the policy's distribution conditioned on the current state."

A trajectory under policy $\pi$ starting from state $s_0$ is the sequence $s_0, a_0, r_0, s_1, a_1, r_1, s_2, \ldots$ generated by sampling $a_t \sim \pi(\cdot \mid s_t)$ and $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ at each step, with $r_t = R(s_t, a_t, s_{t+1})$. The return from time $t$ is the discounted sum

$$
G_t = \sum_{k=0}^{\infty} \gamma^{k} r_{t+k}.
$$

The value of a state under policy $\pi$ is the expected return starting from that state:

$$
V^{\pi}(s) = \mathbb{E}_{\pi}[\,G_t \mid s_t = s\,].
$$

And the Q-value (or action-value) is the expected return starting from state $s$, taking action $a$, and following $\pi$ thereafter:

$$
Q^{\pi}(s, a) = \mathbb{E}_{\pi}[\,G_t \mid s_t = s, a_t = a\,].
$$

Those three objects, return, value, action-value, are the only ones you need to keep in your head for the next two sections. Value iteration and policy iteration in §5.2 are algorithms for computing $V^\pi$ and $Q^\pi$ when $P$ and $R$ are known; Q-learning in §5.3 is the algorithm for estimating $Q^\pi$ when they aren't. Almost everything in modern deep RL is, structurally, a way to fit a parametric approximation of $V$ or $Q$ (the critic) and use it to update a parametric approximation of $\pi$ (the actor).

## A worked gridworld

To make the tuple concrete, here's a small MDP written out explicitly. The state space $\mathcal{S}$ is the sixteen cells of a $4 \times 4$ grid, indexed $(i, j)$ for $i, j \in \{0, 1, 2, 3\}$. The goal cell is $(3, 3)$ and is absorbing. The action space is $\mathcal{A} = \{\text{N}, \text{S}, \text{E}, \text{W}\}$. The transition $P$ is deterministic for now: action $\text{N}$ from $(i, j)$ produces $(i-1, j)$ if that cell exists, and $(i, j)$ otherwise (a no-op against the wall). The reward $R(s, a, s') = -1$ for every step until $s'$ is the goal, at which point the episode ends. The discount is $\gamma = 0.95$.

Three things are worth noticing about this five-line specification. It fits on a napkin, and so does every gridworld variant you'll see for the next two chapters; don't let the formal notation suggest the underlying object is complicated. The reward is sparse-ish: negative-one-per-step is the convention that turns "reach the goal" into "reach the goal as fast as possible," a convention that exists because a purely sparse reward (zero everywhere, one at the goal) is harder to learn from in tabular settings. And the optimal policy is obvious to a human, move toward the goal, but the algorithm that discovers it without being told is the substance of the chapter, and the rest of §5.2 through §5.4 covers how that algorithm works and why it sometimes doesn't.

## What the MDP buys you, and what it costs

The MDP gives you three things the symbolic models of §4.1 don't. A training signal: the reward $R$ is precisely what the symbolic-action engineer was missing when she had to write down all effects by hand. Uncertainty in the dynamics: $P$ is a distribution, and the agent can be optimal in expectation even when the world is non-deterministic. And a compositional notion of optimality: the value function is recursive in the structure of the state space, which is what makes the Bellman equation in §5.2 the workhorse of the field's next thirty years.

What it costs is the assumption that the world fits the tuple. The state space has to be specifiable, the transition has to be Markov in that state space, the reward has to be expressible as a scalar, and the discount has to capture the actual horizon of the task. None of these are automatic on a real robot, and §5.5 is dedicated to the ways they fail. For now, the working agreement is that the MDP is the right vocabulary for talking about reward-driven action models, even when the specific MDP you write down for your robot is wrong in the third decimal place.

Section 5.2 picks up the value function and shows how it can be computed, via value iteration and policy iteration, whenever $P$ and $R$ are known.
