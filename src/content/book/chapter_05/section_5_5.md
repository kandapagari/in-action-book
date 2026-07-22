---
chapter: 5
section: 5.5
title: "The MDP-to-robot translation problem"
target_words: 2000
status: draft
prereqs: §5.1 (MDP tuple, partial observability, Markov property); §5.4 (reward misspecification, shaped rewards); §4.2–4.3 (kinematics, joint-space control)
key_refs:
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapter 3. MIT Press.
  - Kaelbling, Littman & Cassandra (1998). Planning and Acting in Partially Observable Stochastic Domains. Artificial Intelligence 101(1–2).
  - Brockman et al. (2016). OpenAI Gym. arXiv:1606.01540.
  - Tobin et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS.
  - Peng et al. (2018). Sim-to-real transfer of robotic control with dynamics randomization. ICRA.
---

# 5.5  The MDP-to-robot translation problem

The previous section worked on the reward function. It assumed the rest of the MDP tuple was already correctly constructed, that the state space made sense, the action space was a reasonable thing to optimize over, and episodes had sensible boundaries. On a gridworld with sixteen cells, those assumptions are free. On a physical robot, each one is a design decision with its own failure modes.

This section is about that translation step. You have a robot, a task, and now an understanding of MDPs. How do you turn the robot-and-task into a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ that a learning algorithm can actually make progress on? Four decisions dominate: what goes into the state, what counts as an action, how episodes are structured, and what to do about the fact that the $P$ you can train on in simulation isn't the $P$ you'll face on hardware. Getting any one of these wrong is sufficient to make a well-implemented algorithm completely fail to learn.

## State representation: what is Markov?

The Markov property says the state $s_t$ must contain everything relevant about the past. Formally: $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)$. The future is independent of history given the current state. On a gridworld with full visibility, this holds trivially, since the cell you're in tells you everything. On a robot, it almost never holds perfectly, and your choice of state representation determines how badly it's violated.

Consider a robot arm trained to place a cup on a coaster. A minimal state candidate is the joint position and velocity vector, say $q \in \mathbb{R}^{14}$ for a 7-DOF arm. That's Markov for the arm's own dynamics, but not for the task, since it doesn't include the cup's pose or the coaster's location. Add those and you get something like $\mathbb{R}^{14 + 7 + 3}$, but now you need a pose estimator measuring cup and coaster pose at 20 Hz with low enough noise to avoid corrupting the policy's input. Sensor noise and observation delay both break the Markov property in practice.

The end-to-end alternative is feeding the policy a raw image, 224×224×3, or a stack of three frames for temporal context, and letting it learn its own implicit state. This is what OpenVLA (arXiv:2406.09246) and RT-2 (arXiv:2307.15818) do. The image isn't Markov either: a single frame can't distinguish an arm that's accelerating from one at rest, which is why frame-stacking exists. But the failure is bounded and well-understood, and for most manipulation tasks the information lost by discarding velocity costs less than the engineering effort of maintaining a reliable 6-DOF pose estimator for every object in the scene.

When the state representation is systematically incomplete, not just noisy but structurally missing information the policy needs, the MDP becomes a partially observable Markov decision process, or POMDP (Kaelbling, Littman & Cassandra, 1998). A POMDP adds an observation space $\mathcal{O}$ and an observation function $\Omega(o \mid s, a)$: the agent sees $o$, not $s$. Formally this is the right model for most robotics problems. Practically, POMDP solvers are harder to scale, and most practitioners handle partial observability with one of three engineering approximations: add a recurrent layer so the policy can summarize its own history, stack recent observations as a pseudo-state, or choose a richer observation that makes the missing information less load-bearing. None of these recovers the full Markov guarantee; all of them work well enough in practice for a wide class of tasks.

A specific failure mode worth naming: if the task requires remembering what happened several seconds ago, which drawer was opened, which object was already picked, and the state doesn't include that memory, a Markovian policy can't solve the task. No amount of Q-learning will help; the algorithm converges, but to a policy that can't access the information it needs. The correct fix is changing the state representation, not the learning algorithm.

## Action space design

The action space $\mathcal{A}$ isn't fixed by the robot; it's chosen by the engineer. A 7-DOF arm can be controlled through joint torques, joint velocities, joint positions, Cartesian end-effector positions, Cartesian end-effector poses, or sequences of any of the above. Each choice gives a different $\mathcal{A}$, and the choice affects learning difficulty in ways that interact with the task.

Torque control is the most physically fundamental: the action is a vector of joint torques, and the robot's dynamics are governed by the equations of motion from §4.3. This gives the policy maximum expressive power, since any motion the robot is mechanically capable of is reachable. It also places the highest learning burden on the policy, because the mapping from torques to Cartesian endpoint motion passes through the robot's full inertia tensor and gravity compensation. A policy that outputs random torques will flail destructively; the exploration problem from §5.3 becomes a safety problem.

Position control wraps a low-level PD controller around the joint angles, so the policy sets target positions and the controller handles force. This is by far the most common choice in manipulation research. It sacrifices the ability to modulate contact forces (relevant for assembly, insertion, and any task where compliance matters) in exchange for a much calmer action space where random actions produce slow, bounded motion rather than uncontrolled flailing. OpenVLA outputs a 7-vector of joint-position deltas at 5 Hz for exactly this reason.

Chunked or trajectory actions push the abstraction level higher still. Instead of outputting a single control command per step, the policy outputs a short sequence of commands, a 16-step position trajectory, say, and the low-level controller executes the whole chunk before the next policy call. Diffusion Policy (Chapter 10) and ACT both operate in this regime. The MDP timestep is now one chunk, not one control cycle, which reduces the effective horizon and makes credit assignment easier. The cost is reduced reactivity: the robot commits to 16 steps before it can observe the environment again.

Control frequency is a related design variable. A policy running at 1 Hz sees the world as a sequence of slowly changing states; one running at 200 Hz sees rapid oscillations mostly irrelevant to the task. Mismatched frequencies, training at 10 Hz and deploying on hardware that samples at 200 Hz without synchronization, are a common source of sim-to-real failures that look like policy degradation but are actually a different MDP entirely.

There's no universally correct choice. The practical heuristic: use the lowest-level action space you can train stably, since it preserves the most expressiveness for fine-grained tasks. In practice this usually means position deltas at 10 to 20 Hz for most tabletop manipulation, with a separate compliance layer if the task involves contact-rich assembly.

## Episode structure

Every RL training loop has episodes: a start state, a sequence of steps, and a terminal condition. On a gridworld, episodes are tidy, since the agent starts at a fixed cell, reaches the goal or falls into a pit, and resets. On a physical robot, all three components require design decisions.

Resets are expensive and often nondeterministic. A robot that knocks an object off the table requires a human to replace it before the next episode. At 1 minute per episode and 100,000 episodes of training, that's 70 days of human labor per run. Simulators eliminate this cost; real hardware doesn't. Practical strategies include training entirely in simulation and accepting sim-to-real error, training on tasks that are self-resetting (a drawer that can be opened and closed indefinitely, say), and learning from demonstrations rather than RL, which requires far fewer trials.

Horizon length $T$ controls how far ahead the value function must reason. A task that takes 500 steps at 10 Hz lasts 50 seconds; value estimates must propagate over all 500 steps, and the discount factor $\gamma$ must be high enough (close to 1) that the terminal reward isn't discounted to near zero. High $\gamma$ slows value-function convergence; low $\gamma$ causes the agent to discount future outcomes so heavily it behaves myopically. For a task with a natural horizon of 50 seconds, $\gamma = 0.99$ is typical, meaning the reward at the last step is worth about $0.99^{500} \approx 0.007$ at the start, thin but nonzero. For a task with a horizon of 10 minutes, $\gamma$ must be tuned or the task reformulated with intermediate rewards.

Termination conditions control when an episode ends before the horizon. Early termination on success (the object was placed) is almost always good: it keeps episodes short and keeps the replay buffer from clogging with post-success idle steps. Early termination on failure (the arm collided, the object was dropped) is more nuanced. Terminating on every collision teaches the agent that collisions are absorbing, that nothing recoverable ever follows, which is accurate in some tasks and not in others. For contact-rich tasks, excessive early termination on collision teaches the agent to be so collision-averse it never makes contact with objects it needs to manipulate.

## The sim-to-real gap

For any non-trivial task, training on real hardware end-to-end isn't feasible at the sample counts RL typically requires. The standard engineering solution is training in a physics simulator and deploying on the real robot. This introduces the sim-to-real gap: the transition function $P_{\text{sim}}$ you trained on differs from the transition function $P_{\text{real}}$ you deploy on, and the policy has no way to detect the difference.

The gap has several sources. Contact dynamics, what happens when two surfaces touch, are both physically complex and computationally expensive to simulate faithfully. Most simulators used for RL training (MuJoCo, Bullet, Isaac) use simplified contact models that are fast and stable but inaccurate in subtle ways: they underrepresent surface friction variation, ignore deformable objects, and smooth over fast-contact transients. Sensor noise is typically modeled as additive Gaussian; real sensor noise isn't. Actuator dynamics, backlash, delay, saturation, are often ignored entirely in simulation.

The canonical treatment is domain randomization (Tobin et al., 2017; Peng et al., 2018): during training, randomly vary the parameters of the simulator, friction coefficients, object masses, actuator delays, visual textures, so the real world is, in some probabilistic sense, one sample from the distribution the policy has already been exposed to. If the policy has learned to be robust across a wide range of friction values, it will generalize to the real robot's friction even if no single simulation run matched it exactly.

Domain randomization works when the real world is plausibly in the support of the randomized distribution. It fails when the sim-to-real gap is caused by something qualitatively absent from the simulator, compliant objects, wet surfaces, cables, rather than by parameter mismatch. For those cases, the only reliable option is real-robot data, which brings the sample-complexity discussion full circle. Chapter 9 examines world models as a partial answer: if you can learn a good model of real-robot dynamics from a small amount of hardware data, you can train in that learned model rather than in a fixed simulator.

## When all four decisions interact

The practical failure mode usually isn't one bad choice in isolation; it's four mediocre choices interacting. A policy trained with position-control at 10 Hz on a task with a 200-step horizon, using an observation that omits object velocity, in a simulator with no friction randomization, will fail to deploy for at least three simultaneous reasons, and figuring out which one is load-bearing requires ablating each design decision separately. In a physics lab this is called controlling variables. In robot learning it's rarely done systematically, which is one reason published results are hard to reproduce.

The checklist before running RL on a new task runs roughly: confirm the observation is sufficient for the task (if it isn't, fix the observation before writing a reward), choose an action abstraction at the right granularity for the task's required precision, set the episode horizon and reset strategy with explicit knowledge of the training budget, and identify the top two or three sources of sim-to-real gap before training, so you know what to randomize.

None of this is specific to MDP theory. It's engineering. But it's the engineering the theory assumes has been done correctly, and that assumption is almost never true on the first pass.

The next section rounds out Chapter 5 with a summary of what the MDP formalism gives the reader and what it conspicuously leaves open, the latter being, in large part, the motivation for everything in Chapters 6 through 10.
