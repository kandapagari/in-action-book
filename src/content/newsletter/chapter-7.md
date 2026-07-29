---
chapter: 7
subject: "What breaks when a network replaces the Q-table"
---
Tabular Q-learning keeps one number for every state-action pair. Fine on a sixteen-cell gridworld; useless on anything a robot ever sees, where the states run to infinity and the agent almost never meets the same one twice. Chapter 7 swaps the table for a parametric function and then confronts what that swap breaks.

Syntactically, almost everything survives: the update still nudges each value toward its bootstrapped target. The convergence guarantee does not survive, and the distance between "the update looks identical" and "the update still converges" is the whole address of deep reinforcement learning. The chapter names the culprit precisely, the deadly triad of function approximation, bootstrapping, and off-policy training, and uses Baird's counterexample to show the instability is structural, not some quirk of deep nets. From there it walks through the three engineering hacks that turned a known-divergent idea into DQN, the network that learned Atari straight from pixels: a replay buffer, a target network, and a clipped regression loss. Those same ideas keep resurfacing, lightly disguised, across DDPG, TD3, SAC, and PPO later in the chapter. Read it on the site.
