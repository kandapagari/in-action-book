---
chapter: 7
section: 7.6
title: Summary
target_words: 2000
status: draft
prereqs: §7.1–§7.5; function approximation and DQN, policy gradients and the variance problem, PPO, off-policy actor-critic (DDPG/TD3/SAC), and sim-to-real via domain randomization
key_refs:
  - Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518:529-533. (DQN)
  - Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
  - Haarnoja, Zhou, Abbeel & Levine (2018). Soft actor-critic. ICML. arXiv:1801.01290.
  - Tobin et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS 2017.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.). MIT Press.
---

# 7.6  Summary

Chapter 7 was the deep-RL chapter, and it carried two jobs at once. The
first was to assemble the modern reinforcement-learning toolkit — the
algorithms that turn the tabular MDP machinery of Chapter 5 into
something that runs on a high-dimensional robot — in the order the field
actually built them: value-based methods, then policy gradients, then the
actor-critic synthesis that dominates control today. The second, quieter
job was to keep asking which parts of that toolkit survive into the
foundation-model era of Part 4 and which were superseded by the imitation
recipe of Chapter 6. This summary collects the load-bearing ideas and
marks where the later chapters keep reaching back for them.

## The four ideas worth carrying forward

*Function approximation breaks the comfortable guarantees of tabular RL,
and DQN is the engineering that makes it work anyway.* §7.1 traced the
move from a Q-table with one cell per state to a Q-network that
generalizes across states it has never seen — the only option once the
state is an image or a continuous joint configuration. The catch is that
the convergence proofs of Chapter 5 quietly assumed a table; combine
bootstrapping, off-policy data, and function approximation and you have
the "deadly triad" that can diverge outright (Sutton & Barto 2018). DQN
(Mnih et al. 2015) is not a new theory so much as a set of stabilizers —
a replay buffer to break the temporal correlation in the data, and a
slowly-updated target network to stop the regression target from chasing
its own tail. Recognizing those two tricks as answers to a stability
problem, rather than arbitrary implementation details, is what lets you
read every later off-policy method as a variation on the same theme.

*Policy gradients optimize the thing you actually want, and pay for it in
variance.* §7.2 made the case for a different family entirely: instead of
learning a value function and acting greedily, parameterize the policy
directly and push its parameters up the gradient of expected return. This
is the natural fit for continuous action spaces, where the `max` over
actions that Q-learning needs becomes intractable, and it is the lineage
that every VLA action head ultimately belongs to. The price is variance.
The REINFORCE estimator (Williams 1992) is unbiased but noisy, and the
whole craft of §7.2 — baselines, advantage estimation, GAE
(arXiv:1506.02438) — is machinery for cutting that variance without
introducing bias you cannot control. The bias-variance dial introduced
there is the same dial you turn when you tune any modern policy-gradient
method.

*PPO is the workhorse because it is hard to break.* §7.3 built PPO
(arXiv:1707.06347) as the answer to a practical question: how do you take
the largest policy-improvement step the data supports without stepping so
far that the policy collapses? The clipped surrogate objective is a cheap
stand-in for the trust region of TRPO — it simply refuses to reward the
update for moving the action probabilities too far from where they
started. The reason PPO, and not something more sophisticated, is the
default for sim-trained locomotion and for RLHF alike is robustness: it
tolerates sloppy hyperparameters, parallelizes cleanly across simulated
environments, and rarely diverges. When §6.5 and §7.5 spoke of
"fine-tuning a cloned policy with RL," PPO is almost always the verb.

*Off-policy actor-critic buys sample efficiency, which is the currency
that matters on hardware.* §7.4 closed the algorithmic arc with DDPG, TD3,
and SAC (arXiv:1801.01290), the methods that combine a learned critic with
a directly-parameterized actor and reuse a replay buffer the way DQN does.
The throughline from TD3's twin critics to SAC's entropy bonus is a
sequence of fixes for the same failure — an over-optimistic critic that
the actor learns to exploit — and SAC's maximum-entropy objective turns
exploration from a bolted-on heuristic into part of the loss. The reason
the chapter cared is the arithmetic of §7.5: PPO's on-policy data hunger
is fine in a free simulator and ruinous on a real arm, so when learning
must touch hardware, the order-of-magnitude sample-efficiency gain of an
off-policy method is the difference between feasible and not.

## What you should be able to do now

Four concrete capabilities, in roughly the order the rest of the book will
ask for them.

You should be able to *choose between a value-based, on-policy, and
off-policy method for a given problem, and justify the choice*. The
decision turns on a few questions the chapter equipped you to ask. Is the
action space discrete or continuous — DQN-family methods need the former,
policy-gradient and actor-critic methods handle the latter. Is data cheap
(a fast simulator) or expensive (a real robot) — the first tolerates
PPO's on-policy appetite, the second pushes you toward SAC's replay-buffer
efficiency. Do you need a robust default you can get running today, or are
you willing to tune for peak performance? Being able to walk that decision
tree out loud is the chapter's central practical skill.

You should be able to *explain why deep RL is unstable and name the
specific mechanisms that stabilize it*. Not "neural nets make RL hard" but
the deadly triad by name, and the concrete countermeasures: replay buffers
and target networks (§7.1), trust regions and clipping (§7.3), twin
critics and target-policy smoothing and entropy regularization (§7.4).
Each is a response to a named pathology — correlated data, a moving
target, an exploitable critic, premature exploitation. Reading a new RL
paper's "tricks" section as a catalogue of answers to these pathologies,
rather than as folklore, is what this chapter was meant to give you.

You should be able to *read the variance-reduction machinery of a
policy-gradient method and say what each piece buys*. A baseline reduces
variance without adding bias; an advantage function recenters the gradient
on whether an action beat expectations; GAE's λ trades a little bias for a
large variance reduction; PPO's clip keeps the update honest. Naming the
job of each component — rather than treating the loss function as an
indivisible incantation — is what lets you debug a policy-gradient method
that will not learn, and it connects directly back to the bias-variance
framing of §3.4.

You should be able to *frame sim-to-real as a distribution-design problem
and apply domain randomization accordingly*. §7.5's reframing is the
durable idea: the reality gap is ordinary train/test distribution shift,
and because you build the simulator you control the training
distribution. Randomize the parameters you cannot trust — textures and
lighting for vision (Tobin et al. 2017), masses and friction and latency
for dynamics — widely enough that the real robot looks like one more
sample. You should also be able to state the cost: robustness trades away
peak performance, the ranges are hand-chosen, and structural simulator
errors are not fixed by wider sampling.

## Where the chapter has set up the rest of the book

Chapter 7 hands off in three directions. The most immediate is the
diffusion and flow chapters of Part 3. The policy-gradient lineage of §7.2
ends, several chapters later, at the action heads of modern VLAs: a VLA
that outputs a continuous action distribution is parameterizing a policy,
and the question of how to represent and train that distribution is the
question §7.2 opened. Chapter 10's diffusion and flow-matching heads, and
π0's flow-matching objective in Chapter 13 (arXiv:2410.24164), are richer
answers to it than the Gaussian policy SAC assumes.

The deeper handoff is to the data argument that runs through Part 4. §7.5
ended by noting that the multi-robot, multi-environment datasets of
Chapters 12 and 15 are, in effect, domain randomization performed with
real robots instead of a simulator: train across enormous variation so
that the deployment case is just another sample. That instinct — that
robustness comes from the breadth of the training distribution, not from
matching any single condition exactly — is the same instinct that powers
foundation action models, scaled up by orders of magnitude. The reader who
internalized §7.5 will recognize Open X-Embodiment (arXiv:2310.08864) as
the same idea wearing different clothes.

The third handoff is to fine-tuning, in Chapter 16. The BC-then-RL pattern
that §6.5 identified as where the field converged is built from this
chapter's parts: clone a policy from demonstrations, then sharpen it past
human performance with PPO when a simulator and reward exist, or with an
off-policy method when sample budget is tight. Chapter 16's recipe for
adapting a foundation model to a new robot leans on knowing which RL tool
fits which budget — exactly the decision tree above.

## What the chapter has not covered

Two omissions are worth naming. The chapter stayed almost entirely within
*model-free* RL — methods that learn a value function or a policy directly
from transitions, never building an explicit model of the environment's
dynamics. That is a deliberate deferral, not an oversight: model-based RL,
where the agent learns a predictive model and plans or trains inside it,
is the entire subject of Chapter 9's world models, and the sample-
efficiency story that §7.4 told through off-policy reuse has a second,
larger chapter that only makes sense once world models are on the table.
Read this chapter's sample-efficiency discussion as the model-free half of
a two-part argument.

The chapter also treated exploration lightly. Beyond ε-greedy in §7.1 and
SAC's entropy bonus in §7.4, it did not cover the harder exploration
problem — sparse rewards, intrinsic motivation, curiosity, count-based
bonuses — that dominates RL research on tasks where reward is rare. For
robot manipulation the omission is defensible, because §6.1's argument
holds: when demonstrations are available, imitation sidesteps exploration
entirely, and that is precisely why the field's center of gravity moved
from RL to imitation for manipulation. The hard-exploration literature
matters most exactly where demonstrations are not available, which is not
where the rest of this book lives.

Chapter 7's contribution to the book's overall argument is to complete the
"learning from rewards" thread that Chapter 5 opened, and then to mark its
boundary. The deep-RL toolkit is real, powerful, and load-bearing in
locomotion and in fine-tuning, but it is not the engine of the foundation
action models the book builds toward — that engine is the imitation of
Chapter 6, scaled with data. The three findings to carry forward are that
deep RL is fundamentally a stability-engineering problem dressed as a
learning one, that the on-policy/off-policy split is really a data-budget
decision, and that sim-to-real is distribution design rather than physics
fidelity. Part 3 picks up the sequence-model and generative machinery that,
combined with this RL lineage and Chapter 6's imitation lineage, finally
produces a VLA.

§7.x closes the chapter with a hands-on exercise — training PPO and SAC on
the same continuous-control task and comparing their sample efficiency and
wall-clock cost — and the full reading list for the chapter.
