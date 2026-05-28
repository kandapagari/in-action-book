---
chapter: 1
section: 1.4
title: The four families of action models
target_words: 2000
status: draft
prereqs: §1.2 (the three-slot anatomy), §1.3 (the six-era history)
key_refs:
  - Fikes & Nilsson (1971). STRIPS. Artificial Intelligence 2(3–4).
  - LaValle (2006). Planning Algorithms. Cambridge University Press.
  - Mnih et al. (2015). Human-level control through deep RL. Nature 518.
  - Argall et al. (2009). A Survey of Robot Learning from Demonstration. RAS 57(5).
  - Pomerleau (1988). ALVINN. NeurIPS 1988.
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications, Challenges. arXiv:2505.04769.
---

# 1.4  The four families of action models

The history in §1.3 was chronological. This section is taxonomic. The same fifty-five
years of work, regrouped: four families of action model that the rest of the book
will study, named so that you have a vocabulary for the chapters ahead. The
families are *classical/analytical*, *reinforcement-learning*, *imitation*, and
*foundation/VLA*. They are not mutually exclusive, and almost every deployed
modern robot is a stack that contains at least two of them. But the differences
matter, because each family fills the three slots from §1.2 — inputs, outputs,
training signal — in a structurally different way.

A taxonomy that does not earn its keep is just an organizing trick. This one
earns it for two reasons. First, the right family is often determined by the
*data and supervision available* for the task, not by the task itself; once you
can name what kind of supervision you have, the family follows. Second, when a
modern system fails — and they all do, somewhere — the failure mode is usually
characteristic of one family inside the stack. Knowing which family is which
shortens the debugging.

## Family 1 — Classical / analytical action models

The first family treats the action problem as applied mathematics. Given a model
of the robot (its kinematics, dynamics, and contact geometry) and a model of
the task (a goal pose, a path constraint, a stability criterion), you *derive*
the action sequence rather than learning it. There is no training data; there
is no neural network. There is a system of equations and an algorithm that
solves it.

Examples in this family are the workhorses of industrial robotics. STRIPS and
its descendant PDDL (the Era-1 systems from §1.3) compute symbolic plans —
sequences of operators that transform one logical world state into another.
Inverse-kinematics solvers — closed-form for some arm geometries, numerical
(Jacobian pseudo-inverse, damped least squares) for the rest — turn a desired
end-effector pose into joint angles. Motion planners like RRT, RRT*, and PRM
(LaValle 2006) compute collision-free paths through configuration space.
Computed-torque controllers and operational-space controllers turn a desired
trajectory into the motor torques that will track it. Each of these is a
classical action model. None of them learns from data.

In the §1.2 slot anatomy: Slot 1 (inputs) is a structured, low-dimensional
representation of the world — a list of predicates, a goal pose, a known
obstacle map. Slot 2 (outputs) is a symbolic plan, a joint trajectory, or a
torque command, depending on the level of the controller. Slot 3 (training
signal) does not exist; the system is derived, not trained.

The argument *against* this family is the argument that motivated everything in
Eras 3–6: it does not scale to scenes that have not been hand-described.
Classical methods need a clean model of the world, and the world is usually
messy. The argument *for* the family — and it is a strong one — is that when
the model is clean, the methods are correct, fast, and certifiable. Industrial
pick-and-place lines that move billions of parts a year run almost entirely on
classical methods, because the model is clean (a fixturized part on a known
conveyor) and the cost of being wrong is high. Chapter 4 develops the family in
detail and makes the case that the right question for a modern roboticist is
not "classical or learned" but "where does the dividing line go in this
particular system."

## Family 2 — Reinforcement-learning action models

The second family treats the action problem as optimization against a reward
function. You define a reward — a scalar that says how well the robot is doing
— and you train a policy to maximize the expected sum of future rewards. The
training data comes from the robot's own interactions with an environment
(simulated, real, or a mixture), and the supervision signal is the reward.

This is the family that produced the deep-RL successes of Era 4 (DQN, TRPO,
PPO, SAC) and that powers most of the modern legged-locomotion stack. ANYmal,
Cassie, and Spot all use RL controllers for their gait policies, trained
massively in simulation and deployed via sim-to-real transfer with domain
randomization. The locomotion success is not an accident. Walking is the
canonical case where a reward function is easy to write (forward velocity,
upright posture, energy efficiency) and a demonstration is hard to provide
(you cannot puppeteer a quadruped through a stumbling-on-gravel recovery the
way you can puppeteer an arm through a pick).

In slot terms: Slot 1 (inputs) is whatever the policy observes from the
environment — proprioception, often a depth image, sometimes a heightmap. Slot
2 (outputs) is typically joint targets or end-effector velocities, executed by
a low-level controller at higher rate. Slot 3 (training signal) is the reward,
which is *not* a target output but a scalar evaluation of an output, and the
machinery for converting that scalar into a gradient (policy gradients,
Q-learning, actor-critic) is what most of Chapters 5–7 will be about.

The cost of being in this family is the cost of reward design. Writing a
reward function that produces the behavior you actually want, without
producing degenerate strategies that game the reward, is the second-hardest
problem in robot learning (the hardest is the data problem). Reward hacking is
a recurring failure mode: a quadruped that learns to fall forward and slide
because forward velocity is rewarded and standing up is not, a manipulator
that learns to flick the object off the table because "object is no longer
visible" was accidentally part of the success criterion. Chapter 5 spends a
full section on this; for now, what matters is that the family's strength —
"just write down what you want" — is also its weakness, because what you write
down is rarely what you want.

## Family 3 — Imitation action models

The third family treats the action problem as supervised learning from
demonstrations. Instead of a reward, you have a dataset of (observation,
action) pairs collected by a human teleoperator, and the loss is the
prediction error between the policy's action and the demonstrator's.

This is the largest family in modern manipulation, by a wide margin, for the
practical reason §1.1 already named: for most tasks of practical interest,
demonstrations are cheaper than reward functions. ALVINN (1988) was the
prototype. Behavior Transformer (Shafiullah et al., 2022), Diffusion Policy
(Chi et al., 2023), ACT (Zhao et al., 2023), and BC-Z (Jang et al., 2022) are
the modern instances. All of them share a Slot-1/Slot-2 structure inherited
from supervised learning — observations in, actions out — and a Slot-3 that is
straightforward in spirit (match the demonstrator) but pathological in
practice (the closed-loop / open-loop mismatch from §1.1).

The defining problem of this family is *compounding error*. A policy trained
to match demonstrations sees, at training time, only the states the
demonstrator visited; at deployment time, it sees the states *its own
imperfect actions* led to, which are slightly off-distribution. The slight
off-distribution leads to slightly worse actions, which lead to more
off-distribution states, which lead to worse actions still. By the end of an
episode, the policy is operating in a regime it never saw during training,
and the trajectory diverges. DAgger (Ross, Gordon, Bagnell, 2011) is the
canonical fix — query the demonstrator on the policy's own visited states and
add those to the dataset — but DAgger is expensive and is not what modern
practice typically uses. Modern practice uses *more demonstrations* and
*better architectures* (history conditioning, diffusion heads, action
chunking) and accepts that compounding error is the cost of the family.

A clean example of how seriously the field takes this trade-off is Action
Chunking with Transformers (ACT, Zhao et al., 2023): rather than predicting
one action at a time, the policy predicts the next *k* actions in one forward
pass. Predicting a chunk makes the policy more robust to single-step errors
and produces smoother behavior, at the cost of slower reaction to surprises.
The choice of *k* — typically 8 to 16 — is one of the architectural levers
that distinguishes modern imitation systems from naive behavior cloning, and
Chapter 6 returns to it.

## Family 4 — Foundation / VLA action models

The fourth family is the subject of the second half of this book. A foundation
action model — a Vision-Language-Action model, in the dominant subgenre —
takes the imitation-learning recipe of Family 3 and adds two ingredients:
(1) a large-scale *pretraining* stage on internet vision-language data, and
(2) a *cross-embodiment* training stage on robot data aggregated across many
labs and robot types. The result is a policy that can be prompted in natural
language, that inherits world knowledge from a backbone trained on text and
images, and that is intended to generalize across tasks, scenes, and (with
fine-tuning) robot embodiments.

The members of this family are the systems Era 6 has produced. RT-1 (Brohan
et al., 2022, arXiv:2212.06817) is the canonical first instance, an 8-task
transformer trained on 130k demonstrations. RT-2 (Brohan et al., 2023,
arXiv:2307.15818) is the moment a vision-language model was reused as the
policy backbone. OpenVLA (Kim et al., 2024, arXiv:2406.09246) is the
open-source 7B-parameter VLA built on Llama-2 + SigLIP + DINOv2 and trained
on 970k Open X-Embodiment trajectories. Octo (arXiv:2405.12213) and π0
(Black et al., 2024, arXiv:2410.24164) are the continuous-action-head
counterparts. The Sapkota survey (arXiv:2505.04769) and the Pure-VLA survey
(Zhang et al., 2025, arXiv:2509.19012) catalogue the rest, and the Model Zoo
in Appendix F gives names and parameter counts for twenty-four of them.

In slot terms: Slot 1 (inputs) is one or more RGB images plus a natural-
language instruction, often plus proprioception and a short history. Slot 2
(outputs) is a pose delta or trajectory, represented either as discrete
tokens (RT-2, OpenVLA) or as a continuous output from a diffusion or
flow-matching head (Octo, π0). Slot 3 (training signal) is layered:
self-supervised pretraining on internet data, supervised imitation on robot
demonstrations, sometimes RL fine-tuning on top. The slot diagram for a VLA
looks like an imitation policy with a much larger Slot 3 and a much more
heterogeneous Slot 1.

What makes this family genuinely new — and worth a textbook — is the
*compounding* of training signals. A VLA inherits language understanding from
a corpus of trillions of tokens, scene understanding from billions of images,
and action grounding from one million demonstrations. None of the previous
three families can claim that stack. Whether the stack gives you what it
promises — true cross-embodiment generalization, robustness to novel objects,
long-horizon task execution — is the empirical question that drives Chapters
11 through 15.

## Reading the four together

A worked exercise will close the section. Consider four ways to build a robot
that empties a dishwasher.

A *classical* solution writes a PDDL plan ("pick plate from rack, place plate
in cabinet, repeat") and an inverse-kinematics-plus-RRT motion planner that
realizes each step. It needs a clean model of the kitchen.

A *reinforcement-learning* solution defines a reward — say, +1 for each plate
in the cabinet, –1 for each plate broken — and trains a policy in simulation
through millions of episodes, then sim-to-reals it to hardware. It needs a
simulator that captures the dishwasher's geometry and the plates' contact
dynamics.

An *imitation* solution collects fifty teleoperated demonstrations of an
operator emptying the dishwasher and trains a Diffusion Policy on them. It
needs the operator's time and a robot to demonstrate on.

A *foundation/VLA* solution takes OpenVLA, fine-tunes it on those same fifty
demonstrations, and prompts it with "empty the dishwasher." It needs the
demonstrations *and* the OpenVLA checkpoint, but it inherits enough world
knowledge to recognize plates it has not seen, handle a misaligned rack, and
respond to the operator's natural-language correction mid-task.

The fourth option is the most flexible and the most expensive at training
time. The first option is the most reliable when its assumptions hold. The
middle two trade off in different directions. A real deployed system — Figure
02's kitchen demos, GR00T-enabled humanoid pilots, the warehouse arms running
π0 fine-tunes — usually contains pieces of all four. Section 1.5 sets out
which of these the rest of the book covers in depth and which it does not.
