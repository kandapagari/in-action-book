---
chapter: 1
section: 1.3
title: "A short history, from STRIPS to π0"
target_words: 2200
status: draft
prereqs: §1.1 (why action is hard), §1.2 (the three-slot anatomy)
key_refs:
  - Fikes & Nilsson (1971). STRIPS. Artificial Intelligence 2(3–4).
  - Pomerleau (1988). ALVINN. NeurIPS 1988.
  - LaValle (2006). Planning Algorithms. Cambridge University Press.
  - Kober, Bagnell, Peters (2013). RL in Robotics: A Survey. IJRR 32(11).
  - Mnih et al. (2015). Human-level control through deep RL. Nature 518.
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2. arXiv:2307.15818.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
---

# 1.3  A short history, from STRIPS to π0

There is a temptation, when writing the history chapter of a textbook on a hot
new field, to treat the past as a series of failed attempts that have now been
superseded. Resist that temptation. The history of action models is not a
history of failures. It is a history of the bottleneck moving — perception,
planning, language, control, data, deployment — and at each move, the methods
of the previous era kept solving the problems they had been good at all along.
Almost every modern VLA still relies on a classical inverse-kinematics solver
somewhere in the stack, and almost every legged-locomotion controller still
relies on a hand-tuned reinforcement-learning reward. The eras compound; they
do not replace.

What follows is a selective tour, in six eras, intended to give you the names
and the conceptual shifts. It is not a complete bibliography — the Model Zoo
in Appendix F and the chapter-by-chapter references in Appendix E.2 do that
work. Read this section as a map you can hold in one hand, not as a survey.

## Era 1 — Symbolic planning (1970s)

The original action models were symbolic. STRIPS — Fikes and Nilsson, 1971 —
was developed at SRI's Shakey project, the first mobile robot intended to
navigate a physical lab and execute tasks described in natural language. The
representation was logical: the world was a set of predicates ("the door is
open," "the robot is in room B"), actions were operators with preconditions
and effects, and a plan was a sequence of operators that transformed an
initial state into a goal state.

For a textbook on learned action models in 2026, STRIPS looks ancient and is
in many ways the opposite of what we will spend the rest of the book on. Why
include it at all? Because the *abstraction* it introduced — that a robot's
job is to search over a sequence of operators that transform world states —
is still the right one. PDDL (McDermott et al., 1998), the modern descendant
of STRIPS, is alive and well in benchmarks like the International Planning
Competition; modern hierarchical-RL systems and many task-and-motion-planning
frameworks still use it as a representation for the *task* layer even when
the *motion* layer is fully learned. Chapter 4 will argue that the dividing
line between Era 1 and Eras 2–6 is not "symbolic vs. learned" but rather
"manipulating a discrete set of operators vs. emitting continuous low-level
commands" — and modern stacks do both, at different levels.

## Era 2 — Geometric and dynamic robotics (1980s–1990s)

The 1980s and 1990s were the era of *control*: kinematics, dynamics, and
motion planning treated as branches of applied mathematics. The canonical
references — Spong, Hutchinson, and Vidyasagar's *Robot Modeling and Control*;
Murray, Li, and Sastry's *A Mathematical Introduction to Robotic
Manipulation*; LaValle's *Planning Algorithms* — are all still in print, and
the methods they describe are still load-bearing in modern robots. Forward
and inverse kinematics, computed-torque control, RRTs and PRMs for collision-
free path planning, impedance and force control — these are not legacy
techniques. They are the substrate on top of which learned policies operate.

Era 2's contribution to *action models* in the modern sense is mostly
indirect. None of these methods learn from data; they are derived from a
model of the robot and the task. But they established two ideas the rest of
the book will use without saying. First, that the action space is structured
— that joint angles, joint velocities, end-effector poses, and torques are
related to each other through a known geometry, and that switching among them
is a matter of choosing the right level of abstraction. Second, that some
problems are easier to *write down* than to learn. Inverse kinematics for a
six-degree-of-freedom arm is a small system of equations; collision checking
in a known environment is a graph search. When a modern VLA produces an
end-effector pose delta, an inverse-kinematics solver from Era 2 turns it
into joint commands, and an Era-2 collision checker decides whether to
execute it. Chapter 4 treats this hand-off as the load-bearing point it is.

## Era 3 — First learned approaches (late 1980s–2000s)

The first credible attempt to *learn* an action model from data was Pomerleau's
ALVINN (1988), a fully connected neural network with one hidden layer that
mapped a 30×32 camera image of a road to a steering angle. ALVINN drove a van
across the United States in 1995. It is the direct ancestor of every modern
behavior-cloning policy, and it predates the deep-learning revolution by
nearly thirty years.

The 1990s and 2000s saw the parallel growth of reinforcement learning in
robotics. The single best reference for this era is the Kober, Bagnell, and
Peters survey (IJRR, 2013) — it captures everything from policy-search
methods on humanoids to motor-primitive learning on industrial arms. The era
produced impressive demonstrations: pendulum-swing-up, ball-in-cup, robot
table tennis. What it did not produce was generalization. Each system was
hand-engineered for its task, the policies were small (often a few hundred
parameters), and transfer between tasks was a research result rather than a
default. The data problem we identified in §1.1 — sparse, expensive, biased —
was acute, and the methods of the time had no good answer to it.

Era 3's contribution is the existence proof: a robot's action model can be
learned. The methods that did the learning would later be displaced by deep
learning, but the framing — policy as a function approximator, training as
optimization against a reward or a demonstration — is the framing the rest of
the field would inherit.

## Era 4 — Deep reinforcement learning (2013–2017)

The arrival of deep learning in robotics is usually dated to the DeepMind DQN
paper (Mnih et al., Nature 2015) — not because the paper was about robotics,
but because it showed that a convolutional neural network could be the
function approximator inside a Q-learning agent and learn to play Atari
games from pixels. The lesson generalized fast. Within two years, Levine,
Finn, Darrell, and Abbeel had end-to-end visuomotor policies trained with
guided policy search; within four, the algorithmic toolkit for deep RL on
robots — TRPO (Schulman et al. 2015), DDPG (Lillicrap et al. 2016), PPO
(Schulman et al. 2017), SAC (Haarnoja et al. 2018) — had stabilized into the
list of names you still see in every paper.

Era 4 also discovered, expensively, the limits of deep RL on real robots. RL
needed environments to interact with, and interacting with a real robot for
millions of episodes is impractical. Simulation became the workhorse, and
*sim-to-real* — the problem of training in simulation and deploying on
hardware — became a sub-field of its own. Domain randomization (Tobin et al.,
2017) was the era's signature trick. The legged-locomotion successes of the
late 2010s — ANYmal, Cassie, Spot — all rest on it.

The era's contribution to modern action models is conceptual rather than
direct. Deep RL is not, in 2026, the dominant training signal for foundation
action models — most VLAs are trained by imitation, with optional RL fine-
tuning. But the recognition that a deep neural network could be the policy
came from this era, and the algorithmic infrastructure (replay buffers,
target networks, advantage estimation) is still what you reach for when
imitation is not enough. Chapter 7 develops all of this in depth.

## Era 5 — Deep imitation and the sequence-model turn (2017–2021)

While Era 4 was working out deep RL, a quieter parallel track was rediscovering
imitation. The argument was pragmatic: if reward design is the hardest part of
RL, and if demonstrations are cheaper than reward functions for most tasks of
practical interest, then maybe the right thing is to scale up the ALVINN idea
with modern architectures. BC-Z (Jang et al., 2022, CoRL 2021) was the strong
demonstration: a CNN-based behavior-cloning policy that generalized to unseen
tasks given language conditioning. The "language" thread is critical; it is
the first time, in this lineage, that natural-language instructions became
inputs to a policy. The pattern would only accelerate.

In parallel, the transformer architecture started appearing in control. Chen
et al.'s Decision Transformer (NeurIPS 2021) showed that you could cast
reinforcement learning as a sequence-prediction problem: feed in (state,
action, return-to-go) tuples, ask a transformer to predict the next action.
The Trajectory Transformer (Janner, Li, Levine, NeurIPS 2021) did something
similar from a planning perspective. Neither was a robot system; both were
gestures toward a unification of perception, language, and control inside one
architecture. That unification arrived in the next era.

Era 5's contribution is the recognition that the closed-loop *trajectory* is
the natural unit, and that the transformer is the natural architecture for
sequences of mixed-modality tokens. The vocabulary that the next era's
foundation action models would use — action tokens, return-to-go, language
conditioning — was assembled in Era 5.

## Era 6 — Foundation action models (2022 → now)

The era we live in starts with RT-1 (Brohan et al., 2022, arXiv:2212.06817):
a transformer policy trained on 130,000 demonstrations across 700+ tasks at
Google, using language conditioning and an action-tokenization scheme that
would later become canonical. RT-1 was, by 2022 standards, a generalist; by
2026 standards, it is a primitive baseline. The trajectory from there is
fast.

RT-2 (Brohan et al., 2023, arXiv:2307.15818) made the conceptual leap: take
an off-the-shelf vision-language model — PaLI-X, PaLM-E — and *reuse it as
the policy backbone*, with discrete action tokens slotted into the bottom of
the vocabulary. Suddenly a policy was inheriting the world knowledge of a
trillion-token language model trained on internet data. The "VLM-as-policy"
moment is what gives the term Vision-Language-Action model its current
meaning.

The open-source response arrived in 2024 with OpenVLA (Kim et al.,
arXiv:2406.09246): a 7-billion-parameter VLA built on Llama-2, SigLIP, and
DINOv2, trained on 970,000 trajectories from the Open X-Embodiment dataset.
OpenVLA matters not because it is the strongest VLA — it is not — but because
it is the first one a graduate student can actually run, fine-tune, and
modify. The rest of the book uses it as the canonical running example for
exactly that reason.

At the same time, a parallel track abandoned discrete action tokens in favor
of continuous heads. Octo (UC Berkeley, 2024, arXiv:2405.12213) plugged a
diffusion head onto a transformer backbone. π0 (Physical Intelligence, 2024,
arXiv:2410.24164) plugged a flow-matching head onto a PaliGemma backbone and
trained on ten thousand hours of robot data, producing dexterous behavior
(laundry-folding, egg-packing) that earlier discrete-token VLAs could not.

The 2025 generation added a third axis: *dual-system* architectures, in which
a slow VLM "System 2" handles scene understanding and a fast sensorimotor
"System 1" handles real-time control. Figure AI's Helix is deployed on the
Figure 02 humanoid at BMW's Spartanburg plant; NVIDIA's GR00T N1
(arXiv:2503.14734) is the open-research counterpart. Both architectures
exist because a single 7B-parameter forward pass is too slow for the inner
control loop of a humanoid robot — exactly the latency argument we previewed
in §1.1.

And the era is not done. As of mid-2026, the model zoo (Appendix F) contains
24 named systems, including efficiency-frontier checkpoints (SmolVLA, TinyVLA,
RoboMamba) that run on consumer GPUs, long-horizon variants (LiLo-VLA,
Long-VLA, Embodied-R1), 3D-grounded models (LEO), and cross-domain ones
(OpenDriveVLA for driving). The recipe — pretrain on internet vision-language
data, fine-tune on robot demonstrations, decode — is stable. The variants are
where the field's research energy now lives.

## The through-line

Six eras, fifty-five years, one direction of travel. Each era took on a
weaker form of supervision than the one before it. Era 1 took rules; Era 2
took a model of the robot; Era 3 took a hand-crafted reward or a small
dataset; Era 4 took rewards plus simulation; Era 5 took demonstrations and
language; Era 6 takes whatever it can get — internet-scale vision-language
data, cross-embodiment demonstration corpora, and a small amount of robot-
specific fine-tuning. The trend is monotone, and there is no reason to expect
it to stop.

What stays the same across eras is the three-slot anatomy from §1.2. Every
system has inputs, outputs, and a training signal. What changes is which
parts of each slot are designed and which are learned. Section 1.4 picks out
the four broad families this lineage produced and gives each one a name.
