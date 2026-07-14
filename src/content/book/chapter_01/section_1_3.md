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

Writing the history chapter of a textbook on a hot new field tempts you to treat the past as a series of failed attempts, now superseded. Resist that. The history of action models isn't a history of failures. It's a history of the bottleneck moving, through perception, planning, language, control, data, deployment, and at each move, the methods of the previous era kept solving the problems they'd always been good at. Almost every modern VLA still relies on a classical inverse-kinematics solver somewhere in the stack. Almost every legged-locomotion controller still relies on a hand-tuned reinforcement-learning reward. The eras compound. They don't replace each other.

What follows is a selective tour through six eras, meant to give you the names and the conceptual shifts. It isn't a complete bibliography (the Model Zoo in Appendix F and the chapter-by-chapter references in Appendix E.2 do that job). Read this section as a map you can hold in one hand, not a survey.

## Era 1 — Symbolic planning (1970s)

The original action models were symbolic. STRIPS, Fikes and Nilsson's 1971 system, was developed at SRI's Shakey project, the first mobile robot built to navigate a physical lab and execute tasks described in natural language. The representation was logical: the world was a set of predicates ("the door is open," "the robot is in room B"), actions were operators with preconditions and effects, and a plan was a sequence of operators transforming an initial state into a goal state.

For a 2026 textbook on learned action models, STRIPS looks ancient, and in many ways it's the opposite of what fills the rest of this book. So why include it? Because the abstraction it introduced, that a robot's job is to search over a sequence of operators that transform world states, is still the right one. PDDL (McDermott et al., 1998), STRIPS's modern descendant, is alive and well in benchmarks like the International Planning Competition. Modern hierarchical-RL systems and many task-and-motion-planning frameworks still use it to represent the task layer even when the motion layer is fully learned. Chapter 4 argues that the real dividing line between Era 1 and Eras 2 through 6 isn't "symbolic versus learned." It's "manipulating a discrete set of operators versus emitting continuous low-level commands," and modern stacks do both, at different levels of the same system.

## Era 2 — Geometric and dynamic robotics (1980s–1990s)

The 1980s and 1990s were the era of control: kinematics, dynamics, and motion planning treated as branches of applied mathematics. The canonical references, Spong, Hutchinson, and Vidyasagar's *Robot Modeling and Control*, Murray, Li, and Sastry's *A Mathematical Introduction to Robotic Manipulation*, LaValle's *Planning Algorithms*, are all still in print, and the methods they describe are still load-bearing in modern robots. Forward and inverse kinematics, computed-torque control, RRTs and PRMs for collision-free path planning, impedance and force control: none of this is a legacy technique. It's the substrate learned policies operate on top of.

Era 2's contribution to action models in the modern sense is mostly indirect, since none of these methods learn from data; they're derived from a model of the robot and the task instead. But they established two ideas the rest of the book will use without spelling out again. First, the action space is structured: joint angles, joint velocities, end-effector poses, and torques relate to each other through known geometry, and switching among them just means choosing the right level of abstraction. Second, some problems are easier to write down than to learn. Inverse kinematics for a six-degree-of-freedom arm is a small system of equations. Collision checking in a known environment is a graph search. When a modern VLA produces an end-effector pose delta, an inverse-kinematics solver from Era 2 turns it into joint commands, and an Era-2 collision checker decides whether to execute it at all. Chapter 4 treats this hand-off as the load-bearing point it actually is.

## Era 3 — First learned approaches (late 1980s–2000s)

The first credible attempt to learn an action model from data was Pomerleau's ALVINN (1988), a fully connected neural network with one hidden layer that mapped a 30×32 camera image of a road to a steering angle. ALVINN drove a van across the United States in 1995, nearly thirty years before the deep-learning revolution, and it's the direct ancestor of every modern behavior-cloning policy.

The 1990s and 2000s saw parallel growth in reinforcement learning for robotics. The single best reference for this era is the Kober, Bagnell, and Peters survey (IJRR, 2013), which captures everything from policy-search methods on humanoids to motor-primitive learning on industrial arms. The era produced impressive demonstrations: pendulum swing-up, ball-in-cup, robot table tennis. What it didn't produce was generalization. Each system was hand-engineered for its task, the policies were small (often a few hundred parameters), and transfer between tasks stayed a research result rather than a default. The data problem from §1.1, sparse, expensive, biased, was acute, and the methods of the time had no good answer for it.

Era 3's contribution is the existence proof: a robot's action model can be learned. The specific methods that did the learning would later get displaced by deep learning, but the framing, policy as a function approximator, training as optimization against a reward or a demonstration, is exactly the framing the rest of the field inherited.

## Era 4 — Deep reinforcement learning (2013–2017)

The arrival of deep learning in robotics is usually dated to DeepMind's DQN paper (Mnih et al., Nature 2015), not because the paper was about robotics but because it showed a convolutional neural network could serve as the function approximator inside a Q-learning agent and learn to play Atari games from raw pixels. The lesson generalized fast. Within two years, Levine, Finn, Darrell, and Abbeel had end-to-end visuomotor policies trained with guided policy search. Within four, the algorithmic toolkit for deep RL on robots, TRPO (Schulman et al. 2015), DDPG (Lillicrap et al. 2016), PPO (Schulman et al. 2017), SAC (Haarnoja et al. 2018), had stabilized into the list of names you still see in every paper today.

Era 4 also discovered, expensively, the limits of deep RL on real robots. RL needs environments to interact with, and interacting with a real robot for millions of episodes just isn't practical. Simulation became the workhorse, and sim-to-real, the problem of training in simulation and deploying on hardware, turned into a subfield of its own. Domain randomization (Tobin et al., 2017) was the era's signature trick, and the legged-locomotion successes of the late 2010s, ANYmal, Cassie, Spot, all rest on it.

The era's contribution to modern action models runs conceptual rather than direct. Deep RL isn't, in 2026, the dominant training signal for foundation action models; most VLAs train by imitation, with optional RL fine-tuning on top. But the recognition that a deep neural network could be the policy came from this era, and the algorithmic infrastructure it produced, replay buffers, target networks, advantage estimation, is still what you reach for when imitation isn't enough. Chapter 7 develops all of this in depth.

## Era 5 — Deep imitation and the sequence-model turn (2017–2021)

While Era 4 was working out deep RL, a quieter parallel track was rediscovering imitation. The argument was pragmatic: if reward design is the hardest part of RL, and demonstrations are cheaper than reward functions for most tasks anyone actually cares about, maybe the right move is to scale up the ALVINN idea with modern architectures. BC-Z (Jang et al., CoRL 2021) was the strong demonstration, a CNN-based behavior-cloning policy that generalized to unseen tasks given language conditioning. The language thread matters here; it's the first point in this lineage where natural-language instructions became inputs to a policy at all. The pattern only accelerated from there.

In parallel, the transformer architecture started showing up in control. Chen et al.'s Decision Transformer (NeurIPS 2021) showed you could cast reinforcement learning as a sequence-prediction problem: feed in (state, action, return-to-go) tuples, ask a transformer to predict the next action. The Trajectory Transformer (Janner, Li, Levine, NeurIPS 2021) did something similar from a planning angle. Neither one was a robot system. Both gestured toward unifying perception, language, and control inside one architecture, and that unification arrived in the next era.

Era 5's contribution is the recognition that the closed-loop trajectory is the natural unit, and that a transformer is the natural architecture for sequences built from mixed-modality tokens. The vocabulary the next era's foundation action models would use, action tokens, return-to-go, language conditioning, got assembled here first.

## Era 6 — Foundation action models (2022 → now)

The era we live in starts with RT-1 (Brohan et al., 2022, arXiv:2212.06817), a transformer policy trained on 130,000 demonstrations across 700-plus tasks at Google, using language conditioning and an action-tokenization scheme that would later become canonical. RT-1 was a generalist by 2022 standards; by 2026 standards it reads as a primitive baseline. The trajectory from there moves fast.

RT-2 (Brohan et al., 2023, arXiv:2307.15818) made the conceptual leap: take an off-the-shelf vision-language model, PaLI-X, PaLM-E, and reuse it as the policy backbone, with discrete action tokens slotted into the bottom of the vocabulary. Suddenly a policy inherited the world knowledge of a trillion-token language model trained on internet data. This "VLM-as-policy" moment is what gives the term Vision-Language-Action model its current meaning.

The open-source response arrived in 2024 with OpenVLA (Kim et al., arXiv:2406.09246), a 7-billion-parameter VLA built on Llama-2, SigLIP, and DINOv2, trained on 970,000 trajectories from the Open X-Embodiment dataset. OpenVLA matters not because it's the strongest VLA (it isn't) but because it's the first one a graduate student can actually run, fine-tune, and modify. The rest of this book leans on it as the canonical running example for exactly that reason.

At the same time, a parallel track abandoned discrete action tokens for continuous heads. Octo (UC Berkeley, 2024, arXiv:2405.12213) plugged a diffusion head onto a transformer backbone. π0 (Physical Intelligence, 2024, arXiv:2410.24164) plugged a flow-matching head onto a PaliGemma backbone and trained on ten thousand hours of robot data, producing dexterous behavior, laundry-folding, egg-packing, that earlier discrete-token VLAs simply couldn't manage.

The 2025 generation added a third axis: dual-system architectures, where a slow VLM "System 2" handles scene understanding and a fast sensorimotor "System 1" handles real-time control. Figure AI's Helix runs on the Figure 02 humanoid at BMW's Spartanburg plant; NVIDIA's GR00T N1 (arXiv:2503.14734) is the open-research counterpart. Both architectures exist because a single 7B-parameter forward pass runs too slow for the inner control loop of a humanoid robot, the exact latency argument previewed back in §1.1.

And the era isn't finished. As of mid-2026, the model zoo (Appendix F) lists 24 named systems, including efficiency-frontier checkpoints (SmolVLA, TinyVLA, RoboMamba) that run on consumer GPUs, long-horizon variants (LiLo-VLA, Long-VLA, Embodied-R1), 3D-grounded models (LEO), and cross-domain ones (OpenDriveVLA for driving). The recipe itself, pretrain on internet vision-language data, fine-tune on robot demonstrations, decode, has stabilized. The variants are where the field's research energy lives now.

## The through-line

Six eras, fifty-five years, one direction of travel. Each era took on a weaker form of supervision than the one before it: Era 1 took rules, Era 2 took a model of the robot, Era 3 took a hand-crafted reward or a small dataset, Era 4 took rewards plus simulation, Era 5 took demonstrations and language, and Era 6 takes whatever it can get, internet-scale vision-language data, cross-embodiment demonstration corpora, a small amount of robot-specific fine-tuning. The trend runs monotone, and nothing suggests it's about to stop.

What stays constant across eras is the three-slot anatomy from §1.2. Every system has inputs, outputs, and a training signal. What changes is which parts of each slot get designed by hand and which get learned. Section 1.4 picks out the four broad families this lineage produced and gives each one a name.
