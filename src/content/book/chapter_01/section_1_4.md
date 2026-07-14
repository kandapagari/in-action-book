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

The history in §1.3 was chronological. This section is taxonomic. Take the same fifty-five years of work and regroup it into four families of action model, named so you have vocabulary for the chapters ahead: classical/analytical, reinforcement-learning, imitation, and foundation/VLA. They aren't mutually exclusive, and almost every deployed modern robot is a stack containing at least two of them. But the differences matter, because each family fills the three slots from §1.2, inputs, outputs, training signal, in a structurally different way.

A taxonomy that doesn't earn its keep is just an organizing trick. This one earns it for two reasons. The right family is often determined by the data and supervision available for a task, not by the task itself, so once you can name what kind of supervision you actually have, the family follows. And when a modern system fails, and they all do somewhere, the failure mode is usually characteristic of one family inside the stack. Knowing which family is which shortens the debugging considerably.

## Family 1 — Classical / analytical action models

The first family treats the action problem as applied mathematics. Given a model of the robot (its kinematics, dynamics, and contact geometry) and a model of the task (a goal pose, a path constraint, a stability criterion), you derive the action sequence rather than learning it. There's no training data, no neural network involved. Just a system of equations and an algorithm that solves it.

Examples in this family are the workhorses of industrial robotics. STRIPS and its descendant PDDL, the Era-1 systems from §1.3, compute symbolic plans: sequences of operators that transform one logical world state into another. Inverse-kinematics solvers, closed-form for some arm geometries, numerical (Jacobian pseudo-inverse, damped least squares) for the rest, turn a desired end-effector pose into joint angles. Motion planners like RRT, RRT*, and PRM (LaValle 2006) compute collision-free paths through configuration space. Computed-torque controllers and operational-space controllers turn a desired trajectory into the motor torques that track it. Each of these is a classical action model, and none of them learns from data.

In §1.2's slot anatomy: Slot 1 (inputs) is a structured, low-dimensional representation of the world, a list of predicates, a goal pose, a known obstacle map. Slot 2 (outputs) is a symbolic plan, a joint trajectory, or a torque command, depending on the controller's level. Slot 3 (training signal) doesn't exist, since the system is derived, not trained.

The argument against this family is exactly what motivated everything in Eras 3 through 6: it doesn't scale to scenes that haven't been hand-described. Classical methods need a clean model of the world, and the world is usually messy. The argument for the family is strong nonetheless. When the model is clean, the methods are correct, fast, and certifiable. Industrial pick-and-place lines that move billions of parts a year run almost entirely on classical methods, because the model is clean (a fixturized part on a known conveyor) and the cost of being wrong is high. Chapter 4 develops the family in detail and makes the case that the right question for a modern roboticist isn't "classical or learned," it's where the dividing line goes in this particular system.

## Family 2 — Reinforcement-learning action models

The second family treats the action problem as optimization against a reward function. You define a reward, a scalar describing how well the robot is doing, and train a policy to maximize the expected sum of future rewards. Training data comes from the robot's own interactions with an environment (simulated, real, or a mixture), and the supervision signal is the reward itself.

This is the family that produced the deep-RL successes of Era 4 (DQN, TRPO, PPO, SAC) and that powers most of the modern legged-locomotion stack. ANYmal, Cassie, and Spot all use RL controllers for their gait policies, trained massively in simulation and deployed via sim-to-real transfer with domain randomization. The locomotion success isn't an accident: walking is the canonical case where a reward function is easy to write (forward velocity, upright posture, energy efficiency) but a demonstration is hard to provide, since you can't puppeteer a quadruped through a stumbling-on-gravel recovery the way you can puppeteer an arm through a pick.

In slot terms: Slot 1 (inputs) is whatever the policy observes from the environment, proprioception, often a depth image, sometimes a heightmap. Slot 2 (outputs) is typically joint targets or end-effector velocities, executed by a low-level controller running at higher rate. Slot 3 (training signal) is the reward, which isn't a target output but a scalar evaluation of one, and the machinery converting that scalar into a gradient (policy gradients, Q-learning, actor-critic) is what most of Chapters 5 through 7 cover.

The cost of belonging to this family is the cost of reward design. Writing a reward function that produces the behavior you actually want, without producing degenerate strategies that game the reward, ranks as the second-hardest problem in robot learning (the data problem still takes first place). Reward hacking is a recurring failure mode: a quadruped that learns to fall forward and slide because forward velocity gets rewarded and standing up doesn't, a manipulator that learns to flick the object off the table because "object is no longer visible" accidentally became part of the success criterion. Chapter 5 spends a full section on this. For now, what matters is that the family's strength, just write down what you want, doubles as its weakness, because what you write down is rarely what you actually want.

## Family 3 — Imitation action models

The third family treats the action problem as supervised learning from demonstrations. Instead of a reward, you have a dataset of (observation, action) pairs collected by a human teleoperator, and the loss is the prediction error between the policy's action and the demonstrator's.

This is the largest family in modern manipulation by a wide margin, for the practical reason §1.1 already named: for most tasks anyone cares about, demonstrations are cheaper than reward functions. ALVINN (1988) was the prototype. Behavior Transformer (Shafiullah et al., 2022), Diffusion Policy (Chi et al., 2023), ACT (Zhao et al., 2023), and BC-Z (Jang et al., 2022) are the modern instances. All of them share a Slot-1/Slot-2 structure inherited from supervised learning, observations in, actions out, and a Slot 3 that's simple in spirit (match the demonstrator) but pathological in practice, thanks to the closed-loop/open-loop mismatch from §1.1.

The defining problem of this family is compounding error. A policy trained to match demonstrations sees, during training, only the states the demonstrator visited. At deployment, it sees the states its own imperfect actions led to, which sit slightly off-distribution. That slight drift produces slightly worse actions, which produce more off-distribution states, which produce worse actions still. By the end of an episode the policy is operating in a regime it never saw during training, and the trajectory diverges. DAgger (Ross, Gordon, Bagnell, 2011) is the canonical fix: query the demonstrator on the policy's own visited states and fold those into the dataset. But DAgger is expensive, and modern practice rarely uses it as written. Instead it leans on more demonstrations and better architectures (history conditioning, diffusion heads, action chunking), accepting compounding error as the price of admission to this family.

A clean example of how seriously the field takes this trade-off is Action Chunking with Transformers (ACT, Zhao et al., 2023). Rather than predicting one action at a time, the policy predicts the next *k* actions in a single forward pass. Predicting a chunk makes the policy more robust to single-step errors and produces smoother behavior, at the cost of slower reaction to surprises. The choice of *k*, typically 8 to 16, is one of the architectural levers separating modern imitation systems from naive behavior cloning, and Chapter 6 returns to it.

## Family 4 — Foundation / VLA action models

The fourth family is the subject of the second half of this book. A foundation action model, a Vision-Language-Action model in the dominant subgenre, takes the imitation-learning recipe of Family 3 and adds two ingredients: a large-scale pretraining stage on internet vision-language data, and a cross-embodiment training stage on robot data aggregated across many labs and robot types. The result is a policy you can prompt in natural language, one that inherits world knowledge from a text-and-image backbone and is meant to generalize across tasks, scenes, and, with fine-tuning, robot embodiments.

The members of this family are the systems Era 6 produced. RT-1 (Brohan et al., 2022, arXiv:2212.06817) is the canonical first instance, an 8-task transformer trained on 130k demonstrations. RT-2 (Brohan et al., 2023, arXiv:2307.15818) marks the moment a vision-language model got reused as the policy backbone. OpenVLA (Kim et al., 2024, arXiv:2406.09246) is the open-source 7B-parameter VLA built on Llama-2, SigLIP, and DINOv2, trained on 970k Open X-Embodiment trajectories. Octo (arXiv:2405.12213) and π0 (Black et al., 2024, arXiv:2410.24164) are the continuous-action-head counterparts. The Sapkota survey (arXiv:2505.04769) and the Pure-VLA survey (Zhang et al., 2025, arXiv:2509.19012) catalogue the rest, and the Model Zoo in Appendix F names twenty-four of them along with parameter counts.

In slot terms: Slot 1 (inputs) is one or more RGB images plus a natural-language instruction, often with proprioception and a short history added. Slot 2 (outputs) is a pose delta or trajectory, represented either as discrete tokens (RT-2, OpenVLA) or as continuous output from a diffusion or flow-matching head (Octo, π0). Slot 3 (training signal) comes layered: self-supervised pretraining on internet data, supervised imitation on robot demonstrations, sometimes RL fine-tuning stacked on top. The slot diagram for a VLA looks like an imitation policy with a much larger Slot 3 and a far more heterogeneous Slot 1.

What makes this family genuinely new, and worth a textbook, is the compounding of training signals. A VLA inherits language understanding from a corpus of trillions of tokens, scene understanding from billions of images, and action grounding from roughly a million demonstrations. None of the previous three families can claim that stack. Whether the stack delivers what it promises, true cross-embodiment generalization, robustness to novel objects, long-horizon task execution, is the empirical question driving Chapters 11 through 15.

## Reading the four together

A worked exercise closes the section. Consider four ways to build a robot that empties a dishwasher.

A classical solution writes a PDDL plan ("pick plate from rack, place plate in cabinet, repeat") and an inverse-kinematics-plus-RRT motion planner that realizes each step. It needs a clean model of the kitchen.

A reinforcement-learning solution defines a reward, say +1 for each plate reaching the cabinet, −1 for each plate broken, and trains a policy in simulation through millions of episodes before sim-to-realing it to hardware. It needs a simulator that captures the dishwasher's geometry and the plates' contact dynamics.

An imitation solution collects fifty teleoperated demonstrations of an operator emptying the dishwasher and trains a Diffusion Policy on them. It needs the operator's time and a robot to demonstrate on.

A foundation/VLA solution takes OpenVLA, fine-tunes it on those same fifty demonstrations, and prompts it with "empty the dishwasher." It needs the demonstrations and the OpenVLA checkpoint both, but in exchange it inherits enough world knowledge to recognize plates it hasn't seen before, handle a misaligned rack, and respond to the operator's natural-language correction mid-task.

The fourth option is the most flexible and the most expensive at training time. The first is the most reliable when its assumptions hold. The middle two trade off in different directions depending on what's cheap for you to collect. A real deployed system, Figure 02's kitchen demos, GR00T-enabled humanoid pilots, the warehouse arms running π0 fine-tunes, usually contains pieces of all four. Section 1.5 sets out which of these the rest of the book covers in depth, and which it doesn't.
