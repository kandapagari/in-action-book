---
chapter: 1
section: 1.2
title: "Anatomy of an action model: inputs, outputs, training signal"
target_words: 2000
status: draft
prereqs: §1.1 (the four reasons action is hard)
key_refs:
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Sutton & Barto (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
  - Argall et al. (2009). A Survey of Robot Learning from Demonstration. RAS 57(5).
---

# 1.2  Anatomy of an action model: inputs, outputs, training signal

Section 1.1 argued that turning intent into motion is structurally hard and named four reasons why. This section gets concrete. We'll take a modern action model apart and label its components in language the rest of the book will keep using. By the end, you should be able to look at any paper in the field, RT-1, OpenVLA, π0, Helix, GR00T N1, whatever gets published the week after you read this, and identify the three slots that define it: what goes in, what comes out, and what training signal told it the difference between right and wrong.

Three slots. They look simple, and most of the design space of the field lives in the choices made at each one.

## Slot 1 — Inputs: what the policy is allowed to look at

An action model takes some observation of the world and produces something the robot can do with it. The first design choice is what counts as "observation."

The minimum is one RGB camera and a clock. Most modern systems use more. A typical input bundle for a manipulation VLA looks like this:

- One or more RGB images, usually from cameras mounted on the robot: a wrist-mounted camera that sees the gripper, and a stationary "agentview" camera that sees the scene. Image resolution is typically 224×224 or 256×256 after preprocessing, small enough for a pretrained vision encoder to process in milliseconds, large enough that table-scale geometry stays visible.
- A natural-language instruction, usually a short imperative sentence such as "pick up the red block," "open the drawer," or "stack the cups." The instruction is optional in classical control and central in foundation-model-based approaches.
- Proprioception: the robot's own measurements of joint angles, joint velocities, end-effector pose, and gripper width. This data is cheap and always available. Modern VLAs use less of it than you'd expect, though, because they're designed for cross-embodiment generalization, and proprioception schemas differ from robot to robot.
- A short history of recent observations and actions. Some policies condition on the last image, some on the last *k* images, some on a sliding window across the full observation-action trajectory. Long context is more expressive; short context is faster. The choice interacts with the closed-loop/open-loop problem we'll return to in Chapter 6.

A few things about this slot recur throughout the book. The input modality mix isn't fixed by the problem, it's a design choice: the same task can be done by a single-camera policy or a five-camera one, and comparing them isn't trivial, since more cameras means more compute and more surface area for the model to overfit. The inputs are also heterogeneous — pixels, tokens, floating-point joint angles — and combining them is its own architectural question, one we treat in Chapter 8 on tokenization and Chapter 11 on the CLIP-to-RT-1 lineage. And what you decide *not* to look at matters just as much as what you do. A policy with access to a force-torque sensor will learn to use it. A policy without one will quietly substitute visual approximations of contact instead. Both can work. They fail differently.

## Slot 2 — Outputs: what the policy is allowed to do

The second design choice is the action space, the set of things the policy is allowed to output. Few decisions carry more downstream consequences, because this choice determines what kind of motion the policy can express and how it gets trained.

Action spaces come in two cuts: by *type* and by *frame*. The type of an action is what the number physically means. The frame is what coordinate system the number is expressed in.

Four common types, ordered roughly from lowest- to highest-level:

- **Torques.** Direct motor commands, one number per joint, updated at the control loop rate (typically 200 Hz to 1 kHz for an arm). This is the most expressive action space; anything the robot can physically do, you can in principle express as a torque sequence. It's also the hardest to learn. Almost no VLAs operate at this level, though classical computed-torque controllers and some specialized legged-locomotion RL policies do.
- **Joint or end-effector velocities.** "Move joint 3 at 0.2 rad/s," or "move the end-effector at 5 cm/s in the +x direction." One layer above torques, with a low-level controller smoothing the velocity command into actual motor commands. Most teleoperation interfaces use this space, and many imitation-learning policies match it.
- **Pose deltas.** "Move the end-effector 1 cm in +x, 2 mm in -z, rotate the wrist by 0.05 rad, close the gripper." Modern VLAs almost universally emit this kind of action: a small relative motion per timestep, applied by an inverse-kinematics controller that works out the joint commands. OpenVLA's seven-dimensional action (three translation, three rotation, one gripper) is a canonical example.
- **Waypoints or trajectories.** "Move to this pose, then this pose, then this pose." A single inference produces a sequence of future poses rather than one delta. Diffusion policies (Chapter 10) and the continuous-action heads in π0 and Octo (Chapters 12 and 13) work this way.

The frame distinction cuts across all four types. Most action spaces get expressed relative to the current end-effector pose: "go 1 cm forward from where you are" rather than "go to absolute world coordinate (0.45, 0.10, 0.30)." Relative actions generalize better, since they don't depend on how the robot's base frame happens to be calibrated, and they're easier to compose. Nearly every contemporary VLA uses relative actions. The exceptions are systems with strong global scene grounding, some 3D-aware VLAs like LEO in Chapter 15, and end-to-end driving policies like OpenDriveVLA, where the relevant frame is the world rather than the agent.

A second axis cuts across the type: discrete versus continuous representation. Even within "pose deltas," a model can output continuous floating-point numbers, the natural representation, or discretize each dimension into a fixed number of bins and predict bin indices instead. RT-1, RT-2, and OpenVLA all chose discretization (256 bins per axis is standard), because it lets them reuse a language-model decoder head and a cross-entropy loss. π0 and Octo went the other way, with continuous outputs from a diffusion or flow-matching head. We'll spend Chapter 10 on that trade-off. For now, what matters is that the same physical action, "move 1 cm in +x," gets represented and trained differently depending on which side of this choice the architecture landed on.

## Slot 3 — Training signal: how the model learns to fill the gap

The third slot is what tells the model that one mapping from inputs to outputs beats another. This is where action models split into the four families we'll name in Section 1.4, and it's the slot where methods have changed the most over the last fifty years.

Three training-signal types dominate the field today, and they aren't mutually exclusive. Most contemporary systems combine at least two.

- **Supervised imitation from demonstrations.** The signal is a dataset of (observation, action) pairs collected by a human teleoperator. The loss is whatever you'd use for a regular sequence-prediction problem: cross-entropy if the action space is discrete, mean-squared error or a diffusion-style loss if it's continuous. This is the dominant signal in modern VLAs, partly because the data is easiest to collect at scale and partly because it works well enough, even though it inherits the closed-loop/open-loop mismatch flagged in §1.1.
- **Reinforcement learning from rewards.** The signal is a scalar reward function, hand-designed or learned, that says this trajectory accomplished the task and that one didn't. The model learns to maximize expected return. RL fits naturally where a demonstration is hard to produce but a success criterion is easy to write down, locomotion being the clearest case. It also fits fine-tuning a policy already trained by imitation: you start with something that mostly works, then use RL to polish the parts that don't. Chapter 5 develops this in detail.
- **Self-supervised pretraining from large unlabeled corpora.** The signal is a pretext task, predicting masked image patches, predicting the next text token, contrastively aligning images and text, that needs no robot data at all. The model learns useful representations from internet-scale vision and language data, then gets adapted to robots through a smaller second stage. CLIP (Chapter 11) is the canonical example, and every modern VLA inherits most of its capacity from this stage.

The modern recipe, almost without exception, layers all three: self-supervised pretraining on internet data, supervised imitation on robot demonstrations, and optionally reinforcement-learning fine-tuning to close the last gap. When you read a paper, the useful question isn't "which signal does it use?" It's "in what proportions, and in what order?"

## A worked instance: OpenVLA in three slots

The anatomy sticks better once it's pinned to a real model. Take OpenVLA (Kim et al. 2024, arXiv:2406.09246), the open-source VLA you ran in Chapter 2.

- **Inputs.** One RGB image, resized to 224×224, encoded by a combination of SigLIP (a vision-language pretraining objective) and DINOv2 (a self-supervised visual representation). A natural-language instruction, tokenized by the Llama-2 tokenizer. No proprioception, no history beyond the current frame.
- **Outputs.** Seven discrete action tokens, one per axis of an end-effector pose delta: three translation, three rotation (axis-angle), one gripper. Each axis is binned into 256 buckets fitted to the per-axis range of the training data. The tokens come out of the same decoder that would otherwise produce text, repurposed from the bottom 256 entries of Llama's vocabulary.
- **Training signal.** Two stages, both supervised. Stage one: the Llama-2 + SigLIP + DINOv2 backbone arrives pretrained on internet vision-language data, self-supervised, no robot data involved. Stage two: imitation on 970,000 robot trajectories from the Open X-Embodiment dataset, with cross-entropy loss over the discrete action tokens.

Three slots, four design choices each, one paragraph per model. Run this exercise on anything in the Model Zoo (Appendix F) and the shape of the design space falls right out.

## Where this differs from a perception model and from a planner

Two contrasts close out this section, because the entire premise of action models, and of this book, is that they're a distinct object from the two things they get confused with most often.

A perception model (an image classifier, an object detector, a vision-language model) has Slot 1 and Slot 3 but no real Slot 2. Its outputs are labels, segments, or natural-language responses, not commands a robot will execute. A perception model can be a component of an action model, and every modern VLA has one embedded in it, but the embedding isn't free. The perception model has to get wired into a head that emits actions, and a training signal that grounds those actions has to be added on top. Most of the engineering effort in OpenVLA, RT-2, and π0 lives in exactly that wiring.

A planner, think STRIPS, PDDL, motion planners like RRT or PRM, has Slot 2 but typically lacks Slot 1 and Slot 3 in the sense we've been using. It takes a symbolic or geometric description of the world rather than raw sensor input, produces an action or trajectory, and doesn't learn from data at all, since the rules are written by hand. Classical planners are extremely good at certain things action models struggle with, and Chapter 4 makes that case in detail. They're bad at certain things action models handle well, which is why the two coexist in modern robotic stacks instead of one replacing the other.

The action models this book is about sit in the middle. They accept raw high-dimensional sensor input like a perception model, produce executable actions like a planner, and learn the mapping from data rather than having it written down. That combination is what makes them new, and what makes them hard. Section 1.3 traces how the field arrived at that combination, and Section 1.4 names the four families that share the slot structure but differ in how they fill it.
