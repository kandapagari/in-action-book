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

Section 1.1 argued that turning intent into motion is structurally hard and
named four reasons. This section gets concrete. We will take a modern action
model apart and label its components, in language that the rest of the book
will use. By the end of the section you should be able to look at any paper in
the field — RT-1, OpenVLA, π0, Helix, GR00T N1, the next one published the
week after you read this — and identify the three slots that define it: what
goes in, what comes out, and what training signal told it the difference.

Three slots, in that order. They look simple. Most of the design space of the
field lives in the choices you make at each one.

## Slot 1 — Inputs: what the policy is allowed to look at

An action model takes some observation of the world and produces something the
robot can do with it. The first design choice is what counts as "observation."

The minimum is one RGB camera and a clock. Most modern systems use more. The
typical input bundle for a manipulation VLA is:

- One or more RGB images, usually from cameras mounted on the robot (a
  wrist-mounted camera that sees the gripper, and a stationary "agentview"
  camera that sees the scene). Image resolution is typically 224×224 or
  256×256 after preprocessing — small enough that a pretrained vision encoder
  can process it in milliseconds, large enough that table-scale geometry is
  visible.
- A natural-language instruction, usually a short imperative sentence: "pick up
  the red block," "open the drawer," "stack the cups." The instruction is
  optional in classical control and central in foundation-model-based
  approaches.
- Proprioception — the robot's own measurements of its joint angles, joint
  velocities, end-effector pose, and gripper width. This is the cheap data and
  it is always available; surprisingly, modern VLAs use less of it than you
  might expect, because they were designed for cross-embodiment generalization
  and proprioception schemas differ between robots.
- A short history of recent observations and actions. Some policies condition
  on the last image; some on the last *k* images; some on a sliding window of
  the full observation-action trajectory. Long context is more expressive,
  short context is faster, and the choice interacts with the closed-loop /
  open-loop problem we will return to in Chapter 6.

Three observations about this slot that will recur. First, the input modality
mix is *not* fixed by the problem; it is a design choice. The same task can be
done by a single-camera policy or a five-camera one, and the comparison is
non-trivial because more cameras means more compute and more places for the
model to overfit. Second, the inputs are heterogeneous: pixels, tokens,
floating-point joint angles. Combining them is its own architectural question,
which we will treat in Chapter 8 when we discuss tokenization and in Chapter 11
when we trace the CLIP → RT-1 lineage. Third, what you decide *not* to look at
matters. A policy that has access to a force-torque sensor will learn to use
it; a policy that does not will silently substitute visual approximations of
contact. Both can work; they fail differently.

## Slot 2 — Outputs: what the policy is allowed to do

The second design choice is the action space — the set of things the policy is
allowed to output. This choice has more consequences than any other single
decision in the model, because it determines what kind of motion the policy
can express and how the policy is trained.

Action spaces come in two cuts: by *type* and by *frame*. The type of an action
is what the number physically means. The frame is what coordinate system the
number is expressed in.

The four common types, ordered roughly from lowest- to highest-level:

- **Torques.** Direct motor commands. One number per joint, updated at the
  control loop rate (typically 200 Hz–1 kHz for an arm). This is the most
  expressive action space — anything the robot can do, you can in principle
  do with a torque sequence — and the hardest to learn. Almost no VLAs operate
  at this level; classical computed-torque controllers and some specialized
  legged-locomotion RL policies do.
- **Joint or end-effector velocities.** "Move joint 3 at 0.2 rad/s," or "move
  the end-effector at 5 cm/s in the +x direction." A layer above torques,
  with a low-level controller smoothing the velocity command into actual motor
  commands. Most teleoperation interfaces use this space; many imitation-
  learning policies match it.
- **Pose deltas.** "Move the end-effector 1 cm in +x, 2 mm in -z, rotate the
  wrist by 0.05 rad, close the gripper." This is the space modern VLAs almost
  universally emit: a small relative motion per timestep, applied by an
  inverse-kinematics controller that figures out the joint commands. OpenVLA's
  seven-dimensional action (three translation, three rotation, one gripper)
  is a canonical example.
- **Waypoints or trajectories.** "Move to this pose, then this pose, then this
  pose." A single inference produces a sequence of future poses rather than a
  single delta. This is what diffusion policies (Chapter 10) and the
  continuous-action heads in π0 and Octo (Chapters 12–13) produce.

The frame distinction cuts across all four types. Most action spaces are
expressed *relative to the current end-effector pose* — a "go 1 cm forward
from where you are" rather than "go to absolute world coordinate (0.45, 0.10,
0.30)." Relative actions generalize better because they do not depend on the
calibration of the robot's base frame, and they are easier to compose. Almost
every contemporary VLA uses relative actions. The exceptions are systems with
strong global scene grounding (some 3D-aware VLAs like LEO in Chapter 15) and
end-to-end driving policies (OpenDriveVLA), where the relevant frame is the
world, not the agent.

A second axis cuts across the type: **discrete vs. continuous representation**.
Even within "pose deltas," a model can either output continuous floating-point
numbers — the natural representation — or discretize each dimension into a
fixed number of bins and predict bin indices. RT-1, RT-2, and OpenVLA all
chose discretization (256 bins per axis is canonical), because it lets them
reuse a language-model decoder head and a cross-entropy loss. π0 and Octo went
the other way, with continuous outputs from a diffusion or flow-matching head.
We will spend Chapter 10 on the trade-off; for the anatomy, what matters is
that the same physical action — "move 1 cm in +x" — gets represented and
trained differently depending on which side of this choice the architecture
made.

## Slot 3 — Training signal: how the model learns to fill the gap

The third slot is what tells the model that one mapping from inputs to outputs
is better than another. This is where action models split into the four
families we will name in Section 1.4, and it is the slot where the methods
have changed the most over the last fifty years.

Three training-signal types dominate the modern field. They are not exclusive;
most contemporary systems use two of them in combination.

- **Supervised imitation from demonstrations.** The signal is a dataset of
  (observation, action) pairs collected by a human teleoperator. The loss is
  whatever you would use for a regular sequence-prediction problem — typically
  cross-entropy if the action space is discrete, or mean-squared-error /
  diffusion-style loss if it is continuous. This is the dominant signal in
  modern VLAs, partly because the data is the easiest to collect at scale and
  partly because it works well *enough*, even though it inherits the
  closed-loop / open-loop mismatch we flagged in §1.1.
- **Reinforcement learning from rewards.** The signal is a scalar reward
  function, hand-designed or learned, that says "this trajectory accomplished
  the task; that one did not." The model learns to maximize the expected
  return. RL is the natural fit for tasks where a demonstration is hard to
  produce but a success criterion is easy to write down — locomotion, in
  particular. It is also the natural fit for *fine-tuning* a policy that was
  initially trained by imitation; you start with something that mostly works,
  then use RL to polish the parts that do not. Chapter 5 develops this in
  detail.
- **Self-supervised pretraining from large unlabeled corpora.** The signal is
  a pretext task — predicting masked image patches, predicting the next text
  token, contrastively aligning images and text — that does not require robot
  data at all. The model learns useful representations from internet-scale
  vision and language data, and is then adapted to robots through a smaller
  second stage. CLIP (Chapter 11) is the canonical example, and every modern
  VLA inherits most of its capacity from this stage.

The modern recipe, almost without exception, layers all three: self-supervised
pretraining on internet data, supervised imitation on robot demonstrations,
and (optionally) reinforcement-learning fine-tuning to close the last gap.
When you read a paper, the most informative question is not "which signal does
it use?" but "in what proportions, and in what order?"

## A worked instance: OpenVLA in three slots

The anatomy is easier to remember if you pin it to a real model. Take
OpenVLA (Kim et al. 2024, arXiv:2406.09246), the open-source VLA you ran in
Chapter 2.

- **Inputs.** One RGB image, resized to 224×224, encoded by a combination of
  SigLIP (a vision-language pretraining objective) and DINOv2 (a
  self-supervised visual representation). A natural-language instruction,
  tokenized by the Llama-2 tokenizer. No proprioception, no history beyond
  the current frame.
- **Outputs.** Seven discrete action tokens, one per axis of an end-effector
  pose delta: three translation, three rotation (axis-angle), one gripper.
  Each axis is binned into 256 buckets fitted to the per-axis range of the
  training data. The tokens come out of the same decoder that would otherwise
  produce text — they live in the bottom 256 entries of Llama's vocabulary,
  repurposed.
- **Training signal.** Two stages, both supervised. Stage one: the Llama-2
  + SigLIP + DINOv2 backbone arrives pretrained on internet vision-language
  data — self-supervised, no robot data. Stage two: imitation on 970,000
  robot trajectories from the Open X-Embodiment dataset, with cross-entropy
  loss over the discrete action tokens.

Three slots, four design choices each, one paragraph. You can do this exercise
for any model in the Model Zoo (Appendix F) and the structure of the design
space falls out.

## Where this differs from a perception model and from a planner

Two contrasts close out this section, because the entire premise of *action
models* — and of this book — is that they are a distinct object from the two
things they are most often confused with.

A *perception model* — an image classifier, an object detector, a
vision-language model — has Slot 1 and Slot 3 but not a Slot 2 in the sense
above. Its outputs are labels, segments, or natural-language responses, not
commands a robot will execute. A perception model can be a component of an
action model, and indeed every modern VLA has a perception model embedded in
it. But the embedding is not free: the perception model has to be wired into
a head that emits actions, and a training signal that grounds those actions
has to be added. Most of the engineering effort in OpenVLA, RT-2, and π0 lives
in that wiring.

A *planner* — STRIPS, PDDL, motion planners like RRT or PRM — has Slot 2 but
typically not Slot 1 or Slot 3 in the sense above. It takes a symbolic or
geometric description of the world (not raw sensor input), it produces an
action or trajectory (Slot 2), and it does not learn from data — the rules are
written. Classical planners are extremely good at certain things action models
are bad at, and Chapter 4 makes that case in detail. They are bad at certain
things action models are good at, which is why the two coexist in modern
robotic stacks rather than one replacing the other.

The action models this book is about sit in the middle: they accept raw
high-dimensional sensor input like a perception model, they produce executable
actions like a planner, and they learn the mapping from data rather than
having it written. That combination is what makes them new and what makes them
hard. Section 1.3 traces the history of how the field arrived at that
combination — and Section 1.4 names the four families that share the slot
structure but differ in how they fill it.
