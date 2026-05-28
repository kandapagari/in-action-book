---
chapter: 1
section: 1.1
title: Why "action" is the hard part of robotics
target_words: 2000
status: draft
prereqs: linear algebra, basic Python
key_refs:
  - Kober, Bagnell, Peters (2013). RL in Robotics: A Survey. IJRR 32(11).
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications, Challenges. arXiv:2505.04769.
  - Stanford HAI (2025). AI Index Report — Robotics chapter.
---

# 1.1  Why "action" is the hard part of robotics

A useful way to start a book on action models is to admit, up front, that action
is the part of robotics that has resisted us the longest. The bottleneck of the
field has migrated several times over the last forty years, and at each
migration the people working on the previous bottleneck tended to declare the
problem solved. In the 1980s the bottleneck was perception: cameras were noisy,
compute was scarce, and a robot that could reliably segment a coffee cup from a
table was a research result. In the 2010s perception fell to deep learning, and
the bottleneck became language understanding and reasoning: a robot still could
not be told, in plain English, to "make me coffee," because the gap between an
instruction and a sequence of joint commands was unbridged. In the 2020s
language understanding fell — to a different generation of large neural networks
— and the bottleneck once again became what it has, on reflection, always been:
turning intent into motion. The action problem.

This chapter is about why that problem is hard, why it has resisted progress
that perception and language did not, and why — finally — we have something
worth writing a textbook about. The rest of Section 1.1 lays out four properties
of the action problem that explain why it is structurally different from the
problems that came before it. Section 1.2 anatomizes a modern action model and
gives you the vocabulary the rest of the book will use. Sections 1.3 and 1.4
trace the history and the four families of action models we will study.

## A concrete scene

Consider a robot arm with a parallel-jaw gripper, sitting in front of a tabletop
that has on it a coffee cup, a sponge, and a small puddle of spilled milk. You
tell the robot, in English, "clean up the spill." What does it have to do? In
rough order: locate the spill, recognize that the sponge is the relevant tool,
estimate where to grasp the sponge so the wrist clears the cup, plan a path from
the gripper's current pose to that grasp pose, close the fingers with the right
amount of force, lift, traverse to the puddle, lower the sponge until it
contacts the table with the right pressure, drag the sponge across the puddle
with the right normal force and surface velocity, lift again, drop the sponge
somewhere reasonable, and stop. Twenty-odd decisions, half of them analog,
several of them contact-rich, and every one of them dependent on the previous.

A modern vision-language model can look at this scene and tell you exactly what
should happen. It can write you a paragraph of plain English describing the
plan. What it cannot do, in itself, is *do it*. The gap between the
paragraph and the motion is the territory of this book.

## Four reasons action is structurally hard

Why is that gap so wide? Four properties of the action problem set it apart from
its perceptual and linguistic cousins.

### 1. The output space is high-dimensional, continuous, and constrained

A modern image classifier picks one of a thousand labels. A language model picks
one of fifty thousand tokens. A robot policy picks a vector of, typically, six
to thirty real-valued numbers — and it picks one every twenty milliseconds.

The numerical structure matters. The classifier's output space is a probability
simplex over a finite set; you can train it by minimizing cross-entropy and you
can evaluate it by counting hits. The robot's output space is a six- or
thirty-dimensional Euclidean manifold (or its tangent, if you parameterize in
velocities or torques) with hard constraints from kinematics, joint limits,
torque limits, and contact dynamics. Two policies that look numerically close
under any standard distance metric can produce wildly different physical
behaviors: 1 mm of vertical error matters when you are picking up a coin, and
does not when you are stacking boxes. The right notion of "close" is
task-dependent in a way that the right notion of "close" between two images is
not.

The geometric structure also matters. The space of rigid-body rotations is the
3-sphere SO(3), not Euclidean 3-space, and learning a policy that respects this
without you telling it to is at the boundary of what current methods can do.
Most action models cheat — they parameterize rotations as axis-angle deltas in
some local frame, accept the singularities, and hope the training data avoids
them. That works until it does not, and "until it does not" is exactly the
distribution-edge case where students get confused and engineers get paged.

### 2. The training signal is sparse, expensive, and biased

The internet, for the last decade, has been a free training corpus for
perception and language. There are tens of billions of labeled images, and even
more text. For robot actions, there is no equivalent. A robot demonstration —
one trajectory, lasting a few seconds — typically costs a human ten to thirty
seconds of teleoperation, plus the capital cost of the robot itself. Until very
recently, *all* the robot data in the world fit on a small server.

The Open X-Embodiment dataset, a 21-institution collaboration released in 2024,
collected over one million episodes across 22 different embodiments — and that
is, in 2026, considered a watershed moment in the field. By comparison, ImageNet
in 2009 had fourteen million labeled examples and was considered a minimum
viable resource. Robot data is *three orders of magnitude smaller* than vision
data was when modern computer vision started, and the scaling gap is unlikely to
close, because data acquisition for robots requires physical hardware.

The data that does exist is biased in ways that perceptual data is not. Demos
are collected from a particular set of robots, in a particular set of labs,
performing a particular set of tasks under particular lighting conditions, by
particular human teleoperators with particular ergonomic preferences. The
trajectories tend to be too smooth, too uniformly successful, and too short.
Behavior cloning on such data produces policies that work beautifully in the
narrow regime of the training distribution and fail interestingly outside it.
This is why so much of the modern VLA literature is, structurally, dataset
engineering — and why Chapter 15 spends a long time on what robot data actually
looks like under the hood.

### 3. The environment is not a function of your action — it is a process

A perception model maps an input to an output. So does a translation model. So
does — in a deliberately simplified view — a chat model. The relationship is, at
core, a function.

A robot policy is part of a control loop. It produces an action; the environment
responds; the new environment state determines what the next action should be.
This control-theoretic structure changes every part of the machine-learning
pipeline. Loss functions need to account for trajectories, not just single
predictions. Evaluation needs to account for compounding error: a tiny mistake
at step one becomes a large mistake at step one hundred. The training data is
inherently *closed-loop* in spirit — you want the policy to learn what to do
after it has just done something slightly wrong — but is almost always collected
*open-loop* — the demonstrator was doing things correctly. Chapter 6 returns to
this point at length when we discuss compounding error and DAgger; for now, it
is enough to know that the closed-loop / open-loop mismatch is the single most
important reason why imitation learning is harder than supervised learning, even
when the data looks the same.

There is a second consequence of the closed-loop structure: a robot policy has
to run in *real time*. A language model with a forty-second per-token latency
is a research curiosity; a robot policy with a forty-second per-action latency
is a non-functional product. Inference compute budgets for robot policies are
measured in tens of milliseconds, not seconds, and this constraint reaches back
into architecture choices that the rest of deep learning does not have to think
about. The dual-system architectures of Chapter 14 — Helix, GR00T N1 — exist
specifically because a single 7B-parameter forward pass is too slow for the
inner control loop of a humanoid robot.

### 4. The cost of being wrong is asymmetric

A misclassified image causes embarrassment. A wrong word in a chat response
causes confusion. A wrong robot action — at the wrong instant, with the wrong
magnitude — breaks the cup, breaks the robot, or breaks the person nearby.

This asymmetry has technical and methodological consequences. The technical
consequence is that any deployed robot policy must be wrapped in a safety layer
that is *not* trained from data and that has well-understood worst-case
behavior. Chapter 17 spends a whole chapter on this. The methodological
consequence is that the field's success metric is not just average performance
but worst-case behavior across edge cases. A 95% success-rate policy that fails
by dropping objects on the floor is a productivity tool. A 95% success-rate
policy that fails by closing the gripper through a human hand is a recall. The
two policies might have indistinguishable summary statistics.

This is the one of the four properties where progress in machine learning has
helped the least. The other three — high-dimensional output, sparse data,
closed-loop dynamics — are problems that the deep-learning toolkit has been
incrementally chipping at for a decade. The asymmetric cost problem is more or
less unchanged from 1985, and the modern toolkit barely addresses it. When you
read a paper that claims a VLA "generalizes" to a new task, the first question
you should ask is: what was the failure mode of the cases that did not work?

## So why is *now* the moment?

Given those four properties, a fair question is why a textbook on action models
is worth writing in 2026 at all. The answer is in three pieces.

First, foundation models have closed the language-understanding part of the
problem at industrial scale. The "make me coffee" instruction is now a tractable
input, not a research result, because the same vision-language model that powers
your favorite chat assistant can be reused as the front end of a robot policy.
The work of *grounding* that instruction in a specific scene is no longer
gatekept by the difficulty of the grounding step.

Second, the data wall is not falling, but it is getting climbable. Open
X-Embodiment is the watershed; the next generation of multi-institution
collaborative datasets, the steady improvement in teleoperation interfaces, and
the increasing use of synthetic data from simulators that are becoming
photorealistic, all push in the same direction. The Stanford AI Index Report
2025 noted that peer-reviewed papers on foundation-model-based robot control
grew over 60% between 2022 and 2024, outpacing every other applied-AI subfield
— and that growth was, structurally, made possible by the dataset side of the
ecosystem catching up.

Third, the methods that bridge perception, language, and action — the methods
this book is about — have crystallized into a recognizable recipe. Tokenize the
action space. Pretrain on a vision-language corpus. Co-train or fine-tune on
robot demonstrations. Decode. The recipe has variants — discrete action tokens
in RT-2 and OpenVLA, continuous flow-matching heads in π0, dual-system
decompositions in Helix and GR00T N1 — but the recipe is a recipe. A textbook
can teach it.

What the textbook cannot do — and this is the honest version of the answer — is
make any of the four structural problems above go away. They are still hard.
They still bite. A reader who finishes this book will not have made action easy;
they will have learned the vocabulary, the techniques, and the failure modes of
the current generation of approaches, and they will be in a position to push the
field forward in places where the methods do not yet work. That is what we mean
by *robot learning*, and the next section gets specific about what its central
object — the action model itself — actually is.
