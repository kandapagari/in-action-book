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

Start with an admission: action is the part of robotics that has resisted us the longest. The bottleneck of the field has moved several times over the last forty years, and each time it moved, the people working on the previous bottleneck declared the problem solved. In the 1980s the bottleneck was perception. Cameras were noisy, compute was scarce, and getting a robot to reliably segment a coffee cup from a table counted as a research result on its own. Deep learning took care of that in the 2010s, and the bottleneck shifted to language understanding and reasoning, because a robot still couldn't be told, in plain English, to "make me coffee" — the gap between an instruction and a sequence of joint commands stayed unbridged. Language understanding fell in turn, to a different generation of large neural networks, and the bottleneck settled back onto what it had probably always been: turning intent into motion. Call it the action problem.

This chapter explains why that problem is hard, why it resisted progress that perception and language did not, and why we finally have something worth writing a textbook about. The rest of Section 1.1 lays out four properties of the action problem that make it structurally different from what came before. Section 1.2 anatomizes a modern action model and gives you the vocabulary the rest of the book uses. Sections 1.3 and 1.4 trace the field's history and the four families of action models we'll study.

## A concrete scene

Picture a robot arm with a parallel-jaw gripper in front of a tabletop holding a coffee cup, a sponge, and a small puddle of spilled milk. You tell the robot, in English, "clean up the spill." What does it actually have to do? Roughly: locate the spill, recognize the sponge as the relevant tool, estimate where to grasp it so the wrist clears the cup, plan a path from the gripper's current pose to that grasp pose, close the fingers with the right force, lift, traverse to the puddle, lower the sponge until it contacts the table at the right pressure, drag it across the puddle at the right normal force and surface velocity, lift again, drop the sponge somewhere reasonable, and stop. That's twenty-odd decisions. Half of them are analog. Several are contact-rich. Every one depends on the one before it.

A modern vision-language model can look at this scene and describe exactly what should happen. It can write you a paragraph of plain English laying out the plan. What it can't do, on its own, is *do it*. The gap between the paragraph and the motion is the territory of this book.

## Four reasons action is structurally hard

Why is that gap so wide? Four properties of the action problem set it apart from its perceptual and linguistic cousins.

### 1. The output space is high-dimensional, continuous, and constrained

A modern image classifier picks one of a thousand labels. A language model picks one of fifty thousand tokens. A robot policy picks a vector of, typically, six to thirty real-valued numbers, and it picks a new one every twenty milliseconds.

The numerical structure matters here. The classifier's output space is a probability simplex over a finite set, so you train it by minimizing cross-entropy and evaluate it by counting hits. The robot's output space is a six- or thirty-dimensional Euclidean manifold (or its tangent, if you parameterize in velocities or torques), constrained by kinematics, joint limits, torque limits, and contact dynamics. Two policies that look numerically close under any standard distance metric can produce wildly different physical behaviors. A millimeter of vertical error matters when you're picking up a coin and doesn't when you're stacking boxes. What counts as "close" is task-dependent in a way that "close" between two images simply isn't.

Geometry complicates things further. The space of rigid-body rotations is the 3-sphere SO(3), not Euclidean 3-space, and learning a policy that respects this without being told to sits at the edge of what current methods can do. Most action models cheat: they parameterize rotations as axis-angle deltas in some local frame, accept the singularities, and hope the training data avoids them. That works until it doesn't, and "until it doesn't" is exactly the distribution-edge case where students get confused and engineers get paged.

### 2. The training signal is sparse, expensive, and biased

The internet has been a free training corpus for perception and language for the last decade. Tens of billions of labeled images exist, and even more text. Robot actions have no equivalent. A single robot demonstration, one trajectory lasting a few seconds, typically costs a human ten to thirty seconds of teleoperation, plus the capital cost of the robot itself. Until very recently, all the robot data in the world fit on a small server.

The Open X-Embodiment dataset, a 21-institution collaboration released in 2024, collected over one million episodes across 22 different embodiments, and that counts, in 2026, as a watershed moment in the field. ImageNet, by comparison, had fourteen million labeled examples back in 2009 and was considered a minimum viable resource at the time. Robot data is three orders of magnitude smaller than vision data was when modern computer vision got going, and that scaling gap is unlikely to close soon, because acquiring robot data requires physical hardware, not a web crawler.

What data does exist is biased in ways perceptual data isn't. Demos come from a particular set of robots, in a particular set of labs, performing a particular set of tasks under particular lighting, run by particular teleoperators with particular ergonomic habits. The trajectories tend to run too smooth, too uniformly successful, too short. Behavior cloning on data like that produces policies that work beautifully inside the narrow regime of the training distribution and fail in interesting ways outside it. So much of the modern VLA literature is, structurally, dataset engineering. Chapter 15 spends a long time on what robot data actually looks like under the hood.

### 3. The environment is not a function of your action — it's a process

A perception model maps an input to an output. So does a translation model, and so does, in a deliberately simplified view, a chat model. At core, the relationship is a function.

A robot policy is part of a control loop instead. It produces an action, the environment responds, and the new environment state determines what the next action should be. This control-theoretic structure changes every part of the machine-learning pipeline. Loss functions need to account for trajectories, not single predictions. Evaluation needs to account for compounding error, since a tiny mistake at step one becomes a large mistake at step one hundred. The training data is, in spirit, meant to be closed-loop (you want the policy to learn what to do after it's just done something slightly wrong), but almost all of it gets collected open-loop, because the demonstrator was doing things correctly the whole time. Chapter 6 returns to this at length when we cover compounding error and DAgger. For now it's enough to know that this closed-loop/open-loop mismatch is the single biggest reason imitation learning is harder than supervised learning, even when the data looks similar on paper.

There's a second consequence of the closed-loop structure: a robot policy has to run in real time. A language model with forty seconds of per-token latency is a research curiosity. A robot policy with forty seconds of per-action latency is a non-functional product. Inference compute budgets for robot policies get measured in tens of milliseconds, not seconds, and that constraint reaches back into architecture decisions the rest of deep learning never has to think about. The dual-system architectures covered in Chapter 14, Helix and GR00T N1 among them, exist specifically because a single 7B-parameter forward pass runs too slow for the inner control loop of a humanoid robot.

### 4. The cost of being wrong is asymmetric

A misclassified image causes embarrassment. A wrong word in a chat response causes confusion. A wrong robot action, at the wrong instant, with the wrong magnitude, breaks the cup, breaks the robot, or breaks the person standing nearby.

That asymmetry carries real technical and methodological weight. Technically, it means any deployed robot policy needs a safety layer that isn't trained from data and that has well-understood worst-case behavior; Chapter 17 is devoted to this. Methodologically, it means the field's success metric can't just be average performance, it has to include worst-case behavior across edge cases. A 95%-success-rate policy that fails by dropping objects on the floor is a productivity tool. A 95%-success-rate policy that fails by closing the gripper on a human hand is a recall notice. Their summary statistics might look identical.

This is the one property, of the four, where machine learning progress has helped the least. The other three (high-dimensional output, sparse data, closed-loop dynamics) are problems the deep-learning toolkit has been chipping away at for a decade. Asymmetric cost is more or less unchanged since 1985, and the modern toolkit barely touches it. When you read a paper claiming a VLA "generalizes" to a new task, the first question worth asking is what the failure mode looked like for the cases that didn't work.

## So why is *now* the moment?

Given those four properties, it's fair to ask why a textbook on action models is worth writing in 2026 at all. The answer comes in three pieces.

First, foundation models have closed the language-understanding part of the problem at industrial scale. The "make me coffee" instruction is now a tractable input rather than a research result, because the same vision-language model powering your favorite chat assistant can be reused as the front end of a robot policy. Grounding that instruction in a specific scene is no longer gatekept by how hard the grounding step used to be.

Second, the data wall isn't falling, but it's getting climbable. Open X-Embodiment is the watershed moment; the next generation of multi-institution collaborative datasets, steadily improving teleoperation interfaces, and increasingly photorealistic synthetic data from simulators all push in the same direction. The Stanford AI Index Report 2025 noted that peer-reviewed papers on foundation-model-based robot control grew over 60% between 2022 and 2024, outpacing every other applied-AI subfield. That growth was, structurally, made possible by the dataset side of the ecosystem finally catching up.

Third, the methods that bridge perception, language, and action, the methods this book is about, have crystallized into a recognizable recipe: tokenize the action space, pretrain on a vision-language corpus, co-train or fine-tune on robot demonstrations, then decode. The recipe has variants — discrete action tokens in RT-2 and OpenVLA, continuous flow-matching heads in π0, dual-system decompositions in Helix and GR00T N1 — but it is a recipe, and a textbook can teach a recipe.

What a textbook can't do, and here's the honest version of the answer, is make any of the four structural problems above go away. They're still hard. They still bite. A reader who finishes this book won't have made action easy; they'll have learned the vocabulary, the techniques, and the failure modes of the current generation of approaches, and they'll be positioned to push the field forward exactly where the methods don't yet work. That's what we mean by *robot learning*, and the next section gets specific about what its central object, the action model itself, actually is.
