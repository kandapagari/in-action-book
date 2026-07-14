---
chapter: 1
section: 1.6
title: Summary
target_words: 1800
status: draft
prereqs: §1.1–§1.5
key_refs:
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications, Challenges. arXiv:2505.04769.
---

# 1.6  Summary

Chapter 1 was, by design, the chapter with the least code in the book. It laid out the vocabulary, the history, and the shape of the field so the chapters that do contain code, and there's a lot of it coming, can move fast. This summary collects the load-bearing ideas in one place. If you've read the chapters in order, treat what follows as a checklist for what you should be able to do before turning the page. If you skipped ahead and came back, treat it as a refresher.

## The four ideas worth carrying forward

Action is structurally different from perception and language. The output space of a robot policy, a vector of continuous joint commands or pose deltas emitted every twenty milliseconds, isn't the output space of a classifier or a token predictor. §1.1 named four properties that set it apart: the output space is high-dimensional, continuous, and constrained; the policy runs in a closed loop where each action changes the next observation; training distribution and deployment distribution diverge by construction (compounding error); and the cost of being wrong gets paid in physical hardware rather than benchmark points. Every architectural choice in the rest of the book, action chunking, diffusion heads, flow matching, runtime safety monitors, responds to one of these four properties. When you read a paper and ask why this design and not the obvious one, the answer is almost always one of the four.

An action model is a function with three slots. §1.2 introduced the anatomy: Slot 1 is the input (observations and instructions, in whatever modality), Slot 2 is the output (the action representation, symbolic, joint target, end-effector delta, torque, or a tokenized version of any of these), and Slot 3 is the training signal (the loss or reward that produced the function). The anatomy is deliberately model-agnostic. It applies to a STRIPS planner (structured predicates in, symbolic plan out, no training at all), to a SAC policy (observations in, action distribution out, reward as signal), and to OpenVLA (image plus text in, action tokens out, imitation loss on cross-embodiment data). Every paper for the rest of the book can be re-read as a set of choices for these three slots, and that re-reading is more useful than it sounds. It's how you notice that two systems with different names are making the same architectural choice, and that two systems with similar names aren't.

The history is one of bottlenecks moving, not methods replacing each other. §1.3 walked through six eras, symbolic planning, classical control, deep RL, end-to-end imitation, the foundation moment, the VLA scale-up, and the point of that walk was that each era's methods survived the transition into the next one. STRIPS-descendant planners still run inside warehouse robots that ship a VLA on top of them. Inverse-kinematics solvers still convert pose deltas to joint commands at the bottom of the OpenVLA stack. RL still produces the locomotion policies that VLAs ride on. The eras compound rather than replace. The corollary for an engineer: "should I use a classical method or a learned method" is rarely the right question. The right question is where the dividing line goes in this particular system, and the rest of the book is, in large part, a catalogue of where that line has been drawn in published systems and why.

The four families are a vocabulary, not a taxonomy of mutually exclusive camps. §1.4 named them: classical/analytical, reinforcement-learning, imitation, foundation/VLA. They're useful because they correspond to qualitatively different kinds of supervision, derivation from a model, reward maximization, demonstration matching, and pretrained-plus-fine-tuned demonstration matching, and once you can name the supervision, the design space narrows fast. A deployed system in 2026 typically contains pieces of at least two of these families, often three: a classical IK solver inside a learned policy, a VLA prompting a low-level RL controller, an imitation policy fine-tuned with RL on safety-critical edge cases. Reading the family vocabulary fluently is what lets you take a stack like that apart.

## What you should be able to do now

Four concrete things, in increasing order of how much they matter for the rest of the book.

You should be able to read the abstract of a modern VLA paper, RT-1 (arXiv:2212.06817), OpenVLA (arXiv:2406.09246), π0 (arXiv:2410.24164), and identify, sentence by sentence, which slot of the action-model anatomy each claim is talking about. Abstracts that say "we train on diverse data" are talking about Slot 3. Abstracts that say "we propose a new tokenization" are talking about Slot 2. Abstracts that say "we condition on language and a third-person camera" are talking about Slot 1. It's a small skill, but it changes how fast you triage a reading list, and a working roboticist reads on the order of two new VLA papers a week. The time you save on the ones that don't matter is time spent on the ones that do.

You should be able to place a named system in the right family and the right era. ALVINN (Pomerleau, 1988) is imitation, Era 3. DQN-Atari (Mnih et al., 2015) is RL, Era 4. RT-1 is foundation/VLA, Era 6. ANYmal's locomotion controller is RL by method, Era 4, but contemporary by deployment. This sounds pedantic, yet it's exactly how the rest of the book is organized, since Chapters 4 through 14 are structured by family and era, and being able to place a system on that map quickly tells you which chapter it'll show up in and what its likely failure modes are.

You should be able to sketch, on a napkin, the data and supervision pipeline for a new robot task you've just been handed. Say the customer wants the robot to empty a dishwasher. The four-family sketch runs: classical (PDDL plan plus IK plus motion planning, assuming a clean kitchen model), RL (a reward function, a simulator, sim-to-real transfer), imitation (a teleoperation rig, fifty demonstrations, a Diffusion Policy or ACT model), and foundation/VLA (an OpenVLA checkpoint, those same fifty demonstrations as fine-tuning data, a natural-language prompt). Within the first hour of a project you should be able to draw all four and name which one you'd actually build, and why. The real answer is usually some combination, and that combination is the design of the system.

You should be able to predict the failure mode of an action model from which family it sits in. Classical methods fail when the world model is wrong, an obstacle the planner never knew about. RL methods fail by reward hacking, the policy gets the reward in a way you didn't intend. Imitation methods fail by compounding error, the trajectory diverges as soon as the policy hits a state the demonstrator never visited. Foundation methods fail in all three ways at once, plus a fourth: they confabulate, producing actions that look plausible but are wrong because the pretraining prior over-extrapolated. When a production system starts failing, recognizing which family the failure belongs to is the first step toward fixing it, and that recognition is what the rest of the book teaches.

## Where the chapter has set up the rest of the book

Three forward references are worth naming again here, since they structure what comes next. The three-slot anatomy from §1.2 is the spine of Chapter 2 (the first end-to-end VLA you train) and of every model walkthrough from Chapter 11 onward. The four families from §1.4 are the spine of Chapters 4 through 7 (the lineage) and Chapters 11 through 14 (the foundation family in depth). The compounding-error problem from §1.1 is the subject of §6.3 (DAgger), §6.5 (BC versus IRL versus RL trade-offs), and §17.2 (runtime monitoring as a deployment-time response to the same problem). When one of those threads resurfaces in a later chapter, the cross-reference isn't decorative. It's the second half of an argument this chapter started.

The chapter has not set up two things you might have expected. It hasn't defined "intelligence," "embodiment," or "generalization" with the rigor those words deserve. Each is a load-bearing concept in the field, and each gets its own section later on, §11.5 on when scale starts to pay off, §12.5 on what "emergent" actually means, §18.1 on cross-embodiment generalization, where they can be discussed against specific empirical results rather than treated as philosophical abstractions. If those are the questions you came to this book for, the first ten chapters are the infrastructure that makes those discussions productive instead of vacuous.

The chapter also hasn't given you any code. Chapter 2 changes that. The sample chapter already sitting in the project folder (Sample_Chapter_02_Your_First_VLA.docx) walks an OpenVLA fine-tune end to end, on a single GPU, using a small LIBERO-derived dataset (arXiv:2603.28301), and it's where the vocabulary established here first earns its keep.

§1.x closes the chapter with one hands-on exercise, designed to finish in under an hour with no GPU required, and the full reading list for the chapter.
