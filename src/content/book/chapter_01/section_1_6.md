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

Chapter 1 was, by design, the chapter with the least code in the book. It set
out the vocabulary, the history, and the shape of the field, so that the
chapters that do contain code (and lots of it) can move quickly. This summary
collects the load-bearing ideas in one place. If you are reading the chapters
in order, treat the next twelve hundred words as a checklist for what you
should be able to do before you turn the page; if you skipped ahead and came
back, treat it as a single-section refresher.

## The four ideas worth carrying forward

*Action is structurally different from perception and language.* The output
space of a robot policy — a vector of continuous joint commands or pose deltas,
emitted every twenty milliseconds — is not the output space of a classifier or
a token predictor. §1.1 named four properties that make it different: the
output space is high-dimensional, continuous, and constrained; the policy
operates in a closed loop where each action changes the next observation; the
training distribution and the deployment distribution diverge by construction
(compounding error); and the cost of being wrong is paid in physical hardware
rather than in benchmark points. Every architectural choice in the rest of the
book — action chunking, diffusion heads, flow matching, runtime safety
monitors — is a response to one of these four properties. When you read a
paper and ask "why this design and not the obvious one," the answer is almost
always one of the four.

*An action model is a function with three slots.* §1.2 introduced the slot
anatomy: Slot 1 is the input (observations and instructions in whatever
modality), Slot 2 is the output (the action representation — symbolic, joint
target, end-effector delta, torque, or a tokenized version of any of these),
Slot 3 is the training signal (the loss or reward that produced the function).
The anatomy is deliberately model-agnostic. It applies to a STRIPS planner
(structured predicates in, symbolic plan out, no training), to a SAC policy
(observations in, action distribution out, reward as signal), and to OpenVLA
(image plus text in, action tokens out, imitation loss on cross-embodiment
data). Every paper you read for the rest of the book can be re-read as a set
of choices for these three slots. That re-reading is more useful than it
sounds. It is how you notice that two systems with different names are
making the same architectural choice, and that two systems with similar names
are not.

*The history is one of bottlenecks moving, not of methods replacing each
other.* §1.3 walked through six eras — symbolic planning, classical control,
deep RL, end-to-end imitation, the foundation moment, and the VLA scale-up
— and the point of the walk was that each era's methods survived the
transition to the next. STRIPS-descendant planners still run inside warehouse
robots that ship a VLA on top. Inverse-kinematics solvers still convert pose
deltas to joint commands at the bottom of the OpenVLA stack. RL still
produces the locomotion policies that VLAs ride on. The eras compound; they
do not replace. The corollary for an engineer is that "should I use a
classical method or a learned method" is rarely the right question. The
right question is "where does the dividing line go in this particular
system," and the rest of the book is, in large part, a catalogue of where
the line has been drawn in published systems and why.

*The four families are a vocabulary, not a taxonomy of mutually exclusive
camps.* §1.4 named them: classical/analytical, reinforcement-learning,
imitation, foundation/VLA. The families are useful because they correspond
to qualitatively different supervisions — derivation from a model, reward
maximization, demonstration matching, and pretrained-plus-fine-tuned
demonstration matching, respectively — and because once you can name the
supervision, the design space narrows fast. A deployed system in 2026
typically contains pieces of at least two and often three of the families:
a classical IK solver inside a learned policy, a VLA prompting a low-level
RL controller, an imitation policy fine-tuned with RL on safety-critical
edge cases. Reading the family vocabulary fluently is what lets you take
apart such a stack.

## What you should be able to do now

Four concrete things, in increasing order of how much they matter for the
rest of the book.

You should be able to *read the abstract of a modern VLA paper* — RT-1
(arXiv:2212.06817), OpenVLA (arXiv:2406.09246), π0 (arXiv:2410.24164) — and
identify, at a sentence-by-sentence level, which slot of the action-model
anatomy each claim is talking about. Abstracts that say "we train on
diverse data" are talking about Slot 3. Abstracts that say "we propose a
new tokenization" are talking about Slot 2. Abstracts that say "we condition
on language and a third-person camera" are talking about Slot 1. This is a
small skill but it changes how fast you can triage a reading list. A modern
roboticist reads on the order of two new VLA papers a week; the time saved
on the ones that turn out not to matter is the time you spend on the ones
that do.

You should be able to *place a named system in the right family and the
right era*. ALVINN (Pomerleau, 1988) is imitation, Era 2. DQN-Atari (Mnih
et al., 2015) is RL, Era 3. RT-1 is foundation/VLA, Era 5. ANYmal's
locomotion controller is RL, Era 3 by method but contemporary by
deployment. The exercise sounds pedantic but it is how the rest of the
book is structured — Chapters 4 through 14 are organized by family and era
— and being able to place a system on that map quickly is what lets you
predict which chapter it will be discussed in and what its likely failure
modes are.

You should be able to *sketch, on a napkin, the data and supervision pipeline
for a new robot task you have just been handed*. The customer says "the
robot should empty a dishwasher." The four-family napkin sketch is the
classical solution (PDDL plan plus IK plus motion planning, assuming a clean
kitchen model), the RL solution (a reward function, a simulator, sim-to-real
transfer), the imitation solution (a teleoperation rig, fifty demonstrations,
a Diffusion Policy or ACT model), and the foundation/VLA solution (an
OpenVLA checkpoint, the same fifty demonstrations as fine-tuning data, a
natural-language prompt). Within the first hour of a project, you should be
able to draw all four and to name which one you would actually build, and
why. The answer is usually some combination; that combination is the design
of the system.

You should be able to *predict the failure mode of an action model from
which family it sits in*. Classical methods fail when the world model is
wrong (an obstacle the planner did not know about). RL methods fail by
reward hacking (the policy gets the reward in a way you did not intend).
Imitation methods fail by compounding error (the trajectory diverges as
soon as the policy sees a state the demonstrator never visited). Foundation
methods fail in all three ways at once, plus a fourth — they confabulate,
producing actions that look plausible but are wrong because the pretraining
prior over-extrapolated. When a system in production starts failing,
recognizing the family of the failure is the first step in fixing it. That
recognition is what the rest of the book teaches.

## Where the chapter has set up the rest of the book

Three explicit forward references are worth re-naming here, because they
structure what comes next. The three-slot anatomy from §1.2 is the spine of
Chapter 2 (the first end-to-end VLA you train) and of every model
walkthrough from Chapter 11 onward. The four families from §1.4 are the
spine of Chapters 4 through 7 (the lineage) and Chapters 11 through 14 (the
foundation family in depth). The compounding-error problem from §1.1 is the
subject of §6.3 (DAgger), §6.5 (BC versus IRL versus RL trade-offs), and
§17.2 (runtime monitoring as a deployment-time response to the same
problem). When you see one of those threads in a later chapter, the cross-
reference is not decorative; it is the second half of an argument the
present chapter started.

The chapter has *not* set up two things you might have expected. It has not
defined "intelligence," "embodiment," or "generalization" with the rigor
those words deserve. Each is a load-bearing concept in the field, and each
is the subject of a section later in the book — §11.5 on when scale starts
to pay off, §12.5 on what "emergent" means, §18.1 on cross-embodiment
generalization — where they can be discussed against specific empirical
results rather than as philosophical abstractions. If those are the
questions you came to the book for, the first ten chapters are the
infrastructure that makes those discussions productive rather than vacuous.

The chapter has also not given you any code. Chapter 2 is the first place
that changes. The sample chapter that already exists in the project folder
(Sample_Chapter_02_Your_First_VLA.docx) walks an OpenVLA fine-tune end to
end, on a single GPU, using a small LIBERO-derived dataset
(arXiv:2603.28301), and is the chapter where the vocabulary established
here first earns its keep.

§1.x closes the chapter with one hands-on exercise — designed to be
completable in under an hour and with no GPU required — and the full
reading list for the chapter.
