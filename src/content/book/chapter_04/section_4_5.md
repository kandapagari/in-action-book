---
chapter: 4
section: 4.5
title: Summary
target_words: 2000
status: draft
prereqs: §4.1–§4.4; the three-layer cake (symbolic, geometric, dynamic) and the argument that learned methods sit on top of, beside, or inside classical layers rather than replacing them
key_refs:
  - Fikes & Nilsson (1971). STRIPS — A New Approach to the Application of Theorem Proving to Problem Solving. Artificial Intelligence 2(3–4).
  - McDermott et al. (1998). PDDL — The Planning Domain Definition Language. AIPS-98.
  - Helmert (2006). The Fast Downward Planning System. JAIR 26.
  - Garrett, Lozano-Pérez, Kaelbling (2020). PDDLStream. ICAPS.
  - Khatib (1987). A Unified Approach for Motion and Force Control of Robot Manipulators — The Operational Space Formulation. IEEE J. Robotics and Automation 3(1).
  - Hogan (1985). Impedance Control — An Approach to Manipulation. ASME J. Dynamic Systems.
  - Ahn et al. (2022). Do As I Can, Not As I Say (SayCan). arXiv:2204.01691.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
---

# 4.5  Summary

Chapter 4 was the chapter that traced the lineage. The four sections walked
down the classical stack — symbolic plans at the top (§4.1), geometric
trajectories in the middle (§4.2), inverse-dynamics torques at the bottom
(§4.3), and then a frank accounting of which of those layers is still doing
real work inside a deployed 2026 robot (§4.4). The chapter sits in the book
at a particular hinge: it is the first chapter of Part 2 and the chapter
where the four-family taxonomy from §1.4 starts to be unpacked rather than
asserted. This summary collects the load-bearing ideas in one place and
flags which ones Chapters 5 through 14 will lean on hardest.

## The four ideas worth carrying forward

*A classical action model is a function from a structured state to a
structured action.* §4.1 made the case for symbolic planning by reducing
STRIPS to three pieces — predicates, action schemas, planning problems —
and showing that every PDDL domain sits on top of those three. The piece
that newcomers miss is that the *interface* of a symbolic action model
survives even when the planner inside is replaced. SayCan
(arXiv:2204.01691) and Code as Policies (arXiv:2209.07753) both swap a
heuristic-search planner for a language model, but the output is still
"a named action with arguments, drawn from a finite alphabet". When
Chapter 14 dissects dual-system architectures, the top system is
emitting that same kind of object — only the implementation has moved
from Fast Downward to a 70-billion-parameter transformer. The shape of
the interface is the part that matters; the choice of engine behind it
is, increasingly, an engineering preference rather than a research
commitment.

*Geometry is the layer that newcomers most often forget exists.* §4.2
spent its pages on inverse kinematics, RRT-Connect, and the difference
between a sampling-based and an optimization-based motion planner,
because the failure mode of a learned-policy paper read in isolation is
to imagine that the model itself is doing the geometry. It is not. The
seven numbers OpenVLA (arXiv:2406.09246) emits at 5 Hz are a target pose
delta; the IK solver, the joint-limit clipper, the workspace bounding
box, and the short-horizon smoother that sit underneath the model are
the geometric layer, and they are doing the same kind of work that
TRAC-IK and OMPL were doing in 2015. The chapter's most useful
diagnostic — if a VLA paper does not name its IK solver, assume
TRAC-IK; if it does not name its motion planner, assume the
manufacturer's default — is what keeps you from being surprised when a
fine-tune that "should have worked" fails because the target pose was
1.5 cm outside the dexterous workspace.

*Inverse dynamics is the layer that has barely moved in forty years and
probably will not move much in five.* §4.3 worked through the
manipulator equation $M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau$
and the computed-torque, PD-plus-gravity, and impedance-control
patterns that sit on top of it. The reason this layer is so durable is
that the structure of the problem is favorable to derivation: the
dynamics model is known up to parameters, the parameters can be
identified, and the resulting controller has stability properties that
matter when something heavy is moving fast. Legged locomotion is the
visible exception — RL-trained controllers on ANYmal, Cassie, and Atlas
do emit torques and have internalized the inertial terms during
training — but for manipulators in 2026, the bottom of every commercial
stack the authors know of is still a classical controller. Residual
learning (Hogan 1985's impedance plus a learned correction) is the
research frontier *on top of* classical control, not its replacement.

*The right argument about classical methods is layer by layer, not yes
or no.* §4.4 was the chapter's punch line and it is worth repeating in
the summary because it is the single most common rhetorical trap in
the field. "Is classical robotics dead" is a malformed question. The
question that admits an answer is "in this layer of this stack, what
fraction of the work is being done by a classical method and what
fraction by a learned one, and which way is the line moving". The §4.4
table — symbolic mostly classical with LLM-flavored variants in the
top slot; geometric overwhelmingly classical with thinning slices for
short-horizon contact-rich motion; dynamic essentially fully classical
on manipulators, learned on legs — is the form of the answer the rest
of the book will keep returning to. Whenever a paper claims to have
"replaced the classical pipeline", the right reflex is to ask which
layer it has replaced and whether the other two are still there. They
almost always are.

## What you should be able to do now

Four concrete capabilities, in roughly increasing order of how much
the rest of the book will rely on them.

You should be able to *write a small PDDL domain and run a planner on
it*. The chapter's hands-on directory has a five-block tabletop with
`pick-up`, `put-down`, `stack`, and `unstack` schemas; you should be
able to read that domain without reaching for documentation and
extend it with a fifth schema — `push`, say, or `swap` — without
breaking the rest. The skill is small but load-bearing: every
language-model planner in Part 4 (SayCan, Code as Policies, the
top half of GR00T N1 arXiv:2503.14734) is producing something that
can be read as a PDDL plan with the predicates implicit, and being
able to write the explicit version makes the implicit one
intelligible.

You should be able to *derive a 6-DOF inverse-kinematics solution
analytically for a simple geometry, or pick a numerical IK solver
off the shelf for a harder one, and know which case you are in*. §4.2
named the deciding factor: closed-form IK exists when the wrist is
spherical and the geometry is simple enough that the Pieper criterion
applies, and you should reach for it; otherwise you call Damped Least
Squares or TRAC-IK and accept that the solver may return nothing on
some queries. The skill matters because Chapter 16's fine-tuning
recipes assume you can stand up an IK layer underneath the VLA without
ceremony, and Chapter 17's deployment chapter assumes you can debug
"the model outputs a perfectly reasonable pose, but the arm does not
move" by looking at the IK return code rather than at the network.

You should be able to *read a torque-control paper and identify which
terms of the manipulator equation the controller is computing
explicitly, which it is learning, and which it is letting the
manufacturer's onboard firmware handle*. The Franka Panda case from
§4.4 is the canonical illustration: a research user who sends a
"torque-level" command is, in fact, setting the high-level term while
the manufacturer's gravity compensator, friction model, and joint
limits operate underneath. A paper that reports "we trained an RL
policy to output joint torques on a Franka" is not training a
controller that produces motor current; it is training a controller
that produces *the top of the manipulator equation*, with everything
below it already classical. Naming what is learned vs. what is
provided is the difference between reading the paper at face value
and reading the paper correctly.

You should be able to *draw the three-layer cake for any deployed
robot in the literature and label, for each layer, classical vs.
learned*. Pick any system — OpenVLA on a WidowX, π0 on a bimanual
manipulator, GR00T N1 on a humanoid, Aloha on a teleop rig — and
the cake-labeling exercise is what tells you what the system actually
is. OpenVLA: classical top (task is implicit in the language prompt),
learned middle-top (network emits poses), classical middle-bottom
(IK and trajectory generation), classical bottom (joint-level control).
π0: similar, but with a flow-matching middle-top that emits action
*chunks* rather than single poses. GR00T N1: LLM-flavored top, learned
middle, classical bottom. The exercise is mechanical once you can do
it for one system, and it is the skill the rest of Part 4 quietly
assumes you have.

## Where the chapter has set up the rest of the book

Chapter 4 hands off to the next three chapters of Part 2 along a
specific path. Chapter 5 picks up the *reward* slot from §1.2 — the
training-signal slot that classical action models leave empty — and
develops MDPs and reinforcement learning as the family that fills it.
Chapter 6 does the same for demonstrations and behavior cloning, the
family that fills the same slot with imitation data instead of
reward. Chapter 7 is the deep-RL chapter, which is where the
torque-level controllers from §4.3 meet the function-approximation
ideas from §3.1. By the time you finish Part 2, you will have seen the
three other ways of producing an action — reward-driven, imitation-
driven, and the deep-RL hybrid — and the classical methods from this
chapter will have moved from being "the way robots are programmed" to
being "the layer underneath the way modern robots are programmed".

Two specific forward references from this chapter are worth naming
because they will recur. The PDDL interface from §4.1 — named skills
with arguments, drawn from a finite alphabet — is the interface
between Chapter 14's high-level system and its low-level system in a
dual-system architecture. The residual-learning pattern from §4.3
and §4.4 — classical controller plus a small learned correction — is
the architectural pattern Chapter 16 reaches for first when the
question is "how do I add a small amount of learning to an existing
classical stack without breaking it". Both ideas appear later under
different names, and the §4 versions are the small, readable
prototypes.

The chapter has *not* set up two things you might have expected. It
has not covered task-and-motion planning in depth; PDDLStream
(Garrett, Lozano-Pérez, and Kaelbling, 2020) appeared in §4.1 as a
named reference rather than a worked example, and the kind of joint
symbolic-geometric search that TAMP performs is its own subfield with
its own dedicated literature, which we point to rather than reproduce.
It has also not covered the *operational-space formulation* (Khatib,
1987) in detail; it appeared in §4.3 as one of the named patterns,
but the full treatment of force control belongs in a manipulation
textbook, not this one. Both omissions are deliberate. The aim of
Chapter 4 was to give you the classical *vocabulary* the rest of the
book reuses, not to make you a planning-systems engineer or a force-
control engineer.

Part 2 starts here and the four-family taxonomy from §1.4 is now
half-explained: the classical family in Chapter 4, the RL family next,
the imitation family after that, and the foundation/VLA family in
Part 4. Each subsequent chapter adds a single family to the working
vocabulary, and Chapter 4's contribution is the family that defines
the *interfaces* — the named skills, the joint-space trajectories, the
joint torques — that every later family ultimately has to produce
something compatible with. Whatever a VLA outputs in 2030, an IK
solver and an inverse-dynamics controller are still likely to be the
last two functions called before a motor sees current.

§4.x closes Chapter 4 with one hands-on exercise — running Fast
Downward on a five-block PDDL domain, then wiring its symbolic plan to
a TRAC-IK-driven pick-and-place executor in PyBullet — and the full
reading list for the chapter.
