---
chapter: 4
section: 4.1
title: "Symbolic actions: STRIPS, PDDL, and action schemas"
target_words: 2000
status: draft
prereqs: §1.2 (the three-slot anatomy), §1.3 (STRIPS situated in Era 1); high-school logic; the willingness to read a Lisp-flavored syntax for half an hour
key_refs:
  - Fikes & Nilsson (1971). STRIPS — A New Approach to the Application of Theorem Proving to Problem Solving. Artificial Intelligence 2(3–4).
  - McDermott et al. (1998). PDDL — The Planning Domain Definition Language. AIPS-98 Planning Competition.
  - Helmert (2006). The Fast Downward Planning System. JAIR 26.
  - Garrett, Lozano-Pérez, Kaelbling (2020). PDDLStream — Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning. ICAPS 2020.
  - Ahn et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances. (SayCan.) arXiv:2204.01691.
---

# 4.1  Symbolic actions: STRIPS, PDDL, and action schemas

The action models you have met so far emit numbers. OpenVLA emits a 7-vector
every 50 ms; the SmallPolicy from §3.3 emits a 7-vector every step; Diffusion
Policy emits a sequence of 7-vectors at once. Symbolic action models emit
something else entirely: a *name* and a *list of arguments*, drawn from a
finite alphabet that the engineer wrote down ahead of time. `pick(block_a)`
is a symbolic action. So is `move(robot, kitchen, living_room)`, or
`pour(cup_1, bowl_2)`. There is no real-number anywhere in the output. The
output is closer to a programming-language statement than to a vector.

This is the family the field started with — and the family every other
chapter of this book will, in some way, define itself against. Part of the
reason for opening Chapter 4 with it is historical: STRIPS predates every
other family by at least a decade and the rest of the field's vocabulary
inherits its terminology. The more important reason is that symbolic action
models still run, today, inside the *top* layer of many deployed robots. The
hierarchical decomposition that an LLM produces when SayCan
(arXiv:2204.01691) asks "what should the robot do next" is, structurally, a
STRIPS plan. The task scheduler inside a warehouse robot that ships a VLA
for grasping is, structurally, a STRIPS planner. Understanding what a
symbolic action is — and what it is not — is the prerequisite for reading
those modern systems as systems rather than as black boxes.

## The STRIPS world model

STRIPS, the Stanford Research Institute Problem Solver (Fikes and Nilsson,
1971), was the planner that drove Shakey, the first mobile robot intended to
navigate a physical lab. The representation has three pieces, and once you
have these three pieces you can read essentially every symbolic planner that
has been written since.

The first piece is a set of *predicates* — Boolean facts about the world.
`(on block_a block_b)` says block A is on top of block B. `(holding block_a)`
says the robot's gripper is currently holding block A. `(clear block_b)` says
nothing is on top of block B. A *state* is a set of predicates that are
currently true; everything not in the set is assumed false (the closed-world
assumption). For a five-block tabletop, the state might be the four-element
set `{(on a b), (on b table), (clear a), (handempty)}`. That is the entire
world model the planner has.

The second piece is a set of *actions* — operators that take a state to a
new state. Each action has a *name* with typed arguments, a list of
*preconditions* that must hold for the action to apply, and a list of
*effects* that say which predicates become true and which become false after
the action fires. The canonical pick-up action is:

```
action: pick-up(?x - block)
  precondition:  (clear ?x) and (on ?x table) and (handempty)
  effect:        not (on ?x table) and not (clear ?x) and
                 not (handempty) and (holding ?x)
```

The `?x` is a parameter — an action *schema* with a free variable. To
actually execute, the planner *grounds* the schema by binding `?x` to a
specific block, producing concrete instances `pick-up(a)`, `pick-up(b)`, and
so on. Grounding turns a small set of schemas into a much larger set of
actually-executable actions; for ten blocks and four schemas you can easily
end up with a few hundred ground actions, and for a realistic warehouse
domain with thousands of objects, grounding can produce millions of ground
actions and is itself a non-trivial step.

The third piece is the *planning problem* — an initial state, a set of
ground actions derived from the schemas, and a goal expressed as a partial
predicate set. A solution is a sequence of ground actions that, applied in
order, transforms the initial state into one that satisfies the goal. The
planner's job is to search for such a sequence, typically by heuristic
search over the state space (depth-first, breadth-first, A* with a relaxed-
plan heuristic, and various more recent landmark-based heuristics).

That is it. Predicates, action schemas, planning problem. Every PDDL domain
you will ever read sits on top of those three abstractions.

## PDDL: the standard syntax

STRIPS as a representation outlasted STRIPS as a piece of software. By the
mid-1990s every research group had its own incompatible syntax, and the
1998 AIPS planning competition demanded a common one. The result was the
Planning Domain Definition Language (McDermott et al., 1998), known
universally as PDDL. PDDL has gone through several versions; PDDL 2.1
(2003) added numeric fluents and durative actions, PDDL 3 (2005) added
preferences and trajectory constraints, and the various PDDL+ dialects
extend the language for hybrid and probabilistic domains. For Chapter 4 we
will mostly use PDDL 1, the plain STRIPS-flavored subset, because it is
small enough to teach in a section and large enough to express the
exercise at the end of the chapter.

A PDDL specification has two files: the *domain* file, which defines the
predicates and action schemas (and is reusable across many problems), and
the *problem* file, which defines a specific initial state and goal. Here
is a complete domain for a minimal block-stacking world:

```
(define (domain blocks)
  (:requirements :strips :typing)
  (:types block)
  (:predicates
    (on ?x - block ?y - block)
    (on-table ?x - block)
    (clear ?x - block)
    (holding ?x - block)
    (handempty))
  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (on-table ?x) (handempty))
    :effect (and (holding ?x) (not (on-table ?x))
                 (not (clear ?x)) (not (handempty))))
  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (on-table ?x) (clear ?x) (handempty)
                 (not (holding ?x)))))
```

And a problem instance that asks the planner to swap two blocks:

```
(define (problem swap-ab)
  (:domain blocks)
  (:objects a b - block)
  (:init (on a b) (on-table b) (clear a) (handempty))
  (:goal (and (on b a) (on-table a))))
```

Saved as `blocks.pddl` and `swap.pddl`, those two files are enough input
for any STRIPS-compatible planner. The exercise at the end of the chapter
asks you to run Fast Downward (Helmert, 2006) on a slightly larger version
of this domain; for the present section, the point is to read the syntax
without flinching.

Two pieces of PDDL syntax are worth flagging because they trip up
beginners. The `:typing` requirement is what makes `?x - block` mean
"`?x` is of type block"; without `:typing`, the planner has no notion of
types and treats every parameter as an untyped object. The negation form
`(not (on-table ?x))` inside an `:effect` block means "remove the predicate
`(on-table ?x)` from the state"; PDDL uses the same `not` token for two
distinct purposes — a precondition test and an effect deletion — and which
one applies is positional.

## Search, heuristics, and why classical planning is fast

Given a domain and a problem, what does a planner *do*? At the highest
level: it searches over the implicit graph whose nodes are states and whose
edges are ground actions, looking for a path from the initial state to a
state that satisfies the goal. The branching factor — number of ground
actions applicable in a state — is typically dozens to thousands; the depth
— number of actions in an optimal plan — is typically a few to a few
hundred. A naive breadth-first search is hopeless even for medium domains.

Modern planners get speed from heuristics. The single most influential idea
is the *relaxed-plan heuristic*: pretend, for the sake of estimating the
distance to the goal, that actions have only positive effects (negative
effects are ignored). Under that relaxation, the planning problem becomes
polynomial — a kind of reachability calculation — and its solution length
is an admissible-ish estimate of the true distance. The Fast-Forward
planner (Hoffmann and Nebel, 2001) made the relaxed-plan heuristic
practical, and Fast Downward (Helmert, 2006), the current de facto baseline,
uses landmark-based and merge-and-shrink heuristics that build on the same
idea. The headline number is that Fast Downward routinely solves problems
with millions of ground actions in seconds. That kind of speed is what
keeps symbolic planners load-bearing in modern stacks: even when the *low-
level* actions are emitted by a learned policy, a STRIPS-style task layer
can sit on top and decide which subgoal to issue next without slowing the
loop noticeably.

A worked example, briefly: for the swap-ab problem above, an optimal plan
is four actions long — `unstack(a, b); put-down(a); pick-up(b);
stack(b, a)`, assuming we add `stack` and `unstack` schemas alongside
pick-up and put-down. (We omitted those two schemas for space; the full
domain is in the chapter's hands-on directory.) Fast Downward solves this
problem in milliseconds. The search space for a four-block instance is
still tractable for a human; by ten blocks it is not, and by a hundred it
is firmly in planner territory.

## What an action schema is, and what it is not

An action schema is *not* a controller. It says nothing about how the
robot's joints move to accomplish `pick-up(a)`; it asserts only that, *if
the preconditions hold*, the action can be issued, and *after* it issues,
the effects will hold. The actual motion of the arm — the seven-vector at
50 ms that §3.1 spent so much time on — is somebody else's problem.

That somebody else, in a classical stack, is the geometric layer treated in
§4.2 (inverse kinematics, motion planning) and the dynamic layer treated in
§4.3 (computed-torque control). In a modern stack, the somebody else is a
learned policy: a VLA fine-tuned to execute pick primitives, an RL
controller for legged locomotion, a Diffusion Policy for fine manipulation.
The interface between the symbolic layer and the executor layer is the
ground-action name with its bindings — `pick-up(block_a)` — plus whatever
geometric context the executor needs (a target pose, a grasp annotation, a
language string). This *vertical decomposition* is what enables modern
task-and-motion planning (Garrett et al., 2020, ICAPS) and language-grounded
planning (SayCan, arXiv:2204.01691) to combine symbolic reasoning at the
top with learned actuation at the bottom.

Three failure modes of pure-symbolic action models are worth naming because
they directly motivate the rest of the book. First, the *frame problem*:
writing down every effect of every action is brittle, and forgetting a
single effect produces plans that look correct on paper and fail in
reality. Second, the *grounding problem*: the planner assumes its
predicates correspond to real-world facts, but `(on a b)` is not something
a robot can directly observe — it has to be perceived. Third, the *coverage
problem*: any action the engineer did not write a schema for is invisible
to the planner. Learned action models address these one by one. A
Diffusion Policy does not need a frame-perfect effect model because it
operates on observations. A VLA does not need predicate grounding because
it consumes raw pixels. A foundation-scale imitation model does not need
hand-engineered schemas because it acquires its action repertoire from
demonstration.

What it gives up, in exchange, is the *guarantees* a symbolic planner
provides: that the plan is provably correct under its world model, that
sub-optimality bounds are explicit, and that adding a new goal is a one-
line change to a problem file rather than a fine-tune. The trade is
fundamental; we will revisit it explicitly in §4.4 and again in §17.5 when
discussing what we cannot certify.

The next section moves from the symbolic layer down to the geometric one
and asks how a ground action like `pick-up(block_a)` actually becomes a
collision-free trajectory of joint angles.
