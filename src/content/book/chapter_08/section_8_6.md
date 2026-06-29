---
chapter: 8
section: 8.6
title: Summary
target_words: 2000
status: draft
prereqs: §8.1–§8.5; attention and the transformer for control, the Decision Transformer and returns-to-go, the Trajectory Transformer and beam-search planning, what gets tokenized, and the bridge to foundation action models (RT-1/RT-2/OpenVLA) plus the SSM alternative (RoboMamba)
key_refs:
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Janner et al. (2021). Offline reinforcement learning as one big sequence modeling problem (Trajectory Transformer). arXiv:2106.02039.
  - Brohan et al. (2022). RT-1: robotics transformer for real-world control at scale. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2: vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818.
  - Liu et al. (2024). RoboMamba: efficient vision-language-action model for robotic reasoning and manipulation. arXiv:2406.04339.
---

# 8.6  Summary

Chapter 8 was the hinge of the book. Everything before it — classical
planning, MDPs, reinforcement learning, imitation — described ways of
producing actions that were invented for control and stayed in control.
This chapter introduced a machine that was invented for something else
entirely, next-token prediction over text, and showed that pointing it at
robot data produces a controller. That single reframing is what connects
the lineage of Part 2 to the foundation action models of Part 4, and the
chapter's job was to make the reframing concrete enough that the rest of
the book reads as engineering rather than magic. This summary collects the
load-bearing ideas and marks where Parts 3 and 4 keep reaching back for
them.

## The four ideas worth carrying forward

*Control can be cast as sequence modeling, and that reframing is the whole
chapter.* §8.1 and §8.2 made the case that a trajectory — states, actions,
and whatever else you choose to record — is just a sequence of tokens, and
that predicting the next token in that sequence is a way to act. The
Decision Transformer (arXiv:2106.01345) is the cleanest demonstration:
feed it a desired return, the recent history, and the current state, and
it predicts the action a successful trajectory would have taken next. What
makes this more than a notational trick is what it discards. There is no
Bellman backup, no bootstrapping, no value function chasing its own
target — the instability machinery that Chapter 7 spent five sections
taming simply does not appear, because supervised next-token prediction
has none of the moving parts that make temporal-difference learning
fragile. You trade the theoretical scaffolding of RL for the empirical
robustness of supervised learning, and for offline data that is often a
good trade.

*Conditioning is the control knob, and what you condition on defines the
method.* The Decision Transformer conditions on a return-to-go: you tell
it how well to do, and it complies (§8.2). The Trajectory Transformer
(arXiv:2106.02039) instead conditions on nothing in particular and uses
beam search over its own predictions to plan (§8.3). The foundation models
of Part 4 condition on a language instruction. Same architecture, same
next-token objective, three different conditioning signals — and the
conditioning signal is what determines whether you have an offline-RL
method, a planner, or an imitation learner. Internalizing that the
architecture is fixed and the conditioning is the design space is what
lets you see RT-2 and a Decision Transformer as relatives rather than
strangers.

*Tokenization is a modeling decision with consequences, not a
preprocessing chore.* §8.4 was the technical heart of the chapter. Putting
states, actions, returns, and language into one token stream forces a
series of choices — discretize a continuous action into bins, or attach a
continuous head; reuse a vision-language model's existing vocabulary, or
add new symbols — and each choice propagates all the way to control
quality. The action-token overloading trick, where RT-2 (arXiv:2307.15818)
encodes actions as token IDs the vision-language model already owns, is the
move that lets web knowledge survive fine-tuning, and it is only available
because someone treated tokenization as a design problem. The flip side,
which §8.4 was careful to state, is that naive uniform binning wastes
resolution and caps your control frequency, the failure that FAST
(arXiv:2501.09747) and the continuous heads of Chapter 10 exist to fix.

*The recipe is separable from the architecture.* §8.5 made the crossing
into Part 4 explicit and then complicated it productively. Turning a
Decision Transformer into a foundation action model takes exactly four
changes — drop the return for a language instruction, scale the data by
orders of magnitude, initialize from a vision-language model, and
optionally swap the action head — and no new core idea. RoboMamba
(arXiv:2406.04339) then showed that the same recipe runs on a state space
model instead of a transformer, reaching manipulation competence while
updating about 0.1% of its parameters and running several times faster.
The point is not that state space models win; the largest, best-
generalizing action models remain transformers. The point is that you
should read every system in Part 4 by asking which parts are the recipe
and which are the architecture, because conflating the two is how people
end up believing attention is load-bearing when it is often just
incumbent.

## What you should be able to do now

Four concrete capabilities, in roughly the order the rest of the book will
ask for them.

You should be able to *explain how a transformer produces an action, end
to end, without hand-waving the attention step*. Not "it's a neural net
that outputs actions," but the actual path: tokens in, attention as a soft
lookup that lets each token gather information from the others, causal
masking so the model only sees the past, and a prediction head that emits
the next token — which, depending on what you tokenized, is an action.
§8.1 gave you the two-page version specifically so you could read the
architecture sections of Part 4 without stopping to relearn attention each
time.

You should be able to *take a control problem and say what its token
sequence is*. Given a task, you should be able to lay out what gets
tokenized and in what order — which observations, whether a return or a
language instruction does the conditioning, how the action is represented
— and justify each choice against the trade-offs in §8.4. This is the
skill that turns a vague "we'll use a transformer" into an actual design,
and it is the same skill Chapter 11 exercises when it dissects the VLA
recipe component by component.

You should be able to *place the Decision Transformer and the Trajectory
Transformer relative to the offline RL of Chapter 5, and say what each
gives up*. The Decision Transformer trades the Bellman backup for
return-conditioned supervised learning, which buys stability and costs you
the ability to stitch together better-than-demonstrated behavior the way
dynamic programming can (§8.2). The Trajectory Transformer keeps a notion
of planning by searching over its predictions, at the cost of inference
time (§8.3). Naming what each method gained and surrendered, rather than
ranking them, is the analytical habit the chapter was training.

You should be able to *read a foundation action model as a scaled-up
sequence model and locate the four changes*. Shown RT-1, RT-2, OpenVLA, or
RoboMamba, you should be able to point to where the return became a
sentence, where the data scaled, where the vision-language initialization
entered, and what the action head is. That decomposition is the through-
line of Part 4, and §8.5 built it precisely so the later chapters could
assume it.

## Where the chapter has set up the rest of the book

Chapter 8 hands off in three directions, and all three are central to the
remaining parts.

The most immediate is to the generative action heads of Chapter 10. §8.4
ended on the limits of discrete action tokens — wasted resolution, capped
frequency, the awkwardness of binning a smooth motion — and that
unfinished problem is exactly what diffusion and flow-matching heads
solve. When Chapter 10 attaches a diffusion decoder to a transformer
trunk, it is answering the question §8.4 left open: how do you keep the
sequence-model backbone but stop forcing continuous control through a
discrete bottleneck. π0's flow-matching objective in Chapter 13
(arXiv:2410.24164) is the most developed answer.

The deeper handoff is to the entirety of Part 4. §8.5's four-change recipe
is the skeleton on which Chapters 11 through 14 hang their detail.
Chapter 11 walks the recipe from CLIP to RT-1; Chapter 12 takes apart the
data-scaling step with Open X-Embodiment; Chapters 13 and 14 push on the
action head and on splitting one model into two. A reader who holds the
recipe in mind will recognize each of those chapters as elaborating one
change rather than introducing a new paradigm — which is the correct way
to read them, because the paradigm shift already happened, here, in
Chapter 8.

The third handoff is to the latency and efficiency discussion that runs
through Chapter 14 and Chapter 16. §8.1 and §8.5 were blunt about the
quadratic cost of attention and what it means for a robot that must close
a control loop in tens of milliseconds. RoboMamba's linear-cost backbone
was the first answer; Chapter 14's dual-system architectures, which split
a slow deliberative model from a fast reactive one, are the second. The
student who took the cost discussion seriously will not be surprised when
deployment chapters spend as much effort on inference budget as on
accuracy.

## What the chapter has not covered

Two omissions are worth naming. The chapter treated the sequence model
almost entirely as an imitation or offline-RL device and said little about
*online* sequence-model control — using a transformer policy inside a live
RL loop, learning from fresh interaction. That is a deliberate framing
choice, not an oversight: the foundation action models the book builds
toward are imitation learners, and the online-RL fine-tuning of a large
sequence policy belongs with the fine-tuning machinery of Chapter 16,
where the sample-budget arguments of Chapter 7 do the real work. Read this
chapter's silence on online learning as a deferral.

The chapter also stayed deliberately shallow on the internals of the
sequence models themselves. §8.1 gave attention two pages, and §8.5 gave
the selective-state mechanism of Mamba a single sentence, because this is
a book about action models, not about architecture research. A reader who
wants the full derivation of multi-head attention, positional encodings,
or the state space recurrence will need a dedicated transformer reference;
what this chapter supplied is the working understanding required to read
the architecture as a component of a controller. That is the right depth
for the book's purpose, but it is worth being honest that it is a floor,
not a ceiling.

Chapter 8's contribution to the book's overall argument is to perform the
single conceptual move the whole second half depends on: recasting action
generation as next-token prediction, and showing that the move is small in
mechanism and enormous in consequence. The three findings to carry forward
are that control-as-sequence-modeling trades RL's theoretical scaffolding
for supervised robustness, that conditioning — not architecture — is what
distinguishes an offline-RL method from a planner from an imitation
learner, and that the foundation-model recipe is separable from the
transformer that usually carries it. Part 3 now picks up the generative
and world-model machinery that, layered onto this sequence-modeling
foundation, produces the smooth, scalable action models of Part 4.

§8.x closes the chapter with a hands-on exercise — training a small
Decision Transformer on an offline dataset and probing how its behavior
changes as you sweep the target return — and the full reading list for the
chapter.
