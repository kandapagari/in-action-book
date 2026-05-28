---
chapter: 2
section: 2.6
title: Summary
target_words: 1900
status: draft
prereqs: §2.1–§2.5; a working OpenVLA-on-LIBERO loop, the three named silent failures from §2.4, and the chapter map from §2.5
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 2.6  Summary

Chapter 2 was the running-code chapter. You stood up an OpenVLA-7B
checkpoint (Kim et al., 2024, arXiv:2406.09246), pointed it at a LIBERO
tabletop scene (Liu et al., 2023, arXiv:2306.03310), watched it succeed
on the task it was trained on, and watched it fail in three specific ways
when you removed something it depended on. That is the artifact the rest
of the book will keep pointing back to. This summary collects the
load-bearing ideas in one place. It is shorter than §1.6 because the
chapter was shorter; it is structured the same way so you can use it as
a checklist before turning the page.

## The four ideas worth carrying forward

*A VLA is six boxes, not one black box.* §2.1 named the four commitments
(how images become tokens, how the head emits actions, what dataset
taught the model physics, what evaluation discipline is enforced). §2.3
exposed the loop those commitments live inside: a visual encoder, a
language tokenizer, a transformer trunk, an action head that emits
discrete tokens, a detokenizer that maps tokens to a 7-vector, and a
simulator that closes the loop with a fresh image. Six boxes, each one a
component you can swap, debug, or replace independently. The temptation
when reading a new VLA paper is to treat the model as an undifferentiated
blob of parameters; the discipline this chapter pushed on you is to read
the paper as a set of choices for each of those six boxes. When you read
that π0 (Black et al., 2024, arXiv:2410.24164) "uses flow matching for
action," you should now be able to locate that sentence at box four — the
action head — and ask what the same sentence implies for boxes five (the
detokenizer becomes a flow-matching sampler) and three (the training
loss changes from cross-entropy to a flow-matching objective). That
re-reading is the whole point of having stood up the loop.

*Action tokenization is the architectural trick that made the rest
possible.* §2.1 and §2.3 returned to this idea three times because it
keeps mattering. The seven integers OpenVLA emits per step are not text
tokens reused at inference; they are integer IDs in the bottom 256 entries
of the Llama-2 vocabulary that were *repurposed during training* to mean
"discretized action bin 0 through 255." Cross-entropy loss on those
tokens is the entirety of the training objective. The architectural
simplicity that follows — same transformer, same optimizer, same training
infrastructure as a language model — is the single biggest reason a
7-billion-parameter VLA was a tractable engineering project at all. The
cost is the discreteness, and the cost shows up in §2.4 as the third
silent failure: a wrong `unnorm_key` produces seven plausible integers
that decode to a wildly off-scale 7-vector. Chapters 10 and 13 take the
discrete-versus-continuous trade-off apart in detail; what you should
carry forward from Chapter 2 is the fact that the trade-off exists, that
OpenVLA sits on one side of it, and that the model running on your
machine right now is a discrete-action VLA whose smoothness ceiling is
set by the bin width.

*Running and working are not the same loop.* §2.4 is the chapter's
clearest payoff. A loop that completes four hundred steps without
raising an exception is not a loop that solves the task; the gap between
the two is where the silent failures live. The three you saw — flipped
image (because LIBERO returns the agent-view buffer upside down), wrong
prompt template (because OpenVLA's training-time format is not the
obvious one), wrong `unnorm_key` (because per-embodiment normalization
statistics are how the detokenizer recovers physical units) — are not a
complete list. They are a *kind* of bug. Each one has the same shape: a
piece of metadata that was implicit in the training data and that you
have to make explicit at inference time, with no exception raised if you
get it wrong. The general lesson, which §17.4 will return to, is that
the most expensive bugs in a deployed VLA are the ones that do not
crash. Chapter 2 gave you three reproducible examples on a laptop;
Chapter 17 generalizes the diagnosis discipline to hardware. The
intermediate chapters between them are largely about the *kinds* of
metadata — embodiment, camera pose, action normalization, prompt
template — that have to travel with a model from training to inference.

*Evaluation must outlast the cherry-picked clip.* §2.4 closed with the
discipline that a single rollout tells you almost nothing. Twenty
rollouts on twenty logged seeds, on three initial conditions, with one
camera variation, is the minimum unit of evidence that lets you compare
two versions of the same model honestly. The version of that discipline
you ran in §2.4 was small. The version Chapter 15 develops scales to
real-robot evaluation, where variance is higher and the cost of each
trial is in minutes rather than seconds. The skill you should have
internalized in Chapter 2 is logging the seed and the initial condition
*before* you watch the rollout, not after. The temptation to publish the
good clip is a permanent feature of the field; the only protection
against fooling yourself is the rigor of the logs you keep.

## What you should be able to do now

Four concrete things, in increasing order of how much they matter for
the rest of the book.

You should be able to *re-run a single LIBERO rollout from a logged seed
and reproduce the trajectory to the step.* This is the smallest unit of
the discipline §2.4 introduced. If you cannot reproduce your own
rollout, you cannot debug it; if you cannot debug it, you cannot improve
the policy. The recipe is the one in §2.3: pin the simulator seed, pin
the initial condition file, pin the model dtype, and log the action
vector at every step. When you flip to Chapter 16 to fine-tune on your
own data, the same reproducibility hygiene becomes the difference
between a fine-tune that converges and a fine-tune you cannot diagnose.

You should be able to *read a new VLA inference script and place each
line in one of the six boxes from §2.3.* Concretely: when you encounter
a fresh policy on Hugging Face — say, an OpenVLA fine-tune for a new
embodiment, or a community port of Octo (arXiv:2405.12213) — you should
be able to skim its `predict_action` (or equivalent) call, identify
which line tokenizes the image, which line tokenizes the prompt, which
line runs the transformer trunk, which line samples the action, and
which line maps the sample back to physical units. The six-box model is
the reading skill the rest of Part 4 assumes you have. Chapter 11
formalizes it; Chapter 2 gave you the practice.

You should be able to *predict which of the three silent failures from
§2.4 a particular bug is, given only its symptom*. Robot moves in slow
random scribbles → wrong prompt template (the model is not in
action-decoding mode). Robot moves decisively in the wrong direction →
flipped image (the policy is acting on a world that does not exist).
Robot moves at the wrong scale (way too small, way too large) → wrong
`unnorm_key` (the detokenizer is applying the wrong embodiment's
statistics). None of these symptoms throws an exception. The mapping
from symptom to cause is the muscle that distinguishes someone who has
deployed a VLA before from someone who has not, and Chapter 2 gave you
the three reps you need to start.

You should be able to *use the §2.5 map to plan your own reading path.*
If you are short on time, the minimum path is Chapter 3 (only the parts
you do not already know), Chapter 6 (imitation learning), Chapter 8
(transformers), Chapter 11 (CLIP → BC-Z → RT-1, arXiv:2212.06817), Chapter
12 (the scaling moment), and Chapter 16 (fine-tuning). If you are
project-driven and want the canonical reference for a specific model,
Appendix F (the VLA model zoo) and Appendix E.2 (the chapter-by-chapter
reading list) cross-reference everything. The book is designed to be
read non-linearly *after* Chapter 2, and the map is the artifact that
lets you do that without losing your place.

## Where the chapter has set up the rest of the book

Three forward references are worth re-naming because they structure what
comes next. The six-box decomposition from §2.3 is the spine of every
model walkthrough in Part 4 (Chapters 11 through 14): each chapter
re-enters the loop and replaces one or two boxes. The three silent
failures from §2.4 are revisited in §17.2 (runtime monitoring as the
deployment-time response to the same class of bug) and in §15.4
(real-robot evaluation, where the variance is what makes the silent
failures hardest to catch). The four commitments from §2.1 — image
encoding, action head, training data, evaluation discipline — are the
four axes on which Chapter 12 compares RT-2 (arXiv:2307.15818), OpenVLA,
and Octo head-to-head, and the four axes on which Chapter 13 argues for
π0's design.

The chapter has *not* covered three things you might have expected. It
has not trained anything — you ran a pretrained checkpoint and modified
no weights. Chapter 16 is the training chapter. It has not derived any
of the math behind the transformer or the cross-entropy loss; Chapters 3
and 8 do that, in that order. It has not touched a real robot;
everything you ran was in simulation. Chapter 17 covers the gap from
simulation to deployment, and Chapter 15 covers the evaluation
methodology that has to bridge it. Those omissions are deliberate. The
point of Chapter 2 was to keep the first VLA you ran small enough to
hold in your head, with the implicit promise that the rest of the book
widens the aperture without throwing away the picture you have just
formed.

§2.x closes the chapter with one hands-on exercise — a deliberate
variation on the §2.3 script that will reproduce one of the three silent
failures on demand — and the full reading list for the chapter.
