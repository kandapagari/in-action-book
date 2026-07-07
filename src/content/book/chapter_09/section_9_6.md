---
chapter: 9
section: 9.6
title: Summary
target_words: 2000
status: draft
prereqs: §9.1–§9.5; the definition of a world model and its three uses, the RSSM and Dreamer, planning in latent space, video-prediction world models (Genie, V-JEPA), and the world-model-vs-VLA architecture debate
key_refs:
  - Ha & Schmidhuber (2018). World Models. NeurIPS.
  - Hafner et al. (2023). Mastering diverse domains through world models (DreamerV3). arXiv preprint.
  - Hansen et al. (2022). Temporal Difference Learning for Model Predictive Control (TD-MPC). ICML.
  - Bruce et al. (2024). Genie: generative interactive environments. ICML.
  - LeCun (2022). A path towards autonomous machine intelligence. OpenReview position paper.
---

# 9.6  Summary

Chapter 9 was a detour with a purpose. Every architecture the book builds
toward in Part 4 is a policy — a function that reads an observation and a
language instruction and emits an action, with no explicit opinion about
what that action will do. This chapter spent five sections on the road not
taken: models that learn how the world changes and then act by consulting
that model. The point was not to talk you out of the VLA path you are about
to follow, but to make sure you follow it knowing there is a serious
alternative, understanding what it offers and what it costs, and able to
recognize the pieces of it that keep reappearing inside VLA systems anyway.
This summary collects the load-bearing ideas and marks where the rest of the
book reaches back for them.

## The four ideas worth carrying forward

*A world model is supervised learning on the dynamics, and its value is that
it lets you try actions without paying for them.* §9.1 gave the definition
plainly: a world model approximates the transition function
$\hat{P}_\theta(s' \mid s, a)$ and usually the reward $\hat{R}_\theta$, fit
by watching transitions roll past, with the labels free because the next
state is simply what happened next. A policy answers "what should I do?"; a
world model answers "what would happen if I did that?" They point in
opposite directions, and the second question is worth asking because it can
be asked in imagination, where mistakes cost nothing. That single property
generates the three uses the whole chapter organized around — planning over
the model, training a policy inside it (the Dyna idea, from Sutton 1991),
and prediction as an end in itself — and Ha and Schmidhuber's car-racing
controller (2018), trained entirely inside its own hallucinated rollouts,
was the vivid proof that the third loop closes.

*The RSSM works because it splits the latent in two, and expresses dynamics
learning as a KL term.* §9.2 was the technical core. A purely deterministic
latent cannot represent uncertainty; a purely stochastic one cannot
remember across a horizon. The recurrent state-space model keeps both — a
deterministic GRU path carrying memory, a stochastic part riding on top —
and this one design choice is what made latent dynamics learnable at scale.
The mechanism that trains it is the prior/posterior pair: the posterior
corrects a prediction using the observation that arrived, the prior must
learn to predict without it, and driving the KL between them toward zero is
exactly the condition that lets the model dream without looking. Dreamer
then backpropagated imagined returns through the differentiable dynamics
into an actor, so the policy gets an analytic signal for how an action
changes the predicted future — no environment interaction needed to train
it, only to keep the model honest. DreamerV3 (arXiv, Hafner et al. 2023) is
the version to remember, because its contribution was robustness: one fixed
configuration across 150-plus tasks, up to and including a diamond in
Minecraft from scratch.

*Planning turns a one-step predictor into a decision-maker, and every
planner is only as good as the model is honest.* §9.3 developed
model-predictive control in latent space — plan a horizon, execute one
action, replan — and made the case that the replanning, not the forecast,
is what tolerates a mediocre model: you never trust step fifteen because you
replan fourteen times before you get there. The workhorses are
sampling-based, not gradient-based: CEM (which PlaNet used), MPPI (Williams
et al. 2017, the one you meet on real hardware), and PETS's ensemble (Chua
et al. 2018). The horizon problem — short is myopic, long compounds error —
is solved by stapling a learned value function onto the end of a short
rollout, the collaboration between planning and learning that TD-MPC
(Hansen et al. 2022) instantiates cleanly and that MuZero (Schrittwieser et
al. 2020) mirrors with tree search over a model trained to predict only
reward, value, and policy — never appearance. The warning that ties the
section together is model exploitation: a planner optimizes the model, so it
will happily propose whatever the model's *errors* reward, the model-based
cousin of the reward hacking from §5.4.

*Video-prediction world models break the frugality rule on purpose, and the
pixel-versus-representation split is the live disagreement.* §9.4 covered
the models that predict the full sensory future rather than a reward-shaped
latent, and bet that a model which can imagine the world in that fidelity
has learned something more transferable. The motive is data: the internet's
hundreds of thousands of hours of action-free, reward-free video are
unreachable by a DreamerV3-style loop but are exactly what a next-frame
objective can drink from. Genie (Bruce et al. 2024) is the demonstration —
a controllable, playable world learned from 200,000 hours of unlabeled
gameplay via a self-invented latent action vocabulary. V-JEPA (Bardes et al.
2024) is the counter-position made concrete: predict the missing region's
*representation*, not its pixels, following LeCun's (2022) argument that a
model should forecast consequences in an abstract space that has already
discarded unpredictable detail. Genie is generative and steerable at the
cost of modeling every pixel; V-JEPA is compact and transferable at the cost
of never being able to show you the future it imagines. Which pole you want
depends on whether the downstream job needs to *look* at imagined futures or
merely needs a representation that absorbed the world's dynamics.

## What you should be able to do now

Four concrete capabilities, in roughly the order the rest of the book will
ask for them.

You should be able to *state precisely what a world model is and what it is
not*. Not "a model of the world," but the actual object: a learned
approximation of the transition and reward functions, distinct from a policy
(which expresses preferences a model does not), distinct from a hand-built
simulator (which has privileged access to true state), and not necessarily
generative in the sense of producing watchable video. §9.1 drew those three
boundaries specifically so you can read later claims about "world models"
without being misled by the term's elasticity.

You should be able to *explain why the RSSM splits its latent state, and
what the prior, posterior, and KL term each do*. Given the Dreamer training
loop, you should be able to say which distribution is the imagination path,
which is the perception path, why the KL between them is the
dynamics-learning signal, and how a differentiable model lets Dreamer give
its actor an analytic gradient instead of a policy-gradient estimate. This
is the skill that lets §9.5's "world model with a reflex policy bolted on"
read as a precise description rather than a slogan.

You should be able to *set up a latent planning loop and name its failure
modes*. Given a learned model, you should be able to describe the
receding-horizon MPC loop, pick between CEM and MPPI and say why, explain
the value bootstrap that rescues a short horizon, and — most importantly —
anticipate model exploitation and list the mitigations (short horizons,
value bootstrapping, uncertainty penalties). §9.3 built this so that the
model-predictive control still running inside many deployed robots (§4.4)
reads as continuous with the learned-model version.

You should be able to *place any world model on the pixel-to-representation
axis and say what the placement costs*. Shown a new system, you should be
able to ask whether it predicts observations, reward-shaped latents, or
abstract features, and reason about the resulting trade in data
requirements, generality, transfer, and rollout cost. That axis is the one
§9.4 drew between Genie and V-JEPA, and it is the analytical habit the
section was training.

## Where the chapter has set up the rest of the book

Chapter 9 hands off in three directions, and the first is the most
immediate. §9.5's architecture debate is the frame for the entire remainder
of the book. You are about to spend Parts 4 and 5 inside the VLA camp, and
the chapter's job was to make that a choice you understand rather than a
default you inherited — VLAs use the data we can actually get, sidestep the
drift that cripples model rollouts, and inherit generalization from
language-vision pretraining, while the world-model camp keeps the better
claim on passive video, verifiability, and long-horizon reasoning. Hold both
when RT-2 (arXiv:2307.15818), OpenVLA (arXiv:2406.09246), and π0
(arXiv:2410.24164) appear in Part 4 with no explicit model of the world in
sight.

The second handoff is to Chapter 14's dual-system architectures. §9.5 noted
that a slow deliberative module paired with a fast reactive one is
structurally close to "consult a model, then act," and DreamerV3's own
policy-in-imagination is a world model with a reflex policy attached. The
convergence the debate section flagged is not rhetorical; Chapter 14 is
where the hybrid actually gets built, and a reader who saw the two camps
trading parts here will read it as synthesis rather than novelty.

The third handoff is quieter but real: to the video-pretraining thread that
resurfaces in Chapter 15's data discussion and Chapter 18's open problems.
§9.4 argued that a representation trained to predict video is a candidate
visual backbone for an otherwise ordinary VLA — the world model as
ingredient rather than as the whole architecture — and whether that
substrate beats end-to-end action training is one of the unsettled questions
Chapter 18 returns to directly.

## What the chapter has not covered

Two omissions are worth naming. The chapter treated world models almost
entirely at the level of principle and small-scale demonstration, and said
little about the engineering of *deploying* a model-based controller on a
real robot at control rate — the latency budgets, the fallback behavior when
a plan fails, the monitoring of model uncertainty online. That is deliberate:
those concerns belong with the deployment and safety machinery of Chapters
14 and 17, where the model-based system is one case among several, and where
the verifiability argument §9.5 made abstractly gets its concrete test.

The chapter also stayed shallow on the generative machinery itself. It said
that Genie tokenizes frames and that video-prediction models produce pixels,
but it deferred the actual mechanism — how you train a model to generate
high-fidelity output step by step — to Chapter 10, where diffusion and flow
matching are developed properly. This was the right split: §9.4 needed you
to understand *what* a video-prediction model is for and *why* the pixel
objective is contentious, not *how* the pixels get generated. That how is
next, and it is also the machinery the smooth-control VLAs of Chapters 10
and 13 use for their action heads, which is why Chapter 10 does double duty.

Chapter 9's contribution to the book's overall argument is to define the
alternative to the policy-only VLA rigorously enough that the VLA choice
reads as a bet with stakes rather than the only option on the table. The
three findings to carry forward are that a world model buys the ability to
act in imagination and that this generates its every use; that the RSSM's
deterministic/stochastic split and its KL-based dynamics objective are what
made learned latent models practical; and that the field's central
unresolved question — whether predicting actions from enough data really
teaches a network the physics it needs, or a shortcut that fails silently
off distribution — is exactly the question the world-model camp exists to
press. Part 3 now turns to the generative action machinery, diffusion and
flow, that the VLAs of Part 4 actually use to turn a chosen intention into
smooth motion.

§9.x closes the chapter with a hands-on exercise — training a DreamerV3
agent on a pixel-input control task and probing how imagination horizon and
the KL balance affect what it learns — and the full reading list for the
chapter.
