---
chapter: 9
section: 9.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §9.1–§9.6; Python with PyTorch (or JAX) installed; a working understanding of the transition/reward function a world model learns, the RSSM split into deterministic and stochastic state, imagination rollouts, planning by CEM or MPPI in latent space, and the world-model-vs-VLA debate; a pixel-input control task (DeepMind Control's `cartpole-swingup` or `walker-walk` is plenty); about three hours, most of it letting a small Dreamer-style agent train
key_refs:
  - Ha & Schmidhuber (2018). World Models. NeurIPS.
  - Hafner et al. (2019). Learning latent dynamics for planning from pixels (PlaNet). ICML.
  - Hafner et al. (2023). Mastering diverse domains through world models (DreamerV3). arXiv:2301.04104.
  - Bruce et al. (2024). Genie: generative interactive environments. ICML.
  - LeCun (2022). A path towards autonomous machine intelligence. OpenReview position paper.
---

# 9.x  Hands-on exercise + chapter references

Chapter 9 makes one claim that stays abstract until you watch it fail: a
learned world model is only as useful as its rollouts are accurate, and its
rollouts degrade the further into the future you ask them to reach. §9.1
defined the model, §9.2 built it, §9.3 searched it, and §9.5 argued about
whether the whole apparatus is the future of robotics or a detour. None of
that lands until you have trained a model, rolled it forward, and put its
prediction next to what actually happened. The headline exercise does exactly
that — train a compact Dreamer-style agent on a pixel task and lay its
imagined rollout beside the ground-truth frames — which is the canonical
Chapter 9 drill and the fastest way to see both the promise and the drift in
one picture. The remaining drills probe the edges: how horizon length trades
against accuracy, what the RSSM's stochastic branch is actually for, and where
the world-model bet of §9.5 stops being a slogan and becomes a measurement.

Budget about three hours. Most of that is wall-clock training time you spend
doing something else; the code you write is short. The one real cost is the
first exercise's training run, so start it early and read while it runs.

```
pip install torch dm-control
```

You need one pixel-input control task with a low-dimensional action. DeepMind
Control's `cartpole-swingup` trains fastest and is the gentlest place to watch
a latent rollout track reality; `walker-walk` is more interesting and more
punishing if you have the patience. Everything after the install runs offline.

## Exercise 9.x.1 — Train a world model, then watch it imagine

Build a small RSSM agent following §9.2. You do not need to reproduce
DreamerV3's full stack; a scaled-down version is enough to see the phenomena.
The pieces, all named in §9.2: a convolutional encoder from pixels to a
feature vector, a recurrent core that carries the deterministic hidden state
$h_t$ forward, a stochastic state $z_t$ with a prior (predicted from $h_t$
alone) and a posterior (corrected by the current observation), a decoder that
reconstructs the frame, and reward and continue heads. Train it on replay
data — either DreamerV3's own actor collecting experience, or, if you want to
skip the actor entirely for now, a buffer of random-policy episodes. Optimize
the §9.2 loss: reconstruction plus reward prediction plus the KL between prior
and posterior. Let the reconstruction loss plateau; on `cartpole-swingup` that
is roughly a hundred thousand environment steps, well under an hour on a
single GPU and a couple of hours on CPU.

Now the actual experiment, and the one the TOC names. Pick a held-out episode
the model never trained on. Feed the model the first few frames so it can
build up a posterior state — this is the "warm-up" or context window — then
cut off the observations and let it run on the *prior* alone, imagining
forward from its own predictions with the recorded actions as input. Decode
each imagined latent back to a frame. Lay the imagined filmstrip directly
above the ground-truth filmstrip, frame for frame, and look.

The first several frames will be nearly indistinguishable: the pole is where
it should be, the cart is where it should be. Then the two strips begin to
diverge, and the way they diverge is the lesson. Small dynamical errors
compound — a pole angle off by a degree becomes off by ten, then the imagined
pole is falling while the real one is upright. Mark the frame where you can
first tell the two strips apart without squinting. That frame is your model's
honest planning horizon, and it is almost always shorter than you hoped. This
is the drift §9.3 warned about and the reason §9.3's planners re-solve every
control step instead of trusting one long rollout.

Wall clock: training aside, about thirty minutes for the rollout and the
side-by-side plot.

## Exercise 9.x.2 — Find the horizon where planning stops helping

This drill turns the drift you just watched into a number that changes a
design decision. Take the trained model from 9.x.1 and use it to plan, the
§9.3 way: at the current latent state, sample action sequences of length $H$,
score each by predicted return under the model, and execute the first action
of the best sequence — the cross-entropy method or MPPI, either is fine.
Sweep the planning horizon $H$ across a range — say 1, 5, 15, 30, 50 steps —
and record the achieved return in the *real* environment at each setting.
Plot horizon against achieved return.

The curve is not monotonically increasing, and that is the point. Short horizons plan
myopically and leave return on the table. But past some horizon the achieved
return stops climbing and then falls, because the plan is now scoring itself
against imagined frames the model got wrong — you are optimizing an action
sequence to look good in a hallucination. The peak of that curve is the
horizon where your model's accuracy and your planner's foresight are in
balance, and it should sit near the divergence frame you marked in 9.x.1. For
the written part, connect the two numbers explicitly and state the
consequence: this is why §9.3's receding-horizon loop plans short and often,
and why TD-MPC (Hansen et al., 2022) bootstraps a learned value at the end of
a short rollout instead of rolling out to the true horizon at all.

Wall clock: about forty-five minutes reusing 9.x.1's model.

## Exercise 9.x.3 — Delete the stochastic state and break the drawer

§9.2 argued that splitting the latent into a deterministic $h_t$ and a
stochastic $z_t$ is the one design choice that makes the RSSM learnable, and
the argument is easy to nod along to and hard to feel. Feel it. Take your
9.x.1 model and ablate the stochastic branch: force $z_t$ to a deterministic
function of $h_t$, so the model can no longer represent "I don't know yet."
Retrain and re-run the imagination rollout from 9.x.2.

On a fully observed task like `cartpole-swingup` the ablated model may barely
suffer, which is itself worth noting — determinism costs nothing when nothing
is hidden. To make the cost appear, introduce partial observability: mask a
patch of the frame, or switch to a task where an outcome is genuinely
uncertain until revealed (the stapler-or-snake drawer of §9.2 is the mental
image). Now the deterministic model is forced to commit to one guess about the
hidden variable and is punished by reconstruction loss for guessing wrong,
while the full RSSM can spread probability across outcomes and pay less. Report
the reconstruction loss of both models on the partially observed task. The gap
is the concrete value of the stochastic state, and it is exactly the failure
§9.2 predicted rather than one you had to be told about.

Wall clock: about forty minutes if you reuse the training code and only change
the latent.

## Exercise 9.x.4 — Weigh the world-model bet against a policy

§9.5 laid out the disagreement — learn dynamics and plan on top, or map
observations straight to actions and let physics be absorbed implicitly — and
insisted you would spend the rest of the book in the second camp. Before you
do, price the first camp honestly. Take your trained world model and, without
any further environment interaction, train a policy purely inside its
imagination: roll the model forward from replayed start states, let an
actor-critic learn on the imagined trajectories, and deploy the resulting
policy on the real task. This is Dreamer's actual training loop, and the thing
to measure is sample efficiency: how many *real* environment steps did the
whole pipeline consume to reach a given return, counting only the steps used
to fit the model?

Compare that number against a model-free baseline from Chapter 7 — the SAC
agent from Exercise 7.x.2 is the clean comparison — run to the same return on
the same task. The world-model pipeline should reach competence in
dramatically fewer real steps, because it manufactures cheap experience in
imagination; that efficiency is the §9.5 world-model bet stated as a
measurement. Then write the counter-bet in one paragraph: name what the
pipeline cost you in return for that efficiency — the model to build and
debug, the drift that caps the horizon, the reconstruction objective that
spends capacity on pixels a policy would have ignored. That paragraph is the
§9.5 debate in your own words, grounded in a number you produced, and it is
the frame you should carry into Part 4, where every system drops the world
model and bets the other way.

Wall clock: about forty-five minutes if 9.x.1's model is already trained;
longer if you train the SAC baseline from scratch.

## Chapter 9 reading list

The works below are cited in §9.1–§9.6, grouped by purpose. Full
bibliographic entries for everything cited in the book live in Appendix E.2;
this is the chapter-local subset.

### The idea of a learned simulator

- Sutton, R. S. (1991). "Dyna, an Integrated Architecture for Learning,
  Planning, and Reacting." *SIGART Bulletin* 2(4). The original argument §9.1
  rests on: a learned model lets you interleave real experience with planning
  against imagined experience.
- Ha, D., & Schmidhuber, J. (2018). "World Models." *NeurIPS 2018*. The
  compress-pixels-then-predict-latents sketch §9.1 and §9.2 build from; the
  car-racing controller trained entirely inside its own dream.

### Latent dynamics and Dreamer

- Hafner, D., et al. (2019). "Learning Latent Dynamics for Planning from
  Pixels" (PlaNet). *ICML 2019*. The RSSM §9.2 dissects and the
  cross-entropy-method planner §9.3 and Exercise 9.x.2 use.
- Hafner, D., et al. (2020). "Dream to Control: Learning Behaviors by Latent
  Imagination" (Dreamer). *ICLR 2020*. The imagination-trained actor-critic
  Exercise 9.x.4 reproduces.
- Hafner, D., et al. (2023). "Mastering Diverse Domains through World Models"
  (DreamerV3). arXiv:2301.04104. The agent Exercise 9.x.1 scales down; §9.2's
  and §9.5's evidence that one recipe works across many domains.

### Planning against a learned model

- Chua, K., et al. (2018). "Deep Reinforcement Learning in a Handful of Trials
  using Probabilistic Dynamics Models" (PETS). *NeurIPS 2018*. §9.3's
  ensemble-of-models planner and its sample-efficiency argument.
- Williams, G., et al. (2017). "Information-Theoretic MPC for Model-Based
  Reinforcement Learning" (MPPI). *ICRA 2017*. The sampling-based controller
  §9.3 describes and Exercise 9.x.2 sweeps the horizon of.
- Hansen, N., et al. (2022). "Temporal Difference Learning for Model Predictive
  Control" (TD-MPC). *ICML 2022*. §9.3's short-rollout-plus-learned-value
  method; the fix for the horizon cliff Exercise 9.x.2 measures.
- Schrittwieser, J., et al. (2020). "Mastering Atari, Go, Chess and Shogi by
  Planning with a Learned Model" (MuZero). *Nature* 588. §9.3 and §9.4's
  reconstruction-free model that predicts only reward, value, and policy.

### Video-prediction world models

- Bruce, J., et al. (2024). "Genie: Generative Interactive Environments."
  *ICML 2024*. §9.4's pixel-predicting, action-inferring world model learned
  from unlabeled video; one pole of the §9.5 debate.
- Bardes, A., et al. (2024). "V-JEPA: Revisiting Feature Prediction for
  Learning Visual Representations from Video." *TMLR / Meta AI*. §9.4's
  predict-in-representation-space alternative to Genie's pixel reconstruction.

### The architecture debate

- LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence."
  OpenReview position paper. The §9.5 world-model bet in its strongest form:
  understanding of physics must be learned before control is easy.
- Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web
  Knowledge to Robotic Control." arXiv:2307.15818. The §9.5 counter-bet
  incarnate — no explicit dynamics model, physics absorbed implicitly; Part
  4's opening move.
- Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action
  Model." arXiv:2406.09246. The open policy-only system §9.5 contrasts with
  the world-model path.

## Chapter summary

Chapter 9 was the road not taken, walked far enough to know what is on it. You
can now say precisely what a world model is — a learned approximation of the
transition and reward functions, fit by supervised learning on transitions the
agent has already seen, whose value is that it lets you try an action in
imagination before paying for it in reality. You can build the RSSM that makes
this work on pixels, and explain why its split into a deterministic memory and
a stochastic "what I don't yet know" state is the choice that keeps it
learnable — a claim Exercise 9.x.3 turned from assertion into a reconstruction
gap you measured. You can plan against such a model with a receding-horizon
search, and you know from Exercise 9.x.2 why that search must be short and
frequent: rollout accuracy decays with horizon, and past a point the planner
optimizes against its own hallucinations. And you can state the architecture
bet of §9.5 in both directions — learn dynamics first, or map observations to
actions and let physics take care of itself — and back the trade-off with the
sample-efficiency number Exercise 9.x.4 had you produce. That bet is the hinge
of the book. Every system in Part 4 takes the second side of it, and Chapter
10 begins there, with the generative machinery — diffusion and flow models —
that lets a policy emit smooth, multimodal actions without ever modeling the
world it acts in.
