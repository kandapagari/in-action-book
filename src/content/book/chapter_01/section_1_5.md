---
chapter: 1
section: 1.5
title: What you will and will not find in this book
target_words: 2000
status: draft
prereqs: §1.1–§1.4 (the action problem, anatomy, history, four families)
key_refs:
  - Sutton & Barto (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
  - LaValle (2006). Planning Algorithms. Cambridge University Press.
  - Murphy (2022). Probabilistic Machine Learning. MIT Press.
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications, Challenges. arXiv:2505.04769.
  - Zhang et al. (2025). A Survey on Pure VLA Models. arXiv:2509.19012.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
---

# 1.5  What you will and will not find in this book

A book about a moving field has to draw lines. The lines I have drawn here are
not the only reasonable ones, but they are the ones that, after a year of
teaching the material, produced the smallest gap between what students wanted
to know and what they could do at the end of a semester. This section names
the lines and, more usefully, explains why they sit where they do. The aim is
to save you reading three other books to find out that this one does not cover
the topic you actually need.

The short version: this book is about action models *as a class of model*,
with the foundation / VLA family as the dominant case. It is not a robotics
textbook in the LaValle sense, not a controls textbook in the Khalil sense,
and not a deep-learning textbook in the Goodfellow sense. It assumes you have
already read, or are willing to read alongside it, the chapters of those books
that you need. The payoff for that assumption is that the present book can go
deep on the thing those books do not yet cover: how modern action models —
diffusion policies, action-tokenized transformers, flow-matching heads,
vision-language-action stacks — actually work, how they are trained, and how
they fit into a deployed robot.

## What the book covers

Four commitments shape the table of contents.

The first is *the four families, treated as one*. Most existing texts pick a
family and stay inside it. Sutton and Barto stay inside RL. LaValle stays
inside classical planning. Argall's survey (RAS 57(5), 2009) stays inside
imitation. The VLA surveys (Sapkota, arXiv:2505.04769; Zhang, arXiv:2509.19012)
stay inside the foundation family. Each of those treatments is excellent on
its own terms. The cost of staying inside one family is that you cannot see
the trade-off between them, and the trade-off is what determines what you
should actually build. Chapters 4 through 7 walk through the classical, RL,
and imitation families in enough depth that you can read a paper from any of
them; Chapters 11 through 14 then do the same for the foundation family, with
the goal that by the end of Part 4 you can read RT-1, OpenVLA, π0, Helix, and
GR00T N1 as variations on a small number of design choices, not as a
disconnected set of model names.

The second is *modern building blocks before modern systems*. The transformer,
the world-model recurrence, the diffusion or flow-matching action head — each
is a piece of machinery that recurs across many VLAs, and each is best learned
in isolation before you encounter it inside a 7-billion-parameter system whose
paper barely has room to explain it. Part 3 (Chapters 8–10) is the building-
blocks part. Read it before Part 4 even if you are tempted to skip ahead.

The third is *cite the canonical paper, not the survey*. With one exception
(the Sapkota survey, which is the broadest VLA reference I have found and is
cited in several chapters as a map of the field), every model and method gets
one canonical citation — the original paper or the official technical report.
The Appendix E.2 reading list and the Appendix F Model Zoo collect everything
in one place; the chapter sections themselves cite the one paper you need to
go read next.

The fourth is *the engineering of building one, not just the theory*. Part 5
(Chapters 16–17) is about what happens after the model is published. How do
you fine-tune an OpenVLA checkpoint (arXiv:2406.09246) on your own teleoperated
data without burning a week per attempt? How do you build a teleop dataset
that does not waste your time? How do you A/B test on real hardware when
every episode costs three minutes and a fresh setup? How do you put a safety
layer around a learned policy that you cannot certify? Most existing books
stop where the loss curve flattens. The interesting failures all happen after
that, and the chapters in Part 5 are where I have tried to be most concrete.

## What the book does not cover

Several reasonable topics are missing on purpose. Each one is missing because
there is a better existing reference, because including it would double the
book's length without serving its core argument, or — in two cases — because
the topic is moving too fast for a textbook in 2026 to be useful.

*Foundational RL theory beyond what an action model needs.* Sutton and Barto
(2018) is the canonical introduction; the second edition is freely available
and covers the convergence proofs, the contraction-mapping arguments, and the
function-approximation theory in more depth than this book attempts.
Chapter 5 gives you enough RL to read modern locomotion and RL-fine-tuning
papers; Chapters 6 and 7 give you the deep-RL machinery (DQN, PPO, SAC). If
you want the linear-MDP regret bounds, read Sutton and Barto, then Agarwal,
Jiang, Kakade, and Sun's notes.

*Classical motion planning beyond an introductory chapter.* LaValle (2006) is
1,000 pages on this topic and is the right reference. Chapter 4 of the present
book gives you STRIPS, IK, RRT, and computed-torque control at a level that
will let you read a modern paper that combines them with learning. It will
not turn you into a motion-planning researcher. If your application is in a
domain where classical methods dominate (industrial pick-and-place,
high-precision assembly, surgical robotics), read LaValle alongside this book.

*Hardware design, electronics, and mechanical engineering.* This book treats
the robot as a controllable system with a known action space and a known
observation interface. How you build the arm, choose the motors, calibrate
the cameras, or design the gripper is out of scope. Lynch and Park's *Modern
Robotics* (Cambridge, 2017) is one of the cleaner mechanical-side references;
Siciliano et al.'s *Robotics: Modelling, Planning and Control* (Springer, 2010)
covers the dynamics side.

*Computer vision and language modeling, except where they are used.* A
vision-language-action model is, by construction, sitting on top of a
vision encoder and a language model. Chapter 11 explains CLIP at the level
of "what is the contrastive objective and what does the resulting embedding
buy you," but it does not develop the design space of vision transformers,
and it does not derive the scaling laws of language models. For those,
Goodfellow, Bengio, and Courville (2016) and Murphy (2022) are still the
best textbook references; the *Building LLMs from Scratch* community has
several excellent recent treatments.

*Multi-robot systems, swarm robotics, and most of mobile robotics.* The
action models in this book are written as if there is one robot and one task.
Multi-agent coordination, market-based task allocation, and SLAM-heavy mobile
robotics are their own subfields with their own textbooks. The action-model
ideas here transfer (a foundation policy is a foundation policy whether the
robot has legs or wheels), but the surrounding engineering is different
enough that one chapter would have been worse than zero.

*Two topics I would have included if the field had stopped moving long enough.*
The first is *video-pretrained action models without robot data* — V-JEPA-style
self-supervised video models that are turning out to be useful priors for
action prediction. Chapter 9 sketches the world-model side, and §18.3
returns to the open question, but a confident chapter-length treatment is
six months away. The second is *reasoning-plus-action systems* in the
LLM-chain-of-thought sense — RoboBrain-style agents, Embodied-R1
(arXiv:2508.13998) — where the model produces an explicit plan in text before
producing actions. §18.4 catalogues the state of play; a stable treatment is
a year away.

## What you should read alongside this book

A small reading list, in priority order, for the gaps the book leaves open.

If you have *no machine-learning background*, read the first six chapters of
Murphy's *Probabilistic Machine Learning* (MIT Press, 2022) before Chapter 3.
The present book's Chapter 3 is a thirty-minute refresher, not a first
introduction.

If you have *no robotics background*, read the kinematics and dynamics
chapters of Lynch and Park's *Modern Robotics* (Cambridge, 2017) before
Chapter 4. The terminology — joint space, task space, Jacobian, end-effector
pose — is used without re-definition starting in Chapter 2.

If you want a *deeper RL background*, read Sutton and Barto (2018) alongside
Chapters 5 through 7. The present book's RL chapters are written to give you
the model-design vocabulary; Sutton and Barto give you the theory.

If you want to *follow the VLA literature* as it evolves past this book, the
two surveys to keep open in a browser tab are Sapkota et al.
(arXiv:2505.04769) and the Pure VLA Survey (arXiv:2509.19012). They overlap;
read them in that order. The Embodied Arena leaderboard
(arXiv:2509.15273) is the closest thing the field has to a community
benchmark and is worth checking every few months.

## Code, repositories, and what runs on what

The book is paired with a public code repository. Every chapter that contains
a non-trivial code listing has a corresponding folder in the repo with the
full, runnable version. The two extended hands-on exercises — fine-tuning
OpenVLA on a small LIBERO-derived dataset (Chapter 16) and building a
safety-monitor wrapper around a deployed policy (Chapter 17) — are the
points at which I would most encourage you to run code rather than read it.

The default training environment is PyTorch with the HuggingFace `transformers`
stack, because that is what every open-source VLA at the time of writing
(OpenVLA, SmolVLA arXiv:2506.01844, Octo arXiv:2405.12213, π0
arXiv:2410.24164's open weights) ships in. JAX appears in two places — the
Octo reference implementation and the π0 official release — and the JAX-
specific differences are flagged where they matter. The hands-on chapter
exercises are sized to run on a single 24 GB consumer GPU (an RTX 4090 or
similar) for the fine-tuning case, and on a CPU for everything else. The
chapters that describe pretraining at the scale of Open X-Embodiment
(arXiv:2310.08864) are explicitly *describing* it, not asking you to
reproduce it; that workload assumes a cluster.

## A note on the field's pace

The first draft of this book named π0 (Black et al., 2024,
arXiv:2410.24164) as the most recent flow-matching VLA. By the time the
fourth draft was finished, GR00T N1 (arXiv:2503.14734) was out, the FAST
action tokenizer (arXiv:2501.09747) had appeared, the Efficient VLA Survey
(arXiv:2510.24795) had been written, and Helix-02 had been announced. The
chapters that name specific models will date; the chapters that name design
choices — discrete versus continuous actions, single-system versus dual-
system, language-conditioned versus goal-image-conditioned — should not. If
a model name in this book is two years old and looks unfamiliar, treat it as
a representative of its design choice, not as the current state of the art,
and look it up in the most recent survey the appendix points you at.

With the scope of the book settled, §1.6 closes the chapter by collecting the
four ideas you should leave it carrying.
