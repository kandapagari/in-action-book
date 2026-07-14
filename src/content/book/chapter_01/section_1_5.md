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

A book about a moving field has to draw lines somewhere. The lines drawn here aren't the only reasonable ones, but after a year of teaching this material, they produced the smallest gap between what students wanted to know and what they could actually do by the end of a semester. This section names those lines and, more usefully, explains why they sit where they do. The goal is to save you from reading three other books just to discover that this one doesn't cover what you actually need.

Short version: this book treats action models as a class of model, with the foundation/VLA family as the dominant case. It isn't a robotics textbook in the LaValle sense, a controls textbook in the Khalil sense, or a deep-learning textbook in the Goodfellow sense. It assumes you've already read, or are willing to read alongside it, the chapters of those books you personally need. In exchange, this book goes deep on what those others don't yet cover: how modern action models, diffusion policies, action-tokenized transformers, flow-matching heads, vision-language-action stacks, actually work, how they're trained, and how they fit into a deployed robot.

## What the book covers

Four commitments shape the table of contents.

The first is the four families, treated as one. Most existing texts pick a family and stay inside it. Sutton and Barto stay inside RL. LaValle stays inside classical planning. Argall's survey (RAS 57(5), 2009) stays inside imitation. The VLA surveys (Sapkota, arXiv:2505.04769; Zhang, arXiv:2509.19012) stay inside the foundation family. Each treatment is excellent on its own terms, but staying inside one family means you never see the trade-off between them, and that trade-off is what actually determines what you should build. Chapters 4 through 7 walk through the classical, RL, and imitation families in enough depth that you can read a paper from any of them. Chapters 11 through 14 do the same for the foundation family, so that by the end of Part 4 you can read RT-1, OpenVLA, π0, Helix, and GR00T N1 as variations on a small number of design choices rather than a disconnected list of model names.

The second is modern building blocks before modern systems. The transformer, the world-model recurrence, the diffusion or flow-matching action head: each is a piece of machinery that recurs across many VLAs, and each is easier to learn in isolation than inside a 7-billion-parameter system whose paper barely has room to explain it. Part 3 (Chapters 8 through 10) is where the building blocks live. Read it before Part 4, even if you're tempted to skip ahead.

The third is citing the canonical paper rather than the survey. With one exception (the Sapkota survey, the broadest VLA reference available at the time of writing, cited in several chapters as a map of the field), every model and method gets a single canonical citation: the original paper or the official technical report. Appendix E.2's reading list and Appendix F's Model Zoo collect everything in one place, so the chapter sections themselves can just point you to the one paper you actually need next.

The fourth is the engineering of building one, not just the theory. Part 5 (Chapters 16 and 17) covers what happens after the model is published. How do you fine-tune an OpenVLA checkpoint (arXiv:2406.09246) on your own teleoperated data without burning a week per attempt? How do you build a teleop dataset that doesn't waste your time? How do you A/B test on real hardware when every episode costs three minutes and a fresh setup? How do you wrap a safety layer around a learned policy you can't certify? Most existing books stop where the loss curve flattens. The interesting failures all happen after that point, which is where Part 5 tries hardest to be concrete.

## What the book does not cover

Several reasonable topics are missing on purpose: a better existing reference already covers them, including them would double the book's length without serving its core argument, or, in two cases, the topic is moving too fast for a 2026 textbook to say anything useful.

Foundational RL theory beyond what an action model needs. Sutton and Barto (2018) remains the canonical introduction; the second edition is freely available and covers convergence proofs, contraction-mapping arguments, and function-approximation theory in more depth than this book attempts. Chapter 5 gives you enough RL to read modern locomotion and RL-fine-tuning papers, and Chapters 6 and 7 add the deep-RL machinery (DQN, PPO, SAC). If you want linear-MDP regret bounds, read Sutton and Barto, then Agarwal, Jiang, Kakade, and Sun's notes.

Classical motion planning beyond an introductory chapter. LaValle (2006) runs 1,000 pages on this topic alone and remains the right reference. Chapter 4 gives you STRIPS, IK, RRT, and computed-torque control at a level that lets you read a modern paper combining them with learning; it won't turn you into a motion-planning researcher. If your application lives in a domain where classical methods dominate, industrial pick-and-place, high-precision assembly, surgical robotics, read LaValle alongside this book.

Hardware design, electronics, and mechanical engineering. This book treats the robot as a controllable system with a known action space and a known observation interface. How you build the arm, choose the motors, calibrate the cameras, or design the gripper sits outside the scope here. Lynch and Park's *Modern Robotics* (Cambridge, 2017) is one of the cleaner mechanical-side references, and Siciliano et al.'s *Robotics: Modelling, Planning and Control* (Springer, 2010) covers the dynamics side.

Computer vision and language modeling, except where they get used directly. A vision-language-action model sits, by construction, on top of a vision encoder and a language model. Chapter 11 explains CLIP at the level of "what is the contrastive objective and what does the resulting embedding buy you," but doesn't develop the design space of vision transformers and doesn't derive the scaling laws of language models. For those, Goodfellow, Bengio, and Courville (2016) and Murphy (2022) remain the best textbook references, and the *Building LLMs from Scratch* community has produced several excellent recent treatments.

Multi-robot systems, swarm robotics, and most of mobile robotics. The action models in this book assume one robot and one task. Multi-agent coordination, market-based task allocation, and SLAM-heavy mobile robotics are their own subfields with their own textbooks. The action-model ideas here transfer fine (a foundation policy is a foundation policy whether the robot has legs or wheels), but the surrounding engineering differs enough that one chapter would have served worse than zero.

Two topics would have made the cut if the field had held still long enough. The first is video-pretrained action models without robot data: V-JEPA-style self-supervised video models that are turning out to be useful priors for action prediction. Chapter 9 sketches the world-model side and §18.3 returns to the open question, but a confident chapter-length treatment is still six months off. The second is reasoning-plus-action systems in the LLM-chain-of-thought sense, RoboBrain-style agents, Embodied-R1 (arXiv:2508.13998), where the model produces an explicit plan in text before producing actions. §18.4 catalogues the state of play, though a stable treatment is more like a year away.

## What you should read alongside this book

A short reading list, in priority order, for the gaps left open.

No machine-learning background? Read the first six chapters of Murphy's *Probabilistic Machine Learning* (MIT Press, 2022) before Chapter 3. This book's own Chapter 3 is a thirty-minute refresher, not a first introduction.

No robotics background? Read the kinematics and dynamics chapters of Lynch and Park's *Modern Robotics* (Cambridge, 2017) before Chapter 4. Terms like joint space, task space, Jacobian, and end-effector pose get used without re-definition starting in Chapter 2.

Want deeper RL background? Read Sutton and Barto (2018) alongside Chapters 5 through 7. This book's RL chapters aim to give you the model-design vocabulary; Sutton and Barto gives you the theory underneath it.

Want to follow the VLA literature as it evolves past this book? Keep two surveys open in a browser tab: Sapkota et al. (arXiv:2505.04769) and the Pure VLA Survey (arXiv:2509.19012), read in that order since they overlap. The Embodied Arena leaderboard (arXiv:2509.15273) is the closest thing the field has to a community benchmark, and it's worth checking every few months.

## Code, repositories, and what runs on what

The book pairs with a public code repository. Every chapter with a non-trivial code listing has a corresponding folder in the repo containing the full, runnable version. Two extended hands-on exercises, fine-tuning OpenVLA on a small LIBERO-derived dataset (Chapter 16) and building a safety-monitor wrapper around a deployed policy (Chapter 17), are the points where running the code beats reading it.

The default training environment is PyTorch with the HuggingFace `transformers` stack, since that's what every open-source VLA at the time of writing ships in (OpenVLA, SmolVLA arXiv:2506.01844, Octo arXiv:2405.12213, π0's open weights from arXiv:2410.24164). JAX shows up in two places, the Octo reference implementation and the π0 official release, and the JAX-specific differences get flagged where they matter. The hands-on exercises are sized to run on a single 24 GB consumer GPU (an RTX 4090 or similar) for the fine-tuning case, and on a CPU for everything else. The chapters describing pretraining at the scale of Open X-Embodiment (arXiv:2310.08864) are explicitly describing it, not asking you to reproduce it. That workload assumes a cluster.

## A note on the field's pace

The first draft of this book named π0 (Black et al., 2024, arXiv:2410.24164) as the most recent flow-matching VLA. By the time the fourth draft was finished, GR00T N1 (arXiv:2503.14734) was out, the FAST action tokenizer (arXiv:2501.09747) had appeared, the Efficient VLA Survey (arXiv:2510.24795) had been written, and Helix-02 had been announced. The chapters that name specific models will date. The chapters that name design choices, discrete versus continuous actions, single-system versus dual-system, language-conditioned versus goal-image-conditioned, shouldn't. If a model name in this book is two years old and looks unfamiliar, treat it as a representative of its design choice rather than the current state of the art, and look it up in whichever survey the appendix points you at most recently.

With the scope of the book settled, §1.6 closes the chapter by collecting the four ideas you should leave it carrying.
