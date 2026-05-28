---
chapter: 2
section: 2.5
title: What is left for the rest of the book
target_words: 2000
status: draft
prereqs: §2.1 (the four hidden commitments of a VLA), §2.3 (the inference loop), §2.4 (the failure taxonomy); a working mental picture of OpenVLA driving LIBERO; willingness to treat the rest of the book as commentary on this one demo
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Octo Model Team (2024). Octo: An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - Pertsch et al. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
  - NVIDIA (2025). GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
---

# 2.5  What is left for the rest of the book

By this point you have a working OpenVLA running on LIBERO, three named
failure modes you can reproduce, and an honest read on where the model
breaks. That is more than most newcomers to VLAs see in their first month.
It is also less than 5% of what is going on inside the system you just ran.
The remaining 95% is the rest of the book.

This section is a map, not new content. It takes each component of the
loop from §2.3 — the image encoder, the tokenizer, the transformer, the
action detokenizer, the simulator, the training data behind all of it —
and points to the chapter that explains it in depth. Treat the map as a
reading guide. You do not have to read the book in order. You do have to
know where each idea lives.

## The loop, decomposed

The §2.3 loop has six things in it that we have so far treated as black
boxes: a vision encoder, a language tokenizer and a language model trunk
fused with the vision tokens, an action head that produces discrete
tokens, a detokenizer that maps tokens to a 7-vector, a simulator that
steps physics forward, and — invisible at inference but determinative of
everything — the training dataset and the training objective that made the
weights what they are. Six boxes, twelve chapters of unpacking.

The vision encoder you ran — a fused SigLIP-and-DINOv2 stack from Kim et
al. (2024, arXiv:2406.09246) — is one specific instance of a general idea
that is two decades older than VLAs. Chapter 11 traces the path from CLIP
through BC-Z through RT-1 (Brohan et al., 2022, arXiv:2212.06817), which
is the moment language-conditioned robotic policies became possible at
scale. Chapter 12 covers what changed when the vision-language backbone
grew from a few hundred million parameters to several billion, which is
the RT-2 (Brohan et al., 2023, arXiv:2307.15818) and OpenVLA story. If you
want to understand why the SigLIP-and-DINOv2 fusion was chosen over a
single encoder, those two chapters together provide the argument. The
short version, for now: SigLIP gives semantic alignment with text, DINOv2
gives geometric features that survive lighting and viewpoint change; the
fusion buys you both at the cost of slightly more compute per frame.

The language tokenizer is the most boring component in the loop and the
hardest one to think about. It is boring because it is just the standard
Llama-2 byte-pair tokenizer. It is hard because the *re-use* of the bottom
256 token IDs as action bins is the trick that lets a language model do
control without architectural changes. Chapter 11 explains action
tokenization for the first time; Chapter 13 returns to it with FAST
(Pertsch et al., 2025, arXiv:2501.09747), a more compressed encoding that
delivers smoother control on the same backbone. The "256 bins per axis"
choice in OpenVLA is one design point on a Pareto frontier between
resolution, sequence length, and decoding latency. Chapters 11 and 13 are
where that Pareto frontier gets drawn.

The transformer trunk is, mathematically, the same object the reader will
meet in Chapter 8. The two pages on attention there are deliberately
sparse: just enough to read the rest of the book without misconceptions.
Chapter 8 also covers Decision Transformer and Trajectory Transformer,
the lineage that established "control as sequence modeling" before VLAs
took the idea to a foundation-model scale. If your background is in NLP
or CV and you already know transformers, Chapter 8's transformer pages
are skippable; the Decision Transformer pages are not, because they
provide the framing under which a next-token-prediction model can
plausibly be a controller. If your background is in classical robotics
and the word "transformer" is opaque, Chapter 3 first, then Chapter 8.

The action head — the part that emits the seven discrete tokens — is
covered twice in the book, from two different directions. Chapter 10 is
about generative heads in general (diffusion, flow matching, masked
prediction), with Diffusion Policy and ACT as the canonical
imitation-learning representatives. Chapter 13 is about π0 (Black et al.,
2024, arXiv:2410.24164) specifically, which replaces the discrete head
you ran today with a flow-matching head that emits continuous action
chunks. The trade-off, previewed: discrete heads are simpler to train and
debug, but they cap your control resolution and produce jagged
trajectories at high frequency. Continuous heads are smoother and faster
in chunks but harder to train and more sensitive to data quality. The
choice is not yet settled in the field. The book gives both sides their
fair share.

The detokenizer — the unassuming function that maps token 31882 to
`dx=-0.012` — is the smallest piece of code in the loop and the source
of more silent bugs than the rest of the model combined. You met one of
those bugs in §2.4 (the wrong `unnorm_key`). Chapter 11 covers
detokenization formally; Chapter 12 covers how Open X-Embodiment
(O'Neill et al., 2023, arXiv:2310.08864) calibrates per-embodiment
statistics. Chapter 15 explains why "21 different embodiments" is a
statement that requires 21 different detokenizers and what that means for
generalization claims. The detokenizer is one of those components you
do not appreciate until it eats a week of your time.

The simulator is the one box in the loop that this book deliberately
treats as a tool rather than a subject. LIBERO (Liu et al., 2023,
arXiv:2306.03310) is one of several robotics simulators a modern VLA gets
evaluated on, alongside CALVIN, RoboCasa, and SimplerEnv. Chapter 15
gives them all a tour and tells you which one to reach for in which
situation. Appendix D is the practical setup guide. We will not, in this
book, derive contact dynamics or rigid-body integration; that is
adequately covered elsewhere, and we have a finite page budget. What we
do cover is how to evaluate a policy in simulation in a way that
correlates with real-robot success, which is a harder problem than the
simulators themselves.

The training data and objective are the largest topic in the book in
terms of pages, and the most easily underestimated in terms of effect.
OpenVLA's weights are a function of three ingredients: the Llama-2 and
SigLIP-and-DINOv2 backbones (pretrained on internet-scale text and
image-text data), the 970k-trajectory Open X-Embodiment dataset, and
LIBERO-specific fine-tuning data. The first ingredient is covered as
prerequisite material in Chapter 3 and as architectural context in
Chapter 11. The second is the centerpiece of Chapter 12 and of §15.2.
The third is the kind of thing that gets one paragraph in a paper and an
entire Chapter 16 in a book that is trying to teach you to fine-tune your
own model. The objective itself — cross-entropy over action tokens — is
the simplest possible loss for a sequence model and the right place to
start; Chapters 6 and 10 cover the imitation-learning alternatives that
add structure (DAgger, diffusion, flow matching) and Chapters 5 and 7
cover the reinforcement-learning alternatives that change the loss
entirely.

## A map by chapter

Here is the same content rotated 90 degrees: every chapter, in order,
with the part of today's loop it explains.

Part 1 — *Foundations.* Chapter 3 is the math-and-ML prerequisite kit,
in the form most useful for action models specifically: enough linear
algebra to keep up with end-effector kinematics, enough probability to
read RL pseudocode, enough PyTorch to follow a 50-line training loop.
It does not stand on its own as a math reference. It is calibrated to
exactly the prerequisites used elsewhere in the book.

Part 2 — *The lineage.* Chapters 4 through 7 cover the four families of
action models that came before VLAs, in the order they were invented.
Chapter 4 is classical: STRIPS, PDDL, inverse kinematics, computed
torque. Chapter 5 is reinforcement learning from first principles: MDPs,
value iteration, Q-learning. Chapter 6 is imitation learning, which is
the loss family OpenVLA uses. Chapter 7 is deep RL: DQN, PPO, SAC. None
of these methods are obsolete; they are alive in every modern robot
stack, often inside the same robot as a VLA. The book's claim is not
that VLAs replaced them. The claim is that VLAs are best understood as
a particular synthesis of imitation learning and large-scale
pretraining, and you understand the synthesis better when you know what
went in.

Part 3 — *Modern building blocks.* Chapter 8 is transformers and
sequence models for control. Chapter 9 is world models — the
model-based alternative to a feed-forward VLA, with RSSM and Dreamer as
the imitation-free representatives, and Genie and V-JEPA as the
video-pretrained representatives. Chapter 10 is diffusion and flow
matching for action generation. Together these three chapters supply
the architectural vocabulary that the foundation-model chapters use
without re-deriving.

Part 4 — *Foundation action models in depth.* This is where the book
spends its longest sustained argument. Chapter 11 retraces the CLIP →
BC-Z → RT-1 path. Chapter 12 covers the scaling moment: RT-2, OpenVLA,
Octo (arXiv:2405.12213). Chapter 13 covers π0 and the move to
flow-matching action heads. Chapter 14 covers dual-system architectures
like Helix and GR00T N1 (arXiv:2503.14734), which add a slower
high-level planner on top of a fast policy. Chapter 15 covers datasets
and evaluation in detail. By the end of Part 4 the reader should be
able to look at a new VLA paper on arXiv and place it on the map within
about fifteen minutes.

Part 5 — *Building with action models.* Chapter 16 is fine-tuning a VLA
for your own robot, which is the practical question most readers of
this book will eventually want to answer. Chapter 17 is evaluation,
safety, and deployment for VLAs that have to do something other than
look good in a demo. Chapter 18 is the open-problems chapter:
embodiment transfer, long-horizon tasks, video pretraining,
reasoning-plus-action. It is the chapter that has the shortest
shelf-life, because the field will move underneath it; it is also the
chapter that earns its keep by telling you what to read after the book.

## How to read the rest of the book

If you read straight through, you will spend roughly one to two evenings
per chapter for the technical chapters and a weekend each on the
foundation-model chapters. That is honest pacing for an upper-undergrad
or first-year grad student. If you are short on time and want to get
back to standing up your own VLA, the minimum path is Chapter 3 (only
the parts you do not already know), Chapter 6, Chapter 8, Chapter 11,
Chapter 12, Chapter 16. That is six chapters; it covers the imitation
loss, the architecture, the lineage, the current state of the art, and
the practical fine-tuning recipe. The rest is depth and context.

If you came to this book wanting to understand a specific model — π0,
GR00T N1, RT-2 — the table in Appendix F (the VLA model zoo) cross-
references each model to the chapters that explain its components, and
the reading list in Appendix E.2 gives the canonical paper for each.
Reading in that order is also valid; many practitioners do.

The next section closes Chapter 2.
