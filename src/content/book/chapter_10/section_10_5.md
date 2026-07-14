---
chapter: 10
section: 10.5
title: "Action-head choices in modern VLAs"
target_words: 2000
status: draft
prereqs: §10.1–§10.4 (denoising, chunks, flow matching, and the latency/multimodality/smoothness framework). Helpful, §8.4 on tokenizing actions and §2.3 on the OpenVLA inference loop. Forward pointers to Ch. 12 (RT-2, OpenVLA, Octo), Ch. 13 (π0), Ch. 14 (Helix).
key_refs:
  - Brohan, A. et al. (2023). RT-2 — Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Kim, M. J. et al. (2024). OpenVLA — An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Pertsch, K. et al. (2025). FAST — Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
---

# 10.5  Action-head choices in modern VLAs

The previous section gave you three axes and promised the shipping
models would show up here to be measured against them. Now we collect on
that. Four systems — RT-2, OpenVLA, π0, and Helix — each bolted a
different action head onto a vision-language backbone, and each choice is
a legible answer to the three questions from §10.4: how multimodal is the
head, how many network calls does it cost, and what does it do about
smoothness. Read in order, the four also trace an arc: the field started
by reusing the machinery it already had, discovered where that machinery
broke, and then built heads specifically for control. This section is
that arc, told through the head each model chose.

We are looking at the heads only. The backbones, training data, and
scaling stories belong to Part 4 — RT-2 and OpenVLA return in Chapter 12,
π0 in Chapter 13, Helix in Chapter 14. Here the question is narrower and
sharper: given a VLM that can already see and read, how do you get an
action out of it, and what does that decision cost?

## RT-2: actions as words

RT-2's move was the one a language modeler would make first. If the
backbone is a VLM that emits text tokens, then make the action a string
of text tokens. RT-2 discretizes each continuous action dimension into
256 bins and assigns each bin an integer, then reuses tokens from the
model's existing vocabulary to spell out those integers (arXiv:2307.15818).
An action becomes a short sentence — seven numbers for a 6-DoF arm plus a
gripper flag — and the model predicts it the same way it predicts any
other tokens, one at a time, autoregressively.

The appeal is that nothing about the architecture changes. The same
transformer that learned "a photo of a cat" from web data now emits
"1 128 91 …" for a reach, and because the action head *is* the language
head, RT-2 can co-train on web vision-language data and robot data in one
mixture. That co-training is where its celebrated generalization comes
from, and we take it apart in Chapter 12.

Score it on the three axes. Multimodality: real but coarse. An
autoregressive classifier over bins can, in principle, place probability
on two separated bins and sample one — so RT-2 is not mode-averaging the
way an L2 head does. But 256 bins per dimension is a blunt grid, and
every action is snapped to a bin edge before it leaves the model, which
is a quantization error baked into the representation (§10.4). Latency:
this is where the choice hurts. Producing an action means decoding
several tokens in sequence, and RT-2 runs at roughly 1–3 Hz — fine for
slow pick-and-place, hopeless for anything dexterous. Smoothness: the bin
grid is a lattice of small discontinuities, and RT-2 predicts a single
next action rather than a chunk, so there is no within-chunk averaging to
lean on. RT-2's head is the "reuse what we have" corner: maximal
architectural simplicity, paid for in speed and in the granularity of
motion.

## OpenVLA: the same head, made open and studied

OpenVLA inherited RT-2's answer almost wholesale. It is a 7B-parameter
model — a Llama 2 language backbone fed by two vision encoders — and its
action head is again discrete token classification, 256 bins per
dimension, decoded autoregressively (arXiv:2406.09246). You already met
this head in §2.3, where the inference loop's last step turned predicted
token IDs back into a 7-vector. OpenVLA's contribution was not a new head
but an open, reproducible one: weights, code, and a fine-tuning recipe
the rest of us can actually run.

Because it is open, OpenVLA is also where the discrete head's limits got
measured in daylight. The bin grid caps precision; autoregressive
decoding caps frequency at roughly the same single-digit Hz as RT-2; and
fine-tuning to a new robot means the action distribution shifts under a
tokenizer that was frozen to someone else's data. None of these are
fatal — OpenVLA is a strong, widely used baseline precisely because the
discrete head is simple and trains stably — but together they framed the
open question the next two models answer: if the discrete-token head is
what is throttling frequency and smoothness, what replaces it?

There were two answers, and they split.

## FAST: keep the tokens, fix the tokenizer

The first answer kept the autoregressive head and blamed the tokenizer.
Naive binning wastes tokens: consecutive timesteps in a smooth trajectory
are nearly identical, so a per-timestep, per-dimension grid spends most
of its tokens re-encoding "almost the same as last step." FAST
(Frequency-space Action Sequence Tokenization, arXiv:2501.09747) instead
runs the action chunk through a discrete cosine transform — the same
frequency-space compression idea behind JPEG — and tokenizes the
resulting coefficients. Smooth trajectories compress to a handful of
low-frequency coefficients, so the same token budget now buys a longer,
higher-frequency chunk.

FAST matters here because it shows the RT-2/OpenVLA lineage was not a
dead end; the head was fine, the encoding was wasteful. A model with a
FAST tokenizer can express high-frequency, contact-rich motion that a
naive-binning model of the same size cannot, without abandoning the
autoregressive machinery that makes co-training with language easy. It is
the "repair the head we have" branch, and it is the branch π0 reaches for
when it needs a fast-to-train autoregressive variant. But the other
branch threw the discrete head out entirely.

## π0: a flow-matching action expert

π0 keeps the VLM backbone — a PaliGemma-class vision-language model — but
attaches a separate **action expert**, a smaller (~300M-parameter)
transformer that generates continuous action chunks by flow matching
(§10.3), no discretization anywhere (arXiv:2410.24164). The backbone
reads the image and instruction; the action expert takes that context and
integrates a flow field from noise to a chunk of, typically, 50
continuous actions. This is the head built *for* control rather than
inherited from language.

On the three axes it is the near-mirror image of RT-2. Multimodality:
high and sharp — a flow model represents the full action distribution and
samples a committed mode, so π0 handles the bimodal situations that make
a mean-regression head stall (§10.2). Latency: this is the whole point.
Flow matching's few-step sampling (§10.3), spread across a 50-action
chunk by receding-horizon execution (§10.4), is what lets π0 run its
control loop at up to 50 Hz — an order of magnitude past RT-2, on a
larger model. Smoothness: continuous outputs mean no bin lattice, and
chunk prediction gives within-chunk coherence, so the motion is smooth
enough for folding and other high-frequency, contact-rich tasks. π0 is
the "build the head the task needs" corner, and it pays for it with more
moving parts — a second network, a flow objective, sampling to tune —
than a single token head.

Notice that π0 did not have to choose *only* flow matching. The same
architecture ships in an autoregressive-plus-FAST variant for
faster training and a flow-matching variant for the smoothest, fastest
inference. That the same backbone hosts both heads is the cleanest
evidence for §10.4's thesis: the head is a swappable choice matched to a
deployment target, not a fixed property of the model.

## Helix: two heads at two clock rates

Helix, Figure's humanoid VLA, refuses to pick one point on the latency
axis and instead occupies two at once. It is a dual-system design: a
slow vision-language system (call it S2) reasons about the scene and the
instruction at roughly 7–9 Hz, and a fast visuomotor system (S1) produces
continuous motor commands at around 200 Hz, with S2's latent
understanding fed to S1 as conditioning (figure.ai/news/helix). The full
architecture is a Chapter 14 topic; what belongs here is the action-head
logic.

Helix's answer to the three axes is to stop treating them as one
decision. The tight-loop, smoothness-critical part of the problem —
keeping a humanoid's many joints coordinated in real time — goes to a
small, fast continuous head running far above any single big-model
frequency. The slow, multimodal, semantic part — deciding *what* to do,
which of several sane behaviors the instruction calls for — goes to the
big VLM, which can afford to be slow because it does not run every motor
tick. RT-2 put both jobs in one autoregressive head and got a rate that
served neither well; Helix splits the jobs so each runs at the rate it
needs. It is the strongest illustration that "which action head" can be
the wrong question — sometimes the answer is *two heads*, at two clocks.

## What the four choices add up to

Line the four up and the trend is not subtle. RT-2 reused the language
head and discovered its ceiling. OpenVLA made that head open and let the
community measure the ceiling precisely. FAST showed the ceiling was in
the tokenizer, not the head, and lifted it. π0 replaced the head with a
continuous flow-matching expert and broke into the high-frequency,
smooth-control regime the discrete head could not reach. Helix stopped
insisting on one head at all. The through-line is the field moving from
*inherited* action heads toward heads *designed* for the physics of
control — and, at the frontier, toward decoupling the semantic decision
from the motor decision so each can use the head that suits it.

Three practical takeaways survive out of this. First, discrete-token
heads (RT-2, OpenVLA) are simple, train stably, and co-train naturally
with language, at the cost of frequency and precision — a good default
when your loop is slow and your motions are coarse. Second, continuous
flow or diffusion heads (π0) are what you reach for when smoothness and
control frequency are load-bearing, and you accept more architectural
complexity to get them. Third, when no single frequency serves both the
thinking and the moving, a dual-system split (Helix) lets you have both.
Section 10.6 gathers the chapter — diffusion, flow matching, the three
axes, and these four choices — into the short list of things you should
now be able to reason about without looking anything up.
