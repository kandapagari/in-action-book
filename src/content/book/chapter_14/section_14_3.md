---
chapter: 14
section: 14.3
title: "GR00T N1 and its N1.5–N1.7 successors: humanoid-flavored dual systems"
target_words: 2000
status: draft
prereqs: §14.2 (Helix box by box — the continuous latent channel, end-to-end training, and the 7B/80M split this section sets GR00T beside), §14.1 (the two-clocks argument and the System 1 / System 2 vocabulary), §10.2–§10.3 (Diffusion Policy and flow matching, which GR00T's fast path uses as its action module), §12.4 (Open X-Embodiment, part of GR00T's training mixture).
key_refs:
  - Bjorck, J. et al. (2025). GR00T N1, An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Google DeepMind (2025). Gemini Robotics-ER 1.5 and the Embodied Reasoning family. arXiv:2510.03342.
  - Figure AI (2025). Helix, A Vision-Language-Action Model for Generalist Humanoid Control. figure.ai/news/helix.
---

# 14.3  GR00T N1 and its N1.5–N1.7 successors: humanoid-flavored dual systems

Helix showed you one team's dual-system choices, made behind closed weights. GR00T N1 (arXiv:2503.14734), which NVIDIA released in March 2025, makes the same top-level split and then publishes almost all of it: the architecture, the training recipe, the data mixture, and the weights. That openness is what makes it the better teaching object of the two. Where §14.2 had to describe Helix's fast head as "an 80M-parameter transformer" and stop, GR00T lets you read exactly what goes into the seam between the systems, and it fills that seam differently enough from Helix that the contrast tells you which parts of the design are forced and which are one lab's preference.

## The same split, drawn NVIDIA's way

GR00T N1 is, at the top level, the picture from §14.1: a slow vision-language module that reasons, and a fast module that produces the motor commands. NVIDIA labels them System 2 and System 1, borrowing the same Kahneman gloss Figure did.

System 2 is a vision-language model, in the released version a variant of NVIDIA's Eagle VLM, running at roughly 10 Hz. It reads the wrist and head cameras and the language instruction and produces a stream of vision-language tokens, the internal representation a VLM builds before it would normally decode text. System 1 is a diffusion transformer, a DiT of the kind you met in §10.2, running much faster and generating continuous action chunks. The fast path does not decode discrete action tokens the way RT-1 or OpenVLA did; it denoises a chunk of continuous actions across a handful of diffusion steps, conditioned on both System 2's tokens and the robot's proprioception. If you have §10.3 in hand, think of the action module as a flow-matching-style generator sitting where Helix put a plain transformer.

That single substitution is the first real design fork between the two systems. Helix's fast head regresses actions directly; GR00T's samples them from a denoiser. The payoff is the one §10.4 laid out: a diffusion or flow head represents multimodal action distributions cleanly, so when there are two reasonable ways to grasp the tote, the head can keep both alive instead of averaging them into a grasp that reaches for neither. The cost is also the one §10.4 named, extra forward passes per action, which is why the number of denoising steps gets tuned down hard for real-time use.

## What crosses GR00T's seam

The channel between the systems is where §14.2 told you to look, so look here too. In Helix the channel is one continuous latent vector, deliberately thin. In GR00T it is wider: System 2 passes down its vision-language tokens, and the DiT cross-attends to them at every denoising step. So System 1 does not get a single compressed intent, it gets the reasoner's full token-level representation to attend over while it builds the action.

Neither choice is obviously right, and the difference is instructive. A thin latent forces System 2 to commit to a compact summary of its intent, which keeps the fast loop cheap and makes the handoff easy to reason about, at the price of throwing away whatever nuance did not fit in the vector. A wide token channel keeps that nuance available to the fast path, at the price of a heavier cross-attention step on every diffusion iteration and a tighter coupling that makes the two networks harder to develop separately. GR00T bets that the extra context is worth the extra cost; Helix bets the opposite. You cannot settle that from first principles, which is exactly the point of having two open-enough systems to compare.

Both networks train together, end to end, and here GR00T's openness pays off again because you can see what "together" means in data terms. NVIDIA trained on a pyramid: real robot teleoperation at the base narrow layer, a large body of human egocentric video in the middle, and synthetic data generated in simulation on top, including trajectories produced by NVIDIA's own neural "dream" generation. Real teleop teaches the exact embodiment; human video teaches manipulation priors that transfer; simulation fills the long tail cheaply. The mixture is the answer to a problem §14.2 skated past: 500 hours of teleoperation is a lot to collect, and video plus sim is how you get the reasoning module the breadth its VLM pretraining promised without teleoperating every scene by hand. Open X-Embodiment (§12.4) sits inside that real-robot layer.

## Why humanoid-flavored

Both Helix and GR00T target humanoids, and it is worth being precise about what that adds beyond "the robot has legs." A humanoid is a high-degree-of-freedom, bimanual, balance-critical platform, which stresses exactly the two failures §14.1 built the whole chapter around. The degrees of freedom are numerous enough that a single slow policy cannot close a stable loop over all of them; the balance constraint means a stale chunk is not merely suboptimal but a fall. GR00T is built to drive that class of body, and NVIDIA pairs it with reference humanoid hardware and its Isaac simulation stack, so the sim layer of the training pyramid matches the robots the policy is meant to run on.

The concrete demonstration to picture is a bimanual manipulation task on a humanoid: two arms coordinating to move an object from one hand to the other, or to hold a container steady with one hand while the other loads it. System 2 reads the scene at 10 Hz and settles the intent, "steady the box, place the item"; the DiT closes the fast loop, denoising action chunks that keep both arms coordinated and the grip stable as the item's weight shifts. The handoff between hands is the same class of contact event §14.2 traced through Helix, and it lands on the fast module for the same reason: the reaction window is shorter than System 2's cycle.

## The lineage: N1.5 to N1.7

GR00T did not stop at N1, and the successor line is a clean case study in where a dual-system design gets improved once the skeleton is fixed. The changes cluster in two places, and neither is the top-level split, which stays put.

The first place is the reasoning backbone. N1.5 and the later N1.6 swap and upgrade the VLM: N1.6 moves to a Cosmos-family backbone with a substantially larger action module (reported as roughly a 2× DiT). Cosmos matters because it is trained with physics-informed pretraining, NVIDIA's world-foundation-model line, so System 2 arrives already carrying some sense of how objects move and fall rather than only how they look. That is the §9.4 world-model idea leaking into the reasoner of a VLA, and it is worth flagging as a direction §18.3 returns to.

The second place is the data, and this is where N1.7 makes its central bet. N1.7, which reached general availability in mid-2026, is built on a Cosmos-Reason backbone (a Qwen3-VL-architecture VLM with the physics-informed pretraining above) and, more importantly, on a large first-person human-video corpus NVIDIA calls EgoScale, on the order of 20,000 hours of egocentric video feeding the low-level controller. The through-line from N1 to N1.7 is not a new architecture; it is the same dual-system skeleton fed progressively more and better data, especially human video, which is the cheapest way to buy manipulation priors at scale. Read against §14.2, this is GR00T doing openly what Helix's auto-labeled teleop did quietly: turning cheap, abundant video into the breadth the reasoning half was supposed to provide.

## A third family: Gemini Robotics-ER

Helix and GR00T are two instances of one idea, close enough that comparing them box by box is the whole exercise. Google DeepMind's Gemini Robotics line (arXiv:2510.03342) is close enough to belong in this chapter and different enough to be worth naming as a distinct family rather than a third clone.

The split is there: Gemini Robotics-ER, the "Embodied Reasoning" model, plays the System 2 role, and a separate action model plays System 1. What Gemini adds is a heavier emphasis on the reasoning half thinking before it acts. DeepMind calls the pattern "Embodied Thinking," and the shape is the embodied chain-of-thought idea we come back to in §18.4: the reasoner does not just emit an intent, it reasons through spatial and temporal steps, produces intermediate representations like points and trajectories on the image, and then hands a plan to the action model. The other capability worth naming is Motion Transfer, the ability to carry a skill learned on one embodiment over to a different one through the shared reasoning layer, which is one concrete attack on the cross-embodiment problem §18.1 takes up in full.

So the family tree of this chapter has three branches. Helix bets on a thin continuous latent and closed end-to-end training. GR00T bets on a wide token channel, a diffusion action head, and an openly published video-heavy data pyramid. Gemini bets on a reasoning half that thinks explicitly, in embodied chains of thought, before the fast half moves. All three keep the two-clock skeleton from §14.1; they disagree about what the slow clock should produce and how richly it should talk to the fast one.

The comparison also exposes the one question all three punt on, which is how you keep the fast loop stable when the slow loop's output updates underneath it, and how much real-time budget the whole stack actually has once you count the diffusion steps and the cross-attention. Naming the families is the easy part; the next section spends that budget line by line, because a dual-system design that misses its latency target is just two networks that fall over together.
