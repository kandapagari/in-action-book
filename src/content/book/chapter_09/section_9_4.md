---
chapter: 9
section: 9.4
title: "Video-prediction world models (Genie, V-JEPA)"
target_words: 2000
status: draft
prereqs: §9.1 (a world model predicts whatever the downstream use needs; the three uses), §9.2 (RSSM latents, prior vs. posterior, reconstruction loss), §9.3 (MuZero and TD-MPC deliberately do not reconstruct observations), §8.1 (transformers and tokenization), §3.2 (KL, representation learning).
key_refs:
  - Bruce et al. (2024). Genie: Generative Interactive Environments. ICML.
  - Bardes et al. (2024). V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video. TMLR / Meta AI.
  - LeCun (2022). A Path Towards Autonomous Machine Intelligence. Position paper, Open Review.
  - Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). arXiv preprint.
---

# 9.4  Video-prediction world models (Genie, V-JEPA)

The models in §9.2 and §9.3 all shared a frugal instinct. The RSSM
reconstructed observations only because reconstruction was a convenient
training signal; MuZero and TD-MPC dropped even that and predicted nothing
but reward, value, and policy. The guiding rule was: model only what the
decision needs. This section is about the line of work that broke that
rule on purpose. Instead of compressing the world down to the few numbers
a planner reads off, these models predict the future in its full sensory
detail — the next frame of video, pixel by pixel — and bet that a model
which can imagine the world in that fidelity has learned something more
transferable than any reward-shaped latent. The bet is not obviously
right, and the two systems this section centers on, Genie and V-JEPA,
disagree sharply about how to make it. That disagreement is the point.

## Why predict pixels at all

The case for reconstruction-free latents was airtight for a fixed task
with a known reward. It stops being airtight the moment you want one model
to serve many tasks you have not specified yet, and it collapses entirely
when the data you want to learn from has no reward and no action labels at
all.

Consider the data. The internet holds hundreds of thousands of hours of
video — people cooking, driving, assembling furniture, playing games — and
essentially none of it comes annotated with the actions that produced it
or the rewards that mattered. A DreamerV3-style model cannot touch this,
because its whole training loop assumes it collected the data itself and
therefore knows the actions and rewards. A model that learns purely by
predicting the next frame has no such requirement. The supervision is the
video itself: show it frames $1$ through $t$, ask it to predict frame
$t{+}1$, and the pixels grade the answer. This is the same self-supervised
bargain that made language models scale (§8.1), transplanted to
observation streams, and it is why "video prediction" and "world model"
have become nearly synonymous in the foundation-model era.

The second argument is generality. A reward-specific latent is sharp but
narrow — TD-MPC2's state knows exactly what its tasks need and nothing
else. A model that must predict the full next frame is forced to represent
whatever governs how scenes evolve: that unsupported objects fall, that
occluded objects persist, that a pushed cup slides and does not teleport.
Those regularities are task-agnostic. If you can learn them once from
passive video, you have a substrate that many downstream policies can
share, which is exactly the promise Part 4's foundation models chase from
the action side.

## Genie: controllable worlds from action-free video

Genie (Bruce et al. 2024) is the cleanest demonstration that the bargain
pays off. It was trained on over 200,000 hours of publicly available
2-D platformer gameplay video, with no action labels, no rewards, and no
knowledge of which button produced which effect. From that alone it
learned a *generative interactive environment*: give it a single image — a
photograph, a sketch, a frame from a game it never saw — and it turns that
image into something you can play, frame by frame, responding to a control
input at every step.

The trick that makes this work despite the missing action labels is a
**latent action model**. Genie has three learned parts, all trained
together on raw frames. A **video tokenizer** compresses each frame to a
grid of discrete tokens, in the spirit of §8.4's "what gets tokenized"
question, so the rest of the model works over a manageable sequence rather
than raw pixels. A **latent action model** looks at a pair of consecutive
frames and infers a small discrete code that explains the transition
between them — what "action" must have occurred to turn this frame into
the next. Crucially it is trained only from the frames, so it invents its
own action vocabulary; nobody tells it "this code means jump." A
**dynamics model**, an autoregressive transformer, then predicts the next
frame's tokens given the past tokens and one of these latent action codes.

At training time the latent action is inferred from the real next frame,
the way §9.2's posterior read the true observation. At play time you
withhold the future and instead *supply* the latent action yourself —
pick one of the small set of learned codes — and the dynamics model rolls
the world forward under your control. The astonishing result is that these
self-discovered codes are consistent and interpretable: the same code
reliably moves the avatar left across wildly different generated worlds,
another makes it jump, and a human can learn to "play" a generated
environment within seconds even though the control scheme was never
designed. Genie learned a controllable model of how worlds change from
watching, without ever being told what control means.

For a robotics reader the implication is direct and is why Genie sits in
this chapter rather than a graphics one. Robot teleoperation data is
expensive and scarce; passive video of the world is abundant. A world
model that can be trained from action-free video and later steered by a
small learned action interface is a template for pretraining on the cheap
data and grounding on the expensive data — a template later humanoid
efforts, and Genie's own successors aimed at 3-D and robotic domains,
build on directly.

## The objection: pixels are the wrong thing to predict

Genie predicts pixels, and pixels are, in a real sense, a terrible thing
to spend a model's capacity on. Most of the bits in a video frame are
detail that no decision depends on and no model can predict anyway: the
exact texture of gravel, the precise flicker of a shadow, the arrangement
of leaves on a tree the robot will never touch. A next-frame objective
pours enormous modeling effort into getting those unpredictable details
plausible, and a model graded on pixel accuracy is punished for admitting,
correctly, that it cannot know them. This is the same complaint §9.3 made
about long rollouts drifting into fantasy, sharpened into a critique of
the objective itself.

Yann LeCun's position paper (LeCun 2022) turned this complaint into a
design principle. The argument: an intelligent agent should predict the
*consequences* of actions at an abstract level, not render the sensory
future in full. Prediction should happen in a representation space that
has already thrown away the unpredictable noise, so the model spends its
effort on structure that matters and is not penalized for failing to
hallucinate irrelevant detail. This is precisely the lesson MuZero and
TD-MPC learned for reward-driven control in §9.3 — model consequences, not
appearance — promoted to a claim about how any world model should be
built. LeCun's name for the resulting architecture is the *Joint
Embedding Predictive Architecture*, or JEPA.

## V-JEPA: predict the representation, not the frame

V-JEPA (Bardes et al. 2024) is that principle made concrete for video.
The setup looks like masked prediction, but with a decisive twist. Take a
video, encode it, and mask out a large spatiotemporal chunk — a block of
patches across several frames. A generative model like a masked
autoencoder would try to reconstruct the missing *pixels*. V-JEPA instead
predicts the missing region's *representation*: it runs a target encoder
over the full, unmasked video to produce feature vectors for the masked
region, then trains a predictor to match those features from the visible
context alone. The loss is computed entirely in latent space. No pixel is
ever reconstructed.

The mechanism that keeps this from collapsing to the trivial solution
(map everything to a constant and predict the constant) is an asymmetry
borrowed from self-supervised image learning: the target encoder is not
trained by gradient descent but is an exponential moving average of the
context encoder, updated slowly, so it provides a stable, non-trivial
target the predictor must actually work to hit. What comes out is a frozen
video encoder whose features transfer well to downstream recognition and
motion-understanding tasks without any fine-tuning of the backbone —
strong evidence that predicting in representation space captures the
structure of how scenes evolve while spending nothing on rendering.

The contrast with Genie is the whole argument of this section in one line.
Genie predicts frames and gets an interactive, playable, visibly
impressive world you can steer — at the cost of modeling every pixel.
V-JEPA predicts features and gets a compact, transferable understanding of
dynamics — at the cost of not being able to show you the future it
imagines. One is generative and controllable; the other is abstract and
efficient. Which you want depends on whether the downstream job needs to
*look* at imagined futures (planning by rolling out visible states, or
generating training scenarios) or merely needs a representation that has
absorbed the world's dynamics.

## Where this meets action

Neither Genie nor V-JEPA is a policy; both are world models in the §9.1
sense, waiting to be put to one of the three uses. Two connections to
action are worth naming now and returning to later.

The first is planning by video. If a model can predict future frames
conditioned on candidate actions, a planner can, in principle, score those
candidates by imagining their visible outcomes and checking whether a goal
frame is reached — a pixel-space version of §9.3's shooting search. Early
robot systems along these lines conditioned video prediction on a
goal image and selected actions whose predicted rollouts best approached
it. The obstacle is cost and drift: generating high-fidelity video for
hundreds of candidate rollouts at control rate is far more expensive than
rolling an RSSM latent, and the pixel objective's blurring compounds over
the horizon exactly as §9.3 warned.

The second is pretraining. A representation trained to predict video —
whether Genie's tokens or V-JEPA's features — is a candidate visual
backbone for a downstream policy, one that has learned about physical
dynamics from data no teleoperation rig could ever collect. Whether that
substrate actually beats a policy trained end-to-end on action-labeled
data is unsettled, and it is the crux of the architecture debate §9.5
takes up head-on: the world-model camp holds that understanding the world
must come first and control follows, while the VLA camp of Part 4 bets
that pouring action-labeled data through one network learns the relevant
dynamics implicitly, no separate world model required.

For now, hold onto the split this section drew. Video-prediction world
models predict the observation itself, which unlocks learning from
abundant action-free video and produces models you can imagine with in
full sensory detail — a real gain in generality over the reward-shaped
latents of §9.3. But predicting pixels is expensive and partly wasted, and
the JEPA response — predict abstract representations instead — trades away
the visible, controllable rollout for efficiency and transfer. Genie and
V-JEPA are the two poles of that trade, and the question of which pole the
future of embodied intelligence should sit closer to is exactly the
debate we turn to next.
