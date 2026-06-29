---
chapter: 8
section: 8.4
title: "What gets tokenized: states, actions, returns, language"
target_words: 2000
status: draft
prereqs: §8.1 (the transformer, tokens, the embedding table, causal decoding); §8.2 (Decision Transformer, returns-to-go, the modality-specific embedding trick); §8.3 (the Trajectory Transformer's per-dimension binning and the multimodality argument); §5.1 (states, actions, rewards, returns).
key_refs:
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Janner et al. (2021). Offline reinforcement learning as one big sequence modeling problem (Trajectory Transformer). arXiv:2106.02039.
  - Brohan et al. (2022). RT-1: robotics transformer for real-world control at scale. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2: vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818.
  - Pertsch et al. (2025). FAST: efficient action tokenization for vision-language-action models. arXiv:2501.09747.
---

# 8.4  What gets tokenized: states, actions, returns, language

The last three sections kept saying "tokenize the trajectory" as if it
were one operation. It is not. A trajectory is a heterogeneous thing — a
camera frame, a seven-dimensional joint command, a scalar reward, a
scalar return, and, in the models of Part 4, a sentence of English — and
every one of those quantities has to be turned into an entry in a token
sequence before a transformer can touch it. The choices you make there
are not plumbing. They decide what the model can represent, how long its
sequences get, and whether the language pretraining of a vision-language
model survives contact with robot actions at all. This section pulls the
tokenization question out of the background and looks at it directly,
because it is the hinge between the RL-flavored sequence models of this
chapter and the VLAs of Part 4.

The core difficulty is that "token" means two incompatible things. To a
transformer, a token is an integer index into an embedding table — a
discrete symbol, one of a fixed vocabulary of, say, fifty thousand. But
the quantities in a robot trajectory are mostly continuous. Reconciling
those two facts is the whole game, and there are exactly two strategies:
discretize the continuous thing into a symbol, or keep it continuous and
project it into the embedding space with a learned layer. Every system
in this book picks one of these per modality, and the interesting part is
that they do not pick the same one for every modality.

## Two strategies, modality by modality

Take the four quantities in a Decision Transformer trajectory —
state, action, return-to-go, reward — and ask of each: symbol or vector?

The Decision Transformer (arXiv:2106.01345) keeps states, actions, and
returns *continuous*. It does not bin them. Instead it learns a separate
linear projection for each modality — one matrix that maps a state vector
into the embedding space, another for actions, another for the scalar
return — adds a timestep encoding, and feeds the resulting vectors to the
transformer as if they were word embeddings. The transformer never sees
an integer index for these; it sees vectors that happen to come from
three different little networks. This is the projection strategy, and its
virtue is that no resolution is thrown away: a joint angle of 0.4137
radians arrives intact.

The Trajectory Transformer (arXiv:2106.02039), as §8.3 described, does
the opposite. It bins every dimension of the state and action and the
reward and the return into a discrete vocabulary, roughly a hundred bins
per dimension, and trains with plain cross-entropy over those symbols.
This is the discretization strategy. Its virtue is the one §8.3 dwelt on:
a softmax over bins is a full distribution, so the model can be honestly
multimodal, and you can run beam search over the resulting symbols. Its
cost is resolution and sequence length — one token per dimension per
step.

So two papers, published weeks apart, on the same hardware benchmark,
made opposite tokenization calls, and both worked. That is the lesson to
internalize before Part 4: tokenization is a design axis, not a settled
convention, and the right choice depends on what you are going to *do*
with the tokens. If you plan to search (§8.3), you want discrete symbols.
If you plan to regress a single action and care about precision, you lean
toward projection. Hold that tension; it explains the entire RT-1 →
RT-2 → π0 progression.

## Why returns are the strange token

The return-to-go deserves a moment of its own, because it is the token
that has no analogue in language and the one beginners most often
mishandle. In §8.2 the return-to-go was the conditioning signal: you
prepend the performance you *want* and the model produces actions
consistent with it. Tokenizing it is easy — it is a scalar, project it or
bin it like any other — but its *semantics* are unusual. Unlike a state
or an action, the return is not something the world hands you; it is a
command you supply at test time, and it can be a lie. Asking for a return
higher than anything in the dataset is exactly how you probe a Decision
Transformer's extrapolation, and it usually fails gracefully into "the
best behavior I saw," not magic. The point for tokenization is that the
return token occupies a real slot in the vocabulary and the sequence, and
it costs you context length just like everything else. When VLAs in Part
4 drop the return token entirely — RT-1 and successors are imitation
learners with no return conditioning — they are buying back that slot and
that conceptual complication, which is one quiet reason the foundation
models look simpler than the offline-RL models they descend from.

## Tokenizing actions for a language model

Now the bridge the chapter promised. RT-1 (arXiv:2212.06817) is the first
system in this book to tokenize robot actions specifically so they can
live in the same sequence as image and language tokens. Its scheme is
the discretization strategy, applied deliberately: each dimension of the
robot's action — three for end-effector position, three for rotation, one
for the gripper, plus a mode switch — is uniformly discretized into 256
bins. An action becomes a short string of integers in the range 0–255,
and the transformer predicts them autoregressively, exactly as it would
predict the next word.

The number 256 is not arbitrary, and this is the idea that detonates in
RT-2 (arXiv:2307.15818). A pretrained vision-language model already has a
vocabulary of tens of thousands of tokens. RT-2's move is to *reuse* that
vocabulary for actions: it takes 256 token IDs the language model already
has — in the published version, the integer-string tokens, or rarely used
ones reassigned to mean action bins — and declares them to be the action
alphabet. No new embedding table, no new output head. The robot's action
space is overloaded onto symbols the model learned during web
pretraining. Producing an action is then *literally the same operation*
as producing text: emit a token. This is why RT-2 can be fine-tuned from
a web-scale VLM without architectural surgery, and why the web knowledge
partly survives — the model is doing the only thing it ever knew how to
do, predict the next token, and some of those tokens now happen to mean
"move left." When the chapter's learning objectives say to connect
sequence modeling to "the action-token decoding scheme used by
RT-1/RT-2," this overloading is the scheme.

A small worked example makes the sequence concrete. A single RT-2
training example, flattened, looks like

```
<img patch tokens> "pick up the can"
  → 134 211 6 248 9 17 0
```

where the seven trailing integers are the binned action: position,
rotation, gripper, and terminate. The loss is cross-entropy on those
seven tokens. Nothing about the machinery distinguishes them from the
caption tokens that the same model would emit on a web image. That
uniformity is the entire trick, and its limits drive Part 4.

## When discretizing actions stops being free

Uniform binning has a flaw that §8.3 only gestured at and that becomes
acute for high-frequency control. Robot actions are correlated and smooth
over time; a 50 Hz controller emits action vectors that barely change
from step to step. Binning each dimension independently throws that
structure away and spends a fixed token budget per step regardless of how
much actually happened. Worse, fine manipulation needs fine resolution
exactly where uniform bins are coarsest — near zero, where most action
deltas live.

FAST (arXiv:2501.09747) is the clean fix and a good note to end on,
because it shows tokenization is still an active research frontier rather
than a solved preprocessing step. Instead of binning raw actions, FAST
applies a discrete cosine transform to short chunks of the action
trajectory — the same compression idea used in JPEG — and tokenizes the
frequency coefficients. Smooth motion compresses to a handful of tokens;
the scheme spends sequence length in proportion to how much the action
actually varies, and it gives high-frequency continuous-control VLAs an
action vocabulary that is both compact and high-resolution. Concretely,
where uniform binning might spend dozens of tokens encoding a near-static
arm, the frequency representation collapses that stretch into a few
coefficients and reserves resolution for the moments where the motion
actually changes — the budget allocation a 50 Hz controller needs. The
same paper reports that this compression both shortens training sequences
and improves downstream control, a reminder that the right token format
can buy accuracy and efficiency at once rather than trading them off. The
takeaway is the section's thesis restated with a real artifact behind it:
*how* you turn actions into tokens sets a ceiling on how well a
transformer can control a robot, and moving that ceiling has been worth
its own papers.

Two strategies — discretize into symbols or project as vectors — applied
independently to states, actions, returns, and language, and overloaded
onto a pretrained vocabulary when you want web knowledge to come along
for free: that is the tokenization toolkit you carry into the rest of the
book. With it in hand, the only thing left in this chapter is to make the
bridge explicit — to trace exactly how a sequence model trained on tokens
becomes a foundation action model — which is what §8.5 does next.
