---
chapter: 11
section: 11.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §11.1–§11.6; Python with NumPy installed; a firm grasp of action tokenization from §11.3 (per-dimension uniform binning, the resolution floor a bin width sets, cross-entropy over bins as the training loss), the RT-1 recipe from §11.2, and the diversity-versus-volume account of scale from §11.5; any small teleop dataset of continuous action chunks, or the synthetic generator supplied below; about two hours, most of it writing and testing a fifty-line tokenizer
key_refs:
  - Brohan, A. et al. (2022). RT-1 — Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Brohan, A. et al. (2023). RT-2 — Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Pertsch, K. et al. (2025). FAST — Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
  - Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
  - Jang, E. et al. (2022). BC-Z — Zero-Shot Task Generalization with Robotic Imitation Learning. CoRL 2021 / PMLR 164.
---

# 11.x  Hands-on exercise + chapter references

Section 11.3 made a claim you took mostly on trust: chopping a continuous
action into 256 bins throws away precision, but the amount it throws away is
small enough that a language model's cross-entropy loss can drive a real arm.
The way to stop trusting and start knowing is to build the tokenizer, run a
dataset through it, and read the reconstruction error off the numbers you
produced. That is the exercise the TOC names for this chapter, and it is worth
doing even though the tokenizer is fifty lines of NumPy, because the fifty lines
force you to confront every decision RT-1 made quietly: how many bins, uniform
or not, per-dimension range or global, and what error you are willing to eat at
the bin edges. The supporting drills push on the edges §11.3 and §11.5 argued
about: where the resolution floor actually bites, and whether more bins is the
free lunch it looks like.

Budget about two hours. None of it is GPU time; this chapter's exercise is
almost entirely about the data structure between a float and a token, which is
where most of the practical pain in early VLAs actually lived.

## The data you will tokenize

If you have a teleop dataset of your own, use it. Otherwise the generator below
stands in for one: a batch of action chunks, each chunk a short trajectory of
7-DoF end-effector deltas (three for translation, three for rotation, one for
the gripper), with the kind of range mismatch across dimensions that real
teleop always has. Translation deltas are small and centered near zero;
rotations swing wider; the gripper is nearly binary. That mismatch is the whole
reason per-dimension binning matters, and a synthetic set that ignored it would
hide the lesson.

```python
import numpy as np
rng = np.random.default_rng(0)

def fake_teleop(n=4000, horizon=8):
    # 7 dims: dx, dy, dz (small), droll, dpitch, dyaw (wider), gripper (~binary)
    trans = rng.normal(0, 0.02, size=(n, horizon, 3))
    rot   = rng.normal(0, 0.15, size=(n, horizon, 3))
    grip  = rng.choice([-1.0, 1.0], size=(n, horizon, 1))
    grip += rng.normal(0, 0.05, size=grip.shape)   # a little slop
    return np.concatenate([trans, rot, grip], axis=-1)  # (n, horizon, 7)

actions = fake_teleop()
```

Everything that follows runs against this array or your own dataset of the same
shape.

## Exercise 11.x.1 — Build the tokenizer and measure its round-trip error

This is the headline drill. Write a tokenizer that discretizes each action
dimension into $B$ uniform bins, encodes a chunk to integer token IDs, decodes
those IDs back to floats, and reports how much the round trip cost. RT-1 used
$B = 256$ per dimension; start there.

The one decision that matters more than any other is where the bin edges go.
Fit them per dimension, from the data, using a robust range rather than the raw
min and max. A single outlier teleop frame will blow your min-max range wide,
making every bin coarse and wasting most of the vocabulary on values that never
occur. Clip to a percentile range instead, say the 1st and 99th, and let the
tail values saturate at the end bins. That single choice is the difference
between a tokenizer that works and one that quietly ruins your gripper
precision.

```python
class UniformActionTokenizer:
    def __init__(self, data, n_bins=256, lo_pct=1, hi_pct=99):
        flat = data.reshape(-1, data.shape[-1])          # (N, D)
        self.lo = np.percentile(flat, lo_pct, axis=0)    # per-dim edges
        self.hi = np.percentile(flat, hi_pct, axis=0)
        self.n_bins = n_bins

    def encode(self, x):                                 # floats -> token IDs
        z = (x - self.lo) / (self.hi - self.lo)          # to [0, 1]
        z = np.clip(z, 0, 1 - 1e-9)
        return (z * self.n_bins).astype(np.int64)        # (..., D)

    def decode(self, ids):                               # token IDs -> floats
        z = (ids + 0.5) / self.n_bins                    # bin center
        return z * (self.hi - self.lo) + self.lo

tok = UniformActionTokenizer(actions, n_bins=256)
ids  = tok.encode(actions)
recon = tok.decode(ids)
```

Now the measurement. Reconstruction error is not one number; report it per
dimension, because the whole point of §11.3 is that a fixed bin count means
different physical precision on a 2-centimeter translation range and a 0.6-radian
rotation range. Compute per-dimension mean absolute error, and convert it to
physical units so it means something: millimeters for translation, milliradians
for rotation.

```python
mae = np.abs(recon - actions).reshape(-1, 7).mean(axis=0)
print("per-dim MAE:", mae)
print("translation MAE (mm):", mae[:3] * 1000)
print("rotation MAE (mrad):", mae[3:6] * 1000)
```

Read what comes out. Each dimension's MAE should sit near a quarter of its bin
width, which is the expected error of rounding a uniform variable to the nearest
of $B$ centers. The number you care about is whether that error clears your
robot's tolerance. A pick-and-place task forgives a millimeter of slop at the
gripper; threading a connector does not. Write one sentence stating, in physical
units, the finest task your 256-bin tokenizer could support. That sentence is
the resolution floor of §11.3, and you now have it as a measured quantity rather
than an assertion.

## Exercise 11.x.2 — Sweep the bin count and watch precision buy itself

The obvious fix for "not precise enough" is more bins, and the obvious question
is why RT-1 stopped at 256. Turn the knob and find out. Re-tokenize the same
dataset at $B \in \{8, 16, 32, 64, 128, 256, 512, 1024\}$ and plot per-dimension
MAE against $B$ on a log-log axis.

The curve is a clean power law: error falls roughly as $1/B$, because halving
the bin width halves the rounding error. So far more bins look free. They are
not, and the cost lives in a place this exercise cannot show you directly but
you should name anyway. Each bin is a distinct token the policy must learn to
predict, and the training signal for a bin is the number of demonstrations that
land in it. Push $B$ to 1024 and most bins are visited by a handful of chunks or
none, so the cross-entropy classifier of §11.3 has almost nothing to learn from
per bin. The reconstruction error keeps dropping while the *predictability* of
the token quietly collapses. RT-1's 256 is a compromise between a floor that is
fine enough for tabletop manipulation and a vocabulary small enough that every
bin gets seen often. Mark on your plot the point where the round-trip error
first clears your task tolerance; that, not the leftmost achievable error, is
the bin count you would actually ship.

## Exercise 11.x.3 — Break it with a distribution shift

§11.5 argued that a model living below the diversity floor is memorizing, and a
tokenizer fit to a narrow slice of behavior has the same failure mode in
miniature. Make it visible. Fit the tokenizer on one regime of the data, then
run a different regime through it.

Split the synthetic set (or your teleop set) so that the tokenizer sees only
small, slow motions during fitting: keep chunks whose translation magnitude sits
below the median. Fit `lo` and `hi` on that half. Now encode and decode the
*fast* half, the chunks you held out, and recompute per-dimension MAE.

```python
speed = np.linalg.norm(actions[..., :3], axis=-1).mean(axis=1)   # per-chunk
slow, fast = actions[speed < np.median(speed)], actions[speed >= np.median(speed)]

tok_slow = UniformActionTokenizer(slow, n_bins=256)
mae_in   = np.abs(tok_slow.decode(tok_slow.encode(slow)) - slow).mean()
mae_out  = np.abs(tok_slow.decode(tok_slow.encode(fast)) - fast).mean()
print("in-distribution MAE:", mae_in, " shifted MAE:", mae_out)
```

The fast chunks saturate at the end bins the slow-fitted range never covered, so
their large motions all decode to the same clipped value and the error jumps.
This is the tokenizer's version of the compounding-error story from §6.3: fit
your discretization to a narrow behavior distribution and it will silently clip
anything outside it, which is exactly what happens to a VLA asked for a motion
its demonstrations never contained. The lesson RT-1's authors internalized, and
the reason §11.5 kept insisting diversity beats volume, is that the range you
fit is a promise about the behaviors you expect, and the robot keeps that
promise whether or not you meant it to.

## Exercise 11.x.4 — Compare against a smarter tokenizer, on paper

You have now felt the two weaknesses of uniform per-dimension binning: it spends
bins evenly even where the data is not, and its physical precision is hostage to
the widest-ranging dimension in the chunk. Read the FAST paper (Pertsch et al.,
2025, arXiv:2501.09747) with those two weaknesses in front of you. FAST replaces
uniform binning with a discrete cosine transform of the action chunk followed by
byte-pair encoding of the coefficients, which concentrates tokens on the parts
of the signal that actually carry information and lets a chunk of many timesteps
collapse into far fewer tokens.

No code here; the drill is analytical. In two or three sentences, explain which
of your measured failures FAST fixes and which it does not. It attacks the wasted
vocabulary of Exercise 11.x.2, since the DCT-plus-BPE stage puts token capacity
where the motion has structure instead of spreading it uniformly. It does
nothing about the promise-of-range problem from Exercise 11.x.3, because any
learned tokenizer still fits its scheme to the behaviors it saw. Getting that
distinction right is the payoff of the whole exercise: you can now read a new
action-tokenization paper and predict, before the experiments, which of your
own measured pains it will and will not relieve. That skill is exactly what
Chapter 12 asks of you when the model names start arriving three per section.

## Chapter 11 reading list

The works below are cited in §11.1–§11.6, grouped by role. Full bibliographic
entries for everything cited in the book live in Appendix E.2; this is the
chapter-local subset.

### Multimodal pretraining, the inherited half of the recipe

- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural
  Language Supervision" (CLIP). *ICML 2021*. The aligned image-text space §11.1
  builds the whole VLA recipe on top of.
- Jia, C., et al. (2021). "Scaling Up Visual and Vision-Language Representation
  Learning With Noisy Text Supervision" (ALIGN). *ICML 2021*. §11.1's evidence
  that the CLIP result was about scale and noisy web data, not one lab's trick.
- Driess, D., et al. (2023). "PaLM-E: An Embodied Multimodal Language Model."
  arXiv:2303.03378. §11.1's bridge from a frozen aligned encoder to a policy that
  reasons in the same space, the idea Chapter 12 scales.

### Language-conditioned imitation

- Jang, E., et al. (2022). "BC-Z: Zero-Shot Task Generalization with Robotic
  Imitation Learning." *CoRL 2021 / PMLR 164*. §11.2's first demonstration that a
  language-conditioned imitation policy generalizes to unseen task phrasings.
- Brohan, A., et al. (2022). "RT-1: Robotics Transformer for Real-World Control
  at Scale." arXiv:2212.06817. The center of gravity of this chapter: the
  tokenized action space Exercise 11.x.1 rebuilds, the Everyday Robots dataset of
  §11.5, and the scaling framing of §11.4.

### Action tokenization and its successors

- Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web
  Knowledge to Robotic Control." arXiv:2307.15818. §11.3's proof that RT-1's
  action tokens fold cleanly into a web-pretrained VLM's token stream; Chapter 12
  in full.
- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for
  Vision-Language-Action Models." arXiv:2501.09747. The DCT-plus-BPE tokenizer
  Exercise 11.x.4 reads against your own measurements; the answer to uniform
  binning's wasted vocabulary.

### The data that made scale a real question

- Padalkar, A., et al. (2023). "Open X-Embodiment: Robotic Learning Datasets and
  RT-X Models." arXiv:2310.08864. §11.5's evidence on where diversity starts to
  pay off across many robots; the dataset Chapter 12 treats in detail.
- Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action
  Model." arXiv:2406.09246. §11.5's open checkpoint standing on the RT-1 recipe;
  the model Chapter 12 opens up component by component.

## Chapter summary

Chapter 11 assembled the recipe every later chapter reuses, and this closing
exercise made its most quietly consequential ingredient concrete. You can now
trace the path from CLIP's aligned image-text space to a policy that follows a
typed instruction, and say why the alignment had to be paid for on web data
rather than robot data: the demonstrations are too scarce and too expensive to
teach a machine what a mug is, so they are spent on the last mile from aligned
perception to motor command instead. You can build the action tokenizer at the
heart of RT-1, discretizing each dimension into uniform bins, and you have
measured its round-trip error in physical units rather than accepting §11.3's
claim on faith, which means you can state the finest task a 256-bin scheme
supports and why more bins stop helping before the reconstruction error says
they should. You can explain why language conditioning unlocked generalization
that plain behavior cloning could not, and you can read the diversity-versus-
volume story of §11.5 as a statement about which axis of a dataset actually buys
robustness. Carry the last skill into Part 4: given a new tokenizer, a new
backbone, or a new action head, you can now predict which of the pains you
measured here it relieves before you read anyone's experiments. Chapter 12 takes
the RT-1 recipe and asks what happens when the frozen half stops being a vision
encoder and becomes a full web-pretrained vision-language model, starting with
RT-2.
