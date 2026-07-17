---
chapter: 11
section: 11.3
title: "Action tokenization: a small idea with large consequences"
target_words: 2000
status: draft
prereqs: §11.2 (RT-1 discretizes actions and predicts them as tokens), §8.1 (the transformer and next-token prediction), §3.2 (why cross-entropy over bins is a distribution, not a point estimate). Helpful, §10.5 on action-head choices, since tokenization is the discrete alternative to the continuous heads discussed there.
key_refs:
  - Brohan, A. et al. (2022). RT-1, Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Brohan, A. et al. (2023). RT-2, Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Pertsch, K. et al. (2025). FAST, Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
---

# 11.3  Action tokenization: a small idea with large consequences

Section 11.2 slipped one decision past you on purpose. RT-1 does not output a
robot action; it outputs a short string of tokens, one per action dimension,
each chosen from 256 discrete bins. A transformer that was built to predict
the next word now predicts the next slice of a gripper command, and the training
objective is the same cross-entropy loss a language model uses. The move looks
like an implementation detail. It is closer to a fork in the road, and most of
the models in Part 4 are standing on one side of it.

Here is the tension. A robot action is continuous. The commanded velocity of a
joint is a real number, and so is the change in gripper aperture, and so is
every rotation you might send to an arm. Continuous quantities want continuous
methods: regress the number, minimize squared error, done. Chapter 3 spent a
section on exactly that loss. So why would anyone take a smooth, real-valued
target and deliberately chop it into 256 buckets, throwing away all the
precision that lives between bin edges?

## The case for turning actions into tokens

The answer starts with what a transformer is good at, and it is not regression.
A decoder-only transformer trained with cross-entropy is a machine for modeling
distributions over discrete symbols. Feed it text and it learns
$p(\text{next word} \mid \text{context})$. If you want to reuse that machine for
control, and reusing it is the entire premise of the VLA program, the cheapest
path is to make actions look like words. Discretize each action dimension into
bins, treat each bin as a symbol, and control becomes next-token prediction
over an action vocabulary. Nothing about the architecture has to change. The
optimizer, the attention stack, the sampling code, even the pretrained weights
carry over untouched. That reuse is the whole point, and it is why RT-2
(arXiv:2307.15818) could take a vision-language model trained on web text and
images and teach it to drive a robot without redesigning the network: the
actions were folded into the same token stream as everything else.

There is a second reason, and it matters more than the engineering
convenience. Regression to a single number commits the policy to one answer.
Averaged over a dataset where the same observation sometimes precedes a left
turn and sometimes a right, least-squares regression splits the difference and
drives straight into the obstacle between them. This is the multimodality
problem from §10.4, and discretization sidesteps it for free. A distribution
over 256 bins can put mass on the left-turn bin *and* the right-turn bin and
leave the middle empty. Sample from it and you get one or the other, never the
disastrous average. Cross-entropy over bins is a full distribution, not a point
estimate, so the policy can represent "either of these, but not the thing
between them," which is precisely the shape real manipulation data has.

## How RT-1 actually does it

RT-1's scheme is the one everyone learned first, so it is worth stating in
mechanical detail. Take one action dimension, say the commanded x-translation
of the end effector. It lives in some known range, clip it to $[-1, 1]$ after
normalization. Slice that range into 256 equal-width bins. Any real value now
falls into exactly one bin, and you replace the number with that bin's integer
index, a symbol between 0 and 255. Do this independently for each of the action
dimensions RT-1 controls, arm translation and rotation and gripper and the
discrete mode switch, and one continuous action vector becomes a fixed-length
sequence of integers. The transformer predicts them one at a time, left to
right, exactly as it would predict a word.

Recovering the action at run time is the reverse, and it is where the cost hides.
The model hands you a bin index; you map it back to a real number, usually the
center of the bin. That center is off from the true value by up to half a bin
width, always. With 256 bins over a unit range, half a bin is about
$1/512 \approx 0.002$ in normalized units. For coarse pick-and-place at a few
hertz, that quantization error disappears under the noise of the robot itself
and nobody notices. Uniform binning is simple, it round-trips predictably, and
for RT-1's task distribution it was good enough. The hands-on exercise at the
end of this chapter has you build exactly this tokenizer and measure its
reconstruction error on a real teleop dataset, so the number stops being
abstract.

A worked micro-example makes the round trip concrete. Suppose the normalized
x-command is $0.37$. Map $[-1, 1]$ onto bin indices $0$ through $255$: the index
is $\lfloor (0.37 + 1)/2 \times 256 \rfloor = \lfloor 175.4 \rfloor = 175$. The
transformer emits the token `175`. To decode, take the bin center,
$(175 + 0.5)/256 \times 2 - 1 \approx 0.371$. You asked for $0.37$ and got back
$0.371$. The gap is the quantization error, and it is bounded by half a bin no
matter what value you started with.

## Where uniform bins fall apart

Now change the task. Instead of moving a can across a table at 3 Hz, you are
folding a cloth or threading a connector at 50 Hz, and the robot has many joints
moving fast and together. Two things break at once.

The first is resolution. High-frequency control means each consecutive action
is only slightly different from the last, tiny increments arriving fifty times
a second. Slice the range into 256 uniform bins and those tiny increments all
fall into the same bin or two adjacent ones. The tokenizer can no longer tell
consecutive actions apart; it has quantized away the very signal the dexterous
task depends on. You could add more bins, but the vocabulary grows and the model
has to learn a much larger classification, and you are fighting the tool instead
of using it.

The second problem is length, and it compounds the first. A dexterous episode
at 50 Hz has thousands of timesteps, each with many action dimensions. Tokenize
per dimension per timestep and the sequence the transformer has to predict
becomes enormous. Attention cost grows with sequence length, training slows,
and the autoregressive decode at run time, one token at a time, blows the
latency budget a real robot lives under. The FAST paper (Pertsch et al., 2025,
arXiv:2501.09747) reports that naïve per-dimension, per-timestep binning does
not merely degrade on high-frequency dexterous data; on some tasks it fails
completely, producing policies that never learn the skill at all. The small idea
that carried RT-1 does not survive contact with harder robots.

## Compression instead of binning: FAST

The fix in FAST is to stop treating each number in isolation and compress the
action sequence first, the way an audio or image codec compresses a signal
before storing it. A chunk of consecutive actions is highly redundant, since
smooth motion means neighboring values are correlated, and redundancy is exactly
what compression eats. FAST runs the action chunk through a discrete cosine
transform, the same DCT that sits at the heart of JPEG, which re-expresses the
motion as a handful of frequency coefficients. Most of those coefficients are
near zero for smooth trajectories, so you keep the few that carry the energy,
quantize those, and turn *them* into tokens. Slow, smooth motion collapses to a
short token string; only genuinely fast or complex motion needs many tokens.

The payoff is that the token count now tracks the information content of the
action, not the raw sample rate. High-frequency data that drowned uniform
binning becomes tractable, sequences get shorter, and training speeds up. The
FAST authors ship a ready-made tokenizer, FAST+, fit on roughly a million real
robot trajectories, meant to be dropped in as a black box across different robots
and control rates. Paired with the π0 model you will meet in Chapter 13, FAST
let an autoregressive VLA scale to thousands of hours of data and match the
performance of the diffusion-based action heads from Chapter 10, while cutting
training time substantially. That comparison is the one to hold onto: tokenized
autoregressive control and continuous diffusion heads are the two live options,
and §10.5 and Chapter 13 are where the trade-off gets settled.

## Why this small choice ripples so far

Step back and the stakes come into focus. The tokenizer is a contract between
the continuous world the robot lives in and the discrete world the transformer
thinks in, and everything downstream inherits its terms. Pick uniform bins and
you get RT-1's simplicity and RT-2's free reuse of a web-pretrained VLM, at the
price of a resolution ceiling that caps how dexterous the policy can get. Pick
a compression-based scheme like FAST and you buy back the dexterity, at the cost
of a decode step that is no longer a trivial table lookup. Neither is wrong;
they answer to different robots. What you cannot do is ignore the choice, because
a policy can only ever be as precise as the tokens it is allowed to emit. Get
the tokenizer wrong and the fanciest backbone in the world predicts the right
bin for an action the bin cannot express.

That is the sense in which a discretization scheme, three lines of code in its
simplest form, sets a ceiling on an entire model. The next section pulls the
lens back from this one design choice to the model that made it famous, and asks
which of RT-1's headline results came from the transformer and the tokens, and
which came from the 130,000 demonstrations feeding them.
