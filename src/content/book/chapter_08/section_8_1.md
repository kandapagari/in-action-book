---
chapter: 8
section: 8.1
title: "The transformer in two pages, for control"
target_words: 2000
status: draft
prereqs: §3.1 (vectors, matrices, the chain rule); §3.3 (a PyTorch training loop); §5.1 (states, actions, policies, trajectories); §7.2 (why high-variance gradients hurt). Appendix C has a longer transformer refresher if this section moves too fast.
key_refs:
  - Vaswani et al. (2017). Attention is all you need. NeurIPS.
  - Radford et al. (2018). Improving language understanding by generative pre-training (GPT). OpenAI tech report.
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Brohan et al. (2022). RT-1: robotics transformer for real-world control at scale. arXiv:2212.06817.
---

# 8.1  The transformer in two pages, for control

Every architecture in Parts 3 and 4 of this book is a transformer or a
close relative. RT-1, RT-2, OpenVLA, Octo, π0 — strip away the robot
plumbing and what remains is a stack of attention layers reading a
sequence of tokens and predicting the next one. If you understand that
stack, the rest of the book is mostly a matter of asking what the
tokens are. This section builds the transformer from the one operation
that matters for control, attention, and skips the parts you can read
in Appendix C. The promise of "two pages" is a slight exaggeration, but
the load-bearing idea really does fit in a paragraph.

Here is that paragraph. A transformer takes a sequence of vectors,
called tokens, and produces a new sequence of the same length, where
each output vector is a weighted average of all the input vectors. The
weights are computed from the inputs themselves: token *i* looks at
token *j*, decides how relevant *j* is to it, and mixes *j* in
proportionally. Stack a few of these mixing layers with ordinary
neural-network layers in between, and you have a model that can route
information between any two positions in the sequence in a single step.
That is the whole machine. For control, the sequence is a trajectory —
states, actions, maybe rewards, maybe a language instruction — and
"predict the next token" becomes "predict the next action."

## Attention is a soft lookup

The operation underneath everything is called scaled dot-product
attention, introduced in its modern form by Vaswani et al. (2017). It
is easiest to read as a differentiable dictionary lookup.

Each token produces three vectors by multiplying its embedding by three
learned matrices: a *query* $q$, a *key* $k$, and a *value* $v$. Think
of a key as a label on a piece of information and a value as the
information itself; a query is what a token is currently looking for.
To compute the output for token *i*, take its query $q_i$, compare it
against every key $k_j$ in the sequence by dot product, turn those
scores into weights with a softmax, and use the weights to average the
values:

$$
\text{attn}(q_i) = \sum_j \underbrace{\text{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)}_{\text{weight }w_{ij}} \, v_j .
$$

The $\sqrt{d}$ in the denominator (with $d$ the dimension of the key
vectors) keeps the dot products from growing large enough to saturate
the softmax — a small numerical detail with a large effect on whether
the thing trains at all, which is exactly the kind of failure §3.5
warned about. Everything else is a dot product and a weighted sum.

The single property worth memorizing: an output token is a content-
addressed average. Position *i* does not pull from position *i-1*
because of where it sits; it pulls from whichever positions have keys
that match its query, wherever they are in the sequence. A gripper-
contact event 40 steps ago and the instruction token at the very front
are the same distance away as the previous frame — one attention hop.
This is the structural reason transformers handle long-range dependence
better than the recurrent networks they replaced, where information
from 40 steps back had to survive 40 sequential overwrites.

Written as code, the entire operation for a whole sequence is five
lines, and it is worth seeing how little there is:

```python
import torch, torch.nn.functional as F

def attention(x, Wq, Wk, Wv, causal=True):
    q, k, v = x @ Wq, x @ Wk, x @ Wv     # each (L, d)
    scores = q @ k.T / k.shape[-1]**0.5  # (L, L) all pairs
    if causal:                           # forbid looking ahead
        L = x.shape[0]
        scores = scores.masked_fill(
            torch.triu(torch.ones(L, L), 1).bool(), float("-inf"))
    w = F.softmax(scores, dim=-1)        # (L, L) weights per row
    return w @ v                         # (L, d) mixed values
```

The `scores` matrix is the whole story: row *i*, column *j* holds how
much token *i* attends to token *j*. The mask zeroes out the upper
triangle so the past cannot read the future, the softmax turns each row
into weights that sum to one, and the final product is the weighted
average. There is no recurrence and no convolution here — just two
matrix multiplies bracketing a softmax.

In practice a layer runs several attention operations in parallel, each
with its own query/key/value matrices — *multi-head* attention. One
head might track which object the instruction names while another
tracks the gripper's state; the heads are concatenated and mixed by a
final linear layer. The mechanism is identical; there are just several
copies looking for different things.

## A transformer is attention plus the boring parts

A full transformer block wraps that attention operation in machinery
that, for our purposes, is boring and you can treat as fixed.
After attention mixes information across positions, a small two-layer
MLP processes each position independently. Both sub-layers are wrapped
in a residual connection (add the input back to the output) and layer
normalization, the same residual idea that let the deep networks of
§3.1 train at all. Stack $N$ such blocks — 6 in the original paper,
12 to 32 in the models we care about — and that is the network.

Two additions make it usable for sequences.

*Positional encoding.* Attention as written is permutation-invariant:
shuffle the tokens and every output is shuffled the same way but
otherwise unchanged, because a weighted sum does not care about order.
For language or a trajectory, order is the whole point — "place the cup
on the plate" is not "place the plate on the cup," and a state-then-
action ordering means something different from the reverse. So each
token's embedding gets a position signal added to it before the first
layer, either a fixed sinusoidal pattern or a learned per-slot vector.
Now "which position am I" is part of the content that queries and keys
are computed from.

*Causal masking.* A model that predicts the next token must not be
allowed to look at it. During training we feed the whole trajectory at
once for efficiency, but we forbid token *i* from attending to any
token *j > i* by setting those attention weights to zero before the
softmax. This is the difference between a *decoder* (causal, left-to-
right, the GPT family of Radford et al. 2018, and the right choice for
generating actions one at a time) and an *encoder* (bidirectional, used
when the whole input is available up front, such as a fixed image).
Control policies are almost always causal: at deployment time the robot
has its past but not its future.

## Why this is a natural fit for control

Reinforcement learning, as set up in Chapter 5, is the study of
trajectories: sequences $s_0, a_0, r_0, s_1, a_1, r_1, \dots$ A policy
maps history to the next action. That is, almost verbatim, the
sequence-modeling problem a decoder transformer was built for —
condition on the tokens so far, predict the next one. The reframing,
made explicit by the Decision Transformer of Chen et al. (arXiv:
2106.01345) and developed in §8.2, is to stop treating control as a
search for an optimal value function and start treating it as
autoregressive prediction over trajectory tokens. The Bellman backup of
§5.2 disappears; in its place is the same cross-entropy or regression
loss we have been minimizing since §3.4.

Three properties of attention pay off specifically for robots. First,
variable and heterogeneous inputs: a token is just a vector, so an
image patch, a joint-angle reading, a return value, and a word from an
instruction can all be tokens in one sequence, and attention learns how
much each should influence the action. This is precisely how RT-1
(Brohan et al., arXiv:2212.06817) feeds camera images and a language
command into one model — we walk through its tokenizer in §8.4 and §11.3.
Second, long-horizon credit: the content-addressed average means an
action can attend directly to the instruction issued at $t=0$ or to the
moment a grasp first succeeded, without that signal decaying through
intervening steps. Third, scale: the architecture is almost entirely
matrix multiplications with no sequential recurrence inside a forward
pass, so it saturates GPUs, and — the empirical fact that organizes all
of Part 4 — its performance keeps improving as data and parameters grow.

## The cost, stated plainly

Attention's strength is also its bill. Because every token attends to
every other token, an $L$-token sequence requires on the order of
$L^2$ dot products. Double the trajectory length and you quadruple the
compute and memory. For language with thousands of tokens this is the
central engineering headache; for control it bounds how much history
and how many image patches you can afford per step, and it collides
head-on with the real-time deadline of a control loop — a 10 Hz policy
has 100 milliseconds to see, think, and command, full stop. That
tension drives several later design choices: the action *chunking* of
Chapter 10, the dual-system split of Chapter 14 that runs a heavy
transformer slowly and a light one fast, and the state-space-model
alternative (RoboMamba, arXiv:2406.04339) we reach in §8.5, whose
selling point is precisely that it scales linearly rather than
quadratically with sequence length.

You now have the one operation — a content-addressed, softmax-weighted
average of value vectors — and the scaffolding around it: residual
blocks, positional encodings, causal masks. That is enough to read
every architecture diagram in the rest of the book; when a paper shows
a stack of grey boxes labeled "transformer," you know what is inside
each one and why an action can come out the top. The next section spends
that understanding on the first concrete instance, recasting offline RL
as next-token prediction with the Decision Transformer.
