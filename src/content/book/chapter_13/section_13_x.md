---
chapter: 13
section: 13.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §13.3 (the flow-matching objective, its straight-line target, and the two-cluster toy previewed there); §13.1 for the multimodality requirement the toy is meant to expose; §3.3's fifty-line PyTorch training loop, which this exercise is a small variation on; a CPU is enough, no GPU required, and the whole thing runs in a couple of minutes
key_refs:
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Lipman, Y. et al. (2023). Flow Matching for Generative Modeling.
  - Liu, X. et al. (2022). Flow Straight and Fast, Learning to Generate and Transfer Data with Rectified Flow. arXiv:2209.03003.
  - Pertsch, K. et al. (2025). FAST, Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
  - Beyer, L. et al. (2024). PaliGemma, A Versatile 3B VLM for Transfer. arXiv:2407.07726.
---

# 13.x  Hands-on exercise + chapter references

Chapter 12's exercise cost you a GPU afternoon and a fine-tune. This one costs a
couple of minutes on a laptop CPU, and it is arguably the more important of the
two, because it makes the single claim the whole chapter rests on into something
you can watch happen on a plot. §13.3 argued that a squared-error regression loss
produces multimodal, splitting behavior, and it asked you to take the argument
partly on faith. The faith ends here. You are going to build the two-cluster toy
from §13.3, train a tiny velocity field on it with exactly the π0 training step,
and then draw the field over the plane. If the argument in §13.3 is right, you
will see the arrows fork: the left half of the noise cloud swept toward one
cluster, the right half toward the other, and a slack seam down the middle where
no single velocity can decide. If the argument is wrong, you will see the arrows
collapse everything to the midpoint, which is the averaging failure §13.1 warned
about, and you will know the section oversold its case. It does not, but you
should not believe that until the plot shows it.

The TOC names this exercise for Chapter 13 in one line: implement a minimal
π0-style training step on a 2-D toy dataset and visualize the learned flow field.
Two dimensions is not a toy chosen for convenience. It is the smallest setting
where the geometry of the flow is still visible to your eye, and every property
that matters at the scale of a two-arm robot folding laundry is already present
in it. Scale the point count up to an action chunk, condition the field on a
PaliGemma scene encoding (arXiv:2407.07726) instead of nothing, and the code you
write below is π0's action expert. That is not a metaphor. It is the same loss.

## What you need before you start

Almost nothing. PyTorch, matplotlib, and a CPU. No robot, no simulator, no
downloaded checkpoint, no scene conditioning of any kind. The point of stripping
all of that away is to isolate the one mechanism §13.3 hinges on, so that when it
works you know it was the flow-matching objective doing the work and not some
prop the rest of the system was quietly holding up. If you did the fifty-line
training loop in §3.3, this will feel like a variation on it, because it is one:
sample data, compute a loss, step the optimizer, repeat.

## Exercise 13.x.1 — Build the two-mode dataset

Make the data first, because the data is the point. You want a target
distribution with two clearly separated modes, so that "did the model learn both
or did it average them" is a question your eye can answer. Two Gaussian blobs, one
on the left and one on the right, will do.

```python
import torch

def sample_data(n):
    # two clusters: left mode at (-2, 0), right mode at (+2, 0)
    left  = torch.randn(n // 2, 2) * 0.3 + torch.tensor([-2.0, 0.0])
    right = torch.randn(n // 2, 2) * 0.3 + torch.tensor([ 2.0, 0.0])
    return torch.cat([left, right], dim=0)
```

Read the two clusters as two demonstrated action modes for the same scene, the
left fold and the right fold from §13.3. A mean-squared regressor asked to map
one input to this data has exactly one option, the midpoint at the origin, which
is a point the data never visits. That is the failure you are about to route
around. Plot a scatter of a few hundred samples before you go on, so the target
is fixed in your head: two clouds near $x = \pm 2$, nothing at the center.

## Exercise 13.x.2 — Write the π0 training step

Now the velocity field. Keep it small, a three-layer MLP is plenty, and give it
the two things the field is a function of: the current point $a_\tau$ and the
current time $\tau$. No scene input, because there is no scene. This is the only
place the toy simplifies π0's expert, and it simplifies it by deletion rather
than by cheating.

```python
import torch.nn as nn

class VelocityField(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),   # input: (a_x, a_y, tau)
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),              # output: velocity (v_x, v_y)
        )

    def forward(self, a, tau):
        return self.net(torch.cat([a, tau], dim=-1))
```

The training step is the loss from §13.3, transcribed to two dimensions with no
changes to its shape. Sample a batch of real points $a_1$, sample an equal batch
of Gaussian noise $a_0$, pick a random time $\tau$ per example, form the point on
the straight line between them, and regress the field onto the constant velocity
$a_1 - a_0$.

```python
model = VelocityField()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    a1  = sample_data(256)                     # clean targets
    a0  = torch.randn_like(a1)                 # noise
    tau = torch.rand(a1.shape[0], 1)           # random time in [0, 1]

    a_tau    = (1 - tau) * a0 + tau * a1        # point on the straight path
    target_v = a1 - a0                          # constant velocity along it

    pred_v = model(a_tau, tau)
    loss   = ((pred_v - target_v) ** 2).mean()

    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(step, loss.item())
```

Watch the printed loss. It will not go to zero, and that is the tell worth
pausing on. A field that perfectly fit every $(a_\tau, \tau)$ pair to its target
would have to output two different velocities at the same point in the seam
between basins, which is impossible for a function, so the loss floors out at the
irreducible variance the seam contributes. A loss that does drop near zero means
your two clusters overlap too much to be distinct modes, so push them farther
apart and retrain. The residual loss is not a bug. It is the mathematical
signature of the multimodality you asked for.

## Exercise 13.x.3 — Draw the field and find the seam

This is the payoff. Evaluate the trained field on a grid over the plane at a
fixed mid-flow time, say $\tau = 0.5$, and draw it as a quiver plot.

```python
import matplotlib.pyplot as plt

xs = torch.linspace(-4, 4, 20)
ys = torch.linspace(-4, 4, 20)
gx, gy = torch.meshgrid(xs, ys, indexing="xy")
grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
tau  = torch.full((grid.shape[0], 1), 0.5)

with torch.no_grad():
    v = model(grid, tau)

plt.quiver(grid[:, 0], grid[:, 1], v[:, 0], v[:, 1])
plt.scatter([-2, 2], [0, 0], c="red", s=80, zorder=3)
plt.show()
```

Look at the arrows near the vertical line $x = 0$. To the left of it they lean
left, toward the $(-2, 0)$ cluster; to the right they lean right, toward
$(+2, 0)$. Along the line itself, right where a naive regressor would have
planted every prediction, the arrows go short and undecided, some pointing
nowhere in particular, because that is the ambiguous region no single velocity
can resolve. The field learned to split the noise rather than average it, and it
did so from a plain squared-error loss with no special multimodal machinery
anywhere in the code. That splitting picture is §13.3 stated in ink.

## Exercise 13.x.4 — Sample, and count the modes

A field is a promise; sampling is the promise kept. Draw a hundred noise points,
integrate each one forward with the ten-step Euler solver from §13.3, and see
where they land.

```python
def sample(model, n=100, n_steps=10):
    a  = torch.randn(n, 2)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        tau = torch.full((n, 1), k * dt)
        with torch.no_grad():
            a = a + model(a, tau) * dt
    return a

pts = sample(model)
plt.scatter(pts[:, 0], pts[:, 1])
plt.show()
```

Two things to check, and they are the two things that carry over to the real
model. First, the samples should land in both clusters, in roughly the fifty-fifty
proportion the data had, and almost none should sit at the origin. That is honest
multimodal sampling from a regression loss, the claim §13.1 said was hard and
§13.3 said flow matching delivers. Second, drop the step count from ten to two and
resample. The clusters should still be recognizable, a little sloppier, which is
the rectified-flow property (Liu et al., arXiv:2209.03003) that lets π0 run its
expert in "ten or so" steps instead of a diffusion sampler's dozens. Now push it
to one step and watch it fall apart into a smear near the center, which is the
straight-path construction reminding you that few steps is not the same as one.
That failure at one step is the honest boundary of the trick, and it is worth
provoking on purpose so you know where it lives.

If you want the deciding-between-heads objective the TOC lists, run the token
baseline against this: bin each axis into, say, 32 uniform cells, train the same
MLP to classify which cell instead of regressing a velocity, and sample. On two
clean clusters the token head will do fine, which is exactly the point §13.1 made.
The token head fails on resolution and latency, not on this easy multimodality, so
a 2-D toy is the wrong instrument to show its weakness. Seeing the token head
succeed here is a useful reminder that the flow head's advantage is specific,
earned on high-frequency dexterous chunks, and not a blanket superiority you can
demonstrate on a scatter plot.

## Chapter 13 reading list

Cited across §13.1–§13.6, grouped by the job each reference does. Full entries
for everything in the book live in Appendix E.2; this is the chapter-local
subset.

### The model and its lineage

- Black, K., et al. (2024). "π0: A Vision-Language-Action Flow Model for General
  Robot Control." arXiv:2410.24164. The spine of the chapter. The architecture of
  §13.2, the flow-matching objective of §13.3, and the laundry-and-bussing results
  of §13.4 all come from here; the training step you wrote in this exercise is its
  loss at 2-D scale.
- Physical Intelligence (2025). "π0.5: A VLA with Open-World Generalization."
  arXiv:2504.16054. §13.4's evidence that the generalization bottleneck sat in the
  backbone data rather than the action head, and the co-training move that let the
  model walk into houses it had never seen.
- Physical Intelligence (2025). "π0.6: recovery and RL-from-experience." Technical
  report. §13.4's step past the imitation reliability ceiling, and §13.5's example
  of an RL loop most labs cannot afford to run.
- Physical Intelligence (2026). "π0.7: steerable generalist." Technical report.
  §13.4's turn from a capable generalist into a controllable one, and §13.5's
  caution that steerable is not the same as safe.

### The generative machinery underneath

- Lipman, Y., et al. (2023). "Flow Matching for Generative Modeling." The
  objective §13.3 built on: learn a velocity field with a regression loss, no
  simulation of the flow during training.
- Liu, X., et al. (2022). "Flow Straight and Fast: Learning to Generate and
  Transfer Data with Rectified Flow." arXiv:2209.03003. The straight-path idea
  that makes few-step sampling accurate, and the reason Exercise 13.x.4 still
  works at two steps. It is what buys π0 its control-rate latency.

### The token head it replaced

- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for
  Vision-Language-Action Models." arXiv:2501.09747. §13.1's evidence that clever
  tokenization compresses the resolution and sequence-length costs but leaves the
  serial-decode latency untouched, which is the gap π0 was built to close.
- Beyer, L., et al. (2024). "PaliGemma: A Versatile 3B VLM for Transfer."
  arXiv:2407.07726. §13.2's backbone: the web-pretrained vision-language model π0
  keeps intact and pairs with the flow-matching expert.

### The thread that reaches Chapter 18

- Assran, M., et al. (2025). "V-JEPA 2 and V-JEPA 2-AC." arXiv:2506.09985. §13.5's
  candidate answer to where the next 10,000 hours of training data come from when
  teleoperation runs out; developed in full in Chapter 18.

## Chapter summary

Chapter 13 took the action-head fork that Chapter 12 left open and followed the
continuous branch all the way down, using π0 as the one model taken apart from
every side. You can now say precisely when a discrete action head stops paying its
way: a resolution ceiling from uniform binning, a sequence-length blowup on long
high-frequency chunks, and a serial-decode latency that is structural to
autoregression, with FAST fixing the first two and nothing fixing the third short
of leaving tokens behind. You can draw π0's architecture from memory, a
web-pretrained backbone for scene and language beside a flow-matching action
expert for motion, sharing one attention, with the scene encoded once outside the
sampling loop, and you can point to which half holds the transferable knowledge
and which holds the embodiment-specific motor skill. You can explain how a plain
regression loss yields multimodal actions, and after this exercise you can do more
than explain it, because you plotted the splitting field and counted the samples
landing in both modes rather than at their average. And you can read the
π0→π0.5→π0.6→π0.7 releases as one argument, separating the durable contribution,
the flow head that survived all four, from the churn of data and training stacked
around it. §14 picks up the ceiling §13.5 named: π0 reasons and acts in a single
forward pass, so it cannot think longer on a harder problem, and the dual-system
designs of Helix and GR00T N1 are the field's answer to exactly that limit.
