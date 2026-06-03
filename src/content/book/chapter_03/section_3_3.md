---
chapter: 3
section: 3.3
title: A 50-line PyTorch training loop, annotated
target_words: 2000
status: draft
prereqs: §3.1 (vectors, gradients, chain rule), §3.2 (cross-entropy, KL divergence); a working Python/PyTorch installation
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 3.3  A 50-line PyTorch training loop, annotated

The last two sections established the math: the chain rule computes gradients,
cross-entropy is the operational form of KL divergence for discrete targets,
and minimizing the expected loss over a dataset of demonstrations is what behavior
cloning actually does. This section makes all of that executable. We will build
the smallest possible policy network, write its training loop, and annotate every
line that does non-obvious work. The goal is not a toy you should ship — it is a
template you should be able to read in any VLA codebase and recognize.

The policy here is a three-layer MLP with a separate linear head per action
dimension, producing 256-bin logits exactly like the OpenVLA action head
described in §2.3 and §3.2. Inputs are 64-dimensional observation embeddings;
outputs are 7 per-joint discrete distributions over action bins. This is not
a vision model, not a transformer, and not a sequence model. It is simple enough
to fit on one screen and complex enough to exercise every part of the gradient
path you need to understand.

## The model

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SmallPolicy(nn.Module):
    def __init__(self, obs_dim=64, hidden=256, n_bins=256, n_joints=7):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, n_bins) for _ in range(n_joints)]
        )

    def forward(self, obs):           # obs: (B, obs_dim)
        h = self.trunk(obs)           # h: (B, hidden)
        return [head(h) for head in self.heads]   # list of 7 × (B, n_bins)
```

Two architectural decisions are worth pausing on. First, the trunk is shared
across all 7 joints. Every action dimension sees the same representation, which
is correct: the decision about how far to move joint 4 depends on the same
observation state as the decision about how far to move joint 6. Second, the
heads are separate linear layers, not a single linear layer with 7 × 256 outputs.
The difference is purely organizational — they are mathematically equivalent —
but the `ModuleList` makes it easy to later add per-head losses, per-head
regularizers, or per-head frozen weights.

The output is a list of logits, not probabilities. PyTorch's `CrossEntropyLoss`
applies the softmax internally, which is numerically more stable than calling
`F.softmax` first and then `F.nll_loss`. Do not apply softmax before the loss
function; you will get the right answer but with worse numerical conditioning.

## The training loop

```python
def train(n_epochs=50, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy    = SmallPolicy().to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=1e-4, weight_decay=1e-5
    )
    loss_fn = nn.CrossEntropyLoss()

    # synthetic dataset: 4096 (obs, action-bin-vector) pairs
    obs  = torch.randn(4096, 64)
    acts = torch.randint(0, 256, (4096, 7))   # one bin index per joint
    loader = DataLoader(
        TensorDataset(obs, acts), batch_size=64, shuffle=True
    )

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for obs_b, acts_b in loader:
            obs_b  = obs_b.to(device)
            acts_b = acts_b.to(device)

            optimizer.zero_grad()                        # (1)

            logits = policy(obs_b)                       # (2)

            loss = sum(
                loss_fn(logits[j], acts_b[:, j])
                for j in range(7)
            ) / 7.0                                      # (3)

            loss.backward()                              # (4)
            torch.nn.utils.clip_grad_norm_(              # (5)
                policy.parameters(), max_norm=1.0
            )
            optimizer.step()                             # (6)

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(loader)
            print(f"epoch {epoch + 1:3d}  loss {avg:.4f}")

    return policy
```

Six lines are labelled. Here is what each one is actually doing.

### (1) `optimizer.zero_grad()`

PyTorch accumulates gradients by default. Every call to `loss.backward()` adds
to the `.grad` attribute of each parameter rather than replacing it. If you
forget `zero_grad()`, you get gradient accumulation across batches, which gives
the illusion of a larger effective batch size but is almost never what you want
inside a standard training loop. Call `zero_grad()` before every forward pass.
The exception — gradient accumulation as a deliberate technique for simulating
large-batch training on limited memory — is covered in Chapter 16, where we
fine-tune a 7B-parameter VLA under a 40 GB VRAM ceiling. The mechanism is the
same; the intent is different.

### (2) `logits = policy(obs_b)`

This is the forward pass: data flows from the input tensor through the trunk,
through each head, and returns a list of logit tensors. PyTorch's autograd
engine is watching. Every operation that involves a tensor with
`requires_grad=True` (which all model parameters carry by default) is recorded
in a computation graph. After this line, PyTorch knows the complete sequence of
operations that produced `logits` from `policy.parameters()`. The backward pass
will traverse that graph in reverse.

### (3) Loss computation

```python
loss = sum(
    loss_fn(logits[j], acts_b[:, j])
    for j in range(7)
) / 7.0
```

`loss_fn(logits[j], acts_b[:, j])` computes the cross-entropy for joint $j$:
it softmaxes the logits to get a predicted PMF $q_j$, then computes $-\log
q_j(a^{\star}_j)$ for each item in the batch and averages. Summing over 7 joints
and dividing by 7 gives the mean per-joint cross-entropy. This is numerically
identical to what OpenVLA computes on its 7 action-token positions
(arXiv:2406.09246), except that OpenVLA uses a 32,000-token vocabulary masked
to 256 action bins rather than a native 256-class softmax. The gradient structure
is the same.

Note that the sum over 7 joints is part of the computation graph. `loss` is a
scalar tensor, not a Python float. When you call `.item()` to log it, you detach
it from the graph; do not call `.item()` before `.backward()`.

### (4) `loss.backward()`

This is the chain rule, executed. PyTorch walks the computation graph from the
scalar loss back to every parameter, computing vector-Jacobian products at each
node. After this line, every parameter in `policy` has its `.grad` attribute
populated with $\nabla_{\theta} \mathcal{L}$, a tensor of the same shape as the
parameter itself. This is the $n$-dimensional vector we described in §3.1: for
`SmallPolicy`, it has on the order of 64×256 + 256 + 256×256 + 256 + 7×(256×256 +
256) ≈ 490,000 entries. The cost of computing all of them is roughly twice the
cost of the forward pass.

If the loss is a Python sum of `CrossEntropyLoss` values — which it is here —
backprop automatically distributes through the sum via linearity. Each head's
loss contributes a gradient to `self.heads[j]` parameters and also to `trunk`
parameters, because the trunk output feeds into every head. The trunk's gradient
is the sum of 7 head gradients, all propagated through the shared representation.
This is an important structural fact: training multiple heads on the same trunk
is not the same as training 7 independent networks. They regularize each other
through the shared parameters.

### (5) Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
```

Before the optimizer updates the weights, this line rescales the gradient
vector if its L2 norm exceeds 1.0. If $\|\nabla L\|_2 > 1$, the gradient is
replaced by $\nabla L / \|\nabla L\|_2$; otherwise it is left unchanged. This
prevents the exploding-gradient failure mode described in §3.1: a single
unusually large loss value can produce a gradient spike that moves weights far
from a good region. Clipping at 1.0 is a commonly safe default; VLA fine-tuning
recipes (Chapter 16) sometimes lower it to 0.5 for the first few thousand steps.
The cost is a single extra pass over all parameters to compute the norm, which
is negligible.

Note the placement: clipping must happen after `.backward()` (the `.grad`
attributes must exist) and before `.step()` (otherwise you are clipping a stale
gradient that the optimizer has already used). This ordering is the source of
one of the more common training-loop bugs: clipping after `step()` does nothing.

### (6) `optimizer.step()`

This applies the AdamW update rule:

$$
m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad
v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon} - \eta \lambda \theta_{t-1}.
$$

The first two equations maintain a running mean ($m_t$) and running variance
($v_t$) of the gradient. Dividing by the square root of the variance is the
"adaptive" part of Adam: dimensions with high-variance gradients get smaller
effective learning rates, and dimensions with low-variance gradients get larger
ones. The defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
work for the overwhelming majority of robot learning training runs.

The last term, $-\eta \lambda \theta_{t-1}$, is the AdamW weight decay: it
directly decays the parameter toward zero, independently of the gradient. This
is subtly different from L2 regularization in the loss (which couples the
regularizer through the adaptive denominator). For fine-tuning large pretrained
models like OpenVLA, AdamW's decoupled decay is the right choice; plain Adam
with L2-regularized loss can under-regularize large-magnitude parameters.

## Reading the diagnostics

After 50 epochs on random data, the loss should plateau around $\log(256) \approx
5.54$ — the entropy of a uniform distribution over 256 bins. That is the floor
for a model that has learned nothing, because random action-bin targets are
uniformly distributed and the best a model can do is predict uniform logits.
If the loss drops below $\log(256)$, the model has found structure in the
synthetic data — almost certainly overfitting to random correlations. If the
loss stays well above $\log(256)$, the learning rate is probably too small
or the gradients are being clipped to zero.

Two diagnostic quantities beyond the loss are worth logging from the start of
any training run. First, the gradient norm before clipping: if it is consistently
at the clip threshold, you are cutting information every step. Second, the weight
update magnitude — `optimizer.step()` changes the parameters by some amount,
and the ratio of that amount to the parameter norm (sometimes called the update
ratio or the param SNR) should stay in the range $10^{-3}$ to $10^{-2}$ for
stable training. A ratio above $10^{-2}$ indicates the effective learning rate
is too high; a ratio below $10^{-3}$ indicates the model is not learning. Neither
PyTorch nor AdamW provides these quantities automatically; log them with a few
extra lines before `optimizer.step()`, using `torch.nn.utils.clip_grad_norm_`'s
return value (which is the unclipped norm) and a parameter-snapshot comparison.

## Three things to change for a real task

The loop above is a template, not a production recipe. Three changes are needed
to move from random synthetic data to actual robot demonstrations.

**Replace the synthetic dataset.** Real behavior cloning loads `(observation,
action)` pairs from a teleoperation log. In practice this means a DataLoader
over a dictionary of tensors, often loaded from HDF5 or RLDS format. The shapes
change — observations might be $(B, 224, 224, 3)$ RGB images rather than
$(B, 64)$ vectors — but the loop structure does not.

**Replace the loss for continuous actions.** If you switch from discrete-bin
actions to raw continuous vectors, as π0 does (arXiv:2410.24164), the per-joint
cross-entropy becomes per-joint mean-squared error, or a flow-matching loss, or
a diffusion denoising loss. The backward pass and optimizer step are identical;
only `loss_fn` changes. Chapter 10 covers each of these alternatives in detail.

**Add a validation loop.** One loop over the training data measures whether the
model is fitting. A second loop over held-out episodes — without calling
`optimizer.step()` — measures whether it is generalizing. If training loss
drops but validation loss does not, the model is memorizing demonstrations rather
than learning a transferable policy. Chapter 15 discusses how to construct
evaluation splits that are actually informative in the robot setting, where
collecting genuinely out-of-distribution validation data is non-trivial.

With these three modifications, the 50-line loop above is functionally the core
of the training pipelines used in RT-1 (arXiv:2212.06817), OpenVLA
(arXiv:2406.09246), and Octo (arXiv:2405.12213). The transformer architectures
differ, the datasets differ, the hardware scale differs by orders of magnitude —
but `zero_grad`, forward pass, loss, `backward`, clip, `step` is the sequence in
each of them. The next section extends the vocabulary one step further: rather
than a single supervised loss on demonstrated actions, we examine the three loss
families — supervised, reinforcement, and self-supervised — that the rest of the
book draws from, and make precise when each one is the right tool.
