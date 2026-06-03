---
appendix: C
title: "PyTorch and JAX primer"
target_words: 4200
status: draft
prereqs: Python at the level of a comfortable user; Appendix A (matrices) and B (probability); §3.3 read at least once
key_refs:
  - Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS 2019.
  - Bradbury, J. et al. (2018). JAX: composable transformations of Python+NumPy programs. github.com/google/jax.
  - Heek, J. et al. (2024). Flax: A neural network library for JAX. github.com/google/flax.
  - Kim, M. J. et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black, K. et al. (2024). π0. arXiv:2410.24164.
---

# Appendix C.  PyTorch and JAX primer

The body of this book is implementation-shaped. The 50-line training
loop of §3.3 is PyTorch, the OpenVLA checkpoint of Chapter 2 is
PyTorch, the LoRA fine-tune of Chapter 16 is PyTorch. Two important
parts of the modern stack are not: π0 (arXiv:2410.24164) and Octo
(arXiv:2405.12213) ship as JAX/Flax models, and any reader who wants
to do more than fine-tune a checkpoint will eventually have to read
JAX code. This appendix is the cheat sheet for both frameworks. It
assumes Python familiarity and skips installation; if `import torch`
and `import jax` already work, you are in the right place.

## C.1  Tensors and devices

Both frameworks revolve around a tensor type: a multidimensional array
that can live on CPU, GPU, or TPU, and that participates in automatic
differentiation. Three differences are worth knowing immediately.

PyTorch tensors are *mutable* and live on the device specified at
creation:

```python
import torch
x = torch.randn(3, 4, device="cuda")
x[0, 0] = 1.0   # in-place mutation; fine
```

JAX arrays are *immutable* and live on whichever device JAX chose by
default (which you can override with `jax.device_put`):

```python
import jax.numpy as jnp
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
# x[0, 0] = 1.0  -> TypeError: 'ArrayImpl' object does not support item assignment
x = x.at[0, 0].set(1.0)   # functional update; returns a new array
```

The `x.at[i].set(v)` pattern is JAX's answer to in-place writes, and
it shows up everywhere — you will see it in optimizer step functions,
in indexed updates to environment state, and in the body of every
scan loop. It feels awkward for one afternoon and then disappears
into the background.

The second difference: PyTorch dispatches operations *eagerly* by
default, executing each one when its line of code runs. JAX traces
operations *lazily*, building a computational graph that is compiled
the first time you call the wrapping function. The third: PyTorch's
default precision is fp32; JAX's default is fp32 too, but TPU work
often defaults to bf16 because of hardware. Mixed-precision training
in PyTorch goes through `torch.cuda.amp.autocast`; in JAX it goes
through dtype-promotion rules and an explicit policy from `flax.linen`.

## C.2  Autograd in two paragraphs

The single feature that makes either framework worth using is reverse-
mode automatic differentiation. In PyTorch, every tensor remembers
how it was produced if it has `requires_grad=True`, and calling
`.backward()` on a scalar loss populates `.grad` attributes throughout
the graph:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)        # tensor([2., 4., 6.])
```

In JAX, gradients are produced by *function transformations*. The
function `jax.grad(f)` returns a new function that computes the
gradient of `f` with respect to its first argument; the original `f`
is unchanged:

```python
import jax
def loss(x):
    return (x ** 2).sum()
grad_fn = jax.grad(loss)
print(grad_fn(jnp.array([1.0, 2.0, 3.0])))   # [2. 4. 6.]
```

The functional style is what makes JAX composable: `jax.grad`,
`jax.jit`, `jax.vmap`, and `jax.pmap` can be stacked in any order. The
imperative style is what makes PyTorch feel like writing NumPy. In
practice, large-scale VLA training favors PyTorch for its ecosystem
and JAX for its TPU-native parallelism; the field has not converged.

## C.3  A minimal PyTorch training loop

The training loop of §3.3 is the canonical template. The skeleton:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = MyPolicy().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loader = DataLoader(my_dataset, batch_size=64, shuffle=True, num_workers=4)

for epoch in range(num_epochs):
    for batch in loader:
        obs = batch["obs"].cuda(non_blocking=True)
        act = batch["act"].cuda(non_blocking=True)
        pred = model(obs)
        loss = nn.functional.mse_loss(pred, act)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

Six lines of substance, three lines of glue. The substance is: move
batch to GPU, run forward, compute loss, zero gradients, backward,
optimizer step. Every one of those steps has a JAX analogue, but the
JAX version returns a new optimizer state and new parameters rather
than mutating in place; see §C.5.

Four practical refinements you will see almost everywhere in the
body of the book. *Gradient clipping*:
`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` before
`optimizer.step()`, to prevent occasional exploding gradients from
destroying training. *Learning rate scheduling*: typically a linear
warmup for the first 1000 steps followed by cosine decay, implemented
via `torch.optim.lr_scheduler`. *Mixed precision*: wrap the forward in
`with torch.cuda.amp.autocast(dtype=torch.bfloat16):` and use a
`GradScaler` for fp16 (bf16 does not need a scaler). *Checkpoint
saving*: `torch.save({"model": model.state_dict(), "opt":
optimizer.state_dict(), "step": step}, path)` every $N$ steps;
checkpointing optimizer state is what lets you resume from
preemption without re-warmup.

## C.4  Modules, parameters, and state dicts

A PyTorch `nn.Module` is a class that bundles parameters, buffers, and
a `forward` method. The canonical pattern:

```python
class SmallPolicy(nn.Module):
    def __init__(self, obs_dim=20, act_dim=7, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, act_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

`model.parameters()` returns an iterator over every leaf parameter
the module owns; `model.state_dict()` returns an `OrderedDict` mapping
parameter names to tensors. The two are how a checkpoint gets saved
and loaded, and how a fine-tune restricts itself to a subset of
parameters: filter `parameters()` by name and hand only the survivors
to the optimizer. OpenVLA's LoRA fine-tune does exactly this — the
7B base parameters are frozen with `requires_grad = False`, the
LoRA adapters are added with `requires_grad = True`, and only the
adapters appear in the optimizer's parameter list.

A common confusion: `state_dict` includes both parameters (learned)
and buffers (not learned but part of the model state — running mean
and variance of batch-norm layers, for example). Saving only
`parameters()` and not the buffers will silently produce a checkpoint
that "loads cleanly" but evaluates wrong on subsequent runs.

## C.5  The same loop in JAX/Flax

JAX has no notion of a stateful model. A model in Flax is a
declaration of a function-with-parameters, and the parameters are
ordinary PyTree leaves that you pass explicitly. The same small
policy:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

class SmallPolicy(nn.Module):
    hidden: int = 256
    act_dim: int = 7

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.act_dim)(x)

model = SmallPolicy()
params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 20)))
optimizer = optax.adamw(3e-4)
opt_state = optimizer.init(params)

@jax.jit
def update(params, opt_state, batch):
    def loss_fn(p):
        pred = model.apply(p, batch["obs"])
        return jnp.mean((pred - batch["act"]) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

Four things are different. The model's parameters live outside the
model object, in a separate `params` PyTree. Forward passes use
`model.apply(params, x)` rather than `model(x)`. The optimizer is a
separate object whose state is also passed explicitly. The whole
training step is wrapped in `jax.jit`, which compiles it to a single
XLA program the first time it is called — slow on call number one,
fast forever after.

The functional style buys two things. The same `update` function
works on a single device and across multiple devices (with `jax.pmap`
or `jax.shard_map`), without changing the implementation. And the
behavior of `update` is completely determined by its arguments —
there is no hidden state mutated in the background — which makes it
straightforward to reason about, to checkpoint, and to debug.

## C.6  Vectorization, parallelization, and the function-transform family

The transforms JAX exposes are worth listing as a group, because they
are the single feature most often cited as the reason to choose JAX
over PyTorch.

`jax.jit(fn)` compiles `fn` with XLA. Inputs of fixed shape and dtype
are required at trace time, and Python-level control flow that depends
on tensor values must be lifted to `jax.lax.cond` and `jax.lax.scan`.

`jax.grad(fn)` returns the gradient of `fn` with respect to its first
argument. `jax.value_and_grad(fn)` returns both the value and the
gradient in one pass.

`jax.vmap(fn, in_axes=...)` maps `fn` across a leading batch axis as
if it were a Python `for` loop, but produces vectorized code. This is
what eliminates the explicit batch dimension in many JAX models: write
the function for a single example, then `vmap` it.

`jax.pmap(fn)` parallelizes `fn` across multiple devices, replicating
the parameters and sharding the batch. Multi-host setups use
`jax.lax.with_sharding_constraint` for finer-grained control.

PyTorch's analogues exist but are less unified. `torch.compile` is the
JIT story; `torch.vmap` exists but is less integrated than its JAX
counterpart; multi-device training is done through `DistributedDataParallel`
or `FSDP`. The body of the book uses PyTorch's defaults for everything
except the JAX-native models in Chapters 12 and 13.

## C.7  Multi-GPU in one paragraph

Most VLA fine-tunes in this book fit on a single GPU. When they do
not, the standard PyTorch idiom is `torch.nn.parallel.DistributedDataParallel`
(DDP), which replicates the model on each GPU, shards the batch, and
averages gradients via NCCL all-reduce. For models that exceed a
single GPU's memory, *FSDP* (fully sharded data parallel) shards the
parameters themselves across GPUs and reassembles them just in time
for each forward and backward. The launch incantation is

```bash
torchrun --nproc-per-node=8 train.py
```

and inside the script, `init_process_group("nccl")` wires everything
up. JAX's equivalent is `jax.distributed.initialize()` plus the
function transforms above. Single-host multi-GPU is two lines of
boilerplate in either framework; multi-host setups are more involved
and the framework-specific docs are the right place to look.

## C.8  Five pitfalls that have eaten a week of someone's time

A short, unranked list of failure modes that show up often enough to
be worth printing.

*The `.cuda()` call that does not happen.* A model on GPU and data on
CPU produces an error message that names the device mismatch
explicitly; a model on GPU and a single tensor that secretly stayed
on CPU silently runs much slower. Run `next(model.parameters()).device`
to confirm.

*The forgotten `optimizer.zero_grad()`.* PyTorch accumulates gradients
across `backward()` calls; if you forget to zero, the second batch's
gradients are added to the first. Always call `zero_grad(set_to_none=True)`
before each backward, or use the simpler `optimizer.zero_grad()` if
you do not need the small memory savings.

*The shape that broadcasts when you meant it to error.* PyTorch's
broadcasting rules are lenient: a `[B, D]` tensor and a `[D]` tensor
combine without complaint. If you swap an axis upstream and the
resulting shape still broadcasts, the model trains on garbage. Adding
`assert x.shape == (B, D, T)` at module boundaries during development
catches more bugs than it has any right to.

*The data loader that pickles too much.* Each `DataLoader` worker
forks the parent process; if the parent has loaded a 7B-parameter
model before the loader starts, each worker copies it. Construct the
data loader before the model, or use `num_workers=0` during debug.

*The JAX recompilation that fires on every step.* `jax.jit` keys the
compilation cache on the *shapes and dtypes* of the arguments, not
their values. If any argument's shape changes per step — variable-
length sequences, a batch that has one fewer example at the end of
the epoch — JAX recompiles, and the second iteration of training
suddenly takes thirty seconds. Pad to a fixed shape, or use
`jax.lax.dynamic_slice` to handle variable lengths inside a fixed-
shape buffer.

## C.9  Reading π0's JAX code

The book's chapters use PyTorch by default, but Chapter 13's
treatment of π0 (Black et al., 2024, arXiv:2410.24164) leans on the
official JAX implementation. Three reading hints if you have not used
JAX before.

The model class will be a `flax.linen.Module` (or, increasingly,
`flax.nnx` for the newer codebases). Its parameters are not inside
the class; they are produced once by `model.init(...)` and threaded
through every subsequent call. When you see `model.apply(params, x)`
in π0's training script, that is the JAX-native forward pass.

Anywhere you would have a Python `for` loop in PyTorch — over
diffusion timesteps, over decoding tokens, over a recurrent unroll —
JAX uses `jax.lax.scan` instead. `scan` looks like a fold: it carries
a state forward through a fixed number of iterations and accumulates
outputs along the way. Reading `scan`-based code feels foreign for a
few minutes; the payoff is that the entire loop compiles to one XLA
program.

Anywhere you would have a random number in PyTorch (via the global
RNG), JAX takes an *explicit PRNG key* as an argument. The key is
split via `jax.random.split(key)` into one key for each random
operation, and the resulting code is deterministic in a way PyTorch's
randomness is not. The π0 codebase passes PRNG keys through nearly
every function call; the apparent boilerplate is what gives the model
its bit-exact reproducibility.

## C.10  A choice grid

When should you use which framework? Three rough heuristics.

If you are fine-tuning a published VLA, use whatever framework the
authors released the checkpoint in. OpenVLA, RT-1, RT-2, RDT-1B,
TinyVLA, SmolVLA — PyTorch. π0, Octo — JAX. Converting checkpoints
between frameworks is possible but lossy in the kind of details
(parameter naming, optimizer state) that matter for the second epoch.

If you are training a new model from scratch on a single 8-GPU node,
PyTorch is the path of least resistance. The ecosystem (HuggingFace,
PEFT, accelerate, deepspeed) is mature and the debugging story is
better.

If you are training on TPUs or multi-host clusters of $\geq 64$ GPUs,
JAX is the path of least resistance. The compiled-by-default
model and the unified parallelism story (`pmap`, `shard_map`) are
worth the learning curve.

This is the framework chapter. The next one (Appendix D) is the
simulator chapter — how to set up the MuJoCo or Isaac Lab instance
on which the policies you just learned to train will actually be
trained.
