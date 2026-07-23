---
chapter: 10
section: 10.3
title: "Flow matching and rectified flow for action"
target_words: 2000
status: draft
prereqs: §10.1 (diffusion mechanism, noise schedule, sampling ladder), §10.2 (action chunks, receding horizon), §3.1 (gradients and the ODE view of a network), §3.2 (expectations). Helpful, §5.1 for state/action notation.
key_refs:
  - Lipman, Y. et al. (2023). Flow Matching for Generative Modeling. ICLR 2023.
  - Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast — Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023.
  - Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 10.3  Flow matching and rectified flow for action

§10.2 left both action heads generating their chunk from scratch every time, ten denoising steps for Diffusion Policy, one CVAE pass for ACT. The ten steps are the expensive part, and §10.1 already named the reason: the DDPM sampler walks down a long ladder, one network call per rung, because the reverse process it learned is a wandering, curved path from noise back to data. Flow matching asks a sharper question. What if you trained the path to be straight? A straight path from noise to a sample can, in the limit, be traversed in a single step, because you already know the direction, it never changes. This section is how that idea works and why π0 built a foundation model on it.

## From a noise schedule to a velocity field

Start by throwing out the noise schedule. In §10.1 the forward process was a fixed chain of Gaussian corruptions indexed by a discrete step $t \in \{1, \dots, T\}$, and the whole apparatus of $\beta_t$ and $\bar\alpha_t$ existed to describe how much of the original sample survived at each rung. Flow matching replaces that with something simpler to picture: a continuous path in time, $t \in [0, 1]$, that carries a point from pure noise at $t=0$ to a data sample at $t=1$.

The cleanest such path is a straight line. Draw a noise sample $x_0 \sim \mathcal{N}(0, I)$ and a data sample $x_1$ from your dataset, and define the point at time $t$ as the linear interpolation between them:

$$
x_t = (1 - t)\, x_0 + t\, x_1.
$$

At $t=0$ you are sitting on the noise sample; at $t=1$ on the data sample; in between you slide along the segment connecting them. Now ask the obvious calculus question, how fast is the point moving, and in what direction? Differentiate:

$$
\frac{d x_t}{dt} = x_1 - x_0.
$$

The velocity is constant. Along this particular straight segment the point moves in the same direction at the same speed the whole way. That constant vector, $x_1 - x_0$, is the target the network learns to predict, and its constancy is the entire source of flow matching's speed advantage. Where diffusion learned to nudge a sample one noisy notch toward the data manifold, flow matching learns the *velocity of a flow* that transports the whole noise distribution onto the whole data distribution.

## Training: regress the velocity

The training loop is, if anything, blunter than diffusion's. Pick a data sample $x_1$, draw a noise sample $x_0$, pick a random time $t \in [0, 1]$, form the interpolated point $x_t$, and train a network $v_\theta(x_t, t)$ to output the velocity that carried it there:

$$
\mathcal{L} = \mathbb{E}_{\,x_1,\,x_0,\,t}
\left[\ \big\|\ v_\theta(x_t,\ t)\ -\ (x_1 - x_0)\ \big\|^2\ \right].
$$

This is the conditional flow matching objective of Lipman et al. (2023), specialized to straight-line paths, the same specialization Liu et al. (2023) arrived at independently and named rectified flow. Compare it line for line with the diffusion loss from §10.1. Both are mean squared error. Both feed the network a corrupted point and a time index. The only change is the regression target: diffusion predicts the *noise* $\epsilon$ that was added, flow matching predicts the *velocity* $x_1 - x_0$ that points from noise toward data. That is a small edit to the code and a large change in what the model represents.

```python
# v_theta: velocity network, same shape in and out as the sample
def flow_matching_loss(x1, v_theta):
    x0 = torch.randn_like(x1)                 # noise endpoint
    t  = torch.rand(x1.shape[0], 1)           # random time in [0, 1]
    xt = (1 - t) * x0 + t * x1                # point on the segment
    target = x1 - x0                          # constant velocity
    return ((v_theta(xt, t) - target) ** 2).mean()
```

Two things are worth pausing on. First, there is no schedule to tune, no $\beta_t$ curve, no choice of variance-preserving versus variance-exploding parameterization. The interpolation is linear and that is the end of it, which removes a whole category of the fiddly hyperparameters diffusion practitioners argue about. Second, the loss regresses a *conditional* velocity, the exact vector for this one $(x_0, x_1)$ pair, but what the network converges to is the *marginal* velocity averaged over every pair that could have produced $x_t$. That averaging is the subtle part, and it is exactly where rectified flow earns its name, so hold the thought.

## Sampling: integrate an ODE

Generation runs the flow forward in time. Start at a fresh noise sample $x_0 \sim \mathcal{N}(0, I)$ and integrate the ordinary differential equation $\tfrac{dx}{dt} = v_\theta(x, t)$ from $t=0$ to $t=1$. The crudest integrator, forward Euler, is a loop that reads almost like the DDPM step from §10.1 with the stochastic term deleted:

```python
def sample(v_theta, shape, n_steps):
    x = torch.randn(shape)          # start at noise, t = 0
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = i * dt
        x = x + v_theta(x, t) * dt  # step along the velocity
    return x                        # arrives at data, t = 1
```

The number of steps `n_steps` is now an honest dial you turn at inference, not a property baked into a training schedule. Turn it up and the Euler integration hugs the true trajectory more closely; turn it down and you take fewer, larger strides and pay for it in accuracy. This is the knob §10.4 will weigh against control-loop latency.

Here is the catch, and it is the reason "just use a straight line" is not the whole story. The individual training segments are straight, but the *marginal* field the network learns is generally not, because many different straight segments cross the same point $x_t$ heading in different directions, and the network can only output their average there. Follow that averaged field with Euler and your actual trajectory bends. A curved trajectory needs many small steps to integrate accurately, you are back to diffusion's problem, just in nicer notation. Naive flow matching with a linear path buys you a cleaner objective, but not, by itself, one-step sampling.

## Rectified flow: straighten the path

Rectified flow (Liu et al., 2023) is the procedure that removes the curvature, and the trick is almost impudent. Train a first flow model the usual way. Then generate a batch of samples with it, and keep the *pairing*: each generated $x_1$ came from a specific starting noise $x_0$. Now retrain a fresh flow model on these matched pairs instead of on random noise-data pairings. Because the pairs came from the model's own transport, the straight segments between them cross each other far less, so the marginal field the second model learns is much closer to actually straight. Repeat once more if you want it straighter. Each pass is called a *reflow*.

The payoff is that a well-rectified flow can be integrated in a handful of Euler steps, sometimes even one, with little loss in sample quality, because a straight trajectory is exactly the case where one big Euler step lands in the right place. This is the concrete sense in which flow matching "needs fewer steps by construction," the promise §10.1 made when it first pointed forward to this section. Diffusion gets to few-step sampling by distilling a slow teacher into a fast student after the fact; rectified flow builds the straightness into the training objective and its reflow procedure. Both end up fast; flow matching gets there with less machinery.

It is worth being precise about what reflow costs, because §10.4 will hold it against the alternatives. Each reflow pass means generating a dataset from the current model and training another one, real compute, paid once, offline. In exchange you move expense out of the inference loop, where a robot cannot afford it, and into training, where you can. That is usually the trade you want on hardware.

## Flow matching for action

Everything so far has been generic, the sample $x_1$ could be an image. Point it at robot actions the same way §10.2 pointed diffusion at them: let $x_1$ be a chunk of future actions, a stack of, say, fifty joint or end-effector targets, and condition the velocity network on the current observation. The training loss gains a conditioning argument and nothing else changes:

$$
\mathcal{L} = \mathbb{E}\left[\ \big\|\ v_\theta(x_t,\ t,\ o)\ -\ (x_1 - x_0)\ \big\|^2\ \right],
$$

with $o$ the encoded cameras, proprioception, and language instruction. At deployment you draw noise the shape of an action chunk, integrate the observation-conditioned ODE for a few steps, and read off a chunk of continuous actions, no discretization, no binning.

The reference implementation is **π0** (Black et al., 2024, arXiv:2410.24164), the first foundation-scale VLA to use a flow-matching action head, and the model Chapter 13 dissects in full. The one-sentence version: π0 takes a pretrained vision-language model, attaches a flow-matching head that generates action chunks at high frequency, and trains the whole thing on a large cross-embodiment demonstration mixture. The choice of flow matching over diffusion is not incidental. π0 targets smooth, dexterous control at up to 50 Hz, folding laundry, bussing a table, and at that rate a fifty-step diffusion sampler per chunk is untenable. A flow head that produces a good chunk in around ten integration steps, on a continuous action space that never has to be quantized into tokens, is what makes the control frequency reachable. The contrast with the discrete action-token approach of RT-2, which we reach in Chapter 12, is exactly the continuous-versus-discrete axis this chapter keeps circling; π0 is the continuous pole, and flow matching is how it holds that pole cheaply.

Two caveats keep the picture honest. Flow matching does not invent a new kind of multimodality, a flow, like a diffusion model, represents a full distribution over action chunks and samples committed modes from it, so the mug-and-laptop argument of §10.1 carries over unchanged; what flow matching changes is the *cost* of drawing that sample, not the expressiveness of the draw. And the straightening is not free lunch: push toward genuinely one-step generation on a hard, multimodal action distribution and quality does eventually degrade, which is why deployed systems like π0 sit at a few steps rather than one. The right number of steps is a task-dependent choice, and choosing it well is precisely the trade-off §10.4 takes up next.
