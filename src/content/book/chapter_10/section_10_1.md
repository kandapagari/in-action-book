---
chapter: 10
section: 10.1
title: "A 10-minute introduction to diffusion models"
target_words: 2000
status: draft
prereqs: §3.2 (random variables, expectations, KL divergence), §3.4 (supervised loss families), §8.4 (what gets tokenized). Helpful, §5.1 for the state/action notation reused here.
key_refs:
  - Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
  - Song, Y. et al. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. ICLR 2021.
  - Chi, C. et al. (2023). Diffusion Policy — Visuomotor Policy Learning via Action Diffusion. RSS 2023.
---

# 10.1  A 10-minute introduction to diffusion models

Every action head you have met so far commits to one number. The Gaussian policy of §7.2 outputs a mean and a variance and samples once; the discrete action-token classifier you will meet in Chapter 11 picks the argmax bin. Both assume the right action is a single point, or a single blob around a point. That assumption breaks the moment a task has more than one good answer. Reaching for a mug on a cluttered table, you can swing left around the laptop or right around the coffee pot; both work, and the average of the two drives your hand straight into the laptop. A policy trained to regress the mean of the demonstrations learns exactly that average, and exactly that collision.

Diffusion models are the tool the field reached for to fix this. They were invented for images, the flashy text-to-image systems of the early 2020s are diffusion models, but the property that made them good at images is the property robotics needed: they represent a full distribution over outputs, multi-peaked and all, instead of collapsing it to a mean. Before we can talk about diffusing *actions* in §10.2, you need the mechanism itself. This section is the ten-minute version.

## The idea: destroy structure, then learn to rebuild it

Suppose you have a dataset of samples $x_0$ drawn from some complicated distribution $q(x_0)$ you cannot write down, natural images, or, for us, expert action trajectories. You want a model that can produce new samples from that same distribution. The direct approach, writing a formula for $q$ and sampling from it, is hopeless; the distribution lives in a space with thousands of dimensions and has structure no closed form captures.

Diffusion sidesteps the problem with a trick that sounds like it should not work. Take a real sample and gradually wreck it, adding a little Gaussian noise at a time, until after many steps nothing is left but pure noise. That direction is trivial, adding noise requires no learning. Then train a network to undo one step of the wreckage: given a noisy sample, predict what was added. If the network can reliably strip away a little noise, you can start from pure noise and run it many times, and what falls out the far end is a fresh sample from $q$. You have turned "sample from an intractable distribution" into "denoise, repeatedly," which is just supervised regression.

The forward, structure-destroying process is fixed and defined for you. Denoising Diffusion Probabilistic Models (Ho, Jain & Abbeel, 2020), the paper that made the recipe practical, defines it as a chain of $T$ steps, each adding Gaussian noise with a small variance $\beta_t$:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right).
$$

A useful accident of Gaussians is that you never have to run this chain step by step. Composing $t$ Gaussian steps gives another Gaussian, so you can jump straight to any noise level in one shot. Writing $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^{t}\alpha_s$,

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I).
$$

Read that equation as a dial. At $t=0$ you have the clean sample. As $t \to T$, $\bar\alpha_t \to 0$ and the sample dissolves into standard Gaussian noise, all trace of $x_0$ gone. The whole schedule is chosen so that $x_T$ is indistinguishable from noise you could draw yourself, which is what lets you start sampling from nothing.

## Training: predict the noise

The reverse process is the part you learn. In principle you want a network $p_\theta(x_{t-1} \mid x_t)$ that inverts one forward step, and the honest derivation runs through a variational bound on the data log-likelihood, the same machinery behind a VAE. That derivation is worth reading once in the original paper; here is where it lands, which is all you need to implement it.

Because the forward equation tells you exactly which noise $\epsilon$ was mixed into $x_0$ to produce $x_t$, you can train the network to recover that noise. Sample a clean example, pick a random timestep $t$, noise the example to level $t$, and ask the network, call it $\epsilon_\theta(x_t, t)$, to guess the $\epsilon$ that was added. The loss is the plainest thing imaginable, mean squared error between the true noise and the predicted noise:

$$
\mathcal{L} = \mathbb{E}_{x_0,\,t,\,\epsilon}
\left[\ \big\|\ \epsilon - \epsilon_\theta(x_t, t)\ \big\|^2\ \right].
$$

This is the payoff. All the probabilistic scaffolding collapses to a regression problem of the sort you wrote a training loop for in §3.3. The network takes a noisy input and a timestep, outputs a same-shaped noise estimate, and you minimize squared error. There is no adversary to balance as in a GAN, no reconstruction-versus-KL tension as in a VAE (§3.2 for the KL term). Diffusion training is stable for the same reason it is boring, and that stability is a large part of why it took over.

A note on what the network predicts. Estimating the added noise $\epsilon$ is equivalent, up to a rescaling, to estimating the gradient of the log-density $\nabla_{x} \log q(x_t)$, the *score*. Song et al. (2021) arrived at the same family of models from that direction, framing the whole thing as a stochastic differential equation and calling it score-based generative modeling. Noise prediction and score matching are two dialects for one idea; you will see both names, and it is worth knowing they refer to the same object.

## Sampling: run the denoiser backward

To generate, start from $x_T \sim \mathcal{N}(0, I)$ and walk down the ladder. At each step, use $\epsilon_\theta$ to estimate the noise in the current $x_t$, subtract a scaled version of it to get a cleaner $x_{t-1}$, and add back a touch of fresh noise to stay on the distribution the network was trained on. One DDPM sampling step looks like:

```python
# eps_theta: trained noise-prediction network
# alpha[t], alpha_bar[t], beta[t]: from the fixed schedule
def ddpm_step(x_t, t):
    eps = eps_theta(x_t, t)
    mean = (x_t - beta[t] / (1 - alpha_bar[t]).sqrt() * eps) / alpha[t].sqrt()
    if t > 0:
        return mean + beta[t].sqrt() * torch.randn_like(x_t)
    return mean  # last step is deterministic
```

Repeat from $t=T$ down to $t=0$ and the final $x_0$ is your sample.

The catch is right there in the loop: you call the network once per step, and the original DDPM used $T = 1000$. A thousand forward passes to produce one sample is fine for generating a wallpaper offline; it is a disaster for a robot that needs a new action every 30 milliseconds. Most of the engineering since 2020 has been about buying back those steps, DDIM-style deterministic samplers that skip most of the ladder, distillation into a handful of steps, and eventually the flow-matching reformulation of §10.3 that changes the objective so that fewer steps are needed by construction. Hold onto this tension between sample quality and inference latency; it is the axis §10.4 organizes the whole chapter around.

## Conditioning: generating the *right* sample

So far the model generates unconditioned samples, some plausible $x_0$, with no say over which one. A robot needs the opposite: given what the camera sees right now, produce an action that fits *this* situation, not a random draw from every action in the dataset. The fix is to feed the condition into the denoiser. Let $o$ be the current observation (an image embedding, a proprioceptive state, a language instruction, or all three) and train $\epsilon_\theta(x_t, t, o)$ to predict noise given that context. The loss is unchanged; you just widen the network's input. At sampling time you hold $o$ fixed and denoise as before, and the walk down the ladder is now steered toward actions consistent with the observation.

The same conditioning hook is what lets you dial *how strongly* the model obeys the condition. Classifier-free guidance, the standard trick borrowed from text-to-image work, trains the network to run both with and without $o$ and then extrapolates between the two predictions at sampling time, sharpening obedience to the instruction. It matters less for manipulation than it does for image generation, but it is the reason a language-conditioned diffusion policy can be pushed to follow the prompt more literally when you need it to. We flag it here and return to it when it earns its keep.

## Why any of this helps a robot

Return to the mug and the laptop. A diffusion model does not learn the average of the two ways around the obstacle; it learns the distribution that has probability mass on the left path and on the right path and almost none in the collision zone between them. Sample it once and you get a committed left-swing or a committed right-swing, never the averaged smear. That is the multimodality property, and it is the single reason diffusion displaced mean-squared-error regression as the default action head for imitation learning.

There is a second reason, quieter but just as important for control. Nothing in the recipe cares whether $x_0$ is an image, a single action, or a chunk of sixteen consecutive actions stacked into one vector. If you define $x_0$ to be a short trajectory, the model learns to generate whole coherent trajectories at once, the smooth, temporally consistent motion that Diffusion Policy (Chi et al., 2023) gets and that the ACT architecture reaches by a related route. We will make that concrete in §10.2, where the abstract $x_0$ finally becomes a sequence of gripper poses.

None of this comes free. You have already seen the cost: a diffusion head trades one cheap forward pass for many, and a robot's control loop has no patience. Whether the multimodality is worth the latency depends on the task, and answering that question well is a skill this chapter is trying to give you.

With the mechanism in hand, corrupt with a fixed noise schedule, train a network to predict the noise, sample by denoising from pure noise, we can stop talking about images and start diffusing actions.
