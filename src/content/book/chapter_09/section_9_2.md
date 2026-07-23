---
chapter: 9
section: 9.2
title: "Latent dynamics: RSSM and Dreamer"
target_words: 2000
status: draft
prereqs: §9.1 (world model = learned transition/reward function; predict in latent space; the model carries memory), §3.2 (KL divergence, expectations), §3.3 (the training loop), §5.1 (states, actions, rewards), §7.2 (actor-critic and the variance problem).
key_refs:
  - Hafner et al. (2019). Learning Latent Dynamics for Planning from Pixels (PlaNet). ICML.
  - Hafner et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination (Dreamer). ICLR.
  - Hafner et al. (2021). Mastering Atari with Discrete World Models (DreamerV2). ICLR.
  - Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). arXiv preprint.
---

# 9.2  Latent dynamics: RSSM and Dreamer

Section 9.1 ended on a claim: to plan or to train a policy in imagination, you first need a model good enough to imagine with. Ha and Schmidhuber's car-racing controller showed the shape of the answer, compress pixels to a latent, predict the next latent, act on it, but its memory model was a mixture-density RNN trained separately from the vision model, and it worked cleanly on one toy game. The line of work that turned that sketch into a method used across dozens of tasks is the *recurrent state-space model*, or RSSM, and the family of agents built on it: PlaNet, then Dreamer through its three versions. This section is about what the RSSM is and why its one unusual design choice, splitting the latent state in two, is what makes the whole thing learnable.

## Why one latent variable is not enough

Start from the obvious design and watch it fail. You want a recurrent model that carries a latent state $s_t$ forward: given $s_t$ and action $a_t$, produce $s_{t+1}$, and from $s_t$ reconstruct the observation $o_t$ and predict the reward. If you make $s_t$ purely deterministic, an ordinary RNN or GRU hidden state, the model cannot represent uncertainty. A robot that has not yet looked inside a drawer does not know whether it holds a stapler or a snake, and a deterministic state forces the model to commit to one guess. Prediction error then punishes it for being decisive about something it could not have known.

If instead you make $s_t$ purely stochastic, sample it fresh from a learned distribution at every step, you get the opposite failure. The model can express uncertainty, but it struggles to *remember*. Any information it wants to carry across ten steps has to survive being resampled ten times, and the noise washes it out. Long-horizon prediction, which is the entire point of a world model you plan with, falls apart.

The RSSM's answer, from PlaNet (Hafner et al. 2019), is to keep both. The latent state is a pair: a **deterministic** part $h_t$, carried by a GRU, and a **stochastic** part $z_t$, sampled from a distribution that the network predicts. The deterministic path is a reliable wire down which information flows unchanged across many steps; the stochastic part rides on top of it and captures what the model cannot know for certain. This split is the single idea that makes latent dynamics work, and every Dreamer variant keeps it.

## The two distributions: prior and posterior

The RSSM has to do two jobs that pull in different directions, and it handles them with two distributions over the stochastic state.

The **prior** predicts the next stochastic state from the past alone: $\hat{z}_t \sim p_\theta(z_t \mid h_t)$, where $h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$ is the GRU rolling the recurrent state forward. This is the imagination path. When you dream, roll the model forward without looking at the world, you sample from the prior, because there is no new observation to condition on.

The **posterior** corrects that prediction using the observation that actually arrived: $z_t \sim q_\theta(z_t \mid h_t, o_t)$. This is the perception path, used during training and whenever the agent is grounded in a real frame. The posterior sees the drawer's contents; the prior had to guess.

Training pushes these two together. The model reconstructs the observation and predicts the reward from the state $(h_t, z_t)$, and, the part that matters, it minimizes the KL divergence between the posterior and the prior, $\mathrm{KL}\!\left(q_\theta(z_t \mid h_t, o_t)\,\|\, p_\theta(z_t \mid h_t)\right)$. That KL term is the dynamics-learning signal. It says: whatever the observation told you, the model should have been able to predict it from the past. Drive that KL to zero and the prior alone is enough to forecast, which is exactly the condition you need to dream without looking. The whole objective is the variational lower bound (the ELBO from §3.2's probability toolkit), applied to a sequence: reconstruction accuracy minus the KL, summed over the trajectory.

A short, illustrative version of one RSSM step, ignoring batching and the distribution parameterization details:

```python
def rssm_step(h_prev, z_prev, a_prev, obs=None):
    # Deterministic recurrence carries memory forward.
    h = gru(h_prev, concat(z_prev, a_prev))

    # Prior: predict next stochastic state from the past alone.
    prior = mlp_prior(h)                 # imagination path

    if obs is not None:
        # Posterior: correct the prior using the real observation.
        feat = encoder(obs)
        post = mlp_post(concat(h, feat)) # perception path
        z = post.sample()
        kl = kl_divergence(post, prior)  # dynamics-learning signal
    else:
        z = prior.sample()               # dreaming: no observation
        kl = None
    return h, z, kl
```

The `obs is None` branch is not an afterthought. It is the mode the model runs in when it imagines, and the fact that the same recurrence produces both grounded and imagined states, differing only in whether $z$ comes from the posterior or the prior, is what lets a policy trained in imagination transfer back to the real environment.

## PlaNet: plan with the model, skip the policy

PlaNet (Hafner et al. 2019) used the RSSM the way §9.1 called the first use of a world model: pure planning, no learned policy. At each step it ran a sampling-based planner, the cross-entropy method, entirely inside the latent space. It would sample a few hundred candidate action sequences, roll each one forward through the RSSM prior to predict its rewards, keep the best-scoring fraction, refit a distribution to them, and repeat. The first action of the winning sequence gets executed; then the whole search runs again from the new state. This is model-predictive control (§9.3 develops it) with a learned latent model standing in for the physics. On continuous-control tasks from pixels, a cheetah running, a cartpole swinging up, PlaNet matched model-free agents while using a fraction of the environment interaction, because the expensive search happened in imagination.

The catch is cost. Re-planning hundreds of rollouts at every single control step is slow, and the plan is only as good as the search budget. That is the opening Dreamer walked through.

## Dreamer: learn the policy inside the dream

Dreamer (Hafner et al. 2020) kept PlaNet's RSSM unchanged and replaced the online planner with a policy learned *from* imagined rollouts, the Dyna idea from §9.1, now with a deep latent model and an actor-critic. Training runs three interleaved loops. First, fit the RSSM to a replay buffer of real experience, exactly as above. Second, imagine: starting from states the model has seen, roll the prior forward for a fixed horizon (fifteen or so steps) to produce a batch of dreamed trajectories, and label each imagined state with a predicted reward and a learned value. Third, update an **actor** to maximize those imagined returns and a **critic** to estimate them, an ordinary actor-critic (§7.2), except every transition it learns from is synthetic.

The move that makes Dreamer more than Dyna-with-a-neural-net is *how* the actor gets its gradient. Because the RSSM is fully differentiable, Dreamer backpropagates the imagined return through the learned dynamics and straight into the actor's parameters. The policy does not have to discover which actions were good by trial and correlation, the way a policy-gradient method does (§7.2's variance problem); it gets an analytic signal for how a nudge to the action changes the predicted future reward. Real environment interaction is now needed only to keep the world model honest, to collect the experience the RSSM trains on, and not to train the policy at all. On the same continuous-control suite, Dreamer beat PlaNet's scores while dropping the per-step planning cost, because at deployment it just runs the learned actor: one forward pass, no search.

## DreamerV2 and V3: discrete latents and hands-off robustness

Two refinements matter for the reader who wants to know why Dreamer is the default the TOC's hands-on exercise reaches for.

**DreamerV2** (Hafner et al. 2021) changed the stochastic state from a Gaussian to a set of *categorical* variables, a vector of, say, 32 one-hot codes with 32 classes each, trained through the straight-through gradient estimator. Categorical latents turned out to model the sharp, multimodal uncertainty of visually rich environments far better than a smooth Gaussian, and DreamerV2 became the first agent to reach human-level Atari scores purely from a world model, matching methods that learned directly from the game.

**DreamerV3** (Hafner et al. 2023) is the version worth remembering, because its contribution is not a new architecture but *robustness*. A single configuration, one fixed set of hyperparameters, trained working agents across more than 150 tasks spanning continuous control, Atari, and 3-D navigation, with no per-task tuning. It got there through a stack of unglamorous engineering fixes: **symlog** compression of rewards and observations so the same network handles wildly different magnitude scales, **KL balancing** and **free bits** to stop the dynamics KL from either collapsing or dominating, and careful normalization of returns. The headline demonstration was collecting a diamond in Minecraft from scratch, a task with a long, sparse chain of prerequisites that had resisted every prior method, without being shown a human doing it. For a practitioner the point is simpler: DreamerV3 is the world-model agent you can drop onto a new pixel-input task and expect to train without a tuning campaign, which is why it anchors this chapter's exercise.

## What the split bought us

Step back and the through-line is one decision made well. Splitting the latent into a deterministic carrier and a stochastic rider let a single model both remember across long horizons and represent what it does not know, and expressing that split as a prior and a posterior turned dynamics learning into a KL term you can optimize by gradient descent. Everything downstream, from PlaNet's planner to DreamerV3's diamond, is built on states produced by that same recurrence. The remaining question is the one PlaNet answered crudely and Dreamer sidestepped by learning a policy: given a trustworthy latent model, how do you actually search it for good actions? That is planning in latent space, and it is next.
