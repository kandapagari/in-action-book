---
chapter: 6
section: 6.2
title: "Behavior cloning, step by step"
target_words: 2000
status: draft
prereqs: §3.3 (PyTorch training loop); §5.1 (states, actions, policies); §6.1 (why imitation, what BC is in one sentence)
key_refs:
  - Pomerleau (1988). ALVINN: An Autonomous Land Vehicle in a Neural Network. NeurIPS.
  - Mandlekar et al. (2021). What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. arXiv:2108.03298.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
---

# Behavior cloning, step by step

Section 6.1 made the economic case for demonstrations. This section makes the algorithm concrete. Behavior cloning is the simplest member of the imitation family, and its simplicity is exactly the point: it is supervised learning, the machinery you already built in Chapter 3, applied to a dataset whose labels happen to be expert actions. There is no environment interaction during training, no reward, no value function. If you can train an image classifier, you can train a BC policy. The subtleties, and there are several, live in the choices you make around that supervised core: how observations are encoded, how actions are represented, which loss you minimize, and how you decide whether the result is any good.

We walk through those choices in the order you would face them on a real project, using a running example: training a policy to lift a block off a table, with the robomimic `lift` dataset from the Mandlekar et al. (2021) study (arXiv:2108.03298) as the concrete dataset. It contains 200 teleoperated demonstrations of a simulated Franka arm lifting a cube, small enough to train in minutes, real enough to exhibit every failure mode this chapter cares about.

## The objective, formally

Let the demonstration dataset be

$$
\mathcal{D} = \{(o_i, a_i)\}_{i=1}^{N},
$$

where each pair is an observation and the action the expert took when seeing it. The pairs come from slicing expert trajectories into individual time steps; a single 15-second demonstration at 20 Hz contributes 300 pairs. Behavior cloning fits a parameterized policy $\pi_\theta(a \mid o)$ by maximizing the likelihood of the expert's actions:

$$
\theta^\ast = \arg\max_\theta \sum_{i=1}^{N} \log \pi_\theta(a_i \mid o_i).
$$

That is the entire algorithm. When the policy is a Gaussian with fixed variance, maximizing log-likelihood reduces to minimizing mean squared error between predicted and demonstrated actions, which is why much of the literature, and most quick-start code, just writes `mse_loss`. Keep the likelihood view in mind anyway; it becomes important the moment a single Gaussian is the wrong distributional choice, which on real robot data is most of the time. We return to that below.

Notice what the objective does *not* contain: any term involving what happens when the policy's actions are executed. BC treats the dataset as i.i.d. samples from some distribution over observations, exactly as image classification treats photos. The dataset is not i.i.d.; it is a set of trajectories, and at deployment time the policy's own actions determine which observations it sees next. That mismatch is the deepest problem in imitation learning, and it gets its own section (§6.3). For now we make the standard move of ignoring it, which works better than it has any right to, provided the data is plentiful and the horizon is short.

## Step 1: know what is in your dataset

A teleop dataset is a set of trajectories, each a time-indexed record of observations and actions. For robomimic `lift`, one time step contains: an 84×84 RGB image from a front camera, a second image from a wrist camera, the 7-dimensional joint configuration, the end-effector pose, the gripper aperture, and the 7-dimensional action the operator commanded (6-DoF end-effector velocity plus gripper open/close). Multiply by roughly 50 steps per demonstration and 200 demonstrations: about 10,000 training pairs.

Before writing any model code, look at the data. Plot action distributions per dimension. You will find the gripper channel is nearly binary and the translation channels are roughly zero-mean with occasional large spikes. Replay a few demonstrations and watch them. The Mandlekar study found that *demonstrator quality*, whether the data came from one practiced operator or many inconsistent ones, moved task success by tens of percentage points, more than most architectural choices. A BC policy is a mirror: it reproduces the dataset's habits, including hesitation, retries, and the operator's idiosyncratic approach angles. No later step compensates for confused data.

## Step 2: choose the observation encoding

The policy needs a function from raw observations to a feature vector. Three standard options, in increasing order of capability and cost:

**Low-dimensional state.** Concatenate joint angles, end-effector pose, and object poses (available in simulation) into one vector and feed an MLP. Trains in minutes, useless on a real robot where object poses are not directly observable.

**Visual encoder from scratch.** A small CNN over the camera images, trained jointly with the policy head. This is what most robomimic baselines do, and at the 200-demonstration scale it works because the visual variety is low: one table, one cube, one lighting condition.

**Pretrained visual backbone.** A frozen or fine-tuned encoder pretrained on internet-scale data is the route every modern VLA takes. OpenVLA (arXiv:2406.09246) uses fused SigLIP and DINOv2 features feeding a 7B-parameter language-model backbone. The pretrained option buys robustness to visual variation your demonstrations never covered, and Chapter 11 traces how this became the default recipe.

For the running example, a four-layer CNN per camera, features concatenated with proprioception, is plenty.

## Step 3: choose the action representation

This choice matters more than newcomers expect, and Chapters 10–13 are in large part a study of its consequences. The options:

**Continuous regression.** The network outputs a 7-vector directly; train with MSE. Simplest, and the right starting point for `lift`.

**Discretized tokens.** Bin each action dimension (RT-1, arXiv:2212.06817, uses 256 bins per dimension) and predict bins with a cross-entropy loss. This turns control into classification, which plays well with transformer architectures and, critically, can represent multimodal action distributions, since a softmax over bins can place mass on two distant bins at once.

**Action chunks.** Predict the next $k$ actions (say, 16 steps) as one output instead of one step at a time. Chunking reduces compounding error by cutting the number of decisions per episode, and smooths control; it is standard in ACT and Diffusion Policy (Chapter 10).

**Generative heads.** Represent $\pi_\theta(a \mid o)$ with a diffusion or flow-matching model that can sample from arbitrarily complex action distributions. This is the π0 route (arXiv:2410.24164) and the subject of Chapter 13.

Why so much machinery for what regression handles in one line? Because demonstrations are multimodal. Suppose half your demonstrators pass left around an obstacle and half pass right. MSE-trained regression predicts the average, straight into the obstacle. Each option above the first is, at heart, a way of representing "left or right, but not the mean" as a distribution. On `lift` the demonstrations are single-moded enough that plain regression works; on almost any longer-horizon real-robot dataset, it eventually does not.

## Step 4: the training loop

With encoder and action head chosen, the loop is the one from §3.3 with the labels swapped:

```python
policy = BCPolicy(obs_encoder, action_dim=7)
opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for obs, act in loader:            # (B, ...), (B, 7)
        pred = policy(obs)             # (B, 7)
        loss = F.mse_loss(pred, act)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

Three practical notes save a day each. Normalize actions per dimension to zero mean and unit variance using *dataset* statistics, and store those statistics with the checkpoint; at deployment the network's outputs must be un-normalized with the same numbers, and a mismatch produces a policy that moves confidently in the wrong direction. Weight the gripper dimension, or give it its own binary cross-entropy head: it is one of seven dimensions, but failing to close it is 100% of task failure. Use augmentation (random crops, color jitter) on images too; with 10,000 frames of one table, the encoder will otherwise memorize pixels.

## Step 5: evaluation, the part that surprises people

Here is the trap: validation loss on held-out (observation, action) pairs is only weakly correlated with task success. The Mandlekar study documents checkpoints whose validation MSE is nearly identical while their success rates differ by 20 points, and the lowest-validation-loss checkpoint is rarely the best policy. The reason is the i.i.d. fiction from the objective: validation pairs are drawn from *expert* trajectories, but at deployment the policy visits its own states. A policy can match the expert almost everywhere and still drift into states the dataset never covered, where its behavior is undefined.

The only evaluation that counts is the rollout: execute the policy from fresh initial conditions and measure success over enough episodes to get a meaningful estimate. Fifty is a common floor, and Chapter 15 treats the statistics properly. The practical consequence is that checkpoint selection must be done by rollout, in simulation if you have one. This is the first place BC's "it's just supervised learning" slogan breaks down. The training is supervised; the evaluation is not.

Run the full recipe on `lift`, CNN encoder, MSE regression, 200 demonstrations, rollout-based checkpoint selection, and you get a policy succeeding roughly 90% of the time, matching the robomimic baseline. Scale the same recipe up, more data, bigger encoder, discretized actions, and you arrive, with surprisingly few new ideas, at RT-1: 130,000 episodes, 35M parameters, over 700 tasks, and the same maximum-likelihood objective we wrote at the top of this section.

What the recipe does not fix is the drift problem we waved away after the objective: each small imitation error moves the policy slightly off the expert's distribution, where errors get larger, which moves it further off. Section 6.3 quantifies that feedback loop, and shows why its cost grows quadratically with the horizon.
