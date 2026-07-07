---
chapter: 9
section: 9.3
title: "Planning in latent space"
target_words: 2000
status: draft
prereqs: §9.1 (world model = learned transition/reward; planning is one of its three uses), §9.2 (RSSM prior for rollouts; PlaNet planned with the cross-entropy method, Dreamer replaced it with a learned policy), §5.1 (returns and the discount factor), §7.1 (value functions as bootstrap targets).
key_refs:
  - Hafner et al. (2019). Learning Latent Dynamics for Planning from Pixels (PlaNet). ICML.
  - Chua et al. (2018). Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS). NeurIPS.
  - Williams et al. (2017). Information-Theoretic MPC for Model-Based Reinforcement Learning (MPPI). ICRA.
  - Hansen et al. (2022). Temporal Difference Learning for Model Predictive Control (TD-MPC). ICML.
  - Schrittwieser et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero). Nature 588.
---

# 9.3  Planning in latent space

Section 9.2 ended with a trustworthy latent model and a pointed question:
given a model that can roll the future forward, how do you search it for
a good action? Dreamer answered by not searching at all — it trained a
policy inside imagination and, at deployment, ran one forward pass.
PlaNet answered the other way, with no policy and a fresh search at every
control step. This section is about that second answer, because the
search machinery is worth understanding on its own. It is the bridge
between the learned dynamics of this chapter and the model-predictive
control of §4.4, and it is the reason a world model can act well the
first time it is asked, before any policy has been trained.

## The receding-horizon loop

Planning with a model means solving a small optimization problem, over
and over. At the current latent state $s_t$, you look for the sequence of
actions $a_{t:t+H}$ that maximizes predicted return over a horizon of $H$
steps:

$$
a_{t:t+H}^\star = \arg\max_{a_{t:t+H}} \; \mathbb{E}\!\left[\sum_{k=0}^{H-1} \gamma^k \, \hat{R}_\theta(s_{t+k}, a_{t+k})\right],
\qquad s_{t+k+1} \sim \hat{P}_\theta(\cdot \mid s_{t+k}, a_{t+k}).
$$

Both $\hat{P}_\theta$ and $\hat{R}_\theta$ are the learned model from
§9.2 — the RSSM prior rolling latents forward and the reward head reading
them off. You solve for the whole sequence, then execute only the *first*
action $a_t^\star$. The world advances one step, you observe the new
frame, encode it to a fresh latent through the posterior, and solve the
problem again from there. This is model-predictive control (MPC): plan a
horizon, commit one step, replan. The name "receding horizon" describes
the effect — the planning window slides forward with you, always looking
$H$ steps ahead and never further.

Two features of this loop matter before we look at how the $\arg\max$ gets
solved. First, replanning at every step is what makes MPC robust to a
mediocre model. A single fifteen-step forecast will drift; but you never
trust step fifteen, because by the time you get there you have replanned
fourteen times, each time correcting on a real observation. The horizon
buys foresight, the replanning buys correction, and the combination
tolerates a model that would be useless if you followed its full
prediction blindly. Second, the whole loop lives in latent space. Nothing
here renders a pixel. The planner scores candidate futures by the rewards
the model predicts for latent states, exactly as §9.1 promised — the
model must predict whatever the search needs, and here it needs a reward,
not a picture.

## Sampling-based planners: shoot, weight, repeat

The optimization is nasty. The objective is non-convex, the dynamics are
a deep network, and the action sequence is high-dimensional. The methods
that work best in practice do not compute gradients at all; they sample.

The crudest version is **random shooting**. Draw a few hundred action
sequences at random, roll each through the model, sum the predicted
rewards, and keep the single best sequence. It is trivial to implement
and embarrassingly parallel — every rollout is independent, so a GPU
evaluates the whole batch at once. It is also wasteful, because most
random sequences are garbage and the good region of action space never
gets sampled densely.

The **cross-entropy method** (CEM), which PlaNet used, fixes that by
iterating. Sample a batch of sequences from a Gaussian, roll them all
out, keep the top-scoring fraction — the "elites," typically the best ten
percent — and refit the Gaussian's mean and variance to just those
elites. Sample again from the tightened distribution and repeat a handful
of times. The distribution walks toward the high-reward region and
narrows around it. PlaNet ran something like a thousand candidate
sequences, ten elites, a few refinement iterations, at every control
step. The cost is real: hundreds to thousands of model rollouts per
action. That cost is exactly what Dreamer eliminated by amortizing the
search into a policy, and it is why online planning is reserved for
settings where a slow, deliberate decision is worth paying for.

**MPPI** (model-predictive path integral control; Williams et al. 2017)
is the variant you meet most often on real hardware. Instead of a hard
cutoff between elites and the rest, it keeps every sampled sequence and
weights it by the exponential of its return, $w_i \propto \exp(\eta \,
\hat{G}_i)$, then sets the next mean to the reward-weighted average of all
samples. A good sequence pulls the mean strongly, a bad one barely at all,
and nothing is thrown away. MPPI tends to produce smoother action
sequences than CEM's hard selection, which matters for a physical robot
whose motors dislike jerky commands, and it underlies a good deal of
real-time model-based control on drones and manipulators. Chua et al.'s
PETS (2018) is the other reference point here: it paired CEM planning with
an *ensemble* of probabilistic dynamics models, and showed that
model-based control could match model-free RL on continuous benchmarks
with a fraction of the data — the same efficiency argument §9.1 made for
world models, demonstrated with explicit planning rather than a learned
policy.

## The horizon problem, and the value bootstrap that fixes it

A short horizon is cheap but myopic. Plan fifteen steps ahead and the
planner is blind to any reward that arrives on step sixteen; a robot that
must cross a room to reach a reward will never see the point of the first
step if the room is twenty steps wide. A long horizon sees the reward but
is expensive and, worse, compounds model error — each predicted step
feeds the next, and small mistakes snowball into a fantasy by the time the
horizon runs out.

The fix is one of the more elegant ideas in model-based control: cap the
rollout at a short horizon and staple a **learned value function** onto
the end. Instead of summing rewards to the end of the episode, you sum
$H$ steps of predicted reward and then add $\gamma^H \hat{V}(s_{t+H})$ —
the value function's estimate of everything that happens after the
horizon. The value acts as a learned summary of the far future, so a
five-step plan can still account for a reward a hundred steps away,
because the value at step five already knows it is coming. This is where
planning and learning stop being alternatives and start collaborating:
you plan over the near term where the model is reliable, and you trust a
learned value for the long term where rollouts would drift.

**TD-MPC** (Hansen et al. 2022) is the clean instantiation. It learns a
latent dynamics model, a reward model, and a value function together,
then at each step runs MPPI-style planning over a short latent horizon
with the value bootstrapping the tail. Crucially, its model is trained
only to predict rewards and values — not to reconstruct observations —
which lets the latent space discard everything irrelevant to control.
TD-MPC and its successor TD-MPC2 (Hansen et al. 2024) are, as of this
writing, among the strongest methods on continuous-control benchmarks,
and TD-MPC2 in particular showed a single model handling dozens of tasks
across different embodiments — the model-based echo of the generalist
ambition that drives the VLAs of Part 4.

## Gradient-based planning, and why it is the road less taken

The RSSM is differentiable end to end, so an obvious alternative to
sampling is to compute $\partial \hat{G} / \partial a_{t:t+H}$ by
backpropagating the predicted return through the rollout, then do gradient
ascent on the action sequence directly. This is exactly the mechanism
Dreamer used to train its actor (§9.2). For *planning* at test time,
though, it is used less than you might expect. Backpropagating through
many steps of a recurrent model runs into the same exploding and
vanishing gradients that plague any long unrolled network, and the loss
surface over raw actions is riddled with poor local optima that a
gradient walk falls into and a population of samples escapes. The
practical rule of thumb: gradients are excellent for training a policy
offline, where you average over many trajectories and can afford to be
careful, and sampling is more robust for planning online, where you need
one good sequence right now from one state. Some systems split the
difference — seed a sampler with a gradient step, or vice versa — but the
workhorses in latent planning remain CEM and MPPI.

## Tree search over a learned model: MuZero

Everything so far assumed a continuous action space and a shooting-style
search. When actions are discrete and the payoff for deep lookahead is
large — board games, some strategic tasks — the planner of choice is
Monte Carlo tree search, and the landmark system is **MuZero**
(Schrittwieser et al. 2020). MuZero learns a latent model in the spirit of
this chapter, then plans by building a search tree over latent states:
from the current latent, it expands actions, predicts the resulting latent
and its value with the model, and uses those predictions to guide which
branches to explore deeper. What made it notable is what its model does
*not* predict. It never reconstructs the board or the screen; it is
trained only to make its reward, value, and policy predictions match
reality after each imagined step. The latent is whatever internal state
makes planning accurate, nothing more. That is the same lesson TD-MPC
learned in the continuous world — a world model for planning should model
consequences that matter to the decision, not the appearance of the
scene — and it is the design that §9.4 will push in a very different
direction by asking a model to predict appearance in full.

## The failure mode every planner shares

One warning ties the section together. A planner is an optimizer pointed
at a learned model, and an optimizer will find whatever the model rewards
— including the model's mistakes. If the dynamics network wrongly predicts
that driving the gripper into the table yields high reward, the planner
will gleefully propose exactly that, because it is optimizing the model,
not the world. This is model exploitation, and it is the model-based
analogue of the reward-hacking problem from §5.4. The mitigations are the
recurring ones: keep horizons short so errors have less room to compound,
bootstrap with a value learned from real returns, and quantify the model's
uncertainty so the planner can be penalized for wandering into states the
model has never seen — the ensemble in PETS exists for precisely this
reason. A planner is only ever as good as the model is honest, and honesty
degrades the further from the data you push it.

Latent planning, then, is a powerful way to turn a one-step predictor into
a decision-maker without training a policy — but everything it can do is
bounded by how far into the future the model stays trustworthy, and in
the visually rich worlds a robot actually inhabits, that horizon is short.
The response has been to build models that predict the world in far
greater fidelity, all the way to raw video, and that is where we turn
next.
