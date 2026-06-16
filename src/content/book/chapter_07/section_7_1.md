---
chapter: 7
section: 7.1
title: "Function approximation: from Q-tables to Q-networks"
target_words: 2000
status: draft
prereqs: §5.2 (value iteration as a contraction); §5.3 (tabular Q-learning, the TD target, off-policy property, the regression-loss form of the deep update); §3.1 (gradients, the chain rule); §3.3 (a PyTorch training loop); §3.5 (what a diverging loss looks like)
key_refs:
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.), Chapters 9-11. MIT Press.
  - Tsitsiklis & Van Roy (1997). An analysis of temporal-difference learning with function approximation. IEEE Transactions on Automatic Control, 42(5):674-690.
  - Baird (1995). Residual algorithms: reinforcement learning with function approximation. ICML.
  - Riedmiller (2005). Neural fitted Q iteration. ECML.
  - Lin (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning, 8(3-4):293-321.
  - Mnih et al. (2013). Playing Atari with deep reinforcement learning. arXiv:1312.5602.
  - Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518:529-533.
  - van Hasselt, Guez & Silver (2016). Deep reinforcement learning with double Q-learning. AAAI.
---

# 7.1  Function approximation: from Q-tables to Q-networks

The Q-learning of §5.3 stored one number for every state-action pair.
That works on a 4×4 gridworld with 16 states and 4 actions: 64
numbers, all visited many times over a few thousand episodes. It does
not work on anything a robot sees. The state of a manipulator with a
camera is an image plus joint angles; the number of distinct states is
effectively infinite, and the agent will never see the same one twice.
A table indexed by state has no row to look up, no row to update, and
nothing to generalize from. The whole apparatus of §5.3 — the TD
target, the off-policy property, the convergence theorem — assumed a
representation that the real problem does not permit.

This section replaces the table with a parametric function. The good
news is that almost everything from §5.3 survives the substitution
syntactically: the update still pushes $Q(s,a)$ toward
$r + \gamma \max_{a'} Q(s',a')$. The bad news is that the theorem does
not survive, and the gap between "the update looks the same" and "the
update still converges" is where deep RL lives. We will set up the
function-approximation view, name the precise way it can blow up, and
then walk through the three engineering fixes that turned it from a
known-divergent idea into DQN, the network that learned to play Atari
from pixels (Mnih et al. 2015).

## A table is just a one-hot linear model

Start by noticing that the table was never special. Storing $Q(s,a)$
as an array is identical to a linear model
$Q_\theta(s,a) = \theta^\top \phi(s,a)$ where the feature vector
$\phi(s,a)$ is a one-hot indicator: a single 1 in the slot for
$(s,a)$, zeros everywhere else. The parameter vector $\theta$ is the
table, flattened. The tabular update of §5.3 is exactly a
gradient-descent step on the squared TD error under those one-hot
features, with the step size $\alpha$ as the learning rate.

Seen this way, function approximation is one change: pick a $\phi$ that
is *not* one-hot, so that updating $\theta$ for one state-action pair
also changes the predicted value of nearby pairs. That sharing is the
entire point. It is what lets the agent say something about a state it
has never visited, by interpolating from states it has. A linear model
with hand-built features ("distance to goal", "gripper-open flag") was
the standard tool through the 1990s. A neural network replaces the
hand-built $\phi$ with a learned one: $Q_\theta(s,a)$ is a multilayer
network, $\theta$ its weights, and the features in the second-to-last
layer are discovered by gradient descent rather than designed. For a
robot taking pixels as input, the network is a convolutional encoder
followed by a head that outputs one Q-value per discrete action.

## The fitted-Q view: RL as a sequence of regressions

The cleanest way to think about training a Q-network is to treat it as
a sequence of ordinary supervised regressions — this is *fitted Q
iteration*, made neural by Riedmiller (2005) under the name NFQ.
Suppose you have a batch of transitions $(s, a, r, s')$. Freeze the
current network, and for each transition compute a regression target

$$
y \;=\; r + \gamma \max_{a'} Q_{\theta^-}(s', a'),
$$

where $\theta^-$ is the *frozen* copy of the weights. Now fit a fresh
network $Q_\theta$ to the dataset $\{((s,a),\, y)\}$ by minimizing
mean-squared error, exactly as in §3.3 — this is supervised learning,
nothing more. Then set $\theta^- \leftarrow \theta$ and repeat. Each
round is a single application of the Bellman optimality operator from
§5.2, projected onto the space of functions your network can
represent.

The loss for one round is the regression objective we previewed at the
end of §5.3:

$$
\mathcal{L}(\theta) \;=\; \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\Bigl[\bigl(r + \gamma \max_{a'} Q_{\theta^-}(s',a') - Q_\theta(s,a)\bigr)^2\Bigr].
$$

Two details in this expression carry all the weight, and both are
easy to skip past. First, the target uses $\theta^-$, not $\theta$:
we do not differentiate through the $\max$ term. The target is treated
as a fixed label even though it depends on the network's own weights.
Pretending it is fixed is what makes the step a regression rather than
something stranger; it is also a lie, and the size of the lie is what
we have to manage. Second, the expectation is over whatever
distribution $\mathcal{D}$ the transitions were drawn from — and that
distribution is not the one the greedy policy induces. We are about to
see why that combination is dangerous.

## The deadly triad

Tabular Q-learning converges. Fitted Q with a neural network can
diverge — the Q-values run off to infinity and the policy becomes
garbage — and it can do so on problems where a table would have been
fine. Sutton & Barto (2018) call the culprit the *deadly triad*: the
simultaneous presence of (1) function approximation, (2) bootstrapping
— building the target from the network's own estimate rather than from
a full Monte Carlo return — and (3) off-policy training, learning
about the greedy policy from data collected by a different,
exploratory one. Any two of the three are usually safe. All three
together can be unstable.

The mechanism is a feedback loop. Because features are shared, raising
$Q_\theta(s,a)$ to hit its target also raises $Q_\theta$ at the very
states $s'$ that appear inside other targets. Those targets move up,
so the next regression chases them, which raises the values again.
With a table the shared-feature step is absent and the loop cannot
form; with on-policy Monte Carlo returns there is no bootstrap to
amplify; but with all three, nothing pins the values down. Baird
(1995) built a tiny seven-state MDP — *Baird's counterexample* — where
linear TD with off-policy updates provably diverges, which is worth
remembering precisely because it is linear: the instability is not a
quirk of deep networks, it is structural. Tsitsiklis & Van Roy (1997)
gave the matching theory, proving convergence for on-policy linear TD
and exhibiting the off-policy divergence. None of this says deep Q-learning
*will* diverge on your problem; it says nothing prevents it, so the
algorithm needs explicit machinery to stay stable.

## DQN: three fixes that made it work

DQN (Mnih et al. 2013; Mnih et al. 2015) is fitted Q-learning plus
three modifications, each one a direct countermeasure to the
instability above. It is worth knowing them individually, because the
same three ideas reappear, lightly disguised, in every off-policy
actor-critic method in §7.4.

**A replay buffer.** Rather than learn from each transition once, in
the order it arrives, DQN stores transitions in a large buffer
$\mathcal{D}$ (a million frames, in the Atari work) and samples
minibatches uniformly from it — an idea due to Lin (1992). This does
two things. It breaks the temporal correlation between consecutive
samples, which a stochastic-gradient step assumes away, and it lets
each expensive transition be reused many times, which matters
enormously on a robot where collecting data means moving real motors.
Replay is the single most important reason off-policy methods are
sample-efficient, and §7.4 leans on it hard.

**A target network.** The moving-target problem — the label depends on
the weights we are updating — is tamed by holding $\theta^-$ fixed for
a stretch (in DQN, a few thousand gradient steps) and only then copying
the live weights into it. The regression now has a stationary target
for the duration of each stretch, which is the condition under which
the fitted-Q view is honest. Later methods replace the periodic copy
with a slow exponential average, $\theta^- \leftarrow \tau\theta +
(1-\tau)\theta^-$ with small $\tau$, but the purpose is identical: slow
the target down so the network is chasing something that is nearly
still.

**A regression loss with clipping.** DQN minimizes the squared TD
error by SGD, clipping the error term to $[-1, 1]$ (equivalently,
using a Huber loss) so that a single wild target cannot produce a
gradient large enough to wreck the weights. This is the same defensive
move as gradient clipping in §3.5, applied at the level of the TD error.

Here is the inner loop, stripped to its essentials and small enough to
read in one pass:

```python
# s, a, r, s2, done: a minibatch sampled from the replay buffer
with torch.no_grad():
    q_next = q_target(s2).max(dim=1).values          # max_a' Q_theta-(s', a')
    y = r + gamma * q_next * (1.0 - done)            # TD target, frozen target net
q_pred = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)
loss = F.smooth_l1_loss(q_pred, y)                   # Huber: clipped regression
opt.zero_grad(); loss.backward(); opt.step()
# every C steps: q_target.load_state_dict(q_online.state_dict())
```

Run this on CartPole and the pole stays up within a few hundred
episodes; run the convolutional version on Atari frames and you
reproduce, with patience, the result that made the field pay attention
in 2013. The behaviour policy is still $\varepsilon$-greedy from §5.3,
now annealed from near-1 down to a small floor, and the off-policy
property from §5.3 is exactly what licenses learning from a buffer full
of transitions collected by stale, more-random versions of the policy.

## The leftover bias, and what this still cannot do

One crack is worth naming because it motivates a later fix. The
$\max_{a'}$ in the target both selects the best next action and
estimates its value from the same noisy network, which biases the
estimate upward — the network's overestimates get selected
preferentially. van Hasselt, Guez & Silver (2016) remove most of this
with *double Q-learning*: use the online network to pick the
argmax action and the target network to score it. It is a two-line
change with a measurable effect, and it is the kind of correction that
distinguishes a Q-network that limps from one that works.

The deeper limitation is the one §5.3 already flagged and this section
has not removed. A Q-network still acts by computing
$\arg\max_a Q_\theta(s,a)$. With a handful of discrete actions —
Atari's eighteen joystick positions — the argmax is a cheap forward
pass over every action. With a robot's continuous action space, a
seven-dimensional vector of joint velocities, the argmax is itself an
optimization problem at every control step, and enumerating actions is
hopeless. That single obstacle is why value-based methods, for all
their sample efficiency, are not the end of the story for control, and
why the next section turns to methods that represent the policy
directly.

The takeaway from this section is the substitution and its price.
Swapping the Q-table for a Q-network buys generalization across an
enormous or continuous state space, at the cost of the convergence
guarantee — the deadly triad means the update that was provably stable
in §5.3 can now diverge. DQN's replay buffer, target network, and
clipped regression loss are the standard countermeasures, and they
recur throughout the chapter. What they do not fix is the argmax over
actions, which §7.2 confronts by abandoning value-only learning and
parameterizing the policy itself.
