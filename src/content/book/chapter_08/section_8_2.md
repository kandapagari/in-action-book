---
chapter: 8
section: 8.2
title: "Decision Transformer: control as conditional sequence modeling"
target_words: 2000
status: draft
prereqs: §8.1 (the transformer, attention, causal masking); §5.1 (states, actions, rewards, returns); §5.2 (value iteration, the Bellman backup we are about to drop); §6.2 (behavior cloning, which this section generalizes); §7.4 (offline actor-critic, the baseline we compare against).
key_refs:
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Janner et al. (2021). Offline reinforcement learning as one big sequence modeling problem (Trajectory Transformer). NeurIPS.
  - Fu et al. (2020). D4RL: datasets for deep data-driven reinforcement learning. arXiv preprint.
---

# 8.2  Decision Transformer: control as conditional sequence modeling

Section 8.1 ended on a reframing: a policy maps a history of tokens to the next action, which is exactly the job a causal transformer was built for. The Decision Transformer (Chen et al., arXiv:2106.01345) is the clean realization of that idea. It throws out the entire apparatus of value functions, Bellman backups, and temporal-difference error that Chapters 5 and 7 spent so long building, and replaces it with a single question: given everything that has happened so far, and a statement of how well we intend to do, what is the next action? Training is ordinary supervised next-token prediction. There is no bootstrapping, no target network, no discount factor to tune. The surprising part is that this works, and on some offline benchmarks it works better than the methods it discards.

## The trick: condition on the return you want

Behavior cloning (§6.2) has an obvious flaw as an offline RL method. If your dataset is a mix of good and bad trajectories, some expert, some fumbling, then imitating all of it averages the experts and the fumblers together, and you get a mediocre policy. Plain cloning has no way to ask for *only the good behavior*, because it has no notion of how good any given action was.

The Decision Transformer's one idea is to give the model that notion, at the input. Alongside each state and action, it feeds a third quantity: the *return-to-go*, the sum of rewards from the current step to the end of the episode,

$$
\hat{R}_t = \sum_{t'=t}^{T} r_{t'} .
$$

Note what this is and is not. It is not the discounted value estimate of a critic; it is just an arithmetic fact computed from the logged data after the episode finished. During training, the return-to-go is known exactly for every step of every trajectory, because the episode is over and we can add up the rewards. So the model is trained to answer: "in states like this, when the trajectory went on to collect a return of $\hat{R}_t$, what action came next?" That is still pure supervised learning, predict the recorded action, but the action is now *conditioned* on the eventual outcome.

The payoff comes at deployment. We no longer have a logged future to sum up, so instead we *specify* the return we want. We prompt the model with a high target return, say, the best return seen in the dataset, or a bit beyond, and let it generate the action that, in its training experience, preceded outcomes that good. Generate the action, execute it, observe the real reward, subtract it from the target to get the new return-to-go, and repeat. The desired return acts as a knob: ask for a high return and the model produces expert-like behavior even if it was trained on a pile of mediocre data, because it has learned to associate high-return prompts with the competent slices of that data. This is the mechanism that lets it beat behavior cloning on mixed-quality datasets.

## The token sequence

Concretely, a trajectory is flattened into a single sequence with three token types per timestep, in a fixed order:

$$
\tau = \big(\hat{R}_1,\, s_1,\, a_1,\; \hat{R}_2,\, s_2,\, a_2,\; \dots,\; \hat{R}_T,\, s_T,\, a_T \big).
$$

Each of the three is mapped to the model's embedding dimension by its own learned linear layer (or a small convolutional stem, if the state is an image), a timestep embedding is added so the model knows which step a token belongs to, and the whole thing goes into a GPT-style causal decoder of the kind §8.1 described. The causal mask is doing real work here: when the model predicts $a_t$, it is allowed to attend to every token up to and including $s_t$, all past returns, states, and actions, plus the current return-to-go and current state, but not to anything later. The training loss is applied only at the action positions: for continuous control, mean-squared error between predicted and recorded actions; for discrete actions, cross-entropy. Returns and states are fed in but not predicted.

The ordering $(\hat{R}, s, a)$ matters and is worth pausing on. Because the return token comes *before* the state and action at each step, the action prediction can attend to the return-to-go directly. The model literally reads "here is how well things are about to go, here is where you are" and then emits an action. Reverse the order and the conditioning signal would arrive too late to inform the action at that step.

## A minimal forward pass

The architecture is almost entirely the §8.1 transformer plus three input heads. The only fiddly part is interleaving the three token streams so they line up in the sequence:

```python
import torch, torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, d=128, K=20):
        super().__init__()
        self.embed_R = nn.Linear(1, d)          # return-to-go
        self.embed_s = nn.Linear(state_dim, d)  # state
        self.embed_a = nn.Linear(act_dim, d)    # action
        self.embed_t = nn.Embedding(K, d)       # timestep
        self.transformer = CausalTransformer(d) # the §8.1 stack
        self.predict_a = nn.Linear(d, act_dim)  # action head

    def forward(self, R, s, a, t):              # each (batch, K, ·)
        te = self.embed_t(t)
        tokens = torch.stack(                    # interleave R,s,a
            [self.embed_R(R) + te,
             self.embed_s(s) + te,
             self.embed_a(a) + te], dim=2
        ).reshape(s.shape[0], 3 * s.shape[1], -1)
        h = self.transformer(tokens)             # causal attention
        h = h.reshape(s.shape[0], s.shape[1], 3, -1)
        return self.predict_a(h[:, :, 1])        # read off state slots
```

Everything that makes this an RL method is in how the data is prepared, computing returns-to-go, choosing a context window `K` of recent steps, not in the network. The network is a language model that happens to be reading robot trajectories.

## Why dropping the Bellman backup is a big deal

It is worth being explicit about what just disappeared. Q-learning and the actor-critic methods of §7.4 estimate a value function by bootstrapping: the value of this state is defined in terms of the estimated value of the next state, which is itself being learned. That self-reference is the source of most of deep RL's instability, the moving targets, the need for target networks and careful learning-rate schedules, the divergence when function approximation, bootstrapping, and off-policy data combine (the "deadly triad" of §7.1). The Decision Transformer sidesteps all of it. Returns-to-go are computed once from fixed data; there is no bootstrapping because there is no value estimate chasing its own tail. The optimization is as stable as training a language model, which after a decade of engineering is very stable indeed. You trade a hard, unstable optimization for an easy, stable one, and let scale and supervised learning do the rest.

This reframing was not unique to one paper. The Trajectory Transformer (Janner et al., 2021), published essentially concurrently, takes the same "offline RL is sequence modeling" stance but uses the transformer as a *model* of the trajectory distribution and plans over it with beam search, the subject of §8.3. The two papers together marked the moment the field started taking "control as next-token prediction" seriously, and they are the conceptual ancestors of every VLA in Part 4 that decodes actions autoregressively.

## What it cannot do, and the honest comparison

The Decision Transformer is not free magic, and the chapter's learning objective asks you to articulate the trade-offs against offline actor-critic methods like the CQL/SAC variants of §7.4. Three caveats matter.

First, it does not *stitch*. Value-based methods can, in principle, combine the good first half of one trajectory with the good second half of another, because the Bellman backup propagates value backward across trajectories that share a state. The Decision Transformer only recombines along sequences it can attend to within its context window; if no single logged trajectory ever connected two regions of the state space, prompting for a high return will not conjure the bridge. On benchmark suites built specifically to require stitching, value-based offline RL still wins.

Second, the return prompt is an approximation. Asking for a return far beyond anything in the data does not reliably extrapolate; performance typically peaks when you prompt for a return near the top of the training distribution and degrades if you ask for the impossible. The knob has a working range, and finding it is its own small tuning problem.

Third, it inherits the coverage limits of all offline learning. If the dataset never shows competent behavior in some region, no amount of return conditioning recovers it.

Against those costs sit real advantages: training stability, no critic to diverge, natural handling of sparse and delayed rewards (a return-to-go of zero until a final success is perfectly easy to sum and condition on), and an architecture identical to the sequence models the rest of the book relies on. On the standard D4RL offline benchmark (Fu et al., 2020), the suite the chapter's hands-on exercise uses, including the `hopper-medium` task, the Decision Transformer is competitive with strong offline actor-critic baselines across most locomotion datasets, and notably robust on the sparse-reward and mixed-quality settings where behavior cloning falls apart.

The deeper point, and the reason this section sits where it does, is architectural rather than numerical. Once control is next-token prediction over $(\hat{R}, s, a)$ tokens, the door is open to swap any of those tokens for something richer: replace the scalar return with a language instruction, replace the low-dimensional state with image patches, and you have walked the short distance from Decision Transformer to RT-1. We make that substitution explicit in §8.4, after the next section shows the other thing you can do with a trajectory model, not just generate from it, but search over it.
