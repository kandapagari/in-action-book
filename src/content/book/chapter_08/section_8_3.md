---
chapter: 8
section: 8.3
title: "Trajectory Transformer and beam-search planning"
target_words: 2000
status: draft
prereqs: §8.1 (the transformer, attention, causal masking, autoregressive decoding); §8.2 (Decision Transformer, returns-to-go, control as next-token prediction); §5.1 (states, actions, rewards, returns); §5.2 (value iteration — the planning it replaces); §7.4 (offline actor-critic, the baseline we keep comparing against).
key_refs:
  - Janner et al. (2021). Offline reinforcement learning as one big sequence modeling problem (Trajectory Transformer). arXiv:2106.02039.
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Fu et al. (2020). D4RL: datasets for deep data-driven reinforcement learning. arXiv:2004.07219.
---

# 8.3  Trajectory Transformer and beam-search planning

Section 8.2 used a transformer one way: train it to predict the next
action, then at deployment hand it a target return and read off whatever
action it emits. The Trajectory Transformer (Janner et al.,
arXiv:2106.02039) uses the same raw ingredient — a causal transformer
over robot trajectories — for a different job. Instead of asking the
model "what action comes next?", it asks "what is the probability of
this whole future?", and then *searches* for a high-value future the
way a language model searches for a high-likelihood sentence. The
Decision Transformer is a policy; the Trajectory Transformer is a model
of the trajectory distribution that you plan against. The distinction is
the whole point of this section, and it is the cleanest example in the
book of the difference between a policy and a model.

## Two ways to use the same network

It helps to hold the two papers side by side, because they were
published within weeks of each other and are constantly confused.

Both flatten a trajectory into a token sequence and both train with
ordinary next-token prediction. The Decision Transformer predicts only
the action tokens and conditions on a return-to-go you supply at test
time; the policy *is* the forward pass. The Trajectory Transformer
predicts *everything* — next state, next reward, next return, next
action — so the trained model is a joint distribution over how
trajectories unfold. A joint distribution by itself does not tell you
what to do. To get an action, you sample many candidate futures from the
model and keep the one that scores best. That search step is where the
planning happens, and it is what §5.2's value iteration did with a known
dynamics model: look ahead over possible futures and pick the action
that leads somewhere good. The Trajectory Transformer just learned the
dynamics model from offline data instead of being handed it.

So the slogan "offline RL as one big sequence modeling problem" cashes
out two different ways. In §8.2 it meant *amortize* the planning into a
return-conditioned policy. Here it means *learn the model and plan
explicitly at test time*. Same architecture, opposite ends of the
classic model-free / model-based split.

## Discretizing the continuous

The first concrete obstacle is that a `hopper` state is a vector of
real numbers, not a word in a vocabulary, and beam search is an
algorithm over discrete tokens. The Trajectory Transformer's answer is
blunt: discretize every dimension. Each continuous coordinate of the
state, each action dimension, the reward, and the return are quantized
independently — the paper uses per-dimension binning, so each of the,
say, eleven state dimensions of `hopper` becomes one token drawn from a
fixed vocabulary of bins. A single timestep is then a short run of
tokens,

$$
\big(\underbrace{s_t^1, s_t^2, \dots, s_t^N}_{\text{state dims}},\;
\underbrace{a_t^1, \dots, a_t^M}_{\text{action dims}},\; r_t,\; \hat{R}_t\big),
$$

and a trajectory is these runs concatenated end to end. Training is
plain cross-entropy next-token prediction over this vocabulary, exactly
as for text.

To make the bin count concrete: the paper quantizes each dimension into
roughly a hundred bins. For `hopper`, with eleven state dimensions,
three action dimensions, plus a reward and a return token, one timestep
costs sixteen tokens, all drawn from a shared vocabulary of about a
hundred symbols. A trajectory of thirty steps is then almost five
hundred tokens — comfortably inside a transformer's context window, but
a useful reminder of why looking ahead a hundred steps is not free: the
horizon you can plan over is the context window divided by sixteen.

The discretization has a cost and a benefit worth naming. The cost is
resolution: a binned action is coarser than the raw float, and you spend
sequence length — one token per dimension per step — which limits how
far you can look ahead within a fixed context window. The benefit is
that the model now expresses a full, multimodal distribution over each
dimension instead of regressing to a single mean. If, from some state,
two genuinely different actions are both reasonable, a regression head
(the mean-squared-error action head of §8.2) splits the difference and
outputs the average, which may be a bad action belonging to neither
mode. A softmax over bins can put probability mass on both and let the
search pick one. This is the same multimodality argument that will
return, in continuous form, when we reach diffusion and flow-matching
action heads in Chapter 10.

## Beam search as planning

With the model in hand, control becomes a search over token sequences.
Beam search is the standard decoding algorithm from machine
translation: keep a "beam" of the *B* most promising partial sequences,
extend each by one token, score the extensions, prune back to the best
*B*, and repeat. In translation you score by accumulated log-likelihood,
because you want the most probable sentence. The Trajectory
Transformer's one modification is the scoring function. Likelihood alone
would just reproduce the behavior policy — the most *probable* future is
whatever the dataset did on average, which is the behavior-cloning
failure mode again. Instead, sequences are scored by the cumulative
reward (or predicted return-to-go) of the future they describe. The beam
fills with futures the model believes are both plausible *and*
high-value, and the first action of the best surviving beam is the
action you execute.

The loop is the receding-horizon pattern from classical model-predictive
control: plan a short horizon, take the first action, observe the real
next state, and replan from there.

```python
def plan(model, state, beam_width=64, horizon=15):
    beams = [(tokenize(state), 0.0)]          # (sequence, value)
    for _ in range(horizon):
        candidates = []
        for seq, value in beams:
            for nxt in model.sample_next_tokens(seq, k=beam_width):
                r_hat = decode_reward(nxt)     # model's predicted reward
                candidates.append((seq + nxt, value + r_hat))
        beams = top_k(candidates, beam_width)  # prune to best B by value
    best_seq, _ = max(beams, key=lambda x: x[1])
    return first_action(best_seq)              # execute, then replan
```

Two details make this honest rather than hand-wavy. First, the model
predicts rewards and returns as tokens, so `decode_reward` is reading the
network's own estimate of how good a step is — the search is only as
good as the learned model, and a model that overestimates reward in
unseen regions will plan toward fantasies. Second, scoring partial beams
by the predicted return-to-go, not just summed rewards so far, lets the
search value a state for what it leads to rather than only what it has
collected — the same reason value functions exist in §5.2. The
Trajectory Transformer essentially folds an approximate value estimate
into the model's own predictions.

## Why bother, given the Decision Transformer exists?

If §8.2's policy is simpler — one forward pass, no search — why carry the
heavier machinery here? Three reasons, and they map onto the chapter's
objective of articulating trade-offs.

The first is *stitching*, the exact weakness §8.2 admitted. Because the
Trajectory Transformer searches over futures rather than emitting a
single conditioned action, it can chain the good segment of one logged
trajectory onto the good segment of another whenever its learned
dynamics make the join plausible. On the sparse, long-horizon tasks
where stitching matters most — the AntMaze navigation tasks in the D4RL
benchmark (Fu et al., arXiv:2004.07219), where reaching the goal
requires composing route fragments no single demonstration walked
end-to-end — the planning approach is markedly stronger than a
return-conditioned policy.

The second is that beam search gives you a knob the Decision Transformer
lacks: spend more compute at test time and get a better plan. Widen the
beam or lengthen the horizon and the action quality improves, up to the
limits of the model. This is the same test-time-compute trade-off that
later becomes central to large-model reasoning, appearing here, in 2021,
for control.

The third is conceptual and is the reason this section sits between
Decision Transformer and the tokenization discussion of §8.4. Once you
accept that a transformer can *model* trajectories, planning, prediction,
and even system identification all become decoding problems over the
same network. The model that plans your next action can also roll out a
hypothetical future for inspection, which is a short step from the world
models of Chapter 9.

## The costs, stated plainly

Beam search is not free. Every control step requires sampling and
scoring hundreds of token continuations through a transformer, which is
orders of magnitude slower than the Decision Transformer's single
forward pass — a serious problem for anything approaching real-time
control, and a recurring tension we will quantify when we reach latency
budgets in Chapter 14. The per-dimension discretization caps spatial
resolution and burns context length, so very high-dimensional
observations (raw images, for instance) do not tokenize gracefully this
way; this is one reason image-based VLAs in Part 4 lean on the Decision
Transformer's amortized-policy style or on diffusion heads rather than
on explicit beam search. And like every method in this chapter, it is
bounded by the offline data: the learned model hallucinates outside its
coverage, and a planner that trusts those hallucinations will
confidently plan into them.

The takeaway is not that one approach beats the other but that the same
sequence-modeling substrate supports both a fast amortized policy and a
slow deliberate planner, and the right choice depends on whether you can
afford to think at test time. With both the policy view and the model
view of trajectory transformers on the table, the open question is what
exactly we should be turning into tokens — scalars, states, actions,
returns, or words — which is precisely what §8.4 takes up as it builds
the bridge from these RL-flavored sequence models to the
language-conditioned VLAs of Part 4.
