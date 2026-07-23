---
chapter: 8
section: 8.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §8.1–§8.6; Python with PyTorch installed; a working understanding of attention, causal masking, the Decision Transformer and returns-to-go, the Trajectory Transformer, and what gets tokenized; a single offline dataset (D4RL's `hopper-medium` or a logged CartPole buffer is plenty); about two to three hours, most of it training a model small enough to finish on a laptop CPU
key_refs:
  - Chen et al. (2021). Decision Transformer: reinforcement learning via sequence modeling. arXiv:2106.01345.
  - Janner et al. (2021). Offline reinforcement learning as one big sequence modeling problem (Trajectory Transformer). arXiv:2106.02039.
  - Vaswani et al. (2017). Attention is all you need. NeurIPS.
  - Brohan et al. (2023). RT-2: vision-language-action models transfer web knowledge to robotic control. arXiv:2307.15818.
  - Fu et al. (2020). D4RL: datasets for deep data-driven reinforcement learning. arXiv:2004.07219.
---

# 8.x  Hands-on exercise + chapter references

Chapter 8 rests on one claim that is easy to accept on paper and only becomes real once you have trained the model yourself: a transformer with no value function, no Bellman backup, and no notion of optimality can still produce competent actions, purely by predicting the next token of a trajectory the way GPT predicts the next word of a sentence. The drills below make that claim something you have watched happen rather than something you have been told. The headline exercise trains a small Decision Transformer on an offline dataset and then sweeps the target return at test time, so you can see the conditioning knob §8.2 described actually steer behavior. The remaining drills probe the boundaries of the reframing: where return conditioning stops working, what tokenization costs you, and how to read a foundation action model as the same machine scaled up.

The exercises take two to three hours combined. The model is deliberately tiny, a few hundred thousand parameters, two or three attention layers, so a full training run finishes in minutes on a CPU and you can afford to rerun it while you change one thing at a time. After the dependency install nothing here needs internet.

```
pip install torch
```

You need one offline dataset of `(state, action, reward)` trajectories. The cleanest choice is a D4RL task (Fu et al., arXiv:2004.07219) such as `hopper-medium`, which is the dataset the Decision Transformer paper (arXiv:2106.01345) itself reports on; if installing D4RL's MuJoCo dependency is more friction than you want, log a few thousand episodes from a CartPole policy of mixed quality and use that. The lesson does not depend on the task, only on the data being offline and of varied return.

## Exercise 8.x.1 — A Decision Transformer, and the return knob

Build `dt.py` following §8.1 and §8.2. The architecture is the §8.1 transformer with nothing exotic: embed each of the three token types, return-to-go, state, action, into the same width, add a positional (here, timestep) signal, stack two or three causal attention blocks, and put a head on the action positions that predicts the next action. Train with the ordinary supervised loss from §3.4: cross-entropy if your actions are discrete, mean-squared error if they are continuous. There is no target network, no replay-buffer staleness, none of the Chapter 7 machinery; that absence is the point of the chapter, and you should notice how much shorter this training loop is than the SAC loop from Exercise 7.x.2.

Precompute the returns-to-go for every timestep, the sum of rewards from that step to the end of its episode, because that is the quantity the model conditions on. Train until the action-prediction loss plateaus.

Now the actual experiment. At evaluation time you supply the first return-to-go token yourself; it is a request, not a measurement. Roll out the policy in the live environment across a sweep of target returns, say five values from well below the dataset's average return to somewhat above its maximum, and record the *achieved* return at each. Plot requested return on the x-axis against achieved return on the y-axis.

The expected picture, and §8.2's central demonstration, is a band where the two track each other: ask for more, get more. Read the two ends carefully, because they are where the method's nature shows. At the low end the policy will obligingly underperform, proof that you are steering behavior, not just maximizing it. At the high end the curve flattens: ask for a return larger than anything the data contains and the model cannot deliver it, because it only ever learned to imitate trajectories that existed. That ceiling is the honest limit §8.2 named, a return-conditioned imitator cannot invent better-than-demonstrated behavior the way the dynamic programming of §5.2 can stitch one together. Record where your curve stops climbing; that knee is the edge of the dataset.

Wall clock: about forty-five minutes including the sweep.

## Exercise 8.x.2 — Where stitching breaks, and why the planner exists

This drill makes §8.3's policy-versus-model distinction concrete by finding a case the Decision Transformer gets wrong. Construct, or pick from D4RL, a dataset whose good behavior is *fragmented*: no single logged trajectory is high-return end to end, but a high-return path exists if you splice the good first half of one episode onto the good second half of another. The classic illustration is a dataset of trajectories that each do one subtask well and the other badly.

Train your Decision Transformer on it and ask for a high return. It will typically fail to splice, because return-conditioned imitation reproduces whole trajectories and never learned the join. This is the concrete face of the "no stitching" limitation §8.2 and §8.3 kept flagging. If you have the appetite, contrast it with a value-based offline method, even a small CQL or the IQL baseline D4RL ships, which can stitch because it propagates value through the Bellman backup the Decision Transformer threw away. You do not have to implement the planner; the Trajectory Transformer (arXiv:2106.02039) recovers stitching by searching over its own predictions instead, at the cost of inference time §8.3 measured. The deliverable is a short written answer: state precisely what information the Bellman backup carries that return-conditioned supervised learning does not, and why beam search over a trajectory model is a third way to recover it.

Wall clock: about thirty minutes if you reuse 8.x.1's code and skip the value-based baseline; an hour if you run the comparison.

## Exercise 8.x.3 — Tokenize an action, and pay the resolution bill

§8.4 argued that tokenization is a modeling decision with consequences, not a preprocessing chore. Feel the consequence. Take a continuous control task and discretize each action dimension into uniform bins, start with 256 bins, the RT-1-style choice, then retrain the Decision Transformer to predict bin indices with cross-entropy. Sweep the bin count down: 256, 64, 16, 8. Plot achieved return against bin count.

You will watch control quality fall off as the bins coarsen, and the fall-off is not linear; there is a cliff below which the discretization can no longer represent the smooth corrections the task needs, and the policy starts to chatter or stall. That cliff is exactly the wasted-resolution failure §8.4 warned about and the reason FAST (arXiv:2501.09747) and the continuous diffusion and flow-matching heads of Chapter 10 exist. For the written part, estimate the effective control frequency your tokenization implies: if each action costs one token and your model emits tokens at some rate, how fast can the loop close? That back-of-envelope number is the §8.1 cost argument made personal, and it is the constraint Chapter 14's dual-system split is designed to dodge.

Wall clock: about forty minutes including the sweep.

## Exercise 8.x.4 — Read a foundation action model and find the four changes

Pick one Part 4 system from the reading list, RT-1 (arXiv:2212.06817), RT-2 (arXiv:2307.15818), or OpenVLA (arXiv:2406.09246) are the cleanest, and read it with §8.5's recipe in hand. Write a short audit that locates, by figure or paragraph, each of the four changes that turn a Decision Transformer into a foundation action model:

1. **Where did the return become a language instruction?** Find the conditioning input and confirm it is a sentence, not a scalar.
2. **Where did the data scale, and by how much?** Name the dataset and its order of magnitude relative to the few-thousand-episode buffer of Exercise 8.x.1.
3. **Where did the vision-language initialization enter?** Identify the pretrained backbone and what it was trained on before it ever saw a robot.
4. **What is the action head, and how are actions tokenized?** Discrete bins, overloaded vocabulary IDs, or a continuous head, and what §8.4 trade-off that choice accepts.

Then answer one synthesis question: which parts of this system are the recipe and which are the architecture? The §8.6 warning was that conflating the two is how people end up believing attention is load-bearing when it is often just incumbent, and RoboMamba (arXiv:2406.04339), which runs the same recipe on a state space model, is the evidence. The point of the audit is that once you can decompose a VLA into these four moves, every chapter of Part 4 reads as elaborating one of them rather than introducing a new paradigm.

Wall clock: about thirty minutes for the read plus the written audit.

## Chapter 8 reading list

The works below are cited in §8.1–§8.6, grouped by purpose. Full bibliographic entries for everything cited in the book live in Appendix E.2; this is the chapter-local subset.

### The transformer itself

- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*. The scaled dot-product attention §8.1 builds the whole chapter on.
- Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training." OpenAI tech report. GPT; the causal decoder §8.1 says a control policy almost always is.

### Control as sequence modeling

- Chen, L., et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling." arXiv:2106.01345. §8.2's main result and Exercise 8.x.1's algorithm; the return-conditioned imitator whose ceiling Exercise 8.x.1 measures.
- Janner, M., et al. (2021). "Offline Reinforcement Learning as One Big Sequence Modeling Problem" (Trajectory Transformer). arXiv:2106.02039. §8.3's planner; the model-not-policy framing and the beam search that recovers the stitching Exercise 8.x.2 exposes.
- Fu, J., et al. (2020). "D4RL: Datasets for Deep Data-Driven Reinforcement Learning." arXiv:2004.07219. The offline-RL benchmark the exercises draw their datasets from.

### Tokenization and its limits

- Brohan, A., et al. (2022). "RT-1: Robotics Transformer for Real-World Control at Scale." arXiv:2212.06817. The discrete action-binning scheme §8.4 dissects and Exercise 8.x.3 stress-tests.
- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for Vision-Language-Action Models." arXiv:2501.09747. The fix for the wasted-resolution problem Exercise 8.x.3 makes you feel.

### The bridge to foundation action models

- Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818. §8.4's action-token overloading and §8.5's four-change recipe; Exercise 8.x.4's target.
- Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. The open 7B VLA §8.5 points to and Chapter 12 takes apart.
- Liu, J., et al. (2024). "RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation." arXiv:2406.04339. §8.5's state space model alternative; the evidence that the recipe is separable from the transformer.
- Gu, A., & Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." The linear-cost backbone underneath RoboMamba and §8.1's answer to the quadratic-attention bill.

## Chapter summary

Chapter 8 performed the single conceptual move the second half of the book depends on: it recast action generation as next-token prediction and showed that the move is small in mechanism and large in consequence. You can now explain how a transformer produces an action end to end, tokens in, attention as a content-addressed average, causal masking, a prediction head on the action positions, without hand-waving the attention step. You can take a control problem and say what its token sequence is, justifying each tokenization choice against the resolution-versus-frequency trade-off you measured by hand. You can place the Decision Transformer and the Trajectory Transformer against the offline RL of Chapter 5 and name what each gives up: the policy that imitates returns but cannot stitch, the model you plan against but pay inference time for. And you can read a foundation action model as a scaled-up sequence model, locating the four changes, language for return, data at scale, a vision-language initialization, an action head, that §8.5 showed are the whole difference. Part 3 now picks up the generative and world-model machinery that, layered onto this sequence-modeling foundation, produces the smooth, scalable action models of Part 4, beginning, in Chapter 9, with world models and model-based learning.
