---
chapter: 9
section: 9.1
title: "What is a world model, really"
target_words: 2000
status: draft
prereqs: §4.3 (forward vs. inverse dynamics), §5.1 (states, actions, transition function), §7.1 (function approximation), §8.4 (what gets tokenized). Helpful but not required, §3.2 for the probability notation.
key_refs:
  - Sutton (1991). Dyna, an integrated architecture for learning, planning, and reacting. SIGART Bulletin 2(4).
  - Ha & Schmidhuber (2018). World Models. NeurIPS.
  - Hafner et al. (2023). Mastering diverse domains through world models (DreamerV3). arXiv preprint.
  - LeCun (2022). A path towards autonomous machine intelligence. OpenReview position paper.
---

# 9.1  What is a world model, really

Chapter 8 left the policy doing all the work. A Decision Transformer reads a trajectory and predicts the next action; nothing in it ever asks what that action *does*. The model has no opinion about the future beyond the action it is about to emit. That is fine when you have enough demonstrations to imitate, and it is the design every VLA in Part 4 inherits. But it throws away a capability that classical control took for granted: the ability to answer "what happens if I do this?" before committing to it.

A world model is the machine that answers that question. Give it the current state and a candidate action, and it predicts the next state, and usually the reward, and sometimes the raw observation that the robot's cameras would return. It is, in one phrase, a learned simulator of the environment, trained from the agent's own experience rather than written by hand. Everything else in this chapter is a variation on that idea: how to learn the simulator (§9.2), how to use it to choose actions (§9.3), how to scale it to video (§9.4), and whether it should replace the policy-only VLA entirely (§9.5).

## The function you are trying to learn

Recall the transition function from the MDP of §5.1. The environment moves according to

$$
s_{t+1} \sim P(s_{t+1} \mid s_t, a_t), \qquad r_t = R(s_t, a_t).
$$

In a real robot you do not have $P$ or $R$ in closed form. A world model is a parametric approximation of them, call it $\hat{P}_\theta$ and $\hat{R}_\theta$, fit by watching transitions $(s_t, a_t, s_{t+1}, r_t)$ roll past and minimizing prediction error. That is the entire definition. A world model is supervised learning on the dynamics, where the labels come for free because the next state is simply the thing that happened next.

This connects directly to §4.3, where we wrote the forward dynamics of a manipulator as $\ddot{q} = M(q)^{-1}(\tau - C(q,\dot q)\dot q - g(q))$ and called it a model of the robot. That equation *is* a world model, an analytic, hand-derived one, valid for the rigid body and useless the moment you point a camera at a pile of laundry. The learned world models in this chapter are the same object built by a different route: instead of deriving the dynamics from physics, you regress them from data. The trade is the usual one. The analytic model is exact within its assumptions and blind outside them; the learned model is approximate everywhere and, with enough data, approximate in places physics gives you nothing.

## A model predicts; a policy acts

It is worth being pedantic about the distinction, because the rest of Part 3 and Part 4 hinges on it. A *policy* is a function from state to action, $\pi(a \mid s)$, "what should I do?" A *world model* is a function from state and action to next state, $\hat{P}(s' \mid s, a)$, "what would happen if I did that?" They point in opposite directions. You can hold one without the other. A chess engine that evaluates positions but cannot move is missing a policy; a behavior-cloned arm that grasps cups but cannot tell you what the cup will do once grasped is missing a world model.

The reason to want the second function is that it buys you something the first cannot: the ability to try actions without paying for them. A policy commits in the real world, where mistakes cost broken grippers and reset time. A world model lets you commit in imagination, roll the consequences forward, and only then act. The three uses of a world model are exactly the three ways to spend that imagined experience.

## Three things a world model is for

The first use is **planning**. Given a model, you can search over action sequences, simulate each one, and pick the one whose predicted trajectory scores best, model-predictive control, but with a learned model instead of a hand-derived one. Section 9.3 develops this; the short version is that planning turns a one-step predictor into a decision-maker without ever training a policy.

The second use is **learning a policy inside the model**, training in imagination. This is the oldest idea in the chapter and it has a name: Dyna, from Sutton (1991). The agent collects a little real experience, fits a model to it, and then generates large quantities of synthetic experience by rolling the model forward, treating those synthetic transitions as if they were real and feeding them to an ordinary RL update. Real data is expensive on a robot; model rollouts are nearly free. Dreamer (§9.2) is Dyna scaled up with a deep latent model and a neural policy, and it is the reason the technique matters again.

The third use is **prediction as an end in itself**, sometimes called pure world-model evaluation. Here you never plan and never train a policy in the model; you simply ask how good the model is at forecasting, on the bet that a system that can predict the world has learned something worth reusing. The video-prediction models of §9.4, and LeCun's (2022) argument that prediction is the core of intelligence, live here. A model that can imagine the next two seconds of a kitchen has, in some sense, understood kitchens, whether or not it ever picks anything up.

## The canonical example: a controller trained inside a dream

The cleanest illustration is Ha and Schmidhuber's *World Models* (2018), which is worth walking through because every later system in this chapter is a refinement of its three pieces.

The task is a 2-D car-racing game with pixel observations. The system has three components. A **vision model**, a variational autoencoder, compresses each 64×64 frame into a small latent vector $z_t$, call it roughly 30 numbers instead of roughly 12,000 pixels. A **memory model**, a recurrent network with a mixture-density output head, predicts the next latent: $\hat{P}(z_{t+1} \mid z_t, a_t, h_t)$, where $h_t$ is the RNN's hidden state carrying the history. This is the world model proper; it lives entirely in latent space and never touches a pixel. Finally a tiny **controller**, a single linear layer mapping $(z_t, h_t)$ to an action, is the policy.

The striking result was not the racing score. It was that the controller could be trained *entirely inside the memory model's hallucinated rollouts*, the agent learned to drive in its own dream, then transferred to the real game with little loss. That is the Dyna loop made vivid: the world model became a free simulator, and the policy never needed the expensive environment during its own training.

Two design choices from this example recur for the rest of the chapter. First, **predict in latent space, not pixel space.** Forecasting raw images is wasteful, most of the bits in a frame are irrelevant to what the agent should do, and a model that spends capacity rendering the exact texture of the asphalt has spent it badly. Compress first, predict in the compressed space. Section 9.2's RSSM is a more capable version of exactly this two-stage idea. Second, **the model carries memory.** The world is not Markov in its observations, a single frame does not tell you the car's velocity, so the model maintains a recurrent state that makes the prediction Markov in the *latent*. We return to why that matters in §9.2.

## What a world model is not

Three clarifications, because the term gets stretched.

A world model is not necessarily generative in the sense of producing watchable video. The car-racing memory model never renders a frame; it predicts the next 30-number latent and that is enough to plan with. Pixel-perfect video generation, as in Genie (§9.4), is one option on a spectrum, and often an expensive one. What the model must predict is whatever the downstream use needs: a reward for planning, a latent for policy training, a full frame only if a human or a perception module has to read it.

A world model is not a policy with extra steps. You can derive a policy from a model by planning, but the model itself expresses no preferences. This is precisely the dividing line in the §9.5 debate: the VLAs of Part 4 are policies that learned to act by imitation and hold no explicit model of consequences, while the world-model camp argues that learning consequences first is the more data-efficient and more general bet. Neither side disputes the definitions; they disagree about which function is worth learning.

And a world model is not the same as the simulator you set up in Appendix D. A physics simulator like MuJoCo is a hand-built world model with privileged access to the true state, it knows every joint angle and contact force because you told it the equations. A learned world model has to infer all of that from observations, which is harder and also the entire point: it can model things, like cloth or granular media or a human's next move, that you cannot conveniently write down in a simulator's configuration file.

With the object defined, the obvious question is how to learn it well enough to plan with. The dominant answer for the last several years has been the recurrent state-space model behind Dreamer, and that is where we turn next.
