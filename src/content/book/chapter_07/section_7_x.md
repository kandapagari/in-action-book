---
chapter: 7
section: 7.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §7.1–§7.6; Python with PyTorch and gymnasium installed; a working understanding of DQN, policy gradients, PPO, off-policy actor-critic, and domain randomization; about two to three hours of compute (a GPU helps for the SAC run but classic-control tasks finish on CPU)
key_refs:
  - Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518:529-533. (DQN)
  - Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
  - Haarnoja, Zhou, Abbeel & Levine (2018). Soft actor-critic. ICML. arXiv:1801.01290.
  - Tobin et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS 2017.
  - Sutton & Barto (2018). Reinforcement Learning — An Introduction (2nd ed.). MIT Press.
---

# 7.x  Hands-on exercise + chapter references

Chapter 7 made two claims that are easy to nod along to and hard to feel until you have run the code. The first is that on-policy methods like PPO are robust but sample-hungry, while off-policy methods like SAC squeeze far more learning out of the same number of environment steps, at the cost of more moving parts that can quietly break. The second is that a policy trained in one fixed simulator overfits its dynamics, and that randomizing those dynamics during training is what buys you transfer. The four drills below make both claims concrete. They take two to three hours combined. Classic control tasks run on a laptop CPU; the SAC drill is faster on a GPU but does not require one, and after the dependency install nothing here needs internet.

The test beds are the two standard classic-control environments: `CartPole-v1` (discrete actions, the natural home for a DQN- or PPO-style policy) and `Pendulum-v1` (one continuous torque action, the natural home for SAC). They are deliberately tiny, you can watch every rollout and a full training run finishes in minutes, but they are enough to expose the sample-efficiency gap and the sim-to-real failure mode without burning a day of compute. Install the dependencies:

```
pip install torch gymnasium
```

If you would rather not reimplement the algorithms from scratch, `stable-baselines3` ships vetted PPO and SAC implementations and every drill below maps onto it with a few lines; reading a known-correct implementation alongside §7.3 and §7.4 is itself a worthwhile exercise. The instructions assume you are writing your own loops, because the bugs you hit are the lesson.

## Exercise 7.x.1 — PPO on CartPole, and the meaning of a learning curve

Write `ppo_cartpole.py` using the §7.3 skeleton: an actor and a critic (two small MLPs, or one shared trunk with two heads, 64 units per layer is plenty), GAE for the advantage, and the clipped surrogate objective. Collect rollouts in batches of around 2,048 environment steps, then run a handful of epochs of minibatch updates over each batch before discarding it, the on-policy discipline §7.3 insisted on.

Train until the policy reliably balances the pole for the full 500 steps. Plot two curves against environment steps: the average episode return, and the approximate KL divergence between the policy before and after each update. Record one number: the total environment steps to reach a 475-return moving average. Then read the KL curve. If you ever see return collapse from near-500 back toward 20 in a single update, look at the KL spike that preceded it; that is the trust-region violation §7.3 described, the policy stepping so far that the data it just collected no longer describes where it landed. Lowering the clip range or the learning rate should make the collapse disappear. Keeping the step-to-threshold number is the point; it is the baseline the next drill undercuts.

Wall clock: about twenty minutes including the runs.

## Exercise 7.x.2 — SAC on Pendulum, and the sample-efficiency gap

Now switch to the continuous task. Write `sac_pendulum.py` using the §7.4 ingredients: a stochastic Gaussian actor with the squashing correction, twin Q-critics with the min-of-two target to fight overestimation, target networks updated by Polyak averaging, a replay buffer, and the automatic temperature adjustment that tunes the entropy bonus. This is more code than PPO and more ways to get it wrong; if the policy's return is stuck near the floor, the usual culprits are a sign error in the entropy term, forgetting the `tanh` log-prob correction, or updating the target network too fast.

Train SAC on `Pendulum-v1` and, for an honest comparison, also point your PPO implementation at the same environment (PPO handles continuous actions with a Gaussian head, §7.2). Plot both return curves on the same axes against environment steps. The expected picture, and §7.4's headline claim, is that SAC reaches good performance in far fewer environment steps than PPO, because it reuses every transition many times out of the replay buffer rather than throwing each batch away. Record both step-to-threshold numbers and the ratio between them. That ratio is the sample-efficiency gap, and it is the reason §7.4 argued you reach for an off-policy method the moment real-robot samples get expensive, and the reason you tolerate PPO's appetite when, as in §7.5, the samples come from a fast simulator instead.

Wall clock: about forty minutes including both runs.

## Exercise 7.x.3 — Domain randomization and the transfer test

This drill makes §7.5 concrete with the smallest possible "sim-to-real" gap: a mismatch between a training simulator and a different test simulator. Take the Pendulum environment and treat its physical constants, pole mass, pole length, maximum torque, gravity, as the dynamics parameters a real pendulum would differ on.

Train two SAC policies. The first trains on the default parameters only, the fixed-simulator baseline. The second resamples the parameters at the start of every episode from a band around the defaults (say ±30% on mass and length), which is domain randomization in its plainest form. Then build a held-out test set: a grid of pendulum parameter settings the policy never trained on, including a few outside the randomization band, and evaluate both policies on all of them.

The fixed-simulator policy will do well on the exact default it trained on and degrade, sometimes sharply, as the test pendulum drifts away from it. The randomized policy will be slightly worse on the nominal setting and markedly better across the rest of the grid. Tabulate the two policies' average return per test setting side by side. The gap you measure is exactly the §7.5 thesis: randomization trades a little nominal performance for a policy that survives the reality gap, by forcing a single policy to be robust across the whole band rather than tuned to one set of constants. Note also where the randomized policy itself falls off, at the grid points well outside its training band, because that boundary is the honest limit of the method: randomization buys robustness inside the distribution you randomized over, not magic generalization beyond it.

Wall clock: about forty-five minutes including training and the sweep.

## Exercise 7.x.4 — Read a deep-RL paper and audit it against the chapter

Pick one algorithm paper from the reading list, DQN (arXiv:1312.5602 / Nature 2015), PPO (arXiv:1707.06347), or SAC (arXiv:1801.01290) are the cleanest choices, and read it with the chapter's vocabulary in hand. Write a short audit answering five questions:

1. **What problem in the previous method is this paper fixing?** DQN's replay buffer and target network fix the divergence §7.1 warned about; PPO's clip fixes TRPO's expensive trust region from §7.3; SAC's twin critics fix the overestimation §7.4 traced to DDPG.
2. **Which of the chapter's failure modes does it confront directly, and which does it leave to the reader?** Overestimation, on-policy staleness, the exploration-exploitation balance, brittle hyperparameters.
3. **What does the paper report its sample budget as,** and how does that compare to the step counts you measured in Exercises 7.x.1–7.x.2?
4. **Where would this method break on a real robot,** using §7.5's reality-gap framing? What does its benchmark quietly assume that a physical robot will not give you?
5. **If you had to fine-tune a cloned policy (Chapter 6) past human performance with this method, what would the hard part be?** This is the question Chapters 11–16 spend their pages answering.

There is no scoring rubric. The point is that an algorithm paper stops being a wall of equations the moment you can name which earlier failure each design choice is paying down, and these five questions are the ones a researcher actually asks when deciding whether to build on a method.

Wall clock: about thirty minutes for the read plus the written audit.

## Chapter 7 reading list

The works below are cited in §7.1–§7.6, grouped by purpose. Full bibliographic entries for everything cited in the book live in Appendix E.2; this is the chapter-local subset. The classical RL references predate the arXiv mirror in `sources/` and are cited by author, year, and venue.

### Value-based deep RL and function approximation

- Tsitsiklis, J. N., & Van Roy, B. (1997). "An Analysis of Temporal-Difference Learning with Function Approximation." *IEEE TAC* 42(5). The result behind §7.1's warning that off-policy bootstrapping with function approximation can diverge, the "deadly triad."
- Riedmiller, M. (2005). "Neural Fitted Q Iteration." *ECML 2005*. Batch Q-learning with a neural network; §7.1's immediate ancestor to DQN.
- Mnih, V., et al. (2013, 2015). "Playing Atari with Deep Reinforcement Learning" (arXiv:1312.5602) and "Human-level Control through Deep Reinforcement Learning" (*Nature* 518). DQN; the replay buffer and target network that made §7.1's deep value learning stable. Exercise 7.x.4's cleanest target.
- van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-Learning." *AAAI 2016*. The overestimation fix §7.1 and §7.4 both lean on.

### Policy gradients and on-policy methods

- Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning* 8. REINFORCE; the log-derivative estimator at the root of §7.2.
- Sutton, R. S., et al. (2000). "Policy Gradient Methods for Reinforcement Learning with Function Approximation." *NeurIPS 2000*. The policy gradient theorem §7.2 derives.
- Schulman, J., et al. (2016). "High-Dimensional Continuous Control Using Generalized Advantage Estimation." *ICLR 2016*. arXiv:1506.02438. GAE, the variance-reduction tool §7.2 and §7.3 both use.
- Schulman, J., et al. (2015). "Trust Region Policy Optimization." *ICML 2015*. TRPO; the constrained-step idea §7.3 says PPO approximates cheaply.
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347. PPO; §7.3's main result and Exercise 7.x.1's algorithm.

### Off-policy actor-critic

- Silver, D., et al. (2014). "Deterministic Policy Gradient Algorithms." *ICML 2014*. DPG; the deterministic-actor foundation §7.4 builds on.
- Lillicrap, T. P., et al. (2016). "Continuous Control with Deep Reinforcement Learning." *ICLR 2016*. arXiv:1509.02971. DDPG; §7.4's first deep off-policy actor-critic for continuous control.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). "Addressing Function Approximation Error in Actor-Critic Methods." *ICML 2018*. arXiv:1802.09477. TD3; the twin critics and delayed updates §7.4 and Exercise 7.x.2 use.
- Haarnoja, T., et al. (2018). "Soft Actor-Critic." *ICML 2018*. arXiv:1801.01290. SAC; §7.4's maximum-entropy off-policy method and Exercise 7.x.2's algorithm.

### Sim-to-real and domain randomization

- Tobin, J., et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." *IROS 2017*. The randomization idea §7.5 and Exercise 7.x.3 are built on.
- Peng, X. B., et al. (2018). "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization." *ICRA 2018*. Randomizing *dynamics* rather than only appearance, exactly Exercise 7.x.3's setup.
- OpenAI; Akkaya, I., et al. (2019). "Solving Rubik's Cube with a Robot Hand." arXiv:1910.07113. Domain randomization plus automatic curriculum at scale; §7.5's headline demonstration.
- Lee, J., et al. (2020). "Learning Quadrupedal Locomotion over Challenging Terrain." *Science Robotics* 5(47). Randomized-simulation training that transferred to a real legged robot; §7.5's locomotion case.

### Foundations

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning — An Introduction* (2nd ed.). MIT Press. The reference text underneath the entire chapter; Chapters 9–13 cover function approximation through policy gradients.

## Chapter summary

Chapter 7 took the tabular reinforcement learning of Chapter 5 and scaled it up to the neural-network function approximators that modern control actually uses. You can now explain why naive value learning with a function approximator diverges and how DQN's replay buffer and target network tame it; derive the policy gradient and name the variance problem that GAE and a critic exist to solve; implement PPO and read its KL curve well enough to diagnose a trust-region violation, having watched one in your own run; choose between an on-policy method like PPO and an off-policy method like SAC by the sample-efficiency gap you measured rather than by reputation; and apply domain randomization to buy transfer across a dynamics gap, while staying honest about the boundary of the distribution you randomized over. That is the deep-RL toolkit that fine-tunes a cloned policy past human performance when a reward and a simulator are in reach. Part 3 now changes the question from *how do we optimize a policy* to *what architecture should the policy be*, beginning, in Chapter 8, with the sequence models that reframe control itself as a prediction problem.
