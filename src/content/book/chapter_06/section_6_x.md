---
chapter: 6
section: 6.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §6.1–§6.6; Python with PyTorch, gymnasium, and gymnasium-robotics (or robomimic) installed; a working understanding of behavior cloning, compounding error, and DAgger; about two hours of laptop CPU time (a GPU helps but is not required)
key_refs:
  - Pomerleau (1988). ALVINN: An Autonomous Land Vehicle in a Neural Network. NeurIPS.
  - Ross, Gordon & Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS. (DAgger)
  - Ho & Ermon (2016). Generative Adversarial Imitation Learning. NeurIPS.
  - Mandlekar et al. (2021). What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. arXiv:2108.03298.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
---

# 6.x  Hands-on exercise + chapter references

Chapter 6 argued that cloning is where you start and that compounding
error is what eventually bites you. The exercise is the one where you see
both with your own eyes: train a behavior-cloning policy whose offline
loss looks healthy, watch its rollout success rate disappoint, then close
the gap with DAgger and put a number on the improvement. The four drills
below take a combined two hours or so. A GPU shortens the training runs
but nothing here requires one, and after the initial dependency install
no internet is needed.

The test bed is a 2D block-pushing task: a point or planar end-effector
must push a block to a goal region. The task is deliberately simple — you
can watch every rollout — but it has the one property that makes the
exercise worth doing, a horizon long enough for compounding error to show
up. Install the dependencies:

```
pip install torch gymnasium gymnasium-robotics
```

If you prefer the manipulation-flavored version, `robomimic` ships the
block-pushing and lift datasets used in Mandlekar et al. (2021,
arXiv:2108.03298) and the workflow below maps onto it with minimal
changes; the gymnasium `PointMaze`/`Push` environments are the
lighter-weight option and are assumed in the instructions.

## Exercise 6.x.1 — Collect (or load) a demonstration dataset

You need expert demonstrations before you can clone them. Two routes:

**Route A — scripted expert.** Write a short scripted controller that
solves the push task using privileged state (the block pose and goal,
which the *learner* will not see in raw form). A proportional controller
that drives the end-effector to a pre-push pose behind the block and then
toward the goal is enough. Roll it out for 200 episodes, recording at each
step the *observation* the learner will receive (e.g., the rendered image
or the low-dimensional observation vector, not the privileged state) and
the *action* the scripted expert took. Save as `demos.npz` with arrays
`obs` and `act`. The split between what the expert sees (full state) and
what the learner sees (observation) is the whole point — it is what makes
the cloned policy imperfect and therefore interesting.

**Route B — load robomimic.** Download the robomimic `lift` or `can`
proficient-human dataset and use its 200 demonstrations directly. This
skips the scripting and gives you genuine human data with the
multimodality §6.6 warned about.

Record one number before moving on: the expert's own success rate over
50 fresh episodes. It should be at or near 100%. That is your ceiling.

Wall clock: about twenty minutes (Route A) or ten (Route B).

## Exercise 6.x.2 — Train a behavior-cloning policy and measure the offline/online gap

Write `bc.py`. Build a small MLP (two or three hidden layers of 256 units
is plenty for the low-dimensional observation; a tiny CNN if you are
cloning from pixels) that maps observation to action. Train it as
straight supervised regression on `demos.npz` with a mean-squared-error
loss and Adam, the §6.2 loop:

```python
for epoch in range(num_epochs):
    for obs, act in loader:
        pred = policy(obs)
        loss = F.mse_loss(pred, act)
        opt.zero_grad(); loss.backward(); opt.step()
```

Hold out 10% of the demonstrations as a validation set and plot training
and validation loss. Both should fall to a small value and track each
other — the policy fits the demonstrations and is not badly overfitting.
This is the reassuring picture.

Now run the policy in the environment for 50 episodes and record the
*rollout success rate* and the average episode length. Here is the
lesson: the success rate will be meaningfully below the expert's 100%,
often dramatically so on the longer episodes, even though the offline loss
looked fine. Save the two numbers — offline validation loss and online
success rate — side by side. The gap between "fits the data" and
"succeeds on its own rollouts" is §6.3's compounding error, made
concrete. Write one sentence diagnosing where the rollouts fail; you will
typically see the policy do well until a small error pushes the block or
the end-effector into a configuration the demonstrations never showed,
after which it flails — exactly the covariate-shift story.

Wall clock: about thirty minutes including training and evaluation.

## Exercise 6.x.3 — Re-train with DAgger and quantify the gap closed

Write `dagger.py` starting from your BC policy. Implement the DAgger loop
from §6.3:

```python
dataset = load("demos.npz")          # start from the BC demonstrations
policy = train_bc(dataset)
for iteration in range(num_dagger_iters):
    new_obs, new_states = [], []
    for episode in range(rollouts_per_iter):
        # roll out the CURRENT policy, recording the states it visits
        obs, states = rollout(policy)
        new_obs += obs; new_states += states
    # the EXPERT relabels the learner's visited states
    new_act = [scripted_expert(s) for s in new_states]
    dataset = aggregate(dataset, new_obs, new_act)
    policy = train_bc(dataset)        # retrain on the aggregated set
```

The keystone is that the *expert* (your scripted controller from 6.x.1,
which can see privileged state) labels the states the *learner* actually
reached — including the off-distribution ones the pure-BC policy stumbled
into. Run five DAgger iterations, 20 rollouts each. After each iteration,
record the rollout success rate over 50 fresh episodes.

Plot success rate versus DAgger iteration, with the pure-BC result from
6.x.2 as iteration zero and the expert's 100% as a dashed ceiling line.
Save as `dagger_curve.png`. The curve should climb from the disappointing
BC number toward the expert ceiling over the five iterations. The gap it
closes — BC success rate at iteration zero versus DAgger success rate at
iteration five — is the single number this whole exercise exists to
produce. Report it in one sentence: "DAgger raised rollout success from
X% to Y% by relabeling N additional states with expert actions."

A worthwhile variant if you have time: track the *dataset size* alongside
success rate, and note how many expert labels DAgger spent to buy the
improvement. That label budget is the real cost §6.5 weighed when it
placed DAgger one rung above BC — on hardware those labels are a human's
time, not a free function call.

Wall clock: about forty-five minutes for five iterations and the plot.

## Exercise 6.x.4 — Read an imitation-learning paper through the chapter's lens

Open one of: Mandlekar et al. (2021), "What Matters in Learning from
Offline Human Demonstrations for Robot Manipulation" (arXiv:2108.03298),
which is the empirical anchor for most of this chapter's claims; or RT-1
(arXiv:2212.06817), to see cloning at foundation scale. In a text file
`imitation_audit.txt`, answer five questions, one paragraph each:

1. **Signal.** What is the training signal — pure BC, DAgger, IRL,
   offline RL, or a mixture? If BC, what action representation and loss?
   If the paper claims to fight compounding error, how?

2. **Data.** How many demonstrations, collected how (teleop, scripted,
   kinesthetic), at what frequency? Is the data single-task or multi-task,
   single-embodiment or cross-embodiment? Estimate the total labeled
   state-action count, the §6.1 quantity.

3. **Failure mode.** Where does the paper report the policy failing? Map
   the reported failures onto the chapter's vocabulary — covariate shift,
   multimodality smearing, insufficient coverage — or note that they do
   not fit and say what is missing.

4. **The counterfactual.** Would RL or IRL plausibly have done better on
   this task, per §6.5's ladder? What would each have demanded (reward,
   simulator, online interaction) that the authors apparently lacked or
   chose to avoid?

5. **Judgment.** What single change — more data, DAgger relabeling, a
   richer action head, a reward signal — would most improve the reported
   result, and why?

There is no scoring rubric. The point is that the chapter's framing stops
being abstract the moment you apply it to a paper you care about, and
these five questions are the ones a design review of an imitation system
actually asks.

Wall clock: about thirty minutes for the read plus the written audit.

## Chapter 6 reading list

The works below are cited in §6.1–§6.6, grouped by purpose. Full
bibliographic entries for everything cited in the book live in Appendix
E.2; this is the chapter-local subset.

### Behavior cloning: origins and modern practice

- Pomerleau, D. A. (1988). "ALVINN: An Autonomous Land Vehicle in a Neural
  Network." *NeurIPS 1988*. Behavior cloning before it had the name; §6.1's
  historical anchor. Three-layer net, 45 minutes of steering data, drove on
  real roads.
- Mandlekar, A., et al. (2021). "What Matters in Learning from Offline
  Human Demonstrations for Robot Manipulation." arXiv:2108.03298. The
  empirical study §6.1 and §6.5 lean on; the robomimic datasets and the
  finding that BC on a few hundred demos beats from-scratch RL under sparse
  reward. The companion codebase is the test bed for this chapter's
  exercises.
- Florence, P., et al. (2022). "Implicit Behavioral Cloning." *CoRL 2021*.
  The energy-based alternative to regression BC; relevant to §6.6's note
  that a unimodal regression head smears across multimodal expert behavior.

### Compounding error and DAgger

- Ross, S., & Bagnell, J. A. (2010). "Efficient Reductions for Imitation
  Learning." *AISTATS 2010*. The analysis that makes the quadratic-in-
  horizon cost of compounding error precise — §6.3's central result.
- Ross, S., Gordon, G., & Bagnell, J. A. (2011). "A Reduction of Imitation
  Learning and Structured Prediction to No-Regret Online Learning."
  *AISTATS 2011*. The DAgger paper; §6.3's fix and Exercise 6.x.3's
  algorithm.

### Inverse RL and adversarial imitation

- Ng, A. Y., & Russell, S. (2000). "Algorithms for Inverse Reinforcement
  Learning." *ICML 2000*. The paper that named IRL and pointed out its
  ill-posedness — §6.4's starting point.
- Abbeel, P., & Ng, A. Y. (2004). "Apprenticeship Learning via Inverse
  Reinforcement Learning." *ICML 2004*. Feature-expectation matching;
  §6.4's first practical IRL.
- Ziebart, B. D., et al. (2008). "Maximum Entropy Inverse Reinforcement
  Learning." *AAAI 2008*. The MaxEnt formulation that became the field's
  workhorse; §6.4.
- Ho, J., & Ermon, S. (2016). "Generative Adversarial Imitation Learning."
  *NeurIPS 2016*. GAIL; the distribution-matching pivot in §6.4.
- Fu, J., Luo, K., & Levine, S. (2018). "Learning Robust Rewards with
  Adversarial Inverse Reinforcement Learning." *ICLR 2018*. AIRL; recovers
  a dynamics-disentangled, portable reward — §6.4's closing idea.

### Offline RL: the rung between cloning and online RL

- Kumar, A., et al. (2020). "Conservative Q-Learning for Offline
  Reinforcement Learning." *NeurIPS 2020*. CQL; the out-of-distribution
  value-overestimation fix §6.5 names when it places offline RL on the
  ladder.
- Levine, S., et al. (2020). "Offline Reinforcement Learning: Tutorial,
  Review, and Perspectives on Open Problems." arXiv:2005.01643. The survey
  to read if §6.5's offline-RL paragraph leaves you wanting the full story.

### Imitation at foundation scale: the bridge to Part 4

- Brohan, A., et al. (2022). "RT-1: Robotics Transformer for Real-World
  Control at Scale." arXiv:2212.06817. Cloning of 130k demonstrations over
  700+ tasks; the §6.1 proof that imitation scales, revisited in Chapter 11.
- Collaboration, O. X.-E., et al. (2023). "Open X-Embodiment: Robotic
  Learning Datasets and RT-X Models." arXiv:2310.08864. The million-episode
  cross-embodiment dataset behind §6.1's scaling argument; detailed in
  Chapter 15.
- Black, K., et al. (2024). "π0: A Vision-Language-Action Flow Model for
  General Robot Control." arXiv:2410.24164. The BC-then-RL,
  flow-matching-head VLA §6.5 and §6.6 use to show where the chapter's
  ideas land in a current system; Chapter 13.

## Chapter summary

Chapter 6 established imitation as the dominant training signal for modern
robot learning and gave you the tools to use it well. You can now train a
behavior-cloning policy and read its curves without being fooled by a
healthy offline loss; explain compounding error with the horizon argument
intact and apply DAgger to close the gap, having measured that gap
yourself; distinguish cloning from inverse RL and offline RL by the demand
each one makes; and decide, given a dataset and a robot, whether behavior
cloning is the right starting point — which, as §6.5 argued and the
exercises confirmed, it almost always is. Chapter 7 picks up the other
half of that decision, developing the deep-RL machinery that fine-tunes a
cloned policy past human performance when a reward and a simulator are at
hand.
