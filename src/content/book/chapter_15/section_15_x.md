---
chapter: 15
section: 15.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §15.4 (SimplerEnv and its one promise, that a sim score predicts a real score, plus the LIBERO/CALVIN/RoboCasa contrast the exercise sits next to); §15.5 (the binomial model of a success rate, the Wilson interval, and the claim that k/N without an error bar is not a number you can defend); §15.6 (screen in sim for volume, confirm on hardware for truth). A GPU and a downloaded OpenVLA checkpoint make the full run possible in an afternoon; the statistical payoff, which is the part that matters, runs on a CPU in seconds from recorded per-seed counts
key_refs:
  - Li, X., Hsu, K., Fu, J. et al. (2024). Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv). CoRL.
  - Kim, M. J., Pertsch, K., Karamcheti, S. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Ghosh, D., Walke, H., Pertsch, K. et al. (2024). Octo, An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - Liu, B., Zhu, Y., Gao, C. et al. (2023). LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - Atreya, P., Pertsch, K. et al. (2025). RoboArena, Distributed Real-World Evaluation of Generalist Robot Policies.
---

# 15.x  Hands-on exercise + chapter references

§15.5 made a claim you were asked to take on faith: that a single success rate, reported without an interval, hides how little you actually know. Twenty trials, fifteen successes, "75%," and the true rate sitting anywhere from the low fifties to the high eighties. This exercise turns that claim into an interval you compute from runs you did yourself. The TOC states the target plainly: reproduce a published OpenVLA evaluation in SimplerEnv, then compute confidence intervals over five seeds. You are going to run the same policy on the same task five times, changing only the random seed, and watch the "one number" the leaderboard prints fan out into a spread wide enough to change what you would conclude. If §15.5 was overselling, the five seeds will land on top of each other and the interval will be a point. They will not, and the gap between the one number and the honest one is the whole lesson.

There is a practical fork here, and it is worth naming before you start. Running OpenVLA in SimplerEnv for real wants a GPU and a checkpoint download, and it will keep a machine busy for an afternoon. The statistical payoff does not need any of that. So the exercise splits: 15.x.1 and 15.x.2 are the full reproduction for readers with the hardware, and 15.x.3 onward, the part that actually teaches the confidence interval, runs on a laptop CPU from the per-seed counts, whether those counts came from your own GPU run or from the recorded numbers below. Do the full run if you can. If you cannot, skip to 15.x.3 with the recorded counts and you lose nothing of the point.

## What you need

For the full path: a CUDA GPU with enough memory for the 7B OpenVLA checkpoint, the SimplerEnv package (Li et al., 2024), and the OpenVLA weights (arXiv:2406.09246) from the Hugging Face hub. For the statistics path: Python, NumPy, SciPy, and nothing else. No robot either way, because SimplerEnv is exactly the sim-for-real substitute §15.4 built the case for; the physical WidowX and Google robot setups it mirrors are the ones whose real numbers OpenVLA already published against.

## Exercise 15.x.1 — Reproduce one seed

Start by running a single evaluation and checking that your number lands near the paper's. SimplerEnv ships the Google-robot and BridgeData/WidowX task suites already matched to their real-world counterparts, so the setup is mostly wiring a policy to an environment and looping.

```python
import numpy as np
import simpler_env
from simpler_env.policies.openvla import OpenVLAInference  # package's wrapper

def evaluate(env_name, policy, n_episodes=25, seed=0):
    env = simpler_env.make(env_name)
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        policy.reset()
        done = False
        while not done:
            action = policy.step(obs["image"], obs["instruction"])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(info.get("success", False))
    return successes, n_episodes

policy = OpenVLAInference(checkpoint="openvla/openvla-7b")
k, n = evaluate("google_robot_pick_coke_can", policy, n_episodes=25, seed=0)
print(f"seed 0: {k}/{n} = {k / n:.0%}")
```

Two things to check before you trust anything. Your single-seed number should land in the neighborhood of OpenVLA's published SimplerEnv result for that task, not on top of it; a few points off is normal and is itself the first hint of what 15.x.2 is about. And the task string must match the setup the paper reported, because SimplerEnv's whole promise from §15.4 is scene-by-scene fidelity, and a slightly different variant is a different number.

## Exercise 15.x.2 — Run five seeds

Now change one thing, the seed, and hold everything else fixed. Same checkpoint, same task, same episode count. The seed moves object initial poses, and nothing else, which is precisely the initial-condition variance §15.5 said bites hardest.

```python
results = {}
for seed in range(5):
    k, n = evaluate("google_robot_pick_coke_can", policy, n_episodes=25, seed=seed)
    results[seed] = (k, n)
    print(f"seed {seed}: {k}/{n} = {k / n:.0%}")
```

Write the five fractions down. If your run produced something like 19/25, 21/25, 16/25, 20/25, 18/25, you are looking at a range from 64% to 84% from nothing but the seed. That spread is not a bug in SimplerEnv and it is not your policy being unstable; it is what a success rate is, a draw from a distribution, and §15.5 told you to expect exactly this. Readers on the statistics-only path: use those five counts as your recorded numbers and carry on.

## Exercise 15.x.3 — Put an interval on each seed

Here is where the chapter's argument gets settled. For each seed you have k successes out of n. Compute the Wilson interval §15.5 recommended, the one that behaves near 0 and 1 where the textbook normal approximation falls apart.

```python
from scipy.stats import norm

def wilson(k, n, conf=0.95):
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return center - half, center + half

recorded = {0: (19, 25), 1: (21, 25), 2: (16, 25), 3: (20, 25), 4: (18, 25)}
for seed, (k, n) in recorded.items():
    lo, hi = wilson(k, n)
    print(f"seed {seed}: {k/n:.0%}  95% CI [{lo:.0%}, {hi:.0%}]")
```

Every one of those intervals is roughly twenty points wide, because n is 25 and there is no arithmetic that makes 25 trials narrow. Read two seeds against each other: seed 2 at 64% and seed 1 at 84% look like a real difference until you notice their intervals overlap heavily, which means the "difference" is the seed talking, not the policy. This is the same trap as comparing two policies that differ by a trial or two, drawn here from one policy against itself. If a single policy against itself can swing twenty points on the seed, a two-point gap between two different policies is worth nothing.

## Exercise 15.x.4 — Pool the seeds

The fix for a wide interval is more trials, and five seeds of 25 is 125 trials you can pool, provided the seeds were independent draws from the same setup, which by construction they were. Sum the successes, sum the episodes, and put one interval on the pooled count.

```python
K = sum(k for k, _ in recorded.values())
N = sum(n for _, n in recorded.values())
lo, hi = wilson(K, N)
print(f"pooled: {K}/{N} = {K/N:.0%}  95% CI [{lo:.0%}, {hi:.0%}]")
```

The pooled point estimate sits near the mean of the five, around 75%, but the interval has tightened from twenty-odd points to roughly thirteen, because n went from 25 to 125. That shrinkage, proportional to the square root of the sample size, is the only lever you have; there is no clever estimator that gets you a tight number from few trials. If you want the interval to halve, you run four times as many episodes, and now you understand in your fingers why §15.5 called sample size the thing robotics cannot afford and why RoboArena (Atreya, Pertsch et al., 2025) distributes the cost across labs instead of asking one lab to run a thousand rollouts alone.

## Exercise 15.x.5 — Compare two policies honestly

The payoff. Repeat 15.x.2 for a second policy on the same task, Octo (arXiv:2405.12213) is the natural partner since SimplerEnv ships it too, pool each policy's five seeds, and ask whether their intervals overlap.

```python
octo_recorded = {0: (14, 25), 1: (13, 25), 2: (17, 25), 3: (12, 25), 4: (15, 25)}
Ko = sum(k for k, _ in octo_recorded.values()); No = sum(n for _, n in octo_recorded.values())
print("OpenVLA:", wilson(K, N))
print("Octo:   ", wilson(Ko, No))
```

If the two pooled intervals are disjoint, you can claim one policy beats the other on this task and defend it. If they overlap, you cannot, no matter how far apart the point estimates look, and reporting the raw gap anyway is the quiet lie §15.5 named. Run the comparison the honest way from §15.6 when it is on real hardware: interleave the two policies trial by trial rather than batching, so warm motors and drifting light hit both equally. In sim the seed does that work for you, which is one more reason SimplerEnv is where you iterate and hardware is where you confirm.

## Chapter 15 reading list

Cited across §15.1–§15.7, grouped by the job each reference does. Full entries for everything in the book live in Appendix E.2; this is the chapter-local subset.

### The datasets

- Open X-Embodiment Collaboration, Padalkar, A., et al. (2023). "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." arXiv:2310.08864. The spine of §15.2: roughly sixty datasets, twenty-one institutions, twenty-two embodiments, and the point that the contribution was harmonization, not collection.
- Walke, H., et al. (2023). "BridgeData V2: A Dataset for Robot Learning at Scale." CoRL 2023. §15.1's worked example, the single WidowX episode small enough to inspect array by array.
- Cadene, R., Alibert, S., Soare, A., et al. (2024). "LeRobot: State-of-the-art machine learning for real-world robotics in PyTorch." github.com/huggingface/lerobot. §15.3's second storage lineage, PyTorch-first and aimed at the person with one cheap arm; the delta-timestamp query is the piece of engineering to remember.
- Shukor, M., et al. (2025). "SmolVLA." arXiv:2506.01844. §15.3's example of a policy that ships inside the LeRobot ecosystem and loads in two lines.

### The benchmarks

- Liu, B., Zhu, Y., Gao, C., et al. (2023). "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310. §15.4 and §15.6's source for the four-axis factoring of difficulty, object, spatial, goal, long-horizon, worth stealing as a coverage checklist even if you never run it.
- Mees, O., Hermann, L., Rosete-Beas, E., Burgard, W. (2022). "CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation." IEEE RA-L. §15.4's long-horizon chaining benchmark and the ABC→D split that separates a policy that learned the task from one that memorized the rooms.
- Nasiriany, S., Maddukuri, A., Zhang, L., et al. (2024). "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots." RSS. §15.4's scale-through-generated-assets benchmark, where memorization stops being a winning strategy.
- Li, X., Hsu, K., Fu, J., et al. (2024). "Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv)." CoRL. §15.4's sim-that-predicts-real benchmark and the environment this exercise runs in; its measure of merit is correlation with the physical rig, not raw difficulty.

### The policies under the microscope

- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. The policy you reproduce in 15.x.1–15.x.2 and the 970,000-trajectory training figure §15.7 uses to frame the whole chapter.
- Ghosh, D., Walke, H., Pertsch, K., et al. (2024). "Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213. 15.x.5's comparison partner, and §15.7's second data point for what pretraining scale buys.

### Real-robot evaluation

- Atreya, P., Pertsch, K., et al. (2025). "RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies." §15.4, §15.5, and §15.6's recurring reference: the Chatbot-Arena-style pairwise, distributed protocol that gives up sim's reproducibility to buy a signal no single lab can quietly train against.

### The survey that frames the space

- Zhang, J., et al. (2026). "Vision-Language-Action in Robotics: A Survey of Datasets, Benchmarks, and Data Engines." arXiv:2604.23001. §15.1–§15.3's framing reference for how the field's data and benchmarks fit together.

## Chapter summary

Chapter 15 put a floor under every number Part 4 quoted and every claim Part 5 will make. You can now open a robot episode and name what each field means, knowing a seven-vector action might be end-effector deltas or joint velocities and the file will not tell you which. You can explain what pooling data cost Open X-Embodiment, that its real work was harmonizing incompatible conventions, and why the field drifted from one branded corpus toward teleop fleets and LeRobot's low-friction path for a person with one arm. You can read a table of LIBERO, CALVIN, RoboCasa, and SimplerEnv scores and say which comparisons it actually supports and which it only decorates, and after this exercise you can do more than recite §15.5's warning about intervals, because you watched one policy against itself swing twenty points on the seed alone, tightened the interval by pooling, and learned to call two overlapping intervals a tie no matter how far apart their point estimates print. That last skill is the one Chapter 16 leans on the moment it fine-tunes a model and asks whether the number moved or the noise did.
