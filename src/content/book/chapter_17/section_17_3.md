---
chapter: 17
section: 17.3
title: "A/B evaluation on hardware"
target_words: 2000
status: draft
prereqs: §15.5 (real-robot success as a binomial estimate, Wilson intervals, interleaving, failure-mode logging — this section is that protocol turned into a two-policy comparison), §15.x (two overlapping intervals are not a difference test), §16.4 (the screen-then-confirm loop; A/B is the confirm step at higher stakes), §17.1 (safety as its own evaluation axis, the column this comparison must not drop).
key_refs:
  - Atreya, P., Pertsch, K. et al. (2025). RoboArena, Distributed Real-World Evaluation of Generalist Robot Policies.
  - Liu, B. et al. (2023). LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
---

# 17.3  A/B evaluation on hardware

You fine-tuned a new version of your policy. Call it B, and the one already running A. B scores higher on your sim screen. The question this section answers is whether B is actually better on the real robot, or whether it just looks better, and the reason the question is hard is that a real robot is a noisy measuring instrument that drifts under you while you use it. The TOC's phrasing for the goal is exact: an A/B evaluation pipeline that does not lie to you. Most naive comparisons lie, and they lie in ways that look like results.

## The comparison that lies

Here is the tempting, wrong protocol: run A for a day, collect its success rate, run B the next day, compare. It is wrong because everything §15.5 listed as a source of variance moves between the two days. The tape on the gripper wears, the lab lights shift, the motors run warmer in the afternoon, someone nudges a camera. Any of those can swing a success rate by more than the difference between two policies, so a day-over-day comparison measures the day at least as much as the policy. You will conclude B is better, or worse, and you will have measured the weather.

The fix is the one §15.5 already argued for in the single-policy case, now load-bearing: interleave. Alternate A and B trial by trial, so that whatever drifts hits both policies equally. If the light dims halfway through, it dims on A's trials and B's trials in the same proportion, and the comparison survives it. Batching is the enemy; interleaving is the whole trick. RoboArena (Atreya, Pertsch et al., 2025) scaled exactly this intuition across labs, using pairwise comparisons rather than absolute scores precisely because a relative, interleaved judgment is robust to the per-lab conditions an absolute number is not.

## Pair the trials, and blind the judge

Interleaving buys more if you pair it. Where you can reset to the same initial state twice, run A and B from that same state back to back, and you have a paired trial: same object pose, same lighting, same everything, differing only in the policy. Paired comparisons are more powerful than unpaired ones because they cancel the per-trial variance that initial conditions inject, so you detect a real difference with fewer trials. On a robot where trials are expensive and §15.5 already showed intervals are wide at the sample sizes you can afford, that efficiency is worth the extra setup discipline of reproducing initial states.

Blinding matters as much as pairing. The person who resets the scene and scores the trial should not know whether A or B produced the rollout, because knowing leaks into the borderline calls, and manipulation is full of borderline calls: did the block settle in the target or wobble out, was that contact a success or a near-miss. §15.5 named the human in the loop as a source of bias; blinding is how you close it. Randomize the A/B order so it is not predictable, hide the identity from the scorer, and record which was which separately for analysis. If the scorer can guess the policy, the score is contaminated.

## The statistic is the difference, not two intervals

Here is the trap §15.x set up and this section springs. You run the interleaved comparison, you compute A's success rate with its Wilson interval and B's with its own, you see the intervals overlap, and you are tempted to conclude "no difference." That conclusion is wrong, or at least unsupported, because two overlapping marginal intervals do not test a difference. The quantity you care about is the difference itself, and with paired trials it has its own, tighter test.

```python
# paired A/B on the same initial states: each trial gives (a_success, b_success)
import numpy as np
from scipy.stats import binomtest

pairs = [(1, 1), (0, 1), (1, 0), (1, 1), (0, 1), ...]   # 1 = success
b_wins = sum(1 for a, b in pairs if b > a)   # B succeeded, A failed
a_wins = sum(1 for a, b in pairs if a > b)   # A succeeded, B failed
# McNemar: only the discordant pairs carry information about the difference
p = binomtest(b_wins, b_wins + a_wins, 0.5).pvalue
print(f"B>A on {b_wins} pairs, A>B on {a_wins}, discordant p={p:.3f}")
```

The point of that snippet is what it throws away. The pairs where A and B both succeed or both fail carry no information about which is better; only the discordant pairs, where one succeeded and the other did not, do. A McNemar-style test looks at exactly those, which is why a paired design is more sensitive than comparing two marginal rates. Report the difference with an interval, say the same thing in words, and never present two overlapping marginal intervals as if their overlap were the answer.

And decide N before you start. The cardinal sin of expensive-trial evaluation is peeking: running until the result looks significant and then stopping, which inflates your false-positive rate because you gave yourself many chances to cross the line by noise. Pre-register the trial count and the decision rule, run to it, and report what you find. If you must look early, use a sequential test built for it rather than a fixed test you peek at, because the fixed test's p-value means nothing once you have optionally stopped.

## Do not drop the safety column

Everything so far compared success rates, and §17.1 already warned that success rate is the wrong sole axis. A/B evaluation inherits that warning with force, because the most dangerous outcome is a B that improves the mean and regresses the tail. Suppose B lifts success from 82% to 88% and, in doing so, changes its failure mode from "stops when uncertain" to "pushes harder and occasionally exceeds the force limit." On a success table B is the clear winner. On a factory floor B is the recall notice from §1.5, and shipping it because the mean improved is exactly the mistake this chapter exists to prevent.

So the comparison carries a safety column alongside the success column. Log every trial as a full episode with its failure mode, the §15.6 discipline, and count the safety-layer interventions from §17.2 per policy: how often did the shield have to veto B's action versus A's, and did B's failures move toward the benign end of the distribution or the dangerous one. A policy that wins on success and loses on shield-intervention rate has not been shown to be better; it has been shown to be a different risk profile, and the decision to ship it is a decision about that risk, not about the mean. Borrow LIBERO's axis idea (arXiv:2306.03310) here too: break the comparison out by the kinds of variation that matter, because B might beat A on the easy tasks and lose on the ones near a person.

## From bench to production

The A/B protocol so far assumes a test station. Deployment adds two moves worth naming. Shadow mode runs B alongside A on the live robot without executing B's actions, computing what B would have done and comparing it to what A did, which gets you a large, cheap, real-world comparison with no risk because B never touches an actuator. Canary deployment runs B for real on a small fraction of the workload while A handles the rest, so a regression shows up on a slice you can roll back rather than the whole fleet. Both are the confirm step of §16.4's screen-then-confirm loop, extended past the lab into a running deployment, and both feed the logging and rollback machinery §17.4 builds next. An A/B result that says B is better on the bench is a hypothesis; shadow and canary are how you test it where it actually has to hold.

A comparison that interleaves, pairs, blinds, tests the difference rather than two intervals, fixes its sample size in advance, and refuses to drop the safety column is a comparison that does not lie to you. It is more bookkeeping than a leaderboard submission, and it is the price of knowing that the new policy you are about to trust near a person is actually the better one. §17.4 turns the logs this evaluation generates into the production machinery that watches the policy after you have shipped it: what to record, what to alert on, and what to roll back when the number moves the wrong way.
