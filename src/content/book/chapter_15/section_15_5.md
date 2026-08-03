---
chapter: 15
section: 15.5
title: "Real-robot evaluation: variance, success rate, time-to-completion"
target_words: 2000
status: draft
prereqs: §15.4 (sim benchmarks buy reproducibility with a fixed world; RoboArena moves evaluation back onto real hardware, which is where the statistical problems in this section start). Helpful, §5.1 (an episode as one rollout of a policy in an environment) and §6.3 (compounding error, the reason a policy that looks fine for 20 seconds can fall apart at 60).
key_refs:
  - Atreya, P., Pertsch, K. et al. (2025). RoboArena, Distributed Real-World Evaluation of Generalist Robot Policies.
  - Kim, M. J., Pertsch, K., Karamcheti, S. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Black, K., Brown, N., Driess, D. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 15.5  Real-robot evaluation: variance, success rate, time-to-completion

A sim benchmark hands you a number and a promise that anyone can reproduce it. A real robot hands you a number and a shrug. Run the same policy on the same task fifty times and you will not get the same answer, because the lighting drifted, the gripper picked up a little grease, the object landed two centimeters left of where it did yesterday, and the tape on the table has started to peel. This is not sloppiness you can engineer away. Physical evaluation is a noisy measurement of a quantity you cannot observe directly, and the whole job of this section is to treat it that way: how to estimate a success rate you can defend, how to say honestly how uncertain that estimate is, and what to record beyond the pass/fail bit so that "it works" means something.

## Success rate is an estimate, not a score

Start with the number everyone reports. You run the policy N times, it succeeds k times, and you write down k/N as the success rate. That fraction is not the policy's success rate. It is a sample estimate of an unknown probability p, and the gap between the two is where most published robot numbers quietly mislead.

Each trial is close enough to a coin flip with bias p that the binomial model applies: k successes out of N draws. The thing you actually care about, p, is hidden, and k/N is your best guess at it. The trouble is that guesses from small N are wide. Suppose you run 20 trials and get 15 successes. You report 75%. A 95% confidence interval around that number, using the Wilson method that behaves well for proportions near 0 and 1, runs roughly from 53% to 89%. So "75%" is compatible with the true policy being a coin flip on a good day or nearly reliable, and your 20 trials cannot tell those apart. That is not a rounding concern. A difference between 53% and 89% is the difference between a demo and a product.

The lesson is blunt: report the interval, not just the point. A success rate without an N and an uncertainty band is a marketing figure. When you read a paper that says "our policy achieves 90% on the pick-and-place task" with no trial count, assume the worst about N until proven otherwise, because the authors who ran 200 trials almost always tell you so.

How many trials do you need? Enough that the interval is narrow enough to support the claim you want to make. The width of a binomial interval shrinks with the square root of N, which is unforgiving arithmetic. Going from 20 to 80 trials halves your error bar; going from 80 to 320 halves it again. If you want to distinguish a 70% policy from an 80% policy with any confidence, you are looking at well over a hundred trials per policy, and that is why honest real-robot comparisons are expensive and why so many papers dodge them with sim. There is no trick that gets you a tight interval from ten rollouts.

## Where the variance actually comes from

Treating trials as independent coin flips is a useful fiction, and like all useful fictions it leaks. The reasons it leaks are worth naming, because each one is a source of variance you can either control or at least record.

The first is initial-condition variance. Where the object starts, at what angle, under what light, on what surface: change any of these and you have changed the task. A policy can hit 95% when the mug always starts in the same 5-centimeter box and collapse to 40% when you scatter the mug across the whole table. Neither number is wrong; they measure different distributions of starting states. So the initial-state distribution is part of the benchmark, and if you do not fix it and describe it, your success rate is uninterpretable. Good protocols mark starting positions with a grid or randomize them from a written, reproducible scheme, and they report which.

The second is temporal drift, the slow kind. Over a long evaluation session the hardware changes underneath you. Motors warm up and their friction shifts, the gripper's rubber wears, cameras auto-adjust exposure as the afternoon sun moves, and a cable you bumped at trial 30 subtly changes the arm's zero. None of this is in your policy, all of it is in your numbers, and the fix is to interleave rather than batch: if you are comparing policy A against policy B, alternate them trial by trial instead of running all of A in the morning and all of B after lunch. Batching lets the time-of-day confound ride directly on top of the comparison you care about.

The third is the human in the loop. Someone resets the scene between trials, and that someone makes judgment calls: was that a success or did the block wobble off at the last second, is this starting pose close enough to the intended one, should we retry because the object rolled off before the policy even started. Those calls move the number. RoboArena (Atreya, Pertsch et al., 2025) leans into this by making a human's pairwise preference the unit of measurement, which sidesteps the "what counts as success" argument by asking "which of these two was better" instead. For a single-policy success rate you do not get that luxury, so you write the success criterion down before you run anything: contact does not count, the object must be inside the target region for two seconds, a dropped object is a failure even if it lands in the right place. Decide it in advance, because deciding it while watching a marginal trial is how bias walks in.

## Not every failure is the same failure

The pass/fail bit throws away information you paid for. A policy that fails by knocking the mug over is telling you something different from one that fails by freezing, and a policy that fails by grabbing the wrong object is different again. When you log failures, log the mode: no-grasp, wrong-object, dropped-in-transit, missed-placement, collision, timeout. A confusion table of failure modes turns a flat 75% into a diagnosis. The OpenVLA evaluations (arXiv:2406.09246) and the π0 report (arXiv:2410.24164) both break results down by task category rather than dumping one aggregate, and that breakdown is what lets a reader see whether a policy's weakness is perception, grasping, or the long-horizon glue between steps.

This connects straight back to §6.3. Compounding error means failures cluster late in an episode, so a policy that succeeds on the first two subtasks of a five-step chain and dies on the third is not "60% good"; it has a specific failure surface at step three. Aggregate success rate hides that. Per-step or per-subtask success rates expose it.

## Time-to-completion, and why it is not a tiebreaker

Two policies both hit 80% on the same task. One finishes the average successful trial in 14 seconds, the other takes 41. They are not equally good, and success rate alone cannot see the difference. Time-to-completion is the second axis of real-robot evaluation, and it matters for reasons beyond throughput: a policy that takes three times as long is usually taking a more roundabout, hesitant path, which means more chances to knock something over and less headroom when the next task is waiting.

Report time-to-completion only over successful trials, and report its spread, not just its mean. A policy that finishes in 14 seconds give or take 2 is a different animal from one that averages 14 but ranges from 6 to 55; the second one is fast on the easy starts and grinding through the hard ones, and that variance predicts how it will behave when the task gets harder. Because completion times are usually skewed with a long right tail, the median and the interquartile range describe them more honestly than mean and standard deviation. And watch the failure-truncation trap: if your hard trials tend to fail and get excluded, your average completion time is computed only over the easy successes and looks better than the policy deserves. Always read time-to-completion next to success rate, never instead of it.

There is a third quantity worth logging that sits between the two: intervention rate, if your setup allows a human to nudge or reset a stuck policy mid-episode. How often a supervisor has to step in, and for what, is often the number a deployment team cares about more than raw success, because it maps directly onto how many robots one person can babysit. §16 and §17 return to this when we talk about fielding a policy for real work rather than a paper.

## A reporting protocol you can actually defend

Pulling it together, here is what a real-robot evaluation should carry so that someone else can read it and know what they are looking at.

Fix and describe the initial-state distribution before running, whether gridded or randomized from a stated scheme. Choose N large enough that your confidence interval supports your claim, and report N alongside the rate. Interleave policies trial by trial when comparing, never batch. Write the success criterion down in advance and apply it mechanically. Log the failure mode of every failed trial, not just the count. Report success rate with a Wilson (or equivalent) interval, and report time-to-completion over successes as a median with its spread.

That is more bookkeeping than a sim benchmark demands, and it is the price of a number that survives contact with someone else's robot. A distributed effort like RoboArena exists precisely because no single lab's protocol is fully trusted by the rest; spreading evaluation across institutions and using relative comparisons is a way of averaging out one lab's peeling tape and warm motors. You will not run RoboArena at home. But you can run its underlying discipline, which is to treat every reported success rate as an estimate with an error bar and a paper trail, and that discipline is what §15.6 turns into a checklist for building an evaluation of your own from scratch.
