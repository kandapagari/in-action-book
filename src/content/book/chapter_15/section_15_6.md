---
chapter: 15
section: 15.6
title: "Building your own evaluation"
target_words: 2000
status: draft
prereqs: §15.5 (real-robot statistics — success rate as an estimate, Wilson intervals, interleaving, failure-mode logging; this section turns that discipline into a build procedure). Helpful, §15.4 (sim benchmarks you can borrow tasks from) and §15.1 (the episode format your logs should match so evaluation data is reusable as training data).
key_refs:
  - Atreya, P., Pertsch, K. et al. (2025). RoboArena, Distributed Real-World Evaluation of Generalist Robot Policies.
  - Liu, B. et al. (2023). LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - Kim, M. J., Pertsch, K., Karamcheti, S. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
---

# 15.6  Building your own evaluation

By now you can read someone else's evaluation and tell whether to trust it. The harder job is designing one from scratch for a policy nobody has benchmarked yet, which is exactly what you face the first time you fine-tune a model for a robot that no public suite covers. There is no leaderboard for "my WidowX arm sorting my specific bin of parts." You have to build the ruler before you can measure anything, and a badly built ruler will lie to you for weeks before you notice. This section is the procedure: what to decide, in what order, and where the traps sit.

The framing that saves you the most pain is to treat an evaluation as a design artifact with the same care you would give a piece of the policy. It has requirements, it has failure modes, and it can be wrong. A good one answers a question you actually have. A bad one produces a number that feels informative and is not.

## Start from the decision, not the task

Before you pick tasks or count trials, write down the sentence that this evaluation is supposed to let you finish. "I will ship policy B instead of policy A if ___." "I will keep collecting data if ___." "I will trust this on the real robot if ___." The blank is your target: a threshold, a comparison, a go/no-go gate. Everything downstream, how many trials, which tasks, what you log, is set by that sentence, and skipping it is the most expensive mistake in the whole chapter because it produces evaluations that measure something real but useless.

An example of the trap. You want to know whether adding wrist-camera data helped, so you run both policies on your standard ten tasks and policy B scores three points higher overall. Did wrist data help? You cannot tell, because the tasks where a wrist camera should matter, the tight-clearance grasps, are four of your ten, and averaging them with six tasks the wrist never sees washes the signal out. The evaluation measured aggregate competence when your question was about a specific capability. Had you started from the decision, "keep the wrist camera if it improves tight grasps," you would have built a different task set.

So: one evaluation, one question. If you have three questions, you may need three task sets, and pretending one number answers all of them is how teams talk themselves into shipping the wrong model.

## Choose tasks that span the difficulty you care about

Two failure shapes bracket a bad task set. Too easy and every policy scores near 100%, so the evaluation cannot separate them; this is the ceiling effect, and it is common when you design tasks around the demo you already know works. Too hard and everything scores near zero, which is a floor effect and just as blind. The tasks that carry information are the ones where good policies pull apart from bad ones, which means you want a spread: a few your current policy mostly handles, a few it half-handles, a couple it fails. The half-handled ones do most of the discriminating.

Borrow structure rather than inventing it. LIBERO (arXiv:2306.03310) organizes its tasks along axes that are worth stealing even if you never run LIBERO itself: variation in objects, in spatial layout, in goal, and in long-horizon composition. Those axes give you a checklist for coverage. If every task in your set varies the object but none vary the spatial layout, you will not learn whether your policy generalizes across placements, and placement is where §15.5's initial-condition variance bites hardest. Pick tasks so that each axis you care about moves in at least a couple of them.

One more coverage rule that people skip: include a task the policy should refuse or cannot do, if safety matters to you at all. A policy that confidently attempts an impossible grasp and drags the arm into the table is worse than one that stalls, and an evaluation made only of solvable tasks never sees that difference. Chapter 17 builds on this when it treats safety as its own evaluation axis rather than a footnote to success rate.

## Fix the protocol before the first trial

Everything §15.5 argued about noise becomes a set of decisions you write down now, before a single rollout, because deciding any of them mid-evaluation lets bias in through the side door.

The success criterion, first and most important. Write the pass condition for each task as something a stranger could apply without watching you: "the block is fully inside the printed square for two continuous seconds, gripper open." Not "it looks placed." If you cannot state it mechanically, you will adjudicate marginal trials by mood, and your mood correlates with which policy you are rooting for.

The initial-state distribution, second. Decide whether object start poses are gridded or randomized, and if randomized, from what scheme, and write it in a form you can reproduce next month. Mark the positions with tape or a printed template. This is the single biggest lever on the absolute success number, so a rate reported without it is uninterpretable, exactly as in §15.5.

Then the trial count. Work backwards from your decision sentence. If you need to tell a 70% policy from an 85% one, a back-of-envelope power calculation using the binomial spread from §15.5 puts you north of a hundred trials per policy, and no amount of wishing shrinks that. If you only need to know whether a policy clears 50% at all, far fewer will do. Decide N from the gap you must resolve, not from how many rollouts you have patience for, and if patience and statistics disagree, shrink the claim rather than the sample.

Interleaving and blinding round it out. Alternate policies trial by trial so warm motors and drifting light hit both equally. If a human scores the trials, have them score without knowing which policy produced the rollout when you can manage it, because knowing leaks into the borderline calls. RoboArena (Atreya, Pertsch et al., 2025) takes both of these to their logical end with distributed, preference-based scoring across labs; you are running a one-room version of the same idea.

## Log the episode, not the outcome

Here is the move that pays off for months: record every trial in the same shape as a training episode from §15.1, not as a row in a pass/fail spreadsheet. Save the observations, the actions the policy emitted, the language instruction, timestamps, the outcome, and the failure mode when it failed. Two reasons this matters beyond tidiness.

First, a failure you logged fully is a failure you can replay. When policy B misses a grasp on trial 47, a bare "fail" tells you nothing, but the recorded observation stream lets you watch what the policy saw and often see immediately that the object was half out of frame, or the instruction was one you never trained on. The OpenVLA evaluations (arXiv:2406.09246) break results down by task and failure category precisely because the aggregate hides the diagnosis; you get that breakdown for free if you logged the modes as you went.

Second, evaluation rollouts are expensive to collect and identical in format to training data. A hard task your policy fails on is a candidate for your next data-collection session. Throwing that rollout away because your eval harness only kept a boolean is discarding the most valuable data you produce, the data from exactly the states where the policy is weak. Log episodes and your evaluation doubles as a failure-mining pipeline that feeds §16.

A minimal record per trial looks like this:

```python
trial = {
    "policy_id":     "B_wristcam_v3",
    "task_id":       "tight_grasp_bin",
    "seed":          47,             # or gridded init index
    "instruction":   "pick up the hex bolt",
    "episode":       [...],          # obs/action timesteps, §15.1 shape
    "outcome":       "fail",
    "failure_mode":  "no-grasp",     # from a fixed vocabulary
    "duration_s":    None,           # filled only on success
}
```

The `failure_mode` field draws from a small fixed list you settle before running: no-grasp, wrong-object, dropped-in-transit, missed-placement, collision, timeout. A fixed vocabulary is what turns a pile of failures into a confusion table, and a confusion table is what turns "75%" into "it grasps fine but places short," which is a sentence you can act on.

## Sim first, then real, and know what the sim number is worth

If a simulator covers anything close to your task, evaluate there first, because sim buys you the two things real hardware refuses: reproducibility and volume. You can run five hundred seeded trials overnight and get a confidence interval so tight it is almost a point estimate. The chapter's own hands-on exercise leans on this, reproducing an OpenVLA evaluation in SimplerEnv across five seeds and computing intervals over them, and that is the right first move for any new policy.

The catch is the sim-to-real gap from §7.5. A sim success rate is an honest measurement of behavior in the simulator and an optimistic proxy for the real robot, sometimes wildly so. Treat the sim number as a screen, not a verdict: it is excellent for killing bad candidates cheaply and for catching regressions between training runs, and it is not a substitute for the smaller, noisier, more expensive real-robot number you report when it actually matters. A useful habit is to run a handful of real trials on a couple of sim tasks and note the gap; if sim says 90% and hardware says 55% on the same task, you have calibrated how much to discount every other sim figure.

## The checklist

Strip away the reasoning and a defensible home-grown evaluation is this. Write the decision sentence the evaluation must let you finish. Choose tasks that spread across the difficulty and the variation axes you care about, avoiding the ceiling and the floor. Fix the success criterion, the initial-state distribution, the trial count, and the interleaving before the first rollout, all in writing. Log each trial as a full episode with a failure mode from a fixed list, not as a boolean. Screen in sim for volume, confirm on real hardware for truth, and report success rate with a Wilson interval and time-to-completion as a median with spread.

None of this is exotic, and all of it is skipped constantly, which is why so many robot demos evaporate the moment someone else runs them. The discipline is the deliverable. A policy is only as trustworthy as the ruler you measured it with, and once that ruler holds, §15.7 closes the chapter by pulling the dataset, benchmark, and evaluation threads back into a single picture before Part 5 turns to building.
