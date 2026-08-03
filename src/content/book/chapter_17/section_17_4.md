---
chapter: 17
section: 17.4
title: "Logging, alerting, and rollback"
target_words: 2000
status: draft
prereqs: §2.4 (the most expensive bugs in a deployed VLA are the ones that do not crash — this section is the production answer), §15.6 (log every rollout as a full episode with a failure mode; here that discipline becomes a production system), §17.2 (the shield's violation logs, the raw signal this section alerts on), §17.3 (the A/B and canary machinery whose regressions rollback reverts). §15.1 (the episode format the logs should match so they are replayable and reusable as training data).
key_refs:
  - Cadene, R., Alibert, S., Soare, A. et al. (2024). LeRobot, State-of-the-art machine learning for real-world robotics in PyTorch. github.com/huggingface/lerobot.
  - Walke, H. et al. (2023). BridgeData V2, A Dataset for Robot Learning at Scale. CoRL 2023.
---

# 17.4  Logging, alerting, and rollback

A crash is the easy failure. It stops the robot, raises an exception, and points at its own cause. §2.4 named the failure that actually costs you: the bug that does not crash, the flipped image or wrong normalization that produces confident, wrong behavior and raises nothing. On a laptop that bug wasted an afternoon. In production, on a robot running unattended for a shift, the same class of silent degradation runs until something breaks or someone notices, and "someone notices" is not a monitoring strategy. This section is about the machinery that catches the silent kind: what to log so a failure is reconstructable, what to alert on so you learn about it before the damage, and what to roll back so learning about it is survivable.

## Log the whole episode, not the outcome

The unit of a production log is the same unit Chapter 15 used for a dataset: the episode. §15.6 told you to log every evaluation rollout as a full episode with a failure mode; production is that discipline turned always-on. For every run the robot makes, you record the stream of observations, the actions the policy proposed, the actions the shield actually let through, the proprioception, the timestamps, and the shield-intervention log from §17.2. Record it in the episode format from §15.1, the one LeRobot (Cadene et al., 2024) standardizes, and you get two things at once: a log you can replay to reconstruct exactly what happened, and, because it is already in training format, data you can fold straight back into the next fine-tune.

```python
record = {
    "episode_id":   uuid(),
    "model_version": "openvla-ft-2026-08-03-a",   # WHICH policy produced this
    "metadata": {                                  # the stuff §2.4 said travels silently
        "unnorm_key": "my_robot", "prompt_template": "openvla-v1",
        "camera_calib_hash": "3f9a...",
    },
    "timesteps": [ ... ],          # obs, proposed_action, executed_action, force, dt
    "shield_events": [ ... ],      # every veto, with the predicate that fired
    "outcome": {"success": None, "failure_mode": None},   # filled in later
}
```

The two fields people leave out are the two that matter most in an incident. The model version, because when a robot misbehaves the first question is always "which policy was running," and a log that cannot answer it makes every incident an investigation from scratch. And the metadata that §2.4 warned travels implicitly, the `unnorm_key`, the prompt template, the camera calibration, because the silent bugs are precisely mismatches in that metadata, and a log that records it lets you diff a bad run against a good one and find the flipped bit. A log that stores only "success: false" has thrown away everything you need to know why.

## Alert on leading indicators, not on crashes

Alerting on crashes is alerting too late. The signal you want is the drift that precedes the failure, and you already have the sensors for it from §17.2. The shield-intervention rate is the best single leading indicator: if the fraction of actions the shield has to veto is climbing, the policy is proposing more unsafe actions than it used to, which means it is drifting off distribution before any single failure is visible. Action-statistics drift is another, the §6.3 compounding-error signal, flagged when the policy's outputs move away from the distribution it produced in evaluation. Rising time-to-completion, the §15.5 metric, catches a policy that is still succeeding but working harder, which often precedes a policy that stops succeeding. Force readings creeping toward the limit, success rate sliding on a canary slice, all of these move before the robot actually fails.

The discipline is to pick thresholds that fire early enough to act and rarely enough to trust. An alert that fires constantly is an alert everyone ignores, which is worse than no alert because it launders inattention into process. So set thresholds off the evaluation baseline, alert on a sustained shift rather than a single noisy sample, and tier the response: a soft threshold that logs and notifies, a hard threshold that stops the robot and pages a human. The hard threshold is the one that has to fire faster than the damage, which means it lives close to the shield, on the fast clock, not in a dashboard someone checks hourly.

## Rollback: versioning is the prerequisite

Rollback is only possible if you can name what you are rolling back to, so model versioning is the unglamorous prerequisite that makes the rest work. Every deployed policy is a named, stored checkpoint, and every log records which one ran, so that when B regresses you can revert to A without a rebuild. This is the same blue-green and canary machinery §17.3 used to compare policies, now used to retreat: keep the previous version loaded and ready, and make switching back a single deliberate action rather than a redeploy under pressure.

Define the rollback triggers before you deploy, not during the incident. A rollback trigger is a written rule, "if the shield-intervention rate on the canary exceeds twice baseline for five minutes, revert to the previous version," decided in advance and applied mechanically, because deciding it while watching a robot misbehave is how a two-minute rollback becomes a twenty-minute argument. The trigger has to beat the damage: a rollback that takes an hour is no defense against a policy that hurts something in a minute, which is why the fast, hard stop from the alerting tier and the slower version-revert are two different mechanisms. The stop buys you the time; the revert fixes the cause.

## The loop closes here

There is a payoff to logging every run as a replayable episode that goes beyond incident response, and it is the loop this book has been building toward since Chapter 15. The runs where your deployed policy is weakest, the ones that tripped the shield, drifted the statistics, or failed outright, are the most valuable training data you own, exactly as §15.6 and §16.4 argued. Because you logged them in training format, they flow straight back into the next fine-tune, so the failures your deployment surfaces become the data that fixes them. Evaluation, deployment, and data collection are not three activities; they are one loop, and production logging is the arc that closes it. A robot that runs, logs its failures, and feeds them back is a robot that gets better at the tasks it is worst at, which is the only kind of improvement that compounds.

This also reframes what a deployed VLA is. It is not a finished artifact you install and forget; it is a running process you watch, measure, and revise, more like a service in production than a shipped appliance. The monitoring, alerting, and rollback stack is the operational half of that, and it is why deployment is a discipline distinct from fine-tuning: §16 got the policy working, and this chapter keeps it working, and safe, once it leaves the lab. The techniques so far, layers, monitors, honest A/B, logging and rollback, all reduce risk and catch failures. None of them proves the policy safe. §17.5 confronts that directly, and names the residual risks, adversarial and otherwise, that no amount of this machinery currently lets anyone certify away.
