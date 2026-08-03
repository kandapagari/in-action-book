---
chapter: 16
section: 16.2
title: "Building a teleop dataset that does not waste your time"
target_words: 2000
status: draft
prereqs: §16.1 (you have picked a base model and know its expected observation and action interface, which fixes what your data has to contain). §15.1 (the episode format your recordings must match, timestamped observation-action dictionaries), §15.5 (the initial-condition variance that your data has to cover or your policy will not), §6.2–§6.3 (behavior cloning and compounding error, the reason recovery behavior in the data matters more than clean demonstrations).
key_refs:
  - Walke, H. et al. (2023). BridgeData V2, A Dataset for Robot Learning at Scale. CoRL 2023.
  - Cadene, R., Alibert, S., Soare, A. et al. (2024). LeRobot, State-of-the-art machine learning for real-world robotics in PyTorch. github.com/huggingface/lerobot.
  - Ross, S., Gordon, G., Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS.
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
---

# 16.2  Building a teleop dataset that does not waste your time

The most expensive mistake in this whole chapter is collecting the wrong 200 episodes. Teleoperation is slow, it needs a human at the controls, and every session drifts the setup a little, so a dataset that turns out to be unusable is not just wasted GPU time, it is a wasted afternoon of your life you cannot get back by re-running a script. This section is about spending that afternoon well. The governing idea is simple to state and hard to obey: your fine-tuning data has to look like the situations the deployed policy will face, including the ones where things go wrong, and most first datasets fail because they contain only the situations where things went right.

## Why clean demonstrations make bad data

The instinct when you sit down to teleoperate is to perform. You reset the scene neatly, you drive the arm along a smooth confident path, the gripper closes at exactly the right moment, and you log a beautiful success. Do that two hundred times and you have built a dataset that teaches the policy one thing: how to be perfect from a starting state it will never see twice. The trouble is behavior cloning, and §6.3 already named it. A policy trained only on expert trajectories has never seen the states that follow a small mistake, so the first time it drifts even slightly off the demonstrated path, it is in unfamiliar territory and the errors compound. Your gorgeous demonstrations are the exact reason it falls apart.

The fix is not to demonstrate badly. It is to demonstrate recovery. When the grasp misses, do not reset and pretend it did not happen; keep teleoperating, drive the arm back, re-approach, and complete the task. Now the dataset contains the state "gripper is near the object but not aligned" together with the action "back off and re-approach," which is precisely the correction the policy needs when it makes that mistake on its own. This is the human-in-the-loop intuition behind DAgger (Ross et al., 2011) applied at collection time rather than as a separate correction loop: you are deliberately seeding the data with the off-distribution states and their fixes, so the policy learns to get back on track instead of only how to stay on it.

## Coverage beats volume

The second failure mode is a dataset that is large but narrow. Two hundred episodes of the same mug in the same spot under the same lighting is two hundred near-duplicates, and the policy that trains on it will fail the moment the mug moves five centimeters or a cloud passes the window. §15.5 called this initial-condition variance and showed it is where real-robot success rates go to die. Your data is the only place the policy learns to tolerate it, so the variance has to be in the data on purpose.

Vary the things the deployed robot will see vary. Object position and orientation across the reachable workspace, not clustered in one comfortable zone. Lighting, if your deployment site's lighting is not controlled. Distractor objects in the scene, so the policy learns which object the instruction refers to rather than "the only thing on the table." The specific instruction phrasing, if you want language robustness, because a policy trained only on "pick up the red block" can be surprisingly brittle to "grab the red one." A useful discipline is to grid the workspace: mentally divide the reachable area into a handful of zones and make sure each zone gets its share of episodes, rather than letting your hand default to the center where teleoperation is easiest. Fifty well-spread episodes routinely beat two hundred clustered ones, and they take a quarter of the time to collect.

## Match the format to the model, from the first episode

Before you record anything, settle the format, because reformatting a dataset after the fact is the kind of tedious, bug-prone work that eats a day and introduces silent errors. §15.1 laid out the episode anatomy: a list of timesteps, each a dictionary pairing an observation with the action that followed. Your recording has to produce that shape, and it has to produce it in the units and frames your base model expects, which §16.1 told you to check early. This is where the single most common silent failure in the whole pipeline lives, and it deserves its own paragraph.

The action-convention mismatch is the bug that trains a policy that looks fine and behaves like it is drunk. Your teleoperation system logs actions in some convention, absolute end-effector pose, or delta pose, in meters or millimeters, in the base frame or the camera frame, with the gripper as a binary or a continuous width. OpenVLA (arXiv:2406.09246) expects 7-DoF end-effector deltas normalized to its training statistics. If you log absolute poses and feed them to a model expecting deltas, nothing errors out; the numbers are the right shape and the wrong meaning, and the loss will even go down as the policy learns to predict your (wrong-convention) actions. You will discover the problem only on the robot, when it moves confidently to the wrong place. Fix the convention before collection and verify it by replaying a logged episode open-loop on the robot: if the replay reproduces the demonstrated motion, your convention is right, and if it does not, you just saved yourself the fine-tune.

The LeRobot format (§15.3, Cadene et al., 2024) is worth adopting here for a concrete reason beyond tidiness: it is what several base models load natively, its dataset tooling handles the timestamp bookkeeping for you, and its delta-timestamp query lets the same recorded episode feed a policy that wants a single action and one that wants a sixteen-step horizon. Recording straight into that format means your data is loadable by the training loop with no conversion step, which removes an entire class of the reformatting bugs above.

## How much data, and how to tell when you have enough

There is no universal number, but there is a shape to the answer. For fine-tuning a 7B generalist that already knows the visual world, the surprising finding across OpenVLA-style fine-tunes is that a few tens to low hundreds of episodes per task often move the success rate substantially, because the base model is not learning to see, only learning to map what it already sees onto your robot's actions. For a small model that has not memorized as much, you will need more, because it is doing more of the learning from your data alone. The chapter's exercise ablates dataset size directly so you can watch the curve for yourself rather than trust a number I quote.

The practical way to find your enough is to collect in rounds and let evaluation tell you when to stop. Collect a first batch, fine-tune, evaluate with the honest protocol from §15.5 and §15.6, and look at where it fails. Then, and this is the move that makes the whole chapter cohere, collect your next batch targeting those failures specifically. If the policy fails when the object is at the left edge, spend the next session putting the object at the left edge. Evaluation and data collection are one loop, not two activities, and the rollouts where your policy is weakest are the most valuable episodes you will ever record. A dataset built this way grows toward the policy's actual weaknesses instead of padding its existing strengths.

## A collection checklist

Before the session, decide and write down: the exact action convention and units, verified by open-loop replay; the observation the model expects, cameras and any state vector, at the right resolution; the instruction phrasing set; and the workspace zones you will cover. During the session, include recovery behavior rather than resetting after every fumble, spread episodes across the zones instead of the comfortable center, and vary the conditions you know the deployment will vary. After the session, replay a random few episodes to confirm the logs are sane, and check the dataset loads in your training pipeline before you collect the next batch, not after you have collected all of them.

Do that and the two hundred episodes you collect are two hundred you will actually use. Now they need to go into the model, and the question of which parts of a 7B network you update, and at what cost, is the one that decides whether your single GPU is enough. That is §16.3: LoRA against full fine-tuning against touching only the action head.
