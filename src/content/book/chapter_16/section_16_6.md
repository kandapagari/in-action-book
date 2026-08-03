---
chapter: 16
section: 16.6
title: Summary
target_words: 2000
status: draft
prereqs: §16.1–§16.5; how to pick a base model by hard constraints rather than leaderboard rank, why a teleop dataset has to cover variation and demonstrate recovery, what LoRA changes and why it is the default, how a sim-to-real loop screens cheaply and confirms honestly, the three fine-tuning failure modes and their opposite fixes, and the ordered recipe card that ties all of it together
key_refs:
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Hu, E. et al. (2021). LoRA, Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
  - Ghosh, D., Walke, H., Pertsch, K. et al. (2024). Octo, An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - Ross, S., Gordon, G., Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS.
---

# 16.6  Summary

Part 4 left you able to read the papers. Chapter 16 was the first chapter of the book about doing the work: taking a foundation VLA you did not train, and a robot the model has never seen, and closing the distance between them on hardware a normal lab owns. The chapter's argument runs against a common instinct. The instinct is that fine-tuning is mostly a training problem, solved by the right hyperparameters and enough epochs. The chapter's claim is that fine-tuning is mostly a data-and-procedure problem, and that the training itself, once you use LoRA, is the easy part. Most of what decides whether your policy works happens before the first gradient step, in which base model you picked and what your dataset contains, and after it, in whether you ran an honest loop or trained once and gave up.

## The ideas worth carrying forward

*The base model is chosen by constraints, not by scores.* §16.1 put four filters in order: control rate on your deployable GPU, action-head match to your motion, fine-tuning cost against your compute, and interface and license fit. A model that tops a benchmark but cannot hit your control rate on your card is disqualified, not a fixer-upper, because §14.4's latency wall is structural. For most new embodiments the survivor is OpenVLA (arXiv:2406.09246) with LoRA, with Octo (arXiv:2405.12213) as the reach for a diffusion head or an odd action space and SmolVLA as the one built for a single cheap arm. The lesson that carries is that the leaderboard number is the last thing to consult, not the first.

*Clean demonstrations make bad training data, and coverage beats volume.* §16.2 was the chapter's most counterintuitive section. A dataset of flawless expert trajectories teaches a policy to be perfect from states it will never re-enter, and the moment it drifts, the compounding error of §6.3 takes over. The fix is to demonstrate recovery, driving back and re-approaching after a fumble on the same episode, which seeds the data with the off-distribution states and their corrections, the DAgger intuition (Ross et al., 2011) applied at collection time. Paired with that is coverage: fifty episodes spread across the workspace, lighting, and distractors beat two hundred clustered in the comfortable center, because the variance you fail to collect is the variance §15.5 said your success rate will die on.

*LoRA makes a 7B fine-tune sane, and it works because the update is low-rank.* §16.3 explained the method and the decision behind it. Freezing the big weight matrix and learning a small additive correction (Hu et al., 2021) cuts trainable parameters by orders of magnitude, which collapses the compute cost to a single GPU, and it preserves the base model's competence by construction, because you cannot overwrite what you never touch. OpenVLA's own table is the number to keep: LoRA matched full-quality fine-tuning at roughly 8x less compute, while last-layer-only collapsed to 30% and a frozen vision encoder capped out near 47%. That last figure is the trap, since it means a LoRA config that skips the visual pathway silently reproduces the weak result.

*Sim buys volume, hardware buys truth, and neither certifies.* §16.4 turned §15.6's screen-then-confirm procedure into a training loop: fine-tune, screen in sim across seeds with a confidence interval, confirm on a small honest real evaluation, read the failures, collect against them, repeat. Simulation earns its place as a cheap way to kill broken fine-tunes and as randomized data augmentation (the domain-randomization bet from §7.5), but a high sim number is evidence, not proof, and the loop never lets it stand as the final answer.

*The same on-robot symptom has three root causes that demand opposite fixes.* §16.4's diagnostic is the piece practitioners get most wrong. A fine-tune drives its loss to the floor and the robot still fails, and "it does not work" can mean over-specialization (works in exact training conditions, fails when anything shifts; fix by widening data and training less, not more), mode collapse (frozen or blended motion because the policy averaged two valid solutions into one that satisfies neither; fix with sharper conditioning or a multimodal head), or action-token mismatch (confident motion to the wrong place because the action convention is wrong; fix by the open-loop replay check that should have happened before collection). The three-set test, exact conditions, shifted conditions, replay check, tells them apart, and applying the wrong fix makes things worse.

## What you should be able to do now

Four things, in the order the recipe card uses them.

You should be able to *pick a base model for a specific robot and defend the choice*. Given a robot's control-rate requirement, its cameras and action space, your GPU, and your licensing needs, you can run §16.1's filter and say why OpenVLA, Octo, or SmolVLA is the right starting point, and why the highest-scoring checkpoint might not be.

You should be able to *collect a teleop dataset that a fine-tune can actually use*. You know to fix and replay-verify the action convention before recording anything, to spread episodes across the workspace and conditions rather than clustering them, to include recovery behavior instead of resetting after every fumble, and to record in the format your model loads so there is no reformatting step to introduce silent bugs.

You should be able to *fine-tune OpenVLA with LoRA and full fine-tuning and know when each is right*. You can configure a LoRA adapter that reaches the vision pathway, explain why it resists catastrophic forgetting, and say when the larger cost of full fine-tuning is justified (large varied data, evidence LoRA's capacity is the bottleneck) versus when action-head-only suffices (re-targeting an action space the model's perception already matches).

You should be able to *diagnose the three failure modes and write a recipe card for a new embodiment*. Handed a fine-tune that fails on the robot, you can run the three-set test, name whether it over-specialized, collapsed a mode, or has a token mismatch, and apply the matching fix. And you can compress the whole process into §16.5's ordered card, understanding why each step precedes the next.

## Where the chapter has set up the rest of the book

Chapter 16 gets a policy working in the lab. Chapter 17 asks the harder question of whether it should be allowed to run outside one. The screen-then-confirm loop here becomes §17.3's A/B evaluation on hardware, the same interleaved, blinded protocol at higher stakes. The episode-logging discipline, every trial recorded as a full episode with a failure mode, becomes §17.4's production logging and rollback, watching a running robot instead of a test harness. And the three failure modes get a fourth, adversarial sibling in §17.5, where the question is not whether the policy fails on its own but whether someone can make it fail on purpose.

There is also a thread the chapter deliberately left open. §16.5's card assumed a robot shaped like an arm, and said nothing about fine-tuning a whole-body or loco-manipulation policy, where a failure can mean the robot fell over and the real-time budget of §14.4 collides with the success statistics of §15.5. That combined problem is one the field has barely standardized, and §18 returns to why the humanoid case makes it harder.

Chapter 16's contribution to the book's argument is to convert everything Part 4 explained into something you can do on Monday. A foundation VLA is not a finished product; it is a starting distribution, and the work of pointing it at your robot is mostly the unglamorous discipline of good data and an honest loop, with the clever low-rank training trick doing far less of the lifting than its reputation suggests. §16.x makes this concrete: you will fine-tune OpenVLA with LoRA on a small dataset and then ablate two knobs, the dataset size and the adapter rank, so the curves the chapter described in words become curves you plotted, and the diminishing returns of both become something you measured rather than took on trust.
