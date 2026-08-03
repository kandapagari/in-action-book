---
chapter: 16
section: 16.5
title: "A recipe card for new embodiments"
target_words: 2000
status: draft
prereqs: §16.1–§16.4 (the whole chapter; this section compresses it). Everything here is a pointer back to a section that argued for it, so the card is usable on its own and the prose says where each step's justification lives.
key_refs:
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Hu, E. et al. (2021). LoRA, Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
  - Li, X., Hsu, K., Fu, J. et al. (2024). Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv). CoRL.
---

# 16.5  A recipe card for new embodiments

The chapter's last learning objective is the most practical: write a one-page recipe card you could hand to someone starting a fine-tune on a robot neither of you has worked with before. This section is that card, followed by the reasoning that keeps it from being a cargo cult. The card is ordered, and the order is load-bearing, because the expensive mistakes in this chapter all come from doing a later step before an earlier one, collecting data before fixing the action convention, or training hard before checking coverage.

## The card

**Step 0 — Write the decision sentence.** Before anything, write the sentence the whole project has to let you finish: "this policy picks up the bin and places it on the shelf, at least 80% of the time, across the six starting positions we care about." Everything downstream, how much data, which base model, what counts as success, is set by this sentence. Skipping it is how projects drift into collecting data for a task nobody defined.

**Step 1 — Pick the base model by constraint, not by leaderboard (§16.1).** Filter in this order: control rate on your deployable GPU, action-head match to your motion (smooth/multimodal versus slow/simple), fine-tuning cost against your compute, interface fit to your cameras and action space, and license fit to your intended use. For most new embodiments the survivor is OpenVLA with LoRA; reach for Octo if you need a diffusion head or a non-standard action space, and SmolVLA if your whole budget is one cheap arm and one consumer card.

**Step 2 — Fix and verify the action convention (§16.2).** Settle units, frame, delta-versus-absolute, and gripper encoding to match the base model's expectation. Then verify by open-loop replay: log one teleoperated episode, replay the logged actions on the robot, and confirm the motion reproduces. Do not proceed until the replay matches. This single check prevents the action-token mismatch failure (§16.4) that otherwise surfaces only on the robot, weeks later.

**Step 3 — Collect a first dataset built to cover and to recover (§16.2).** Grid the workspace and spread episodes across zones rather than the comfortable center. Vary object pose, lighting, distractors, and instruction phrasing to match what deployment will vary. Include recovery: when a grasp misses, drive back and re-approach on the same episode rather than resetting. Aim for a few tens to low hundreds of episodes per task for a 7B base model, more for a small one. Record straight into the format the model loads (LeRobot, typically) so there is no reformatting step.

**Step 4 — Fine-tune with LoRA (§16.3).** Start at rank 16. Make sure the adapters reach the vision pathway or the encoder is otherwise trainable, because a frozen encoder reproduces OpenVLA's weak ~47% result. Keep the base weights frozen so you cannot catastrophically forget. Save the adapter, not a full model copy.

**Step 5 — Screen in sim, then confirm on hardware (§16.4).** If a correlated sim covers your task, run the fine-tune across several seeds and compute a confidence interval (§15.x) before spending robot time. Then run a small honest real evaluation: fixed success criterion written in advance, initial states from a stated distribution, every trial logged as a full episode with a failure mode from a fixed list, success rate reported with a Wilson interval (§15.5).

**Step 6 — Diagnose failures with the three-set test (§16.4).** Evaluate on exact training conditions, mildly shifted conditions, and an open-loop replay check. Fails replay: token mismatch, go back to Step 2. Passes training but fails shifted: over-specialization, widen data and lower rank or steps, do not train longer. Fails even training with frozen or blended motion: mode collapse, sharpen conditioning or move to a multimodal head.

**Step 7 — Collect the next batch against the failures, and repeat (§16.4).** The rollouts where the policy is weakest are the most valuable data you can record. Target the next collection session at exactly those conditions, fine-tune again, and turn the crank until the decision sentence from Step 0 is satisfied or you have evidence the base model cannot get there.

## Why the order is the way it is

A recipe card is dangerous if it is followed without understanding, so here is what each ordering choice is defending against.

Step 0 comes first because the most common way a fine-tuning project wastes a month is not a technical failure at all; it is collecting a beautiful dataset for a task whose success criterion nobody pinned down, so the evaluation at the end cannot say whether it worked. The decision sentence is what makes every later number interpretable.

Steps 1 and 2 come before any data collection because both are cheap to get right early and ruinous to fix late. Choosing the base model after you have collected data risks discovering your data does not match its interface. Verifying the convention after you have collected two hundred episodes risks discovering all two hundred are in the wrong frame. The replay check in Step 2 takes ten minutes and saves the single most demoralizing failure in the chapter, the policy that trained perfectly and moves confidently to the wrong place.

Step 3 puts coverage and recovery ahead of volume because §16.2's whole argument is that fifty well-spread episodes with recovery behavior beat two hundred clean clustered ones. The card's instruction to grid the workspace is not fussiness; it is the direct countermeasure to the initial-condition variance (§15.5) that otherwise silently caps your success rate.

Step 4 defaults to LoRA at rank 16 with a trainable vision path because that is the configuration OpenVLA's own experiments found best-per-dollar, matching full-quality fine-tuning at roughly 8x less compute (§12.2). The rank-16 starting point is a starting point, not a law, which is why the exercise ablates it; but starting elsewhere without a reason is how people end up either overfitting a tiny dataset with too much rank or starving a hard task with too little.

Steps 5 and 6 separate screening from confirming and separate the three failure modes because the same on-robot symptom, "it does not work", has three different root causes that demand opposite fixes. The three-set test is the cheapest way to tell over-specialization (fix: less training, more data variety) from mode collapse (fix: sharper conditioning or a multimodal head) from token mismatch (fix: go back to Step 2). Apply the wrong fix and you make it worse, which is why the diagnosis is a step in its own right rather than an afterthought.

Step 7 makes the whole thing a loop rather than a line because a single pass almost never hits the target, and a project that treats fine-tuning as train-once-test-once concludes, wrongly, that the method failed. The loop is what converts a stalled fine-tune into a converging one, and it is the same evaluation-feeds-collection cycle §15.6 and §16.4 built.

## What the card does not cover

The card assumes manipulation on a robot roughly shaped like an arm, because that is where the base models, the datasets, and the tooling are richest. It does not cover fine-tuning a whole-body or loco-manipulation policy, where a failure can mean the robot fell over and the real-time jitter budget of §14.4 shares a page with the success statistics of §15.5. It also stops at the point where the policy works in the lab; getting it to survive a real deployment, with monitoring, logging, and rollback, is a different discipline and the subject of Chapter 17. And it says nothing about certifying that a working policy is safe, because, as §17.5 will argue, nobody yet can.

With the card in hand you have the whole chapter as a procedure. §16.6 steps back from the procedure to the ideas, and names the four things you should now be able to do with a base model, a robot, and an afternoon.
