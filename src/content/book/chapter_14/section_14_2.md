---
chapter: 14
section: 14.2
title: "Helix: a high-level VLM and a low-level sensorimotor model"
target_words: 2000
status: draft
prereqs: §14.1 (the two-clocks argument and the System 1 / System 2 vocabulary this section makes concrete), §13.2 (π0's single-backbone-plus-expert asymmetry, the design Helix breaks apart onto two networks), §12.2 (OpenVLA as the 7B single-system point of comparison). Helpful, §11.1 on the CLIP-style pretraining that gives System 2 its open-world grounding.
key_refs:
  - Figure AI (2025). Helix, A Vision-Language-Action Model for Generalist Humanoid Control. figure.ai/news/helix.
  - Figure AI (2026). Helix-02. figure.ai/news/helix-02.
  - Bjorck, J. et al. (2025). GR00T N1, An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 14.2  Helix: a high-level VLM and a low-level sensorimotor model

Section 14.1 gave you the argument for splitting a robot policy across two clocks. Helix (figure.ai/news/helix) is the version of that argument you can point at a running humanoid. Figure released it in February 2025 as, by their description, the first VLA to control a full humanoid upper body at high rate from a single set of weights: two wrists, the torso, the head, and all the fingers on both hands, roughly thirty-five degrees of freedom moving together under one policy. What makes it a good teaching example is not that it works but that the seam between its two systems is drawn so plainly. You can name what crosses that seam and what does not.

## The two halves, sized

System 2 is a 7B open-source vision-language model, pretrained on internet-scale image and text data before Figure ever touched it. That pretraining is the whole point: it is what lets System 2 look at a mug it has never seen on this robot and still know it is a mug, graspable by the rim, part of the category "dishes" that the instruction "put the dishes away" refers to. System 2 takes in the robot's cameras and the language instruction, runs at roughly 7 to 9 Hz, and produces one thing: a continuous latent vector. Not an action, not a joint angle, not a word. A vector that encodes intent, and that System 1 has learned to read.

System 1 is the small one, an 80M-parameter transformer, and it runs at 200 Hz. Twenty-plus times faster than System 2, on a network almost a hundred times smaller. It takes the latest latent vector from System 2 as a standing goal, combines it with its own view of the cameras and the robot's proprioception (joint positions, velocities, wrist forces), and outputs the actual continuous actions that drive the motors. When System 2's latent says "reach for the near mug's rim," System 1 is the part that turns that into a stream of joint targets, cycle after cycle, adjusting for the mug's real position and the hand's real trajectory as they unfold.

The size gap is not an accident, it is the design. System 2 gets to be big because it runs slowly, and it can run slowly because semantic decisions age slowly; "pick up the mug next" is still true a quarter-second later. System 1 has to be small because it runs fast, and it has to run fast because balance and contact will not wait. Each network is sized to the clock its job runs on. That is the §14.1 principle turned into two concrete parameter counts.

## What crosses the seam

The single most important design choice in Helix is what System 2 hands down to System 1, and the answer is: as little as possible, in the form of one latent vector.

Contrast this with the obvious alternative. You could have System 2 emit language, an actual string like "grasp the blue mug's handle and lift slowly," and have System 1 parse it. Figure did not do that, and the reason is instructive. A discrete language token is a bottleneck; you cannot smoothly interpolate between "grasp" and "grasp a little more to the left," and you pay a tokenizer's latency to encode and decode text on every handoff. A continuous latent has neither problem. It can shift by a small amount to mean a slightly different intent, and System 1 can follow that shift continuously rather than waiting for the next discrete word. The channel between the systems is deliberately thin and deliberately continuous, which is what lets a 7 Hz reasoner steer a 200 Hz controller without either one stuttering.

This thin continuous channel is also what makes the two systems trainable as one thing. Because the latent is a differentiable vector rather than a sampled word, gradients from System 1's behavior-cloning loss can flow back through the latent into System 2 during training. Helix is trained end to end on human teleoperation data, roughly 500 hours of it, with an auto-labeling step that runs a VLM over the recordings after the fact to generate the natural-language instruction each clip should be paired with. No task-specific heads, no per-skill fine-tuning: one network learns to fold "put away the dishes" and "hand me the bottle" and a long tail of other instructions into the same latent vocabulary that System 1 knows how to execute.

## Why this beats a single pass, in one task

Take the demo Figure led with: two Figure humanoids standing at a kitchen counter, given groceries they have never seen, told to put them away, cooperating without a script. Watch one robot pick up a bag of cookies and hand it to the other, which places it in a drawer.

System 2 on the first robot reads the scene a handful of times a second. It recognizes the cookie bag as a graspable item, decides the intent "pick up the bag, orient it for a handoff," and holds that intent steady while the hand closes; there is no reason to re-reason mid-grasp, and it could not afford to anyway. System 1, meanwhile, is doing the part System 2 never sees. The bag is deformable, so as the fingers close the contact geometry keeps changing, and the 200 Hz loop keeps adjusting finger positions against the wrist-force readings to hold the bag without crushing it. When the second robot's hand arrives to take the bag, the first robot's System 1 feels the load transfer and releases at the right moment. None of that contact choreography went through System 2. It happened on the fast clock, against fresh sensor data, exactly where §14.1 argued it had to happen.

A single-system policy at 7 Hz would have committed to a grasp and then been blind to the bag deforming under its fingers for 140 milliseconds at a stretch, which for a soft object is the difference between holding it and dropping it or flattening it. A single-system policy at 200 Hz could not have afforded the 7B model that recognized the bag as a bag in the first place. Helix runs both, on the same robot, on two onboard GPUs, one carrying System 2 and one carrying System 1, drawing embedded-scale power rather than a datacenter's.

## From two systems to three: Helix-02 and System 0

Helix as shipped in early 2025 controlled the upper body and assumed something else kept the robot standing. That something was conventional: a hand-written whole-body controller, on the order of 100,000 lines of C++, doing balance and locomotion the classical way from §4.3, a fast model-based loop tracking references and rejecting disturbances. Helix reasoned and manipulated; hand-coded control kept the legs under it.

Helix-02 (figure.ai/news/helix-02), which Figure detailed in early 2026, folds that layer into the learned stack as well. Figure calls it "System 0," a neural whole-body controller that replaces the hand-written locomotion code with a learned policy running below System 1. The naming is telling: not System 3, sitting above the reasoner, but System 0, sitting below the fast controller, closest to the metal and running at the highest rate of the three. So the stack now reads top to bottom as a slow VLM setting intent, a fast visuomotor policy turning intent into arm-and-hand actions, and a still-faster learned controller keeping the whole body balanced under whatever those actions do to the robot's center of mass. This is worth flagging as a boundary quietly dissolving. Chapter 4 drew a clean line between classical model-based control and learned policies; Helix-02's System 0 erases part of that line by learning the locomotion that used to be derived from a dynamics model. We come back to what that means for the field in §18.1.

The other Helix-02 result worth naming is coordination. Figure showed two robots working a shared task with no designated leader, each running its own Helix stack, negotiating a joint activity through what they observe of each other rather than through a central planner handing out subtasks. A May 2026 demonstration had two of them tidying a bedroom together on this leaderless basis. For the argument of this chapter the significance is narrow but real: the dual-system design does not just scale down to one robot's two clocks, it composes across robots, because each robot's slow System 2 is already doing the kind of scene-level reasoning that noticing-what-the-other-robot-is-doing requires.

## Where Helix sits relative to what's next

Helix is one instantiation of the dual-system idea, and it made specific choices: a continuous latent channel, end-to-end training, an 80M fast head, and eventually a learned System 0 underneath. NVIDIA's GR00T N1 (arXiv:2503.14734) makes the same top-level split for humanoids but fills in the boxes differently, most visibly by using a diffusion-transformer action module in the fast path rather than Helix's plain transformer, and by publishing its weights and training recipe where Figure has kept theirs closed. Comparing those two choice-by-choice is the fastest way to see which parts of the dual-system design are load-bearing and which are one team's taste, so that is exactly what the next section does, walking through GR00T N1 and setting it beside Helix box for box.
