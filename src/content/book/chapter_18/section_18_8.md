---
chapter: 18
section: 18.8
title: "Beyond vision: tactile sensing as a foundation-model modality"
target_words: 2000
status: draft
prereqs: §18.2 (long-horizon and dexterous tasks; contact bandwidth as the bottleneck this section addresses), §14.4 (latency budgets and the multi-rate control argument), §14.5 (dexterity-focused foundation models GENE and RLDX-1, all vision-first), §10.4 (why smooth multimodal action heads matter for contact-rich motion), §9.7 (world-model prediction as a policy component, here applied to touch), §16.2 (what a dataset for a new modality would cost you).
key_refs:
  - Zhou, J., Hong, F., Li, Y. et al. (2026). TouchWorld, A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation. arXiv:2607.07287.
  - Genesis AI (2026). GENE-26.5 dexterous foundation model, and the 1:1:1 data-glove pipeline. Company announcement.
  - RLWRLD (2026). RLDX-1 Technical Report. arXiv:2605.03269.
  - mimic Robotics (2026). M1 hand, U1 wearable exoskeleton, full-stack dexterous platform. Company announcement, 18 July 2026.
---

# 18.8  Beyond vision: tactile sensing as a foundation-model modality

Every model in this book takes pictures and words as input. The V in VLA is a camera, the L is language, and the A comes out the other end. §18.2 explained why that combination struggles with dexterity, and one clause in that explanation deserves its own section, because it names a missing sense rather than a missing amount of data: contact dynamics are stiff and near-discontinuous, and a policy has to react to a slip before the object falls. A camera cannot see a slip. It can see the consequence of one, a few frames late, which for a falling mug is too late.

This is the argument for touch, and it has become a foundation-model argument rather than a sensor-engineering one. The interesting recent work does not add a tactile input channel to an existing VLA. It asks what kind of model touch wants to be part of, and answers that it wants to sit on a faster clock than vision does, which turns out to have the same shape as the dual-system argument of Chapter 14 and the world-model argument of §9.7.

## What vision cannot tell you

Be precise about the gap, because "add touch" sounds like a hardware shopping list and the reason it matters is informational. Vision and language give you semantics and geometry: what the object is, roughly where it is, what the instruction refers to. They do not reveal the quantities that decide whether a contact-rich manipulation succeeds. How much force is on the fingertip. Whether the object is slipping against the skin right now. Whether the grasp is stable or one perturbation from failing. Whether the surfaces are seated or merely touching.

These are hidden states in the §9.1 sense: not noisy observations of something the camera could see in principle, but variables occluded by the contact itself. The fingertip is between the camera and the thing you want to know about. A person closing a hand on a glass knows the grip is adequate without looking, and revises it continuously, from a signal that has no visual counterpart.

There is a second gap that is about rate rather than content. §14.4 established that a VLA running at five to twenty hertz is doing acceptable manipulation and marginal contact control. Slip correction wants to be faster than that. A tactile signal is cheap to sample at hundreds of hertz, and the correction it drives is small and local, adjust this finger's force slightly, which is exactly the kind of computation you do not want behind a large transformer. So touch does not just add information; it adds information whose useful timescale is an order of magnitude off from the rate the rest of the policy runs at, and any architecture that treats it as one more observation stream into the main model wastes it.

## TouchWorld, and the shape of the answer

TouchWorld (arXiv:2607.07287, Zhou et al., July 2026) is the most complete attempt so far to build touch in structurally rather than bolt it on, and its diagnosis is stated in exactly the terms above: most existing policies treat touch as a low-frequency observation inside a monolithic action model, coupling slow task reasoning, action generation, and fast contact feedback in a single loop.

Their architecture separates those jobs into a hierarchy of four, and the layering is the contribution. A high-level planning layer does vision-language subtask decomposition, and, in the move that gives the model its name, predicts *tactile subgoals*, what the contact should feel like if the subtask is going correctly. A visuo-tactile goal-conditioned policy generates nominal action chunks against those goals. And a tactile-conditioned refinement policy runs on top of the chunk, performing online residual correction from recent tactile and proprioceptive feedback at high frequency.

Two ideas in that stack are worth extracting independently of whether TouchWorld itself is the design that survives.

The first is a predicted tactile subgoal, which is §9.7's insight applied to a different sense. A world model earns its keep by predicting what the next observation should be; here the model predicts what the *contact* should be, and the difference between predicted and measured touch is an error signal available immediately, without waiting for the visual consequence. That reframes touch from a feedback channel into a reference channel, and it gives the policy something a purely reactive tactile controller lacks, a notion of what correct feels like at this point in the task.

The second is residual correction as the fast layer. The refinement policy does not choose what to do; it adjusts what the slower layer already decided, which keeps the fast loop small enough to run fast and keeps the semantic generalization of the vision-language layers intact. This is the §14.1 dual-system argument arriving for a third time, first for reasoning versus control, then in §14.2's System 0 for whole-body balance, now for contact. The recurring lesson is that the field keeps discovering that different parts of a manipulation problem want different clock rates, and that stacking them beats unifying them.

The reported numbers are modest in absolute terms and interesting in their spread. Across six long-horizon, contact-rich tasks, TouchWorld reports 65.0% success in the clean setting and 53.7% under human perturbations, beating the strongest baseline by 15.7 and 18.5 percentage points respectively. Notice that the margin *grows* under perturbation. That is the pattern you would predict if the tactile loop is doing real work, since perturbation is precisely the condition where a policy needs to feel what went wrong, and it is a more convincing argument for the architecture than the clean-setting number.

## Why this is a data problem before it is a model problem

The reason tactile foundation models arrived years after visual ones is not that nobody thought of touch. It is that the data does not exist at anything like the scale vision data does, and the reasons are structural enough to be worth understanding before you plan a project around them.

Tactile sensors are not standardized. A camera is a camera; images from a thousand sources compose into one dataset, which is what made §12.4's Open X-Embodiment and the whole internet-pretraining story possible. Tactile hardware is a zoo of vision-based sensors, capacitive arrays, piezoresistive skins, and magnetic designs, each with its own resolution, dynamic range, and failure mode. A tactile dataset collected on one fingertip does not obviously transfer to another, so the cross-embodiment aggregation that carried visual VLAs is much harder here.

There is also no passive corpus. §9.4's central advantage for video was that the internet is full of it, and §9.7's WAM result got a 42% relative improvement out of unlabeled human video. There is no YouTube of touch. Every tactile sample has to be physically collected by something instrumented, which puts the field in the position visual robot learning was in before Open X-Embodiment, and it explains why the dexterity efforts of §14.5 all built bespoke collection pipelines: Genesis AI's data glove with its 1:1:1 mapping between human hand, glove, and robot hand, and mimic Robotics' pairing of an M1 hand with a U1 wearable exoskeleton, announced in July 2026 as a full-stack platform. Those are not incidental hardware products. They are data-acquisition strategies for a modality with no free corpus, and reading them that way tells you what the companies think the bottleneck is.

The honest consequence for a reader is that tactile work is hardware-gated in a way the rest of this book is not. You can fine-tune a VLA with a borrowed arm and a public checkpoint (Chapter 16). You cannot study tactile foundation models without a tactile sensor, and your results will be entangled with which one you bought.

## Where this sits among the open problems

Touch is the clearest case in this chapter of a capability gap that is not obviously a scale problem. §18.2 laid out a live disagreement about dexterity, whether it emerges from scaling a generalist or requires purpose-built models, and tactile sensing gives the specialist camp its strongest argument: no amount of visual data contains the information a fingertip provides, so this is not a gap scale can close. If that argument is right, the generalist trajectory this book has traced from RT-1 through π0.7 has a ceiling on contact-rich tasks that no additional teleoperation of camera-and-instruction data will lift.

The counter-argument deserves stating, because it is not weak. Humans deprived of tactile feedback manipulate poorly but not hopelessly, vision plus proprioception plus a good predictive model of contact carries you a long way, and §9.7's WAMs are precisely a bet that a model with a strong physics prior can anticipate contact it cannot feel. It is possible that predicted contact from a good world model substitutes for measured contact more than anyone expects. TouchWorld's own design hedges toward the middle by predicting tactile subgoals rather than only reacting to tactile input, which is a world model for touch and a tactile sensor working together.

What would settle it is an ablation someone should run and publish: the same task suite, the same policy architecture, with the tactile channel present, ablated, and replaced by a predicted tactile signal from a world model. TouchWorld's perturbation results are suggestive but confounded with everything else its hierarchy changes. Until that experiment exists, tactile sensing belongs in this chapter as an open problem rather than a solved ingredient, and it belongs near the top of the list, because it is the one place where the field may be missing a sense rather than a scale.
