---
chapter: 13
section: 13.5
title: "Open questions in continuous-action foundation models"
target_words: 2000
status: draft
prereqs: §13.4 (the π0 → π0.5 → π0.6 → π0.7 lineage, especially the RL-from-experience thread and the 10,000-hour data bill), §13.3 (flow matching as the objective), §13.1 (why discrete token heads fail on fast, dexterous, long-horizon tasks). Helpful, Chapter 7 on RL and Chapter 11 on scaling.
key_refs:
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Physical Intelligence (2025). π0.5, A VLA with Open-World Generalization. arXiv:2504.16054.
  - Assran, M. et al. (2025). V-JEPA 2 and V-JEPA 2-AC. arXiv:2506.09985.
---

# 13.5  Open questions in continuous-action foundation models

§13.4 sold you on the flow-matching head. A student reading only that section could walk away thinking the continuous-action problem is solved and the rest is data plumbing. It is not solved, and the gaps are not minor engineering to-dos that a bigger cluster will close. Four of them are worth naming precisely, because each one is a place where the field currently has a workaround rather than an answer, and knowing which is which will save you from believing a demo video.

## The reward loop nobody wants to run

§13.4 ended on a promissory note: π0.6 fixes imitation's ceiling with reinforcement learning on the robot's own experience, and on-robot RL is "slow and finicky in ways a simulator hides." That parenthetical hides the single largest unsolved problem in the whole lineage.

Here is the shape of it. Imitation learning is cheap to run once you have the data, because you collect demonstrations offline and then train on a GPU with no robot in the loop. RL is the opposite. To improve from experience the robot has to actually attempt the task, thousands of times, on real hardware that wears out, needs resetting between trials, and occasionally throws a shirt on the floor that a human has to pick up before the next attempt can start. A simulated agent learning to fold laundry can run a million episodes overnight; a real bimanual rig folding real shirts runs maybe a few hundred a day if someone babysits it, and the reward signal for "did it fold well" is not something you can read off a sensor. Somebody, or some learned model, has to score each attempt.

So the open question is not "does RL help" (§13.4 says it does) but "how do you make the loop cheap enough to be worth running." Three partial answers exist and none is satisfying. You can build a simulator good enough that policies trained in it transfer to hardware, which works for rigid objects and falls apart on cloth, fluids, and contact-rich assembly where the physics is exactly the part sim gets wrong. You can learn a reward model from human preference so the robot scores itself, which then imports every failure mode of reward hacking. Or you can accept the slow real-world loop and pour engineering at reset automation and throughput, which is what the well-funded labs quietly do and what a university group cannot afford. The honest state of things is that RL-from-experience works when you have a fleet and a budget, and remains out of reach for everyone else.

## Where do the next 10,000 hours come from

π0 needed roughly 10,000 hours of cross-embodiment teleoperation. That number is the barrier §13.4 flagged, and it points at a question flow matching does not answer on its own: teleoperation does not scale the way web text scaled for language models. Every hour of robot demonstration costs an operator an hour, a working robot, and a physical setup. There is no equivalent of scraping the internet, because the internet does not contain many recordings of a specific gripper's joint velocities.

Two escape routes are under active work and both belong in the "open" column. The first is learning action-relevant structure from human video, of which the internet has an ocean. V-JEPA 2-AC (arXiv:2506.09985) is the sharpest recent evidence that this can work: Meta's team pretrained a world model on unlabeled video, then post-trained it into a zero-shot pick-and-place policy using under 62 hours of robot interaction data, and it reached roughly 80% on a held-out task where Octo (§12.1) managed about 15%. That is a real result, and it says the video prior carries a lot of the load that teleoperation used to carry alone. It is also pick-and-place, not laundry, so nobody should extrapolate it to dexterous long-horizon control yet.

The second route is distilling human first-person video into robot-usable demonstrations, which is part of how the recent large open models get their hours. LingBot-VLA 2.0 (Ant Group, July 2026) reports training on 60,000 hours built from 50,000 hours of robot teleoperation plus 10,000 hours of distilled first-person human video, and running one policy across twenty robot morphologies. Whether the human-video fraction actually substitutes for teleoperation, or just pads the total, is the open question, and the papers do not yet isolate it cleanly. The field is betting heavily that video is the way out of the data trap. The bet is not settled.

## Nobody agrees on how to measure this

Suppose you build a new continuous-action head and want to claim it beats π0. On what? For image classification you report ImageNet accuracy and the field trusts the number. Robotics has no such number, and the substitutes are worse than they look.

The simulation benchmarks a student will meet first (LIBERO, CALVIN, SimplerEnv) run in a simulator, which means they measure performance on exactly the rigid-body, clean-physics tasks that were never the hard part. A policy can top LIBERO and still fumble a real shirt, because the benchmark never tests the cloth deformation, the sensor noise, or the compounding contact dynamics that decide real success. Reporting a sim score for a dexterous policy is a bit like grading a swimmer on a dry-land treadmill: the numbers are precise and mostly beside the point. Real-robot evaluation avoids that trap and creates a new one, because "we folded 8 of 10 shirts in our lab" is not reproducible by anyone who lacks your robot, your shirts, and your kitchen, and small differences in setup swing the number more than the algorithm does.

RoboArena is the most interesting attempt to fix this, borrowing the head-to-head, human-judged comparison format that ranked chatbots and applying it to policies evaluated across many labs, so no single group's shirts define the score. It is early, and it does not solve the deeper problem that dexterous manipulation success is partly a matter of taste, since "folded well" has no crisp definition. Until the field agrees on evaluation, published comparisons between continuous-action models should be read as suggestive, not decisive, and a student reviewing this literature should always ask what "success rate" was measured against before believing the ranking.

## Steerable, but not yet controllable, and not yet safe

Flow matching earns its place by keeping genuine multimodality alive: when a shirt affords two reasonable folds, the head represents both instead of averaging them into a fold that grabs nothing (§13.3). That is the right behavior right up until you want a specific fold, at which point the model's willingness to pick either mode becomes the operator's problem. π0.7 pushed on steerability (§13.4) and made instructions bite at a finer grain, yet the underlying tension has no clean resolution. A policy that commits hard to your instruction loses the flexible generalization that made it useful; a policy that stays flexible is hard to pin down when you need one exact outcome. The lineage manages the trade-off empirically and cannot yet dial it on purpose.

Sitting under steerability is a harder issue the chapter has mostly deferred: these models offer no guarantees. A flow-matching head is a learned velocity field integrated by an ODE solver, and nothing in that machinery certifies that the resulting motion is safe, stays inside a workspace, or refuses an instruction it should refuse. Worse, the same continuous output that makes the head expressive also gives an attacker a smooth surface to push on. Recent work on adversarial patches for VLA policies shows that a printed pattern in the scene can hijack the action output, and the continuous-action models are not exempt. We take safety and adversarial robustness apart in Chapter 17; for now the point is that "it folds the shirt" and "it is safe to deploy near a person" are different claims, and the second is much further from settled than the demos suggest.

## Why does any of this work, and how far does one forward pass reach

The last open question is the one a theorist would have asked first: we do not really know why flow matching is the right objective for control, only that it is empirically better than the alternatives from §13.1. There is no scaling law for action the way there is for language, no clean curve telling you that ten times the data buys a predictable drop in error, so labs discover the returns to scale by spending the money and looking. Chapter 11 gave the scaling intuition; this chapter is where you see how coarse that intuition still is once the output is a robot trajectory instead of a token.

And there is a ceiling built into π0's shape that the next chapter is entirely about. π0 reasons and acts in one forward pass through one model, which is what makes it fast, and which also means it cannot think for longer on a harder problem the way a person pauses before a tricky fold. When the task needs deliberation on a slow clock and reaction on a fast one at the same time, a single head has to serve both, and something gives. Splitting the reasoner from the sensorimotor controller is the response the field converged on, and it is exactly the dual-system design Helix and GR00T N1 use.

Those are the four open problems worth carrying forward: a reinforcement-learning loop too expensive for most labs to run, a data supply that teleoperation cannot fill and video has not yet proven it can, an evaluation culture with no trusted number, and a controllability-and-safety gap that demos paper over. None of them is closed by the flow-matching head that made the rest of the chapter work, which is the useful thing to remember about foundation models for action: the architecture is very good and the surrounding problems are wide open, and confusing the first for the second is how you end up trusting a laundry video too much. §13.6 pulls the chapter's threads together.
