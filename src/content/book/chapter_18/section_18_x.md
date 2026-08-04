---
chapter: 18
section: 18.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §18.1–§18.5 (the four open problems and the year-1 plan this exercise starts), §18.5 specifically (the reading path and the 500-word-note discipline), §15.6 (evaluation-first thinking, which the note has to respect to be honest). No GPU or robot needed; this exercise is reading and writing, and it is the on-ramp to a real project rather than a toy.
key_refs:
  - Open X-Embodiment Collaboration (2023). arXiv:2310.08864.
  - Assran, M. et al. (2025). V-JEPA 2. arXiv:2506.09985.
  - ERVLA (2026). arXiv:2606.03784.
  - Zheng, J. et al. (2025). UniAct. arXiv:2501.10105.
---

# 18.x  Hands-on exercise + chapter references

Every other chapter's exercise had you run code. This one, the last in the book, has you do the thing that actually starts a research career: read the current literature on one open problem and take a position on it in writing. The TOC states the target plainly: pick one open problem from the chapter, find three recent papers on it, and write a 500-word position note. That sounds soft next to wrapping a safety shield around OpenVLA, and it is the harder skill, because a position note forces you to say what you think is true and what you would do about it, which running someone else's code never does. §18.5 argued that a method without an evaluation is a demo. A field without a position is a reading list, and this exercise is how you stop being a reading list.

No hardware is required, which is the point. The gap between a person who has read this book and a person who contributes to the field is not a GPU; it is the habit of turning reading into a claim. This exercise builds that habit once, in 500 words, so that the year-1 plan from §18.5 has a first page.

## Exercise 18.x.1 — Pick the problem that matches what you have

Do not pick the problem that sounds most exciting. Pick the one you could actually run an experiment on, because a position note you cannot follow with a project is an essay, and §18.5 matched each open problem to the setup it needs. If you have one robot arm, the long-horizon failure taxonomy (§18.2) or the cross-embodiment where-does-it-break study (§18.1) is yours. If you have a GPU and no robot, video pretraining's data-efficiency question (§18.3) or the ERVLA inference-reasoning question (§18.4) is yours. Write one sentence naming the problem and one sentence naming the experiment you could run on it. If the second sentence is hard to write, pick a different problem; the difficulty is telling you the truth.

## Exercise 18.x.2 — Find three recent papers, the right way

Three papers, and the selection matters more than the number. Do not pick three that agree. Pick the one that best states the current approach, one that reports a result complicating it, and one from the last few months that you found yourself, not from this book's reference list, because the skill being tested is finding the frontier after the book stops. Use the arXiv listing for cs.RO and cs.LG, follow the citation graph forward from a paper you know using Semantic Scholar or Connected Papers, and read each paper's failure section before its results, per §18.5. For each, write two lines: what it claims, and what it admits it could not do. That second line is where your position will come from.

## Exercise 18.x.3 — Write the 500 words, with a claim in the first sentence

Now the note. Five hundred words, and the first sentence is a claim, not a background statement. "Cross-embodiment transfer will not be solved by a universal action space, because the three papers below all report the same failure at the same place" is a first sentence. "Cross-embodiment transfer is an important open problem" is not, because it commits to nothing. Structure the rest in four short movements rather than a five-paragraph essay: the claim, the evidence from your three papers, the strongest objection to your claim stated in one honest sentence, and the experiment you would run to settle it. Keep the experiment small enough to actually do, because §18.5's year-1 plan says the note is the first page of the project, and a project you cannot start is a project you will not finish.

Two failure modes to avoid, both common. The hedging note, which surveys the three papers evenhandedly and concludes "more research is needed," is not a position and gets no credit; the whole exercise is refusing that sentence. The overconfident note, which claims the problem is trivially solved by an idea you have not tested, is worse, because it skips the objection movement that keeps you honest. A good note is a claim you would defend and an experiment that could prove you wrong, which is the shape of every result worth publishing.

## Going further

Two extensions turn the exercise into the project. Run the small experiment the note ends with, even a rough version on a subset, and add a paragraph reporting what happened, because a position note backed by one data point outranks a dozen that are pure argument. And send the note to one person working on the problem, a paper author, a lab's public channel, a course instructor, and ask the single question the note raised for you; the field is small and young enough (§18.5) that this works more often than you expect, and a real answer from someone in the middle of the problem is worth more than another paper.

## Chapter 18 reading list

Cited across §18.1–§18.6, grouped by the bet each reference serves. Full entries for the whole book live in Appendix E.2; this is the chapter-local subset, and it doubles as the ordered reading path §18.5 recommended.

### Cross-embodiment generalization

- Open X-Embodiment Collaboration (2023). "Open X-Embodiment: Robotic Learning Datasets and RT-X Models." arXiv:2310.08864. §18.1's anchor: the pooled multi-robot dataset and the positive-transfer result every later cross-embodiment paper argues with.
- Octo Model Team (2024). "Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213. §18.1's heterogeneity-absorbing design, with swappable input and output heads; also §18.3's conventional-generalist baseline that V-JEPA 2-AC beat.
- Zheng, J., et al. (2025). "Universal Actions for Enhanced Embodied Foundation Models (UniAct)." arXiv:2501.10105. §18.1's learned universal action space, the cleanest statement of the claim that there is a shared behavior level above joints and below language.
- Google DeepMind (2026). "Gemini Robotics 2." Announcement, 2026-07-30. §18.1's whole-body loco-manipulation result, the first VLA to drive legs, torso, arms, and hands under one policy, dissolving the §4 locomotion boundary.

### Long-horizon and dexterous tasks

- Long-VLA (2025). "Long-VLA: Unleashing Long-Horizon Capability of Vision-Language-Action Models." arXiv:2508.19958. §18.2's learned task-decomposition attack on the compounding-error problem.
- Fang, H., et al. (2025). "LiLo-VLA." arXiv:2602.21531. §18.2's companion long-horizon result in the same decomposition vein.
- Physical Intelligence (2025). "π*0.6 / RECAP: Learning from Experience." pi.website/download/pistar06.pdf. §18.2's reinforcement-learning-from-experience result on a foundation policy, the return of §6.5's tool for the robustness cloning cannot buy.

### Video-pretrained action models

- Assran, M., et al. (2025). "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." arXiv:2506.09985. §18.3's central result: a world model from over a million hours of video, post-trained into a zero-shot pick-and-place policy on under 62 hours of robot data.
- Bruce, J., et al. (2024). "Genie: Generative Interactive Environments." ICML 2024. §18.3's latent-action branch, inferring a made-up action space from action-free video, the cleaner statement of recovering actions video never recorded.
- Nair, S., et al. (2022). "R3M: A Universal Visual Representation for Robot Manipulation." CoRL 2022. §18.3's representation-only precursor: video pretraining that improves the visual front end but still needs robot demonstrations for the policy.

### Reasoning joined to action

- Zawalski, M., et al. (2024). "Robotic Control via Embodied Chain-of-Thought Reasoning (ECoT)." §18.4's canonical embodied chain-of-thought, reasoning tokens generated before the action in one autoregressive stream.
- ERVLA (2026). "Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation." arXiv:2606.03784. §18.4's sharpest move: train with reasoning traces, drop them at inference via CoT-dropout, paying none of the latency for the learning benefit.
- Gemini Robotics-ER (2026). "Embodied Thinking." arXiv:2510.03342. §18.4's foundation-scale embodied reasoning, and §17.5's analyzability argument turned into a product choice: legible reasoning over an opaque pass.
- Embodied-R1 (2025). arXiv:2508.13998. §18.4's grounded-spatial-reasoning line, where the reasoning output, a grasp point, is verifiable in a way a paragraph of chain-of-thought is not.

### Surveys, for breadth after the primary sources

- VLA Models: Concepts, Progress, Applications and Challenges (2025). arXiv:2505.04769. §18.5's recommended map of the whole area, useful once you know the territory and useless as a first read.
- Efficient VLA Survey (2025). arXiv:2510.24795. §18.5's companion survey on the latency and compute constraints that §14.4 and §18.4 both run into.

## Chapter summary

Chapter 18 took the open problems that Chapter 17's honesty exposed and turned them into a map and a plan. You can now summarize the four live research bets as specific questions with specific failure modes rather than as buzzwords: cross-embodiment transfer as a representation problem that works within a robot class and breaks across bodies, with whole-body neural control now dissolving the classical locomotion boundary from Chapter 4; long-horizon and dexterity as two distinct failures, one of accumulated error and one of contact bandwidth, neither near a human; video pretraining as the data bottleneck's most concrete answer yet, proven on short tasks by V-JEPA 2 and unproven on hard ones; and reasoning-plus-action as a real subfield whose central tension is the latency of thinking, with ERVLA's train-then-drop trick the sharpest current resolution. You can map these onto one picture as coordinated attacks on the same capability shortage, identify which one your actual resources fit, and leave with an ordered reading path and a year-1 arc that puts reproduction and evaluation before method. And after this exercise you have written the first 500 words of that arc: a position note that makes a claim, backs it with three papers you chose, states its own strongest objection, and ends in an experiment small enough to run. That note is where this book stops being something you read and starts being something you do. The field is young, the useful corners are full of unmeasured failures, and the person who finishes here and pushes on one of them is a participant in it, not an observer.
