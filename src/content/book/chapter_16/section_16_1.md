---
chapter: 16
section: 16.1
title: "Picking a base model: OpenVLA, Octo, or a smaller distilled checkpoint"
target_words: 2000
status: draft
prereqs: §12.2 (OpenVLA's architecture and its fine-tuning decision table, the LoRA-at-68% result you will lean on all chapter), §12.3 (Octo's diffusion head and why the action representation is the axis these models differ on), §10.5 (action-head choices), §14.4 (the latency wall a 7B backbone hits, which decides whether a checkpoint can control your robot at all). Helpful, §15.3 (LeRobot, the ecosystem several of these ship inside).
key_refs:
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Ghosh, D., Walke, H., Pertsch, K. et al. (2024). Octo, An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - Shukor, M. et al. (2025). SmolVLA, A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844.
  - Wen, J. et al. (2024). TinyVLA, Towards Fast, Data-Efficient Vision-Language-Action Models. arXiv:2409.12514.
---

# 16.1  Picking a base model: OpenVLA, Octo, or a smaller distilled checkpoint

Part 4 handed you a shelf of trained VLAs and explained how each works. Part 5 asks a blunter question: which one do you download and fine-tune tonight, on the robot you actually own? The honest answer is that the choice is mostly made for you by four constraints you cannot wish away, and only after those have done their filtering does taste enter. Those constraints are the control rate your robot needs, the GPU you have to serve the model on, the shape of the data you can collect, and the license you are allowed to build on. Get those straight and the field of candidates usually collapses to one or two.

Start with the constraint that kills the most options first, because it is the one people ignore until their robot is jerking around: the control rate. §12.2 gave OpenVLA's honest latency, about 6 Hz on an RTX 4090 in bfloat16, and §14.4 explained why that number decides everything downstream. A tabletop pick-and-place that tolerates a deliberate, one-action-per-sixth-of-a-second cadence is fine on a 7B model. A robot that has to react to contact, correct a slipping grasp, or hold a balance loop is not, and no amount of fine-tuning fixes a base model whose forward pass is slower than the loop it has to close. So before comparing checkpoints on quality, write down the frequency your task demands, and treat any model that cannot hit it as disqualified rather than as a fixer-upper.

## The three tiers on the shelf

The open checkpoints worth starting from in 2026 sort into three rough tiers by size, and the tiers matter more than the individual names because they trade the same things against each other.

The 7B generalists are OpenVLA (arXiv:2406.09246) and π0 (arXiv:2410.24164). These are the models with the most competence per download: trained on the largest pools, best at absorbing a new task from few demonstrations, and the ones a paper will compare against. They also want a real GPU to serve, land in the single-digit-hertz range for the token-head ones, and take the longest to fine-tune. OpenVLA is the safe default here, and most of this chapter uses it, because its fine-tuning story is the most thoroughly documented open recipe in existence and its decision table (§12.2) is something you can act on directly.

The mid-size generalists are Octo, at 27M and 93M parameters (arXiv:2405.12213). Octo is an order of magnitude smaller than OpenVLA, carries a diffusion action head instead of a token classifier, and was built from the start to be fine-tuned onto new observation and action spaces. It runs faster, fits on smaller hardware, and its diffusion head gives you smoother, more multimodal trajectories than OpenVLA's discrete bins can (the §10.5 argument, made concrete). The cost is raw generalization: a 93M model has not memorized the visual world the way a 7B backbone has, so it leans harder on your fine-tuning data to make up the difference.

The small, deployment-oriented models are SmolVLA (arXiv:2506.01844) and TinyVLA (arXiv:2409.12514). SmolVLA is a roughly 450M-parameter VLA built inside the LeRobot ecosystem (§15.3) and trained largely on community-contributed datasets, explicitly aimed at the person with one low-cost arm and a single consumer GPU. TinyVLA pushes the same direction, small backbone, fast inference, data-efficient fine-tuning. Neither will top a leaderboard against a 7B model on hard generalization, and neither is trying to. They exist so that the fine-tune-deploy loop fits on hardware a hobbyist or a small lab already has, and they close the loop fast enough that a few hundred milliseconds of latency stops being the wall it is for the big models.

## What actually separates them

Four axes decide which tier and which checkpoint you land on. None of them is "which scored highest," because the highest-scoring model on someone else's benchmark can be the wrong tool for your robot.

The first is the action representation, and it is the one with the longest downstream shadow. OpenVLA emits discrete action tokens; Octo, π0, and SmolVLA carry continuous heads (diffusion or flow-matching). If your task needs smooth, high-frequency, multimodal motion, a continuous head starts you closer to the goal and a token head fights you the whole way, because 256 bins per dimension (§11.3) put a hard resolution floor under how fine a motion the policy can even represent. If your task is slow and mostly unimodal, the token head's simplicity is a feature, and OpenVLA's discrete recipe means every part of the training stack is boring and well-trodden.

The second is fine-tuning economics. §12.2's table is the number to internalize: OpenVLA with LoRA reaches about 68% where full fine-tuning of just the last layer collapses to 30%, and LoRA does it with roughly an 8x compute reduction on a single A100. That is what makes the 7B tier reachable at all for a normal lab. Octo and the small models are cheaper still per step simply because they are smaller, so the calculus is less about clever adaptation and more about whether you can afford to touch the whole network. If your compute budget is one consumer GPU and a weekend, that budget alone may push you from the 7B tier down to SmolVLA regardless of what you would prefer on quality.

The third is the observation and action interface, meaning how much surgery it takes to make the model's inputs and outputs match your robot. OpenVLA expects a single third-person image and a language string, and emits a 7-DoF end-effector action; it has no slot for a wrist camera, a proprioceptive state vector, or a bimanual action space without you modifying the model. Octo was designed for exactly this flexibility, with input tokenizers you can add or swap and an action head you can re-target to a new action dimension, which is why it is often the better starting point for an embodiment that does not look like a single 7-DoF arm. Check this early. A model whose interface is close to your robot's saves you the most error-prone part of the whole job.

The fourth is licensing and provenance, the constraint people discover last and regret most. Weights, training code, and the datasets underneath all carry licenses, and a permissive research checkpoint is not automatically something you can ship in a product. OpenVLA and Octo released weights and code openly and are widely used in commercial-adjacent settings, but the datasets pooled into Open X-Embodiment (§15.2) carry their own terms, and a few contributing datasets are research-only. If your fine-tune is going into anything you sell, read the license on the base weights and on the pretraining corpus before you invest a month of data collection on top of them.

## A short decision procedure

Put the four axes in order and the choice usually makes itself. Run through them like a filter.

First, does the model meet your control rate on the GPU you can deploy on? If not, drop to a smaller tier; a fast, slightly-less-capable policy beats a brilliant one that misses its deadline. Second, does its action head match the motion your task needs, smooth and multimodal versus slow and simple? Third, can you afford to fine-tune it, given LoRA for the 7B tier and full fine-tuning for the small tier? Fourth, does its input/output interface sit close to your robot's cameras and action space, or will you be doing model surgery before you even start? Fifth, does the license permit what you actually intend to do with the result?

For most readers of this book, the default that survives that filter is OpenVLA fine-tuned with LoRA, and that is the setup the rest of the chapter and the hands-on exercise are built around. It is the best-documented recipe, it runs on hardware a lab owns, and its failure modes are the ones the literature has mapped. If your robot needs smoother or faster control than a 6 Hz token head can give, Octo is the reach for its diffusion head and flexible interface. If your entire budget is one cheap arm and one consumer card, SmolVLA is the model that was built for your exact situation and will not punish you for it.

There is a temptation, once you have picked, to treat the base model as sacred and pour all your effort into the fine-tune. Resist it. The base model is a starting distribution, not a finished product, and the single largest lever on your final success rate is not which of these checkpoints you chose but the quality of the data you fine-tune it on. That is where a project quietly succeeds or wastes a month, and it is what §16.2 is about: building a teleoperation dataset that earns its collection time instead of burning it.
