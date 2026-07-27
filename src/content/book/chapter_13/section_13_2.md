---
chapter: 13
section: 13.2
title: "π0's architecture, end to end"
target_words: 2000
status: draft
prereqs: §13.1 (why fast, dexterous, long-horizon tasks break the discrete token head, and the three-part spec a continuous head has to meet), §11.1 (CLIP-style vision-language pretraining, which is the knowledge PaliGemma carries in), §12.2 (OpenVLA as the token-head 7B baseline π0 is reacting against), §10.5 (action-head choices). Flow matching itself is deferred to §13.3.
key_refs:
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Beyer, L. et al. (2024). PaliGemma, A Versatile 3B VLM for Transfer. arXiv:2407.07726.
  - Lipman, Y. et al. (2023). Flow Matching for Generative Modeling.
  - Pertsch, K. et al. (2025). FAST, Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
---

# 13.2  π0's architecture, end to end

Section 13.1 ended with a specification, not a model. A continuous action head has to emit a full action chunk in one shot, represent multimodal action distributions without collapsing to the mean, and run fast enough to close a 50 Hz loop. π0 (arXiv:2410.24164) is Physical Intelligence's answer, and the cleanest way to understand it is to watch how it splits one job the token head was doing badly into two jobs handled by two parts of the network that were built for different things.

The two parts are a pretrained vision-language backbone and a smaller action expert bolted onto it. The backbone reads pixels and language and produces a representation of what is going on in the scene; the action expert reads that representation plus the robot's proprioceptive state and produces the motion. Total parameter count is about 3.3B, which is less than half of OpenVLA's 7B, and yet it drives a two-arm robot at frequencies OpenVLA cannot touch. The parameter count is not the point. The division of labor is.

## The backbone: PaliGemma, doing exactly what it already knew how to do

π0 does not train a vision-language model from scratch. It starts from PaliGemma (arXiv:2407.07726), a 3B open VLM that pairs a SigLIP image encoder with a Gemma language model, already pretrained on the kind of image-caption and visual-question-answering data that teaches a network to connect the word "mug" to the round ceramic thing in the corner of a frame. This is the payoff of everything in Part 4. The semantic grounding that RT-2 got by co-finetuning on web data, π0 inherits for free by initializing from a model that already has it.

Feed the backbone three things at each timestep: the camera images (π0 uses multiple views, wrist and overhead), the language instruction as text tokens, and nothing else that it has to reason about semantically. It runs its transformer stack over that multimodal input the same way PaliGemma does when answering a question about a photograph. The output is a sequence of hidden states that encode, roughly, "there is a shirt here, the instruction says fold it, the left gripper is near the collar." No action has been produced yet. The backbone's only job is to understand the situation, and because it was pretrained to understand situations described in images and words, that job costs almost nothing to teach.

Here is the design decision that matters, and it is easy to skate past. The backbone is not asked to output actions at all. RT-2 and OpenVLA overloaded the VLM's own token vocabulary with action bins, which forced the language model to spend its output distribution on motor commands and dragged every action through the full weight of the language decoder, one token at a time. π0 refuses that. The VLM stays a VLM. Motion generation is handed off to a separate set of weights.

## The action expert: a second transformer sharing the same attention

The action expert is a smaller transformer, roughly 300M parameters, and the way it connects to the backbone is the part worth being precise about, because "cross-attention into a separate head" is the loose description and it is not quite what π0 does.

π0 uses what the paper calls a mixture of experts arrangement, though it is not the sparse routing you might associate with that phrase. There is one shared sequence of tokens and one shared attention operation running over it, but the tokens come in two flavors and each flavor is processed by its own set of weights. Image and language tokens go through the PaliGemma weights. Robot-state and action tokens go through the action-expert weights. Both sets of tokens sit in the same attention matrix, so an action token can attend to an image token and vice versa, but the feed-forward and projection parameters that transform each are separate. Think of it as two transformers interleaved into one attention computation rather than two transformers connected by a bridge. The action expert sees everything the backbone saw, through attention, without inheriting the backbone's mass on its own forward pass.

Why bother with the separation instead of one homogeneous stack? Two reasons, and they are the same two reasons the split exists at all. First, the action expert can be small and therefore fast, since it does not need billions of parameters to turn an already-understood scene into a trajectory; the understanding is upstream. Second, the two halves want different numerics. The backbone operates on discrete language and image tokens; the action expert operates on continuous, noisy action vectors under a flow-matching objective, which we get to in §13.3. Forcing both through identical weights would compromise both. Keeping them separate lets each be what it is.

## What actually flows through at inference

Walk one control step end to end, because the shape of the data is where the design becomes concrete.

The robot presents its observation: two or three camera images, the current joint positions and gripper state, and the standing language instruction. Images and text enter the backbone and get encoded. The proprioceptive state gets projected into a token and handed to the action expert. Now π0 wants to produce not a single action but an action chunk, a block of the next H actions, where H is around 50. At 50 Hz that is roughly one second of motion committed in a single generation. Chunking is not incidental; it is how π0 keeps the control loop fast and the motion temporally coherent, and it is the same trick ACT and Diffusion Policy used in Chapter 10 for the same reasons.

The action expert generates that chunk by flow matching. Starting from a chunk of pure Gaussian noise the same shape as the action block, it runs a small number of integration steps, ten or so, each step a forward pass through the action expert conditioned on the backbone's scene representation via the shared attention. Each step nudges the noisy chunk toward a clean trajectory. After the last step you have H real-valued action vectors, at full continuous resolution, no bins anywhere. The robot executes some prefix of the chunk, the observation updates, and the whole thing runs again. Ten small forward passes through a 300M expert is a very different latency budget than dozens of serial token decodes through a 7B decoder, which is precisely how π0 buys back the speed §13.1 said autoregression was bleeding.

Notice what did not happen. The 3.3B backbone did not run ten times. It ran once, produced the scene representation, and that representation conditioned all ten cheap expert steps. The expensive part happens once per observation; the iterative part is cheap. That asymmetry is the whole efficiency argument, and it falls straight out of putting the semantics in one place and the motion sampling in another.

## The three-part spec, checked off

Return to §13.1's specification and hold the architecture against it.

One shot, full chunk. The action expert emits all H actions of the chunk together, not one dimension at a time, so there is no per-dimension per-timestep sequence to blow up the way FAST (arXiv:2501.09747) documented for tokenizers. Multimodality preserved. Flow matching, developed next section, is a genuine generative model over the action distribution; it can place probability on a left trajectory and a right trajectory and sample one, rather than averaging them into the obstacle, which was the failure that killed naive regression. Fast enough. The heavy backbone runs once per observation and the light expert iterates a handful of times, landing control rates around 50 Hz on real two-arm hardware. All three met, and met by structural choices rather than by tuning.

## Where the knowledge lives, and why that matters for Part 5

One consequence of this architecture is worth flagging now because it shapes everything in the fine-tuning chapters. π0 keeps almost all of its transferable, task-general knowledge in the PaliGemma backbone and almost all of its embodiment-specific, motor knowledge in the action expert. That is not a tidy accident; it is close to a design goal. When you later fine-tune π0 for a new robot, the interesting question becomes how much of the backbone you can freeze, since the semantics of "fold the shirt" do not change when you swap arms, and how much of the action expert you have to retrain, since the motor commands very much do. We return to exactly this split in Chapter 16, and π0's clean separation is part of why it fine-tunes as gracefully as it does.

There is a data story stacked underneath all of this that the architecture alone does not tell. π0 was trained on roughly 10,000 hours of robot trajectories spanning several embodiments, a corpus Physical Intelligence assembled well beyond what any single academic dataset offers, and the flow-matching head is only as good as the demonstrations it regressed onto. Architecture makes the capability possible; the data makes it real. We take the dataset side apart properly in Chapter 15.

The one piece deliberately left as a black box here is the flow-matching objective itself: how you train the action expert so that ten integration steps turn noise into a clean, multimodal action chunk, and why that beats the many denoising steps a diffusion sampler would need for the same job. That objective is the engine inside the expert, and it is the whole of the next section.
