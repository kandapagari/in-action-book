---
chapter: 12
section: 12.1
title: "RT-2: a VLM that also outputs actions"
target_words: 2000
status: draft
prereqs: §11.1 (CLIP-style vision-language pretraining and the alignment a policy inherits), §11.3 (action tokenization and its resolution ceiling), §11.4 (what RT-1's conditioning stack got right and why the field routed around it). Helpful, §8.4 on what gets tokenized, since RT-2's whole trick is to put actions in the same token stream as words.
key_refs:
  - Brohan, A. et al. (2023). RT-2, Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Brohan, A. et al. (2022). RT-1, Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Driess, D. et al. (2023). PaLM-E, An Embodied Multimodal Language Model. arXiv:2303.03378.
  - Padalkar, A. et al. (2023). Open X-Embodiment, Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
---

# 12.1  RT-2: a VLM that also outputs actions

Chapter 11 ended on an uncomfortable finding: RT-1's competence came mostly from 130,000 real demonstrations, not from anything clever in the network, and a 13-robot fleet teleoperating in kitchens for 17 months is not a lever most labs can pull. RT-2 (arXiv:2307.15818) responds to that finding in the only way that scales. If robot data is scarce and web data is not, stop trying to learn perception and reasoning from teleop, and inherit them from a model that already read the internet. The idea is one sentence long. Take a vision-language model that can already answer questions about images, and teach it to emit robot actions as if they were just more words. That's it. The rest of this section is why that sentence works, what it buys you that RT-1 could never reach, and what it costs at 3 in the morning when the robot has to decide where to move in the next 300 milliseconds.

This is the first section of a chapter about scale, so keep the shape of the argument in view. RT-2 is the proof that a web-pretrained backbone transfers to control. OpenVLA (§12.2) makes that proof open and reproducible, and Octo (§12.3) shows a different action head bolted to the same idea. All three sit on the dataset that §12.4 is about. RT-2 is where the recipe first cooks.

## The one trick: actions are strings

Recall from §11.3 how RT-1 handled actions. A 7-dimensional command, three numbers for end-effector translation, three for rotation, one for the gripper, gets each dimension quantized into 256 bins, and the policy predicts the bin with a cross-entropy loss. RT-2 keeps the discretization but changes what the bins are. Instead of a fresh action head with its own 256-way output, it writes each action as a short string of integers and feeds that string through the VLM's existing text vocabulary.

Concretely, an action becomes something like `1 128 91 241 5 101 127`: the first integer is the discretized gripper command, the next six are the position and rotation bins, all in the range 0 to 255. The model already has tokens for those numbers, or is made to, by overwriting the 256 least-frequently-used entries in its tokenizer with dedicated action-bin tokens (this is the PaLI-X variant; the PaLM-E variant reuses number tokens the model already owns). The upshot is that "predict the next action" and "predict the next word" are now the same operation running through the same softmax. No separate regression head, no auxiliary loss, no architectural surgery on the backbone. A robot episode is just a training example where the target output happens to be seven numbers instead of a sentence.

Two backbones carried the paper. One is PaLI-X, a vision-language model in the tens of billions of parameters trained on web image-text and visual question answering. The other is PaLM-E (arXiv:2303.03378), the embodied language model from §11.1 that already interleaves images and text in one token stream. Both were built to caption pictures and answer questions about them. Neither had ever touched a robot. RT-2's contribution is showing that the same weights, lightly retrained, can drive one.

## Co-fine-tuning, or how not to lobotomize your VLM

Here's the failure mode that the naive version of this idea walks straight into. Take your beautiful web-pretrained VLM, fine-tune it only on robot trajectories, and within an epoch or two it forgets almost everything it knew. The demonstrations contain no photos of the Eiffel Tower, no trivia about mammals, no arithmetic, so gradient descent happily overwrites all of that with kitchen-table object-picking. You'd end up with an expensive RT-1 that has thrown away the exact web knowledge you paid for. Catastrophic forgetting is the standard name for this, and it's the whole reason a smaller model trained from scratch on robot data was the sensible 2022 choice.

RT-2's fix is co-fine-tuning: during robot training, keep feeding the original web data too, mixing visual-question-answering batches in alongside the robot-trajectory batches. The model never stops practicing "what is in this image" while it learns "what action comes next," so the web competence stays resident. The paper is direct that this matters, reporting that co-fine-tuning beats fine-tuning on robot data alone by a wide margin on the generalization tests, and that the ratio drifts toward more robot data over the course of training. The web knowledge isn't a nice-to-have you preserve out of thrift. It's the ingredient that produces everything interesting in the next subsection, and forgetting it collapses RT-2 back into a worse RT-1.

At deployment there's one more wrinkle worth naming, because it trips people up. A VLM asked to produce free-form text could, in principle, emit "I think the robot should" instead of `1 128 91`. RT-2 constrains decoding so that when the model is acting, only valid action tokens are allowed out. The sampling is restricted to the action vocabulary and the format is fixed at seven numbers, which turns the general-purpose language head into a well-behaved policy without changing any weights.

## What web knowledge buys: the emergent skills

Now the payoff, and it's the part that made RT-2 famous. Because the backbone still knows what it learned from the web, RT-2 can act on concepts that appear in zero robot demonstrations. The paper groups these into a few kinds of generalization that RT-1 simply could not do.

Symbol and number understanding: told to "move the banana to the sum of two plus one," RT-2 moves it to the object labeled 3, chaining arithmetic it learned from text with a grounding of digits it learned from images, none of it taught by teleop. Novel object categories: asked to "pick up the extinct animal" from a lineup that includes a toy dinosaur alongside ordinary objects, it picks the dinosaur, because "dinosaur" and "extinct" are linked in its web knowledge and the demonstrations never had to enumerate that. It can pick out a specific person's face, identify a flag, or move an object toward a logo, all riding on associations that live in the VLM and were never in the action data.

The mechanism is the one §11.1 set up. The demonstrations teach the mapping from aligned, language-aware features to motor commands, a small thing, and the enormous prior on what the world contains and how words refer to it comes free from the backbone. RT-1's generalization stopped at recombinations of skills and phrasings it had practiced; RT-2's extends to semantic categories it has never physically manipulated. That is a different kind of generalization, and it's the first time a robot policy exhibited it.

RT-2 also inherits a faint version of chain-of-thought. Prompted to reason in text before acting, some variants can produce an intermediate plan, "to pick up the object that could be used as a hammer, I should grab the rock," and then emit the action, using the language model's reasoning to bridge an abstract instruction to a concrete grasp. It's brittle and slow, and it foreshadows the explicit high-level reasoning that Chapter 14's dual-system architectures make a first-class citizen rather than a prompting trick. Don't oversell it here; RT-2's reasoning is a hint of what's coming, not a solved capability.

## What it costs

The price of putting your policy inside a tens-of-billions-parameter VLM is that your policy now runs like a tens-of-billions-parameter VLM. RT-2's largest model is 55B parameters, far too big to run on the robot, so the paper serves it from a cloud TPU and queries it over the network, landing control frequencies in the low single digits to about 10 Hz depending on model size. For the slow tabletop pick-and-place tasks in the paper that's tolerable. For anything needing fast, reactive, contact-rich control it is not, and this latency wall is a running theme through the rest of Part 4: §13's flow-matching heads and §14's dual-system split are both, in large part, answers to "the big VLM is too slow to close a loop at."

The second cost is openness, or the lack of it. PaLI-X and PaLM-E are proprietary, the training data mix is Google's, and nobody outside could download the checkpoint and poke at it. For a field trying to build on a result, an unreproducible result is only half a gift. That gap is precisely the hole OpenVLA (arXiv:2406.09246) was built to fill, which is why §12.2 comes next.

And a quieter limitation, easy to miss under the flashy demos: RT-2's new skills are almost all about *what* to manipulate, not *how*. The web taught it that a dinosaur is extinct and that 2 plus 1 is 3; it did not teach it a new grasp, a new force profile, or a motion it never saw demonstrated. The emergent generalization lives in the semantic, perceptual half of the problem. The motor half is still bounded by the demonstration distribution, same as RT-1, a point the Open X-Embodiment work (arXiv:2310.08864) leans on when it argues that pooling *action* data across robots is the way to widen the motor half too.

## Why this section anchors the chapter

RT-2 settled an argument. Before it, "use a big pretrained model as the policy backbone" was a plausible bet; after it, the bet had numbers on real hardware and a set of skills no from-scratch policy had shown. Every model in this chapter and the next assumes that settled result and asks a follow-up question. Can we make it open? Can we make the action head continuous instead of discrete? Can we make it fast enough to deploy? RT-2 doesn't answer any of those well, but it's the reason they became the right questions to ask.

The most reproducible of those follow-ups is the first one, and it's where we go next: OpenVLA takes RT-2's recipe, swaps the closed backbone for open weights, and publishes everything, giving us a checkpoint we can actually take apart component by component.
