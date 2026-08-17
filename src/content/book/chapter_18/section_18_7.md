---
chapter: 18
section: 18.7
title: "Does embodied reasoning actually reason?"
target_words: 2000
status: draft
prereqs: §18.4 (reasoning + action, which raised faithfulness as a worry and left it open; this section reports the measurements), §15.4–§15.5 (why static benchmarks mislead, the machinery of the argument here), §12.5 (what "emergent" really means, and the shortcut-versus-capability question), §14.3 (Gemini's "Embodied Thinking" as the industrial bet on legible reasoning), §17.5 (analyzability, which depends on the reasoning being real).
key_refs:
  - Xu, H., Zheng, S., Lu, Z. et al. (2026). Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models (BeTTER). arXiv:2604.18000.
  - Foutter, M., Cercola, M., Wild, L., Pavone, M. et al. (2026). Do Vision-Language-Action Models Mean What They Say? On the Role of Faithfulness in Embodied Reasoning (Pinocchio). arXiv:2607.04681.
  - ERVLA (2026). Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation. arXiv:2606.03784.
  - Zawalski, M. et al. (2024). Robotic Control via Embodied Chain-of-Thought Reasoning (ECoT).
---

# 18.7  Does embodied reasoning actually reason?

§18.4 ended on a suspicion. If a policy writes out a plan and then acts, we naturally read the plan as the cause of the action, and that reading might be wrong. A model can produce fluent reasoning that has little to do with the computation that actually produced the motion, in which case legible reasoning buys the appearance of transparency and not the substance. I called the question open and under-measured, and recommended the obvious experiment: perturb the reasoning, see whether the action changes.

Two papers have since run experiments in that spirit, from different directions, and both come back with results that should make you more skeptical than §18.4 left you. This section reports them. It is the least comfortable section in the chapter, because the target is not a technique the field is trying and failing to scale, it is a technique the field is already shipping.

## Stress-testing the reasoning, not the success rate

The first result attacks the measurement problem head-on. BeTTER (arXiv:2604.18000, Xu et al., April 2026, from Peking University, Tsinghua, and BeingBeyond) is a diagnostic benchmark built on a premise §15.5 will recognize: high success rates on standard benchmarks may be measuring the wrong thing, so build an evaluation designed to separate reasoning failure from execution failure and see what survives.

The design choice that makes their results interpretable is what they call kinematic isolation. If a policy fails a task, you normally cannot tell whether it misunderstood the instruction or simply could not execute the motion, and that ambiguity has protected the reasoning story for years. BeTTER holds the low-level difficulty fixed and applies targeted causal interventions to the high-level structure instead: shift the spatial layout, extrapolate the task in time, recompose known primitives into an order the model has not seen. A model that reasons should degrade gently. A model that has memorized a mapping should fall off a cliff.

State-of-the-art VLAs fall off the cliff. The paper's language is "catastrophically fail in dynamic scenarios," and its four findings are worth listing separately because they describe four different illusions.

The first is apparent instruction understanding without grounding. The policy behaves as though it parsed the instruction, and the behavior is explained by a shortcut from words to motions, what the authors call a lexical-kinematic shortcut, rather than by any link between the language and the objects actually present. The second is failure of subgoal composition: a model that can do A and can do B cannot reliably do A-then-B in a new order, which is exactly the compositional generalization §12.5 warned you to be careful about calling emergent. The third is failure of causal state tracking, and it shows up as behavioral inertia, the policy continuing along a trajectory the world has already invalidated, because it is not maintaining a representation of what has changed. The fourth is failure of fine-grained semantic grounding, a collapse in the ability to distinguish objects that are lexically or visually close.

The mechanistic half of the paper is what lifts it above a complaint. The authors trace these symptoms to architectural bottlenecks in the VLM-to-VLA conversion, capacity compression and myopic downsampling, that degrade the semantic representation the base model arrived with. The claim is not that the reasoning was never there. It is that adapting a vision-language model into an action model damages the very representations the reasoning depended on, and then static evaluation lets the damaged model overfit to sensorimotor priors and score well anyway. If that is right, embodied reasoning is not a capability we are slowly adding to VLAs. It is partly a capability the VLA recipe destroys, and the benchmarks conceal the destruction.

## Measuring whether the words bind the actions

The second result asks §18.4's question directly and gives it a vocabulary the field needed. "Do Vision-Language-Action Models Mean What They Say?" (arXiv:2607.04681, Foutter et al., July 2026, Stanford with NVIDIA Research and collaborators) separates two things that were being conflated: *functional* reasoning, where the reasoning improves task performance, and *faithful* reasoning, where the reasoning reflects the policy's actual decision process.

That distinction does a lot of work, and it is the concept to carry out of this section. Everything §18.4 reported as a success, ECoT's generalization gains, ERVLA's benchmark numbers, is evidence of functional reasoning. None of it is evidence of faithful reasoning. The two can come apart, and the paper's argument is that current alignment strategies admit reasoning whose intermediate steps mask the causal links to the action, reasoning that is ungrounded in the environment or internally disconnected, and that this ungroundedness limits generalization rather than being a merely cosmetic defect.

Their empirical entry point is a human evaluation of a state-of-the-art driving reasoning model, and the finding is an inconsistent coupling between reasoning quality and trajectory improvement. Better-looking reasoning did not reliably produce better driving. Take that seriously for a moment: the assumption underlying every legible-reasoning safety argument, including the one §17.5 leaned on and the one Gemini's "Embodied Thinking" (§14.3) represents as a product decision, is that reasoning quality and behavioral quality move together. On at least one serious system, measured by humans, they do not.

The constructive half is the more hopeful part. They operationalize faithfulness behaviorally, scoring observation grounding and stepwise coherence with a learned critic they name Pinocchio, and then use the critic as a dense reward for reinforcement-learning post-training. Faithfulness improves by 4% over the alignment baseline and 18% over trajectory-error post-training, task performance stays competitive, and, most interesting, on a synthetic out-of-distribution set the faithfulness-trained policy is about 1.6× more responsive to rare counterfactual scenarios. That last number is the one to hold onto, because it converts faithfulness from an interpretability nicety into a capability claim: a policy whose reasoning is actually bound to its observations reacts better to situations it has never seen.

## What this does to the chapter's other arguments

Three consequences, and none of them are comfortable.

The safety argument weakens. §17.5 accepted that we cannot verify a VLA, and treated legible reasoning as a partial substitute, something a monitor could read. §17.2's monitor patterns lean the same way. If reasoning can be unfaithful, a monitor reading the chain of thought is reassured by a plan the policy may not follow, and an unfaithful-but-plausible trace is worse than no trace, because it defeats the human review it invites. The mitigation is not to abandon legibility but to stop treating it as evidence unless faithfulness has been measured, and the Pinocchio-style critic is the first tool for measuring it.

§18.4's clever fix looks different in this light. ERVLA's move, train with reasoning traces and drop them at inference, was presented there as elegantly sidestepping latency. It also sidesteps this section's problem, by declining to make a claim the model cannot support: if the reasoning never appears at runtime, nobody is tempted to read it as an explanation. But that cuts the other way too. A policy that reasons only in training gives up the runtime deliberation that motivated the idea, and it gives up the inspectable intermediate as well. The design space here has a shape §18.4 did not draw: reasoning can be functional-and-hidden (ERVLA), functional-and-visible-but-possibly-unfaithful (most embodied CoT today), or functional-visible-and-verified-faithful, which is where Pinocchio points and where almost nothing yet sits.

And the benchmark critique of §15.4 lands harder than it did there. BeTTER's finding that static protocols mask representational degradation is the same argument §15.5 made about variance and §18.2 made about long-horizon measurement, arriving now with a mechanism: it is not merely that our benchmarks are easy, it is that their easiness is load-bearing for the reported capability. LIBERO-Plus and BeTTER exist because a benchmark that never intervenes cannot tell overfitting from understanding.

## Where a reader can push

This is the most accessible open problem in the chapter, and it needs no fleet. The perturbation experiment §18.4 asked for is a weekend of work on a single arm: take a policy that emits reasoning, corrupt the reasoning in a controlled way, swap a referent, break the step order, describe a scene that is not there, and measure how much the action distribution moves. If the actions barely change, the reasoning is decoration on that policy for that task, and you have a publishable negative result. Recomposition tests are similarly cheap: train on A-then-B, evaluate on B-then-A, and watch for the composition failure BeTTER reports.

The honest summary is that embodied reasoning is functionally useful and epistemically unproven. It helps, on the tasks §18.4 described, for reasons that may have nothing to do with the model reasoning in any sense a person would recognize, and the two 2026 results that looked closely both found less than the framing promised. That is not a reason to abandon the direction; the faithfulness-as-reward result suggests the gap is at least partly fixable, and fixing it buys out-of-distribution robustness rather than just interpretability. It is a reason to distrust any system, including a commercial one, whose safety case rests on reasoning it has not shown to be faithful. §18.5's advice on where to contribute applies with unusual force here, because the measurement tools are new, the negative results are cheap to produce, and the field is currently shipping the assumption these papers are undermining.
