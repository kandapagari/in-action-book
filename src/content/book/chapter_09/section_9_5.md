---
chapter: 9
section: 9.5
title: "World models vs. VLAs: the architecture debate"
target_words: 2000
status: draft
prereqs: §9.1 (a world model predicts whatever the downstream use needs), §9.3 (planning by rolling out a learned model; drift over long horizons), §9.4 (the pixel-vs-representation split; Genie and V-JEPA as poles), §8.5 (bridge to foundation action models), §6.1 (imitation as the dominant signal), Part 4 (VLAs, previewed).
key_refs:
  - LeCun (2022). A Path Towards Autonomous Machine Intelligence. Position paper, OpenReview.
  - Hafner et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). arXiv preprint.
  - Bruce et al. (2024). Genie: Generative Interactive Environments. ICML.
  - RT-2 (Brohan et al. 2023), arXiv:2307.15818.
  - OpenVLA (Kim et al. 2024), arXiv:2406.09246.
  - π0 (Black et al. 2024), arXiv:2410.24164.
---

# 9.5  World models vs. VLAs: the architecture debate

The last four sections built up a machine that learns how the world changes and then acts by consulting that machine. Part 4 of this book is about a machine that skips the middle step: it maps observations and a language instruction straight to actions, and never builds an explicit model of dynamics at all. Both machines work. Both have produced systems that control real robots. And their proponents disagree, sometimes heatedly, about which one is a detour and which one is the road. This section lays out the disagreement plainly, because you will spend the rest of the book inside the VLA camp and it is worth knowing what the other camp thinks you are getting wrong.

The dispute is not really about neural network layers. It is about where understanding of physics is supposed to live. One side says a competent agent must first learn a model of how the world behaves, and that control is a comparatively easy search on top of that model. The other side says that understanding is a means, not an end, and that if you have enough data pairing observations with correct actions, a single network will absorb whatever physics it needs implicitly, on the way to predicting the action. Everything else, pixels versus features, planning versus reflex, video pretraining versus teleoperation, is downstream of that split.

## The world-model bet

State the strong version, the one Yann LeCun's position paper (LeCun 2022) makes without hedging. A system that only ever learns stimulus-to-response mappings is brittle by construction. It has no way to answer a question it was not trained on, because answering novel questions requires simulating consequences, "if I push here, what happens?", and simulation requires a model. Intelligence, on this view, *is* a predictive world model plus a cheap planner that queries it. The action policy is almost an afterthought; the hard-won asset is the model, and once you have a good one, steering it toward any goal is a search problem you already know how to solve (§9.3).

Three arguments prop this bet up, and each has real weight.

The first is data. Correct-action data is the scarcest thing in robotics, every trajectory costs a human on a teleoperation rig (§6.1). Video of the world is nearly free and effectively infinite. A world model can drink from the free source: Genie learned controllable dynamics from 200,000 hours of unlabeled gameplay (§9.4), and V-JEPA learned transferable video features with no action labels at all. If most of what a robot needs to know about physics can be learned from passive observation, then the architecture that can *use* passive observation has an enormous head start over one that can only learn from action-labeled demonstrations.

The second is generalization. A model of dynamics is task-agnostic by nature. Gravity, contact, occlusion, and momentum do not change when you switch from stacking cups to folding towels. If those regularities are captured once, in a reusable model, then a new task is a new goal handed to the same planner, no retraining of the physics. A stimulus-response policy, the argument goes, has to relearn the relevant slice of physics inside every new behavior it acquires, which is why such policies are famously data-hungry and famously bad at anything outside their training distribution.

The third is verifiability. A planner that rolls out an explicit model can, at least in principle, be inspected. You can ask what future it predicted, check whether that future was physically plausible, and reject a plan whose imagined rollout is nonsense. A policy that emits an action from an opaque forward pass offers no such handle, a point that returns with force in Chapter 17, where we try to make deployed systems safe. Model-based systems wear their reasoning on the outside.

## The VLA counter-bet

Now the other side, which is where the field's momentum and most of its recent results actually sit. The VLA bet is deflationary: an explicit world model is a nice idea that keeps losing to the crude approach of predicting actions directly, and it loses for reasons that are not accidental.

Start with the observation that dooms pixel-space planning in practice. §9.3 and §9.4 already showed the mechanism, model rollouts drift, and a next-frame objective spends most of its capacity rendering detail no decision depends on. Push that further. To act by planning, you must roll your world model forward tens or hundreds of times per decision, at control rate, and every rollout inherits the model's errors compounded over the horizon. The VLA camp's response is: why pay that tax at all? A policy trained by imitation does the equivalent of the whole plan-and-select loop in one forward pass, with no rollout to drift, and it learns to do so from exactly the data, expert trajectories, that already encode good behavior.

Then there is the awkward empirical fact. When you scale a vision-language model that was pretrained on internet images and text and then fine-tune it to output actions, you get a system that generalizes to new objects and new phrasings of a task strikingly well, RT-2 (arXiv:2307.15818) was the demonstration that a VLM's semantic knowledge transfers to control without any explicit dynamics model in sight. OpenVLA (arXiv:2406.09246) and π0 (arXiv:2410.24164) doubled down on the recipe and did not add a world model; they added data, better action heads, and bigger backbones. The generalization the world-model camp promised as the payoff of understanding-first turned out to be available, at least in part, straight from language-and-vision pretraining poured through an action objective. Physics, on this reading, gets learned implicitly and well enough, as a side effect of predicting what an expert did.

The counter-bet, sharpened to one sentence: understanding the world is instrumental, and if the instrument you actually need is "output the right action," then training directly for that objective is more efficient than training for the harder, wasteful proxy of "predict the entire future." The world-model camp is solving a superset of the problem, and paying for the extra generality with data, compute, and drift it does not need.

## Where the honest uncertainty is

It would be tidy to declare a winner. The tidy version is wrong, and pretending otherwise would misinform you about a live research question.

The VLA camp is winning on today's benchmarks, but today's benchmarks mostly measure short-horizon manipulation from large demonstration sets, precisely the regime where reflexive imitation shines and where long-horizon reasoning, the world model's home turf, barely gets tested. The tasks that would actually separate the two bets, long, novel, multi-step problems that demand planning through states no demonstration covered, are the ones the field is least able to evaluate, a gap Chapter 15 examines directly. Absence of a world-model advantage on cup-stacking is not evidence it will stay absent on assembly.

The two approaches are also converging in ways that blur the fight. DreamerV3 (Hafner et al. 2023) is a world model that trains its policy inside imagined rollouts, a world model with a reflex policy bolted on. The dual-system VLAs of Chapter 14 pair a slow deliberative module with a fast reactive one, which is structurally close to "consult a model, then act." And a growing line of work uses video-prediction models (§9.4) purely as pretraining for the visual backbone of an otherwise ordinary VLA, the world model as ingredient rather than as the whole architecture. The cleanest statement of the two bets is a useful map, but the territory is filling in with hybrids that take something from each.

The deepest open question is the one neither camp can currently settle: does predicting actions from enough data really teach a network the physics it needs, or does it teach a shortcut that happens to work on the training distribution and fails silently off it? If the former, the world model is redundant and the VLA camp is simply right. If the latter, VLAs are accumulating a debt that comes due exactly when we ask them to do something truly new, and the understanding-first camp will have been early rather than wrong. Nobody has the experiment that decides this cleanly, which is why the debate is a debate and not a settled fact.

## What to carry forward

For the rest of this book you will follow the VLA road, and you should now understand that this is a choice with a live alternative, not a foregone conclusion. The reasons the VLA path dominates the coming chapters are honest ones: it uses the data we can actually get, it sidesteps the drift that cripples model rollouts, and it inherits generalization from language-vision pretraining for free. The reasons to keep an eye on the world-model camp are equally honest: passive video is a data source VLAs struggle to exploit, planning offers a verifiability that opaque policies lack, and the tasks most likely to expose the difference are the ones we have barely started to test.

Hold both. When Part 4 shows you RT-2, OpenVLA, and π0 mapping pixels and language to actions with no explicit model of the world, remember that a serious research tradition thinks this works despite the missing model, not because it is unnecessary, and that the two traditions are already trading parts. We turn next to a summary of the chapter, and then leave world models behind for the action-generation machinery, diffusion and flow, that modern VLAs actually use.
