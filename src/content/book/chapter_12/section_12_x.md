---
chapter: 12
section: 12.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §12.1–§12.6; the OpenVLA setup from Chapter 2 (weights pulled, LIBERO installed, a working inference loop); a GPU with at least 24 GB of memory for the fine-tune, or patience with a rented one; a firm grasp of LoRA versus full fine-tuning from §12.2, the discrete-token action head OpenVLA uses, and Octo's diffusion head from §12.3 as the contrast baseline; about half a day, most of it waiting on the fine-tune to converge
key_refs:
  - Kim, M. J. et al. (2024). OpenVLA — An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Octo Model Team, Ghosh, D., Walke, H. et al. (2024). Octo — An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - Brohan, A. et al. (2023). RT-2 — Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Open X-Embodiment Collaboration, Padalkar, A. et al. (2023). Open X-Embodiment — Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Hu, E. et al. (2021). LoRA — Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
---

# 12.x  Hands-on exercise + chapter references

Chapter 11 gave you a fifty-line tokenizer and no GPU bill. This chapter's
exercise is the opposite, and that is the point. Every claim in §12.2 about
OpenVLA being the checkpoint people reach for first rests on one property you
have not yet tested for yourself: that a 7B model pretrained on somebody else's
970,000 trajectories can be bent to a task it never saw, using a few dozen of
your own demonstrations, in an afternoon rather than a month. The exercise the
TOC names for Chapter 12 is exactly that test. Fine-tune OpenVLA on a single new
task with about fifty demonstrations, then put it next to three baselines that
had to learn the task from those same fifty demonstrations with no web
pretraining behind them. The gap between the fine-tuned VLA and the baselines is
the whole argument of Part 4, rendered as a success-rate number you produced.

Budget half a day, and know where the time goes. Almost none of it is your
thinking; most of it is the fine-tune converging and the evaluation rollouts
running one slow episode at a time. Set the job going, read the π0 abstract for
Chapter 13 while you wait, and come back to numbers.

## What you need before you start

You already did the hard setup in Chapter 2: the OpenVLA weights are on disk,
LIBERO runs, and the inference loop from §2.3 turns an image and an instruction
into a 7-DoF action. This exercise reuses all of it. Pick one LIBERO task suite
and hold out a single task as your target, say "put the black bowl in the
drawer." Collect or subsample roughly fifty demonstration trajectories for that
one task. LIBERO ships demonstrations, so you can pull fifty from the suite
rather than teleoperating them yourself, which keeps the exercise about the
learning question and not about your joystick skills.

One honest caveat up front. Full fine-tuning of a 7B model wants more memory
than a single consumer card has, which is why §12.2 spent a paragraph on LoRA
and 4-bit quantization. Use them here. A LoRA adapter over a 4-bit base fits on
one 24 GB GPU and is the configuration the OpenVLA authors themselves report for
adaptation, so you are not cutting a corner; you are doing what the paper did.

## Exercise 12.x.1 — Fine-tune OpenVLA on the held-out task

This is the headline drill. Load the pretrained checkpoint, attach a LoRA
adapter, and train on your fifty demonstrations until the action-token accuracy
on a small validation split stops climbing. The recipe is deliberately close to
the one in the OpenVLA repository, because reproducing a known-good recipe is
the right way to learn what its knobs do before you start turning them on your
own robot in Chapter 16.

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch

proc  = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    load_in_4bit=True, trust_remote_code=True)

lora = LoraConfig(r=32, lora_alpha=16, target_modules="all-linear")
model = get_peft_model(model, lora)          # ~110M trainable of ~7.5B total
model.print_trainable_parameters()
```

The single number to watch during training is the fraction of trainable
parameters that line prints: on the order of one to two percent of the model.
Everything else is frozen, which is the mechanism behind the whole result. You
are not teaching OpenVLA what a bowl is or how a drawer opens; the web
pretraining and the Open X-Embodiment pretraining already paid for that. You are
teaching it to map perception it already understands onto the specific action
distribution of this one task. Watch the validation token accuracy climb past
roughly ninety percent within a few hundred steps and stall. When it stalls,
stop; more steps past that point start memorizing your fifty trajectories rather
than generalizing across them, and the rollout success rate will tell on you if
you overtrain.

Record two things: wall-clock time to convergence, and peak GPU memory. You will
want both when Chapter 16 asks you to estimate what adapting a VLA to a real
robot actually costs.

## Exercise 12.x.2 — Build the three baselines

The fine-tuned VLA means nothing without something to beat. Stand up three
comparisons, each of which strips away one thing the VLA had.

The first is zero-shot OpenVLA: the same checkpoint, no fine-tuning, evaluated
directly on your held-out task. This isolates what the fifty demonstrations
bought, separate from what pretraining bought. Expect it to do something
non-trivial and still fail the task most of the time, because the target task's
exact action distribution was not in its pretraining mix in the form your LIBERO
setup presents it.

The second is behavior cloning from scratch: a small ResNet-plus-MLP policy,
maybe fifteen million parameters, trained only on your fifty demonstrations with
no pretraining behind it. This is §6.2's plain BC, and it is the honest control
for the question "did we need a foundation model at all, or would any network
trained on fifty demos do?" Keep the architecture unremarkable; the point is the
absence of transferred knowledge, not a clever design.

The third is Diffusion Policy, the §10.2 action head trained from scratch on the
same fifty demonstrations. This one matters because it is the strongest
non-foundation baseline and because it shares an action-head philosophy with
Octo (§12.3). If Diffusion Policy from scratch matches your fine-tuned OpenVLA on
this narrow task, that is a real and interesting finding about how much a single
task actually needs pretraining, and you should report it rather than bury it.

```python
# evaluation harness shared by all four policies
def rollout(policy, env, task, n_episodes=50, max_steps=400):
    wins = 0
    for _ in range(n_episodes):
        obs, done, t = env.reset(task=task), False, 0
        while not done and t < max_steps:
            action = policy.act(obs["image"], task.language)
            obs, _, done, info = env.step(action)
            t += 1
        wins += int(info.get("success", False))
    return wins / n_episodes
```

Run the same harness on all four policies so the success rates are comparable.
Fifty episodes per policy is the floor; real-robot evaluation variance (§15.4,
when you get there) is large enough that fewer than that tells you almost
nothing.

## Exercise 12.x.3 — Read the numbers honestly

Now put the four success rates in a table and read them against what §12.2 and
§12.5 led you to expect. The pattern you are most likely to see, and the one the
OpenVLA paper reports across its own tasks, is fine-tuned OpenVLA on top, the two
from-scratch baselines well below it, and zero-shot OpenVLA somewhere in between
or at the bottom depending on how far your task sits from its pretraining
distribution.

The interesting cases are the ones that break that pattern. If Diffusion Policy
from scratch beats your fine-tuned VLA, ask whether your task is narrow and
repetitive enough that pretrained breadth is dead weight; a foundation model
earns its keep on variety, and a single fixed task is the case where it has the
least to offer. If zero-shot OpenVLA beats your fine-tune, you almost certainly
overtrained the adapter, so revisit where you stopped in Exercise 12.x.1. Write
one paragraph explaining your table. The paragraph is the deliverable, not the
numbers; anyone can produce a table, and the skill Chapter 12 is trying to build
is the reading of it.

## Exercise 12.x.4 — Predict a result before you read it

The last learning objective for this chapter is the one that pays off for the
rest of the book: read a paper from this era and predict its main result before
the experiments section. Practice it now, on a paper you have not studied. Octo
(arXiv:2405.12213) is a good choice, because §12.3 told you its design without
walking you through its numbers.

Cover the results. From the architecture alone, write down three predictions.
Will Octo, with its diffusion head, produce smoother trajectories than
token-based OpenVLA on a task that rewards smoothness? Its modular design lets
you attach a new camera without retraining the trunk, so does the paper claim
cheap adaptation to a new observation setup, and roughly how cheap? Octo is
smaller than OpenVLA by an order of magnitude, so where do you expect it to lose,
and by how much? Commit the guesses to paper, then uncover the results and score
yourself. Getting a prediction wrong is more useful here than getting it right,
because a wrong prediction tells you exactly which part of your mental model of
these architectures is miscalibrated. That calibration is what carries you into
Chapter 13, where the model names arrive faster than any exercise can keep up
with, and reading them by their design rather than their leaderboard row becomes
the only way to keep your footing.

## Chapter 12 reading list

The works below are cited across §12.1–§12.6, grouped by the role they play.
Full bibliographic entries for everything cited in the book live in Appendix E.2;
this is the chapter-local subset.

### The recipe scaled up

- Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web
  Knowledge to Robotic Control." arXiv:2307.15818. §12.1's proof that a
  web-pretrained VLM transfers to control when actions ride in its token stream;
  the closed-weights result the rest of the chapter reproduces in the open.
- Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action
  Model." arXiv:2406.09246. The center of the chapter and of this exercise: the
  open 7B checkpoint of §12.2, the LoRA and 4-bit adaptation recipe you ran, and
  the discrete-token head you compared against a diffusion baseline.
- Octo Model Team, Ghosh, D., Walke, H., et al. (2024). "Octo: An Open-Source
  Generalist Robot Policy." arXiv:2405.12213. §12.3's diffusion-head, modular
  alternative to OpenVLA, and the paper Exercise 12.x.4 asks you to predict
  before reading.

### The data underneath all of it

- Open X-Embodiment Collaboration, Padalkar, A., et al. (2023). "Open
  X-Embodiment: Robotic Learning Datasets and RT-X Models." arXiv:2310.08864.
  §12.4's shared corpus: the 970,000 and 800,000 trajectories that OpenVLA and
  Octo pretrained on, and the RT-X evidence that one policy across many robots
  beats a policy per robot.

### The tools and the caution

- Hu, E., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
  arXiv:2106.09685. The low-rank adapter that made Exercise 12.x.1 fit on one
  GPU; the reason fine-tuning a 7B VLA is an afternoon and not a cluster job.
- Wei, J., et al. (2022). "Emergent Abilities of Large Language Models."
  *Transactions on Machine Learning Research*. §12.5's source for the honest
  version of the emergence claim.
- Schaeffer, R., Miranda, B., Koyejo, S. (2023). "Are Emergent Abilities of Large
  Language Models a Mirage?" *NeurIPS 2023*. §12.5's counterweight: the argument
  that emergence is often an artifact of a discontinuous metric, the skepticism
  you should carry into every "emergent" claim in Part 4.

## Chapter summary

Chapter 12 took the RT-1 recipe of Chapter 11 and turned the scale up, and this
exercise made the payoff of doing so a number on your own screen. You can now
explain how RT-2 reuses an off-the-shelf vision-language model as a policy
backbone, folding actions into the same token stream the model already uses for
words, and why that inheritance is the answer to the data scarcity that limited
RT-1. You can open the OpenVLA checkpoint and name every part of it as frozen,
fine-tuned, or added, which is exactly what you did when you attached a LoRA
adapter and watched two percent of the parameters carry a new task. You can
describe how Octo's diffusion action head differs from a token classifier and
what its modularity buys, and you tested that contrast directly by putting a
from-scratch Diffusion Policy in your baseline table. Above all you can read a
robot-learning paper from this era and predict its headline result from its
design before the experiments section confirms or embarrasses your guess, a
habit that turns the flood of model names in the coming chapters from noise into
signal. Chapter 13 spends that habit immediately, on π0, where the discrete
action tokens you just fine-tuned give way to a flow-matching head built for the
smooth, dexterous control that binning could never quite reach.
