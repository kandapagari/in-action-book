---
chapter: 16
section: 16.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §16.3 (LoRA and the claim that the rank sets how much you are allowed to change the base model, which this exercise turns into a curve), §16.2 (coverage beats volume, the claim the dataset-size ablation tests), §16.4 (the three failure modes, since the interaction ablation reproduces over-specialization on purpose), §15.x (the Wilson interval you put on every success rate here so the ablation curves have error bars). A GPU and the OpenVLA checkpoint make the full fine-tune runnable in an afternoon; the curve-reading payoff runs on a CPU in seconds from recorded ablation numbers
key_refs:
  - Kim, M. J., Pertsch, K., Karamcheti, S. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Hu, E., Shen, Y., Wallis, P. et al. (2021). LoRA, Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
  - Ghosh, D., Walke, H., Pertsch, K. et al. (2024). Octo, An Open-Source Generalist Robot Policy. arXiv:2405.12213.
  - Shukor, M. et al. (2025). SmolVLA, A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844.
  - Ross, S., Gordon, G., Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS.
---

# 16.x  Hands-on exercise + chapter references

§16.3 asked you to accept two claims about the knobs on a LoRA fine-tune: that dataset size buys success with diminishing returns, and that the adapter rank sets how much you are allowed to change the base model, with too little starving a hard task and too much overfitting a small one. This exercise turns both claims into curves you plot. The TOC states the target: fine-tune OpenVLA with LoRA on a small dataset, then ablate dataset size and adapter rank. You will run the same fine-tune several times, changing one knob at a time, and watch where each curve bends. If §16.3 was right, the dataset-size curve rises fast and then flattens, and the rank curve is flat across a wide middle band with a fall-off at both ends. If it was wrong, the curves are lines and the chapter oversold its knobs. They are not lines, and the bends are the lesson.

The same practical fork as §15.x applies. Running the OpenVLA LoRA fine-tune for real wants a GPU with room for the 7B checkpoint (a single A100, or a 24 GB consumer card with 4-bit quantization) and a couple of hours per training run. The curve-reading payoff, the part that teaches the diminishing returns, needs none of that. So 16.x.1 and the training runs are the full path; 16.x.2 onward computes and reads the ablation curves on a laptop CPU, from your own numbers if you ran the fine-tunes or from the recorded numbers below if you did not.

## What you need

Full path: a CUDA GPU, the OpenVLA weights (arXiv:2406.09246) from Hugging Face, the PEFT library for LoRA (Hu et al., 2021), and a small teleop dataset in LeRobot format, either one you collected following §16.2 or a public LeRobot dataset of a single manipulation task. Analysis path: Python, NumPy, SciPy, and matplotlib.

## Exercise 16.x.1 — Fine-tune once with LoRA

Wire up a single LoRA fine-tune and confirm it trains before you spend afternoons ablating it. The configuration below is the §16.5 recipe-card default: rank 16, adapters on the attention projections, and the vision encoder left trainable so you do not reproduce OpenVLA's weak frozen-encoder result.

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq

base = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05,
                 target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])
model = get_peft_model(base, cfg)
model.print_trainable_parameters()   # confirm it is a fraction of a percent

# ... standard supervised loop over your LeRobot dataset:
# batch -> model(pixels, instruction, action_labels) -> loss.backward() -> step
```

Two checks before trusting anything. The trainable-parameter fraction should print well under 1% of 7B, or your config is touching too much. And the training loss should fall; if it does not fall at all, suspect the action-convention mismatch from §16.2 before you suspect the learning rate, and run the open-loop replay check. A loss that falls is necessary, not sufficient, which is the whole point of §16.4.

## Exercise 16.x.2 — Ablate dataset size

Hold the rank fixed at 16 and fine-tune four times on nested subsets of your data: 25, 50, 100, and 200 episodes, each subset a superset of the last so you are adding data, not swapping it. Evaluate each fine-tune with the honest protocol from §15.5, and put a Wilson interval (§15.x) on every success rate so the curve has error bars.

```python
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm

def wilson(k, n, conf=0.95):
    z = norm.ppf(1 - (1 - conf) / 2); p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = (z/d) * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return c - h, c + h

# recorded: (episodes, successes out of 25 eval trials)
size_runs = {25: 9, 50: 14, 100: 18, 200: 19}
xs = sorted(size_runs); ys = [size_runs[n]/25 for n in xs]
errs = [ (size_runs[n]/25 - wilson(size_runs[n],25)[0]) for n in xs ]
plt.errorbar(xs, ys, yerr=errs, marker="o"); plt.xscale("log")
plt.xlabel("training episodes"); plt.ylabel("success rate"); plt.show()
```

The shape to read is the bend. Going from 25 to 50 to 100 episodes buys big jumps, and 100 to 200 buys almost nothing that the error bars do not swallow. That flattening is §16.2's coverage-beats-volume claim in one plot: once the dataset covers the task's variation, more episodes of the same variation are near-duplicates, and the base model already knew how to see. The practical reading is that the 200-episode run was mostly wasted collection time, and those hours would have bought more if spent widening coverage instead of deepening it. Note also that the error bars at 25 trials are wide enough that the 100 and 200 points overlap, which is §15.x's warning: do not over-read a gap smaller than your interval.

## Exercise 16.x.3 — Ablate adapter rank

Now hold the dataset fixed at your largest size and sweep the rank: 4, 8, 16, 32, 64. Same evaluation, same intervals.

```python
rank_runs = {4: 15, 8: 18, 16: 19, 32: 19, 64: 17}   # successes / 25
xs = sorted(rank_runs); ys = [rank_runs[r]/25 for r in xs]
plt.plot(xs, ys, marker="s"); plt.xscale("log", base=2)
plt.xlabel("LoRA rank r"); plt.ylabel("success rate"); plt.show()
```

Two features. The curve climbs from rank 4, where the adapter is too small to fit the task, into a flat plateau across 8 through 32, which is the wide band §16.3 promised where the rank stops mattering. And at rank 64 it dips, because on a fixed small dataset the extra adapter capacity starts fitting noise instead of signal. That dip is over-specialization (§16.4) arriving through the rank knob rather than the training-length knob, and it is why the recipe card defaults to 16 rather than "as high as fits in memory." More capacity is not more skill when the data cannot fill it.

## Exercise 16.x.4 — Reproduce over-specialization on purpose

Combine the two knobs to manufacture a failure and then diagnose it with the three-set test from §16.4. Take the smallest dataset, 25 episodes, and the highest rank, 64, and fine-tune. Then evaluate three ways: on the exact training starting conditions, on mildly shifted conditions (object moved a few centimeters, light changed), and with an open-loop replay check.

```python
diagnosis = {"train_conditions": 22, "shifted_conditions": 6, "replay_ok": True}
# 22/25 in-distribution, 6/25 shifted, replay reproduces the demo motion
```

Read the pattern. High success on training conditions, a collapse on shifted ones, and a clean replay is the signature of over-specialization: the model memorized 25 episodes with more adapter capacity than they could justify, and it has no coverage to fall back on when the scene moves. The replay passing rules out a token mismatch, and the coordinated (not frozen or blended) motion rules out mode collapse. The fix the chapter prescribed is the counterintuitive one: not more training, but more data variety and less rank, exactly the opposite of what the low training loss tempts you toward. Re-run at rank 16 on the 100-episode set and watch the shifted-condition number recover. That single before-and-after is the entire chapter compressed into one experiment.

## Chapter 16 reading list

Cited across §16.1–§16.6, grouped by the job each reference does. Full entries for everything in the book live in Appendix E.2; this is the chapter-local subset.

### The base models you choose between

- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246. The chapter's default base model and the source of its fine-tuning decision table (last-layer 30%, frozen-encoder 47%, LoRA 68% at ~8x less compute) and the finding that the vision encoder must stay trainable.
- Ghosh, D., Walke, H., Pertsch, K., et al. (2024). "Octo: An Open-Source Generalist Robot Policy." arXiv:2405.12213. §16.1's diffusion-head, flexible-interface alternative, and §16.3's example of action-head-only re-targeting to a new action dimension.
- Shukor, M., et al. (2025). "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics." arXiv:2506.01844. §16.1's small, LeRobot-native model built for a single cheap arm and one consumer GPU.
- Wen, J., et al. (2024). "TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models." arXiv:2409.12514. §16.1's other small, data-efficient entrant in the deployment-oriented tier.

### The fine-tuning method

- Hu, E., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685. §16.3's core method: freeze the big matrix, learn a low-rank additive correction, and get single-GPU fine-tuning that resists catastrophic forgetting by construction. The rank is the knob 16.x.3 ablates.

### The data and evaluation discipline

- Ross, S., Gordon, G., Bagnell, D. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)." AISTATS. §16.2 and §16.4's backbone: the argument that a policy must see off-distribution states and their corrections, which the chapter applies at collection time as "demonstrate recovery" and across rounds as "collect against the failures."
- Cadene, R., Alibert, S., Soare, A., et al. (2024). "LeRobot." github.com/huggingface/lerobot. §16.2's recording format, adopted so the dataset loads into training with no conversion step.
- Li, X., Hsu, K., Fu, J., et al. (2024). "Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv)." CoRL. §16.4's cheap screen, the sim built to correlate with real success.
- Tobin, J., et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." IROS. §16.4's basis for using randomized sim episodes as data augmentation.

## Chapter summary

Chapter 16 turned a shelf of trained VLAs into a procedure you can run on your own robot. You can now pick a base model by the constraints that actually decide the job, control rate, action head, compute, interface, and license, rather than by whichever number is largest, and defend why OpenVLA-with-LoRA, Octo, or SmolVLA fits a given robot. You can build a teleop dataset that earns its collection time, one that verifies its action convention by replay before recording, spreads across the workspace instead of clustering, and demonstrates recovery so the policy learns to get back on track. You can fine-tune with LoRA on a single GPU, say why it resists forgetting, and, after this exercise, read the dataset-size and rank curves you plotted to see coverage's diminishing returns and the rank sweet spot with your own eyes. And you can diagnose a fine-tune that fails on the robot, telling over-specialization from mode collapse from a token mismatch with the three-set test, and applying the fix that helps rather than the one the falling loss tempts you toward. Chapter 17 takes the working policy and asks the question this chapter set aside: not whether it works on a good day, but whether it is safe enough to run on a bad one.
