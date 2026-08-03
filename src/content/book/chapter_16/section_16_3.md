---
chapter: 16
section: 16.3
title: "LoRA vs. full fine-tuning vs. action-head-only"
target_words: 2000
status: draft
prereqs: §16.2 (you have a format-correct dataset in hand). §12.2 (OpenVLA's fine-tuning decision table and the finding that the vision encoder must adapt, plus the crucial detail that LoRA matched sandwich fine-tuning at ~68% for ~8x less compute), §3.1 (matrix rank and why a low-rank update is a small number of parameters), §10.5 (what an action head is, since "action-head-only" means updating just that part).
key_refs:
  - Hu, E. et al. (2021). LoRA, Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Ghosh, D., Walke, H., Pertsch, K. et al. (2024). Octo, An Open-Source Generalist Robot Policy. arXiv:2405.12213.
---

# 16.3  LoRA vs. full fine-tuning vs. action-head-only

You have a 7B model and a dataset of 150 episodes. Now the question is what, exactly, gets updated when you train, because a 7B model has more parameters than your 150 episodes have supervision to move responsibly, and the three standard answers, update everything, update a small low-rank slice, or update only the action head, trade the same three things against each other: how much compute you spend, how much of the base model's competence you keep, and how far you can push the policy toward a task that looks nothing like its pretraining.

## Full fine-tuning: the most capable and the most dangerous

Full fine-tuning updates every weight in the network, backbone included. It is the most expressive option, because nothing is held fixed, and on a large, diverse fine-tuning set it gives the best ceiling. It is also the option most likely to hurt you on a small one, for two reasons that both come down to having too much capacity chasing too little data.

The first is cost. Updating all 7B parameters of OpenVLA means storing gradients and optimizer state for all of them, which pushes memory well past a single consumer GPU and into multi-GPU or high-memory-datacenter-card territory. §12.2 put the LoRA-versus-full gap at roughly 8x compute, and that multiplier is the difference between a weekend on one card and a week on a cluster. For most readers that alone settles it.

The second is catastrophic forgetting, the failure mode where the model, in learning your task, overwrites the general competence that made it worth starting from. Train all 7B parameters hard on 150 episodes of one mug on one table and the model can forget how to generalize across objects it used to handle, because gradient descent has no reason to preserve capabilities your loss function never rewards. You end up with a model that aces your exact setup and has quietly become worse than the checkpoint you started from at anything else. On a large and varied fine-tuning corpus this is not a concern; on a small specialized one it is the default outcome unless you guard against it.

## LoRA: the recommended default, and why it works

LoRA, low-rank adaptation (Hu et al., 2021, arXiv:2106.09685), is the method that makes fine-tuning a 7B VLA sane on a single GPU, and it is OpenVLA's own recommendation for good reason. The idea rests on one empirical observation: the update a fine-tune applies to a big weight matrix is usually low-rank, meaning it can be well approximated by the product of two much smaller matrices. So instead of updating the weight matrix W directly, you freeze it and learn a small additive correction, W + BA, where B and A are skinny matrices whose inner dimension r (the rank) is tiny, often 8, 16, or 32. During training only B and A get gradients; W never moves.

The consequences are worth spelling out because they explain every practical advantage. The number of trainable parameters drops by orders of magnitude, since two rank-16 matrices are a rounding error next to a full weight matrix, which is what collapses the memory and compute cost. The frozen W keeps the base model's competence intact by construction, because you literally cannot overwrite it, which is why LoRA resists the catastrophic forgetting that stalks full fine-tuning. And the learned adapter is small enough to store and swap, so you can keep one base model and a folder of task-specific adapters, loading whichever one you need, rather than a full 7B copy per task.

```python
# LoRA applied to OpenVLA's backbone, using the PEFT library
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                       # rank: the one knob that matters most
    lora_alpha=16,              # scaling; commonly set equal to r
    target_modules=["q_proj", "v_proj"],   # which projections get adapters
    lora_dropout=0.05,
)
model = get_peft_model(base_openvla, config)
model.print_trainable_parameters()   # e.g. 0.1% of 7B is trainable
```

The rank r is the knob to understand, and the chapter exercise ablates it directly. A higher rank gives the adapter more capacity to fit your task, at more memory and more risk of overfitting a small dataset; a lower rank is cheaper and more conservative. The useful mental model is that r sets how much you are allowed to change the base model, and the right value scales with how far your task is from pretraining and how much data you have. A task close to the pretraining distribution with little data wants a small rank; a task far from it with a lot of data can use a larger one. §12.2's headline holds across a wide band of ranks, though: LoRA at a sensible rank matched full-quality sandwich fine-tuning for a fraction of the cost, which is why it is the default this book recommends and the exercise walks you through.

One detail from §12.2 that interacts with LoRA and is easy to get wrong: OpenVLA found the vision encoder had to adapt during fine-tuning, not stay frozen, because internet-pretrained visual features lack the fine spatial geometry a manipulation policy needs. So when you configure LoRA, make sure the adapters actually reach the visual pathway or that the encoder is otherwise trainable, rather than adapting only the language backbone and leaving the eyes locked. A LoRA config that targets only the LLM's attention projections and skips the encoder reproduces the weak "frozen vision" result from the decision table, near 47%, and you will wonder why your well-collected data is underperforming.

## Action-head-only: the cheapest, for the narrowest case

The third option updates only the action head and freezes the entire backbone. For OpenVLA this means touching just the repurposed output vocabulary slots; for a model like Octo (arXiv:2405.12213) with a separate diffusion head, it means training that head while the transformer trunk stays fixed. This is the cheapest thing you can do, trains in minutes, and needs almost no memory.

It is also the most limited, and §12.2 quantified how limited: OpenVLA's last-layer-only fine-tune landed around 30%, the worst option in the table. The reason is intuitive. If you only touch the head, the representation feeding it, everything the backbone computes about the scene, is frozen at its pretraining state, so the head can only re-slice features the model already extracts. That is enough when your task is genuinely just a re-mapping of the base model's existing competence onto a slightly different action space, and it is nowhere near enough when your robot's cameras, objects, or dynamics differ from pretraining in ways the frozen backbone never learned to represent.

Where action-head-only earns its place is a specific, common situation: adapting a model to a new action space or action dimension without changing what it perceives. Octo was designed with exactly this in mind, letting you attach a new action head for a robot with a different number of degrees of freedom and train just that head, precisely because the flexible-interface design of §16.1 assumed you would. If your robot sees roughly what the base model was trained to see but acts in a different space, head-only adaptation is a fast, honest first try. If it sees something new, you will need LoRA or more.

## Choosing, in one paragraph

Default to LoRA. It is the setup that fits a single GPU, resists forgetting, produces swappable adapters, and matched full-quality fine-tuning in OpenVLA's own experiments, which is why the rest of this chapter uses it. Reach for full fine-tuning only when you have a large, varied dataset (thousands of episodes across many conditions), the compute to run it, and evidence that LoRA's capacity is actually the bottleneck rather than your data. Reach for action-head-only when you are re-targeting the action space of a model whose perception already matches your robot, and you want an answer in minutes. And whichever you pick, remember the failure the next section is about, because these methods do not fail loudly: a fine-tune can drive its training loss to the floor and still produce a policy that is worse than the base model on everything except a memorized demo. §16.4 puts that inside the sim-to-real loop, where the gap between a low training loss and a working robot is widest.
