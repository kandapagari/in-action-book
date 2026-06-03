---
appendix: F
title: "Model zoo"
target_words: 2200
status: draft
prereqs: none — Appendix F is a reference table, not a chapter
key_refs:
  - All numerical entries are sourced from each model's primary reference; see Appendix E.2.
---

# Appendix F.  Model zoo

This appendix is a reference table of the 24 vision-language-action
and related foundation action models discussed in the book. Each row
summarizes the developer, the backbone architecture, the parameter
count, a one-line distinguishing feature, and the primary reference.
The "Chapter" column points to the chapter where each model is treated in
depth; many models appear briefly in adjacent chapters as well.

Numerical entries are taken from each model's primary reference, as
cited in Appendix E.2 and in the relevant chapter bodies. A dash (—)
indicates the primary reference does not report that figure, or
reports a range that does not fit in a single cell; consult the
chapter for the qualified version.

## F.1  The table

| Model | Developer | Backbone | Params | Distinguishing feature | Primary ref | Chapter |
|---|---|---|---|---|---|---|
| PaLM-E | Google DeepMind | PaLM + ViT-22B | up to 562B | Injects continuous sensors into LLM embedding space; embodied multimodal LLM. | arXiv:2303.03378 | Ch. 12 |
| RT-1 | Google DeepMind | Robotics Transformer (decoder-only) | — | FiLM multimodal fusion; image and action tokenization; 130k demos across 700+ tasks. | arXiv:2212.06817 | Ch. 11 |
| RT-2 | Google DeepMind | PaLI-X / PaLM-E | 5B–55B | Co-fine-tuned on web + robotics data; actions as discrete text tokens; chain-of-thought reasoning. | arXiv:2307.15818 | Ch. 12 |
| RT-2-X | 22-institution collaboration | — | — | Multi-embodiment generalization across 20+ robot hardware types. | arXiv:2310.08864 | Ch. 12, 15 |
| OpenVLA | Stanford / Berkeley / TRI / DeepMind | Llama-2 7B + SigLIP + DinoV2 | 7B | Open-source generalist VLA; LoRA touches only 1.4% of parameters; supports quantization. | arXiv:2406.09246 | Ch. 2, 12, 16 |
| Octo | Berkeley / Stanford / CMU / DeepMind | ViT-S/B transformer | 27M–93M | Block-wise attention to add or remove sensory inputs; open source; reward-free imitation. | arXiv:2405.12213 | Ch. 12, 16 |
| RDT-1B | Tsinghua | Diffusion Transformer | 1B | Large-scale diffusion model for bimanual manipulation; zero-shot generalization. | arXiv:2410.07864 | Ch. 13 |
| π0 | Physical Intelligence | PaliGemma + flow-matching action expert | 3.3B | Continuous actions at 50 Hz; ~10,000 hours of robotic trajectories; dexterous multi-stage tasks. | arXiv:2410.24164 | Ch. 13 |
| π0-FAST | Physical Intelligence | Autoregressive π0 + FAST tokenizer | — | Frequency-space (DCT) action-sequence tokenization; ~5× faster training. | arXiv:2501.09747 | Ch. 13 |
| SimVLA | Frontier Robotics | Standard VLM + lightweight action transformer | 0.5B–0.8B | Decoupled perception and control; flow-matching denoising; on par with π0.5 on LIBERO. | arXiv:2602.18224 | Ch. 13 |
| Xiaomi-Robotics-0 | Xiaomi Robotics | Qwen3-VL-4B + flow-matching DiT | 4.7B | Real-time execution; flow-matching action expert. | arXiv:2602.12684 | Ch. 13 |
| SmolVLA | Hugging Face | SmolVLM2 encoder-decoder + flow-matching transformer | 450M | Consumer-GPU and MacBook deployable; asynchronous inference decouples VLM from execution. | arXiv:2506.01844 | Ch. 13, 16 |
| TinyVLA | Midea / academic collab. | Lightweight pretrained VLM (<1.4B) + diffusion decoder | <1.4B | Diffusion-based policy decoder; distilled from larger VLAs; fast, data-efficient inference. | arXiv:2409.12514 | Ch. 16 |
| RoboMamba | Tongji University | Mamba SSM + frozen CLIP | — | Linear-scaling SSM backbone replaces transformer; 3.7M-parameter MLP policy head. | arXiv:2406.04339 | Ch. 8, 16 |
| Helix / Helix 02 | Figure AI | Dual-system (7B VLM + 80M visuomotor) | 7.08B | Full upper-body 35-DoF continuous control; on-board embedded GPUs; "Sport Mode". | figure.ai/news/helix and helix-02 | Ch. 14 |
| GR00T N1 | NVIDIA | Dual-system (System 2 VLM + System 1 reactive) | 2.2B | Humanoid-centric; 93.3% language-following; 3000h+ of human video + robot + synthetic data. | arXiv:2503.14734 | Ch. 14 |
| ρα (Rho-alpha) | Microsoft Research | Phi family + action expert | — | Tactile sensing + online learning from human corrections; BusyBox benchmark. | MSR technical report | Ch. 14, 17 |
| Embodied-R1 | Tianjin University / Huawei Noah's Ark | Qwen-2.5-VL-3B-Instruct | 3B | Reinforced fine-tuning (RFT); multi-task reward curriculum; affordance prediction. | arXiv:2508.13998 | Ch. 15 |
| RoboBrain2.0 | BAAI | Qwen2.5-VL-72B-Instruct (fine-tuned) | 7B–32B | Reinforced embodied reasoning; spatial-referring data; pointing fine-tuning. | BAAI technical report | Ch. 12 |
| LiLo-VLA | UNC / Georgia Tech / CMU | OpenVLA-OFT or π0.5 backbone | — | Modular: decouples transport (reaching) from interaction; object-centric visual masking. | arXiv:2602.21531 | Ch. 15 |
| Long-VLA | Westlake / Zhejiang / Xi'an Jiaotong | MDT + GPT-2-style transformer | — | Phase-aware input masking; long-horizon skill chaining; L-CALVIN benchmark. | arXiv:2508.19958 | Ch. 15 |
| LEO | BAAI | Transformer-based 3D model | — | Two-stage 3D vision–language alignment; embodied 3D scene understanding. | arXiv:2311.12871 | Ch. 15 |
| UniAct | Tsinghua / Shanghai AI Lab | Autoregressive transformer | — | Universal atomic actions for cross-embodiment heterogeneity. | arXiv:2501.10105 | Ch. 15, 18 |
| OpenDriveVLA | — | Autoregressive trajectory generator | — | Unified 2D / 3D perception → driving trajectories; closed-loop end-to-end control. | arXiv:2503.23463 | Ch. 15 |

## F.2  How to read the zoo

The table is laid out so that consecutive rows roughly correspond to
the order in which the models are introduced in the book. Rows 1–4
(PaLM-E through RT-2-X) are the Google DeepMind lineage covered in
Chapters 11–12. Rows 5–6 (OpenVLA, Octo) are the open-source
generalist policies treated in Chapter 12 and revisited in
Chapter 16. Rows 7–13 (RDT-1B through RoboMamba) are the continuous-
action and efficient-VLA family of Chapter 13 and §16.x. Rows 14–17
(Helix through ρα) are the dual-system models of Chapter 14. Rows
18–24 (Embodied-R1 through OpenDriveVLA) are the specialized and
adjacent models — long-horizon, 3D, cross-embodiment, driving —
covered in Chapter 15.

Three caveats on the parameter-count column. The number reported is
the total parameter count of the deployed model unless otherwise
noted; for dual-system models the column reports the sum of the slow
(VLM) and fast (sensorimotor) components and the breakdown is given
in the corresponding chapter. For models that ship in multiple sizes
(RT-2 at 5B and 55B, Octo at 27M and 93M, RoboBrain2.0 at 7B and 32B),
the column reports the range; the chapter treatment uses the size
the authors emphasize. For models that report only an active
parameter count or a sparse-expert total, the chapter body explains
the convention used; the zoo cell is the total.

## F.3  What the zoo deliberately omits

A few well-known systems are absent from the table by design.
Standalone vision-language models (Llama-3 family, Qwen-VL, PaLI-X)
are excluded — they are backbones for VLAs in this table, not VLAs
themselves. Pure robotics policies without language conditioning
(Diffusion Policy, ACT) are excluded — they appear in Chapter 10 as
*architectures* rather than as *foundation models*, and the
distinction matters: a Diffusion Policy trained on one task is not a
foundation model, but it is part of the lineage that produced the
action heads of π0 and RDT-1B. Pure planners with LLM "brains"
(SayCan, Code as Policies, Inner Monologue) are excluded — they emit
symbolic plans rather than low-level actions, and so technically
belong to the lineage of Chapter 4 rather than the zoo of Part 4.

The boundary between "foundation action model" and "large policy" is
not crisp; the inclusion criterion used here is that the model (i)
ingests both vision and language and emits actions, (ii) has been
pretrained or co-trained on a corpus that includes general web-scale
content rather than only robotics data, and (iii) is presented by
its authors as a step toward generality rather than as a single-task
demonstration. This is the same criterion the surveys cited in
Chapter 1 use, and disagreement at the boundary (Octo vs. RT-2; LEO
vs. PaLM-E) is the kind of disagreement that the introductory
chapter of each survey spends a section on.

## F.4  Cross-references

A small number of cross-cutting facts may be useful when comparing
rows.

The single largest parameter count in the table belongs to PaLM-E
(up to 562B); the single smallest is SmolVLA (450M). The ratio is
about 1,250×, which is the dynamic range the field currently
operates over.

The four models that use a flow-matching action head — π0,
SimVLA, Xiaomi-Robotics-0, SmolVLA — share the architectural lineage
laid out in Chapter 10 and revisited in Chapter 13. The two that use
a diffusion-transformer head — RDT-1B and TinyVLA — share a
different lineage; the contrast is the substance of §10.5.

The two dual-system models — Helix and GR00T N1 — share the System-
1 / System-2 architectural pattern but disagree on the latency
budget assigned to each system; Chapter 14's §14.4 has the breakdown.

The open-source weights, in order of permissiveness, are: OpenVLA
(Apache 2.0), Octo (MIT), RDT-1B (Apache 2.0), SmolVLA (Apache 2.0),
TinyVLA (research license), GR00T N1 (NVIDIA research license),
π0 (Physical Intelligence research license). RT-1, RT-2, PaLM-E, and
the Helix family are not openly weighted. This matters for
Chapter 16; the recipes there assume you have a checkpoint, and the
open-weight column of the table is the candidate list.

For the full bibliographic entries of every primary reference in
this table, see Appendix E.2. For the lineage that produced this
zoo — CLIP → RT-1 → RT-2 → OpenVLA → π0 → Helix / GR00T N1 — see
Chapters 11 through 14. For the open scientific problems that remain
across the zoo, see Chapter 18.
