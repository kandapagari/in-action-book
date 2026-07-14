---
chapter: 1
section: 1.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §1.1–§1.6
key_refs:
  - Fikes & Nilsson (1971). STRIPS. Artificial Intelligence 2(3–4).
  - Pomerleau (1988). ALVINN. NeurIPS 1988.
  - LaValle (2006). Planning Algorithms. Cambridge University Press.
  - Argall et al. (2009). A Survey of Robot Learning from Demonstration. RAS 57(5).
  - Kober, Bagnell, Peters (2013). RL in Robotics: A Survey. IJRR 32(11).
  - Mnih et al. (2015). Human-level control through deep RL. Nature 518.
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2. arXiv:2307.15818.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - O'Neill et al. (2023). Open X-Embodiment. arXiv:2310.08864.
  - Black et al. (2024). π0. arXiv:2410.24164.
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications, Challenges. arXiv:2505.04769.
---

# 1.x  Hands-on exercise + chapter references

The Chapter 1 exercise needs no GPU, no code, and no robot. It's a reading-and-classification exercise, designed to take about ninety minutes, meant to make the four-family vocabulary from §1.4 and the three-slot anatomy from §1.2 actually stick. Resist the urge to skip it. Students who do it report afterward that it changed how they read the rest of the book. Students who skip it tend to do it anyway, implicitly and more slowly, somewhere around Chapter 11.

## Exercise 1.x.1 — The four-paper triage

Pick four papers, one from each family, read each in abstract-plus-figures-only mode (twenty minutes per paper, no full read), and fill in a table. Recommended choices for a first pass:

- *Classical/analytical:* LaValle (2006), Chapter 5 of *Planning
  Algorithms*. RRT and its variants. (A textbook chapter rather than a paper
  because the classical family does not have a single "canonical paper" the
  way the others do.)
- *Reinforcement-learning:* Mnih et al. (2015), "Human-level control through
  deep reinforcement learning." *Nature* 518.
- *Imitation:* Pomerleau (1988), "ALVINN: An Autonomous Land Vehicle in a
  Neural Network." NeurIPS 1988. Or, if you want a modern instance, Chi et al.
  (2023), "Diffusion Policy," arXiv:2303.04137 (not in the local source
  mirror but freely available on arXiv).
- *Foundation/VLA:* Kim et al. (2024), OpenVLA, arXiv:2406.09246.

For each paper, fill in seven cells:

1. *Family* (one of the four).
2. *Era* (one of the six from §1.3).
3. *Slot 1 — input.* What does the model see? Be specific: image resolution,
   number of cameras, presence of proprioception, language conditioning.
4. *Slot 2 — output.* What does the model emit? Symbolic plan, joint
   targets, end-effector deltas, action tokens, continuous chunks? At what
   rate?
5. *Slot 3 — training signal.* Derivation, reward, demonstration, or
   pretraining-plus-fine-tune? On what data?
6. *Compounding-error response.* What does the paper do (if anything) to
   handle the closed-loop / open-loop mismatch from §1.1?
7. *One sentence on why this paper mattered.* No hedging.

A worked example, for ALVINN:

| Cell | Value |
|------|-------|
| Family | Imitation |
| Era | 3 (first learned approaches, 1988 prototype) |
| Slot 1 | 30×32 grayscale camera, single front-facing |
| Slot 2 | Steering angle (a single scalar), at ~10 Hz |
| Slot 3 | Supervised regression on human-driving demonstrations |
| Compounding-error response | None explicitly; trajectory divergence was
the dominant failure mode |
| Why it mattered | First demonstration that a neural network could output a
control signal directly from pixels, on real hardware |

Doing this for four papers takes ninety minutes the first time and goes faster on the second pass. The table is itself the artifact, so don't write prose around it.

## Exercise 1.x.2 — The dishwasher problem, written down

Re-do the dishwasher worked example from §1.4 in writing, for a robot and kitchen you've actually seen. Choose any household task, emptying a dishwasher, sorting recycling, putting laundry in a basket, and write four short paragraphs (under 150 words each), one per family, describing how you'd build a robot to do it. For each, name the data or model the family would need (a clean kitchen geometry, a reward function, a teleoperation rig and some number of demonstrations, a base VLA checkpoint), the most likely failure mode (a missing obstacle, reward hacking, compounding-error trajectory divergence, a confabulated action), and one concrete reason you would or wouldn't pick this family for the task.

The exercise is the writing itself, not the answer. There's no scoring rubric. The value comes from noticing where you hesitate, and the places you hesitate are exactly the places later chapters will fill in.

## Exercise 1.x.3 — Read one survey, end to end

Open Sapkota et al. (arXiv:2505.04769) and read it in one sitting. It's long but readable, and the local source mirror has it cached. Mark, with a pencil or a highlighter, three places where it refers to a model or method that doesn't yet make sense to you. Those three places will get answered by specific chapters of this book, usually one of Chapters 11 through 14. At the end of each of those chapters, come back to the survey and re-read the marked passages. The before-and-after delta is the single best self-assessment for whether the book is working.

If you can't find three places to mark, meaning every sentence of the Sapkota survey already makes sense, you've read the prerequisites this book assumes and can probably skip Chapter 3 (the math refresher) and parts of Chapter 8 (transformers for control), heading straight to Part 4 instead. Most readers will mark more than three.

## Exercise 1.x.4 — A short language exercise

Pick five terms from the chapter and write a one-sentence definition for each without looking back: compounding error, the three-slot anatomy, action tokenization, the dividing line (between classical and learned components in a stack), and cross-embodiment generalization.

Then look back and check. The point isn't the score, it's noticing which of the five you can't define cleanly. Each one is a recurring concept, and the ones you can't yet define are the ones to keep a finger on while reading the next chapter.

## Chapter 1 reading list

The works below are the ones cited or referenced across §1.1 through §1.6, grouped by purpose. Full bibliographic entries for everything cited in the whole book live in Appendix E.2; this list is just the chapter-local subset.

### Foundational references

- Fikes, R. E., & Nilsson, N. J. (1971). "STRIPS: A new approach to the
  application of theorem proving to problem solving." *Artificial
  Intelligence* 2(3–4), 189–208.
- Pomerleau, D. A. (1988). "ALVINN: An Autonomous Land Vehicle in a Neural
  Network." *NeurIPS 1988*.
- LaValle, S. M. (2006). *Planning Algorithms.* Cambridge University Press.
  Freely available at planning.cs.uiuc.edu.
- Argall, B. D., Chernova, S., Veloso, M., & Browning, B. (2009). "A Survey
  of Robot Learning from Demonstration." *Robotics and Autonomous Systems*
  57(5), 469–483.
- Kober, J., Bagnell, J. A., & Peters, J. (2013). "Reinforcement Learning
  in Robotics: A Survey." *International Journal of Robotics Research*
  32(11), 1238–1274.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.

### Deep RL and policy learning

- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control
  through deep reinforcement learning." *Nature* 518, 529–533.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2016). "Continuous
  control with deep reinforcement learning." (DDPG.) *ICLR 2016*.
- Schulman, J., Wolski, F., Dhariwal, P., et al. (2017). "Proximal Policy
  Optimization Algorithms." arXiv:1707.06347.
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). "Soft
  Actor-Critic." *ICML 2018*.

### Imitation and behavior cloning

- Ross, S., Gordon, G., & Bagnell, D. (2011). "A Reduction of Imitation
  Learning and Structured Prediction to No-Regret Online Learning."
  (DAgger.) *AISTATS 2011*.
- Jang, E., Irpan, A., Khansari, M., et al. (2022). "BC-Z: Zero-Shot Task
  Generalization with Robotic Imitation Learning." *CoRL 2022*.
- Shafiullah, N. M., Cui, Z., Altanzaya, A., & Pinto, L. (2022). "Behavior
  Transformers: Cloning k modes with one stone." arXiv:2206.11251.
- Chi, C., Feng, S., Du, Y., et al. (2023). "Diffusion Policy: Visuomotor
  Policy Learning via Action Diffusion." arXiv:2303.04137.
- Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). "Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware." (ACT.)
  arXiv:2304.13705.

### Foundation / VLA models

- Brohan, A., Brown, N., Carbajal, J., et al. (2022). "RT-1: Robotics
  Transformer for Real-World Control at Scale." arXiv:2212.06817.
- Driess, D., Xia, F., Sajjadi, M., et al. (2023). "PaLM-E: An Embodied
  Multimodal Language Model." arXiv:2303.03378.
- Brohan, A., Brown, N., Carbajal, J., et al. (2023). "RT-2: Vision-
  Language-Action Models Transfer Web Knowledge to Robotic Control."
  arXiv:2307.15818.
- O'Neill, A., Rehman, A., Maddukuri, A., et al. (2023). "Open X-Embodiment:
  Robotic Learning Datasets and RT-X Models." arXiv:2310.08864.
- Octo Model Team (2024). "Octo: An Open-Source Generalist Robot Policy."
  arXiv:2405.12213.
- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). "OpenVLA: An
  Open-Source Vision-Language-Action Model." arXiv:2406.09246.
- Black, K., Brown, N., Driess, D., et al. (2024). "π0: A Vision-Language-
  Action Flow Model for General Robot Control." arXiv:2410.24164.
- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for
  Vision-Language-Action Models." arXiv:2501.09747.
- NVIDIA (2025). "GR00T N1: An Open Foundation Model for Generalist
  Humanoid Robots." arXiv:2503.14734.

### Surveys and field-state references

- Sapkota, R., et al. (2025). "Vision-Language-Action Models: Concepts,
  Progress, Applications and Challenges." arXiv:2505.04769.
- Zhang, et al. (2025). "A Survey on Pure Vision-Language-Action Models."
  arXiv:2509.19012.
- "Embodied Arena: A Benchmark for Foundation Models in Embodied AI."
  arXiv:2509.15273.
- "Efficient Vision-Language-Action Models: A Survey." arXiv:2510.24795.
- Stanford HAI (2025). *AI Index Report — Robotics chapter.*

### Background textbooks worth keeping nearby

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT
  Press.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction.*
  MIT Press.
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics: Mechanics, Planning,
  and Control.* Cambridge University Press.
- Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2010). *Robotics:
  Modelling, Planning and Control.* Springer.

## Chapter summary

Chapter 1 set out a vocabulary for thinking about action models: the three-slot anatomy of inputs, outputs, and training signal, the six-era history that moved from STRIPS to π0, and the four families, classical, RL, imitation, foundation/VLA, that organize the rest of the book. With that vocabulary in hand, you can read a modern VLA abstract and pull out its design choices, place a published system in the right family and era, sketch four candidate solutions to a new robot task, and predict an action model's failure mode from the family it belongs to. Chapter 2 is where all of this first earns its keep: a complete, end-to-end fine-tune of an OpenVLA checkpoint on a small benchmark, on a single GPU, in roughly seven pages.
