---
appendix: E
title: "Canonical references and the offline source mirror"
target_words: 2400
status: draft
prereqs: none — this appendix is consulted, not read straight through
key_refs:
  - This appendix consolidates the reference lists at the end of every chapter.
---

# Appendix E.  Canonical references and the offline source mirror

Each chapter cites its primary references in the chapter body. This
appendix consolidates them into a single reading list organized by
chapter and documents the offline source mirror that ships with the
project's working directory so citations can be verified without an
internet connection.

## E.1  How to use the offline source mirror

All canonical arXiv PDFs cited in the book are mirrored under
`sources/pdfs/` in the project's working directory, with filenames of
the form `arxiv_<id>_<short-title>.pdf` — for example,
`arxiv_2406.09246_OpenVLA.pdf`. Companion blog posts, vendor
technical reports, and product pages that lack arXiv equivalents are
saved as HTML snapshots under `sources/html/` with descriptive
filenames matching their domain — for example,
`huggingface_blog_pi0_and_pi0_fast.html` or `figure_ai_news_helix.html`.
The machine-readable index `sources/SOURCES_INDEX.txt` lists every
URL → local-file mapping, plus the curl-based `download_sources.sh`
script used to (re)generate the mirror.

Four citation-hygiene rules apply throughout the book.

Prefer the arXiv ID (for example, arXiv:2406.09246) over a hosted URL.
arXiv IDs are stable; URLs are not.

When a model is best described by a vendor blog or technical report
rather than a peer-reviewed paper — Helix, Helix 02, RoboBrain2.0,
ρα (Rho-alpha) — the canonical URL is given and the snapshot date is
recorded in `SOURCES_INDEX.txt`.

Survey citations are kept up to date in each chapter's reference list
and in §E.2 of this appendix. A reader who wants a single point of
entry to the literature starts with the surveys listed under
Chapter 1.

Numerical claims — parameter counts, training-hours figures, success
rates — are always cited to the model's primary reference. The Model
Zoo in Appendix F reuses those numbers verbatim and points back to
the primary-reference column rather than re-citing the originals.

## E.2  Reading list, by chapter

### Chapter 1. The robot learning problem

- Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement
  Learning in Robotics: A Survey. *International Journal of Robotics
  Research*, 32(11), 1238–1274.
- Sapkota, R. et al. (2025). Vision-Language-Action Models: Concepts,
  Progress, Applications and Challenges. arXiv:2505.04769.
- Zhang et al. (2025). Pure Vision Language Action (VLA) Models — A
  Comprehensive Survey. arXiv:2509.19012.
- Ni et al. (2025). Embodied Arena — A Comprehensive, Unified, and
  Evolving Evaluation Platform. arXiv:2509.15273.
- Xu et al. (2025). An Anatomy of Vision-Language-Action Models — From
  Modules to Milestones and Challenges. arXiv:2512.11362.

### Chapter 2. Your first VLA, end-to-end

- Kim, M. J. et al. (2024). OpenVLA — An Open-Source Vision-Language-
  Action Model. arXiv:2406.09246.
- Liu, B. et al. (2023). LIBERO — Benchmarking Knowledge Transfer for
  Lifelong Robot Learning. *NeurIPS 2023 Datasets and Benchmarks
  track*.
- Garcia et al. (2025). LIBERO-Para — A Diagnostic Benchmark and
  Metrics for Paraphrase Robustness in VLA Models. arXiv:2603.28301.

### Chapter 3. Math and ML prerequisites in 30 minutes

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep
  Learning*. MIT Press.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
  Springer.
- Paszke, A. et al. (2019). PyTorch — An Imperative Style, High-
  Performance Deep Learning Library. *NeurIPS 2019*.

### Chapter 4. Classical action models: planning and inverse dynamics

- Fikes, R. E., & Nilsson, N. J. (1971). STRIPS — A New Approach to
  the Application of Theorem Proving to Problem Solving. *Artificial
  Intelligence*, 2(3–4), 189–208.
- McDermott, D. et al. (1998). PDDL — The Planning Domain Definition
  Language. *Yale Center for Computational Vision and Control TR-98-003*.
- Helmert, M. (2006). The Fast Downward Planning System. *Journal of
  Artificial Intelligence Research*, 26, 191–246.
- Garrett, C. R., Lozano-Pérez, T., & Kaelbling, L. P. (2020).
  PDDLStream — Integrating Symbolic Planners and Blackbox Samplers via
  Optimistic Adaptive Planning. *ICAPS 2020*.
- Ahn, M. et al. (2022). Do As I Can, Not As I Say — Grounding
  Language in Robotic Affordances (SayCan). arXiv:2204.01691.
- LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University
  Press.
- Featherstone, R. (2008). *Rigid Body Dynamics Algorithms*. Springer.
- Khatib, O. (1987). A Unified Approach for Motion and Force Control
  of Robot Manipulators — The Operational Space Formulation. *IEEE
  Journal of Robotics and Automation*, 3(1).
- Hogan, N. (1985). Impedance Control — An Approach to Manipulation.
  *ASME Journal of Dynamic Systems, Measurement, and Control*.

### Chapter 5. Learning from rewards: MDPs and reinforcement learning

- Bellman, R. (1957). *Dynamic Programming*. Princeton University
  Press.
- Howard, R. A. (1960). *Dynamic Programming and Markov Processes*.
  MIT Press.
- Puterman, M. L. (1994). *Markov Decision Processes — Discrete
  Stochastic Dynamic Programming*. Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning — An
  Introduction* (2nd ed.). MIT Press.
- Kober, J., Bagnell, J. A., & Peters, J. (2013). Reinforcement
  Learning in Robotics — A Survey. *International Journal of Robotics
  Research*, 32(11).
- Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998).
  Planning and Acting in Partially Observable Stochastic Domains.
  *Artificial Intelligence*, 101(1–2).

### Chapter 6. Learning from demonstrations: behavior cloning and imitation learning

- Pomerleau, D. A. (1988). ALVINN — An Autonomous Land Vehicle in a
  Neural Network. *NeurIPS 1988*.
- Ross, S., Gordon, G. J., & Bagnell, J. A. (2011). A Reduction of
  Imitation Learning and Structured Prediction to No-Regret Online
  Learning (DAgger). *AISTATS 2011*.
- Ng, A. Y., & Russell, S. J. (2000). Algorithms for Inverse
  Reinforcement Learning. *ICML 2000*.
- Argall, B. D., Chernova, S., Veloso, M., & Browning, B. (2009). A
  Survey of Robot Learning from Demonstration. *Robotics and
  Autonomous Systems*, 57(5).

### Chapter 7. Deep RL for control: DQN to SAC and PPO

- Mnih, V. et al. (2015). Human-level Control through Deep
  Reinforcement Learning (DQN). *Nature*, 518(7540).
- Lillicrap, T. P. et al. (2016). Continuous Control with Deep
  Reinforcement Learning (DDPG). *ICLR 2016*.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms
  (PPO). arXiv:1707.06347.
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function
  Approximation Error in Actor-Critic Methods (TD3). *ICML 2018*.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. *ICML 2018*.
- Tobin, J. et al. (2017). Domain Randomization for Transferring Deep
  Neural Networks from Simulation to the Real World. *IROS 2017*.

### Chapter 8. Sequence models meet control

- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Chen, L. et al. (2021). Decision Transformer — Reinforcement
  Learning via Sequence Modeling. *NeurIPS 2021*.
- Janner, M., Li, Q., & Levine, S. (2021). Offline Reinforcement
  Learning as One Big Sequence Modeling Problem (Trajectory
  Transformer). *NeurIPS 2021*.
- Fu, J. et al. (2020). D4RL — Datasets for Deep Data-Driven
  Reinforcement Learning. arXiv:2004.07219.
- Liu et al. (2024). RoboMamba — Multimodal State Space Model for
  Efficient Robot Reasoning and Manipulation. arXiv:2406.04339.
- Wu et al. (2025). A Survey on Efficient Vision-Language-Action
  Models. arXiv:2510.24795.

### Chapter 9. World models and model-based learning

- Ha, D., & Schmidhuber, J. (2018). World Models. *NeurIPS 2018*.
- Hafner, D. et al. (2019). Learning Latent Dynamics for Planning from
  Pixels (PlaNet). *ICML 2019*.
- Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to
  Control — Learning Behaviors by Latent Imagination (Dreamer). *ICLR
  2020*.
- Hafner, D. et al. (2023). Mastering Diverse Domains through World
  Models (DreamerV3). arXiv:2301.04104.
- Bruce, J. et al. (2024). Genie — Generative Interactive Environments.
  *ICML 2024*.
- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
  OpenReview position paper.

### Chapter 10. Diffusion and flow models for action generation

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion
  Probabilistic Models. *NeurIPS 2020*.
- Song, Y. et al. (2021). Score-Based Generative Modeling through
  Stochastic Differential Equations. *ICLR 2021*.
- Chi, C. et al. (2023). Diffusion Policy — Visuomotor Policy Learning
  via Action Diffusion. *RSS 2023*.
- Zhao, T. Z., Kumar, V., Levine, S., & Finn, C. (2023). Learning
  Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).
  *RSS 2023*.
- Lipman, Y. et al. (2023). Flow Matching for Generative Modeling.
  *ICLR 2023*.
- Liu, X., Gong, C., & Liu, Q. (2023). Flow Straight and Fast —
  Learning to Generate and Transfer Data with Rectified Flow. *ICLR
  2023*.

### Chapter 11. The VLA recipe: from CLIP to RT-1

- Radford, A. et al. (2021). Learning Transferable Visual Models from
  Natural Language Supervision (CLIP). *ICML 2021*.
- Jang, E. et al. (2022). BC-Z — Zero-Shot Task Generalization with
  Robotic Imitation Learning. *CoRL 2021*.
- Brohan, A. et al. (2022). RT-1 — Robotics Transformer for Real-World
  Control at Scale. arXiv:2212.06817.
- Reed, S. et al. (2022). A Generalist Agent (Gato). *TMLR*.

### Chapter 12. Scaling up: PaLM-E, RT-2, OpenVLA, Octo (and friends)

- Driess, D. et al. (2023). PaLM-E — An Embodied Multimodal Language
  Model. arXiv:2303.03378.
- Brohan, A. et al. (2023). RT-2 — Vision-Language-Action Models
  Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
- Kim, M. J. et al. (2024). OpenVLA. arXiv:2406.09246.
- Octo Model Team / Ghosh, D. et al. (2024). Octo — An Open-Source
  Generalist Robot Policy. arXiv:2405.12213.
- O'Neill, A. et al. (2024). Open X-Embodiment — Robotic Learning
  Datasets and RT-X Models. arXiv:2310.08864.
- Liu et al. (2024). RDT-1B — A Diffusion Foundation Model for
  Bimanual Manipulation. arXiv:2410.07864.

### Chapter 13. Smooth control: π0 and flow-matching action heads

- Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model
  for General Robot Control. arXiv:2410.24164 (companion technical
  report at pi.website/download/pi0.pdf).
- Pertsch, K. et al. (2025). FAST — Efficient Action Tokenization for
  Vision-Language-Action Models. arXiv:2501.09747.
- Frontier Robotics (2025). SimVLA — A Simple VLA Baseline for
  Robotic Manipulation. arXiv:2602.18224.
- Xiaomi Robotics (2025). Xiaomi-Robotics-0. arXiv:2602.12684.
- Hugging Face (2025). SmolVLA. arXiv:2506.01844.

### Chapter 14. Dual-system architectures: Helix and GR00T N1

- Figure AI (2024). Helix — A Vision-Language-Action Model for
  Generalist Humanoid Control. figure.ai/news/helix.
- Figure AI (2026). Helix 02 — A General-Purpose Humanoid System.
  figure.ai/news/helix-02.
- NVIDIA (2025). GR00T N1 — An Open Foundation Model for Generalist
  Humanoid Robots. arXiv:2503.14734.
- Driess, D. et al. (2023). PaLM-E (antecedent of VLM + controller
  decoupling). arXiv:2303.03378.
- Ahn, M. et al. (2022). Do As I Can, Not As I Say (SayCan). *CoRL
  2022*.

### Chapter 15. Datasets, benchmarks, and evaluation; specialized and adjacent VLAs

- Wei et al. (2025). LiLo-VLA — Compositional Long-Horizon
  Manipulation via Linked Object-Centric Policies. arXiv:2602.21531.
- Long-VLA Team (2025). Unleashing Long-Horizon Capability of Vision-
  Language Models for Manipulation (Long-VLA). arXiv:2508.19958.
- Tianjin University / Huawei (2025). Embodied-R1 — Reinforced
  Embodied Reasoning for Manipulation. arXiv:2508.13998.
- Huang et al. (2024). LEO — An Embodied Generalist Agent in 3D
  World. arXiv:2311.12871.
- OpenDriveVLA Team (2025). OpenDriveVLA — Towards End-to-End
  Autonomous Driving with VLAs. arXiv:2503.23463.
- Zheng et al. (2025). UniAct — A Universal Action Model for Cross-
  Embodiment Robot Manipulation. arXiv:2501.10105.
- TinyVLA Team (2024). TinyVLA — Toward Fast, Data-Efficient Vision-
  Language-Action Models for Robotic Manipulation. arXiv:2409.12514.

### Chapter 16. Fine-tuning a VLA for your robot

- Hu, E. J. et al. (2022). LoRA — Low-Rank Adaptation of Large
  Language Models. *ICLR 2022*.
- Kim, M. J. et al. (2024). OpenVLA. arXiv:2406.09246.
- Octo Model Team / Ghosh, D. et al. (2024). Octo. arXiv:2405.12213.
- Wu et al. (2025). A Survey on Efficient Vision-Language-Action
  Models. arXiv:2510.24795.
- TinyVLA Team (2024). TinyVLA. arXiv:2409.12514.
- Hugging Face (2025). SmolVLA. arXiv:2506.01844.
- Liu et al. (2024). RoboMamba. arXiv:2406.04339.

### Chapter 17. Evaluation, safety, and deployment

- García, J., & Fernández, F. (2015). A Comprehensive Survey on Safe
  Reinforcement Learning. *Journal of Machine Learning Research*, 16.
- Alshiekh, M. et al. (2018). Safe Reinforcement Learning via
  Shielding. *AAAI 2018*.
- Brunke, L. et al. (2022). Safe Learning in Robotics — From Learning-
  Based Control to Safe Reinforcement Learning. *Annual Review of
  Control, Robotics, and Autonomous Systems*, 5.
- Amodei, D. et al. (2016). Concrete Problems in AI Safety.
  arXiv:1606.06565.
- Garcia et al. (2025). LIBERO-Para. arXiv:2603.28301.
- Ni et al. (2025). Embodied Arena. arXiv:2509.15273.

### Chapter 18. Open problems and what comes next

- Zhang et al. (2025). Pure Vision Language Action (VLA) Models — A
  Comprehensive Survey. arXiv:2509.19012.
- Sapkota, R. et al. (2025). Vision-Language-Action Models — Concepts,
  Progress, Applications and Challenges. arXiv:2505.04769.
- Xu et al. (2025). An Anatomy of Vision-Language-Action Models — From
  Modules to Milestones and Challenges. arXiv:2512.11362.
- Wu et al. (2025). A Survey on Efficient Vision-Language-Action
  Models. arXiv:2510.24795.
- Stanford HAI (2025). *AI Index Report 2025* — Robotics chapter.
- Firoozi, R. et al. (2023). Foundation Models in Robotics —
  Applications, Challenges, and the Future. arXiv:2312.07843.

## E.3  Where the mirror is incomplete

Three categories of citation in this book are not held in the offline
mirror. *Textbooks* — Sutton and Barto, Bishop, Strang, LaValle,
Featherstone, Murray-Li-Sastry — are commercial works and not
mirrored; the reader is expected to either own a copy or to access
them through institutional channels. *Older pre-arXiv papers* —
Fikes and Nilsson (1971), Pomerleau (1988), Khatib (1987), Hogan
(1985), McDermott et al. (1998) — predate arXiv and are cited by
venue; canonical scans exist on author homepages and university
repositories and are linked to in `SOURCES_INDEX.txt` where stable
URLs were available at snapshot time. *Vendor technical reports* —
the Figure AI Helix and Helix 02 posts, the BAAI RoboBrain2.0 report,
the Microsoft ρα report — are HTML pages that have been snapshotted
to `sources/html/` with the date of capture recorded; if the live
version has since changed, the snapshot is the canonical reference
for what is claimed in the book.

A reader who finds a missing reference is asked to open an issue on
the book's companion repository so the next edition's mirror script
can pick it up.
