# Book progress — Action Models for Robot Learning

Working draft of all chapter sections. The daily scheduled task reads this file,
identifies the next pending section, drafts it, and updates the status here.

**Schema:**
- `[ ]` pending — not yet drafted
- `[~]` in progress — picked up by today's session
- `[x]` drafted — first-draft markdown exists under `book/chapter_NN/`
- `[r]` revised — Pavan has done a pass on it

**Pace target:** one section per day, ~1,500–2,500 words.
**Total sections:** 18 chapters × ~6 sections = ~108 sections + appendices.

## Conventions

- Section files live at `book/chapter_NN/section_N_N.md`. Use two-digit chapter
  numbers (`chapter_01`, `chapter_18`) and underscored section numbers
  (`section_1_1.md` for §1.1).
- Each section file opens with a YAML-style frontmatter block listing the
  section number, title, target word count, and chapter prerequisites — the
  daily task expects this format.
- All citations use the canonical references compiled in
  `Action_Models_Detailed_TOC.docx` Appendix E.2, with arXiv IDs from
  `sources/SOURCES_INDEX.txt`. Prefer arXiv IDs over URLs.
- The daily task drafts in `.md` first; we promote sections to chapter `.docx`
  files at the end of each chapter for editorial review.

## Status

### Part 1 — Foundations and a first taste of VLAs

#### Chapter 1. The robot learning problem

- [x] 1.1 Why "action" is the hard part of robotics
- [x] 1.2 Anatomy of an action model: inputs, outputs, training signal
- [x] 1.3 A short history, from STRIPS to π0
- [x] 1.4 The four families of action models
- [x] 1.5 What you will and will not find in this book
- [x] 1.6 Summary
- [x] 1.x Hands-on exercise + chapter references

#### Chapter 2. Your first VLA, end-to-end

Sample-chapter draft `Sample_Chapter_02_Your_First_VLA.docx` (7 pages, organized as 2.1–2.8) is the raw material; sections below promote and expand it to the TOC's 2.1–2.6 spine.

- [x] 2.1 What we are going to build, and what is hidden inside
- [x] 2.2 Setting up the environment (OpenVLA weights, LIBERO simulator)
- [x] 2.3 Walking through the inference loop one line at a time
- [x] 2.4 When it works and when it does not
- [x] 2.5 What is left for the rest of the book
- [x] 2.6 Summary
- [x] 2.x Hands-on exercise + chapter references

#### Chapter 3. Math and ML prerequisites in 30 minutes

- [x] 3.1 Vectors, matrices, gradients, and why the chain rule rules robotics
- [x] 3.2 Random variables, expectations, KL divergence
- [x] 3.3 A 50-line PyTorch training loop, annotated
- [x] 3.4 Three loss families: supervised, RL, self-supervised
- [x] 3.5 Debugging a model that will not train
- [x] 3.6 Summary
- [x] 3.x Hands-on exercise + chapter references

### Part 2 — The lineage that produced VLAs

#### Chapter 4. Classical action models: planning and inverse dynamics

- [x] 4.1 Symbolic actions: STRIPS, PDDL, and action schemas
- [x] 4.2 Geometric actions: inverse kinematics and motion planning
- [x] 4.3 Inverse dynamics and computed-torque control
- [x] 4.4 Where classical methods are still load-bearing in modern robots
- [x] 4.5 Summary
- [x] 4.x Hands-on exercise + chapter references

#### Chapter 5. Learning from rewards: MDPs and reinforcement learning

- [x] 5.1 States, actions, rewards, and policies
- [x] 5.2 Value iteration and policy iteration
- [x] 5.3 Q-learning and the role of exploration
- [x] 5.4 Why reward design is the hardest part
- [x] 5.5 The MDP-to-robot translation problem
- [x] 5.6 Summary
- [x] 5.x Hands-on exercise + chapter references

#### Chapter 6. Learning from demonstrations: behavior cloning and imitation learning

- [x] 6.1 Why imitation is the dominant signal in modern robotics
- [x] 6.2 Behavior cloning, step by step
- [x] 6.3 Compounding error and DAgger
- [x] 6.4 A glance at IRL and adversarial imitation
- [x] 6.5 Choosing between BC, IRL, and RL
- [x] 6.6 Summary
- [x] 6.x Hands-on exercise + chapter references

#### Chapter 7. Deep RL for control: DQN to SAC and PPO

- [x] 7.1 Function approximation: from Q-tables to Q-networks
- [x] 7.2 Policy gradients and the variance problem
- [x] 7.3 PPO in 100 lines
- [x] 7.4 Off-policy actor-critic: DDPG, TD3, SAC
- [x] 7.5 Sim-to-real: domain randomization in one slide
- [x] 7.6 Summary
- [x] 7.x Hands-on exercise + chapter references

### Part 3 — Modern building blocks

#### Chapter 8. Sequence models meet control

- [x] 8.1 The transformer in two pages, for control
- [x] 8.2 Decision Transformer: control as conditional sequence modeling
- [x] 8.3 Trajectory Transformer and beam-search planning
- [x] 8.4 What gets tokenized: states, actions, returns, language
- [x] 8.5 Bridge to foundation action models — and to the SSM alternative (RoboMamba)
- [x] 8.6 Summary
- [x] 8.x Hands-on exercise + chapter references

#### Chapter 9. World models and model-based learning

- [x] 9.1 What is a world model, really
- [x] 9.2 Latent dynamics: RSSM and Dreamer
- [x] 9.3 Planning in latent space
- [x] 9.4 Video-prediction world models (Genie, V-JEPA)
- [x] 9.5 World models vs. VLAs: the architecture debate
- [x] 9.6 Summary
- [x] 9.x Hands-on exercise + chapter references

#### Chapter 10. Diffusion and flow models for action generation

- [x] 10.1 A 10-minute introduction to diffusion models
- [x] 10.2 Diffusion Policy and ACT
- [x] 10.3 Flow matching and rectified flow for action
- [x] 10.4 Trade-offs: latency, multimodality, smoothness
- [x] 10.5 Action-head choices in modern VLAs
- [x] 10.6 Summary
- [x] 10.x Hands-on exercise + chapter references

### Part 4 — Foundation action models in depth

#### Chapter 11. The VLA recipe: from CLIP to RT-1

- [x] 11.1 CLIP and the multimodal pretraining moment
- [x] 11.2 Language-conditioned imitation: BC-Z, RT-1
- [x] 11.3 Action tokenization: a small idea with large consequences
- [x] 11.4 What RT-1 changed and what it did not
- [x] 11.5 The data side: when does scale start to pay off
- [x] 11.6 Summary
- [x] 11.x Hands-on exercise + chapter references

#### Chapter 12. Scaling up: RT-2, OpenVLA, and Octo

- [x] 12.1 RT-2: a VLM that also outputs actions
- [x] 12.2 OpenVLA: an open-source 7B-parameter VLA
- [x] 12.3 Octo: a generalist policy with a diffusion head
- [ ] 12.4 Open X-Embodiment: the dataset that made all of this possible
- [ ] 12.5 What "emergent" really means in this context
- [ ] 12.6 Summary
- [ ] 12.x Hands-on exercise + chapter references

#### Chapter 13. Smooth control: π0 and flow-matching action heads

- [ ] 13.1 The trouble with discrete action tokens
- [ ] 13.2 π0's architecture, end to end
- [ ] 13.3 Flow matching as a control objective
- [ ] 13.4 What π0 can do that earlier VLAs cannot
- [ ] 13.5 Open questions in continuous-action foundation models
- [ ] 13.6 Summary
- [ ] 13.x Hands-on exercise + chapter references

#### Chapter 14. Dual-system architectures: Helix and GR00T N1

- [ ] 14.1 Why a single forward pass is not always enough
- [ ] 14.2 Helix: a high-level VLM and a low-level sensorimotor model
- [ ] 14.3 GR00T N1: humanoid-flavored dual systems
- [ ] 14.4 Latency budgets and real-time control
- [ ] 14.5 Deployment case studies (Figure 02, GR00T-enabled humanoids)
- [ ] 14.6 Summary
- [ ] 14.x Hands-on exercise + chapter references

#### Chapter 15. Datasets, benchmarks, and evaluation

- [ ] 15.1 What a robot dataset looks like, by example
- [ ] 15.2 Open X-Embodiment in detail
- [ ] 15.3 Sim benchmarks (LIBERO, CALVIN, RoboCasa, SimplerEnv)
- [ ] 15.4 Real-robot evaluation: variance, success rate, time-to-completion
- [ ] 15.5 Building your own evaluation
- [ ] 15.6 Summary
- [ ] 15.x Hands-on exercise + chapter references

### Part 5 — Building with action models

#### Chapter 16. Fine-tuning a VLA for your robot

- [ ] 16.1 Picking a base model
- [ ] 16.2 Building a teleop dataset that does not waste your time
- [ ] 16.3 LoRA vs. full fine-tuning vs. action-head-only
- [ ] 16.4 Sim-to-real fine-tuning loops
- [ ] 16.5 A recipe card for new embodiments
- [ ] 16.6 Summary
- [ ] 16.x Hands-on exercise + chapter references

#### Chapter 17. Evaluation, safety, and deployment

- [ ] 17.1 Safety as a layer, not a property
- [ ] 17.2 Runtime monitors and shielding
- [ ] 17.3 A/B evaluation on hardware
- [ ] 17.4 Logging, alerting, and rollback
- [ ] 17.5 What we still cannot certify
- [ ] 17.6 Summary
- [ ] 17.x Hands-on exercise + chapter references

#### Chapter 18. Open problems and what comes next

- [ ] 18.1 Generalization across embodiments
- [ ] 18.2 Long-horizon and dexterous tasks
- [ ] 18.3 Video-pretrained action models
- [ ] 18.4 Reasoning + action: LLM chains of thought meet control
- [ ] 18.5 What to read next, and how to contribute
- [ ] 18.6 Summary
- [ ] 18.x Hands-on exercise + chapter references

### Appendices

- [x] A. Linear algebra refresher
- [x] B. Probability and information theory
- [x] C. PyTorch and JAX primer
- [x] D. Setting up a robotics simulator
- [x] E. Canonical references (full bibliography, by chapter)
- [x] F. VLA model zoo

## Daily session log

(The scheduled task appends one line per run, format: `YYYY-MM-DD — drafted §N.M (~W words)`.)

- 2026-05-13 — drafted §1.1 Why "action" is the hard part of robotics (~2060 words)
- 2026-05-13 — drafted §1.2 Anatomy of an action model: inputs, outputs, training signal (~2135 words)
- 2026-05-13 — drafted §1.3 A short history, from STRIPS to π0 (~2152 words)
- 2026-05-14 — drafted §1.4 The four families of action models (~2033 words)
- 2026-05-15 — drafted §1.5 What you will and will not find in this book (~1909 words)
- 2026-05-15 — drafted §1.6 Summary (~1606 words)
- 2026-05-15 — drafted §1.x Hands-on exercise + chapter references (~1789 words)
- 2026-05-15 — drafted §2.1 What we are going to build, and what is hidden inside (~1722 words)
- 2026-05-15 — drafted §2.2 Setting up the environment (OpenVLA weights, LIBERO simulator) (~1905 words)
- 2026-05-15 — drafted §2.3 Walking through the inference loop one line at a time (~1905 words)
- 2026-05-16 — drafted §2.4 When it works and when it does not (~1983 words)
- 2026-05-19 — drafted §2.5 What is left for the rest of the book (~1846 words)
- 2026-05-19 — drafted §2.6 Summary (~1761 words)
- 2026-05-20 — drafted §2.x Hands-on exercise + chapter references (~2000 words)
- 2026-05-21 — drafted §3.1 Vectors, matrices, gradients, and why the chain rule rules robotics (~1734 words)
- 2026-05-22 — drafted §3.2 Random variables, expectations, KL divergence (~2093 words)
- 2026-05-23 — drafted §3.3 A 50-line PyTorch training loop, annotated (~2040 words)
- 2026-05-24 — drafted §3.4 Three loss families: supervised, RL, self-supervised (~2191 words)
- 2026-05-25 — drafted §3.5 Debugging a model that will not train (~2124 words)
- 2026-05-26 — drafted §3.6 Summary (~1866 words)
- 2026-05-27 — drafted §3.x Hands-on exercise + chapter references (~2225 words)
- 2026-05-27 — drafted §4.1 Symbolic actions: STRIPS, PDDL, and action schemas (~2038 words)
- 2026-05-29 — drafted §4.2 Geometric actions: inverse kinematics and motion planning (~1770 words)
- 2026-05-30 — drafted §4.3 Inverse dynamics and computed-torque control (~2148 words)
- 2026-05-31 — drafted §4.4 Where classical methods are still load-bearing in modern robots (~2052 words)
- 2026-06-01 — drafted §4.5 Summary (~2030 words)
- 2026-06-02 — drafted §4.x Hands-on exercise + chapter references (~2510 words)
- 2026-06-03 — drafted §5.1 States, actions, rewards, and policies (~2069 words)
- 2026-06-03 — drafted Appendix A: Linear algebra refresher (~2471 words)
- 2026-06-03 — drafted Appendix B: Probability and information theory (~2226 words)
- 2026-06-03 — drafted Appendix C: PyTorch and JAX primer (~2302 words)
- 2026-06-03 — drafted Appendix D: Setting up a robotics simulator (~2053 words)
- 2026-06-03 — drafted Appendix E: Canonical references and offline source mirror (~2269 words)
- 2026-06-03 — drafted Appendix F: Model zoo (~1598 words)
- 2026-06-04 — drafted §5.2 Value iteration and policy iteration (~1819 words)
- 2026-06-05 — drafted §5.3 Q-learning and the role of exploration (~2144 words)
- 2026-06-07 — drafted §5.4 Why reward design is the hardest part (~1949 words)
- 2026-06-08 — drafted §5.5 The MDP-to-robot translation problem (~2241 words)
- 2026-06-08 — drafted §5.6 Summary (~2092 words)
- 2026-06-09 — drafted §5.x Hands-on exercise + chapter references (~2041 words)
- 2026-06-10 — drafted §6.1 Why imitation is the dominant signal in modern robotics (~1922 words)
- 2026-06-11 — drafted §6.2 Behavior cloning, step by step (~1650 words)
- 2026-06-12 — drafted §6.3 Compounding error and DAgger (~1928 words)
- 2026-06-15 — drafted §6.4 A glance at IRL and adversarial imitation (~1743 words)
- 2026-06-15 — drafted §6.5 Choosing between BC, IRL, and RL (~1778 words)
- 2026-06-15 — drafted §6.6 Summary (~1747 words)
- 2026-06-15 — drafted §6.x Hands-on exercise + chapter references (~1962 words)
- 2026-06-16 — drafted §7.1 Function approximation: from Q-tables to Q-networks (~1964 words)
- 2026-06-17 — drafted §7.2 Policy gradients and the variance problem (~1983 words)
- 2026-06-18 — drafted §7.3 PPO in 100 lines (~1711 words)
- 2026-06-19 — drafted §7.4 Off-policy actor-critic: DDPG, TD3, SAC (~1878 words)
- 2026-06-20 — drafted §7.5 Sim-to-real: domain randomization in one slide (~1629 words)
- 2026-06-21 — drafted §7.6 Summary (~1790 words)
- 2026-06-22 — drafted §7.x Hands-on exercise + chapter references (~2000 words)
- 2026-06-23 — drafted §8.1 The transformer in two pages, for control (~1624 words)
- 2026-06-24 — drafted §8.2 Decision Transformer: control as conditional sequence modeling (~1630 words)
- 2026-06-25 — drafted §8.3 Trajectory Transformer and beam-search planning (~1703 words)
- 2026-06-26 — drafted §8.4 What gets tokenized: states, actions, returns, language (~1580 words)
- 2026-06-27 — drafted §8.5 Bridge to foundation action models + SSM alternative (RoboMamba) (~1849 words)
- 2026-06-28 — drafted §8.6 Summary (~1792 words)
- 2026-06-29 — drafted §8.x Hands-on exercise + chapter references (~1980 words)
- 2026-06-30 — drafted §9.1 What is a world model, really (~1570 words)
- 2026-07-01 — drafted §9.2 Latent dynamics: RSSM and Dreamer (~1696 words)
- 2026-07-02 — drafted §9.3 Planning in latent space (~1835 words)
- 2026-07-03 — drafted §9.4 Video-prediction world models (Genie, V-JEPA) (~1836 words)
- 2026-07-04 — drafted §9.5 World models vs. VLAs: the architecture debate (~1650 words)
- 2026-07-05 — drafted §9.6 Summary (~1980 words)
- 2026-07-06 — drafted §9.x Hands-on exercise + chapter references (~2175 words)
- 2026-07-07 — drafted §10.1 A 10-minute introduction to diffusion models (~1760 words)
- 2026-07-08 — drafted §10.2 Diffusion Policy and ACT (~1685 words)
- 2026-07-09 — drafted §10.3 Flow matching and rectified flow for action (~1738 words)
- 2026-07-10 — drafted §10.4 Trade-offs: latency, multimodality, smoothness (~1720 words)
- 2026-07-11 — drafted §10.5 Action-head choices in modern VLAs (~1675 words)
- 2026-07-12 — drafted §10.6 Summary (~1836 words)
- 2026-07-13 — drafted §10.x Hands-on exercise + chapter references (~2100 words)
- 2026-07-14 — drafted §11.1 CLIP and the multimodal pretraining moment (~1699 words)
- 2026-07-15 — drafted §11.2 Language-conditioned imitation: BC-Z, RT-1 (~1658 words)
- 2026-07-15 — drafted §11.3 Action tokenization: a small idea with large consequences (~1650 words)
- 2026-07-16 — drafted §11.4 What RT-1 changed and what it did not (~1580 words)
- 2026-07-17 — drafted §11.5 The data side: when does scale start to pay off (~1825 words)
- 2026-07-18 — drafted §11.6 Summary (~1898 words)
- 2026-07-19 — drafted §11.x Hands-on exercise + chapter references (~1950 words)
- 2026-07-20 — drafted §12.1 RT-2: a VLM that also outputs actions (~1656 words)
- 2026-07-22 — drafted §12.2 OpenVLA: an open-source 7B-parameter VLA (~1708 words)
- 2026-07-23 — drafted §12.3 Octo: a generalist policy with a diffusion head (~1791 words)
