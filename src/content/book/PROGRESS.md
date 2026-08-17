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

## Field-tracking notes (2026-07-23)

Gap review against chapters 13–18 (all still pending as of this date). Sources to
pull into Appendix E / chapter references when each section is drafted:

- π0 lineage: π0.5 (arXiv 2504.16054, open-world generalization), π0.6 (Physical
  Intelligence, Nov 2025, RL-from-experience), π0.7 (Apr 2026, steerable generalist).
- Helix-02 (Figure, Jan 2026): "System 0" neural whole-body controller replacing
  ~109k lines of hand-coded C++ locomotion; May 2026 leaderless two-robot bedroom demo.
- GR00T N1.5 → N1.6 (Cosmos-2B VLM backbone, 2× DiT) → N1.7 (whole-body,
  GEAR-SONIC); NVIDIA/Unitree reference humanoid hardware, late 2026.
- Gemini Robotics 1.5 / Robotics-ER 1.5–1.6 (arXiv 2510.03342): "Embodied
  Thinking" and Motion Transfer — a third dual-system-style family, not yet
  represented in the TOC.
- Dexterity-focused foundation models: Genesis AI GENE-26.5, RLWRLD RLDX-1.
- V-JEPA 2 → V-JEPA 2-AC (arXiv 2506.09985; Meta AI, 2025): world model
  post-trained into a zero-shot pick-and-place policy from <62h unlabeled video,
  80% vs. Octo's 15% on a held-out task — cross-reference from §9.4.
- Embodied reasoning: embodied chain-of-thought, latent-reasoning VLAs
  (LaRA-VLA, arXiv 2602.01166), HALO, ThinkAct.
- Safety: VLA-safety survey (arXiv 2604.23775), runtime action authorization
  survey (arXiv 2606.00090), TRAP adversarial-patch CoT hijacking (arXiv
  2603.23117).
- Benchmarks: RoboArena as a red-team-style eval suite alongside LIBERO/CALVIN/
  SimplerEnv; Open X-Embodiment has no single branded successor — growth now
  comes from large teleop fleets (AGIBot Genie-1, bimanual YAM, Unitree G1).

## Field-tracking notes (2026-07-23b)

Second pass same day — fresh searches turned up items not in the entry above:

- LingBot-VLA 2.0 (Ant Group/Robbyant, released 2026-07-08): 6B-parameter open-source
  VLA, Apache 2.0 weights + code, one policy running unmodified across 20 robot
  morphologies from 17 manufacturers (Unitree, AgiBot, Franka, Fourier, etc.);
  reported to beat π0.5 on the GM-100 bimanual benchmark. Trained on 60k hours of
  real-world data (50k robot teleop + 10k distilled first-person human video).
  Relevant to §13.4/§13.5 and §15.2 (teleop-fleet successors to Open X-Embodiment).
- GR00T N1.7 reached General Availability (NVIDIA, ~Apr–Jul 2026): swaps the Eagle
  VLM backbone for Cosmos-Reason2-2B (Qwen3-VL architecture, physics-informed
  pretraining via Cosmos World Foundation Model); central new contribution is
  EgoScale, ~20,854 hours of first-person human video pretraining feeding a
  32-layer DiT low-level controller. Refines the N1.5→N1.7 lineage note already in
  §14.3.
- New VLA adversarial-safety papers not yet listed: VLA-Hijack, a transferable
  patch attack with cross-architecture black-box transfer across OpenVLA/UniVLA/
  CronusVLA (arXiv 2605.28083); partially-observable adversarial patch attacks on
  VLA policies (arXiv 2606.03556); ForesightSafety-VLA, a diagnostic safety
  benchmark (arXiv 2606.27079). Relevant to §17.5 alongside the already-tracked
  TRAP paper.
- Broader dataset/benchmark survey: "Vision-Language-Action in Robotics: A Survey
  of Datasets, Benchmarks, and Data Engines" (arXiv 2604.23001) — useful framing
  reference for §15.1–§15.3.

## Field-tracking notes (2026-08-01)

Gap review against chapters 16–18 (all still pending as of this date; 15.6–15.x
also pending but generic/exercise sections with no specific finding to attach).

- Gemini Robotics 2 / Gemini Robotics-ER 2 / Gemini Robotics On-Device 2 (Google
  DeepMind, announced 2026-07-30): first VLA to drive a full humanoid — legs,
  torso, arms, and multi-finger hands — under one unified policy, addressing
  loco-manipulation directly rather than handing off between separate
  locomotion and manipulation controllers; demonstrated on Apptronik's Apollo 2.
  The on-device variant reportedly adapts to a new embodiment from only a few
  hours of data. A fourth major dual-system-style family alongside Helix,
  GR00T N1, and the earlier Gemini Robotics-ER 1.5/1.6 line already noted above.
  (deepmind.google/blog, MarkTechPost 2026-07-30, roboticsandautomationnews.com)
- WholebodyVLA (OpenDriveLab, ICLR 2026): "Towards Unified Latent VLA for
  Whole-body Loco-manipulation Control" — an open research counterpart to the
  Gemini Robotics 2 loco-manipulation direction (github.com/OpenDriveLab/WholebodyVLA).
- ERVLA, "Revisiting Embodied Chain-of-Thought for Generalizable Robot
  Manipulation" (arXiv 2606.03784, June 2026): trains with CoT reasoning traces
  via a CoT-dropout strategy but predicts actions directly at inference,
  sidestepping autoregressive CoT latency; reports 86.9% on LIBERO-Plus and
  53.2% on VLABench. Adds to the embodied-reasoning family already tracked
  (LaRA-VLA, HALO, ThinkAct).
- VLA runtime-monitoring: "Hide-and-Seek in Trajectories: Discovering Failure
  Signals for VLA Runtime Monitoring" (arXiv 2605.30834, May 2026) — addresses
  label noise in step-level failure annotation for runtime monitors.
- VLA adversarial safety: CVPR 2026 AdvML Workshop Challenge on adversarial
  multimodal attacks against driving VLAs (arXiv 2607.11560, July 2026); "Lost
  in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs"
  (arXiv 2605.21446, May 2026). Both extend the attack-surface list alongside
  the already-tracked TRAP, VLA-Hijack, and partially-observable patch papers.

## Field-tracking notes (2026-08-15)

All 108 sections + appendices are at `[x]`, so there are no pending `[ ]` bullets to
revise; findings below are logged for the revision (`[r]`) passes, plus one new
pending section added under the new-section rule.

- **World Action Models (WAMs) — a named architectural paradigm positioned against
  the VLA recipe.** NVIDIA replaces the VLM backbone with a video world model and
  post-trains a single network to jointly predict future video *and* actions
  ("World Action Models are Zero-shot Policies", arXiv 2602.15922; NVIDIA
  Technical Blog "Beyond VLAs: How World Action Models Reshape Robot
  Manipulation", 2026-08-04; NVIDIA glossary entry for "world action model").
  The argument: a VLM backbone learns to *describe* the world, not predict how it
  evolves, so VLAs generalize semantically but not physically. Substrate is
  Cosmos 3, a Mixture-of-Transformers omni world foundation model (4B Edge / 16B
  Nano / 64B Super; ~767M images, 348M videos, 8M action samples), with released
  policies Cosmos3-Nano-Policy-DROID and Cosmos3-Edge-Policy-DROID (Hugging Face,
  commercial-use license). Reported ablation: same recipe/data/compute, omni
  checkpoint vs. base raises RoboLab success 28.1% → 36.8%; Edge 4B runs 15 Hz
  real-time control with 32-action chunks on Jetson Thor. Jim Fan's framing
  ("VLAs are dead, long live World Action Models") is the rhetorical version.
  Third-party WAM work is already appearing: unified 4D world-action modeling with
  asynchronous denoising (arXiv 2604.26694), HiMem-WAM hierarchical memory-gated
  WAMs (arXiv 2606.10363), and a world-model-for-robot-learning survey (arXiv
  2605.00080).
  → **New section warranted, not a parenthetical:** §9.5 frames "world models vs.
  VLAs" as a debate between two separate model classes; WAM is a third thing — a
  world model *as* the policy backbone, with its own pretraining story, deployment
  tiers, and named model family. Added as pending §9.7 in Chapter 9. Should also
  be cross-referenced from §18.3 and §14.3 during revision (GR00T's Cosmos-Reason
  backbone is the same lineage arriving from the other direction).
- Benchmarks: RoboCasa365 (arXiv 2603.04356, ICLR 2026; Nasiriany et al., UT
  Austin + NVIDIA Research) — 365 household mobile-manipulation tasks across 2,500
  kitchen scenes, ~600h human + ~1,600h synthetic demonstrations, explicitly built
  for multi-task / foundation-model / lifelong-learning evaluation. This is the
  concrete "successor to RoboCasa" for §15.4. Also LIBERO-Plus, a robustness
  variant of LIBERO perturbing seven factors (layout, viewpoint, initial state,
  instruction phrasing, lighting, background texture, sensor noise) — previously
  referenced here only as a score ERVLA reported against, but it belongs in §15.4
  as a benchmark in its own right.
- Policy evaluation via world models: GigaWorld-1 (arXiv 2607.02642), a roadmap for
  using world models as the evaluation substrate for robot policies — relevant to
  §15.5/§15.6, where real-robot evaluation variance is the central problem.
- Embodied-reasoning skepticism (a counterweight the book currently lacks):
  "Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models"
  (arXiv 2604.18000) reports reasoning deficits in SOTA VLAs traced to
  representational bottlenecks in VLM→VLA adaptation, persisting on real hardware;
  "Do Vision-Language-Action Models Mean What They Say? On the Role of Faithfulness
  in Embodied Reasoning" (arXiv 2607.04681) asks whether verbalized CoT reflects
  the policy's actual decision process. Both belong in §18.4 alongside the
  constructive work already tracked (ECoT, LaRA-VLA, ERVLA, HALO, ThinkAct) and in
  §14.3's Gemini "Embodied Thinking" discussion. Constructive additions: dense
  embodied-CoT supervision (arXiv 2606.30552), Guide-Think-Act interactive embodied
  reasoning (arXiv 2605.13632).
- Safety, **training-time** threats (a category §17.5 does not cover; it treats
  inference-time patches and sensor perturbation only): AttackVLA (arXiv
  2511.12149, Li et al.) is a unified attack framework spanning the VLA
  development lifecycle, and it makes a useful negative observation — existing
  attacks mostly induce *untargeted* failure or a frozen action state, so
  precise targeted control was unexplored. Its BackdoorVLA fills that gap: a
  trigger that compels an attacker-specified long-horizon action sequence,
  58.4% average targeted success rate, 100% on selected tasks, validated in
  simulation *and* on real hardware. ATAAT (arXiv 2605.08612, Chen et al.,
  2026-05-09) is also an **attack**, not a defense, despite the "adversarial
  tuning" framing — it diagnoses "gradient interference" as why prior VLA
  backdoors failed and reaches TASR > 80% at a 5% poisoning rate.
  *Correction to note: an earlier draft of this bullet described ATAAT as a
  defense. It is not.* The honest state of the field is that the attack
  literature is well ahead of the defense literature, which is itself worth
  saying in the book. The Awesome-VLA-Safety list
  (github.com/LiQiiiii/Awesome-VLA-Safety) accompanying arXiv 2604.23775 is a
  usable living bibliography for §17.5.
- Dexterity/tactile: TouchWorld, a predictive-and-reactive tactile foundation model
  for dexterous manipulation (arXiv 2607.07287) — tactile sensing as a foundation-
  model modality, orthogonal to the vision-first GENE/RLDX-1 line already tracked;
  mimic Robotics announced a full-stack dexterous platform (M1 hand, U1 wearable
  exoskeleton, AI platform) on 2026-07-18 (roboticsandautomationnews.com). Both
  feed §14.5 and §18.2.
- Datasets: Open-H-Embodiment (arXiv 2604.21017), a cross-embodiment
  multi-institution medical-robotics corpus — 119 datasets, ~780h paired video +
  kinematics, 50+ institutions. Evidence for the §15.2 thesis that OXE's successors
  are domain-specific consortia rather than one branded aggregate.

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
- [x] 9.7 World Action Models: the world model as the policy backbone — jointly predicting future video and actions (DreamZero / Cosmos 3), why a physics prior generalizes where a semantic prior does not, and how this reframes the §9.5 debate. (Added 2026-08-15; placed after 9.x to avoid renumbering the existing Summary/exercise bullets.)

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
- [x] 12.4 Open X-Embodiment: the dataset that made all of this possible
- [x] 12.5 What "emergent" really means in this context
- [x] 12.6 Summary
- [x] 12.x Hands-on exercise + chapter references

#### Chapter 13. Smooth control: π0 and flow-matching action heads

- [x] 13.1 The trouble with discrete action tokens
- [x] 13.2 π0's architecture, end to end
- [x] 13.3 Flow matching as a control objective
- [x] 13.4 What π0 can do that earlier VLAs cannot — and the π0.5 → π0.6 → π0.7 lineage (open-world generalization, RL-from-experience, steerable generalist behavior)
- [x] 13.5 Open questions in continuous-action foundation models
- [x] 13.6 Summary
- [x] 13.x Hands-on exercise + chapter references

#### Chapter 14. Dual-system architectures: Helix and GR00T N1

- [x] 14.1 Why a single forward pass is not always enough
- [x] 14.2 Helix: a high-level VLM and a low-level sensorimotor model — through Helix-02's "System 0" whole-body controller (replacing hand-coded locomotion) and leaderless multi-robot coordination
- [x] 14.3 GR00T N1 and its N1.5–N1.7 successors: humanoid-flavored dual systems — plus Gemini Robotics-ER as a third family (embodied reasoning + VLA, "Embodied Thinking," Motion Transfer)
- [x] 14.4 Latency budgets and real-time control
- [x] 14.5 Deployment case studies (Figure Helix-02, GR00T-enabled humanoids, dexterity-focused entrants such as Genesis AI's GENE and RLWRLD's RLDX-1)
- [x] 14.6 Gemini Robotics 2: unified whole-body loco-manipulation (a fourth dual-system family; first VLA to run legs, torso, arms, and multi-finger hands under one learned policy, demoed on Apptronik Apollo 2)
- [x] 14.7 Summary
- [x] 14.x Hands-on exercise + chapter references

#### Chapter 15. Datasets, benchmarks, and evaluation

- [x] 15.1 What a robot dataset looks like, by example
- [x] 15.2 Open X-Embodiment in detail — and what has (and has not) succeeded it: large teleop fleets (AGIBot Genie-1, bimanual YAM, Unitree G1) feeding newer foundation models rather than a single branded dataset
- [x] 15.3 LeRobot: Hugging Face's dataset format, hub, and community-contributed robot data
- [x] 15.4 Sim benchmarks (LIBERO, CALVIN, RoboCasa, SimplerEnv) and red-team-style suites such as RoboArena
- [x] 15.5 Real-robot evaluation: variance, success rate, time-to-completion
- [x] 15.6 Building your own evaluation
- [x] 15.7 Summary
- [x] 15.x Hands-on exercise + chapter references

### Part 5 — Building with action models

#### Chapter 16. Fine-tuning a VLA for your robot

- [x] 16.1 Picking a base model
- [x] 16.2 Building a teleop dataset that does not waste your time
- [x] 16.3 LoRA vs. full fine-tuning vs. action-head-only
- [x] 16.4 Sim-to-real fine-tuning loops
- [x] 16.5 A recipe card for new embodiments
- [x] 16.6 Summary
- [x] 16.x Hands-on exercise + chapter references

#### Chapter 17. Evaluation, safety, and deployment

- [x] 17.1 Safety as a layer, not a property
- [x] 17.2 Runtime monitors and shielding — VLM-as-monitor patterns (Code-as-Monitor), runtime action authorization, and step-level failure-signal detection (arXiv 2605.30834)
- [x] 17.3 A/B evaluation on hardware
- [x] 17.4 Logging, alerting, and rollback
- [x] 17.5 What we still cannot certify — plus emerging attack surfaces (adversarial patches that hijack VLA chain-of-thought reasoning, sensor-perturbation fragility, and the CVPR 2026 AdvML adversarial-VLA challenge)
- [x] 17.6 Summary
- [x] 17.x Hands-on exercise + chapter references
- [x] 17.7 Training-time attacks: poisoned data, backdoors, and a defense literature that has not caught up — supply-chain exposure in aggregated datasets and public checkpoints, AttackVLA/BackdoorVLA targeted long-horizon triggers, ATAAT's 5%-poisoning result, and why clean-task evaluation is blind to all of it. (Added 2026-08-15; placed after 17.x to avoid renumbering existing bullets.)

#### Chapter 18. Open problems and what comes next

- [x] 18.1 Generalization across embodiments — including the shift toward whole-body neural control (Helix-02's System 0, and now Gemini Robotics 2's unified loco-manipulation policy) blurring the classical/learned locomotion boundary from Chapter 4
- [x] 18.2 Long-horizon and dexterous tasks — dexterity remains below human level even as dedicated foundation models (GENE, RLDX-1) target it directly
- [x] 18.3 Video-pretrained action models — now with a concrete result: V-JEPA 2 → V-JEPA 2-AC, zero-shot pick-and-place from under 62 hours of unlabeled robot video
- [x] 18.4 Reasoning + action: LLM chains of thought meet control — no longer speculative; an active subfield (embodied CoT, latent-reasoning VLAs, ERVLA's CoT-dropout at inference, Gemini's Embodied Thinking)
- [x] 18.5 What to read next, and how to contribute
- [x] 18.6 Summary
- [x] 18.x Hands-on exercise + chapter references
- [x] 18.7 Does embodied reasoning actually reason? — the BeTTER stress tests (lexical-kinematic shortcuts, composition failure, behavioral inertia), functional vs. faithful reasoning, and the Pinocchio faithfulness critic; answers the question §18.4 left open. (Added 2026-08-15; placed after 18.x to avoid renumbering existing bullets.)
- [x] 18.8 Beyond vision: tactile sensing as a foundation-model modality — what a camera cannot see about contact, TouchWorld's predicted tactile subgoals plus high-frequency residual refinement, and why touch has no free corpus. (Added 2026-08-15; placed after 18.x to avoid renumbering existing bullets.)

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
- 2026-07-23 — drafted §12.4 Open X-Embodiment: the dataset that made all of this possible (~1532 words)
- 2026-07-24 — drafted §12.5 What "emergent" really means in this context (~1732 words)
- 2026-07-24 — drafted §12.6 Summary (~1920 words)
- 2026-07-24 — drafted §12.x Hands-on exercise + chapter references (~1990 words)
- 2026-07-27 — drafted §13.1 The trouble with discrete action tokens (~1580 words)
- 2026-07-27 — drafted §13.2 π0's architecture, end to end (~1575 words)
- 2026-07-27 — drafted §13.3 Flow matching as a control objective (~1605 words)
- 2026-07-27 — drafted §13.4 What π0 can do that earlier VLAs cannot (π0.5→π0.6→π0.7 lineage) (~1810 words)
- 2026-07-27 — drafted §13.5 Open questions in continuous-action foundation models (~1770 words)
- 2026-07-27 — drafted §13.6 Summary (~1894 words)
- 2026-07-27 — drafted §13.x Hands-on exercise + chapter references (~1840 words)
- 2026-07-28 — drafted §14.1 Why a single forward pass is not always enough (~1740 words)
- 2026-07-28 — drafted §14.2 Helix: a high-level VLM and a low-level sensorimotor model (~1543 words)
- 2026-07-28 — drafted §14.3 GR00T N1 and its N1.5–N1.7 successors + Gemini Robotics-ER (~1595 words)
- 2026-07-28 — drafted §14.4 Latency budgets and real-time control (~1791 words)
- 2026-07-28 — drafted §14.5 Deployment case studies (Figure/BMW, GR00T, GENE/RLDX-1) (~1656 words)
- 2026-07-28 — drafted §14.7 Summary (~2359 words)
- 2026-07-28 — drafted §14.x Hands-on exercise + chapter references (~1835 words)
- 2026-07-29 — drafted §15.1 What a robot dataset looks like, by example (~1812 words)
- 2026-07-29 — drafted §15.2 Open X-Embodiment in detail (and its fleet-scale successors) (~1610 words)
- 2026-07-29 — drafted §15.3 LeRobot: Hugging Face's dataset format, hub, and community data (~1750 words)
- 2026-07-30 — drafted §15.4 Sim benchmarks (LIBERO, CALVIN, RoboCasa, SimplerEnv) + RoboArena (~1565 words)
- 2026-07-31 — drafted §15.5 Real-robot evaluation: variance, success rate, time-to-completion (~1694 words)
- 2026-08-01 — drafted §15.6 Building your own evaluation (~1777 words)
- 2026-08-02 — drafted §15.7 Summary (~2041 words)
- 2026-08-03 — drafted §15.x Hands-on exercise + chapter references (~2075 words)
- 2026-08-03 — drafted §16.1 Picking a base model (~1470 words)
- 2026-08-03 — drafted §16.2 Building a teleop dataset that does not waste your time (~1426 words)
- 2026-08-03 — drafted §16.3 LoRA vs. full fine-tuning vs. action-head-only (~1375 words)
- 2026-08-03 — drafted §16.4 Sim-to-real fine-tuning loops (~1429 words)
- 2026-08-03 — drafted §16.5 A recipe card for new embodiments (~1315 words)
- 2026-08-03 — drafted §16.6 Summary (~1329 words)
- 2026-08-03 — drafted §16.x Hands-on exercise + chapter references (~1764 words)
- 2026-08-03 — drafted §14.6 Gemini Robotics 2: unified whole-body loco-manipulation (~1640 words)
- 2026-08-03 — renumbered Chapter 14: Gemini Robotics 2 section moved to §14.6 (was §14.7) so it precedes the Summary; Summary moved to §14.7. Cross-references in §14.1 and §14.x updated.
- 2026-08-03 — drafted §17.1 Safety as a layer, not a property (~1446 words)
- 2026-08-03 — drafted §17.2 Runtime monitors and shielding (~1362 words)
- 2026-08-03 — drafted §17.3 A/B evaluation on hardware (~1420 words)
- 2026-08-03 — drafted §17.4 Logging, alerting, and rollback (~1188 words)
- 2026-08-03 — drafted §17.5 What we still cannot certify (~1436 words)
- 2026-08-03 — drafted §17.6 Summary (~1628 words)
- 2026-08-03 — drafted §17.x Hands-on exercise + chapter references (~1951 words)
- 2026-08-03 — drafted §18.1 Generalization across embodiments (~1598 words)
- 2026-08-03 — drafted §18.2 Long-horizon and dexterous tasks (~1556 words)
- 2026-08-03 — drafted §18.3 Video-pretrained action models (~1541 words)
- 2026-08-03 — drafted §18.4 Reasoning + action: where LLM-style chains of thought meet control (~1630 words)
- 2026-08-03 — drafted §18.5 What to read next, and how to contribute (~1569 words)
- 2026-08-03 — drafted §18.6 Summary (~1571 words)
- 2026-08-03 — drafted §18.x Hands-on exercise + chapter references (~1627 words)
- 2026-08-03 — Chapter 18 complete. All 18 chapters and appendices A–F now drafted; book is fully drafted at first-draft [x] status. Remaining editorial work is revision ([r]) passes by Pavan.
- 2026-08-15 — field check: appended Field-tracking notes (2026-08-15); added pending §9.7 (World Action Models) under the new-section rule. No chapter content touched.
- 2026-08-15 — drafted §9.7 World Action Models: the world model as the policy backbone (~1665 words)
- 2026-08-15 — drafted §17.7 Training-time attacks: poisoned data, backdoors, and a defense literature that has not caught up (~1540 words)
- 2026-08-15 — drafted §18.7 Does embodied reasoning actually reason? (~1555 words)
- 2026-08-15 — drafted §18.8 Beyond vision: tactile sensing as a foundation-model modality (~1600 words)
- 2026-08-15 — corrected the 2026-08-15 field-tracking bullet on ATAAT (arXiv 2605.08612): it is an attack framework, not a defense. Section §17.7 is written on the corrected reading. Nothing in a drafted chapter had relied on the wrong version.
- 2026-08-04 — No pending [ ] sections found. Book remains fully drafted (all 108 sections + appendices A–F at [x]). No drafting action taken. Next step is revision ([r]) passes by Pavan; the daily draft task has no remaining first-draft work.
