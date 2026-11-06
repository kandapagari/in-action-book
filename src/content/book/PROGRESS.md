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

## Field-tracking notes (2026-08-31)

All sections remain `[x]`/`[r]`; no pending `[ ]` bullets exist to revise. Findings are
logged for the revision passes, plus one new pending section under the new-section rule.

- **GR00T N2 — the GR00T lineage switches paradigms, from VLA to world action model.**
  Previewed in Huang's GTC keynote (NVIDIA press release, 2026-03-16): built on the
  DreamZero world-action-model architecture rather than a VLM backbone, reported to
  succeed at new tasks in new environments more than twice as often as leading VLAs,
  ranked No. 1 on MolmoSpaces and RoboArena for generalist policies, GA slated for
  end of 2026. Same release: GR00T N1.7 moved to early access with commercial
  licensing, and Cosmos 3 was announced as a unified world foundation model.
  Not previously captured here — the 2026-08-15 entry tracked WAMs as an NVIDIA
  research direction (arXiv 2602.15922) without noting that the *flagship humanoid
  foundation model itself* is being rebuilt on it. §14.3's N1.5→N1.7 lineage and
  §9.7's WAM framing both need this during revision: the "VLA vs. WAM" debate is
  settled inside NVIDIA's own product line, not just in its blog posts.
  (nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)
- **MolmoSpaces (Ai2, arXiv 2602.11337)** — a large-scale open ecosystem for robot
  navigation and manipulation: 230k+ indoor scenes, 130k+ object models, 42M+
  annotated grasps, assets convertible to MuJoCo / ManiSkill / Isaac Lab via USD,
  plus MolmoSpaces-Bench and a public leaderboard (molmospaces.allen.ai/leaderboard)
  built specifically around generalization under systematic controlled perturbation.
  Belongs in §15.4 next to LIBERO / CALVIN / RoboCasa365 / SimplerEnv / RoboArena;
  it is now one of the two leaderboards vendors quote (see GR00T N2 above).
  Companion MolmoBot work trains manipulation entirely in simulation
  (allenai.org/blog/molmobot-robot-manipulation).
- mimic-video (arXiv 2512.15692, Mees et al.): a *video-action model* that leaves the
  pretrained internet-scale video backbone frozen and conditions a small flow-matching
  action decoder on its latents, treating the decoder as an inverse dynamics model.
  Reports 10× sample efficiency and 2× faster convergence over VLA architectures, and
  77% average success with a single episode per task (~2% of the action data). A
  cheaper, decoder-only point in the WAM design space from §9.7 — worth contrasting
  there with DreamZero's jointly-trained 14B model. Follow-on FLUX-mimic scaling work
  at mimicrobotics.com/blog/introducing-flux-mimic.
- **VLA backdoor *defenses* have started to appear**, which qualifies §17.7's claim
  that the defense literature has not caught up. TrustVLA (arXiv 2607.12571) is an
  inference-time, no-retraining defense: it adapts the Dirichlet evidence framework to
  monitor per-token, per-layer epistemic uncertainty, localizes the "compact causal
  footprint" a trigger leaves in attention, and inpaints or fails safe. Also newly
  surfaced on the attack side: State Backdoor (arXiv 2601.04266), which uses the arm's
  *initial state* as the trigger; Robot Collapse (arXiv 2411.11683), supply-chain
  backdoors against VLM-based manipulation; and backdoor-based *ownership verification*
  (arXiv 2605.09005), which inverts the threat model and uses a planted trigger as a
  watermark. §17.7 should soften from "no defenses exist" to "defenses exist but are
  inference-time patches on frozen checkpoints, and none address supply-chain
  provenance" — a more defensible and more interesting claim.
- Tactile: FTP-1, "A Generalist Foundation Tactile Policy Across Tactile Sensors for
  Contact-Rich Manipulation" (arXiv 2606.13102) — cross-*sensor* generalization, the
  tactile analogue of cross-embodiment. Pairs with TouchWorld (arXiv 2607.07287) in
  §18.8, and sharpens that section's "touch has no free corpus" point: the obstacle is
  not only corpus size but sensor heterogeneity.
- **Figure Index (figure.ai/news/introducing-index, 2026-08-25)** — Figure came out of
  stealth with a consumer smartphone app that pays members of the public ("Creators")
  to record first-person video of ordinary tasks. Reported figures: 264,000 downloads
  across 108 countries, 44,000 weekly active users, 16M videos uploaded, 30 minutes of
  video ingested every second (~4.9 human-years of work per day), $15M paid out to
  date, and a stated commitment of over $1B on data and compute in the next 12 months.
  Per 1,000 hours collected: 373 unique tasks, 1,146 unique manipulated objects, 116
  unique environments. Five-stage pipeline — automated quality filtering, human fraud
  review, embedding-based deduplication against a similarity threshold, task-quota and
  cluster-based rebalancing, hierarchical text captioning.
  → **New section warranted, not a parenthetical:** §15.2 frames OXE's succession as
  teleop fleets, and §15.1–§15.3 assume data comes from robots or from lab-curated
  video. Paid public crowdsourcing is a third acquisition regime with problems none of
  those sections cover: economic incentive to submit garbage (hence a fraud-review
  stage), diversity as a quantity you must actively defend by deduplication rather than
  something you get for free, and a human-video-to-robot-action gap that must be closed
  downstream. Added as pending §15.8 in Chapter 15. Should be cross-referenced from
  §11.5 (when scale pays off) and §16.2 (building a teleop dataset) during revision,
  and read against GR00T's EgoScale (§14.3) as the other large first-person-video corpus.

## Field-tracking notes (2026-09-01)

Only one day since the prior entry, so most items below are gaps the earlier sweeps
missed rather than same-day news. One new pending section added under the new-section rule.

- **Skild AI is entirely absent from this log, and its flagship model breaks an
  assumption Chapter 16 is built on.** Skild Brain 1.0 (late May 2026) is a
  ~500B-parameter "omni-bodied" robot foundation model — quadrupeds, humanoids,
  tabletop arms, mobile manipulators under one set of weights. S1, announced
  2026-08-25 (skild.ai/blogs/s1; The Robot Report 2026-08-31), replaces language
  instructions with *in-context learning*: a single human video of the task goes
  into the prompt and the shipped weights execute it, no post-training. Targets
  long-horizon tasks up to ~10 minutes (repotting a plant, brewing coffee, cooking
  pancakes); reported 66% vs. a 9% baseline. Pathak's stated pretraining position is
  that none of the four data sources suffices alone — teleop (high quality, slow,
  narrow), human video (abundant, far from the robot), simulation (scalable, gapped),
  data-capture gloves (in between) — so Skild mixes all four. Company context: $1.4B
  Series C (Jan 2026, SoftBank-led, ~$14B valuation, ~$1.7B raised total), Fetch
  Robotics assets acquired from Zebra.
  → **New section warranted, not a parenthetical:** every adaptation mechanism in
  Chapter 16 — LoRA, full fine-tuning, action-head-only, sim-to-real loops, the recipe
  card — assumes you adapt by *training*. S1's claim is that you adapt by *prompting*,
  which changes what §16.1 (picking a base model) and §16.5 (recipe card) are even
  choosing between, and it needs the honest caveats too (single vendor, self-reported
  numbers, no public weights, Pathak himself says this is not yet robotics' ChatGPT
  moment). Added as pending §16.7 in Chapter 16. Cross-reference from §12.5 during
  revision — "emergent" now has a concrete new referent in Skild's claim that spatula
  flipping appeared with no flipping examples in the corpus.
- **DYNA-2 (Dyna Robotics, 2026-08-10)** — a world-action model pretrained on
  >1M hours of egocentric *human* video and no robot data at all, with a dual
  next-frame / next-action objective. Reports a human-data scaling law across four
  orders of magnitude (1k → 1M hours) that transfers to unseen robot data: mean
  normalized performance across 14 physical tasks rising 20% → 53%, with power-law
  fits (held-out MSE ∝ D^-0.0184, R²=0.919). Claims both the data *and* the
  world-modeling objective are necessary for cross-embodiment transfer to emerge.
  (prnewswire 2026-08-10; MarkTechPost 2026-08-13; dyna.co/dyna-2). Belongs in §9.7
  as a third WAM point beside DreamZero and mimic-video — and it is the sharpest
  version of that section's argument, since it removes robot data from pretraining
  entirely. Also feeds §11.5 (when scale pays off) and §15.8.
- Broader WAM literature has consolidated fast enough to warrant a survey pointer for
  §9.7: Awesome-WAM (github.com/OpenMOSS/Awesome-WAM), plus DSWAM dual-system WAM
  (arXiv 2607.04927), MotuBrain (arXiv 2604.27792), OA-WAM object-addressable
  (arXiv 2605.06481), and MotionWAM for real-time humanoid loco-manipulation
  (arXiv 2606.09215) — the last connecting §9.7 to §14.6's loco-manipulation thread.
- ACoT-VLA, "Action Chain-of-Thought for Vision-Language-Action Models" (arXiv
  2601.11404, CVPR 2026, AgiBot): reasons in coarse *action intents* rather than
  language, via an Explicit Action Reasoner (lightweight transformer synthesizing
  coarse trajectories) plus an Implicit Action Reasoner (latent action priors from the
  VLM backbone). 98.5% LIBERO / 84.1% LIBERO-Plus / 47.4% VLABench; official baseline
  for the AgiBot World Challenge @ ICRA 2026. Sharpens §18.7's functional-vs-faithful
  framing — if the reasoning is not in language, the faithfulness question changes
  shape — and belongs in §18.4 beside ERVLA and ECoT.
- Pre-VLA, "Preemptive Runtime Verification for Reliable Vision-Language-Action and
  World-Model Rollouts" (arXiv 2605.22446): verification *before* execution rather than
  post-hoc failure detection, and covers world-model rollouts as well as VLA policies.
  §17.2 currently lists only reactive monitors (Code-as-Monitor, runtime action
  authorization, arXiv 2605.30834's failure signals); this is the preemptive category.
- Cross-embodiment: "One Policy, Many Embodiments — Unified Camera-Centric Action
  Geometry Pre-training" (arXiv 2608.26058), 4.03k h robot+sim plus 2.34k h human
  demonstrations, 98.3% LIBERO / 82.0% zero-shot LIBERO-Plus / 62.0% RoboCasa GR-1.
  A concrete recent data point for §18.1.
- Benchmark methodology, a counterweight §15.4/§15.5 lack: "What Are We Actually
  Benchmarking in Robot Manipulation?" (arXiv 2606.04233) questions what the standard
  suites measure; UMI-Bench 1.0 (arXiv 2606.10382) is an open reproducible *real-world*
  tabletop benchmark, i.e. an attempt at the reproducibility §15.5 says real-robot
  evaluation lacks.
- Field-scale framing stat for §18.5 or the introduction: ICLR 2026 received 164 VLA
  paper submissions, against a single submission in 2024 (semiengineering.com,
  "Vision-Language-Action Models Arrive"). Secondary source — verify against ICLR's own
  numbers before it goes into print.

## Field-tracking notes (2026-09-01b)

Second pass same day, prompted by the run schedule. Everything below is a gap the
earlier sweeps missed rather than news from the last 24 hours. One new pending
section added under the new-section rule.

- **Discrete diffusion VLAs are a whole action-head class this log has never
  mentioned, and they were one of the dominant threads at ICLR 2026.** Discrete
  Diffusion VLA (arXiv 2508.20072) models discretized action chunks with a discrete
  diffusion process *inside* the VLM backbone rather than bolting a separate
  continuous-diffusion head onto the outside. Adaptive decoding order (easy action
  elements first) plus secondary re-masking to revisit uncertain predictions;
  reports 96.3% on LIBERO, 71.2% visual matching on SimplerEnv-Fractal, 54.2% on
  SimplerEnv-Bridge, and cuts function evaluations from the sequence length (56 for
  autoregressive) to ~12 refinement rounds via parallel decoding. Related work:
  Unified Diffusion VLA (ICLR 2026, Reuss/Pari/Agrawal/Lioutikov — discrete
  diffusion for unified generation on a VL backbone), Fast-dVLA (arXiv 2603.25661,
  pushing discrete diffusion VLAs to real-time), dVLA-RL (arXiv 2606.23623, RL over
  denoising trajectories).
  → **New section warranted, not a parenthetical:** Chapter 10 presents the
  action-head design space as a two-way choice — discrete autoregressive tokens
  (§11.3, §13.1) versus continuous diffusion/flow heads outside the backbone
  (§10.2–§10.5) — and §13.1 is built on discrete tokens being the thing flow
  matching *fixed*. Discrete diffusion is a third option that keeps discretization,
  drops the left-to-right order, and folds the head back into the backbone, so it
  falsifies the clean dichotomy both chapters rely on. Added as pending §10.7 in
  Chapter 10. Cross-reference from §13.1 and §10.5 during revision.
- ICLR 2026 VLA submission count now has a better source than the
  semiengineering.com secondary reporting flagged in the prior entry: Moritz Reuss's
  "State of Vision-Language-Action Research at ICLR 2026"
  (mbreuss.github.io/blog_post_iclr_26_vla.html) is a first-hand analysis of the 164
  VLA submissions, with breakdowns by discrete diffusion, reasoning VLAs, and
  benchmark usage (LIBERO / CALVIN / SIMPLER). Use this for §18.5 rather than the
  trade-press number, and it doubles as a "what the field is currently working on"
  snapshot for §18.5's reading list.
- **Tacta Systems / TactaBot (announced 2026-07-27)** — missed by the tactile sweeps.
  Three components: the Tacta Hand (15 independently actuated joints, proprietary
  "fluidic tendon" actuation), Skill Capture (data collection), and a "Dexterous
  Intelligence" model. The sensor spec is the citable part: dynamic range 250 Pa to
  700 kPa, 400 Hz sampling, temperature resolution ~0.1 °C. Deployment stated for
  early 2027 in electronics, AI-infrastructure, and automotive manufacturing.
  (prnewswire 2026-07-27; therobotreport.com; automate.org interview.) Feeds §18.8 —
  it is a *hardware* answer to that section's sensor-heterogeneity point, where
  TouchWorld (arXiv 2607.07287) and FTP-1 (arXiv 2606.13102) are the model-side
  answers — and §14.5 as another dexterity-focused entrant beside GENE, RLDX-1, and
  mimic.
- Runtime monitoring: SAFECAST (arXiv 2608.04246, 2026-08-04) trains hidden-state
  failure probes on *contrast sets* (minimally perturbed near-miss trajectories) and
  calibrates for deployment-time shift, reporting significant ROC-AUC gains over the
  prior SOTA on real DROID and simulated LIBERO across several VLM backbones. Belongs
  in §17.2 beside arXiv 2605.30834's label-noise work and Pre-VLA's preemptive
  framing — the three together give that section a clean progression: reactive →
  label-noise-aware → preemptive. Also not previously tracked: SafeVLA (arXiv
  2503.03480), safety *alignment* via constrained learning, i.e. a training-time
  defense direction distinct from §17.7's inference-time patches.
- Deployment evidence for §14.5, which currently leans on Figure/BMW and
  GR00T-enabled humanoids: Chinese industrial humanoid programs have moved to
  volume, and VLA capability is being reported as the commercial differentiator.
  UBTECH Walker S2 order book above ¥800M from BYD, Foxconn and others, with a
  Liuzhou line targeting 5,000 units/year; AgiBot reportedly past its 15,000th unit
  and a large share of H1 2026 shipments. (techtimes.com 2026-08-26;
  iiot-world.com; merics.org.) All secondary trade reporting with self-reported
  vendor figures — attribute loosely, or use only the qualitative claim.
- §9.7 supporting material: NVIDIA's second WAM explainer, "Pretrained to Imagine,
  Fine-Tuned to Act: The Rise of World-Action Models" (developer.nvidia.com/blog), is
  more useful pedagogically than the "Beyond VLAs" post already tracked, and it names
  Cosmos Policy's actual trick — the pretrained video-diffusion architecture is left
  essentially unchanged and actions, future states, and values are encoded as extra
  latent "frames" in the original diffusion sequence. DreamZero also now has public
  artifacts (dreamzero0.github.io, github.com/dreamzero0/dreamzero) with concrete
  numbers for the section: 14B autoregressive video diffusion running closed-loop at
  7 Hz, >40% improvement on unseen tasks from 10–20 minutes of *video-only*
  demonstrations from other robots or humans, and adaptation to a new embodiment from
  ~30 minutes of play data.
- Citation upgrade: π0.7 now has an arXiv entry — "π0.7: a Steerable Generalist
  Robotic Foundation Model with Emergent Capabilities" (arXiv 2604.15483; blog at
  pi.website/blog/pi07). The 2026-07-23 entry cited it only as "Physical
  Intelligence, Apr 2026". Reported claim worth quoting in §13.4: the single π0.7
  model matches or beats the RL-tuned π*0.6 specialists on tasks like espresso making
  and box assembly.
- Human-video-to-robot bridging: HandEdit (arXiv 2608.12122), a unified benchmark for
  egocentric human-to-robot dexterous hand image editing — i.e. the retargeting step
  that has to happen between a phone video and a robot action. Small, but it is the
  concrete instance of the gap §15.8 is being written around.

## Field-tracking notes (2026-09-15)

Two weeks since the prior entry. All sections remain `[x]`/`[r]` except the three pending
bullets added by recent runs (§10.7, §15.8, §16.7). One new pending section added under
the new-section rule.

- **Reinforcement-learning post-training of VLAs is a whole adaptation regime this log
  has never tracked, and it is now large enough to have its own survey and its own
  frameworks.** The setting: start from a BC/SFT-initialized VLA, then let the policy
  collect its own interaction data and optimize task-level success signal rather than
  log-likelihood of demonstrations. Survey and living bibliography: "A Survey on
  Reinforcement Learning of Vision-Language-Action Models for Robotic Manipulation"
  (github.com/Denghaoyuan123/Awesome-RL-VLA). Infrastructure: RLinf-VLA (arXiv
  2510.06710), a unified efficient RL training framework for VLAs. Methods: RIPT /
  interactive post-training (arXiv 2505.17016), Z-1 efficient RL for VLAs (arXiv
  2606.31846), EXPO-FT sample-efficient RL fine-tuning (arXiv 2605.25477), STARE-VLA
  progressive stage-aware RL (arXiv 2512.05107), OmniVLA-RL (arXiv 2604.17706),
  VLA-Precision (CoRL 2026; asymmetric co-bootstrapping for real-world *online* RL).
  Flow- and reasoning-specific variants: πRL, ReinFlow, VLA-R1. Continual-learning
  adjacent: "Simple Recipe Works: VLAs are Natural Continual Learners with RL" (arXiv
  2603.11653), CRL-VLA (arXiv 2602.03445).
  → **New section warranted, not a parenthetical:** every adaptation mechanism in
  Chapter 16 optimizes a supervised objective — LoRA, full fine-tuning,
  action-head-only, sim-to-real loops, the §16.5 recipe card — and §16.7 (Skild S1) is
  the "don't train at all" alternative. RL post-training is a third regime: you do
  train, but against reward rather than demonstrations, which changes the data
  requirement (interaction, not teleop), the failure modes (reward hacking, catastrophic
  forgetting of the pretrained prior), and the safety story (the robot has to be allowed
  to fail on hardware). Chapter 7 covers deep RL as a standalone family and §13.4
  mentions π0.6's RL-from-experience in passing, but no section covers RL *as a
  post-training step on a pretrained VLA*. Added as pending §16.8 in Chapter 16.
  Cross-reference from §7.4/§7.5, §13.4, and §16.3 during revision.
- **Generalist AI's GEN-1 is entirely absent from this log** despite being a major 2026
  model family. Announced 2026-04-06: an embodied foundation model trained from scratch
  on Generalist's own corpus of ~500,000 hours of real-world data, with the headline
  claim being *mastery* rather than generality — 99% average success on tasks where
  prior models reach 64%, roughly 3× faster task completion, from ~1 hour of robot data
  per task. A follow-up ("Towards Machines with a Thousand Hands") extends GEN-1 across
  end effectors, from five-finger anthropomorphic hands to specialized tools with novel
  actuation, with one base model transferring sensorimotor policies across radically
  different ways of contacting the world. Commercial traction: Elite Robots
  collaboration announced 2026-07-21. (therobotreport.com; siliconangle.com 2026-04-06;
  generalistai.com/blog/gen-1; prnewswire 2026-07-21.) Feeds §14.5 (deployment case
  studies — note this log already tracks "Genesis AI GENE-26.5" as a dexterity entrant;
  Generalist AI is a different company and GEN-1 a different model, worth disambiguating
  in §14.5 and Appendix F), §16.1 (base-model choice), and §11.5 (when scale pays off —
  the "1 hour per task" claim is a strong data point about where the pretraining/
  adaptation split now sits). Self-reported vendor numbers; attribute as such.
- **Safety-alignment *data* is a category §17.1/§17.5 do not cover.** ROBOSHACKLES
  (arXiv 2606.18632) is a 10,000-clip robotic-video safety dataset spanning two
  direct-harm and four indirect-harm categories. The construction story is the
  pedagogically interesting part: real-world data of robots harming humans cannot be
  ethically collected, so the corpus is *synthesized* — starting from real DROID
  observations, through scene understanding, hazard-aware image editing, temporal prompt
  generation, and single-pass video rollout synthesis, with human verification. The
  finding is the alarming part: under a strict refusal-based criterion, all six evaluated
  embodied foundation models produced actions in every safety category — a 100% unsafe
  action generation rate, covering both explicitly harmful commands and benign-looking
  instructions with unsafe downstream consequences. Belongs in §17.1 (safety as a layer)
  and §17.5 (what we cannot certify) during revision; it is the cleanest available
  evidence that today's EFMs have no refusal behavior at all.
- Attack-surface additions for §17.5, beyond the TRAP / VLA-Hijack /
  partially-observable-patch / sensor-perturbation work already tracked: FreezeVLA
  (arXiv 2509.19870), whose contribution is less the attack than the *detectability*
  argument — an action-freezing attack leaves the robot persistently unresponsive in a
  way that reads as normal standby or successful completion, so it slips past standard
  safety monitors, which is a direct problem for the §17.2 monitor designs;
  trajectory-level redirection attacks (arXiv 2606.12978); planner-agnostic attacks on
  tolerance-aware manipulation (arXiv 2607.03758); RoboJailBench (arXiv 2605.19328),
  attacks *and* defenses for embodied robotic agents. Second living bibliography
  alongside Awesome-VLA-Safety: github.com/TheFatInsect/Awesome-Embodied-AI-Safety-Security.
- **The defense side keeps filling in, which further softens §17.7.** Structure-Aware
  Robust Fine-Tuning (arXiv 2608.03231) defends VLAs against physical attention
  hijacking at *training* time — a different category from TrustVLA's inference-time
  patch (tracked 2026-08-31) and from SafeVLA's constrained alignment; and
  "Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models"
  (arXiv 2510.13237) pairs the two directions in one paper. §17.7's claim should now
  read: defenses exist across inference-time, alignment-time, and robust-fine-tuning
  categories, but none address supply-chain provenance and none are evaluated against
  each other on a common benchmark.
- **AGIBOT WORLD 2026 (AgiBot, open-source) — the concrete teleop-fleet dataset §15.2
  predicts.** 13.6 TB of entirely real-world robot data in a LeRobot v2.1-style layout,
  collected on AgiBot's G2 platform, releasing in five phases each aligned to one
  research direction (first phase: imitation learning). What makes it worth a paragraph
  rather than a name-drop: the G2 captures synchronized RGB(D), *tactile* signals, lidar
  point clouds, IMU, and full-body joint states in one pipeline, so it is the first
  fleet-scale corpus where the tactile modality §18.8 is written around arrives free
  alongside the vision data. Also new: RoboMIND 2.0, a multimodal bimanual
  mobile-manipulation dataset (arXiv 2512.24653). Both feed §15.2 and — because of the
  LeRobot layout — §15.3. (therobotreport.com;
  huggingface.co/datasets/agibot-world/AgiBotWorld2026.)
- §9.7 literature has kept consolidating: LaWAM latent world-action models (arXiv
  2606.15768), EWAM for closed-loop online adaptation (arXiv 2606.12690), Kairos, a
  regret-aware native world-action-model stack (arXiv 2606.16533), Feedback World Model
  for guiding diffusion policies (arXiv 2605.15705), and VLA-JEPA (arXiv 2602.10098),
  which takes the opposite route to DreamZero — keep the VLA and *add* a latent world
  model rather than replacing the backbone. VLA-JEPA is the useful foil for §9.7's
  argument, since it concedes the physics-prior point without conceding the architecture.
- Dexterity: **Motus2** (unveiled at the Bund Summit, 2026-09-10; joint with Tsinghua
  University) reports 75% task success on dexterous manipulation by combining a world
  model with action generation *and* a value-assessment head, simulating environmental
  change before execution. Plausibly the same lineage as MotuBrain (arXiv 2604.27792,
  tracked 2026-09-01) — verify before citing the two as separate systems. Single
  secondary source (hyper.ai) with self-reported numbers; use qualitatively in §14.5 or
  §9.7, not as a benchmark claim. Also a venue pointer for §18.8: the IROS 2026 workshop
  "Scalable Tactile Sensing for Dexterous Manipulation" (2026-09-27, Pittsburgh) —
  tactile skins, tactile simulation, visuo-tactile learning, tactile foundation models,
  sim-to-real.
- Whole-body: HAF (arXiv 2608.16837) adapts *generalist* VLAs to humanoid whole-body
  loco-manipulation via hierarchical action flow plus spectral latent RL — the
  adaptation-shaped counterpart to §14.6's train-it-unified approach, and a second data
  point for the new §16.8 (its RL half). Citation upgrade for the 2026-08-01 entry:
  WholeBodyVLA now has an arXiv entry (arXiv 2512.11047, ICLR 2026) with concrete
  numbers — latent actions learned from action-free egocentric video via a Latent Action
  Model plus a loco-manipulation-oriented RL policy, verified on the AgiBot X2 humanoid,
  +21.3% over the prior baseline. That entry cited only the GitHub repo.
- Citation upgrade for §18.4: the ERVLA paper (arXiv 2606.03784, tracked 2026-08-01) has
  quotable corpus figures — 978,743 trajectories, 226.3M samples, 2,592.5 hours,
  described as the largest embodied-CoT corpus to date — and frames embodied CoT as
  *representation-shaping supervision* rather than mandatory test-time reasoning, which
  is a sharper statement of the CoT-dropout idea than the earlier note captured. Related
  and not previously tracked: "Continuous Reasoning for Vision-Language-Action" (arXiv
  2606.00229).
- Survey pointers for §18.5's reading list, all new here: "An Anatomy of
  Vision-Language-Action Models: From Modules to Milestones and Challenges" (arXiv
  2512.11362); "Foundation Models in Robotics: A Comprehensive Review of Methods,
  Models, Datasets, Challenges and Future Research Directions" (arXiv 2604.15395); and a
  domain-extension review covering UAVs and bimanual manipulation (arXiv 2607.06706),
  useful precisely because this book is manipulation-only and should say so explicitly.

## Field-tracking notes (2026-10-01)

Two weeks since the prior entry. Pending bullets are still §10.7, §15.8, §16.7, §16.8.
No new section added this run: every finding below fits an existing bullet, and the two
that land on pending sections (§15.8, §16.7) are handled as parenthetical revisions.

- **Figure Helix 2.5 (figure.ai/news/helix-2-5-zero-shot-30-home-generalization,
  2026-09-17) is the first controlled ablation of whether Index pays off.** Three household
  tasks (tidy 13–15 toys, fold towels, make beds) evaluated zero-shot in 30 Bay Area homes
  with no data collected in any of them. Two policies trained on identical task data, one
  from random init and one from the Index-pretrained checkpoint, everything else fixed:
  full-task success 9% → 56%, no partial credit. Also uses half the task-specific data of a
  representative Helix-02 behavior. Self-reported, but the experimental design is the kind
  §15.8 needs, so it goes into that bullet. Also feeds §14.2/§14.5 during revision.
- **GEN-1.5 (Generalist AI, generalistai.com/blog/gen-1.5, Sept 2026): "Embodied
  Foundation Models are One-Shot Learners."** 3–12 s of sensorimotor demonstration in a 30 s
  context, no gradient updates: 59% average across 10 short tasks (jars, zippers, purses),
  83% after ten gradient steps on five minutes of data. Two prompts in context chain into a
  longer behavior. Generalist says no meta-learning objective was used; the capability is
  attributed to pretraining scale. Unlike Skild S1, the prompt is robot sensorimotor data,
  not a human video. That makes it a second vendor making the in-context claim, which §16.7
  previously lacked. Same-month commentary worth citing in §16.7: Turing Post (2026-09-21)
  notes that S1's 66% is cumulative per-step success with human interventions to recover
  failures, *not* a fully autonomous completion rate.
- SAIL (Sakana AI + U. Tokyo, arXiv 2603.08269, presented at IROS 2026; sakana.ai/sail):
  test-time scaling for in-context imitation. MCTS over complete trajectories generated by a
  frozen VLM, scored by simulating them and reading subtask completion from video. Going
  from 1 to 45 candidates raises success from 25% to 73% across six sim tasks, and 5/6 on a
  real SO-101 arm. A third, search-based point in §16.7's no-training corner.
- **Runway Praxis-1 (runway.com/research/introducing-praxis-1, Sept 2026)**: an
  *open-weight* world action model built on Runway's video pretraining. In early partner
  testing (Noble Machines bimanual, Standard Bots 6-DoF RO1, Ultra mobile base), the same
  policy moves across embodiments and environments without retraining; public release is
  promised "in the coming months." The citable figure: web-video pretraining matches
  teleop-robot-video pretraining on post-finetune placement error (16.1 vs 16.0 cm, ±1 SEM
  over 93 pairs). Runway also reports 0.95 correlation between world-model policy simulation
  and real outcomes. For §9.7 during revision it is the first open-weight WAM from a
  video-generation company rather than a robotics lab, and the policy-evaluation claim
  belongs next to GigaWorld-1 in §15.5/§15.6.
- Light-O1 (Light Origins, prnewswire 2026-09-23; github.com/lightorigins/Light-O1):
  pretrains a human action prior from structured human motion recovered from internet video,
  then adapts it to robots. Six runs of a 4B base from 3.75B to 120B multimodal tokens (the
  largest is ~100k hours of human action) give power-law error decline across three
  downstream datasets. Open-sourced Light-O1-Preview, a ~6B reasoning text-to-whole-body-
  action model. A second human-video scaling-law claim beside DYNA-2; it goes into §15.8.
- WAM consolidation for §9.7: a proper survey now exists. "World-Action Models for Robot
  Learning and Control: A Survey" (arXiv 2609.16074, 2026-09-13; Laptev, Danfei Xu, Eric Xing
  et al.; rcl-robotics.github.io/Awesome-World-Action-Models) gives a taxonomy covering
  representations, transition modeling, action interfaces, and training pipelines. Also new:
  OpenWAM, open modular WAM pretraining (arXiv 2609.07398); WLA³ world latent action
  modeling (arXiv 2609.15870); WholeBodyWAM (CoRL 2026), WAM priors for humanoid
  loco-manipulation, which links §9.7 to §14.6.
- Safety: "Backdoors in Learning-Based Industrial Robotic Arm Manipulation: An Empirical
  Security Study" (arXiv 2609.26868) runs VLA backdoors on two *commercial* industrial arms
  (FANUC, xArm) and adds an online runtime trigger-detection/neutralization pipeline. It is
  real-hardware evidence for §17.7 and another inference-time defense beside TrustVLA. Also
  SpecGuard, inference-time backdoor detection (arXiv 2609.11799); check whether it was
  evaluated on VLAs before citing it in §17.7. Humanoid safe stopping: Safe-Stop
  (junfeng-long.github.io/safestop) learns whether a moving humanoid can still reach a stable
  stance and otherwise falls back to damping. Reports 96.4% over 179,650 Unitree G1 sim
  episodes; for §17.1/§17.4.
- Benchmarks for §15.4/§15.6: RoboSPA, long-horizon complex-scene VLA evaluation (arXiv
  2609.05324); a graph-based autonomous reset harness for long-horizon real-robot
  evaluation (arXiv 2609.19413). The second goes straight at §15.5's cost-of-evaluation
  problem.
- Contact-rich: Facet-0 (pine-lab-ntu.github.io/facet-0) conditions on wrist
  force/torque as well as images and language, predicts actions and their contact effects,
  then improves from real-world attempts. It reports 82% vs 15% on five computer-assembly
  tasks. Force/torque, not tactile, is the cheaper non-visual modality; it is worth a
  sentence in §18.8 and is a candidate example for §16.8's real-world RL loop.
- Industry context only, not for chapter content: OpenAI confirmed plans to build humanoid
  and other robots (Forbes, 2026-09-03), and Agility unveiled Digit 5 with an independent
  safety controller (agilityrobotics.com).

## Field-tracking notes (2026-10-15)

Two weeks since the prior entry. Pending bullets are still §10.7, §15.8, §16.7, §16.8. No new
section this run. Every finding fits an existing bullet, and the one that lands on a pending
section (§15.8) goes in as a parenthetical.

- **FLUX 3 Action (Black Forest Labs, 2026-09-23/24; bfl.ai/models/flux-3-action;
  MarkTechPost 2026-09-24; The Batch).** A 7B open-weight world action model and the first
  open release from the FLUX 3 multimodal backbone. Inputs are camera frames, robot state, and
  text; each call returns 32 actions and can optionally return 32 predicted video frames. BFL
  reports 42.92% on NVIDIA's RoboLab-120 against 36.8% for Cosmos3-Nano-Policy and 28.0% for
  π0.5, plus 41 ms inference on a B200 and fine-tuning from about 200 teleop episodes, with
  LeRobot integration. Caveats: BFL supplied the score before NVIDIA's leaderboard listed it,
  the headline number comes from a DROID-fine-tuned checkpoint, and commercial license terms
  are still unclear (FLUX Kommunity License). For §9.7 during revision: after Runway Praxis-1,
  this is the second WAM from an image/video-generation company, and the first whose weights
  are actually out. It is also the first WAM whose public RoboLab comparison runs directly
  against NVIDIA's own Cosmos policy.
- **Natura-1 (Nature Will, 2026-10-13; prnewswire; Forbes/Koetsier 2026-10-13).** A
  hands-only 50B mixture-of-experts model trained jointly on language, images, video, and
  action. The company reports pretraining on more than 1M hours of egocentric human video, a
  100 ms control loop, and zero-shot instruction following on unseen objects. All claims are
  self-reported with no independent benchmark yet. It joins GENE-26.5, RLDX-1, GEN-1, and
  TactaBot in §14.5's dexterity list. It is also a third million-hour-scale human-video
  pretraining claim beside DYNA-2 and Light-O1, so it goes into the pending §15.8 bullet.
- **The safety arms race moved in the same week, one attack and one defense.** "Black-Box
  Adversarial Patch Attacks on VLAs via Ancestor VLM Exploitation" (arXiv 2610.09708,
  2026-10-07) optimizes patches on the *publicly released VLM a VLA was adapted from*, with no
  access to the target VLA, and they transfer. The interesting negative result is that the
  attack nearly vanishes on policies whose adaptation rewrites the shared visual-semantic
  representation. That makes it a supply-chain-shaped threat, and it connects §17.5 to §17.7's
  provenance point. "Detect and Suppress" (arXiv 2610.03498, 2026-10-02) is a mechanistic
  inference-time defense. A sparse-autoencoder feature correlated with patch presence feeds a
  linear-probe detector and is then suppressed without fine-tuning. On LIBERO-10, conditional
  intervention helps, but always-on suppression substantially hurts the clean policy. That is
  a usable worked example of the monitor-cost trade-off in §17.2.
- Evaluation for §15.4/§17.3: JitterVLA (arXiv 2610.13628, 2026-10-09) is a replay benchmark
  for *motion quality* rather than task success. Its argument is that VLAs send sparse,
  late, irregular targets, so a plausible action can still produce abrupt motion near people.
  No suite in §15.4 currently measures this.
- WAM preprints continue to pile up for §9.7: ProWAM, which jointly predicts actions and
  ordered sparse visual sub-goals, 70.0% zero-shot real-world (arXiv 2610.02508); XGenAct,
  geometry-enhanced WAMs predicting future depth and segmentation (arXiv 2610.03516); a
  humanoid WAM with joint state–action generation (arXiv 2610.12026), which adds to the
  §14.6 link; and "Rethinking World-Action Model for Compositional and In-Context Robotic
  Manipulation" (arXiv 2610.02368). I only saw the last one's title. Read it before deciding
  whether it belongs in §16.7 as well.
- Embodied-reasoning gap fill for §18.4: CoTinyVLA (arXiv 2607.25487) distills CoT into a
  sub-billion-parameter VLA. That is the small-model end of the ERVLA/ZR-0 "train with CoT,
  skip it at inference" thread.
- Not logged, could not verify: a "Gemini 4 Argon for Robotics" Google Developers Blog post
  dated 2026-10-13 showed up in one search, but a targeted follow-up found nothing. Check
  deepmind.google and developers.googleblog.com before the next run cites it.

## Field-tracking notes (2026-11-01)

Two and a half weeks since the prior entry. Pending bullets are still §10.7, §15.8, §16.7, §16.8.
No new section this run. One finding lands on a pending section (§16.8) and goes in as a
parenthetical. The rest belong to drafted chapters and wait for revision.

- **Gemini 4 Argon is now verified, which resolves the 2026-10-15 open item.** Google's launch
  post (blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-argon)
  went out 2026-10-13, and AndroidHeadlines (2026-10-14) and Interesting Engineering covered it.
  The robotics claims are all Google-reported: 87.0% on ERQA, and a real2sim pipeline that
  watches a raw human video and writes the simulation code itself. In the demo, the model
  transferred a person's cornhole throws to Apptronik Apollo, which sank seven in a row. Google
  also says sim data generated by Argon raised a real robot's task success from 56% to 75%. The
  German coverage (mixed-news.com) adds a pick-and-place time cut from 58.9 s to 17.4 s. For
  §14.3/§14.6 during revision, this is the ER side of the Gemini line moving to a general
  frontier model. The video-to-sim step also belongs next to Index and DYNA-2 when §15.8 gets
  drafted, as a different way to turn human video into robot training signal.
- **Skild AI, "Physical Self-Play" (skild.ai/blogs/physical-self-play, 2026-09-23; Robotics &
  Automation News 2026-10-16).** RL post-training of the S1 foundation model through self-play
  in Isaac Sim, about 140 simulated years, with a single reward: score a goal. The policy then
  transfers to a humanoid. Skild says dribbling and tackling emerged without shaped rewards, and
  that early 4-agent teams pass. Caveats: no win rate, baseline, or benchmark table, and Skild
  calls the result preliminary. It is the clearest vendor example yet of the §16.8 regime
  (adapt a pretrained VLA against reward, not demonstrations), so it goes into that bullet.
- **NEURA Robotics Neuron AI (Business Wire / Robotics & Automation News, 2026-10-14).** NEURA
  calls it a "Perception-Language-Action" model. It fuses 3D vision, touch, audio, robot state,
  and language, keeps memory of objects after they leave the camera view, and runs on-robot
  across five NEURA embodiments, including 4NE1 Gen 3/4. NEURA showed an eleven-minute laundry
  sequence at GTC Berlin. There are no benchmarks yet. For §18.8 during revision, it is a
  commercial model treating touch and audio as first-class inputs. Memory is also the axis
  where AR-VLA (INSAIT, RSS 2026; an autoregressive action expert with history) gives the
  academic counterpart.
- Runtime monitoring for §17.2: **Progress Probes P²** (arXiv 2610.31613). Linear probes on a
  VLA's internal features flag deviation from expected task progress. They train on successful
  demos only, with no failure data and no external VLM, and the authors report detection at or
  above Robometer and SAFE at lower cost. Gap fill from July that earlier runs missed: ActFovea
  (arXiv 2607.29169) checks spatiotemporal consistency between visual motion and
  proprioception/action, and only accepts a recovery chunk after verifying it.
- Safety tooling, industry context: NVIDIA's Open Agent Safety Platform
  (investor.nvidia.com, 2026-09-28) pairs the OpenShell runtime boundary with the Sentry
  hardware monitor on BlueField-4. Figure, Skild, and Gecko Robotics are named as exploring it
  for physical systems. Worth one sentence in §17.1's "safety as a layer" framing. It is not a
  certified safety system.
- Infrastructure: Isaac Lab 3.0 went GA around GTC Berlin (2026-10-21), with a multi-backend
  (Newton/PhysX) architecture per docs.nvidia.com/isaaclab/3.0. This is an Appendix D note. An
  NVIDIA human-video dataset was reported alongside it, but I found only an aggregator source,
  so it is not logged.
- Industry context only: Altman confirmed an OpenAI humanoid demo for 2027 (TechTimes
  2026-10-19), which updates the 2026-10-01 note. Agile Robots showed Agile ONE SN, a compact
  research humanoid, at GTC Berlin (Coffee & Robots roundup, 2026-10-29).
- Searched, nothing new: Physical Intelligence (latest is still π0.7), Figure, 1X, and
  Generalist had no new model releases after 2026-10-15. I also found no new dexterity-model
  release beyond Natura-1, which is already logged.

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
- [ ] 10.7 Discrete diffusion for action: the third action-head class — keeping discretized action chunks but dropping left-to-right order, and folding the denoiser back inside the VLM backbone instead of attaching a head outside it; adaptive decoding order and re-masking as error correction, why parallel decoding cuts function evaluations from sequence-length to a handful of refinement rounds, and what this does to the discrete-vs-continuous dichotomy §10.2–§10.5 and §13.1 are built on (Discrete Diffusion VLA, Unified Diffusion VLA, Fast-dVLA, dVLA-RL). (Added 2026-09-01; placed after 10.x to avoid renumbering the existing Summary/exercise bullets.)

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
- [ ] 15.8 Paying the public for data: crowdsourced human video as a third acquisition regime — Figure's Index app, the throughput and diversity numbers it reports, and the pipeline problems that only appear when contributors are paid strangers (quality filtering, fraud review, embedding deduplication, task rebalancing, captioning); why diversity has to be defended rather than assumed, and what still has to happen between a phone video and a robot action (read against DYNA-2's claimed 1k→1M-hour scaling law on human egocentric video, which is the strongest evidence to date that this regime pays off; HandEdit, arXiv 2608.12122, is the concrete benchmark for the human-hand-to-robot-hand retargeting step in that gap) (update 2026-10-01: Helix 2.5 supplies the first controlled ablation of Index itself, with zero-shot full-task success going from 9% to 56% across 30 unseen homes when only the pretraining init changes; Light-O1 is a second human-video power-law scaling claim, ~100k hours of human action recovered from internet video) (update 2026-10-15: Nature Will's Natura-1 is a third such claim, a hands-only 50B MoE pretrained on >1M hours of egocentric human video, all figures self-reported). (Added 2026-08-31; placed after 15.x to avoid renumbering the existing Summary/exercise bullets.)

### Part 5 — Building with action models

#### Chapter 16. Fine-tuning a VLA for your robot

- [x] 16.1 Picking a base model
- [x] 16.2 Building a teleop dataset that does not waste your time
- [x] 16.3 LoRA vs. full fine-tuning vs. action-head-only
- [x] 16.4 Sim-to-real fine-tuning loops
- [x] 16.5 A recipe card for new embodiments
- [x] 16.6 Summary
- [x] 16.x Hands-on exercise + chapter references
- [ ] 16.7 Adapting without training: in-context learning for robots — Skild's S1 claim that a single human video in the prompt replaces post-training, what "omni-bodied" pretraining across teleop + human video + simulation + glove data buys, how this reframes the §16.1 base-model choice and the §16.5 recipe card, and the caveats a textbook has to state (self-reported numbers, no public weights, one vendor) — the "don't train at all" corner of the adaptation space, to be read against §16.8's "train, but against reward" (update 2026-10-01: Generalist's GEN-1.5 is a second vendor with the in-context claim, prompting with 3–12 s of robot sensorimotor data rather than human video and getting 59% with no updates vs. 83% after ten gradient steps, a useful prompt-vs-fine-tune comparison point; Sakana's SAIL, arXiv 2603.08269, is a search-based variant using test-time MCTS over a frozen VLM; and the caveat that S1's 66% is per-step success with human recovery interventions, not autonomous completion). (Added 2026-09-01; placed after 16.x to avoid renumbering the existing Summary/exercise bullets.)
- [ ] 16.8 Adapting with reward: RL post-training of a pretrained VLA — the third adaptation regime beside supervised fine-tuning (§16.3) and prompting (§16.7), where the policy collects its own interaction data and optimizes task success instead of demonstration likelihood; what changes relative to Chapter 7's from-scratch RL (you are protecting a pretrained prior, not building one), the practical machinery (RLinf-VLA as the framework layer, RIPT/Z-1/EXPO-FT/STARE-VLA/VLA-Precision as the method layer, πRL and ReinFlow for flow-matching heads), the failure modes a reader will actually hit (reward hacking, catastrophic forgetting of pretrained generality, sim-versus-hardware rollout cost), and the safety precondition nobody mentions — the robot has to be allowed to fail on real hardware; closes the loop with §13.4's π0.6 RL-from-experience note and §10.7's dVLA-RL (update 2026-11-01: Skild's "Physical Self-Play" is a vendor-scale example of this regime, self-play RL on the S1 foundation model in Isaac Sim with a single score-a-goal reward, about 140 simulated years, transferred to a humanoid; it reports no win rate or baseline). (Added 2026-09-15; placed after 16.7 to avoid renumbering the existing Summary/exercise bullets.)

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
- 2026-08-31 — field check: appended Field-tracking notes (2026-08-31); added pending §15.8 (crowdsourced human video / Figure Index) under the new-section rule. No chapter content touched.
- 2026-09-01 — field check: appended Field-tracking notes (2026-09-01); added pending §16.7 (in-context learning / Skild S1) under the new-section rule; lightly revised the pending §15.8 bullet to note DYNA-2. No chapter content touched.
- 2026-09-01 — field check (second run same day): appended Field-tracking notes (2026-09-01b); added pending §10.7 (discrete diffusion action heads) under the new-section rule; lightly revised the pending §15.8 bullet to note HandEdit. No chapter content touched.
- 2026-09-15 — field check: appended Field-tracking notes (2026-09-15); added pending §16.8 (RL post-training of a pretrained VLA) under the new-section rule; lightly revised the pending §16.7 bullet to point at §16.8 as the reward-based counterpart. No chapter content touched.
- 2026-10-15 — field check: appended Field-tracking notes (2026-10-15); no new section; lightly revised the pending §15.8 bullet to note Natura-1. No chapter content touched.
- 2026-08-04 — No pending [ ] sections found. Book remains fully drafted (all 108 sections + appendices A–F at [x]). No drafting action taken. Next step is revision ([r]) passes by Pavan; the daily draft task has no remaining first-draft work.
- 2026-11-06 — errata pass (11 entries from the FMR video-script fact checks): §1.1 ImageNet 2009 size (3.2M, later ~14M), dropped "three orders of magnitude", Open X-Embodiment dated Oct 2023; §1.4 RT-1 (130k demos, 700+ instructions), ACT chunk k≈100 vs Diffusion Policy 8–16, Spot removed from RL gait examples; §2.1/§2.3 "bottom 256" → last 256 tokens (IDs 31744–31999), 21 institutions, 32,000 tokenizer vs 32,064 padded rows; §2.4 reworded-prompt and `bridge_orig` failures rewritten from our 6 Nov run (purposeful wrong motion; ~21× too small). Same errors also fixed downstream in §1.2, §2.3 diagnostics, §2.5, §2.6, §2.x.
