---
chapter: 15
section: 15.7
title: Summary
target_words: 2000
status: draft
prereqs: §15.1–§15.6; what a single robot episode contains at the level of arrays, how Open X-Embodiment pools sixty datasets and what harmonization costs, LeRobot as the second storage lineage aimed at the person with one cheap arm, what the sim benchmarks (LIBERO, CALVIN, RoboCasa, SimplerEnv) each measure and hide, why a real-robot success rate is an estimate with a confidence interval rather than a score, and the build procedure for an evaluation nobody has benchmarked yet
key_refs:
  - Open X-Embodiment Collaboration, Padalkar, A. et al. (2023). Open X-Embodiment, Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Walke, H. et al. (2023). BridgeData V2, A Dataset for Robot Learning at Scale. CoRL 2023.
  - Cadene, R., Alibert, S., Soare, A. et al. (2024). LeRobot, State-of-the-art machine learning for real-world robotics in PyTorch. github.com/huggingface/lerobot.
  - Liu, B. et al. (2023). LIBERO, Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS Datasets and Benchmarks.
  - Atreya, P., Pertsch, K. et al. (2025). RoboArena, Distributed Real-World Evaluation of Generalist Robot Policies.
---

# 15.7  Summary

Part 4 spent six chapters treating data and evaluation as background: OpenVLA
trained on 970,000 trajectories, Octo on 800,000, and the numbers that ranked one
VLA above another came pre-computed from some benchmark you were asked to trust.
Chapter 15 stopped taking either for granted. It opened the pile of trajectories to
see what one episode actually is, followed the corpus up from a single WidowX arm
to sixty pooled datasets, and then asked the question every one of those model
comparisons quietly assumed had an answer: when a paper says a policy "works," what
was measured, and would it survive someone else running it? §15.1 took an episode
apart array by array, §15.2 scaled that to Open X-Embodiment and its fleet-scale
successors, §15.3 set LeRobot beside it as the second storage lineage, §15.4 walked
the sim benchmarks the field quotes, §15.5 turned real-robot success into a
statistics problem, and §15.6 turned all of it into a procedure you can run on a
robot no leaderboard covers. The chapter's job was to make sure that when Part 5
fine-tunes a model and asks whether it improved, you know what an honest answer to
that question costs.

## The ideas worth carrying forward

*A robot episode is a list of timestamped dictionaries, and the anatomy is the same
everywhere even when the conventions are not.* §15.1 used one BridgeData V2 episode
(Walke et al., 2023) because it is small enough to inspect by hand. Each timestep
carries an observation that is more than an image, an action written in some lab's
private convention of units and frames, a language instruction that arrived by a
route worth knowing, and a scalar or two of proprioception. The lesson that carries
is that the shape is universal and the semantics are not: two datasets can both
store a seven-vector action where one means end-effector deltas in meters and the
other means joint velocities in radians per second, and nothing in the file tells
you which. That gap is the seed of every problem in §15.2, and it is why the
episode format you log your own evaluations in should match this shape, so your
eval data is reusable as training data later.

*Open X-Embodiment is plumbing, not a dataset, and the plumbing is the hard part.*
§15.2 scaled one pile of dictionaries up to sixty. OXE (arXiv:2310.08864) pooled
roughly sixty datasets from twenty-one institutions, covering twenty-two robot
embodiments and more than a million real trajectories, and its contribution was
less the data than the harmonization: re-serializing everyone's incompatible
conventions into one format a single policy could train across. That harmonization
is exactly the MDP-to-robot translation problem from §5.5 wearing work clothes,
and it is imperfect on purpose, because forcing every embodiment into one action
space throws away detail no matter how you do it. What has succeeded OXE is not a
bigger branded pool but large teleoperation fleets, AGIBot's Genie-1 collection,
bimanual YAM rigs, Unitree G1 data, feeding newer foundation models directly, and
LingBot-VLA 2.0's one-policy-across-twenty-morphologies result is the clearest sign
that the pooled corpus was a phase, not an endpoint.

*LeRobot is the same episode with a different address and a different audience.*
§15.3 drew the contrast that matters. Where OXE grew out of TensorFlow, RLDS, and
academic labs comfortable in that ecosystem, LeRobot (Cadene et al., 2024) is
PyTorch-first and built for the person with one low-cost arm who wants to record a
hundred episodes this afternoon and train tonight. It bundles a dataset format, a
hosting hub, pretrained policies you load in two lines, and driver support for
hardware you can build from a parts list. The delta-timestamp query trick, asking
for a window of past and future frames by time rather than index, is the piece of
engineering worth remembering, because it is what lets the same on-disk episode
feed a diffusion head that wants a sixteen-step action horizon and a single-step
policy that wants one. The community-dataset count going from a handful to several
thousand in about a year is the evidence that lowering friction moved more than any
architecture did.

*A sim benchmark is a fixed world sold as a fair comparison, and each one trades
honesty for reproducibility differently.* §15.4 walked four. LIBERO
(arXiv:2306.03310) is the fine-tuning yardstick, organized along axes of object,
spatial, goal, and long-horizon variation that are worth stealing even if you never
run it. CALVIN pushes on long horizons and language chaining; RoboCasa buys scale
through generated assets; SimplerEnv exists for one purpose, to make a sim number
predict the real number, by rebuilding real evaluation scenes closely enough that
the ranking transfers. The through-line is that reproducibility and realism pull
against each other, and every benchmark picks a point on that line and hides the
cost of its choice in the tasks it happens not to include. RoboArena (Atreya,
Pertsch et al., 2025) is the reaction: give up on sim, run distributed
preference-scored evaluation across many labs on real hardware, and accept noise as
the price of credibility.

*A real-robot success rate is a sample estimate of a probability you cannot
observe, and reporting it without an interval is the most common quiet lie in the
literature.* §15.5 was the statistics chapter. You run a policy N times, it
succeeds k times, and k/N is not the policy's success rate; it is one draw from a
binomial whose spread is wide at the sample sizes robotics can afford. Twenty
trials at 15 successes is consistent with a true rate anywhere from about 53% to
88% at 95% confidence, which is why a Wilson interval belongs on every reported
number and why two policies that differ by one or two successes out of twenty
differ by nothing you can defend. The variance is real and physical, drifting light
and grease and a two-centimeter shift in object pose, and the fixes are procedural:
interleave policies trial by trial, log failure modes from a fixed vocabulary
rather than a bare boolean, and report time-to-completion as a median with spread
rather than a mean that one timeout destroys.

## What you should be able to do now

Four things, in the order Part 5 uses them.

You should be able to *read a dataset off disk and know what every field means and
does not mean*. Given an unfamiliar episode, you can name the observation, action,
instruction, and proprioception channels, and you know to check the action
convention before trusting it, because a seven-vector that looks like end-effector
deltas might be joint velocities and the file will not warn you. This is the
inspection §15.1 walked, and it is what keeps you from feeding a policy actions in
the wrong units and blaming the architecture.

You should be able to *explain what pooling robot data costs, and what replaced
pooling*. You can say why Open X-Embodiment's real contribution was harmonization
rather than collection, why that harmonization loses embodiment-specific detail,
and why the field moved from one branded corpus toward large teleop fleets and
one-policy-across-many-morphologies results. You can also place LeRobot as the
second lineage and say who each format is actually for.

You should be able to *tell an honest sim benchmark result from a misleading one*.
You know what LIBERO, CALVIN, RoboCasa, and SimplerEnv each measure and each ignore,
you know that a sim number is a screen and not a verdict because of the sim-to-real
gap from §7.5, and you know why RoboArena gave up on sim to buy credibility. Handed
a table of benchmark scores, you can say which comparisons the benchmark actually
supports.

You should be able to *design and run an evaluation for a policy nobody has
benchmarked*. You start from the decision sentence the evaluation must let you
finish, choose tasks that spread across difficulty rather than clustering at the
ceiling, fix the success criterion and initial-state distribution and trial count
in writing before the first rollout, log each trial as a full episode with a failure
mode, screen in sim for volume, and confirm on hardware for truth. This is the
§15.6 procedure, and it is the deliverable Chapter 16 will lean on the moment it
asks whether fine-tuning helped.

## Where the chapter has set up the rest of the book

Chapter 15 is the hinge into Part 5, and it hands forward a specific pairing. The
whole point of Chapter 16 is to fine-tune a base VLA onto your own robot, and every
step of that depends on this chapter: you need to build a teleop dataset in a format
(§15.1, §15.3) that a training loop can consume, and you need an evaluation (§15.5,
§15.6) honest enough to tell you whether the fine-tune moved the number or just
moved the noise. §16 will say "pick a base model and collect data that does not
waste your time," and the reason that sentence is not empty is that this chapter
told you what good data and a defensible measurement look like.

The safety thread runs into Chapter 17. §15.6's rule about including a task the
policy should refuse, and §15.5's insistence on logging failure modes rather than a
boolean, are the seeds of treating safety as its own evaluation axis rather than a
footnote to success rate. When §17.3 builds A/B evaluation on hardware, it is
running the interleaved, blinded protocol §15.5 argued for, on higher stakes. When
§17.4 builds logging and rollback, it is extending the episode-logging discipline
from §15.6 into a production system that watches a running robot instead of a test
harness.

One thread points at data collection specifically. §15.6's move, logging every
evaluation rollout as a full episode so a failure becomes replayable and a hard task
becomes a candidate for the next collection session, turns evaluation into a
failure-mining pipeline that feeds §16.2's data-collection loop. Evaluation and data
collection are not two activities in this book; they are one loop, and the rollouts
where your policy is weakest are the most valuable data you will ever record.

## What the chapter has not covered

Two omissions worth naming. The chapter treated evaluation almost entirely as
measuring task success, and said little about evaluating the properties that do not
reduce to a success rate: how smooth the motion is, how hard the policy is on the
hardware, whether it fails safely or violently, how it degrades as the scene drifts
from the training distribution. Some of those get picked up as safety in Chapter 17,
but the general problem of scoring a policy on axes other than "did it finish" is
one the field has barely standardized, and this book does not pretend otherwise.

The chapter also stayed on manipulation and mostly on single-arm tabletop tasks,
which is where the datasets and benchmarks are richest. It touched bimanual and
whole-body data through the teleop-fleet successors in §15.2 but did not develop how
you evaluate a loco-manipulation policy, where a "failure" can mean the robot fell
over rather than missed a grasp, and where the statistics of §15.5 have to share a
page with the real-time jitter budget of §14.4. That combined evaluation problem is
still an open one, and §18 returns to why the humanoid case makes it harder.

Chapter 15's contribution to the book's argument is to put a floor under every
number in Part 4 and every claim in Part 5. A model comparison is only as good as
the dataset conventions underneath it and the evaluation protocol on top of it, and
this chapter showed both: what a trajectory contains and where its conventions can
mislead, how the field pools and formats data and what that costs, what the sim
benchmarks measure and hide, and why a real success rate is an estimate that needs
a confidence interval before it means anything. With data and evaluation understood
at this depth, Part 5 can build, because now you can tell whether what you built
actually works.

§15.x closes the chapter with a hands-on exercise, reproducing an OpenVLA
evaluation in SimplerEnv across five seeds and computing a confidence interval over
the runs, so the gap between one sim number and a defensible one becomes something
you measure rather than something you take on trust, followed by the chapter's full
reading list.
