---
chapter: 14
section: 14.4
title: "Latency budgets and real-time control"
target_words: 2000
status: draft
prereqs: §14.1 (the two-clocks argument and why balance and contact set hard deadlines), §14.2 (Helix's 7–9 Hz System 2 and 200 Hz System 1, and the thin latent channel between them), §14.3 (GR00T's diffusion action head and wide token channel, which change where the time goes), §13.2 (π0's chunking, one backbone pass amortized over a one-second chunk).
key_refs:
  - Figure AI (2025). Helix, A Vision-Language-Action Model for Generalist Humanoid Control. figure.ai/news/helix.
  - Bjorck, J. et al. (2025). GR00T N1, An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 14.4  Latency budgets and real-time control

The last three sections argued that a dual-system design earns its complexity by letting reasoning and reacting run on different clocks. That argument only pays off if the fast clock actually keeps time. A design that splits into two networks and then misses its control deadline has not solved the latency problem; it has hidden it inside a more complicated failure. This section is about the accounting that decides whether the split works: where the milliseconds go, what "real-time" means when the ground truth is a robot that can fall, and the tricks that buy a fast loop its deadline without starving the slow one.

## What "real-time" actually demands

Start with the word, because it gets used loosely. Real-time control does not mean fast on average. It means a guarantee: every control cycle produces an action within a fixed deadline, and the guarantee holds on the worst cycle, not the median one. A loop that runs at 200 Hz has 5 milliseconds per cycle, and if one cycle in a thousand takes 40 milliseconds, that one cycle is the whole story for a balancing humanoid. Average throughput is a benchmark number. Worst-case latency is what keeps the robot upright.

Where does the deadline come from? From the physics, not the software. Section 14.1 gave the numbers that matter: a bipedal robot is an unstable system whose balance can go from fine to falling in under a hundred milliseconds, and the reaction window when a grasp slips is on the order of tens of milliseconds. Those are not targets someone chose. They are properties of the mechanism, the way a pendulum's fall time is a property of its length. The control loop has to close faster than the fastest disturbance it needs to reject, and everything else in the latency budget is negotiable against that fixed ceiling.

This is why the field talks about a budget at all. You have a fixed amount of time per cycle, set by physics, and a list of things that have to happen inside it: read the sensors, run the fast network, send the command to the motors. Add them up. If the sum fits under the deadline with margin to spare, the loop is real-time. If it does not, no amount of clever architecture upstream will save you, because the body will have moved on before the action arrives.

## Tracing the milliseconds through a dual-system stack

Take Helix's shape from §14.2 and walk a single action from photons to torque. There are two loops running at once, so trace them separately.

The fast loop is System 1, the 80M-parameter transformer at 200 Hz. Its per-cycle budget is 5 milliseconds, and into that window it has to fit: grabbing the latest camera frame and proprioception, one forward pass of the small network, and pushing the resulting joint targets out to the motor controllers. An 80M transformer on an embedded GPU runs a forward pass in a couple of milliseconds if the model is well optimized, which leaves a thin margin for the sensor read and the actuator write. That margin is the entire reason System 1 is 80M parameters and not 800M. The size was chosen backward from the deadline.

The slow loop is System 2, the 7B VLM at 7 to 9 Hz. Its budget is generous by comparison, roughly 110 to 140 milliseconds per cycle. A 7B model reading two camera streams and an instruction, then producing a latent vector, spends most of that budget on the forward pass itself, which is fine, because the semantic decision it produces stays valid across many System-1 cycles. The slow loop can afford to be slow precisely because its output ages slowly.

Now notice the thing that makes the dual-system design work and also makes it subtle: the two loops are not synchronized. System 1 does not wait for System 2. It runs every 5 milliseconds against whatever the most recent latent vector happens to be, and System 2 refreshes that vector whenever its own slower cycle completes. Between refreshes, roughly 20 to 40 fast cycles reuse the same standing intent. The latent sitting in the buffer is always a little stale, by up to one System-2 period, and the design bets that staleness is harmless because semantic intent does not change on a 5-millisecond timescale. That bet is the same one π0's chunking made in §13.2, moved from inside one network to the seam between two.

## Chunking and asynchronous inference

The staleness bet is worth dwelling on, because it is the general trick that lets slow models drive fast robots, and it shows up in three forms across the models this book has covered.

π0 (arXiv:2410.24164) uses action chunking. One heavy backbone pass produces a whole one-second chunk of actions, and the robot plays that chunk out at high rate while the backbone is busy computing the next one. The backbone runs slowly; the action stream is fast because it is pre-computed in batches. The cost, from §14.1, is that the chunk is open-loop within itself: the fast corrections inside that second are replayed from a decision made at the top of the chunk, not computed against sensor data arriving mid-chunk.

The dual-system models fix that open-loop gap by putting a real closed loop on the fast clock. System 1 is not replaying a pre-computed chunk; it is genuinely reacting to fresh sensors every 5 milliseconds, using the slow latent only as a standing goal. This is more expensive than chunking, because you pay for a live fast network instead of a cheap buffer readout, and it is the reason the design is reserved for tasks dynamic enough to need it.

The third form is asynchronous execution, and it is the piece that keeps the whole thing from stalling. If System 1 ever blocked waiting for System 2 to finish a forward pass, the fast loop would inherit the slow loop's latency, and the split would buy nothing. So the systems run on separate hardware, in Helix's case two onboard GPUs, and communicate through a shared buffer rather than a blocking call. System 2 writes its newest latent into the buffer when ready; System 1 reads whatever is there, right now, without waiting. GR00T's wide token channel from §14.3 changes what sits in that buffer, a full set of vision-language tokens rather than one vector, and it changes the cost of reading it, since the diffusion head must cross-attend to those tokens on every denoising step. The asynchronous shape is the same. The fast path never blocks on the slow path.

## The number that actually bites: jitter

Throughput is the easy number and the one every demo reports. The one that decides whether a robot ships is jitter: the variation in cycle time, the gap between the median cycle and the worst one. A loop that averages 5 milliseconds but occasionally spikes to 30 is not a 200 Hz loop in any sense the control engineer cares about, because control stability depends on cycles arriving on a predictable cadence. A controller tuned for a 5-millisecond period behaves badly when handed an action that is 25 milliseconds late, and "behaves badly" for a balancing humanoid means a stumble.

Jitter comes from the messy parts of the stack, not the network math. Garbage collection pauses in a memory-managed runtime. The operating system scheduling something else onto the GPU for a few milliseconds. A camera frame arriving late because the USB bus hiccuped. Memory allocation that usually hits cache and occasionally does not. None of these show up in a throughput benchmark run on a quiet laptop, and all of them show up on a robot that has been running for six hours in a warehouse. This is why the hands-on exercise for this chapter asks you to measure worst-case jitter and not just mean latency: the mean is the number that looks good in a paper, and the tail is the number that determines whether the robot survives a shift.

Serious real-time robotics fights jitter with the same tools any hard-real-time system uses. Pin the fast loop to a dedicated core so the scheduler cannot preempt it. Pre-allocate every buffer the loop will ever touch so there is no allocation on the hot path. Keep the fast network small and its execution deterministic, no data-dependent branching that makes some cycles longer than others. A learned System 1 is friendlier here than it might seem, because a fixed-size transformer doing a fixed-size forward pass has almost no data-dependent variation in runtime; it does the same arithmetic every cycle regardless of what the input says. That determinism is an underrated reason the fast half of these systems is a plain feedforward network and not something with a variable number of steps.

## Where the diffusion head complicates the budget

GR00T's fast path (§14.3) denoises action chunks instead of regressing them directly, and that choice reaches straight into the latency budget. Every denoising step is a forward pass through the DiT, so a fast head that takes ten steps to produce a chunk costs ten times the compute of a single-pass head for the same output. The multimodality payoff from §10.4 is real, but it is not free, and the place it gets paid for is here, in milliseconds per action.

This is why the number of denoising steps is one of the first things tuned down for deployment. A diffusion model trained with fifty steps might run with four or five at inference, trading some sample quality for the latency headroom the control deadline demands. Flow-matching heads of the π0 kind push this further, since a well-trained flow model can produce a usable action in a handful of integration steps rather than dozens. The design tension is clean: more steps buy better, more multimodal actions, and fewer steps buy the deadline. A dual-system humanoid does not get to pick both, and the deadline wins, because an elegant action that arrives after the robot has fallen is worth nothing.

Latency budgeting, then, is where the abstract "two clocks" story from §14.1 turns into engineering you can be wrong about. You can pick the right architecture and still miss the deadline by choosing a fast network that is 20M parameters too big, a denoiser that takes three steps too many, or a runtime that garbage-collects on the hot path. The next section moves from the budget to the robots that have met it in the field, looking at Figure's deployed Helix stack and the GR00T-enabled humanoids, to see what these numbers look like when the robot is doing real work for real hours rather than closing a loop on a bench.
