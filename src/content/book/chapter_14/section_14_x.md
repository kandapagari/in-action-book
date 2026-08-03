---
chapter: 14
section: 14.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §14.4 (the latency budget, and the claim that worst-case jitter rather than mean latency is the number that keeps a controller stable); §14.1 for the two-clocks argument the exercise is built to expose; §14.2–§14.3 for the Helix and GR00T timing figures the two stand-in loops are sized against; a CPU is enough, no GPU or downloaded checkpoint required, and the whole thing runs in well under a minute
key_refs:
  - Figure AI (2025). Helix, A Vision-Language-Action Model for Generalist Humanoid Control. figure.ai/news/helix.
  - Figure AI (2026). Helix-02. figure.ai/news/helix-02.
  - Bjorck, J. et al. (2025). GR00T N1, An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Google DeepMind (2025). Gemini Robotics-ER 1.5 and the Embodied Reasoning family. arXiv:2510.03342.
---

# 14.x  Hands-on exercise + chapter references

§14.4 asked you to accept a claim on the strength of an argument: that a control
loop is judged by its worst cycle, not its average one, and that this is the whole
reason a dual-system stack exists. This exercise turns the claim into two numbers
you compute yourself. You are going to build a stand-in for a single-system policy
and a stand-in for a dual-system one, run each inside a simulated control loop,
and measure two things the chapter kept insisting on: end-to-end latency per cycle
and the worst-case jitter across a run. If §14.4 is right, the single-system loop
will show a latency floor no amount of averaging can hide, and the dual-system
loop will keep its fast cycle cheap and steady even while a heavy reasoner grinds
away beside it. If §14.4 is wrong, the two histograms will look the same and you
will have caught the chapter overselling. It does not, but a plot you made
yourself is worth more than a paragraph I wrote.

The TOC states the exercise plainly: profile a single-system VLA and a
dual-system VLA, measure end-to-end latency and worst-case jitter. You will not
download a 7B checkpoint to do it, because the effect has nothing to do with what
the networks compute and everything to do with how the loop is shaped. A small
matrix multiply standing in for the fast controller and a large one standing in
for the reasoner reproduce the timing structure exactly, and they run on a laptop
CPU in seconds. The size ratio is the part that carries meaning: pick it to match
the Helix figures from §14.2, an 80M fast network beside a 7B slow one, and the
arithmetic you plot is the arithmetic Figure ships.

## What you need before you start

NumPy and matplotlib. No robot, no simulator, no GPU, no checkpoint. The only
thing you are simulating is time, so the networks can be dummies whose sole job is
to burn a predictable number of floating-point operations. That is the point.
When the jitter appears, you will know it came from the loop structure and the
occasional interruption, not from anything the model was thinking about.

## Exercise 14.x.1 — Time a single forward pass at two sizes

Start by building the two networks and measuring what one forward pass costs. Make
them plain dense layers whose compute scales with a width parameter, so a small
width gives you a System-1-sized model and a large width gives you a
System-2-sized one.

```python
import numpy as np, time

def make_net(width, depth=3, d_in=64):
    layers = [np.random.randn(d_in, width).astype(np.float32)]
    for _ in range(depth - 1):
        layers.append(np.random.randn(width, width).astype(np.float32))
    return layers

def forward(layers, x):
    for W in layers:
        x = np.maximum(x @ W, 0.0)   # dense layer + ReLU
    return x

fast = make_net(width=256)     # stand-in for the 80M System 1
slow = make_net(width=4096)    # stand-in for the 7B System 2
x = np.random.randn(1, 64).astype(np.float32)

for name, net in [("fast", fast), ("slow", slow)]:
    forward(net, x)                        # warm up caches
    t = []
    for _ in range(200):
        s = time.perf_counter(); forward(net, x); t.append(time.perf_counter() - s)
    t = np.array(t) * 1e3                   # milliseconds
    print(f"{name}: mean {t.mean():.3f} ms, p99 {np.percentile(t, 99):.3f} ms")
```

The exact milliseconds depend on your machine, and that does not matter. What
matters is the ratio: the wide network costs something like one to two orders of
magnitude more per pass than the narrow one, which is the 7B-versus-80M gap of
§14.2 showing up as wall-clock time. Note the second number too. The p99 sits
above the mean even here, on a dummy net doing nothing but matrix multiplies,
because the operating system, memory allocator, and cache all conspire to make
some passes slower than others. That gap between mean and p99 is jitter in its
simplest form, and it only gets worse from here.

## Exercise 14.x.2 — Run the single-system loop

Now put the slow network inside a control loop the way a single-system VLA forces
you to. Every cycle has to produce an action, and in a one-network design the only
network you have is the big one, so every cycle pays the full reasoner cost.

```python
def run_single_system(n_cycles=1000):
    latencies = []
    for _ in range(n_cycles):
        s = time.perf_counter()
        forward(slow, x)          # the ONLY thing that can produce an action
        latencies.append((time.perf_counter() - s) * 1e3)
    return np.array(latencies)

lat = run_single_system()
print(f"single-system: mean {lat.mean():.2f} ms, "
      f"max {lat.max():.2f} ms, jitter {lat.max() - lat.mean():.2f} ms")
```

Read the numbers against a control deadline. §14.4 put the fast loop's budget near
5 milliseconds, a 200 Hz cadence. If the mean here is already above that, the
single-system loop cannot hold 200 Hz at all, which is the structural problem, not
a tuning one. And the max will sit well above the mean, so even when the average
squeaks under budget, the worst cycle blows through it. On a robot that worst cycle
is a late action, and a late action on a balance controller is a stumble. The loop
is not "mostly fast enough." It has no fast cycle to fall back on.

## Exercise 14.x.3 — Run the dual-system loop and compare jitter

Here is the design the chapter is about. Run the fast network every cycle, and run
the slow network only once every `k` cycles, caching its output as the current
intent. The fast network reads that cached latent and closes the loop against it.

```python
def run_dual_system(n_cycles=1000, k=25):
    latencies = []
    intent = forward(slow, x)               # initial slow pass
    for i in range(n_cycles):
        s = time.perf_counter()
        if i % k == 0:
            intent = forward(slow, x)        # refresh intent occasionally
        forward(fast, x)                     # fast controller, every cycle
        latencies.append((time.perf_counter() - s) * 1e3)
    return np.array(latencies)

dual = run_dual_system()
print(f"dual-system: mean {dual.mean():.2f} ms, "
      f"max {dual.max():.2f} ms, jitter {dual.max() - dual.mean():.2f} ms")
```

Two features should jump out. The mean drops toward the fast network's cost,
because 24 of every 25 cycles pay only the cheap forward pass. But the max stays
high, spiking on exactly the cycles where `i % k == 0` and the slow pass runs
inline. That is the trap §14.4 warned about: naively bolting a slow reasoner onto a
fast loop moves the mean but not the worst case, and worst case is what the
controller lives or dies by. A histogram makes it obvious.

```python
import matplotlib.pyplot as plt
plt.hist(lat,  bins=40, alpha=0.5, label="single-system")
plt.hist(dual, bins=40, alpha=0.5, label="dual-system (inline slow pass)")
plt.axvline(5.0, color="red", linestyle="--", label="5 ms budget")
plt.xlabel("cycle latency (ms)"); plt.legend(); plt.show()
```

You will see the dual-system distribution pile up near the fast cost with a thin
tail of tall spikes reaching into single-system territory. The spikes are the
unshielded slow passes. Every one of them is a missed deadline waiting to happen.

## Exercise 14.x.4 — Decouple the clocks and watch the tail collapse

The fix, and the actual dual-system design, is to stop running the slow pass
inside the fast loop. §14.4 described two loops on separate hardware talking
through a shared buffer, so the fast controller never blocks on the reasoner and
just reads whatever intent is currently sitting in the buffer. You can simulate
that decoupling in a single thread by charging the slow pass to a separate clock
and never letting it land inside a fast cycle's timing.

```python
def run_decoupled(n_cycles=1000, k=25):
    fast_latencies, intent = [], forward(slow, x)
    slow_budget = 0.0                        # "time owed" to the slow clock
    for i in range(n_cycles):
        s = time.perf_counter()
        forward(fast, x)                     # fast loop: ONLY the cheap net
        fast_latencies.append((time.perf_counter() - s) * 1e3)
        # slow pass runs on its own clock, off the fast loop's critical path
        if i % k == 0:
            ss = time.perf_counter(); intent = forward(slow, x)
            slow_budget += (time.perf_counter() - ss) * 1e3
    return np.array(fast_latencies), slow_budget

fast_lat, _ = run_decoupled()
print(f"decoupled fast loop: mean {fast_lat.mean():.2f} ms, "
      f"max {fast_lat.max():.2f} ms, jitter {fast_lat.max() - fast_lat.mean():.2f} ms")
```

The number to stare at is the max, and it should now sit close to the mean,
because the only work timed inside the fast loop is the fast network. The tall
spikes are gone. The reasoner still runs, still costs what it costs, but its cost
no longer shows up on the clock the controller is graded against. That is the
entire trick of the dual-system architecture reduced to one loop and a comment:
the fast half's worst-case jitter is bounded by the fast network alone, and the
slow half is free to be as heavy and as slow as reasoning demands. Plot all three
histograms together and the story is complete, with the single-system loop far to
the right of the budget line, the inline dual-system loop pulled left but keeping a
spiky tail across it, and the decoupled loop packed tight against the fast cost
with almost no tail at all.

If you want the piece §14.4 flagged about diffusion heads, add a variable number
of denoising steps to the fast network in Exercise 14.x.3, looping the fast
forward pass three or five times per cycle. Watch the fast loop's own mean climb
and its budget headroom shrink. That is why GR00T's denoising-step count
(arXiv:2503.14734) is one of the first knobs tuned down for deployment, and why a
flow-matching head that needs only a handful of steps has room to spare where a
many-step sampler does not.

## Chapter 14 reading list

Cited across §14.1–§14.7, grouped by the job each reference does. Full entries for
everything in the book live in Appendix E.2; this is the chapter-local subset.

### The two flagship dual-system models

- Figure AI (2025). "Helix: A Vision-Language-Action Model for Generalist Humanoid
  Control." figure.ai/news/helix. The spine of §14.2. The 7B/80M split, the 7–9 Hz
  and 200 Hz clocks, the thin continuous latent between the two systems, and the
  roughly 500 hours of auto-labeled teleoperation all come from here.
- Figure AI (2026). "Helix-02." figure.ai/news/helix-02. §14.2 and §14.5's source
  for "System 0," the neural whole-body controller that replaced about 100,000
  lines of hand-written C++ locomotion, and the leaderless two-robot demonstration.
  It is also the setup for the certification worry §17.5 inherits.
- Bjorck, J., et al. (2025). "GR00T N1: An Open Foundation Model for Generalist
  Humanoid Robots." arXiv:2503.14734. §14.3's open counterexample to Helix: the
  diffusion action head, the wide token channel that cross-attends to full
  vision-language tokens on every denoising step, and the N1.5–N1.7 lineage where
  the skeleton held fixed while the backbone (Cosmos) and the data (EgoScale, on
  the order of 20,000 hours of first-person video) did the changing. Its
  denoising-step count is the knob Exercise 14.x.4 asks you to feel.

### The single-system baseline and its ceiling

- Black, K., et al. (2024). "π0: A Vision-Language-Action Flow Model for General
  Robot Control." arXiv:2410.24164. §14.1's starting point. π0's one-heavy-pass,
  cheap-expert asymmetry is the design that bends the two-clock constraint without
  breaking it, and the limit it leaves open (one forward pass, one clock) is the
  ceiling this whole chapter answers.

### The third and fourth families

- Google DeepMind (2025). "Gemini Robotics-ER 1.5 and the Embodied Reasoning
  family." arXiv:2510.03342. §14.3's third branch: the two-clock split with a
  heavier reasoning half, the "Embodied Thinking" pattern that emits intermediate
  points and trajectories before the fast half moves, and Motion Transfer across
  embodiments. The embodied-chain-of-thought idea here is developed in full in
  §18.4, and Motion Transfer feeds the cross-embodiment discussion in §18.1.
- Google DeepMind (2026). "Gemini Robotics 2, Gemini Robotics-ER 2, and Gemini
  Robotics On-Device 2." deepmind.google/blog (announced 2026-07-30). §14.6's
  fourth branch: the first VLA to drive a full humanoid, legs included, under one
  learned policy, demonstrated on Apptronik's Apollo 2 and erasing the
  locomotion-versus-manipulation seam every earlier stack kept. The On-Device 2
  variant's few-hours re-targeting to a new body feeds §18.1, and putting balance
  inside the manipulation policy sharpens the certification worry of §17.5.
- OpenDriveLab (2026). "WholebodyVLA: Towards Unified Latent VLA for Whole-body
  Loco-manipulation Control." ICLR 2026. §14.6's open research counterpart to
  Gemini Robotics 2, the readable version of unified whole-body control.

### The backward thread

- Featherstone, R. (2008). *Rigid Body Dynamics Algorithms.* Springer. §14.1 and
  §14.7's reminder that the fast-inner-loop, slow-outer-loop structure predates
  learning by decades. The dual-system VLA is §4.3's layered control with both
  layers now learned, and Helix-02's System 0 is where that rediscovery reaches the
  locomotion layer classical methods used to own.

## Chapter summary

Chapter 14 took the ceiling Chapter 13 named and built the field's answer to it.
You can now explain why a single forward pass cannot serve a general robot: the
semantic decision of what to do stays valid for a second or two, while balance and
contact demand corrections inside tens of milliseconds, so one network is forced
to be either too slow to stand or too small to reason. You can draw both Helix and
GR00T N1 from memory, name what crosses the seam in each (Helix's thin continuous
latent against GR00T's wide token channel, Helix's closed end-to-end training
against GR00T's open video-heavy recipe), and place Gemini Robotics-ER as a third
family that loads the reasoning half harder. You can do the latency accounting that
decides whether a split ships, and after this exercise you can do more than recite
it, because you plotted the single-system loop stranded past its budget, watched an
inline slow pass leave a spiky tail across the deadline, and saw the tail collapse
once the two clocks were decoupled. And you can read a deployment claim for the
number that matters, consistency across a full shift rather than success on a lucky
run, knowing the failures that show up over hours are jitter and contact-timing
failures, not reasoning ones. Chapter 15 picks up the question this chapter kept
deferring: what data trains these systems, and how does anyone measure whether they
actually work.
