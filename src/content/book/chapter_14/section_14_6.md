---
chapter: 14
section: 14.6
title: Summary
target_words: 2000
status: draft
prereqs: §14.1–§14.5; why reasoning and reacting want clocks an order of magnitude apart and one forward pass can only serve one of them, Helix's thin continuous latent between a 7B System 2 and an 80M System 1, GR00T N1's wide token channel and diffusion action head plus its N1.5–N1.7 data lineage, Gemini Robotics-ER as an embodied-reasoning third family, the worst-case-jitter latency budget that decides whether a stack ships, and the deployment record that shows the fast half is where a shift lives or dies
key_refs:
  - Figure AI (2025). Helix, A Vision-Language-Action Model for Generalist Humanoid Control. figure.ai/news/helix.
  - Figure AI (2026). Helix-02. figure.ai/news/helix-02.
  - Bjorck, J. et al. (2025). GR00T N1, An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Google DeepMind (2025). Gemini Robotics-ER 1.5 and the Embodied Reasoning family. arXiv:2510.03342.
---

# 14.6  Summary

Chapter 13 handed this chapter a single working model and one built-in limit.
π0 (arXiv:2410.24164) runs its heavy backbone once and lets a cheap flow expert
iterate over the frozen scene, which is why it closes a control loop where
token-decoding VLAs stall. That asymmetry buys a fast action stream, but it does
not separate the two jobs a general robot is actually doing, and §14.1 opened by
naming them: a slow semantic job that decides what to do, and a fast sensorimotor
job that keeps the body upright and the grasp from slipping. Fold both into one
forward pass and you are stuck picking one clock for work that wants two clocks
more than an order of magnitude apart. Chapter 14 spent five sections on the
design that stops picking. Run two networks at two rates at once, a slow reasoner
setting intent and a fast controller closing the loop against fresh sensors, and
each half runs at the rate its job can actually tolerate. §14.1 built the
argument, §14.2 walked Helix end to end, §14.3 set GR00T N1 beside it box by box
and named Gemini Robotics-ER as a third family, §14.4 did the latency arithmetic
that decides whether the split works, and §14.5 checked the whole idea against
robots that have run for real hours on real floors. The chapter's job was to make
the two-clock structure concrete enough that Part 5's fine-tuning and safety
questions read as questions about a specific machine rather than about VLAs in
the abstract.

## The ideas worth carrying forward

*Reasoning and reacting run on clocks that differ by more than an order of
magnitude, and one forward pass can only offer one clock.* §14.1 is the whole
chapter in miniature. The semantic decision "pick up the blue mug next" stays
valid for a second or two, so re-deriving it two hundred times a second wastes
compute to reach the same answer. Balance and contact will not wait: a bipedal
robot can go from stable to falling in under a hundred milliseconds, and the
window to react to a slipping grasp is tens of milliseconds. A single network
forced to serve both ends up too slow to balance or too small to reason, and
π0's chunking bends that constraint without breaking it, because the fast
corrections inside a chunk are replayed from a decision made at the top of it
rather than computed against sensors arriving mid-chunk. The fix is to stop
sharing the forward pass. This is also a rediscovery, not an invention: §4.3's
classical robots already ran a fast inner control loop inside a slow outer
planner, and the dual-system VLA is that layered structure with both layers now
learned.

*Helix draws the seam between the two systems as plainly as any deployed system
does, and the choice that matters most is what crosses it.* §14.2 sized the two
halves. System 2 is a 7B vision-language model running at roughly 7 to 9 Hz;
System 1 is an 80M transformer running at 200 Hz, almost a hundred times smaller
and twenty times faster, driving thirty-five-odd degrees of freedom of a humanoid
upper body. What passes between them is deliberately thin, one continuous latent
vector rather than a language string, because a continuous channel interpolates
smoothly and carries no tokenizer latency, which is what lets a 7 Hz reasoner
steer a 200 Hz controller without either stuttering. That thin differentiable
channel is also what makes the pair trainable as one thing on roughly 500 hours
of auto-labeled teleoperation. Helix-02 then folded balance itself into the
learned stack as "System 0," a neural whole-body controller replacing about
100,000 lines of hand-written C++, which quietly erases part of the classical
boundary §4.3 drew and hands the question to §18.1.

*GR00T N1 makes the same top-level split and fills the boxes differently, and the
disagreement is the lesson.* §14.3 leaned on GR00T (arXiv:2503.14734) because
NVIDIA published the architecture, the recipe, the data, and the weights, so you
can read what Figure kept closed. Two forks stand out. GR00T's fast path is a
diffusion transformer that denoises action chunks rather than regressing them, so
it represents multimodal action distributions cleanly at the cost of extra
forward passes per action. And its channel is wide where Helix's is thin: the DiT
cross-attends to System 2's full vision-language tokens on every denoising step,
keeping nuance the reasoner would otherwise have to compress away, at the price of
tighter coupling and a heavier fast loop. Neither bet wins from first principles,
which is exactly why having two open-enough systems to compare is worth more than
either alone. The N1.5-to-N1.7 lineage then makes a second point: across releases
the skeleton stayed fixed and the changes clustered in the backbone (Cosmos,
physics-informed) and the data (EgoScale, on the order of 20,000 hours of
first-person video), the cheapest way to buy manipulation breadth at scale.

*There is a third family, and it is close enough to belong here and different
enough to name separately.* Google DeepMind's Gemini Robotics-ER
(arXiv:2510.03342) keeps the two-clock split but loads the reasoning half harder.
Its "Embodied Thinking" pattern has System 2 reason through spatial and temporal
steps, emit intermediate points and trajectories on the image, and hand a plan
down rather than a bare intent, which is the embodied chain-of-thought idea §18.4
develops in full. Its Motion Transfer carries a skill learned on one embodiment
to another through the shared reasoning layer, a direct attack on the
cross-embodiment problem §18.1 takes up. So the chapter's family tree has three
branches that agree on the skeleton and disagree about what the slow clock should
produce and how richly it should talk to the fast one: Helix bets thin latent and
closed end-to-end training, GR00T bets wide token channel and open video-heavy
data, Gemini bets on a reasoner that thinks explicitly before the fast half moves.

*A dual-system design that misses its deadline is just two networks that fall over
together, and the number that decides this is jitter, not mean latency.* §14.4 was
the reality check. Real-time control is a worst-case guarantee, not an average: a
loop that runs 5 milliseconds most cycles and spikes to 30 on one in a thousand is
not a 200 Hz loop in any sense the control engineer cares about, because the
controller is tuned for a cadence and a late action means a stumble. The fast
network's 80M size was chosen backward from that 5-millisecond budget, and its
fixed-size feedforward shape is a feature because it runs the same arithmetic
every cycle with almost no data-dependent variation. The split only works because
the loops run asynchronously on separate hardware through a shared buffer, so
System 1 never blocks on System 2 and instead reads whatever latent is currently
there, accepting staleness of up to one slow-clock period as the price of never
stalling. GR00T's diffusion head reaches straight into this budget, which is why
the denoising-step count is one of the first things tuned down for deployment, and
why flow-matching heads that need only a handful of steps have an edge here.

## What you should be able to do now

Four things, in the order the rest of the book uses them.

You should be able to *decide whether a robot needs a dual-system design at all*.
Given a task, you can say which of the two failures π0 leaves open actually bites:
a stale plan, or a missed fast correction on a contact or balance event. A slow
tabletop arm sliding a mug needs neither and should not pay for two networks and a
communication protocol; a humanoid picking a tote off a moving belt needs both,
because the plan ages fine but the grip slip and the shoulder bump land inside a
window one clock cannot serve. This is the diagnosis §14.1 built, and it is what
keeps you from reaching for a humanoid-grade stack when a single-system policy
would do.

You should be able to *draw either Helix or GR00T from memory and say what crosses
the seam*. Slow VLM setting intent, fast controller closing the loop, an
asynchronous buffer between them. You can name Helix's 7B/80M split and thin
continuous latent, contrast it with GR00T's diffusion head and wide token channel,
and say what each choice buys and costs. You can also place Helix-02's System 0
below System 1 and explain why absorbing locomotion into the learned stack matters
for §17.5's certification problem.

You should be able to *do the latency accounting that decides whether the split
ships*. You can lay out the fast loop's 5-millisecond budget, explain why
worst-case jitter and not mean throughput is the number that keeps a humanoid
upright, and point to where the milliseconds go: the forward pass, the sensor
read, the actuator write, and the denoising steps if the head is a diffusion
model. You can say why the two loops must run asynchronously and why the fast
network is kept small and deterministic on purpose.

You should be able to *read a deployment claim for the number that actually
matters*. Consistency across a full shift, not success rate on a good run. You know
that a 95% policy is a jam every twenty parts on a line that counts every
repetition, that the failures which show up over hours are jitter and
contact-timing failures rather than reasoning failures, and that the teams whose
claims aged well reported repetition counts and shift lengths and admitted where
dexterity still falls short of a human hand.

## Where the chapter has set up the rest of the book

Chapter 14 hands forward more than it resolves. The largest handoff is to Part 5.
§14.5 showed that both Helix's vertically integrated stack and GR00T's transferable
foundation model exist because fine-tuning a base model onto your own robot is the
only affordable door for most labs, which is precisely the decision Chapter 16
takes apart: pick a base model, build a teleop dataset that does not waste your
time, and decide how much of the two halves to freeze. GR00T N1.7's EgoScale video
prior is why that fine-tuning can be cheap, since most of the manipulation prior is
already baked in and the lab's teleop only has to teach the specifics of its
embodiment.

The safety thread runs straight into Chapter 17. Helix-02's System 0 made the
balance layer as opaque as the manipulation layer, so you can no longer read the
locomotion controller's source to prove it will not step wrong, which is the
certification gap §17.5 confronts directly. And the deployment lesson from §14.5,
that the fast half run for hours against drifting sensors is where a shift lives or
dies, is the argument for the runtime monitors and rollback machinery §17.2 and
§17.4 build.

The dataset story stacked under §14.3 sets up Chapter 15. GR00T's training pyramid,
real teleop at the base, human egocentric video in the middle, simulation on top,
is the concrete answer to a question Chapter 15 takes apart in full: what has and
has not succeeded the pooled-teleop corpus of Open X-Embodiment (§12.4). LingBot-VLA
2.0's one-policy-across-twenty-morphologies result from §14.5 is the same thread,
and the evaluation complaint that a demo is not a shift is the problem Chapter 15
confronts when it gets to real-robot evaluation and its variance.

One thread runs backward, the way it did in Chapters 12 and 13. The dual-system
design is a rediscovery of §4.3's layered control, a learned planner over a learned
tracking loop, and Helix-02's System 0 pushes that rediscovery down to the metal by
learning the locomotion that used to be derived from a dynamics model. §18.1 is
where the book returns to what that dissolving boundary means for the field.

## What the chapter has not covered

Two omissions worth naming. The chapter treated the dual-system split as the answer
to the two-clock problem and said almost nothing about how you decide the split's
internal boundary, how wide the latent channel should be, how many degrees of
freedom belong on the fast clock versus the slow one, or whether the two are best
trained jointly or separately. §14.3 showed Helix and GR00T disagreeing on the
channel width and left the disagreement open on purpose, because the field has not
settled it and this book will not pretend to. That is a design frontier, not a
solved recipe.

The chapter also stayed almost entirely on humanoids and the manipulation tasks
they run. It did not develop the dexterity-first models §14.5 flagged, GENE and
RLDX-1, beyond noting that they aim at the hardest end of the fast loop and remain,
by their creators' own framing, below reliable human-level dexterity. That
frontier gets its own treatment in §18.2. Nor did the chapter touch how a
dual-system policy should behave when its slow reasoner produces an intent the fast
controller cannot safely execute, which is a safety question §17.2 owns.

Chapter 14's contribution to the book's argument is to answer the ceiling Chapter
13 named. A single forward pass couples reasoning and reacting to one clock, and
splitting the policy into a slow vision-language reasoner and a fast sensorimotor
controller, joined by a thin asynchronous channel, lets each run at the rate its
job requires, which is what a balance-critical dexterous humanoid needs and a
single-system VLA cannot give it. Three families, Helix, GR00T, and Gemini
Robotics-ER, agree on that skeleton and disagree productively about its details.
The design earns its complexity only on tasks fast or dynamic enough to need it,
its latency budget is decided by worst-case jitter rather than average throughput,
and its deployment record says the fast half run for real hours is where success
is actually won or lost. With the two-clock structure understood at this depth,
Chapter 15 can ask what data trains these systems and how anyone is supposed to
measure whether they work.

§14.x closes the chapter with a hands-on exercise, measuring worst-case jitter
rather than mean latency on a simulated two-rate control loop so the number §14.4
insisted on becomes something you plot rather than something you take on faith,
followed by the chapter's full reading list.
