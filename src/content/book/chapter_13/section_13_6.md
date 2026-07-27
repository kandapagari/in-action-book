---
chapter: 13
section: 13.6
title: Summary
target_words: 2000
status: draft
prereqs: §13.1–§13.5; the three costs of discrete action tokens, π0's split of a PaliGemma backbone from a flow-matching action expert, flow matching as a regression objective that keeps multimodality in the noise, the π0→π0.5→π0.6→π0.7 lineage that leaves the head fixed and changes the data-and-training story, and the four open problems the flow head does not close
key_refs:
  - Black, K. et al. (2024). π0, A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - Pertsch, K. et al. (2025). FAST, Efficient Action Tokenization for Vision-Language-Action Models. arXiv:2501.09747.
  - Beyer, L. et al. (2024). PaliGemma, A Versatile 3B VLM for Transfer. arXiv:2407.07726.
  - Physical Intelligence (2025). π0.5, A VLA with Open-World Generalization. arXiv:2504.16054.
  - Assran, M. et al. (2025). V-JEPA 2 and V-JEPA 2-AC. arXiv:2506.09985.
---

# 13.6  Summary

Chapter 12 left the action head as a two-way fork, discrete bins on one branch
and a diffusion model on the other, and then set flow matching aside on purpose
so the OpenVLA-versus-Octo comparison stayed clean. Chapter 13 picked the piece
back up and spent a whole chapter on the third branch. The through-line is one
model taken apart from every side. π0 (arXiv:2410.24164) keeps the web-pretrained
backbone that Part 4 spent three chapters building, throws out the token head,
and regresses continuous actions with a flow-matching objective instead. §13.1
was the diagnosis of why you would want to. §13.2 laid out the architecture that
results. §13.3 opened up the flow-matching engine that makes the action expert
work. §13.4 showed what the finished model buys on real hardware and traced the
π0.5→π0.6→π0.7 lineage that grew out of it. §13.5 named the four problems the
head does not touch. The chapter's job was to make one continuous-action
foundation model legible enough that Chapter 14's dual-system designs read as a
response to a specific ceiling rather than a new direction out of nowhere.

## The ideas worth carrying forward

*Discrete action tokens fail on exactly the tasks a foundation model is supposed
to be good at, and one of the three failures cannot be engineered away.* §13.1
named the costs: a resolution ceiling from uniform binning, a sequence-length
blowup when you tokenize a long high-frequency chunk, and serial-decode latency
because autoregression emits one token at a time by construction. FAST
(arXiv:2501.09747) compresses the first two and stretches the token head further
than you would guess. It does nothing for the third. On a 50 Hz laundry-folding
task all three bite at once, and the latency one stays bitten no matter how
clever the tokenizer gets. That is the specification π0 was built to meet: emit
a full continuous action chunk in one shot, represent genuinely multimodal action
distributions, and do it fast enough to close a real control loop.

*π0 works by splitting the two jobs the token head was doing at once.* §13.2
walked the architecture. A PaliGemma backbone (arXiv:2407.07726) handles the
scene and the language exactly as it already knew how to, and a separate
flow-matching action expert, a second transformer sharing the same attention,
does the continuous motion generation. The scene gets encoded once, outside the
sampling loop; the expert then integrates a noise draw into a clean action chunk
in around ten steps. That split is not cosmetic. It puts the transferable,
task-general knowledge in the backbone and the embodiment-specific motor
knowledge in the expert, which is the fact Chapter 16 leans on when it asks how
much of each you can freeze while fine-tuning for a new robot.

*Flow matching is a plain regression loss that somehow produces multimodal
behavior, and the trick is where the multimodality lives.* §13.3 was the part
that should worry a careful reader until it doesn't. You train the expert to
predict a velocity field, the loss is mean-squared error against a straight-line
target between noise and data, and a regression loss is exactly the averaging
trap §13.1 warned about. It escapes the trap because the multimodality is routed
through the noise, not emitted by the network in one shot: which mode a sample
lands in depends on where its noise particle started, so the field can split the
noise cloud toward two demonstrated motions with a slack seam between them rather
than averaging to the dead center. The 2-D toy in the hands-on exercise makes the
splitting field visible. Ten integration steps against a diffusion sampler's many
denoising steps is also where the latency π0 clawed back becomes arithmetic
rather than assertion.

*The finished model does dexterous long-horizon manipulation token heads cannot,
and its successors kept the head and changed everything around it.* §13.4 ran the
laundry test and then read four models as one argument. π0 proved the
flow-matching continuous head could fold laundry and bus tables where discrete
heads stalled. π0.5 (arXiv:2504.16054) found the generalization bottleneck was
backbone data rather than the action head, and co-trained on the non-robot world
to enter houses it had never seen. π0.6 added reinforcement learning on the
robot's own experience once imitation hit its reliability ceiling. π0.7 turned a
capable generalist into a steerable one. Not one of them replaced the flow head;
each stacked a different data-and-training story on top of it. For a student
that pattern is worth noticing early, because the durable contribution is the
architecture and almost everything after π0 is what you wrap around it.

*The head that made the chapter work closes none of the field's hard problems.*
§13.5 was deliberately deflationary. The RL loop π0.6 uses is too expensive for
most labs to run. The next 10,000 hours of training data have to come from
somewhere teleoperation cannot fill and video has not yet proven it can, which is
where V-JEPA 2-AC (arXiv:2506.09985) comes back into view in Chapter 18. There is
no trusted evaluation number and no scaling law for action the way there is for
language, so labs find the returns to scale by spending the money and looking.
And "steerable" is not the same as controllable or safe. Confusing a very good
architecture for solved surrounding problems is how you end up trusting a laundry
video more than it has earned.

## What you should be able to do now

Four things, in the order the rest of the book uses them.

You should be able to *say precisely when a discrete action head stops paying and
why FAST only fixes two-thirds of it*. Given a task, you can predict whether the
resolution ceiling, the sequence-length blowup, or the serial-decode latency is
the binding constraint, and you know the latency one is structural to
autoregression, so no tokenizer removes it. This is the diagnosis §13.1 built,
and it is what lets you read a continuous-head paper as an answer to a stated
problem instead of a fashion.

You should be able to *draw π0's architecture from memory and say what each half
knows*. Backbone for scene and language, action expert for motion, one shared
attention, scene encoded once outside the sampling loop. You can point at where
the task-general knowledge sits versus the embodiment-specific motor knowledge,
which is the exact partition Chapter 16 turns into a freeze-versus-retrain
decision.

You should be able to *explain how a regression objective yields multimodal
actions*. You can state the velocity-field loss, explain that it looks like the
averaging trap and isn't, and locate the multimodality in the noise particle's
starting point rather than in anything the network emits at once. You can also
say why ten integration steps beats a diffusion sampler's many, and connect that
directly to running at control rate.

You should be able to *read the π0 lineage as one argument and separate the
durable part from the churn*. π0 fixed the action head; π0.5 fixed backbone data;
π0.6 fixed reliability with RL; π0.7 fixed steerability. You can name which
bottleneck each release attacked and see that the flow head survived all four,
which is the habit §13.4 was building so the model zoo does not read as a parade
of replacements.

## Where the chapter has set up the rest of the book

Chapter 13 hands forward more than it closes. The largest handoff is to Chapter
14, and §13.5 wrote the setup explicitly: π0 reasons and acts in one forward pass
through one model, which is what makes it fast and also means it cannot think
longer on a harder problem the way a person pauses before a tricky fold. When a
task needs deliberation on a slow clock and reaction on a fast one at the same
time, a single head has to serve both and something gives. Splitting the reasoner
from the sensorimotor controller is the field's answer, and it is exactly the
dual-system design Helix and GR00T N1 use.

The freeze-versus-retrain partition from §13.2 is the on-ramp to Chapter 16. π0's
clean separation of backbone knowledge from motor knowledge is part of why it
fine-tunes gracefully, and "how much of each do I retrain for my robot" is the
question the recipe card in Part 5 answers. The data story stacked under §13.2
and §13.5, roughly 10,000 hours of multi-embodiment trajectories that Physical
Intelligence assembled beyond any academic dataset, is the setup for Chapter 15,
which takes the dataset side apart and asks what has and has not succeeded the
pooled-teleop corpus. And the evaluation complaint from §13.5, that there is no
trusted number for this, is the problem Chapter 15 confronts head-on when it gets
to real-robot evaluation and its variance.

One thread runs the same way it did in Chapter 12, back into the classical
material. π0 is still behavior cloning at heart, a map from observations to
actions regressed onto demonstrations, with the compounding-error exposure
Chapter 6 warned about. The flow-matching head made the imitation smoother and
multimodal; it did not repeal the imitation-learning failure modes underneath,
which is why π0.6 needed RL to push past the reliability ceiling in the first
place. Chapter 17 on safety is where that unpaid debt returns.

## What the chapter has not covered

Two omissions worth naming. The chapter treated π0 as the representative
continuous-action foundation model and said little about the diffusion-head
alternative it competes with directly, Octo and the ACT-style policies from
Chapter 10. That was a choice: §13.1 established that diffusion policies are one
valid answer to the same specification, and π0 picked flow matching partly for
its few-step sampling, but comparing the two families head-to-head on the same
hardware is an evaluation question the book parks until Chapter 15. Here it was
enough to develop one lineage in depth.

The chapter also stayed inside a single model doing one forward pass, the same
frame Chapter 12 kept. It did not touch task decomposition, an explicit slow
planner over a fast controller, or any structured split between deliberation and
reaction. That absence is not an oversight; §13.5 named it as π0's built-in
ceiling and handed it to Chapter 14, whose entire premise is the two-system
structure π0 deliberately does without.

Chapter 13's contribution to the book's argument is to settle the action-head
question left open in Chapter 12 by developing the continuous branch in full. A
web-pretrained backbone can be paired with a flow-matching action expert that
emits a whole action chunk in one shot, keeps genuine multimodality by routing it
through the noise, and samples fast enough to run a real control loop, which is
what token heads could not manage on fast dexterous tasks. The head proved
durable enough that three successive releases changed the data and the training
without touching it. And the problems that head does not solve, an unaffordable RL
loop, a data supply teleoperation cannot fill, no trusted evaluation number, and a
controllability-and-safety gap, are the ones the rest of the book keeps returning
to. With one continuous-action model understood at this depth, Chapter 14 can ask
what happens when you stop trying to do reasoning and control in the same forward
pass.

§13.x closes the chapter with a hands-on exercise, training the 2-D flow-matching
toy from §13.3 so the splitting velocity field becomes something you plot rather
than something you take on faith, followed by the chapter's full reading list.
