---
chapter: 18
section: 18.5
title: What to read next, and how to contribute
target_words: 2000
status: draft
prereqs: §18.1–§18.4 (the four open problems this section turns into a plan), §16 (fine-tuning, the skill that makes a one-robot contribution possible), §15.6 (building your own evaluation, the part of a research project people skip and shouldn't), §2.x and §16.x (the exercises that already gave you a working policy to build on).
key_refs:
  - Open X-Embodiment Collaboration (2023). arXiv:2310.08864.
  - Kim, M. J. et al. (2024). OpenVLA. arXiv:2406.09246.
  - Physical Intelligence (2024). π0. arXiv:2410.24164.
  - Assran, M. et al. (2025). V-JEPA 2. arXiv:2506.09985.
---

# 18.5  What to read next, and how to contribute

A book like this has a failure mode: the reader closes it impressed and does nothing, because the field looks like it belongs to a dozen labs with thousand-GPU clusters and robot fleets. That impression is half true and half a trap. The frontier results do come from big labs. But the open problems in this chapter have corners that a person with one robot, or no robot and a good simulator, can push on in a year, and several of the field's useful contributions came from exactly that. This section is the practical exit from the book: what to read to go deeper, and how to turn reading into a project that produces something real.

## What to read, and in what order

Reading everything is not a plan. Reading the right five things and then building is. Here is the short path through the primary sources, chosen so each one earns its time.

Start with the papers you have already been using. OpenVLA (arXiv:2406.09246) is the model you ran in Chapter 2 and fine-tuned in Chapter 16, and rereading it now, with the whole book behind you, is different from reading it cold; the design choices that were opaque in Chapter 2 are legible after Chapters 10 through 13. Then π0 (arXiv:2410.24164), because it is the cleanest statement of the flow-matching action head that Chapter 13 built up, and because the π-family is where the continuous-action lineage keeps moving. These two are your anchors; you can run one and you understand the other.

For the open problems specifically, read one representative paper per bet rather than the survey. Open X-Embodiment (arXiv:2310.08864) for cross-embodiment, because the dataset and the RT-X transfer result are the thing every later cross-embodiment paper is arguing with. V-JEPA 2 (arXiv:2506.09985) for video pretraining, because §18.3 argued it is the first result that made the video bet concrete and the paper is where the 62-hours-to-a-policy claim is laid out. For reasoning-plus-action, read ERVLA (arXiv:2606.03784) not because it is the most famous but because its CoT-dropout idea is the most instructive: it shows you a place where a simple, sharp question, does the reasoning need to run at inference, produced a real result. Save the surveys for when you need breadth. The VLA survey at arXiv:2505.04769 and the efficient-VLA survey at arXiv:2510.24795 are good maps once you already know the territory, and useless as a first read, because a survey teaches you the names of things and not how any of them works.

Two habits make the reading compound. Keep the arXiv IDs, not the titles, because the field renames and re-releases and the ID is the stable handle, which is why this book cites by ID wherever it can (Appendix E.2 has the full list). And read the appendices and the failure sections of these papers, not just the results, because the appendix is where a paper admits what it could not get to work, and that admission is where the open problems actually live.

## Where a one-robot researcher can push

The four bets are not equally accessible. Some need a fleet; some need a weekend. Matching the problem to what you have is the difference between a project you finish and one you abandon.

Cross-embodiment transfer (§18.1) sounds like it needs many robots, and the headline results do, but the tractable corner is the opposite: take a released cross-embodiment policy and study where it breaks on your one robot. The universal-action idea from UniAct (arXiv:2501.10105) makes a testable claim, that there is a shared action space above joints, and you can test it by fine-tuning a generalist onto a robot it never saw and measuring how much of its cross-robot knowledge survives. That is a one-robot experiment with a real result at the end.

Long-horizon and dexterous tasks (§18.2) are the most accessible of the four, which is the section's happiest surprise. The compounding-error curve reproduces on a tabletop: build one honestly multi-step task, run your fine-tuned policy on it fifty times, and log where in the sequence it fails and why. You will produce the per-step success-rate breakdown that most papers gesture at and few publish, and the failure taxonomy you build is exactly the kind of unglamorous, useful artifact the field is short on. Dexterity is harder because it needs a good hand, but if you have one, the generalization-to-contact question, does the grasp survive a slipperier object, is wide open and cheap to probe.

Video pretraining (§18.3) is the bet that most rewards having no robot at all, because most of the work is in simulation and on video datasets, and the label-gap problem is conceptual before it is physical. If your access is a GPU and not a robot, this is your corner: take a video-pretrained representation and measure how much robot data it actually saves on a fixed downstream task, which is the number V-JEPA 2 reported once and almost nobody has replicated.

Reasoning-plus-action (§18.4) sits in between. The ERVLA question, how much reasoning has to survive to inference, is answerable in simulation on LIBERO with a single GPU, and the grounded-spatial-reasoning line from Embodied-R1 (arXiv:2508.13998) has verifiable outputs a laptop can score. If you like the language-model side of the field more than the control side, this is where your existing intuitions transfer.

## A year-1 research plan

Here is a concrete arc, the shape of a first serious project, written for someone with one arm or a simulator and a year of part-time attention.

The first two months are not research; they are reproduction, and skipping this step is the most common way a project dies. Reproduce a result you did not produce: fine-tune OpenVLA on a public dataset and match a published success rate within a few points, using the Chapter 16 recipe. If you cannot reproduce a known number, you cannot trust a new one, and the debugging you do here (§3.5) is the skill the rest of the project runs on.

Months three and four are for building the evaluation before the method, which §15.6 argued and which everyone ignores until it burns them. Pick one open problem, define the task, and build the eval that would detect progress on it: the multi-step task and its per-step logging for long horizons, the held-out embodiment for transfer, the data-efficiency curve for video pretraining. An honest evaluation you built yourself is worth more than a method, because a method without an evaluation is a demo, and the field has enough demos.

The back half of the year is the actual attempt: one change to one model, measured against the evaluation you already built, with the negative results kept. Most changes will not help, and the discipline is to report that they did not rather than to keep tuning until a number moves, which is how the reproducibility problem the field has got made. Write it up at 500 words first, the position note the §18.x exercise asks for, before you write it up at eight pages. If the result is real, the short version will already be clear; if it is not, the short version will show you why.

One more piece of advice, because it saves months. Pick a problem where you can get a signal in days, not one where the first measurement takes a quarter to set up. A student who chooses "improve dexterity on a five-fingered hand" needs the hand, the teleop rig, the data pipeline, and a working baseline before a single experiment runs, and most of that time produces no result. A student who chooses "measure how ERVLA's accuracy changes as you drop more of the reasoning trace" can get a curve on LIBERO in a week with one GPU, and the curve is a finding whether it goes up, down, or flat. Fast feedback is not a luxury for a first project; it is the thing that keeps the project alive long enough to teach you the field. The ambitious problem will still be there in year two, and you will be far better equipped to attack it having shipped something small first.

There is also a quieter kind of contribution worth naming, because it suits some people better than novel methods. Reproduction and negative results are undersupplied and genuinely valued: a clean writeup showing that a published number does not replicate on a second robot, or that a celebrated trick does not help outside its original benchmark, is a real service to a field with a reproducibility problem, and it asks for rigor rather than invention. If you are better at careful measurement than at inventing architectures, that is not a lesser path; it is the path the field is shortest on.

The through-line of this book has been that these models are more capable than they are understood, and that gap is not a reason to stand back. It is the opening. The frontier moves fast enough that a careful person with a modest setup and a real evaluation can find something the big labs skipped, because the big labs are optimizing headline numbers and the useful corners are full of unmeasured failures. The last content section made this personal on purpose. The summary that follows collects what the chapter established, and the closing exercise turns this plan into the first 500 words of it.
