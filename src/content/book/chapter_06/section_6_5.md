---
chapter: 6
section: 6.5
title: "Choosing between BC, IRL, and RL"
target_words: 2000
status: draft
prereqs: §6.1 (why imitation dominates; the four-way map of methods); §6.2 (BC objective); §6.3 (compounding error, DAgger); §6.4 (IRL, GAIL); §5.4–5.5 (reward design difficulty, MDP-to-robot gap)
key_refs:
  - Mandlekar et al. (2021). What Matters in Learning from Offline Human Demonstrations for Robot Manipulation. arXiv:2108.03298.
  - Ross, Gordon & Bagnell (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS. (DAgger)
  - Ho & Ermon (2016). Generative Adversarial Imitation Learning. NeurIPS.
  - Kumar et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS. (CQL)
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# Choosing between BC, IRL, and RL

You have now seen four ways to turn experience into a policy: behavior cloning (§6.2), DAgger (§6.3), inverse reinforcement learning and its adversarial cousin GAIL (§6.4), and, from Chapter 5, ordinary reinforcement learning against a hand-specified reward. A fifth, offline RL, was named in §6.1 and gets its due below. The natural question after meeting all of them is the one a working engineer actually faces: given *this* dataset, *this* robot, and *this* deadline, which do I reach for first?

The honest short answer, and the one the rest of this book is built around, is that you almost always start with behavior cloning. But "start with BC" is a default, not a law, and the point of this section is to make the default legible: to say precisely which question each method answers, what each one demands from you in return, and what observations should push you off the default toward something more expensive.

## The three questions that decide it

Every method in this chapter is, underneath, an answer to a different question about what you have and what you can get.

Behavior cloning asks: *do I have demonstrations of the behavior I want?* If yes, and if I will mostly be deploying in situations the demonstrations cover, BC fits a state-to-action map and is done. It needs no reward, no simulator, and no online interaction. Its weakness is the one §6.3 dissected: it has no mechanism to recover from states the demonstrations never visited, so its error compounds over a long horizon.

DAgger asks a follow-up: *can I put an expert back in the loop while the learner runs?* If a human (or a privileged scripted controller) can label the states the policy actually reaches, DAgger closes BC's covariate-shift gap directly. It needs interactive expert access during training, cheap in simulation, expensive and sometimes unsafe on hardware.

Inverse RL and GAIL ask: *do I want the expert's objective rather than the expert's reflexes?* If the task will be deployed in configurations no demonstration covers, and a portable reward would generalize where a cloned trajectory will not, reward inference earns its cost. The price, from §6.4, is a simulator the policy can explore in for many steps and a two-player optimization that is finicky to stabilize.

Reinforcement learning asks the bluntest question of all: *do I have a reward and a place to fail safely?* If you can write a reward and afford millions of trial-and-error samples, which in practice means a simulator, RL can discover behavior no demonstrator produced. Chapter 5 laid out why those two preconditions are rarely both met on a real robot.

Notice that the questions form a rough ladder of cost. BC asks only for data you may already have. Each rung above it asks for something harder to obtain, interactive expert access, a simulator, a reward function, in exchange for a specific kind of generalization that the rung below cannot give you. The engineering discipline is to climb no higher than your problem forces you to.

## Where offline RL sits

The ladder has a rung between pure cloning and full online RL that is worth naming on its own, because it is increasingly where serious robot systems live. *Offline RL* uses a fixed dataset, the same demonstrations BC would consume, but optimizes a reward signal against it, without ever interacting with the environment. It answers a question BC cannot address: *my demonstrations are mixed quality, and I have at least a sparse reward; can I do better than copying the average?*

BC copies every demonstration with equal weight, good and bad alike; if half your teleop episodes fumbled the grasp, BC dutifully learns to fumble half the time. Offline RL, given a reward that distinguishes the good episodes from the bad, can prefer the actions that led to high return and downweight the rest. The catch is subtle and was the field's central difficulty for years: a value function trained on a fixed dataset will confidently overestimate the value of actions the dataset never contains, and a naive offline RL run chases those phantom values straight off a cliff. Conservative methods such as CQL (Kumar et al. 2020) fix this by explicitly penalizing value estimates for out-of-distribution actions, keeping the learned policy honest about what the data actually supports. The result is a method that extracts more than BC from the same fixed dataset when a reward is available, while sidestepping the safety and sample problems of online exploration. It is the natural choice when you have demonstrations *and* a reward but no safe way to collect fresh experience.

## A worked decision

Make it concrete. Suppose you are deploying a tabletop arm that must pick assorted parts from a bin and place them on a fixture, and you have collected 300 teleop demonstrations.

Start at the bottom of the ladder. You have demonstrations, so behavior cloning is the first thing to train: it is a day of work, it needs nothing you do not already have, and it gives you a baseline that tells you how hard the task actually is. Mandlekar et al. (2021, arXiv:2108.03298) is the empirical anchor here: across a suite of manipulation tasks, BC on a few hundred human demonstrations was a strong baseline, and outperformed RL trained from scratch under the sparse rewards typical of real tasks. Do not skip the baseline on a hunch that you need something fancier; measure first.

Now read the failure mode. If the BC policy succeeds on parts and poses like those in the data but drifts and stalls on novel ones, you are looking at compounding error (§6.3), and the cheapest fix is more coverage, either more demonstrations of the failure regions, or DAgger if you can afford to have an operator correct the policy as it runs. Reach for DAgger before reaching for reward inference; it solves the covariate-shift problem you actually have without forcing you to invent a reward.

You climb past DAgger only when the problem changes character. If the fixture moves to positions no demonstration covered, and you want the policy to *figure out* a placement rather than interpolate a remembered one, you are now asking for the expert's objective, not the expert's trajectories, the §6.4 situation, and IRL or GAIL becomes defensible, assuming you can stand up a simulator to optimize in. And if your demonstrations are a mixed bag and you happen to have a reward, say a force sensor that flags bruised parts, offline RL lets you prefer the gentle episodes over the rough ones without discarding the rough ones entirely.

The thing to resist is starting at the top. Online RL from scratch on this task would mean writing a dense bin-picking reward (weeks, per §5.4), building a simulator faithful enough to transfer (§5.5), and burning millions of samples, to solve a problem that 300 demonstrations and an afternoon of BC may largely dispatch.

## A checklist you can actually apply

Strip the discussion to the signals that should move your decision:

- **You have demonstrations and will deploy near them.** Behavior cloning. This is the default and it is the default for good reasons.
- **BC works in-distribution but drifts off it, and you can correct it live.** Add DAgger. Cheapest possible fix for compounding error.
- **Your demonstrations are mixed quality and you have a reward.** Offline RL (e.g., CQL). Squeezes more out of the same fixed data without online risk.
- **You need the expert's goal to transfer to unseen configurations, and you have a simulator.** IRL / GAIL. You are buying a portable objective, at the cost of a two-player optimization.
- **You have a reward, a simulator, and a sample budget, and want behavior no human demonstrated.** Online RL (Chapter 7). The most powerful and the most demanding.
- **You have a competent BC policy and want to push past human performance on a specific deployment.** BC-then-RL fine-tuning. This is the pattern modern systems actually use.

That last row is where the field has converged, and it is worth stating plainly because it dissolves the false choice the section title implies. The production answer is rarely "BC *or* RL." It is BC *then* RL: pretrain a policy on a large demonstration corpus, where imitation's data-scaling advantage (§6.1) does the heavy lifting, then fine-tune with online RL on the narrow slice of behavior where a reward is cheap to specify and the policy is already competent enough to explore sensibly. π0 (arXiv:2410.24164) is exactly this: a flow-matching BC objective on broad demonstration data, with optional RL fine-tuning per deployment. The imitation signal teaches the policy what good behavior roughly looks like; the reward signal sharpens it where it matters. Neither alone is enough, and the ordering is not arbitrary. A policy that has never seen the task explores hopelessly under RL, the same observation §6.1 made and the reason this book spends four chapters on imitation before returning to deep RL.

## Why the default holds for VLAs

For the foundation action models this book is building toward, the ladder collapses almost entirely onto its bottom rung, and it is worth being explicit about why. Foundation VLAs are trained across hundreds of tasks and dozens of embodiments at once. There is no single reward function that spans "fold a towel," "stack blocks," and "wipe a counter," so reward inference and online RL, both of which presuppose a task-specific objective, do not scale to the multi-task, cross-embodiment regime. Behavior cloning does, because its supervisory signal is just states and actions, which are comparable across tasks and robots in a way that scalar rewards are not. That is the structural reason RT-1, OpenVLA, Octo, and π0 are all, at their core, cloning at scale. The methods higher on the ladder do not disappear; they re-enter as *fine-tuning* tools for a single robot and a single task, which is precisely the subject of Chapter 16.

With this section, the chapter's core question is answered. You can now train a behavior-cloning policy and read its failure modes; explain compounding error and the DAgger fix in your own words; distinguish cloning from reward inference and from offline RL, and say what each demands; and decide, given a dataset and a robot, whether BC is the right place to start, which, more often than not, it is. The summary that follows consolidates these into the handful of claims worth carrying into the rest of the book.
