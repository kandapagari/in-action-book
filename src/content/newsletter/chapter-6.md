---
chapter: 6
subject: "Why demonstrations, not rewards, power today's robots"
---
Reinforcement learning hands you a rigorous training signal, but it's a single scalar, it's hard to design, and squeezing a policy out of it takes an enormous number of samples. Chapter 6 explains why that ledger pushed the field somewhere else: demonstrations, not reward, are what actually drive RT-1, OpenVLA, and most of the commercially relevant robot learning of the past five years.

The case rests on two asymmetries. Reward labels an outcome; a demonstration labels every decision on the way to it, so one fifteen-second teleoperated trajectory carries hundreds of time steps, each an action the expert judged right. Demonstration data also scales linearly and parallelizes, while reward-driven sample complexity blows up as the state space grows. With that established, the chapter lines up behavior cloning, DAgger, inverse reinforcement learning, and offline RL, then traces the arc from Pomerleau's ALVINN in 1988 to billion-parameter VLAs. The framing stays practical the whole way: you've got a dataset, a robot, and a deadline, so which method, and how much data? Pick the wrong branch of the imitation family for your constraints and you fail in a common, avoidable way. This chapter is how you dodge it. The full argument is on the site.
