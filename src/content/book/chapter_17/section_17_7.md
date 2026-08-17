---
chapter: 17
section: 17.7
title: "Training-time attacks: poisoned data, backdoors, and a defense literature that has not caught up"
target_words: 2000
status: draft
prereqs: §17.5 (inference-time attack surfaces, the half of the threat model this section completes), §17.1 (safety as a layer; why the non-learned layer is what stands between an attack and a person), §16.2–§16.3 (where your fine-tuning data comes from, and who touched it), §15.3 (community dataset hubs as a supply chain), §12.4 (Open X-Embodiment, aggregation across 21 institutions), §17.4 (logging and rollback, the detection story when prevention fails).
key_refs:
  - Li, J., Zhao, Y., Zheng, X. et al. (2025). AttackVLA, Benchmarking Adversarial and Backdoor Attacks on Vision-Language-Action Models. arXiv:2511.12149.
  - Chen, K., Long, Y., Li, S., Shang, M. (2026). ATAAT, Adaptive Threat-Aware Adversarial Tuning Framework against Backdoor Attacks on Vision-Language-Action Models. arXiv:2605.08612.
  - Survey (2026). Vision-Language-Action Safety, Threats, Challenges, Evaluations, and Mechanisms. arXiv:2604.23775.
  - Living bibliography: github.com/LiQiiiii/Awesome-VLA-Safety.
---

# 17.7  Training-time attacks: poisoned data, backdoors, and a defense literature that has not caught up

§17.5 gave you an attacker who works at inference time. They print a patch, place it in the workcell, and the policy misbehaves while the weights sit untouched. That is the attack the literature found first, and it is not the worst one. The worse one arrives earlier, before the robot ever runs, when someone influences what the policy learns. This section is about training-time threats, and about a fact the chapter should not hide: on this axis the attack results are considerably stronger than the defenses, and a book that reported the attacks without saying so would leave you with a false sense of the state of play.

## Why the training pipeline is the softer target

Look at where a modern VLA's training data actually comes from and the exposure becomes obvious. §12.4's Open X-Embodiment aggregates contributions from twenty-one institutions. §15.3's LeRobot hub takes community uploads. §16.2 has you collecting teleoperation data, plausibly with contractors, students, or a vendor. Chapter 16's whole recipe assumes you start from public pretrained weights and fine-tune. Every one of those steps is a place where data or parameters you did not produce enter a policy you will point at a person.

Compare that to the assumptions a patch attack needs. To place a physical patch, an attacker needs access to your workcell, or at least to something in the camera's field of view. To poison a dataset, they need to upload a file. The economics run the wrong way for the defender: contributing to the supply chain is cheap, legitimate, and mostly unaudited, whereas standing in a robot cell is conspicuous. The classical software analogy is exact enough to be useful, this is a dependency-confusion problem wearing robotics clothes, and the robotics field has not built the equivalent of signed packages and reproducible builds.

The threat also survives the mitigations §17.5 recommended. Constraining the operating domain does nothing about a poisoned checkpoint, because the poison travels with the weights into whatever domain you constrain. Keeping humans out of the cell helps only until the robot is trusted around them. And task-success evaluation, the entire machinery of Chapter 15, is designed to be blind here: a well-built backdoor leaves clean-task performance intact, which is the property that makes it a backdoor rather than damage.

## What the attacks can currently do

AttackVLA (arXiv:2511.12149, Li et al.) is worth reading first, less for its attack than for its audit of the field. The authors build one framework aligned to the VLA development lifecycle, data construction, training, inference, reimplement the existing attacks on VLAs plus several adapted from the vision-language literature, and run them all in simulation and on real hardware. Two of their observations matter more than any single number.

The first is a reproducibility problem that should sound familiar from §11.3. Action tokenizers differ across VLA architectures, and because they differ, attack results from different papers were not comparable. A field cannot tell whether it is making progress if every result is measured against a different action representation, and the tokenizer choice §11.3 described as "a small idea with large consequences" turns out to have consequences for security research too.

The second is a gap in what the attacks accomplished. Almost all prior work induced *untargeted* failure, the robot does something wrong, or froze the policy into a static action state, the robot stops. Both are denial of service. Neither gives the attacker control. AttackVLA's authors point out that nobody had shown precise targeted control of a VLA's behavior, and then close the gap themselves with BackdoorVLA: a trigger that, when present, compels the policy to execute an attacker-specified *long-horizon action sequence*. They report a 58.4% average targeted success rate, reaching 100% on selected tasks, validated in simulation and in real robotic settings.

Sit with what that means before moving on. Not "the robot fails when I show it a sticker." Closer to "the robot performs a multi-step task of my choosing when I show it a sticker, and behaves normally the rest of the time." A denial-of-service attack on a robot is a safety problem you handle with the emergency stop. A targeted long-horizon backdoor is closer to an insider threat, and the layered architecture of §17.1 does not obviously stop it, because the commanded sequence may be perfectly legal: within the force limit, inside the workspace, smooth, and wrong.

ATAAT (arXiv:2605.08612, Chen et al., May 2026) is the second result to know, and it needs a label correction that I made myself while tracking this literature. The title says "Adaptive Threat-Aware Adversarial Tuning Framework *against* Backdoor Attacks," which reads as a defense. It is an attack. The paper's contribution is a diagnosis of why earlier VLA backdoors underperformed, which they attribute to "gradient interference," an optimization failure where the poisoning objective and the task objective fight each other during end-to-end training, and a method that picks a gradient-decoupling strategy according to the adversary's capabilities. The reported result is a targeted attack success rate above 80% at a 5% poisoning rate, handling semantic-level triggers. Five percent is the number to remember: an attacker who controls a twentieth of your fine-tuning data can install reliable targeted behavior.

I flag my own misreading deliberately, because it illustrates something about reading this literature quickly. Adversarial-ML paper titles routinely describe the threat they study rather than the side they take, and a reader skimming for defenses will collect attacks by mistake. If you are building a bibliography for a deployment review, check which direction each paper actually points.

## The uncomfortable asymmetry

Put the two papers together with §17.5's inference-time results and the shape of the field is clear. There is a mature, quantified, real-hardware-validated body of work on breaking VLAs, spanning patches, transferable black-box patches, sensor perturbation, semantic triggers, data poisoning, and targeted long-horizon backdoors. There is no comparable body of work on reliably detecting or removing any of it. The VLA safety survey (arXiv:2604.23775) organizes both sides and the imbalance is visible in the organization: the threat taxonomy is detailed and populated, the defense taxonomy is largely a list of directions imported from image classification, where the models are smaller, the input space is narrower, and the failure is a wrong label rather than a moving arm.

This is not a criticism of the researchers. Attacks are easier to publish because success is unambiguous, and defenses in adversarial ML have a long history of being broken by the next paper. But it changes what an honest deployment posture looks like, and it extends §17.5's conclusion rather than repeating it. There, the argument was that you cannot certify the policy, so you certify the layer beneath it. Here the argument has to go further, because the mitigations that work against a policy's own mistakes do not obviously work against a policy that has been taught to misbehave on cue.

Three practices follow, and they are provenance practices rather than technical defenses, which is itself the finding.

Know where your weights and data came from, and record it. Which checkpoint, which commit, which dataset revision, which contributor, which teleoperator. §17.4 built logging for post-incident debugging; the same discipline applied to training inputs is what lets you answer "what else was trained on that data" after a bad day. Treat a fine-tuning corpus the way you would treat a dependency manifest.

Treat the clean-task evaluation as silent on this question. A policy that passes your A/B (§17.3) at parity has told you nothing about triggers, because that is precisely the design goal of the attack. If you want any signal, you need evaluation that looks for anomalous behavior under unusual inputs rather than average behavior under representative ones, which is the direction diagnostic safety benchmarks like ForesightSafety-VLA (§17.5) are groping toward and nobody has solved.

Keep the non-learned layer load-bearing, and recognize its limit here. The force limit and workspace bound still constrain a triggered policy, and that is worth a great deal, it converts many possible attacks into ones that merely stop the robot. But a legal-looking sequence of legal motions can still be the wrong sequence, and no ten-line safety check knows the difference between putting a part in a bin and putting it somewhere it should not go. Where the consequence of a wrong-but-legal action is serious, the mitigation has to be procedural, restricted physical access to the cell, restricted authority over what the robot may be commanded to do, and a human who notices.

That is a thinner answer than the rest of this chapter offered, and the thinness is the point. §17.5 said we cannot certify that a policy is safe. This section says we also cannot currently certify that a policy is *ours*, in the sense of doing only what we trained it to do, and that the research needed to change that is behind the research showing why we should worry. Chapter 18 treats this as an open problem alongside the others, and of the open problems in this book it is the one where a careful student could most plausibly publish something useful within a year, because the defense side is close to empty.
