---
chapter: 17
section: 17.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §17.1 (safety as a non-learned layer with stateable worst-case behavior — the claim this exercise makes real), §17.2 (the predicate shield and its safe fallback, the object you build here), §2.4 (the silent-failure class one of the induced failures reproduces), §16.x (the fine-tuned OpenVLA this shield wraps). A GPU and the fine-tuned checkpoint let you wrap a live policy; the shield logic itself, which is the point, runs on a CPU in seconds against scripted actions.
key_refs:
  - Alshiekh, M. et al. (2018). Safe Reinforcement Learning via Shielding. AAAI.
  - Zhou, Q. et al. (2025). Code-as-Monitor, Constraint-Aware Visual Programming for Reactive and Proactive Robotic Failure Detection. CVPR 2025.
  - Survey (2026). Safety of Vision-Language-Action Models, A Survey. arXiv:2604.23775.
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
---

# 17.x  Hands-on exercise + chapter references

§17.1 asked you to accept that a safety layer has worst-case behavior you can state, where the policy it wraps does not. This exercise makes you build the layer and then attack it, so the claim stops being a sentence and becomes three logged vetoes you produced on purpose. The TOC states the target: add a force-and-workspace safety layer in front of a fine-tuned OpenVLA and demonstrate it catches three concrete failure modes you induce deliberately. The three are the ones §17.2 said a shield exists for: a force spike from unexpected contact, a workspace breach from an oversized action, and a velocity spike from a hallucinated jump. If §17.1 was right, the shield vetoes all three and logs why. If it was wrong, one slips through, and you have found a hole in your own safety layer, which is exactly the kind of thing you want to find on a laptop rather than on a floor.

The same honest fork as the last two chapters applies. Wrapping a live fine-tuned OpenVLA (§16.x) wants a GPU and the checkpoint. The shield itself is pure, cheap, non-learned code, which is the whole reason it can be certified, so it runs on a CPU in seconds and you can test it against scripted actions that stand in for whatever the policy might emit. Build and attack the shield on the CPU path; wrap the real policy if you have the hardware. The lesson is identical either way, because the shield does not care where the action came from.

## What you need

Full path: a GPU, a fine-tuned OpenVLA checkpoint from §16, and a robot or simulator to execute on. Shield path: Python and NumPy. No policy required, because you will feed the shield the offending actions directly, which is both easier and more rigorous, since a scripted attack hits the exact predicate you mean to test.

## Exercise 17.x.1 — Build the shield

Start from §17.2's shield and make it concrete. Four predicates, a defined safe fallback, and a violation log. Keep every check cheap, because a shield that needs a network cannot run on the fast clock.

```python
import numpy as np

class Limits:
    max_force = 20.0            # newtons
    box = (np.array([0.2, -0.3, 0.0]), np.array([0.7, 0.3, 0.5]))  # xyz min, max
    max_step = 0.05            # meters per control step

def in_box(xyz, box):
    lo, hi = box
    return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

def shield(state, action, L=Limits):
    next_xyz = state["ee_xyz"] + action["delta_xyz"]
    checks = {
        "force":     state["wrist_force"] < L.max_force,
        "workspace": in_box(next_xyz, L.box),
        "velocity":  float(np.linalg.norm(action["delta_xyz"])) < L.max_step,
    }
    if all(checks.values()):
        return action, None
    violated = [k for k, ok in checks.items() if not ok]
    return {"delta_xyz": np.zeros(3), "hold": True}, violated   # decelerate-and-hold
```

The fallback returns a hold, not a no-op, for the §17.2 reason: on a moving robot "hold position" is defined and "do nothing" is not. The second return value is the violation list, which is the log entry §17.4 alerts on. Confirm a benign action passes: a small in-box step with low force should return `(action, None)`.

## Exercise 17.x.2 — Induce the force failure

The first attack is the contact spike. Simulate the robot meeting something it should not, a fixture, a wall, a hand, by handing the shield a state whose wrist force is over the limit, with an action that would keep pushing.

```python
state = {"ee_xyz": np.array([0.4, 0.0, 0.2]), "wrist_force": 35.0}  # 35 N > 20 N
action = {"delta_xyz": np.array([0.02, 0.0, 0.0])}                  # keep pushing in
safe, violated = shield(state, action)
assert violated == ["force"] and safe["hold"]
print("force veto:", violated)
```

The point to sit with is that the shield never asked why the force was high. It did not need to know whether the policy hallucinated, the calibration drifted, or a person reached in. A rising force with a commanded push is refused on the reading alone, which is the property that makes the check certifiable: its behavior is stated over the sensor value, not over the network that produced the action.

## Exercise 17.x.3 — Induce the workspace failure, the §2.4 way

The second attack reproduces a silent bug from §2.4 rather than an obvious one. Feed the shield an action with the wrong normalization scale, the kind a wrong `unnorm_key` produces, so the delta is an order of magnitude too large and would fling the end-effector out of the workspace.

```python
state = {"ee_xyz": np.array([0.65, 0.0, 0.2]), "wrist_force": 2.0}
# a detokenizer with the wrong unnorm_key emits deltas at the wrong scale:
action = {"delta_xyz": np.array([0.5, 0.0, 0.0])}   # 0.5 m step, off the table
safe, violated = shield(state, action)
assert "workspace" in violated and "velocity" in violated
print("oversized-action veto:", violated)
```

This is the exercise's most important case, because it is the failure that has no exception and no telltale in the loss. The policy is confident, the numbers are the right shape, and only their magnitude is wrong. Two predicates catch it at once, the workspace bound and the velocity ceiling, which is defense in depth (§17.1) doing its job: the oversized action trips more than one independent check, so even if you had mis-set the box, the velocity limit still stops it. A success-rate benchmark would have scored this as a failed grasp and moved on; the shield turns it into a logged, attributable veto.

## Exercise 17.x.4 — Induce the velocity spike, and read the log

The third attack is the hallucinated jump: an in-box, low-force action that is simply too large a single step, the discontinuity a policy sometimes emits when the scene confuses it.

```python
state = {"ee_xyz": np.array([0.4, 0.0, 0.2]), "wrist_force": 1.0}
action = {"delta_xyz": np.array([0.0, 0.09, 0.0])}   # 9 cm in one step, inside the box
safe, violated = shield(state, action)
assert violated == ["velocity"]

# now run a stream and collect the log the way §17.4 wants it
log = []
for a in scripted_attack_stream():          # your three attacks, interleaved with benign steps
    _, v = shield(current_state(), a)
    if v: log.append({"t": now(), "violated": v})
print(f"{len(log)} vetoes; by predicate:",
      {k: sum(k in e['violated'] for e in log) for k in ['force','workspace','velocity']})
```

The per-predicate tally at the end is the seed of §17.4's alerting: a shield-intervention rate broken out by which check fired tells you not just that the policy is misbehaving but how, and a rising count in one column points at the cause. Force vetoes climbing means contact trouble; workspace and velocity vetoes climbing together means the scale bug from 17.x.3. The log is the difference between a shield that keeps you safe and a shield that also tells you why you needed it.

## Going further

Two extensions worth the time if you have the hardware. Wrap the actual fine-tuned OpenVLA (arXiv:2406.09246) so the shield sees real policy outputs, and measure the shield-intervention rate on a normal run, which is your safety baseline for §17.4's alerting. And add a fourth, semantic check the predicates cannot express, "do not move toward the region tagged as a person", using a VLM-as-monitor on the slow clock in the Code-as-Monitor (Zhou et al., 2025) style, then confirm the cheap shield still runs underneath it and still wins on any disagreement. That is the §17.2 layering made real: expressive checks on top, certifiable checks at the bottom.

## Chapter 17 reading list

Cited across §17.1–§17.6, grouped by the job each reference does. Full entries for everything in the book live in Appendix E.2; this is the chapter-local subset.

### The safety layer and runtime monitoring

- Alshiekh, M., et al. (2018). "Safe Reinforcement Learning via Shielding." AAAI. §17.2's source for the shield: a correct-by-construction layer between policy and actuators that overrides unsafe actions. The idea predates VLAs and is what the predicate monitor implements.
- Zhou, Q., et al. (2025). "Code-as-Monitor: Constraint-Aware Visual Programming for Reactive and Proactive Robotic Failure Detection." CVPR 2025. §17.1 and §17.2's VLM-as-monitor: compiling free-form constraints into visual-program checks, and the reactive-versus-proactive distinction, with the caveat that a learned monitor reintroduces the opacity it watches.
- Runtime Action Authorization for VLA Models, A Survey (2026). arXiv:2606.00090. §17.2's framing of the shield as an authorization gate, and the design space of what authorizes an action against what specification.
- Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring (2026). arXiv:2605.30834. §17.2's learned failure detector and its honest problem: step-level failure annotation is noisy, so a monitor trained on it inherits the noise.

### Evaluation on hardware

- Atreya, P., Pertsch, K., et al. (2025). "RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies." §17.3's model for interleaved, pairwise, drift-robust comparison, scaled across labs.
- Liu, B., et al. (2023). "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310. §17.3's borrowed axis idea: break an A/B comparison out by the kinds of variation that matter, because B can win on easy tasks and lose near a person.

### The residual risks and the attack surface

- Safety of Vision-Language-Action Models, A Survey (2026). arXiv:2604.23775. §17.5's map of the whole safety area, cited as a survey because the field moves fast enough that any single result is provisional.
- TRAP: Adversarial-Patch Hijacking of VLA Chain-of-Thought Reasoning (2026). arXiv:2603.23117. §17.5's canonical physical attack: a printed pattern that steers the action without any digital access.
- VLA-Hijack: Transferable Black-Box Patch Attacks Across VLA Architectures (2026). arXiv:2605.28083. §17.5's transferability result: one patch hijacks OpenVLA, UniVLA, and CronusVLA without the target's weights, so an attacker needs a model like yours, not yours.
- Partially-Observable Adversarial Patch Attacks on VLA Policies (2026). arXiv:2606.03556. §17.5's extension where the patch need not stay in full view to work.
- Lost in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs (2026). arXiv:2605.21446. §17.5's non-adversarial fragility: ordinary fog degrades reasoning, the brittleness you cannot keep out of a workcell.
- ForesightSafety-VLA: A Diagnostic Safety Benchmark (2026). arXiv:2606.27079, and the CVPR 2026 AdvML adversarial-VLA challenge (arXiv:2607.11560). §17.5's early attempts to measure safety failures rather than task successes.

### The backward thread

- Featherstone, R. (2008). *Rigid Body Dynamics Algorithms.* Springer. §17.1's impedance-control layer, the compliant contact behavior from §4.3 reused as a hardware safety property that holds even when the checks above it fail. The certification the chapter cannot give the policy is the certification §4.1's symbolic planner could give its plan, which is the trade §4.4 named.

## Chapter summary

Chapter 17 converted the book's recurring caution, that these models are more capable than they are analyzable, into a deployment discipline and an honest boundary. You can now wrap a VLA in a runtime safety layer of cheap, certifiable predicates with a defined safe fallback, and after this exercise you have watched that layer catch a force spike, an oversized-action scale bug, and a hallucinated jump that you induced on purpose. You can run a hardware A/B comparison that interleaves, pairs, blinds, tests the difference rather than two intervals, and refuses to drop the safety column, so the new policy you ship is actually the better and safer one rather than the luckier one. You can build the production logging, leading-indicator alerting, and pre-defined rollback that catch the silent failures a crash-only monitor misses, and close the loop that turns your worst rollouts into your next training data. And you can state, without softening it, what none of this certifies: that a learned policy cannot be formally verified, that it carries an adversarial and natural-perturbation attack surface your success benchmarks never test, and that deploying one is managed risk watched closely, not a guarantee. Chapter 18 takes the open problems this honesty exposes, cross-embodiment transfer, long-horizon and dexterous tasks, video pretraining, and reasoning joined to action, and asks where the field goes from here.
