---
chapter: 17
section: 17.2
title: "Runtime monitors and shielding"
target_words: 2000
status: draft
prereqs: §17.1 (safety is a layer of non-learned checks around the policy; this section builds that layer). §3.x (the predicate drills you wrote, which become the monitor's kernel here), §2.4 (the three silent failures — flipped image, wrong prompt template, wrong unnorm_key — a monitor is the deployment-time response to that class of bug), §6.3 (compounding error, which a monitor detects as drift and interrupts), §14.4 (the latency budget a monitor has to fit inside), §14.7 (a slow reasoner can propose an intent the fast layer must refuse).
key_refs:
  - Zhou, Q. et al. (2025). Code-as-Monitor, Constraint-Aware Visual Programming for Reactive and Proactive Robotic Failure Detection. CVPR 2025.
  - Survey (2026). Runtime Action Authorization for Vision-Language-Action Models, A Survey. arXiv:2606.00090.
  - Authors (2026). Hide-and-Seek in Trajectories, Discovering Failure Signals for VLA Runtime Monitoring. arXiv:2605.30834.
  - Alshiekh, M. et al. (2018). Safe Reinforcement Learning via Shielding. AAAI.
---

# 17.2  Runtime monitors and shielding

§17.1 argued that safety is a layer of non-learned checks wrapped around the policy. This section builds the layer. The component that does the work has a name in the safe-control literature, a shield (Alshiekh et al., 2018): a piece of code that sits between the policy's output and the robot's actuators, watches every proposed action against a set of constraints, and overrides any action that would violate one before it executes. The policy proposes; the shield has veto power; the veto is the safety.

## The monitor is a set of predicates

Strip a runtime monitor down and it is a list of predicates, boolean functions of the robot's state and the proposed action, each of which must be true for the action to pass. You already wrote a first sketch of these back in §3.x, the drills that checked whether a state satisfied some condition; here those same predicates become the kernel of a thing running on a real robot at control rate. The shield evaluates them every cycle, and when one fails it does not ask the policy to try again, it substitutes a safe action of its own.

```python
def shield(state, proposed_action, limits):
    # each predicate returns True if the action is SAFE on that axis
    checks = {
        "force":     state.wrist_force < limits.max_force,
        "workspace": in_box(forward_kinematics(state, proposed_action), limits.box),
        "velocity":  norm(proposed_action.delta) < limits.max_step,
        "collision": not collides(planned_path(state, proposed_action), limits.scene),
    }
    if all(checks.values()):
        return proposed_action           # policy's action is allowed through
    log_violation(checks, state, proposed_action)
    return safe_fallback(state)           # e.g. hold position, or decelerate to stop
```

Two things about that sketch matter more than its details. The fallback is not "do nothing", it is a specific, defined safe action, usually decelerate-and-hold, because on a moving robot "do nothing" is ambiguous and "hold position under compliant control" is not. And the violation is logged, every time, with the state and action that triggered it, because those logs are the raw material §17.4 turns into alerting and the failure-mining loop from §15.6. A shield that silently clips is a shield you cannot debug.

## What the monitor catches that testing did not

A monitor earns its place on exactly the failures your evaluation could not enumerate. Recall the three silent bugs from §2.4: a flipped image, a wrong prompt template, a wrong `unnorm_key` feeding the detokenizer the wrong per-embodiment statistics. Each produced confident, wrong actions and raised no exception. A monitor is the deployment-time answer to that whole class, because it does not care why the action is wrong, only that it violates a constraint. A detokenizer fed the wrong normalization statistics emits actions whose magnitudes are systematically off, and a velocity ceiling catches them at the first oversized step, long before you have diagnosed the `unnorm_key`. The monitor converts a silent failure into a loud, logged intervention, which is the whole point.

The same logic covers drift. §6.3's compounding error says a policy that wanders off its training distribution produces states it has never seen and actions that get worse from there. A monitor cannot fix the drift, but it can detect it: an action-statistics check that flags when the policy's outputs leave the distribution it produced in evaluation, or a stall detector that notices the robot is making no progress, gives you a trigger to stop and hand off rather than letting a confidently-diverging policy run to the point of damage. Stopping safely on a detected anomaly is almost always better than continuing, because the failures that show up in deployment, as §14.5 found, are the ones long runs expose and short demos hide.

## The latency the monitor has to fit inside

A monitor runs inside the control loop, so it spends the budget §14.4 was strict about. A predicate shield is cheap by construction: a force comparison, a forward-kinematics call, a bounds check, a fast collision query, all of which run in microseconds and none of which involves a network. That cheapness is not incidental, it is why the safety layer can be non-learned and still keep up with a 200 Hz fast loop. The moment a check needs a neural forward pass, it can no longer live on the fast clock, which is the structural reason the intelligent monitors below run on the slow clock instead.

This is also where §14.7's problem lands. A dual-system policy's slow reasoner can propose an intent the fast controller should not execute, a reach that would exceed a force limit, a path through a person. The shield is the arbiter: it sits on the fast clock, it does not care that the intent came from a 7B model that reasoned carefully, and it vetoes the action on the same cheap predicates it applies to everything else. The reasoner's intelligence buys it no exemption from the force limit, which is exactly the property you want.

## VLM-as-monitor: power and the opacity it reintroduces

Predicate shields have a ceiling: they can only check constraints you can write as cheap code. "Do not exceed 20 N" is easy; "do not put the knife down with the blade toward the person" is not a geometric box. A research line attacks this by using a vision-language model as the monitor, translating a free-form constraint into something checkable and watching the scene for violations. Code-as-Monitor (Zhou et al., 2025) is the clearest instance: it uses a VLM to compile constraints into small visual-programming checks, and it distinguishes reactive detection, catching a failure as it happens, from proactive detection, flagging that the current plan is heading toward one. That is genuinely more expressive than a predicate list, and it moves the opacity problem rather than solving it, because a learned monitor has the same undefined tail §17.1 warned about in the learned policy. You have added a second network that can also be wrong on the input nobody anticipated.

The honest way to use a VLM monitor is as an outer, slow-clock layer over an inner, fast, non-learned shield, never as a replacement for it. The cheap predicates guarantee the worst-case behavior you can state; the VLM catches the semantic violations the predicates cannot express; and when the two disagree, the cheap layer wins, because it is the one whose failure modes you understand. Defense in depth (§17.1) with a learned outer layer is fine. A learned layer alone is just the original problem with an extra step.

## Authorization and learned failure signals

Two research directions are worth naming because they are where this subfield is moving. The first is runtime action authorization, surveyed in arXiv:2606.00090, which frames the shield as an authorization gate: every action must be authorized against an explicit policy before it reaches an actuator, and the survey organizes the design space of what does the authorizing and against what specification. It is the same shield idea given a security vocabulary, and the vocabulary helps, because it makes you name the authority and the policy rather than scattering `if` statements through a control loop.

The second is learning the monitor itself. Hide-and-Seek in Trajectories (arXiv:2605.30834) trains a model to detect failure signals in a policy's trajectory for runtime monitoring, and its central contribution is honest about the hard part: step-level failure annotation is noisy, because the moment a trajectory "went wrong" is rarely a clean single frame, and a monitor trained on noisy step labels inherits that noise. This is the same measurement-is-hard theme from Chapter 15 reappearing inside the safety layer, and it is why a learned failure detector, like a VLM monitor, belongs on top of a predicate shield rather than in place of one.

You now have the layer §17.1 promised, built from cheap predicates on the fast clock and expressive learned checks on the slow one. A monitor tells you when to intervene on a single robot in real time. The next question is a different one: across many runs, is version B of your policy actually better and safer than version A, or does it just look that way? §17.3 builds the A/B evaluation on hardware that answers it, and it runs into every statistical trap Chapter 15 warned about, now with a safety column added.
