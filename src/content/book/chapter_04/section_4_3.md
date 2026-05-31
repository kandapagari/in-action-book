---
chapter: 4
section: 4.3
title: "Inverse dynamics and computed-torque control"
target_words: 2000
status: draft
prereqs: §4.2 (the geometric layer hands us a kinematic trajectory q(t), q̇(t), q̈(t)); §3.1 (matrices, Jacobians, the chain rule); basic Newtonian mechanics (F = ma in vector form); willingness to read one matrix equation slowly
key_refs:
  - Craig, J. J. (2005). Introduction to Robotics — Mechanics and Control, 3rd ed. Pearson.
  - Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). Robot Modeling and Control. Wiley.
  - Lynch, K. M., & Park, F. C. (2017). Modern Robotics — Mechanics, Planning, and Control. Cambridge University Press.
  - Featherstone, R. (2008). Rigid Body Dynamics Algorithms. Springer.
  - Luh, J. Y. S., Walker, M. W., & Paul, R. P. C. (1980). On-Line Computational Scheme for Mechanical Manipulators. ASME J. Dynamic Systems, Measurement, and Control 102(2).
  - Khatib, O. (1987). A Unified Approach for Motion and Force Control of Robot Manipulators — The Operational Space Formulation. IEEE J. Robotics and Automation 3(1).
  - Hogan, N. (1985). Impedance Control — An Approach to Manipulation. ASME J. Dynamic Systems, Measurement, and Control 107(1).
  - Slotine, J.-J. E., & Li, W. (1987). On the Adaptive Control of Robot Manipulators. IJRR 6(3).
---

# 4.3  Inverse dynamics and computed-torque control

§4.2 ended with a kinematic trajectory: a sequence of joint configurations
`(q_0, q_1, …, q_K)` with timestamps, smoothed and collision-free. That
trajectory describes *where* the joints should be at each instant. It says
nothing about *what force* the motors have to apply to put them there.
Closing that gap is the inverse-dynamics problem, and the controller that
uses its solution is called *computed-torque control*.

The reason the gap exists is mechanical. A robot arm is not a free
particle. Gravity pulls on every link. Lifting a payload changes the
inertia the shoulder has to fight. Moving the elbow fast generates
Coriolis forces on the wrist. Friction in the harmonic drive of joint
three eats torque proportional to angular velocity. A position-commanded
joint that ignores all of this works for a slow industrial arm with high-
ratio gearboxes — the gearbox absorbs the discrepancy. It does not work
for a fast, low-inertia, torque-controlled arm like a Franka Panda or
KUKA iiwa, which is precisely the class of robots most modern manipulation
research runs on. For those, the controller has to ask what torque produces
the desired acceleration, then send that torque.

## The manipulator equation

Every textbook on robot dynamics arrives at the same equation, often
called the *manipulator equation* or the *equation of motion in joint
space*:

```
M(q) q̈ + C(q, q̇) q̇ + g(q) + τ_f(q̇) = τ
```

Each term is a vector in `R^n`, where `n` is the number of joints. Reading
it left to right:

- `q, q̇, q̈ ∈ R^n` are the joint positions, velocities, and accelerations.
- `M(q) ∈ R^{n×n}` is the *mass matrix*, also called the joint-space
  inertia matrix. It is symmetric positive-definite, and its entries
  depend on the current configuration because moving the elbow changes
  how much inertia the shoulder feels.
- `C(q, q̇) q̇` packs the Coriolis and centripetal terms — forces that
  appear because joints are moving with respect to each other.
- `g(q)` is the gravity vector — the torque each joint has to produce just
  to hold the arm static against gravity in configuration `q`.
- `τ_f(q̇)` collects friction and other velocity-dependent losses. It is
  notoriously hard to model precisely; viscous and Coulomb terms are the
  usual cartoon.
- `τ ∈ R^n` is the vector of joint torques the controller commands.

The whole equation is Newton's second law lifted to a chain of rigid
bodies. The derivation, via Lagrangian or Newton-Euler mechanics, is a
standard chapter in Spong et al. (2006), Craig (2005), and Lynch and Park
(2017). For our purposes, what matters is the *use* of the equation, not
its derivation.

*Inverse dynamics* is the question: given `q, q̇, q̈`, compute `τ`. That is
a direct evaluation of the right-hand side — substitute the numbers, get
the torque. It is the easy direction. *Forward dynamics* is the inverse
direction: given `q, q̇, τ`, compute `q̈`. That requires inverting `M(q)`
and is what physics simulators (MuJoCo, Bullet, Drake) do every step.
Manipulation control is mostly inverse dynamics; physics simulation is
mostly forward dynamics.

## Computing inverse dynamics: RNEA in one paragraph

A naive implementation of `M(q) q̈ + C(q, q̇) q̇ + g(q)` constructs each
matrix and multiplies, costing `O(n^3)` per call. The actual implementation
everyone uses is the *Recursive Newton-Euler Algorithm* (RNEA), due to Luh,
Walker, and Paul (1980) and developed in full generality by Featherstone
(2008). RNEA computes `τ` directly without ever forming `M` or `C`, in two
sweeps along the kinematic chain. The forward sweep propagates velocities
and accelerations outward from the base to the end-effector, computing the
linear and angular acceleration of each link's center of mass. The backward
sweep propagates forces and moments inward, summing up the torque each
joint must apply to produce that link's acceleration plus the loads
transmitted by its children. The total cost is `O(n)`. For a 7-DOF arm,
one RNEA call takes a few microseconds in C++, fast enough to run inside
a 1 kHz control loop with thousands of cycles to spare. RNEA is what
`pinocchio`, `RBDL`, and the Drake dynamics library compute under the
hood; you almost never write the recursion yourself.

## The computed-torque law

Suppose the motion planner has handed us a reference trajectory:
`q_d(t), q̇_d(t), q̈_d(t)` — desired position, velocity, and acceleration
at time `t`. The controller measures the actual `q(t), q̇(t)` and defines
the *tracking error*:

```
e = q_d - q
ė = q̇_d - q̇
```

The *computed-torque control law*, sometimes called the *inverse-dynamics
control law*, is:

```
τ = M(q) [ q̈_d + K_d ė + K_p e ] + C(q, q̇) q̇ + g(q) + τ_f(q̇)
```

`K_p` and `K_d` are diagonal positive-definite gain matrices. Read it
piece by piece. The bracketed term is the desired acceleration plus a PD
correction that pulls the error toward zero. Multiplying by `M(q)` turns
that desired acceleration into the torque required to produce it.
Adding `C(q, q̇) q̇ + g(q) + τ_f(q̇)` cancels the gravitational, Coriolis,
and frictional torques the arm is already feeling. The net effect is
striking: substituting this `τ` into the manipulator equation and
simplifying gives `ë + K_d ė + K_p e = 0`, a linear, decoupled, second-order
error dynamics on every joint. The robot, which was a coupled nonlinear
mechanical system, has been transformed into `n` independent linear
springs. This trick is called *feedback linearization* and is one of the
load-bearing ideas in classical robot control.

In practice the controller does not invert `M(q)` symbolically; it calls
RNEA twice. Once with `q̈_d + K_d ė + K_p e` as the acceleration argument
to get the feedforward + linearizing torque, and a second time with
acceleration zero to get the gravity term separately if needed. Or, more
commonly, once with the full corrected acceleration, since RNEA returns
the full right-hand side in one pass. A minimal pseudocode sketch:

```python
def computed_torque(q, qd, q_des, qd_des, qdd_des, Kp, Kd):
    e   = q_des  - q
    ed  = qd_des - qd
    qdd_cmd = qdd_des + Kd * ed + Kp * e
    tau = rnea(q, qd, qdd_cmd)   # one O(n) call
    return tau
```

Two RNEA calls per timestep at 1 kHz is unproblematic on a modern CPU.

## Why the model is never quite right, and what to do about it

Computed-torque control assumes you know `M, C, g, τ_f` exactly. You do
not. The arm's link masses are known to maybe 1%; the inertia tensors to
worse; friction is wildly nonlinear and changes with temperature; the
moment a payload is attached, every parameter in `M(q)` shifts. The
literature has three standard responses.

The first is to *not invert the full model* — keep only the easy parts.
*PD-plus-gravity* control uses

```
τ = K_p e + K_d ė + g(q)
```

dropping the inertial and Coriolis cancellation entirely. The arm tracks
worse, especially at high speed, but the law is robust: as long as the
gravity model is approximately right and the gains are stable, the arm
holds its trajectory acceptably. Most teaching robots and many ROS
controllers ship this configuration by default.

The second is *robust* or *adaptive control*. Slotine and Li (1987) showed
that you can adapt the parameter estimates online: define a parameter
vector `θ̂` that contains the estimated masses, inertias, and friction
coefficients; update `θ̂` according to a Lyapunov-derived rule each step;
use the current estimate inside the computed-torque law. Under reasonable
conditions the tracking error and the parameter error both converge.
Adaptive control is elegant and was, for a decade, the right answer for
high-performance arm control. It is now overshadowed in practice by
*learning* the residual: estimate the model from data and let a small
network correct what the parameterized model gets wrong.

The third response is *impedance control* (Hogan, 1985). Instead of
forcing the arm to track a position trajectory exactly, command it to
behave like a virtual spring-damper attached to the desired pose:

```
τ = J(q)^T [ K_imp (x_d - x) + D_imp (ẋ_d - ẋ) ] + g(q) + (model terms)
```

where `x = T(q)` is the end-effector pose and `J(q)` is the Jacobian.
Now if the gripper hits something unexpectedly, it deflects rather than
crashing. This is the control mode the Franka Panda exposes by default;
it is also what makes safe physical human-robot interaction possible. We
revisit impedance control in Chapter 17 when we discuss safety as a
layer.

## Operational-space control, briefly

Joint-space computed-torque control places gains on each joint
independently. That is the wrong unit of analysis when the task is
specified in Cartesian space — "move the gripper 5 cm forward, then
press down with 5 N". The *operational-space formulation* of Khatib (1987)
rewrites the same equations in end-effector coordinates:

```
F = Λ(q) ẍ_d + μ(q, q̇) + p(q)
τ = J(q)^T F + (null-space term)
```

where `Λ` is the task-space inertia matrix and `μ, p` are the
Coriolis and gravity contributions projected into task space. The joint
torques are then computed by multiplying the task-space force `F` by the
Jacobian transpose. The redundant nulls­pace (extra DOF beyond the six
needed for the task) is filled by a secondary objective — joint-limit
avoidance, manipulability maximization, or simply staying away from a
configuration the engineer dislikes. Operational-space control is the
standard formulation for humanoid whole-body control and underpins
modern dual-system architectures we will see in Chapter 14, where a
high-level VLM names a task pose and a low-level operational-space
controller realizes it.

## Where this connects to the rest of the book

A learned policy can choose to take responsibility for as much or as
little of the manipulator equation as the engineer assigns it. The
spectrum is wide and worth seeing as a spectrum.

At one extreme, OpenVLA and Diffusion Policy emit *joint positions* or
*end-effector poses* and rely on a downstream position controller —
typically PD-plus-gravity, sometimes full computed-torque — to actually
produce the torques. The learned model never sees `τ`. This is the
default in nearly all VLA literature, and it works because the
underlying classical controller is doing the dynamics-aware work.

A step further in, RL controllers for legged locomotion (ANYmal, Cassie,
Atlas) typically emit *joint torques directly*. They learn an
end-to-end policy whose output is `τ`, and the manipulator equation
appears only inside the simulator they trained in. There is no
inverse-dynamics layer in the deployed stack; the policy has internalized
whatever inertial and gravitational compensation the task requires.

Further still, *residual* and *hybrid* approaches treat classical
computed-torque as a strong prior and learn only the correction. A
common pattern, used in fine manipulation and contact-rich assembly:
classical impedance control provides 80% of the right torque, and a
small learned network adds a residual to handle hard-to-model contact.
This is one of the more practical recipes for getting a learned policy
into a production robot quickly.

The reason §4.3 deserves its own section in a book mostly about VLAs is
that even when the controller is learned, the *language* the rest of the
robotics stack uses to describe its job is the language of the
manipulator equation. "We had a tracking error of 3 cm at the wrist when
the payload doubled" is not a sentence you can debug without `M(q)`. A
foundation policy that produces beautiful joint targets is still
shipping torques through a classical controller most of the time, and
when that controller misbehaves, the only useful description of the
misbehavior is in inverse-dynamics terms.

The next section, §4.4, takes the three layers we have now built —
symbolic, geometric, dynamic — and asks where each is still load-bearing
in robots actually shipping in 2026, even as learned components take
over more of the stack.
