---
chapter: 4
section: 4.4
title: "Where classical methods are still load-bearing in modern robots"
target_words: 2000
status: draft
prereqs: §4.1 (symbolic plans, PDDL), §4.2 (IK and motion planning), §4.3 (computed-torque and impedance control); a working mental model of "the stack" — perception, planning, control — even if you have not implemented one yourself
key_refs:
  - Garrett, C. R., Lozano-Pérez, T., & Kaelbling, L. P. (2020). PDDLStream — Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning. ICAPS.
  - Ahn, M. et al. (2022). Do As I Can, Not As I Say — Grounding Language in Robotic Affordances. arXiv:2204.01691. (SayCan.)
  - Liang, J. et al. (2022). Code as Policies — Language Model Programs for Embodied Control. arXiv:2209.07753.
  - Kim, M. J. et al. (2024). OpenVLA — An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Black, K. et al. (2024). π0 — A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
  - NVIDIA (2025). GR00T N1 — An Open Foundation Model for Generalist Humanoid Robots. arXiv:2503.14734.
  - Khatib, O. (1987). A Unified Approach for Motion and Force Control of Robot Manipulators — The Operational Space Formulation. IEEE J. Robotics and Automation 3(1).
  - Hogan, N. (1985). Impedance Control — An Approach to Manipulation. ASME J. Dynamic Systems, Measurement, and Control 107(1).
  - Padalkar, A. et al. (2023). Open X-Embodiment — Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
---

# 4.4  Where classical methods are still load-bearing in modern robots

The previous three sections built a three-layer cake. At the top, a
symbolic planner produces a sequence of named actions. In the middle, a
geometric layer turns each action into a joint-space trajectory. At the
bottom, an inverse-dynamics controller turns the trajectory into torques.
A reader who has been following along might reasonably ask the next
question: in 2026, with VLAs trained on a million teleoperated episodes
and π0 outputting smooth continuous actions at 50 Hz, how much of that
cake is still in the kitchen?

The honest answer is: most of it. The learned components have grown in,
mostly at the top and the middle, but they have rarely *replaced* a
classical layer outright. They sit on top of it, or beside it, or inside
it as a residual term. This section catalogues where each of the three
classical layers is still doing real work in deployed systems, and
where it is being squeezed out. The taxonomy is useful because the rest
of the book will keep returning to it. When Chapter 14 discusses
dual-system architectures, the "low-level" system is almost always one
of the classical controllers from §4.3. When Chapter 16 discusses
fine-tuning, the cheapest gains often come from leaving the geometric
and dynamic layers alone and training only the perception-to-pose head.
A learned action model is rarely a learned *stack*; it is a learned
*part* of a stack that still has classical bones.

## The symbolic layer: alive and well at the top

The cleanest survivor is the top layer. Almost no commercial robot
executes a long-horizon task as a single forward pass through a learned
policy. Warehouse picking, restaurant assembly, surgical assistance,
and household tidying all share a structure: a high-level component
decides *what to do next*, and a low-level component decides *how*.
That high-level component is, structurally, a symbolic planner. The
implementation has changed — increasingly it is an LLM rather than Fast
Downward — but the *contract* is the same. The output is a discrete
action drawn from a finite alphabet of skills, optionally with
arguments. The downstream system then has to ground each call in
geometry.

SayCan (arXiv:2204.01691) is the cleanest published example. The high
level is a language model that, given a goal like "bring me the
sponge", produces a candidate sequence of skill calls drawn from a
predefined library — `find(sponge)`, `pick(sponge)`, `bring_to(user)`.
The model also scores each candidate using a value function that
estimates how feasible the skill is from the current state. The output
is a STRIPS-flavored plan, just with the planner replaced by a much
better engine. Code as Policies (arXiv:2209.07753) makes the same
substitution more literal: the LLM emits Python that calls a library of
parameterized skills, and the Python is the plan. In both cases the
*shape* of the top layer is what §4.1 described.

The serious modern engineering happens in *task and motion planning*
(TAMP), where the symbolic and geometric layers are interleaved rather
than stacked. PDDLStream (Garrett, Lozano-Pérez, and Kaelbling, 2020)
lets a symbolic planner call out to motion-planning samplers
mid-search: the planner cannot commit to `pick(block_a)` until a
geometric subroutine has confirmed there is a collision-free grasp;
once the grasp is found, its result is added back into the symbolic
state. This kind of architecture is the standard top-of-stack for
household robots that need to actually open a drawer and find the spoon
inside it. The symbolic layer is doing logical bookkeeping that no
end-to-end network has yet been shown to handle reliably for tasks
longer than a handful of steps.

The squeeze on this layer is coming from below, in two ways. First,
VLAs are getting better at short-horizon multi-step tasks — π0
(arXiv:2410.24164) executes "fold this shirt" or "set the table" in
one shot, where five years ago each would have been three or four
symbolic steps. Second, LLM-based planners are eroding the strict
distinction between "symbolic plan" and "natural-language script". A
robot's task list is increasingly a list of English sentences fed to a
VLA, with the LLM acting more like a teacher than like a logician. But
even with those changes, *some* discrete planner decides which
sentences come in what order, and that planner inherits its
vocabulary from STRIPS.

## The geometric layer: hidden but indispensable

The middle layer is the one most newcomers underestimate. VLA papers
tend to elide it. A model "outputs end-effector poses" and the reader
imagines that the model is doing the geometry. It is not. Almost
universally, the VLA outputs a *target pose at 5–50 Hz* and a classical
IK solver, a workspace bounding box, and a short-horizon trajectory
generator sit underneath, turning that target into joint commands and
keeping the motion smooth and collision-free.

OpenVLA (arXiv:2406.09246) is a representative case. The model emits
seven numbers — six for the delta end-effector pose, one for the
gripper. Those numbers feed into a server-side controller that does the
IK, clips the result against the robot's joint limits, blends it with
the previous command for smoothness, and forwards a joint-space target
to the arm's low-level controller. The model does not know what
"unreachable" means. If the user puts the cup on the floor and asks
the robot to grab it, the IK solver — usually TRAC-IK or whatever ships
with the manufacturer's SDK — is the component that returns "no
solution", and the VLA never sees the failure as a learning signal
during deployment.

The same is true of motion planning. A learned policy emitting poses
at 5 Hz is producing waypoints, not trajectories. Real motion planning
— collision-free path, smooth interpolation, joint-limit projection,
self-collision avoidance — is happening somewhere, and unless the
research paper explicitly says "we replaced motion planning with the
network", that somewhere is a classical planner. RRT-Connect or OMPL is
still running. The most ambitious systems that *try* to learn motion
planning end-to-end, such as some legged-locomotion controllers, do so
because the geometry is simple (a few-DOF foot trajectory) and the
contact dynamics are not. For manipulation in cluttered environments,
classical motion planning remains the default. A clear way to see this
is to inspect the GR00T N1 (arXiv:2503.14734) reference stack: the
foundation model produces high-level intent and short-horizon action
chunks, but the deployment-time controller still includes a
collision-aware geometric layer that vetoes commands which would put a
link through a table.

Two places where the geometric layer is genuinely shrinking are
free-space motions and learned contact-rich manipulation. For
free-space reaching, modern VLAs produce trajectories that are smooth
enough on their own that downstream geometry is reduced to safety
checking. For contact-rich tasks — insertion, wiping, peg-in-hole — the
model is learning the *contact phase* of the motion that classical
geometric planners struggle with anyway, while the *approach phase* is
still a planned trajectory. So even here the cake is layered, just
with a thinner geometric slice in the middle.

## The dynamic layer: torques are still classical

The bottom of the stack is the most strongly classical of all, and the
prediction is that it will stay that way for the foreseeable future on
manipulators. Once a joint target comes down from the upper layers,
*something* has to turn it into motor current. That something is, in
every commercially deployed manipulator the authors are aware of in
2026, a classical controller — PD-plus-gravity, computed-torque, or
impedance control, in the language of §4.3.

The Franka Panda is the canonical example. The arm ships with several
control interfaces — joint position, joint velocity, joint impedance,
Cartesian impedance, torque — and *every one of them*, including
torque, is consumed by the manufacturer's onboard controller that
applies its own dynamics-aware compensation before driving the motors.
A research user who selects the torque interface still has the
manufacturer's gravity compensation, friction model, and joint-limit
shielding active underneath. The user has not "replaced" the dynamic
layer; they have set the high-level term in the manipulator equation
and let the low-level term take care of the rest. This is also why
papers report "torque-level control" of a Franka with a learned
policy without ever actually estimating `M(q)` in the learned network:
the manufacturer's controller is doing it.

The strong exception is legged robotics. RL-trained controllers for
ANYmal, Cassie, and Atlas typically emit joint torques (or position
targets that drive a high-bandwidth, low-impedance underlying joint
PD), and the policy has internalized the inertial and gravitational
terms during training in simulation. The reason this works for legs
and not for manipulator end-effectors is that legged-locomotion
controllers run in closed loop with proprioception at 200–1000 Hz on a
relatively low-dimensional output, with a clear and easily simulated
reward, and crucially with no *contact specification* — the policy
chooses where to step, but the contact itself is a constraint the
physics imposes rather than a target the controller has to enforce.
Manipulators have to push on objects with specified forces; legs only
have to push on the ground.

Where the dynamic layer is being augmented but not replaced is the
*residual* pattern (mentioned at the end of §4.3): classical
impedance control provides 80% of the right torque, and a small
network learns the residual that handles deformable objects, cloth,
contact transitions, or fast in-hand reorientation. Production
deployments of such residuals at companies like Covariant, Figure, and
1X are not heavily documented in the literature, but the architectural
pattern is consistent across the few systems whose papers do describe
it.

## A summary table you can keep in your head

If you have to argue with someone about whether classical robotics is
"dead", the right shape of the argument is layer by layer:

- **Symbolic layer.** Mostly classical, increasingly LLM-driven; the
  *interface* (named skills with arguments) is unchanged since STRIPS.
  Replaced wholesale by a learned policy only on tasks short enough to
  fit in a single VLA call.
- **Geometric layer.** Almost universally classical inside deployed
  manipulators in 2026. Learned policies emit poses; IK and motion
  planning sit underneath. Genuinely shrinking only for short-horizon
  contact-rich motions and some legged controllers.
- **Dynamic layer.** Classical, full stop, for manipulators. RL has
  taken the dynamic layer on legged platforms because the structure of
  the task is friendly to torque-level policies. Residual learning is
  the active research frontier on top of classical control rather than
  a replacement for it.

The reason this matters for the rest of the book is that, when we
discuss what a foundation action model *outputs* in Chapters 11–14, we
are almost always discussing what enters the geometric layer from
above. The picture of "VLA → motor" you may have built up while
reading the OpenVLA paper is missing two layers in the middle. They
are still there, still being maintained by the same kind of engineer
who maintained them in 2005, and they are still where most
production failures live.

§4.5 closes Chapter 4 with a summary that consolidates the layered
view and previews how each subsequent part of the book attacks the
layers from below.
