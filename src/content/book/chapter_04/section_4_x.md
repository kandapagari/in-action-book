---
chapter: 4
section: 4.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §4.1–§4.5; a working Python 3 environment; Fast Downward installed (or the dockerized variant); PyBullet installed; the TRAC-IK Python bindings or a fallback damped-least-squares IK; about three hours of laptop CPU time (no GPU required)
key_refs:
  - Fikes & Nilsson (1971). STRIPS — A New Approach to the Application of Theorem Proving to Problem Solving. Artificial Intelligence 2(3–4).
  - McDermott et al. (1998). PDDL — The Planning Domain Definition Language. AIPS-98.
  - Helmert (2006). The Fast Downward Planning System. JAIR 26.
  - Garrett, Lozano-Pérez, Kaelbling (2020). PDDLStream. ICAPS.
  - Kuffner & LaValle (2000). RRT-Connect. ICRA.
  - Beeson & Ames (2015). TRAC-IK — An Improved Inverse Kinematics Solver. Humanoids.
  - Ahn et al. (2022). Do As I Can, Not As I Say (SayCan). arXiv:2204.01691.
  - Liang et al. (2022). Code as Policies. arXiv:2209.07753.
  - Kim et al. (2024). OpenVLA. arXiv:2406.09246.
  - Black et al. (2024). π0. arXiv:2410.24164.
---

# 4.x  Hands-on exercise + chapter references

Chapter 4 was the classical-lineage chapter; the exercise is where you build a small version of the three-layer cake on your laptop and stack one block on top of another with no neural network anywhere in the loop. The four drills below take a combined three hours on a laptop CPU. The point is leaving Chapter 4 with a working symbolic-to-geometric-to-dynamic pipeline on disk, so that when Chapters 11 through 14 start replacing the top and middle of the cake with learned components, you have something to replace into.

## Exercise 4.x.1 — A five-block PDDL domain and a planner call

Create a directory `drills_ch4/` with two files: `blocksworld.pddl` (the domain) and `problem_5.pddl` (the instance). The domain encodes the standard Blocksworld formulation from §4.1, predicates `(on ?x ?y)`, `(on-table ?x)`, `(clear ?x)`, `(holding ?x)`, `(arm-empty)`, and four action schemas: `pick-up`, `put-down`, `stack`, `unstack`. Write the domain from memory and check it against §4.1 afterwards. If you can't reproduce the `stack` schema's add and delete lists without peeking, you haven't yet internalized what an action schema is.

The problem file describes five blocks `a` through `e` arranged as two towers, `(a on b)`, `(b on c)`, `(c on-table)`, `(d on e)`, `(e on-table)`, plus the matching `clear` and `arm-empty` predicates, and a goal that stacks all five into a single column with `a` on top and `e` on the bottom: `(on a b) (on b c) (on c d) (on d e) (on-table e)`. Fast Downward should return one of several plans of length 12.

Install Fast Downward (Helmert, 2006), from source, via the dockerized variant, or via the `pyperplan` wrapper for a simpler alternative, and run it on the two files. Save the returned plan to `plan.txt`. The deliverable is the plan plus a one-line note recording which heuristic the planner used (`seq-opt-lmcut` and `lama-first` are the two defaults worth knowing). Runtime should be well under a second. If the planner reports `unsolvable`, your goal or initial state is inconsistent, most often a missing `clear` predicate, and §4.1's discussion of the closed-world assumption is the place to re-read.

Wall clock: about thirty minutes including the Fast Downward install.

## Exercise 4.x.2 — Wire the plan to a PyBullet pick-and-place executor

Copy the plan from Exercise 4.x.1 into a new Python file `execute_plan.py`. Its job is taking the plan and producing joint-space motion on a simulated Franka Panda in PyBullet. The architecture is the §4.5 three-layer cake in code form:

1. Symbolic layer (already done): the plan from 4.x.1, read in as a list of `(action_name, arguments)` tuples.
2. Geometric layer (this exercise): a function `solve_action(action, world_state)` that, for each symbolic action, computes a sequence of Cartesian waypoints (pre-grasp pose, grasp pose, lift pose, transport pose, place pose, retract pose), passes each waypoint through TRAC-IK to obtain joint targets, and concatenates the joint targets into a trajectory. Use the TRAC-IK Python bindings (Beeson and Ames, 2015) if installed; otherwise fall back to the damped-least-squares IK from §4.2 with a damping factor of 0.05 and a maximum of 200 iterations. RRT-Connect (Kuffner and LaValle, 2000) is overkill for tabletop blocksworld, since straight-line motion in joint space between IK solutions is sufficient, but drop OMPL in if you have it handy.
3. Dynamic layer (PyBullet provides it): set joint targets with `setJointMotorControlArray` in position-control mode. PyBullet's internal PD-plus-gravity controller does the §4.3 work for you; the point of this exercise is not implementing computed-torque from scratch, but noticing that the simulator is silently providing the bottom layer of the cake.

The world is a flat table at `z=0`, five colored 4 cm cubes whose initial positions match `problem_5.pddl`, and a Franka mounted at the table edge. PyBullet ships the URDF at `pybullet_data/franka_panda/panda.urdf`. A skeleton for the execution loop, illustrative rather than exhaustive:

```python
plan = read_plan("plan.txt")
world = init_pybullet_world(blocks=["a", "b", "c", "d", "e"])
arm = load_panda(world)
ik = TracIKSolver(arm.urdf, "panda_link0", "panda_hand")

for action_name, args in plan:
    waypoints = waypoints_for(action_name, args, world)
    for pose in waypoints:
        q_target = ik.solve(pose, q_seed=arm.current_q())
        if q_target is None:
            raise RuntimeError(f"IK failed for {action_name} at {pose}")
        arm.move_to(q_target, duration=1.0)
    world.apply_symbolic_effects(action_name, args)
```

Run the script and watch the arm execute the twelve-action plan in about thirty seconds of wall clock simulation time. Save a screenshot of the final state, a single tower with `a` on top, to `final_state.png`. Save the joint trajectory to `trajectory.npz`; you'll reuse it in Exercise 4.x.3.

The most common failure is IK returning `None` for the pre-grasp pose of `c`, because the dexterous workspace of the Panda doesn't quite reach a block at table height directly below the base. The §4.2 fix is lowering the table by 5 cm or mounting the arm 10 cm higher. If the IK succeeds but the gripper passes through an adjacent block on approach, you've rediscovered the problem PDDLStream (Garrett, Lozano-Pérez, and Kaelbling, 2020) was written to solve: the symbolic planner doesn't know about geometric infeasibility, and a true TAMP system would interleave the two searches. We don't implement that here; leaving the bug visible is the lesson.

Wall clock: about ninety minutes including TRAC-IK installation.

## Exercise 4.x.3 — Replace the bottom layer with explicit computed-torque

Copy `execute_plan.py` to `execute_plan_torque.py` and change one thing: instead of using PyBullet's position controller, drive the arm in pure torque mode (`setJointMotorControlArray(... , controlMode=p.TORQUE_CONTROL, forces=tau)`) and compute `tau` explicitly from the §4.3 computed-torque law

$$
\begin{aligned}
\tau &= M(q)(\ddot{q}_d + K_d (\dot{q}_d - \dot{q}) + K_p (q_d - q)) \\
&\quad + C(q,\dot{q})\dot{q} + g(q).
\end{aligned}
$$

PyBullet exposes the inertia matrix and the bias terms through `calculateMassMatrix` and `calculateInverseDynamics`, so you don't have to derive the Panda's dynamics from URDF by hand. Choose `K_p = 100` and `K_d = 20` on every joint as a starting point. The reference trajectory `q_d(t)` is the trajectory you saved in 4.x.2; numerically differentiate it once for `q̇_d` and twice for `q̈_d` with a simple central difference, or fit a cubic spline first if the differentiated signal comes out too noisy.

The arm executes the plan again, with modest overshoot at each waypoint that decays in under half a second. Plot `q(t)`, `q_d(t)`, and `tau(t)` for joint 4 (the elbow, which carries the largest gravity load) and save as `joint4_torque.png`. The plot should make three things visible: `tau(t)` isn't zero when the arm is stationary (the static gravity term), peaks in `tau(t)` line up with the lift-and-transport phases (dynamic torques scale with acceleration), and tracking error grows when `K_d` is reduced; try `K_d = 5` and watch the arm wobble. That wobble is what §4.3 cited as the motivation for matching damping coefficients to the inertia matrix, and the residual-learning pattern from §4.4 is what a research group reaches for when the wobble persists even with well-tuned gains.

This is the exercise where the three-layer cake stops being a metaphor. You've written the geometric and dynamic layers explicitly; the symbolic layer is the text file from 4.x.1; the learned layer is the empty slot the rest of the book fills in.

Wall clock: about forty-five minutes.

## Exercise 4.x.4 — A PDDL diff against a SayCan transcript

Open the SayCan paper (Ahn et al., 2022, arXiv:2204.01691) to one of the appendices listing a full kitchen-task transcript; the "bring me a snack" examples typically run several pages. Pick one transcript and, on paper or in a text file, transcribe the language-model output into PDDL. Each numbered step the model produces is one ground action; figure out which predicates would have to hold in the precondition and which would be added or deleted.

You'll discover three things. SayCan plans are short, four to eight steps, well within Fast Downward's reach. The predicate set runs much larger than blocksworld's (`in-hand`, `at-location`, `is-open`, `contains`), since a real kitchen has more state than five blocks. And several SayCan steps will be hard to express as STRIPS schemas at all without admitting the language model is doing implicit hierarchical decomposition that the symbolic formalism would express with HTN-style methods rather than pure STRIPS.

The deliverable is the transcribed PDDL plus a half-page note on which steps mapped cleanly, which didn't, and what would have been needed to make them clean. The §4.5 claim that the interface of a symbolic action model survives, with only the engine having moved, holds true in a limited sense, and you should have first-hand experience with where that limit sits. Chapter 14 returns to this when it dissects the high-level system of a dual-system architecture.

Wall clock: about thirty minutes.

## Chapter 4 reading list

The works below are the ones cited across §4.1 through §4.5, grouped by purpose. Full bibliographic entries for everything cited in the whole book live in Appendix E.2; this list is just the chapter-local subset.

### Symbolic planning (§4.1)

- Fikes, R. E., & Nilsson, N. J. (1971). "STRIPS — A New Approach to the Application of Theorem Proving to Problem Solving." *Artificial Intelligence* 2(3–4). The paper §4.1 starts from.
- McDermott, D., et al. (1998). "PDDL — The Planning Domain Definition Language." AIPS-98 Planning Competition. The lingua franca §4.1 recommends learning first.
- Helmert, M. (2006). "The Fast Downward Planning System." *JAIR* 26. The reference planner used in Exercise 4.x.1.
- Hoffmann, J., & Nebel, B. (2001). "The FF Planning System." *JAIR* 14. The other heuristic-search planner worth knowing, and the lineage behind Fast Downward's `lama-first` configuration.
- Garrett, C. R., Lozano-Pérez, T., & Kaelbling, L. P. (2020). "PDDLStream — Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning." ICAPS. The TAMP reference §4.1 and Exercise 4.x.2 both point to.
- Ahn, M., et al. (2022). "Do As I Can, Not As I Say — Grounding Language in Robotic Affordances." (SayCan.) arXiv:2204.01691. The LLM-as-symbolic-planner system §4.1 and Exercise 4.x.4 use.
- Liang, J., et al. (2022). "Code as Policies — Language Model Programs for Embodied Control." arXiv:2209.07753. The companion LLM-planner paper §4.4 cites.

### Geometric actions: IK and motion planning (§4.2)

- Craig, J. J. (2005). *Introduction to Robotics — Mechanics and Control*, 3rd ed. Pearson. The standard manipulator-kinematics textbook.
- Lynch, K. M., & Park, F. C. (2017). *Modern Robotics — Mechanics, Planning, and Control.* Cambridge University Press. The modern geometric-algebra-flavored alternative to Craig.
- LaValle, S. M. (2006). *Planning Algorithms.* Cambridge University Press. The reference textbook for motion planning.
- Kavraki, L. E., Švestka, P., Latombe, J.-C., & Overmars, M. H. (1996). "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces." *IEEE T-RA* 12(4). The PRM paper.
- Kuffner, J. J., & LaValle, S. M. (2000). "RRT-Connect — An Efficient Approach to Single-Query Path Planning." ICRA. The single-query planner §4.2 recommends as the default.
- Karaman, S., & Frazzoli, E. (2011). "Sampling-based Algorithms for Optimal Motion Planning." *IJRR* 30(7). RRT* and PRM*.
- Ratliff, N., Zucker, M., Bagnell, J. A., & Srinivasa, S. (2009). "CHOMP." ICRA. The optimization-based planner §4.2 contrasts with sampling.
- Beeson, P., & Ames, B. (2015). "TRAC-IK — An Improved Inverse Kinematics Solver." Humanoids. The numerical IK solver used in Exercise 4.x.2.
- Şucan, I. A., Moll, M., & Kavraki, L. E. (2012). "The Open Motion Planning Library." *IEEE RAM* 19(4). The OMPL reference; the planner every real ROS stack reaches for.

### Inverse dynamics and control (§4.3)

- Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control.* Wiley. The textbook §4.3 follows for the manipulator equation derivation.
- Featherstone, R. (2008). *Rigid Body Dynamics Algorithms.* Springer. The reference for recursive algorithms, RNEA, CRBA, that any real inverse-dynamics implementation uses.
- Luh, J. Y. S., Walker, M. W., & Paul, R. P. C. (1980). "On-Line Computational Scheme for Mechanical Manipulators." *ASME J. Dynamic Systems* 102(2). The original RNEA paper.
- Khatib, O. (1987). "A Unified Approach for Motion and Force Control of Robot Manipulators — The Operational Space Formulation." *IEEE J. Robotics and Automation* 3(1). The operational-space reference.
- Hogan, N. (1985). "Impedance Control — An Approach to Manipulation." *ASME J. Dynamic Systems* 107(1). The impedance-control paper §4.3 and §4.4 cite as the residual-learning antecedent.
- Slotine, J.-J. E., & Li, W. (1987). "On the Adaptive Control of Robot Manipulators." *IJRR* 6(3). The adaptive-control reference for when the dynamic parameters aren't perfectly known.

### Where classical methods still live in modern stacks (§4.4)

- Kim, M. J., et al. (2024). "OpenVLA — An Open-Source Vision-Language-Action Model." arXiv:2406.09246. The §4.4 worked example for what the seven numbers per step actually feed into.
- Black, K., et al. (2024). "π0 — A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164. The continuous-action VLA §4.4 contrasts with OpenVLA's discrete tokens. Chapter 13 is the full treatment.
- NVIDIA (2025). "GR00T N1 — An Open Foundation Model for Generalist Humanoid Robots." arXiv:2503.14734. The humanoid dual-system example §4.4 cites; Chapter 14 unpacks it.
- Padalkar, A., et al. (2023). "Open X-Embodiment — Robotic Learning Datasets and RT-X Models." arXiv:2310.08864. The dataset every §4.4 modern-stack reference was trained on. Chapter 12 is the full treatment.

## Chapter summary

Chapter 4 was the classical-action chapter, sitting at the hinge between Part 1 (the foundations) and Part 2 (the lineage that produced VLAs). You can now write a small PDDL domain and run Fast Downward on it. You can stand up a tabletop IK-plus-motion-planning layer that takes a symbolic plan and turns it into joint-space motion on a simulated Franka. You can write the manipulator equation and the computed-torque law without looking them up, and explain which terms PyBullet, the manufacturer's onboard firmware, and the application code are each responsible for. And you can read any modern VLA paper and draw its three-layer cake, labeling each layer classical or learned. Chapter 5 begins the next chapter of Part 2, reinforcement learning, the family that fills the training-signal slot the classical models leave empty, and the controllers from §4.3 return there as the policies an MDP optimizes over.
