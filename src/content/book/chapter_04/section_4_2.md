---
chapter: 4
section: 4.2
title: "Geometric actions: inverse kinematics and motion planning"
target_words: 2000
status: draft
prereqs: §4.1 (a ground action like pick-up(block_a) is what comes in); §3.1 (matrices, Jacobians); high-school trigonometry; willingness to think in joint-angle vectors rather than Cartesian poses
key_refs:
  - Craig, J. J. (2005). Introduction to Robotics — Mechanics and Control, 3rd ed. Pearson.
  - Lynch, K. M., & Park, F. C. (2017). Modern Robotics — Mechanics, Planning, and Control. Cambridge University Press.
  - LaValle, S. M. (2006). Planning Algorithms. Cambridge University Press.
  - Kavraki, L. E., Švestka, P., Latombe, J.-C., & Overmars, M. H. (1996). Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces. IEEE T-RA 12(4).
  - Kuffner, J. J., & LaValle, S. M. (2000). RRT-Connect — An Efficient Approach to Single-Query Path Planning. ICRA.
  - Karaman, S., & Frazzoli, E. (2011). Sampling-based Algorithms for Optimal Motion Planning. IJRR 30(7).
  - Ratliff, N., Zucker, M., Bagnell, J. A., & Srinivasa, S. (2009). CHOMP — Gradient Optimization Techniques for Efficient Motion Planning. ICRA.
  - Schulman, J. et al. (2013). Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization. RSS.
  - Beeson, P., & Ames, B. (2015). TRAC-IK — An Improved Inverse Kinematics Solver. Humanoids.
  - Şucan, I. A., Moll, M., & Kavraki, L. E. (2012). The Open Motion Planning Library. IEEE RAM 19(4).
---

# 4.2  Geometric actions: inverse kinematics and motion planning

A STRIPS planner hands the executor a string: `pick-up(block_a)`. The executor's first job is turning that string into something a motor controller can consume, a sequence of joint-angle vectors, sampled at maybe 100 Hz, that moves the gripper from where it is now to a pose letting it close around block A without colliding with the table, with block B, or with itself. That translation is this section's topic. It's a layer almost every modern robot still runs, sitting either under a symbolic planner (the classical task-and-motion stack) or under a learned high-level policy (a VLM proposing subgoals to a geometric controller).

The translation splits into two distinct subproblems. The first is inverse kinematics (IK): given a desired pose for the end effector, what joint angles produce it? The second is motion planning: given a starting configuration and a goal configuration, what continuous, collision-free path connects them? IK asks about one instant in time; motion planning asks about the curve through configuration space. Confusingly, both get called "geometric" because both ignore the forces and torques actually driving the joints, that's §4.3's problem. Here, the robot is treated as a rigid mechanism that can be commanded to any kinematically reachable configuration, and the question is which configuration, and how to get there.

## Forward kinematics, then inverse

A serial manipulator with `n` joints is parameterized by a vector `q ∈ R^n` of joint angles (or, for prismatic joints, displacements). Forward kinematics is the map from `q` to the pose of the end effector, written `T(q) ∈ SE(3)`, a 4×4 homogeneous transform encoding both position and orientation. The map is built by chaining one transform per link, using Denavit–Hartenberg parameters or the more modern product-of-exponentials formulation (Lynch and Park, 2017, Ch. 4). It's closed-form, smooth, and cheap: for a 7-DOF arm, evaluating `T(q)` takes roughly two dozen multiplications. There's no algorithmic difficulty in forward kinematics, only bookkeeping.

Inverse kinematics runs the opposite direction: given a target pose `T_desired`, find a `q` with `T(q) = T_desired`. The difficulty here is real. For a generic 6-DOF arm, IK is a system of six nonlinear equations in six unknowns. For arms with three intersecting wrist axes, Puma, Kuka KR series, most industrial designs, Pieper's solution from 1968 gives a closed-form answer: position decouples from orientation, and each piece solves with trigonometric identities. For arms without that geometric niceness, and for any robot with more than six joints, which includes most modern collaborative arms (Franka Panda, KUKA iiwa, Universal Robots), there's no closed form, and the problem becomes either numerical or underdetermined. A 7-DOF arm reaching a 6-DOF target has a one-dimensional manifold of solutions, a continuous family of elbow positions, the "elbow swivel" parameter. The IK solver has to pick one, and the choice matters; picking poorly puts the elbow into a doorframe.

## How an IK solver actually works

In practice almost no one writes their own IK solver anymore. Everyone calls into TRAC-IK (Beeson and Ames, 2015), KDL, or IKFast (the analytic generator distributed with OpenRAVE). It's still worth understanding the two underlying recipes, since their failure modes show up right at the interfaces.

The closed-form recipe is what IKFast does. It takes the URDF of the robot, symbolically derives the trigonometric equations from the forward-kinematics chain, and emits a C++ file that, given a target pose, returns the up-to-sixteen analytic solutions in microseconds. It's exact, has no hyperparameters, and is the default choice when the arm has the right geometry. It fails when the arm doesn't, which covers most 7-DOF arms, because the symbolic derivation simply never terminates.

The numerical recipe handles everything else. It starts from a current guess `q_0` and iterates a small update `q_{k+1} = q_k + Δq` until the error `|T(q_k) - T_desired|` falls below a tolerance. The update uses the kinematic Jacobian `J(q) = ∂T/∂q` (a 6×n matrix mapping joint velocities to end-effector twists), inverted in some sense to map the pose error back to a joint update. The simplest version is `Δq = J^+ · e`, where `J^+` is the Moore-Penrose pseudoinverse and `e` is the 6-vector pose error. In practice everyone uses damped least squares instead, `Δq = J^T (JJ^T + λ²I)^{-1} e` (Nakamura and Hanafusa, 1986), because it stays well-conditioned near singularities, configurations where `J` loses rank and the arm momentarily can't move in some Cartesian direction. TRAC-IK runs damped least squares and sequential quadratic programming in parallel and returns whichever converges first.

The failure modes are worth a paragraph, since they're what an executor above the IK layer has to handle. No solution exists: the target pose sits outside the workspace, or inside a self-collision. A solution exists but the solver missed it: numerical IK is a local method, and a bad initial guess can sit in a basin that never reaches the global optimum. Multiple solutions exist: which one comes back depends on the seed, and naively re-seeding between calls produces a robot that flips its elbow between consecutive waypoints. Every production system wraps IK to handle these cases, seeding with the previous solution, filtering by collision, choosing among solutions by a "distance from current pose" heuristic.

## Configuration space and the motion-planning problem

Suppose IK has succeeded and returned a goal configuration `q_goal`. The robot currently sits at `q_start`. Why not just interpolate the joint angles linearly between them? Because the straight line in joint space typically passes through configurations where the arm sits inside the table, inside itself, or inside the block it's trying to grasp.

The cleanest way to think about this is configuration space, or C-space: the space of all joint vectors `q ∈ R^n` (with appropriate periodicity for revolute joints). Each pose of the robot is one point in C-space. Each obstacle in the workspace, the table, the second block, the gripper itself, projects to a forbidden region `C_obs ⊂ R^n`, and the free region `C_free = R^n \ C_obs` is where the robot is allowed to be. The motion-planning problem becomes finding a continuous curve from `q_start` to `q_goal` lying entirely inside `C_free`.

`C_free` is the catch. It's implicitly defined; there's no closed-form description of it, only a black-box collision-check taking a `q` and returning true or false. For a 7-DOF arm in a typical tabletop scene, one collision check costs a few hundred microseconds (FCL, the standard library, broadphases AABB intersections then runs GJK on candidate pairs). Motion planners get judged, in large part, by how few of these checks they need.

## Sampling-based motion planning

The dominant approach for robotic arms is sampling-based motion planning (LaValle, 2006). The idea is simple almost to the point of embarrassment: rather than explicitly modeling `C_free`, sample points uniformly at random from the joint-space box, discard the ones that collide, and build a graph out of the survivors. Two algorithms dominate.

Probabilistic Roadmap (PRM), introduced by Kavraki et al. (1996), samples thousands of configurations once, connects each to its k nearest neighbors with straight-line edges (checking each edge for collisions along its midpoint and recursively), and stores the resulting roadmap. At query time, `q_start` and `q_goal` get connected to the roadmap and Dijkstra finds a path. PRM is multi-query: you amortize the roadmap build across many planning queries in the same scene.

Rapidly-exploring Random Tree (RRT), introduced by LaValle (1998), builds a tree rooted at `q_start`. Each iteration samples a random `q`, finds the nearest node in the tree, and extends the tree by stepping a small distance toward the sample, adding the new node if the edge is collision-free. Bias the sampling toward `q_goal` with probability 0.05, and the tree finds the goal quickly. RRT-Connect (Kuffner and LaValle, 2000) grows two trees, one from start and one from goal, and connects them, empirically the fastest single-query planner for high-DOF arms and the default in most modern stacks.

Neither RRT nor PRM produces optimal paths. The RRT* and PRM* variants from Karaman and Frazzoli (2011) recover asymptotic optimality by rewiring edges as new nodes arrive; in the limit of infinite samples, the returned path converges to the shortest one. In practice, the post-processing step that matters most is shortcutting: take the returned waypoint sequence, pick random pairs of waypoints, and replace the segment between them with a straight line if that line is collision-free. A 200-waypoint RRT output typically shortens to about 20 waypoints after shortcutting.

The reference implementation everyone uses is OMPL (Şucan, Moll, and Kavraki, 2012), wrapped in MoveIt for ROS. A typical IK-plus-RRT-Connect-plus-shortcut pipeline for a 7-DOF arm in a known scene plans a pick motion in 100 to 500 ms.

## Optimization-based motion planning

The other family is optimization-based. CHOMP (Ratliff et al., 2009) formulates planning as gradient descent on a trajectory cost, a sum of smoothness (joint acceleration squared) and obstacle cost (the distance field of the scene, smoothed). It starts from a straight-line initialization in joint space and iteratively pushes the trajectory away from obstacles. TrajOpt (Schulman et al., 2013) does the same thing with sequential convex optimization and explicit collision constraints rather than soft penalties. Both are local methods: they need a reasonable initial trajectory and return a local optimum, not the global one.

The trade-off against sampling-based planners is fairly clean. Sampling-based methods are probabilistically complete, meaning given infinite time they find a solution if one exists, but produce jagged paths and ignore costs other than path length. Optimization-based methods produce smooth, low-cost paths but get stuck in local minima around obstacles. Production stacks typically run RRT-Connect first to find a path, then hand the result to TrajOpt as a warm start for polishing.

## What the geometric layer hands to §4.3

The output of this whole pipeline, IK to get `q_goal`, motion planning to get a path, shortcutting and smoothing, is a discrete sequence of joint configurations `(q_0, q_1, …, q_K)` with associated timestamps. This is a kinematic trajectory: it describes positions over time but says nothing about whether the robot's motors can actually generate those positions under gravity and inertia. Sending the trajectory directly to position-controlled joints often works fine for slow motions and stiff industrial arms; sending it to a torque-controlled compliant arm moving fast will produce trajectory tracking error of centimeters, more than enough to miss the block entirely.

Closing that gap, turning a kinematic trajectory into joint torques, is the inverse-dynamics problem of §4.3.
