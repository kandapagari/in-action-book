---
appendix: D
title: "Setting up a robotics simulator"
target_words: 3000
status: draft
prereqs: Appendix C; Ubuntu 22.04 or macOS 14+ on a machine with an NVIDIA GPU for Isaac Lab; a working Python 3.10+ install
key_refs:
  - Todorov, E., Erez, T., & Tassa, Y. (2012). MuJoCo — A physics engine for model-based control. IROS 2012.
  - NVIDIA (2024). Isaac Lab Documentation. isaac-sim.github.io/IsaacLab.
  - Liu, B. et al. (2023). LIBERO — Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023 D&B.
  - Mittal, M. et al. (2023). Orbit — A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE RA-L.
---

# Appendix D.  Setting up a robotics simulator

The chapters that ask you to run code — Chapter 2's LIBERO demo,
Chapter 5's gridworld, Chapter 7's PPO on HalfCheetah, Chapter 16's
fine-tuning loop — all assume a working simulator on your machine. This
appendix is the step-by-step setup for the two simulators the body of
the book uses: MuJoCo (for the small benchmarks and the Decision
Transformer work) and Isaac Lab (for the large-scale, photo-realistic
manipulation and locomotion work). It also covers attaching a USD
asset so that you can extend an existing scene with your own robot,
and the basic teleop loop so that you can record a small dataset for
fine-tuning. The simulator landscape changes more often than the
algorithms do, so where versions are specified, treat them as known-
good at the time of writing and update against the official docs if
those have moved.

## D.1  Why two simulators, and not one

MuJoCo is small, fast, and CPU-friendly. A single MuJoCo environment
runs at hundreds of frames per second on a laptop without a GPU, and
the engine has been the backbone of academic RL benchmarks since
Todorov, Erez, and Tassa (2012, IROS). LIBERO
(Liu et al., 2023, NeurIPS) — the manipulation benchmark Chapter 2
loads — is a MuJoCo derivative, and so are D4RL, MetaWorld, and most
of the published benchmarks that Chapters 5 through 10 reach for. If
you only have a laptop, MuJoCo is the simulator you will use for the
first three-quarters of the book.

Isaac Lab is large, photo-realistic, and GPU-native. Built on
NVIDIA's Omniverse and Isaac Sim, it runs thousands of parallel
environments on a single GPU at 60 Hz, with PhysX-backed rigid-body
and contact dynamics and renderable RGB-D output. The trade-off is
the install footprint (Omniverse alone is ~30 GB), a hard NVIDIA-GPU
requirement, and a learning curve that is steeper than MuJoCo's by
the better part of an afternoon. The chapters that fine-tune a VLA
with sim-to-real (Chapter 16) and the chapters that run dexterous
manipulation experiments (Chapters 13 and 15) need Isaac Lab; the
earlier chapters do not.

If you can install only one, install MuJoCo first. The book's first
runnable code lives inside it.

## D.2  Installing MuJoCo

MuJoCo has been open source since 2022 and is pip-installable. On
Ubuntu 22.04 and macOS 14+:

```bash
python3.10 -m venv ~/venvs/action-models
source ~/venvs/action-models/bin/activate
pip install --upgrade pip
pip install mujoco==3.2.5 dm-control gymnasium[mujoco] numpy
```

Confirm the install with the canonical one-liner:

```python
import mujoco
import mujoco.viewer
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body><geom type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
</mujoco>""")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
```

If a window opens with a small white cube sitting at the origin,
MuJoCo is working. On macOS the viewer needs the `mjpython` launcher
in place of `python` for the GUI to render correctly:
`mjpython -m mujoco.viewer --mjcf=path/to/scene.xml`. On Ubuntu under
WSL2, the viewer requires WSLg (Windows 11) or an X server forwarded
to a Linux X11 client.

LIBERO sits on top of `robosuite`, which sits on top of MuJoCo. The
install adds two more lines:

```bash
pip install robosuite
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

A LIBERO benchmark task can then be loaded as:

```python
from libero.libero import benchmark
bm = benchmark.get_benchmark_dict()["libero_object"]()
task = bm.get_task(0)        # the first of the ten "Object" tasks
env = task.env(render=False)
obs = env.reset()
```

A working LIBERO install is the prerequisite for the OpenVLA loop in
Chapter 2; the chapter's hands-on directory has a `verify_libero.py`
that exercises the loop end to end and prints the observation shape.

## D.3  Installing Isaac Lab

Isaac Lab requires an NVIDIA GPU with at least 8 GB of memory
(16 GB is recommended), driver version 535+, and Ubuntu 22.04 or
Windows 11 with WSL2. macOS is not supported. The 2026 install path
is through Isaac Sim 4.5 or later; the older "standalone Omniverse
launcher" path is deprecated.

```bash
# 1. Install the NVIDIA driver and CUDA 12.x via Ubuntu's package manager.
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
# 2. Install Isaac Sim. The recommended method in 2026 is the pip
#    wheel; the older Omniverse Launcher GUI install is also supported.
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
# 3. Install Isaac Lab from its GitHub repo.
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab && ./isaaclab.sh --install
```

Confirm the install by running a canonical example:

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/standalone/demos/quadrupeds.py
```

If a window opens with a herd of small quadruped robots running on a
plane, Isaac Lab is working. First-launch compilation takes a few
minutes; subsequent launches are fast.

Two common install failures. *Driver mismatch*: `nvidia-smi` works
but Isaac Sim crashes on launch with a kernel error. The fix is
almost always to update the driver to 535+ and reboot. *X-server
mismatch*: the GUI launches but renders a black window. This is a
WSLg or VirtualGL issue on remote machines; the workaround is to
run with `--headless` and use the recording-and-playback tools
described in §D.5.

## D.4  Loading a USD asset

Both Isaac Sim and Isaac Lab use the Universal Scene Description
(USD) format pioneered by Pixar. A USD file (`.usd`, `.usda`, or
`.usdc`) describes a scene as a hierarchical tree of "prims" —
geometric primitives, references, materials, lights — and is the
format every NVIDIA robotics asset ships in. Most academic robotics
assets ship in URDF (XML-flavored) or MJCF (also XML-flavored, the
MuJoCo dialect); both need to be converted to USD before Isaac Lab
will accept them.

The conversion utility:

```bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
    --input  path/to/my_robot.urdf \
    --output path/to/my_robot.usd \
    --merge-joints
```

After conversion, the asset can be loaded into a scene as:

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

my_robot_cfg = ArticulationCfg(
    prim_path="/World/MyRobot",
    spawn=sim_utils.UsdFileCfg(usd_path="path/to/my_robot.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        joint_pos={"joint_0": 0.0, "joint_1": -0.5},
    ),
)
```

Three things commonly go wrong on the first conversion. URDF inertia
tags are sometimes wrong by a factor of 10 (a 1-kg link claiming
10 kg·m² of inertia); the simulator will solve but the dynamics will
feel wrong, and PhysX may flag the asset as unstable. Mesh
collisions default to the visual mesh, which is usually too detailed
to simulate efficiently; pre-compute a convex decomposition (the
`--collision-approximation convexDecomposition` flag) and Isaac Lab
will use that for contact. Joint limits in radians vs. degrees:
PhysX expects radians, URDF allows either, and a 90 in radians is a
joint that flies out the floor on the first step.

## D.5  Recording a teleop dataset

The fine-tuning recipe of Chapter 16 needs demonstrations. The
canonical way to produce them is teleoperation: a human operator
drives the robot through the task, the simulator records (observation,
action, reward, done) tuples, and the result is a dataset of episodes
ready for behavior cloning. Both MuJoCo and Isaac Lab support
keyboard and 3D-mouse teleop out of the box; SpaceMouse hardware
(the 3Dconnexion family) is the standard.

In MuJoCo/robosuite, a basic teleop loop:

```python
import robosuite as suite
from robosuite.devices import SpaceMouse

env = suite.make("Lift", robots="Panda", has_renderer=True,
                  has_offscreen_renderer=False, control_freq=20)
device = SpaceMouse()
device.start_control()

episodes = []
for ep in range(10):
    obs = env.reset()
    transitions = []
    done = False
    while not done:
        action = device.get_controller_state()["dpos_drot"]
        next_obs, reward, done, info = env.step(action)
        transitions.append((obs, action, reward, next_obs, done))
        env.render()
        obs = next_obs
    episodes.append(transitions)
```

Saving the result to an HDF5 file in the format the rest of the book
expects (one group per episode, datasets `observations`, `actions`,
`rewards`, `dones`) is twenty more lines of glue; the chapter's hands-
on directory has the working version.

In Isaac Lab, teleop goes through the `Teleop` workflow in
`isaaclab.envs`; the API is busier but the principle is the same. The
canonical example to crib from is
`source/standalone/workflows/teleoperation/teleop_se3_agent.py`.

Three pragmatic notes about teleop datasets. *Frequency*: 20 Hz is
the standard for end-effector-pose teleop; 50 Hz for joint-velocity
teleop. Sampling faster than the human can react produces redundant
frames; sampling slower produces jerky trajectories. *Reset
discipline*: the operator should re-randomize the scene between
episodes (block position, lighting, target pose), otherwise the
dataset over-represents one initial condition. *First-episode
junk*: the first one or two episodes of any teleop session contain
the operator learning the controls; throw them out before training.

## D.6  Headless rendering and recording

Most training runs do not need a viewer. To run a MuJoCo environment
without rendering at all, just construct it without the viewer call
and step forward; the engine runs orders of magnitude faster than
real time on a CPU. To run Isaac Lab without rendering, append
`--headless` to the launch command:

```bash
./isaaclab.sh -p source/standalone/tutorials/00_sim/spawn_prims.py --headless
```

Headless runs still produce camera observations via the
`isaaclab.sensors.Camera` API; the cameras render to GPU buffers
that the policy reads directly, no window required. Recording a
video from a headless run uses `imageio`:

```python
import imageio
frames = []
for step in range(200):
    obs = env.step(action)
    frames.append(obs["camera"]["rgb"])
imageio.mimsave("rollout.mp4", frames, fps=20)
```

Videos are how you debug a policy that "works in success rate" but
"feels weird"; the eyeball is the highest-bandwidth diagnostic
instrument in robot learning.

## D.7  A platform-specific cheat sheet

A condensed reference for the install and launch commands by
operating system.

*Ubuntu 22.04 with an NVIDIA GPU:* MuJoCo via pip, Isaac Lab via the
two-command install above, both run natively. The most common
failure is a stale CUDA driver; `sudo apt upgrade nvidia-driver-535
&& reboot` fixes most things.

*Ubuntu 22.04 without an NVIDIA GPU:* MuJoCo via pip; Isaac Lab is
not supported. Use Gymnasium environments for the earlier chapters
and reach for a cloud GPU only when Chapter 16 demands it.

*macOS (Apple Silicon, M-series):* MuJoCo via pip works out of the
box, with the `mjpython` launcher for the GUI; LIBERO works with the
same caveat. Isaac Lab is unsupported; use a Linux machine or a
cloud GPU instance.

*Windows 11 with WSL2:* MuJoCo works under WSL2 with WSLg.
Isaac Lab requires both WSLg and a recent NVIDIA driver; the install
is otherwise identical to native Ubuntu.

*Cloud (a single A100 or L40S on RunPod, Lambda, or vast.ai):*
MuJoCo and Isaac Lab both run unmodified; the install is identical
to a native Ubuntu install. The hassle is filesystem persistence —
preemptible instances erase state on every shutdown — so put
checkpoints on attached object storage and not on the instance's
local disk.

## D.8  What to do when the simulator misbehaves

Three failure modes that every reader will encounter at least once.

*The physics blew up.* The robot's joints fly out the floor at
simulation step 0. The cause is almost always an asset issue — bad
inertia tags, a self-colliding mesh, joint limits in the wrong units.
Run the asset's `analyze` step first (`mujoco._analyze` or Isaac
Lab's USD validator) and fix what it complains about.

*The policy outputs values the simulator clips.* The model emits an
action of magnitude 50 but the simulator's joint torque limit is 10.
The clipping is happening silently and the model never gets a useful
gradient. Add an `assert action.abs().max() < limit` somewhere in
the data loader and confirm the bound is what you think it is.

*The episode is too long or too short.* The default episode length
of a benchmark may be 200 steps; your task takes 500. The model
learns a policy that "finishes" at step 199 by giving up. Reading
the environment config and confirming `max_episode_steps` against
your task's actual duration is a two-minute check that saves a
half-day of confusion.

A working simulator is the unsexy prerequisite for everything else
in this book. With MuJoCo running for the small benchmarks and Isaac
Lab running for the large ones, every code listing in Chapters 2
through 17 should produce the result the chapter describes. The
appendix that comes next (Appendix E) is the consolidated reading
list; the appendix after that (F) is the model zoo table.
