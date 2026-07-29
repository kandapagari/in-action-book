---
chapter: 15
section: 15.1
title: "What a robot dataset looks like, by example"
target_words: 2000
status: draft
prereqs: §6.2 (behavior cloning — every trajectory in these datasets is an observation-action pair that a BC loss consumes, and this section is really just BC's training data seen up close), §12.4 (Open X-Embodiment — the corpus this section samples a single episode from, treated in detail next in §15.2). Helpful, §11.3 on action tokenization, since the raw continuous actions shown here are what a tokenizer later discretizes.
key_refs:
  - Brohan, A. et al. (2022). RT-1, Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Open X-Embodiment Collaboration, Padalkar, A. et al. (2023). Open X-Embodiment, Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Walke, H. et al. (2023). BridgeData V2, A Dataset for Robot Learning at Scale. CoRL 2023.
---

# 15.1  What a robot dataset looks like, by example

Part 4 kept talking about datasets as if you already knew what one contained. OpenVLA trained on 970,000 trajectories; Octo on 800,000; the whole scale argument of Chapter 12 rested on piles of robot data whose internal shape we never opened up. This chapter opens it. Before you can critique a published evaluation or build your own, you have to know what a single robot episode actually is at the level of bytes and arrays, and the fastest way to learn that is to take one apart.

So take one apart. We will use an episode from BridgeData V2 (Walke et al., 2023, CoRL), a public tabletop-manipulation dataset collected on a low-cost WidowX arm, because it is small enough to inspect by hand and it is one of the larger contributors to Open X-Embodiment. The specifics differ across datasets; the anatomy does not. Once you have seen how Bridge lays out an episode, RT-1's data (arXiv:2212.06817) and the other fifty-nine datasets in the pool read as variations on the same template.

## An episode is a list of timesteps

The unit of a robot dataset is not an image or an action. It is an *episode*: one continuous attempt at one task, from the moment the robot starts moving to the moment it stops. "Put the spoon in the pot" is an episode. "Fold the cloth" is another. An episode that ran for six seconds at a 5 Hz control rate is a list of thirty timesteps, and each timestep is a dictionary that pairs what the robot saw with what it did next.

Concretely, one Bridge timestep holds roughly this:

```python
timestep = {
    "observation": {
        "image_0":      uint8[256, 256, 3],   # wrist-mounted RGB camera
        "image_1":      uint8[256, 256, 3],   # over-the-shoulder RGB camera
        "state":        float32[7],           # end-effector pose + gripper
    },
    "action":           float32[7],           # what the teleoperator commanded
    "language_instruction": "put the spoon on the towel",
    "is_terminal":      False,
    "is_first":         True,                 # only on timestep 0
    "is_last":          False,
}
```

Read that dictionary slowly, because the entire rest of the book has been feeding on its contents without showing them to you. The `observation` is the policy's input, the thing a VLA's vision encoder and proprioceptive channel actually consume. The `action` is the training target, the label a behavior-cloning loss regresses toward, exactly as §6.2 described. The `language_instruction` is what turns a plain imitation dataset into a *language-conditioned* one, the ingredient that made everything from RT-1 onward possible. The boolean flags are bookkeeping so the data loader knows where episodes begin and end when it streams thousands of them back to back.

Stack thirty of these dictionaries and you have an episode. Stack sixty thousand episodes and you have BridgeData V2. That is all a robot dataset is: a very large list of lists of these dictionaries.

## The observation: more than an image

The word "vision" in vision-language-action hides how much of the observation is not pixels. Bridge gives you two camera streams, and the two-camera setup is not incidental. A wrist camera sees the gripper and whatever it is about to touch, which is where the fine motor action happens; a fixed external camera sees the whole scene, which is where the object you were told to grab actually sits. Drop either one and a class of tasks becomes unsolvable from the images alone. This is why §16.2, when it gets to collecting your own data, spends real time on camera placement rather than treating it as a detail.

The `state` vector is the part beginners skip and then regret skipping. Those seven floats are the robot's proprioception: the (x, y, z) position of the end effector, its orientation, and the gripper's open-or-closed value. Proprioception matters because pixels alone are ambiguous about the robot's own configuration. Two frames can look nearly identical while the arm is in genuinely different poses, and a policy that only sees images has to infer its own joint state from the visual scene, which it does badly. Feeding the state in directly removes the guesswork.

One warning that costs people days. The images are stored as `uint8`, values 0 to 255, because that is how cameras produce them and how you save disk. Almost every model expects `float32` normalized into some range the vision backbone was pretrained on. The conversion happens in the data loader, not in the file, and forgetting it is one of the most common reasons a freshly loaded dataset trains to garbage.

## The action: whose convention, in what units

The `action` field is where robot datasets stop resembling image datasets and start being their own hard problem. Seven floats again, but these seven mean something completely different from the seven in `state`. In Bridge they are a *delta*: how much to move the end effector along each axis, how much to rotate it, and whether to change the gripper, all relative to the current pose. The policy is not predicting where the arm should be. It is predicting how it should change over the next timestep.

Three facts about that action vector decide whether two datasets can even be trained together, and none of them is visible until you check:

- **Frame of reference.** Is the action expressed in the robot's base frame or the end-effector frame? Bridge uses end-effector-relative deltas; other datasets use absolute base-frame poses. Mix them without conversion and the policy learns contradictions.
- **Units and scale.** Meters or centimeters, radians or a normalized (-1, 1) range. RT-1 discretizes each action dimension into 256 bins before training; Bridge keeps them continuous. The tokenization choice from §11.3 is a choice made *about this vector*.
- **Control rate.** Bridge runs near 5 Hz, so each action covers a fifth of a second of motion. A dataset recorded at 50 Hz packs ten times as many, much smaller, actions into the same physical motion. Same task, same robot, wildly different-looking action sequences.

This is the concrete face of the MDP-to-robot translation problem from §5.5, and it is why §15.2 can spend an entire section on how Open X-Embodiment harmonized sixty of these action conventions into one loader. Right now the point is narrower: an action is not a number, it is a number plus a convention, and the convention lives in the dataset's documentation rather than in the array.

## The language instruction, and how it got there

RT-1 and Bridge both attach a natural-language string to every episode, and that string is doing quiet, heavy work. It is the bridge between the web-scale language pretraining a VLA inherits and the specific motion in front of it. Without it you have plain behavior cloning, a policy that does one thing; with it you have a policy you can *tell* what to do.

Where the string comes from is less magical than it sounds and worth knowing before you trust it. In most tabletop datasets a human wrote it, either the teleoperator narrating what they were about to do or an annotator labeling the clip afterward. That means the labels are as noisy as human labels always are. "Put the spoon on the towel," "place spoon on cloth," and "move the spoon" might all describe the identical motion in three different episodes, and a few instructions will simply be wrong because someone mislabeled a clip at the end of a long session. Later datasets started auto-generating or templating instructions to cut that noise, which trades human sloppiness for template rigidity. Neither is clean. When a language-conditioned policy behaves oddly on a phrasing that should work, the instruction distribution in its training data is the first place to look.

## How it is stored: RLDS and why the format is not a footnote

Sixty datasets from twenty-one labs arrived in sixty formats. The thing that let anyone train across all of them at once was an agreement on storage, and for Open X-Embodiment that agreement is RLDS, the Reinforcement Learning Datasets format, which sits on top of TensorFlow Datasets. RLDS is just a convention for serializing exactly the episode-of-timesteps structure we walked through, plus the metadata a loader needs to shuffle and batch episodes efficiently without reading them all into memory.

Efficiency here is not academic. A single Bridge image is a quarter-megabyte before you even have the second camera or the next twenty-nine timesteps, so a serious pretraining mixture runs into terabytes and cannot live in RAM. The data loader streams episodes from disk, decodes the JPEG-compressed frames on the fly, normalizes the `uint8` images to `float32`, samples a window of timesteps, and hands the model a batch, all while the GPU is busy on the previous batch. Get that pipeline wrong and your expensive accelerator sits idle waiting for data, which is the single most common way a robot-learning run wastes money. §15.2 opens the RLDS plumbing further; Appendix D shows the loader setup end to end.

## Simulation datasets look almost the same

Everything above described real robot data, teleoperated by humans on physical arms. A large share of the benchmarks you will meet in §15.3 are simulated instead, and the reassuring news is that a simulated episode has the identical shape: observations, actions, a language instruction, terminal flags. LIBERO (Liu et al., 2023), the benchmark Chapter 2 already ran OpenVLA against, serves episodes that slot into the same data loader as Bridge with barely a change.

The differences are two, and they matter for opposite reasons. Simulated data is cheap and clean: no teleoperator fatigue, no camera calibration drift, perfectly consistent labels, and you can generate ten thousand episodes overnight instead of paying humans for a month. Simulated data is also fake in ways that leak: the physics of contact and friction are approximated, the rendered images carry a visual signature no real camera produces, and a policy that only ever saw simulation frames tends to stumble on the sim-to-real gap we first hit in §7.5. The dataset format is portable across the sim-real boundary. The competence a policy learns from it is not.

## What you can do now

You can read the phrase "we pretrained on 800,000 trajectories" and know what was actually on disk: a list of episodes, each a list of timesteps, each timestep a dictionary pairing multi-camera images and a proprioceptive state vector with a commanded action and a language string, serialized in a streaming format like RLDS so a loader can feed a GPU without melting. You can spot the three ways two datasets silently disagree, in action frame, in units, and in control rate. And you know that a simulation episode wears the same clothes as a real one while hiding a different body underneath. The next section takes the one dataset the whole field leans on, Open X-Embodiment, and shows how sixty of these piles were forced into a single trainable corpus.
