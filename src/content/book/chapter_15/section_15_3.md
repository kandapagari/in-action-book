---
chapter: 15
section: 15.3
title: "LeRobot: Hugging Face's dataset format, hub, and community-contributed robot data"
target_words: 2000
status: draft
prereqs: §15.1 (an episode's anatomy as a list of timestep dictionaries — LeRobotDataset is one concrete on-disk encoding of exactly that), §15.2 (Open X-Embodiment and RLDS — LeRobot is the other storage lineage, and the contrast between them is the point). Helpful, §12.3 (Octo's diffusion head) and §13.2 (π0), both of which ship as loadable policies inside LeRobot.
key_refs:
  - Cadene, R., Alibert, S., Soare, A., Gallouedec, Q., Zouitine, A., Wolf, T. et al. (2024). LeRobot, State-of-the-art machine learning for real-world robotics in PyTorch. Hugging Face. github.com/huggingface/lerobot.
  - Shukor, M. et al. (2025). SmolVLA, A Vision-Language-Action Model for Affordable and Efficient Robotics. arXiv:2506.01844.
  - Open X-Embodiment Collaboration, Padalkar, A. et al. (2023). Open X-Embodiment, Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
---

# 15.3  LeRobot: Hugging Face's dataset format, hub, and community-contributed robot data

Section 15.2 left you with a picture of the field's storage habits circa 2023: sixty academic datasets, RLDS serialization on top of TensorFlow Datasets, sampling weights tuned by hand. That lineage produced Open X-Embodiment and the models trained on it. It also came with a tax. RLDS is a TensorFlow-first format, the tooling assumes you are comfortable in that ecosystem, and contributing a new dataset meant learning a serialization convention that most robotics labs touched exactly once. A second lineage grew up alongside it, aimed at a different user: the person with one low-cost arm on a desk who wants to record a hundred episodes this afternoon and train tonight. That lineage is LeRobot.

LeRobot is Hugging Face's open-source robotics library, released in 2024 and written for PyTorch rather than TensorFlow (Cadene et al., 2024). It bundles four things that the RLDS world kept separate: a dataset format, a hosting hub, a set of pretrained policies you can load in two lines, and driver support for cheap hardware you can build from a parts list. The bet behind it is that robot learning was gated less by ideas than by friction, and that collapsing the distance between "I have an arm" and "I have a trained policy" would matter more than any single architecture. Whether that bet paid off is something you can now check on the Hub yourself: the number of community-uploaded robot datasets there went from a handful at launch to several thousand within about a year.

## LeRobotDataset: the same episode, a different on-disk shape

Recall the timestep dictionary from §15.1: images, a proprioceptive state vector, an action, a language instruction, some boolean flags. LeRobotDataset stores exactly that structure, but the encoding decisions differ from RLDS in ways that reveal what the format is optimized for.

The tabular data (states, actions, timestamps, episode indices, task indices) lives in Parquet files, the columnar format from the Apache Arrow world that Hugging Face's `datasets` library already speaks. The camera streams do not sit in those Parquet files as raw arrays. They are encoded as MP4 video, one file per camera per episode (or chunked across episodes in later format versions), and the Parquet rows carry pointers plus timestamps into those videos rather than the pixels themselves. This is the single most consequential design choice in the format, so it is worth dwelling on.

Why video? Go back to §15.1's arithmetic. A single 256×256 RGB frame is a quarter-megabyte uncompressed; a two-camera, 30-timestep episode is fifteen megabytes of raw pixels, and a serious dataset is tens of thousands of episodes. Storing frames as JPEG helps, but consecutive frames in a robot episode are almost identical (the arm moves a centimeter, the background does not move at all) and JPEG throws away none of that inter-frame redundancy. Video codecs exist precisely to exploit it. Encoding an episode's camera stream as H.264 or AV1 routinely shrinks it by an order of magnitude over per-frame JPEG, because the codec stores one keyframe and then only the deltas. The cost is that reading a random timestep now means seeking into a compressed video and decoding, which is more expensive per-frame than reading a JPEG; LeRobot's loader hides this behind the same streaming-and-prefetch machinery §15.1 described, decoding on background workers while the GPU chews the previous batch. For datasets dominated by camera data, which is nearly all of them, the trade lands well on the storage side.

A LeRobotDataset on disk, then, is a directory with three kinds of content: the Parquet files holding the numeric time series, the MP4 files holding the video, and a small pile of JSON metadata. The metadata is where the format earns its keep for anyone loading a dataset they did not create. An `info.json` declares the schema: which features exist, their shapes and dtypes, the frames-per-second, the codec used for the video. Separate files hold the mapping from task-index to natural-language instruction and the per-feature statistics (mean, std, min, max) that a policy needs for input normalization. That last item deserves emphasis: the normalization statistics travel with the dataset. One of the recurring ways a fine-tuning run in §16 fails is a mismatch between the statistics a checkpoint was trained under and the statistics of the new data, and LeRobot's decision to ship stats as first-class metadata is a direct attempt to make that failure loud instead of silent.

## Querying time: the delta-timestamp trick

Here is a problem every robot data loader has to solve and RLDS solves less ergonomically. A policy almost never wants a single timestep. Behavior-cloning models predict a chunk of future actions at once (ACT and Diffusion Policy from Chapter 10 both do), and many architectures condition on a short stack of past frames. So the loader has to hand you windows, not points: "give me the current observation, the previous two frames, and the next sixteen actions."

LeRobotDataset expresses this with a `delta_timestamps` argument. You ask for a feature at a list of time offsets relative to the current frame, in seconds, and the loader assembles the window for you:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

delta_timestamps = {
    "observation.image": [-0.2, -0.1, 0.0],   # two past frames + current
    "action": [t / 30 for t in range(16)],     # next 16 actions at 30 fps
}
dataset = LeRobotDataset("lerobot/aloha_static_coffee", delta_timestamps=delta_timestamps)
sample = dataset[0]
# sample["observation.image"] -> tensor [3, C, H, W]
# sample["action"]            -> tensor [16, action_dim]
```

The offsets are in seconds, not indices, which is the detail that makes the format robust across recording rates. A dataset logged at 30 fps and one logged at 50 fps both answer "the action 0.1 seconds from now" correctly, because the loader resolves the request against the real timestamps stored in the Parquet rows rather than counting rows. This is the frequency-mismatch headache from §15.2 handled at the query layer instead of being left for the model to average over. It does not make two datasets recorded at different rates identical, but it does mean your code asks for physical time and gets it, regardless of how densely a given dataset sampled that time.

## The Hub, and what "community-contributed" actually buys you

RLDS datasets tend to live wherever the lab that made them chose to park them. A LeRobotDataset lives on the Hugging Face Hub by default, addressed by a `repo_id` like `lerobot/aloha_static_coffee`, versioned like a Git repository, and loadable by anyone with the string. `LeRobotDataset("some-user/their_dataset")` pulls it the same way you pull a language-model checkpoint. Uploading your own is a single `push_to_hub()` call after recording.

The effect of putting robot data behind the same one-line pull as a pretrained model is quantitative, not just cosmetic. It dropped the cost of sharing a dataset close to zero, and the Hub filled with community uploads: teleoperation runs on SO-100 and SO-101 arms, ALOHA bimanual data, Koch-arm datasets, task-specific collections someone recorded for a workshop and left public. Several thousand robot datasets now sit there, most of them small, many of them noisy, a few of them excellent. This is a different animal from Open X-Embodiment. OXE is a curated, harmonized corpus assembled by a collaboration; the LeRobot Hub is a bazaar. The upside is coverage and immediacy: if you own an SO-101, someone has probably already published data on a task near yours. The downside is exactly what you would expect from a bazaar: no guarantee of quality, no harmonization across uploads, and camera setups, control rates, and labeling conventions that vary wildly from one `repo_id` to the next. §15.1's warning about silently disagreeing datasets applies with full force here, minus the curation layer OXE provided.

Both storage lineages now coexist, and the field is bilingual rather than converged. Conversion tools move OXE's RLDS datasets into LeRobotDataset form so PyTorch-first practitioners can use them, and the two formats trade influence rather than one replacing the other.

## Policies and hardware, in the same box

Two features push LeRobot past being merely a dataset format. First, it ships pretrained and trainable policies as importable modules: ACT, Diffusion Policy, TDMPC, VQ-BeT, and, the reason this matters for a book about foundation models, π0 and SmolVLA. You can load a π0 checkpoint and run inference on a LeRobotDataset without leaving the library, which is why Chapter 16's fine-tuning walkthrough leans on it.

SmolVLA (arXiv:2506.01844) is the clearest expression of the project's thesis. It is a compact VLA, roughly 450 million parameters, trained substantially on community-contributed LeRobot Hub datasets rather than on a proprietary fleet corpus, and designed to run on consumer hardware including a single GPU or even a CPU. It exists to prove that the bazaar produces something usable: that data recorded by hobbyists and small labs on sub-$1,000 arms, pooled through the Hub, can train a VLA that works. Set it beside the fleet-scale models of §15.2, the million-trajectory teleoperation programs behind LingBot-VLA and AGIBot, and you have the two poles of the field's data strategy. One pole pours capital into private industrial-scale collection; the other tries to make the open long tail add up to something. SmolVLA is the evidence that the second pole is not just idealism.

Second, LeRobot supports the low-cost hardware that makes the whole loop reachable. The SO-100 and SO-101 arms, the Koch and Moss designs, and ALOHA-style bimanual setups all have drivers in the library, with parts lists and assembly instructions that put a working teleoperation rig in the low hundreds of dollars. This closes the circuit that Chapter 16 will walk in full: build the arm, teleoperate it to record a LeRobotDataset, push it to the Hub, fine-tune SmolVLA or π0 on it, deploy the policy back to the same arm. The format, the hub, the policy, and the robot are one toolchain, and that integration is the actual product.

## What you can do now

You can read `LeRobotDataset("some-user/task")` and know what lands on disk: Parquet files holding the numeric time series, MP4 files holding each camera as compressed video, and JSON metadata carrying the schema, the task strings, and the normalization statistics that keep a fine-tuning run honest. You can explain why the format stores video instead of frames and what that trade costs at read time. You can use `delta_timestamps` to pull past-and-future windows in physical seconds, sidestepping the control-rate mismatch that RLDS left to the model. And you can place the LeRobot Hub correctly against Open X-Embodiment: not a curated corpus but an open bazaar of several thousand community datasets, whose payoff — proof that pooled hobbyist data can train a working VLA — is SmolVLA. The next section leaves datasets for evaluation and walks through the simulation benchmarks, LIBERO, CALVIN, RoboCasa, and SimplerEnv, that let you compare two policies without owning any of this hardware.
