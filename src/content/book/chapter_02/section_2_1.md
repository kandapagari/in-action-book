---
chapter: 2
section: 2.1
title: What we are going to build, and what is hidden inside
target_words: 2000
status: draft
prereqs: §1.2 (the three-slot anatomy), §1.4 (the foundation/VLA family); basic Python
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2. arXiv:2307.15818.
---

# 2.1  What we are going to build, and what is hidden inside

Chapter 1 was the vocabulary chapter. This one is the running-code chapter. By the end of it you'll have a Vision-Language-Action model, specifically OpenVLA-7B (Kim et al., 2024, arXiv:2406.09246), running on your machine: taking an RGB image of a simulated tabletop plus a natural-language instruction ("put the red block in the bowl") as input, and emitting low-level robot commands that drive a successful execution in the LIBERO simulator (Liu et al., 2023, arXiv:2306.03310). You'll also have it failing, repeatedly, in three characteristic ways that the rest of the book is structured to explain.

The order matters here: working first, theory after. The hard ideas in Chapters 3 through 10, what the chain rule means for a thirty-dimensional action space, how a policy gradient differs from a supervised gradient, why a transformer trained on text turns out useful for motor control, read easier once you've already watched a VLA do the right thing on your own laptop and the wrong thing on your colleague's. This chapter is the first-encounter artifact those later chapters keep referring back to. Don't skip the code. The muscle memory of standing up a model isn't something you acquire any other way.

## The end state, in one paragraph

Here's concretely what you're about to run. An RGB image of a simulated tabletop scene, call it 256×256×3, uint8, comes off the LIBERO simulator. OpenVLA tokenizes the image (through a fused SigLIP and DINOv2 visual backbone) and the natural-language instruction (through a Llama-2-7B text tokenizer). The combined token sequence passes through a 7-billion-parameter transformer, which produces a short sequence of discrete action tokens, seven of them, one per axis of an end-effector pose delta plus a gripper bit. A detokenizer maps those tokens back into a 7-vector of continuous values. LIBERO steps the simulator with that 7-vector as the commanded action, returns the next image, and the loop repeats at roughly 5 to 15 Hz. Twenty to two hundred steps later, the task either succeeds, the episode times out, or the robot does something interesting and wrong. There's no magic between the picture and the motion, just a transformer doing next-token prediction over a vocabulary whose bottom 256 entries happen to mean "discretized action bins."

That paragraph is the spine of the chapter. §2.2 covers the environment setup so the spine can run. §2.3 walks the inference loop line by line. §2.4 covers what happens when the loop fails, which is most of the time on the first try. §2.5 maps each piece of the loop to the chapter that explains it in depth. §2.6 closes things out.

## What is hidden inside a VLA

A modern VLA isn't a monolith labeled "AI." It's a stack of commitments, and much of the rest of this book is a guide to choosing among the alternatives at each layer. Four of those commitments matter at the level of the inference loop you're about to run.

The first commitment is how images become tokens. OpenVLA uses a fused SigLIP-plus-DINOv2 visual encoder: two pretrained vision transformers running in parallel, their patch embeddings concatenated, projected down, and fed into the language-model trunk. Other VLAs choose differently. RT-2 (Brohan et al., 2023, arXiv:2307.15818) uses PaLI-X's native vision encoder; π0 (Black et al., 2024, arXiv:2410.24164) uses a separate vision trunk with a flow-matching head. This commitment determines how the model sees objects it hasn't been shown before. Chapter 11 dissects vision-language pretraining, and Chapter 12 compares OpenVLA's choice against its contemporaries.

The second commitment is how the policy turns intent into action. OpenVLA belongs to the discrete-token family: actions are a fixed-length sequence of categorical tokens, predicted with ordinary cross-entropy loss and decoded greedily at inference time. The alternative is a continuous-action head, a small diffusion or flow-matching model conditioned on the transformer's hidden state, sampling a smooth action chunk instead. RT-1 and RT-2 are discrete; π0 and Octo (arXiv:2405.12213) are continuous. Each choice trades off differently between simplicity, smoothness, and inference latency. Chapter 10 covers these heads in depth. For now, just know which family the model you're running belongs to (discrete), and that discrete-versus-continuous is the single largest architectural fork in modern VLAs.

The third commitment is what dataset taught the model physics. OpenVLA was pretrained on 970,000 robot trajectories drawn from Open X-Embodiment (O'Neill et al., 2023, arXiv:2310.08864), an aggregated corpus of teleoperated demonstrations from 22 institutions across 22 distinct robot embodiments. This mixture matters because the action vocabulary gets calibrated on the empirical distribution of motions inside it: bins sit where the training data spent its time. A motion common in the training data gets fine-grained resolution; a rare motion gets a single bin and a coarse output. Chapter 12 returns to Open X-Embodiment in detail. For §2.4, the practical upshot is that failures are often distributional failures, not modeling failures.

The fourth commitment, and the one most people miss, is what evaluation looks like. A cherry-picked clip of a robot succeeding once is easy to produce. A success rate measured across twenty seeds, three initial conditions, and two camera variations is hard to produce, and it's the only thing that should actually make you believe a result. Chapter 15 covers evaluation in depth; the version you'll do in §2.4 is small, a handful of LIBERO rollouts with logged seeds, but the discipline is the same discipline, and it scales.

## A first concrete example

Suppose the LIBERO task is `libero_object` task 0, and the instruction is "put the apple in the bowl." A single inference step looks like this:

```
input image:   (256, 256, 3)  uint8   tabletop with apple, bowl, distractor
input prompt:  "In: What action should the robot take to put the apple in the bowl?\nOut: "
visual tokens: 256 patches × 2 encoders → 256 fused image tokens
text tokens:   ~20 tokens from the Llama-2 tokenizer
forward pass:  7B-parameter decoder; greedy decode for 7 steps
output tokens: [31882, 31894, 31881, 31905, 31888, 31875, 31999]
detokenized:   dx=-0.012, dy=+0.034, dz=-0.018, droll=+0.001, dpitch=-0.002, dyaw=+0.020, grip=1.0
sim step:      env.step([-0.012, 0.034, -0.018, 0.001, -0.002, 0.020, 1.0])
```

Those seven output tokens aren't text. They're integer IDs that happen to sit in the bottom 256 entries of the Llama-2 vocabulary, repurposed during training to mean "discretized action bin 0 through 255." When token 31882 gets decoded, the detokenizer subtracts a vocabulary base, gets a bin index in [0, 255], and maps that bin to a continuous value using the per-axis minimum and maximum from the training-data normalization statistics. The `unnorm_key` parameter in OpenVLA's `predict_action` call selects which embodiment's statistics to use. Pass the wrong key, and your output silently goes off-scale.

That seven-token output is the entire policy decision. Everything else, the billion parameters, the vision encoder, the language model, is upstream machinery whose whole job is choosing those seven integers well. When this chapter teaches you to debug, the seven tokens are the first thing you print.

## Three load-bearing ideas you will meet again

Hidden inside this chapter's cheerful demo are three ideas that resurface again and again through the rest of the book.

Action tokenization turns control into a sequence-modeling problem. The architectural simplicity that follows, same transformer, same cross-entropy loss, same training infrastructure as a language model, is the reason a 7-billion-parameter VLA was feasible to train in the first place. The cost is discreteness: a 256-bin bucket over ±10 cm of translation gives you about 0.78 mm of resolution, fine for tabletop pick-and-place, coarse for threading a needle. Chapters 10 and 13 cover what reclaiming continuous resolution buys and what it costs.

Pretraining on internet-scale vision-language data buys semantics that pure robot datasets can't. OpenVLA's language understanding doesn't come from its 970,000 robot trajectories; it comes from the trillions of tokens of text and billions of images its backbones saw before any robot data ever touched them. That's what lets it parse "put the apple in the bowl" rather than only "task 17 of suite 4." Chapter 11 retraces the path, CLIP, BC-Z, RT-1, that made this transfer possible.

Evaluation is easy to fake in a cherry-picked clip and hard to do honestly across seeds, cameras, and initial conditions. A VLA that succeeds 9 out of 10 times on one initial pose isn't the same as one that succeeds 5 out of 10 across twenty randomized poses. §2.4 introduces the discipline of logging seeds and initial conditions, and Chapter 15 generalizes it. The temptation to publish the good clip is a permanent feature of this field. The only protection against fooling yourself is the rigor of the logs you actually keep.

## What the chapter does not cover

A short list of things explicitly out of scope here. None of the training happens in this chapter; you'll run a pretrained checkpoint and won't modify its weights. Fine-tuning waits for Chapter 16. None of the math behind the transformer gets derived here, since Chapter 8 handles that. None of the safety wrapping a real deployment would require shows up here either, that's Chapter 17's job. Real-robot transfer isn't discussed at all; everything in this chapter runs in simulation. These omissions exist to keep the first VLA you run small enough to hold in your head. The rest of the book widens the aperture from here.

§2.2 begins the practical setup: GPU and software requirements, the weights download, and the LIBERO smoke test you should run before the model touches anything.
