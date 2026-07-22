---
chapter: 12
section: 12.2
title: "OpenVLA: an open-source 7B-parameter VLA"
target_words: 2000
status: draft
prereqs: §12.1 (RT-2's recipe — a web-pretrained VLM taught to emit actions as tokens, and the closed-weights problem it left behind), §11.3 (256-bin action tokenization and its resolution ceiling), §11.1 (what a vision-language backbone contributes to a policy). Helpful, Chapter 10's diffusion heads as the contrast case, since OpenVLA deliberately does not use one.
key_refs:
  - Kim, M. J. et al. (2024). OpenVLA, An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan, A. et al. (2023). RT-2, Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818.
  - Padalkar, A. et al. (2023). Open X-Embodiment, Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Hu, E. et al. (2021). LoRA, Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
---

# 12.2  OpenVLA: an open-source 7B-parameter VLA

RT-2 settled the argument and then walked off with the evidence. The checkpoint sat on Google's TPUs, the backbone was proprietary, and the training mix was internal. A lab that wanted to build on the result could read the paper and admire the demos, but it could not download the weights, inspect what the fine-tuning had touched, or run its own ablation. OpenVLA (arXiv:2406.09246) exists to close that gap. Kim and colleagues took RT-2's core recipe, rebuilt it on open components, trained it on the public Open X-Embodiment corpus, and released everything: weights, PyTorch training code, and fine-tuning notebooks. The headline number is that this open 7B model beats the closed 55B RT-2-X by 16.5 percentage points of absolute success across 29 evaluation tasks, with seven times fewer parameters. That is the kind of result that reorganizes a field's default starting point, and for most of 2024 and 2025 OpenVLA was the checkpoint people reached for first.

This section does three things. It opens up the architecture so you can name which pieces are frozen, which get fine-tuned, and which are new, because the chapter's learning objective is precisely that you can inspect this checkpoint and account for every component. It explains why a plain discrete action head beat a fancier baseline. And it walks through the fine-tuning and quantization findings, since those are the parts you will actually touch when Chapter 16 has you adapt a VLA to your own robot.

## What the checkpoint is made of

OpenVLA is not built on a bespoke network. It sits on Prismatic-7B, an off-the-shelf open vision-language model, and the whole design philosophy is to change as little as possible. Prismatic has the three-part shape that every VLM in §11.1 shared: a visual encoder that turns an image into a set of patch embeddings, a small projector that maps those embeddings into the language model's input space, and a large language backbone that does the actual sequence modeling. The backbone is Llama 2 7B. The projector is a two-layer MLP. The visual encoder is the interesting choice.

Rather than a single vision transformer, Prismatic fuses two pretrained encoders whose features get concatenated: DINOv2, which is strong on spatial and geometric structure, and SigLIP, which is strong on semantic, language-aligned content. Together they run about 600M parameters. The bet is that a policy needs both halves. It has to know *what* the object is, which is SigLIP's job, and *where* it sits in the frame, which is where DINOv2 earns its place. The paper's ablations credit this fused encoder as one of the reasons OpenVLA generalizes better than a single-encoder baseline would.

The input is deliberately spare. One third-person camera image at 224×224, plus the language instruction as text. The authors tried 384×384 and found no improvement for the extra compute, which is a small but telling result: higher resolution helps generic VLM benchmarks but did not, at least here, help control. No wrist camera, no proprioceptive state vector, no history of past frames. A single picture and a sentence go in; seven numbers come out.

Those seven numbers are the added component. Following RT-2, OpenVLA discretizes each dimension of the 7-DoF end-effector action into 256 bins, with the bin edges set to span the 1st-to-99th percentile of that dimension in the training data rather than its full range, which keeps a few outlier actions from stretching the bins and wasting resolution on motions the robot almost never makes. The clever, cheap trick is how those bins enter the vocabulary. Llama's tokenizer has thousands of tokens; OpenVLA overwrites the 256 least-frequently-used ones (the last 256 entries) and reassigns them as action tokens. Nothing about the architecture changes. The output softmax is Llama's original language head, now occasionally emitting token IDs that happen to mean "translate +3cm in x" instead of a rare subword. Predicting an action is still just predicting the next token, trained with the same cross-entropy loss the language model was born with.

So the inventory reads: Llama 2 7B backbone (pretrained, then fine-tuned), fused DINOv2+SigLIP encoder (pretrained, then fine-tuned), MLP projector (fine-tuned), and 256 repurposed vocabulary slots (the only genuinely new parameters, and there are almost none of them). Pretraining on the 970k robot episodes fine-tunes all of it end to end.

## The vision-encoder finding worth remembering

There is one result in the pretraining setup that contradicts standard VLM practice, and it is the kind of detail that separates people who read the paper from people who skimmed it. When you train a vision-language model, the received wisdom is to *freeze* the visual encoder. A frozen encoder keeps the robust, internet-pretrained features intact, and unfreezing it usually costs you generalization. Prismatic itself was trained that way.

OpenVLA does the opposite, and the authors are blunt that fine-tuning the vision encoder during robot training was crucial for good performance. Their reading is that internet-pretrained features capture plenty of high-level semantics but not enough of the fine spatial detail a manipulation policy needs, the precise geometry of a gripper approaching a drawer handle, so the encoder has to adapt to the control task rather than stay locked to its captioning-era weights. If you were going to guess wrong about this checkpoint, this is where you would do it. Note it now, because it comes back when we discuss which layers to unfreeze during your own fine-tuning.

## Why a discrete head, and how it stacks up

By 2024 the field already had Diffusion Policy and ACT (Chapter 10) showing that continuous action heads produce smoother, more multimodal trajectories than a discrete classifier can. OpenVLA looked at that and chose the discrete head anyway. The reason is architectural honesty about what the paper was testing. Bolting a diffusion head onto a Llama backbone means the action no longer flows through the language model's own output layer, which breaks the "actions are just tokens" simplicity that lets you reuse the entire VLM training stack unchanged. OpenVLA's contribution is the open recipe and the fine-tuning story, not a new action representation, so it kept the representation dead simple and spent its novelty budget elsewhere. Octo (§12.3) makes the other choice, and comparing the two is the cleanest way to see what a diffusion head buys and what it costs, which is why the chapter puts them back to back.

The simple head was not a handicap. Across the 29-task generalist evaluation spanning multiple embodiments, OpenVLA outscored RT-2-X by 16.5 points absolute, and in the multi-task-from-scratch fine-tuning comparisons it beat Diffusion Policy by 20.4 points. The lesson is not that discrete beats continuous in general; it doesn't, and Chapter 13 is largely about the cases where it loses badly. The lesson is that a 7B web-pretrained backbone trained on nearly a million diverse episodes has enough raw competence that the crudeness of its action encoding stops being the bottleneck, at least for the slow tabletop manipulation these benchmarks measure.

## Fine-tuning: what to unfreeze, and what it costs

This is the part of OpenVLA that mattered most for adoption, and it is the second half of §12.1's complaint answered. RT-2 never showed you how to efficiently specialize the model to a new robot; OpenVLA ran the experiment. The authors compared several strategies for adapting the pretrained checkpoint to a new task, and the results form a small decision table you can more or less memorize.

Fine-tuning only the last layer plus the token embeddings: cheap, and bad, landing around 30% success. Freezing the vision encoder and tuning everything else: also weak, near 47%, which is the mirror image of the pretraining finding above; if adapting the visual features matters during pretraining, it matters during fine-tuning too. "Sandwich" fine-tuning, which unfreezes the vision encoder, the token embeddings, and the last layer while leaving the middle of the LLM alone: much better at roughly 62%, and lighter on memory because it never touches the full backbone. LoRA (arXiv:2106.09685), the low-rank adaptation method that inserts small trainable matrices into the frozen backbone: best overall at about 68%, matching sandwich on quality while using far less compute.

LoRA is the recommendation you walk away with. It fine-tunes OpenVLA on a single A100 in a fraction of the time full fine-tuning demands, roughly an 8x compute reduction, and it does so without a quality penalty. That is what makes the "fine-tune a foundation VLA on your own 50 demonstrations" workflow of the chapter exercise realistic on hardware a lab actually owns, instead of a hyperscaler's cluster.

## Running it: quantization and the honest latency number

The last open contribution is inference cost. Loaded in bfloat16, OpenVLA needs about 15GB of GPU memory and runs at roughly 6Hz on a single RTX 4090, a consumer card. The paper then shows you can quantize the model to 4-bit precision for serving with no measurable drop in success rate, which pushes it onto smaller GPUs still. Six hertz is worth sitting with. It is fast enough for the deliberate pick-and-place tasks in the benchmarks, and it is nowhere near fast enough for reactive, contact-rich control that needs to close a loop at 50Hz or more. This is the same latency wall RT-2 hit, just at a friendlier price point, and it is exactly the pressure that §13's flow-matching heads and §14's dual-system split are built to relieve. An open 7B model you can run on a gaming GPU is a real gift; it is not a real-time controller.

## Where this leaves us

OpenVLA converted RT-2 from a demonstration into infrastructure. Anyone could now download a state-of-the-art generalist policy, see exactly which components were pretrained and which were adapted, fine-tune it on consumer hardware with LoRA, and serve it quantized. The dataset that made this possible, Open X-Embodiment (arXiv:2310.08864), is important enough that §12.4 is devoted to it. But OpenVLA also drew a clean boundary around its own choices: a discrete token head, a single image, no history, six hertz. Each of those was a decision, not a law, and the next model in the chapter reopens the first one. Octo keeps the open, reproducible spirit and asks what changes when the action head is a small diffusion model instead of a token classifier.
