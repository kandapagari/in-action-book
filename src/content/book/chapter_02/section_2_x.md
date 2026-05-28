---
chapter: 2
section: 2.x
title: Hands-on exercise + chapter references
target_words: 2000
status: draft
prereqs: §2.1–§2.6; a working OpenVLA-on-LIBERO loop; the three silent failures from §2.4; willingness to break the loop on purpose
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310.
  - O'Neill et al. (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864.
  - Brohan et al. (2022). RT-1. arXiv:2212.06817.
  - Brohan et al. (2023). RT-2. arXiv:2307.15818.
  - Octo Model Team (2024). Octo. arXiv:2405.12213.
  - Black et al. (2024). π0. arXiv:2410.24164.
  - Pertsch et al. (2025). FAST. arXiv:2501.09747.
  - NVIDIA (2025). GR00T N1. arXiv:2503.14734.
  - Sapkota et al. (2025). VLA Models: Concepts, Progress, Applications and Challenges. arXiv:2505.04769.
---

# 2.x  Hands-on exercise + chapter references

The Chapter 1 exercise was a reading-and-classification exercise; you sat
with four papers and a table. The Chapter 2 exercise is a coding exercise.
You already have a working OpenVLA-on-LIBERO loop on your machine, so you
already have everything the exercise needs. What you do not yet have is the
muscle that turns a working loop into a reliable diagnostic instrument, and
the only way to build it is to break the loop on purpose and read the
symptoms cold. Budget about two hours. The deliverable is a short text file
— not a model, not a notebook, not a chart — that captures three reproduced
failures, the diagnostic signature of each, and the one-line fix.

## Exercise 2.x.1 — Reproduce the three silent failures on demand

Take the §2.3 inference loop and copy it to a new file called
`failure_drills.py`. Wrap the inner loop in a function that accepts three
toggles, defaulting to the correct values:

```python
def run_episode(flip_image=True,
                prompt_template="default",
                unnorm_key="libero_object",
                seed=0,
                max_steps=400):
    ...
```

Inside the function, branch on each toggle. `flip_image=False` removes the
`[::-1]` slice. `prompt_template="rephrased"` substitutes the string
`"What action should the robot take? {instruction}"` for the canonical
`In:`/`Out:` scaffold. `unnorm_key="bridge_orig"` substitutes the Bridge V2
embodiment statistics from Open X-Embodiment (O'Neill et al., 2023,
arXiv:2310.08864) for the LIBERO ones. Seed both the LIBERO suite and the
Python `random` and `numpy` generators from the `seed` argument so the
initial condition is fixed across runs.

Run the function four times: once with all defaults (the baseline), then
three times with exactly one toggle flipped per run. Log to a `.jsonl`
file, one line per step, with the five fields from §2.3: step index,
`img_mean`, the seven-element rounded action vector, `reward`, and a
single string tag (`"baseline"`, `"no_flip"`, `"bad_prompt"`,
`"bad_unnorm"`) for which run produced the line. The file should end up
between 1600 and 2400 lines for four runs of roughly 400 to 600 steps
each; that is small enough to grep.

The diagnostic step is the one that matters. Open the `.jsonl` in whatever
column-oriented tool you prefer (`jq`, pandas, a spreadsheet) and write
three predicates — one for each failure mode — that take only the action
vector and `img_mean` columns as input and classify a row as belonging to
one of the four runs. Concretely:

1. *Bad prompt:* mean over time of `np.linalg.norm(action[:6])` is below
   roughly 0.05, every step, regardless of `img_mean`. The arm twitches
   near zero; the language tower never enters action-decoding mode.
2. *Bad `unnorm_key`:* `np.max(np.abs(action[:3]))` exceeds 0.1 m on a
   non-trivial fraction of steps. The translation magnitudes are off by
   the LIBERO-vs-Bridge ratio (roughly 3 to 5×).
3. *No flip:* harder. `img_mean` is unchanged from baseline, and the
   action magnitudes are reasonable. The signature is in the *trajectory
   shape*, not in any per-step statistic. The clearest tell is that the
   episode never raises `reward > 0` and `done` never goes True, but
   `action` magnitudes look healthy and gripper commands cycle at roughly
   the same rate as the baseline. The diagnosis is by elimination — the
   model behaves like a working model on a world that does not exist.

Write the three predicates as functions, run them on the four logs, and
verify that each tagged run matches exactly one predicate. If the
predicates disagree with the tags — say, the `no_flip` log fires the
bad-prompt predicate — your toggle is wrong, not the predicate, and you
should re-read §2.4 before moving on.

Save the four `.jsonl` files and the predicate file to a directory called
`drills/`. You will return to it in Chapter 17, where the same predicates
become the kernel of a runtime monitor that runs on a real robot. The
drills you write today are a first sketch of the diagnostic vocabulary
that §17.2 will formalize.

## Exercise 2.x.2 — The 50-episode sweep

Take the corrected loop (all defaults, no toggles flipped). Sweep it across
fifty seeds on `libero_object` task 0. Each episode takes roughly 60
seconds on a 4090, so the sweep is about 50 minutes of wall clock; do this
overnight or in parallel with something else. Log the same five fields per
step, plus the per-episode final `reward`, the step at first nonzero
gripper command (the seventh action element crossing 0.5), and the
wall-clock time.

The deliverable is a single number — the success rate over fifty episodes,
with the standard error — and a one-sentence comparison with the published
number from Kim et al. (2024, arXiv:2406.09246), which reports roughly 88
to 90% on `libero_object` for the released checkpoint. If your number is
within ±3 percentage points, your install is healthy. If it is more than
five points below the paper, treat the gap as a bug, not a result, and
work backwards through §2.2 (environment) and §2.3 (loop) until you find
the discrepancy. The most common cause is mixed-precision drift from a
non-bfloat16 dtype; the second most common is a stale
`dataset_statistics.json` from a pre-release OpenVLA checkpoint.

Then produce one descriptive plot. The plot is not a metric; it is a
discipline. On the x-axis, the step at first nonzero gripper command;
on the y-axis, a histogram of episodes, colored by final outcome
(success vs. failure). You should see what §2.4 promised: successful
episodes cluster between step 60 and 100, failed episodes are bimodal at
"never closes" and "closes immediately." That single plot is the most
informative one-pager you can produce about a discrete-action VLA, and
the construction generalizes — Chapter 15 builds the same plot for
fine-tuned policies and real-robot rollouts.

## Exercise 2.x.3 — A one-page paper read

Open the OpenVLA paper (Kim et al., 2024, arXiv:2406.09246) and read
Section 3 only, the architecture section. Do not read the experimental
results yet; do not read the related work. For each of the six boxes in
§2.3 — vision encoder, language tokenizer, transformer trunk, action
head, detokenizer, simulator/embodiment — find the sentence in Section 3
that names the design choice and write it down. The vision encoder is
the easy one (a fused SigLIP and DINOv2); the detokenizer is the
trickiest one and lives in the brief discussion of per-embodiment action
normalization.

The exercise is the *practice of reading a VLA paper by box*. When you
get to Chapter 12, you will do the same thing for RT-2 (Brohan et al.,
2023, arXiv:2307.15818) and Octo (Octo Model Team, 2024, arXiv:2405.12213);
when you get to Chapter 13, for π0 (Black et al., 2024, arXiv:2410.24164).
The structure is identical. Doing it once now, on the model you have
already run, makes the next three times much faster.

## Exercise 2.x.4 — One sentence per failure

Close the laptop. On paper or in a text file, write three sentences. Each
sentence has the form: *"If the robot is doing X, the most likely cause
is Y, and the diagnostic is Z."* The three X's are the symptom
descriptions from §2.4: "moves in slow random scribbles near the start
pose", "moves decisively in the wrong direction", "slams the arm toward
joint limits." If you can write all three from memory without checking
§2.4, you have internalized the silent-failure taxonomy. If you cannot,
re-read §2.4 once and try again the next day. The three-sentence form is
deliberately compact; it is the version you will want to have in your
head when you are debugging a real fine-tune at 11pm and have no time
to grep.

## Chapter 2 reading list

The works below are the ones cited in §2.1–§2.6. They are grouped by
purpose. The full bibliography for the whole book is in Appendix E.2;
this list is the chapter-local subset.

### The model and its lineage

- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). "OpenVLA: An
  Open-Source Vision-Language-Action Model." arXiv:2406.09246. The model
  you ran end to end. Read Section 3 for the architecture and Section 5
  for the LIBERO numbers.
- Brohan, A., Brown, N., Carbajal, J., et al. (2022). "RT-1: Robotics
  Transformer for Real-World Control at Scale." arXiv:2212.06817. The
  paper that introduced the action-tokenization-into-a-transformer
  recipe OpenVLA inherits.
- Brohan, A., Brown, N., Carbajal, J., et al. (2023). "RT-2: Vision-
  Language-Action Models Transfer Web Knowledge to Robotic Control."
  arXiv:2307.15818. The proof that a VLM backbone could carry actions;
  the immediate predecessor of OpenVLA's design.
- Octo Model Team (2024). "Octo: An Open-Source Generalist Robot Policy."
  arXiv:2405.12213. The continuous-action sibling of OpenVLA, with a
  diffusion head rather than discrete tokens. Useful as a contrast.
- Black, K., Brown, N., Driess, D., et al. (2024). "π0: A Vision-
  Language-Action Flow Model for General Robot Control." arXiv:2410.24164.
  The flow-matching successor; revisited in Chapter 13.
- Pertsch, K., et al. (2025). "FAST: Efficient Action Tokenization for
  Vision-Language-Action Models." arXiv:2501.09747. The follow-up
  tokenization scheme that addresses the discreteness ceiling raised in
  §2.4.
- NVIDIA (2025). "GR00T N1: An Open Foundation Model for Generalist
  Humanoid Robots." arXiv:2503.14734. The dual-system humanoid VLA that
  Chapter 14 returns to.

### The data and the simulator

- O'Neill, A., Rehman, A., Maddukuri, A., et al. (2023). "Open
  X-Embodiment: Robotic Learning Datasets and RT-X Models."
  arXiv:2310.08864. The dataset that underwrites OpenVLA pretraining
  and the source of the `bridge_orig` `unnorm_key` from §2.4. Chapter
  12 covers it in depth.
- Liu, B., Zhu, Y., Gokmen, C., et al. (2023). "LIBERO: Benchmarking
  Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310.
  The simulator you ran. Read the task-suite breakdown (`libero_object`,
  `libero_spatial`, `libero_goal`, `libero_10`) to understand what
  Chapter 15 will measure.

### Field surveys (for breadth)

- Sapkota, R., et al. (2025). "Vision-Language-Action Models: Concepts,
  Progress, Applications and Challenges." arXiv:2505.04769. The survey
  you were asked to mark up in Exercise 1.x.3. By now several more of
  its sentences should make sense.
- "A Survey on Pure Vision-Language-Action Models." arXiv:2509.19012.
- "Efficient Vision-Language-Action Models: A Survey." arXiv:2510.24795.
- "Anatomy of a VLA: Modules, Milestones, Challenges." arXiv:2512.11362.

### Background you may want nearby

- Touvron, H., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned
  Chat Models." The LLM whose vocabulary OpenVLA repurposes for action
  tokens; the bottom-256-entries trick from §2.1 lives or dies on this.
- Zhai, X., et al. (2023). "Sigmoid Loss for Language Image Pre-Training."
  (SigLIP.) One half of OpenVLA's vision tower.
- Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features
  without Supervision." arXiv:2304.07193. The other half.
- Todorov, E., Erez, T., & Tassa, Y. (2012). "MuJoCo: A Physics Engine
  for Model-Based Control." IROS 2012. The simulator underneath LIBERO.

## Chapter summary

Chapter 2 stood up a complete VLA on a laptop and ran it. You watched it
succeed, watched it fail in three named ways, and now have three reproduced
drills, a fifty-episode sweep, and a one-page architectural read of the
OpenVLA paper sitting in a directory on your disk. With that artifact in
hand, you can locate the six boxes inside a `predict_action` call you have
never seen before, predict the cause of a silent failure from its symptom,
estimate the success-rate variance of a VLA on a finite-episode evaluation,
and place any new VLA paper into the design-choice grid the rest of Part 4
will use. Chapter 3 returns to the math the rest of the book assumes —
linear algebra, probability, and a fifty-line PyTorch training loop —
before Chapter 4 begins the lineage that produced the model you just ran.
