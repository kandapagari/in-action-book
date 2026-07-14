---
chapter: 11
section: 11.1
title: "CLIP and the multimodal pretraining moment"
target_words: 2000
status: draft
prereqs: §3.2 (KL and cross-entropy as a training signal), §6.1 (why imitation is the dominant signal), §8.1 (the transformer, for control). Helpful, §8.4 on what gets tokenized, since the vision encoder here becomes a token source in §11.2.
key_refs:
  - Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021.
  - Jia, C. et al. (2021). Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (ALIGN). ICML 2021.
  - Driess, D. et al. (2023). PaLM-E: An Embodied Multimodal Language Model. arXiv:2303.03378.
  - Brohan, A. et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
---

# 11.1  CLIP and the multimodal pretraining moment

The four chapters ahead of you are about the same trick applied at
larger and larger scale, so it is worth naming the trick before we get
lost in model names. A vision-language-action model is a policy that
takes a camera image and a sentence and emits an action, and it works
because someone else already spent millions of dollars teaching a
network that a picture of a mug and the word "mug" belong together. The
policy inherits that alignment for free and only has to learn the last
step, from aligned perception to motor commands. Every model in Part 4
is a variation on where that inherited alignment comes from and how the
action step is bolted on.

This section is about where the alignment came from. In January 2021 two
groups, one at OpenAI and one at Google, published the same idea within
weeks of each other: stop labeling images and let the caption be the
label. The OpenAI version is called CLIP (Radford et al., 2021), the
Google version ALIGN (Jia et al., 2021), and CLIP is the one the
robotics field ended up building on, so it gets top billing here. The
reason it mattered for us has almost nothing to do with the task CLIP was
built for and everything to do with a side effect: it produced an image
encoder whose features already speak the same language as text. Chapter
6 taught you that imitation is the dominant training signal in robotics;
this chapter is the story of what happens when you point that signal at a
perception stack that was pretrained to understand language. But first,
the perception stack.

## The problem CLIP was actually solving

Before 2021, the standard way to get a good image encoder was to train it
on ImageNet: a million photographs, each stamped with exactly one of a
thousand class labels. The resulting network was excellent at the
thousand things it had seen and useless at the thousand-and-first. If
your robot needed to recognize a "spatula" and spatula was not one of the
thousand classes, you collected a spatula dataset and fine-tuned. The
label set was a fence, and everything outside it was invisible.

The fence was not a modeling limitation; it was a data-collection
decision. Someone had to sit down and decide the thousand categories,
then pay annotators to sort images into them. That does not scale, and
worse, it throws away most of what an image's caption already tells you.
A web photo captioned "our golden retriever puppy chewing a blue slipper
on the porch" contains a dozen concepts and their relationships, and the
ImageNet pipeline reduces all of it to the single token "dog," if that.

CLIP's move was to treat the caption as the supervision and skip the
fixed label set entirely. Radford and colleagues scraped roughly 400
million image-text pairs off the public web — no human relabeling, just
whatever caption happened to sit next to each image — and trained on the
raw pairs. ALIGN pushed the same idea to over a billion pairs and made a
virtue of the noise, showing that at that scale you do not even need to
clean the captions. The lesson both papers drove home is one this book
returns to constantly: a mediocre signal at enormous scale beats a clean
signal at small scale, and the web is the only place the enormous scale
lives.

## Contrastive learning in one matrix

Here is the mechanism, because it is simpler than its reputation. You
have two encoders. One is an image encoder $f$ (a vision transformer or a
ResNet) that turns a photo into a vector; the other is a text encoder $g$
(a transformer) that turns a caption into a vector. Both project into the
same $d$-dimensional space, and both outputs are normalized to unit
length so that a dot product between them is just a cosine similarity.

Take a batch of $N$ image-caption pairs that actually go together. Encode
all $N$ images and all $N$ captions, and form the $N \times N$ matrix of
similarities $S_{ij} = f(\text{image}_i) \cdot g(\text{caption}_j)$. The
$N$ entries on the diagonal are the real pairs; the $N^2 - N$ off-diagonal
entries are mismatches — image $i$ against some other image's caption.
The training objective is a sentence long: make the diagonal large and
everything else small. Concretely, treat each row as a classification
problem over $N$ options where the correct answer is the diagonal entry,
apply the cross-entropy loss you already know from §3.2, and do the same
down each column. That is it. No new loss function, no adversary, no
reconstruction term. It is the InfoNCE contrastive loss, and in PyTorch
the core is about six lines:

```python
# image_emb, text_emb: (N, d), each row L2-normalized
logits = image_emb @ text_emb.T          # (N, N) similarity matrix
logits = logits * temperature.exp()      # learned scalar sharpens the softmax
labels = torch.arange(N, device=logits.device)   # correct match is the diagonal
loss_i = F.cross_entropy(logits, labels)          # each image picks its caption
loss_t = F.cross_entropy(logits.T, labels)        # each caption picks its image
loss = (loss_i + loss_t) / 2
```

The batch supplies its own negatives, which is the quiet efficiency of
the scheme: with $N = 32{,}768$, every pair is contrasted against tens of
thousands of wrong answers at once, no negative mining required. Scale
the batch up and each gradient step gets sharper on its own.

## The side effect that mattered: zero-shot

CLIP was pitched as a zero-shot image classifier, and the demo is worth
walking through because it shows what the shared space buys you. Suppose
you want to classify a photo as cat, dog, or spatula, and you never
trained on any of those labels. Write each label into a sentence — "a
photo of a cat," "a photo of a dog," "a photo of a spatula" — and run all
three through the text encoder. Run the photo through the image encoder.
Whichever caption vector sits closest to the image vector is your
prediction. The class list is now just text you type at inference time,
so adding "spatula" costs you a sentence instead of a dataset. The fence
is gone.

That is a neat classification result, but the deeper fact is the shape of
the space it lives in. CLIP did not learn "cat" and "dog" as opaque
category IDs. It learned a geometry in which visual features and word
meanings are laid out together, so that directions in image space
correspond to directions in language space. A model that can find the
mug in a picture because you asked for "mug" is one small step from a
model that can pick up the mug because you asked it to — the perception
half of that sentence is already solved, and solved in a way that
generalizes to objects and phrasings nobody enumerated in advance.

## Why a robotics book cares

Return to the trick from the top of the section. A policy trained by pure
imitation (Chapter 6) learns to map pixels to actions, and its
understanding of the world is bounded by the demonstrations it saw. Show
it a thousand demos of stacking red blocks and it stacks red blocks; ask
for the green one and you are outside the fence again. The demonstrations
are expensive — a human teleoperates every one — so the fence is low and
you cannot buy your way past it the way CLIP bought its way past
ImageNet. Robot data does not exist on the web at 400-million-pair scale.

Language-conditioned pretraining is the escape. If your policy's vision
front-end is a CLIP-style encoder, it arrives already knowing that mugs,
cups, and "the thing you drink from" point in similar directions, and
that knowledge was paid for with web images, not teleop hours. The
policy's own scarce demonstrations then only need to teach the mapping
from *aligned, language-aware features* to motor commands, which is a far
smaller thing to learn than perception-plus-control from scratch. This is
exactly the claim in the chapter's third learning objective — that
language conditioning unlocked generalization imitation alone could not
reach — and §11.2 makes it concrete with BC-Z and RT-1, the first
policies to cash the check.

The bridge is not hypothetical. PaLM-E (Driess et al., 2023,
arXiv:2303.03378) took the idea to its logical end by injecting robot
sensor data directly into a large language model's token stream, so that
"multimodal sentences" interleaving words, images, and states could be
reasoned over by one network; it is the clearest early demonstration that
a language-pretrained backbone transfers to embodied tasks. RT-1 (Brohan
et al., 2022, arXiv:2212.06817), which anchors the rest of this chapter,
made the more modest and more copied choice: use a pretrained language
embedding to condition a comparatively small transformer policy, and
spend the saved capacity on real-robot data. We will pull RT-1 apart in
§11.2 and §11.4.

## What CLIP did not solve

Two cautions, so you do not overclaim. First, alignment is not grounding
in action. CLIP knows that an image of an open drawer and the words "open
drawer" are near each other; it has no idea how to move a gripper to
produce that outcome, and nothing in its training touched a robot. The
action half of vision-language-action is new work that Chapters
11 through 14 are about, and it is the hard half. Second, CLIP's
world is a single still image and a short caption. It has no notion of
time, force, or the difference between a mug that is empty and one that is
full to the brim. A control policy lives and dies on exactly those
distinctions, and inheriting a static-image encoder means inheriting its
blind spots along with its strengths. Keep both limits in mind; several
design decisions in the models ahead are attempts to work around them.

With the perception stack pretrained and its features aligned to
language, the recipe has its first ingredient. The next section adds the
second — a way to condition a policy on an instruction and turn those
aligned features into motor commands — and gives it a name: RT-1.
