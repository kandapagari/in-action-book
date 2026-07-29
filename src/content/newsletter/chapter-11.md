---
chapter: 11
subject: "How CLIP handed every VLA its perception for free"
---
A vision-language-action model works largely because someone else already burned millions of dollars teaching a network that a photo of a mug and the word "mug" belong together. The policy inherits that alignment for nothing and only has to learn the final step, from aligned perception to motor commands. Chapter 11 traces where the inherited alignment came from and how the action step gets bolted onto it.

It starts with CLIP, which in 2021 stopped labeling images and let the caption be the label, training on something like four hundred million image-text pairs scraped off the web. The chapter boils contrastive learning down to a single matrix and six lines of code, then pulls out the side effect that mattered for robots: an image encoder whose features already speak the language of text, arranged so that directions in image space line up with directions in language space. A model that can find the mug because you said "mug" is a short step from one that picks it up because you asked. You'll also see plainly what CLIP left undone, alignment isn't grounding in action, and a static-image encoder carries its blind spots along with its gifts, before the chapter cashes the check with BC-Z and RT-1. The full story is on the site.
