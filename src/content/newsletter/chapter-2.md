---
chapter: 2
subject: "Run your first vision-language-action model, end to end"
---
Chapter 2 is the one where you run something. By the end you'll have OpenVLA-7B live on your own machine, taking an RGB frame of a simulated tabletop plus an instruction like "put the red block in the bowl" and emitting the low-level commands that carry it out in the LIBERO simulator. Then you'll watch it fail, again and again, in three characteristic ways the rest of the book exists to explain.

The ordering is on purpose: get it working, understand it later. The chapter follows the whole path from a 256×256 image to a seven-number action and shows there's no magic in the gap, just a transformer doing next-token prediction over a vocabulary whose bottom entries happen to mean discretized action bins. Sitting inside that path are four quiet commitments every VLA makes: how pixels turn into tokens, how an instruction turns into motion, which dataset taught the model any physics it knows, and what an honest evaluation actually measures. The muscle memory of standing a model up comes no other way than doing it once, and this first build is the artifact later chapters keep pointing back to. Follow the walkthrough on the site.
