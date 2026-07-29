---
chapter: 10
subject: "Diffusion and flow models for generating action"
---
Reach for a mug behind a laptop and you can go left or right; both work fine, and the average of the two drives your hand into the laptop. A policy trained to regress the mean of its demonstrations learns that average, and with it the collision. Chapter 10 is about the fix the field settled on: action heads that model a whole multi-peaked distribution over what to do next instead of flattening it to a single point.

The chapter gives diffusion in its ten-minute form. Wreck the structure of the data with a fixed noise schedule, train a network to undo the damage one step at a time, and sampling from an ugly, intractable distribution becomes plain supervised regression. The same recipe holds whether the target is an image, one action, or a chunk of consecutive actions stacked together, which is how Diffusion Policy and ACT put out motion that's smooth and temporally coherent. None of it is free. A diffusion head swaps one cheap forward pass for many, and a control loop has no patience for that. That standoff between expressive multimodality and hard latency is the axis the chapter turns on, and it leads straight into flow matching and the action-head decisions inside modern VLAs. Read it on the site.
