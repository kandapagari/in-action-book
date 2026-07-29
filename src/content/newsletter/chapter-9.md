---
chapter: 9
subject: "World models: learning a simulator from experience"
---
A policy predicts the next action and never stops to ask what that action will do. A world model answers precisely the question the policy skips: given where things stand and an action you're considering, what happens next? Chapter 9 defines it as a learned simulator, a parametric stand-in for the environment's transition and reward functions, fit by watching transitions roll past.

Why bother? Because it lets you try an action without paying for it. In the real world a policy commits, and mistakes cost broken grippers and reset time; inside a world model it can commit in imagination, run the consequences forward, and only then move. The chapter sorts the uses of that imagined experience into planning, training a policy entirely inside the model in the spirit of Dyna, and treating prediction as a goal of its own, then walks through Ha and Schmidhuber's result where a controller learned to drive inside its own hallucinated rollouts. Two lessons recur afterward: predict in a compressed latent space rather than raw pixels, and give the model memory so its predictions stay Markov. From here the book heads into RSSM, Dreamer, video predictors like Genie and V-JEPA, and the sharpening argument between world models and VLAs. Continue on the site.
