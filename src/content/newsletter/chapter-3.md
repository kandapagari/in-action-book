---
chapter: 3
subject: "The math that actually runs a robot policy, in 30 minutes"
---
Every action model touches the world through a small vector of real numbers, the seven-vector OpenVLA hands the simulator every fifty milliseconds. Everything upstream of that vector is linear algebra and the chain rule. Chapter 3 exists to make that sentence stop sounding like a slogan.

Instead of re-teaching mathematics, the chapter picks four objects you'll meet on nearly every later page and watches each do its specific job inside a policy. A robot's state is literally a vector. Forward kinematics is a function whose derivative is the manipulator Jacobian; a gradient is the best linear approximation that lets an optimizer make progress; backpropagation is just the chain rule applied carefully from pixels down to motor torques. That's enough working fluency to read the rest of the book without translating in your head. The chapter also plants two warnings that come due later: rotation conventions are a quiet minefield that breaks learned policies without a word, and the same chain rule that makes training possible is what makes gradients vanish or blow up. This is the substrate every architecture ahead sits on. Pick it up on the site before the networks arrive.
