---
chapter: 3
section: 3.1
title: Vectors, matrices, gradients, and why the chain rule rules robotics
target_words: 2000
status: draft
prereqs: Chapter 1 (vocabulary of action models), Chapter 2 (a running OpenVLA loop); high-school calculus and a passing acquaintance with linear algebra
key_refs:
  - Kim et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246.
  - Brohan et al. (2022). RT-1: Robotics Transformer for Real-World Control at Scale. arXiv:2212.06817.
  - Black et al. (2024). π0: A Vision-Language-Action Flow Model for General Robot Control. arXiv:2410.24164.
---

# 3.1  Vectors, matrices, gradients, and why the chain rule rules robotics

Chapter 2 ended with a 7-vector arriving at a simulator every 50 ms. That 7-vector is where the entire stack, a 7-billion-parameter transformer, a vision encoder pretrained on internet images, a tokenizer trained on a hundred thousand teleoperation episodes, finally touches the world. Everything upstream of it is linear algebra and the chain rule, and this chapter's job is to make sure that sentence doesn't feel like a slogan. We won't re-teach linear algebra from scratch; Appendix A is there for a full refresher. What this section does instead is pick out four objects you'll see on almost every page of the rest of the book, vectors, matrices, gradients, and the chain rule, and show them doing the specific work they do inside a robot policy.

If you can already write the Jacobian of a 2-link arm without looking it up, skim this section and move to §3.2. If the phrase "gradient of the loss with respect to the weights" still feels like an incantation, sit with this one.

## Vectors and matrices: the right datatype for a robot

A robot's instantaneous state is a vector. Not metaphorically, literally a list of numbers your program holds in memory. For a 7-DoF Franka arm, the joint configuration $q \in \mathbb{R}^{7}$ stacks the seven joint angles. The joint velocity $\dot q \in \mathbb{R}^{7}$ is another 7-vector. The end-effector pose, depending on representation, is a 6-vector (3 for position, 3 for axis-angle orientation) or a 7-vector (position plus a unit quaternion). The action OpenVLA emitted in Chapter 2 was itself a 7-vector: three for end-effector translation deltas, three for rotation deltas, one for the gripper. Calling these "states" or "poses" or "actions" hides their common type signature. They're all points in $\mathbb{R}^{n}$, with $n$ small.

Matrices show up the moment one vector has to become another. Forward kinematics is a function $f : \mathbb{R}^{7} \to \mathbb{R}^{6}$ taking joint angles to end-effector pose. Differentiate it and you get the manipulator Jacobian $J(q) \in \mathbb{R}^{6 \times 7}$, the matrix mapping joint velocities to end-effector velocities:

$$
\dot x = J(q)\, \dot q.
$$

Inverse-kinematics solvers, impedance controllers, and damped-least-squares trackers all live inside that one equation. If you've ever wondered why classical roboticists obsess over the singular values of $J(q)$, the answer is that the matrix sometimes goes near-singular, and inverting it then produces wildly large joint commands. We'll return to Jacobians in Chapter 4, where they're the load-bearing object in classical control, and again in Chapter 13, where π0's flow-matching head produces continuous actions that an outer loop still has to map through a Jacobian to reach the joints (Black et al., 2024, arXiv:2410.24164).

Images are matrices too, or more precisely third-order tensors of shape $H \times W \times 3$. The OpenVLA visual encoder from §2.3 took a $224 \times 224 \times 3$ tensor, ran it through a stack of convolutions and attention layers, and emitted a sequence of 256 512-dimensional vectors. Every "embedding," "feature map," and "representation" you'll read about for the next fifteen chapters is some tensor, and every learned layer is a function turning one tensor into another. Only three operations account for the bulk of the work: matrix-vector multiplication, elementwise nonlinearity, and reduction (sum, mean, max). A transformer block is a particular arrangement of those three. A convolution is a structured matrix multiplication with weight sharing. A normalization layer is a reduction followed by an elementwise rescale. Once you see the underlying datatype, architecture diagrams stop looking like a set of incompatible boxes.

One pragmatic warning: robotics conventions are a minefield. The same 6-vector can mean position-then-rotation or rotation-then-position. The same rotation can be a quaternion (with $w$ first, last, or absent), an axis-angle vector, or three Euler angles in any of twelve orderings. RT-1 (Brohan et al., 2022, arXiv:2212.06817), OpenVLA (Kim et al., 2024, arXiv:2406.09246), and π0 each pick a convention, document it somewhere in the codebase, and silently break if you feed them the wrong layout. When a learned policy outputs garbage, "the rotation convention is flipped" sits in the top three culprits more often than it has any right to. Read the data-loader, not the paper.

## Gradients are linear approximations

A function $L : \mathbb{R}^{n} \to \mathbb{R}$, say the loss of a policy network as a function of its weights, has a gradient $\nabla L \in \mathbb{R}^{n}$. The gradient isn't "the slope" in any single direction. It's the vector that, dotted with any small step $\Delta w$, predicts the change in $L$:

$$
L(w + \Delta w) \approx L(w) + \nabla L(w)^{\top} \Delta w.
$$

That's the whole idea. Every line about "moving downhill," every diagram with a loss landscape, every optimizer trick is a corollary of that one linear approximation. The gradient is the best linear approximation in the sense that it's the only vector for which the equation above holds correct to first order, and it points toward steepest ascent because a dot product with a unit vector is largest when the two are aligned. Gradient descent, taking $w \leftarrow w - \eta\, \nabla L(w)$, is the obvious move once you have it.

The catch is that policies carry a lot of weights. OpenVLA-7B has, approximately, seven billion of them. The gradient is a 7-billion-entry vector, the same shape as the parameter vector itself. You won't compute it by naively perturbing each weight in turn; that would take roughly seven billion forward passes per training step. Deep learning is practical at all because there's an algorithm that computes the full gradient in about the time of two forward passes. That algorithm is backpropagation, and backpropagation is just the chain rule applied carefully.

A worked toy example helps here. Suppose the policy is a single linear layer followed by a mean-squared-error loss: $L(W) = \tfrac{1}{2} \| W x - a^{\star} \|^{2}$, where $x \in \mathbb{R}^{d}$ is an observation, $a^{\star} \in \mathbb{R}^{k}$ is the target action, and $W \in \mathbb{R}^{k \times d}$ is the weight. Differentiating gives $\nabla_{W} L = (W x - a^{\star}) x^{\top}$, a $k \times d$ matrix, exactly the shape of $W$, which is the first sanity check worth running on every gradient derivation you write. The outer-product structure is informative too: the gradient with respect to row $i$ of $W$ is the $i$-th component of the prediction error scaled by the input $x$. Wrong dimension of the output? Adjust the row that produced it, in proportion to the input that fed it. That intuition holds for a linear regressor, and it's the same template that keeps working for transformer attention heads. They just stack many such updates on top of each other.

## The chain rule, and why it rules robotics

Now stack the layers. Let $a = g(h)$ and $h = f(x; W)$, with the same $L$ as before. To update $W$, you need $\partial L / \partial W$. The chain rule says:

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot
\frac{\partial a}{\partial h} \cdot \frac{\partial h}{\partial W}.
$$

Each factor is a Jacobian. The product is, in general, a tensor, but the practical magic is that you never materialize the full Jacobians. You materialize the vector-Jacobian product at each layer, walking backwards from $L$ to the inputs. That's what `loss.backward()` does under the hood in PyTorch, and it's what keeps the asymptotic cost of a gradient at about twice the cost of a forward pass instead of seven billion times that cost. We'll write that loop out by hand in §3.3.

The phrase "chain rule rules robotics" isn't a pun. It's a claim about which equation does the most work in modern robot learning. Pick any modern action model and trace the loss back to the parameters it updates. In RT-1 (arXiv:2212.06817), the discrete-action cross-entropy loss flows back through a transformer trunk, a token embedding, a FiLM-conditioned CNN, and into the EfficientNet visual encoder; every arrow in that data-flow diagram is one factor of one chain-rule product. In π0, the flow-matching loss on a continuous 7-vector flows back through an action-expert transformer, a language-conditioned VLM trunk, and a tokenized image encoder. Same structure, different blocks. Even classical methods we'll meet in Chapter 4, iterative inverse-kinematics solvers, computed-torque controllers, trajectory optimizers, are walking the chain rule through the kinematic and dynamic equations of the robot. The objects differ (an explicit Jacobian instead of an autodiff graph), but the operation stays the same: compose local linearizations and read off the global one.

The chain rule has one further consequence worth flagging now, since it comes back to haunt later chapters. If any factor in the product is very small, the whole product shrinks toward zero, which is the vanishing gradient problem, why ReLU replaced sigmoid as the default activation, and why transformer trunks add residual connections letting the gradient bypass deep stacks. If any factor is very large, the whole product blows up, the exploding gradient problem, why gradient clipping exists, and why training a 7B-parameter VLA without mixed-precision care often produces NaNs by step 200. The fixes are mostly engineering. The underlying diagnosis is always the chain rule multiplying things it shouldn't.

You now have the vocabulary: a vector is a state or an action, a matrix maps one to another, a gradient is the linear approximation that lets optimization make progress, and the chain rule composes those linear approximations across the entire stack from pixels to motor torques. That's the substrate. The next section adds the second pillar, randomness, that turns this substrate into a probabilistic model of a robot interacting with an uncertain world.
