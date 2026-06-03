---
appendix: A
title: "Linear algebra refresher"
target_words: 3600
status: draft
prereqs: high-school algebra; one semester of calculus; the willingness to look at a matrix without flinching
key_refs:
  - Strang, G. (2016). Introduction to Linear Algebra (5th ed.). Wellesley-Cambridge Press.
  - Trefethen, L. N., & Bau, D. (1997). Numerical Linear Algebra. SIAM.
  - Murray, R. M., Li, Z., & Sastry, S. S. (1994). A Mathematical Introduction to Robotic Manipulation. CRC Press.
---

# Appendix A.  Linear algebra refresher

The body of this book treats linear algebra as a working tool, not a
subject. Vectors, matrices, eigenvalues, and singular value
decompositions all appear in chapter examples without ceremony, on the
assumption that the reader has seen them before and needs only the
fingertip-level fluency to keep up. This appendix is the safety net.
Every object discussed here is illustrated with a robotics example
from the body of the book — a manipulator Jacobian, a covariance
matrix, an action-token codebook — so that the math arrives attached to
a thing it does, rather than as a list of definitions. The treatment is
intentionally short. If you find yourself wanting more, Strang (2016)
is the canonical undergraduate text and Trefethen and Bau (1997) is the
numerical-linear-algebra reference the rest of the field reaches for.

## A.1  Vectors and inner products

A vector is an ordered tuple of real numbers. The notation
$x \in \mathbb{R}^{n}$ says $x$ has $n$ components, indexed
$x_1, x_2, \ldots, x_n$. In robotics, the most frequent vectors you
will write down are joint configurations $q \in \mathbb{R}^{7}$ for a
Franka-class arm, end-effector poses $p \in \mathbb{R}^{6}$ when
orientation is expressed as an axis-angle vector, and the seven-vector
of end-effector deltas plus gripper command that OpenVLA
(arXiv:2406.09246) emits at 5 Hz.

Two vectors $x, y \in \mathbb{R}^{n}$ have an *inner product*
$\langle x, y \rangle = x^\top y = \sum_{i} x_i y_i$. The inner product
is a scalar; geometrically it equals $\lVert x \rVert\, \lVert y
\rVert \cos\theta$, where $\theta$ is the angle between $x$ and $y$
and $\lVert x \rVert = \sqrt{x^\top x}$ is the Euclidean norm. Three
properties of inner products do load-bearing work in the body of the
book. Bilinearity — linear in each argument when the other is fixed —
is what makes a transformer's attention pattern $\mathrm{softmax}(QK^\top
/ \sqrt{d})$ computable in matrix form. Symmetry, $\langle x, y \rangle
= \langle y, x \rangle$, is why a Gram matrix $X^\top X$ is symmetric.
Positive-definiteness, $\langle x, x \rangle \geq 0$ with equality
only at $x = 0$, is what lets the loss function in §3.3 be bounded
below by zero.

Three derived quantities you will see often. The *Euclidean distance*
$\lVert x - y \rVert$ measures how far apart two points are in
configuration space. The *cosine similarity* $\langle x, y \rangle /
(\lVert x \rVert\, \lVert y \rVert)$ is the inner product after
length normalization; CLIP-style contrastive losses (Radford et al.,
2021, ICML 2021) compute cosine similarity between image and text
embeddings and that is the entire learning signal. The *projection*
of $x$ onto a unit vector $u$ is $(u^\top x)\, u$; projection onto the
column space of a matrix is the operation a least-squares solver
performs under the hood.

## A.2  Matrices as linear maps

A matrix $A \in \mathbb{R}^{m \times n}$ is two objects at once. As
data, it is an $m$-by-$n$ array of numbers, indexed $A_{ij}$ for
$i \in \{1, \ldots, m\}$ and $j \in \{1, \ldots, n\}$. As a function,
it is a *linear map* from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$: every
vector $x \in \mathbb{R}^{n}$ has an image $Ax \in \mathbb{R}^{m}$,
and the map satisfies $A(\alpha x + \beta y) = \alpha\, Ax + \beta\,
Ay$ for any scalars. The second view is the one that matters; once you
read $A$ as a function, every operation the rest of the book performs
on it becomes intelligible.

Three matrix examples that recur. The *manipulator Jacobian* $J(q)
\in \mathbb{R}^{6 \times 7}$ of §3.1 maps joint velocities $\dot q \in
\mathbb{R}^{7}$ to end-effector velocities $\dot p = J(q)\, \dot q \in
\mathbb{R}^{6}$. The *covariance matrix* $\Sigma \in \mathbb{R}^{n
\times n}$ of a Gaussian random vector encodes the pairwise covariances
of its components; it is symmetric and positive semi-definite, and the
diagonal entries are the per-component variances. The *weight matrix*
$W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ of a fully
connected layer maps a $d_{\text{in}}$-dimensional input embedding to
a $d_{\text{out}}$-dimensional output embedding; the entire weight
budget of OpenVLA — about seven billion numbers — is a stack of such
matrices.

Matrix-matrix multiplication $C = AB$, where $A \in \mathbb{R}^{m
\times k}$ and $B \in \mathbb{R}^{k \times n}$, produces $C \in
\mathbb{R}^{m \times n}$ with entries $C_{ij} = \sum_{\ell} A_{i\ell}\,
B_{\ell j}$. As linear maps, $AB$ is the *composition* of $B$ followed
by $A$: $(AB)x = A(Bx)$. Composition is associative but not
commutative, which is why the order of matrix factors in a transformer
attention computation matters and why most GPU training time on a VLA
is spent inside one of a small number of large matrix multiplies.

The *transpose* $A^\top \in \mathbb{R}^{n \times m}$ is the matrix with
swapped indices, $(A^\top)_{ij} = A_{ji}$. The transpose appears
ubiquitously because $\langle Ax, y \rangle = \langle x, A^\top y
\rangle$ — moving a linear map between the two arguments of an inner
product just transposes it — and that identity is the backbone of
backpropagation, where the gradient of a loss with respect to an input
of a layer is computed by multiplying the upstream gradient by the
layer's transposed weight.

## A.3  Rank, null space, and the four fundamental subspaces

The *rank* of a matrix $A \in \mathbb{R}^{m \times n}$ is the dimension
of its column space — the subspace of $\mathbb{R}^{m}$ spanned by its
columns. Rank is at most $\min(m, n)$. When rank equals
$\min(m, n)$ the matrix is *full rank*; otherwise it is *rank-deficient*
and behaves degenerately under inversion.

The *null space* of $A$, written $\ker(A)$ or $\mathrm{null}(A)$, is
$\{x \in \mathbb{R}^{n} : Ax = 0\}$. Its dimension is $n - \mathrm{rank}(A)$.
For the manipulator Jacobian of a redundant arm (7-DOF arm, 6-DOF task
space, so $J \in \mathbb{R}^{6 \times 7}$), the null space is a
one-dimensional subspace of joint-velocity vectors that produce zero
end-effector velocity — that is, the directions the joints can move
without the gripper moving. Null-space control exploits this freedom
to keep the elbow up while the gripper does work, and is one of the
reasons 7-DOF arms are useful when 6-DOF ones would also reach the
target.

The four fundamental subspaces — column space, null space, row space,
left null space — are connected by the *fundamental theorem of linear
algebra*: column space and left null space are orthogonal complements
in $\mathbb{R}^{m}$; row space and null space are orthogonal
complements in $\mathbb{R}^{n}$. The practical implication for the book
is that a least-squares solution always exists and is unique up to a
shift in the null space, and that the Moore-Penrose pseudoinverse —
used by every numerical inverse-kinematics solver in §4.2 — picks the
shift that minimizes the norm of $x$.

## A.4  Eigenvalues and eigenvectors

An *eigenvector* of a square matrix $A \in \mathbb{R}^{n \times n}$ is
a nonzero vector $v$ such that $Av = \lambda v$ for some scalar
$\lambda$, the corresponding *eigenvalue*. Geometrically, $v$ is a
direction along which $A$ acts as pure scaling. A symmetric matrix has
$n$ orthogonal real eigenvectors and $n$ real eigenvalues; the
*spectral theorem* writes $A = V \Lambda V^\top$ where $V$ stacks the
eigenvectors in columns and $\Lambda$ is the diagonal matrix of
eigenvalues.

Eigenvalues do interpretable work in three places in the book. In
control, the eigenvalues of the closed-loop dynamics matrix tell you
whether a controller is stable (all eigenvalues in the open left half
of the complex plane, for continuous-time systems). In probability,
the eigenvalues of a covariance matrix are the variances along the
principal axes of the Gaussian; this is the math under principal
component analysis. In optimization, the eigenvalues of the Hessian of
a loss function are the local curvatures; a learning-rate schedule
that works well on a flat eigenvalue spectrum will diverge on a steep
one, which is one motivation for adaptive optimizers like Adam.

A diagonalizable matrix $A = V \Lambda V^{-1}$ is essentially trivial
to power: $A^{k} = V \Lambda^{k} V^{-1}$, where $\Lambda^{k}$ is the
diagonal matrix with $\lambda_i^{k}$ on its diagonal. This is the
trick value iteration in §5.2 implicitly uses: the Bellman operator is
contractive with rate $\gamma$, which is its spectral radius, and the
iteration converges geometrically at exactly that rate.

## A.5  Singular value decomposition

The singular value decomposition (SVD) is the workhorse of numerical
linear algebra. Every matrix $A \in \mathbb{R}^{m \times n}$ — not just
symmetric, not just square — admits a factorization $A = U \Sigma
V^\top$ where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n
\times n}$ are orthogonal (columns orthonormal) and $\Sigma \in
\mathbb{R}^{m \times n}$ is rectangular-diagonal with non-negative
entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m, n)}
\geq 0$. The $\sigma_i$ are the *singular values* of $A$.

Three uses of SVD recur. First, the *pseudoinverse*: $A^{+} = V
\Sigma^{+} U^\top$, where $\Sigma^{+}$ inverts the nonzero singular
values and zeros the rest. The damped least squares IK solver of §4.2
is structurally a pseudoinverse with the small singular values
truncated, which is what prevents joint velocities from blowing up
near a Jacobian singularity. Second, *low-rank approximation*: the
best rank-$k$ approximation of $A$ in the Frobenius norm is obtained
by zeroing all but the top $k$ singular values; this is the math
under LoRA (Hu et al., 2022, ICLR 2022) and every adapter-based
fine-tuning technique in Chapter 16. Third, *conditioning*: the ratio
$\sigma_1 / \sigma_{\min(m, n)}$ is the condition number of $A$, and a
large condition number means the matrix is close to singular and
inverting it amplifies noise.

## A.6  Quadratic forms and positive-definiteness

A *quadratic form* is a function $f(x) = x^\top A x$ for some
symmetric matrix $A$. The form is *positive definite* if
$x^\top A x > 0$ for all nonzero $x$, *positive semi-definite* if
$x^\top A x \geq 0$, and *negative definite* if the inequality
reverses. The Hessian of the loss function in §3.5 is symmetric and
locally positive semi-definite near a minimum; the inertia matrix
$M(q)$ in §4.3's manipulator equation is positive definite for every
physically realizable configuration, which is exactly why the
computed-torque controller from that section is well-posed.

Positive-definite matrices have all-positive eigenvalues, are
invertible, and admit a *Cholesky factorization* $A = LL^\top$ with
$L$ lower triangular. Cholesky is what every Gaussian sampler in the
book uses internally: to sample $x \sim \mathcal{N}(0, A)$, draw
$z \sim \mathcal{N}(0, I)$ and return $Lz$.

## A.7  Rotation matrices, $\mathrm{SO}(3)$, and the rest of the rigid-body story

Robotics adds a piece of structured linear algebra that pure ML does
not: rotations. A *rotation matrix* in three dimensions is a matrix
$R \in \mathbb{R}^{3 \times 3}$ with $R^\top R = I$ and $\det R = +1$.
The set of all such matrices is the *special orthogonal group*
$\mathrm{SO}(3)$; it is a three-dimensional manifold inside the
nine-dimensional space of $3 \times 3$ matrices.

The body of the book treats $\mathrm{SO}(3)$ mostly through proxy
representations — axis-angle vectors, quaternions, Euler angles —
because that is what neural-network policies emit. Three things to
remember about all of them. They are not unique: every rotation has
two quaternion representations $\pm q$, and twelve Euler-angle
conventions exist. Linear interpolation between rotation
representations is not rotation interpolation: averaging two
quaternions component-wise and renormalizing is not the same as the
geodesic between them on $\mathrm{SO}(3)$, and the difference shows
up as wobble in §10.2's Diffusion Policy outputs if the model
parameterizes orientation wrong. The Jacobian of a rotation
representation with respect to its parameters is non-trivial; the
right object for differentiating through rotations is the *exponential
map* from the Lie algebra $\mathfrak{so}(3)$ (skew-symmetric matrices)
to $\mathrm{SO}(3)$. Murray, Li, and Sastry (1994) is the canonical
reference for the Lie-group story; the body of the book uses the
exponential map without proof.

## A.8  Numerical linear algebra in two paragraphs

Almost every matrix operation in the body of the book is computed
numerically by a library — NumPy, PyTorch, JAX — that uses BLAS
underneath. The relevant practical knowledge is small. Floating-point
matrix multiplication is associative in mathematics but not in code:
$(AB)C$ and $A(BC)$ can differ at the last few bits, and a long chain
of matrix multiplications can accumulate small errors that compound.
This is why training in mixed precision (fp16 or bf16) sometimes
diverges where fp32 does not, and why VLA training recipes are picky
about which layers are kept in fp32.

Matrix inversion is almost always the wrong operation to call.
Instead of computing $A^{-1} b$, solve the linear system $Ax = b$
directly with an LU or QR factorization; the result is the same and
the conditioning is better. Pseudoinverses for non-square matrices
are computed via SVD, not by $A^\top (A A^\top)^{-1}$ or analogues;
the latter formula is correct on paper and numerically catastrophic
when $A$ is poorly conditioned, which is the most common case in
inverse-kinematics work near a singularity.

## A.9  A quick reference of identities

A handful of identities are used so often in the body of the book that
they are worth memorizing.

The chain rule for matrix-valued functions: if $f : \mathbb{R}^{n} \to
\mathbb{R}^{m}$ and $g : \mathbb{R}^{m} \to \mathbb{R}^{p}$, then the
Jacobian of the composition $g \circ f$ at $x$ is $J_{g \circ f}(x) =
J_g(f(x))\, J_f(x)$. This is the matrix form of the chain rule and the
mathematical backbone of backpropagation.

The matrix-multiplication identity for transposes: $(AB)^\top = B^\top
A^\top$. Used implicitly every time gradients flow backward through a
linear layer.

The Woodbury identity for low-rank updates of an inverse:
$(A + UV^\top)^{-1} = A^{-1} - A^{-1} U (I + V^\top A^{-1} U)^{-1}
V^\top A^{-1}$. This is the identity under LoRA's training-time
inference path, and the identity Kalman filters use to update a
covariance matrix from a low-dimensional measurement.

The trace cyclicity property: $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) =
\mathrm{tr}(CAB)$. This is used in derivations of every loss whose
gradient involves a trace, and it is the identity behind the
"reparameterization trick" in §3.4's discussion of VAEs.

The Frobenius inner product: $\langle A, B \rangle_F = \mathrm{tr}(A^\top
B) = \sum_{ij} A_{ij} B_{ij}$. This is the inner product that makes
the space of matrices into a vector space; the Frobenius norm
$\lVert A \rVert_F = \sqrt{\langle A, A \rangle_F}$ is what
"distance between weight matrices" almost always means.

Five identities, four sections of background, one quick-reference list
— the rest is in the body of the book. If you forget what
$A^\top A$ is, look here; if you forget why the singular values matter
to a controller, look at §4.2 and then come back.
