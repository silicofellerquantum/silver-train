"""Section C - Quantum Machine Learning Fundamentals (data encoding, feature maps and quantum kernels, variational
circuits, QNN/VQC/QCNN architectures, hybrid training, parameter-shift gradients, loss functions and the QCNN of
QML.ipynb).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""A two-qubit variational quantum classifier (VQC) encodes the features $(x_0, x_1)$ with $R_Y(x_0)$ on qubit 0 and
$R_Y(x_1)$ on qubit 1, applies a CNOT (control 0, target 1) and then one trainable gate $R_Y(\theta)$ on qubit 1. Its output
is $f(\mathbf x; \theta) = \langle Z_1\rangle$, the prediction is $\operatorname{sign} f$, and the loss of one example is $L = (f - y)^2$.
\begin{center}
\begin{tikzpicture}[x=1cm, y=0.8cm, font=\small]
\draw (0,0) node[left]{$q_0:\ \ket0$} -- (5,0); \draw (0,-1) node[left]{$q_1:\ \ket0$} -- (5,-1);
\node[draw, fill=white] at (1,0) {$R_Y(x_0)$}; \node[draw, fill=white] at (1,-1) {$R_Y(x_1)$};
\fill (2.4,0) circle (2pt); \draw (2.4,0) -- (2.4,-1.18); \draw (2.4,-1) circle (5pt);
\node[draw, fill=brand!15] at (3.6,-1) {$R_Y(\theta)$}; \node[draw, fill=white] at (4.7,-1) {$\langle Z\rangle$};
\end{tikzpicture}
\end{center}""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""\texttt{QML.ipynb} encodes a pixel value $x \in [0, 1]$ with \texttt{ry(np.pi * x, q)} on a qubit in $\ket0$. For a
pixel of brightness $x = \frac13$ the probability of measuring $1$ on that qubit and its $\langle Z\rangle$ (before any other gate) are""",
         opts=[r"$P(1) = \frac14$, $\langle Z\rangle = \frac12$", r"$P(1) = \frac34$, $\langle Z\rangle = -\frac12$",
               r"$P(1) = \frac13$, $\langle Z\rangle = \frac13$", r"$P(1) = \frac12$, $\langle Z\rangle = 0$"],
         sol=r"""$R_Y(\pi x)\ket0 = \cos\frac{\pi x}{2}\ket0 + \sin\frac{\pi x}{2}\ket1$. For $x = \frac13$ the half-angle is $\frac\pi6$:
$P(1) = \sin^2\frac\pi6 = \frac14$, $P(0) = \frac34$, so $\langle Z\rangle = \frac34 - \frac14 = \frac12 = \cos\frac\pi3 = \cos(\pi x)$.
A dark pixel ($x = 0$) stays $\ket0$, a lit pixel ($x = 1$) becomes $\ket1$, intermediate values give superpositions."""),

    dict(id="C2", kind="mcq",
         q=r"""The vector $\mathbf x = (1, 2, 4, 2)$ is amplitude-encoded: $\ket{\psi_{\mathbf x}} = \sum_j \frac{x_j}{\|\mathbf x\|}\ket j$ with
$\ket0 = \ket{00}, \ket1 = \ket{01}, \ket2 = \ket{10}, \ket3 = \ket{11}$ (qubit 0 written first). The number of qubits and the
probability that a measurement of qubit 0 gives 1 are""",
         opts=[r"$2$ qubits; $0.80$", r"$2$ qubits; $0.32$", r"$4$ qubits; $0.80$", r"$2$ qubits; $0.60$"],
         sol=r"""Four amplitudes need $\log_2 4 = 2$ qubits. $\|\mathbf x\| = \sqrt{1 + 4 + 16 + 4} = 5$, so the amplitudes are
$(0.2, 0.4, 0.8, 0.4)$ on $\ket{00}, \ket{01}, \ket{10}, \ket{11}$. Qubit 0 is 1 in $\ket{10}$ and $\ket{11}$:
$P = 0.8^2 + 0.4^2 = 0.64 + 0.16 = 0.80$. (Qubit 1 is 1 in $\ket{01}, \ket{11}$: $0.16 + 0.16 = 0.32$.) Amplitude encoding is
exponentially compact in qubits, but preparing a general such state needs $O(2^n)$ gates."""),

    dict(id="C3", kind="mcq",
         q=r"""A data point has $d = 256$ real features, each stored with 8-bit precision. The numbers of qubits needed by
\emph{basis} encoding (every bit on its own qubit), \emph{angle} encoding (one feature per qubit) and \emph{amplitude} encoding are""",
         opts=[r"$2048$, $256$, $8$", r"$256$, $256$, $8$", r"$2048$, $8$, $256$", r"$8$, $256$, $2048$"],
         sol=r"""Basis encoding writes the bit string into computational basis states: $256 \times 8 = 2048$ bits, one qubit each.
Angle encoding uses one rotation angle per feature: $256$ qubits. Amplitude encoding stores $256 = 2^8$ numbers as the
amplitudes of $8$ qubits. Fewer qubits come at the price of a deeper, more complicated state-preparation circuit."""),

    dict(id="C4", kind="mcq",
         q=r"""With the notebook's angle encoding $\ket{\psi(\mathbf x)} = \bigotimes_i R_Y(\pi x_i)\ket0$, the fidelity (quantum) kernel
$k(\mathbf x, \mathbf x') = |\braket{\psi(\mathbf x)|\psi(\mathbf x')}|^2$ for $\mathbf x = (1, 0.5)$ and $\mathbf x' = (0.5, 0)$ is""",
         opts=[r"$\frac14$", r"$\frac12$", r"$\frac1{16}$", r"$0$"],
         sol=r"""For one qubit $\braket{\psi(x)|\psi(x')} = \cos\frac{\pi x}{2}\cos\frac{\pi x'}{2} + \sin\frac{\pi x}{2}\sin\frac{\pi x'}{2} = \cos\frac{\pi(x - x')}{2}$.
For a product state the overlap is the product: $k = \prod_i\cos^2\frac{\pi(x_i - x'_i)}{2}$.
Here $x_0 - x'_0 = 0.5$ and $x_1 - x'_1 = 0.5$, so $k = \cos^2\frac\pi4\cdot\cos^2\frac\pi4 = \frac12\cdot\frac12 = \frac14$.
(The overlap itself is $\frac12$; the kernel is its square.) Note that this kernel has a simple closed form --- it can be evaluated
classically in $O(d)$ time."""),

    dict(id="C5", kind="mcq",
         q=r"""A quantum-kernel SVM is trained on $M = 20$ points. Every kernel entry $k_{ij} = |\braket{\psi_i|\psi_j}|^2$, with $\ket{\psi_i} = \ket{\psi(\mathbf x_i)}$,
is estimated by its own circuit. Using $k(\mathbf x, \mathbf x) = 1$ and the symmetry $k_{ij} = k_{ji}$, the number of distinct
circuits needed for the training Gram matrix is""",
         opts=[r"$190$", r"$400$", r"$380$", r"$20$"],
         sol=r"""The $20\times20$ Gram matrix has $400$ entries. The $20$ diagonal entries are known ($=1$), leaving $380$
off-diagonal entries, and symmetry halves them: $\binom{20}{2} = \frac{20\cdot19}{2} = 190$ circuits (each run with many shots).
Prediction on a new point needs another $M = 20$ kernel values. The quadratic growth $O(M^2)$ is a practical limitation of kernel methods."""),

    dict(id="C6", kind="mcq",
         q=r"""In a swap test between the states $\ket\psi$ and $\ket\phi$ the ancilla is measured in state $0$ with probability $0.8$.
The fidelity kernel $|\braket{\psi|\phi}|^2$ is""",
         opts=[r"$0.6$", r"$0.8$", r"$0.3$", r"$0.89$"],
         sol=r"""The swap test (Hadamard on the ancilla, controlled-SWAP, Hadamard, measure) gives
$P(0) = \frac12\big(1 + |\braket{\psi|\phi}|^2\big)$. Solving: $|\braket{\psi|\phi}|^2 = 2P(0) - 1 = 2(0.8) - 1 = 0.6$.
Orthogonal states give $P(0) = \frac12$, identical states $P(0) = 1$. (The compute--uncompute test $U(\mathbf x')^\dagger U(\mathbf x)$
instead gives $|\braket{\psi|\phi}|^2$ directly as the probability of the all-zero outcome.)"""),

    dict(id="C7", kind="mcq",
         q=r"""A parameter $\theta$ enters a circuit only through one gate $R_Y(\theta)$. The measured expectation values are
$f(\theta + \frac\pi2) = 0.30$ and $f(\theta - \frac\pi2) = -0.50$. The exact derivative $f'(\theta)$ is""",
         opts=[r"$0.40$", r"$0.80$", r"$-0.40$", r"$0.25$"],
         sol=r"""The two-term parameter-shift rule for a gate $e^{-i\theta P/2}$ ($P$ a Pauli operator) is
$f'(\theta) = \frac12\big[f(\theta + \frac\pi2) - f(\theta - \frac\pi2)\big]$. Here $\frac12(0.30 - (-0.50)) = 0.40$.
The rule is \emph{exact} (no step-size error), because $f(\theta) = a + b\cos\theta + c\sin\theta$:
$\frac12[f(\theta + \frac\pi2) - f(\theta - \frac\pi2)] = \frac12[-2b\sin\theta + 2c\cos\theta] = f'(\theta)$."""),

    dict(id="C8", kind="mcq",
         q=r"""In the default 8-qubit QCNN of \texttt{QML.ipynb}, the parameter $\theta_{1,1}$ = \texttt{theta[0]} is used by how many
$R_Y$ gates, and how many circuit evaluations does an exact parameter-shift derivative with respect to it need for one image
(shifting each occurrence separately)?""",
         opts=[r"$7$ gates; $14$ circuits", r"$1$ gate; $2$ circuits", r"$4$ gates; $8$ circuits", r"$11$ gates; $22$ circuits"],
         sol=r"""\texttt{theta[0]} is the first parameter of layer 1; it is applied by \texttt{ry(theta[t], qubits[a])} in \emph{every}
convolution block of that layer. Layer 1 acts on all 8 qubits: pairs $(0,1), (2,3), (4,5), (6,7)$ and then $(1,2), (3,4), (5,6)$,
i.e.\ $4 + 3 = 7$ blocks, so 7 gates share $\theta_{1,1}$ (weight sharing, as in a CNN filter). By the product rule
$\partial_\theta f = \sum_{g=1}^{7}\partial_{\theta^{(g)}} f$, and each occurrence needs the two-term shift (2 circuits):
$7\times2 = 14$ circuits. Shifting the shared value once (2 circuits) does \emph{not} give the derivative."""),

    dict(id="C9", kind="mcq",
         q=r"""Using the notebook's layout rule (pool pairs of active qubits, keep the sinks; an odd last qubit moves on unchanged;
4 parameters per layer), a QCNN on $N = 20$ qubits (e.g.\ $4\times5$ pixels) has""",
         opts=[r"$5$ layers and $20$ parameters", r"$4$ layers and $16$ parameters",
               r"$5$ layers and $40$ parameters", r"$19$ layers and $76$ parameters"],
         sol=r"""Track the number of active qubits: pooling keeps $\lceil n/2\rceil$ of $n$ (the sinks, plus the odd last qubit).
$20 \to 10 \to 5 \to 3 \to 2 \to 1$: five layers. With four shared parameters per layer: $4\times5 = 20$ parameters.
In general the layer count is $\lceil\log_2 N\rceil$ and the parameter count $4\lceil\log_2 N\rceil$ --- logarithmic in $N$, while a
dense classical layer on $N$ inputs has $O(N)$ weights."""),

    dict(id="C10", kind="mcq",
         q=r"""The pooling block of \texttt{QML.ipynb} on source qubit $a$ and sink qubit $b$ is
\texttt{rz.ctrl(t3, q[a], q[b]); x(q[a]); rx.ctrl(t4, q[a], q[b]); x(q[a])}. As an operator it equals""",
         opts=[r"$\ket1\!\bra1_a\otimes R_Z(t_3)_b + \ket0\!\bra0_a\otimes R_X(t_4)_b$",
               r"$\ket0\!\bra0_a\otimes R_Z(t_3)_b + \ket1\!\bra1_a\otimes R_X(t_4)_b$",
               r"$R_Z(t_3)_b\,R_X(t_4)_b$ (independent of qubit $a$)",
               r"$\ket1\!\bra1_b\otimes R_Z(t_3)_a + \ket0\!\bra0_b\otimes R_X(t_4)_a$"],
         sol=r"""\texttt{rz.ctrl(t3, q[a], q[b])} $= \ket0\!\bra0_a\otimes I + \ket1\!\bra1_a\otimes R_Z(t_3)$. The pair \texttt{x(q[a]) \dots\ x(q[a])}
turns the following controlled gate into an \emph{anti}-controlled one (it acts when $a$ is $\ket0$):
$\ket1\!\bra1_a\otimes I + \ket0\!\bra0_a\otimes R_X(t_4)$. Multiplying the two (they act on orthogonal control subspaces):
$\ket1\!\bra1_a\otimes R_Z(t_3) + \ket0\!\bra0_a\otimes R_X(t_4)$. Qubit $a$ is never used again; by the deferred-measurement
principle this equals ``measure $a$, then rotate $b$ according to the result''."""),

    dict(id="C11", kind="mcq",
         q=r"""For $N = 16$ qubits the notebook's layout has $26$ convolution blocks and $15$ pooling blocks. Counting the $N$
encoding rotations and every gate of the blocks as \texttt{cudaq.estimate\_resources} does, the circuit has""",
         opts=[r"$154$ gates", r"$124$ gates", r"$138$ gates", r"$170$ gates"],
         sol=r"""Per convolution block: 2 \texttt{ry} + 1 CNOT = 3 gates. Per pooling block: \texttt{crz}, \texttt{x}, \texttt{crx}, \texttt{x} = 4 gates.
Total $= N + 3n_{\text{conv}} + 4n_{\text{pool}} = 16 + 3(26) + 4(15) = 16 + 78 + 60 = 154$.
\emph{Check} with the default: $8 + 3(11) + 4(7) = 8 + 33 + 28 = 69$, the recorded value. (Doubling 69 gives 138, but the
block counts do not scale linearly.)"""),

    dict(id="C12", kind="mcq",
         q=r"""A classifier with outputs in $[-1, 1]$ and prediction $\operatorname{sign} f$ gives $f = 0.6,\ -0.2,\ 0.9,\ 0.1$ on four
images with labels $y = +1,\ -1,\ +1,\ -1$. Its mean squared error (as in \texttt{QML.ipynb}) and accuracy are""",
         opts=[r"$0.505$ and $75\,\%$", r"$2.02$ and $75\,\%$", r"$0.60$ and $75\,\%$", r"$0.505$ and $50\,\%$"],
         sol=r"""Errors $f - y$: $-0.4,\ 0.8,\ -0.1,\ 1.1$; squares $0.16,\ 0.64,\ 0.01,\ 1.21$; sum $2.02$; mean $\mathcal L = 0.505$.
Signs: $+, -, +, +$ against labels $+, -, +, -$: three correct, accuracy $75\,\%$. The second image is classified correctly even
though its error is large ($0.8$): MSE punishes small margins, accuracy only looks at the sign."""),

    dict(id="C13", kind="mcq",
         q=r"""With its defaults (20 training images, \texttt{MAX\_ITERATIONS = 100} loss evaluations, 20 test images) and the
recorded time of $1.7$ ms per circuit on \texttt{qpp-cpu}, the number of circuits \texttt{QML.ipynb} runs for training plus testing
and the resulting estimate of the run time are""",
         opts=[r"$2020$ circuits, about $3.4$ s", r"$2000$ circuits, about $0.17$ s",
               r"$100$ circuits, about $0.17$ s", r"$2020$ circuits, about $34$ s"],
         sol=r"""Every loss evaluation runs the QCNN once per training image: $100\times20 = 2000$ circuits; evaluating the test set adds 20:
$2020$ circuits (the notebook prints exactly this number). Time: $2020\times1.7\ \text{ms} = 3.43$ s; the recorded training time was
$3.4$ s. The classical optimizer itself costs almost nothing --- simulation dominates."""),

    dict(id="C14", kind="mcq",
         q=r"""A single-qubit model alternates trainable unitaries with the data-encoding gate $R_Y(x)$, which appears $L = 3$ times
(``data re-uploading''). Its output $f(x) = \langle Z\rangle$ is a trigonometric series in $x$. The highest frequency it can contain is""",
         opts=[r"$3$", r"$1$", r"$6$", r"$8$"],
         sol=r"""Each $R_Y(x) = e^{-ixY/2}$ has generator eigenvalues $\pm\frac12$. In $f = \braket{\psi|M|\psi}$ each occurrence contributes
phases $e^{\pm ix/2}$ from the ket and from the bra; with $L$ occurrences the exponents are differences of sums of $L$ terms $\pm\frac12$,
so the frequencies are the integers $0, 1, \dots, L$ (Schuld, Sweke \& Meyer 2021). With one encoding gate $f(x) = a + b\cos x + c\sin x$;
with $L = 3$ the spectrum reaches frequency 3. Repeating the encoding increases expressivity."""),

    dict(id="C15", kind="match",
         q=r"""Match each data encoding (List-I) with its description (List-II); $d$ = number of features.""",
         match=([r"basis encoding of a $d$-bit string", r"angle encoding $\bigotimes_i R_Y(x_i)\ket0$",
                 r"amplitude encoding of $d = 2^n$ real numbers", r"ZZ (IQP-type) feature map"],
                [r"$\log_2 d$ qubits; preparing a general state needs $O(d)$ gates",
                 r"$d$ qubits; an $X$ gate on every qubit whose bit is 1; basis states only",
                 r"$d$ qubits; Hadamards, then phases $e^{-i\phi_i(\mathbf x)Z_i}$ and $e^{-i\phi_{ij}(\mathbf x)Z_iZ_j}$; entangling; conjectured hard to simulate classically",
                 r"$d$ qubits; depth 1; product state, so its kernel factorises"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item Basis encoding: $\ket{b_1\dots b_d}$ prepared by $X$ gates (2).
\item Angle encoding (used by \texttt{QML.ipynb} with angles $\pi x_i$): one rotation per qubit, no entanglement (4).
\item Amplitude encoding: exponentially few qubits, but generic state preparation is expensive (1).
\item The ZZ feature map of Havl\'{\i}\v{c}ek et al.\ (2019) uses entangling $Z_iZ_j$ phases; its kernel is believed to be hard to
compute classically for suitable parameters (3).
\end{itemize}
Hence P$\to$2, Q$\to$4, R$\to$1, S$\to$3."""),

    dict(id="C16", kind="match",
         q=r"""Match each QML model (List-I) with what is trained (List-II).""",
         match=([r"quantum-kernel SVM", r"variational quantum classifier (VQC)", r"QCNN", r"quantum GAN"],
                [r"a parameterised generator circuit, trained adversarially against a discriminator",
                 r"no circuit parameters: the feature map is fixed; the classical dual coefficients $\alpha_m$ are trained",
                 r"a feature map followed by a trainable ansatz $U(\boldsymbol\theta)$; $\boldsymbol\theta$ is updated by a classical optimizer",
                 r"convolution and pooling layers with shared parameters; $O(\log N)$ parameters; one output qubit"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item A kernel SVM uses the quantum computer only to estimate $k(\mathbf x_i, \mathbf x_j)$; training is a convex classical problem for $\alpha$ (2).
\item A VQC trains the circuit parameters in a hybrid loop (3).
\item The QCNN of \texttt{QML.ipynb}: 3 layers $\times$ 4 shared parameters on 8 qubits, readout $\langle Z_7\rangle$ (4).
\item A quantum GAN trains a generator circuit to fool a (classical or quantum) discriminator (1).
\end{itemize}
Hence P$\to$2, Q$\to$3, R$\to$4, S$\to$1."""),

    dict(id="C17", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} \texttt{QML.ipynb} trains its QCNN without evaluating any gradient circuits.\\[1mm]
\textbf{Reason (R):} Several QCNN parameters enter controlled rotations (\texttt{rz.ctrl}, \texttt{rx.ctrl}), for which the two-term
parameter-shift rule with shifts $\pm\frac\pi2$ is not exact.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} the default optimizer is COBYLA (Nelder--Mead and Powell are the alternatives); all three are
derivative-free --- they only call \texttt{loss(theta)}.
\textbf{R is true:} the generator of a controlled rotation, $\ket1\!\bra1\otimes\frac P2$, has eigenvalues $0, \pm\frac12$, so $f$ contains the
frequencies $\frac12$ and $1$ and needs a four-term shift rule.
But R is \emph{not} the reason for A: the notebook would use COBYLA even for a circuit with only $R_Y$ gates; the choice of a
gradient-free optimizer is a design decision, not forced by the gate set. Answer: (B)."""),

    dict(id="C18", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} For any feature map, the exact Gram matrix $K_{ij} = |\braket{\psi(\mathbf x_i)|\psi(\mathbf x_j)}|^2$ is
symmetric and positive semi-definite.\\[1mm]
\textbf{Statement II:} When the entries $K_{ij}$ are \emph{estimated} with a finite number of shots, the estimated matrix can have small
negative eigenvalues.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $|\braket{\psi_i|\psi_j}|^2 = \operatorname{tr}(\rho_i\rho_j)$ with $\rho_i = \ket{\psi_i}\!\bra{\psi_i}$, a Hilbert--Schmidt inner product, so
$\sum_{ij}c_ic_jK_{ij} = \operatorname{tr}\big[(\sum_ic_i\rho_i)^2\big] \ge 0$ for real $c$. \textbf{II is true:} each estimate carries independent
statistical noise of order $1/\sqrt S$; the perturbed symmetric matrix need not be PSD, especially when $M$ is large and the
true smallest eigenvalues are near 0. In practice the spectrum is clipped or a small multiple of $I$ is added. Answer: (A)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The model output $f(\mathbf x; \theta) = \langle Z_1\rangle$ equals""",
         opts=[r"$\cos\theta\cos x_0\cos x_1 - \sin\theta\sin x_1$", r"$\cos\theta\cos x_0\cos x_1 + \sin\theta\sin x_1$",
               r"$\cos(\theta + x_1)\cos x_0$", r"$\cos\theta\cos x_1 - \sin\theta\sin x_0\sin x_1$"],
         sol=r"""Work backwards with the observable (Heisenberg picture). (1) Through $R_Y(\theta)$ on qubit 1:
$Z_1 \to \cos\theta\,Z_1 - \sin\theta\,X_1$ (from $R_Y^\dagger ZR_Y = \cos\theta Z - \sin\theta X$). (2) Through the CNOT: $Z$ on the target
becomes $Z_0Z_1$, while $X$ on the target is unchanged. (3) On the product state $R_Y(x_0)\ket0\otimes R_Y(x_1)\ket0$:
$\langle Z_0\rangle = \cos x_0$, $\langle Z_1\rangle = \cos x_1$, $\langle X_1\rangle = \sin x_1$.
Hence $f = \cos\theta\cos x_0\cos x_1 - \sin\theta\sin x_1$."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""For $\mathbf x = (0, \frac\pi3)$, $\theta = \frac\pi6$ and label $y = +1$, the shifted outputs $f(\theta + \frac\pi2)$, $f(\theta - \frac\pi2)$ and
the loss gradient $\partial L/\partial\theta$ are""",
         opts=[r"$-1,\ +1,\ +2$", r"$-1,\ +1,\ -2$", r"$+1,\ -1,\ +2$", r"$-1,\ +1,\ -1$"],
         sol=r"""With $x_0 = 0$, $x_1 = \frac\pi3$: $f(\theta) = \frac12\cos\theta - \frac{\sqrt3}{2}\sin\theta = \cos(\theta + \frac\pi3)$.
$f(\frac\pi6) = \cos\frac\pi2 = 0$; $f(\frac\pi6 + \frac\pi2) = f(\frac{2\pi}3) = \cos(\frac{2\pi}3 + \frac\pi3) = \cos\pi = -1$;
$f(\frac\pi6 - \frac\pi2) = f(-\frac\pi3) = \cos(-\frac\pi3 + \frac\pi3) = 1$.
Parameter shift: $f'(\theta) = \frac12(-1 - 1) = -1$ (exact; check $f' = -\sin(\theta + \frac\pi3) = -1$).
Chain rule: $\frac{\partial L}{\partial\theta} = 2(f - y)f' = 2(0 - 1)(-1) = +2$. A gradient step $\theta \leftarrow \theta - \eta\cdot2$ decreases $\theta$,
which increases $f$ towards the label $+1$."""),
]
