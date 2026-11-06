"""Section B - Quantum Mechanics (states, measurement, gates, tensor products, entanglement, phase kickback).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed JEE option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""\textbf{Hadamard test.} A control qubit starts in $\ket{0}$ and a target register starts in an eigenstate $\ket{u}$
of a unitary $U$ with $U\ket{u} = e^{2\pi i\varphi}\ket{u}$, $0 \le \varphi < 1$. The circuit applies, in order:
(i) $H$ on the control; (ii) controlled-$U$ (control qubit $\to$ target register); (iii) $H$ on the control;
(iv) a measurement of the control qubit in the computational basis. This is the one-counting-qubit version of quantum
phase estimation.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""A two-qubit state is $\ket{\psi} = c\,\big(\ket{00} + 2i\ket{01} - \ket{10} + \sqrt2\,\ket{11}\big)$ with $c > 0$.
If only the \emph{first} qubit is measured in the computational basis, the probability of obtaining $1$ is""",
         opts=[r"$\dfrac38$", r"$\dfrac14$", r"$\dfrac58$", r"$\dfrac12$"],
         sol=r"""Normalisation: $c^2\big(|1|^2 + |2i|^2 + |-1|^2 + |\sqrt2|^2\big) = c^2(1 + 4 + 1 + 2) = 8c^2 = 1$, so $c = \frac{1}{2\sqrt2}$.
By the Born rule, the probability that the first qubit gives 1 is the total weight of the basis states whose first bit is 1:
\[ P(1) = c^2\big(|-1|^2 + |\sqrt2|^2\big) = \frac{1 + 2}{8} = \frac38 . \]
(The phases $i$ and $-1$ do not affect probabilities in this basis; $\frac58$ is $P(0)$; $\frac14$ and $\frac12$ come from
forgetting to normalise or from counting states instead of weights.)"""),

    dict(id="B2", kind="mcq",
         q=r"""Which pair of single-qubit state vectors describes the \emph{same} physical state (i.e.\ gives identical
measurement statistics in every basis)?""",
         opts=[r"$\ket{0} + i\ket{1}\;$ and $\;i\ket{0} - \ket{1}$",
               r"$\ket{0} + i\ket{1}\;$ and $\;\ket{0} - i\ket{1}$",
               r"$\ket{0} + \ket{1}\;$ and $\;\ket{0} - \ket{1}$",
               r"$\ket{0} + i\ket{1}\;$ and $\;i\ket{0} + \ket{1}$"],
         sol=r"""Two vectors represent the same physical state iff they differ by an overall (global) factor $e^{i\gamma}$.
$i\ket{0} - \ket{1} = i\big(\ket{0} + i\ket{1}\big)$, since $i\cdot i = -1$: global phase $i$, so the pair is physically identical.
The other pairs differ by a \emph{relative} phase: $\ket{0} \pm i\ket{1}$ (relative phase $\pm i$), $\ket{0} \pm \ket{1}$
(relative phase $\pm1$), and $i\ket{0} + \ket{1} = i(\ket{0} - i\ket{1})$, which is $\ket{0} - i\ket{1}$ again.
A relative phase is measurable: e.g.\ $\ket{0}+\ket{1} \propto \ket{+}$ and $\ket{0} - \ket{1} \propto \ket{-}$ are perfectly
distinguishable in the $X$ basis. Relative phases are exactly what the QFT writes information into."""),

    dict(id="B3", kind="mcq",
         q=r"""The state $\frac{1}{\sqrt2}\big(\ket{0} + e^{i\phi}\ket{1}\big)$ passes through a Hadamard gate and is then measured
in the computational basis. The probability of outcome $0$ is""",
         opts=[r"$\cos^2(\phi/2)$", r"$\sin^2(\phi/2)$", r"$\dfrac12$", r"$\cos^2\phi$"],
         sol=r"""With $H\ket{0} = \frac{\ket0 + \ket1}{\sqrt2}$ and $H\ket{1} = \frac{\ket0 - \ket1}{\sqrt2}$:
\[ H\,\frac{\ket{0} + e^{i\phi}\ket{1}}{\sqrt2} = \frac{1 + e^{i\phi}}{2}\ket{0} + \frac{1 - e^{i\phi}}{2}\ket{1}. \]
$|1 + e^{i\phi}|^2 = (1 + e^{i\phi})(1 + e^{-i\phi}) = 2 + 2\cos\phi = 4\cos^2(\phi/2)$, so
$P(0) = \frac{4\cos^2(\phi/2)}{4} = \cos^2(\phi/2)$ (and $P(1) = \sin^2(\phi/2)$).
The Hadamard converts an invisible \emph{relative phase} into a visible \emph{probability}: $\phi = 0$ gives $0$ surely,
$\phi = \pi$ gives $1$ surely. A single-qubit QFT is exactly this $H$."""),

    dict(id="B4", kind="mcq",
         q=r"""For the qubit state $\ket{\psi} = \cos\frac{\theta}{2}\ket{0} + e^{i\phi}\sin\frac{\theta}{2}\ket{1}$ the quantity
$\langle X\rangle + i\langle Y\rangle$, where $\langle A\rangle = \bra{\psi}A\ket{\psi}$ and $X, Y$ are Pauli matrices, equals""",
         opts=[r"$\sin\theta\; e^{i\phi}$", r"$\sin\theta\; e^{-i\phi}$", r"$\cos\theta\; e^{i\phi}$", r"$\sin\frac{\theta}{2}\, e^{i\phi}$"],
         sol=r"""Write $a = \cos\frac\theta2$, $b = e^{i\phi}\sin\frac\theta2$. With $X = \begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}$,
$Y = \begin{psmallmatrix}0&-i\\i&0\end{psmallmatrix}$:
\[ \langle X\rangle = \bar a b + \bar b a = 2\,\mathrm{Re}(\bar a b), \qquad \langle Y\rangle = -i\bar a b + i\bar b a = 2\,\mathrm{Im}(\bar a b). \]
Hence $\langle X\rangle + i\langle Y\rangle = 2\bar a b = 2\cos\frac\theta2\sin\frac\theta2\,e^{i\phi} = \sin\theta\,e^{i\phi}$.
So $\langle X\rangle = \sin\theta\cos\phi$, $\langle Y\rangle = \sin\theta\sin\phi$ (and $\langle Z\rangle = \cos\theta$): the Bloch vector
$(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. The \emph{azimuth} $\phi$ is the phase that QFT rotations change."""),

    dict(id="B5", kind="mcq",
         q=r"""Which of the following two-qubit states is \emph{entangled} (cannot be written as $\ket{a}\otimes\ket{b}$)?""",
         opts=[r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)$",
               r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} + \ket{11}\big)$",
               r"$\frac12\big(\ket{00} - \ket{01} + \ket{10} - \ket{11}\big)$",
               r"$\frac12\big(\ket{00} + i\ket{01} - \ket{10} - i\ket{11}\big)$"],
         sol=r"""For $\sum c_{jk}\ket{jk}$, a product $(a_0\ket0 + a_1\ket1)(b_0\ket0 + b_1\ket1)$ has $c_{jk} = a_jb_k$, so
$c_{00}c_{11} = a_0a_1b_0b_1 = c_{01}c_{10}$. Conversely, $c_{00}c_{11} = c_{01}c_{10}$ (the $2\times2$ coefficient matrix has
determinant 0, i.e.\ rank 1) guarantees a product. Test each option (factor $\frac12$ dropped):
\begin{itemize}\itemsep0pt
\item $(1, 1, 1, -1)$: $c_{00}c_{11} = -1 \neq 1 = c_{01}c_{10}$ $\Rightarrow$ \textbf{entangled}.
\item $(1, 1, 1, 1)$: $1 = 1$, product $\ket{+}\ket{+}$.
\item $(1, -1, 1, -1)$: $-1 = -1$, product $\ket{+}\ket{-}$.
\item $(1, i, -1, -i)$: $-i = -i$, product $\frac{1}{2}(\ket0 - \ket1)(\ket0 + i\ket1)$.
\end{itemize}
The entangled state is $\mathrm{CZ}\ket{+}\ket{+}$: a controlled-phase gate (like those in the QFT) \emph{can} entangle.
(This is not in conflict with C14: the QFT acting on a \emph{basis} state happens to give a product state.)"""),

    dict(id="B6", kind="mcq",
         q=r"""A CNOT gate (qubit 1 = control, qubit 2 = target) acts on $\ket{+}\ket{-}$, where
$\ket{\pm} = \frac{1}{\sqrt2}(\ket0 \pm \ket1)$. The output is""",
         opts=[r"$\ket{-}\ket{-}$", r"$\ket{+}\ket{-}$", r"$\ket{-}\ket{+}$", r"$\frac{1}{\sqrt2}\big(\ket{0}\ket{-} + \ket{1}\ket{+}\big)$"],
         sol=r"""$\ket{-}$ is an eigenvector of $X$ with eigenvalue $-1$: $X\ket{-} = \frac{\ket1 - \ket0}{\sqrt2} = -\ket{-}$. Therefore
\[ \mathrm{CNOT}\,\ket{+}\ket{-} = \frac{1}{\sqrt2}\big(\ket0\ket{-} + \ket1\,X\ket{-}\big) = \frac{1}{\sqrt2}\big(\ket0 - \ket1\big)\ket{-}
   = \ket{-}\ket{-}. \]
The target is unchanged; the eigenvalue $-1$ has been ``kicked back'' onto the control as a relative phase. This
\emph{phase kickback} is the mechanism by which controlled-$U^{2^j}$ gates load the eigenphase into the counting
register in phase estimation (see B8)."""),

    dict(id="B7", kind="mcq",
         q=r"""In the basis $\ket{00}, \ket{01}, \ket{10}, \ket{11}$ (first qubit = control), the matrix of controlled-$R_3$, where
$R_k = \operatorname{diag}\big(1, e^{2\pi i/2^k}\big)$, is""",
         opts=[r"$\operatorname{diag}\big(1, 1, 1, e^{i\pi/4}\big)$", r"$\operatorname{diag}\big(1, 1, 1, e^{i\pi/8}\big)$",
               r"$\operatorname{diag}\big(1, e^{i\pi/4}, 1, e^{i\pi/4}\big)$", r"$\operatorname{diag}\big(1, 1, e^{i\pi/4}, e^{i\pi/4}\big)$"],
         sol=r"""$R_3 = \operatorname{diag}(1, e^{2\pi i/8}) = \operatorname{diag}(1, e^{i\pi/4})$ (the $T$ gate). Controlled-$R_3$ applies $R_3$
to the target only when the control is $\ket1$:
\[ \mathrm{C}R_3 = \ket0\!\bra0 \otimes I + \ket1\!\bra1 \otimes R_3 = \operatorname{diag}(1, 1, 1, e^{i\pi/4}). \]
Only $\ket{11}$ picks up a phase, so the gate is \emph{symmetric}: it does not matter which qubit is called the control.
$\operatorname{diag}(1, e^{i\pi/4}, 1, e^{i\pi/4}) = I\otimes R_3$ and $\operatorname{diag}(1,1,e^{i\pi/4},e^{i\pi/4}) = R_3 \otimes I$ are
uncontrolled; $e^{i\pi/8}$ confuses $2\pi/2^3$ with $\pi/2^3$."""),

    dict(id="B8", kind="mcq",
         q=r"""$U\ket{u} = e^{2\pi i\varphi}\ket{u}$. A control qubit in $\ket{+}$ and the register in $\ket{u}$ pass through a
controlled-$U^{2^j}$ gate. The output state is""",
         opts=[r"$\frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\,2^j\varphi}\ket1\big)\ket{u}$",
               r"$\frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\,\varphi/2^j}\ket1\big)\ket{u}$",
               r"$\frac{1}{\sqrt2}\,e^{2\pi i\,2^j\varphi}\big(\ket0 + \ket1\big)\ket{u}$",
               r"$\frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\,j\varphi}\ket1\big)\ket{u}$"],
         sol=r"""$U^{2^j}\ket{u} = \big(e^{2\pi i\varphi}\big)^{2^j}\ket{u} = e^{2\pi i\,2^j\varphi}\ket{u}$. The gate acts only on the $\ket1$ branch:
\[ \frac{1}{\sqrt2}\big(\ket0\ket u + \ket1\,U^{2^j}\ket u\big) = \frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\,2^j\varphi}\ket1\big)\ket u. \]
The phase is \emph{relative} (only on $\ket1$), not global, so it is measurable after a Hadamard / inverse QFT. If
$\varphi = 0.\varphi_1\varphi_2\dots$ in binary, $2^j\varphi$ shifts the binary point $j$ places and the phase is
$e^{2\pi i\,0.\varphi_{j+1}\varphi_{j+2}\dots}$ --- exactly the factor in the QFT product formula."""),

    dict(id="B9", kind="mcq",
         q=r"""The state $\frac{1}{\sqrt3}\big(\ket{00} + \ket{01} + \ket{11}\big)$ is prepared and only the \emph{first} qubit is
measured, giving $0$. The probability of this outcome and the post-measurement state of the second qubit are""",
         opts=[r"$\frac23$ and $\ket{+}$", r"$\frac13$ and $\ket{+}$", r"$\frac23$ and $\ket{0}$", r"$\frac13$ and $\ket{0}$"],
         sol=r"""Group by the first qubit: $\frac{1}{\sqrt3}\Big(\ket0\big(\ket0 + \ket1\big) + \ket1\ket1\Big)
= \sqrt{\tfrac23}\,\ket0\ket{+} + \sqrt{\tfrac13}\,\ket1\ket1$.
Probability of first qubit $= 0$: $\big|\sqrt{2/3}\big|^2 = \frac23$. By the projection postulate the state collapses to the
normalised $\ket{0}\ket{+}$, so the second qubit is $\ket{+} = \frac{1}{\sqrt2}(\ket0 + \ket1)$.
(Partial measurement keeps the \emph{relative} amplitudes of the surviving branch; this is why in Shor's algorithm measuring
the work register leaves the counting register in a periodic superposition.)"""),

    dict(id="B10", kind="mcq",
         q=r"""Which of the following matrices can \emph{not} be a quantum gate?""",
         opts=[r"$\frac{1}{\sqrt2}\begin{pmatrix}1&1\\ i&i\end{pmatrix}$", r"$\frac{1}{\sqrt2}\begin{pmatrix}1&1\\ 1&-1\end{pmatrix}$",
               r"$\frac{1}{\sqrt2}\begin{pmatrix}1&i\\ i&1\end{pmatrix}$", r"$\begin{pmatrix}0&1\\ i&0\end{pmatrix}$"],
         sol=r"""Closed-system evolution is unitary: $U^\dagger U = I$, i.e.\ the columns are orthonormal.
For $\frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\ i&i\end{psmallmatrix}$ both columns equal $\frac{1}{\sqrt2}(1, i)^T$, so their inner
product is $1 \neq 0$ (also $\det = 0$): it maps $\ket0$ and $\ket1$ to the same state and would destroy information --- not unitary.
Checks: $H$ is unitary; for $\frac{1}{\sqrt2}\begin{psmallmatrix}1&i\\ i&1\end{psmallmatrix}$ the column inner product is
$\frac12(\bar1\cdot i + \bar i\cdot 1) = \frac12(i - i) = 0$ and each column has norm 1; $\begin{psmallmatrix}0&1\\ i&0\end{psmallmatrix}$ has
columns $(0, i)$ and $(1, 0)$, orthonormal. (Unitarity is also why the QFT is invertible: $F^{-1} = F^\dagger$.)"""),

    dict(id="B11", kind="mcq",
         q=r"""Which two-qubit gate does \emph{not} commute with controlled-$R_2 = \operatorname{diag}(1, 1, 1, i)$?""",
         opts=[r"$H\otimes I$", r"$Z\otimes Z$", r"$\mathrm{SWAP}$", r"$S\otimes I$, where $S = \operatorname{diag}(1, i)$"],
         sol=r"""$Z\otimes Z$ and $S \otimes I$ are diagonal; diagonal matrices always commute.
SWAP exchanges $\ket{01} \leftrightarrow \ket{10}$; $\mathrm{C}R_2$ has the same entry ($1$) on both, so
$\mathrm{SWAP}\cdot\mathrm{C}R_2\cdot\mathrm{SWAP} = \mathrm{C}R_2$: they commute (controlled-phase gates are symmetric).
$H\otimes I$ mixes $\ket{0x}$ with $\ket{1x}$. For $x = 1$ the diagonal entries of $\mathrm{C}R_2$ on $\ket{01}$ and $\ket{11}$ are
$1$ and $i$ --- different --- so the two operators do not commute. Explicitly
$(H\otimes I)\,\mathrm{C}R_2\ket{01} = \frac{1}{\sqrt2}(\ket{01} + \ket{11})$ but $\mathrm{C}R_2(H\otimes I)\ket{01} = \frac{1}{\sqrt2}(\ket{01} + i\ket{11})$.
This is why the order ``$H$ first, then the controlled rotations'' in the QFT circuit matters."""),

    dict(id="B12", kind="mcq",
         q=r"""A state-vector simulator stores each amplitude as a single-precision complex number (two 4-byte floats).
The memory needed for the full state of $32$ qubits is""",
         opts=[r"$32$ GiB", r"$16$ GiB", r"$64$ GiB", r"$256$ bytes"],
         sol=r"""$n$ qubits $\Rightarrow 2^n$ amplitudes. Each amplitude occupies $2 \times 4 = 8$ bytes. Memory $= 2^{32}\times 8 = 2^{35}$ bytes
$= 2^{5}\times 2^{30}$ bytes $= 32$ GiB. Double precision (16 bytes) would need 64 GiB; 256 bytes ($32\times8$) is the
wrong idea that each qubit needs one number. Every extra qubit doubles the memory --- the reason the course notebooks
default to 15 qubits ($2^{15}\times 8$ B $= 256$ KiB) on a laptop."""),

    dict(id="B13", kind="mcq",
         q=r"""$H\otimes H$ is applied to the Bell state $\frac{1}{\sqrt2}\big(\ket{00} - \ket{11}\big)$, followed by a measurement of both
qubits. The probability of the outcome $01$ is""",
         opts=[r"$\dfrac12$", r"$0$", r"$\dfrac14$", r"$1$"],
         sol=r"""$H\otimes H\ket{00} = \frac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$ and
$H\otimes H\ket{11} = \frac12(\ket{00} - \ket{01} - \ket{10} + \ket{11})$ (sign $(-1)^{x\cdot y}$). Subtract and divide by $\sqrt2$:
\[ \frac{1}{2\sqrt2}\big(2\ket{01} + 2\ket{10}\big) = \frac{1}{\sqrt2}\big(\ket{01} + \ket{10}\big). \]
Hence $P(01) = P(10) = \frac12$ and $P(00) = P(11) = 0$. The perfect ``equal bits'' correlation in the $Z$ basis became a
perfect ``opposite bits'' correlation in the $X$ basis --- interference of four amplitudes, not a product-state result
($\frac14$)."""),

    dict(id="B14", kind="mcq",
         q=r"""A source emits a qubit that is $\ket0$ with probability $\frac12$ and $\ket1$ with probability $\frac12$ (a classical
coin decides; nobody knows which). A Hadamard gate is applied and the qubit is measured. The probability of outcome $0$ is""",
         opts=[r"$\dfrac12$", r"$1$", r"$0$", r"$\dfrac14$"],
         sol=r"""This is a \emph{mixture}, not a superposition. Treat the two cases separately and average:
$H\ket0 = \ket{+}$ gives $P(0) = \frac12$; $H\ket1 = \ket{-}$ gives $P(0) = \frac12$. Hence $P(0) = \frac12\cdot\frac12 + \frac12\cdot\frac12 = \frac12$.
(Density matrix: $\rho = \frac12 I$, and $H\rho H^\dagger = \frac12 I$.) Contrast: the superposition
$\ket{+} = \frac{1}{\sqrt2}(\ket0 + \ket1)$ gives $H\ket{+} = \ket0$ and $P(0) = 1$. Amplitudes interfere, classical probabilities do
not --- QFT peaks need coherence, which is why decoherence (dephasing) destroys them."""),

    dict(id="B15", kind="match",
         q=r"""Match each single-qubit gate in List-I with its action in List-II. ($R_k = \operatorname{diag}(1, e^{2\pi i/2^k})$.)""",
         match=([r"$X$", r"$H$", r"$R_2$", r"$R_3$"],
                [r"$\ket{1} \mapsto \frac{1}{\sqrt2}\big(\ket0 - \ket1\big)$", r"$\ket{1} \mapsto e^{i\pi/4}\ket{1}$",
                 r"$\ket{0} \leftrightarrow \ket{1}$", r"$\ket{1} \mapsto i\ket{1}$"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4",
               r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""$X$ is the bit flip: $\ket0 \leftrightarrow \ket1$ (3). $H\ket1 = \frac{1}{\sqrt2}(\ket0 - \ket1) = \ket{-}$ (1).
$R_2 = \operatorname{diag}(1, e^{2\pi i/4}) = \operatorname{diag}(1, i) = S$, so $\ket1 \mapsto i\ket1$ (4).
$R_3 = \operatorname{diag}(1, e^{2\pi i/8}) = T$, so $\ket1 \mapsto e^{i\pi/4}\ket1$ (2).
Hence P$\to$3, Q$\to$1, R$\to$4, S$\to$2. Note $R_1 = Z$, $R_2 = S$, $R_3 = T$, and $R_{k}^{2} = R_{k-1}$."""),

    dict(id="B16", kind="match",
         q=r"""Match each two-qubit state in List-I with its statistics when \emph{both} qubits are measured in the computational
basis (List-II).""",
         match=([r"$\frac{1}{\sqrt2}\big(\ket{00} + \ket{11}\big)$", r"$\frac{1}{\sqrt2}\big(\ket{01} - \ket{10}\big)$",
                 r"$\frac12\big(\ket0 + i\ket1\big)\big(\ket0 - \ket1\big)$", r"$\ket{0}\ket{+}$"],
                [r"all four outcomes, each with probability $\frac14$", r"the two bits are always equal; $00$ and $11$ each $\frac12$",
                 r"first bit always $0$; second bit $0$ or $1$ with $\frac12$ each", r"the two bits always differ; $01$ and $10$ each $\frac12$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3"],
         sol=r"""P ($\Phi^+$): amplitudes only on $\ket{00}$, $\ket{11}$, each $\frac{1}{\sqrt2}$ $\Rightarrow$ equal bits, $\frac12$ each (2).
Q ($\Psi^-$): only $\ket{01}$, $\ket{10}$ $\Rightarrow$ bits always differ (4); the minus sign is invisible in this basis.
R: expand $\frac12(\ket{00} - \ket{01} + i\ket{10} - i\ket{11})$; every amplitude has modulus $\frac12$ $\Rightarrow$ all four outcomes
with probability $\frac14$ (1). The phases $i$ and $-1$ do not show up --- just like the output of a QFT on a basis state.
S: $\ket0\ket{+} = \frac{1}{\sqrt2}(\ket{00} + \ket{01})$ $\Rightarrow$ first bit 0, second random (3).
Hence P$\to$2, Q$\to$4, R$\to$1, S$\to$3."""),

    dict(id="B17", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} The states $\ket{+}$ and $\ket{-}$ cannot be distinguished by a measurement in the computational
basis $\{\ket0, \ket1\}$.\\[1mm]
\textbf{Reason (R):} $\ket{+}$ and $\ket{-}$ differ only by a global phase.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} for both states $P(0) = P(1) = \frac12$, because $|\pm\frac{1}{\sqrt2}|^2 = \frac12$.
\textbf{R is false:} $\ket{-} = \frac{1}{\sqrt2}(\ket0 - \ket1)$ is not $e^{i\gamma}\ket{+}$ for any $\gamma$; the two differ by a
\emph{relative} phase $-1$ on $\ket1$. They are orthogonal ($\braket{+|-} = 0$) and therefore perfectly distinguishable in the
$X$ basis (apply $H$, then measure: $\ket{+}\to\ket0$, $\ket-\to\ket1$). States that differ by a global phase could never be
distinguished in \emph{any} basis. Answer: (C)."""),

    dict(id="B18", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} The Bell state $\frac{1}{\sqrt2}\big(\ket{00} + \ket{11}\big)$ can be written as $\ket{a}\otimes\ket{b}$ for
suitable single-qubit states $\ket a$, $\ket b$.\\[1mm]
\textbf{Statement II:} A two-qubit state $c_{00}\ket{00} + c_{01}\ket{01} + c_{10}\ket{10} + c_{11}\ket{11}$ is a product state if and only
if $c_{00}c_{11} = c_{01}c_{10}$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{II is true.} ($\Rightarrow$) If the state is $(a_0\ket0 + a_1\ket1)(b_0\ket0 + b_1\ket1)$ then $c_{jk} = a_jb_k$ and
$c_{00}c_{11} = a_0b_0a_1b_1 = c_{01}c_{10}$. ($\Leftarrow$) If $c_{00}c_{11} = c_{01}c_{10}$, the matrix $C = \begin{psmallmatrix}c_{00}&c_{01}\\ c_{10}&c_{11}\end{psmallmatrix}$
has $\det C = 0$, so it has rank 1 and $C = \mathbf{a}\mathbf{b}^T$, i.e.\ $c_{jk} = a_jb_k$.
\textbf{I is false:} for $\Phi^+$, $c_{00}c_{11} = \frac12 \neq 0 = c_{01}c_{10}$. Answer: (D)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""The probability that the control qubit is found in state $\ket1$ is""",
         opts=[r"$\sin^2(\pi\varphi)$", r"$\cos^2(\pi\varphi)$", r"$\sin^2(2\pi\varphi)$", r"$\sin^2(\pi\varphi/2)$"],
         sol=r"""After (i): $\frac{1}{\sqrt2}(\ket0 + \ket1)\ket u$. After (ii), by phase kickback (B8 with $j = 0$):
$\frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\varphi}\ket1\big)\ket u$. After (iii) (B3 with $\phi = 2\pi\varphi$):
\[ \Big(\frac{1 + e^{2\pi i\varphi}}{2}\ket0 + \frac{1 - e^{2\pi i\varphi}}{2}\ket1\Big)\ket u . \]
$|1 - e^{2\pi i\varphi}|^2 = 2 - 2\cos(2\pi\varphi) = 4\sin^2(\pi\varphi)$, so $P(1) = \sin^2(\pi\varphi)$ and $P(0) = \cos^2(\pi\varphi)$.
$\varphi = 0 \Rightarrow$ always 0; $\varphi = \frac12 \Rightarrow$ always 1: one qubit resolves exactly one binary digit
$\varphi = 0.\varphi_1$. More digits need more counting qubits and an inverse QFT."""),

    dict(id="B20", kind="para", passage="B",
         q=r"""Take $\varphi = \frac13$. The circuit is run repeatedly (fresh $\ket0$ control each time). The minimum number of runs
needed so that the probability of observing outcome $1$ \emph{at least once} exceeds $0.99$ is""",
         opts=[r"$4$", r"$3$", r"$2$", r"$7$"],
         sol=r"""From B19, $P(1) = \sin^2(\pi/3) = \left(\frac{\sqrt3}{2}\right)^2 = \frac34$, so a single run gives 0 with probability $\frac14$.
Runs are independent: $P(\text{no } 1 \text{ in } k \text{ runs}) = \left(\frac14\right)^k$. We need $1 - 4^{-k} > 0.99$, i.e.\ $4^k > 100$.
$4^3 = 64 < 100$ (failure probability $0.0156$), $4^4 = 256 > 100$ (failure probability $0.0039$). Hence $k = 4$.
(Repetition amplifies confidence; it does not make the single-shot estimate of $\varphi = \frac13 = 0.0101\ldots_2$ exact.)"""),
]
