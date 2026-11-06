"""Section B - Quantum Computing Foundations (qubits and states, Born rule, Bloch sphere, Pauli algebra, gates and
unitaries, tensor products, entanglement, measurement statistics, circuits and parameterized circuits).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""A two-qubit parameterized circuit starts in $\ket{00}$, applies $R_Y(\theta)$ to qubit 0 and then a CNOT with
control qubit 0 and target qubit 1:
\begin{center}
\begin{tikzpicture}[x=1cm, y=0.8cm, font=\small]
\draw (0,0) node[left]{$q_0:\ \ket0$} -- (4,0); \draw (0,-1) node[left]{$q_1:\ \ket0$} -- (4,-1);
\node[draw, fill=white, minimum width=1.2cm] at (1.2,0) {$R_Y(\theta)$};
\fill (2.7,0) circle (2pt); \draw (2.7,0) -- (2.7,-1.18); \draw (2.7,-1) circle (5pt);
\node[draw, fill=white] at (3.6,0) {\tiny M}; \node[draw, fill=white] at (3.6,-1) {\tiny M};
\end{tikzpicture}
\end{center}
Both qubits are then measured in the computational basis; $0 \le \theta \le \pi$.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""For the qubit state $\ket\psi = \frac12\ket0 + \frac{\sqrt3}{2}e^{i\pi/4}\ket1$ the expectation values
$\big(\langle X\rangle, \langle Z\rangle\big)$ are""",
         opts=[r"$\big(\tfrac{\sqrt6}{4},\ -\tfrac12\big)$", r"$\big(\tfrac{\sqrt3}{2},\ -\tfrac12\big)$",
               r"$\big(\tfrac{\sqrt6}{4},\ \tfrac12\big)$", r"$\big(\tfrac{\sqrt6}{2},\ -\tfrac12\big)$"],
         sol=r"""For $\ket\psi = \alpha\ket0 + \beta\ket1$: $\langle Z\rangle = |\alpha|^2 - |\beta|^2$ and $\langle X\rangle = 2\operatorname{Re}(\alpha^*\beta)$.
Here $|\alpha|^2 = \frac14$, $|\beta|^2 = \frac34$, so $\langle Z\rangle = -\frac12$. And
$\langle X\rangle = 2\cdot\frac12\cdot\frac{\sqrt3}{2}\cos\frac\pi4 = \frac{\sqrt3}{2}\cdot\frac{\sqrt2}{2} = \frac{\sqrt6}{4} \approx 0.612$.
(Check: $\langle Y\rangle = 2\operatorname{Im}(\alpha^*\beta) = \frac{\sqrt6}{4}$ and $\langle X\rangle^2 + \langle Y\rangle^2 + \langle Z\rangle^2 = \frac38 + \frac38 + \frac14 = 1$: a pure state lies on the Bloch sphere.)"""),

    dict(id="B2", kind="mcq",
         q=r"""For real numbers $a, b, c$ the operator $(aX + bY + cZ)^2$ equals""",
         opts=[r"$(a^2 + b^2 + c^2)\,I$", r"$(a + b + c)^2\,I$",
               r"$(a^2 + b^2 + c^2)\,I + 2i(abZ + bcX + caY)$", r"$a^2X + b^2Y + c^2Z$"],
         sol=r"""Expand: $a^2X^2 + b^2Y^2 + c^2Z^2 + ab(XY + YX) + bc(YZ + ZY) + ca(ZX + XZ)$.
Every Pauli matrix squares to $I$, and different Pauli matrices \emph{anticommute}: $XY = iZ = -YX$, $YZ = iX = -ZY$,
$ZX = iY = -XZ$, so all cross terms vanish. Result: $(a^2 + b^2 + c^2)I$. Consequence: $\hat n\cdot\vec\sigma$ with a unit vector
$\hat n$ has eigenvalues $\pm1$ --- every such observable is a ``rotated $Z$''."""),

    dict(id="B3", kind="mcq",
         q=r"""A circuit contains the parameter $\theta$ in exactly one gate $R_Y(\theta) = e^{-i\theta Y/2}$, and
$f(\theta) = \braket{\psi(\theta)|M|\psi(\theta)}$ is the expectation value of an observable $M$ measured at the end. Although
$R_Y(\theta + 2\pi) = -R_Y(\theta)$, the smallest period that $f$ is guaranteed to have is""",
         opts=[r"$2\pi$", r"$4\pi$", r"$\pi$", r"$\pi/2$"],
         sol=r"""Write the circuit as $\ket{\psi(\theta)} = V R_Y(\theta) W\ket0$. Replacing $\theta$ by $\theta + 2\pi$ multiplies the state by
$-1$, a \emph{global} phase, and $f = \braket{\psi|M|\psi}$ contains $\ket\psi$ and $\bra\psi$, so the sign cancels:
$f(\theta + 2\pi) = f(\theta)$. In general $f(\theta) = a + b\cos\theta + c\sin\theta$ (only the frequency 1 appears), whose
period is exactly $2\pi$ unless $b = c = 0$; $\pi$ or $\pi/2$ would require extra symmetry. The operator itself has period $4\pi$."""),

    dict(id="B4", kind="mcq",
         q=r"""Which of the following two-qubit states is a \emph{product} (non-entangled) state?""",
         opts=[r"$\frac12\big(\ket{00} - \ket{01} + \ket{10} - \ket{11}\big)$", r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)$",
               r"$\frac1{\sqrt2}\big(\ket{00} + \ket{11}\big)$", r"$\frac1{\sqrt2}\big(\ket{01} - \ket{10}\big)$"],
         sol=r"""Write $\ket\psi = \sum c_{ab}\ket{ab}$ and arrange the coefficients as a $2\times2$ matrix $C = (c_{ab})$. A pure state is a
product state iff $\det C = c_{00}c_{11} - c_{01}c_{10} = 0$ (rank 1).
First state: $\frac12\begin{psmallmatrix}1 & -1\\ 1 & -1\end{psmallmatrix}$, determinant $\frac14(-1 + 1) = 0$. Indeed it equals
$\frac{1}{\sqrt2}(\ket0 + \ket1)\otimes\frac{1}{\sqrt2}(\ket0 - \ket1) = \ket+\ket-$. The other three have $\det C = -\frac12$,
$\frac12$ and $\frac12$: all entangled."""),

    dict(id="B5", kind="mcq",
         q=r"""For the Bell state $\ket{\Phi^+} = \frac{1}{\sqrt2}(\ket{00} + \ket{11})$ the expectation values
$\big(\langle Z_0\rangle,\ \langle Z_0Z_1\rangle,\ \langle X_0X_1\rangle\big)$ are""",
         opts=[r"$(0,\ 1,\ 1)$", r"$(0,\ 0,\ 0)$", r"$(0,\ 1,\ 0)$", r"$(1,\ 1,\ 1)$"],
         sol=r"""$\langle Z_0\rangle$: outcomes $00$ and $11$ each with probability $\frac12$, so qubit 0 gives $+1$ and $-1$ equally: $0$.
$\langle Z_0Z_1\rangle$: $Z_0Z_1\ket{ab} = (-1)^{a+b}\ket{ab}$; both $00$ and $11$ have even parity, so $+1$.
$\langle X_0X_1\rangle$: $X\otimes X$ maps $\ket{00} \leftrightarrow \ket{11}$, so $X_0X_1\ket{\Phi^+} = \ket{\Phi^+}$: eigenvalue $+1$.
The individual qubits look completely random, but their correlations are perfect --- in two different bases."""),

    dict(id="B6", kind="mcq",
         q=r"""Starting from $\ket{00}$, a circuit applies $H$ to qubit 0, a CNOT (control 0, target 1) and then $R_Y(\theta)$ to
qubit 1. The expectation value $\langle Z_1\rangle$ of the final state is""",
         opts=[r"$0$ for every $\theta$", r"$\cos\theta$", r"$\cos^2(\theta/2)$", r"$-\sin\theta$"],
         sol=r"""After $H$ and CNOT the state is $\ket{\Phi^+} = \frac{1}{\sqrt2}(\ket{00} + \ket{11})$. The reduced state of qubit 1 is
$\rho_1 = \operatorname{tr}_0\ket{\Phi^+}\!\bra{\Phi^+} = \frac12 I$ (maximally mixed). A local unitary on qubit 1 maps it to
$R_Y\,\frac I2\,R_Y^\dagger = \frac I2$, so $\langle Z_1\rangle = \operatorname{tr}(\rho_1 Z) = 0$ for every $\theta$.
A single-qubit readout of an entangled qubit can be completely insensitive to a local parameter
(here $\langle Z_0Z_1\rangle = \cos\theta$ does depend on $\theta$) --- a small version of the trainability problems of Section D."""),

    dict(id="B7", kind="mcq",
         q=r"""A qubit is prepared and measured $1000$ times in the computational basis; outcome $0$ occurs $620$ times. The
estimate of $\langle Z\rangle$ and its standard error are approximately""",
         opts=[r"$0.24 \pm 0.031$", r"$0.62 \pm 0.015$", r"$0.24 \pm 0.0009$", r"$0.24 \pm 0.015$"],
         sol=r"""$\widehat{\langle Z\rangle} = \hat P(0) - \hat P(1) = 0.62 - 0.38 = 0.24$. One shot gives $\pm1$ with variance
$1 - \langle Z\rangle^2$, so the mean of $S$ shots has standard error $\sqrt{(1 - \langle Z\rangle^2)/S} = \sqrt{(1 - 0.0576)/1000}
= \sqrt{0.000942} = 0.0307$. (Equivalently $2\sqrt{\hat p(1-\hat p)/S} = 2\sqrt{0.62\cdot0.38/1000} = 2(0.01535)$.)
Halving the error needs four times the shots."""),

    dict(id="B8", kind="mcq",
         q=r"""The state $\big(R_Y(\pi/2)\otimes R_Y(\pi)\big)\ket{00}$ (the first factor acts on qubit 0), written as the amplitude
vector in the basis order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$, is""",
         opts=[r"$\frac{1}{\sqrt2}(0,\ 1,\ 0,\ 1)$", r"$\frac{1}{\sqrt2}(0,\ 0,\ 1,\ 1)$",
               r"$\frac{1}{\sqrt2}(1,\ 0,\ 1,\ 0)$", r"$\frac{1}{\sqrt2}(0,\ 1,\ 0,\ -1)$"],
         sol=r"""$R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$. Qubit 0: $R_Y(\pi/2)\ket0 = \frac{1}{\sqrt2}(\ket0 + \ket1)$.
Qubit 1: $R_Y(\pi)\ket0 = \ket1$. The product is $\frac{1}{\sqrt2}(\ket0 + \ket1)\otimes\ket1 = \frac{1}{\sqrt2}(\ket{01} + \ket{11})$,
i.e.\ the vector $\frac{1}{\sqrt2}(0, 1, 0, 1)$ (qubit 0 written first). The vector $(0, 0, 1, 1)/\sqrt2$ is what one gets with the
qubit order reversed --- conventions matter when reading state vectors."""),

    dict(id="B9", kind="mcq",
         q=r"""Which of the following matrices is \emph{not} unitary?""",
         opts=[r"$\frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix}$", r"$\frac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$",
               r"$\begin{pmatrix}1 & 0\\ 0 & i\end{pmatrix}$", r"$\begin{pmatrix}c & -is\\ -is & c\end{pmatrix}$ with $c = \cos\frac\pi8$, $s = \sin\frac\pi8$"],
         sol=r"""$U$ is unitary iff $U^\dagger U = I$, i.e.\ its columns are orthonormal. For $\frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\1&1\end{psmallmatrix}$ both
columns are equal, $\frac{1}{\sqrt2}(1,1)$: they are not orthogonal, and $U^\dagger U = \begin{psmallmatrix}1&1\\1&1\end{psmallmatrix} \neq I$
(the matrix is singular, $\det = 0$). The others are the Hadamard $H$, the phase gate $S$ and $R_X(\pi/4)$, all unitary.
Only unitary gates (plus measurement) are allowed in a quantum circuit."""),

    dict(id="B10", kind="mcq",
         q=r"""Which pair of states gives \emph{identical} outcome statistics for every possible measurement?""",
         opts=[r"$\frac{1}{\sqrt2}(\ket0 + i\ket1)$ and $\frac{e^{i\pi/3}}{\sqrt2}(\ket0 + i\ket1)$",
               r"$\frac{1}{\sqrt2}(\ket0 + \ket1)$ and $\frac{1}{\sqrt2}(\ket0 - \ket1)$",
               r"$\frac{1}{\sqrt2}(\ket0 + i\ket1)$ and $\frac{1}{\sqrt2}(\ket0 - i\ket1)$",
               r"$\frac{1}{\sqrt2}(\ket0 + \ket1)$ and $\frac{1}{\sqrt2}(\ket0 + i\ket1)$"],
         sol=r"""Two state vectors describe the same physical state iff they differ by a \emph{global} phase $e^{i\phi}$: every probability
$|\braket{m|\psi}|^2$ is unchanged by $\ket\psi \to e^{i\phi}\ket\psi$. The first pair differs by $e^{i\pi/3}$.
The other pairs differ by a \emph{relative} phase between $\ket0$ and $\ket1$: they have different Bloch vectors
($\pm\hat x$; $\pm\hat y$; $\hat x$ vs.\ $\hat y$) and are distinguished by measuring $X$ or $Y$ (e.g.\ $H$ then $Z$)."""),

    dict(id="B11", kind="mcq",
         q=r"""For $R_Y(\theta) = e^{-i\theta Y/2}$, the conjugated observable $R_Y(\theta)^\dagger\, Z\, R_Y(\theta)$ equals""",
         opts=[r"$\cos\theta\,Z - \sin\theta\,X$", r"$\cos\theta\,Z + \sin\theta\,X$",
               r"$\cos\theta\,Z - \sin\theta\,Y$", r"$Z$"],
         sol=r"""Use $R_Y(\theta) = \cos\frac\theta2 I - i\sin\frac\theta2 Y$ and $YZ = iX$, $ZY = -iX$:
$R_Y^\dagger Z R_Y = (c + isY)Z(c - isY) = c^2Z + ics(YZ - ZY) + s^2YZY$ with $c = \cos\frac\theta2$, $s = \sin\frac\theta2$.
$YZ - ZY = 2iX$, so $ics\cdot2iX = -2csX = -\sin\theta X$; and $YZY = -Z$, so $s^2YZY = -s^2Z$. Total:
$(c^2 - s^2)Z - \sin\theta X = \cos\theta Z - \sin\theta X$.
\emph{Check} on $\ket0$: $R_Y(\theta)\ket0$ has $\langle Z\rangle = \cos\theta$, and $\braket{0|(\cos\theta Z - \sin\theta X)|0} = \cos\theta$. On $\ket+$:
$R_Y(\theta)\ket+$ has $\langle Z\rangle = -\sin\theta$, matching $-\sin\theta\braket{+|X|+}$."""),

    dict(id="B12", kind="mcq",
         q=r"""In the basis order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$ (qubit 0 first), the controlled-$R_Z(\theta)$ gate with
control qubit 0 and target qubit 1 --- \texttt{rz.ctrl(theta, q[0], q[1])} in CUDA-Q --- is""",
         opts=[r"$\operatorname{diag}\big(1,\ 1,\ e^{-i\theta/2},\ e^{i\theta/2}\big)$", r"$\operatorname{diag}\big(e^{-i\theta/2},\ e^{i\theta/2},\ 1,\ 1\big)$",
               r"$\operatorname{diag}\big(1,\ 1,\ 1,\ e^{i\theta}\big)$", r"$\operatorname{diag}\big(1,\ e^{-i\theta/2},\ 1,\ e^{i\theta/2}\big)$"],
         sol=r"""Controlled-$U$ $= \ket0\!\bra0\otimes I + \ket1\!\bra1\otimes U$ with $U = R_Z(\theta) = \operatorname{diag}(e^{-i\theta/2}, e^{i\theta/2})$.
For control $0$ ($\ket{00}, \ket{01}$) nothing happens; for control $1$ the target gets $e^{-i\theta/2}$ ($\ket{10}$) or $e^{i\theta/2}$ ($\ket{11}$).
So the matrix is $\operatorname{diag}(1, 1, e^{-i\theta/2}, e^{i\theta/2})$. Note that it is \emph{not} the controlled-phase
$\operatorname{diag}(1,1,1,e^{i\theta})$: the two differ by the phase $e^{-i\theta/2}$ on the control-$1$ subspace, which is a
\emph{relative} phase and is observable. This gate is the first half of the QCNN pooling block."""),

    dict(id="B13", kind="mcq",
         q=r"""A state-vector simulator stores every amplitude as a double-precision complex number (16 bytes). The memory
needed for one 30-qubit state is""",
         opts=[r"$16$ GiB (about $17.2$ GB)", r"$480$ bytes", r"$8$ GiB", r"$1$ GiB"],
         sol=r"""An $n$-qubit state has $2^n$ amplitudes: $2^{30} = 1\,073\,741\,824$. Memory $= 16 \times 2^{30}$ bytes
$= 2^{34}$ bytes $= 16$ GiB $\approx 17.2\times10^9$ bytes. Each additional qubit doubles it; single precision (8 bytes)
halves it. The 8-qubit default of \texttt{QML.ipynb} needs only $16\times2^8 = 4096$ bytes."""),

    dict(id="B14", kind="mcq",
         q=r"""A single qubit is prepared as $R_X(\theta_2)R_Y(\theta_1)\ket0$ (first $R_Y$, then $R_X$). The expectation value
$f(\theta_1, \theta_2) = \langle Z\rangle$ is""",
         opts=[r"$\cos\theta_1\cos\theta_2$", r"$\cos(\theta_1 + \theta_2)$",
               r"$\tfrac12(\cos\theta_1 + \cos\theta_2)$", r"$\cos\theta_1 - \sin\theta_1\sin\theta_2$"],
         sol=r"""Follow the Bloch vector $\vec r = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$. Start: $(0, 0, 1)$.
$R_Y(\theta_1)$ rotates about $y$: $\vec r = (\sin\theta_1, 0, \cos\theta_1)$.
$R_X(\theta_2)$ rotates about $x$: $(y, z) \to (y\cos\theta_2 - z\sin\theta_2,\ y\sin\theta_2 + z\cos\theta_2)$, so
$z' = 0\cdot\sin\theta_2 + \cos\theta_1\cos\theta_2$. Hence $f = \cos\theta_1\cos\theta_2$: rotations about \emph{different} axes do
not add angles. This is the simplest ``variational'' model: $f$ is trigonometric in each parameter separately."""),

    dict(id="B15", kind="match",
         q=r"""Match each gate (List-I) with a correct property (List-II). $S = \operatorname{diag}(1, i)$.""",
         match=([r"$H$", r"$S$", r"$R_Y(\pi)$", r"$R_X(\pi)$"],
                [r"maps $\ket0 \mapsto \ket1$ and $\ket1 \mapsto -\ket0$", r"equals $-iX$",
                 r"conjugation exchanges $X$ and $Z$: $(\cdot)Z(\cdot)^\dagger = X$", r"conjugation maps $X$ to $Y$: $(\cdot)X(\cdot)^\dagger = Y$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1",
               r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item $H$: $HZH = X$ and $HXH = Z$ (3).
\item $S$: $SXS^\dagger = \begin{psmallmatrix}1&0\\0&i\end{psmallmatrix}\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}\begin{psmallmatrix}1&0\\0&-i\end{psmallmatrix} = \begin{psmallmatrix}0&-i\\i&0\end{psmallmatrix} = Y$ (4).
\item $R_Y(\pi) = \cos\frac\pi2 I - i\sin\frac\pi2 Y = -iY = \begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$: $\ket0\mapsto\ket1$, $\ket1\mapsto-\ket0$ (1).
\item $R_X(\pi) = \cos\frac\pi2 I - i\sin\frac\pi2 X = -iX$ (2): an $X$ gate up to a global phase.
\end{itemize}
Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="B16", kind="match",
         q=r"""Match each state (List-I) with its Bloch vector $(\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$ (List-II).""",
         match=([r"$\ket+ = \frac{1}{\sqrt2}(\ket0 + \ket1)$", r"$\frac{1}{\sqrt2}(\ket0 + i\ket1)$",
                 r"$\cos\frac\pi8\ket0 + \sin\frac\pi8\ket1$", r"$\ket1$"],
                [r"$(0,\ 0,\ -1)$", r"$\big(\tfrac{1}{\sqrt2},\ 0,\ \tfrac{1}{\sqrt2}\big)$", r"$(1,\ 0,\ 0)$", r"$(0,\ 1,\ 0)$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1"],
         sol=r"""A state $\cos\frac\vartheta2\ket0 + e^{i\varphi}\sin\frac\vartheta2\ket1$ has Bloch vector
$(\sin\vartheta\cos\varphi,\ \sin\vartheta\sin\varphi,\ \cos\vartheta)$.
\begin{itemize}\itemsep0pt
\item $\ket+$: $\vartheta = \frac\pi2$, $\varphi = 0$: $(1, 0, 0)$ (3).
\item $\frac{1}{\sqrt2}(\ket0 + i\ket1)$: $\vartheta = \frac\pi2$, $\varphi = \frac\pi2$: $(0, 1, 0)$ (4).
\item $\cos\frac\pi8\ket0 + \sin\frac\pi8\ket1$: $\vartheta = \frac\pi4$, $\varphi = 0$: $(\frac{1}{\sqrt2}, 0, \frac{1}{\sqrt2})$ (2) --- this is the
notebook's angle encoding of a pixel value $x = \frac14$: $R_Y(\pi x)\ket0$.
\item $\ket1$: south pole $(0, 0, -1)$ (1).
\end{itemize}
Hence P$\to$3, Q$\to$4, R$\to$2, S$\to$1."""),

    dict(id="B17", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} For the state $R_X(\theta)\ket0$ one has $\langle Z\rangle = \cos\theta$, so $\theta$ and $-\theta$ give
identical statistics for a measurement in the computational basis.\\[1mm]
\textbf{Reason (R):} $R_X(\theta)\ket0$ and $R_X(-\theta)\ket0$ are the same quantum state.""",
         opts=AR_OPTIONS,
         sol=r"""$R_X(\theta)\ket0 = \cos\frac\theta2\ket0 - i\sin\frac\theta2\ket1$, so $P(0) = \cos^2\frac\theta2$, $P(1) = \sin^2\frac\theta2$ and
$\langle Z\rangle = \cos\theta$, an even function: \textbf{A is true}.
$R_X(-\theta)\ket0 = \cos\frac\theta2\ket0 + i\sin\frac\theta2\ket1$ differs by the \emph{relative} phase $-1$ on $\ket1$; the
Bloch vectors are $(0, \mp\sin\theta, \cos\theta)$, distinguishable by measuring $Y$. \textbf{R is false} (for $\sin\theta \neq 0$).
Answer: (C). (Same $Z$-statistics, different states: a $Z$ readout alone cannot tell them apart.)"""),

    dict(id="B18", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} A CNOT gate acting on a product state always produces an entangled state.\\[1mm]
\textbf{Statement II:} A pure two-qubit state $c_{00}\ket{00} + c_{01}\ket{01} + c_{10}\ket{10} + c_{11}\ket{11}$ is a product state if and
only if $c_{00}c_{11} - c_{01}c_{10} = 0$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} $\mathrm{CNOT}\ket{00} = \ket{00}$, $\mathrm{CNOT}\ket{1}\ket{+} = \ket1\ket+$ and
$\mathrm{CNOT}\ket+\ket+ = \ket+\ket+$ are all product states. CNOT entangles only for suitable inputs, e.g.\ $\ket+\ket0 \mapsto \ket{\Phi^+}$.
\textbf{II is true:} $(a_0\ket0 + a_1\ket1)\otimes(b_0\ket0 + b_1\ket1)$ has $c_{ab} = a_ab_b$, a rank-1 matrix with determinant 0;
conversely a $2\times2$ matrix with determinant 0 has rank $\le 1$ and factorises as an outer product $\mathbf a\mathbf b^{\mathsf T}$.
Answer: (D)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""The expectation values $\langle Z_1\rangle$ and $\langle Z_0Z_1\rangle$ of the final state are""",
         opts=[r"$\cos\theta$ and $1$", r"$\cos\theta$ and $\cos^2\theta$", r"$1$ and $\cos\theta$", r"$\cos^2(\theta/2)$ and $1$"],
         sol=r"""$R_Y(\theta)\ket0\otimes\ket0 = \cos\frac\theta2\ket{00} + \sin\frac\theta2\ket{10}$; the CNOT flips qubit 1 when qubit 0 is 1:
$\ket\psi = \cos\frac\theta2\ket{00} + \sin\frac\theta2\ket{11}$.
$\langle Z_1\rangle = \cos^2\frac\theta2 - \sin^2\frac\theta2 = \cos\theta$. Both components have even parity, so
$\langle Z_0Z_1\rangle = 1$ for every $\theta$: the outcomes of the two qubits always agree. (Product of the marginals would give
$\cos^2\theta$, which is wrong because the qubits are correlated.)"""),

    dict(id="B20", kind="para", passage="B",
         q=r"""The value of $\theta$ in $(0, \pi]$ at which the two qubits are \emph{maximally} entangled, and the purity
$\operatorname{tr}\rho_1^2$ of the reduced state of qubit 1 at that value, are""",
         opts=[r"$\theta = \pi/2$ and $\operatorname{tr}\rho_1^2 = \tfrac12$", r"$\theta = \pi$ and $\operatorname{tr}\rho_1^2 = \tfrac12$",
               r"$\theta = \pi/2$ and $\operatorname{tr}\rho_1^2 = 1$", r"$\theta = \pi/4$ and $\operatorname{tr}\rho_1^2 = \tfrac34$"],
         sol=r"""Tracing out qubit 0 from $\cos\frac\theta2\ket{00} + \sin\frac\theta2\ket{11}$ gives $\rho_1 = \operatorname{diag}(\cos^2\frac\theta2, \sin^2\frac\theta2)$
and purity $\operatorname{tr}\rho_1^2 = \cos^4\frac\theta2 + \sin^4\frac\theta2 = \frac{1 + \cos^2\theta}{2}$. It is minimal (most mixed,
most entangled) when $\cos\theta = 0$: $\theta = \frac\pi2$, giving $\frac12$ --- the Bell state $\ket{\Phi^+}$.
At $\theta = \pi$ the state is $\ket{11}$ (product, purity 1); at $\theta = \frac\pi4$ the purity is $\frac{1 + 1/2}{2} = \frac34$."""),
]
