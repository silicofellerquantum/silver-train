"""Section A - Mathematical and Quantum Foundations (probability, complex numbers, matrices, density matrices, pure
and mixed states, the Bloch sphere, partial trace and linear operators).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""A qubit should have been prepared in $\ket+ = \frac{1}{\sqrt2}(\ket0 + \ket1)$. State tomography of the real (noisy)
preparation gives the density matrix
\[ \rho = \begin{pmatrix}0.55 & 0.35\\ 0.35 & 0.45\end{pmatrix}. \]""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""The density matrix of the pure state $\ket\psi = \frac{1}{\sqrt2}(\ket0 + i\ket1)$ is""",
         opts=[r"$\frac12\begin{pmatrix}1 & -i\\ i & 1\end{pmatrix}$", r"$\frac12\begin{pmatrix}1 & i\\ -i & 1\end{pmatrix}$",
               r"$\frac12\begin{pmatrix}1 & 1\\ 1 & 1\end{pmatrix}$", r"$\frac12\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}$"],
         sol=r"""$\rho = \ket\psi\!\bra\psi$ with $\ket\psi = \frac{1}{\sqrt2}\begin{pmatrix}1\\ i\end{pmatrix}$ and $\bra\psi = \frac{1}{\sqrt2}(1,\ -i)$ (complex conjugate
transpose). Hence $\rho = \frac12\begin{pmatrix}1\cdot1 & 1\cdot(-i)\\ i\cdot1 & i\cdot(-i)\end{pmatrix} = \frac12\begin{pmatrix}1 & -i\\ i & 1\end{pmatrix}$.
\emph{Check}: Hermitian, trace 1, $\rho^2 = \rho$ (pure). Its Bloch vector is $(0, 1, 0)$."""),

    dict(id="A2", kind="mcq",
         q=r"""The purity $\operatorname{Tr}\rho^2$ of $\rho = \begin{pmatrix}0.7 & 0.3\\ 0.3 & 0.3\end{pmatrix}$ is""",
         opts=[r"$0.76$", r"$0.58$", r"$1$", r"$0.52$"],
         sol=r"""$\operatorname{Tr}\rho^2 = \sum_{ij}|\rho_{ij}|^2 = 0.7^2 + 0.3^2 + 0.3^2 + 0.3^2 = 0.49 + 0.09 + 0.09 + 0.09 = 0.76$.
\emph{Check} with the Bloch vector: $x = 2\operatorname{Re}\rho_{01} = 0.6$, $z = \rho_{00} - \rho_{11} = 0.4$, $|\vec r|^2 = 0.52$ and
$\operatorname{Tr}\rho^2 = \frac{1 + |\vec r|^2}{2} = 0.76$. Since $0.76 < 1$ the state is mixed."""),

    dict(id="A3", kind="mcq",
         q=r"""The Bloch vector $(r_x, r_y, r_z)$ of $\rho = \begin{pmatrix}0.8 & 0.2 - 0.1i\\ 0.2 + 0.1i & 0.2\end{pmatrix}$ is""",
         opts=[r"$(0.4,\ 0.2,\ 0.6)$", r"$(0.4,\ -0.2,\ 0.6)$", r"$(0.2,\ 0.1,\ 0.6)$", r"$(0.4,\ 0.2,\ 0.8)$"],
         sol=r"""Write $\rho = \frac12(I + r_xX + r_yY + r_zZ) = \frac12\begin{pmatrix}1 + r_z & r_x - ir_y\\ r_x + ir_y & 1 - r_z\end{pmatrix}$.
Then $r_z = \rho_{00} - \rho_{11} = 0.6$, and $\rho_{10} = \frac12(r_x + ir_y) = 0.2 + 0.1i$ gives $r_x = 0.4$, $r_y = 0.2$.
Equivalently $r_k = \operatorname{Tr}(\rho\sigma_k)$. $|\vec r|^2 = 0.56 < 1$: a valid mixed state."""),

    dict(id="A4", kind="mcq",
         q=r"""Which of the following matrices is \emph{not} a valid density matrix?""",
         opts=[r"$\begin{pmatrix}0.6 & 0.5\\ 0.5 & 0.4\end{pmatrix}$", r"$\begin{pmatrix}0.5 & 0.5\\ 0.5 & 0.5\end{pmatrix}$",
               r"$\begin{pmatrix}0.9 & 0\\ 0 & 0.1\end{pmatrix}$", r"$\begin{pmatrix}0.5 & -0.3i\\ 0.3i & 0.5\end{pmatrix}$"],
         sol=r"""A density matrix must be Hermitian, have trace 1 and be positive semi-definite. All four are Hermitian with trace 1; for a
$2\times2$ matrix PSD $\Leftrightarrow$ $\det\rho \ge 0$. $\det\begin{psmallmatrix}0.6&0.5\\0.5&0.4\end{psmallmatrix} = 0.24 - 0.25 = -0.01 < 0$:
one eigenvalue is negative (a ``probability'' below zero). The others have determinants $0$ (pure $\ket+$), $0.09$ and $0.25 - 0.09 = 0.16$.
Equivalently the Bloch vector of the first has length $\sqrt{1^2 + 0.2^2} > 1$."""),

    dict(id="A5", kind="mcq",
         q=r"""The operators $K_0 = \begin{pmatrix}1 & 0\\ 0 & 0.8\end{pmatrix}$ and $K_1 = \begin{pmatrix}0 & b\\ 0 & 0\end{pmatrix}$ ($b \ge 0$) form a
trace-preserving set of Kraus operators exactly when $b$ equals""",
         opts=[r"$0.6$", r"$0.36$", r"$0.8$", r"$0.4$"],
         sol=r"""Trace preservation requires $\sum_kK_k^\dagger K_k = I$. $K_0^\dagger K_0 = \operatorname{diag}(1, 0.64)$ and
$K_1^\dagger K_1 = \begin{psmallmatrix}0&0\\b&0\end{psmallmatrix}\begin{psmallmatrix}0&b\\0&0\end{psmallmatrix} = \operatorname{diag}(0, b^2)$.
So $0.64 + b^2 = 1$, $b = 0.6$. This is amplitude damping with $\gamma = 0.36$: $K_0 = \operatorname{diag}(1, \sqrt{1-\gamma})$, $K_1 = \sqrt\gamma\ket0\!\bra1$."""),

    dict(id="A6", kind="mcq",
         q=r"""A qubit is prepared in $\frac{1}{\sqrt2}(\ket0 + e^{i\varphi}\ket1)$ where the phase $\varphi$ is random and uniformly distributed in
$[-\frac\pi2, \frac\pi2]$. The average value of $\langle X\rangle$ is""",
         opts=[r"$2/\pi \approx 0.64$", r"$0$", r"$1/2$", r"$\pi/4 \approx 0.79$"],
         sol=r"""For fixed $\varphi$, $\langle X\rangle = 2\operatorname{Re}(\alpha^*\beta) = \cos\varphi$. Averaging over the uniform distribution:
$\frac1\pi\int_{-\pi/2}^{\pi/2}\cos\varphi\,d\varphi = \frac{1}{\pi}[\sin\varphi]_{-\pi/2}^{\pi/2} = \frac2\pi \approx 0.637$.
The averaged state $\bar\rho$ has its off-diagonal element reduced from $\frac12$ to $\frac1\pi$: random phases are the origin of
\emph{dephasing}. Populations ($\langle Z\rangle = 0$) are unchanged."""),

    dict(id="A7", kind="mcq",
         q=r"""A circuit has 20 gates; each fails independently with probability $p = 0.01$. The probability that \emph{at least one} gate fails is""",
         opts=[r"$0.182$", r"$0.200$", r"$0.818$", r"$0.165$"],
         sol=r"""$P(\text{no failure}) = (1 - p)^{20} = 0.99^{20} = e^{20\ln0.99} = e^{-0.2010} = 0.818$. Hence
$P(\ge1) = 1 - 0.818 = 0.182$. The linear estimate $20p = 0.200$ over-counts runs with two or more failures;
$P(\text{exactly one}) = 20(0.01)(0.99)^{19} = 0.165$."""),

    dict(id="A8", kind="mcq",
         q=r"""An outcome has true probability $0.9$. It is estimated from $10\,000$ shots. The standard error of the estimate is""",
         opts=[r"$0.003$", r"$0.009$", r"$0.0009$", r"$0.03$"],
         sol=r"""The count is binomial; the estimated frequency has $\mathrm{SE} = \sqrt{P(1 - P)/S} = \sqrt{0.9\cdot0.1/10\,000} = \sqrt{9\times10^{-6}} = 0.003$.
Shot noise is a \emph{statistical} error of the estimate, not a property of the quantum state: it shrinks like $1/\sqrt S$ even on a noiseless simulator."""),

    dict(id="A9", kind="mcq",
         q=r"""An $X$ gate is implemented as $R_X(\pi + \varepsilon)$ instead of $R_X(\pi)$, with $\varepsilon = 0.1$ rad. Applied to $\ket0$, the
probability of measuring $0$ (an error) is closest to""",
         opts=[r"$0.0025$", r"$0.05$", r"$0.1$", r"$0.01$"],
         sol=r"""$R_X(\theta)\ket0 = \cos\frac\theta2\ket0 - i\sin\frac\theta2\ket1$, so $P(0) = \cos^2\frac{\pi + \varepsilon}{2} = \sin^2\frac\varepsilon2$.
For $\varepsilon = 0.1$: $\sin^2(0.05) = 0.0025$. A small \emph{coherent} angle error gives an error probability quadratic in $\varepsilon$
(but such errors add up in amplitude when repeated --- see Section D)."""),

    dict(id="A10", kind="mcq",
         q=r"""The fidelity $F = \braket{+|\rho|+}$ between the target $\ket+$ and the mixed state $\rho = 0.8\ket+\!\bra+ + 0.2\,\frac I2$ is""",
         opts=[r"$0.9$", r"$0.8$", r"$0.95$", r"$0.6$"],
         sol=r"""By linearity, $\braket{+|\rho|+} = 0.8\braket{+|+}\braket{+|+} + 0.2\cdot\frac12\braket{+|+} = 0.8 + 0.1 = 0.9$.
The maximally mixed part still overlaps the target with probability $\frac12$. (For a pure target, $F = \braket{\psi|\rho|\psi}$; some texts use the square root.)"""),

    dict(id="A11", kind="mcq",
         q=r"""For the two-qubit state $\ket\psi = \sqrt{0.8}\ket{00} + \sqrt{0.2}\ket{11}$, the purity $\operatorname{Tr}\rho_A^2$ of the reduced state of qubit A is""",
         opts=[r"$0.68$", r"$1$", r"$0.5$", r"$0.8$"],
         sol=r"""$\rho_A = \operatorname{Tr}_B\ket\psi\!\bra\psi = 0.8\ket0\!\bra0 + 0.2\ket1\!\bra1$ (the cross terms $\ket{00}\!\bra{11}$ have orthogonal B parts and vanish).
$\operatorname{Tr}\rho_A^2 = 0.64 + 0.04 = 0.68$. The global state is pure, but its parts are mixed because they are entangled ---
the same mechanism by which a qubit entangled with its environment decoheres."""),

    dict(id="A12", kind="mcq",
         q=r"""The larger eigenvalue of $\rho = \begin{pmatrix}0.7 & 0.3\\ 0.3 & 0.3\end{pmatrix}$ is closest to""",
         opts=[r"$0.861$", r"$0.700$", r"$1.000$", r"$0.760$"],
         sol=r"""For a qubit, $\lambda_\pm = \frac{1 \pm |\vec r|}{2}$. Here $\vec r = (0.6, 0, 0.4)$, $|\vec r| = \sqrt{0.52} = 0.7211$, so
$\lambda_+ = 0.8606$, $\lambda_- = 0.1394$. \emph{Check}: $\lambda_+ + \lambda_- = 1 = \operatorname{Tr}\rho$ and $\lambda_+\lambda_- = 0.12 = \det\rho = 0.21 - 0.09$;
$\lambda_+^2 + \lambda_-^2 = 0.76 = \operatorname{Tr}\rho^2$."""),

    dict(id="A13", kind="mcq",
         q=r"""Two independent qubits are in the states $\rho_A = \operatorname{diag}(0.9, 0.1)$ and $\rho_B = \operatorname{diag}(0.8, 0.2)$. In the
state $\rho_A\otimes\rho_B$, the probability that both qubits give the \emph{same} outcome is""",
         opts=[r"$0.74$", r"$0.72$", r"$0.26$", r"$0.50$"],
         sol=r"""$\rho_A\otimes\rho_B = \operatorname{diag}(0.9\cdot0.8,\ 0.9\cdot0.2,\ 0.1\cdot0.8,\ 0.1\cdot0.2) = \operatorname{diag}(0.72, 0.18, 0.08, 0.02)$
in the order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$. Same outcome: $0.72 + 0.02 = 0.74$."""),

    dict(id="A14", kind="mcq",
         q=r"""A qubit with Bloch vector $(r_x, r_y, r_z)$ is acted on by $\rho \mapsto X\rho X$. Its new Bloch vector is""",
         opts=[r"$(r_x,\ -r_y,\ -r_z)$", r"$(-r_x,\ r_y,\ r_z)$", r"$(r_x,\ r_y,\ -r_z)$", r"$(-r_x,\ -r_y,\ -r_z)$"],
         sol=r"""$X\rho X = \frac12(I + r_xXXX + r_yXYX + r_zXZX)$. Since $X$ commutes with itself and anticommutes with $Y$ and $Z$:
$XXX = X$, $XYX = -Y$, $XZX = -Z$. So $(r_x, r_y, r_z) \mapsto (r_x, -r_y, -r_z)$: a rotation by $\pi$ about the $x$-axis.
The bit-flip channel $(1 - p)\rho + pX\rho X$ therefore keeps $r_x$ and multiplies $r_y, r_z$ by $1 - 2p$."""),

    dict(id="A15", kind="match",
         q=r"""Match each state (List-I) with its Bloch vector (List-II).""",
         match=([r"$\ket1$", r"$\frac I2$", r"$\frac12\begin{psmallmatrix}1 & -i\\ i & 1\end{psmallmatrix}$", r"$\operatorname{diag}(0.75,\ 0.25)$"],
                [r"$(0,\ 0,\ 0.5)$", r"$(0,\ 0,\ 0)$", r"$(0,\ 0,\ -1)$", r"$(0,\ 1,\ 0)$"]),
         opts=[r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1", r"P$\to$1, Q$\to$2, R$\to$4, S$\to$3",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4"],
         sol=r"""Use $r_z = \rho_{00} - \rho_{11}$, $r_x + ir_y = 2\rho_{10}$.
\begin{itemize}\itemsep0pt
\item $\ket1$: south pole $(0, 0, -1)$ (3).
\item $I/2$: the centre of the ball (2) --- maximally mixed.
\item $\rho_{10} = \frac i2$: $r_y = 1$, i.e.\ $(0, 1, 0)$ (4), the pure state $\frac{1}{\sqrt2}(\ket0 + i\ket1)$.
\item $\operatorname{diag}(0.75, 0.25)$: $r_z = 0.5$ (1), a mixed state on the $z$-axis.
\end{itemize}
Hence P$\to$3, Q$\to$2, R$\to$4, S$\to$1."""),

    dict(id="A16", kind="match",
         q=r"""Match each matrix (List-I) with its property (List-II).""",
         match=([r"$\begin{psmallmatrix}0 & 1\\ 1 & 0\end{psmallmatrix}$", r"$\frac12\begin{psmallmatrix}1 & 1\\ 1 & 1\end{psmallmatrix}$",
                 r"$\begin{psmallmatrix}1 & 0\\ 0 & i\end{psmallmatrix}$", r"$\begin{psmallmatrix}0 & 1\\ 0 & 0\end{psmallmatrix}$"],
                [r"unitary but not Hermitian", r"Hermitian and unitary",
                 r"neither Hermitian nor unitary (a Kraus operator of amplitude damping, up to a factor)", r"a rank-1 projector and a valid density matrix"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item $X$: $X^\dagger = X$ and $X^\dagger X = I$ (2); it has trace 0, so it is not a density matrix.
\item $\ket+\!\bra+$: Hermitian, $P^2 = P$, trace 1, PSD (4); not unitary ($\det = 0$).
\item $S = \operatorname{diag}(1, i)$: $S^\dagger S = I$ but $S^\dagger = \operatorname{diag}(1, -i) \neq S$ (1).
\item $\ket0\!\bra1$: nilpotent, $\neq$ its adjoint, not invertible (3); $\sqrt\gamma\ket0\!\bra1$ is the ``jump'' of amplitude damping.
\end{itemize}
Hence P$\to$2, Q$\to$4, R$\to$1, S$\to$3."""),

    dict(id="A17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} Every mixed single-qubit state lies strictly inside the Bloch ball ($|\vec r| < 1$).\\[1mm]
\textbf{Reason (R):} For $\rho = \frac12(I + \vec r\cdot\vec\sigma)$ one has $\operatorname{Tr}\rho^2 = \frac{1 + |\vec r|^2}{2}$, which equals 1 only when $|\vec r| = 1$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true}: $\rho^2 = \frac14(I + 2\vec r\cdot\vec\sigma + (\vec r\cdot\vec\sigma)^2) = \frac14((1 + |\vec r|^2)I + 2\vec r\cdot\vec\sigma)$ because
$(\vec r\cdot\vec\sigma)^2 = |\vec r|^2I$; the trace is $\frac{1 + |\vec r|^2}{2}$. \textbf{A is true}: ``mixed'' means $\operatorname{Tr}\rho^2 < 1$,
which by R is exactly $|\vec r| < 1$. R is the reason for A. Answer: (A)."""),

    dict(id="A18", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} Every $2\times2$ Hermitian matrix with trace 1 is a valid density matrix.\\[1mm]
\textbf{Statement II:} The diagonal entries of a density matrix in the computational basis are the probabilities of the outcomes of a
computational-basis measurement.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false}: positivity is also required; $\operatorname{diag}(1.2, -0.2)$ is Hermitian with trace 1 but has a negative eigenvalue.
\textbf{II is true}: $P(k) = \operatorname{Tr}(\ket k\!\bra k\rho) = \rho_{kk}$ (Born rule for mixed states). This is how the notebook turns an exact noisy
$\rho$ into outcome probabilities on the density-matrix path. Answer: (D)."""),

    dict(id="A19", kind="para", passage="A",
         q=r"""The Bloch vector and the purity of the prepared state are""",
         opts=[r"$(0.7,\ 0,\ 0.1)$ and $0.75$", r"$(0.35,\ 0,\ 0.1)$ and $0.75$", r"$(0.7,\ 0,\ 0.1)$ and $0.50$", r"$(0.7,\ 0,\ 0.55)$ and $0.85$"],
         sol=r"""$r_x = 2\operatorname{Re}\rho_{01} = 0.7$, $r_y = 0$ (real matrix), $r_z = 0.55 - 0.45 = 0.1$.
$\operatorname{Tr}\rho^2 = 0.55^2 + 0.45^2 + 2(0.35)^2 = 0.3025 + 0.2025 + 0.245 = 0.75 = \frac{1 + 0.49 + 0.01}{2}$. The vector is shorter than 1
(mixed: decoherence) and tilted towards $\ket0$ (a population imbalance, e.g.\ relaxation)."""),

    dict(id="A20", kind="para", passage="A",
         q=r"""The fidelity $\braket{+|\rho|+}$ of the preparation, and the probability that a computational-basis measurement gives $1$, are""",
         opts=[r"$0.85$ and $0.45$", r"$0.70$ and $0.45$", r"$0.85$ and $0.55$", r"$0.92$ and $0.45$"],
         sol=r"""$\braket{+|\rho|+} = \frac12(\rho_{00} + \rho_{01} + \rho_{10} + \rho_{11}) = \frac12(0.55 + 0.35 + 0.35 + 0.45) = 0.85 = \frac{1 + r_x}{2}$.
$P(1) = \rho_{11} = 0.45$. ($0.92 = \sqrt{0.85}$ is the root fidelity used in some texts; $0.70$ is $r_x$ itself.)"""),
]
