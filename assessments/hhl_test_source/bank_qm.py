"""Section B - Quantum computing foundations (qubits, superposition, entanglement, unitary and controlled gates,
measurement and post-selection, the quantum Fourier transform and phase estimation).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""A unitary $U$ has the eigenstate $\ket u$ with $U\ket u = e^{2\pi i\varphi}\ket u$, $\varphi = \tfrac14$. Phase estimation uses
$n_c = 2$ clock qubits $c_0c_1$ ($c_0$ = most significant bit). Both start in $\ket0$ and receive a Hadamard; $c_1$ (weight $2^0$) controls $U$
and $c_0$ (weight $2^1$) controls $U^2$, so that the state becomes $\frac12\sum_{k=0}^{3}e^{2\pi i\varphi k}\ket{k}\ket u$. Finally the inverse
QFT is applied to the clock and the clock is measured.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""A qubit is in the state $\ket\psi = \frac{1}{\sqrt5}\left(\ket0 + 2i\ket1\right)$. The probability of measuring $1$ is""",
         opts=[r"$\tfrac45$", r"$\tfrac25$", r"$\tfrac{2}{\sqrt5}$", r"$-\tfrac45$"],
         sol=r"""Born rule: $P(1) = |\braket{1|\psi}|^2 = \left|\frac{2i}{\sqrt5}\right|^2 = \frac{|2i|^2}{5} = \frac{4}{5}$.
Check the normalisation: $\frac{1}{5}(|1|^2 + |2i|^2) = \frac{1 + 4}{5} = 1$. The factor $i$ is a \emph{relative phase}: it does not change
the computational-basis probabilities but would change the outcome of a measurement in the $X$ basis."""),

    dict(id="B2", kind="mcq",
         q=r"""The default HHL circuit uses $7$ qubits. A state-vector simulator stores one complex double-precision number
($16$ bytes) per amplitude. The number of amplitudes and the memory of the state are""",
         opts=[r"$128$ amplitudes; $\ 2048$ bytes", r"$7$ amplitudes; $\ 112$ bytes", r"$49$ amplitudes; $\ 784$ bytes",
               r"$128$ amplitudes; $\ 1024$ bytes"],
         sol=r"""An $n$-qubit state has $2^n$ amplitudes: $2^7 = 128$. Memory $= 16 \times 2^7 = 2048$ bytes $= 2$ KiB.
In general $16\times2^n$ bytes: every extra qubit doubles the memory, $30$ qubits need $17.2$ GB and $40$ qubits $17.6$ TB.
(A real \texttt{complex128} has $8$ bytes for the real part and $8$ for the imaginary part; $1024$ bytes would be single precision.)"""),

    dict(id="B3", kind="mcq",
         q=r"""The product $HZH$ equals""",
         opts=[r"$X$", r"$Z$", r"$Y$", r"$-X$"],
         sol=r"""\[ HZH = \frac12\begin{pmatrix}1&1\\1&-1\end{pmatrix}\begin{pmatrix}1&0\\0&-1\end{pmatrix}\begin{pmatrix}1&1\\1&-1\end{pmatrix}
         = \frac12\begin{pmatrix}1&-1\\1&1\end{pmatrix}\begin{pmatrix}1&1\\1&-1\end{pmatrix} = \frac12\begin{pmatrix}0&2\\2&0\end{pmatrix} = X. \]
The Hadamard exchanges the $Z$ and $X$ bases ($H\ket0 = \ket+$, $H\ket1 = \ket-$), so it maps $Z$ to $X$ and vice versa ($HXH = Z$)."""),

    dict(id="B4", kind="match",
         q=r"""Match each rotation applied to $\ket0$ (List-I) with the resulting state (List-II).""",
         match=([r"$R_Y(\pi)\ket0$", r"$R_Y(\pi/2)\ket0$", r"$R_Y(-\pi/2)\ket0$", r"$R_Y(\pi/3)\ket0$"],
                [r"$\ket+$", r"$\frac{\sqrt3}{2}\ket0 + \frac12\ket1$", r"$\ket1$", r"$\ket-$"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2",
               r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4"],
         sol=r"""$R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$ (first column of the matrix in the conventions).
\begin{itemize}\itemsep0pt
\item P: $\theta = \pi$: $\cos\frac\pi2\ket0 + \sin\frac\pi2\ket1 = \ket1$ (3).
\item Q: $\theta = \frac\pi2$: $\frac{1}{\sqrt2}(\ket0 + \ket1) = \ket+$ (1).
\item R: $\theta = -\frac\pi2$: $\cos(-\frac\pi4) = \frac{1}{\sqrt2}$, $\sin(-\frac\pi4) = -\frac{1}{\sqrt2}$, giving $\ket-$ (4).
\item S: $\theta = \frac\pi3$: $\cos\frac\pi6\ket0 + \sin\frac\pi6\ket1 = \frac{\sqrt3}{2}\ket0 + \frac12\ket1$ (2).
\end{itemize}
In HHL the angle $\theta = 2\arcsin(C/\lambda)$ is chosen so that the amplitude of $\ket1$ is exactly $\sin\frac\theta2 = C/\lambda$."""),

    dict(id="B5", kind="mcq",
         q=r"""Which of the following two-qubit states is \emph{entangled}?""",
         opts=[r"$\tfrac12(\ket{00} + \ket{01} + \ket{10} - \ket{11})$", r"$\tfrac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$",
               r"$\tfrac12(\ket{00} - \ket{01} + \ket{10} - \ket{11})$", r"$\tfrac12(\ket{00} + \ket{01} - \ket{10} - \ket{11})$"],
         sol=r"""Write $\ket\psi = \sum a_{ij}\ket{ij}$. It is a product state iff the coefficient matrix $\begin{psmallmatrix}a_{00}&a_{01}\\a_{10}&a_{11}\end{psmallmatrix}$
has rank 1, i.e.\ $a_{00}a_{11} - a_{01}a_{10} = 0$.
\begin{itemize}\itemsep0pt
\item $(1, 1, 1, -1)/2$: $1\cdot(-1) - 1\cdot1 = -2 \neq 0$: \textbf{entangled}.
\item $(1, 1, 1, 1)/2 = \ket+\ket+$; $(1, -1, 1, -1)/2 = \ket+\ket-$; $(1, 1, -1, -1)/2 = \ket-\ket+$: determinant $0$, product states.
\end{itemize}
In HHL, phase estimation entangles the clock with the system ($\sum_j\beta_j\ket{u_j}\ket{k_j}$); uncomputation removes this entanglement again."""),

    dict(id="B6", kind="mcq",
         q=r"""The control qubit is in $\ket+$ and the target in an eigenstate $\ket u$ of $U$ with $U\ket u = e^{i\phi}\ket u$. After a controlled-$U$
the joint state is""",
         opts=[r"$\frac{1}{\sqrt2}\left(\ket0 + e^{i\phi}\ket1\right)\otimes\ket u$", r"$\ket+\otimes e^{i\phi}\ket u$ (the phase stays on the target)",
               r"$\frac{1}{\sqrt2}\left(e^{i\phi}\ket0 + \ket1\right)\otimes\ket u$", r"$\frac{1}{\sqrt2}\left(\ket0\ket u + \ket1 U^\dagger\ket u\right)$, entangled"],
         sol=r"""Linearity: $\text{C-}U\,\frac{1}{\sqrt2}(\ket0 + \ket1)\ket u = \frac{1}{\sqrt2}\left(\ket0\ket u + \ket1U\ket u\right) = \frac{1}{\sqrt2}\left(\ket0\ket u + e^{i\phi}\ket1\ket u\right)$.
Because $U\ket u$ is $\ket u$ times a number, the number can be attached to the control: $\frac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)\otimes\ket u$.
This is \textbf{phase kickback}: the eigenvalue becomes a \emph{relative} phase of the control, the target is unchanged and no entanglement
is created. Phase estimation reads these kicked-back phases with the inverse QFT."""),

    dict(id="B7", kind="mcq",
         q=r"""The inverse QFT on two qubits is applied to $\ket\psi = \frac12\sum_{y=0}^{3}e^{2\pi i\,3y/4}\ket y$. The result is""",
         opts=[r"$\ket{11}$", r"$\ket{01}$", r"$\ket{10}$", r"$\frac12\sum_{y}\ket y$"],
         sol=r"""By definition $\mathrm{QFT}\ket x = \frac12\sum_y e^{2\pi i\,xy/4}\ket y$; for $x = 3$ this is exactly $\ket\psi$. Hence
$\mathrm{QFT}^{-1}\ket\psi = \ket3 = \ket{11}$. Explicitly the amplitudes are $\frac12(1, -i, -1, i)$: the phase advances by $3\cdot\frac{2\pi}{4}$ per step.
(The equivalent reading $e^{2\pi i\,3y/4} = e^{-2\pi i\,y/4}$ means ``$-1$'', which is $3$ modulo $4$ --- not $1$.)"""),

    dict(id="B8", kind="mcq",
         q=r"""Phase estimation with $n_c = 3$ clock qubits is run on an eigenstate with $\varphi = 0.375$. The measured clock bit string
($c_0$ = most significant bit) is""",
         opts=[r"$011$ with probability $1$", r"$110$ with probability $1$", r"$011$ with probability $\approx 0.81$",
               r"$100$ with probability $1$"],
         sol=r"""$\varphi = 0.375 = \tfrac38 = 0.011_2$ is an exact 3-bit binary fraction. After the controlled powers the clock holds
$\frac{1}{\sqrt8}\sum_k e^{2\pi i\,3k/8}\ket k = \mathrm{QFT}\ket3$, so the inverse QFT returns $\ket{3} = \ket{011}$ with certainty and
$\tilde\varphi = 3/8$. Probabilities below $1$ (like the $4/\pi^2 \approx 0.405$ bound) occur only when $\varphi\,2^{n_c}$ is \emph{not} an integer.
$110$ is the bit-reversed string (a classic qubit-ordering slip); $100 = 4$ would mean $\varphi = 0.5$."""),

    dict(id="B9", kind="mcq",
         q=r"""Phase estimation with $n_c = 3$ clock qubits is run on an eigenstate with $\varphi = \tfrac13$. The most likely outcome $k$ and its
probability are""",
         opts=[r"$k = 3$; $\ \approx 0.69$", r"$k = 3$; $\ 1$", r"$k = 2$; $\ \approx 0.69$", r"$k = 3$; $\ \approx 0.41$"],
         sol=r"""$\varphi\,2^{n_c} = 8/3 \approx 2.667$ is not an integer, so the outcome is random, peaked at the nearest integer $k = 3$.
For outcome $k$, with $\delta = \varphi - k/8$:
\[ P(k) = \left|\frac18\sum_{m=0}^{7}e^{2\pi i\,m\delta}\right|^2 = \frac{\sin^2(8\pi\delta)}{64\sin^2(\pi\delta)}. \]
For $k = 3$: $\delta = \frac13 - \frac38 = -\frac1{24}$, $\sin^2(\pi/3) = 0.75$, $64\sin^2(\pi/24) = 64\times0.017037 = 1.0904$, so $P(3) \approx 0.688$.
(For $k = 2$: $\delta = \frac1{12}$, $P(2) = \frac{\sin^2(2\pi/3)}{64\sin^2(\pi/12)} = \frac{0.75}{4.287} \approx 0.175$.) The bound $4/\pi^2 \approx 0.405$
is only a worst-case guarantee. In HHL such rounding of $\lambda t2^{n_c}/2\pi$ is the main source of error."""),

    dict(id="B10", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} A unitary gate preserves the norm of every state vector.\\[1mm]
\textbf{Statement II:} A projective measurement is reversible: the pre-measurement state can always be recovered from the post-measurement state.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $\lVert U\psi\rVert^2 = \braket{\psi|U^\dagger U|\psi} = \braket{\psi|\psi}$.
\textbf{II is false:} measurement maps $\sum_k a_k\ket k$ to a single $\ket k$; all the other amplitudes are lost and different states
(e.g.\ $\ket+$ and $\ket-$) give the same post-measurement states. This irreversibility is why HHL's ancilla measurement must be the
\emph{last} step on the system, and why a failed run (ancilla $= 0$) cannot be repaired --- it must be repeated. Answer: (C)."""),

    dict(id="B11", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} The states $\ket\psi$ and $e^{i\alpha}\ket\psi$ give identical statistics for every measurement.\\[1mm]
\textbf{Reason (R):} The relative phase between the components of a superposition affects interference and hence measurement results.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} every probability has the form $|\braket{m|\psi}|^2$ and $|e^{i\alpha}\braket{m|\psi}|^2 = |\braket{m|\psi}|^2$ --- a \emph{global} phase is unobservable.
\textbf{R is true:} e.g.\ $\ket+$ and $\ket-$ differ only by a relative phase, yet $H$ maps them to $\ket0$ and $\ket1$.
R does not explain A: A is about a phase common to all components, R about phases \emph{between} components. Answer: (B).
(In HHL, the sign of $C/\lambda_j$ is a relative phase between eigencomponents and therefore matters.)"""),

    dict(id="B12", kind="mcq",
         q=r"""A controlled-$R_Y(\theta)$ with $\theta = 2\arcsin\tfrac12$ acts on control $\ket1$ and target (ancilla) $\ket0$. The probability of then
measuring the ancilla as $1$ is""",
         opts=[r"$\tfrac14$", r"$\tfrac12$", r"$\tfrac34$", r"$\tfrac1{16}$"],
         sol=r"""$\theta = 2\arcsin\frac12 = 2\cdot\frac\pi6 = \frac\pi3$. The control is $\ket1$, so the rotation acts:
$R_Y(\pi/3)\ket0 = \cos\frac\pi6\ket0 + \sin\frac\pi6\ket1 = \frac{\sqrt3}{2}\ket0 + \frac12\ket1$. Hence $P(1) = (\frac12)^2 = \frac14$.
In HHL this is the eigenvalue inversion for $C/\lambda = \frac12$ (e.g.\ $C = 1$, $\lambda = 2$): the amplitude is $C/\lambda$ and the success
probability of that component is $(C/\lambda)^2$."""),

    dict(id="B13", kind="mcq",
         q=r"""A 3-qubit clock register $c_0c_1c_2$ ($c_0$ = most significant bit) controls a multi-controlled $R_Y$ that fires only when all controls are $1$.
To make it act only on the clock value $k = 2$, the code places $X$ gates before and after the rotation. Which $X$ gates are needed?""",
         opts=[r"$X$ on $c_0$ and $c_2$, before and after (4 gates)", r"$X$ on $c_1$ only, before and after (2 gates)",
               r"$X$ on $c_1$ and $c_2$, before and after (4 gates)", r"none: $k = 2$ is selected by the rotation angle"],
         sol=r"""$k = 2 = 010_2$: $c_0 = 0$, $c_1 = 1$, $c_2 = 0$. A multi-controlled gate fires on $\ket{111}$, so every control that must be $0$
is flipped by $X$ first: $X$ on $c_0$ and $c_2$ maps $\ket{010} \to \ket{111}$. After the rotation the same $X$ gates restore the clock.
Total $2\times2 = 4$ $X$ gates. This is exactly the \texttt{zero\_bits} loop in \texttt{HHL.ipynb}: \texttt{kernel.x(clock[i])} for every bit $i$ of $k$ that is $0$."""),

    dict(id="B14", kind="mcq",
         q=r"""An ancilla $a$ and a system register are in $\frac12\ket0_a\ket{\psi_0} + \frac{\sqrt3}{2}\ket1_a\ket{\psi_1}$ ($\ket{\psi_0}, \ket{\psi_1}$ normalised).
The ancilla is measured and the result $1$ is kept (post-selection). The probability of this and the resulting system state are""",
         opts=[r"$\tfrac34$; $\ \ket{\psi_1}$", r"$\tfrac{\sqrt3}{2}$; $\ \ket{\psi_1}$", r"$\tfrac34$; $\ \frac{1}{\sqrt2}(\ket{\psi_0} + \ket{\psi_1})$",
               r"$\tfrac14$; $\ \ket{\psi_0}$"],
         sol=r"""$P(a = 1) = \left\lVert\frac{\sqrt3}{2}\ket{\psi_1}\right\rVert^2 = \frac34$. After the measurement only the $a = 1$ branch survives and is renormalised:
$\frac{\frac{\sqrt3}{2}\ket{\psi_1}}{\sqrt{3/4}} = \ket{\psi_1}$. HHL works in exactly this way: the ancilla-$1$ branch holds $\propto A^{-1}\ket b$, the ancilla-$0$
branch is discarded, and the probability of keeping it is the success probability."""),

    dict(id="B15", kind="mcq",
         q=r"""A circuit applies $U_1$, then $U_2$, then $U_3$ (all unitary). The circuit that undoes it applies""",
         opts=[r"$U_3^\dagger$, then $U_2^\dagger$, then $U_1^\dagger$", r"$U_1^\dagger$, then $U_2^\dagger$, then $U_3^\dagger$",
               r"$U_3$, then $U_2$, then $U_1$", r"$U_1^\dagger U_2^\dagger U_3^\dagger$ applied as a single gate in the same order as the original"],
         sol=r"""The circuit implements the product $W = U_3U_2U_1$ (the first gate acts first, so it stands on the right).
$W^{-1} = W^\dagger = U_1^\dagger U_2^\dagger U_3^\dagger$; read right to left this means: apply $U_3^\dagger$ first, then $U_2^\dagger$, then $U_1^\dagger$.
Reverse the order \emph{and} invert every gate. HHL's uncomputation does exactly this with phase estimation: the QFT (inverse of the inverse QFT)
first, then the controlled $U^{-2^j}$ in reverse order, then the Hadamards."""),

    dict(id="B16", kind="match",
         q=r"""The textbook QFT on $n = 4$ qubits (as used, inverted, on the clock register of the default HHL circuit). Match each quantity (List-I)
with its value (List-II).""",
         match=([r"number of Hadamard gates", r"number of controlled-phase gates", r"number of SWAP gates",
                 r"magnitude of the smallest controlled-phase angle"],
                [r"$2$", r"$\pi/8$", r"$6$", r"$4$"]),
         opts=[r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: one Hadamard per qubit: $4$ (4).
\item Q: $n(n-1)/2 = 4\cdot3/2 = 6$ controlled phases (3).
\item R: $\lfloor n/2\rfloor = 2$ SWAPs to reverse the qubit order (1).
\item S: angles $2\pi/2^k$ for $k = 2, \dots, n$; the smallest is $2\pi/2^4 = \pi/8$ (2).
\end{itemize}
Consistency with \texttt{HHL.ipynb}: the recorded statistics show $12$ \texttt{cr1} and $4$ \texttt{swap} gates (inverse QFT + QFT for uncomputation:
$2\times6$ and $2\times2$) and $16$ \texttt{h} gates ($4$ initial + $4$ + $4$ in the two QFTs + $4$ final)."""),

    dict(id="B17", kind="mcq",
         q=r"""A Hadamard is applied to each of four qubits in $\ket{0000}$. The amplitude of $\ket{1011}$ is""",
         opts=[r"$\tfrac14$", r"$\tfrac1{16}$", r"$-\tfrac14$", r"$\tfrac{1}{\sqrt8}$"],
         sol=r"""$H\ket0 = \frac{1}{\sqrt2}(\ket0 + \ket1)$ for each qubit, so $H^{\otimes4}\ket{0000} = \frac{1}{(\sqrt2)^4}\sum_{y=0}^{15}\ket y = \frac14\sum_y\ket y$.
Every one of the $16$ basis states has amplitude $\frac14$ (probability $\frac1{16}$), with a $+$ sign because the input is $\ket{0\dots0}$
(signs $(-1)^{x\cdot y}$ appear only for inputs $x \neq 0$). This is the first step of phase estimation on the HHL clock register."""),

    dict(id="B18", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} Every two-qubit pure state can be written as $\ket a\otimes\ket b$ for suitable single-qubit states $\ket a$, $\ket b$.\\[1mm]
\textbf{Statement II:} For the Bell state $\frac{1}{\sqrt2}(\ket{00} + \ket{11})$ the reduced state of each qubit is the maximally mixed state $\frac12 I$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} product states form a small subset; the Bell state is a counter-example (its coefficient matrix $\frac{1}{\sqrt2}I$ has rank 2).
\textbf{II is true:} tracing out the second qubit gives $\rho_A = \frac12(\ket0\!\bra0 + \ket1\!\bra1) = \frac12I$; a measurement of one qubit alone
gives $0$ or $1$ with probability $\frac12$ each.
In HHL, if the clock were \emph{not} uncomputed, the reduced state of the system register would be mixed like this instead of the pure $\ket x$.
Answer: (D)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""Before the inverse QFT, the amplitude of the clock state $\ket{11}$ (with $\ket u$ factored out) is""",
         opts=[r"$-\tfrac i2$", r"$\tfrac i2$", r"$-\tfrac12$", r"$\tfrac12$"],
         sol=r"""The clock amplitudes are $\frac12e^{2\pi i\varphi k}$ with $\varphi = \frac14$, i.e.\ $\frac12 i^k$. For $\ket{11}$, $k = 3$:
$\frac12 i^3 = \frac12(-i) = -\frac i2$. The four amplitudes are $\frac12(1, i, -1, -i)$: each controlled power adds the phase $2\pi\varphi\cdot 2^j$
to the clock qubit of weight $2^j$ (phase kickback, B6)."""),

    dict(id="B20", kind="para", passage="B",
         q=r"""After the inverse QFT the clock measurement gives""",
         opts=[r"$01$ with probability $1$, so $\tilde\varphi = \tfrac14$", r"$10$ with probability $1$, so $\tilde\varphi = \tfrac12$",
               r"$01$ or $11$ with probability $\tfrac12$ each", r"$11$ with probability $1$, so $\tilde\varphi = \tfrac34$"],
         sol=r"""The clock state $\frac12\sum_k e^{2\pi i\,k/4}\ket k$ is $\mathrm{QFT}\ket1$ (definition with $N = 4$, $x = 1$). Hence the inverse QFT gives $\ket1 = \ket{01}$
with certainty, and $\tilde\varphi = 1/2^{n_c} = 1/4$ --- exact, because $\varphi\cdot2^{n_c} = 1$ is an integer. In HHL the same mechanism writes
$k = \lambda t2^{n_c}/2\pi$ into the clock, one value per eigencomponent of $\ket b$."""),
]
