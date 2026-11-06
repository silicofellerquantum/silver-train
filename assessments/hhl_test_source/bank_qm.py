"""Section B - Quantum computing foundations for HHL (states, measurement, post-selection, controlled gates, phase
kickback, QFT and inverse QFT, phase estimation, two's complement, rotations, memory).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""Quantum phase estimation with $t = 2$ counting qubits is applied to an eigenstate $\ket u$ of a unitary $U$, $U\ket u = e^{2\pi i\varphi}\ket u$. The counting
register ends in $\sum_y\alpha_y\ket y$ with $\alpha_y = \frac14\sum_{j=0}^{3}e^{2\pi i j(\varphi - y/4)}$, and the outcome $y$ estimates $\varphi \approx y/4$.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""Phase estimation with 4 counting qubits is applied to an eigenstate of $U$ with eigenvalue $e^{2\pi i\cdot3/16}$. The measured counting register is""",
         opts=[r"$\ket{0011}$ with probability 1", r"$\ket{1100}$ with probability 1", r"$\ket{0011}$ with probability $\tfrac{3}{16}$", r"$\ket{0011}$ or $\ket{0100}$, each with probability $\tfrac12$"],
         sol=r"""When $\varphi = k/2^t$ is exactly representable, the controlled powers produce $\frac{1}{\sqrt{16}}\sum_je^{2\pi i jk/16}\ket j = \mathrm{QFT}\ket k$, and the inverse QFT returns
exactly $\ket k$. Here $k = 3 = 0011_2$ (most significant bit first): probability 1. $\ket{1100}$ is the bit-reversed string (the swaps of the QFT fix
the order). In the notebook's default every eigenvalue maps to such an exact clock value, which is why its fidelity is 1."""),

    dict(id="B2", kind="mcq",
         q=r"""In HHL, phase estimation uses $U = e^{iAt}$, so an eigenvalue $\lambda$ gives the clock value $k = \lambda t\,2^{n_c}/(2\pi)$. With $n_c = 4$ and $t = \pi/8$ (the notebook's
default), the eigenvalue $\lambda = 4$ gives""",
         opts=[r"$k = 4$, i.e.\ $\ket{0100}$", r"$k = 8$", r"$k = 2$", r"$k = 4\pi$"],
         sol=r"""$e^{i\lambda t} = e^{2\pi i\varphi}$ with $\varphi = \lambda t/(2\pi)$; the clock reads $k = \varphi\,2^{n_c}$. So $k = 4\cdot\frac{\pi}{8}\cdot\frac{16}{2\pi} = 4$. With $t = \pi/8$ every eigenvalue
$\lambda$ is read as $k = \lambda$ (the \texttt{"auto"} time maps $|\lambda|_{\min} = 1$ to clock value 1), so the default eigenvalues $1, 2, 3, 4$ give the exact clock values
$1, 2, 3, 4$ printed by the notebook."""),

    dict(id="B3", kind="match",
         q=r"""Match each step of quantum phase estimation (List-I) with its effect (List-II).""",
         match=([r"Hadamards on the counting register", r"controlled $U^{2^j}$ gates", r"inverse QFT on the counting register", r"measurement of the counting register"],
                [r"converts the pattern of phases $e^{2\pi i2^j\varphi}$ into the binary number $\approx 2^t\varphi$",
                 r"returns one estimate $y$ of $2^t\varphi$",
                 r"creates the uniform superposition $\frac{1}{\sqrt{2^t}}\sum_j\ket j$",
                 r"write the phase $e^{2\pi i2^j\varphi}$ onto counting qubit $j$ (phase kickback)"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: Hadamards give the uniform superposition (3).
\item Q: controlled $U^{2^j}$ on the eigenstate kicks the phase back to the control (4).
\item R: the counting register is then $\mathrm{QFT}\ket k$ (exactly or approximately); the inverse QFT turns it into $\ket k$ (1).
\item S: measurement gives one estimate (2). In HHL this measurement is \emph{not} done: the clock register stays coherent for the eigenvalue inversion.
\end{itemize}"""),

    dict(id="B4", kind="mcq",
         q=r"""A control qubit in $\ket+$ and a target in an eigenstate $\ket u$ of $U$ ($U\ket u = e^{i\phi}\ket u$) pass through a controlled-$U$. The result is""",
         opts=[r"$\tfrac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)\otimes\ket u$", r"$\ket+\otimes e^{i\phi}\ket u$", r"$\tfrac{1}{\sqrt2}(\ket0\ket u + \ket1\ket{u'})$ with $\ket{u'}$ orthogonal to $\ket u$", r"$\tfrac{1}{\sqrt2}(e^{i\phi}\ket0 + \ket1)\otimes\ket u$"],
         sol=r"""$\tfrac{1}{\sqrt2}(\ket0\ket u + \ket1U\ket u) = \tfrac{1}{\sqrt2}(\ket0\ket u + e^{i\phi}\ket1\ket u) = \tfrac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)\ket u$. The target is unchanged; the phase
appears as a \emph{relative} phase on the control (phase kickback). $\ket+\otimes e^{i\phi}\ket u$ would be a global phase --- unobservable. This is how
phase estimation extracts eigenvalues without disturbing the eigenvector."""),

    dict(id="B5", kind="mcq",
         q=r"""The inverse QFT on two qubits is applied to $\tfrac12\big(\ket{00} + i\ket{01} - \ket{10} - i\ket{11}\big)$. The result is""",
         opts=[r"$\ket{01}$", r"$\ket{11}$", r"$\ket{10}$", r"$\tfrac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$"],
         sol=r"""$\mathrm{QFT}\ket k = \frac12\sum_y i^{ky}\ket y$ ($\omega = e^{2\pi i/4} = i$). For $k = 1$: $\frac12(1, i, i^2, i^3) = \frac12(1, i, -1, -i)$ --- exactly the given state. So
$\mathrm{QFT}^{-1}$ returns $\ket1 = \ket{01}$. ($k = 3$ would give $\frac12(1, -i, -1, i)$.) This is the last step of phase estimation, the link between HHL and the QFT lecture."""),

    dict(id="B6", kind="mcq",
         q=r"""The notebook builds the inverse QFT (in phase estimation) and the QFT (in the uncomputation) on $n_c = 4$ clock qubits from controlled phase gates
\texttt{cr1} and \texttt{swap}s. How many \texttt{cr1} and \texttt{swap} gates does the whole HHL circuit contain?""",
         opts=[r"$12$ and $4$", r"$6$ and $2$", r"$16$ and $4$", r"$12$ and $8$"],
         sol=r"""One QFT on $n$ qubits has $\binom n2 = n(n-1)/2$ controlled rotations and $\lfloor n/2\rfloor$ swaps: for $n = 4$, $6$ and $2$. HHL needs the inverse QFT once and the QFT once
(uncomputation): $12$ \texttt{cr1} and $4$ \texttt{swap} --- exactly the recorded statistics (\texttt{cr1: 12, swap: 4})."""),

    dict(id="B7", kind="mcq",
         q=r"""An ancilla in $\ket0$ is rotated by $R_Y(\theta) = e^{-i\theta Y/2}$ with $\theta = 2\arcsin(\tfrac12)$. The probability of then measuring $\ket1$ is""",
         opts=[r"$\tfrac14$", r"$\tfrac12$", r"$\tfrac34$", r"$\sin^2(\tfrac12) \approx 0.23$"],
         sol=r"""$R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$. With $\theta = 2\arcsin c$: $\sin\frac\theta2 = c = \tfrac12$, so the amplitude of $\ket1$ is $\tfrac12$ and $P(1) = \tfrac14$; $\theta = \pi/3$.
This is HHL's eigenvalue inversion: $c = C/\lambda$ (here $C = 1$, $\lambda = 2$) puts amplitude $C/\lambda$ on the ancilla's $\ket1$."""),

    dict(id="B8", kind="mcq",
         q=r"""To apply a rotation only when the 4-qubit clock register is in $\ket k$, the notebook surrounds a fully controlled \texttt{cry} with $X$ gates on the clock qubits whose bit of $k$ is
0 (clock[0] = most significant bit). For $k = 5$, the $X$ gates act on""",
         opts=[r"clock[0] and clock[2]", r"clock[1] and clock[3]", r"clock[0] only", r"clock[2] and clock[3]"],
         sol=r"""$5 = 0101_2$: bits (clock[0], clock[1], clock[2], clock[3]) $= (0, 1, 0, 1)$. The zero bits are clock[0] and clock[2]; $X$ on them maps $\ket{0101}$ to $\ket{1111}$, which triggers the
all-ones control, and the second pair of $X$ gates restores the register. Summed over $k = 1, \dots, 15$ these ``polarity'' gates account for the
$56$ \texttt{x} gates in the recorded statistics (twice the 28 zero bits)."""),

    dict(id="B9", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} A controlled-$U^{2^j}$ can be implemented as $2^j$ consecutive controlled-$U$ gates.\\[1mm]
\textbf{Statement II:} For $U = e^{iAt}$, $U^{2^j} = e^{iA\,t\,2^j}$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} the control condition is the same for every factor, so $(cU)^{2^j} = c(U^{2^j})$ --- but the cost doubles with $j$, which is why efficient Hamiltonian simulation
is essential. \textbf{II is true:} $A$ commutes with itself, so $(e^{iAt})^m = e^{iAmt}$. The notebook registers each $U^{2^j}$ directly as a matrix
(\texttt{np.linalg.matrix\_power(U, 2**j)}). Answer: (A)."""),

    dict(id="B10", kind="match",
         q=r"""HHL reads the 4-qubit clock as a signed (two's complement) number: values $k \ge 8$ mean $k - 16$. Match each clock value (List-I) with the signed value
(List-II).""",
         match=([r"$k = 3$", r"$k = 8$", r"$k = 12$", r"$k = 15$"],
                [r"$-1$", r"$-4$", r"$3$", r"$-8$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3"],
         sol=r"""Rule: signed $= k$ for $k < 8$, $k - 16$ for $k \ge 8$.
\begin{itemize}\itemsep0pt
\item P: $3 < 8$: $3$ (3). \quad Q: $8 - 16 = -8$ (4), the most negative value. \quad R: $12 - 16 = -4$ (2). \quad S: $15 - 16 = -1$ (1).
\end{itemize}
A negative eigenvalue $\lambda$ gives $e^{i\lambda t}$ with phase $1 + \lambda t/2\pi$ (mod 1), near the top of the clock range; the notebook uses
\texttt{signed = k\_val - 2**nc if k\_val >= 2**(nc - 1) else k\_val}."""),

    dict(id="B11", kind="mcq",
         q=r"""A state $\ket\Psi = \sqrt{1 - p}\,\ket{\psi_0}\ket0 + \sqrt p\,\ket{\psi_1}\ket1$ (normalised $\ket{\psi_0}$, $\ket{\psi_1}$) is measured on the last qubit and the result 1 is kept (post-selection).
Which statement is correct?""",
         opts=[r"The outcome 1 occurs with probability $p$ and leaves the rest in $\ket{\psi_1}$",
               r"The outcome 1 occurs with probability $\sqrt p$",
               r"The remaining state is $\sqrt p\,\ket{\psi_1}$, not normalised",
               r"Post-selection succeeds with probability 1 if $p > 0$"],
         sol=r"""Born rule: $P(1) = \|\sqrt p\,\ket{\psi_1}\|^2 = p$; the post-measurement state is renormalised: $\ket{\psi_1}$. On average $1/p$ repetitions are needed. In HHL $p = P_{\text{success}} = C^2\|A^{-1}b\|^2$
and $\ket{\psi_1} = \ket x$; the notebook recorded $p = 0.3559$ (7086 accepted of 20000 shots)."""),

    dict(id="B12", kind="mcq",
         q=r"""After phase estimation in HHL the system and clock registers are in $\sum_j\beta_j\ket{u_j}\ket{k_j}$ (distinct $k_j$). When is this a product state of the two registers?""",
         opts=[r"when $\ket b$ is an eigenvector of $A$ (only one $\beta_j \ne 0$)", r"always", r"never", r"when all $\beta_j$ are equal"],
         sol=r"""With distinct clock values, the Schmidt coefficients are the $|\beta_j|$; the state is a product iff only one is non-zero, i.e.\ $\ket b$ is an eigenvector. Otherwise system and clock are entangled
--- which is why the clock must be uncomputed after the inversion: leaving it entangled would leave the system in a mixture rather than $\ket x$.
Equal $\beta_j$ (the notebook's default) give \emph{maximal} entanglement over the four eigenvalues."""),

    dict(id="B13", kind="mcq",
         q=r"""A state vector of $n$ qubits needs $16\times2^n$ bytes (fp64). The notebook's default HHL circuit has 7 qubits, yet it prints ``Estimated memory: 4.10 KB''. Why?""",
         opts=[r"The state needs $2048$ B and the notebook reserves one extra copy (for reading the full state), giving $4096$ B",
               r"Each qubit needs $16 \times 7 = 112$ B and there are 36 of them",
               r"The density matrix of 7 qubits needs exactly 4.10 KB",
               r"Custom gates add $2048$ B of matrix storage per qubit"],
         sol=r"""$16\times2^7 = 2048$ B. The memory check is called with \texttt{extra\_copies=1} because \texttt{cudaq.get\_state} returns a second copy of the state: $2\times2048 = 4096$ B $= 4.10$ KB.
A density matrix would need $16\times4^7 = 262\,144$ B. Even with 8 clock qubits (11 qubits) the state needs only $32$ kB: HHL simulations of small systems are cheap;
the circuit depth, not memory, grows quickly (Section D)."""),

    dict(id="B14", kind="mcq",
         q=r"""Preparing an arbitrary normalised $n$-qubit amplitude vector $\ket b$ from $\ket{0\dots0}$ with one- and two-qubit gates requires in general""",
         opts=[r"$O(2^n)$ gates", r"$O(n)$ gates", r"$O(\log n)$ gates", r"a single gate per qubit"],
         sol=r"""An arbitrary state has $2^{n+1} - 2$ real degrees of freedom, and each gate fixes only $O(1)$ of them, so $\Omega(2^n)$ gates are needed in general (standard constructions use $O(2^n)$
uniformly controlled rotations). HHL's exponential advantage therefore requires $\ket b$ to be \emph{efficiently} preparable (structured $b$, or qRAM). The notebook
side-steps this on the simulator with \texttt{kernel.qalloc(vector)}, which initialises the amplitudes directly --- not possible on hardware."""),

    dict(id="B15", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} In HHL, measuring the clock register immediately after phase estimation would destroy the superposition over eigencomponents that $\ket x$ needs.\\[1mm]
\textbf{Reason (R):} The inverse QFT is a unitary operation.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} measuring the clock in $\sum_j\beta_j\ket{u_j}\ket{k_j}$ collapses it to one term $\ket{u_j}\ket{k_j}$ with probability $|\beta_j|^2$; the relative weights $\beta_j/\lambda_j$ of $\ket x$ are lost.
\textbf{R is true} but does not explain A: the reason is the collapse caused by measurement, not the unitarity of the inverse QFT. Answer: (B)."""),

    dict(id="B16", kind="mcq",
         q=r"""A swap test between states $\ket\psi$ and $\ket\phi$ gives the ancilla outcome 0 with probability $P(0) = \tfrac12(1 + |\braket{\psi|\phi}|^2)$. If $P(0) = 0.9$ is measured, the overlap
$|\braket{\psi|\phi}|^2$ is""",
         opts=[r"$0.8$", r"$0.9$", r"$0.4$", r"$0.95$"],
         sol=r"""$|\braket{\psi|\phi}|^2 = 2P(0) - 1 = 0.8$. Swap tests are one way to extract information from HHL's output $\ket x$ (e.g.\ its overlap with a reference) without reading all amplitudes.
The notebook instead reads the simulated state directly and reports the fidelity $|\braket{x_{\text{classical}}|x_{\text{quantum}}}|^2$."""),

    dict(id="B17", kind="para", passage="B",
         q=r"""For $\varphi = \tfrac14$, the measured counting register is""",
         opts=[r"$\ket{01}$ with probability 1", r"$\ket{10}$ with probability 1", r"$\ket{01}$ with probability $\tfrac14$", r"$\ket{00}$ with probability 1"],
         sol=r"""$\varphi = \tfrac14 = 1/2^2$ is exact: $\alpha_y = \frac14\sum_je^{2\pi ij(1/4 - y/4)}$ is $1$ for $y = 1$ and $0$ otherwise (geometric sum of roots of unity). So $y = 1 = \ket{01}$ with
certainty, $\varphi = y/4 = \tfrac14$ exactly."""),

    dict(id="B18", kind="para", passage="B",
         q=r"""For $\varphi = \tfrac13$, the most likely outcome and its probability are""",
         opts=[r"$\ket{01}$, about $0.70$", r"$\ket{01}$, exactly $1$", r"$\ket{10}$, about $0.70$", r"all four outcomes, $0.25$ each"],
         sol=r"""$\tfrac13 = 0.0101\dots_2$ is not a 2-bit fraction. $P(y) = |\alpha_y|^2$ evaluates to $(0.0625, 0.6998, 0.1875, 0.0502)$ for $y = 0, 1, 2, 3$: the nearest value $y = 1$ ($\varphi \approx 0.25$) wins with
$\approx 0.70$, but the rest leaks to neighbours. (The general guarantee for the nearest outcome is $\ge 4/\pi^2 \approx 0.405$.) In HHL such leakage spreads an eigenvalue
over several clock values, each inverted differently --- the fidelity drops below 1 (e.g.\ $0.953$ for the default with only 3 clock qubits)."""),

    dict(id="B19", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} The inverse QFT is the adjoint of the QFT, so applying it after the QFT returns the original state.\\[1mm]
\textbf{Statement II:} The QFT on $n$ qubits requires a number of gates exponential in $n$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} the QFT is unitary, $\mathrm{QFT}^{-1} = \mathrm{QFT}^\dagger$ (reverse the gate order and conjugate the angles --- the notebook's two QFT blocks are mirror images).
\textbf{II is false:} $n$ Hadamards, $n(n-1)/2$ controlled rotations and $\lfloor n/2\rfloor$ swaps: $O(n^2)$ gates. Phase estimation's cost is dominated by the controlled powers
$U^{2^j}$, not by the QFT. Answer: (C)."""),

    dict(id="B20", kind="mcq",
         q=r"""With $n_c = 4$ and $t = \pi/8$, the matrix $A$ is replaced by $A + I$. How do the ideal clock values change?""",
         opts=[r"each clock value increases by $1$", r"they are unchanged, because $I$ only adds a global phase", r"each clock value doubles", r"each clock value increases by $\pi/8$"],
         sol=r"""$e^{i(A + I)t} = e^{it}e^{iAt}$ (since $I$ commutes with $A$). Without control this would be a global phase, but in \emph{controlled}-$U$ it becomes a relative phase on the control: every eigenvalue
$\lambda$ becomes $\lambda + 1$ and every clock value $k = \lambda t\,2^{n_c}/(2\pi)$ increases by $t\,2^{n_c}/(2\pi) = 1$. Global phases of a gate matter once the gate is controlled ---
phase estimation measures exactly these."""),
]
