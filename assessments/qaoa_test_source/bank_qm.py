"""Section B - Quantum mechanics for QAOA (states, Born rule, Pauli algebra, rotations, CNOT, expectation values,
entanglement, symmetry, memory of a state vector).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""A single edge $(0,1)$ with weight 1 is treated with one QAOA layer in the notebook's convention. Starting from
$\ket{00}$ the gates are: $H$ on both qubits; CNOT (control 0, target 1); $R_Z(\theta)$ on qubit 1 with $\theta = \gamma w = \gamma$;
CNOT (control 0, target 1); $R_X(2\beta)$ on both qubits. The angles are $\gamma = \pi/2$ and $\beta = \pi/8$.
Amplitudes are listed in the order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$ (qubit 0 written first).""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""The 15-qubit state $\ket{+}^{\otimes 15}$ is measured in the computational basis. The notebook's graph has 4 optimal
bitstrings. The probability of obtaining an optimal bitstring, and the factor by which the previously recorded QAOA value
$P_{\text{opt}} = 0.06$ exceeds it, are""",
         opts=[r"$1.22\times10^{-4}$; about $490$", r"$3.05\times10^{-5}$; about $1970$",
               r"$2.44\times10^{-4}$; about $245$", r"$6.10\times10^{-5}$; about $980$"],
         sol=r"""$\ket{+}^{\otimes n} = 2^{-n/2}\sum_z \ket z$: every one of the $2^{15} = 32\,768$ bitstrings has amplitude $2^{-15/2}$ and
probability $|2^{-15/2}|^2 = 1/32\,768$ (Born rule). Four optimal strings: $P = 4/32\,768 = 1.22\times10^{-4}$.
Enhancement: $0.06 / (1.22\times10^{-4}) = 0.06 \times 8192 \approx 491.5$. QAOA concentrates probability on good cuts;
the initial uniform superposition is simply random guessing."""),

    dict(id="B2", kind="mcq",
         q=r"""The two-qubit operator $(Z\otimes Z)(X\otimes X)$ equals""",
         opts=[r"$-\,Y\otimes Y$", r"$Y\otimes Y$", r"$i\,Y\otimes Y$", r"$-i\,Y\otimes Y$"],
         sol=r"""Tensor products multiply factor by factor: $(Z\otimes Z)(X\otimes X) = (ZX)\otimes(ZX)$.
Single-qubit Pauli algebra: $ZX = iY$ (check: $ZX = \begin{psmallmatrix}1&0\\0&-1\end{psmallmatrix}\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}
= \begin{psmallmatrix}0&1\\-1&0\end{psmallmatrix} = i\begin{psmallmatrix}0&-i\\i&0\end{psmallmatrix} = iY$).
So $(iY)\otimes(iY) = i^2\,Y\otimes Y = -Y\otimes Y$. Also $(X\otimes X)(Z\otimes Z) = (XZ)\otimes(XZ) = (-iY)\otimes(-iY) = -Y\otimes Y$:
the same --- $Z\otimes Z$ and $X\otimes X$ commute, although $Z$ and $X$ anticommute."""),

    dict(id="B3", kind="match",
         q=r"""Match each operator (List-I) with its matrix (List-II). Two-qubit matrices use the basis order $\ket{00}, \ket{01},
\ket{10}, \ket{11}$.""",
         match=([r"$R_X(2\beta)$", r"$R_Z(\theta)$", r"$e^{-i\theta\, Z\otimes Z/2}$", r"$H$"],
                [r"$\operatorname{diag}\big(e^{-i\theta/2}, e^{i\theta/2}, e^{i\theta/2}, e^{-i\theta/2}\big)$",
                 r"$\begin{pmatrix}\cos\beta & -i\sin\beta\\ -i\sin\beta & \cos\beta\end{pmatrix}$",
                 r"$\tfrac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$",
                 r"$\operatorname{diag}\big(e^{-i\theta/2}, e^{i\theta/2}\big)$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3"],
         sol=r"""For a Pauli $P$ (so $P^2 = I$): $e^{-i\phi P} = \cos\phi\, I - i\sin\phi\, P$.
\begin{itemize}\itemsep0pt
\item P: $R_X(2\beta) = e^{-i\beta X} = \cos\beta\, I - i\sin\beta\, X$ (2).
\item Q: $R_Z(\theta) = e^{-i\theta Z/2} = \operatorname{diag}(e^{-i\theta/2}, e^{+i\theta/2})$ (4).
\item R: $Z\otimes Z = \operatorname{diag}(+1, -1, -1, +1)$, so $e^{-i\theta Z\otimes Z/2} = \operatorname{diag}(e^{-i\theta/2}, e^{i\theta/2}, e^{i\theta/2}, e^{-i\theta/2})$ (1):
equal bits get $e^{-i\theta/2}$, different bits get $e^{+i\theta/2}$.
\item S: Hadamard (3).
\end{itemize}"""),

    dict(id="B4", kind="mcq",
         q=r"""The state $R_Z(\theta)\ket{+}$ is passed through a Hadamard gate and then measured in the computational basis. The
probability of the outcome 0 is""",
         opts=[r"$\cos^2(\theta/2)$", r"$\tfrac12$", r"$\cos^2\theta$", r"$\sin^2(\theta/2)$"],
         sol=r"""$R_Z(\theta)\ket+ = \tfrac{1}{\sqrt2}(e^{-i\theta/2}\ket0 + e^{i\theta/2}\ket1) = e^{-i\theta/2}\,\tfrac{1}{\sqrt2}(\ket0 + e^{i\theta}\ket1)$.
Apply $H$: $\tfrac{1}{\sqrt2}(\ket0 + e^{i\theta}\ket1) \mapsto \tfrac12\big[(1 + e^{i\theta})\ket0 + (1 - e^{i\theta})\ket1\big]$.
$P(0) = \tfrac14|1 + e^{i\theta}|^2 = \tfrac14(2 + 2\cos\theta) = \cos^2(\theta/2)$.
Without the Hadamard the result would be $\tfrac12$ for every $\theta$: a $Z$-rotation only changes a \emph{relative phase},
which becomes visible only after interference. In QAOA the mixer plays this interfering role for the cost phases."""),

    dict(id="B5", kind="mcq",
         q=r"""For one unit edge $(0,1)$ the qubits are in the state $\ket\psi = \tfrac12\ket{00} + \tfrac{\sqrt3}{2}\ket{01}$. The expected
cut value $\langle C\rangle = \big\langle \tfrac12(1 - Z_0Z_1)\big\rangle$ is""",
         opts=[r"$\tfrac34$", r"$\tfrac14$", r"$-\tfrac12$", r"$\tfrac{\sqrt3}{2}$"],
         sol=r"""$Z_0Z_1$ is diagonal with eigenvalue $+1$ on $\ket{00}$ and $-1$ on $\ket{01}$. Born rule: $P(00) = \tfrac14$, $P(01) = \tfrac34$.
$\langle Z_0Z_1\rangle = \tfrac14(+1) + \tfrac34(-1) = -\tfrac12$, so $\langle C\rangle = \tfrac12\big(1 - (-\tfrac12)\big) = \tfrac34$.
Directly: the edge is cut exactly in outcome $01$, which has probability $\tfrac34$."""),

    dict(id="B6", kind="mcq",
         q=r"""Which of the following operators commutes with $Z_0Z_1$ (on two or more qubits)?""",
         opts=[r"$X_0X_1$", r"$X_0$", r"$Y_0Z_1$", r"$X_1$"],
         sol=r"""For Pauli products, count the positions where the two factors anticommute: an \emph{even} count means the products
commute, odd means they anticommute.
$X_0X_1$ vs $Z_0Z_1$: positions 0 ($X$ vs $Z$, anticommute) and 1 ($X$ vs $Z$, anticommute) --- two, so they commute.
$X_0$: one anticommuting position. $Y_0Z_1$: position 0 anticommutes ($Y$ vs $Z$), position 1 commutes --- one.
$X_1$: one. So only $X_0X_1$ commutes with $Z_0Z_1$ (consistent with B2)."""),

    dict(id="B7", kind="match",
         q=r"""Conjugation by a CNOT with control qubit 0 and target qubit 1 maps a Pauli operator $P$ to
$\mathrm{CNOT}\,P\,\mathrm{CNOT}$. Match each $P$ (List-I) with its image (List-II).""",
         match=([r"$X\otimes I$", r"$I\otimes X$", r"$Z\otimes I$", r"$I\otimes Z$"],
                [r"$Z\otimes Z$", r"$X\otimes X$", r"$I\otimes X$", r"$Z\otimes I$"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4",
               r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1"],
         sol=r"""Rules (CNOT$^2 = I$, so conjugation is $P \mapsto \mathrm{CNOT}\,P\,\mathrm{CNOT}$):
\begin{itemize}\itemsep0pt
\item $X$ on the control \emph{spreads} to the target: $X\otimes I \to X\otimes X$ (2). (Flipping the control first changes whether the target is flipped.)
\item $X$ on the target is unchanged: $I\otimes X \to I\otimes X$ (3) (the target flip commutes with ``flip if control is 1'').
\item $Z$ on the control is unchanged: $Z\otimes I \to Z\otimes I$ (4) (the control value is never changed).
\item $Z$ on the target spreads \emph{back} to the control: $I\otimes Z \to Z\otimes Z$ (1): after the CNOT the target holds $z_0\oplus z_1$.
\end{itemize}
The last rule is the heart of the cost gate: CNOT$\cdot(I\otimes R_Z(\theta))\cdot$CNOT $= e^{-i\theta\,Z\otimes Z/2}$."""),

    dict(id="B8", kind="mcq",
         q=r"""The state $e^{-i\gamma\, Z\otimes Z/2}\ket{+}\ket{+}$ is maximally entangled for which $\gamma \in (0, \pi)$?""",
         opts=[r"$\gamma = \pi/2$", r"$\gamma = \pi/4$", r"$\gamma = 3\pi/4$", r"for no $\gamma$: it is always a product state"],
         sol=r"""Amplitudes: $c_{00} = c_{11} = \tfrac12 e^{-i\gamma/2}$, $c_{01} = c_{10} = \tfrac12 e^{i\gamma/2}$.
A two-qubit pure state is a product state iff $c_{00}c_{11} - c_{01}c_{10} = 0$; the concurrence is
$\mathcal C = 2|c_{00}c_{11} - c_{01}c_{10}| = 2\cdot\tfrac14|e^{-i\gamma} - e^{i\gamma}| = |\sin\gamma|$.
$\mathcal C = 1$ (maximal) requires $\sin\gamma = \pm1$, i.e.\ $\gamma = \pi/2$ in $(0, \pi)$; at $\gamma = \pi/4$ or $3\pi/4$, $\mathcal C = 0.71$.
The cost layer is the only entangling part of QAOA; the mixer acts on single qubits."""),

    dict(id="B9", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} Replacing $H_C$ by $H_C + c\,I$ (any real constant $c$) changes the QAOA state
$\ket{\boldsymbol\gamma, \boldsymbol\beta}$ only by a global phase.\\[1mm]
\textbf{Statement II:} Consequently, a QAOA circuit that uses $e^{-i\gamma C}$ with $C = \tfrac W2 - H_C$ has the same optimal
angles $(\gamma^*, \beta^*)$ as the notebook's circuit with $e^{-i\gamma H_C}$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $cI$ commutes with everything, so $e^{-i\gamma(H_C + cI)} = e^{-i\gamma c}\,e^{-i\gamma H_C}$; the factors
$e^{-i\gamma_l c}$ multiply the final state by an unobservable global phase.
\textbf{II is false:} $C = \tfrac W2 I - H_C$ contains $-H_C$, so $e^{-i\gamma C} = e^{-i\gamma W/2}\,e^{+i\gamma H_C}$. Apart from a global phase
this is the notebook's cost unitary with $\gamma \to -\gamma$. The optimal angles are related by $\gamma^* \to -\gamma^*$ (the
constant is harmless, the \emph{sign} is not). This is why textbook formulas (written with $e^{-i\gamma C}$) must be translated
before they are used with the notebook's convention. Answer: (C)."""),

    dict(id="B10", kind="mcq",
         q=r"""Two qubits are in a product state in which $P(z_0 = 1) = 0.2$ and $P(z_1 = 1) = 0.7$. The value of $\langle Z_0Z_1\rangle$ is""",
         opts=[r"$-0.24$", r"$+0.24$", r"$0.62$", r"$0.14$"],
         sol=r"""For a product state the expectation factorises: $\langle Z_0Z_1\rangle = \langle Z_0\rangle\langle Z_1\rangle$, with
$\langle Z\rangle = P(0) - P(1)$. So $\langle Z_0\rangle = 0.8 - 0.2 = 0.6$ and $\langle Z_1\rangle = 0.3 - 0.7 = -0.4$, giving $-0.24$.
Check via the cut probability: $P(z_0 \neq z_1) = 0.2\cdot0.3 + 0.8\cdot0.7 = 0.62$ and $\langle Z_0Z_1\rangle = 1 - 2\cdot0.62 = -0.24$."""),

    dict(id="B11", kind="mcq",
         q=r"""Conjugating $Z$ by the mixer gate gives $e^{i\beta X}\,Z\,e^{-i\beta X} =$""",
         opts=[r"$\cos 2\beta\; Z + \sin 2\beta\; Y$", r"$\cos\beta\; Z + \sin\beta\; Y$", r"$\cos 2\beta\; Z - \sin 2\beta\; Y$",
               r"$\cos 2\beta\; Z + \sin 2\beta\; X$"],
         sol=r"""Because $XZ = -ZX$, moving $Z$ through $e^{i\beta X}$ flips the sign of the exponent: $e^{i\beta X}Z = Z\,e^{-i\beta X}$. Hence
\[ e^{i\beta X}Ze^{-i\beta X} = Z\,e^{-2i\beta X} = Z(\cos 2\beta - i\sin 2\beta\, X) = \cos2\beta\, Z - i\sin2\beta\, ZX . \]
With $ZX = iY$: $-i\sin2\beta\,(iY) = +\sin 2\beta\, Y$. Result $\cos2\beta\,Z + \sin2\beta\,Y$: a rotation of the Bloch vector by $2\beta$
about the $x$ axis (the angle doubles, as for every $e^{-i\phi P}$). This identity is the first step of the $p = 1$ derivation."""),

    dict(id="B12", kind="mcq",
         q=r"""A state vector needs $16 \times 2^n$ bytes in fp64 and $8\times 2^n$ bytes in fp32. Ignoring simulator workspace, the
largest $n$ whose state fits on a single 80 GB ($80\times10^9$ bytes) GPU is""",
         opts=[r"$32$ (fp64) and $33$ (fp32)", r"$33$ (fp64) and $34$ (fp32)", r"$32$ (fp64) and $64$ (fp32)",
               r"$36$ (fp64) and $37$ (fp32)"],
         sol=r"""fp64: $16\times2^{32} = 6.87\times10^{10}$ bytes $= 68.7$ GB $\le 80$ GB, but $16\times2^{33} = 137.4$ GB $> 80$ GB, so $n = 32$.
fp32 halves the memory: $8\times2^{33} = 68.7$ GB fits, $8\times2^{34} = 137.4$ GB does not, so $n = 33$.
Halving the precision buys exactly one qubit. The notebook's 15-qubit state needs $16\times2^{15} = 524\,288$ bytes (``524.29 KB'')."""),

    dict(id="B13", kind="mcq",
         q=r"""The gate $R_X(2\beta)$ with $\beta = \pi/8$ is applied to $\ket0$. The probability of measuring 1 is""",
         opts=[r"$\tfrac{2 - \sqrt2}{4} \approx 0.146$", r"$\tfrac12$", r"$\tfrac{2 + \sqrt2}{4} \approx 0.854$", r"$\sin(\pi/8) \approx 0.383$"],
         sol=r"""$R_X(2\beta)\ket0 = \cos\beta\ket0 - i\sin\beta\ket1$ (first column of the matrix in B3), so $P(1) = \sin^2\beta$.
For $\beta = \pi/8$: $\sin^2(\pi/8) = \tfrac{1 - \cos(\pi/4)}{2} = \tfrac{1 - \sqrt2/2}{2} = \tfrac{2 - \sqrt2}{4} \approx 0.146$.
$\sin(\pi/8)$ is the amplitude's modulus, not the probability."""),

    dict(id="B14", kind="mcq",
         q=r"""Which statement about $\ket{+}^{\otimes n}$ and $H_M = \sum_{i} X_i$ is correct?""",
         opts=[r"It is the eigenstate of $H_M$ with the \emph{largest} eigenvalue, $+n$",
               r"It is the ground state of $H_M$, with eigenvalue $-n$",
               r"It is an eigenstate of $H_M$ with eigenvalue $0$",
               r"It is not an eigenstate of $H_M$"],
         sol=r"""$X\ket+ = +\ket+$, so each $X_i$ gives $+1$ on $\ket+^{\otimes n}$ and $H_M\ket+^{\otimes n} = n\,\ket+^{\otimes n}$.
The eigenvalues of $H_M$ range from $-n$ (state $\ket-^{\otimes n}$) to $+n$, so $\ket+^{\otimes n}$ is the \emph{highest} eigenstate of
$H_M$ (equivalently the ground state of $-H_M$). This matters for the annealing interpretation: adiabatic evolution keeps a
state in the \emph{same} position of the spectrum, so with $+H_M$ the matching target is the highest state of the final
Hamiltonian (see C14)."""),

    dict(id="B15", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} A state $\ket\psi$ and its complex conjugate $\ket{\psi^*}$ (every amplitude conjugated) give
identical computational-basis measurement statistics.\\[1mm]
\textbf{Reason (R):} The initial QAOA state $\ket{+}^{\otimes n}$ has only real amplitudes.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} the outcome probabilities are $|a_z|^2$ and $|a_z^*|^2 = |a_z|^2$ for every $z$, for \emph{any} state.
\textbf{R is true:} all amplitudes of $\ket+^{\otimes n}$ equal $2^{-n/2}$.
But R does not explain A: A holds for every state, real or not; the explanation is $|a^*| = |a|$.
(R \emph{is} the reason for a different fact: since $H_C$, $H_M$ and $\ket+^{\otimes n}$ are real,
$\ket{\boldsymbol\gamma,\boldsymbol\beta}^* = \ket{-\boldsymbol\gamma, -\boldsymbol\beta}$, so $F_p(-\boldsymbol\gamma, -\boldsymbol\beta) = F_p(\boldsymbol\gamma, \boldsymbol\beta)$.) Answer: (B)."""),

    dict(id="B16", kind="mcq",
         q=r"""For the triangle $K_3$ (unit weights) a state has $\langle Z_0Z_1\rangle = -0.6$, $\langle Z_1Z_2\rangle = -0.2$ and
$\langle Z_0Z_2\rangle = 0.4$. Its expected cut $\langle C\rangle$ is""",
         opts=[r"$1.7$", r"$1.3$", r"$-0.2$", r"$1.9$"],
         sol=r"""By linearity $\langle H_C\rangle = \tfrac12(-0.6 - 0.2 + 0.4) = -0.2$, and $\langle C\rangle = \tfrac W2 - \langle H_C\rangle = 1.5 + 0.2 = 1.7$.
Edge by edge: $\langle C_{ij}\rangle = \tfrac12(1 - \langle Z_iZ_j\rangle)$ gives $0.8 + 0.6 + 0.3 = 1.7$.
This is how \texttt{cudaq.observe} works: it evaluates $\langle H\rangle = \sum_k c_k\langle P_k\rangle$ term by term. The value is below
$C_{\max} = 2$, as it must be."""),

    dict(id="B17", kind="mcq",
         q=r"""On $n$ qubits, $e^{-i(\pi/2) H_M}$ with $H_M = \sum_i X_i$ equals ..., and therefore (since $C(z) = C(\bar z)$) $F_p$ is
periodic in every single $\beta_l$ with period ...""",
         opts=[r"$(-i)^n X^{\otimes n}$; $\ \pi/2$", r"$I$; $\ \pi/2$", r"$(-i)^n X^{\otimes n}$; $\ \pi$", r"$X^{\otimes n}$; $\ 2\pi$"],
         sol=r"""$e^{-i(\pi/2)X} = \cos\tfrac\pi2 I - i\sin\tfrac\pi2 X = -iX$, and the mixer factorises over qubits, so
$e^{-i(\pi/2)H_M} = (-iX)^{\otimes n} = (-i)^n X^{\otimes n}$: a global phase times ``flip every bit''.
Shifting $\beta_l \to \beta_l + \pi/2$ inserts $X^{\otimes n}$ after layer $l$. $X^{\otimes n}$ commutes with every $Z_iZ_j$ (two sign
flips) and with $H_M$, so it can be moved to the end, where it maps $\ket z \to \ket{\bar z}$; since $C(\bar z) = C(z)$ the
expectation is unchanged. Hence period $\pi/2$ in each $\beta_l$ --- one quarter of the naive $2\pi$."""),

    dict(id="B18", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} $Z_iZ_j$ has eigenvalue $-1$ exactly on the computational basis states with $z_i \neq z_j$.\\[1mm]
\textbf{Statement II:} $\langle Z_iZ_j\rangle$ can be estimated from the same computational-basis shots that are used to sample
candidate splits.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $Z_iZ_j\ket z = s_is_j\ket z$ with $s = (-1)^z$, and $s_is_j = -1 \iff z_i \neq z_j$.
\textbf{II is true:} $Z_iZ_j$ is diagonal in the computational basis, so each shot $z$ directly yields the value
$(-1)^{z_i\oplus z_j}$; averaging over shots estimates $\langle Z_iZ_j\rangle$ (and $\langle H_C\rangle$) with no extra basis change.
This is what happens on hardware targets, where the notebook estimates the objective from shots. Answer: (A)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""Immediately after the second CNOT (before the mixer), the phase of the amplitude of $\ket{01}$ relative to the amplitude
of $\ket{00}$ is""",
         opts=[r"$90^\circ$", r"$45^\circ$", r"$0^\circ$", r"$180^\circ$"],
         sol=r"""After the Hadamards every amplitude is $\tfrac12$. CNOT$\cdot(I\otimes R_Z(\gamma))\cdot$CNOT $= e^{-i\gamma Z_0Z_1/2}$: the first CNOT
writes $z_0\oplus z_1$ on qubit 1, $R_Z(\gamma)$ gives phase $e^{-i\gamma/2}$ if that bit is 0 and $e^{+i\gamma/2}$ if it is 1, and the
second CNOT restores the bits. With $\gamma = \pi/2$:
$\ket{00}, \ket{11} \to \tfrac12 e^{-i\pi/4}$ and $\ket{01}, \ket{10} \to \tfrac12 e^{+i\pi/4}$.
Relative phase of $\ket{01}$ to $\ket{00}$: $e^{i\pi/4}/e^{-i\pi/4} = e^{i\pi/2}$, i.e.\ $90^\circ$ ($= \gamma$). The measurement
probabilities are still all $\tfrac14$."""),

    dict(id="B20", kind="para", passage="B",
         q=r"""After the mixer, the probability of measuring a \emph{cut} bitstring ($01$ or $10$) is""",
         opts=[r"$0$", r"$1$", r"$\tfrac12$", r"$\tfrac34$"],
         sol=r"""For one edge with one layer in this convention $\langle Z_0Z_1\rangle = \sin4\beta\,\sin\gamma$ (derived in the Answer Book of C7;
the derivation uses B11 twice and the conjugation of $Y_0Z_1$ by the cost gate). At $\gamma = \pi/2$, $\beta = \pi/8$:
$\sin(\pi/2)\sin(\pi/2) = 1$, so $\langle Z_0Z_1\rangle = +1$ and $P(\text{cut}) = \tfrac12(1 - \langle Z_0Z_1\rangle) = 0$.
Explicitly, the final state has support only on $\ket{00}$ and $\ket{11}$ (state-vector simulation gives the probabilities
$0.5, 0, 0, 0.5$ for $\ket{00}, \ket{01}, \ket{10}, \ket{11}$). With $\beta = -\pi/8$ instead, the probability of a cut bitstring is $1$. The sign of $\beta$
relative to $\gamma$ decides whether the layer helps or hurts."""),
]
