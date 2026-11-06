"""Section B - Quantum computing foundations for VQE (qubits, Bloch sphere, rotations, Pauli algebra, tensor products,
measurement bases, CNOT, qubit ordering, exp_pauli, memory, variance).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""A two-qubit transverse-field Ising Hamiltonian $H = -Z_0Z_1 - \tfrac12(X_0 + X_1)$ (the notebook's \texttt{"tfim"} model with $N = 2$,
$J = 1$, $h = 0.5$) is treated with a tiny hardware-efficient circuit: starting from $\ket{00}$, apply $R_Y(\theta_0)$ to qubit 0 and $R_Y(\theta_1)$
to qubit 1, then a CNOT with control 0 and target 1. The angles are $\theta_0 = \pi/2$, $\theta_1 = 0$.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""The state $R_Y(\pi/3)\ket0$ is measured in the computational basis. The probability of the outcome 1 is""",
         opts=[r"$\tfrac14$", r"$\tfrac34$", r"$\tfrac12$", r"$\sin^2(\pi/3) = \tfrac34$ for outcome 0"],
         sol=r"""$R_Y(\theta) = e^{-i\theta Y/2} = \begin{psmallmatrix}\cos\frac\theta2 & -\sin\frac\theta2\\ \sin\frac\theta2 & \cos\frac\theta2\end{psmallmatrix}$, so $R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$
(real amplitudes). With $\theta = \pi/3$: $P(1) = \sin^2(\pi/6) = \tfrac14$, $P(0) = \cos^2(\pi/6) = \tfrac34$. The notebook's hardware-efficient
ansatz is built from exactly these \texttt{ry} rotations."""),

    dict(id="B2", kind="mcq",
         q=r"""The two-qubit operator $(X\otimes Y)(Y\otimes X)$ equals""",
         opts=[r"$Z\otimes Z$", r"$-Z\otimes Z$", r"$i\,Z\otimes Z$", r"$Y\otimes Y$"],
         sol=r"""Multiply factor-wise: $(X\otimes Y)(Y\otimes X) = (XY)\otimes(YX)$. Pauli table: $XY = iZ$, $YX = -iZ$. So the product is
$(iZ)\otimes(-iZ) = -i^2\,Z\otimes Z = Z\otimes Z$. Products of Pauli strings are again Pauli strings up to a phase $\pm1, \pm i$ --- the
algebra behind Hamiltonian decompositions such as the notebook's $\texttt{XXYY}$, $\texttt{XYYX}$, \dots\ terms."""),

    dict(id="B3", kind="match",
         q=r"""Match each single-qubit state (List-I) with its Bloch vector $(\braket X, \braket Y, \braket Z)$ (List-II).""",
         match=([r"$\ket0$", r"$\ket- = \tfrac{1}{\sqrt2}(\ket0 - \ket1)$", r"$\tfrac{1}{\sqrt2}(\ket0 + i\ket1)$", r"$R_Y(\pi/3)\ket0$"],
                [r"$(0, 1, 0)$", r"$(\tfrac{\sqrt3}{2}, 0, \tfrac12)$", r"$(0, 0, 1)$", r"$(-1, 0, 0)$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2"],
         sol=r"""For $\ket\psi = a\ket0 + b\ket1$: $\braket Z = |a|^2 - |b|^2$, $\braket X = 2\,\mathrm{Re}(a^*b)$, $\braket Y = 2\,\mathrm{Im}(a^*b)$.
\begin{itemize}\itemsep0pt
\item P: $a = 1$, $b = 0$: $(0, 0, 1)$ (3).
\item Q: $a = -b = \frac{1}{\sqrt2}$: $\braket X = -1$: $(-1, 0, 0)$ (4).
\item R: $a^*b = \frac i2$: $\braket Y = 1$: $(0, 1, 0)$ (1).
\item S: $R_Y(\theta)\ket0$ has Bloch vector $(\sin\theta, 0, \cos\theta)$; $\theta = \pi/3$: $(\frac{\sqrt3}{2}, 0, \frac12)$ (2). $R_Y$ rotates in the $x$--$z$ plane:
real amplitudes, $\braket Y = 0$.
\end{itemize}"""),

    dict(id="B4", kind="match",
         q=r"""To estimate $\braket P$ from computational-basis measurements, each Pauli string is first rotated into a string of $Z$'s and $I$'s. Match each
observable (List-I) with the gates applied before measuring (List-II).""",
         match=([r"$X$", r"$Y$", r"$Z$", r"$X_0Z_1$"],
                [r"$H$ on qubit 0 only", r"nothing", r"$H$", r"$S^\dagger$, then $H$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1"],
         sol=r"""If $UPU^\dagger = Z$, measuring $Z$ after applying $U$ gives the eigenvalues of $P$: $\braket{\psi|P|\psi} = \braket{U\psi|Z|U\psi}$.
\begin{itemize}\itemsep0pt
\item P: $HXH = Z$: apply $H$ (3).
\item Q: $(HS^\dagger)\,Y\,(HS^\dagger)^\dagger = Z$: apply $S^\dagger$ first, then $H$ (4) (checked numerically in \texttt{verify\_answers.py}).
\item R: $Z$ is already diagonal: nothing (2).
\item S: rotate only the $X$ factor: $H$ on qubit 0; then $\braket{X_0Z_1}$ is the average of $(-1)^{b_0\oplus b_1}$ (1).
\end{itemize}
The notebook's text describes the same rule (``$H$ for $X$, $S^\dagger H$ for $Y$''); \texttt{cudaq.observe} does it automatically with shots."""),

    dict(id="B5", kind="mcq",
         q=r"""For the Bell state $\ket{\Phi^+} = \tfrac{1}{\sqrt2}(\ket{00} + \ket{11})$, the expectation value $\braket{Y\otimes Y}$ is""",
         opts=[r"$-1$", r"$+1$", r"$0$", r"$-\tfrac12$"],
         sol=r"""$Y\ket0 = i\ket1$, $Y\ket1 = -i\ket0$. So $(Y\otimes Y)\ket{00} = i\cdot i\ket{11} = -\ket{11}$ and $(Y\otimes Y)\ket{11} = (-i)(-i)\ket{00} = -\ket{00}$:
$(Y\otimes Y)\ket{\Phi^+} = -\ket{\Phi^+}$, an eigenstate with eigenvalue $-1$. (Consistency: $\braket{XX} = 1$, $\braket{ZZ} = 1$ and
$XX\cdot YY\cdot ZZ$-relation $(X\otimes X)(Y\otimes Y) = -(Z\otimes Z)$ gives $1\cdot\braket{YY} = -1$.)"""),

    dict(id="B6", kind="mcq",
         q=r"""The Hartree--Fock state of the notebook's H$_2$ example is $\ket{1100}$, written with qubit 0 first. In the vector returned by \texttt{cudaq.get\_state},
which lists amplitudes with qubit 0 as the \emph{least} significant bit, this basis state has the index""",
         opts=[r"$3$", r"$12$", r"$0$", r"$6$"],
         sol=r"""Qubit 0 first means the string $q_0q_1q_2q_3 = 1100$. With qubit 0 as the least significant bit the index is
$\sum_k q_k2^k = 1\cdot1 + 1\cdot2 + 0\cdot4 + 0\cdot8 = 3$. Read as a big-endian number the same string would be $12$. Mixing up the two
orderings is a classic source of wrong amplitudes; the repository's lecture scripts reorder CUDA-Q state vectors before comparing them
with NumPy for exactly this reason."""),

    dict(id="B7", kind="mcq",
         q=r"""Starting from $\ket{00}$, apply $R_Y(\pi/2)$ to qubit 0 and then a CNOT with control 0 and target 1. The resulting state is""",
         opts=[r"$\tfrac{1}{\sqrt2}(\ket{00} + \ket{11})$", r"$\tfrac{1}{\sqrt2}(\ket{00} + \ket{10})$", r"$\tfrac{1}{\sqrt2}(\ket{00} - \ket{11})$",
               r"$\tfrac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$"],
         sol=r"""$R_Y(\pi/2)\ket0 = \cos\frac\pi4\ket0 + \sin\frac\pi4\ket1 = \frac{1}{\sqrt2}(\ket0 + \ket1)$, so after the first gate $\frac{1}{\sqrt2}(\ket{00} + \ket{10})$.
The CNOT flips qubit 1 when qubit 0 is 1: $\ket{10} \to \ket{11}$. Result: the Bell state $\frac{1}{\sqrt2}(\ket{00} + \ket{11})$ --- entangled. This is
how the CNOT ladder of the hardware-efficient ansatz creates entanglement between neighbouring qubits."""),

    dict(id="B8", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} Any circuit of gates applied to $\ket{0\dots0}$ prepares a normalised state, so the ansatz state $\ket{\psi(\boldsymbol\theta)}$ needs no
explicit normalisation.\\[1mm]
\textbf{Statement II:} Measuring in the computational basis directly reveals the eigenvalue of an $X_i$ term, without any basis change.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} gates are unitary, $\|U\ket\psi\| = \|\ket\psi\|$, so $\ket{\psi(\boldsymbol\theta)} = U(\boldsymbol\theta)\ket{0\dots0}$ has norm 1 for every $\boldsymbol\theta$. This is why
VQE can use $\braket{\psi|H|\psi}$ directly, without dividing by $\braket{\psi|\psi}$ as in a Rayleigh quotient.
\textbf{II is false:} a $Z$-basis measurement yields eigenvalues of $Z$; for $X_i$ a Hadamard on qubit $i$ is needed first (B4). Answer: (C)."""),

    dict(id="B9", kind="mcq",
         q=r"""Consider the four exchange terms of the notebook's H$_2$ Hamiltonian, $\texttt{XXYY}$, $\texttt{XYYX}$, $\texttt{YXXY}$, $\texttt{YYXX}$. Which statement is
correct?""",
         opts=[r"All six pairs commute, but no pair is qubit-wise commuting",
               r"All six pairs are qubit-wise commuting, so one measurement setting suffices",
               r"Every pair anticommutes",
               r"Only $\texttt{XXYY}$ and $\texttt{YYXX}$ commute"],
         sol=r"""Two Pauli strings commute iff they differ by anticommuting single-qubit factors at an \emph{even} number of positions.
$\texttt{XXYY}$ vs $\texttt{XYYX}$: positions 1 and 3 differ ($X$/$Y$) --- two: commute. $\texttt{XXYY}$ vs $\texttt{YXXY}$: positions 0 and 2 --- commute.
$\texttt{XXYY}$ vs $\texttt{YYXX}$: all four positions --- commute. Likewise for the remaining three pairs.
\emph{Qubit-wise} commuting requires, at every position, equal letters or an $I$; every pair here has positions with $X$ against $Y$, so
no pair qualifies. Consequence: with simple qubit-wise grouping these 4 terms need 4 settings; with general commuting grouping (a
joint basis change containing CNOTs) they need only one (C14)."""),

    dict(id="B10", kind="mcq",
         q=r"""CUDA-Q's \texttt{exp\_pauli(theta, q, word)} applies $e^{i\theta P}$ for the Pauli string $P$ given by \texttt{word}. Applied with the word
$\texttt{"ZZ"}$ to the basis state $\ket{01}$, it gives""",
         opts=[r"$e^{-i\theta}\ket{01}$", r"$e^{+i\theta}\ket{01}$", r"$\cos\theta\ket{01} + i\sin\theta\ket{10}$", r"$\ket{10}$"],
         sol=r"""$Z\otimes Z\ket{01} = (+1)(-1)\ket{01} = -\ket{01}$: an eigenstate with eigenvalue $-1$. On an eigenvector, $e^{i\theta P}$ acts as the number
$e^{i\theta\lambda} = e^{-i\theta}$. A diagonal Pauli string only adds phases; it never changes bit values. The sign convention ($e^{+i\theta P}$,
verified with CUDA-Q 0.16) matters for the notebook's UCC state $\cos\theta\ket{1100} - \sin\theta\ket{0011}$."""),

    dict(id="B11", kind="mcq",
         q=r"""The four-qubit Pauli string $X_0X_1X_2Y_3$ (word $\texttt{"XXXY"}$) acting on $\ket{1100}$ gives""",
         opts=[r"$i\ket{0011}$", r"$-i\ket{0011}$", r"$\ket{0011}$", r"$i\ket{1100}$"],
         sol=r"""Qubit by qubit: $X\ket1 = \ket0$ (qubits 0, 1), $X\ket0 = \ket1$ (qubit 2), $Y\ket0 = i\ket1$ (qubit 3). Product: $i\ket{0011}$.
Hence $e^{i\theta P}\ket{1100} = \cos\theta\ket{1100} + i\sin\theta\,(i\ket{0011}) = \cos\theta\ket{1100} - \sin\theta\ket{0011}$: the notebook's
UCC state. The single Pauli string moves both electrons from orbital 0 (qubits 0, 1) to orbital 1 (qubits 2, 3) --- a double excitation."""),

    dict(id="B12", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} For a Hermitian $H$, the expectation value $\braket{\psi|H|\psi}$ is a real number for every state $\ket\psi$.\\[1mm]
\textbf{Reason (R):} The identity $\braket{\psi|H|\psi} = \braket{\psi|H|\psi}^*$ holds only for normalised states.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} $\braket{\psi|H|\psi}^* = \braket{\psi|H^\dagger|\psi} = \braket{\psi|H|\psi}$, so it equals its own complex conjugate.
\textbf{R is false:} the identity uses only $H = H^\dagger$; it holds for every vector, normalised or not (the norm $\braket{\psi|\psi}$ is real anyway).
Real energies are why a VQE energy can be compared with $E_0$ at all. Answer: (C)."""),

    dict(id="B13", kind="mcq",
         q=r"""A complex fp64 amplitude needs 16 bytes. For LiH in STO-3G with Jordan--Wigner (12 qubits), the memory needed for the state vector and
for the density matrix ($2^n\times2^n$ amplitudes) is""",
         opts=[r"$65.5$ kB and $268$ MB", r"$65.5$ kB and $131$ kB", r"$192$ B and $2.3$ kB", r"$268$ MB and $65.5$ kB"],
         sol=r"""State vector: $16\times2^{12} = 16\times4096 = 65\,536$ bytes $\approx 65.5$ kB. Density matrix: $16\times4^{12} = 16\times16\,777\,216 = 268\,435\,456$ bytes
$\approx 268$ MB. Density matrices (needed for exact noisy simulation, target \texttt{density-matrix-cpu}) square the cost; at the notebook's 15
qubits the density matrix would need $17.2$ GB versus $524$ kB for the state vector."""),

    dict(id="B14", kind="mcq",
         q=r"""Two qubits start in $\ket{00}$; $R_Y(\theta_0)$ and $R_Y(\theta_1)$ are applied to qubits 0 and 1, followed by a CNOT (control 0, target 1). The
expectation value $\braket{Z_0Z_1}$ of the final state is""",
         opts=[r"$\cos\theta_1$", r"$\cos\theta_0\cos\theta_1$", r"$\cos\theta_0$", r"$\cos(\theta_0 + \theta_1)$"],
         sol=r"""Heisenberg picture: $\braket{Z_0Z_1}_{\text{after}} = \braket{\mathrm{CNOT}\,Z_0Z_1\,\mathrm{CNOT}}_{\text{before}}$. CNOT conjugation maps $Z_1 \to Z_0Z_1$, hence
$Z_0Z_1 \to Z_0\cdot Z_0Z_1 = Z_1$. So $\braket{Z_0Z_1}_{\text{after}} = \braket{Z_1}_{\text{before}} = \cos\theta_1$ (Bloch vector of $R_Y(\theta_1)\ket0$).
Direct check with the bits: after the CNOT, $z_1' = z_0\oplus z_1$, so $z_0 \oplus z_1' = z_1$ and $(-1)^{z_0\oplus z_1'} = (-1)^{z_1}$. Before the CNOT the
answer would have been $\cos\theta_0\cos\theta_1$."""),

    dict(id="B15", kind="mcq",
         q=r"""Which unitary $U$ satisfies $UYU^\dagger = Z$, so that measuring $Z$ after applying $U$ measures $Y$?""",
         opts=[r"$U = HS^\dagger$ (first $S^\dagger$, then $H$)", r"$U = H$", r"$U = SH$ (first $H$, then $S$)", r"$U = S$"],
         sol=r"""With $S = \operatorname{diag}(1, i)$: $SXS^\dagger = \begin{psmallmatrix}1&0\\0&i\end{psmallmatrix}\begin{psmallmatrix}0&1\\1&0\end{psmallmatrix}\begin{psmallmatrix}1&0\\0&-i\end{psmallmatrix} = \begin{psmallmatrix}0&-i\\i&0\end{psmallmatrix} = Y$,
hence $S^\dagger YS = X$. Therefore
\[ (HS^\dagger)\,Y\,(HS^\dagger)^\dagger = H\,(S^\dagger YS)\,H = HXH = Z . \]
In circuit order: apply $S^\dagger$, then $H$, then measure in the computational basis. $H$ alone maps $Y \to -Y$; $S$ maps $Y \to SYS^\dagger = -X$;
neither diagonalises $Y$. (Verified numerically in \texttt{verify\_answers.py}.)"""),

    dict(id="B16", kind="para", passage="B",
         q=r"""The energy $\braket{\psi|H|\psi}$ of the prepared state is""",
         opts=[r"$-1$", r"$-1.5$", r"$-\sqrt2 \approx -1.414$", r"$0$"],
         sol=r"""The circuit prepares the Bell state $\frac{1}{\sqrt2}(\ket{00} + \ket{11})$ (B7). Term by term: $\braket{Z_0Z_1} = 1$ (both components have equal bits);
$\braket{X_0} = 0$ because $X_0$ maps $\ket{00} \to \ket{10}$ and $\ket{11} \to \ket{01}$, states orthogonal to the Bell state; likewise $\braket{X_1} = 0$.
$E = -1\cdot1 - \tfrac12(0 + 0) = -1$. The state wins the full $ZZ$ energy but none of the field energy."""),

    dict(id="B17", kind="para", passage="B",
         q=r"""The exact ground-state energy $E_0$ of this $H$, and the relative error of the prepared state, are""",
         opts=[r"$-\sqrt2 \approx -1.414$; $29.3\,\%$", r"$-1.5$; $33.3\,\%$", r"$-1$; $0\,\%$", r"$-\sqrt2$; $41.4\,\%$"],
         sol=r"""$H$ commutes with $X_0X_1$, so it splits into two $2\times2$ blocks. The ground state lies in the block $X_0X_1 = +1$, spanned by
$\ket{\Phi^+} = \frac{1}{\sqrt2}(\ket{00} + \ket{11})$ and $\ket{\Psi^+} = \frac{1}{\sqrt2}(\ket{01} + \ket{10})$. Matrix elements: $\braket{\Phi^+|H|\Phi^+} = -1$ (from $-Z_0Z_1$),
$\braket{\Psi^+|H|\Psi^+} = +1$, and since $X_0\ket{\Psi^+} = X_1\ket{\Psi^+} = \ket{\Phi^+}$: $\braket{\Phi^+|H|\Psi^+} = -\tfrac12(1 + 1) = -1$. Eigenvalues of $\begin{psmallmatrix}-1&-1\\-1&1\end{psmallmatrix}$: $\pm\sqrt2$, so $E_0 = -\sqrt2$
(free-fermion formula and exact diagonalisation agree). Relative error $\frac{-1 - (-\sqrt2)}{\sqrt2} = \frac{0.414}{1.414} = 29.3\,\%$.
Choosing $\theta_1 \neq 0$ (or more layers) lets the ansatz mix in $\ket{\Psi^+}$ and lower the energy."""),

    dict(id="B18", kind="mcq",
         q=r"""Every Hermitian $2^n\times2^n$ matrix is a real linear combination of the $n$-qubit Pauli strings. For $n = 4$ the number of Pauli strings,
and the number actually needed for the notebook's H$_2$ Hamiltonian, are""",
         opts=[r"$256$ and $15$", r"$16$ and $15$", r"$64$ and $15$", r"$256$ and $256$"],
         sol=r"""Each qubit carries one of $I, X, Y, Z$: $4^n$ strings; $n = 4$: $256$. They form an orthogonal basis of the $4^n$-dimensional space of
$2^n\times2^n$ matrices ($\operatorname{tr}(PQ) = 2^n\delta_{PQ}$), so $c_P = 2^{-n}\operatorname{tr}(PH)$. Molecular Hamiltonians are \emph{sparse} in this basis:
H$_2$ needs only 15 of the 256 (symmetries such as particle-number and spin conservation remove the rest)."""),

    dict(id="B19", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} Two Pauli strings either commute or anticommute.\\[1mm]
\textbf{Statement II:} A set of mutually commuting Pauli strings can be measured simultaneously after one common basis change.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} position by position the factors commute or anticommute, so $PQ = (-1)^kQP$ with $k$ = number of anticommuting positions.
\textbf{II is true:} commuting Hermitian operators share an eigenbasis; a Clifford circuit rotating that basis to the computational basis lets
one set of shots estimate all of them (for qubit-wise commuting sets this circuit is just single-qubit rotations). This is the basis of
measurement grouping. Answer: (A)."""),

    dict(id="B20", kind="mcq",
         q=r"""For $H = -Z + \tfrac12X$ (one qubit), the energy variance $\mathrm{Var}(H) = \braket{H^2} - \braket H^2$ in the state $\ket0$, and in the exact ground state, are""",
         opts=[r"$0.25$ and $0$", r"$1.25$ and $0$", r"$0$ and $0.25$", r"$0.25$ and $1.25$"],
         sol=r"""$H^2 = Z^2 - \tfrac12(ZX + XZ) + \tfrac14X^2 = \tfrac54I$ (the cross terms cancel because $ZX = -XZ$). So $\braket{H^2} = 1.25$ in every state.
For $\ket0$: $\braket H = -1$, variance $1.25 - 1 = 0.25$. For the ground state: $\braket H = -\sqrt{5/4}$, $\braket H^2 = 1.25$, variance $0$.
\emph{Zero-variance principle}: exact eigenstates have zero energy variance, so the variance measures how far a VQE state is from an
eigenstate --- and how many shots an energy estimate needs (shot noise $\propto \sqrt{\mathrm{Var}/S}$)."""),
]
