"""Section A - Mathematical and quantum-chemistry foundations (eigenproblems, Rayleigh quotient, Hermitian matrices,
Schroedinger equation, Born-Oppenheimer, second quantisation, Pauli decomposition, energy units, correlation energy).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""A single qubit has the Hamiltonian $H = -Z + \tfrac12 X = \begin{pmatrix}-1 & \tfrac12\\ \tfrac12 & 1\end{pmatrix}$.
A one-parameter trial state is prepared as $\ket{\psi(\theta)} = R_Y(\theta)\ket0 = \cos\tfrac\theta2\ket0 + \sin\tfrac\theta2\ket1$, and its
energy $E(\theta) = \braket{\psi(\theta)|H|\psi(\theta)}$ is minimised over $\theta \in (-\pi, \pi]$.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""The ground-state energy (smallest eigenvalue) of $H = \begin{pmatrix}1 & 2\\ 2 & -2\end{pmatrix}$ and a normalised eigenvector for it are""",
         opts=[r"$-3$ and $\tfrac{1}{\sqrt5}(1, -2)^{\mathsf T}$", r"$-3$ and $\tfrac{1}{\sqrt5}(2, 1)^{\mathsf T}$",
               r"$2$ and $\tfrac{1}{\sqrt5}(2, 1)^{\mathsf T}$", r"$-2$ and $(0, 1)^{\mathsf T}$"],
         sol=r"""Characteristic polynomial: $\det(H - \lambda I) = (1 - \lambda)(-2 - \lambda) - 4 = \lambda^2 + \lambda - 6 = (\lambda + 3)(\lambda - 2)$,
so $\lambda \in \{-3, 2\}$ and $E_0 = -3$.
Eigenvector for $-3$: $(H + 3I)v = 0$ gives $4v_1 + 2v_2 = 0$, i.e.\ $v_2 = -2v_1$: $v = (1, -2)/\sqrt5$.
Check: $Hv = (1 - 4, 2 + 4)/\sqrt5 = (-3, 6)/\sqrt5 = -3v$. The vector $(2, 1)/\sqrt5$ is orthogonal to it and belongs to $\lambda = 2$."""),

    dict(id="A2", kind="mcq",
         q=r"""For the same matrix $H = \begin{pmatrix}1 & 2\\ 2 & -2\end{pmatrix}$, the Rayleigh quotient $R(v) = \dfrac{v^{\mathsf T}Hv}{v^{\mathsf T}v}$ of the trial
vector $v = (1, -1)^{\mathsf T}$ is""",
         opts=[r"$-2.5$, which lies above $E_0 = -3$", r"$-5$, which lies below $E_0$", r"$-3$, exactly $E_0$", r"$0.5$"],
         sol=r"""$Hv = (1 - 2,\ 2 + 2) = (-1, 4)$, so $v^{\mathsf T}Hv = 1\cdot(-1) + (-1)\cdot4 = -5$, and $v^{\mathsf T}v = 2$: $R(v) = -5/2 = -2.5$.
The Rayleigh quotient of any non-zero vector lies between the smallest and largest eigenvalue: $-3 \le -2.5 \le 2$.
Eigenbasis check: with $u_0 = (1, -2)/\sqrt5$ (eigenvalue $-3$), the weight of $u_0$ in $v/\|v\|$ is $\langle v, u_0\rangle^2/\|v\|^2 = (3/\sqrt5)^2/2 = 0.9$;
the remaining $0.1$ belongs to eigenvalue $2$: $R = 0.9(-3) + 0.1(2) = -2.5$. This is the variational principle in its simplest form."""),

    dict(id="A3", kind="match",
         q=r"""For the real symmetric matrix $M = \begin{pmatrix}2 & 1 & 0\\ 1 & 2 & 0\\ 0 & 0 & 5\end{pmatrix}$, match each quantity (List-I) with its value (List-II).""",
         match=([r"$\operatorname{tr} M$", r"$\det M$", r"smallest eigenvalue of $M$", r"largest eigenvalue of $M$"],
                [r"$15$", r"$1$", r"$9$", r"$5$"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4", r"P$\to$1, Q$\to$3, R$\to$2, S$\to$4",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""$M$ is block diagonal: the block $\begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$ has eigenvalues $2 \pm 1 = 1, 3$ (eigenvectors $(1, \mp1)/\sqrt2$), and the
block $(5)$ has eigenvalue $5$. Spectrum $\{1, 3, 5\}$.
\begin{itemize}\itemsep0pt
\item P: trace = sum of diagonal entries $= 2 + 2 + 5 = 9$ = sum of eigenvalues (3).
\item Q: determinant = product of eigenvalues $= 1\cdot3\cdot5 = 15$ (also $(4 - 1)\cdot5$) (1).
\item R: smallest eigenvalue $1$ (2) --- the ``ground state'' of $M$.
\item S: largest eigenvalue $5$ (4).
\end{itemize}"""),

    dict(id="A4", kind="mcq",
         q=r"""An energy eigenstate $\ket{\psi_E}$ of a time-independent Hamiltonian ($H\ket{\psi_E} = E\ket{\psi_E}$) evolves under the time-dependent
Schr\"odinger equation $i\hbar\,\partial_t\ket{\psi(t)} = H\ket{\psi(t)}$. Which statement is correct?""",
         opts=[r"$\ket{\psi(t)} = e^{-iEt/\hbar}\ket{\psi_E}$: only a global phase changes, so all measurement probabilities are constant in time",
               r"$\ket{\psi(t)} = e^{-Et/\hbar}\ket{\psi_E}$: the state decays in time",
               r"The probabilities oscillate with angular frequency $E/\hbar$",
               r"The state is constant only if $E = 0$; otherwise it moves to other eigenstates"],
         sol=r"""The formal solution is $\ket{\psi(t)} = e^{-iHt/\hbar}\ket{\psi(0)}$. On an eigenvector $H$ acts as the number $E$, so
$e^{-iHt/\hbar}\ket{\psi_E} = e^{-iEt/\hbar}\ket{\psi_E}$ (expand the exponential as a power series). The factor has modulus 1: a
\emph{global} phase, invisible in every measurement. Hence eigenstates are called \emph{stationary states}, and finding them
reduces to the time-independent equation $H\ket\psi = E\ket\psi$ --- the eigenvalue problem VQE solves. A superposition of
eigenstates with different energies, by contrast, acquires \emph{relative} phases and its probabilities can oscillate."""),

    dict(id="A5", kind="mcq",
         q=r"""Using $1\ \text{Hartree} = 627.51$ kcal/mol, an energy error of $0.005$ Hartree corresponds to about ... and is ... chemically
accurate (chemical accuracy: 1 kcal/mol $\approx$ 1.6 mHa).""",
         opts=[r"$3.14$ kcal/mol; not", r"$3.14$ kcal/mol; still", r"$0.31$ kcal/mol; still", r"$125.5$ kcal/mol; not"],
         sol=r"""$0.005 \times 627.51 = 3.138$ kcal/mol. Chemical accuracy means an error below $1$ kcal/mol $= 1/627.51$ Ha $= 1.594$ mHa
($\approx 1.6$ mHa, the threshold printed by the notebook). $5$ mHa is about three times too large, so the result is \emph{not}
chemically accurate. The threshold matters because reaction energies and barriers decide chemistry at the kcal/mol scale."""),

    dict(id="A6", kind="mcq",
         q=r"""In the Born--Oppenheimer approximation the nuclei are held fixed and only the electrons are treated quantum mechanically. In the
notebook's 4-qubit H$_2$ Hamiltonian, the nuclear--nuclear repulsion energy appears""",
         opts=[r"inside the coefficient of the identity string $\texttt{IIII}$, as a constant energy shift",
               r"as the coefficient of the $\texttt{ZZII}$ term",
               r"spread over the four $\texttt{XXYY}$-type exchange terms",
               r"nowhere, because nuclear repulsion is neglected in quantum chemistry"],
         sol=r"""The molecular Hamiltonian is $H = T_e + V_{eN} + V_{ee} + V_{NN}$. With fixed nuclei (Born--Oppenheimer), $V_{NN} = \frac{Z_AZ_B}{R}$ is
just a number for a given bond length $R$. A number multiplies the identity operator, so after the fermion-to-qubit mapping it
ends up in the coefficient of $\texttt{IIII}$ (together with constant pieces of the electronic terms). The notebook's comment
says so: ``the constant term includes the nuclear repulsion energy''. It shifts every eigenvalue equally and needs no
measurement, but it must be included to compare with total energies such as the FCI value $-1.137284$ Ha."""),

    dict(id="A7", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} For every normalised state $\ket\psi$, $\braket{\psi|H|\psi} \ge E_0$, where $E_0$ is the smallest eigenvalue of the
Hermitian operator $H$.\\[1mm]
\textbf{Reason (R):} Expanding $\ket\psi = \sum_k c_k\ket{E_k}$ in the orthonormal eigenbasis of $H$ gives $\braket{\psi|H|\psi} = \sum_k |c_k|^2E_k$ with
$\sum_k|c_k|^2 = 1$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} a Hermitian $H$ has an orthonormal eigenbasis (spectral theorem). Then
$\braket{\psi|H|\psi} = \sum_{k,l} c_k^*c_l\braket{E_k|H|E_l} = \sum_{k,l}c_k^*c_lE_l\delta_{kl} = \sum_k|c_k|^2E_k$, and normalisation gives $\sum|c_k|^2 = 1$.
\textbf{A follows from R:} a weighted average with non-negative weights summing to 1 cannot be below the smallest value:
$\sum_k|c_k|^2E_k \ge \sum_k|c_k|^2E_0 = E_0$; equality iff all weight sits on ground states. R is the correct explanation. Answer: (A)."""),

    dict(id="A8", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} A real symmetric matrix can have complex (non-real) eigenvalues.\\[1mm]
\textbf{Statement II:} Eigenvectors of a Hermitian matrix that belong to different eigenvalues are orthogonal.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} real symmetric matrices are Hermitian; if $Hv = \lambda v$ with $\|v\| = 1$ then
$\lambda = v^\dagger Hv = (v^\dagger Hv)^* = \lambda^*$, so $\lambda$ is real. (Complex eigenvalues occur for non-symmetric real matrices such as rotations.)
\textbf{II is true:} from $Hu = \lambda u$, $Hv = \mu v$: $\mu\,u^\dagger v = u^\dagger Hv = (Hu)^\dagger v = \lambda u^\dagger v$, so $(\mu - \lambda)u^\dagger v = 0$ and
$u^\dagger v = 0$ when $\lambda \neq \mu$. Real energies and orthogonal eigenstates are exactly what the variational argument uses. Answer: (D)."""),

    dict(id="A9", kind="mcq",
         q=r"""H$_2$ in the STO-3G basis has 2 spatial orbitals, i.e.\ 4 spin-orbitals, and 2 electrons. The number of 2-electron Slater determinants,
and the number of them with one spin-up and one spin-down electron ($S_z = 0$), are""",
         opts=[r"$6$ and $4$", r"$16$ and $4$", r"$6$ and $2$", r"$4$ and $2$"],
         sol=r"""A Slater determinant is fixed by choosing which spin-orbitals are occupied: $\binom{4}{2} = 6$ ways (the 4-qubit Fock space has $2^4 = 16$
basis states for \emph{all} electron numbers). With one $\alpha$ electron (2 choices of spatial orbital) and one $\beta$ electron (2 choices):
$2\times2 = 4$ determinants with $S_z = 0$. Exact diagonalisation within such a determinant space is \emph{full configuration
interaction} (FCI). The notebook's reference $-1.137284$ Ha is the FCI energy; its ground state uses only two of these
determinants, $\ket{1100}$ and $\ket{0011}$ (see C19)."""),

    dict(id="A10", kind="mcq",
         q=r"""LiH in the minimal STO-3G basis has one $1s$ orbital on H and the orbitals $1s, 2s, 2p_x, 2p_y, 2p_z$ on Li. With the Jordan--Wigner
mapping (one qubit per spin-orbital, no reductions), the number of qubits and the number of electrons are""",
         opts=[r"$12$ qubits, $4$ electrons", r"$6$ qubits, $4$ electrons", r"$12$ qubits, $3$ electrons", r"$24$ qubits, $4$ electrons"],
         sol=r"""Spatial orbitals: $5$ (Li) $+ 1$ (H) $= 6$. Each holds a spin-up and a spin-down spin-orbital: $12$ spin-orbitals $\Rightarrow$ $12$
qubits under Jordan--Wigner. Electrons: Li has 3, H has 1: $4$. (Checked with PySCF in \texttt{verify\_answers.py}.) Freezing the Li $1s$
core and using symmetries reduces the qubit count further (D17). LiH is not part of the notebook; it is a standard
illustration of how molecular problems grow."""),

    dict(id="A11", kind="match",
         q=r"""Fermionic creation and annihilation operators $a_p^\dagger$, $a_p$ satisfy the canonical anticommutation relations. Match each expression
(List-I) with its value (List-II); $\ket{\mathrm{vac}}$ is the empty (vacuum) state.""",
         match=([r"$\{a_p, a_q^\dagger\} = a_pa_q^\dagger + a_q^\dagger a_p$", r"$a_p^\dagger a_p^\dagger$",
                 r"spectrum of $n_p = a_p^\dagger a_p$", r"$a_p\ket{\mathrm{vac}}$"],
                [r"$\{0, 1\}$", r"$\delta_{pq}\,I$", r"$0$ (the zero vector)", r"$0$ (the zero operator)"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: canonical anticommutation relation $\{a_p, a_q^\dagger\} = \delta_{pq}I$ (2).
\item Q: $\{a_p^\dagger, a_p^\dagger\} = 2(a_p^\dagger)^2 = 0$, so $(a_p^\dagger)^2 = 0$ as an \emph{operator}: no orbital can be filled twice --- Pauli exclusion (4).
\item R: $n_p^2 = a_p^\dagger(a_pa_p^\dagger)a_p = a_p^\dagger(1 - a_p^\dagger a_p)a_p = n_p$, so $n_p$ is a projector with eigenvalues $0$ and $1$ (empty/occupied) (1).
\item S: there is nothing to remove from the vacuum: $a_p\ket{\mathrm{vac}}$ is the zero \emph{vector} (3).
\end{itemize}
These relations are what the Jordan--Wigner $Z$-strings must reproduce on qubits."""),

    dict(id="A12", kind="mcq",
         q=r"""In second quantisation $H = \sum_{pq}h_{pq}a_p^\dagger a_q + \tfrac12\sum_{pqrs}h_{pqrs}a_p^\dagger a_q^\dagger a_ra_s$ (+ constant). For $N$ spin-orbitals the number of
terms in this Hamiltonian grows as""",
         opts=[r"$O(N^4)$, from the two-electron integrals $h_{pqrs}$", r"$O(N^2)$, from the one-electron integrals only",
               r"$O(2^N)$, like the dimension of the Fock space", r"$O(N)$, one term per orbital"],
         sol=r"""The one-body part has $N^2$ coefficients $h_{pq}$, the two-body part $N^4$ coefficients $h_{pqrs}$ (four free indices). Permutation symmetries
reduce the count by a constant factor (about 8) but not the power, so the total is $O(N^4)$. After a fermion-to-qubit mapping each
term becomes a few Pauli strings, so the number of Pauli terms --- and the measurement cost of VQE --- also grows like $N^4$.
The Hilbert space is $2^N$-dimensional, but the Hamiltonian itself has only polynomially many terms; that is what makes
$\braket{H} = \sum_kc_k\braket{P_k}$ practical. H$_2$ (STO-3G): 15 Pauli terms."""),

    dict(id="A13", kind="mcq",
         q=r"""For a Pauli operator $P$ ($P^2 = I$), $e^{i\theta P} = \cos\theta\,I + i\sin\theta\,P$. Hence $e^{i\pi X/4}$ equals""",
         opts=[r"$\dfrac{1}{\sqrt2}\begin{pmatrix}1 & i\\ i & 1\end{pmatrix}$", r"$\dfrac{1}{\sqrt2}\begin{pmatrix}1 & -i\\ -i & 1\end{pmatrix}$",
               r"$\dfrac{1}{\sqrt2}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$", r"$\begin{pmatrix}0 & i\\ i & 0\end{pmatrix}$"],
         sol=r"""Power series: even powers of $P$ give $I$, odd powers give $P$, so $e^{i\theta P} = \sum_k\frac{(i\theta)^k}{k!}P^k = \cos\theta\,I + i\sin\theta\,P$.
With $\theta = \pi/4$: $\cos\frac\pi4 = \sin\frac\pi4 = \frac{1}{\sqrt2}$, so $e^{i\pi X/4} = \frac{1}{\sqrt2}(I + iX) = \frac{1}{\sqrt2}\begin{psmallmatrix}1 & i\\ i & 1\end{psmallmatrix}$.
The same identity, with $P = X_0X_1X_2Y_3$, is how CUDA-Q's \texttt{exp\_pauli(theta, q, "XXXY")} acts in the notebook's UCC ansatz.
$\frac{1}{\sqrt2}(I - iX)$ would be $e^{-i\pi X/4} = R_X(\pi/2)$."""),

    dict(id="A14", kind="para", passage="A",
         q=r"""The energy function $E(\theta)$ and the exact ground-state energy $E_0$ are""",
         opts=[r"$E(\theta) = -\cos\theta + \tfrac12\sin\theta$ and $E_0 = -\tfrac{\sqrt5}{2} \approx -1.118$",
               r"$E(\theta) = -\cos\theta + \tfrac12\sin\theta$ and $E_0 = -1$",
               r"$E(\theta) = -\cos\tfrac\theta2 + \tfrac12\sin\tfrac\theta2$ and $E_0 = -\tfrac{\sqrt5}{2}$",
               r"$E(\theta) = -\cos\theta + \sin\theta$ and $E_0 = -\sqrt2$"],
         sol=r"""For $\ket\psi = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$: $\braket Z = \cos^2\frac\theta2 - \sin^2\frac\theta2 = \cos\theta$ and
$\braket X = 2\cos\frac\theta2\sin\frac\theta2 = \sin\theta$. By linearity $E(\theta) = -\braket Z + \tfrac12\braket X = -\cos\theta + \tfrac12\sin\theta$.
Eigenvalues of $\begin{psmallmatrix}-1&1/2\\1/2&1\end{psmallmatrix}$: $\lambda^2 - 1 - \tfrac14 = 0$, $\lambda = \pm\sqrt{5/4} = \pm\frac{\sqrt5}{2}$, so $E_0 = -1.1180$.
Consistency: $E(\theta) = -\sqrt{1 + \tfrac14}\cos(\theta + \varphi)$ has minimum $-\frac{\sqrt5}{2}$ --- the one-parameter $R_Y$ ansatz reaches $E_0$ exactly,
because the ground state of a real $2\times2$ Hamiltonian is a real vector."""),

    dict(id="A15", kind="para", passage="A",
         q=r"""The optimal angle $\theta^*$, and the energy error $E(0) - E_0$ of the starting point $\theta = 0$ (the state $\ket0$), are""",
         opts=[r"$\theta^* = -\arctan\tfrac12 \approx -0.464$ rad; error $\approx 0.118$",
               r"$\theta^* = +\arctan\tfrac12 \approx 0.464$ rad; error $\approx 0.118$",
               r"$\theta^* = -\arctan 2 \approx -1.107$ rad; error $\approx 0.118$",
               r"$\theta^* = -\arctan\tfrac12 \approx -0.464$ rad; error $0$"],
         sol=r"""$E'(\theta) = \sin\theta + \tfrac12\cos\theta = 0 \Rightarrow \tan\theta = -\tfrac12$. The minimum needs $E'' = \cos\theta - \tfrac12\sin\theta > 0$: with
$\theta^* = -\arctan\tfrac12 = -0.4636$, $\cos\theta^* = 0.894 > 0$ and $\sin\theta^* = -0.447$, so $E'' > 0$ and
$E(\theta^*) = -0.894 - 0.224 = -1.118 = E_0$. (The other root $\pi - 0.4636$ is the maximum $+1.118$.)
$E(0) = -1$, so the starting error is $-1 - (-1.118) = 0.118$. The negative angle mixes in $\ket1$ with a \emph{negative} amplitude because the
off-diagonal element $+\tfrac12$ is positive."""),

    dict(id="A16", kind="mcq",
         q=r"""The Hermitian matrix $M = \begin{pmatrix}3 & 1 - 2i\\ 1 + 2i & -1\end{pmatrix}$ written as $aI + bX + cY + dZ$ is""",
         opts=[r"$I + X + 2Y + 2Z$", r"$I + X - 2Y + 2Z$", r"$2I + X + 2Y + Z$", r"$3I + X + 2Y - Z$"],
         sol=r"""Use $c_P = \tfrac12\operatorname{tr}(PM)$ (Paulis are orthogonal: $\operatorname{tr}(PQ) = 2\delta_{PQ}$).
$a = \tfrac12(3 - 1) = 1$; $d = \tfrac12(3 + 1) = 2$; $b = \tfrac12(M_{01} + M_{10}) = \tfrac12\cdot2 = 1$; $c = \tfrac12\operatorname{tr}(YM) = \tfrac12(-iM_{10} + iM_{01}) = \tfrac12(-i(1 + 2i) + i(1 - 2i)) = \tfrac12(2 + 2) = 2$.
Check: $X + 2Y = \begin{psmallmatrix}0 & 1 - 2i\\ 1 + 2i & 0\end{psmallmatrix}$ \checkmark{} and $I + 2Z = \operatorname{diag}(3, -1)$ \checkmark{}. Real coefficients are guaranteed because $M$ is Hermitian --- the
reason the notebook rejects complex coefficients in \texttt{CUSTOM\_HAMILTONIAN}."""),

    dict(id="A17", kind="mcq",
         q=r"""The diagonal two-qubit Hamiltonian $H = \operatorname{diag}(1, 2, 3, 4)$ (basis $\ket{00}, \ket{01}, \ket{10}, \ket{11}$) written as a sum of Pauli strings is""",
         opts=[r"$2.5\,II - 1.0\,ZI - 0.5\,IZ$", r"$2.5\,II + 1.0\,ZI + 0.5\,IZ$", r"$2.5\,II - 0.5\,ZI - 1.0\,IZ$",
               r"$2.5\,II - 1.0\,ZI - 0.5\,IZ + 0.5\,ZZ$"],
         sol=r"""Coefficients $c_P = \tfrac14\operatorname{tr}(PH)$. On the basis states, $ZI = \operatorname{diag}(1, 1, -1, -1)$, $IZ = \operatorname{diag}(1, -1, 1, -1)$, $ZZ = \operatorname{diag}(1, -1, -1, 1)$.
$c_{II} = \frac{1+2+3+4}{4} = 2.5$; $c_{ZI} = \frac{1+2-3-4}{4} = -1$; $c_{IZ} = \frac{1-2+3-4}{4} = -0.5$; $c_{ZZ} = \frac{1-2-3+4}{4} = 0$.
Check $\ket{11}$: $2.5 + 1 + 0.5 = 4$ \checkmark{}. The first qubit (more significant) changes the energy by 2, so its $Z$ coefficient is $-1$."""),

    dict(id="A18", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} For H$_2$ at large bond length, the restricted Hartree--Fock (single closed-shell determinant) energy lies far above
the exact energy.\\[1mm]
\textbf{Reason (R):} The nuclear--nuclear repulsion $1/R$ tends to zero as $R \to \infty$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} the RHF determinant puts both electrons in the bonding orbital $\sigma_g = (1s_A + 1s_B)/\sqrt2$. Expanded in atomic
orbitals it contains ionic configurations (H$^+$H$^-$) with $50\,\%$ weight, which cost a lot of energy for separated atoms. The exact
state is $\approx \cos\theta\ket{\sigma_g^2} - \sin\theta\ket{\sigma_u^2}$ with $\theta \to \pi/4$: static correlation. (Same structure as the notebook's UCC
state $\cos\theta\ket{1100} - \sin\theta\ket{0011}$.)
\textbf{R is true} but is \emph{not} the explanation: $V_{NN}$ is identical in RHF and the exact calculation and cancels in the error. Answer: (B)."""),

    dict(id="A19", kind="mcq",
         q=r"""For the notebook's H$_2$ Hamiltonian, the Hartree--Fock energy is $E_{\mathrm{HF}} = \braket{1100|H|1100} = -1.116759$ Ha and the FCI energy is
$-1.137284$ Ha. The correlation energy $E_{\mathrm{corr}} = E_{\mathrm{FCI}} - E_{\mathrm{HF}}$ is about""",
         opts=[r"$-20.5$ mHa", r"$+20.5$ mHa", r"$-2.05$ mHa", r"$-1137$ mHa"],
         sol=r"""$E_{\mathrm{corr}} = -1.137284 - (-1.116759) = -0.020525$ Ha $= -20.5$ mHa. It is negative because FCI is variationally
\emph{lower} than HF (HF is one particular state in the same space). $20.5$ mHa $\approx 12.9$ kcal/mol, far above chemical accuracy:
HF alone is not good enough, and VQE has to recover this correlation energy. The UCC parameter does exactly that (see C19--C20)."""),

    dict(id="A20", kind="mcq",
         q=r"""A trial state of H$_2$ has $90\,\%$ weight on the ground state ($E_0 = -1.13728$ Ha) and $10\,\%$ on another eigenstate of the 4-qubit Hamiltonian with energy
$E_1 = -0.53821$ Ha. Its energy expectation and its error above $E_0$ are""",
         opts=[r"$-1.0774$ Ha and $59.9$ mHa", r"$-1.1373$ Ha and $0$ mHa", r"$-1.1971$ Ha and $-59.9$ mHa", r"$-0.8378$ Ha and $299.5$ mHa"],
         sol=r"""$E = \sum_k|c_k|^2E_k = 0.9(-1.13728) + 0.1(-0.53821) = -1.02355 - 0.05382 = -1.07738$ Ha.
Error $E - E_0 = 0.1(E_1 - E_0) = 0.1\times0.59907 = 0.0599$ Ha $= 59.9$ mHa. The error is \emph{linear} in the contaminating weight and is
always non-negative (variational principle). The energy is lower than the energies of all excited states but still above $E_0$."""),
]
