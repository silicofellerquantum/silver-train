"""Section D - Advanced VQE and implementation (Jordan-Wigner, Bravyi-Kitaev, UCC, molecular scaling, shots, noise,
error mitigation, backends, exact references, benchmarking and interpretation of the recorded run).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""\emph{Previously recorded default run of \texttt{VQE.ipynb}} (CUDA-Q 0.16.0, target \texttt{qpp-cpu}, 4 vCPU): Ising chain with $N = 15$, $J = 1$,
$h = 0.5$ (29 Pauli terms); hardware-efficient ansatz with 2 layers (45 parameters, 73 gates); COBYLA stopped with ``the objective function has
been evaluated MAXFUN times'' after 300 evaluations; $E_{\mathrm{VQE}} = -15.0533898284$, exact $E_0 = -15.0825179883$; quantum execution time
$69.10$ s; the optimised state gave $\ket{000\dots0}$ with probability $0.7005$. The product state $\ket{000\dots0}$ itself has energy $-14$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""With the Jordan--Wigner transformation $a_j^\dagger = \big(\prod_{k<j}Z_k\big)\,\tfrac12(X_j - iY_j)$ (qubit $\ket1$ = occupied), the creation operator $a_2^\dagger$ on
4 spin-orbitals becomes""",
         opts=[r"$\tfrac12\,Z_0Z_1(X_2 - iY_2)$", r"$\tfrac12(X_2 - iY_2)$", r"$\tfrac12\,Z_3(X_2 - iY_2)$", r"$\tfrac12\,Z_0Z_1(X_2 + iY_2)$"],
         sol=r"""$\tfrac12(X - iY) = \begin{psmallmatrix}0&0\\1&0\end{psmallmatrix} = \ket1\bra0$ fills an empty orbital. The string $Z_0Z_1$ counts the parity of the orbitals \emph{before}
$j = 2$ and supplies the sign $(-1)^{n_0 + n_1}$ that makes operators on different orbitals \emph{anti}commute. Result: $\tfrac12 Z_0Z_1X_2 - \tfrac i2Z_0Z_1Y_2$
(OpenFermion gives the same: \texttt{ZZXI}: $0.5$, \texttt{ZZYI}: $-0.5i$). Without the string (option 2) $\{a_0, a_2^\dagger\}$ would not vanish;
$\tfrac12(X + iY)$ is the annihilation operator."""),

    dict(id="D2", kind="mcq",
         q=r"""Under Jordan--Wigner, the hopping term $a_0^\dagger a_1 + a_1^\dagger a_0$ between neighbouring spin-orbitals 0 and 1 becomes""",
         opts=[r"$\tfrac12(X_0X_1 + Y_0Y_1)$", r"$\tfrac12(X_0X_1 - Y_0Y_1)$", r"$X_0X_1$", r"$\tfrac12(Z_0 + Z_1)$"],
         sol=r"""$a_0^\dagger = \tfrac12(X_0 - iY_0)$ and $a_1 = Z_0\,\tfrac12(X_1 + iY_1)$ (the Jordan--Wigner string of orbital 1 is $Z_0$). Since
$(X - iY)Z = XZ - iYZ = -iY + X$, we get $(X_0 - iY_0)Z_0 = X_0 - iY_0$, so
\[ a_0^\dagger a_1 = \tfrac14(X_0 - iY_0)(X_1 + iY_1) = \tfrac14\big(X_0X_1 + Y_0Y_1 + iX_0Y_1 - iY_0X_1\big). \]
The Hermitian conjugate $a_1^\dagger a_0$ has the opposite imaginary parts, so the sum is $\tfrac12(X_0X_1 + Y_0Y_1)$. It swaps $\ket{01} \leftrightarrow \ket{10}$ (one electron
hops) and annihilates $\ket{00}$, $\ket{11}$. Verified with OpenFermion (\texttt{XX}: $0.5$, \texttt{YY}: $0.5$). $\tfrac12(X_0X_1 - Y_0Y_1)$ would create or destroy \emph{pairs}."""),

    dict(id="D3", kind="mcq",
         q=r"""The Pauli weight (number of non-identity factors) of the image of $a_7^\dagger$ on 8 spin-orbitals is ... under Jordan--Wigner and ... under Bravyi--Kitaev.""",
         opts=[r"$8$; $4$", r"$1$; $4$", r"$8$; $8$", r"$4$; $8$"],
         sol=r"""Jordan--Wigner stores occupations locally but the parity of all previous orbitals non-locally: $a_7^\dagger = Z_0\cdots Z_6\,\tfrac12(X_7 - iY_7)$, weight $7 + 1 = 8$ --- $O(N)$.
Bravyi--Kitaev stores partial sums of occupations in a binary-tree pattern so that both occupation and parity information are spread over only
$O(\log N)$ qubits; for $N = 8$ the strings of $a_7^\dagger$ have weight $\log_28 + 1 = 4$ (OpenFermion: maximum weight 4). Lower weight means
shorter circuits for $e^{i\theta P}$ on hardware. The notebook uses neither mapping explicitly; its H$_2$ coefficients are already Jordan--Wigner."""),

    dict(id="D4", kind="match",
         q=r"""Match each fermion-to-qubit encoding (List-I) with its characteristic property (List-II).""",
         match=([r"Jordan--Wigner", r"Bravyi--Kitaev", r"parity mapping", r"qubit tapering"],
                [r"qubit $j$ stores the parity $n_0 \oplus\dots\oplus n_j$; total parity and $S_z$-parity sit on single qubits, so two qubits can be removed",
                 r"symmetries ($\mathbb Z_2$ stabilisers commuting with $H$) are used to remove qubits from any encoding",
                 r"qubit $j$ stores the occupation $n_j$; strings of length $O(N)$",
                 r"qubits store partial sums of occupations on a binary tree; strings of length $O(\log N)$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: Jordan--Wigner: occupation basis, $Z$-strings (3).
\item Q: Bravyi--Kitaev: binary-tree partial sums, logarithmic weight (4).
\item R: parity mapping: cumulative parities; with interleaved/blocked spin ordering the electron-number parity and spin parity become single-qubit
$Z$'s, which allows the well-known reduction of H$_2$ from 4 to 2 qubits (1).
\item S: tapering (Bravyi et al.\ 2017) is a general symmetry reduction applied after any encoding (2).
\end{itemize}"""),

    dict(id="D5", kind="mcq",
         q=r"""For 4 modes, the Bravyi--Kitaev qubit values are $b = Bn \bmod 2$ with $B = \begin{psmallmatrix}1&0&0&0\\1&1&0&0\\0&0&1&0\\1&1&1&1\end{psmallmatrix}$ and $n$ the occupation vector.
The Hartree--Fock occupation $n = (1, 1, 0, 0)$ is encoded as""",
         opts=[r"$\ket{1000}$", r"$\ket{1100}$", r"$\ket{1101}$", r"$\ket{0100}$"],
         sol=r"""$b_0 = n_0 = 1$; $b_1 = n_0 + n_1 = 2 \equiv 0$; $b_2 = n_2 = 0$; $b_3 = n_0 + n_1 + n_2 + n_3 = 2 \equiv 0$. So $\ket{1000}$.
Check with OpenFermion: the Bravyi--Kitaev Hamiltonian of H$_2$ has $\braket{1000|H_{\mathrm{BK}}|1000} = E_{\mathrm{HF}}$, while $\braket{1100|H_{\mathrm{BK}}|1100}$ is a different
energy. The same physical state has different bitstrings in different encodings --- an ansatz must start from the encoding-correct reference."""),

    dict(id="D6", kind="mcq",
         q=r"""Counting excitations in spin-orbitals without spin restrictions, LiH in STO-3G (12 spin-orbitals, 4 electrons) has how many single and double
excitations from the Hartree--Fock determinant?""",
         opts=[r"$32$ singles and $168$ doubles", r"$32$ singles and $1024$ doubles", r"$8$ singles and $28$ doubles", r"$48$ singles and $168$ doubles"],
         sol=r"""Occupied spin-orbitals $n_{\mathrm{occ}} = 4$, virtual $n_{\mathrm{virt}} = 12 - 4 = 8$. Singles: choose one occupied and one virtual: $4\times8 = 32$.
Doubles: choose two occupied and two virtual: $\binom42\binom82 = 6\times28 = 168$. UCCSD has one parameter per excitation (fewer if spin and
spatial symmetry are imposed). For H$_2$: $\binom22\binom22 = 1$ double --- the notebook's single parameter."""),

    dict(id="D7", kind="mcq",
         q=r"""Under Jordan--Wigner, the anti-Hermitian double-excitation generator $a_3^\dagger a_2^\dagger a_1a_0 - a_0^\dagger a_1^\dagger a_2a_3$ becomes a sum of how many Pauli strings,
and what does the notebook's UCC kernel use?""",
         opts=[r"8 strings with coefficients $\pm i/8$; the notebook keeps the single string $\texttt{XXXY}$",
               r"1 string; the notebook uses all of it",
               r"16 strings; the notebook keeps 8",
               r"4 strings with coefficients $\pm\tfrac12$; the notebook keeps $\texttt{XXYY}$"],
         sol=r"""Each of the four operators contributes $\tfrac12(X \mp iY)$, giving $2^4 = 16$ products; the Hermitian conjugate cancels half of them, leaving 8 strings
with an odd number of $Y$'s and coefficients $\pm i/8$ (OpenFermion: $\texttt{XXXY}, \texttt{XXYX}, \texttt{XYXX}, \texttt{YXXX}, \texttt{XYYY}, \dots$). Full UCC
would exponentiate all 8 (they commute, so $e^{\theta(T - T^\dagger)}$ factorises exactly). The notebook keeps only $e^{i\theta\,\texttt{XXXY}}$: starting from $\ket{1100}$ it
already rotates within $\mathrm{span}\{\ket{1100}, \ket{0011}\}$ (B11), which contains the exact ground state (C9), so nothing is lost for H$_2$."""),

    dict(id="D8", kind="match",
         q=r"""Match each practical VQE difficulty (List-I) with a standard remedy (List-II).""",
         match=([r"barren plateaus (vanishing gradients for deep random circuits)", r"shot noise in energy estimates",
                 r"gate errors on hardware", r"local minima of $E(\boldsymbol\theta)$"],
                [r"error mitigation, e.g.\ zero-noise extrapolation", r"chemically motivated ansatz, local cost functions, careful (e.g.\ Hartree--Fock) initialisation",
                 r"several restarts / good initial parameters", r"measurement grouping and more shots for terms with large $|c_k|$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: barren plateaus (McClean et al.\ 2018): gradient variance decays exponentially with $n$ for random deep ansätze; structured ansätze, local cost
functions and good initialisation help (2).
\item Q: shot noise $\propto \sqrt{\mathrm{Var}/S}$: group commuting terms, allocate shots by $|c_k|$ (4).
\item R: gate noise biases the energy upwards; mitigation such as zero-noise extrapolation (D12) (1).
\item S: local minima: multiple starts, Hartree--Fock or INTERP-like initialisation (3).
\end{itemize}"""),

    dict(id="D9", kind="mcq",
         q=r"""An energy estimator has a per-shot standard deviation of $0.1$ Ha. How many shots are needed for a standard error of $1.6$ mHa (chemical accuracy)?""",
         opts=[r"about $3.9\times10^3$", r"about $63$", r"about $2.6\times10^{-4}$", r"about $1.6\times10^{5}$"],
         sol=r"""$\mathrm{SE} = \sigma/\sqrt S \le 0.0016 \Rightarrow S \ge (0.1/0.0016)^2 = 62.5^2 = 3906$. $63$ forgets to square. Precision costs quadratically: halving
the error needs four times the shots, and every optimizer step needs such an estimate --- the main practical cost of VQE on hardware."""),

    dict(id="D10", kind="mcq",
         q=r"""Under complete depolarisation the state becomes $\rho = I/2^n$. For the notebook's 4-qubit H$_2$ Hamiltonian the measured energy would then be""",
         opts=[r"$-0.0971$ Ha, the coefficient of $\texttt{IIII}$", r"$-1.1373$ Ha", r"$0$ Ha", r"$-1.1168$ Ha"],
         sol=r"""$E = \operatorname{tr}(\rho H) = 2^{-n}\operatorname{tr}H = \sum_kc_k\,2^{-n}\operatorname{tr}P_k = c_{IIII}$, since every non-identity Pauli string is traceless. Here $-0.0970663$ Ha.
Noise thus pulls the energy \emph{up} towards $c_{IIII}$, far above $E_{\mathrm{FCI}} = -1.1373$; partial depolarisation interpolates between the ideal
value and this limit."""),

    dict(id="D11", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} With a finite number of shots, an estimated VQE energy can never fall below $E_0$.\\[1mm]
\textbf{Reason (R):} For every density matrix $\rho$ (pure or mixed, noisy or not), $\operatorname{tr}(\rho H) \ge E_0$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $\rho = \sum_jp_j\ket{\psi_j}\bra{\psi_j}$ with $p_j \ge 0$, $\sum p_j = 1$, so $\operatorname{tr}(\rho H) = \sum_jp_j\braket{\psi_j|H|\psi_j} \ge E_0$: the variational principle survives noise.
\textbf{A is false:} a finite-shot \emph{estimate} fluctuates around $\operatorname{tr}(\rho H)$ with standard error $\sigma/\sqrt S$, and can fall below $E_0$ when the true value is
close to it. The notebook warns about this (``only shot noise can push single evaluations slightly below''). Answer: (D)."""),

    dict(id="D12", kind="mcq",
         q=r"""Zero-noise extrapolation: an energy measured at noise scale factors $\lambda = 1$ and $\lambda = 3$ gives $-1.10$ Ha and $-1.00$ Ha. A linear extrapolation to
$\lambda = 0$ gives""",
         opts=[r"$-1.15$ Ha", r"$-1.20$ Ha", r"$-1.10$ Ha", r"$-1.05$ Ha"],
         sol=r"""Slope: $\frac{-1.00 - (-1.10)}{3 - 1} = 0.05$ Ha per unit $\lambda$. $E(0) = E(1) - 0.05 = -1.15$ Ha. Extrapolation removes the bias of noise but amplifies
statistical errors, and the extrapolated value carries \emph{no} variational guarantee: it can land below $E_0$."""),

    dict(id="D13", kind="mcq",
         q=r"""\texttt{VQE.ipynb} is run with \texttt{BACKEND = "gpu"} on a machine where \texttt{cudaq.num\_available\_gpus()} returns 0. What happens?""",
         opts=[r"\texttt{resolve\_backend} raises a \texttt{RuntimeError} explaining that no NVIDIA GPU is visible; no GPU target is selected",
               r"CUDA-Q silently falls back to \texttt{qpp-cpu}",
               r"The notebook runs on the GPU emulator \texttt{tensornet} instead",
               r"The run starts and crashes later with an out-of-memory error"],
         sol=r"""The shared utilities check \texttt{if backend in ("gpu", "multi-gpu") and num\_gpus == 0} and raise \texttt{RuntimeError("BACKEND='gpu' was requested but CUDA-Q
sees no NVIDIA GPU ...")}. GPU targets are never selected without a GPU because selecting \texttt{tensornet} on such a machine aborted the
Python process (README, known limitations). Only \texttt{BACKEND = "auto"} falls back to the CPU. GPU paths are implemented but were not executed."""),

    dict(id="D14", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} For the Ising chain, the notebook computes its reference energy by sparse diagonalisation for every $N$.\\[1mm]
\textbf{Statement II:} For \texttt{"H2"} and \texttt{"custom"} Hamiltonians, an exact reference is computed only up to \texttt{REFERENCE\_MAX\_QUBITS = 18} qubits.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} the Ising reference uses the free-fermion formula \texttt{tfim\_exact\_energy} for any $N$; sparse diagonalisation is only an
\emph{additional cross-check} for $N \le 16$ (it printed $-15.0825179883$, identical).
\textbf{II is true:} \texttt{elif n\_qubits <= REFERENCE\_MAX\_QUBITS: reference\_energy = exact\_ground\_energy\_sparse(...)}; above that, ``not computed''. Answer: (D)."""),

    dict(id="D15", kind="mcq",
         q=r"""The notebook's exact Ising reference maps the chain to free fermions and computes $E_0 = -\tfrac12\sum|\lambda(iA)|$ for a real antisymmetric matrix $A$. For
$N = 100$ qubits, $A$ has size""",
         opts=[r"$200\times200$", r"$100\times100$", r"$2^{100}\times2^{100}$", r"$4\times4$"],
         sol=r"""Each spin becomes two Majorana operators ($a_j$, $b_j$), so $A$ is $2N\times2N$: $200\times200$, diagonalised in milliseconds. The Hamiltonian matrix
would be $2^{100}\times2^{100}$ --- impossible to store. Exactly solvable models give VQE benchmarks at any size; molecules generally do not."""),

    dict(id="D16", kind="mcq",
         q=r"""In the recorded run the quantum execution time was $69.10$ s for $300$ energy evaluations, while the notebook's pre-run estimate was $\sim102.9$ s.
The average time per evaluation, and the ratio estimate/measured cell time ($70.1$ s), are about""",
         opts=[r"$0.23$ s and $1.5$", r"$0.23$ s and $0.68$", r"$69$ s and $1.5$", r"$0.0032$ s and $1.5$"],
         sol=r"""$69.10/300 = 0.230$ s per \texttt{cudaq.observe} call (73-gate circuit, 29 terms, 15 qubits). $102.9/70.1 = 1.47$: the estimate was about $50\,\%$ high,
within the stated $\pm3\times$ (it assumed an extra state pass for the energy, ``x2 rough allowance''). $0.0032$ s is the time per \emph{gate}
($0.230/73$), not per evaluation; $0.68$ is the inverse ratio."""),

    dict(id="D17", kind="mcq",
         q=r"""For LiH (STO-3G, 12 spin-orbitals under Jordan--Wigner), freezing the doubly occupied Li $1s$ core orbital removes how many qubits?""",
         opts=[r"$2$, leaving $10$ qubits", r"$1$, leaving $11$ qubits", r"$4$, leaving $8$ qubits", r"$0$; frozen orbitals still need qubits"],
         sol=r"""One spatial orbital = two spin-orbitals ($\alpha$, $\beta$) = two qubits under Jordan--Wigner. Freezing it (always occupied, absorbed into the constant and
one-body terms) leaves $12 - 2 = 10$ qubits and 2 active electrons. Further symmetry tapering and removing weakly occupied virtual orbitals reduce the
count further; every such reduction is an approximation choice except tapering, which is exact."""),

    dict(id="D18", kind="mcq",
         q=r"""Generated with PySCF and OpenFermion (STO-3G, Jordan--Wigner, bond length 1.595 \AA), the LiH qubit Hamiltonian has how many Pauli terms, compared with
15 for H$_2$?""",
         opts=[r"$631$", r"$30$", r"$4096$", r"$144$"],
         sol=r"""The number of terms grows like $N^4$ (A12): from $N = 4$ to $N = 12$ spin-orbitals the naive factor is $3^4 = 81$; symmetries remove many, and the
generated Hamiltonian has 631 Pauli terms (checked in \texttt{verify\_answers.py}). $4096 = 2^{12}$ is the dimension of the state, not a term count.
More terms mean more measurement settings and shots per energy evaluation, the dominant cost of molecular VQE."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""Measured from the product state $\ket{000\dots0}$ ($E = -14$), what fraction of the remaining energy gap to $E_0$ did VQE recover?""",
         opts=[r"$97.3\,\%$", r"$99.8\,\%$", r"$70.1\,\%$", r"$2.7\,\%$"],
         sol=r"""Gap from the product state to $E_0$: $-14 - (-15.0825180) = 1.0825180$. Recovered: $-14 - (-15.0533898) = 1.0533898$.
Fraction $1.0533898/1.0825180 = 0.973$. The relative error $0.193\,\%$ (C15) looks tiny because most of the energy ($-14$) comes from the trivial
ordered state; measured against this baseline the ansatz still misses $2.7\,\%$ of the quantum-fluctuation energy. $99.8\,\%$ is $1 - 0.193\,\%$."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Which change is most likely to reduce the remaining error of this run?""",
         opts=[r"More evaluations (the budget was exhausted) and, if needed, more ansatz layers",
               r"More final sampling shots",
               r"Switching from the free-fermion reference to sparse diagonalisation",
               r"Reducing \texttt{MAX\_EVALUATIONS} to stop earlier"],
         sol=r"""COBYLA stopped because of \texttt{MAXFUN}, not because of a convergence criterion, so the optimiser had not finished; a 2-layer ansatz may also lack
expressiveness (the notebook suggests more layers and evaluations, e.g.\ 4 layers and 800 evaluations for the harder critical point). The final
sampling only displays bitstrings; it does not change the energy. Both references are exact and agree to $10^{-10}$. Fewer evaluations can only
make the error larger or equal."""),
]
