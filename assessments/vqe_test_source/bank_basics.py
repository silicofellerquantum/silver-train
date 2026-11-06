"""Section C - VQE fundamentals (variational principle, Rayleigh-Ritz, ansatz, Hartree-Fock, expectation values,
Hamiltonian decomposition, cost function, optimizers, convergence, notebook code).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""In the notebook's H$_2$ example the UCC state $\ket{\psi(\theta)} = \cos\theta\ket{1100} - \sin\theta\ket{0011}$ only involves two basis states. Within
this two-dimensional space the Hamiltonian is the real $2\times2$ matrix
\[ \begin{pmatrix} H_{11} & H_{12}\\ H_{12} & H_{22}\end{pmatrix} = \begin{pmatrix} -1.116759 & 0.181210\\ 0.181210 & 0.462618\end{pmatrix}\ \text{Ha}, \]
with $H_{11} = \braket{1100|H|1100}$, $H_{22} = \braket{0011|H|0011}$, $H_{12} = \braket{1100|H|0011}$ (computed from the notebook's 15 Pauli terms), so that
$E(\theta) = H_{11}\cos^2\theta + H_{22}\sin^2\theta - H_{12}\sin2\theta$.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""A VQE run with \emph{exact} expectation values (no shots, no noise) reports $E(\boldsymbol\theta) = -15.10$ for the 15-qubit Ising chain whose exact ground
energy is $E_0 = -15.0825$. The most likely explanation is""",
         opts=[r"an error in the Hamiltonian, the reference energy or the state normalisation --- a correct calculation cannot go below $E_0$",
               r"the optimizer found a state better than the exact ground state",
               r"shot noise pushed the estimate below $E_0$",
               r"a deeper ansatz always produces energies below the exact value"],
         sol=r"""Variational principle: for every normalised state $\braket{\psi|H|\psi} \ge E_0$ (A7). Exact expectation values of a normalised circuit state
(unitary gates, B8) therefore can never fall below $E_0$. A value below it signals a bug: a wrong Hamiltonian term or sign, a wrong
reference, or an expectation value of an unnormalised vector. With shots, single estimates \emph{can} fluctuate below $E_0$, but the
question excludes shots. This is why the notebook prints ``the VQE energy must stay $\ge E_0$''."""),

    dict(id="C2", kind="mcq",
         q=r"""In the Rayleigh--Ritz method, $H$ is diagonalised inside an $m$-dimensional trial subspace, giving Ritz values $\varepsilon_1 \le \dots \le \varepsilon_m$. Which
statement is correct?""",
         opts=[r"$\varepsilon_k \ge E_{k-1}$ for every $k$: each Ritz value is an upper bound to the corresponding exact eigenvalue",
               r"Only $\varepsilon_1$ is bounded; the other Ritz values can lie below the exact eigenvalues",
               r"The Ritz values are lower bounds to the exact eigenvalues",
               r"The Ritz values are exact whenever $m \ge 2$"],
         sol=r"""Write $P$ for the projector onto the trial space. By the Courant--Fischer min--max theorem the $k$-th eigenvalue of $PHP$ restricted to the
subspace is $\ge$ the $k$-th eigenvalue of $H$ (Hylleraas--Undheim--MacDonald theorem): $\varepsilon_k \ge E_{k-1}$ (labelling exact levels
$E_0 \le E_1 \le \dots$). For $k = 1$ this is the variational principle. Enlarging the subspace can only lower every Ritz value. The passage for
C19 and C20 is a Rayleigh--Ritz calculation with $m = 2$; there $\varepsilon_1$ happens to be exact because the ground state lies in the subspace."""),

    dict(id="C3", kind="mcq",
         q=r"""The notebook's hardware-efficient ansatz uses $L$ layers of [$R_Y$ on every qubit, CNOT ladder over neighbours] plus a final $R_Y$ layer. For
$N = 10$ qubits and $L = 3$, the numbers of parameters, CNOTs and gates are""",
         opts=[r"$40$, $27$ and $67$", r"$30$, $27$ and $57$", r"$40$, $30$ and $70$", r"$40$, $9$ and $49$"],
         sol=r"""Parameters: one angle per $R_Y$: $(L + 1)N = 4\cdot10 = 40$. CNOTs: each ladder has $N - 1 = 9$ CNOTs, $L$ ladders: $27$.
Gates: $40 + 27 = 67$ (the notebook's estimate \texttt{num\_parameters + ANSATZ\_LAYERS * (n\_qubits - 1)}).
Default ($N = 15$, $L = 2$): $45$ parameters, $28$ CNOTs, $73$ gates --- exactly the recorded statistics \texttt{ry 45, cx 28}."""),

    dict(id="C4", kind="mcq",
         q=r"""The UCC kernel starts with \texttt{x(qubits[0]); x(qubits[1])}. In the Jordan--Wigner picture with spin-orbitals ordered (orbital 0 $\alpha$, orbital 0 $\beta$,
orbital 1 $\alpha$, orbital 1 $\beta$), this prepares""",
         opts=[r"the Hartree--Fock determinant $\ket{1100}$: both electrons in the lower (bonding) spatial orbital, one with each spin",
               r"the state with both electrons in the antibonding orbital",
               r"an equal superposition of all 2-electron determinants",
               r"a triplet state with both electrons spin-up"],
         sol=r"""Under Jordan--Wigner, qubit $j$ in $\ket1$ means spin-orbital $j$ is occupied. $X$ on qubits 0 and 1 turns $\ket{0000}$ into $\ket{1100}$: spin-orbitals
0 ($\sigma_g\alpha$) and 1 ($\sigma_g\beta$) occupied. This closed-shell determinant is the restricted Hartree--Fock state; its energy
$\braket{1100|H|1100} = -1.116759$ Ha is the HF energy (C8). VQE starts there ($\theta = 0$) and the double excitation adds the correlation."""),

    dict(id="C5", kind="mcq",
         q=r"""Let $\ket u$, $\ket v$ be orthonormal, $H_{uu} = a$, $H_{vv} = c$, $H_{uv} = H_{vu} = b$ (real). For the trial state $\ket{\psi(\theta)} = \cos\theta\ket u - \sin\theta\ket v$, the energy is""",
         opts=[r"$a\cos^2\theta + c\sin^2\theta - b\sin2\theta$", r"$a\cos^2\theta + c\sin^2\theta + b\sin2\theta$",
               r"$a\cos\theta + c\sin\theta - b$", r"$a\cos^2\theta - c\sin^2\theta$"],
         sol=r"""$E = \braket{\psi|H|\psi} = \cos^2\theta\,H_{uu} + \sin^2\theta\,H_{vv} - \sin\theta\cos\theta(H_{uv} + H_{vu}) = a\cos^2\theta + c\sin^2\theta - 2b\sin\theta\cos\theta$,
and $2\sin\theta\cos\theta = \sin2\theta$. The minus sign comes from the $-\sin\theta$ amplitude. Minimising: $E'(\theta) = (c - a)\sin2\theta - 2b\cos2\theta = 0$,
i.e.\ $\tan2\theta^* = \frac{2b}{c - a}$, and the minimum equals the lower eigenvalue of $\begin{psmallmatrix}a&b\\b&c\end{psmallmatrix}$."""),

    dict(id="C6", kind="match",
         q=r"""Match each VQE ingredient (List-I) with the code in \texttt{VQE.ipynb} that implements it (List-II).""",
         match=([r"Hartree--Fock reference state", r"energy $E(\boldsymbol\theta) = \braket{\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)}$",
                 r"classical parameter update", r"Hamiltonian as a sum of Pauli strings"],
                [r'\texttt{minimize(energy, initial\_theta, method="COBYLA", ...)}',
                 r"\texttt{coefficient * cudaq.SpinOperator.from\_word(word)}, summed",
                 r"\texttt{x(qubits[0]); x(qubits[1])}",
                 r"\texttt{cudaq.observe(kernel, hamiltonian, *args).expectation()}"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: two $X$ gates occupy spin-orbitals 0 and 1 (3).
\item Q: \texttt{cudaq.observe} evaluates $\sum_kc_k\braket{P_k}$ for the ansatz kernel; \texttt{.expectation()} returns the number (4).
\item R: SciPy's \texttt{minimize} proposes new angles from the returned energies (1).
\item S: \texttt{build\_spin\_operator} turns each \texttt{(coefficient, word)} pair into a \texttt{SpinOperator} and adds them (2).
\end{itemize}"""),

    dict(id="C7", kind="mcq",
         q=r"""$H = 0.5\,Z_0 - 0.3\,Z_1 + 0.2\,X_0X_1$. A trial state has $\braket{Z_0} = 0.6$, $\braket{Z_1} = -0.2$ and $\braket{X_0X_1} = 0.5$. Its energy is""",
         opts=[r"$0.46$", r"$0.34$", r"$0.26$", r"$1.3$"],
         sol=r"""Linearity: $E = 0.5(0.6) - 0.3(-0.2) + 0.2(0.5) = 0.30 + 0.06 + 0.10 = 0.46$. Each $\braket{P_k}$ is estimated separately (or in commuting groups) and the
energy is the coefficient-weighted sum --- exactly what \texttt{cudaq.observe} does term by term."""),

    dict(id="C8", kind="mcq",
         q=r"""Only the identity and the $Z$-type terms of the notebook's H$_2$ Hamiltonian contribute to $\braket{1100|H|1100}$. With coefficients
$c_{IIII} = -0.097066$, $c_{ZIII} = c_{IZII} = 0.171413$, $c_{IIZI} = c_{IIIZ} = -0.223432$, $c_{ZZII} = 0.168689$, $c_{ZIZI} = c_{IZIZ} = 0.120625$,
$c_{ZIIZ} = c_{IZZI} = 0.165928$, $c_{IIZZ} = 0.174413$, the Hartree--Fock energy is""",
         opts=[r"$-1.116759$ Ha", r"$-1.137284$ Ha", r"$-0.097066$ Ha", r"$+0.462618$ Ha"],
         sol=r"""On $\ket{1100}$: $Z_0 = Z_1 = -1$ (occupied), $Z_2 = Z_3 = +1$. The $XY$-type terms flip bits and have zero diagonal element.
\[ E_{\mathrm{HF}} = -0.097066 - 2(0.171413) - 2(0.223432) + 0.168689 - 2(0.120625) - 2(0.165928) + 0.174413 . \]
$= -0.097066 - 0.342826 - 0.446863 + 0.168689 - 0.241250 - 0.331856 + 0.174413 = -1.116759$ Ha.
(Signs: $Z_0Z_1 = (+1)$, $Z_2Z_3 = (+1)$, mixed pairs $(-1)$.) $-1.137284$ is FCI; $+0.462618$ is $\braket{0011|H|0011}$."""),

    dict(id="C9", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} The notebook's one-parameter UCC ansatz can reach the exact (FCI) ground energy of its H$_2$ Hamiltonian.\\[1mm]
\textbf{Reason (R):} The exact ground state of that Hamiltonian lies in $\mathrm{span}\{\ket{1100}, \ket{0011}\}$, and $\cos\theta\ket{1100} - \sin\theta\ket{0011}$ covers every
real normalised vector of that span (up to sign).""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} exact diagonalisation of the $16\times16$ matrix gives a ground state with weights $0.98733$ on $\ket{1100}$ and $0.01267$ on $\ket{0011}$
and nothing else (particle-number, spin and spatial symmetry forbid single excitations and the other determinants). Any real unit vector
$(\cos\theta, -\sin\theta)$ in that plane is reached by some $\theta$.
\textbf{A follows:} at $\theta^* = 0.112783$ the energy equals $-1.137283834$ Ha to $10^{-15}$ (NumPy and CUDA-Q). R explains A. Answer: (A)."""),

    dict(id="C10", kind="mcq",
         q=r"""For the default Ising chain ($N = 15$, $J = 1$, $h = 0.5$, $H = -J\sum Z_iZ_{i+1} - h\sum X_i$), the energy of $\ket{00\dots0}$ --- close to where the
hardware-efficient ansatz starts, since its initial angles are drawn from $\mathcal N(0, 0.1^2)$ --- is""",
         opts=[r"$-14$", r"$-15$", r"$-21.5$", r"$-7.5$"],
         sol=r"""On $\ket{0\dots0}$ every $Z_iZ_{i+1} = +1$: $14$ bonds give $-14$. $\braket{X_i} = 0$ for a computational basis state. So $E = -14$, compared with
$E_0 = -15.0825$: the field term contributes about $-1.08$ in the true ground state through small admixtures of flipped spins.
Small random angles start the optimizer close to this classical ordered state (not exactly at it, to break symmetries)."""),

    dict(id="C11", kind="mcq",
         q=r"""Each of the 45 angles of the default hardware-efficient ansatz enters exactly one $R_Y$ gate. The number of circuit evaluations for one exact
parameter-shift gradient, and for one forward finite-difference gradient (L-BFGS-B), are""",
         opts=[r"$90$ and $46$", r"$45$ and $46$", r"$90$ and $90$", r"$2$ and $46$"],
         sol=r"""Parameter shift for a gate $e^{-i\theta Y/2}$: $\partial_\theta E = \tfrac12[E(\theta + \tfrac\pi2) - E(\theta - \tfrac\pi2)]$ --- 2 evaluations per parameter, exact: $2\times45 = 90$.
Forward differences: $E(\boldsymbol\theta)$ once plus one shifted evaluation per parameter: $45 + 1 = 46$ (the notebook's table: ``N\_params + 1
runs per gradient''), approximate and fragile under shot noise. COBYLA (default) uses no gradients at all."""),

    dict(id="C12", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} COBYLA needs no gradient evaluations.\\[1mm]
\textbf{Statement II:} The notebook reports the energy of the optimizer's final point \texttt{opt.x} as the VQE energy.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} COBYLA builds linear models from function values inside a trust region (initial radius \texttt{rhobeg = 0.3} in the notebook).
\textbf{II is false:} the code stores every evaluation and uses \texttt{best\_index = int(np.argmin(energy\_history))}; the reported energy and the sampled
state belong to the best evaluated angles, which need not be \texttt{opt.x}. Answer: (C)."""),

    dict(id="C13", kind="mcq",
         q=r"""Four VQE runs on the notebook's H$_2$ Hamiltonian (exact expectation values; FCI $= -1.137284$ Ha) report the energies below. Which one is a valid,
chemically accurate result?""",
         opts=[r"$-1.13600$ Ha", r"$-1.13500$ Ha", r"$-1.11676$ Ha", r"$-1.13900$ Ha"],
         sol=r"""Errors $E - E_{\mathrm{FCI}}$: $-1.13600$: $+1.28$ mHa (above FCI, below $1.6$ mHa: chemically accurate, valid).
$-1.13500$: $+2.28$ mHa (valid but not chemically accurate). $-1.11676$: $+20.5$ mHa, the Hartree--Fock energy ($\theta = 0$).
$-1.13900$: $-1.72$ mHa, \emph{below} FCI, impossible with exact expectation values (C1)."""),

    dict(id="C14", kind="mcq",
         q=r"""The notebook's H$_2$ Hamiltonian has the identity, 10 $Z$-type terms and 4 terms $\texttt{XXYY}$, $\texttt{XYYX}$, $\texttt{YXXY}$, $\texttt{YYXX}$. With
\emph{qubit-wise commuting} grouping, the minimum number of measurement settings is ..., and with general commuting grouping ...""",
         opts=[r"$5$; $2$", r"$15$; $5$", r"$2$; $1$", r"$4$; $1$"],
         sol=r"""The identity needs no measurement. All $Z$-type terms are qubit-wise commuting: one setting (plain $Z$ measurement). No two of the four
exchange terms are qubit-wise commuting (B9), so each needs its own setting: $1 + 4 = 5$. Under general commutation the four exchange
terms commute pairwise and form one group (measured with an entangling basis change), so $1 + 1 = 2$. (Whether the $Z$-group and the
exchange group could merge: $\texttt{ZIII}$ anticommutes with $\texttt{XXYY}$, so no.) Grouping directly reduces the shot cost of VQE."""),

    dict(id="C15", kind="mcq",
         q=r"""In the previously recorded default run, $E_{\mathrm{VQE}} = -15.0533898284$ and the exact $E_0 = -15.0825179883$. The relative error is""",
         opts=[r"$0.193\,\%$", r"$1.93\,\%$", r"$0.0291\,\%$", r"$-0.193\,\%$"],
         sol=r"""Absolute error $E_{\mathrm{VQE}} - E_0 = -15.0533898 + 15.0825180 = 0.0291282$ (positive, as the variational principle requires). Relative:
$0.0291282/15.0825180 = 0.00193 = 0.193\,\%$ --- the notebook prints ``2.913e-02 (relative 0.193\%)''. Note that TFIM energies are dimensionless
(in units of $J$); chemical accuracy does not apply to this model."""),

    dict(id="C16", kind="mcq",
         q=r"""The notebook's hardware-efficient ansatz uses only $R_Y$ rotations and CNOTs, so it produces states with \emph{real} amplitudes. Why is this
adequate for its Ising and H$_2$ Hamiltonians?""",
         opts=[r"Both Hamiltonians are real symmetric matrices, so a ground state can always be chosen real",
               r"Complex amplitudes cannot be measured, so they are never needed",
               r"Real amplitudes always give lower energies than complex ones",
               r"CNOT gates generate all complex phases automatically"],
         sol=r"""$R_Y(\theta)$ and CNOT have real matrices, so they map real vectors to real vectors. The Ising Hamiltonian ($Z$ and $X$ only) is real; the H$_2$
Hamiltonian is real too ($\texttt{XXYY}$-type strings contain an even number of $Y$'s, each $Y$ contributes $\pm i$, so the products are real).
For a real symmetric $H$ the real and imaginary parts of an eigenvector are eigenvectors themselves, so a real ground state exists. For
Hamiltonians with genuinely complex matrix elements (e.g.\ magnetic fields) a real ansatz would fail."""),

    dict(id="C17", kind="mcq",
         q=r"""The recorded optimised state of the Ising chain gives $\ket{000\dots0}$ with probability $0.70$ but $\ket{111\dots1}$ (not in the top 8) with nearly zero
probability, although $H$ is symmetric under flipping all spins. The best explanation is""",
         opts=[r"for $h < J$ the two ordered configurations are nearly degenerate; VQE settled on one of them (symmetry breaking), which costs only an energy exponentially small in $N$",
               r"the simulator has a bias towards $\ket0$",
               r"the exact ground state contains only $\ket{000\dots0}$",
               r"$\ket{111\dots1}$ has a higher energy than $\ket{000\dots0}$"],
         sol=r"""$\prod_iX_i$ commutes with $H$, so the exact ground state is symmetric: $P(0\dots0) = P(1\dots1)$. For $h < J$ it is close to the ``cat'' state
$\frac{1}{\sqrt2}(\ket{\Uparrow} + \ket{\Downarrow})$ (dressed by spin flips), whose symmetric and antisymmetric combinations are split only by an energy that
decreases exponentially with $N$. A shallow ansatz started near $\ket{0\dots0}$ cannot build the cat state cheaply and settles near one branch, as the
notebook's text explains; the energy cost is tiny compared with the recorded error $0.029$. Both configurations have the same energy."""),

    dict(id="C18", kind="match",
         q=r"""Match each optimizer offered by the notebook (List-I) with its working principle (List-II).""",
         match=([r"COBYLA", r"Nelder--Mead", r"Powell", r"L-BFGS-B"],
                [r"successive line searches along a set of conjugate directions, derivative-free",
                 r"quasi-Newton method with a limited-memory Hessian estimate, using gradients (finite differences here) and box bounds",
                 r"linear models of the objective inside a shrinking trust region, derivative-free",
                 r"a simplex of $n+1$ points updated by reflection, expansion and contraction"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: COBYLA = Constrained Optimization BY Linear Approximation, trust region (3) --- default, \texttt{maxiter = 300}, \texttt{rhobeg = 0.3}.
\item Q: Nelder--Mead simplex (4) --- \texttt{maxfev} budget.
\item R: Powell's conjugate-direction method (1).
\item S: L-BFGS-B (2) --- the only one needing gradients; without analytic gradients SciPy uses $n + 1$ evaluations per gradient.
\end{itemize}"""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The lowest eigenvalue of the $2\times2$ matrix --- the best energy the UCC ansatz can reach --- is""",
         opts=[r"$-1.137284$ Ha", r"$-1.116759$ Ha", r"$-0.654141$ Ha", r"$+0.483143$ Ha"],
         sol=r"""For $\begin{psmallmatrix}a&b\\b&c\end{psmallmatrix}$: $\lambda_\pm = \frac{a+c}{2} \pm \sqrt{\left(\frac{c-a}{2}\right)^2 + b^2}$.
$\frac{a+c}{2} = \frac{-1.116759 + 0.462618}{2} = -0.327071$; $\frac{c-a}{2} = 0.789689$; $\sqrt{0.789689^2 + 0.181210^2} = \sqrt{0.623608 + 0.032837} = 0.810213$.
$\lambda_- = -0.327071 - 0.810213 = -1.137284$ Ha $= E_{\mathrm{FCI}}$; $\lambda_+ = +0.483143$ (an excited state of the full Hamiltonian, too).
The coupling $H_{12}$ lowers the energy below $H_{11} = E_{\mathrm{HF}}$ by $20.5$ mHa, the correlation energy."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""The optimal angle $\theta^*$ and the probability of measuring $\ket{0011}$ in the optimal state are""",
         opts=[r"$\theta^* \approx 0.1128$ rad and $0.0127$", r"$\theta^* \approx 0.1128$ rad and $0.1125$",
               r"$\theta^* \approx -0.1128$ rad and $0.0127$", r"$\theta^* \approx 0.2295$ rad and $0.0518$"],
         sol=r"""$E'(\theta) = (H_{22} - H_{11})\sin2\theta - 2H_{12}\cos2\theta = 0 \Rightarrow \tan2\theta^* = \frac{2H_{12}}{H_{22} - H_{11}} = \frac{0.362420}{1.579377} = 0.229471$,
$2\theta^* = 0.225566$, $\theta^* = 0.112783$ rad (the minimum, $E'' > 0$). Probability of $\ket{0011}$: $\sin^2\theta^* = 0.0127$ (amplitude $-0.1125$).
So the exact H$_2$ ground state is $98.7\,\%$ Hartree--Fock and $1.3\,\%$ doubly excited; this small admixture is worth $20.5$ mHa.
Negative $\theta$ would raise the energy; $0.2295$ is $\tan2\theta^*$, not $\theta^*$."""),
]
