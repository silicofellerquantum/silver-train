"""Section D - Advanced HHL and implementation (complexity, condition number, success probability and amplitude
amplification, read-out cost, clock precision and aliasing, embedding and padding, Hamiltonian simulation, newer
algorithms, dequantisation, the notebook's backends, recorded results and limitations).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""\emph{Previously recorded default run of \texttt{HHL.ipynb}} (CUDA-Q 0.16.0, target \texttt{qpp-cpu}): $4\times4$ system, $\kappa = 4$, 7 qubits, $\|b\| = 1$, $C = 1$.
Success probability: $0.355903$ from the state, $0.3543$ from $20\,000$ shots ($7086$ accepted). Exact $|x_i|^2/\|x\|^2 = (0.7622, 0.0598, 0.1476, 0.0305)$; from the accepted shots
$(0.7691, 0.0587, 0.1420, 0.0302)$. Fidelity of the state read-out $1.0000000000$, relative error $1.655\times10^{-15}$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""The original HHL analysis (sparse $d\times d$ matrix with sparsity $s$, condition number $\kappa$, precision $\varepsilon$) gives a running time of""",
         opts=[r"$\tilde O\!\left(\log(d)\,s^2\kappa^2/\varepsilon\right)$", r"$O(d^3)$", r"$O(d\,s\sqrt\kappa\log(1/\varepsilon))$", r"$O(\log d)$, independent of $\kappa$ and $\varepsilon$"],
         sol=r"""Harrow, Hassidim and Lloyd (2009): $\tilde O(\log(d)s^2\kappa^2/\varepsilon)$ --- logarithmic in $d$ thanks to amplitude encoding and sparse Hamiltonian simulation, but polynomial in $\kappa$ and
$1/\varepsilon$. $O(d^3)$ is Gaussian elimination, $O(ds\sqrt\kappa\log(1/\varepsilon))$ is conjugate gradient. The exponential advantage in $d$ only holds if $\kappa$ and $s$ stay small,
$\ket b$ is easy to prepare and only a property of $\ket x$ is needed."""),

    dict(id="D2", kind="mcq",
         q=r"""HHL succeeds (ancilla $= 1$) with probability $p = 0.01$. On average, how many repetitions are needed with plain repetition, and with amplitude amplification?""",
         opts=[r"about $100$ and about $10$", r"about $10$ and about $100$", r"about $100$ and about $1$", r"about $1$ and about $10$"],
         sol=r"""Repeat-until-success needs $1/p = 100$ trials on average. Amplitude amplification rotates the success amplitude $\sqrt p$ towards 1 in $O(1/\sqrt p) = O(10)$ iterations. Since
$p \sim 1/\kappa^2$ in the worst case, amplification turns HHL's $\kappa^2$ repetition cost into $\kappa$ --- part of the original HHL analysis. The notebook simply post-selects (here $p = 0.356$)."""),

    dict(id="D3", kind="mcq",
         q=r"""To estimate all $d = 1024$ probabilities $|x_i|^2$ of $\ket x$ to absolute precision $\varepsilon = 0.01$ by sampling, roughly how many post-selected shots are needed?""",
         opts=[r"about $10^7$", r"about $10^3$", r"about $10$", r"about $10^{14}$"],
         sol=r"""Each estimate needs $\sim1/\varepsilon^2 = 10^4$ samples' worth of precision, and $d$ components share the budget: the usual scaling is $d/\varepsilon^2 = 1024\times10^4 \approx 10^7$ shots. This is
linear in $d$ and kills the exponential speed-up --- the reason HHL is useful for \emph{properties} of $x$ (e.g.\ $x^\dagger Mx$), not for printing $x$. The notebook's text says the same
(``about $d/\epsilon^2$ shots'')."""),

    dict(id="D4", kind="match",
         q=r"""For exact clock values the notebook needs $\kappa \le 2^{n_c - 1} - 1$. Match each condition number (List-I) with the smallest suitable number of clock qubits $n_c$
(List-II).""",
         match=([r"$\kappa = 3$", r"$\kappa = 7$", r"$\kappa = 20$", r"$\kappa = 40$"],
                [r"$6$", r"$3$", r"$7$", r"$4$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3"],
         sol=r"""Largest positive signed clock value $2^{n_c-1} - 1$: $n_c = 3 \to 3$, $4 \to 7$, $5 \to 15$, $6 \to 31$, $7 \to 63$.
\begin{itemize}\itemsep0pt
\item P: $\kappa = 3 \le 3$: $n_c = 3$ (2). \quad Q: $\kappa = 7 \le 7$: $n_c = 4$ (4).
\item R: $\kappa = 20$: $15 < 20 \le 31$: $n_c = 6$ (1) --- the notebook pairs \texttt{RANDOM\_CONDITION\_NUMBER = 20} with \texttt{NUM\_CLOCK\_QUBITS = 6}.
\item S: $\kappa = 40$: $31 < 40 \le 63$: $n_c = 7$ (3).
\end{itemize}
One clock qubit is the sign bit, hence $2^{n_c-1}$ rather than $2^{n_c}$."""),

    dict(id="D5", kind="mcq",
         q=r"""With \texttt{NUM\_CLOCK\_QUBITS = 8}, how many multi-controlled eigenvalue-inversion rotations does the notebook's circuit contain, and how many qubits does the
default $4\times4$ system then use?""",
         opts=[r"$255$ rotations, $11$ qubits", r"$8$ rotations, $11$ qubits", r"$256$ rotations, $12$ qubits", r"$255$ rotations, $14$ qubits"],
         sol=r"""One rotation per non-zero clock value: $2^8 - 1 = 255$, each controlled by all 8 clock qubits. Qubits: $8 + 2 + 1 = 11$ (the notebook's table: ``11 qubits in total''). The lookup-table
inversion is exponential in $n_c$; scalable implementations compute $1/\lambda$ arithmetically or use polynomial approximations (QSVT)."""),

    dict(id="D6", kind="mcq",
         q=r"""With \texttt{MATRIX\_A = [[1, 2], [0, 1]]} (not Hermitian) and $n_c = 4$, the notebook's circuit uses how many qubits, and where is $x$ in the solution of the embedded system?""",
         opts=[r"$7$ qubits; $x$ is the second half of the solution", r"$6$ qubits; $x$ is the whole solution", r"$7$ qubits; $x$ is the first half of the solution", r"$8$ qubits; $x$ is the second half"],
         sol=r"""Embedding $\begin{psmallmatrix}0 & A\\ A^\dagger & 0\end{psmallmatrix}\begin{psmallmatrix}y\\ x\end{psmallmatrix} = \begin{psmallmatrix}b\\ 0\end{psmallmatrix}$ gives $Ax = b$ (and $A^\dagger y = 0$, so $y = 0$). Dimension $4 \Rightarrow n_b = 2$; total $4 + 2 + 1 = 7$. The code sets
\texttt{solution\_slice = slice(d0, 2 * d0)}: the second half. The embedding costs one extra system qubit and its $\kappa$ is that of the singular values ($5.83$ here)."""),

    dict(id="D7", kind="mcq",
         q=r"""A $3\times3$ Hermitian matrix with eigenvalues $5, 10, 20$ is padded by the notebook to $4\times4$ with an identity block. What happens to $\kappa$?""",
         opts=[r"it grows from $4$ to $20$, because the padding adds the eigenvalue $1$", r"it stays $4$", r"it becomes $1$", r"it becomes $5$"],
         sol=r"""Before padding $\kappa = 20/5 = 4$. The identity block adds eigenvalue $1$ (with $b$ padded by zeros, its eigenvector carries no weight of $\ket b$), but the clock and $C$ must still cover
$\lambda = 1$ under the \texttt{"auto"} rule: $\kappa = 20/1 = 20$. A padding value inside the spectrum (e.g.\ $5$--$20$) would avoid this; the notebook's identity padding is simple but can
worsen the conditioning that sets the clock size."""),

    dict(id="D8", kind="mcq",
         q=r"""``Dequantised'' algorithms (Tang, 2018 and later) showed that for some quantum linear-algebra tasks""",
         opts=[r"classical randomized algorithms with comparable sampling access achieve similar poly-logarithmic dependence on $d$ for low-rank matrices",
               r"HHL can run without phase estimation",
               r"classical computers can prepare $\ket x$ on a quantum computer",
               r"the condition number no longer affects quantum algorithms"],
         sol=r"""Given ``sample and query'' access to the data (analogous to the qRAM assumed by quantum algorithms), classical sampling methods solve low-rank problems (recommendation systems,
low-rank linear systems) in time polylogarithmic in $d$, polynomial in the rank and $\kappa$. Exponential quantum advantage therefore needs sparse, full-rank, well-conditioned structure and
efficient state preparation --- not just ``a big matrix''."""),

    dict(id="D9", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} HHL gives an exponential speed-up over classical algorithms for every linear system $Ax = b$.\\[1mm]
\textbf{Reason (R):} For $s$-sparse, well-conditioned $A$ with efficiently preparable $\ket b$, HHL prepares $\ket x$ in time polynomial in $\log d$, $s$, $\kappa$ and $1/\varepsilon$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true} (the HHL theorem, under its assumptions).
\textbf{A is false:} dense or ill-conditioned matrices ($\kappa \sim d$), expensive state preparation, the need to read all of $x$ ($d/\varepsilon^2$ shots), and dequantised classical algorithms
each remove the advantage. HHL's speed-up is conditional. Answer: (D)."""),

    dict(id="D10", kind="mcq",
         q=r"""Take $C = \lambda_{\min}$ and a matrix with $\kappa = 20$. If $\ket b$ is the eigenvector of $\lambda_{\max}$, the success probability is""",
         opts=[r"$1/\kappa^2 = 0.0025$", r"$1$", r"$1/\kappa = 0.05$", r"$0.5$"],
         sol=r"""$P = \sum_j|\beta_j|^2C^2/\lambda_j^2$ with all weight on $\lambda_{\max}$: $P = (\lambda_{\min}/\lambda_{\max})^2 = 1/\kappa^2 = 1/400$. If $\ket b$ were the eigenvector of $\lambda_{\min}$, $P = 1$. In general
$1/\kappa^2 \le P \le 1$; the notebook's default ($\kappa = 4$, equal weights) gives $0.356$."""),

    dict(id="D11", kind="mcq",
         q=r"""With $n_c = 4$ and $t = \pi/8$ (clock value $k = \lambda$), how is an eigenvalue $\lambda = 9$ treated?""",
         opts=[r"It aliases: clock value $9$ is read as the signed value $-7$, so the wrong (negative) $1/\lambda$ is applied",
               r"It is read correctly as $9$",
               r"It is read as $7$, the largest positive value",
               r"Phase estimation fails with an error"],
         sol=r"""$e^{i\lambda t}$ only knows $\lambda t$ modulo $2\pi$: $k = 9 \pmod{16}$, and $9 \ge 8$ is interpreted as $9 - 16 = -7$. The inversion applies amplitude $C/(-7)$ instead of $C/9$ --- wrong
magnitude and sign. The notebook warns ``some $|k_j|$ exceed the representable range -> eigenvalues alias; reduce EVOLUTION\_TIME'' (it requires $|\lambda|t < \pi$, here $|\lambda| < 8$)."""),

    dict(id="D12", kind="match",
         q=r"""Match each quantum linear-system algorithm (List-I) with its key idea (List-II).""",
         match=([r"HHL (2009)", r"Ambainis (2012), variable-time amplitude amplification", r"Childs--Kothari--Somma (2017)", r"QSVT-based solvers"],
                [r"implement a polynomial approximation of $1/x$ on the singular values of a block-encoded $A$",
                 r"phase estimation + controlled rotation $C/\lambda$ + post-selection; cost $\kappa^2/\varepsilon$",
                 r"amplify different eigenvalue branches at different times: $\kappa$ dependence close to linear ($\kappa\log^3\kappa$)",
                 r"linear combination of unitaries (Fourier/Chebyshev series of $1/x$): $\mathrm{poly}\log(1/\varepsilon)$ precision dependence"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: HHL --- the scheme implemented in the notebook (2).
\item Q: variable-time amplitude amplification improves the $\kappa$ dependence (3).
\item R: Childs--Kothari--Somma replace phase estimation by LCU, improving $1/\varepsilon$ to $\mathrm{poly}\log(1/\varepsilon)$ (4).
\item S: quantum singular value transformation unifies these with polynomial approximations (1). (Adiabatic/discrete-adiabatic methods reach optimal $\kappa\log(1/\varepsilon)$.)
\end{itemize}"""),

    dict(id="D13", kind="mcq",
         q=r"""With \texttt{SIMULATION\_METHOD = "density\_matrix"} the notebook multiplies its recovered $x$ by a phase factor before comparing with NumPy. Why?""",
         opts=[r"A density matrix $\ket\psi\bra\psi$ carries no global phase, so the vector extracted from it has an arbitrary phase that must be aligned",
               r"The density-matrix simulator adds noise that rotates the phase",
               r"The density-matrix simulator uses fp32 precision",
               r"Post-selection on a density matrix flips the sign of every component"],
         sol=r"""$\rho = \ket\psi\bra\psi$ is invariant under $\ket\psi \to e^{i\alpha}\ket\psi$. The notebook takes the dominant eigenvector of $\rho$ (\texttt{evecs[:, -1]}), whose phase is arbitrary, and aligns it with
\texttt{overlap / abs(overlap)} --- printing ``a density matrix carries no global phase''. No noise model is used in the HHL notebook, so the density-matrix result is still exact."""),

    dict(id="D14", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} \texttt{HHL.ipynb} refuses hardware backends, because exact matrix exponentials as custom gates and amplitude initialisation cannot run on hardware targets.\\[1mm]
\textbf{Statement II:} The controlled powers $U^{2^j}$ are registered as custom unitary matrices with \texttt{cudaq.register\_operation}.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} \texttt{if backend\_cfg["device"] == "hardware": raise ValueError("This HHL implementation uses exact matrix exponentials as custom gates and amplitude encoding,
which hardware targets cannot execute...")}.
\textbf{II is true:} \texttt{cudaq.register\_operation(name, np.linalg.matrix\_power(U, 2 ** j))}. On hardware, $e^{iAt}$ would need a gate-level Hamiltonian simulation (e.g.\ Trotterised Pauli
rotations), which the notebook does not implement. Answer: (A)."""),

    dict(id="D15", kind="mcq",
         q=r"""Run with only \texttt{NUM\_CLOCK\_QUBITS = 3} on the default system ($\kappa = 4 > 3$ levels), the notebook rescales $t$ so that $\lambda_{\max} = 4$ maps to clock value 3. The ideal
clock values of $\lambda = 1, 2, 3, 4$ and the effect are""",
         opts=[r"$0.75, 1.5, 2.25, 3$ --- non-integer values are rounded by phase estimation, so the fidelity drops below 1",
               r"$1, 2, 3, 4$ --- still exact",
               r"$0.75, 1.5, 2.25, 3$ --- the fidelity stays exactly 1",
               r"$3, 6, 9, 12$ --- all eigenvalues alias"],
         sol=r"""$t = 2\pi\cdot3/(8\cdot4) = 3\pi/16$, so $k = \lambda t\,2^3/(2\pi) = 0.75\lambda$: $0.75, 1.5, 2.25, 3$. Only $\lambda = 4$ is exact; the others leak over neighbouring clock values and are inverted
with wrong $1/\lambda$. Simulation of the notebook's circuit (CUDA-Q) gives a fidelity of about $0.953$ instead of 1 --- the notebook's ``Things to try next'' predicts exactly this drop."""),

    dict(id="D16", kind="mcq",
         q=r"""With \texttt{NUM\_CLOCK\_QUBITS = 8} the default circuit has 11 qubits. The fp64 state vector needs ... bytes, and the notebook's memory line (state + one host copy) shows about ...""",
         opts=[r"$32\,768$; $65.5$ kB", r"$2\,048$; $4.1$ kB", r"$32\,768$; $32.8$ kB", r"$4\,194\,304$; $8.4$ MB"],
         sol=r"""$16\times2^{11} = 32\,768$ B; with the extra copy for \texttt{get\_state}: $65\,536$ B $\approx 65.5$ kB. Memory is never the limit for these small HHL instances; gate count and depth are
($255$ multi-controlled rotations at $n_c = 8$). $4\,194\,304 = 16\times2^{18}$ would be an 18-qubit state."""),

    dict(id="D17", kind="mcq",
         q=r"""Which statement correctly describes how \texttt{HHL.ipynb} implements $U = e^{iAt}$?""",
         opts=[r"It computes $e^{iAt}$ exactly with \texttt{scipy.linalg.expm} and applies each power $U^{2^j}$ as a custom controlled gate; this is exact on simulators but is not a scalable Hamiltonian simulation",
               r"It uses first-order Trotterisation of $A$'s Pauli decomposition",
               r"It uses qubitisation with a block encoding of $A$",
               r"It approximates $e^{iAt}$ by $I + iAt$"],
         sol=r"""\texttt{U\_step = expm(1j * A\_work * t\_evolution)}; powers via \texttt{np.linalg.matrix\_power} and \texttt{cudaq.register\_operation}. A $2^{n_b}\times2^{n_b}$ matrix per gate is fine for 2 system qubits,
but costs exponential classical effort and has no gate decomposition. Real implementations need sparse Hamiltonian simulation (product formulas, qubitisation), whose cost enters HHL's
$s$ and $\kappa$ dependence. $I + iAt$ is not even unitary."""),

    dict(id="D18", kind="mcq",
         q=r"""Which use of HHL can, in principle, keep an exponential advantage?""",
         opts=[r"estimating a quantity such as $x^\dagger Mx$ for a sparse, well-conditioned $A$ with efficiently preparable $\ket b$",
               r"printing every component of $x$ for a dense $10^6\times10^6$ matrix",
               r"solving a dense, ill-conditioned system with $\kappa \sim d$",
               r"replacing Gaussian elimination for $4\times4$ systems"],
         sol=r"""An expectation value needs only $O(1/\varepsilon^2)$ shots of $\ket x$, independent of $d$, so the $\log d$ scaling survives if $s$, $\kappa$ and state preparation are cheap. Printing $x$ costs $d/\varepsilon^2$ shots;
$\kappa \sim d$ destroys the advantage through the $\kappa$ dependence; small systems are trivially solved classically (the notebook's $4\times4$ example is a demonstration, not a speed-up)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""The sampled $|x_0|^2/\|x\|^2 = 0.7691$ differs from the exact $0.7622$ by $0.0069$. With $7086$ accepted shots, its binomial standard error is about ..., so the deviation is ...""",
         opts=[r"$0.0051$; about $1.4$ standard errors --- consistent with shot noise", r"$0.0051$; about $14$ standard errors --- a bug", r"$0.0119$; about $0.6$ standard errors", r"$0.0034$; about $2$ standard errors"],
         sol=r"""$\sqrt{p(1 - p)/S} = \sqrt{0.7622\times0.2378/7086} = \sqrt{2.558\times10^{-5}} = 0.0051$; $0.0069/0.0051 \approx 1.4$ --- an ordinary fluctuation. The notebook's quoted scale $1/\sqrt{7086} = 0.0119$ is a conservative
upper bound (even the worst case $\sqrt{p(1 - p)} = \tfrac12$ gives only $0.0059$). $0.0034$ is the standard error of the overall success fraction from $20\,000$ shots."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Using $P_{\text{success}} = C^2\|x\|^2/\|b\|^2$, the norm $\|x\|$ estimated from the shot-based success fraction is about""",
         opts=[r"$0.595$", r"$0.354$", r"$0.125$", r"$1.68$"],
         sol=r"""$\|x\| = \frac{\|b\|}{C}\sqrt{P} = \sqrt{0.3543} = 0.595$ (exact $\sqrt{0.355903} = 0.5966$). So even on hardware the norm of $x$ can be estimated from the success rate, while the
components need the histogram (magnitudes) plus extra measurements (signs). $0.354$ forgets the square root; $1.68 = 1/0.595$."""),
]
