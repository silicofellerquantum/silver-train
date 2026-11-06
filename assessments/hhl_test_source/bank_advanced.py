"""Section D - Advanced HHL and implementation (complexity, sparsity and kappa, error analysis, post-selection and
amplitude amplification, circuit resources, CUDA-Q implementation and memory, noise, benchmarking of recorded
results, limitations, dequantisation and applications).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""The previously recorded default run of \texttt{HHL.ipynb} (7 qubits, \texttt{qpp-cpu}) printed:\\[2pt]
{\small\ttfamily Success probability: state 0.355903, shots 0.3543 (7086 of 20000), ideal 0.355903\\
Fidelity |<x\_classical|x\_quantum>|\^{}2 : 1.0000000000\\
Relative error of the recovered vector: 1.655e-15}\\[2pt]
In the eigenbasis of the default $A$ (eigenvalues $1, 2, 3, 4$) the vector $\ket b = \ket{00}$ has all four coefficients $\beta_j = \tfrac12$; $C = 1$ and $\lVert\vec b\rVert = 1$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""For an $s$-sparse Hermitian $N\times N$ matrix with condition number $\kappa$, target error $\epsilon$, and efficient preparation of $\ket b$, the run time
of the original HHL algorithm (2009) scales as""",
         opts=[r"$\tilde O\!\left(\log(N)\,s^2\kappa^2/\epsilon\right)$", r"$O(N^3)$", r"$\tilde O(N\,s\,\kappa)$", r"$O(\log N)$, independent of $\kappa$ and $\epsilon$"],
         sol=r"""Harrow, Hassidim and Lloyd showed a run time $\tilde O(\log(N)s^2\kappa^2/\epsilon)$: $\log N$ from the number of qubits and sparse Hamiltonian
simulation, $s^2$ from simulating $e^{iAt}$ for an $s$-sparse matrix, one factor $\kappa/\epsilon$ from the phase-estimation precision and one factor $\kappa$
from the $O(\kappa^2)$ repetitions reduced to $O(\kappa)$ by amplitude amplification. $O(N^3)$ is Gaussian elimination; $\tilde O(Ns\kappa)$ is roughly
classical conjugate gradient ($Ns\sqrt\kappa$ for positive-definite $A$). No algorithm is independent of $\kappa$."""),

    dict(id="D2", kind="mcq",
         q=r"""Suppose $P_{\text{success}} = 1/\kappa^2$ with $\kappa = 10$. The expected number of repetitions needed for one successful run without, and with,
amplitude amplification is about""",
         opts=[r"$100$ without; $\ 10$ with", r"$10$ without; $\ \sqrt{10}$ with", r"$100$ without; $\ 100$ with (no improvement is possible)",
               r"$1$ without; $\ 1$ with"],
         sol=r"""Repeat-until-success: the number of trials is geometric with mean $1/P = \kappa^2 = 100$.
Amplitude amplification (Grover-type rotations around the success subspace) raises the success amplitude $\sqrt P = 1/\kappa$ to $O(1)$ with
$O(1/\sqrt P) = O(\kappa) \approx 10$ applications of the HHL circuit and its inverse --- a quadratic saving. This is how the original analysis
turns $\kappa^2$ repetitions into a factor $\kappa$ in the run time. (It requires coherent repetitions, not just re-running and measuring.)"""),

    dict(id="D3", kind="match",
         q=r"""Match each quantum linear-system algorithm (List-I) with its characteristic improvement or complexity (List-II).""",
         match=([r"HHL (Harrow, Hassidim, Lloyd, 2009)", r"Variable-time amplitude amplification (Ambainis, 2012)",
                 r"Childs, Kothari and Somma (2017)", r"Discrete adiabatic method (Costa et al., 2022)"],
                [r"precision dependence $\mathrm{polylog}(1/\epsilon)$ via Fourier/Chebyshev fitting of $1/x$ (LCU)", r"almost linear dependence on $\kappa$",
                 r"$\tilde O(\kappa^2/\epsilon)$ dependence on $\kappa$ and $\epsilon$", r"optimal $O(\kappa\log(1/\epsilon))$ query complexity"]),
         opts=[r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4",
               r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: HHL uses phase estimation, which gives $\kappa^2/\epsilon$ (3).
\item Q: Ambainis' variable-time amplitude amplification stops ``branches'' with large eigenvalues early and reaches $\tilde O(\kappa\log^3\kappa)$ (2).
\item R: Childs--Kothari--Somma replace phase estimation by a linear combination of unitaries approximating $1/x$, giving $\mathrm{polylog}(1/\epsilon)$ (1).
\item S: Costa et al.\ use a discrete adiabatic walk and reach the optimal $O(\kappa\log(1/\epsilon))$ (4). Quantum singular value transformation (QSVT) gives
a unified framework for these methods.
\end{itemize}"""),

    dict(id="D4", kind="mcq",
         q=r"""Why can HHL \emph{not} efficiently output all $d$ components of $\vec x$ for $d = 2^{20}$?""",
         opts=[r"Sampling gives each $|x_i|^2$ only statistically; estimating all $d$ values needs $\Omega(d)$ repetitions (about $d/\epsilon^2$ for precision $\epsilon$), destroying the $\log d$ scaling",
               r"The no-cloning theorem forbids measuring the system register", r"Complex amplitudes cannot be measured at all",
               r"The ancilla measurement destroys $\ket x$ every time, so the system register never contains $\ket x$"],
         sol=r"""One run yields one sample $i$ with probability $|x_i|^2/\lVert\vec x\rVert^2$. To see all $d$ components even once needs at least $\sim d$ samples, and to estimate
each with additive precision $\epsilon$ needs about $d/\epsilon^2$ (tomography); signs and phases need further measurement bases. Each sample costs a full
HHL run, so the total is $\Omega(d)$ --- no better than classical. Measurement itself is allowed (no-cloning only forbids \emph{copying} unknown states);
a successful ancilla outcome \emph{leaves} $\ket x$ in the system register."""),

    dict(id="D5", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} HHL gives an exponential speed-up for solving any linear system $A\vec x = \vec b$ and printing the solution $\vec x$.\\[1mm]
\textbf{Reason (R):} Under suitable assumptions (sparse, well-conditioned $A$, efficient preparation of $\ket b$, only a property of $\ket x$ wanted)
the run time of HHL is polylogarithmic in the dimension.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is false}, for three independent reasons: ``any'' system ignores the sparsity and $\kappa$ requirements (for $\kappa \sim d$ the advantage vanishes),
loading an arbitrary $\vec b$ can cost $O(d)$, and printing $\vec x$ needs $\Omega(d)$ repetitions (D4).
\textbf{R is true}: this is precisely the HHL result, $\tilde O(\log(N)s^2\kappa^2/\epsilon)$. Aaronson's ``read the fine print'' lists exactly these caveats. Answer: (D)."""),

    dict(id="D6", kind="mcq",
         q=r"""The notebook is run with \texttt{NUM\_CLOCK\_QUBITS = 3} on the default matrix (eigenvalues $1, 2, 3, 4$). Since $\kappa = 4$ exceeds the $2^{n_c-1} - 1 = 3$ positive levels,
``auto'' sets $t = 2\pi\cdot3/(2^{n_c}\lambda_{\max})$. The ideal clock values $k_j = \lambda_jt2^{n_c}/2\pi$ are""",
         opts=[r"$0.75,\ 1.5,\ 2.25,\ 3$: only $\lambda = 4$ is exactly representable", r"$1,\ 2,\ 3,\ 4$: all exact",
               r"$0.5,\ 1,\ 1.5,\ 2$: $\lambda = 2$ and $4$ exact", r"$1.5,\ 3,\ 4.5,\ 6$: aliasing of $\lambda = 3, 4$"],
         sol=r"""$t = \frac{2\pi\cdot3}{8\cdot4} = \frac{3\pi}{16}$. Then $k_j = \lambda_j\cdot\frac{3\pi}{16}\cdot\frac{8}{2\pi} = \frac34\lambda_j$: $0.75, 1.5, 2.25, 3$.
Only $3$ is an integer, so phase estimation rounds the other three (with spread-out probability, B9) and the eigenvalue inversion uses wrong
values of $1/\lambda$: the fidelity drops below $1$. Also $C = 2\pi/(8t) = \frac43 > \lambda_{\min} = 1$: the eigenvalue $1$ is read either as clock value $1$ ($\tilde\lambda = \frac43$, wrong $1/\lambda$) or as $0$ (no rotation, component lost).
$0.5, 1, 1.5, 2$ would be $t = \pi/8$ unchanged. This is the ``Things to try next'' experiment of the notebook."""),

    dict(id="D7", kind="mcq",
         q=r"""Using the rule of thumb $n_c \approx \lceil\log_2(\kappa/\epsilon)\rceil$ for the number of clock qubits, a system with $\kappa = 100$ and target relative precision
$\epsilon = 0.01$ needs about""",
         opts=[r"$14$ clock qubits", r"$7$ clock qubits", r"$10$ clock qubits", r"$100$ clock qubits"],
         sol=r"""$\kappa/\epsilon = 100/0.01 = 10^4$, $\log_2 10^4 = 4\log_2 10 \approx 4\times3.32 = 13.29$, rounded up: $14$.
Reason: the smallest eigenvalue $\lambda_{\min} = \lambda_{\max}/\kappa$ must be resolved with relative error $\epsilon$, i.e.\ absolute resolution $\epsilon\lambda_{\min}$, while
the clock must also span $\lambda_{\max}$: $2^{n_c} \gtrsim \lambda_{\max}/(\epsilon\lambda_{\min}) = \kappa/\epsilon$. ($7 = \lceil\log_2100\rceil$ ignores $\epsilon$.)
The phase-estimation cost (total evolution time $\propto 2^{n_c}$) grows like $\kappa/\epsilon$."""),

    dict(id="D8", kind="mcq",
         q=r"""The notebook's state-vector approach is used for a Hermitian $2^{20}\times2^{20}$ matrix with $n_c = 10$ (fp64, $16$ bytes per amplitude). The number of qubits
and the memory of the state vector are""",
         opts=[r"$31$ qubits; $\ \approx 34.4$ GB", r"$31$ qubits; $\ \approx 17.2$ GB", r"$2^{20} + 11$ qubits; $\ $ beyond any computer",
               r"$31$ qubits; $\ 496$ bytes"],
         sol=r"""$n_b = 20$, $n_c = 10$, ancilla $1$: $31$ qubits. Memory $16\times2^{31} = 2^{35}$ bytes $= 34.36\times10^9$ bytes $\approx 34.4$ GB --- more than most
single GPUs, i.e.\ a multi-GPU run. ($17.2$ GB is $2^{30}$ amplitudes, one qubit fewer.) Note also that registering $e^{iAt2^j}$ as a dense
$2^{20}\times2^{20}$ custom gate would need $16\times2^{40}$ bytes $\approx 17.6$ TB per matrix: exact matrix exponentials do not scale; real
implementations use Hamiltonian simulation."""),

    dict(id="D9", kind="mcq",
         q=r"""$e^{iAt}$ with $A = A_1 + A_2$ ($[A_1, A_2] \neq 0$) is approximated by $r$ first-order Trotter steps $\left(e^{iA_1t/r}e^{iA_2t/r}\right)^r$. To keep the total
error below $\epsilon$, $r$ must grow as""",
         opts=[r"$r = O\!\left(t^2\lVert[A_1, A_2]\rVert/\epsilon\right)$", r"$r = O(\log(1/\epsilon))$, independent of $t$", r"$r = O(t/\epsilon^2)$",
               r"$r = 1$ always suffices, because the error cancels"],
         sol=r"""Baker--Campbell--Hausdorff: $e^{iA_1\tau}e^{iA_2\tau} = e^{i(A_1+A_2)\tau - \frac{\tau^2}{2}[A_1, A_2] + O(\tau^3)}$, so each step with $\tau = t/r$ has error
$\approx\frac{t^2}{2r^2}\lVert[A_1, A_2]\rVert$. Over $r$ steps the errors add: $\approx\frac{t^2}{2r}\lVert[A_1, A_2]\rVert \le \epsilon \Rightarrow r = O(t^2\lVert[A_1,A_2]\rVert/\epsilon)$.
Higher-order product formulas, qubitisation or QSVT reduce this cost. (The notebook avoids the issue entirely with exact $e^{iAt}$ gates.)"""),

    dict(id="D10", kind="stmt", correct=2,
         q=r"""About \texttt{HHL.ipynb}:\\[1mm]
\textbf{Statement I:} The controlled evolutions $e^{iAt2^j}$ are implemented as exact matrix exponentials (\texttt{scipy.linalg.expm}) registered as custom gates, not
by Hamiltonian simulation.\\[1mm]
\textbf{Statement II:} The notebook executes HHL on quantum-hardware targets when \texttt{BACKEND = "hardware"}.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} \texttt{U\_step = expm(1j * A\_work * t\_evolution)}; each power is registered with \texttt{cudaq.register\_operation}.
\textbf{II is false:} the backend cell stops with an error for \texttt{"hardware"}: ``This HHL implementation uses exact matrix exponentials as custom gates and
amplitude encoding, which hardware targets cannot execute. Use a simulator backend.'' The allowed methods are \texttt{statevector} and
\texttt{density\_matrix}; the recorded results come from the CPU simulator \texttt{qpp-cpu}. Answer: (C)."""),

    dict(id="D11", kind="mcq",
         q=r"""In the recorded run the ideal success probability is $0.355903$ and $7086$ of $20\,000$ shots ($0.3543$) were accepted. The binomial standard error of the
measured fraction and the verdict are""",
         opts=[r"$\approx0.0034$; the deviation is $\approx0.5\sigma$, consistent with shot noise", r"$\approx0.0034$; the deviation is $\approx4.7\sigma$, a circuit error",
               r"$\approx0.012$; the deviation is $\approx0.1\sigma$", r"$\approx0.0007$; the deviation is $\approx2.3\sigma$"],
         sol=r"""$\sigma = \sqrt{\frac{p(1-p)}{N}} = \sqrt{\frac{0.355903\times0.644097}{20000}} = \sqrt{1.146\times10^{-5}} \approx 0.00339$.
Deviation: $0.3543 - 0.3559 = -0.0016$, i.e.\ $-0.0016/0.0034 \approx -0.47\sigma$ --- well within statistical fluctuations. ($0.012 \approx 1/\sqrt{7086}$ is the
notebook's rough error scale for the \emph{accepted} shots, not this standard error.) The state-vector run gives exactly $0.355903$."""),

    dict(id="D12", kind="mcq",
         q=r"""In the recorded run, the post-selected shots ($7086$ accepted) estimate $|x_0|^2/\lVert\vec x\rVert^2$ as $0.7691$; the exact value is $0.7622$. The binomial standard
error of this estimate and the deviation in units of it are""",
         opts=[r"$\approx0.0051$; $\ \approx1.4\sigma$", r"$\approx0.0051$; $\ \approx14\sigma$", r"$\approx0.012$; $\ \approx0.6\sigma$, the exact standard error",
               r"$\approx0.0030$; $\ \approx2.3\sigma$"],
         sol=r"""$\sigma = \sqrt{\frac{p(1-p)}{N_{\text{acc}}}} = \sqrt{\frac{0.7622\times0.2378}{7086}} = \sqrt{2.558\times10^{-5}} \approx 0.00506$.
Deviation $0.7691 - 0.7622 = 0.0069 \approx 1.4\sigma$: an ordinary fluctuation. The notebook prints $1/\sqrt{7086} \approx 0.012$ as a conservative scale
(since $\sqrt{p(1-p)} \le \frac12$, the true standard error is at most $0.0059$). Only the accepted shots count: the effective sample size is
$P_{\text{success}}\times\text{SHOTS}$, so a low success probability directly increases the read-out error."""),

    dict(id="D13", kind="mcq",
         q=r"""The default circuit ($n_c = 4$, $4\times4$ matrix) has $15$ multi-controlled rotations and $8$ controlled $e^{\pm iAt2^j}$ gates on $7$ qubits. With the
same matrix and $n_c = 5$ the notebook's circuit has""",
         opts=[r"$31$ rotations, $\ 10$ controlled $e^{\pm iAt2^j}$, $\ 8$ qubits", r"$16$ rotations, $\ 10$ controlled $e^{\pm iAt2^j}$, $\ 8$ qubits",
               r"$31$ rotations, $\ 5$ controlled $e^{\pm iAt2^j}$, $\ 8$ qubits", r"$31$ rotations, $\ 10$ controlled $e^{\pm iAt2^j}$, $\ 7$ qubits"],
         sol=r"""Rotations: one per non-zero clock value, $2^{n_c} - 1 = 31$ (exponential in $n_c$ --- this naive inversion does not scale; practical versions compute
$1/\lambda$ arithmetically or use QSVT). Controlled evolutions: $n_c$ in phase estimation plus $n_c$ in uncomputation, $2\times5 = 10$.
Qubits: $5 + 2 + 1 = 8$. (Each rotation is now controlled by $5$ qubits.)"""),

    dict(id="D14", kind="match",
         q=r"""Match each error source (List-I) with its main effect on HHL (List-II).""",
         match=([r"bit flip on the ancilla just before measurement", r"phase error on a clock qubit during phase estimation",
                 r"imperfect uncomputation of the clock", r"finite number of shots"],
                [r"statistical error $\sim 1/\sqrt{N_{\text{acc}}}$ in the estimated $|x_i|^2$", r"wrong eigenvalue estimate, so a wrong factor $1/\tilde\lambda$ is applied",
                 r"failed runs are accepted (and good ones rejected): biased $P_{\text{success}}$ and contaminated output",
                 r"residual clock--system entanglement: the output is a mixed state"]),
         opts=[r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: the ancilla is the success flag; flipping it mixes the $\sqrt{1 - C^2/\lambda^2}$ branch (not $\propto A^{-1}$) into the accepted data (3).
\item Q: QPE reads phases; a phase error shifts the estimated $k$, and the rotation then uses the wrong $\tilde\lambda$ (2).
\item R: if $\ket{k_j}$ is not reset to $\ket0$, the eigencomponents stay labelled and do not interfere: mixed output (4).
\item S: sampling noise with the \emph{accepted} shots as sample size (1).
\end{itemize}
HHL's deep circuits (controlled $e^{iAt2^j}$ with large $2^j$) make it far beyond today's noisy hardware for non-trivial sizes."""),

    dict(id="D15", kind="mcq",
         q=r"""Which output can HHL provide efficiently (under its assumptions)?""",
         opts=[r"An estimate of an expectation value $\braket{x|M|x}$ for an efficiently measurable observable $M$", r"The complete vector $\vec x$ to 16 significant digits",
               r"The determinant of $A$", r"The full inverse matrix $A^{-1}$"],
         sol=r"""Measuring an observable $M$ on $\ket x$ repeatedly estimates $\braket{x|M|x}$ to precision $\epsilon$ with $O(1/\epsilon^2)$ runs (or $O(1/\epsilon)$ with amplitude estimation),
independently of $d$. Examples: the total weight of a region of $\vec x$, an overlap $|\braket{y|x}|^2$ via a swap test, a cost function in machine learning.
Printing $\vec x$ needs $\Omega(d)$ runs (D4); $\det A$ and $A^{-1}$ ($d^2$ numbers) are not produced by HHL at all."""),

    dict(id="D16", kind="mcq",
         q=r"""``Dequantisation'' results (Tang 2019 and successors) give classical algorithms with run time polylogarithmic in $d$ for linear systems when""",
         opts=[r"$A$ has low rank and $\ell^2$-norm sample-and-query access to $A$ and $\vec b$ is available", r"$A$ is Hermitian",
               r"$A$ is sparse and well conditioned with $\kappa = \mathrm{polylog}(d)$, and $\vec b$ is given as an efficient circuit",
               r"$\vec b$ is a computational basis vector"],
         sol=r"""Tang's quantum-inspired algorithms show that if one can sample rows/entries with probability proportional to their squared norms (the classical analogue
of amplitude encoding / QRAM), low-rank problems such as recommendation systems and low-rank linear systems can be solved classically in
time $\mathrm{poly}(\text{rank}, \kappa, 1/\epsilon)\,\mathrm{polylog}(d)$. For sparse, well-conditioned, full-rank systems with efficiently prepared $\ket b$, HHL-type problems
are BQP-complete and no such classical algorithm is expected. Hermiticity or a basis vector $\vec b$ alone gives no classical shortcut."""),

    dict(id="D17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} At fixed $N$, $s$ and $\epsilon$, doubling the condition number roughly quadruples the run time of the original HHL algorithm.\\[1mm]
\textbf{Reason (R):} The original HHL run time scales as $\tilde O(\log(N)s^2\kappa^2/\epsilon)$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true} (D1). \textbf{A is true and follows from R}: with everything else fixed, $T \propto \kappa^2$, so $\kappa \to 2\kappa$ gives $T \to 4T$ (up to
logarithmic factors). The two factors of $\kappa$ come from phase-estimation precision ($t \sim \kappa/\epsilon$) and from the repetitions after
amplitude amplification ($\sim\kappa$). Improved algorithms (D3) reduce this to linear. Answer: (A)."""),

    dict(id="D18", kind="mcq",
         q=r"""For a non-Hermitian invertible $A$ the notebook solves $\begin{pmatrix}0 & A\\ A^\dagger & 0\end{pmatrix}\begin{pmatrix}\vec y\\ \vec z\end{pmatrix} = \begin{pmatrix}\vec b\\ \vec 0\end{pmatrix}$. The solution satisfies""",
         opts=[r"$\vec z = A^{-1}\vec b$ (second half) and $\vec y = \vec 0$", r"$\vec y = A^{-1}\vec b$ (first half) and $\vec z = \vec 0$",
               r"$\vec y = A^{-1}\vec b$ and $\vec z = (A^\dagger)^{-1}\vec b$", r"$\vec y = \vec z = A^{-1}\vec b/\sqrt2$"],
         sol=r"""Multiply out: the first block row gives $A\vec z = \vec b$, so $\vec z = A^{-1}\vec b = \vec x$; the second gives $A^\dagger\vec y = \vec 0$, so $\vec y = \vec 0$ ($A^\dagger$ is invertible).
Hence the solution sits in the \emph{second} half of the doubled system register (code: \texttt{solution\_slice = slice(d0, 2 * d0)}). The extra
qubit distinguishes the halves; post-selecting it on $1$ isolates $\ket x$ (here it is $1$ with certainty, since $\vec y = \vec 0$)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""The exact value of the ideal success probability $0.355903$ is""",
         opts=[r"$\dfrac{205}{576}$", r"$\dfrac14$", r"$\dfrac{25}{48}$", r"$\dfrac{205}{144}$"],
         sol=r"""$P_{\text{success}} = \sum_j|\beta_j|^2\frac{C^2}{\lambda_j^2} = \frac14\left(1 + \frac14 + \frac19 + \frac1{16}\right) = \frac14\cdot\frac{144 + 36 + 16 + 9}{144} = \frac{205}{576} = 0.3559028$.
It also equals $C^2\lVert\vec x\rVert^2/\lVert\vec b\rVert^2 = \frac{25^2 + 7^2 + 11^2 + 5^2}{48^2} = \frac{820}{2304} = \frac{205}{576}$. ($\frac{205}{144}$ forgets $|\beta_j|^2 = \frac14$;
$\frac{25}{48}$ is $x_0$.) Since every $\lambda_j$ is exactly representable, the state-vector run reproduces it to all printed digits."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Without knowing $\vec x$ in advance, the norm $\lVert\vec x\rVert$ can be inferred from the success probability. Its value is""",
         opts=[r"$\dfrac{\sqrt{205}}{24} \approx 0.597$", r"$\dfrac{205}{576} \approx 0.356$", r"$1$, because quantum states are normalised", r"$\dfrac{24}{\sqrt{205}} \approx 1.676$"],
         sol=r"""The accepted branch is $\vec a = \frac{C}{\lVert\vec b\rVert}\vec x$ (C13) and $\lVert\vec a\rVert^2 = P_{\text{success}}$, hence
\[ \lVert\vec x\rVert = \frac{\lVert\vec b\rVert}{C}\sqrt{P_{\text{success}}} = \sqrt{\frac{205}{576}} = \frac{\sqrt{205}}{24} \approx 0.5966 . \]
Check: $\frac1{48}\sqrt{820} = \frac{2\sqrt{205}}{48} = \frac{\sqrt{205}}{24}$. On hardware the measured success rate ($0.3543 \pm 0.0034$) gives $\lVert\vec x\rVert \approx 0.595 \pm 0.003$.
The normalised state $\ket x$ has norm $1$, but the solution vector does not."""),
]
