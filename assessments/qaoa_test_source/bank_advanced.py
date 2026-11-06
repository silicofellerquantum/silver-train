"""Section D - Advanced QAOA and implementation (light cones, gradients, optimizers, shot noise, memory and backends,
noise, variants, parameter strategies, benchmarking and interpretation of the recorded run).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""\emph{Previously recorded run of \texttt{QAOA.ipynb}} (default settings, CUDA-Q 0.16.0, target \texttt{qpp-cpu}):
15 nodes, 30 unit edges ($W = 30$), exact optimum $C_{\max} = 25$ with the four optimal bitstrings listed in A14,
$p = 2$, COBYLA with 80 evaluations, best expected cut $\langle C\rangle = 20.5336$, $4000$ shots. The four most frequent
bitstrings were all optimal:
$\texttt{110010001001101}$ (68 shots), $\texttt{001101110110010}$ (67), $\texttt{110010001011101}$ (56), $\texttt{001101110100010}$ (49);
in total 240 shots were optimal.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""On a ring graph, the expectation $\langle Z_uZ_v\rangle$ for one edge $(u, v)$ of a depth-$p$ QAOA state depends on at most how
many qubits when $p = 3$?""",
         opts=[r"$8$", r"$6$", r"$7$", r"all $n$ qubits"],
         sol=r"""Work in the Heisenberg picture: $\langle Z_uZ_v\rangle = \bra{+}^{\otimes n}U^\dagger Z_uZ_vU\ket{+}^{\otimes n}$. Conjugating by a mixer layer
(single-qubit gates) keeps the support. Conjugating by a cost layer can only add the neighbours of the current support (an
$e^{-i\gamma Z_aZ_b/2}$ acts trivially unless it touches the support). Each layer enlarges the support by one node on each side
of the ring: $\{u, v\}$ (2 qubits) $\to$ 4 $\to$ 6 $\to$ 8 after $p = 3$ layers, i.e.\ $2p + 2 = 8$ qubits.
This \emph{light cone} is why fixed-$p$ QAOA is local, and why large instances of bounded degree can be analysed with small
subgraphs."""),

    dict(id="D2", kind="mcq",
         q=r"""For even rings $C_n$ with $n \ge 2p + 2$, numerical optimisation of the angles gives the best $p = 2$ approximation ratio""",
         opts=[r"$5/6$", r"$3/4$", r"$7/8$", r"$1$"],
         sol=r"""For $n \ge 2p + 2$ the light cone of an edge (previous question) is a path of $2p + 2$ nodes that does not close around the
ring, so every edge sees the same local problem and the ratio does not depend on $n$. Optimising $F_2$ numerically for
$C_6$, $C_8$ and $C_{10}$ (\texttt{verify\_answers.py}) gives $r = 0.83333 = 5/6$; for $p = 1$ it gives $3/4$ (C8) and for $p = 3$, $7/8$.
These values follow the pattern $r_p = \frac{2p+1}{2p+2}$ conjectured by Farhi et al.\ for the ring: the ratio improves with depth
but reaches $1$ only as $p \to \infty$ for large rings."""),

    dict(id="D3", kind="mcq",
         q=r"""For the default circuit ($n = 15$, $|E| = 30$, $p = 2$), compare two ways of computing the gradient of $F_2$ with respect to
all $2p$ angles: (i) the exact parameter-shift rule applied to every rotation gate separately (each $R_Z$ and $R_X$ gate is shifted
by $\pm\pi/2$ on its own), and (ii) a forward finite-difference gradient as used by L-BFGS-B. The numbers of circuit evaluations are""",
         opts=[r"$180$ and $5$", r"$8$ and $5$", r"$180$ and $4$", r"$90$ and $5$"],
         sol=r"""\textbf{(i)} $\gamma_l$ enters $|E|$ gates $R_Z(\gamma_lw_e)$; by the chain rule $\partial F/\partial\gamma_l = \sum_e w_e\,\partial F/\partial\theta_e$, and each
$\partial F/\partial\theta_e$ needs two runs (shifted by $\pm\pi/2$, A16). So $2|E|$ runs per $\gamma_l$. Likewise $\beta_l$ enters $n$ gates
$R_X(2\beta_l)$: $2n$ runs. Total $2p(|E| + n) = 2\cdot2\cdot(30 + 15) = 180$.
A single two-term shift of $\gamma_l$ as a whole is \emph{not} exact: its generator $H_C$ has many different eigenvalue gaps.
\textbf{(ii)} $f(\boldsymbol\theta)$ once plus one shifted run per parameter: $2p + 1 = 5$ (the notebook's table: ``$2p+1$ circuit runs per gradient'').
Finite differences are cheaper but approximate and sensitive to shot noise."""),

    dict(id="D4", kind="mcq",
         q=r"""The notebook optimises the angles with SciPy's \texttt{minimize(..., method="COBYLA")} and the options
\texttt{\{"maxiter": 80, "rhobeg": 0.2\}}. The option \texttt{rhobeg = 0.2} sets""",
         opts=[r"the initial size of the trust region, i.e.\ the first steps change the angles by about $0.2$ rad",
               r"a learning rate multiplying the gradient",
               r"the tolerance at which COBYLA declares convergence",
               r"the maximum number of iterations divided by 400"],
         sol=r"""COBYLA (Constrained Optimization BY Linear Approximation) is derivative-free: it builds linear models of the objective
from function values at a simplex of points and takes steps inside a trust region of radius $\rho$. \texttt{rhobeg} is the
starting radius: the first probes move each angle by about $0.2$ rad. $\rho$ then shrinks towards \texttt{tol}/\texttt{rhoend} as the
search converges. No gradients (and therefore no learning rate) are used. With only 80 evaluations a too small \texttt{rhobeg}
would explore little; a too large one would jump across the oscillating landscape."""),

    dict(id="D5", kind="mcq",
         q=r"""With \texttt{OBJECTIVE\_SHOTS = 1000}, $\langle H_C\rangle$ is estimated from 1000 shots. If the cut values of the sampled
bitstrings have a standard deviation of $2.5$, the standard error of the estimated objective is about""",
         opts=[r"$0.079$", r"$0.0025$", r"$2.5$", r"$0.0063$"],
         sol=r"""The shot estimate is the sample mean of $H_C(z) = \tfrac W2 - C(z)$; its spread equals that of $C(z)$, $\sigma = 2.5$.
Standard error of a mean of $S$ independent samples: $\sigma/\sqrt S = 2.5/\sqrt{1000} = 2.5/31.6 \approx 0.079$.
Differences between neighbouring optimizer steps near the optimum are often of this size or smaller, so the optimizer starts
to chase noise; that is why the notebook warns that shot-based objectives ``make optimisation harder''."""),

    dict(id="D6", kind="mcq",
         q=r"""For $n = 22$ (\texttt{BRUTE\_FORCE\_MAX\_NODES}), the notebook's \texttt{brute\_force\_maxcut} creates the arrays
\texttt{index = np.arange(2**N, dtype=np.int64)} and \texttt{values = np.zeros(2**N)}. Together they occupy about""",
         opts=[r"$67$ MB", r"$34$ MB", r"$4.2$ MB", r"$0.5$ MB"],
         sol=r"""$2^{22} = 4\,194\,304$ entries per array. \texttt{int64} and \texttt{float64} both use $8$ bytes:
$2 \times 8 \times 4\,194\,304 = 67\,108\,864$ bytes $\approx 67$ MB (plus temporaries of the same size created per edge while
evaluating \texttt{(index >> a) \^{} (index >> b)}). $34$ MB is one array; $4.2$ MB is the number of \emph{entries} read as bytes.
Each additional node doubles both memory and time, which is why the exact reference stops at 22 nodes."""),

    dict(id="D7", kind="mcq",
         q=r"""A 36-qubit QAOA state vector is simulated in fp64. If it is split evenly over a power-of-two number of 80 GB GPUs (as the
multi-GPU state-vector target does), ignoring workspace, the minimum number of GPUs is""",
         opts=[r"$16$", r"$14$", r"$8$", r"$32$"],
         sol=r"""Memory: $16\times2^{36} = 1.0995\times10^{12}$ bytes $\approx 1.10$ TB. Dividing by $80$ GB: $13.7$, so at least $14$ GPUs by memory
alone. The state is distributed in power-of-two slices ($2^k$ GPUs each holding $2^{36-k}$ amplitudes): $2^3 = 8$ GPUs would need
$137$ GB each (too much), $2^4 = 16$ GPUs need $68.7$ GB each (fits). So 16 GPUs, launched with one MPI rank per GPU
(\texttt{BACKEND = "multi-gpu"}, \texttt{mpiexec}). These GPU paths are implemented in the notebook but were not executed on the
CPU-only test machine."""),

    dict(id="D8", kind="match",
         q=r"""Match each setting of \texttt{BACKEND} and \texttt{SIMULATION\_METHOD} in \texttt{QAOA.ipynb} (List-I) with what the notebook's
\texttt{resolve\_backend} does (List-II).""",
         match=([r'\texttt{"cpu"} + \texttt{"statevector"}', r'\texttt{"cpu"} + \texttt{"density\_matrix"}',
                 r'\texttt{"gpu"} + \texttt{"mps"}', r'\texttt{"cpu"} + \texttt{"mps"}'],
                [r"target \texttt{tensornet-mps}", r"stops with an error: the MPS simulator needs an NVIDIA GPU",
                 r"target \texttt{qpp-cpu}", r"target \texttt{density-matrix-cpu}"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1",
               r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""From \texttt{\_TARGET\_TABLE} and \texttt{\_UNSUPPORTED\_HINTS} in the shared utilities (QAOA allows the methods
\texttt{statevector}, \texttt{mps}, \texttt{tensornet}, \texttt{density\_matrix}):
\begin{itemize}\itemsep0pt
\item P: \texttt{("cpu", "statevector")} $\to$ \texttt{qpp-cpu} (OpenMP state vector, fp64) (3) --- the target of the recorded run.
\item Q: \texttt{("cpu", "density\_matrix")} $\to$ \texttt{density-matrix-cpu} (4); memory $16\times4^n$ bytes.
\item R: \texttt{("gpu", "mps")} $\to$ \texttt{tensornet-mps} (1).
\item S: \texttt{("cpu", "mps")} is not in the table; the hint says ``CUDA-Q's MPS simulator (target 'tensornet-mps') needs an
NVIDIA GPU'' and a \texttt{ValueError} is raised (2).
\end{itemize}"""),

    dict(id="D9", kind="mcq",
         q=r"""For which instance is a matrix-product-state (MPS) simulation of a depth-$p$ QAOA circuit most likely to be efficient (small
bond dimension), with the qubits ordered along a line $0, 1, \dots, n-1$?""",
         opts=[r"A ring $C_n$ with small $p$: every cut of the line is crossed by only 2 edges",
               r"A dense random graph with edge probability $0.5$: more edges average out the entanglement",
               r"A complete graph $K_n$: its symmetry makes the MPS exact with bond dimension 1",
               r"Any graph, because QAOA states are always product states before measurement"],
         sol=r"""An MPS across a cut between qubits $k$ and $k+1$ needs a bond dimension at least the Schmidt rank there. Each cost gate
$e^{-i\theta Z_aZ_b/2}$ that straddles the cut can at most double that rank, so after $p$ layers $\chi \le 2^{p\cdot m_k}$, where $m_k$ is the
number of edges crossing the cut. For a ring in natural order $m_k = 2$ for every cut, so $\chi \le 4^p$: small for small $p$.
Dense random graphs have $m_k = \Theta(n^2)$ crossings: the bond dimension can grow exponentially. $K_n$ is not product-like
either: its cost layer entangles all pairs (B8). The cost layer creates entanglement, so QAOA states are not product states."""),

    dict(id="D10", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} When \texttt{BACKEND = "hardware"}, the notebook estimates the objective from shots even if
\texttt{OBJECTIVE\_SHOTS = None}.\\[1mm]
\textbf{Reason (R):} Hardware (and hardware-emulation) targets only return measurement samples; they cannot give a state vector or an
exact expectation value.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} the code sets
\texttt{objective\_shots = OBJECTIVE\_SHOTS if OBJECTIVE\_SHOTS is not None else (SHOTS if backend\_cfg["device"] == "hardware" else None)},
so on hardware $\langle H_C\rangle$ is estimated with \texttt{SHOTS} $= 4000$ shots per evaluation.
\textbf{R is true and is the reason:} a QPU only produces bitstrings; exact expectation values exist only in simulators
(README: ``Hardware targets support sampling only''). Answer: (A)."""),

    dict(id="D11", kind="mcq",
         q=r"""A simple global-depolarising noise model treats the noisy QAOA output as a mixture: with probability $F = (1 - \varepsilon)^{N_{\text{CNOT}}}$
the ideal state, otherwise a uniformly random bitstring. For the default circuit ($N_{\text{CNOT}} = 120$, ideal $\langle C\rangle = 20.53$,
$W/2 = 15$) and $\varepsilon = 0.005$ per CNOT, the predicted noisy $\langle C\rangle$ is about""",
         opts=[r"$18.0$", r"$11.3$", r"$17.2$", r"$15.0$"],
         sol=r"""$F = 0.995^{120} = e^{120\ln0.995} = e^{-0.6015} \approx 0.548$.
Mixture: $\langle C\rangle_{\text{noisy}} = F\,\langle C\rangle_{\text{ideal}} + (1 - F)\,\tfrac W2 = 0.548\cdot20.53 + 0.452\cdot15 = 11.25 + 6.78 \approx 18.0$.
$11.3$ forgets the random part (a random bitstring still cuts $W/2$ on average); $17.2$ uses the linearisation
$F \approx 1 - 120\varepsilon = 0.4$, which is poor when $120\varepsilon$ is not small. The model is illustrative only (real noise is
not exactly global depolarising), but it shows how noise pulls QAOA towards the random baseline and limits useful depth."""),

    dict(id="D12", kind="match",
         q=r"""Match each QAOA variant (List-I) with its key idea (List-II).""",
         match=([r"warm-start QAOA", r"ADAPT-QAOA", r"Quantum Alternating Operator Ansatz (Hadfield et al.)",
                 r"recursive QAOA (RQAOA)"],
                [r"problem-specific mixers (e.g.\ XY mixers) that keep the state inside the feasible subspace of a constrained problem",
                 r"use measured correlations $\langle Z_iZ_j\rangle$ to fix and eliminate variables, then solve the smaller instance",
                 r"start from a state biased towards a classical (e.g.\ relaxed/SDP) solution instead of $\ket+^{\otimes n}$",
                 r"grow the mixer layer by layer from an operator pool, choosing the operator with the largest energy gradient"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: warm-start QAOA (Egger, Mare\v{c}ek, Woerner 2021) changes the initial state (and matching mixer) using a classical relaxation (3).
\item Q: ADAPT-QAOA (Zhu et al.\ 2022) selects mixers adaptively from a pool by gradient (4).
\item R: the Quantum Alternating Operator Ansatz (Hadfield et al.\ 2019) generalises the mixer to respect constraints (1).
\item S: RQAOA (Bravyi et al.\ 2020) eliminates variables using the strongest correlations, recursively (2).
\end{itemize}
None of these is implemented in \texttt{QAOA.ipynb}, which uses the standard $X$ mixer and $\ket+^{\otimes n}$."""),

    dict(id="D13", kind="mcq",
         q=r"""The INTERP initialisation (Zhou et al.\ 2020) builds depth-$(p+1)$ starting angles from optimised depth-$p$ angles by
$\gamma_i^{(p+1)} = \tfrac{i-1}{p}\,\gamma_{i-1}^{(p)} + \tfrac{p-i+1}{p}\,\gamma_i^{(p)}$ for $i = 1, \dots, p+1$ (with $\gamma_0^{(p)} = \gamma_{p+1}^{(p)} = 0$).
From $\boldsymbol\gamma^{(2)} = (0.4, 0.8)$ it gives $\boldsymbol\gamma^{(3)} =$""",
         opts=[r"$(0.4,\ 0.6,\ 0.8)$", r"$(0.4,\ 0.8,\ 0.8)$", r"$(0.4,\ 0.8,\ 0)$", r"$(0.27,\ 0.53,\ 0.8)$"],
         sol=r"""$p = 2$.
$i = 1$: $\tfrac02\gamma_0 + \tfrac22\gamma_1 = 0.4$.
$i = 2$: $\tfrac12\gamma_1 + \tfrac12\gamma_2 = \tfrac12(0.4 + 0.8) = 0.6$.
$i = 3$: $\tfrac22\gamma_2 + \tfrac02\gamma_3 = 0.8$.
So $(0.4, 0.6, 0.8)$: the optimised schedule is linearly interpolated onto one more point, keeping its smooth shape. This is a
general strategy from the literature; the notebook itself starts every depth from its fixed ramp."""),

    dict(id="D14", kind="stmt", correct=3,
         q=r"""For graphs with more nodes than \texttt{BRUTE\_FORCE\_MAX\_NODES}, the notebook replaces the exact optimum by the best cut found
by \texttt{local\_search\_maxcut} and still reports $r = \langle C\rangle/\texttt{max\_cut}$.\\[1mm]
\textbf{Statement I:} In that case the reported \texttt{max\_cut} is guaranteed to be at least the true $C_{\max}$.\\[1mm]
\textbf{Statement II:} In that case the reported ratio is at least the true ratio $\langle C\rangle/C_{\max}$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""The heuristic returns the cut of an actual split, so $\texttt{max\_cut} \le C_{\max}$ (it can only miss the optimum, never exceed it).
\textbf{I is false} (the inequality goes the other way).
\textbf{II is true:} dividing the same positive $\langle C\rangle$ by a smaller (or equal) number gives a larger (or equal) value:
$\langle C\rangle/\texttt{max\_cut} \ge \langle C\rangle/C_{\max}$. The reported ratio is an \emph{upper bound} on the true one, which is why
the notebook labels the reference ``NOT guaranteed optimal''. Answer: (D)."""),

    dict(id="D15", kind="mcq",
         q=r"""In the recorded run, the quantum execution time was $40.01$ s for the $80$ objective evaluations (\texttt{cudaq.observe}
calls) of the $225$-gate circuit on 15 qubits. The average time per evaluation and per gate are about""",
         opts=[r"$0.50$ s and $2.2$ ms", r"$40$ s and $0.18$ s", r"$0.010$ s and $44\ \mu$s", r"$0.18$ s and $0.8$ ms"],
         sol=r"""Per evaluation: $40.01/80 \approx 0.50$ s. Per gate: $0.50/225 \approx 2.2\times10^{-3}$ s $= 2.2$ ms.
This matches the notebook's own calibration printed before execution: ``$1.0921$ ms/gate at 14 qubits, extrapolated $\times2^1$
to 15 qubits'' $\to 2.18$ ms/gate, and the estimate $\sim39.8$ s versus $40.84$ s measured for the whole cell.
($0.010$ s divides by the $4000$ shots, which play no role in exact \texttt{observe} calls.)"""),

    dict(id="D16", kind="mcq",
         q=r"""The notebook estimates run time as (number of evaluations) $\times$ (gates per circuit) $\times$ (time per gate), with the time per
gate scaling as $2^n$ for a state vector. Going from the default ($n = 15$, $|E| = 30$, $p = 2$) to $n = 25$ with $|E| = 90$ edges at
the same $p$ and evaluation budget, the run time grows by a factor of about""",
         opts=[r"$2.8\times10^{3}$", r"$1.0\times10^{3}$", r"$2.7$", r"$2.9\times10^{6}$"],
         sol=r"""Gates: $n + p(3|E| + n)$: $15 + 2(90 + 15) = 225$ before, $25 + 2(270 + 25) = 615$ after; ratio $615/225 = 2.73$.
Time per gate: $2^{25-15} = 1024$ times larger. Total factor $2.73\times1024 \approx 2.8\times10^3$, i.e.\ the recorded
$\approx 40$ s would become $\approx 31$ hours on the same 4-core CPU --- the reason the notebook suggests a GPU for 24--30 nodes.
$1.0\times10^3$ ignores the extra gates; $2.7$ ignores the exponential state size; $2.9\times10^6$ uses the $4^{\Delta n}$ scaling of a
density matrix. (The model is the notebook's rough estimate, quoted as accurate to about $\pm3\times$.)"""),

    dict(id="D17", kind="mcq",
         q=r"""Which statement about QAOA for Max-Cut reflects the current state of knowledge?""",
         opts=[r"No quantum advantage has been demonstrated: the classical Goemans--Williamson algorithm guarantees $0.878$ in polynomial time, and fixed-depth QAOA is limited by locality",
               r"QAOA with $p = 1$ beats Goemans--Williamson on every 3-regular graph",
               r"QAOA solves Max-Cut exactly in polynomial time for any fixed $p$",
               r"The recorded result $r = 0.82$ on 15 nodes proves a quantum speed-up"],
         sol=r"""Known facts: $p = 1$ QAOA guarantees $0.6924$ on 3-regular graphs (C9), below the Goemans--Williamson $0.878$. Locality
(light cone, D1) limits any fixed-$p$ QAOA: e.g.\ Bravyi et al.\ (2020) showed that on certain bipartite regular graphs low-depth
QAOA stays far from the optimum. Max-Cut is NP-hard, so no polynomial exact algorithm is expected. A 15-node simulation that a
laptop brute-forces in $0.01$ s cannot show a speed-up; QAOA's value is as a well-understood, tunable heuristic and a benchmark."""),

    dict(id="D18", kind="mcq",
         q=r"""With \texttt{GRAPH = "custom"} and \texttt{CUSTOM\_EDGES = [(0, 1, 1.0), (1, 2, 2.5), (0, 2, -1.0)]} (3 nodes), the maximum cut and
the random-split baseline printed by the notebook are""",
         opts=[r"$3.5$ and $1.25$", r"$4.5$ and $2.25$", r"$3.5$ and $2.25$", r"$2.5$ and $1.25$"],
         sol=r"""Negative weights are allowed (``weights may be any real numbers''); a cut edge with weight $-1$ \emph{lowers} the cut.
The non-trivial partitions isolate one node: $\{0\}$: edges $(0,1), (0,2)$: $1 - 1 = 0$. $\{1\}$: $(0,1), (1,2)$: $1 + 2.5 = 3.5$.
$\{2\}$: $(1,2), (0,2)$: $2.5 - 1 = 1.5$. So $C_{\max} = 3.5$ (strings $010$ and $101$).
Baseline $W/2$ with $W = 1 + 2.5 - 1 = 2.5$: $1.25$. The rest of the pipeline is unchanged: the edge gets \texttt{rz(gamma * w)} with
$w = -1$ and $H_C$ the term $-0.5\,Z_0Z_2$."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""Estimated from the recorded counts, the probability of sampling the partition described by $\texttt{110010001001101}$
(either this bitstring or its complement) is""",
         opts=[r"$0.0338$", r"$0.0170$", r"$0.0600$", r"$0.0263$"],
         sol=r"""The complement of $\texttt{110010001001101}$ is $\texttt{001101110110010}$ (flip every bit), which is the second most frequent
string. Both describe the same partition, so the estimate is $(68 + 67)/4000 = 135/4000 = 0.03375 \approx 0.0338$.
The other optimal partition ($\texttt{110010001011101}$ / $\texttt{001101110100010}$) has $(56 + 49)/4000 = 0.02625$; together
$0.03375 + 0.02625 = 0.06 = P_{\text{opt}}$. The nearly equal counts within each pair ($68$ vs $67$, $56$ vs $49$) reflect the exact
symmetry $|\braket{z|\psi}| = |\braket{\bar z|\psi}|$ of the QAOA state, up to shot noise."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""A stricter figure of merit measures the improvement over random guessing:
$\eta = \dfrac{\langle C\rangle - W/2}{C_{\max} - W/2}$. For the recorded run, $\eta$ is""",
         opts=[r"$0.553$", r"$0.821$", r"$0.600$", r"$0.221$"],
         sol=r"""$\eta = \dfrac{20.5336 - 15}{25 - 15} = \dfrac{5.5336}{10} \approx 0.553$.
The usual ratio $r = 20.5336/25 = 0.821$ gives random guessing already $15/25 = 0.6$, so $r$ looks better than the actual
progress. $\eta = 0$ for random guessing and $1$ for the optimum: QAOA at $p = 2$ closed about $55\,\%$ of the gap between random
guessing and the optimum."""),
]
