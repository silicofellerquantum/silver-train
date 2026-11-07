"""Section D - Advanced Noise Analysis and Implementation (process and average gate fidelity, error rates, randomized
benchmarking, error mitigation: zero-noise extrapolation, readout mitigation, dynamical decoupling; quantum error
correction; coherent vs stochastic errors; CUDA-Q implementation and practical limitations).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""For the lecture (illustrative experiment, not part of the notebook), zero-noise extrapolation (ZNE) was applied to the parity
$\langle X^{\otimes4}\rangle$ of a 4-qubit GHZ state. The circuit was simulated exactly on \texttt{density-matrix-cpu} with the notebook's
depolarizing noise model at the scaled probability $p = 0.01\lambda$ (noise scale factor $\lambda$). The ideal value is $1$.
\begin{center}\small
\begin{tabular}{cccc}\toprule $\lambda$ & 1 & 2 & 3\\ \midrule $\langle X^{\otimes4}\rangle$ & 0.9055 & 0.8189 & 0.7396\\ \bottomrule\end{tabular}
\end{center}""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""For a single-qubit depolarizing channel (CUDA-Q convention: Kraus operators $\sqrt{1-p}\,I$, $\sqrt{p/3}\,X, Y, Z$) the process (entanglement) fidelity is
$F_{\text{pro}} = 1 - p$. Using $F_{\text{avg}} = \frac{dF_{\text{pro}} + 1}{d + 1}$, the average gate fidelity for $p = 0.003$ is""",
         opts=[r"$0.9980$", r"$0.9970$", r"$0.99775$", r"$0.9990$"],
         sol=r"""$d = 2$: $F_{\text{avg}} = \frac{2(1 - p) + 1}{3} = 1 - \frac{2p}{3} = 1 - 0.002 = 0.9980$; the average gate infidelity (``error rate'') is $r = 0.002$.
$F_{\text{pro}} = 0.997$ itself is lower because it does not average over input states. $1 - \frac{3p}{4}$ mixes up the $\lambda = \frac{4p}{3}$ convention."""),

    dict(id="D2", kind="mcq",
         q=r"""Single-qubit randomized benchmarking fits the survival probability to $A\alpha^m + B$ ($m$ = sequence length) and finds $\alpha = 0.99$. The error per Clifford
$r = \frac{(d - 1)(1 - \alpha)}{d}$ is""",
         opts=[r"$0.005$", r"$0.010$", r"$0.0067$", r"$0.0025$"],
         sol=r"""$d = 2$: $r = \frac{1\cdot(1 - 0.99)}{2} = 0.005$. RB is robust to state-preparation and measurement errors (absorbed in $A$, $B$) and reports an
\emph{average} error per gate under twirled (effectively depolarizing) noise. $1 - \alpha = 0.01$ forgets the factor $\frac{d-1}{d}$."""),

    dict(id="D3", kind="mcq",
         q=r"""Zero-noise extrapolation measures $E(1) = 0.80$ at the native noise and $E(2) = 0.64$ at doubled noise. The \emph{linear} extrapolation to $\lambda = 0$ gives""",
         opts=[r"$0.96$", r"$1.00$", r"$0.80$", r"$0.88$"],
         sol=r"""Line through $(1, 0.80)$ and $(2, 0.64)$: slope $-0.16$, so $E(0) = 0.80 + 0.16 = 0.96$ (formula $2E(1) - E(2)$).
An exponential model $E = ae^{-b\lambda}$ would give $E(1)^2/E(2) = 1.00$: the result depends on the assumed model --- ZNE is a biased estimator."""),

    dict(id="D4", kind="mcq",
         q=r"""ZNE with three noise levels gives $E(1) = 0.80$, $E(2) = 0.65$, $E(3) = 0.53$. The quadratic (Richardson) extrapolation $E(0) = 3E(1) - 3E(2) + E(3)$ equals""",
         opts=[r"$0.98$", r"$0.95$", r"$0.53$", r"$0.985$"],
         sol=r"""$3(0.80) - 3(0.65) + 0.53 = 2.40 - 1.95 + 0.53 = 0.98$. \emph{Check}: the parabola through the three points has second difference
$0.53 - 2(0.65) + 0.80 = 0.03$, so $E(\lambda) = 0.98 - 0.195\lambda + 0.015\lambda^2$. The linear estimate from the first two points is $0.95$; an exponential fit through $\lambda = 1, 2$ gives $0.8^2/0.65 = 0.985$.
The coefficients $(3, -3, 1)$ amplify the statistical errors: $\sqrt{9 + 9 + 1} \approx 4.4$ times the error of one point."""),

    dict(id="D5", kind="mcq",
         q=r"""A circuit has 10 gates of equal error. To amplify noise, 4 of them are ``folded'' ($G \to GG^\dagger G$), the others are left unchanged. Assuming the noise grows with
the number of gates, the effective noise scale factor is""",
         opts=[r"$1.8$", r"$1.4$", r"$3.0$", r"$2.2$"],
         sol=r"""Each folded gate becomes 3 gates: $4\times3 + 6\times1 = 18$ gates instead of 10, so $\lambda = 18/10 = 1.8$. Folding all gates gives $\lambda = 3$.
Folding keeps the ideal unitary unchanged ($GG^\dagger G = G$) while multiplying the noise --- the hardware analogue of the lecture's ``scale $p$'' experiment."""),

    dict(id="D6", kind="mcq",
         q=r"""Readout calibration of one qubit gives $P(\text{read }0|0) = 0.98$ and $P(\text{read }0|1) = 0.05$. An experiment reads $0$ with frequency $0.60$. The mitigated
probability of the true state $\ket0$ is about""",
         opts=[r"$0.591$", r"$0.600$", r"$0.612$", r"$0.604$"],
         sol=r"""Solve $\begin{psmallmatrix}0.98 & 0.05\\ 0.02 & 0.95\end{psmallmatrix}\begin{psmallmatrix}p_0\\ p_1\end{psmallmatrix} = \begin{psmallmatrix}0.60\\ 0.40\end{psmallmatrix}$ with
$p_0 + p_1 = 1$: $0.98p_0 + 0.05(1 - p_0) = 0.60 \Rightarrow 0.93p_0 = 0.55 \Rightarrow p_0 = 0.5914$. $0.6/0.98 = 0.612$ ignores the $1\to0$ errors;
$0.604$ comes from the transposed confusion matrix."""),

    dict(id="D7", kind="mcq",
         q=r"""Which statement about readout-error mitigation by inverting the calibrated confusion matrix $A$ is correct?""",
         opts=[r"$A^{-1}\mathbf q$ can contain small negative ``quasi-probabilities'', and the full $2^N\times2^N$ matrix becomes impractical for many qubits",
               r"it also corrects gate errors that happened before the measurement",
               r"it reduces the number of shots needed, because $A^{-1}$ removes the shot noise",
               r"it requires quantum error correction to estimate $A$"],
         sol=r"""Inversion only undoes the classical misassignment of outcomes. Shot noise in $\mathbf q$ propagates (and is amplified) through $A^{-1}$, which can produce slightly
negative entries --- the lecture's 3-qubit example gave a minimum of $-0.0009$; they are usually clipped or handled by constrained least squares. $A$ is measured
by preparing and reading basis states; for many qubits one assumes tensor-product or correlated sparse models instead of the full $4^N$-entry matrix."""),

    dict(id="D8", kind="mcq",
         q=r"""A qubit in $\ket+$ idles for $t$ with a random but \emph{static} detuning $\delta \sim \mathcal N(0, \sigma^2)$, $\sigma = 1$ rad/$\mu$s, so it acquires the phase $\delta t$. At
$t = 2\,\mu$s the average coherence $\langle X\rangle$ without and with a Hahn echo ($X$ pulse at $t/2$ and again at the end) is""",
         opts=[r"$0.135$ and $1.0$", r"$0.368$ and $1.0$", r"$0.135$ and $0.135$", r"$0.607$ and $1.0$"],
         sol=r"""Without echo: $\langle X\rangle = \mathbb E\cos(\delta t) = e^{-\sigma^2t^2/2} = e^{-2} = 0.135$ (Gaussian characteristic function).
With echo: $R_Z(\varphi/2)\,X\,R_Z(\varphi/2) = X$ because $XR_Z(a)X = R_Z(-a)$; the two halves cancel for \emph{every} static $\delta$, so $\langle X\rangle = 1$.
($e^{-1} = 0.368$ uses $e^{-\sigma^2t^2/4}$; $0.607 = e^{-1/2}$ uses $t = 1$.)"""),

    dict(id="D9", kind="mcq",
         q=r"""Now the dephasing is Markovian: an independent phase-flip channel with $p = 0.02$ acts after each of 20 idle steps on $\ket+$. With a Hahn echo inserted at step 10,
$\langle X\rangle$ at the end is""",
         opts=[r"$0.442$, the same as without echo", r"$1.000$", r"$0.668$", r"$0$"],
         sol=r"""Each step multiplies $\langle X\rangle$ by $1 - 2p = 0.96$: $0.96^{20} = 0.442$. The $X$ pulses commute with the phase-flip channel up to the sign that the second
$X$ undoes, and independent random flips in the two halves do not cancel. The lecture's density-matrix simulation gives $0.4420$ both with and without echo:
dynamical decoupling only fights noise that is slow (correlated in time) compared with the pulse spacing."""),

    dict(id="D10", kind="mcq",
         q=r"""In the 3-qubit bit-flip code each qubit flips independently with probability $p$, and encoding and correction are perfect. The logical error probability at $p = 0.1$, and
the value of $p$ above which encoding is worse than an unencoded qubit, are""",
         opts=[r"$0.028$ and $p = 0.5$", r"$0.030$ and $p = 0.5$", r"$0.028$ and $p = 1/3$", r"$0.010$ and $p = 0.5$"],
         sol=r"""Majority vote fails when 2 or 3 qubits flip: $P_L = 3p^2(1-p) + p^3 = 3p^2 - 2p^3 = 0.03 - 0.002 = 0.028$ (the lecture's CUDA-Q density-matrix simulation
reproduces this formula to $10^{-16}$). $P_L < p \Leftrightarrow 3p - 2p^2 < 1 \Leftrightarrow (2p - 1)(p - 1) > 0 \Leftrightarrow p < \frac12$. $3p^2 = 0.030$ drops the
$-2p^3$ term."""),

    dict(id="D11", kind="mcq",
         q=r"""A surface-code memory has logical error rate $p_L \approx A(p/p_{\text{th}})^{(d+1)/2}$ for code distance $d$. With $p/p_{\text{th}} = 0.1$, increasing the distance from
$d = 3$ to $d = 5$ reduces $p_L$ by a factor of""",
         opts=[r"$10$", r"$100$", r"$2$", r"none: $p_L$ increases by a factor of 10"],
         sol=r"""The exponent goes from $2$ to $3$: $p_L$ is multiplied by $(p/p_{\text{th}})^{1} = 0.1$, i.e.\ reduced by a factor of 10. Below threshold ($p < p_{\text{th}}$) larger
codes suppress errors exponentially in $d$; above threshold they make things worse. QEC needs many physical qubits and fast feedback --- it is not
implemented in this repository."""),

    dict(id="D12", kind="mcq",
         q=r"""Each of $n = 60$ idle gates over-rotates coherently by $\varepsilon = 0.05$ rad about $x$. Compare with a stochastic bit flip of the same per-gate error probability
$p = \sin^2(\varepsilon/2)$. Starting from $\ket0$, the probabilities of finding $1$ after 60 gates are approximately""",
         opts=[r"$0.995$ (coherent) and $0.036$ (stochastic)", r"$0.036$ and $0.036$", r"$0.0375$ and $0.036$", r"$0.995$ and $0.995$"],
         sol=r"""Coherent: the rotations add up, $R_X(\varepsilon)^{60} = R_X(3)$: $P(1) = \sin^2(1.5) = 0.995$. Stochastic: $p = \sin^2(0.025) = 6.25\times10^{-4}$ per gate and
$P(1) = \frac12[1 - (1 - 2p)^{60}] = \frac12(1 - 0.9278) = 0.036 \approx np$. Coherent errors grow like $n^2\varepsilon^2$ (amplitudes add), stochastic ones like $np$.
Both values were computed with CUDA-Q for the lecture. Randomized compiling/Pauli twirling turns the former into the latter."""),

    dict(id="D13", kind="mcq",
         q=r"""Which statement about Pauli twirling (randomized compiling) is correct?""",
         opts=[r"averaging a gate over random Pauli frames that cancel ideally turns its coherent error into a stochastic Pauli channel with the same average fidelity",
               r"it removes all errors, so the circuit becomes noiseless",
               r"it requires a density-matrix simulator and cannot be done on hardware",
               r"it converts stochastic Pauli noise into coherent errors to make them easier to correct"],
         sol=r"""Inserting random Paulis $P$ before a gate and the compensating $P'$ after it leaves the ideal circuit unchanged; averaged over many random compilations the
effective channel is diagonal in the Pauli basis --- a Pauli (stochastic) channel. This prevents the quadratic, worst-case accumulation of coherent errors and makes
noise models such as those of the notebook (Pauli channels) realistic. It does not reduce the average error itself."""),

    dict(id="D14", kind="mcq",
         q=r"""The noise notebook runs on a CPU (no GPU visible) with the settings\\
\hspace*{1em}\texttt{CIRCUIT = "ghz"},\ \ \texttt{NOISE\_MODEL = "amplitude\_damping"},\ \ \texttt{SIMULATION\_METHOD = "auto"}.\\
Which method does it choose?""",
         opts=[r"\texttt{density\_matrix} (target \texttt{density-matrix-cpu})", r"\texttt{stabilizer} (target \texttt{stim})",
               r"\texttt{trajectory} (target \texttt{nvidia})", r"\texttt{statevector} (target \texttt{qpp-cpu})"],
         sol=r"""The \texttt{"auto"} rule: GPU visible $\to$ \texttt{trajectory}; otherwise \texttt{stabilizer} only for GHZ \emph{with Pauli noise}; otherwise \texttt{density\_matrix}.
Amplitude damping is not a Pauli channel, so \texttt{stim} would silently ignore it --- the notebook even raises an error if \texttt{stabilizer} is forced with it.
\texttt{statevector} is refused because noiseless simulators ignore noise models."""),

    dict(id="D15", kind="match",
         q=r"""Match each technique (List-I) with its description (List-II).""",
         match=([r"zero-noise extrapolation", r"readout-error mitigation", r"dynamical decoupling", r"quantum error correction"],
                [r"invert a calibrated confusion matrix applied to the measured distribution",
                 r"run the circuit at amplified noise levels and extrapolate the observable to zero noise",
                 r"encode logical qubits redundantly, measure syndromes and correct; needs physical errors below a threshold",
                 r"apply pulse sequences during idle periods to refocus slow (quasi-static) dephasing"]),
         opts=[r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3", r"P$\to$1, Q$\to$2, R$\to$4, S$\to$3",
               r"P$\to$2, Q$\to$1, R$\to$3, S$\to$4", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item ZNE: an \emph{estimator} for expectation values, post-processing (2);
\item readout mitigation: classical post-processing of counts (1);
\item DD: changes the physical evolution during idling (4);
\item QEC: protects the quantum state itself during the computation (3).
\end{itemize}
Only QEC corrects errors on the quantum state; ZNE and readout mitigation correct estimates; noise \emph{simulation} does neither. Hence P$\to$2, Q$\to$1, R$\to$4, S$\to$3."""),

    dict(id="D16", kind="match",
         q=r"""Match each physical cause (List-I) with its category (List-II).""",
         match=([r"a pulse amplitude 1\,\% too large on every $X$ gate", r"a random Pauli error after each gate",
                 r"energy loss of $\ket1$ to the environment", r"the discriminator assigns some $\ket1$ shots to outcome 0"],
                [r"decoherence (non-unital, $T_1$)", r"classical measurement (readout) error",
                 r"coherent error (unitary, systematic)", r"stochastic (incoherent) gate noise"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item over-rotation: the same unitary error every time (3);
\item random Pauli: a probabilistic mixture --- the notebook's model (4);
\item energy loss: amplitude damping, $T_1$ (1);
\item misassignment: acts on the classical outcome after the quantum measurement (2).
\end{itemize}
Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="D17", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} Zero-noise extrapolation increases the number of circuit executions needed to reach a given statistical precision.\\[1mm]
\textbf{Reason (R):} ZNE corrects errors on the quantum state during the computation, in the same way as quantum error correction.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true}: several noise levels must be run, and extrapolation coefficients (e.g.\ $3, -3, 1$) amplify the statistical error, so more shots are needed for the same
error bar. \textbf{R is false}: ZNE never touches the quantum state; it post-processes noisy expectation values and returns a (biased) estimate. Only QEC protects the
state, and it can do so for arbitrary-length computations. Answer: (C)."""),

    dict(id="D18", kind="stmt", correct=1,
         q=r"""\textbf{Statement I:} \texttt{Noise\_Simulation.ipynb} includes measurement (readout) errors in its noise model.\\[1mm]
\textbf{Statement II:} The recorded results of \texttt{Noise\_Simulation.ipynb} were obtained on a physical quantum processor.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false}: \texttt{build\_noise\_model} attaches channels only to \texttt{h}, \texttt{x}, \texttt{ry}, \texttt{rz} and CNOTs; measurements are ideal.
\textbf{II is false}: the recorded run used the \texttt{stim} simulator on a CPU (no GPU visible); the notebook even refuses \texttt{BACKEND = "hardware"}, because noise
models are a simulator feature. Answer: (B)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""The linear extrapolation from $\lambda = 1, 2$ and the Richardson (quadratic) extrapolation from $\lambda = 1, 2, 3$ give""",
         opts=[r"$0.9921$ and $0.9994$", r"$0.9055$ and $0.9921$", r"$0.9921$ and $1.0021$", r"$0.9872$ and $0.9994$"],
         sol=r"""Linear: $2E(1) - E(2) = 1.8110 - 0.8189 = 0.9921$. Richardson: $3E(1) - 3E(2) + E(3) = 2.7165 - 2.4567 + 0.7396 = 0.9994$.
Both reduce the error of the raw value $0.9055$ ($9.5\,\%$) to below $1\,\%$. $1.0021$ is the exponential fit, which overshoots the physical maximum."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Which conclusion about this ZNE experiment is correct?""",
         opts=[r"the extrapolated values depend on the fitted model (an exponential fit gives $1.0021 > 1$, unphysical) and, with shot noise, would carry larger error bars than the raw value",
               r"ZNE has corrected the GHZ state itself, so a subsequent circuit could use the improved state",
               r"because the exact density matrix was used, ZNE is no longer needed: the raw value $0.9055$ is already the ideal one",
               r"on hardware the same result is obtained without scaling the noise, since $\lambda$ is a software parameter"],
         sol=r"""ZNE estimates an expectation value; the state remains noisy. Different extrapolation models give $0.9921$, $0.9994$, $1.0021$ --- a model-dependent
bias, and $1.0021$ exceeds the allowed range of a parity. Here the inputs were exact; with $S$ shots per point, the Richardson combination would multiply the
statistical error by about $\sqrt{19} \approx 4.4$. On hardware the noise cannot be scaled by changing $p$; it is amplified physically, e.g.\ by gate folding or pulse
stretching."""),
]
