"""Section C - Noise Models and Simulation (Kraus operators, quantum channels and CPTP maps, density-matrix evolution,
noisy circuits, sampling, gate noise, readout noise, simulator configurations and the code of Noise_Simulation.ipynb).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""The previously recorded run of \texttt{Noise\_Simulation.ipynb} simulates the 15-qubit GHZ circuit ($H$ on qubit 0 and a chain of
14 CNOTs) with depolarizing noise after every gate on the stabilizer target \texttt{stim} (10\,000 shots, seed 42). Part of its table:
\begin{center}\small
\begin{tabular}{rcccr}\toprule
$p$ & population $P(0\cdots0) + P(1\cdots1)$ & parity $\langle X^{\otimes15}\rangle$ & $(1-p)^G$ & distinct outcomes\\ \midrule
0.01 & 0.8936 & 0.6910 & 0.8601 & 96\\
0.1 & 0.3232 & 0.0160 & 0.2059 & 1165\\ \bottomrule
\end{tabular}
\end{center}""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""On the target \texttt{density-matrix-cpu} a kernel prepares $\ket0$ and applies \texttt{rx(t, q[0])}, with \texttt{t = 0.0} passed as a kernel
argument. The noise model attaches \texttt{cudaq.DepolarizationChannel(0.3)} to every \texttt{rx}. \texttt{cudaq.observe} of $Z$ returns""",
         opts=[r"$0.6$", r"$0.7$", r"$0.4$", r"$1.0$"],
         sol=r"""\texttt{rx(0)} is the identity, so the state before the channel is $\ket0$ ($r_z = 1$). CUDA-Q's depolarizing channel has Kraus operators
$\sqrt{1-p}\,I$, $\sqrt{p/3}\,X$, $\sqrt{p/3}\,Y$, $\sqrt{p/3}\,Z$; it multiplies the Bloch vector by $1 - \frac{4p}{3} = 1 - 0.4 = 0.6$.
So $\langle Z\rangle = 0.6$ (measured with CUDA-Q 0.16 for the lecture). $\rho_{11} = \frac{2p}{3} = 0.2$."""),

    dict(id="C2", kind="mcq",
         q=r"""The same experiment is repeated, but the kernel now contains \texttt{rx(0.0, q[0])} with the angle written as a \emph{literal constant}. With CUDA-Q 0.16
the result is $\langle Z\rangle = 1.0$. The best explanation is""",
         opts=[r"the compiler removes the constant identity rotation, so there is no \texttt{rx} gate left to carry the noise channel",
               r"the depolarizing channel acts only on non-zero rotation angles by definition",
               r"\texttt{density-matrix-cpu} ignores noise models",
               r"noise channels are applied only before a gate, never after it"],
         sol=r"""Noise models attach channels to \emph{gate occurrences} in the compiled circuit. A rotation by a constant $0$ is an identity and is optimised away
during compilation, so the channel never fires; with a run-time argument the gate survives (Question C1). The channel itself does not depend on the
angle, the density-matrix target does apply noise, and channels are applied \emph{after} the gate. The lecture's experiments therefore pass idle angles
as arguments."""),

    dict(id="C3", kind="mcq",
         q=r"""Which set of operators defines a valid (completely positive, trace-preserving) quantum channel?""",
         opts=[r"$\{\sqrt{0.7}\,I,\ \sqrt{0.3}\,Y\}$", r"$\{I,\ \sqrt{0.1}\,X\}$",
               r"$\Big\{\begin{psmallmatrix}1&0\\0&0.6\end{psmallmatrix},\ \begin{psmallmatrix}0&0.6\\0&0\end{psmallmatrix}\Big\}$",
               r"$\{\sqrt{0.5}\,I,\ \sqrt{0.5}\,X,\ \sqrt{0.5}\,Z\}$"],
         sol=r"""Any map $\rho \mapsto \sum K\rho K^\dagger$ is completely positive; it is trace preserving iff $\sum K^\dagger K = I$.
$0.7I + 0.3Y^2 = I$ \checkmark. $I + 0.1X^2 = 1.1I$; $\operatorname{diag}(1, 0.36) + \operatorname{diag}(0, 0.36) = \operatorname{diag}(1, 0.72)$; $0.5 + 0.5 + 0.5 = 1.5$:
the other three create or destroy probability. The first is the bit-phase-flip channel with $p = 0.3$."""),

    dict(id="C4", kind="mcq",
         q=r"""The bit-flip channel with $p = 0.2$ acts on $\rho = \begin{pmatrix}0.9 & 0.3\\ 0.3 & 0.1\end{pmatrix}$. The output is""",
         opts=[r"$\begin{pmatrix}0.74 & 0.30\\ 0.30 & 0.26\end{pmatrix}$", r"$\begin{pmatrix}0.74 & 0.18\\ 0.18 & 0.26\end{pmatrix}$",
               r"$\begin{pmatrix}0.90 & 0.18\\ 0.18 & 0.10\end{pmatrix}$", r"$\begin{pmatrix}0.82 & 0.30\\ 0.30 & 0.18\end{pmatrix}$"],
         sol=r"""$X\rho X$ swaps both rows and columns: $\begin{psmallmatrix}0.1&0.3\\0.3&0.9\end{psmallmatrix}$. Then
$0.8\rho + 0.2X\rho X = \begin{psmallmatrix}0.72 + 0.02 & 0.24 + 0.06\\ 0.24 + 0.06 & 0.08 + 0.18\end{psmallmatrix} = \begin{psmallmatrix}0.74 & 0.30\\ 0.30 & 0.26\end{psmallmatrix}$.
The real off-diagonal ($r_x = 0.6$) is untouched; $r_z$ shrinks from $0.8$ to $0.8(1 - 2p) = 0.48$."""),

    dict(id="C5", kind="mcq",
         q=r"""Two depolarizing channels (CUDA-Q convention) with $p_1 = 0.03$ and $p_2 = 0.06$ act one after the other. They equal a single depolarizing channel with $p$ equal to""",
         opts=[r"$0.0876$", r"$0.0900$", r"$0.0882$", r"$0.1168$"],
         sol=r"""Each channel multiplies the Bloch vector by $1 - \frac{4p}{3}$: $0.96$ and $0.92$. Composed: $0.96\times0.92 = 0.8832 = 1 - \frac{4p}{3}$, so
$p = \frac34(1 - 0.8832) = 0.0876$. Probabilities do not simply add ($0.09$); composing ``no-error'' probabilities, $1 - 0.97\cdot0.94 = 0.0882$, ignores that two
Pauli errors can cancel."""),

    dict(id="C6", kind="mcq",
         q=r"""In \texttt{Noise\_Simulation.ipynb} the 15-qubit GHZ circuit has $G$ noisy gate locations (1 Hadamard and 14 CNOTs, each followed by a depolarizing channel).
The notebook's no-error probability $(1 - p)^G$ at $p = 0.02$ is""",
         opts=[r"$0.7386$", r"$0.7000$", r"$0.9800$", r"$0.5455$"],
         sol=r"""For depolarizing noise each gate location (one- or two-qubit) fails with probability $p$, so $G = 1 + 14 = 15$ and
$(0.98)^{15} = e^{15\ln0.98} = e^{-0.3030} = 0.7386$ (printed in the notebook). $1 - 15p = 0.70$ is the linear approximation; $0.98^{30} = 0.5455$
would count two failure chances per CNOT, which is the rule the notebook uses for bit-flip and phase-flip noise, not for depolarizing noise."""),

    dict(id="C7", kind="mcq",
         q=r"""For \texttt{NOISE\_MODEL = "bit\_flip"}, the notebook's CNOT channel is \texttt{Pauli2} with probabilities $p(1-p)$ for $IX$ and $XI$ and $p^2$ for $XX$. The
no-error probability of the 15-qubit GHZ circuit at $p = 0.01$ is""",
         opts=[r"$0.7472$", r"$0.8601$", r"$0.7397$", r"$0.9801$"],
         sol=r"""The CNOT channel equals independent bit flips on both qubits: no error with probability $(1-p)^2$. With one Hadamard and 14 CNOTs:
$(1-p)\cdot(1-p)^{2\cdot14} = (1-p)^{29} = 0.99^{29} = e^{-0.2915} = 0.7472$. This is the notebook's \texttt{no\_error\_probability} for non-depolarizing
kinds ($1 - (1-p)^2$ per two-qubit gate). $0.8601$ is the depolarizing value."""),

    dict(id="C8", kind="mcq",
         q=r"""The density-matrix method stores $4^N$ complex numbers of 16 bytes. The memory for $N = 13$ qubits is about""",
         opts=[r"$1.07$ GB", r"$131$ kB", r"$268$ MB", r"$4.29$ GB"],
         sol=r"""$16\times4^{13} = 16\times67\,108\,864 = 1.074\times10^9$ bytes $\approx 1.07$ GB. $131$ kB is a state vector ($16\cdot2^{13}$); $268$ MB is $N = 12$ and
$4.29$ GB is $N = 14$ (values quoted in the notebook). Every extra qubit multiplies the memory by 4."""),

    dict(id="C9", kind="mcq",
         q=r"""Which simulation can the stabilizer target \texttt{stim} perform \emph{correctly}?""",
         opts=[r"a GHZ circuit ($H$, CNOT) with depolarizing noise after every gate",
               r"the notebook's mirror circuit with random $R_Y$, $R_Z$ rotations and depolarizing noise",
               r"a GHZ circuit with amplitude-damping noise",
               r"a GHZ circuit followed by a $T$ gate, with bit-flip noise"],
         sol=r"""The stabilizer formalism (Gottesman--Knill) efficiently simulates Clifford circuits ($H$, $S$, CNOT, Paulis, measurement) with \emph{Pauli} noise, which
maps stabilizer states to probabilistic mixtures of stabilizer states. Arbitrary rotations and $T$ are non-Clifford; amplitude damping is not a Pauli channel.
The notebook raises an error in those cases instead of silently ignoring the noise."""),

    dict(id="C10", kind="mcq",
         q=r"""On the density-matrix path the notebook calls \texttt{as\_counts\_from\_rho(rho, n, shots)}. What does this function do?""",
         opts=[r"it takes the diagonal of the exact noisy $\rho$ as outcome probabilities, reorders the indices so that qubit 0 is written first, and draws \texttt{shots} counts from a multinomial distribution",
               r"it measures the density-matrix simulator \texttt{shots} times with \texttt{cudaq.sample}",
               r"it rounds every entry of $\rho$ to the nearest multiple of $1/\texttt{shots}$",
               r"it takes the diagonal of $\rho$ and returns it unchanged as counts, without sampling"],
         sol=r"""The function clips and normalises \texttt{np.real(np.diag(rho))}, applies \texttt{bit\_reverse\_order} (CUDA-Q stores qubit 0 as the least significant
bit), draws \texttt{counts\_rng.multinomial(shots, probs)} with a seeded NumPy generator, and also returns the exact distribution. Because measurement
probabilities are exactly the diagonal of $\rho$, this is statistically identical to noisy sampling --- and much faster than noisy \texttt{cudaq.sample} on the
density-matrix target in CUDA-Q 0.16 (about 115 s for 10 qubits, according to the README)."""),

    dict(id="C11", kind="mcq",
         q=r"""The ideal distribution of a 2-qubit circuit is $P(00) = P(11) = 0.5$. The noisy distribution is $Q(00) = Q(11) = 0.45$, $Q(01) = Q(10) = 0.05$. The Hellinger
fidelity $F_H = \big(\sum_x\sqrt{P_xQ_x}\big)^2$ and the total variation distance $\frac12\sum_x|P_x - Q_x|$ are""",
         opts=[r"$0.900$ and $0.100$", r"$0.949$ and $0.100$", r"$0.900$ and $0.200$", r"$0.810$ and $0.100$"],
         sol=r"""$\sum_x\sqrt{P_xQ_x} = 2\sqrt{0.5\times0.45} = 2(0.4743) = 0.9487$; squared: $F_H = 0.900$. Only the two ideal outcomes contribute.
TVD $= \frac12(0.05 + 0.05 + 0.05 + 0.05) = 0.100$. For a balanced two-peak ideal distribution $F_H$ equals the population $Q(00) + Q(11)$ --- which is why the
notebook's Hellinger fidelity and GHZ population columns coincide."""),

    dict(id="C12", kind="mcq",
         q=r"""A 2-qubit GHZ state is measured after Hadamards on both qubits ($X$ basis): counts \texttt{\{00: 450, 11: 400, 01: 80, 10: 70\}}. The parity
$\langle X\otimes X\rangle$ is""",
         opts=[r"$0.70$", r"$0.85$", r"$0.40$", r"$0.15$"],
         sol=r"""Even parity ($00$, $11$) counts $+1$, odd parity ($01$, $10$) counts $-1$: $\langle XX\rangle = \frac{(450 + 400) - (80 + 70)}{1000} = \frac{850 - 150}{1000} = 0.70$.
This is the notebook's \texttt{parity\_expectation}. The ideal GHZ value is 1; phase and bit errors both lower it."""),

    dict(id="C13", kind="mcq",
         q=r"""For $N = 20$ qubits, by what factor does the memory of exact density-matrix simulation exceed that of one trajectory (state vector) in the same precision?""",
         opts=[r"about $10^6$", r"$2$", r"$20$", r"about $10^{12}$"],
         sol=r"""$\frac{16\cdot4^{20}}{16\cdot2^{20}} = 2^{20} \approx 1.05\times10^6$. Trajectory (Monte-Carlo wave-function) simulation keeps one $2^N$ state and inserts random
Kraus jumps; averaging many trajectories reproduces $\rho$ statistically. It trades memory for repetitions --- CUDA-Q runs it on GPUs (\texttt{nvidia} target with a
noise model), which was not executed for these materials."""),

    dict(id="C14", kind="mcq",
         q=r"""In \texttt{build\_noise\_model} the line \texttt{model.add\_all\_qubit\_channel("x", two, num\_controls=1)} means""",
         opts=[r"after every CNOT (an \texttt{x} gate with one control), the two-qubit channel \texttt{two} acts on its control and target qubits",
               r"after every single-qubit \texttt{x} gate, the channel \texttt{two} acts on that qubit and its neighbour",
               r"the channel \texttt{two} is applied once, at the end of the circuit, to the first two qubits",
               r"the channel \texttt{two} replaces the CNOT gate"],
         sol=r"""\texttt{add\_all\_qubit\_channel(name, channel, num\_controls)} attaches a channel to every occurrence of the named gate with that number of controls, on all qubits it
acts on. Plain \texttt{x} gates (no control) get the single-qubit channel \texttt{one} from the loop over \texttt{("h", "x", "ry", "rz")}. The channel is applied
\emph{in addition to}, and after, the ideal gate."""),

    dict(id="C15", kind="match",
         q=r"""Match each CUDA-Q simulation target (List-I) with its property for noisy simulation (List-II).""",
         match=([r"\texttt{stim}", r"\texttt{density-matrix-cpu}", r"\texttt{nvidia} with a noise model", r"\texttt{qpp-cpu}"],
                [r"Monte-Carlo trajectories on a GPU, memory $\propto 2^N$, statistical",
                 r"noiseless state vector: not offered by the noise notebook (it ignores noise models)",
                 r"Clifford circuits with Pauli noise, polynomial cost, hundreds of qubits",
                 r"exact $\rho$, any Kraus channel, memory $16\cdot4^N$ bytes"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item \texttt{stim}: the default on a CPU for GHZ with Pauli noise (recorded run) (3);
\item \texttt{density-matrix-cpu}: exact but $4^N$ (4);
\item \texttt{nvidia} + noise model: trajectory method, selected automatically when a GPU is visible (1);
\item \texttt{qpp-cpu}: a noiseless state-vector simulator; the notebook refuses it for noise (2).
\end{itemize}
Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="C16", kind="match",
         q=r"""Match each figure of merit of the notebook (List-I) with what it is sensitive to or requires (List-II).""",
         match=([r"$Z$-basis GHZ population", r"$X$-basis parity $\langle X^{\otimes N}\rangle$", r"state fidelity $\braket{\psi|\rho|\psi}$", r"purity $\operatorname{Tr}\rho^2$"],
                [r"needs the full density matrix; also the off-diagonal (coherence) elements", r"bit flips (errors that change computational-basis outcomes)",
                 r"phase errors, which are invisible in the $Z$ basis", r"mixedness of the state, whatever the target"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4",
               r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$2, Q$\to$1, R$\to$3, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item population: only diagonal elements; drops with bit flips (2);
\item $X$ parity: $Z$ errors flip its sign (3);
\item state fidelity: for GHZ $F = \frac12(\rho_{0,0} + \rho_{d,d} + 2\operatorname{Re}\rho_{0,d})$ --- needs the coherence $\rho_{0\cdots0,1\cdots1}$, available only on the density-matrix path (1);
\item purity: 1 for pure states, $2^{-N}$ for the maximally mixed state; independent of the target (4).
\end{itemize}
Hence P$\to$2, Q$\to$3, R$\to$1, S$\to$4."""),

    dict(id="C17", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} For depolarizing noise, the GHZ state fidelity $\braket{\mathrm{GHZ}|\rho|\mathrm{GHZ}}$ is \emph{exactly} equal to the
no-error probability $(1 - p)^G$.\\[1mm]
\textbf{Reason (R):} For Pauli noise, a run without any error produces the ideal state, and no error occurs with probability $(1 - p)^G$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true}: a Pauli channel is a mixture ``ideal with probability $1 - p$, Pauli error otherwise''; the error-free branch alone contributes
$(1-p)^G$ to the fidelity, which makes $(1-p)^G$ a \emph{lower bound}. \textbf{A is false}: some error branches also leave the GHZ state unchanged ---
e.g.\ $Z\otimes Z$ after a CNOT ($ZZ\ket{\mathrm{GHZ}} = \ket{\mathrm{GHZ}}$) --- so the fidelity is strictly larger. The lecture's 6-qubit density-matrix run gives
$F = 0.9492$ at $p = 0.01$, while $(0.99)^6 = 0.9415$. (For non-Pauli noise such as amplitude damping the notebook does not claim the bound at all.) Answer: (D)."""),

    dict(id="C18", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} In CUDA-Q 0.16 the density-matrix path of the notebook calls \texttt{set\_noise} before \texttt{get\_state},
because \texttt{get\_state} has no \texttt{noise\_model} argument.\\[1mm]
\textbf{Statement II:} The density-matrix method needs $16\cdot4^N$ bytes, so $N = 14$ needs about $4.3$ GB.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true}: the notebook says so in a comment and uses \texttt{set\_noise} / \texttt{unset\_noise} in a \texttt{try/finally} block (with a deprecation
warning filtered). \textbf{II is true}: $16\times4^{14} = 4.29\times10^9$ bytes. Answer: (A)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The probability mass found \emph{outside} the two ideal GHZ outcomes at $p = 0.01$ and at $p = 0.1$ is""",
         opts=[r"$0.1064$ and $0.6768$", r"$0.1399$ and $0.7941$", r"$0.3090$ and $0.9840$", r"$0.8936$ and $0.3232$"],
         sol=r"""It is $1 - \text{population}$: $1 - 0.8936 = 0.1064$ and $1 - 0.3232 = 0.6768$ (this also equals the total variation distance to the ideal distribution
for a balanced population). $1 - (1-p)^G$ ($0.1399$, $0.7941$) is the probability that \emph{some} error occurred --- larger, because many errors (e.g.\ $Z$ errors) do
not move probability out of the two peaks. $1 - $ parity measures phase damage."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""Why is the population at $p = 0.01$ ($0.8936$) \emph{larger} than $(1 - p)^G = 0.8601$, while the parity ($0.6910$) is smaller?""",
         opts=[r"$(1-p)^G$ counts every error, but $Z$-type errors leave the $Z$-basis population intact; the parity is also sensitive to those phase errors (and to Hadamard noise in the $X$-basis circuit)",
               r"the stabilizer simulator under-reports errors because it cannot simulate depolarizing noise",
               r"$(1-p)^G$ is an upper bound on the population, so the population must lie below it --- the table contains a mistake",
               r"10\,000 shots are too few: with more shots the population would fall to exactly $0.8601$"],
         sol=r"""$(1-p)^G$ is the weight of the error-free branch, a \emph{lower} bound for Pauli noise. Among the depolarizing errors, $Z$-type errors (e.g.\ $Z$ on one qubit, or $ZZ$ after a CNOT) keep the outcome in $\{0\cdots0, 1\cdots1\}$ --- so does an $X$ error on qubit 0 right after the Hadamard, which the CNOT chain copies to every qubit ---
so the population exceeds the bound. The parity measurement
needs extra Hadamards (also noisy), and $Z$-type errors flip its sign, so it is the more demanding test. The statistical error at 10\,000 shots is only
$\sqrt{0.89\cdot0.11/10^4} \approx 0.003$; \texttt{stim} simulates Pauli channels exactly."""),
]
