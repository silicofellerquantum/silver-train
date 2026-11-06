"""Section B - Quantum Noise Fundamentals (decoherence, bit flips, phase flips, bit-phase flips, depolarizing noise,
amplitude damping, phase damping, T1 and T2, measurement errors, coherent vs stochastic errors).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""A superconducting qubit has $T_1 = 80\,\mu$s and $T_2 = 60\,\mu$s. It is prepared in $\ket+$ and then idles for $t = 20\,\mu$s.
Use the standard model: populations relax towards $\ket0$ as $\rho_{11}(t) = \rho_{11}(0)e^{-t/T_1}$, and the coherence decays as
$|\rho_{01}(t)| = |\rho_{01}(0)|e^{-t/T_2}$, with $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\varphi}$ ($T_\varphi$ = pure-dephasing time).""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""Which single-qubit state is left \emph{unchanged} by the bit-flip channel $\rho \mapsto (1 - p)\rho + pX\rho X$ for every $p$?""",
         opts=[r"$\ket+$", r"$\ket0$", r"$\frac{1}{\sqrt2}(\ket0 + i\ket1)$", r"$\ket1$"],
         sol=r"""A state is invariant iff $X\rho X = \rho$, i.e.\ its Bloch vector satisfies $(r_x, -r_y, -r_z) = (r_x, r_y, r_z)$: $r_y = r_z = 0$.
$\ket+$ has $\vec r = (1, 0, 0)$ (indeed $X\ket+ = \ket+$). $\ket0$, $\ket1$ have $r_z = \pm1$ and become mixed; $\frac{1}{\sqrt2}(\ket0 + i\ket1)$ has
$r_y = 1$, which shrinks to $1 - 2p$. The eigenstates of the error operator are immune to that error."""),

    dict(id="B2", kind="mcq",
         q=r"""A phase-flip channel with $p = 0.1$ acts on $\ket+$. The state is then measured in the $X$ basis. The probability of the
outcome ``$+$'' is""",
         opts=[r"$0.9$", r"$0.8$", r"$1.0$", r"$0.95$"],
         sol=r"""$\rho' = 0.9\ket+\!\bra+ + 0.1Z\ket+\!\bra+Z = 0.9\ket+\!\bra+ + 0.1\ket-\!\bra-$ (since $Z\ket+ = \ket-$). So $P(+) = 0.9$ and
$\langle X\rangle = 0.9 - 0.1 = 1 - 2p = 0.8$. In the $Z$ basis nothing changes ($P(0) = P(1) = \frac12$): phase errors are invisible there."""),

    dict(id="B3", kind="mcq",
         q=r"""A depolarizing channel $\rho \mapsto (1 - p)\rho + \frac p3(X\rho X + Y\rho Y + Z\rho Z)$ with $p = 0.03$ acts on $\ket0$. The probability of
then measuring $1$ is""",
         opts=[r"$0.02$", r"$0.03$", r"$0.01$", r"$0.04$"],
         sol=r"""$X$ and $Y$ flip $\ket0$ to $\ket1$ (up to a phase), $Z$ does not. So $P(1) = \frac p3 + \frac p3 = \frac{2p}{3} = 0.02$.
Equivalently the Bloch vector shrinks by $1 - \frac{4p}{3} = 0.96$: $r_z = 0.96$, $P(1) = \frac{1 - r_z}{2} = 0.02$.
(This is CUDA-Q's \texttt{DepolarizationChannel(p)} convention; the same channel is $\rho \mapsto (1 - \lambda)\rho + \lambda\frac I2$ with $\lambda = \frac{4p}{3}$.)"""),

    dict(id="B4", kind="mcq",
         q=r"""Amplitude damping with $\gamma = 0.36$ acts on $\ket+$. The resulting density matrix is""",
         opts=[r"$\begin{pmatrix}0.68 & 0.40\\ 0.40 & 0.32\end{pmatrix}$", r"$\begin{pmatrix}0.68 & 0.32\\ 0.32 & 0.32\end{pmatrix}$",
               r"$\begin{pmatrix}0.50 & 0.40\\ 0.40 & 0.50\end{pmatrix}$", r"$\begin{pmatrix}0.68 & 0.50\\ 0.50 & 0.32\end{pmatrix}$"],
         sol=r"""With $K_0 = \operatorname{diag}(1, \sqrt{1-\gamma})$, $K_1 = \sqrt\gamma\ket0\!\bra1$:
$\rho_{11}' = (1 - \gamma)\rho_{11}$, $\rho_{00}' = \rho_{00} + \gamma\rho_{11}$, $\rho_{01}' = \sqrt{1 - \gamma}\,\rho_{01}$.
For $\ket+$ ($\rho_{ij} = \frac12$): $\rho_{11}' = 0.64\cdot0.5 = 0.32$, $\rho_{00}' = 0.5 + 0.18 = 0.68$, $\rho_{01}' = 0.8\cdot0.5 = 0.40$.
Populations move towards $\ket0$ and the coherence shrinks by $\sqrt{1 - \gamma}$, not by $1 - \gamma$."""),

    dict(id="B5", kind="mcq",
         q=r"""Phase damping with parameter $\lambda$ multiplies the off-diagonal elements by $\sqrt{1 - \lambda}$; a phase-flip channel multiplies them by $1 - 2p$.
The phase-flip probability $p$ that has the same effect as phase damping with $\lambda = 0.36$ is""",
         opts=[r"$0.10$", r"$0.18$", r"$0.36$", r"$0.20$"],
         sol=r"""Both channels leave the populations unchanged, so they are equal iff they scale the coherence by the same factor:
$1 - 2p = \sqrt{1 - 0.36} = 0.8 \Rightarrow p = 0.1$. Phase damping and the phase-flip channel are the same channel in different Kraus
representations (Kraus operators are not unique)."""),

    dict(id="B6", kind="mcq",
         q=r"""A qubit with $T_1 = 50\,\mu$s starts in $\ket1$. After what time is $P(1) = \frac12$?""",
         opts=[r"$34.7\,\mu$s", r"$25\,\mu$s", r"$50\,\mu$s", r"$69.3\,\mu$s"],
         sol=r"""$P(1) = e^{-t/T_1} = \frac12 \Rightarrow t = T_1\ln2 = 50\times0.693 = 34.7\,\mu$s. At $t = T_1$ the population is $e^{-1} = 0.368$, not $\frac12$;
$69.3\,\mu$s would give $\frac14$."""),

    dict(id="B7", kind="mcq",
         q=r"""A qubit has $T_1 = 100\,\mu$s and pure-dephasing time $T_\varphi = 200\,\mu$s. Using $\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\varphi}$, $T_2$ equals""",
         opts=[r"$100\,\mu$s", r"$66.7\,\mu$s", r"$150\,\mu$s", r"$200\,\mu$s"],
         sol=r"""$\frac{1}{T_2} = \frac{1}{200} + \frac{1}{200} = \frac{1}{100}\ \mu\text{s}^{-1}$, so $T_2 = 100\,\mu$s. Relaxation contributes only $\frac{1}{2T_1}$ because amplitude
damping multiplies the coherence by $\sqrt{1 - \gamma} \approx e^{-t/2T_1}$. Using $\frac1{T_1}$ instead would give $66.7\,\mu$s."""),

    dict(id="B8", kind="mcq",
         q=r"""Which pair $(T_1, T_2)$ is \emph{physically impossible} in the standard model?""",
         opts=[r"$T_1 = 50\,\mu$s, $T_2 = 120\,\mu$s", r"$T_1 = 50\,\mu$s, $T_2 = 100\,\mu$s",
               r"$T_1 = 100\,\mu$s, $T_2 = 30\,\mu$s", r"$T_1 = 80\,\mu$s, $T_2 = 80\,\mu$s"],
         sol=r"""Since $\frac1{T_\varphi} = \frac1{T_2} - \frac1{2T_1} \ge 0$, every qubit satisfies $T_2 \le 2T_1$. $120 > 2\times50 = 100$ violates it. $T_2 = 2T_1$
(no pure dephasing) is the allowed limit; $T_2 < T_1$ is common in practice."""),

    dict(id="B9", kind="mcq",
         q=r"""Gates take $50$ ns and the qubit has $T_2 = 100\,\mu$s. Requiring the coherence factor $e^{-t/T_2}$ to stay at least $0.9$, the largest number of
sequential gates is about""",
         opts=[r"$210$", r"$2000$", r"$21$", r"$105$"],
         sol=r"""$e^{-nt_g/T_2} \ge 0.9 \Leftrightarrow n \le \frac{T_2\ln(1/0.9)}{t_g} = \frac{100\,\mu\text{s}\times0.1054}{0.05\,\mu\text{s}} = 210.7$, so about $210$ gates.
$T_2/t_g = 2000$ is the number of gates in one $T_2$, when the coherence has already fallen to $e^{-1} = 0.37$."""),

    dict(id="B10", kind="mcq",
         q=r"""A detector reads $1$ when the qubit is in $\ket0$ with probability $0.02$, and reads $0$ when it is in $\ket1$ with probability $0.05$. For a perfectly
prepared $\ket+$, the probability of reading $1$ is""",
         opts=[r"$0.485$", r"$0.500$", r"$0.475$", r"$0.515$"],
         sol=r"""Law of total probability: $P(\text{read }1) = P(0)P(1|0) + P(1)P(1|1) = 0.5(0.02) + 0.5(0.95) = 0.01 + 0.475 = 0.485$.
Readout errors are classical (they act on the recorded bit) and are usually \emph{asymmetric}: $\ket1$ can relax during the measurement."""),

    dict(id="B11", kind="mcq",
         q=r"""A perfect 3-qubit GHZ state is measured with the readout errors of the previous question ($P(1|0) = 0.02$, $P(0|1) = 0.05$, independent per qubit).
The probability of observing $000$ or $111$ is about""",
         opts=[r"$0.899$", r"$0.941$", r"$0.857$", r"$0.950$"],
         sol=r"""$P(000\text{ read correctly}) = 0.98^3 = 0.9412$, $P(111\text{ read correctly}) = 0.95^3 = 0.8574$. With probability $\frac12$ each:
$\frac12(0.9412 + 0.8574) = 0.8993$; the cross terms ($000 \to 111$: $0.02^3$, $111 \to 000$: $0.05^3$) add only $7\times10^{-5}$. Readout
errors grow with the number of measured qubits."""),

    dict(id="B12", kind="mcq",
         q=r"""Which of the following is a \emph{coherent} (unitary) error rather than decoherence or a measurement error?""",
         opts=[r"a systematic over-rotation of every $X$ gate caused by a miscalibrated pulse amplitude",
               r"energy relaxation of $\ket1$ to $\ket0$ (a $T_1$ process)",
               r"a random phase that differs from shot to shot and averages to dephasing",
               r"the discriminator labelling a $\ket1$ outcome as $0$"],
         sol=r"""A coherent error is a deterministic, unitary deviation $U_{\text{actual}} = U\,e^{-i\varepsilon H}$, identical in every run; it keeps the state pure
and its effects can add up in \emph{amplitude} (quadratically in probability). $T_1$ relaxation is non-unitary decoherence; a phase that is random from run
to run produces a mixed state (stochastic dephasing); a misassigned outcome is a classical readout error."""),

    dict(id="B13", kind="mcq",
         q=r"""After a CNOT, CUDA-Q's \texttt{Depolarization2(p)} applies each of the 15 non-identity two-qubit Paulis with probability $p/15$. For $p = 0.015$, the
probability that the outcome of a $Z$-basis measurement of the control qubit is flipped by this error is""",
         opts=[r"$0.008$", r"$0.015$", r"$0.010$", r"$0.0075$"],
         sol=r"""The $Z$ outcome of the control flips iff the Pauli on the control is $X$ or $Y$. Of the 15 non-identity pairs $P_c\otimes P_t$, those with
$P_c \in \{X, Y\}$ are $2\times4 = 8$. Probability $= 8\cdot\frac{p}{15} = \frac{8(0.015)}{15} = 0.008$."""),

    dict(id="B14", kind="mcq",
         q=r"""A 15-qubit GHZ state suffers only phase-flip noise. Which measurement reveals the errors?""",
         opts=[r"the parity $\langle X^{\otimes15}\rangle$ measured after a Hadamard on every qubit",
               r"the $Z$-basis population $P(0\cdots0) + P(1\cdots1)$",
               r"the number of distinct outcomes in the $Z$ basis",
               r"the probability $P(0\cdots0)$ alone"],
         sol=r"""$Z$ errors commute with $Z$-basis measurements, so all $Z$-basis statistics are unchanged: the population stays 1 and only the outcomes $0\cdots0$,
$1\cdots1$ occur. A single $Z$ error turns $\ket{0\cdots0} + \ket{1\cdots1}$ into $\ket{0\cdots0} - \ket{1\cdots1}$, which flips the sign of $\langle X^{\otimes N}\rangle$.
This is why \texttt{Noise\_Simulation.ipynb} also measures in the $X$ basis (\texttt{MEASURE\_X\_BASIS = True})."""),

    dict(id="B15", kind="match",
         q=r"""Match each channel (List-I) with a set of Kraus operators (List-II).""",
         match=([r"bit flip", r"phase flip", r"amplitude damping", r"phase damping"],
                [r"$\operatorname{diag}(1, \sqrt{1-\lambda})$, $\operatorname{diag}(0, \sqrt\lambda)$",
                 r"$\sqrt{1-p}\,I$, $\sqrt p\,Z$",
                 r"$\begin{psmallmatrix}1&0\\0&\sqrt{1-\gamma}\end{psmallmatrix}$, $\begin{psmallmatrix}0&\sqrt\gamma\\0&0\end{psmallmatrix}$",
                 r"$\sqrt{1-p}\,I$, $\sqrt p\,X$"]),
         opts=[r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1",
               r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item bit flip: apply $X$ with probability $p$ (4);
\item phase flip: apply $Z$ with probability $p$ (2);
\item amplitude damping: the ``jump'' $\sqrt\gamma\ket0\!\bra1$ and the no-jump $K_0$ (3);
\item phase damping: diagonal Kraus operators, populations untouched, coherence $\times\sqrt{1-\lambda}$ (1).
\end{itemize}
Each set satisfies $\sum K^\dagger K = I$. Hence P$\to$4, Q$\to$2, R$\to$3, S$\to$1."""),

    dict(id="B16", kind="match",
         q=r"""Match each channel (List-I) with its action on the Bloch vector $(r_x, r_y, r_z)$ (List-II).""",
         match=([r"depolarizing (CUDA-Q convention, $p$)", r"amplitude damping ($\gamma$)", r"phase damping ($\lambda$)", r"bit flip ($p$)"],
                [r"$r_x, r_y$ scaled, $r_z$ unchanged",
                 r"all three components multiplied by the same factor $1 - \frac{4p}{3}$",
                 r"$r_x$ unchanged, $r_y$ and $r_z$ multiplied by $1 - 2p$",
                 r"$r_x, r_y$ multiplied by $\sqrt{1-\gamma}$, $r_z \mapsto \gamma + (1-\gamma)r_z$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item depolarizing: an isotropic shrink of the whole ball (2);
\item amplitude damping: shrinks and pushes the ball towards the north pole $\ket0$ --- the only non-unital one here (4);
\item phase damping: shrinks towards the $z$-axis (1);
\item bit flip: shrinks towards the $x$-axis (3).
\end{itemize}
Hence P$\to$2, Q$\to$4, R$\to$1, S$\to$3."""),

    dict(id="B17", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} Amplitude damping is not unital: it does not map the maximally mixed state $I/2$ to itself.\\[1mm]
\textbf{Reason (R):} Amplitude damping multiplies the off-diagonal elements of $\rho$ by $\sqrt{1 - \gamma}$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true}: $I/2 \mapsto \operatorname{diag}(\frac{1+\gamma}{2}, \frac{1-\gamma}{2}) \neq I/2$ --- energy flows to $\ket0$.
\textbf{R is true} as well, but it does not explain A: phase damping shrinks the off-diagonals in the same way and \emph{is} unital. Non-unitality
comes from the population transfer $\rho_{00} \mapsto \rho_{00} + \gamma\rho_{11}$ (the jump operator $\ket0\!\bra1$). Answer: (B)."""),

    dict(id="B18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} The phase-flip channel with $p = \frac12$ maps every single-qubit state to a state with zero off-diagonal elements.\\[1mm]
\textbf{Statement II:} The phase-flip channel with $p = 1$ destroys all quantum information.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true}: the coherence is multiplied by $1 - 2p = 0$ (complete dephasing). \textbf{II is false}: with $p = 1$ the channel is
$\rho \mapsto Z\rho Z$, a \emph{unitary} that can be undone by applying $Z$ again. Information is lost only when the error is random ($0 < p < 1$),
most for $p = \frac12$. Answer: (C)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""After the idle period, $\rho_{11}$ and $|\rho_{01}|$ are approximately""",
         opts=[r"$0.389$ and $0.358$", r"$0.389$ and $0.389$", r"$0.500$ and $0.358$", r"$0.389$ and $0.316$"],
         sol=r"""$\rho_{11}(0) = |\rho_{01}(0)| = \frac12$ for $\ket+$. $\rho_{11} = \frac12e^{-20/80} = \frac12e^{-0.25} = 0.389$.
$|\rho_{01}| = \frac12e^{-20/60} = \frac12e^{-1/3} = 0.358$. The $T_2$ decay already \emph{includes} the relaxation contribution; multiplying by
$e^{-t/2T_1}$ again would double-count it ($0.316$)."""),

    dict(id="B20", kind="para", passage="B",
         q=r"""The fidelity $\braket{+|\rho|+}$ after the idle period, and the pure-dephasing time $T_\varphi$, are""",
         opts=[r"$0.858$ and $96\,\mu$s", r"$0.716$ and $96\,\mu$s", r"$0.858$ and $43.6\,\mu$s", r"$0.889$ and $96\,\mu$s"],
         sol=r"""$\braket{+|\rho|+} = \frac12(\rho_{00} + \rho_{11}) + \operatorname{Re}\rho_{01} = \frac12 + 0.358 = 0.858$ (populations sum to 1 whatever their split).
$\frac{1}{T_\varphi} = \frac{1}{T_2} - \frac{1}{2T_1} = \frac{1}{60} - \frac{1}{160} = 0.01667 - 0.00625 = 0.01042\ \mu\text{s}^{-1}$, so $T_\varphi = 96\,\mu$s.
Adding instead of subtracting gives $43.6\,\mu$s."""),
]
