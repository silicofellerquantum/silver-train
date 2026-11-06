"""Section C - HHL fundamentals (registers, encoding, phase estimation with exp(iAt), eigenvalue inversion, the
constant C, controlled rotations, uncomputation, post-selection, norm recovery, the notebook's circuit and code).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""HHL in the notebook's convention is applied to $A = \operatorname{diag}(1, 2)$ and $b = \tfrac{1}{\sqrt2}(1, 1)$ with $n_c = 3$ clock qubits and $t = \pi/4$, so
$U = e^{iAt}$, the clock value of an eigenvalue $\lambda$ is $k = \lambda t\,2^{n_c}/(2\pi) = \lambda$, and $C = 2\pi/(2^{n_c}t) = 1$. The eigenvalue inversion applies
$R_Y(2\arcsin(C/\lambda_k))$ to the ancilla for each clock value; the system register has one qubit.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""After phase estimation and the controlled rotation (before uncomputation), the ideal HHL state is (system $\otimes$ clock $\otimes$ ancilla)""",
         opts=[r"$\sum_j\beta_j\ket{u_j}\ket{k_j}\Big(\sqrt{1 - \tfrac{C^2}{\lambda_j^2}}\,\ket0 + \tfrac{C}{\lambda_j}\ket1\Big)$",
               r"$\sum_j\beta_j\ket{u_j}\ket{k_j}\Big(\sqrt{1 - C^2\lambda_j^2}\,\ket0 + C\lambda_j\ket1\Big)$",
               r"$\sum_j\tfrac{\beta_j}{\lambda_j}\ket{u_j}\ket0\ket1$",
               r"$\sum_j\beta_j\ket{u_j}\ket{0}\Big(\sqrt{1 - \tfrac{C^2}{\lambda_j^2}}\,\ket0 + \tfrac{C}{\lambda_j}\ket1\Big)$"],
         sol=r"""Write $\ket b = \sum_j\beta_j\ket{u_j}$. Phase estimation entangles each eigencomponent with its clock value: $\sum_j\beta_j\ket{u_j}\ket{k_j}\ket0$. The controlled rotation reads $k_j$
and rotates the ancilla by $R_Y(2\arcsin(C/\lambda_j))$: $\ket0 \to \sqrt{1 - C^2/\lambda_j^2}\ket0 + (C/\lambda_j)\ket1$. Uncomputation then resets the clock to $\ket0$, and the ancilla-$\ket1$
branch is $C\sum_j(\beta_j/\lambda_j)\ket{u_j} = C\,A^{-1}\ket b$. Option 2 multiplies by $\lambda$ (that would apply $A$, not $A^{-1}$); option 3 is the final
\emph{unnormalised} target; option 4 has already uncomputed the clock."""),

    dict(id="C2", kind="mcq",
         q=r"""Using the notebook's layout (clock + system + one ancilla), an $8\times8$ Hermitian system with $n_c = 5$ clock qubits needs how many qubits?""",
         opts=[r"$9$", r"$14$", r"$8$", r"$13$"],
         sol=r"""System: $n_b = \log_28 = 3$ qubits (amplitude encoding of $b$). Total $n_c + n_b + 1 = 5 + 3 + 1 = 9$ --- the notebook prints ``Qubit requirement: n\_b + n\_c + 1''. The
default ($4\times4$, $n_c = 4$) needs $4 + 2 + 1 = 7$. $8 + 5 + 1 = 14$ forgets that the system register is logarithmic in the dimension."""),

    dict(id="C3", kind="mcq",
         q=r"""The notebook sets $C = 2\pi/(2^{n_c}t)$, the smallest non-zero eigenvalue magnitude the clock can represent. Why must $C \le |\lambda_j|$ for every eigenvalue?""",
         opts=[r"The rotation amplitude $C/\lambda_j$ must satisfy $|C/\lambda_j| \le 1$ so that $\arcsin(C/\lambda_j)$ exists",
               r"Otherwise $e^{iAt}$ would not be unitary",
               r"Otherwise phase estimation needs more counting qubits than system qubits",
               r"$C$ must be the largest eigenvalue so that the success probability is 1"],
         sol=r"""The ancilla gets amplitude $C/\lambda_j$ on $\ket1$; amplitudes must have modulus $\le 1$, so $C \le \min_j|\lambda_j|$. The success probability $\sum|\beta_j|^2C^2/\lambda_j^2$ is largest when $C$ is
as large as allowed, $C = |\lambda|_{\min}$ --- exactly what the \texttt{"auto"} time achieves (default: $C = 1 = \lambda_{\min}$). Unitarity of $e^{iAt}$ needs only Hermitian $A$."""),

    dict(id="C4", kind="mcq",
         q=r"""With $C = 1$, the eigenvalue-inversion rotation for the clock value of $\lambda = 2$ is $R_Y(\theta)$ with""",
         opts=[r"$\theta = 2\arcsin\tfrac12 = \tfrac\pi3$", r"$\theta = \arcsin\tfrac12 = \tfrac\pi6$", r"$\theta = 2\arcsin2$", r"$\theta = \tfrac\pi2$"],
         sol=r"""$\theta = 2\arcsin(C/\lambda) = 2\arcsin\tfrac12 = 2\cdot\tfrac\pi6 = \tfrac\pi3$; then $R_Y(\theta)\ket0 = \cos\tfrac\pi6\ket0 + \sin\tfrac\pi6\ket1 = \tfrac{\sqrt3}{2}\ket0 + \tfrac12\ket1$, amplitude $\tfrac12 = 1/\lambda$ on $\ket1$.
The factor 2 comes from the half-angle convention $R_Y(\theta) = e^{-i\theta Y/2}$; $\arcsin 2$ does not exist (why $C \le |\lambda|$)."""),

    dict(id="C5", kind="mcq",
         q=r"""For the default system ($\beta_j = \tfrac12$ for the eigenvalues $1, 2, 3, 4$; $C = 1$), the ideal success probability $P_{\text{success}} = \sum_j|\beta_j|^2C^2/\lambda_j^2$ is""",
         opts=[r"$0.3559$", r"$0.5208$", r"$0.25$", r"$1.4236$"],
         sol=r"""$P = \tfrac14\left(1 + \tfrac14 + \tfrac19 + \tfrac1{16}\right) = \tfrac14\cdot1.423611 = 0.355903$ --- the value printed by the notebook (state $0.355903$, shots $0.3543$). Equivalently
$P = C^2\|A^{-1}b\|^2/\|b\|^2 = \|x\|^2 = 0.5966^2$. $1.4236$ forgets the weights $|\beta_j|^2 = \tfrac14$; $0.5208$ is $x_0$."""),

    dict(id="C6", kind="match",
         q=r"""Match each stage of the notebook's HHL circuit (List-I) with what it does to the registers (List-II).""",
         match=([r"amplitude initialisation", r"phase estimation", r"eigenvalue inversion", r"uncomputation"],
                [r"$\ket{u_j}\ket{k_j}\ket0 \to \ket{u_j}\ket{k_j}\big(\sqrt{1 - C^2/\lambda_j^2}\ket0 + \tfrac{C}{\lambda_j}\ket1\big)$",
                 r"returns the clock register to $\ket{0\dots0}$ so that it is no longer entangled with the system",
                 r"prepares $\ket{0\dots0}_{\text{clock}}\ket b\ket0_{\text{anc}}$",
                 r"$\ket b\ket0 \to \sum_j\beta_j\ket{u_j}\ket{k_j}$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: \texttt{kernel.qalloc(...)} loads $\ket0\ket b\ket0$ in one step (3).
\item Q: Hadamards, controlled $U^{2^j}$, inverse QFT write the eigenvalue into the clock (4).
\item R: the multi-controlled $R_Y$ rotations (1).
\item S: QFT, inverse controlled powers, Hadamards (2).
\end{itemize}"""),

    dict(id="C7", kind="mcq",
         q=r"""With \texttt{EVOLUTION\_TIME = "auto"} the notebook chooses $t = 2\pi/(2^{n_c}|\lambda|_{\min})$ whenever $\kappa \le 2^{n_c - 1} - 1$. For $n_c = 4$ and eigenvalues $0.5, 1, 2, 3$, it picks""",
         opts=[r"$t = \pi/4$ and $C = 0.5$", r"$t = \pi/8$ and $C = 1$", r"$t = \pi/4$ and $C = 1$", r"$t = 2\pi/3$ and $C = 3$"],
         sol=r"""$\kappa = 3/0.5 = 6 \le 2^3 - 1 = 7$, so $t = 2\pi/(16\cdot0.5) = \pi/4$ and $C = 2\pi/(16t) = 0.5 = |\lambda|_{\min}$. The clock values $k = \lambda t\cdot16/(2\pi) = 2\lambda$ are $1, 2, 4, 6$ --- all exact
and below the largest positive value $7$. ($t = \pi/8$, $C = 1$ is right for the default with $\lambda_{\min} = 1$.)"""),

    dict(id="C8", kind="mcq",
         q=r"""Why does HHL uncompute phase estimation after the eigenvalue inversion instead of simply measuring the ancilla?""",
         opts=[r"Without it the clock stays entangled with the system, so the post-selected system register would be a mixture instead of $\ket x$ (unless the clock is also post-selected on a fixed value)",
               r"Uncomputation doubles the success probability",
               r"Uncomputation is needed to make $e^{iAt}$ unitary",
               r"It converts the amplitudes $C/\lambda_j$ into probabilities"],
         sol=r"""The ancilla-$\ket1$ branch before uncomputation is $C\sum_j(\beta_j/\lambda_j)\ket{u_j}\ket{k_j}$: tracing out the clock leaves the incoherent mixture $\sum_j|\beta_j/\lambda_j|^2\ket{u_j}\bra{u_j}$, not the
superposition $\ket x$. Reversing phase estimation maps $\ket{k_j} \to \ket0$ for every $j$, so the branch becomes $C\ket x\otimes\ket0$. The notebook post-selects ancilla $= 1$ \emph{and}
clock $= 0$. The success probability is unchanged by uncomputation."""),

    dict(id="C9", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} From the simulated state vector, the notebook recovers the full solution $x$ including its norm.\\[1mm]
\textbf{Reason (R):} In the post-selected branch (clock $= 0$, ancilla $= 1$) the system amplitudes are $\tfrac{C}{\|b\|}\,x$, so $x = \tfrac{\|b\|}{C}\times(\text{amplitudes})$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} the branch is $C\,A^{-1}\ket b = \tfrac{C}{\|b\|}A^{-1}b$ because $\ket b = b/\|b\|$.
\textbf{A follows:} the code computes \texttt{x\_quantum\_full = amplitudes * b\_norm / C\_constant} and gets $x = (0.520833, 0.145833, 0.229167, 0.104167)$ with relative error $1.7\times10^{-15}$.
(On hardware only $|x_i|^2/\|x\|^2$ is sampled.) Answer: (A)."""),

    dict(id="C10", kind="mcq",
         q=r"""The notebook's eigenvalue inversion applies one multi-controlled rotation for every non-zero clock value. With $n_c$ clock qubits this is""",
         opts=[r"$2^{n_c} - 1$ rotations ($15$ for $n_c = 4$)", r"$n_c$ rotations", r"$2^{n_c}$ rotations, including $k = 0$", r"one rotation per eigenvalue of $A$"],
         sol=r"""Clock values $k = 1, \dots, 2^{n_c} - 1$ each get their own $R_Y(2\arcsin(C/\lambda_k))$; $k = 0$ would mean $\lambda = 0$ and is skipped. For $n_c = 4$: $15$, matching \texttt{ccccry: 15}.
The rotation count grows exponentially with the clock size --- a lookup table, not an arithmetic $1/\lambda$ circuit; the matrix itself is not consulted."""),

    dict(id="C11", kind="mcq",
         q=r"""With $n_c = 4$ and $t = \pi/8$, the notebook assigns to the clock value $k = 12$ the eigenvalue""",
         opts=[r"$-4$", r"$12$", r"$4$", r"$-12$"],
         sol=r"""$12 \ge 2^{n_c-1} = 8$, so the signed value is $12 - 16 = -4$, and $\lambda_k = 2\pi\cdot(-4)/(16\cdot\pi/8) = -4$. The rotation angle is $2\arcsin(C/\lambda) = 2\arcsin(-\tfrac14) < 0$: the ancilla
amplitude $-\tfrac14$ carries the sign of $1/\lambda$. Without two's complement, negative eigenvalues would be misread as large positive ones."""),

    dict(id="C12", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} If some $\lambda_jt\,2^{n_c}/(2\pi)$ is not an integer, phase estimation spreads that component over several clock values and the fidelity of $\ket x$ drops below 1.\\[1mm]
\textbf{Statement II:} The notebook's \texttt{"auto"} evolution time guarantees exact clock values for every matrix.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} a non-integer clock value gives the leaky QPE distribution, and each leaked clock value applies the wrong $1/\lambda$ (with only 3 clock qubits the default reaches fidelity $0.953$).
\textbf{II is false:} \texttt{"auto"} only maps $|\lambda|_{\min}$ to clock value 1; other eigenvalues are exact only if they are integer multiples of $|\lambda|_{\min}$ (the notebook prints ``(rounded -> error)''
otherwise). Answer: (C)."""),

    dict(id="C13", kind="mcq",
         q=r"""For the notebook's $2\times2$ example ($\lambda = \tfrac23, \tfrac43$) with $n_c = 4$ and \texttt{"auto"} time, the values of $t$, $C$ and the clock values are""",
         opts=[r"$t = \tfrac{3\pi}{16}$, $C = \tfrac23$, clock values $1$ and $2$", r"$t = \tfrac\pi8$, $C = 1$, clock values $\tfrac23$ and $\tfrac43$",
               r"$t = \tfrac{3\pi}{16}$, $C = \tfrac43$, clock values $1$ and $2$", r"$t = \tfrac{3\pi}{8}$, $C = \tfrac23$, clock values $2$ and $4$"],
         sol=r"""$\kappa = 2 \le 7$: $t = 2\pi/(16\cdot\tfrac23) = \tfrac{3\pi}{16}$, $C = 2\pi/(16t) = \tfrac23$. Clock values $k = \lambda t\cdot16/(2\pi) = \lambda\cdot\tfrac32$: $1$ and $2$ --- exact, so HHL is exact (simulated fidelity 1,
$x = (1.125, 0.375)$, $P_{\text{success}} = 0.625$). $C = \tfrac43$ would exceed $\lambda_{\min}$."""),

    dict(id="C14", kind="mcq",
         q=r"""In the sampling read-out the notebook keeps a shot only if""",
         opts=[r"all clock bits are 0 and the ancilla bit is 1", r"the ancilla bit is 0", r"the clock bits encode an eigenvalue", r"the system bits are all 0"],
         sol=r"""Code: \texttt{if bits[:n\_c] == "0" * n\_c and bits[-1] == "1"}. Ancilla $= 1$ selects the branch carrying $C/\lambda$; clock $= 0$ selects the correctly uncomputed part (in the exact case
this is automatic). The system bits of the accepted shots are then histogrammed to estimate $|x_i|^2/\|x\|^2$."""),

    dict(id="C15", kind="mcq",
         q=r"""The ideal success probability is $p = 0.3559$ and the notebook uses $20\,000$ shots. The standard error of the measured fraction of accepted shots is about""",
         opts=[r"$0.0034$", r"$0.0119$", r"$0.0001$", r"$0.048$"],
         sol=r"""Binomial: $\sqrt{p(1 - p)/S} = \sqrt{0.3559\times0.6441/20000} = 0.0034$. The recorded fraction $0.3543$ differs from $0.3559$ by $0.0016$, about half a standard error --- consistent.
$0.0119 = 1/\sqrt{7086}$ is the notebook's per-component scale for the \emph{accepted} shots."""),

    dict(id="C16", kind="mcq",
         q=r"""On hardware, the post-selected system bitstrings of HHL directly estimate""",
         opts=[r"$|x_i|^2/\|x\|^2$ only --- signs/phases and the norm need further measurements",
               r"every component $x_i$ including its sign",
               r"$x_i/\|x\|$ including its sign",
               r"$\|x\|$ only"],
         sol=r"""Computational-basis measurement gives probabilities $|\braket{i|x}|^2$. Signs need interference with other bases; the norm needs $P_{\text{success}}$ (with $\|b\|$ and $C$). The notebook shows this:
sampled $|x_i|^2$ $(0.7691, 0.0587, 0.1420, 0.0302)$ vs.\ exact $(0.7622, 0.0598, 0.1476, 0.0305)$, magnitudes only. Reading all $d$ components to precision $\varepsilon$
costs about $d/\varepsilon^2$ shots."""),

    dict(id="C17", kind="mcq",
         q=r"""The recorded circuit statistics of the default HHL circuit are\\
\hspace*{1em}\texttt{h: 16, ccustom\_unitary\_constant: 8, swap: 4, cr1: 12, x: 56, ccccry: 15}.\\
How many multi-qubit gates is that?""",
         opts=[r"$39$", r"$111$", r"$23$", r"$95$"],
         sol=r"""Multi-qubit: $8$ controlled $U^{\pm2^j}$ + $4$ swaps + $12$ \texttt{cr1} + $15$ multi-controlled $R_Y$ $= 39$, the recorded ``multi-qubit gates: 39''. Single-qubit: $16 + 56 = 72$; total $111$.
The $8$ custom gates are $4$ controlled $U^{2^j}$ in phase estimation and $4$ inverses in the uncomputation."""),

    dict(id="C18", kind="match",
         q=r"""Match each HHL ingredient (List-I) with the code that implements it in \texttt{HHL.ipynb} (List-II).""",
         match=([r"preparation of $\ket0\ket b\ket0$", r"controlled powers $U^{2^j}$", r"eigenvalue inversion", r"extraction of the post-selected amplitudes"],
                [r"\texttt{kernel.cry(theta, clock, ancilla)} between $X$ gates", r"\texttt{postselected\_amplitudes(state\_vector)}",
                 r"\texttt{kernel.qalloc(bit\_reverse\_order(initial, n\_tot))}", r"\texttt{cudaq.register\_operation(name, matrix\_power(U, 2**j))}"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: amplitude initialisation of the whole register (qubit order converted for CUDA-Q) (3).
\item Q: each power is registered as a custom unitary and applied controlled by one clock qubit (4).
\item R: the multi-controlled $R_Y$ with polarity $X$ gates (1).
\item S: the helper picks the system amplitudes with clock $= 0$ and ancilla $= 1$ (2).
\end{itemize}"""),

    dict(id="C19", kind="para", passage="C",
         q=r"""After phase estimation the state of system $\otimes$ clock (ancilla still $\ket0$) is""",
         opts=[r"$\tfrac{1}{\sqrt2}\big(\ket0\ket{001} + \ket1\ket{010}\big)$", r"$\tfrac{1}{\sqrt2}\big(\ket0 + \ket1\big)\ket{001}$",
               r"$\tfrac{1}{\sqrt2}\big(\ket0\ket{100} + \ket1\ket{010}\big)$", r"$\tfrac{1}{\sqrt2}\big(\ket0\ket{001} + \ket1\ket{011}\big)$"],
         sol=r"""$A$ is diagonal, so $\ket0$ and $\ket1$ are eigenvectors with $\lambda = 1, 2$, and $\beta = (\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2})$. Clock values $k = \lambda$: $1 = \ket{001}$ and $2 = \ket{010}$ (most significant
bit first), both exact. Result: $\tfrac{1}{\sqrt2}(\ket0\ket{001} + \ket1\ket{010})$ --- system and clock entangled. $\ket{100}$ is $1$ written in reverse bit order."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""The success probability and the normalised output state $\ket x$ are""",
         opts=[r"$0.625$ and $\tfrac{1}{\sqrt5}(2, 1)$", r"$0.625$ and $\tfrac{1}{\sqrt2}(1, 1)$", r"$0.75$ and $\tfrac{1}{\sqrt5}(2, 1)$", r"$1$ and $\tfrac{1}{\sqrt5}(1, 2)$"],
         sol=r"""Ancilla amplitudes $C/\lambda = 1$ and $\tfrac12$: post-selected branch $\tfrac{1}{\sqrt2}(1\cdot\ket0 + \tfrac12\ket1)$, probability $\tfrac12(1 + \tfrac14) = 0.625$. Normalising: $(1, \tfrac12)/\sqrt{1.25} = \tfrac{1}{\sqrt5}(2, 1)$,
proportional to $A^{-1}b = \tfrac{1}{\sqrt2}(1, \tfrac12)$ \checkmark. (Full circuit simulation: $P = 0.625$, $\ket x = (0.8944, 0.4472)$.) $(1, 2)/\sqrt5$ would be $A\ket b$ normalised."""),
]
