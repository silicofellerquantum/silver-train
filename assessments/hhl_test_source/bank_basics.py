"""Section C - HHL algorithm fundamentals (problem encoding, state preparation, Hamiltonian simulation, phase
estimation, eigenvalue inversion, controlled rotation, ancilla, uncomputation, measurement and the notebook's code).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""HHL is run by hand on $A = \begin{pmatrix}\frac32&\frac12\\[2pt] \frac12&\frac32\end{pmatrix}$, $\vec b = (1, 0)$ ($n_b = 1$). The eigenpairs are
$\lambda_1 = 1$, $\ket{u_1} = \frac{1}{\sqrt2}(\ket0 - \ket1)$ and $\lambda_2 = 2$, $\ket{u_2} = \frac{1}{\sqrt2}(\ket0 + \ket1)$. Phase estimation uses $n_c = 2$
clock qubits and $t = \pi/2$; because both eigenvalues are positive, the clock value $k$ is read as an \emph{unsigned} integer here
($\tilde\lambda_k = kC$). States are written as $\ket{\text{clock}}\ket{\text{system}}$; the ancilla starts in $\ket0$ and is not shown until the rotation.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""For a Hermitian $A$ and a prepared $\ket b$, a successful run of the HHL algorithm produces""",
         opts=[r"the quantum state $\ket x \propto A^{-1}\ket b$ in the system register", r"the classical list of all $d$ numbers $x_1, \dots, x_d$",
               r"the matrix $A^{-1}$ stored in the clock register", r"the eigenvalues of $A$ as a classical list"],
         sol=r"""HHL transforms $\ket b = \sum_j\beta_j\ket{u_j}$ into $\ket x = \frac{\sum_j(\beta_j/\lambda_j)\ket{u_j}}{\lVert\cdot\rVert}$, a normalised quantum state
whose amplitudes are proportional to the solution vector. The $d$ numbers are \emph{encoded} in $\log_2d$ qubits; reading them all out
needs $\Omega(d)$ repetitions (Section D), so HHL is useful when only a property of $\vec x$ (an expectation value, an overlap) is needed.
The eigenvalues are used only temporarily inside the circuit and are uncomputed; $A^{-1}$ is never stored."""),

    dict(id="C2", kind="mcq",
         q=r"""With $\ket b = \sum_j\beta_j\ket{u_j}$ and exactly representable clock values $k_j$, the state after phase estimation (clock $\otimes$ system $\otimes$ ancilla) is""",
         opts=[r"$\sum_j\beta_j\ket{k_j}\ket{u_j}\ket0$", r"$\sum_j\frac{\beta_j}{\lambda_j}\ket{k_j}\ket{u_j}\ket0$",
               r"$\left(\frac{1}{\sqrt{d}}\sum_j\ket{k_j}\right)\otimes\ket b\otimes\ket0$", r"$\ket{0\dots0}\ket x\ket1$"],
         sol=r"""Phase estimation is linear. On each eigencomponent $\ket{u_j}$, $U = e^{iAt}$ acts as the phase $e^{i\lambda_jt} = e^{2\pi i\,\varphi_j}$ with
$\varphi_j = \lambda_jt/2\pi$, and QPE writes $k_j = \varphi_j2^{n_c}$ into the clock: $\ket0\ket{u_j} \mapsto \ket{k_j}\ket{u_j}$. Summing with weights $\beta_j$:
\[ \ket{0}\ket b\ket0 = \sum_j\beta_j\ket0\ket{u_j}\ket0 \;\longmapsto\; \sum_j\beta_j\ket{k_j}\ket{u_j}\ket0 . \]
The clock is now \emph{entangled} with the system; no factor $1/\lambda_j$ has appeared yet --- that is the job of the controlled rotation."""),

    dict(id="C3", kind="mcq",
         q=r"""In the default notebook run ($n_c = 4$, $t = \pi/8$), the eigenvalue $\lambda = 3$ is written into the clock register as the bit string""",
         opts=[r"$0011$", r"$1100$", r"$0110$", r"$0010$"],
         sol=r"""$k = \dfrac{\lambda t\,2^{n_c}}{2\pi} = \dfrac{3\cdot\frac\pi8\cdot16}{2\pi} = \dfrac{6\pi}{2\pi} = 3$. With $n_c = 4$ bits and the most significant bit first, $3 = 0011_2$.
In fact $t = \pi/8$ was chosen (``auto'') so that $k = \lambda$ for every eigenvalue: $1, 2, 3, 4 \to 0001, 0010, 0011, 0100$, all exact, which is
why the recorded fidelity is $1$. $1100$ is the bit-reversed string; $0110 = 6$ comes from dividing by $\pi$ instead of $2\pi$."""),

    dict(id="C4", kind="mcq",
         q=r"""For the default notebook run ($n_c = 4$, $t = \pi/8$) the constant $C$ and the rotation angle $\theta$ applied for the clock value of $\lambda = 2$ are""",
         opts=[r"$C = 1$; $\ \theta = \pi/3$", r"$C = 1$; $\ \theta = \pi/6$", r"$C = \pi/8$; $\ \theta = 2\arcsin(\pi/16)$",
               r"$C = \tfrac1{16}$; $\ \theta = 2\arcsin\tfrac1{32}$"],
         sol=r"""$C = \dfrac{2\pi}{2^{n_c}t} = \dfrac{2\pi}{16\cdot\pi/8} = \dfrac{2\pi}{2\pi} = 1$ --- the eigenvalue that maps to clock value $1$.
For $\lambda = 2$: $\theta = 2\arcsin\frac{C}{\lambda} = 2\arcsin\frac12 = 2\cdot\frac\pi6 = \frac\pi3$, so the ancilla becomes
$\cos\frac\pi6\ket0 + \sin\frac\pi6\ket1 = \frac{\sqrt3}{2}\ket0 + \frac12\ket1$: amplitude $C/\lambda = \frac12$ on $\ket1$. (Note $R_Y(\theta)$ rotates by $\theta/2$ in amplitude, hence the factor $2$.)"""),

    dict(id="C5", kind="mcq",
         q=r"""With $C = 1$, the eigencomponent with $\lambda = 4$ leaves the ancilla in $\sqrt{1 - C^2/\lambda^2}\,\ket0 + \frac C\lambda\ket1$. For this component the amplitude
on $\ket1$ and the probability of measuring the ancilla as $1$ are""",
         opts=[r"$\tfrac14$ and $\tfrac1{16}$", r"$\tfrac14$ and $\tfrac14$", r"$\tfrac12$ and $\tfrac14$", r"$4$ and $1$"],
         sol=r"""Amplitude $C/\lambda = 1/4$; probability $|1/4|^2 = 1/16 = 0.0625$. The $\ket0$ amplitude is $\sqrt{15/16} = 0.968$ so the ancilla is normalised.
Large eigenvalues are therefore strongly suppressed, exactly as $1/\lambda$ requires; the price is that components with large $\lambda$ rarely
pass post-selection. ($\frac12$ would correspond to $1/\sqrt\lambda$, and an amplitude $4$ is impossible --- that is why $C \le |\lambda|_{\min}$ is needed.)"""),

    dict(id="C6", kind="match",
         q=r"""Match each part of the HHL circuit (List-I) with its role (List-II).""",
         match=([r"clock register ($n_c$ qubits)", r"system register ($n_b$ qubits)", r"ancilla qubit",
                 r"inverse phase estimation after the rotation"],
                [r"is measured; outcome $1$ flags a successful inversion", r"holds $\ket b$ at the start and $\ket x$ after success",
                 r"holds the binary eigenvalue estimates $k_j$", r"returns the clock to $\ket{0\dots0}$ and removes its entanglement with the system"]),
         opts=[r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4",
               r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: QPE writes $k_j = \lambda_jt2^{n_c}/2\pi$ into the clock (3).
\item Q: the system starts in $\ket b$ (amplitude encoding) and ends in $\ket x$ when the run succeeds (2).
\item R: the ancilla carries the amplitude $C/\tilde\lambda$ on $\ket1$; measuring $1$ post-selects the $A^{-1}$ branch (1).
\item S: uncomputation maps $\ket{k_j}\ket{u_j} \to \ket{0}\ket{u_j}$ so that the eigencomponents can interfere into $\ket x$ (4).
\end{itemize}
Total qubits: $n_c + n_b + 1$ ($4 + 2 + 1 = 7$ in the notebook)."""),

    dict(id="C7", kind="mcq",
         q=r"""HHL is run with $A = \operatorname{diag}(1, 2)$, $\ket b = \frac{1}{\sqrt2}(\ket0 + \ket1)$, $C = 1$ and exactly representable eigenvalues. The success probability is""",
         opts=[r"$0.625$", r"$0.75$", r"$0.5$", r"$1$"],
         sol=r"""The eigenvectors are $\ket0, \ket1$ with $\lambda = 1, 2$ and $\beta_1 = \beta_2 = \frac{1}{\sqrt2}$.
\[ P_{\text{success}} = \sum_j|\beta_j|^2\frac{C^2}{\lambda_j^2} = \frac12\cdot\frac11 + \frac12\cdot\frac14 = \frac58 = 0.625. \]
Equivalently $P = C^2\lVert A^{-1}\ket b\rVert^2$ with $A^{-1}\ket b = \frac{1}{\sqrt2}(1, \frac12)$. ($0.75$ forgets to square $C/\lambda$.) The post-selected
state is $\propto \ket0 + \frac12\ket1$, i.e.\ $\ket x = \frac{1}{\sqrt5}(2\ket0 + \ket1)$."""),

    dict(id="C8", kind="mcq",
         q=r"""An $8\times8$ Hermitian system is solved with the notebook's circuit and $n_c = 5$ clock qubits. The total number of qubits is""",
         opts=[r"$9$", r"$8$", r"$14$", r"$6$"],
         sol=r"""$n_b = \log_2 8 = 3$ system qubits, $n_c = 5$ clock qubits and $1$ ancilla: $3 + 5 + 1 = 9$. The matrix dimension enters only
logarithmically --- the source of HHL's potential advantage. ($14 = 8 + 5 + 1$ confuses the dimension $d$ with the number of qubits $\log_2d$.)"""),

    dict(id="C9", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} If the clock register is not uncomputed, the post-selected system register is in general not in the pure state $\ket x$,
because it remains entangled with the clock.\\[1mm]
\textbf{Statement II:} Uncomputation can be replaced by simply measuring the clock register, with the same output $\ket x$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""After the rotation and post-selection the state is $\propto\sum_j\frac{\beta_j}{\lambda_j}\ket{k_j}\ket{u_j}$.
\textbf{I is true:} with distinct $\ket{k_j}$ the clock labels the eigencomponents; tracing it out leaves the mixture $\sum_j p_j\ket{u_j}\!\bra{u_j}$,
not the superposition $\ket x$.
\textbf{II is false:} measuring the clock gives one value $k_j$ and collapses the system to the single eigenvector $\ket{u_j}$ --- the
interference between eigencomponents that forms $\ket x$ is destroyed. Uncomputation \emph{coherently} resets $\ket{k_j} \to \ket0$. Answer: (C)."""),

    dict(id="C10", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} In HHL the constant $C$ must satisfy $C \le \min_j|\lambda_j|$ (more precisely, $\le$ the smallest $|\tilde\lambda|$ that can occur).\\[1mm]
\textbf{Reason (R):} The controlled rotation puts the amplitude $C/\tilde\lambda$ on the ancilla state $\ket1$, and an amplitude cannot exceed $1$ in magnitude.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$ with $\sin\frac\theta2 = C/\tilde\lambda$, which requires $|C/\tilde\lambda| \le 1$.
\textbf{A is true} and follows from R: for every eigenvalue that occurs, $|C/\tilde\lambda| \le 1 \iff C \le |\tilde\lambda|$, so $C$ is bounded by the smallest one.
(The notebook sets $C = 2\pi/(2^{n_c}t)$, the smallest non-zero representable value, and clips the argument of $\arcsin$ to $[-1, 1]$.)
Choosing $C$ as large as allowed maximises the success probability $\propto C^2$. Answer: (A)."""),

    dict(id="C11", kind="mcq",
         q=r"""In the default notebook circuit ($n_c = 4$, $t = \pi/8$, $C = 1$, two's-complement clock), the clock value $k = 14$ ($1110$) is treated as the eigenvalue
$\tilde\lambda$ and rotated by $\theta$, where""",
         opts=[r"$\tilde\lambda = -2$, $\ \theta = -\pi/3$", r"$\tilde\lambda = 14$, $\ \theta = 2\arcsin\tfrac1{14}$",
               r"$\tilde\lambda = -2$, $\ \theta = +\pi/3$", r"$\tilde\lambda = 2$, $\ \theta = \pi/3$"],
         sol=r"""$k = 14 \ge 2^{n_c-1} = 8$, so it represents $14 - 16 = -2$ (two's complement), and $\tilde\lambda = -2\,C = -2$.
(Physically: $e^{-2it} = e^{i(2\pi - 2t)}$, so a negative eigenvalue produces the phase of $k = 16 - 2 = 14$.)
$\theta = 2\arcsin\frac{C}{\tilde\lambda} = 2\arcsin(-\tfrac12) = -\frac\pi3$: the ancilla amplitude is $-\frac12$. The \emph{sign} matters --- it is the
sign of $1/\lambda$ in $A^{-1}$; $+\pi/3$ would solve the system with $|A|$ instead of $A$."""),

    dict(id="C12", kind="mcq",
         q=r"""The eigenvalue-inversion loop of \texttt{HHL.ipynb} runs over $k = 1, \dots, 2^{n_c}-1$; for each $k$ it applies \texttt{kernel.x} to every clock qubit whose bit
of $k$ is $0$, then the multi-controlled \texttt{cry}, then the same \texttt{x} gates again. For $n_c = 4$ the total number of \texttt{x} gates is""",
         opts=[r"$56$", r"$60$", r"$28$", r"$32$"],
         sol=r"""For each $k$ the number of zero bits is $4 - \mathrm{popcount}(k)$, and each zero bit costs $2$ gates. Summed over $k = 1, \dots, 15$:
$\sum_k\mathrm{popcount}(k) = 4\cdot2^3 = 32$ (each bit is $1$ in half of the $16$ values; $k = 0$ adds nothing), so
\[ 2\sum_{k=1}^{15}(4 - \mathrm{popcount}(k)) = 2\,(60 - 32) = 56. \]
This matches the recorded circuit statistics (\texttt{'x': 56}, \texttt{'ccccry': 15}). $28$ forgets the second (restoring) layer."""),

    dict(id="C13", kind="mcq",
         q=r"""Let $\vec a$ be the (unnormalised) amplitudes of the system register in the branch clock $= 0\dots0$, ancilla $= 1$ of an exact HHL run. The notebook recovers
the solution, including its norm, as""",
         opts=[r"$\vec x = \dfrac{\lVert\vec b\rVert}{C}\,\vec a$", r"$\vec x = \dfrac{C}{\lVert\vec b\rVert}\,\vec a$", r"$\vec x = \dfrac{\vec a}{\sqrt{P_{\text{success}}}}$",
               r"$\vec x = \vec a$"],
         sol=r"""The surviving branch is $\vec a = \sum_j\beta_j\frac{C}{\lambda_j}\ket{u_j} = C\,A^{-1}\ket b = C\,A^{-1}\frac{\vec b}{\lVert\vec b\rVert} = \frac{C}{\lVert\vec b\rVert}\vec x$.
Solving for $\vec x$: $\vec x = \frac{\lVert\vec b\rVert}{C}\vec a$ (code: \texttt{x\_quantum\_full = amplitudes * b\_norm / C\_constant}).
$\vec a/\sqrt{P_{\text{success}}}$ is the \emph{normalised} $\ket x$ (correct direction, norm $1$), not $\vec x$. Note that $\lVert\vec a\rVert^2 = P_{\text{success}}$, so on hardware the norm
of $\vec x$ can be estimated from the measured success rate."""),

    dict(id="C14", kind="mcq",
         q=r"""For the notebook's default system ($A$ with eigenvalues $1, 2, 3, 4$ and $\vec b = (1, 0, 0, 0)$, see conventions) the exact solution is""",
         opts=[r"$\vec x = \tfrac1{48}(25,\ 7,\ 11,\ 5)$", r"$\vec x = \tfrac{1}{\sqrt{820}}(25,\ 7,\ 11,\ 5)$", r"$\vec x = (1,\ \tfrac12,\ \tfrac13,\ \tfrac14)$",
               r"$\vec x = \tfrac1{48}(25,\ -7,\ -11,\ 5)$"],
         sol=r"""$A = H_4\Lambda H_4$ with $H_4 = \frac12\begin{psmallmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{psmallmatrix}$ ($H_4^2 = I$), so $A^{-1} = H_4\Lambda^{-1}H_4$.
$H_4\vec b = \frac12(1, 1, 1, 1)$ (all $\beta_j = \frac12$); $\Lambda^{-1}$ gives $\frac12(1, \frac12, \frac13, \frac14) = \frac1{24}(12, 6, 4, 3)$; then
\[ H_4\cdot\tfrac{1}{24}(12, 6, 4, 3) = \tfrac1{48}(12{+}6{+}4{+}3,\ 12{-}6{+}4{-}3,\ 12{+}6{-}4{-}3,\ 12{-}6{-}4{+}3) = \tfrac1{48}(25, 7, 11, 5). \]
$= (0.520833, 0.145833, 0.229167, 0.104167)$, exactly the recorded output. Check the first row: $2.5\cdot25 - 0.5\cdot7 - 1\cdot11 = 48$.
$\frac{1}{\sqrt{820}}(\dots)$ is the normalised $\ket x$; $(1, \frac12, \frac13, \frac14)$ inverts the eigenvalues without changing basis."""),

    dict(id="C15", kind="match",
         q=r"""Match each step of HHL (List-I) with the CUDA-Q / Python code from \texttt{HHL.ipynb} that implements it (List-II).""",
         match=([r"amplitude encoding of $\ket b$", r"controlled $U^{2^j} = e^{iAt2^j}$", r"eigenvalue inversion", r"post-selection of the shots"],
                [r"\texttt{kernel.cry(theta, clock, ancilla)}", r"\texttt{cudaq.register\_operation(name, matrix\_power(U, 2**j))}",
                 r"\texttt{bits[:n\_c] == \textquotesingle0\textquotesingle*n\_c and bits[-1] == \textquotesingle1\textquotesingle}",
                 r"\texttt{kernel.qalloc(initial\_state\_vector)}"]),
         opts=[r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3",
               r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: the initial state $\ket{0\dots0}_{\text{clock}}\ket b\ket0$ is loaded in one step with \texttt{qalloc(vector)} (4).
\item Q: each $U^{2^j}$ is an exact matrix registered as a custom gate and applied controlled by one clock qubit (2).
\item R: the multi-controlled $R_Y(\theta_k)$ on the ancilla, $\theta_k = 2\arcsin(C/\tilde\lambda_k)$ (1).
\item S: in the sampled bit strings (clock $|$ system $|$ ancilla) keep clock $= 0\dots0$ and ancilla $= 1$ (3).
\end{itemize}"""),

    dict(id="C16", kind="mcq",
         q=r"""Starting from $\ket{00}$, which circuit prepares $\ket b = \frac12(\ket{00} - \ket{01} + \ket{10} - \ket{11})$, i.e.\ $\vec b \propto (1, -1, 1, -1)$?""",
         opts=[r"$H$ on both qubits, then $Z$ on $q_1$", r"$H$ on both qubits, then $Z$ on $q_0$", r"$H$ on both qubits only",
               r"$X$ on both qubits, then $H$ on both qubits"],
         sol=r"""Factorise: $\frac12(\ket0 + \ket1)_{q_0}(\ket0 - \ket1)_{q_1} = \ket+\ket-$. $H\ket0 = \ket+$ on $q_0$; on $q_1$, $ZH\ket0 = Z\ket+ = \ket-$.
\begin{itemize}\itemsep0pt
\item $Z$ on $q_0$ gives $\ket-\ket+ \propto (1, 1, -1, -1)$; $H\otimes H$ alone gives $(1, 1, 1, 1)$; $X\otimes X$ then $H\otimes H$ gives $\ket-\ket- \propto (1, -1, -1, 1)$.
\end{itemize}
Structured vectors like this are cheap to prepare; a \emph{general} $\vec b$ needs $O(d)$ gates (or a QRAM), which can cancel HHL's speed-up --- the notebook
side-steps this with exact amplitude initialisation on a simulator."""),

    dict(id="C17", kind="mcq",
         q=r"""Inside the phase-estimation part of HHL, the inverse QFT on the clock register""",
         opts=[r"turns the kicked-back phases $e^{2\pi i\,kj/2^{n_c}}$ into the computational-basis state $\ket{k}$ (the binary eigenvalue estimate)",
               r"computes $1/\lambda$ for every eigenvalue", r"prepares $\ket b$ in the eigenbasis of $A$",
               r"measures the eigenvalues and sends them to the classical computer"],
         sol=r"""After the Hadamards and controlled powers the clock holds $\frac{1}{\sqrt{2^{n_c}}}\sum_{j}e^{i\lambda t j}\ket j = \mathrm{QFT}\ket{k}$ with $k = \lambda t2^{n_c}/2\pi$
(phase kickback). The inverse QFT maps this Fourier-basis state back to $\ket k$, so the eigenvalue is written as a binary number that a controlled
rotation can read. The division by $\lambda$ happens later (rotation), nothing is measured, and the eigenbasis expansion of $\ket b$ is never
computed explicitly --- linearity does it."""),

    dict(id="C18", kind="stmt", correct=0,
         q=r"""Assume $\ket b$ is itself an eigenvector $\ket{u_j}$ of $A$ and $\lambda_j$ is exactly representable by the clock.\\[1mm]
\textbf{Statement I:} The success probability is $C^2/\lambda_j^2$.\\[1mm]
\textbf{Statement II:} After phase estimation the clock register is in the single basis state $\ket{k_j}$ (not entangled with the system).""",
         opts=STATEMENT_OPTIONS,
         sol=r"""With $\beta_j = 1$ and all other $\beta$ zero:
\textbf{I is true:} $P = \sum|\beta|^2C^2/\lambda^2 = C^2/\lambda_j^2$.
\textbf{II is true:} QPE maps $\ket0\ket{u_j} \to \ket{k_j}\ket{u_j}$, a product state; with an exactly representable phase the clock is a single basis state.
The output is $\ket x = \ket{u_j}$ (since $A^{-1}\ket{u_j} = \lambda_j^{-1}\ket{u_j}$), as expected. Answer: (A)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The state after phase estimation is""",
         opts=[r"$\frac{1}{\sqrt2}\left(\ket{01}\ket{u_1} + \ket{10}\ket{u_2}\right)$", r"$\frac{1}{\sqrt2}\left(\ket{01} + \ket{10}\right)\otimes\ket0$",
               r"$\frac{1}{\sqrt2}\left(\ket{10}\ket{u_1} + \ket{01}\ket{u_2}\right)$", r"$\frac{1}{\sqrt2}\left(\ket{00}\ket{u_1} + \ket{01}\ket{u_2}\right)$"],
         sol=r"""Expand $\ket b = \ket0 = \frac{1}{\sqrt2}(\ket{u_1} + \ket{u_2})$, so $\beta_1 = \beta_2 = \frac{1}{\sqrt2}$.
Clock values: $k = \lambda t2^{n_c}/2\pi = \lambda\cdot\frac\pi2\cdot\frac{4}{2\pi} = \lambda$, so $\lambda_1 = 1 \to \ket{01}$ and $\lambda_2 = 2 \to \ket{10}$, both exact.
Hence $\frac{1}{\sqrt2}(\ket{01}\ket{u_1} + \ket{10}\ket{u_2})$: clock and system are entangled. Also $C = 2\pi/(2^{n_c}t) = 2\pi/(4\cdot\frac\pi2) = 1$.
(Option with $\ket0$ factored out ignores this entanglement; the swapped option exchanges the eigenvalues.)"""),

    dict(id="C20", kind="para", passage="C",
         q=r"""After the controlled rotation, uncomputation and a successful ancilla measurement, the success probability and the system state are""",
         opts=[r"$P = \tfrac58$; $\ \frac{1}{\sqrt{10}}(3\ket0 - \ket1)$", r"$P = \tfrac58$; $\ \frac{1}{\sqrt{10}}(3\ket0 + \ket1)$",
               r"$P = \tfrac34$; $\ \frac{1}{\sqrt{10}}(3\ket0 - \ket1)$", r"$P = 1$; $\ \ket0$"],
         sol=r"""Rotation: $\ket{01} \to$ ancilla amplitude $C/1 = 1$; $\ket{10} \to C/2 = \frac12$. After uncomputation the ancilla-$1$ branch is
\[ \tfrac{1}{\sqrt2}\Big(1\cdot\ket{u_1} + \tfrac12\ket{u_2}\Big) = \tfrac12\Big[(\ket0 - \ket1) + \tfrac12(\ket0 + \ket1)\Big] = \tfrac34\ket0 - \tfrac14\ket1 . \]
$P = (\frac34)^2 + (\frac14)^2 = \frac{10}{16} = \frac58$ $\big(= \frac12\cdot1 + \frac12\cdot\frac14\big)$, and the normalised state is $\frac{1}{\sqrt{10}}(3\ket0 - \ket1)$.
Classical check: $A^{-1}\vec b = \frac12\begin{psmallmatrix}3/2&-1/2\\-1/2&3/2\end{psmallmatrix}(1, 0)^T = (\frac34, -\frac14)$ --- with $\lVert\vec b\rVert = C = 1$ the branch \emph{is} $\vec x$.
($P = \frac34$ forgets to square the amplitudes.)"""),
]
