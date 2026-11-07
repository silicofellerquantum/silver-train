"""Section B - Quantum Computing Foundations for Shor's algorithm (registers and superposition, entanglement between the
counting and work registers, unitary modular multiplication, eigenvalues and eigenvectors, phase kickback, Hadamard
and controlled gates, interference, reversibility, measurement, memory of simulation, phase estimation).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""Quantum phase estimation (QPE) with $t = 3$ counting qubits is applied to a unitary $U$ and an eigenvector $\ket u$ with
$U\ket u = e^{2\pi i\varphi}\ket u$, $0 \le \varphi < 1$. After the Hadamards, the controlled-$U^{2^j}$ gates and the inverse QFT, the counting register is
measured and gives $y \in \{0, \dots, 7\}$ with probability
\[ P(y) = \Big|\frac18\sum_{x=0}^{7}e^{2\pi ix(\varphi - y/8)}\Big|^2 = \frac{\sin^2\big(8\pi(\varphi - y/8)\big)}{64\,\sin^2\big(\pi(\varphi - y/8)\big)}\quad(P = 1\text{ if }\varphi = y/8). \]
The estimate of the phase is $y/8$; $y$ is written in binary with the most significant bit first.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""After $H^{\otimes8}$ acts on $\ket{0}^{\otimes8}$ (the counting register of the notebook for $N = 15$), the amplitude of each basis state $\ket x$ is""",
         opts=[r"$1/16$", r"$1/256$", r"$1/8$", r"$1/\sqrt8$"],
         sol=r"""$H^{\otimes t}\ket{0}^{\otimes t} = \frac{1}{\sqrt{2^t}}\sum_{x=0}^{2^t-1}\ket x$. With $t = 8$: $\frac{1}{\sqrt{256}} = \frac1{16}$. The probabilities are $\frac{1}{256}$ each.
This superposition is the input on which the modular exponentiation acts ``for all $x$ at once''."""),

    dict(id="B2", kind="mcq",
         q=r"""The state $\frac{1}{\sqrt8}\sum_{x=0}^{7}\ket x\ket{2^x \bmod 15}$ is prepared and the \emph{second} register is measured with result $4$. The probability
of this result and the post-measurement state of the first register are""",
         opts=[r"$1/4$; $\ \frac{1}{\sqrt2}(\ket2 + \ket6)$", r"$1/8$; $\ \ket2$", r"$1/4$; $\ \frac{1}{\sqrt2}(\ket2 - \ket6)$", r"$1/2$; $\ \frac{1}{\sqrt2}(\ket2 + \ket6)$"],
         sol=r"""$2^x \bmod 15$ for $x = 0, \dots, 7$: $1, 2, 4, 8, 1, 2, 4, 8$. The value $4$ occurs for $x = 2$ and $x = 6$, so
$P = 2\times\frac18 = \frac14$, and the first register collapses to $\frac{1}{\sqrt2}(\ket2 + \ket6)$ (equal positive amplitudes, renormalised). It is periodic
with period $r = 4$ and an unknown offset --- the QFT removes the offset from the probabilities."""),

    dict(id="B3", kind="mcq",
         q=r"""Why does the operator $U_a\ket y = \ket{ay \bmod N}$ (for $0 \le y < N$; identity for $y \ge N$) require $\gcd(a, N) = 1$ to be a valid quantum gate?""",
         opts=[r"only then is $y \mapsto ay \bmod N$ a permutation of $\{0, \dots, N-1\}$, so $U_a$ is a permutation matrix and hence unitary",
               r"because otherwise $U_a$ would not be Hermitian",
               r"because the eigenvalues of $U_a$ must be real",
               r"because a quantum gate must map $\ket0$ to $\ket0$"],
         sol=r"""If $\gcd(a, N) = d > 1$, two inputs collide: e.g.\ $a = 3$, $N = 15$: $3\cdot0 \equiv 3\cdot5 \equiv 0$. A non-injective map cannot be unitary.
For $\gcd(a, N) = 1$, $a$ is invertible modulo $N$, the map is a bijection, and a permutation matrix satisfies $U^\dagger U = I$. Gates need not be
Hermitian, and their eigenvalues are complex phases."""),

    dict(id="B4", kind="mcq",
         q=r"""On the subspace spanned by $\{\ket1, \ket2, \ket4, \ket8\}$, the operator $U_2\ket y = \ket{2y \bmod 15}$ has the eigenvalues""",
         opts=[r"$1, i, -1, -i$", r"$1, 2, 4, 8$", r"$1, -1$ (each twice)", r"$e^{2\pi i/15}, e^{4\pi i/15}, e^{8\pi i/15}, e^{16\pi i/15}$"],
         sol=r"""$U_2$ cycles $\ket1 \to \ket2 \to \ket4 \to \ket8 \to \ket1$: a cyclic shift of length $r = 4$, so $U_2^4 = I$ there and the eigenvalues are the fourth
roots of unity $e^{2\pi is/4}$, $s = 0, 1, 2, 3$. In general the eigenvalues on the orbit of $\ket1$ are $e^{2\pi is/r}$: the order $r$ is encoded in the
eigenphases $s/r$ --- this is what phase estimation reads out."""),

    dict(id="B5", kind="mcq",
         q=r"""With $\ket{u_s} = \frac{1}{\sqrt r}\sum_{k=0}^{r-1}e^{-2\pi isk/r}\ket{a^k \bmod N}$, the superposition $\frac{1}{\sqrt r}\sum_{s=0}^{r-1}\ket{u_s}$ equals""",
         opts=[r"$\ket1$", r"$\ket0$", r"$\frac{1}{\sqrt N}\sum_{y=0}^{N-1}\ket y$", r"$\ket a$"],
         sol=r"""$\frac{1}{\sqrt r}\sum_s\ket{u_s} = \frac1r\sum_k\Big(\sum_se^{-2\pi isk/r}\Big)\ket{a^k} = \sum_k\delta_{k,0}\ket{a^k} = \ket{a^0} = \ket1$, because the sum of
$r$-th roots of unity vanishes unless $k \equiv 0$. That is why the work register is initialised to $\ket1$: it is an equal superposition of all eigenvectors,
and QPE returns a random $s/r$."""),

    dict(id="B6", kind="mcq",
         q=r"""A control qubit in $\ket+$ controls a unitary $U$ acting on its eigenvector $\ket u$, $U\ket u = e^{i\phi}\ket u$, with $\phi = 2\pi/3$. Afterwards, the
expectation value $\langle X\rangle$ of the control qubit is""",
         opts=[r"$-1/2$", r"$1/2$", r"$1$", r"$-\sqrt3/2$"],
         sol=r"""Phase kickback: $\frac{1}{\sqrt2}(\ket0 + \ket1)\ket u \mapsto \frac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)\ket u$ --- the work register is unchanged, the phase moves to
the control. For $\frac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)$, $\langle X\rangle = \cos\phi = \cos\frac{2\pi}{3} = -\frac12$ (and $\langle Y\rangle = \sin\phi = \frac{\sqrt3}{2}$)."""),

    dict(id="B7", kind="mcq",
         q=r"""In phase estimation, counting qubit $j$ controls $U^{2^j}$. If $U\ket u = e^{2\pi i/6}\ket u$, the phase kicked back onto counting qubit $j = 2$ is""",
         opts=[r"$4\pi/3$", r"$2\pi/3$", r"$\pi/3$", r"$2\pi$"],
         sol=r"""$U^{2^j}\ket u = e^{2\pi i\cdot2^j/6}\ket u$; for $j = 2$: $2\pi\cdot\frac46 = \frac{4\pi}{3}$ (equivalently $-\frac{2\pi}{3}$). Each counting qubit receives the phase
doubled relative to the previous one --- the binary-fraction pattern that the inverse QFT decodes."""),

    dict(id="B8", kind="mcq",
         q=r"""$(H\otimes H)\ket{11}$ equals""",
         opts=[r"$\frac12(\ket{00} - \ket{01} - \ket{10} + \ket{11})$", r"$\frac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$",
               r"$\frac12(\ket{00} - \ket{01} - \ket{10} - \ket{11})$", r"$\frac{1}{\sqrt2}(\ket{00} - \ket{11})$"],
         sol=r"""$H\ket1 = \frac{1}{\sqrt2}(\ket0 - \ket1)$, so $H\ket1\otimes H\ket1 = \frac12(\ket0 - \ket1)(\ket0 - \ket1) = \frac12(\ket{00} - \ket{01} - \ket{10} + \ket{11})$.
In general $H^{\otimes n}\ket x = 2^{-n/2}\sum_y(-1)^{x\cdot y}\ket y$: signs depend on the parity of $x\cdot y$."""),

    dict(id="B9", kind="mcq",
         q=r"""Two computational paths lead to the same outcome with amplitudes $\frac12$ and $\frac12e^{i\theta}$. For $\theta = 2\pi/3$ the probability of that outcome is""",
         opts=[r"$1/4$", r"$3/4$", r"$1/2$", r"$0$"],
         sol=r"""Amplitudes add before squaring: $\big|\frac12(1 + e^{i\theta})\big|^2 = \frac14|1 + e^{i\theta}|^2 = \frac14(2 + 2\cos\theta) = \cos^2\frac\theta2 = \cos^2\frac\pi3 = \frac14$.
Interference concentrates probability where the phases agree; in Shor's algorithm the inverse QFT arranges this for $y \approx k2^t/r$."""),

    dict(id="B10", kind="mcq",
         q=r"""Why does Shor's circuit compute $\ket x\ket1 \mapsto \ket x\ket{a^x \bmod N}$ instead of $\ket x \mapsto \ket{a^x \bmod N}$ in place?""",
         opts=[r"the map $x \mapsto a^x \bmod N$ is not injective ($x$ and $x + r$ give the same value), so it cannot be a unitary on a single register; keeping $x$ makes it reversible",
               r"because a single register cannot hold numbers larger than $N$",
               r"because measurement requires two registers",
               r"because the Hadamard gates must act on a different register than the QFT"],
         sol=r"""Quantum gates are unitary, hence invertible. Since $a^{x + r} \equiv a^x$, an in-place map would send different inputs to the same output. Keeping the input
$x$ and writing the result into a second register (here by multiplying the work register $\ket1$ by $a^x$) gives a reversible map and entangles the
registers: $\sum_x\ket x\ket{a^x \bmod N}$."""),

    dict(id="B11", kind="mcq",
         q=r"""A controlled-SWAP (Fredkin) gate is compiled as $\mathrm{CNOT}_{b\to a}\cdot\mathrm{Toffoli}_{c,a\to b}\cdot\mathrm{CNOT}_{b\to a}$, and each Toffoli uses the
standard decomposition with 6 CNOTs. The number of CNOTs per controlled-SWAP is""",
         opts=[r"$8$", r"$6$", r"$3$", r"$9$"],
         sol=r"""$1 + 6 + 1 = 8$ CNOTs (plus single-qubit $H$, $T$, $T^\dagger$ gates). A plain SWAP needs $3$ CNOTs, but the control turns the middle CNOT into a Toffoli.
Gate-level modular multipliers are built from many such gates, which is why compiled Shor circuits are much larger than the notebook's matrix gates."""),

    dict(id="B12", kind="mcq",
         q=r"""\texttt{Shor.ipynb} measures only the counting register. Compared with also measuring the work register first, the distribution of the counting
outcome $y$ is""",
         opts=[r"exactly the same: a measurement of the other register that is never used cannot change the marginal statistics of $y$",
               r"different: without measuring the work register no periodic structure appears",
               r"different: the peaks become twice as wide",
               r"undefined, because entangled registers cannot be measured separately"],
         sol=r"""The work register is not acted on after the controlled multiplications. Measuring it (or not) only changes which conditional state the counting
register is in; averaged over the unseen result, the reduced state of the counting register --- and hence $P(y)$ --- is identical (principle of implicit
measurement). Textbook derivations ``measure the second register'' only to simplify the algebra."""),

    dict(id="B13", kind="mcq",
         q=r"""In \texttt{Shor.ipynb} the work register has $n = $ \texttt{N.bit\_length()} qubits and the counting register $t = 2n$. The total number of qubits for
$N = 143$ is""",
         opts=[r"$24$", r"$16$", r"$22$", r"$21$"],
         sol=r"""$143 = 10001111_2$ has $8$ bits, so $n = 8$, $t = 16$ and $t + n = 24$ (the notebook's comment lists ``143 -> 24''). $16$ forgets the work register;
$22 = 2\lceil\log_2 143\rceil + 6$ has no basis."""),

    dict(id="B14", kind="mcq",
         q=r"""The memory of a double-precision state vector (16 bytes per amplitude) for the 24-qubit circuit of the previous question is about""",
         opts=[r"$268$ MB", r"$384$ bytes", r"$16.8$ MB", r"$4.3$ GB"],
         sol=r"""$16\times2^{24} = 268\,435\,456$ bytes $\approx 268$ MB (the notebook's table). $384 = 16\times24$ is linear; $16.8$ MB is $2^{20}\times16$ (20 qubits);
$4.3$ GB corresponds to 28 qubits. Each extra bit of $N$ adds 3 qubits, i.e.\ multiplies the memory by 8."""),

    dict(id="B15", kind="match",
         q=r"""Match each operation of the order-finding circuit (List-I) with its role (List-II).""",
         match=([r"$H$ on every counting qubit", r"controlled $U_{a^{2^j} \bmod N}$", r"inverse QFT on the counting register", r"$X$ on the last work qubit"],
                [r"turns the phases $2\pi\cdot2^js/r$ into a binary estimate $y \approx 2^ts/r$",
                 r"prepares the work register in $\ket1$, an equal superposition of the eigenvectors $\ket{u_s}$",
                 r"creates the uniform superposition over all exponents $x$",
                 r"performs modular exponentiation / kicks the eigenphases back onto the counting qubits"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item Hadamards: superposition over $x$ (3);
\item controlled powers of $U_a$: $\ket x\ket1 \mapsto \ket x\ket{a^x}$, equivalently phase kickback (4);
\item inverse QFT: interference that converts the periodic phase pattern into peaks at $y \approx k2^t/r$ (1);
\item $X$ on the least significant work qubit: $\ket{0001} = \ket1$ (2).
\end{itemize}
Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="B16", kind="match",
         q=r"""The phase gate is $R_1(\lambda) = \operatorname{diag}(1, e^{i\lambda})$ (controlled versions appear in the QFT). Match each angle (List-I) with the named gate (List-II).""",
         match=([r"$R_1(\pi)$", r"$R_1(\pi/2)$", r"$R_1(\pi/4)$", r"$R_1(-\pi/2)$"],
                [r"$S^\dagger$", r"$T$", r"$Z$", r"$S$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2"],
         sol=r"""$R_1(\pi) = \operatorname{diag}(1, -1) = Z$ (3); $R_1(\pi/2) = \operatorname{diag}(1, i) = S$ (4); $R_1(\pi/4) = \operatorname{diag}(1, e^{i\pi/4}) = T$ (2);
$R_1(-\pi/2) = \operatorname{diag}(1, -i) = S^\dagger$ (1). The inverse QFT of the notebook uses \texttt{cr1(-pi/2**(k-j))}: controlled $S^\dagger$, $T^\dagger$, \dots"""),

    dict(id="B17", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} When the target register is in an eigenvector of $U$, a controlled-$U$ gate leaves the target register unchanged.\\[1mm]
\textbf{Reason (R):} Controlled gates never change the state of their control qubit.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true}: $\ket c\ket u \mapsto \ket c\,e^{ic\phi}\ket u$ --- the target stays $\ket u$, only a phase appears. \textbf{R is false}: that phase belongs to the
control branch $\ket1$, so a control in superposition changes, $\ket+ \mapsto \frac{1}{\sqrt2}(\ket0 + e^{i\phi}\ket1)$ (phase kickback). This is precisely how
phase estimation and Shor's algorithm extract information. Answer: (C)."""),

    dict(id="B18", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} Because the circuit evaluates $a^x \bmod N$ for all $x$ in superposition, measuring the registers directly reveals the period $r$.\\[1mm]
\textbf{Statement II:} Interference produced by the inverse QFT is needed to turn the periodic structure into high-probability measurement outcomes.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false}: a measurement returns one random $x$ and one value $a^x \bmod N$ --- no more information than a single classical evaluation. Quantum
parallelism alone does not solve factoring. \textbf{II is true}: the inverse QFT makes the amplitudes interfere constructively near $y = k2^t/r$, so that one
measurement carries information about $r$. Answer: (D)."""),

    dict(id="B19", kind="para", passage="B",
         q=r"""For $\varphi = 3/8$ the measurement gives""",
         opts=[r"$y = 3$ (bits $011$) with probability $1$", r"$y = 3$ with probability $1/2$",
               r"$y = 6$ (bits $110$) with probability $1$", r"$y = 3$ or $y = 4$ with probability $1/2$ each"],
         sol=r"""$\varphi = 3/8 = 0.011_2$ is exactly representable with three bits, so all terms in the sum have phase $0$ for $y = 3$: $P(3) = 1$, and $P(y) = 0$ for every
other $y$ (the geometric sum of eighth roots of unity vanishes). $110$ is the bit-reversed string --- the inverse QFT's final SWAPs make the most significant
bit come first."""),

    dict(id="B20", kind="para", passage="B",
         q=r"""For $\varphi = 1/3$ the most likely outcome and its probability are""",
         opts=[r"$y = 3$ with $P \approx 0.69$", r"$y = 3$ with $P = 1$", r"$y = 3$ with $P \approx 0.41$", r"$y = 2$ or $3$ with $P = 1/2$ each"],
         sol=r"""$8\varphi = 2.67$, nearest integer $y = 3$, $\delta = \varphi - 3/8 = -\frac{1}{24}$. $P(3) = \frac{\sin^2(8\pi/24)}{64\sin^2(\pi/24)} = \frac{\sin^2(\pi/3)}{64\sin^2(7.5^\circ)}
= \frac{0.75}{64\times0.017037} = \frac{0.75}{1.0904} \approx 0.688$. Because $1/3$ has no finite binary expansion, the probability spreads over neighbouring $y$;
$\frac{4}{\pi^2} \approx 0.41$ is only the worst-case lower bound for the nearest outcome."""),
]
