"""Section C - Grover Basics (angle, iteration, oracle and diffusion circuits, optimal k, several solutions, over-rotation).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""Grover's algorithm runs on $n = 3$ qubits with the single marked item $x^\ast = \ket{101}$. It starts from the uniform state $\ket{s}$, in which every
amplitude is $a = 1/\sqrt8$. One iteration is the oracle $U_\omega$ followed by the diffusion $D$. Trace the amplitudes by hand with inversion about the mean.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""Grover's algorithm is run on $n = 2$ qubits ($N = 4$) with one marked item. After exactly one iteration, the probability of measuring the marked item is""",
         opts=[r"$1$", r"$\dfrac14$", r"$\dfrac34$", r"$\dfrac12$"],
         sol=r"""$\sin\theta = \frac{1}{\sqrt4} = \frac12$, so $\theta = 30^\circ$. After $k = 1$ iteration $P = \sin^2(3\theta) = \sin^2 90^\circ = 1$.
In amplitudes: after the oracle the entries are $(\frac12, \frac12, \frac12, -\frac12)$ with mean $\frac14$; inversion about the mean gives $(0, 0, 0, 1)$."""),

    dict(id="C2", kind="mcq",
         q=r"""For $N = 8$ with one marked item, the probability of measuring the marked item after one iteration is""",
         opts=[r"$\dfrac{25}{32}$", r"$\dfrac58$", r"$\dfrac{9}{16}$", r"$1$"],
         sol=r"""Write $a = 1/\sqrt8$. After the oracle: seven entries $a$, one entry $-a$. Mean $= \frac{6a}{8} = \frac34a$. The marked entry becomes $2\cdot\frac34a + a = \frac52a$, so
$P = \frac{25}{4}a^2 = \frac{25}{4}\cdot\frac18 = \frac{25}{32}$. \emph{Check:} $\sin^2 3\theta = s(3 - 4s)^2$ with $s = \frac18$ gives $\frac18\cdot\frac{25}{4} = \frac{25}{32}$."""),

    dict(id="C3", kind="mcq",
         q=r"""For $n = 4$ qubits with one marked item ($\theta \approx 0.2527$ rad), the optimal number of iterations $k_{\rm opt}$ and the resulting success probability are""",
         opts=[r"$k_{\rm opt} = 3$ and $P \approx 0.961$", r"$k_{\rm opt} = 2$ and $P \approx 0.908$",
               r"$k_{\rm opt} = 4$ and $P \approx 0.582$", r"$k_{\rm opt} = 3$ and $P = 1$"],
         sol=r"""$\frac{\pi}{4\theta} - \frac12 = \frac{0.7854}{0.2527} - 0.5 = 3.108 - 0.5 = 2.608$, which rounds to $k_{\rm opt} = 3$. Then
$P = \sin^2(7\theta) = \sin^2(1.769) \approx 0.961$. The angle $7\theta$ does not hit $\frac\pi2$ exactly, so $P < 1$.
For comparison, $k = 2$ gives $\sin^2 5\theta \approx 0.908$ and $k = 4$ gives $\sin^2 9\theta \approx 0.582$."""),

    dict(id="C4", kind="mcq",
         q=r"""For $n = 2$ ($N = 4$), the matrix of the diffusion operator $D = 2\ket{s}\!\bra{s} - I$ has""",
         opts=[r"diagonal entries $-\frac12$ and off-diagonal entries $+\frac12$",
               r"diagonal entries $+\frac12$ and off-diagonal entries $-\frac12$",
               r"all entries equal to $\frac12$",
               r"diagonal entries $-1$ and off-diagonal entries $+1$"],
         sol=r"""Every entry of $\ket{s}\!\bra{s}$ is $\frac{1}{\sqrt4}\cdot\frac{1}{\sqrt4} = \frac14$, so every entry of $2\ket{s}\!\bra{s}$ is $\frac12$. Subtracting $I$ changes only the diagonal: $\frac12 - 1 = -\frac12$. In general the diagonal is $\frac2N - 1$ and the off-diagonal is $\frac2N$.
\emph{Check:} acting on the amplitudes $(\frac12, \frac12, \frac12, -\frac12)$, the last row gives $\frac12(\frac12 + \frac12 + \frac12) - \frac12(-\frac12) = \frac34 + \frac14 = 1$, the same as inversion about the mean."""),

    dict(id="C5", kind="mcq",
         q=r"""An oracle for the marked string $\ket{0101}$ is built as: $X$ gates, then the multi-controlled $Z$ on all four qubits, then the same $X$ gates. The $X$ gates act on""",
         opts=[r"qubits 1 and 3", r"qubits 2 and 4", r"all four qubits", r"no qubit"],
         sol=r"""The multi-controlled $Z$ flips the sign of $\ket{1111}$ only, so the $X$ layer must map the marked string to $\ket{1111}$. That means flipping exactly the bits where
$x^\ast$ has a $0$. In $0101$ these are qubits 1 and 3. The second $X$ layer undoes the first, so the sign flip ends up on $\ket{0101}$."""),

    dict(id="C6", kind="mcq",
         q=r"""Use the circuit in the conventions box and count each multi-controlled $Z$ as one gate. The number of gates in \emph{one} Grover iteration for $n = 6$, $x^\ast = 010011$ is""",
         opts=[r"$32$", r"$26$", r"$30$", r"$38$"],
         sol=r"""The oracle has $2z$ $X$ gates, where $z$ is the number of $0$s in $x^\ast$ (here $z = 3$), and one multi-controlled $Z$: $6 + 1 = 7$ gates.
The diffusion has $2n$ $H$, $2n$ $X$ and one multi-controlled $Z$: $12 + 12 + 1 = 25$ gates. Total $= 4n + 2z + 2 = 24 + 6 + 2 = 32$."""),

    dict(id="C7", kind="mcq",
         q=r"""Let $C$ be the multi-controlled $Z$ on $n$ qubits. The circuit $H^{\otimes n}X^{\otimes n}\,C\,X^{\otimes n}H^{\otimes n}$ implements""",
         opts=[r"$I - 2\ket{s}\!\bra{s} = -D$", r"$2\ket{s}\!\bra{s} - I = D$", r"$I - 2\ket{0\dots0}\!\bra{0\dots0}$", r"$I - 2\ket{1\dots1}\!\bra{1\dots1}$"],
         sol=r"""$C = I - 2\ket{1\dots1}\!\bra{1\dots1}$. Conjugating by $X^{\otimes n}$ gives $I - 2\ket{0\dots0}\!\bra{0\dots0}$, and conjugating by $H^{\otimes n}$ gives $I - 2\ket{s}\!\bra{s}$, since $H^{\otimes n}\ket{0\dots0} = \ket{s}$.
That is $-D$: the circuit equals the diffusion up to the global phase $-1$, which has no observable effect."""),

    dict(id="C8", kind="mcq",
         q=r"""For $N = 16$ with one marked item, the amplitude of each \emph{unmarked} item after one iteration is""",
         opts=[r"$\dfrac{3}{16}$", r"$\dfrac{11}{16}$", r"$\dfrac14$", r"$\dfrac{7}{32}$"],
         sol=r"""Start with amplitudes $\frac14$. After the oracle the mean is $\frac{15 - 1}{16}\cdot\frac14 = \frac{7}{32}$. An unmarked entry becomes $2\cdot\frac{7}{32} - \frac14 = \frac{7}{16} - \frac{4}{16} = \frac{3}{16}$,
and the marked entry becomes $\frac{7}{16} + \frac{4}{16} = \frac{11}{16}$. \emph{Check:} $15\cdot\frac{9}{256} + \frac{121}{256} = 1$."""),

    dict(id="C9", kind="mcq",
         q=r"""Let $N = 16$ with $M = 4$ marked items. After one iteration the probability of measuring a marked item is""",
         opts=[r"$1$", r"$\dfrac14$", r"$\dfrac34$", r"$\dfrac{25}{32}$"],
         sol=r"""$\sin^2\theta = \frac{M}{N} = \frac14$, so $\theta = 30^\circ$ and $P_1 = \sin^2 3\theta = \sin^2 90^\circ = 1$. With $M = N/4$ a single iteration succeeds with certainty, for every $N$ for which $M = N/4$ is an integer.
The final state is the uniform superposition of the four marked items."""),

    dict(id="C10", kind="mcq",
         q=r"""Let $N = 8$ with $M = 4$ marked items. The probability of a marked outcome after $k$ iterations is""",
         opts=[r"$\frac12$ for every $k$", r"$1$ for $k = 1$", r"$0$ for $k = 1$", r"$1$ for even $k$ and $0$ for odd $k$"],
         sol=r"""$\sin^2\theta = \frac12$ gives $\theta = 45^\circ$, and $P_k = \sin^2\big((2k+1)45^\circ\big)$. The angles $45^\circ, 135^\circ, 225^\circ, \dots$ all have $\sin^2 = \frac12$.
The iteration rotates by $2\theta = 90^\circ$, which exchanges the two diagonals and never brings the state closer to the marked axis. Grover gives no gain at $M = N/2$."""),

    dict(id="C11", kind="mcq",
         q=r"""Let $N = 8$ with $M = 6$ marked items. The probability of a marked outcome after one iteration is""",
         opts=[r"$0$", r"$\dfrac34$", r"$1$", r"$\dfrac14$"],
         sol=r"""$\sin^2\theta = \frac34$ gives $\theta = 60^\circ$, so $P_1 = \sin^2 180^\circ = 0$. Amplitudes confirm this: with $a = 1/\sqrt8$ the oracle gives six entries $-a$ and two entries $+a$; the mean is $-\frac12a$;
the marked entries become $-a + a = 0$. Measuring \emph{without} any iteration ($P_0 = \frac34$) is far better."""),

    dict(id="C12", kind="mcq",
         q=r"""On $2^{15}$ items with one marked item, $k_{\rm opt} = 142$. If $k = 284$ iterations are used instead, the probability of the marked item is about""",
         opts=[r"$3\times10^{-6}$", r"$1$", r"$\dfrac12$", r"$\dfrac{1}{32768}$"],
         sol=r"""The angle is $(2k+1)\theta = 569\times0.0055243 = 3.1433$ rad, which is $0.0017$ past $\pi$. So $P = \sin^2(569\theta) \approx (0.0017)^2 \approx 3\times10^{-6}$.
This is over-rotation: twice the optimal number of iterations brings the state back near the unmarked axis, and the marked item is then \emph{less} likely than a random guess ($1/32768 \approx 3\times10^{-5}$)."""),

    dict(id="C13", kind="mcq",
         q=r"""For $n = 20$ qubits with one marked item, Grover needs $804$ iterations. A classical random search needs on average about $2^{19}$ queries. The classical cost divided by the quantum cost is about""",
         opts=[r"$650$", r"$1000$", r"$2$", r"$20$"],
         sol=r"""$2^{19} = 524\,288$ and $524\,288/804 \approx 652$. In general the ratio is $\frac{N/2}{(\pi/4)\sqrt N} = \frac{2}{\pi}\sqrt N$, and here $\frac2\pi\times1024 \approx 652$. The speed-up is quadratic, not exponential."""),

    dict(id="C14", kind="mcq",
         q=r"""A Grover circuit with success probability $P = 0.9613$ is run for $2000$ shots. The expected marked count and its binomial standard deviation are about""",
         opts=[r"$1923 \pm 9$", r"$1923 \pm 44$", r"$2000 \pm 0$", r"$1923 \pm 77$"],
         sol=r"""Mean $= SP = 2000\times0.9613 \approx 1923$. Standard deviation $= \sqrt{SP(1-P)} = \sqrt{2000\times0.9613\times0.0387} = \sqrt{74.4} \approx 8.6$.
$\pm44 = \sqrt{1923}$ is the Poisson estimate, which is wrong here because $P$ is close to $1$."""),

    dict(id="C15", kind="match",
         q=r"""Match the number of qubits (one marked item) in List-I with $k_{\rm opt}$ in List-II. Use $k_{\rm opt} = \operatorname{round}\big(\frac{\pi}{4\theta} - \frac12\big)$.""",
         match=([r"$n = 2$", r"$n = 6$", r"$n = 8$", r"$n = 10$"], [r"$12$", r"$25$", r"$1$", r"$6$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""$n = 2$: $\theta = \frac\pi6$, $\frac{\pi}{4\theta} - \frac12 = 1$ (3). $\;n = 6$: $\theta \approx 0.1253$, giving $5.77 \to 6$ (4). $\;n = 8$: $\theta \approx 0.0625$, giving $12.06 \to 12$ (1).
$\;n = 10$: $\theta \approx 0.03126$, giving $24.63 \to 25$ (2). Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="C16", kind="match",
         q=r"""Match each part of the Grover circuit in List-I with its effect in List-II.""",
         match=([r"$H^{\otimes n}$ on $\ket{0\dots0}$", r"$X$ gates on the $0$ bits of $x^\ast$, multi-controlled $Z$, the same $X$ gates",
                 r"$H^{\otimes n}X^{\otimes n}$, multi-controlled $Z$, $X^{\otimes n}H^{\otimes n}$", r"measuring all qubits"],
                [r"returns one bit string with its Born probability", r"creates the uniform superposition $\ket{s}$",
                 r"reflects about $\ket{s}$, up to a global phase", r"flips the sign of $\ket{x^\ast}$ only"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1",
               r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3"],
         sol=r"""P: $H^{\otimes n}\ket{0\dots0} = \ket{s}$ (2). Q: the oracle recipe of C5 flips only $\ket{x^\ast}$ (4). R: this is the diffusion circuit of C7 (3). S: the measurement samples one string from $|c_x|^2$ (1).
Hence P$\to$2, Q$\to$4, R$\to$3, S$\to$1."""),

    dict(id="C17", kind="ar", correct=1, opts=AR_OPTIONS,
         q=r"""\textbf{Assertion (A):} The success probability of Grover's algorithm does not depend on which single item is marked.\\[1mm]
\textbf{Reason (R):} The oracle is a diagonal matrix in the computational basis.""",
         sol=r"""\textbf{A is true:} only $\braket{x^\ast|s} = 1/\sqrt N$ enters the analysis, and this is the same for every $x^\ast$.
\textbf{R is true:} $U_\omega = \operatorname{diag}\big((-1)^{f(x)}\big)$. However, being diagonal does not explain A: an arbitrary diagonal gate need not treat all items alike. The explanation is the symmetry of $\ket{s}$, which has the same overlap with every basis state. Answer: (B)."""),

    dict(id="C18", kind="stmt", correct=3, opts=STATEMENT_OPTIONS,
         q=r"""\textbf{Statement I:} In Grover's algorithm, applying more iterations never lowers the success probability.\\[1mm]
\textbf{Statement II:} For one marked item, the optimal number of iterations grows as $\sqrt N$.""",
         sol=r"""\textbf{I is false:} $P_k = \sin^2((2k+1)\theta)$ oscillates. For $N = 4$ we have $P_1 = 1$ but $P_2 = \sin^2 150^\circ = \frac14$.
\textbf{II is true:} $k_{\rm opt} \approx \frac{\pi}{4\theta} \approx \frac\pi4\sqrt N$. Answer: (D)."""),

    dict(id="C19", kind="para", passage="C", opts=[
             r"marked $\frac52a$; each unmarked $\frac12a$", r"marked $\frac74a$; each unmarked $\frac34a$",
             r"marked $\frac52a$; each unmarked $\frac34a$", r"marked $\frac32a$; each unmarked $\frac12a$"],
         q=r"""After the first iteration the amplitudes are""",
         sol=r"""After $U_\omega$: seven entries $a$ and one entry $-a$; the mean is $\frac{7a - a}{8} = \frac34a$. Inversion about the mean gives: unmarked $2\cdot\frac34a - a = \frac12a$; marked $2\cdot\frac34a + a = \frac52a$.
\emph{Check:} $(7\cdot\frac14 + \frac{25}{4})a^2 = 8a^2 = 1$."""),

    dict(id="C20", kind="para", passage="C", opts=[r"$\dfrac{121}{128}$", r"$\dfrac{25}{32}$", r"$1$", r"$\dfrac{11}{16}$"],
         q=r"""After the second iteration the probability of measuring $\ket{101}$ is""",
         sol=r"""Start from the result of C19: unmarked $\frac12a$, marked $\frac52a$. After $U_\omega$ the marked entry is $-\frac52a$, and the mean is $\frac{7\cdot\frac12a - \frac52a}{8} = \frac{a}{8}$.
The marked entry becomes $2\cdot\frac a8 + \frac52a = \frac{11}{4}a$, so $P = \frac{121}{16}a^2 = \frac{121}{16\cdot8} = \frac{121}{128}$. (Each unmarked entry becomes $\frac a4 - \frac a2 = -\frac a4$.)
\emph{Check:} $P_2 = \sin^2 5\theta \approx 0.945$ with $\sin^2\theta = \frac18$. A third iteration would over-rotate."""),
]
