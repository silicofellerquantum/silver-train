"""Section C - Shor's Algorithm Fundamentals (reduction of factoring to order finding, choice of a, the order-finding
circuit, register sizes, measurement outcomes, continued fractions, success and failure cases, repetition,
classical post-processing and the functions of Shor.ipynb).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""Shor's algorithm is applied to $N = 39$ with base $a = 2$, using the register sizes of \texttt{Shor.ipynb}: the work register has
$n = $ \texttt{N.bit\_length()} $= 6$ qubits and the counting register $t = 2n = 12$ qubits, so the peaks of the outcome distribution lie near
$y \approx k\cdot4096/r$. A measured value $y$ is post-processed with \texttt{Fraction(y, 2**t).limit\_denominator(N)}; the denominator $r'$ of the
result, or the smallest multiple $kr'$ with $a^{kr'} \equiv 1 \pmod N$, is taken as the order. The powers of $2$ modulo $39$ are
$2, 4, 8, 16, 32, 25, 11, 22, 5, 10, 20, 1$.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""Which sequence describes Shor's factoring algorithm for an odd composite $N$ that is not a prime power?""",
         opts=[r"pick random $a$; if $\gcd(a, N) > 1$ stop (factor found); find the order $r$ of $a$ by quantum phase estimation and continued fractions; if $r$ is even and $a^{r/2} \not\equiv -1$, output $\gcd(a^{r/2} \pm 1, N)$; otherwise repeat",
               r"apply the QFT to $\ket N$; measure; the outcome is a factor of $N$",
               r"find the order $r$ by quantum phase estimation; the factors are $r$ and $N/r$",
               r"pick random $a$; measure $a^x \bmod N$ in superposition; the most frequent value is a factor"],
         sol=r"""The quantum computer is used only for \emph{order finding}; everything else --- the $\gcd$ test, continued fractions, the checks on $r$ and the
final $\gcd(a^{r/2} \pm 1, N)$ --- is classical. Option 3 confuses the order with a factor (for $N = 15$, $a = 7$: $r = 4$ does not divide $15$)."""),

    dict(id="C2", kind="mcq",
         q=r"""For $N = 15$ and $a = 2$, the order $r$ and the factors obtained from $\gcd(a^{r/2} \pm 1, N)$ are""",
         opts=[r"$r = 4$; $\ \gcd(3, 15) = 3$, $\gcd(5, 15) = 5$", r"$r = 2$; $\ 3$ and $5$", r"$r = 4$; $\ 1$ and $15$", r"$r = 8$; $\ 3$ and $5$"],
         sol=r"""$2^1, 2^2, 2^3, 2^4 \equiv 2, 4, 8, 1$, so $r = 4$. $2^{r/2} = 4$, $4 \not\equiv -1 \equiv 14$. $\gcd(4 - 1, 15) = 3$ and $\gcd(4 + 1, 15) = 5$, and $3\times5 = 15$.
$r = 8$ is a multiple of the order, not the order (the smallest positive exponent)."""),

    dict(id="C3", kind="mcq",
         q=r"""For $N = 15$ and $a = 14$, the order finding succeeds perfectly. The classical post-processing then""",
         opts=[r"fails: $r = 2$ but $14^{1} \equiv -1 \pmod{15}$, so $\gcd(14 \pm 1, 15)$ gives only the trivial factors $15$ and $1$; another $a$ is needed",
               r"succeeds with factors $3$ and $5$",
               r"fails because $r$ is odd",
               r"fails because $\gcd(14, 15) = 1$"],
         sol=r"""$14^2 = 196 = 13\cdot15 + 1$, so $r = 2$ (even). But $a^{r/2} = 14 \equiv -1$: $\gcd(13, 15) = 1$ and $\gcd(15, 15) = 15$ --- trivial. For any $N$, $a = N - 1$ is always such a
failing base. $\gcd(a, N) = 1$ is required, not a reason for failure."""),

    dict(id="C4", kind="mcq",
         q=r"""For $N = 21$ and $a = 4$, the outcome of the classical checks is""",
         opts=[r"$r = 3$ is odd, so $a^{r/2}$ is not an integer power; the attempt fails and a new $a$ is chosen",
               r"$r = 6$, factors $3$ and $7$",
               r"$r = 3$, factors $\gcd(4^1 - 1, 21) = 3$ and $\gcd(4^1 + 1, 21) = 1$",
               r"$\gcd(4, 21) = 7$, a lucky factor"],
         sol=r"""$4^1, 4^2, 4^3 \equiv 4, 16, 64 = 3\cdot21 + 1$, so $r = 3$, odd. The reduction needs $r$ even (then $(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0$). A failed attempt must be
reported as a failure, not as a success; one simply repeats with another base ($a = 2$ gives $r = 6$ and the factors $3$, $7$)."""),

    dict(id="C5", kind="mcq",
         q=r"""For $N = 21$ the randomly chosen base is $a = 6$. What happens?""",
         opts=[r"$\gcd(6, 21) = 3$ is already a non-trivial factor; no quantum computation is needed",
               r"order finding gives $r = 6$ and the factors $3$, $7$",
               r"the algorithm fails because $a$ is even",
               r"$U_6$ is a valid unitary and phase estimation returns $r = 2$"],
         sol=r"""The first classical step is $\gcd(a, N)$. Here $\gcd(6, 21) = 3$, so $21 = 3\times7$ immediately (a ``lucky'' base). Moreover $U_6$ would not be unitary
($6\cdot0 \equiv 6\cdot7 \equiv 0 \pmod{21}$), and $6$ has no order modulo $21$. Even bases are fine in general (e.g.\ $a = 2$)."""),

    dict(id="C6", kind="mcq",
         q=r"""With the register sizes of \texttt{Shor.ipynb} ($n = $ \texttt{N.bit\_length()}, $t = 2n$), the circuit for $N = 33$ uses""",
         opts=[r"$18$ qubits ($n = 6$, $t = 12$)", r"$12$ qubits", r"$15$ qubits ($n = 5$, $t = 10$)", r"$33$ qubits"],
         sol=r"""$33 = 100001_2$ has $6$ bits, so $n = 6$, $t = 12$ and $t + n = 18$. Five bits only reach $31 < 33$, so $n = 5$ is too small. The count is $3n$ in this construction
--- far from the $N$ qubits of option 4."""),

    dict(id="C7", kind="mcq",
         q=r"""For $N = 15$, $a = 11$ and $t = 8$ counting qubits, the ideal outcome distribution and the result of post-processing $y = 128$ are""",
         opts=[r"peaks at $y \in \{0, 128\}$ with probability $1/2$ each; $128/256 = 1/2 \Rightarrow r = 2$, $\gcd(10, 15) = 5$, $\gcd(12, 15) = 3$",
               r"peaks at $y \in \{0, 64, 128, 192\}$; $r = 4$",
               r"peaks at $y \in \{0, 128\}$; $r = 2$, but $11 \equiv -1$, so the factors are trivial",
               r"a uniform distribution, because $11$ is prime"],
         sol=r"""$11^2 = 121 = 8\cdot15 + 1$, so $r = 2$ and the peaks are at $k\cdot256/2 = 0, 128$, each with probability $\frac12$ (exact, since $r$ divides $2^t$). $y = 128$
gives $s/r = 1/2$, $r = 2$; $11 \not\equiv -1 \equiv 14$, and $\gcd(10, 15) = 5$, $\gcd(12, 15) = 3$. $y = 0$ gives no information."""),

    dict(id="C8", kind="mcq",
         q=r"""For $N = 33$, $a = 5$ and $t = 12$, the measurement gives $y = 1229$. The continued-fraction step and the factors are""",
         opts=[r"$1229/4096 \approx 3/10 \Rightarrow r = 10$; $5^5 \equiv 23$, $\gcd(22, 33) = 11$, $\gcd(24, 33) = 3$",
               r"$1229/4096 \approx 3/10 \Rightarrow r = 3$; factors $3$ and $11$",
               r"$1229/4096 \approx 1/3 \Rightarrow r = 3$; the attempt fails",
               r"$1229/4096 \approx 3/10 \Rightarrow r = 10$; $5^5 \equiv -1$, trivial factors"],
         sol=r"""$1229/4096 = 0.30005$; \texttt{limit\_denominator(33)} gives $3/10$ (within $\frac{1}{2\cdot4096}$). $5^{10} \equiv 1 \pmod{33}$ and no smaller power is $1$, so $r = 10$.
$5^5 = 3125 = 94\cdot33 + 23$; $23 \ne 32$; $\gcd(22, 33) = 11$, $\gcd(24, 33) = 3$. The numerator $3$ is $s$, not the order."""),

    dict(id="C9", kind="mcq",
         q=r"""For $N = 33$, $a = 5$, $t = 12$, the outcome is $y = 2048$. The procedure of \texttt{candidate\_orders} returns""",
         opts=[r"the fraction $1/2$ and, after testing multiples of $2$, the order $10$ (since $5^2, 5^4, 5^6, 5^8 \not\equiv 1$, $5^{10} \equiv 1$)",
               r"the order $2$, because the denominator is always the order",
               r"no candidate, because $y = 2^{t-1}$ is never useful",
               r"the order $5$"],
         sol=r"""$2048/4096 = 1/2$: the measured $s/r = 5/10$ has $\gcd(s, r) = 5$, so it reduces to $1/2$. Testing $2k$: $5^2 \equiv 25$, $5^4 \equiv 31$, $5^6 \equiv 16$, $5^8 \equiv 4$,
$5^{10} \equiv 1$ --- the first multiple that works is $10$. Without the multiples test this outcome would be lost."""),

    dict(id="C10", kind="mcq",
         q=r"""The textbook condition $2^t \ge N^2$ guarantees that continued fractions recover $s/r$ from the nearest outcome. For $N = 33$, the smallest such
$t$ and the value used by \texttt{Shor.ipynb} are""",
         opts=[r"$t = 11$; the notebook uses $t = 2n = 12$", r"$t = 10$; the notebook uses $12$", r"$t = 12$; the notebook uses $11$", r"$t = 6$; the notebook uses $12$"],
         sol=r"""$N^2 = 1089$: $2^{10} = 1024 < 1089 \le 2^{11} = 2048$, so $t = 11$. The notebook uses $t = 2\cdot$\texttt{bit\_length} $= 12$, which always satisfies $2^{2n} \ge N^2$
because $N < 2^n$. Then two distinct fractions with denominators $\le N$ differ by more than $1/N^2 \ge 1/2^t$."""),

    dict(id="C11", kind="mcq",
         q=r"""For the notebook's default ($N = 15$, $a = 7$, $t = 8$) the ideal outcomes are $y \in \{0, 64, 128, 192\}$, each with probability $1/4$. Using the
multiples test, the exact probability that one shot yields the factors is""",
         opts=[r"$3/4$ --- only $y = 0$ fails", r"$1/2$ --- only $y = 64$ and $192$ succeed", r"$1$", r"$1/4$ --- only $y = 64$ succeeds"],
         sol=r"""$y = 64$, $192$ give $1/4$, $3/4$, i.e.\ $r = 4$ directly; $y = 128$ gives $1/2$, and the multiples test finds $4$; $y = 0$ gives no denominator. $7^2 = 49 \equiv 4$,
$\gcd(3, 15) = 3$, $\gcd(5, 15) = 5$. Hence $3/4$; the recorded run gave $0.766$ (finite-shot fluctuation around $0.75$)."""),

    dict(id="C12", kind="mcq",
         q=r"""Suppose $r = 12$ divides $2^t$, so every outcome is exactly $y = s2^t/r$ with $s$ uniform in $\{0, \dots, 11\}$. \emph{Without} the multiples test,
the fraction of shots whose reduced denominator equals $r$ is""",
         opts=[r"$\varphi(12)/12 = 1/3$", r"$1/12$", r"$1/2$", r"$11/12$"],
         sol=r"""The reduced fraction $s/12$ has denominator $12$ exactly when $\gcd(s, 12) = 1$: $s \in \{1, 5, 7, 11\}$, i.e.\ $\varphi(12) = 4$ of $12$ values. In general the
probability is $\varphi(r)/r = \Omega(1/\log\log r)$, so $O(\log\log N)$ repetitions (or the multiples/lcm trick) suffice."""),

    dict(id="C13", kind="mcq",
         q=r"""What is the role of the (inverse) quantum Fourier transform \emph{within} Shor's algorithm?""",
         opts=[r"it is a subroutine of phase estimation that turns the periodic phase pattern on the counting register into peaks near $y = k2^t/r$; it does not factor $N$ by itself",
               r"it computes $a^x \bmod N$",
               r"it is the whole of Shor's algorithm; the remaining steps are optional",
               r"it finds the factors $p$ and $q$ directly from $\ket N$"],
         sol=r"""Shor's algorithm $=$ classical reduction $+$ order finding (modular exponentiation, QFT, measurement) $+$ classical post-processing. The QFT is one
component; the dominant cost of real implementations is the modular exponentiation, not the QFT ($O(n^2)$ gates, or fewer when approximated)."""),

    dict(id="C14", kind="mcq",
         q=r"""\texttt{build\_shor\_kernel} computes $c_j = a^{2^j} \bmod N$ classically and skips $c_j = 1$. For $N = 15$, $a = 7$, $t = 8$, the number of controlled
multiplications actually applied is""",
         opts=[r"$2$ ($c_0 = 7$, $c_1 = 4$; $c_j = 1$ for $j \ge 2$)", r"$8$", r"$4$", r"$1$"],
         sol=r"""$7^1 = 7$, $7^2 = 49 \equiv 4$, $7^4 \equiv 16 \equiv 1$, and then $1$ forever. Only two multipliers remain --- matching the recorded gate counts
(\texttt{ccustom\_unitary\_constant}: 2; operations \texttt{mulmod\_7\_mod\_15}, \texttt{mulmod\_4\_mod\_15}). This shortcut uses the known order and is
specific to the tiny example."""),

    dict(id="C15", kind="match",
         q=r"""Match each function of \texttt{Shor.ipynb} (List-I) with what it does (List-II).""",
         match=([r"\texttt{mulmod\_matrix}", r"\texttt{build\_shor\_kernel}", r"\texttt{candidate\_orders}", r"\texttt{classical\_order}"],
                [r"brute-force order of $a$ modulo $m$ as a classical reference (feasible only for tiny $m$)",
                 r"continued-fraction denominator of $y/2^t$ and the smallest multiple $k r'$ with $a^{kr'} \equiv 1$",
                 r"builds the CUDA-Q circuit: Hadamards, controlled $U_{a^{2^j}}$, inverse QFT, measurement",
                 r"permutation matrix $\ket y \mapsto \ket{cy \bmod N}$ for $y < N$, identity above"]),
         opts=[r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1",
               r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1"],
         sol=r"""\texttt{mulmod\_matrix} builds the unitary (4); \texttt{build\_shor\_kernel} assembles the circuit with \texttt{cudaq.register\_operation} gates (3);
\texttt{candidate\_orders} is the classical post-processing (2); \texttt{classical\_order} is the brute-force check (1). Hence P$\to$4, Q$\to$3, R$\to$2, S$\to$1."""),

    dict(id="C16", kind="match",
         q=r"""For $N = 33$, $a = 5$ ($r = 10$) and $t = 12$, match each outcome $y$ (List-I) with \texttt{Fraction(y, 4096).limit\_denominator(33)} (List-II).""",
         match=([r"$410$", r"$819$", r"$1638$", r"$2867$"],
                [r"$2/5$", r"$7/10$", r"$1/10$", r"$1/5$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""$410/4096 = 0.1001 \approx 1/10$; $819/4096 = 0.19995 \approx 1/5$; $1638/4096 = 0.3999 \approx 2/5$; $2867/4096 = 0.69995 \approx 7/10$. Only $410$ and $2867$ give the
denominator $10$ directly ($s = 1, 7$ coprime to $10$); $819$ and $1638$ give $5$, and the multiples test then finds $10$."""),

    dict(id="C17", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} Shor's order-finding reduction is the method of choice for factoring $N = 81 = 3^4$.\\[1mm]
\textbf{Reason (R):} For an odd prime power $N = p^k$, the only square roots of $1$ modulo $N$ are $\pm1$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is false}: prime powers are detected classically (integer $k$-th roots) --- \texttt{Shor.ipynb} does so with \texttt{prime\_power} before any quantum step.
\textbf{R is true}: $\mathbb Z_{p^k}^*$ is cyclic, so $x^2 \equiv 1$ has only the solutions $\pm1$; hence $a^{r/2}$ would always be $\pm1$ and $\gcd(a^{r/2} \pm 1, N)$ would
be trivial. Answer: (D)."""),

    dict(id="C18", kind="stmt", correct=1,
         q=r"""\textbf{Statement I:} A measured value $y = 0$ means that the order is $r = 1$.\\[1mm]
\textbf{Statement II:} When $r$ does not divide $2^t$, every measured $y$ yields the exact order through continued fractions.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false}: $y = 0$ is the peak $s = 0$, present for every $r$; it carries no information (the notebook returns no candidate). \textbf{II is false}: only
outcomes close to $s2^t/r$ with $\gcd(s, r) = 1$ give $r$ directly; others give a divisor of $r$ or nothing, and the probability mass is spread over neighbouring $y$.
Answer: (B)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The order $r$ and the factors $\gcd(2^{r/2} - 1, 39)$, $\gcd(2^{r/2} + 1, 39)$ are""",
         opts=[r"$r = 12$; $\ 3$ and $13$", r"$r = 6$; $\ 3$ and $13$", r"$r = 12$; $\ 1$ and $39$", r"$r = 12$; $\ 7$ and $13$"],
         sol=r"""From the list, $2^{12} \equiv 1$ is the first $1$: $r = 12$, even. $2^6 \equiv 25 \not\equiv 38$. $\gcd(24, 39) = 3$ and $\gcd(26, 39) = 13$; $3\times13 = 39$.
$r = 6$ would require $2^6 \equiv 1$, but $2^6 \equiv 25$."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""The outcome $y = 1707$ is measured. The continued-fraction step gives""",
         opts=[r"$5/12$, so $r = 12$ directly", r"$5/12$, so $r = 5$", r"$1/2$, so the multiples test is needed", r"$2/5$, so $r = 5$"],
         sol=r"""$1707/4096 = 0.41675$, and $5/12 = 0.41667$; $|1707/4096 - 5/12| = 8.1\times10^{-5} < 1/(2\cdot39^2)$, so \texttt{limit\_denominator(39)} returns $5/12$.
$\gcd(5, 12) = 1$, so the denominator is the order $12$. The numerator $5$ is the random index $s$, not the order."""),
]
