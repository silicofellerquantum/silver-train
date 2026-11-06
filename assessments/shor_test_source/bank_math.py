"""Section A - Mathematical and Number Theory Foundations (primes and composites, gcd and the Euclidean algorithm,
modular inverses, Euler's totient and theorem, multiplicative order, modular exponentiation, the Chinese remainder
theorem, square roots of 1, classical factoring cost).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""Shor's reduction is applied by hand to $N = 91 = 7\times13$ with the base $a = 3$ (note $\gcd(3, 91) = 1$). Recall that the
order $r$ of $a$ modulo $N$ is the least $r > 0$ with $a^r \equiv 1 \pmod N$, and that if $r$ is even and $a^{r/2} \not\equiv -1 \pmod N$,
then $\gcd(a^{r/2} \pm 1, N)$ are non-trivial factors.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""Using the Euclidean algorithm, $\gcd(1071, 462)$ equals""",
         opts=[r"$21$", r"$7$", r"$42$", r"$147$"],
         sol=r"""$1071 = 2\cdot462 + 147$; $\ 462 = 3\cdot147 + 21$; $\ 147 = 7\cdot21 + 0$. The last non-zero remainder is $21$.
\emph{Check}: $1071 = 3^2\cdot7\cdot17$ and $462 = 2\cdot3\cdot7\cdot11$, common part $3\cdot7 = 21$. The algorithm needs $O(\log N)$ division steps,
which is why the gcd steps of Shor's algorithm are classically cheap."""),

    dict(id="A2", kind="mcq",
         q=r"""The multiplicative inverse of $3$ modulo $20$ (the $b$ with $3b \equiv 1 \pmod{20}$, $0 < b < 20$) is""",
         opts=[r"$7$", r"$13$", r"$17$", r"$3$"],
         sol=r"""Extended Euclid: $20 = 6\cdot3 + 2$, $3 = 1\cdot2 + 1$, so $1 = 3 - 2 = 3 - (20 - 6\cdot3) = 7\cdot3 - 20$. Hence $3\cdot7 = 21 \equiv 1 \pmod{20}$,
$b = 7$. The inverse exists because $\gcd(3, 20) = 1$; the same condition $\gcd(a, N) = 1$ makes multiplication by $a$ modulo $N$ reversible."""),

    dict(id="A3", kind="mcq",
         q=r"""Euler's totient $\varphi(91)$ equals""",
         opts=[r"$72$", r"$90$", r"$84$", r"$78$"],
         sol=r"""$91 = 7\cdot13$ with distinct primes, so $\varphi(91) = (7-1)(13-1) = 6\cdot12 = 72$: the number of $1 \le a \le 91$ coprime to $91$.
$90 = N - 1$ would hold only for a prime $N$."""),

    dict(id="A4", kind="mcq",
         q=r"""The multiplicative order of $2$ modulo $11$ is""",
         opts=[r"$10$", r"$5$", r"$2$", r"$11$"],
         sol=r"""Powers of 2 mod 11: $2, 4, 8, 5, 10, 9, 7, 3, 6, 1$ --- the first 1 appears at exponent $10$. The order divides $\varphi(11) = 10$, so the only
candidates are $1, 2, 5, 10$: $2^2 = 4$, $2^5 = 32 \equiv 10 \equiv -1$, so the order is $10$ ($2$ is a primitive root mod 11)."""),

    dict(id="A5", kind="mcq",
         q=r"""Using repeated squaring, $3^{13} \bmod 17$ equals""",
         opts=[r"$12$", r"$3$", r"$5$", r"$16$"],
         sol=r"""$3^1 = 3$, $3^2 = 9$, $3^4 = 81 \equiv 13$, $3^8 = 169 \equiv 16 \pmod{17}$. With $13 = 8 + 4 + 1$:
$3^{13} \equiv 16\cdot13\cdot3 = 624 = 36\cdot17 + 12 \equiv 12$. \emph{Check}: $3^{16} \equiv 1$ (Fermat), so $3^{13} \equiv 3^{-3} = 27^{-1} \equiv 10^{-1} \equiv 12$
since $10\cdot12 = 120 = 7\cdot17 + 1$."""),

    dict(id="A6", kind="mcq",
         q=r"""Square-and-multiply computes $a^{1000} \bmod N$ from the binary expansion $1000 = 1111101000_2$. The number of modular multiplications
(squarings plus extra multiplications) is""",
         opts=[r"$14$", r"$999$", r"$10$", r"$16$"],
         sol=r"""Left-to-right: start with $a$ (leading bit), then for each of the remaining $9$ bits square once ($9$ squarings) and multiply by $a$ when the bit is
1. The bits after the leading one are $111101000$: five 1s, so $5$ extra multiplications. Total $9 + 5 = 14$. Naively $999$ multiplications would be needed.
This is the classical idea behind computing $a^{2^j} \bmod N$ in Shor's circuit."""),

    dict(id="A7", kind="mcq",
         q=r"""For $N = 35$, which of the following can \emph{not} be the multiplicative order of any $a$ with $\gcd(a, 35) = 1$?""",
         opts=[r"$5$", r"$12$", r"$4$", r"$6$"],
         sol=r"""By the Chinese remainder theorem the order modulo $35$ is $\operatorname{lcm}(r_5, r_7)$ with $r_5 \mid 4$ and $r_7 \mid 6$, so every order divides the
Carmichael value $\lambda(35) = \operatorname{lcm}(4, 6) = 12$ (and $\varphi(35) = 24$). Orders $12$ (e.g.\ $a = 2$), $4$ (e.g.\ $a = 8$: $8^2 = 64 \equiv 29$,
$8^4 \equiv 1$) and $6$ (e.g.\ $a = 4$: $4^3 = 64 \equiv 29$, $4^6 \equiv 1$) occur; $5$ divides neither $12$ nor $24$."""),

    dict(id="A8", kind="mcq",
         q=r"""For $N = 15$, a base $a$ is drawn uniformly from $\{2, 3, \dots, 13\}$. The probability that $\gcd(a, 15) > 1$, so that a factor is found classically
without any quantum step, is""",
         opts=[r"$1/2$", r"$7/15$", r"$8/15$", r"$1/3$"],
         sol=r"""The multiples of 3 or 5 in $\{2, \dots, 13\}$ are $3, 5, 6, 9, 10, 12$: six of the twelve values, probability $\frac12$. (For RSA-sized $N = pq$ this
probability is about $\frac1p + \frac1q$, astronomically small --- the lucky gcd is only an artefact of tiny examples.)"""),

    dict(id="A9", kind="mcq",
         q=r"""The unique $x \in \{0, \dots, 14\}$ with $x \equiv 2 \pmod 3$ and $x \equiv 3 \pmod 5$ is""",
         opts=[r"$8$", r"$13$", r"$2$", r"$11$"],
         sol=r"""Candidates with $x \equiv 3 \pmod 5$: $3, 8, 13$; their residues mod 3 are $0, 2, 1$. So $x = 8$. The Chinese remainder theorem guarantees one solution
modulo $15$; it is the reason why orders modulo $N = pq$ are least common multiples of the orders modulo $p$ and $q$."""),

    dict(id="A10", kind="mcq",
         q=r"""The complete set of solutions of $x^2 \equiv 1 \pmod{15}$ with $0 \le x < 15$ is""",
         opts=[r"$\{1, 4, 11, 14\}$", r"$\{1, 14\}$", r"$\{1, 4, 14\}$", r"$\{1, 11\}$"],
         sol=r"""By CRT, $x^2 \equiv 1$ mod 15 iff $x \equiv \pm1 \pmod 3$ and $x \equiv \pm1 \pmod 5$: four combinations, $x = 1, 4, 11, 14$
($4^2 = 16$, $11^2 = 121 = 8\cdot15 + 1$). The non-trivial roots $4$ and $11$ give factors: $\gcd(4 - 1, 15) = 3$, $\gcd(4 + 1, 15) = 5$. Shor's algorithm
finds such a root as $x = a^{r/2}$."""),

    dict(id="A11", kind="mcq",
         q=r"""For $N = 21$ and $a = 5$, the order $r$ and the outcome of Shor's classical reduction are""",
         opts=[r"$r = 6$; the reduction fails because $5^3 \equiv -1 \pmod{21}$", r"$r = 6$; it gives the factors $3$ and $7$",
               r"$r = 3$; it fails because $r$ is odd", r"$r = 12$; it gives the factors $3$ and $7$"],
         sol=r"""$5^1 = 5$, $5^2 = 25 \equiv 4$, $5^3 \equiv 20 \equiv -1$, $5^6 \equiv 1$: $r = 6$ (even). But $a^{r/2} = 5^3 \equiv -1$, so
$\gcd(5^3 + 1, 21) = \gcd(21, 21) = 21$ and $\gcd(5^3 - 1, 21) = \gcd(19, 21) = 1$: only trivial factors. The algorithm must restart with another $a$."""),

    dict(id="A12", kind="mcq",
         q=r"""$2^{100} \bmod 13$ equals""",
         opts=[r"$3$", r"$9$", r"$1$", r"$12$"],
         sol=r"""Fermat: $2^{12} \equiv 1 \pmod{13}$. $100 = 8\cdot12 + 4$, so $2^{100} \equiv 2^4 = 16 \equiv 3$. Reducing the exponent modulo the order (or
$\varphi$) is what makes $a^x \bmod N$ periodic in $x$."""),

    dict(id="A13", kind="mcq",
         q=r"""Trial division factors $N = 10403$ by testing divisors $d \ge 2$ in increasing order. The largest $d$ that may have to be tested is""",
         opts=[r"$101$", r"$102$", r"$5201$", r"$103$"],
         sol=r"""If $N$ is composite it has a factor $\le\sqrt N$. $101^2 = 10201 \le 10403 < 102^2 = 10404$, so testing up to $\lfloor\sqrt N\rfloor = 101$ suffices
--- and $10403 = 101\times103$ is found at $d = 101$. For an $n$-bit $N$ this is about $2^{n/2}$ divisions: exponential in $n$."""),

    dict(id="A14", kind="mcq",
         q=r"""For an odd $N$ with $k$ distinct prime factors (not a prime power) and $a$ uniformly random among the residues coprime to $N$, the probability
that $r$ is even and $a^{r/2} \not\equiv -1 \pmod N$ is at least $1 - 2^{-(k-1)}$. For $N = 105 = 3\cdot5\cdot7$ this lower bound is""",
         opts=[r"$3/4$", r"$1/2$", r"$7/8$", r"$1/8$"],
         sol=r"""$k = 3$: $1 - 2^{-2} = \frac34$. For RSA moduli ($k = 2$) the bound is $\frac12$, so on average at most two bases are needed. This bound comes from
the CRT: the reduction fails only if the 2-adic parts of the orders modulo the prime powers all agree."""),

    dict(id="A15", kind="match",
         q=r"""Match each quantity for $N = 15$ (List-I) with its value (List-II).""",
         match=([r"$\varphi(15)$", r"order of $4$ modulo $15$", r"Carmichael value $\lambda(15) = \operatorname{lcm}(\lambda(3), \lambda(5))$",
                 r"$7^{-1} \bmod 15$"],
                [r"$13$", r"$8$", r"$2$", r"$4$"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1",
               r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item $\varphi(15) = 2\cdot4 = 8$ (2).
\item $4^2 = 16 \equiv 1$: order $2$ (3).
\item $\lambda(15) = \operatorname{lcm}(2, 4) = 4$ (4): every order divides it.
\item $7\cdot13 = 91 = 6\cdot15 + 1$: $7^{-1} = 13$ (1).
\end{itemize}
Hence P$\to$2, Q$\to$3, R$\to$4, S$\to$1."""),

    dict(id="A16", kind="match",
         q=r"""Match each classical algorithm (List-I) with its running time for an $n$-bit input $N$ (List-II).""",
         match=([r"trial division", r"Pollard's rho", r"general number field sieve", r"Euclidean $\gcd(a, N)$"],
                [r"$\exp\big((64/9)^{1/3}(\ln N)^{1/3}(\ln\ln N)^{2/3}(1 + o(1))\big)$ (sub-exponential)",
                 r"$O(N^{1/4})$ expected, i.e.\ $O(2^{n/4})$ (heuristic)",
                 r"$O(n)$ division steps (polynomial)",
                 r"$O(\sqrt N) = O(2^{n/2})$ divisions"]),
         opts=[r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3",
               r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item trial division tests divisors up to $\sqrt N$ (4);
\item Pollard's rho finds a factor $p$ in about $\sqrt p \le N^{1/4}$ steps (2);
\item the GNFS is the fastest known general method, sub-exponential but super-polynomial (1);
\item the gcd is polynomial (3) --- not a factoring algorithm on its own.
\end{itemize}
Hence P$\to$4, Q$\to$2, R$\to$1, S$\to$3. Not all classical factoring is exponential."""),

    dict(id="A17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} If $x^2 \equiv 1 \pmod N$ but $x \not\equiv \pm1 \pmod N$, then $\gcd(x - 1, N)$ is a non-trivial factor of $N$.\\[1mm]
\textbf{Reason (R):} In that case $N$ divides $(x - 1)(x + 1)$ but divides neither $x - 1$ nor $x + 1$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true}: $x^2 - 1 = (x-1)(x+1) \equiv 0$, while $x \not\equiv \pm1$ means $N \nmid x \mp 1$. \textbf{A follows from R}: if $\gcd(x - 1, N)$ were
$1$, then $N \mid (x-1)(x+1)$ would force $N \mid x + 1$; if it were $N$, then $N \mid x - 1$ --- both excluded. So it is a proper divisor. This is exactly
Shor's factor-extraction step with $x = a^{r/2}$. Answer: (A)."""),

    dict(id="A18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} If $\gcd(a, N) = 1$, then $a^{\varphi(N)} \equiv 1 \pmod N$.\\[1mm]
\textbf{Statement II:} The multiplicative order of every $a$ coprime to $N$ equals $\varphi(N)$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true} (Euler's theorem). \textbf{II is false}: the order only \emph{divides} $\varphi(N)$; e.g.\ $4$ has order $2$ modulo $15$ while
$\varphi(15) = 8$. Finding the actual order is the hard problem that Shor's quantum period finding solves. Answer: (C)."""),

    dict(id="A19", kind="para", passage="A",
         q=r"""The order $r$ of $3$ modulo $91$ is""",
         opts=[r"$6$", r"$12$", r"$3$", r"$72$"],
         sol=r"""By CRT, $r = \operatorname{lcm}(r_7, r_{13})$. Mod 7: $3, 2, 6, 4, 5, 1$: order $6$. Mod 13: $3, 9, 27 \equiv 1$: order $3$. So $r = \operatorname{lcm}(6, 3) = 6$.
\emph{Check}: $3^6 = 729 = 8\cdot91 + 1$. ($72 = \varphi(91)$ is only a multiple of the order.)"""),

    dict(id="A20", kind="para", passage="A",
         q=r"""The values of $\gcd(3^{r/2} - 1, 91)$ and $\gcd(3^{r/2} + 1, 91)$ are, respectively,""",
         opts=[r"$13$ and $7$", r"$1$ and $91$", r"$26$ and $28$", r"$7$ and $13$"],
         sol=r"""$r = 6$ is even and $3^3 = 27 \not\equiv -1 \equiv 90$. $\gcd(26, 91) = 13$ (since $91 = 7\cdot13$, $26 = 2\cdot13$) and $\gcd(28, 91) = 7$
($28 = 4\cdot7$). Both factors are found: $91 = 7\times13$. The order of the two numbers matters: $a^{r/2} - 1 = 26$ shares $13$ with $91$."""),
]
