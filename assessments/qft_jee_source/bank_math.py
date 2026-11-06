"""Section A - Mathematics (complex numbers, roots of unity, DFT, binary fractions, linear algebra, number theory).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed JEE option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""Let $N = 32$ and define the sequence $a_x = 1$ if $x \equiv 2 \pmod 5$ and $a_x = 0$ otherwise, for
$x = 0, 1, \dots, 31$. Its (un-normalised) Fourier coefficients are
$A_y = \sum_{x=0}^{31} a_x\, e^{2\pi i\,xy/32}$, $\;y = 0, 1, \dots, 31$. (The sequence is ``periodic with period 5'', but
5 does not divide 32.)""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""Let $\omega = e^{2\pi i/8}$. The value of $\displaystyle S = \sum_{k=0}^{7} k\,\omega^{k}$ is""",
         opts=[r"$\dfrac{8}{\omega - 1}$", r"$\dfrac{8}{1-\omega}$", r"$0$", r"$\dfrac{8\omega}{\omega-1}$"],
         sol=r"""Use $\sum_{k=0}^{M-1} k z^k = \dfrac{z\big(1 - M z^{M-1} + (M-1) z^{M}\big)}{(1-z)^2}$ (differentiate the geometric
series $\sum z^k = \frac{1-z^M}{1-z}$ and multiply by $z$). With $z = \omega$, $M = 8$, $\omega^8 = 1$ and $\omega^7 = \omega^{-1}$:
\[ S = \frac{\omega\,(1 - 8\omega^{-1} + 7)}{(1-\omega)^2} = \frac{8\omega - 8}{(1-\omega)^2} = \frac{-8(1-\omega)}{(1-\omega)^2}
     = \frac{8}{\omega - 1}. \]
\emph{Check:} $S \neq 0$ because the weights $k$ break the perfect cancellation of $\sum_k \omega^k = 0$;
$\frac{8}{1-\omega}$ has the wrong sign. Numerically $S = -4 + 9.657i$."""),

    dict(id="A2", kind="mcq",
         q=r"""The number of ordered pairs $(x, y)$ with $x, y \in \{0, 1, \dots, 15\}$ for which
$\displaystyle \sum_{z=0}^{15} e^{2\pi i\,(x+y)z/16} \neq 0$ is""",
         opts=[r"$16$", r"$1$", r"$15$", r"$31$"],
         sol=r"""The sum is a sum of 16th roots of unity: $\sum_{z=0}^{15} \omega^{kz} = 16$ if $16 \mid k$ and $0$ otherwise
($\omega = e^{2\pi i/16}$, geometric series). Here $k = x + y \in \{0, \dots, 30\}$, so we need $x + y \in \{0, 16\}$.
$x + y = 0$: one pair $(0,0)$. $x + y = 16$: $x = 1, \dots, 15$ with $y = 16 - x$: 15 pairs. Total $1 + 15 = 16$.
(Equivalently: $y \equiv -x \pmod{16}$ has exactly one solution $y$ for every $x$.)"""),

    dict(id="A3", kind="mcq",
         q=r"""Let $x = 45$, written with $n = 6$ bits as $x_1x_2\dots x_6$. The value of $e^{2\pi i\,x/2^4}$ is""",
         opts=[r"$e^{i\,13\pi/8}$", r"$e^{i\,5\pi/8}$", r"$e^{i\,3\pi/8}$", r"$e^{-i\,13\pi/8}$"],
         sol=r"""$45 = 101101_2$. Dividing by $2^4$ moves the binary point four places: $\dfrac{45}{16} = 10.1101_2 = 2 + 0.8125$.
Since $e^{2\pi i\cdot 2} = 1$, only the fractional part survives:
\[ e^{2\pi i\,45/16} = e^{2\pi i\,0.1101_2} = e^{2\pi i\,\frac{13}{16}} = e^{i\,13\pi/8}. \]
In QFT language: $e^{2\pi i\,x/2^l}$ depends only on the last $l$ bits $0.x_{n-l+1}\dots x_n$ (here $0.x_3x_4x_5x_6 = 0.1101_2$)."""),

    dict(id="A4", kind="mcq",
         q=r"""The matrix $U = \dfrac12\begin{pmatrix}1&1&1&1\\ 1&i&-1&-i\\ 1&-1&1&-1\\ 1&c&-1&d\end{pmatrix}$ is unitary. Then $(c, d)$ equals""",
         opts=[r"$(-i,\ i)$", r"$(i,\ -i)$", r"$(-1,\ 1)$", r"$(1,\ -1)$"],
         sol=r"""For a unitary matrix the rows are orthonormal. Let $r_4 = (1, c, -1, d)$.
\[ \langle r_1, r_4\rangle = 1 + c - 1 + d = c + d = 0 \;\Rightarrow\; d = -c. \]
\[ \langle r_2, r_4\rangle = \bar1\cdot1 + \bar i\,c + \overline{(-1)}(-1) + \overline{(-i)}\,d = 2 - ic + id = 2 - 2ic = 0 \;\Rightarrow\; c = \tfrac{1}{i} = -i, \]
hence $d = i$. Check $\langle r_3, r_4\rangle = 1 - c - 1 - d = -(c+d) = 0$ and $\|r_4\|^2 = \tfrac14(1+1+1+1) = 1$.
$U$ is exactly the 2-qubit QFT matrix $F_4$ with entries $i^{xy}/2$ (row 4: $i^{3x} = 1, -i, -1, i$)."""),

    dict(id="A5", kind="mcq",
         q=r"""Let $F$ be the $4\times4$ unitary Fourier matrix with entries $F_{yx} = \tfrac12\, i^{xy}$ ($x, y = 0, \dots, 3$).
The trace of $F$ is""",
         opts=[r"$1 + i$", r"$0$", r"$2$", r"$1 - i$"],
         sol=r"""$\operatorname{tr} F = \sum_x F_{xx} = \tfrac12\sum_{x=0}^{3} i^{x^2}$ (a \emph{Gauss sum}).
$x^2 = 0, 1, 4, 9$ give $i^0 = 1$, $i^1 = i$, $i^4 = 1$, $i^9 = i$. Hence $\operatorname{tr} F = \tfrac12(2 + 2i) = 1 + i$.
Consistency: $F^4 = I$, so its eigenvalues lie in $\{1, -1, i, -i\}$; the eigenvalues of this $F$ are $1, 1, -1, i$ and their
sum is $1 + i$."""),

    dict(id="A6", kind="mcq",
         q=r"""Let $f(\theta) = \big|\,1 + e^{i\theta} + e^{2i\theta} + e^{3i\theta}\,\big|^2$. The maximum value of $f$ and the smallest
positive $\theta$ with $f(\theta) = 0$ are, respectively,""",
         opts=[r"$16$ and $\pi/2$", r"$4$ and $\pi/2$", r"$16$ and $\pi/4$", r"$4$ and $\pi$"],
         sol=r"""Geometric series: $\sum_{j=0}^{3} e^{ij\theta} = \dfrac{1 - e^{4i\theta}}{1 - e^{i\theta}}
= e^{3i\theta/2}\,\dfrac{\sin 2\theta}{\sin(\theta/2)}$, so
\[ f(\theta) = \frac{\sin^2 2\theta}{\sin^2(\theta/2)} . \]
Maximum: all four arrows aligned at $\theta = 0$, $f = |4|^2 = 16$ (by the triangle inequality $f \le 16$).
Zeros: $\sin 2\theta = 0$ with $\sin(\theta/2) \neq 0$, i.e. $\theta = \pi/2, \pi, 3\pi/2, \dots$ The smallest is $\pi/2$
(four arrows pointing in the four directions $1, i, -1, -i$ cancel). This $\sin^2(K\theta/2)/\sin^2(\theta/2)$ shape is the
peak shape of every QFT output."""),

    dict(id="A7", kind="mcq",
         q=r"""The best rational approximation $p/q$ to $\dfrac{11}{64}$ with $1 \le q \le 10$ (as produced by the continued-fraction
algorithm) is""",
         opts=[r"$1/6$", r"$1/5$", r"$2/9$", r"$1/7$"],
         sol=r"""Continued fraction of $\frac{11}{64}$: $64 = 5\cdot 11 + 9$, $11 = 1\cdot 9 + 2$, $9 = 4\cdot 2 + 1$, $2 = 2\cdot 1$, so
$\frac{11}{64} = [0;\,5,\,1,\,4,\,2]$. Convergents: $\frac{0}{1}, \frac15, \frac16, \frac{5}{29}, \frac{11}{64}$.
The last convergent with $q \le 10$ is $\frac16$. Direct check: $\left|\frac{11}{64} - \frac16\right| = \frac{2}{384} \approx 0.0052$,
while $\left|\frac{11}{64} - \frac15\right| \approx 0.028$, $\frac17$ differs by $0.029$ and $\frac29$ by $0.050$.
This is exactly the step that turns a QFT measurement $y = 11$ out of $M = 64$ into the period $r = 6$."""),

    dict(id="A8", kind="mcq",
         q=r"""The order $r$ of $a = 3$ modulo $N = 35$ (the least $r > 0$ with $a^r \equiv 1 \pmod{35}$), and the numbers
$\gcd(a^{r/2} - 1, N)$ and $\gcd(a^{r/2} + 1, N)$, are""",
         opts=[r"$r = 12$; $\;7$ and $5$", r"$r = 6$; $\;5$ and $7$", r"$r = 12$; $\;1$ and $35$", r"$r = 4$; $\;5$ and $7$"],
         sol=r"""Powers of 3 modulo 35: $3, 9, 27, 11, 33, 29, 17, 16, 13, 4, 12, 1$, so $r = 12$ (by Euler, $r \mid \varphi(35) = 24$;
$3^{6} \equiv 29 \not\equiv 1$ and $3^{4} \equiv 11 \not\equiv 1$, so $r$ is neither 6 nor 4).
$a^{r/2} = 3^6 \equiv 29 \pmod{35}$, which is not $\equiv -1 = 34$, so Shor's reduction works:
\[ \gcd(29 - 1, 35) = \gcd(28, 35) = 7, \qquad \gcd(29 + 1, 35) = \gcd(30, 35) = 5 . \]
Indeed $35 = 5 \times 7$."""),

    dict(id="A9", kind="mcq",
         q=r"""A general pure state of 10 qubits is described by $D$ complex amplitudes. After removing normalisation and
the unobservable global phase, the number of independent \emph{real} parameters is $P$. Then $(D, P)$ is""",
         opts=[r"$(1024,\ 2046)$", r"$(1024,\ 2048)$", r"$(20,\ 40)$", r"$(1024,\ 2047)$"],
         sol=r"""$n$ qubits span a $2^n$-dimensional space: $D = 2^{10} = 1024$ complex numbers $= 2048$ real numbers.
Normalisation $\sum|\alpha_x|^2 = 1$ removes one real parameter and the global phase $e^{i\gamma}$ removes another:
$P = 2048 - 2 = 2046$. (Option $(1024, 2047)$ forgets the global phase; $(20, 40)$ counts as if qubits were independent.)"""),

    dict(id="A10", kind="mcq",
         q=r"""For $n = 4$, $H^{\otimes 4}|x\rangle = \frac14\sum_{y}(-1)^{x\cdot y}|y\rangle$ where $x\cdot y = \sum_j x_jy_j$.
For $x = 1011_2$, the number of $y \in \{0,\dots,15\}$ with a \emph{negative} amplitude is""",
         opts=[r"$8$", r"$4$", r"$3$", r"$12$"],
         sol=r"""The amplitude is negative iff $x\cdot y = y_1 + y_3 + y_4$ is odd (because $x = 1011$ has $x_1 = x_3 = x_4 = 1$).
$y_2$ is free (2 choices) and among the 8 choices of $(y_1, y_3, y_4)$ exactly half have odd sum, so the count is $2 \times 4 = 8$.
In general, for any $x \neq 0$ exactly $2^{n-1}$ of the $2^n$ signs are negative (the characters $(-1)^{x\cdot y}$ of
$\mathbb{Z}_2^n$ sum to zero)."""),

    dict(id="A11", kind="mcq",
         q=r"""A classical FFT on a vector of length $2^n$ needs about $n\,2^n$ operations, while the QFT circuit needs
$\frac{n(n+1)}{2} + \lfloor n/2\rfloor$ gates. For $n = 20$ the ratio (FFT operations)/(QFT gates) is closest to""",
         opts=[r"$10^{5}$", r"$10^{3}$", r"$10^{7}$", r"$10^{9}$"],
         sol=r"""FFT: $20 \times 2^{20} = 20 \times 1\,048\,576 = 20\,971\,520$. QFT: $\frac{20\cdot21}{2} + 10 = 210 + 10 = 220$ gates.
Ratio $= 20\,971\,520 / 220 \approx 9.5 \times 10^{4} \approx 10^5$. Caveat (important for interpretation): the QFT output is
stored in amplitudes and cannot be read out completely, so this ratio is \emph{not} a speed-up for computing all coefficients."""),

    dict(id="A12", kind="mcq",
         q=r"""With $\omega = e^{2\pi i/8}$, the value of $\displaystyle\prod_{k=1}^{7}\big|1 - \omega^{k}\big|$ is""",
         opts=[r"$8$", r"$0$", r"$1$", r"$7$"],
         sol=r"""The roots of $z^8 - 1$ are $\omega^0, \dots, \omega^7$, so $z^8 - 1 = \prod_{k=0}^{7}(z - \omega^k)$ and
\[ \frac{z^8-1}{z-1} = 1 + z + \dots + z^7 = \prod_{k=1}^{7}(z - \omega^k). \]
Put $z = 1$: $\prod_{k=1}^{7}(1-\omega^k) = 8$, hence the product of the moduli is $8$. (Geometrically: the product of the
distances from one vertex of a regular octagon inscribed in the unit circle to the other seven vertices.)"""),

    dict(id="A13", kind="mcq",
         q=r"""The normalised vector $\alpha = \frac13(1, 2, 2, 0)$ is transformed by the unitary $4\times4$ Fourier matrix
$\beta_y = \frac12\sum_{x}\alpha_x\, i^{xy}$. The value of $|\beta_0|^2$ is""",
         opts=[r"$\dfrac{25}{36}$", r"$\dfrac56$", r"$\dfrac14$", r"$\dfrac{25}{12}$"],
         sol=r"""$\beta_0 = \frac12\sum_x \alpha_x = \frac12\cdot\frac{1+2+2+0}{3} = \frac56$, so $|\beta_0|^2 = \frac{25}{36}$.
Check the other options: $\frac56$ is the amplitude, not the probability; $\frac{25}{12} > 1$ is impossible because
$\sum_y|\beta_y|^2 = \sum_x|\alpha_x|^2 = 1$ (Parseval, $F$ is unitary). The remaining probability $\frac{11}{36}$ is spread over $y = 1,2,3$."""),

    dict(id="A14", kind="mcq",
         q=r"""Let $\varphi = 0.\varphi_1\varphi_2\varphi_3$ (binary, $\varphi_j \in \{0,1\}$). Then $e^{2\pi i\,4\varphi}$ equals""",
         opts=[r"$(-1)^{\varphi_3}$", r"$(-1)^{\varphi_1}$", r"$i^{\varphi_2}$", r"$e^{2\pi i\,0.\varphi_2\varphi_3}$"],
         sol=r"""$4\varphi = \varphi_1\varphi_2.\varphi_3$ in binary, i.e. $4\varphi = (2\varphi_1 + \varphi_2) + 0.\varphi_3$. The integer part is a
whole number of turns, so $e^{2\pi i\,4\varphi} = e^{2\pi i\,0.\varphi_3} = e^{i\pi\varphi_3} = (-1)^{\varphi_3}$.
This is why, in phase estimation, the counting qubit controlling $U^{4}$ carries only the \emph{last} bit $\varphi_3$, and the
inverse QFT reads the bits from there."""),

    dict(id="A15", kind="match",
         q=r"""Let $\omega = e^{2\pi i/8}$. Match List-I with List-II.""",
         match=([r"$\omega^{4}$", r"$\sum_{k=0}^{7}\omega^{2k}$", r"$\omega\cdot\omega^{7}$", r"$\omega + \omega^{7}$"],
                [r"$\sqrt2$", r"$-1$", r"$0$", r"$1$"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2"],
         sol=r"""P: $\omega^4 = e^{i\pi} = -1$ (2). Q: $\omega^2 = i$ is a primitive 4th root, and $\sum_{k=0}^{7} i^k = 2(1 + i - 1 - i) = 0$ (3).
R: $\omega\,\omega^7 = \omega^8 = 1$ (4). S: $\omega + \omega^7 = \omega + \bar\omega = 2\cos\frac{\pi}{4} = \sqrt2$ (1).
Hence P$\to$2, Q$\to$3, R$\to$4, S$\to$1."""),

    dict(id="A16", kind="match",
         q=r"""$F$ is the $4\times4$ unitary Fourier matrix ($F_{yx} = i^{xy}/2$), $H$ the Hadamard matrix and
$CT = \operatorname{diag}(1, 1, 1, e^{i\pi/4})$. Match each matrix in List-I with the \emph{only} property in List-II that it has.""",
         match=([r"$F$", r"$F^2$", r"$H\otimes H$", r"$CT$"],
                [r"a permutation matrix that is its own inverse",
                 r"real, symmetric, its own inverse, but not a permutation matrix",
                 r"diagonal with determinant $e^{i\pi/4}$",
                 r"$M^4 = I$ but $M^2 \neq I$"]),
         opts=[r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3",
               r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4"],
         sol=r"""$(F^2)_{zx} = \frac14\sum_y i^{y(x+z)} = \delta_{z,\,-x \bmod 4}$, so $F^2$ is the permutation $|x\rangle \mapsto |{-x}\rangle$
($0\to0$, $1\leftrightarrow3$, $2\to2$), which squares to $I$: Q$\to$1. Hence $F^4 = I$ while $F^2 \neq I$: P$\to$4.
$H\otimes H = \frac12\begin{psmallmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{psmallmatrix}$ is real, symmetric and
$(H\otimes H)^2 = I$, but its entries are $\pm\frac12$, so it is not a permutation: R$\to$2.
$CT$ is diagonal with $\det = e^{i\pi/4}$; note $CT^4 = \operatorname{diag}(1,1,1,-1) \neq I$, so it does not have property 4: S$\to$3."""),

    dict(id="A17", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} For every integer $k$ that is not a multiple of $N$, $\displaystyle\sum_{j=0}^{N-1} e^{2\pi i\,jk/N} = 0$.\\[1mm]
\textbf{Reason (R):} For every integer $k$ that is not a multiple of $N$, the $N$ numbers $e^{2\pi i\,jk/N}$
($j = 0, \dots, N-1$) are the $N$ distinct vertices of a regular $N$-gon centred at the origin.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} with $z = e^{2\pi i k/N} \neq 1$, $\sum_{j=0}^{N-1} z^j = \frac{1-z^N}{1-z} = \frac{1 - e^{2\pi i k}}{1-z} = 0$.
\textbf{R is false:} if $g = \gcd(k, N) > 1$ the numbers repeat. Example $N = 8$, $k = 2$: the values are
$1, i, -1, -i, 1, i, -1, -i$ --- a square traversed twice, not 8 distinct vertices of an octagon. (The sum is still zero,
because a regular $N/g$-gon is traversed $g$ times.) Answer: (C)."""),

    dict(id="A18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} Every unitary matrix $U$ satisfies $|\det U| = 1$.\\[1mm]
\textbf{Statement II:} The unitary Fourier matrix $F_N$ ($N = 2^n$, entries $\omega^{xy}/\sqrt N$) has $\det F_N = \pm 1$ for every $n$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $|\det U|^2 = \det(U^\dagger U) = \det I = 1$.
\textbf{II is false:} the determinant is the product of the eigenvalues, which lie in $\{1,-1,i,-i\}$ because $F^4 = I$.
For $N = 4$ the eigenvalues are $1, 1, -1, i$ (compare the trace $1 + i$ in A5), so $\det F_4 = 1\cdot1\cdot(-1)\cdot i = -i \neq \pm1$.
(For $N = 2$, $F_2 = H$ and $\det H = -1$, so the claim happens to hold there --- one counterexample is enough.) Answer: (C)."""),

    dict(id="A19", kind="para", passage="A",
         q=r"""The value of $\displaystyle\sum_{y=0}^{31} |A_y|^2$ is""",
         opts=[r"$192$", r"$6$", r"$36$", r"$1152$"],
         sol=r"""The non-zero samples are $x = 2, 7, 12, 17, 22, 27$, i.e. $K = 6$ ones. For the un-normalised transform
$A = \sqrt{N}\,F a$ with unitary $F$, Parseval gives
\[ \sum_y |A_y|^2 = N\sum_x |a_x|^2 = 32 \times 6 = 192 . \]
(36 is $|A_0|^2$ alone; 6 is $\sum_x|a_x|^2$; 1152 would be $N \cdot 36$.)"""),

    dict(id="A20", kind="para", passage="A",
         q=r"""Among $y = 1, 2, \dots, 15$, the value of $y$ for which $|A_y|$ is largest is""",
         opts=[r"$13$", r"$6$", r"$7$", r"$16$ is the only peak; all of $1,\dots,15$ give $|A_y| = 0$"],
         sol=r"""$A_y = \sum_{j=0}^{5} e^{2\pi i (5j+2)y/32} = e^{4\pi i y/32}\sum_{j=0}^{5} e^{2\pi i\,5jy/32}$, so
\[ |A_y| = \left|\frac{\sin(6\cdot 5\pi y/32)}{\sin(5\pi y/32)}\right| = \left|\frac{\sin(30\pi y/32)}{\sin(5\pi y/32)}\right| . \]
Peaks occur where $5y/32$ is close to an integer, i.e. $y \approx 32k/5 = 6.4,\ 12.8,\ \dots$ The closer $5y/32$ is to an
integer, the higher the peak: $5\cdot13/32 = 2.03$ (distance $0.03$) beats $5\cdot6/32 = 0.94$ (distance $0.06$) and
$5\cdot7/32 = 1.09$. Numerically $|A_{13}| = 5.67$, $|A_6| = 4.74$, $|A_7| = 3.38$, $|A_{12}| = 1.85$; and $|A_{16}| = 0$ since
$\sin(15\pi) = 0$. Hence $y = 13$. Lesson: with a period that does not divide $N$, the QFT peaks ``leak'' and their
heights depend on how close $ky/N$ is to an integer."""),
]
