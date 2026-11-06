"""Section A - Mathematics (trigonometry, reflections and rotations, averages, counting, probability, scaling).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""Define $p(k) = \sin^2\big((2k+1)\theta\big)$ for integers $k \ge 0$, with $\theta = \pi/18 = 10^\circ$. (This is the
shape of the Grover success curve, but the two questions below need only trigonometry.) Note that $(2k+1)\theta$ takes the
values $10^\circ, 30^\circ, 50^\circ, \dots$""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""If $\sin\theta = \tfrac14$, the value of $\sin^2 3\theta$ is""",
         opts=[r"$\dfrac{121}{256}$", r"$\dfrac{9}{16}$", r"$\dfrac{11}{16}$", r"$\dfrac{1}{16}$"],
         sol=r"""Use the triple-angle identity $\sin 3\theta = 3\sin\theta - 4\sin^3\theta$:
\[ \sin 3\theta = \frac34 - 4\cdot\frac{1}{64} = \frac{12}{16} - \frac{1}{16} = \frac{11}{16},
   \qquad \sin^2 3\theta = \frac{121}{256}. \]
This is exactly the success probability of one Grover iteration on $N = 16$ items.
The value $\frac{9}{16} = (3\sin\theta)^2$ is the small-angle estimate; it ignores the cubic correction."""),

    dict(id="A2", kind="mcq",
         q=r"""Let $u = \big(\tfrac35, \tfrac45\big)^{T}$. The matrix $2uu^{T} - I$ equals""",
         opts=[r"$\dfrac{1}{25}\begin{pmatrix}-7 & 24\\ 24 & 7\end{pmatrix}$",
               r"$\dfrac{1}{25}\begin{pmatrix}7 & -24\\ -24 & -7\end{pmatrix}$",
               r"$\dfrac{1}{25}\begin{pmatrix}18 & 24\\ 24 & 32\end{pmatrix}$",
               r"$\dfrac{1}{25}\begin{pmatrix}-7 & 12\\ 12 & 7\end{pmatrix}$"],
         sol=r"""$uu^{T} = \frac{1}{25}\begin{psmallmatrix}9 & 12\\ 12 & 16\end{psmallmatrix}$, so
$2uu^{T} = \frac{1}{25}\begin{psmallmatrix}18 & 24\\ 24 & 32\end{psmallmatrix}$. Subtracting $I = \frac{1}{25}\begin{psmallmatrix}25 & 0\\ 0 & 25\end{psmallmatrix}$ gives
$\frac{1}{25}\begin{psmallmatrix}-7 & 24\\ 24 & 7\end{psmallmatrix}$.
\emph{Check:} it fixes $u$, since $\frac{1}{25}\cdot\frac15(-21 + 96,\ 72 + 28) = \frac{1}{125}(75, 100) = (\frac35, \frac45)$. Its trace is $0$ and its
determinant is $-1$, as for every reflection in the plane."""),

    dict(id="A3", kind="mcq",
         q=r"""A vector in the plane is reflected first in the line through the origin at angle $10^\circ$ to the $x$-axis, and then in the
line through the origin at angle $35^\circ$. The combined map is""",
         opts=[r"a rotation by $+50^\circ$", r"a rotation by $+25^\circ$", r"a rotation by $-50^\circ$",
               r"a reflection in the line at angle $45^\circ$"],
         sol=r"""A reflection in the line at angle $\alpha$ sends a vector at angle $\phi$ to angle $2\alpha - \phi$. Applying $\alpha = 10^\circ$ and then
$\beta = 35^\circ$:
\[ \phi \;\to\; 20^\circ - \phi \;\to\; 70^\circ - (20^\circ - \phi) = \phi + 50^\circ . \]
The result is a rotation by $2(\beta - \alpha) = +50^\circ$: \textbf{twice} the angle between the lines, in the direction from the
first line to the second. This is the geometric heart of Grover's algorithm."""),

    dict(id="A4", kind="mcq",
         q=r"""Let $\sin\theta = \tfrac14$ with $0 < \theta < \tfrac\pi2$, and let
$G = \begin{psmallmatrix}\cos 2\theta & -\sin 2\theta\\ \sin 2\theta & \cos 2\theta\end{psmallmatrix}$. The eigenvalues of $G$ are""",
         opts=[r"$\dfrac{7 \pm i\sqrt{15}}{8}$", r"$\dfrac{1 \pm i\sqrt{15}}{4}$", r"$\dfrac{7 \pm \sqrt{15}}{8}$", r"$\pm 1$"],
         sol=r"""A rotation by angle $\alpha$ has eigenvalues $e^{\pm i\alpha} = \cos\alpha \pm i\sin\alpha$ (trace $2\cos\alpha$, determinant $1$).
Here $\alpha = 2\theta$, with
\[ \cos 2\theta = 1 - 2\sin^2\theta = 1 - \frac{2}{16} = \frac78, \qquad
   \sin 2\theta = 2\cdot\frac14\cdot\frac{\sqrt{15}}{4} = \frac{\sqrt{15}}{8}. \]
Therefore the eigenvalues are $\frac{7 \pm i\sqrt{15}}{8}$; their modulus is $\sqrt{(49 + 15)/64} = 1$, as required."""),

    dict(id="A5", kind="mcq",
         q=r"""Let $z = \dfrac{1 + i}{\sqrt2}$. The value of $z^6$ is""",
         opts=[r"$-i$", r"$i$", r"$1$", r"$-1$"],
         sol=r"""Write $z$ in polar form: $|z| = \sqrt{\frac12 + \frac12} = 1$ and its angle is $45^\circ$, so $z = e^{i\pi/4}$. Powers multiply the angle:
\[ z^6 = e^{6i\pi/4} = e^{3i\pi/2} = \cos 270^\circ + i\sin 270^\circ = -i. \]
Since $|z| = 1$, multiplying an amplitude by $z$ changes only its phase and never its probability."""),

    dict(id="A6", kind="mcq",
         q=r"""Let $\sin\theta = \tfrac{1}{10}$, $0 < \theta < \tfrac\pi2$; numerically $\theta = 0.10017$. Among the integers $0 \le k \le 10$, the one that makes
$\sin^2\big((2k+1)\theta\big)$ largest is""",
         opts=[r"$7$", r"$8$", r"$10$", r"$5$"],
         sol=r"""We want $(2k+1)\theta$ as close as possible to $\frac\pi2$, i.e.\ $k$ closest to $\frac{\pi}{4\theta} - \frac12$. Here
\[ \frac{\pi}{4\theta} - \frac12 = \frac{3.1416}{0.40067} - 0.5 = 7.841 - 0.5 = 7.341, \]
so $k = 7$: $15\theta = 1.5025$ rad is $0.068$ below $\frac\pi2$, while $17\theta = 1.7028$ rad is $0.132$ above it. Then
$\sin^2 15\theta = 0.9953$ and $\sin^2 17\theta = 0.9827$.
The tempting estimate $\frac\pi4\cdot 10 = 7.85$ rounds to $8$, which is wrong: it omits the $-\frac12$."""),

    dict(id="A7", kind="mcq",
         q=r"""The vector $v \in \mathbb{R}^{8}$ has seven entries equal to $+1$ and one entry equal to $-1$. Each entry $v_j$ is replaced by $2\bar v - v_j$,
where $\bar v$ is the mean of the entries. The new value of the entry that was $-1$ is""",
         opts=[r"$\dfrac52$", r"$\dfrac12$", r"$\dfrac74$", r"$\dfrac32$"],
         sol=r"""The mean is $\bar v = \frac{7 - 1}{8} = \frac34$. The special entry becomes $2\cdot\frac34 - (-1) = \frac52$; each other entry becomes
$\frac32 - 1 = \frac12$. \emph{Check:} the length is preserved, $7\cdot\frac14 + \frac{25}{4} = 8 = 7 + 1$.
Inversion about the mean reflects every entry through the mean, so an entry far below the mean ends up far above it."""),

    dict(id="A8", kind="mcq",
         q=r"""A box holds $31$ cards of which exactly $3$ are marked. Cards are drawn one at a time at random without replacement until the first
marked card appears. The expected number of draws is""",
         opts=[r"$8$", r"$\dfrac{31}{3}$", r"$16$", r"$\dfrac{31}{4}$"],
         sol=r"""The $M = 3$ marked cards cut the random order of the $N - M = 28$ unmarked cards into $M + 1 = 4$ gaps, and by symmetry each gap
holds on average $\frac{28}{4} = 7$ unmarked cards. The first marked card therefore comes after $7$ unmarked ones on average, on draw number $8$.
In general the expected number of draws is $\frac{N+1}{M+1} = \frac{32}{4} = 8$. The value $\frac{31}{3}$ would be right
for drawing \emph{with} replacement."""),

    dict(id="A9", kind="mcq",
         q=r"""A key, which may be absent, is looked up in a list of $1023$ distinct entries using classical comparisons. The worst-case numbers of comparisons are, for binary search on the \emph{sorted} list and for linear search on the \emph{unsorted} list, respectively,""",
         opts=[r"$10$ and $1023$", r"$10$ and $512$", r"$1023$ and $1023$", r"$32$ and $1023$"],
         sol=r"""Each comparison in binary search halves the remaining range. $1023 = 2^{10} - 1$ entries are exhausted after exactly $10$ comparisons. An unsorted list has no structure to exploit: if the key is absent or in the last place, all $1023$ entries must be checked.
Grover's algorithm addresses the second, \emph{unstructured}, case. ($512$ is about the \emph{average} cost of a linear search when the key is present.)"""),

    dict(id="A10", kind="mcq",
         q=r"""Let $f(x_1, x_2, x_3, x_4) = (x_1 \lor x_2) \land (x_3 \oplus x_4)$ on $\{0, 1\}^4$. The fraction of inputs with $f = 1$ is""",
         opts=[r"$\dfrac38$", r"$\dfrac14$", r"$\dfrac58$", r"$\dfrac12$"],
         sol=r"""The two factors involve different variables, so the counts multiply. $x_1 \lor x_2 = 1$ for $3$ of the $4$ pairs $(x_1, x_2)$;
$x_3 \oplus x_4 = 1$ for $2$ of the $4$ pairs $(x_3, x_4)$. So $M = 3\times2 = 6$ of $N = 16$ inputs, a fraction $\frac{6}{16} = \frac38$.
This fraction is the quantity $\sin^2\theta$ in a Grover search for the solutions of $f$."""),

    dict(id="A11", kind="mcq",
         q=r"""A Boolean function $f$ on $4$ bits takes the value $1$ on exactly $3$ inputs. The value of $\displaystyle\sum_{x\in\{0,1\}^4}(-1)^{f(x)}$ is""",
         opts=[r"$10$", r"$13$", r"$6$", r"$7$"],
         sol=r"""Each of the $N - M = 13$ inputs with $f = 0$ contributes $+1$, and each of the $M = 3$ inputs with $f = 1$ contributes $-1$:
\[ \sum_x (-1)^{f(x)} = (N - M) - M = N - 2M = 16 - 6 = 10. \]
Divided by $N$ this is the factor $1 - \frac{2M}{N}$ that appears in the mean amplitude after the oracle."""),

    dict(id="A12", kind="mcq",
         q=r"""The vector $\ket\psi = c\,\big(\ket{00} + i\ket{01} + (1 + i)\ket{10} + 2\ket{11}\big)$ with $c > 0$ is normalised. The value of $c$ is""",
         opts=[r"$\dfrac{1}{2\sqrt2}$", r"$\dfrac18$", r"$\dfrac{1}{\sqrt7}$", r"$\dfrac15$"],
         sol=r"""Normalisation means $\sum_x|c_x|^2 = 1$, where each term is a squared \emph{modulus}: $|1|^2 = 1$, $|i|^2 = 1$, $|1 + i|^2 = 1^2 + 1^2 = 2$, $|2|^2 = 4$. So
\[ c^2(1 + 1 + 2 + 4) = 8c^2 = 1 \;\Rightarrow\; c = \frac{1}{\sqrt8} = \frac{1}{2\sqrt2}. \]"""),

    dict(id="A13", kind="mcq",
         q=r"""Let $g(n) = \tfrac\pi4\sqrt{2^n}$. Then $g(n + 6)/g(n)$ equals""",
         opts=[r"$8$", r"$64$", r"$6$", r"$\sqrt6$"],
         sol=r"""$\frac{g(n+6)}{g(n)} = \sqrt{\frac{2^{n+6}}{2^n}} = \sqrt{2^6} = 2^3 = 8$. Six extra bits make the search space $64$ times larger but the
number of Grover iterations only $8$ times larger."""),

    dict(id="A14", kind="mcq",
         q=r"""Let $u = \frac{1}{\sqrt2}(1, i, 0)^{T}$ and $v = \frac{1}{\sqrt3}(1, 1, 1)^{T}$. The inner product $\braket{u|v}$ is""",
         opts=[r"$\dfrac{1 - i}{\sqrt6}$", r"$\dfrac{1 + i}{\sqrt6}$", r"$\dfrac{2}{\sqrt6}$", r"$0$"],
         sol=r"""$\braket{u|v} = \sum_k \overline{u_k}v_k$: the \emph{first} vector is conjugated. Thus
\[ \braket{u|v} = \frac{1}{\sqrt2}\cdot\frac{1}{\sqrt3}\big(\bar1\cdot1 + \bar i\cdot1 + 0\big) = \frac{1 - i}{\sqrt6}. \]
Forgetting the conjugate gives $\frac{1 + i}{\sqrt6}$, which is $\braket{v|u}$. The probability $|\braket{u|v}|^2 = \frac{2}{6} = \frac13$ is the same either way."""),

    dict(id="A15", kind="match",
         q=r"""Match the value of $\sin^2\theta$ (List-I) with the value of $\sin^2 3\theta$ (List-II).""",
         match=([r"$\sin^2\theta = \frac18$", r"$\sin^2\theta = \frac14$", r"$\sin^2\theta = \frac12$", r"$\sin^2\theta = \frac34$"],
                [r"$1$", r"$0$", r"$\frac{25}{32}$", r"$\frac12$"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""Since $\sin 3\theta = \sin\theta\,(3 - 4\sin^2\theta)$, with $s = \sin^2\theta$ we get $\sin^2 3\theta = s(3 - 4s)^2$.
$s = \frac18$: $\frac18\cdot\frac{25}{4} = \frac{25}{32}$ (3). $\;s = \frac14$: $\frac14\cdot 4 = 1$ (1). $\;s = \frac12$: $\frac12\cdot 1 = \frac12$ (4).
$\;s = \frac34$: $\frac34\cdot 0 = 0$ (2). Hence P$\to$3, Q$\to$1, R$\to$4, S$\to$2."""),

    dict(id="A16", kind="match",
         q=r"""Let $\ket{u}$ be a unit vector in $\mathbb{C}^{N}$, with $N \ge 3$. Match each matrix in List-I with its trace in List-II.""",
         match=([r"$2\ket{u}\!\bra{u} - I$", r"$I - 2\ket{u}\!\bra{u}$", r"$\ket{u}\!\bra{u}$", r"$I$"],
                [r"$N$", r"$N - 2$", r"$1$", r"$2 - N$"]),
         opts=[r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1",
               r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1"],
         sol=r"""The key fact is $\operatorname{tr}\ket{u}\!\bra{u} = \braket{u|u} = 1$, and $\operatorname{tr} I = N$. So
$\operatorname{tr}(2\ket{u}\!\bra{u} - I) = 2 - N$ (4); $\;\operatorname{tr}(I - 2\ket{u}\!\bra{u}) = N - 2$ (2); $\;\operatorname{tr}\ket{u}\!\bra{u} = 1$ (3);
$\;\operatorname{tr} I = N$ (1). Hence P$\to$4, Q$\to$2, R$\to$3, S$\to$1."""),

    dict(id="A17", kind="ar", correct=2, opts=AR_OPTIONS,
         q=r"""\textbf{Assertion (A):} Rounding $\frac\pi4\sqrt N$ to the nearest integer does not always give the integer $k$ for which
$(2k+1)\theta$ is closest to $\frac\pi2$, where $\sin\theta = 1/\sqrt N$.\\[1mm]
\textbf{Reason (R):} For $0 < x < 1$ we have $\arcsin x < x$.""",
         sol=r"""\textbf{A is true.} For $N = 100$, $\frac\pi4\sqrt N = 7.85$ rounds to $8$, but the true optimum is $k = 7$ (see A6).
\textbf{R is false:} the inequality goes the other way. $\arcsin x = x + \frac{x^3}{6} + \dots > x$, for example $\arcsin 0.1 = 0.10017$.
The actual reason for A is the omitted $-\frac12$ in $k = \frac{\pi}{4\theta} - \frac12$. Answer: (C)."""),

    dict(id="A18", kind="stmt", correct=0, opts=STATEMENT_OPTIONS,
         q=r"""Let $\ket{u}$ be a unit vector in $\mathbb{C}^{N}$, with $N \ge 2$, and let $D = 2\ket{u}\!\bra{u} - I$.\\[1mm]
\textbf{Statement I:} $D^2 = I$.\\[1mm]
\textbf{Statement II:} $D$ has the eigenvalue $+1$ with multiplicity $1$ and the eigenvalue $-1$ with multiplicity $N - 1$.""",
         sol=r"""\textbf{I is true:} $D^2 = 4\ket{u}\braket{u|u}\bra{u} - 4\ket{u}\!\bra{u} + I = I$, because $\braket{u|u} = 1$.
\textbf{II is true:} $D\ket{u} = 2\ket{u} - \ket{u} = \ket{u}$, and for any $\ket{w}$ orthogonal to $\ket{u}$ we have $D\ket{w} = -\ket{w}$; such vectors span
an $(N-1)$-dimensional space. \emph{Check:} the trace is $1 - (N-1) = 2 - N$, in agreement with A16. Answer: (A)."""),

    dict(id="A19", kind="para", passage="A", opts=[
             r"first maximum at $k = 4$; smallest period $9$", r"first maximum at $k = 4$; smallest period $18$",
             r"first maximum at $k = 5$; smallest period $9$", r"first maximum at $k = 9$; smallest period $18$"],
         q=r"""The smallest $k$ with $p(k) = 1$, and the smallest positive period of the sequence $p(k)$, are""",
         sol=r"""$p(k) = 1$ requires $(2k+1)\cdot 10^\circ = 90^\circ$ (the first odd multiple of $90^\circ$), i.e.\ $2k + 1 = 9$, so $k = 4$.
Moving from $k$ to $k + q$ adds $20^\circ q$ to the angle. $\sin^2$ has period $180^\circ$, so we need $20^\circ q = 180^\circ m$; the smallest
solution is $q = 9$. Hence period $9$ ($q = 18$ corresponds to the period $360^\circ$ of $\sin$ and is not the smallest)."""),

    dict(id="A20", kind="para", passage="A", opts=[r"$6$", r"$3$", r"$9$", r"$4$"],
         q=r"""The number of integers $k$ with $0 \le k \le 17$ for which $p(k) \ge \tfrac34$ is""",
         sol=r"""$\sin^2\alpha \ge \frac34$ means $|\sin\alpha| \ge \frac{\sqrt3}{2}$, i.e.\ $\alpha$ lies within $30^\circ$ of $90^\circ$ or of $270^\circ$. For $k = 0, \dots, 17$ the angle
$(2k+1)\cdot10^\circ$ runs over $10^\circ, 30^\circ, \dots, 350^\circ$. The values in $[60^\circ, 120^\circ]$ are $70^\circ, 90^\circ, 110^\circ$ ($k = 3, 4, 5$);
the values in $[240^\circ, 300^\circ]$ are $250^\circ, 270^\circ, 290^\circ$ ($k = 12, 13, 14$). That makes $6$ values of $k$."""),
]
