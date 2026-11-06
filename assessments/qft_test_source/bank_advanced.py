"""Section D - Advanced QFT (periodic states, shift theorem, phase estimation accuracy, Shor's algorithm,
approximate and semiclassical QFT, spectral properties, applications).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""\textbf{Shor's algorithm for $N = 21$ with $a = 2$.} A counting register of $t = 10$ qubits ($M = 2^{10} = 1024$) and a
5-qubit work register initialised to $\ket{1}$ are used. Steps: (i) $H^{\otimes 10}$ on the counting register;
(ii) $\ket{x}\ket{1} \mapsto \ket{x}\ket{2^x \bmod 21}$; (iii) inverse QFT on the counting register; (iv) measure the counting
register, obtaining $y$. Classical post-processing replaces $y/1024$ by the closest fraction $k/r$ with $r < 21$ (continued
fractions) and keeps $r$ if $2^r \equiv 1 \pmod{21}$. The true order of $2$ modulo $21$ is $6$, since $2^6 = 64 = 3\cdot21 + 1$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""On $n = 6$ qubits ($N = 64$) the state $\frac{1}{\sqrt8}\sum_{j=0}^{7}\ket{3 + 8j}$ is prepared, the QFT is applied and all
qubits are measured. The possible outcomes $y$ and their probabilities are""",
         opts=[r"$y \in \{0, 8, 16, \dots, 56\}$, each with probability $\frac18$",
               r"$y \in \{3, 11, 19, \dots, 59\}$, each with probability $\frac18$",
               r"only $y = 8$, with probability $1$",
               r"all $64$ values of $y$, each with probability $\frac{1}{64}$"],
         sol=r"""With $\omega = e^{2\pi i/64}$,
\[ \beta_y = \frac{1}{\sqrt8}\cdot\frac{1}{\sqrt{64}}\sum_{j=0}^{7}\omega^{(3+8j)y} = \frac{\omega^{3y}}{\sqrt{512}}\sum_{j=0}^{7} e^{2\pi i\,jy/8}. \]
The inner sum is $8$ if $8 \mid y$ and $0$ otherwise. Hence $\beta_y = \frac{8\,\omega^{3y}}{\sqrt{512}} = \frac{\omega^{3y}}{\sqrt8}$ for
$y = 0, 8, \dots, 56$ (8 values, probability $\frac18$ each) and $\beta_y = 0$ elsewhere. A state with period $r$ (here $8$)
dividing $N$ produces $r$ peaks at the multiples of $N/r$; the offset $3$ only changes phases (see D3)."""),

    dict(id="D2", kind="mcq",
         q=r"""For the state $\frac{1}{\sqrt8}\sum_{j=0}^{7}\ket{3 + 8j}$ on $6$ qubits, the phase (in $[0^\circ, 360^\circ)$) of the amplitude of
$\ket{16}$ after the QFT is""",
         opts=[r"$270^\circ$", r"$90^\circ$", r"$135^\circ$", r"$0^\circ$"],
         sol=r"""From the D1 calculation, $\beta_{16} = \frac{1}{\sqrt8}\,\omega^{3\cdot16} = \frac{1}{\sqrt8}\,e^{2\pi i\,48/64} = \frac{1}{\sqrt8}\,e^{2\pi i\cdot\frac34}$.
Phase $= \frac34\times360^\circ = 270^\circ$ (the amplitude is $-\frac{i}{\sqrt8}$). The offset $3$ is encoded in these phases:
$\beta_{8k} \propto e^{2\pi i\,3k/8}$, which is invisible to a single measurement but is what the shift theorem predicts."""),

    dict(id="D3", kind="mcq",
         q=r"""Let $\mathrm{QFT}\sum_x\alpha_x\ket{x} = \sum_y\beta_y\ket{y}$ on $n$ qubits ($N = 2^n$, $\omega = e^{2\pi i/N}$). For a fixed integer $s$,
$\mathrm{QFT}\sum_x\alpha_x\ket{(x + s) \bmod N}$ equals""",
         opts=[r"$\sum_y\omega^{sy}\beta_y\ket{y}$", r"$\sum_y\omega^{-sy}\beta_y\ket{y}$", r"$\sum_y\beta_{(y+s)\bmod N}\ket{y}$",
               r"$\sum_y\beta_y\ket{(y + s)\bmod N}$"],
         sol=r"""By linearity it is enough to look at one basis state:
$\mathrm{QFT}\ket{x+s} = \frac{1}{\sqrt N}\sum_y\omega^{(x+s)y}\ket y = \sum_y\omega^{sy}\Big(\frac{\omega^{xy}}{\sqrt N}\Big)\ket y$
($\omega^{N} = 1$ makes ``mod $N$'' harmless). Summing with weights $\alpha_x$ gives $\sum_y\omega^{sy}\beta_y\ket y$.
\textbf{Shift theorem:} a shift in $x$ becomes a phase ramp in $y$, and $|\beta_y|^2$ is unchanged. This is why the random,
unknown offset $x_0$ left after measuring the work register in Shor's algorithm does not matter."""),

    dict(id="D4", kind="mcq",
         q=r"""To obtain $\varphi$ accurate to $n$ bits with probability at least $1 - \varepsilon$, phase estimation uses
$t = n + \big\lceil \log_2\!\big(2 + \frac{1}{2\varepsilon}\big)\big\rceil$ counting qubits. For $n = 5$ and $\varepsilon = 0.1$, $t$ equals""",
         opts=[r"$8$", r"$7$", r"$9$", r"$10$"],
         sol=r"""$2 + \frac{1}{2\times0.1} = 2 + 5 = 7$ and $\log_2 7 \approx 2.807$, so $\lceil\log_2 7\rceil = 3$ and $t = 5 + 3 = 8$.
The extra 3 qubits make the outcome distribution (whose peak has width about one grid point $2^{-t}$, with tails
$\propto 1/\delta^2$) concentrated enough that the first 5 bits are right with probability $\ge 0.9$."""),

    dict(id="D5", kind="mcq",
         q=r"""Phase estimation with $t = 3$ counting qubits (inverse QFT readout) is run on an eigenstate with $\varphi = \frac58$.
The probability that the counting register reads $101$ is""",
         opts=[r"$1$", r"$\dfrac18$", r"$\dfrac{4}{\pi^2}$", r"$\dfrac58$"],
         sol=r"""After the controlled-$U^{2^j}$ gates the counting register holds
$\frac{1}{\sqrt8}\sum_{k=0}^{7}e^{2\pi i\varphi k}\ket{k} = \frac{1}{\sqrt8}\sum_k\omega^{5k}\ket k = \mathrm{QFT}\ket{5}$ ($\omega = e^{2\pi i/8}$),
because $\varphi = \frac58 = 0.101_2$ is exactly representable with 3 bits. The inverse QFT returns $\ket{5} = \ket{101}$ with
certainty. The bound $\frac{4}{\pi^2}$ applies only when $\varphi$ is \emph{not} a multiple of $2^{-t}$."""),

    dict(id="D6", kind="mcq",
         q=r"""If $\varphi$ is not exactly a $t$-bit binary fraction, phase estimation returns the \emph{single} closest $t$-bit
approximation with probability at least""",
         opts=[r"$\dfrac{4}{\pi^2} \approx 0.405$", r"$\dfrac12$", r"$\dfrac{8}{\pi^2} \approx 0.81$", r"$\dfrac1\pi \approx 0.318$"],
         sol=r"""Let $b$ be the closest grid point and $\delta = \varphi - b/2^t$, with $0 < |\delta| \le 2^{-(t+1)}$. The amplitude of $b$ is
\[ \alpha_b = \frac{1}{2^t}\sum_{k=0}^{2^t-1}e^{2\pi i k\delta}, \qquad |\alpha_b| = \frac{|\sin(\pi 2^t\delta)|}{2^t|\sin(\pi\delta)|}. \]
Since $\pi2^t|\delta| \le \frac\pi2$, use $\sin x \ge \frac{2x}{\pi}$ there (concavity) in the numerator and $\sin x \le x$ in the
denominator: $|\alpha_b| \ge \frac{2\cdot2^t|\delta|}{2^t\pi|\delta|} = \frac{2}{\pi}$, so $P(b) \ge \frac{4}{\pi^2} \approx 0.405$.
($\frac{8}{\pi^2}$ is the bound for getting \emph{one of the two} nearest grid points.)"""),

    dict(id="D7", kind="mcq",
         q=r"""Phase estimation with $t = 2$ counting qubits is run with $\varphi = \frac13$. The probability of the outcome $y = 1$
(i.e.\ the estimate $\varphi \approx \frac14$) is""",
         opts=[r"$\dfrac{6 + 3\sqrt3}{16}$", r"$\dfrac{6 - 3\sqrt3}{16}$", r"$\dfrac13$", r"$\dfrac{4}{\pi^2}$"],
         sol=r"""$P(y) = \frac{1}{16}\left|\sum_{k=0}^{3}e^{2\pi i k\delta}\right|^2 = \frac{1}{16}\,\frac{\sin^2(4\pi\delta)}{\sin^2(\pi\delta)}$ with
$\delta = \varphi - \frac{y}{4}$. For $y = 1$: $\delta = \frac13 - \frac14 = \frac{1}{12}$, so
\[ P(1) = \frac{1}{16}\,\frac{\sin^2(\pi/3)}{\sin^2(\pi/12)} = \frac{1}{16}\cdot\frac{3/4}{(1 - \cos\frac\pi6)/2} = \frac{1}{16}\cdot\frac{3}{2 - \sqrt3}
   = \frac{3(2 + \sqrt3)}{16} = \frac{6 + 3\sqrt3}{16} \approx 0.700 . \]
For completeness $P(0) = \frac{1}{16}$, $P(2) = \frac{3}{16}$, $P(3) \approx 0.050$; the four add to $1$. It exceeds the guaranteed
$\frac{4}{\pi^2} \approx 0.405$ because $\frac13$ is not at the worst position (midway between grid points)."""),

    dict(id="D8", kind="mcq",
         q=r"""For Shor's algorithm with a work register of $n = \lceil\log_2 N\rceil$ qubits and a counting register of $2n$ qubits,
the total number of qubits needed to factor $N = 21$ is""",
         opts=[r"$15$", r"$10$", r"$21$", r"$12$"],
         sol=r"""$\log_2 21 \approx 4.39$, so $n = 5$ (5 bits hold the residues $0, \dots, 20$). Counting register $2n = 10$ qubits, work
register $5$ qubits: total $15$. The $2n$ counting qubits make $M = 2^{2n} \ge N^2$, which guarantees that each peak
$y/M$ is within $\frac{1}{2r^2}$ of some $k/r$, so continued fractions can recover $r$. (The passage uses exactly these
$10 + 5$ qubits.)"""),

    dict(id="D9", kind="mcq",
         q=r"""Order finding for $a = 7$, $N = 15$ uses a counting register of $t = 8$ qubits ($M = 256$). The set of possible
measurement outcomes $y$ is""",
         opts=[r"$\{0, 64, 128, 192\}$", r"$\{0, 4, 8, 12\}$", r"$\{0, 128\}$", r"$\{0, 85, 171\}$"],
         sol=r"""Powers of 7 mod 15: $7, 49 \equiv 4, 28 \equiv 13, 91 \equiv 1$, so $r = 4$. After measuring (or ignoring) the work register,
the counting register is periodic with period 4: $\frac{1}{8}\sum_{j=0}^{63}\ket{x_0 + 4j}$. Since $4 \mid 256$, by D1 the
outcomes are exactly the multiples of $M/r = 64$: $y \in \{0, 64, 128, 192\}$, each with probability $\frac14$, with no
leakage. ($\{0,85,171\}$ would correspond to $r = 3$.)"""),

    dict(id="D10", kind="mcq",
         q=r"""In the setting of the previous question (order finding for $a = 7$, $N = 15$, $M = 256$), the continued-fraction step
turns $y/256$ into $k/r'$. Which outcome leads to the \emph{wrong} candidate $r' = 2$?""",
         opts=[r"$y = 128$", r"$y = 64$", r"$y = 192$", r"$y = 0$"],
         sol=r"""$\frac{64}{256} = \frac14 \Rightarrow r' = 4$ \checkmark; $\frac{192}{256} = \frac34 \Rightarrow r' = 4$ \checkmark;
$\frac{128}{256} = \frac12 \Rightarrow r' = 2$; $\frac{0}{256} = \frac01 \Rightarrow r' = 1$ (no information).
For $y = 128$, $k = 2$ shares the factor 2 with $r = 4$, so the fraction reduces and only a \emph{divisor} of $r$ appears.
The check $7^2 = 49 \equiv 4 \not\equiv 1 \pmod{15}$ rejects it; one repeats the run or takes the lcm of several candidates.
With the correct $r = 4$: $\gcd(7^2 - 1, 15) = \gcd(48, 15) = 3$, $\gcd(7^2 + 1, 15) = \gcd(50, 15) = 5$."""),

    dict(id="D11", kind="mcq",
         q=r"""An approximate QFT on $n = 20$ qubits keeps only the controlled-$R_k$ gates with $k \le m = 6$ and drops the rest.
The number of controlled rotations dropped is""",
         opts=[r"$105$", r"$85$", r"$95$", r"$114$"],
         sol=r"""In the textbook circuit the gate $R_k$ (for $k = 2, \dots, n$) appears $n - k + 1$ times (once for each target qubit $j$
with $j + k - 1 \le n$). Total $= \binom{20}{2} = 190$. Kept: $k = 2, \dots, 6$: $19 + 18 + 17 + 16 + 15 = 85$. Dropped:
$190 - 85 = 105$. The approximate QFT therefore needs only $O(nm)$ gates instead of $O(n^2)$, at the price of a small error
(D17)."""),

    dict(id="D12", kind="mcq",
         q=r"""On $n = 6$ qubits ($N = 64$) the uniform superposition of all $x \equiv 0 \pmod 6$, $0 \le x \le 63$, is prepared and the
QFT applied. The probability of the outcome $y = 32$ is""",
         opts=[r"$\dfrac{11}{64}$", r"$\dfrac16$", r"$\dfrac{1}{64}$", r"$\dfrac{1}{11}$"],
         sol=r"""The allowed $x$ are $0, 6, 12, \dots, 60$: $K = 11$ terms ($6\times10 = 60 \le 63 < 66$). Then
\[ \beta_{32} = \frac{1}{\sqrt{11}}\cdot\frac{1}{\sqrt{64}}\sum_{j=0}^{10}e^{2\pi i\,6j\cdot32/64} = \frac{1}{\sqrt{704}}\sum_{j=0}^{10}e^{2\pi i\cdot3j}
   = \frac{11}{\sqrt{704}}, \]
so $P(32) = \frac{121}{704} = \frac{11}{64} \approx 0.172$. The period $6$ does not divide $64$, so the ideal answer $\frac16$ is
only approximate, and most peaks ($y \approx 10.67k$) are smeared over neighbouring integers; $y = 32 = 3\cdot\frac{64}{6}$
happens to fall exactly on a grid point."""),

    dict(id="D13", kind="mcq",
         q=r"""Every eigenvalue $\lambda$ of the $n$-qubit QFT matrix ($n \ge 1$) satisfies""",
         opts=[r"$\lambda \in \{1, -1, i, -i\}$", r"$\lambda^{N} = 1$, and all $N$ of the $N$-th roots of unity occur",
               r"$\lambda \in \{1, -1\}$", r"$|\lambda| < 1$ for at least one eigenvalue"],
         sol=r"""From C8, $\mathrm{QFT}^2\ket x = \ket{-x}$, hence $\mathrm{QFT}^4 = I$. If $F\mathbf v = \lambda\mathbf v$ then
$\mathbf v = F^4\mathbf v = \lambda^4\mathbf v$, so $\lambda^4 = 1$: $\lambda \in \{1, -1, i, -i\}$. The QFT is unitary, so $|\lambda| = 1$
always. Imaginary eigenvalues do occur (e.g.\ $i$ for $N = 4$, where $\operatorname{tr}F = 1 + i$), so $\{1, -1\}$ is too small;
for $N \ge 8$ all four values occur, but never all $N$ roots of unity once $N > 4$."""),

    dict(id="D14", kind="mcq",
         q=r"""In phase estimation the inverse QFT is immediately followed by measurement of every counting qubit. Using the
semiclassical (Griffiths--Niu) QFT, the number of \emph{two-qubit} gates needed for this inverse QFT on $n$ qubits is""",
         opts=[r"$0$", r"$n - 1$", r"$\dfrac{n(n-1)}{2}$", r"$n$"],
         sol=r"""Controlled-phase gates are symmetric (B7), so in each $\mathrm{C}R_k^\dagger$ the qubit that is measured \emph{first} can be
taken as the control. Once a qubit has been measured, its value $y_j$ is a classical bit, and ``controlled on a measured
qubit'' becomes ``apply $R_k^\dagger$ if $y_j = 1$'': a classically controlled single-qubit gate. Measuring the qubits one
after the other and feeding each result forward removes all $\frac{n(n-1)}{2}$ two-qubit gates. This needs only
mid-circuit measurement plus feed-forward, and even allows one counting qubit to be recycled $n$ times."""),

    dict(id="D15", kind="match",
         q=r"""Match each algorithm in List-I with the role played by the QFT, or the related step, in List-II.""",
         match=([r"Quantum phase estimation", r"Shor's factoring algorithm", r"Draper's quantum adder", r"HHL linear-system solver"],
                [r"adds a number by applying controlled phase rotations to a register held in the Fourier basis",
                 r"turns the phases $e^{2\pi i\,2^j\varphi}$ in the counting register into the binary digits of $\varphi$",
                 r"after a QPE step, applies an ancilla rotation proportional to $1/\lambda$ for each eigenvalue $\lambda$",
                 r"extracts an order $r$ from peaks near $kM/r$ using continued fractions"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3",
               r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1", r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3"],
         sol=r"""QPE: controlled-$U^{2^j}$ writes $\frac{1}{\sqrt{2^t}}\sum_k e^{2\pi i\varphi k}\ket k \approx \mathrm{QFT}\ket{2^t\varphi}$; the inverse QFT reads
the digits (2). Shor: QPE applied to modular multiplication, whose eigenphases are $k/r$; continued fractions give $r$ (4).
Draper: in the Fourier basis, adding $a$ to $\ket{x}$ is just the phase ramp $\omega^{ay}$ (D3), done with single-qubit or
controlled rotations, then an inverse QFT (1). HHL: QPE puts eigenvalues $\lambda_j$ in a register, a controlled rotation
of an ancilla by an angle with $\sin\theta \propto 1/\lambda_j$ performs $A^{-1}$, then QPE is uncomputed (3).
Hence P$\to$2, Q$\to$4, R$\to$1, S$\to$3."""),

    dict(id="D16", kind="match",
         q=r"""On $n = 4$ qubits ($N = 16$) the QFT is applied to each input in List-I and the result measured. Match with List-II.""",
         match=([r"$\ket{0}$", r"$\frac12\big(\ket0 + \ket4 + \ket8 + \ket{12}\big)$", r"$\frac{1}{\sqrt8}\sum_{x\ \mathrm{even}}\ket{x}$",
                 r"$\frac14\sum_{x=0}^{15}\ket{x}$"],
                [r"only $y = 0$ and $y = 8$, each with probability $\frac12$", r"all 16 outcomes, each with probability $\frac{1}{16}$",
                 r"only $y = 0, 4, 8, 12$, each with probability $\frac14$", r"$y = 0$ with certainty"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$2, Q$\to$1, R$\to$3, S$\to$4", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1"],
         sol=r"""P: a basis state transforms to equal-magnitude amplitudes (C2): uniform over 16 (2).
Q: period $r = 4$ dividing 16: peaks at multiples of $16/4 = 4$, i.e.\ $0, 4, 8, 12$, each $\frac14$ (3).
R: the even numbers have period $r = 2$: peaks at multiples of $16/2 = 8$, i.e.\ $0$ and $8$, each $\frac12$ (1).
S: the uniform superposition has period 1 and goes to $\ket0$ (C9) (4).
Pattern: period $r$ in $x$ $\Leftrightarrow$ $r$ peaks spaced $N/r$ in $y$ --- the ``uncertainty principle'' of the Fourier
transform. Hence P$\to$2, Q$\to$3, R$\to$1, S$\to$4."""),

    dict(id="D17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} For large $n$, dropping every controlled-$R_k$ with $k > m$, where $m$ is only slightly larger than
$\log_2 n$, changes the output of the QFT circuit only slightly.\\[1mm]
\textbf{Reason (R):} Each dropped gate differs from the identity only by a phase $2\pi/2^{k}$, so its error is at most about
$2\pi/2^{k}$; at most $n$ gates are dropped for each $k$, giving a total error of order $n\cdot 2\pi/2^{m}$, which is small once
$2^m \gg n$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $\|\mathrm{C}R_k - I\| = |e^{2\pi i/2^k} - 1| = 2\sin(\pi/2^k) \le 2\pi/2^k$. Errors of a product of unitaries add
at most linearly (triangle inequality), and $R_k$ appears $n - k + 1 \le n$ times, so the total error is at most
$\sum_{k > m} n\cdot\frac{2\pi}{2^k} < \frac{2\pi n}{2^{m}}$. \textbf{A is true} \emph{because} of R: choosing $m = \log_2 n + c$ makes the error
at most $2\pi/2^{c}$, independent of $n$. (Coppersmith's approximate QFT; in practice gate noise is larger than the
error from these tiny rotations, so dropping them even \emph{improves} real results.) Answer: (A)."""),

    dict(id="D18", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} For $\varphi = \frac13$, phase estimation with any finite number $t$ of counting qubits never returns an
outcome $y$ with $y/2^t = \varphi$ exactly, with certainty or otherwise.\\[1mm]
\textbf{Statement II:} The number $\frac13$ has no terminating binary expansion ($\frac13 = 0.010101\ldots_2$).""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{II is true:} $\frac13 = \frac14 + \frac1{16} + \frac1{64} + \dots = 0.\overline{01}_2$; a terminating expansion would mean
$\frac13 = \frac{y}{2^t}$, i.e.\ $2^t = 3y$, impossible since $3 \nmid 2^t$. \textbf{I is true:} the outcome is always an integer
$y$, so the estimate is $y/2^t \ne \frac13$ for every $t$ by the same argument. The output distribution is peaked near
$2^t/3$ (see D7: $t = 2$ gives $y = 1$ with probability $\approx 0.70$). Both statements are true. Answer: (A)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""Which of the following measured values $y$ leads the post-processing to the correct order $r = 6$?""",
         opts=[r"$171$", r"$341$", r"$512$", r"$683$"],
         sol=r"""The peaks lie near $y \approx k\cdot\frac{1024}{6} = 170.67k$: $0, 170.67, 341.33, 512, 682.67, 853.33$.
\begin{itemize}\itemsep0pt
\item $\frac{171}{1024} = 0.16699$: continued fraction $[0;\,5,\,1,\,84,\,2]$, convergents $\frac15, \frac16, \frac{85}{509}, \frac{171}{1024}$; the last with
      denominator $< 21$ is $\frac16$ $\Rightarrow$ $r = 6$, and $2^6 \equiv 1$ \checkmark.
\item $\frac{341}{1024} \approx \frac13$ ($k = 2$), $\frac{512}{1024} = \frac12$ ($k = 3$), $\frac{683}{1024} \approx \frac23$ ($k = 4$): $\gcd(k, 6) > 1$,
      so only the divisors $3, 2, 3$ appear and $2^3 = 8$, $2^2 = 4$ are $\not\equiv 1 \pmod{21}$.
\end{itemize}
Factoring: $2^{r/2} = 8$, $\gcd(8 - 1, 21) = 7$ and $\gcd(8 + 1, 21) = 3$; indeed $21 = 3\times7$. Only $k = 1, 5$ (coprime to 6)
give $r$ directly, i.e.\ about $\frac{\varphi(6)}{6} = \frac13$ of the runs."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""The probability that the measured value is exactly $y = 512$ is closest to""",
         opts=[r"$0.167$", r"$0.333$", r"$0.500$", r"$0.001$"],
         sol=r"""After step (ii) the state is $\frac{1}{\sqrt{1024}}\sum_{x=0}^{1023}\ket{x}\ket{2^x \bmod 21}$. Group $x = x_0 + 6j$ with $x_0 \in \{0, \dots, 5\}$;
because $1024 = 6\cdot170 + 4$, the offsets $x_0 = 0,1,2,3$ have $K = 171$ terms and $x_0 = 4,5$ have $K = 170$ terms
(check: $4\cdot171 + 2\cdot170 = 1024$). The six work values $2^{x_0} \bmod 21$ are distinct, so the branches do not interfere. After
the inverse QFT, the amplitude of $\ket{y}$ in branch $x_0$ is $\frac{1}{1024}\sum_{j}e^{-2\pi i(x_0 + 6j)y/1024}$. For $y = 512$,
$e^{-2\pi i\cdot6j\cdot512/1024} = e^{-6\pi i j} = 1$, so its modulus is $K_{x_0}/1024$ and
\[ P(512) = \frac{4\cdot171^2 + 2\cdot170^2}{1024^2} = \frac{174\,764}{1\,048\,576} \approx 0.16667 \approx \frac16 . \]
$y = 512 = 3\cdot\frac{1024}{6}$ falls exactly on a grid point, so this peak is ``perfect''; the peaks at $k = 1, 2, 4, 5$ are smeared
over neighbouring $y$ (the nearest integers $171, 341, 683, 853$ get only about $0.11$ each)."""),
]
