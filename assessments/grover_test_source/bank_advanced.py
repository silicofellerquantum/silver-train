"""Section D - Advanced Grover (eigenstructure, unknown M, exact and fixed-point search, amplitude amplification, counting, limits).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""\textbf{Quantum counting.} A search space of $N = 64$ items contains an unknown number $M$ of marked items; in fact $M = 16$. The Grover operator $G$ has
eigenvalues $e^{\pm 2i\theta}$ on the plane containing $\ket{s}$, with $\sin^2\theta = M/N$. Phase estimation of $G$ with $t = 3$ counting qubits is run on $\ket{s}$, which has equal weight on the two eigenvectors. The outcome $y \in \{0, \dots, 7\}$ gives the
phase estimate $\varphi \approx y/8$, where $e^{2\pi i\varphi}$ is the eigenvalue. From this one sets $\tilde\theta = \pi y/8$ and $\tilde M = N\sin^2\tilde\theta$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""For $N = 1024$ with $M = 16$ marked items, the optimal number of iterations is""",
         opts=[r"$6$", r"$25$", r"$5$", r"$8$"],
         sol=r"""$\sin\theta = \sqrt{16/1024} = \frac18$, so $\theta = 0.12533$ rad. Then $\frac{\pi}{4\theta} - \frac12 = 6.267 - 0.5 = 5.767$, which rounds to $6$. The success probability is $\sin^2 13\theta \approx 0.9966$.
($25$ is the value for one marked item; $8 = \sqrt{N/M}$ forgets $\frac\pi4$.)"""),

    dict(id="D2", kind="mcq",
         q=r"""When $M$ is unknown, the algorithm of Boyer, Brassard, H{\o}yer and Tapp picks $k$ uniformly from $\{0, \dots, m-1\}$. The average success probability is at least $\frac14$ once $m \ge 1/\sin2\theta$.
For $N = 2^{20}$ and $M = 1$ this threshold on $m$ is about""",
         opts=[r"$512$", r"$1024$", r"$804$", r"$2^{20}$"],
         sol=r"""$\sin\theta = 2^{-10}$ is small, so $\sin 2\theta \approx 2\sin\theta = 2^{-9}$ and $\frac{1}{\sin2\theta} \approx 2^{9} = 512$. In general the threshold is about $\frac12\sqrt{N/M}$, which comes from the identity
$\frac1m\sum_{k=0}^{m-1}\sin^2((2k+1)\theta) = \frac12 - \frac{\sin 4m\theta}{4m\sin2\theta}$; the last term is at most $\frac14$ in size once $m \ge 1/\sin2\theta$."""),

    dict(id="D3", kind="mcq",
         q=r"""To obtain success probability exactly $1$, the angle is reduced from $\theta$ to $\theta'$ so that $(2k+1)\theta' = \pi/2$ with $k = \lceil\frac{\pi}{4\theta} - \frac12\rceil$. For $N = 16$, $M = 1$ this requires""",
         opts=[r"$k = 3$ and $\theta' = \pi/14$", r"$k = 3$ and $\theta' = \pi/12$", r"$k = 2$ and $\theta' = \pi/10$", r"$k = 4$ and $\theta' = \pi/18$"],
         sol=r"""$\frac{\pi}{4\theta} - \frac12 = 2.608$, so the ceiling is $k = 3$. Then $7\theta' = \frac\pi2$ gives $\theta' = \frac{\pi}{14} \approx 0.2244$, slightly below $\theta \approx 0.2527$ as required.
One way to reduce the angle is to mark only with amplitude $\sin\theta'/\sin\theta$ by adding an ancilla rotation. The ceiling matters: rounding down to $k = 2$ would need $\theta' = \frac{\pi}{10} > \theta$, which cannot be reached by reducing the marked weight."""),

    dict(id="D4", kind="mcq",
         q=r"""Which statement about the query complexity of unstructured search is correct?""",
         opts=[r"Every quantum algorithm needs $\Omega(\sqrt N)$ oracle queries, so Grover's algorithm is optimal up to a constant",
               r"Quantum algorithms need only $O(\log N)$ queries, because $n = \log_2 N$ qubits hold all $N$ items",
               r"The lower bound $\Omega(\sqrt N)$ holds only for classical algorithms",
               r"Grover's algorithm is optimal only when $N$ is a power of $2$"],
         sol=r"""The lower bound of Bennett, Bernstein, Brassard and Vazirani states that any quantum algorithm with black-box access needs $\Omega(\sqrt N)$ queries; Zalka showed that $\frac\pi4\sqrt N$ is optimal even including the constant.
Superposition of $N$ items does not give $O(\log N)$ search, since a measurement returns only one string and the marked amplitude grows slowly."""),

    dict(id="D5", kind="mcq",
         q=r"""The search space of size $N$ is split evenly among $16$ independent quantum computers, each running Grover on its share. Compared with one computer on the whole space, the wall-clock time is divided by""",
         opts=[r"$4$, and the total number of oracle queries is multiplied by $4$",
               r"$16$, and the total number of oracle queries is unchanged",
               r"$4$, and the total number of oracle queries is unchanged",
               r"$16$, and the total number of oracle queries is multiplied by $4$"],
         sol=r"""Each computer searches $N/16$ items: time $\frac\pi4\sqrt{N/16} = \frac14\cdot\frac\pi4\sqrt N$, so the time falls by $\sqrt{16} = 4$. Total queries: $16\times\frac14 = 4$ times the single-computer count.
Grover parallelises only with a factor $\sqrt p$, in contrast to the factor $p$ for classical search."""),

    dict(id="D6", kind="mcq",
         q=r"""Amplitude amplification boosts a procedure $\mathcal{A}$ that succeeds with probability $a = 0.01$ ($\theta \approx 0.10017$). At the optimal $k$, one uses $\mathcal{A}$ or $\mathcal{A}^{-1}$ in total""",
         opts=[r"$15$ times", r"$7$ times", r"$100$ times", r"$8$ times"],
         sol=r"""Optimal $k$: $\frac{\pi}{4\theta} - \frac12 = 7.34$, so $k = 7$. The initial state uses $\mathcal{A}$ once, and each iteration $Q = -\mathcal{A}S_0\mathcal{A}^{-1}S_\chi$ uses $\mathcal{A}$ and $\mathcal{A}^{-1}$ once each. Total $2k + 1 = 15$ uses.
Classical repetition needs about $1/a = 100$ runs on average."""),

    dict(id="D7", kind="mcq",
         q=r"""For $N = 16$, $M = 4$, the phases $\varphi \in [0, 1)$ of the eigenvalues $e^{2\pi i\varphi}$ of $G$ on the plane containing $\ket{s}$ are""",
         opts=[r"$\dfrac16$ and $\dfrac56$", r"$\dfrac1{12}$ and $\dfrac{11}{12}$", r"$\dfrac13$ and $\dfrac23$", r"$\dfrac14$ and $\dfrac34$"],
         sol=r"""$\sin^2\theta = \frac14$ gives $\theta = \frac\pi6$. $G$ rotates by $2\theta$, so its eigenvalues are $e^{\pm 2i\theta} = e^{\pm i\pi/3}$. Writing $e^{2\pi i\varphi}$: $\varphi = \frac{\theta}{\pi} = \frac16$, and $1 - \frac16 = \frac56$.
$\frac{1}{12}$ confuses $\theta$ with $2\pi\varphi$."""),

    dict(id="D8", kind="mcq",
         q=r"""A cipher with a $128$-bit key is attacked by Grover's algorithm (one correct key, cost measured in iterations). The number of iterations is about""",
         opts=[r"$\frac\pi4\cdot2^{64}$", r"$2^{127}$", r"$\frac\pi4\cdot2^{127}$", r"$128$"],
         sol=r"""$N = 2^{128}$, so $k \approx \frac\pi4\sqrt N = \frac\pi4\cdot 2^{64}$. The classical average is $2^{127}$. Doubling the key length restores the original margin.
Each iteration also has to run the cipher reversibly, so the real cost is much larger than the iteration count."""),

    dict(id="D9", kind="mcq",
         q=r"""A simple noise model multiplies the signal by $f$ per iteration: $P_k \approx f^{k}\sin^2((2k+1)\theta) + (1 - f^{k})/N$. For $n = 10$, $k = 25$, $f = 0.99$ (take $0.99^{25} \approx 0.78$), the success probability is about""",
         opts=[r"$0.78$", r"$0.99$", r"$0.25$", r"$0.50$"],
         sol=r"""For $n = 10$, $k = 25$ is optimal and $\sin^2(51\theta) \approx 0.9995$, while $(1 - f^k)/N \approx 0.22/1024$ is negligible. So $P \approx 0.78\times 0.9995 \approx 0.78$. Under this model, a depth of $k$ iterations costs a factor $f^k$;
since $k \propto \sqrt N$, this decay is what limits large searches on noisy hardware."""),

    dict(id="D10", kind="mcq",
         q=r"""A multi-controlled $Z$ with $m$ controls can be built with $m - 1$ clean ancillas, $2(m-1)$ Toffoli gates and one CZ. For the $n = 10$ Grover register ($m = 9$), one full iteration (oracle and diffusion) needs how many Toffoli gates?""",
         opts=[r"$32$", r"$16$", r"$18$", r"$36$"],
         sol=r"""One multi-controlled $Z$ with $m = 9$: $2(9 - 1) = 16$ Toffolis. An iteration has two of them, one in the oracle and one in the diffusion, so $32$ Toffolis. The gate count per iteration is linear in $n$."""),

    dict(id="D11", kind="mcq",
         q=r"""On a state-vector simulator the work grows as (iterations) $\times$ (amplitudes per gate) $\sim 2^{n/2}\cdot2^{n}$. Adding $2$ qubits multiplies the total work by about""",
         opts=[r"$8$", r"$4$", r"$16$", r"$2\sqrt2$"],
         sol=r"""Work $\propto 2^{1.5n}$, so $n \to n + 2$ gives $2^{1.5\times2} = 2^3 = 8$: amplitudes grow by $4$ and iterations by $2$. The number of gates per iteration also grows slowly with $n$, which this estimate ignores."""),

    dict(id="D12", kind="mcq",
         q=r"""In the basis $\{\ket{r}, \ket{m}\}$ of the Grover plane, $G = \begin{psmallmatrix}\cos2\theta & -\sin2\theta\\ \sin2\theta & \cos2\theta\end{psmallmatrix}$. Eigenvectors of $G$ are""",
         opts=[r"$\frac{1}{\sqrt2}\big(\ket{r} \mp i\ket{m}\big)$ with eigenvalues $e^{\pm2i\theta}$",
               r"$\frac{1}{\sqrt2}\big(\ket{r} \pm \ket{m}\big)$ with eigenvalues $e^{\pm2i\theta}$",
               r"$\ket{r}$ and $\ket{m}$ with eigenvalues $e^{\pm2i\theta}$",
               r"$\frac{1}{\sqrt2}\big(\ket{r} \mp i\ket{m}\big)$ with eigenvalues $\pm1$"],
         sol=r"""Apply $G$ to $(1, -i)^{T}$: the first entry is $\cos2\theta + i\sin2\theta = e^{2i\theta}$; the second is $\sin2\theta - i\cos2\theta = -i\,e^{2i\theta}$. So $(1, -i)^{T}$ has eigenvalue $e^{+2i\theta}$, and by conjugation $(1, +i)^{T}$ has $e^{-2i\theta}$.
These eigenvectors are complex: a rotation by an angle other than $0$ or $\pi$ has no real eigenvector."""),

    dict(id="D13", kind="mcq",
         q=r"""In Grover's fixed-point ($\pi/3$) search, one recursion step turns a procedure with failure probability $\varepsilon$ into one with failure probability $\varepsilon^3$. Starting from $\varepsilon = 0.1$, two steps give""",
         opts=[r"$10^{-9}$", r"$10^{-3}$", r"$10^{-6}$", r"$10^{-27}$"],
         sol=r"""After one step $\varepsilon_1 = (0.1)^3 = 10^{-3}$; after the second $\varepsilon_2 = (10^{-3})^3 = 10^{-9}$. In general $\varepsilon_j = \varepsilon^{3^j}$. Unlike ordinary Grover, additional steps never reduce the success probability, so the method cannot over-rotate; the price is that it loses the quadratic speed-up in this basic form."""),

    dict(id="D14", kind="mcq",
         q=r"""Let $N = 8$, $M = 1$, and repeat a run with a fixed $k$ until the marked item is found, counting only oracle queries in the Grover iterations. Then $P_1 = \frac{25}{32}$ and $P_2 = \frac{121}{128}$. The expected numbers of queries for $k = 1$ and $k = 2$ are about""",
         opts=[r"$1.28$ and $2.12$", r"$1$ and $2$", r"$0.78$ and $1.89$", r"$2.12$ and $1.28$"],
         sol=r"""The number of runs is geometric with mean $1/P_k$, and each run costs $k$ queries, so the expected cost is $k/P_k$. $k = 1$: $\frac{32}{25} = 1.28$. $k = 2$: $2\times\frac{128}{121} \approx 2.12$. So $k = 1$ is cheaper here,
although $k = 2$ is the value that maximises the success probability of a single run: the two criteria are different."""),

    dict(id="D15", kind="match",
         q=r"""Match each task in List-I with its query complexity in List-II.""",
         match=([r"Unstructured search, classical", r"Unstructured search, quantum", r"Collision finding in a 2-to-1 function, quantum",
                 r"Element distinctness, quantum"],
                [r"$\Theta(N^{1/3})$", r"$\Theta(N^{2/3})$", r"$\Theta(N)$", r"$\Theta(N^{1/2})$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""Classical search must examine a constant fraction of the items: $\Theta(N)$ (3). Grover: $\Theta(N^{1/2})$ (4). Brassard--H{\o}yer--Tapp collision finding, optimal by Aaronson--Shi: $\Theta(N^{1/3})$ (1). Ambainis's quantum walk for element distinctness: $\Theta(N^{2/3})$ (2).
Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="D16", kind="match",
         q=r"""Match each method in List-I with its main idea in List-II.""",
         match=([r"Amplitude amplification", r"Quantum counting", r"BBHT search", r"Fixed-point search"],
                [r"uses randomly chosen $k$ from a growing range when $M$ is unknown", r"changes the reflection phases so that extra steps cannot over-rotate",
                 r"replaces $H^{\otimes n}$ by any state-preparation procedure $\mathcal{A}$", r"applies phase estimation to $G$ to find $M$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""Amplitude amplification generalises $\ket{s}$ to $\mathcal{A}\ket{0}$ (3). Quantum counting reads $\theta$ from the eigenvalues $e^{\pm2i\theta}$ of $G$, which yields $M$ (4). BBHT draws $k$ at random from a range that grows geometrically (1).
Fixed-point search uses reflections with phase $\pi/3$ and similar choices rather than $\pi$ (2). Hence P$\to$3, Q$\to$4, R$\to$1, S$\to$2."""),

    dict(id="D17", kind="ar", correct=3, opts=AR_OPTIONS,
         q=r"""\textbf{Assertion (A):} Grover's algorithm solves NP-complete problems in polynomial time.\\[1mm]
\textbf{Reason (R):} For a search over $2^n$ candidates, Grover's algorithm uses about $\frac\pi4 2^{n/2}$ iterations.""",
         sol=r"""\textbf{R is true:} this is $\frac\pi4\sqrt N$ with $N = 2^n$. \textbf{A is false:} $2^{n/2}$ is still exponential in $n$. The speed-up is quadratic, and the black-box lower bound shows that generic search cannot do better. Answer: (D)."""),

    dict(id="D18", kind="stmt", correct=0, opts=STATEMENT_OPTIONS,
         q=r"""\textbf{Statement I:} The BBHT algorithm finds a marked item with an expected $O(\sqrt{N/M})$ queries without knowing $M$ in advance.\\[1mm]
\textbf{Statement II:} If exactly half of the items are marked, ordinary Grover iterations leave the success probability at $\frac12$.""",
         sol=r"""\textbf{I is true:} a growing range for the random $k$ reaches $m \sim \sqrt{N/M}$ after a geometric series of attempts, whose total cost is dominated by the final terms.
\textbf{II is true:} $\theta = 45^\circ$ and $\sin^2((2k+1)45^\circ) = \frac12$ for all $k$ (see C10). Answer: (A)."""),

    dict(id="D19", kind="para", passage="D", opts=[
             r"$y = 1$ and $y = 7$", r"$y = 2$ and $y = 6$", r"$y = 1$ and $y = 2$", r"$y = 0$ and $y = 4$"],
         q=r"""The two most likely outcomes are""",
         sol=r"""$\theta = \arcsin\sqrt{16/64} = \arcsin\frac12 = \frac\pi6$, so the eigenphases are $\varphi = \frac{\theta}{\pi} = \frac16$ and $\frac56$. On the grid $y/8$ these lie at $y = \frac86 = 1.33$ and $y = \frac{40}{6} = 6.67$; the closest integers are $1$ and $7$.
Each has probability about $0.35$. (The next most likely are $y = 2$ and $y = 6$, each about $0.09$.)"""),

    dict(id="D20", kind="para", passage="D", opts=[r"$\tilde M \approx 9.4$", r"$\tilde M = 16$", r"$\tilde M = 32$", r"$\tilde M = 8$"],
         q=r"""If $y = 1$ is obtained, the estimate of $M$ is""",
         sol=r"""$\tilde\theta = \frac\pi8$ and $\tilde M = 64\sin^2\frac\pi8 = 64\cdot\frac{1 - \cos(\pi/4)}{2} = 32\big(1 - \frac{\sqrt2}{2}\big) \approx 9.37$. The true value $16$ lies between the estimates from $y = 1$ ($9.4$) and $y = 2$ ($32$): three counting qubits are too few for an accurate count.
Each additional counting qubit halves the error in $\theta$."""),
]
