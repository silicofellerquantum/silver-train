"""Section D - Advanced Shor's Algorithm and Implementation (resource estimation and published estimates, qubit-saving
constructions, gate-level compilation, approximate QFT, noise, simulation memory, what the notebook can and cannot run,
RSA, elliptic curves, post-quantum cryptography and fault tolerance).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""A gate-level version of the order-finding circuit for $N = 15$, $a = 7$ uses $t = 3$ counting qubits and $4$ work qubits. The
multiplications by $7$ and $4 = 7^2 \bmod 15$ are compiled into controlled SWAPs (each CNOT--Toffoli--CNOT, Toffoli $=$ 6 CNOTs) and CNOTs; the
inverse QFT uses $3$ CNOTs (one SWAP) and $3$ controlled phases. In total the circuit has $127$ CNOTs. It was simulated with the CUDA-Q
\texttt{density-matrix-cpu} target and a depolarizing noise model: a one-qubit channel with parameter $p$ after every $H$, $R_z$ and $X$, and a
two-qubit channel with parameter $p$ after every CNOT and controlled phase. The probability that one shot yields the factors $3 \times 5$ was
(re-executed, exact from the density matrix):
\begin{center}\begin{tabular}{lccccc}\toprule
$p$ & $0$ & $0.001$ & $0.003$ & $0.01$ & $0.03$\\\midrule
success probability & $0.750$ & $0.740$ & $0.726$ & $0.703$ & $0.665$\\\bottomrule
\end{tabular}\end{center}
If the counting register were completely random (uniform $y$), the success probability would be $0.375$. These are simulator results, not
measurements on a quantum processor.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""Beauregard's circuit for Shor's algorithm uses $2n + 3$ logical qubits. For RSA-2048 ($n = 2048$) this is""",
         opts=[r"$4099$", r"$6147$", r"$2051$", r"$4096$"],
         sol=r"""$2\times2048 + 3 = 4099$. $6147 = 3\times2048 + 3$ resembles the $3n$ register count of \texttt{Shor.ipynb} ($6144$). These are \emph{logical}
(error-corrected) qubits; the physical count is orders of magnitude larger."""),

    dict(id="D2", kind="mcq",
         q=r"""Which pair of published resource estimates for factoring RSA-2048 is correct?""",
         opts=[r"Gidney and Ekerå (2019): about $20$ million noisy qubits in about $8$ hours; Gidney (2025): fewer than one million noisy qubits in less than a week",
               r"Gidney and Ekerå (2019): $4099$ physical qubits in one second",
               r"Gidney (2025): about $20$ billion qubits in about $8$ hours",
               r"both estimates state that RSA-2048 was factored on existing hardware"],
         sol=r"""Both are \emph{estimates} under explicit assumptions (surface code, physical error rate $10^{-3}$, $\mu$s cycle times), not experiments. Gidney--Ekerå 2019:
$\approx 20$ million physical qubits, $\approx 8$ hours. Gidney 2025 (approximate residue arithmetic, yoked surface codes, magic-state cultivation): $< 1$
million qubits, $< 1$ week. No cryptographically relevant number has been factored by Shor's algorithm."""),

    dict(id="D3", kind="mcq",
         q=r"""With schoolbook arithmetic, modular exponentiation costs $O(n^3)$ gates. Doubling the bit length $n$ of $N$ multiplies the gate count by about""",
         opts=[r"$8$", r"$2$", r"$4$", r"$2^{n}$"],
         sol=r"""$(2n)^3 = 8n^3$. The cost is polynomial: $t = 2n$ controlled multiplications, each $O(n^2)$ gates. Classical sub-exponential algorithms (GNFS) grow much faster,
which is the source of Shor's advantage --- but only on a fault-tolerant machine. $2^n$ is the exponential cost of trial division."""),

    dict(id="D4", kind="mcq",
         q=r"""In the ``semiclassical'' (one-control-qubit) implementation of the inverse QFT used by Beauregard, the $t = 2n$ counting qubits are replaced by""",
         opts=[r"a single qubit that is measured and reset $2n$ times; earlier measurement results classically control the phase corrections",
               r"$n$ qubits, because the QFT needs only half the bits",
               r"no qubits: the QFT is computed classically before the circuit",
               r"$2n$ ancilla qubits that are never measured"],
         sol=r"""The inverse QFT followed by measurement can be done bit by bit (Griffiths--Niu): measure, then apply classically controlled phases to the next qubit.
One control qubit is recycled, so the count drops from $3n$ to about $2n + 3$ ($n$ work, $n + 2$ ancilla for the Fourier-space adder, $1$ control). The
measurement statistics are unchanged."""),

    dict(id="D5", kind="mcq",
         q=r"""For $N = 21$, $a = 2$, $t = 10$, an approximate inverse QFT keeps only controlled phases between qubits at distance $k - j \le m$. With $m = 2$ the
number of controlled phases and the (re-executed, exact) success probability are""",
         opts=[r"$17$ phases; $0.689$ (full QFT: $45$ phases, $0.817$)", r"$45$ phases; $0.817$", r"$2$ phases; $0.689$", r"$17$ phases; $0.817$ --- truncation never changes the result"],
         sol=r"""Distance $1$: $9$ pairs, distance $2$: $8$ pairs, total $17$ (full QFT: $\binom{10}{2} = 45$). The re-executed simulation gave success $0.689$ for $m = 2$, $0.787$ for $m = 3$,
$0.811$ for $m = 4$ and $0.817$ for the full QFT. Very small cut-offs lose accuracy; $m \approx \log_2 n$ plus a few suffices asymptotically."""),

    dict(id="D6", kind="mcq",
         q=r"""The full inverse QFT on $t = 8$ counting qubits (notebook default for $N = 15$) contains how many controlled phase gates?""",
         opts=[r"$28$", r"$8$", r"$64$", r"$56$"],
         sol=r"""One for each pair $j < k$: $\binom82 = \frac{8\cdot7}{2} = 28$ --- the recorded gate count \texttt{cr1: 28}. In general $t(t - 1)/2 = O(t^2)$, which for
large $t$ is dwarfed by the $O(n^3)$ cost of the modular exponentiation."""),

    dict(id="D7", kind="mcq",
         q=r"""The recorded gate counts of \texttt{Shor.ipynb} for $N = 15$ are \texttt{x: 1, ccustom\_unitary\_constant: 2, swap: 4, h: 16, cr1: 28} (and
\texttt{mz: 8}). Why are there $16$ Hadamard gates?""",
         opts=[r"$8$ create the superposition on the counting qubits and $8$ belong to the inverse QFT",
               r"each controlled multiplication contains $8$ Hadamards",
               r"$16 = 2^4$ is the dimension of the work register",
               r"Hadamards are inserted before every measurement of the work register"],
         sol=r"""$t = 8$ Hadamards start the algorithm, and the inverse QFT applies one $H$ per qubit ($8$ more). The multipliers are single custom gates (no internal $H$),
and the work register is not measured. Total gates without \texttt{mz}: $1 + 2 + 4 + 16 + 28 = 51$, depth $20$."""),

    dict(id="D8", kind="mcq",
         q=r"""Why does \texttt{Shor.ipynb} refuse to run on hardware targets?""",
         opts=[r"its modular multipliers are exact custom unitaries (permutation matrices) registered with \texttt{cudaq.register\_operation}; hardware would need them decomposed into native gates, which the notebook does not do",
               r"Shor's algorithm is mathematically impossible on hardware",
               r"CUDA-Q has no hardware targets",
               r"the inverse QFT cannot be executed on hardware"],
         sol=r"""The notebook raises a \texttt{ValueError} for hardware and points to the resource estimates. A dense $2^n\times2^n$ matrix gate is fine for a simulator but has no
native implementation; compiling it costs $O(n^2)$--$O(n^3)$ gates per multiplier. The inverse QFT itself uses standard gates."""),

    dict(id="D9", kind="mcq",
         q=r"""In the gate-level circuit, ``multiply by $7$ modulo $15$'' is built as three left rotations of the 4 work bits (each $3$ SWAPs) followed by $X$ on all
four bits. Its controlled version, with controlled SWAPs of $8$ CNOTs and controlled-$X$ $=$ CNOT, uses how many CNOTs?""",
         opts=[r"$76$", r"$9$", r"$72$", r"$40$"],
         sol=r"""$9$ controlled SWAPs $\times\,8 = 72$ CNOTs, plus $4$ CNOTs for the controlled $X$ gates: $76$. (Multiplication by $7$ is valid on $y = 1, \dots, 14$:
$7y \bmod 15 = 15 - (y\cdot2^3 \bmod 15)$.) With the controlled $\times4$ ($6$ controlled SWAPs $= 48$) and the QFT SWAP ($3$) the circuit has $127$ CNOTs."""),

    dict(id="D10", kind="mcq",
         q=r"""In the gate-level circuit ($t = 3$, $a = 7$, $N = 15$) the counting outcome becomes uniformly random in the fully depolarized limit. The success
probability with the notebook's post-processing is then""",
         opts=[r"$3/8$: $y = 2, 4, 6$ give $r = 4$ and the factors; $y = 0$ gives nothing; odd $y$ give $r' = 8$, and $7^4 \equiv 1$ gives only trivial factors",
               r"$0$",
               r"$3/4$, the same as without noise",
               r"$1/8$"],
         sol=r"""$y/8$: $2/8 = 1/4$, $6/8 = 3/4$ give $4$; $4/8 = 1/2$ gives $4$ via multiples. For odd $y$ the denominator $8$ satisfies $7^8 \equiv 1$, but $7^{4} \equiv 1$, so
$\gcd(0, 15) = 15$, $\gcd(2, 15) = 1$ --- trivial. Hence $3/8 = 0.375$: a noisy run can still ``succeed'' by luck, which is why a success rate must be compared
with this random baseline."""),

    dict(id="D11", kind="mcq",
         q=r"""Using the notebook's $3n$-qubit construction, the state-vector memory (16 bytes per amplitude) for $N = 1023$ is about""",
         opts=[r"$17.2$ GB ($30$ qubits)", r"$16.4$ kB", r"$1.1$ TB ($36$ qubits)", r"$268$ MB ($24$ qubits)"],
         sol=r"""$1023$ has $10$ bits, so $3n = 30$ qubits: $16\times2^{30} = 17.18\times10^9$ bytes. This matches the notebook's reverse estimate that $80$ GB allows at most
$32$ qubits, i.e.\ $N \le 1023$. $N = 1024$ would need $11$ bits and $33$ qubits ($137$ GB)."""),

    dict(id="D12", kind="mcq",
         q=r"""Toy RSA: $p = 5$, $q = 11$, $e = 7$. The private exponent $d$ and the ciphertext of $m = 2$ are""",
         opts=[r"$d = 23$; $c = 18$", r"$d = 3$; $c = 18$", r"$d = 23$; $c = 14$", r"$d = 7$; $c = 2$"],
         sol=r"""$\varphi(55) = 4\cdot10 = 40$; $7d \equiv 1 \pmod{40}$: $7\cdot23 = 161 = 4\cdot40 + 1$, so $d = 23$. $c = 2^7 \bmod 55 = 128 - 110 = 18$; check $18^{23} \bmod 55 = 2$. Shor's
algorithm breaks RSA by factoring $N = 55$, which reveals $\varphi(N)$ and hence $d$."""),

    dict(id="D13", kind="mcq",
         q=r"""Which standard specifies NIST's lattice-based key-encapsulation mechanism intended to replace RSA/ECDH key exchange?""",
         opts=[r"FIPS 203 (ML-KEM)", r"FIPS 197 (AES)", r"FIPS 186 (DSA/ECDSA)", r"SP 800-90A (random bit generators)"],
         sol=r"""NIST published FIPS 203 (ML-KEM, from CRYSTALS-Kyber), FIPS 204 (ML-DSA, from Dilithium) and FIPS 205 (SLH-DSA, from SPHINCS+) in August 2024. AES (FIPS 197) is
symmetric and not broken by Shor; FIPS 186 contains the signature schemes Shor's algorithm does break."""),

    dict(id="D14", kind="mcq",
         q=r"""Which statement about the cryptographic impact of a large fault-tolerant quantum computer is correct?""",
         opts=[r"Shor's algorithm breaks RSA and elliptic-curve cryptography; AES is affected only by Grover's quadratic speed-up, so AES-256 remains adequate",
               r"Shor's algorithm breaks AES-256 in polynomial time",
               r"only RSA is affected; elliptic-curve cryptography is immune",
               r"hash functions such as SHA-256 are broken by Shor's algorithm"],
         sol=r"""Shor's period finding solves factoring and discrete logarithms in any abelian group, including elliptic-curve groups (ECDH, ECDSA). Symmetric ciphers and hash
functions have no such structure; Grover search gives at most a square-root speed-up, mitigated by doubling key length."""),

    dict(id="D15", kind="match",
         q=r"""Match each fault-tolerance term (List-I) with its meaning (List-II).""",
         match=([r"logical qubit", r"physical qubit", r"code distance $d$", r"magic state"],
                [r"minimum number of physical errors that can cause an undetected logical error",
                 r"a special resource state that enables non-Clifford gates such as $T$ or Toffoli in the surface code",
                 r"an error-corrected qubit encoded in many noisy qubits",
                 r"a noisy hardware qubit with error rates around $10^{-3}$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""Logical qubit: encoded, error-corrected (3); physical qubit: the noisy hardware element (4); code distance: the number of errors needed for an undetected
logical error --- a surface-code patch uses about $2d^2$ physical qubits (1); magic states: distilled or cultivated resources for $T$/Toffoli (2). Shor's
arithmetic is dominated by Toffoli gates, so magic-state production dominates its cost."""),

    dict(id="D16", kind="match",
         q=r"""Match each construction for an $n$-bit $N$ (List-I) with its logical-qubit count (List-II).""",
         match=([r"\texttt{Shor.ipynb} ($t = 2n$ counting $+$ $n$ work)", r"Beauregard (2003)", r"Gidney--Ekerå (2019)",
                 r"one recycled counting qubit $+$ $n$ work qubits with exact (oracle) multipliers"],
                [r"$n + 1$", r"$3n + 0.002\,n\log_2 n$", r"$3n$", r"$2n + 3$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1",
               r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2"],
         sol=r"""Notebook: $2n + n = 3n$ (3). Beauregard: $2n + 3$ (4). Gidney--Ekerå: $3n + 0.002n\log_2 n$, about $6190$ for $n = 2048$ (2). With the semiclassical QFT and
idealised matrix multipliers one needs only $n + 1$ (1) --- but real multipliers need ancilla qubits."""),

    dict(id="D17", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} Migrating to post-quantum cryptography is urgent even though no current quantum computer can run Shor's algorithm at
cryptographic sizes.\\[1mm]
\textbf{Reason (R):} Shor's algorithm also solves the discrete-logarithm problem.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true}: encrypted data recorded today can be decrypted later (``harvest now, decrypt later''), and migrations take years. \textbf{R is true}: period
finding solves discrete logarithms, so Diffie--Hellman and ECC fall too. But R does not explain the \emph{urgency} --- that comes from data lifetime and
migration time. Answer: (B)."""),

    dict(id="D18", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} In the notebook's construction, a $32$-bit $N$ needs $96$ qubits.\\[1mm]
\textbf{Statement II:} Repeated squaring lets $a^x \bmod N$ be implemented with $t$ controlled multiplications by the classically precomputed constants $a^{2^j} \bmod N$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true}: $3n = 96$ (the notebook's reverse estimate: $100$ qubits allow $33$ bits). \textbf{II is true}: $a^x = \prod_j(a^{2^j})^{x_j}$, so counting qubit $j$
controls the multiplication by $a^{2^j} \bmod N$, computed classically. Answer: (A)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""At $p = 0.01$, the probability that \emph{all} $127$ CNOTs are free of error is about $0.99^{127}$. Compare it with the success probability.""",
         opts=[r"$0.99^{127} \approx 0.28$, yet the success probability is $0.703$: many errors do not change $y$ or still lead to a correct order",
               r"$0.99^{127} \approx 0.28$, so the success probability must be at most $0.28$",
               r"$0.99^{127} \approx 0.88$, equal to the success probability",
               r"$0.99^{127} \approx 0.01$, so the result is pure noise"],
         sol=r"""$0.99^{127} = e^{127\ln0.99} = e^{-1.276} = 0.279$. The success probability is not bounded by the error-free probability: errors on work qubits after their last
use, phase errors that commute with measurement, or errors that shift $y$ to another successful value do not cause failure. Also, a random $y$ still succeeds
with probability $0.375$."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Which conclusion is supported by the table?""",
         opts=[r"the success probability decreases monotonically from $0.750$ towards the random baseline $0.375$; at $p = 0.03$ it is $0.665$, still above the baseline",
               r"noise increases the success probability",
               r"at $p = 0.03$ the circuit is completely random",
               r"the results show that a quantum processor factored $15$"],
         sol=r"""The values fall $0.750 \to 0.740 \to 0.726 \to 0.703 \to 0.665$; the gap to the uniform baseline ($0.375$) shrinks but has not vanished. These are
density-matrix \emph{simulations} with an idealised noise model --- no quantum processor was used, and factoring $15$ with a compiled circuit that uses the known
order is a demonstration, not evidence of quantum advantage."""),
]
