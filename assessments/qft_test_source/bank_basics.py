"""Section C - QFT Basics (definition, product formula, circuit, gate counts, simple inputs, inverse QFT).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""The textbook 3-qubit QFT circuit is applied to the input $\ket{x_1x_2x_3} = \ket{011}$ ($x = 3$). The gates, in order, are:
$H$ on qubit 1; controlled-$R_2$ (control qubit 2, target qubit 1); controlled-$R_3$ (control qubit 3, target qubit 1);
$H$ on qubit 2; controlled-$R_2$ (control qubit 3, target qubit 2); $H$ on qubit 3; SWAP of qubits 1 and 3.
Here $R_k = \operatorname{diag}(1, e^{2\pi i/2^k})$, and ``the phase of qubit $j$'' means the angle $\theta$ in its state
$\frac{1}{\sqrt2}(\ket0 + e^{i\theta}\ket1)$, taken in $[0^\circ, 360^\circ)$.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""For $n = 2$, the amplitude of $\ket{10}$ in $\mathrm{QFT}\ket{11}$ is""",
         opts=[r"$-\dfrac12$", r"$\dfrac12$", r"$\dfrac{i}{2}$", r"$-\dfrac{i}{2}$"],
         sol=r"""$N = 4$, $\omega = e^{2\pi i/4} = i$, $x = \ket{11} = 3$, $y = \ket{10} = 2$. From the definition
\[ \braket{y|\mathrm{QFT}|x} = \frac{1}{\sqrt N}\,\omega^{xy} = \frac12\, i^{6} = \frac12\,(i^4)(i^2) = -\frac12 . \]
Full output: $\mathrm{QFT}\ket{3} = \frac12\big(\ket0 - i\ket1 - \ket2 + i\ket3\big)$, i.e.\ $\frac12(\ket{00} - i\ket{01} - \ket{10} + i\ket{11})$."""),

    dict(id="C2", kind="mcq",
         q=r"""A 5-qubit QFT is applied to the basis state $\ket{x}$ with $x = 19$ and the output is measured. The probability of
obtaining $y = 7$ is""",
         opts=[r"$\dfrac{1}{32}$", r"$0$", r"$\dfrac{1}{\sqrt{32}}$", r"$\dfrac{7}{32}$"],
         sol=r"""$\mathrm{QFT}\ket{x} = \frac{1}{\sqrt{32}}\sum_y \omega^{xy}\ket y$ and $|\omega^{xy}| = 1$, so every outcome has probability
$\left|\frac{1}{\sqrt{32}}\right|^2 = \frac{1}{32}$, whatever $x$ and $y$ are. ($\frac{1}{\sqrt{32}}$ is the modulus of the amplitude,
not a probability.) All the information about $x$ sits in the phases $\omega^{19y}$."""),

    dict(id="C3", kind="mcq",
         q=r"""For $n = 3$, $\mathrm{QFT}\ket{110}$ equals (qubit 1 written first)""",
         opts=[r"$\frac{1}{\sqrt8}\big(\ket0 + \ket1\big)\big(\ket0 - \ket1\big)\big(\ket0 - i\ket1\big)$",
               r"$\frac{1}{\sqrt8}\big(\ket0 - i\ket1\big)\big(\ket0 - \ket1\big)\big(\ket0 + \ket1\big)$",
               r"$\frac{1}{\sqrt8}\big(\ket0 + \ket1\big)\big(\ket0 - \ket1\big)\big(\ket0 + i\ket1\big)$",
               r"$\frac{1}{\sqrt8}\big(\ket0 - \ket1\big)\big(\ket0 - \ket1\big)\big(\ket0 - i\ket1\big)$"],
         sol=r"""\emph{Product formula.} Write $y = \sum_{l=1}^{n} y_l 2^{\,n-l}$; then $\omega^{xy} = \prod_{l}e^{2\pi i\,x y_l/2^{l}}$ and
$e^{2\pi i\,x/2^l} = e^{2\pi i\,0.x_{n-l+1}\dots x_n}$ (only the last $l$ bits of $x$ survive), so the sum over $y$ factorises into one
factor per output qubit. For $n = 3$: $\mathrm{QFT}\ket{x_1x_2x_3} = \frac{1}{\sqrt8}
\big(\ket0 + e^{2\pi i\,0.x_3}\ket1\big)\big(\ket0 + e^{2\pi i\,0.x_2x_3}\ket1\big)\big(\ket0 + e^{2\pi i\,0.x_1x_2x_3}\ket1\big)$.
With $x_1x_2x_3 = 110$: $0.x_3 = 0.0 = 0 \Rightarrow e^{0} = 1$; $\;0.x_2x_3 = 0.10_2 = \frac12 \Rightarrow e^{i\pi} = -1$;
$\;0.x_1x_2x_3 = 0.110_2 = \frac34 \Rightarrow e^{3\pi i/2} = -i$. Hence
$\frac{1}{\sqrt8}(\ket0 + \ket1)(\ket0 - \ket1)(\ket0 - i\ket1)$.
\emph{Check} with the definition: amplitude of $y$ is $\omega^{6y}/\sqrt8 = (-i)^{y}/\sqrt8$; for $y = 4y_1 + 2y_2 + y_3$,
$(-i)^{4y_1}(-i)^{2y_2}(-i)^{y_3} = 1\cdot(-1)^{y_2}(-i)^{y_3}$ --- the same three factors. The option
$\frac{1}{\sqrt8}(\ket0 - i\ket1)(\ket0 - \ket1)(\ket0 + \ket1)$ is the bit-reversed order (what you get without the final SWAPs)."""),

    dict(id="C4", kind="mcq",
         q=r"""The textbook QFT circuit on $n = 12$ qubits contains how many controlled-$R_k$ gates?""",
         opts=[r"$66$", r"$78$", r"$72$", r"$144$"],
         sol=r"""Qubit $j$ receives one controlled rotation from each qubit below it, i.e.\ $n - j$ of them. Total
\[ \sum_{j=1}^{n}(n - j) = (n-1) + (n-2) + \dots + 0 = \frac{n(n-1)}{2} = \frac{12\cdot11}{2} = 66 . \]
($78 = \frac{n(n+1)}{2}$ also counts the 12 Hadamards; $144 = n^2$.)"""),

    dict(id="C5", kind="mcq",
         q=r"""On hardware a SWAP is implemented with 3 CNOTs. Counting every controlled-$R_k$ as one two-qubit gate, the total number
of two-qubit gates in the 12-qubit textbook QFT (including the final SWAPs) is""",
         opts=[r"$84$", r"$72$", r"$66$", r"$102$"],
         sol=r"""Controlled rotations: $\frac{12\cdot11}{2} = 66$ (C4). SWAPs: $\lfloor 12/2\rfloor = 6$, each costing 3 CNOTs $= 18$.
Total $66 + 18 = 84$. ($72 = 66 + 6$ counts each SWAP as one gate; $102 = 66 + 36$ uses 6 CNOTs per SWAP.)
In practice the SWAPs are often removed by simply relabelling the output qubits, leaving 66."""),

    dict(id="C6", kind="mcq",
         q=r"""The smallest rotation angle appearing in the textbook QFT circuit on $6$ qubits is""",
         opts=[r"$5.625^\circ$", r"$11.25^\circ$", r"$2.8125^\circ$", r"$6^\circ$"],
         sol=r"""The rotations used are $R_2, \dots, R_n$ with angle $\frac{2\pi}{2^k}$. The smallest is $R_n = R_6$:
$\frac{360^\circ}{2^6} = \frac{360^\circ}{64} = 5.625^\circ$. ($11.25^\circ$ is $R_5$; $2.8125^\circ$ would be $R_7$, which needs 7 qubits.)
These tiny angles are why the approximate QFT drops the smallest rotations (Section D)."""),

    dict(id="C7", kind="mcq",
         q=r"""In the textbook QFT circuit acting on $\ket{x_1x_2\dots x_n}$, \emph{before} the final SWAPs, qubit 1 is in the state
$\frac{1}{\sqrt2}\big(\ket0 + e^{2\pi i\,\phi}\ket1\big)$ with $\phi$ equal to""",
         opts=[r"$0.x_1x_2\dots x_n$", r"$0.x_n$", r"$0.x_nx_{n-1}\dots x_1$", r"$0.x_1$"],
         sol=r"""$H$ on qubit 1 gives $\frac{1}{\sqrt2}(\ket0 + e^{2\pi i\,0.x_1}\ket1)$ because $(-1)^{x_1} = e^{2\pi i\,x_1/2}$.
Each controlled-$R_k$ from qubit $k$ (for $k = 2, \dots, n$) multiplies the $\ket1$ part by $e^{2\pi i\,x_k/2^k}$, appending the
binary digit $x_k$ at position $k$: $0.x_1 \to 0.x_1x_2 \to \dots \to 0.x_1x_2\dots x_n$.
This factor belongs to the \emph{last} output qubit in the product formula, so the SWAPs move it from qubit 1 to qubit $n$."""),

    dict(id="C8", kind="mcq",
         q=r"""For any $n$-qubit basis state, $\mathrm{QFT}^2\ket{x}$ equals ($N = 2^n$)""",
         opts=[r"$\ket{(-x) \bmod N}$", r"$\ket{x}$", r"$\ket{2x \bmod N}$", r"$\ket{N - 1 - x}$"],
         sol=r"""\[ \mathrm{QFT}^2\ket x = \frac{1}{N}\sum_{z}\Big(\sum_y \omega^{y(x+z)}\Big)\ket z . \]
The inner geometric sum equals $N$ if $x + z \equiv 0 \pmod N$ and $0$ otherwise. Hence $\mathrm{QFT}^2\ket x = \ket{-x \bmod N}$.
Examples ($n = 2$): $\ket0 \to \ket0$, $\ket1 \to \ket3$, $\ket2 \to \ket2$. Consequently $\mathrm{QFT}^4 = I$.
($\ket{N-1-x}$ is the bitwise complement, a different map: it sends $\ket0 \to \ket{N-1}$.)"""),

    dict(id="C9", kind="mcq",
         q=r"""The QFT is applied to the uniform superposition $\frac{1}{\sqrt N}\sum_{x=0}^{N-1}\ket{x}$. The output is""",
         opts=[r"$\ket{0}$", r"the uniform superposition itself", r"$\ket{N-1}$", r"$\ket{N/2}$"],
         sol=r"""\[ \frac{1}{\sqrt N}\sum_x \frac{1}{\sqrt N}\sum_y\omega^{xy}\ket y = \frac1N\sum_y\Big(\sum_x\omega^{xy}\Big)\ket y = \ket0, \]
since $\sum_x \omega^{xy} = N\,\delta_{y,0}$. A constant (``DC'') signal has only the zero-frequency component.
Alternatively: the uniform superposition is $\mathrm{QFT}\ket0$ (product formula with $x = 0$), so applying the QFT once more gives
$\mathrm{QFT}^2\ket0 = \ket{-0 \bmod N} = \ket0$ by C8."""),

    dict(id="C10", kind="mcq",
         q=r"""For $n$ qubits, $\mathrm{QFT}\ket{10\dots0}$ (i.e.\ $x = 2^{n-1}$) equals""",
         opts=[r"$\ket{+}^{\otimes(n-1)}\otimes\ket{-}$", r"$\ket{-}\otimes\ket{+}^{\otimes(n-1)}$", r"$\ket{-}^{\otimes n}$", r"$\ket{+}^{\otimes n}$"],
         sol=r"""Amplitude of $y$: $\frac{1}{\sqrt N}\omega^{2^{n-1}y} = \frac{1}{\sqrt N}e^{2\pi i\,y/2} = \frac{1}{\sqrt N}(-1)^{y}$. The sign depends only on
the parity of $y$, i.e.\ on its last bit $y_n$ (qubit $n$). So the state factorises as
$\ket{+}\otimes\dots\otimes\ket{+}\otimes\ket{-}$. Product-formula check: $x_1 = 1$, others $0$, so every factor has phase
$e^{2\pi i\,0.0\dots}=1$ except the last, $e^{2\pi i\,0.x_1\dots x_n} = e^{2\pi i\,0.1} = -1$. (This is the ``highest frequency''
alternating signal $+,-,+,-,\dots$.)"""),

    dict(id="C11", kind="mcq",
         q=r"""The inverse QFT on 3 qubits is built by reversing the textbook QFT circuit and conjugating every gate. Ignoring the
SWAP, the \emph{first} gate of the inverse circuit is""",
         opts=[r"$H$ on qubit 3", r"$H$ on qubit 1", r"controlled-$R_2^\dagger$ between qubits 1 and 2",
               r"controlled-$R_3^\dagger$ between qubits 1 and 3"],
         sol=r"""QFT $= \mathrm{SWAP}_{13}\cdot H_3\cdot \mathrm{C}R_2^{(3\to2)}\cdot H_2\cdot\mathrm{C}R_3^{(3\to1)}\cdot \mathrm{C}R_2^{(2\to1)}\cdot H_1$
(rightmost acts first). Since $(ABC)^{-1} = C^{-1}B^{-1}A^{-1}$:
\[ \mathrm{QFT}^{-1} = H_1\cdot \mathrm{C}R_2^{\dagger\,(2\to1)}\cdot\mathrm{C}R_3^{\dagger\,(3\to1)}\cdot H_2\cdot \mathrm{C}R_2^{\dagger\,(3\to2)}\cdot H_3\cdot \mathrm{SWAP}_{13}, \]
so the order of application is SWAP, then $H$ on qubit 3, then $\mathrm{C}R_2^\dagger$ (3$\to$2), $H_2$, \dots, finally $H_1$.
$H^\dagger = H$ and $R_k^\dagger = \operatorname{diag}(1, e^{-2\pi i/2^k})$."""),

    dict(id="C12", kind="mcq",
         q=r"""For $n = 4$, the phase (in degrees, in $[0^\circ, 360^\circ)$) of the amplitude of $\ket{y}$, $y = 3$, in $\mathrm{QFT}\ket{x}$, $x = 5$, is""",
         opts=[r"$337.5^\circ$", r"$22.5^\circ$", r"$15^\circ$", r"$168.75^\circ$"],
         sol=r"""The amplitude is $\frac14\omega^{xy}$ with $\omega = e^{2\pi i/16}$, so the phase is
$\frac{2\pi\,xy}{16} = 2\pi\cdot\frac{15}{16}$, i.e.\ $\frac{15}{16}\times360^\circ = 337.5^\circ$ (equivalently $-22.5^\circ$; the option
$22.5^\circ$ has the sign convention reversed)."""),

    dict(id="C13", kind="mcq",
         q=r"""NumPy's two transforms of a length-$N$ vector $a$ are
\[ \texttt{fft}(a)_y = \sum_x a_x\, e^{-2\pi i\,xy/N}, \qquad \texttt{ifft}(a)_y = \frac1N\sum_x a_x\, e^{+2\pi i\,xy/N}. \]
If $\mathrm{QFT}\sum_x\alpha_x\ket x = \sum_y\beta_y\ket y$ (convention box), then the vector $\beta$ equals""",
         opts=[r"$\sqrt N\cdot\texttt{ifft}(\alpha)$", r"$\texttt{fft}(\alpha)/\sqrt N$", r"$\texttt{fft}(\alpha)$", r"$\texttt{ifft}(\alpha)/\sqrt N$"],
         sol=r"""$\beta_y = \frac{1}{\sqrt N}\sum_x\alpha_x e^{+2\pi i\,xy/N}$. The sign of the exponent is $+$, which matches \texttt{ifft}, and
$\frac{1}{\sqrt N} = \sqrt N\cdot\frac1N$, so $\beta = \sqrt N\,\texttt{ifft}(\alpha)$ (\texttt{numpy.fft.ifft(alpha, norm="ortho")}).
$\texttt{fft}(\alpha)/\sqrt N$ is the \emph{inverse} QFT. This is how the course notebooks verify CUDA-Q's state vectors."""),

    dict(id="C14", kind="mcq",
         q=r"""For an $n$-qubit computational basis state $\ket x$, the state $\mathrm{QFT}\ket x$ is""",
         opts=[r"always a product (unentangled) state", r"always maximally entangled",
               r"a product state only when $x$ is even", r"entangled unless $x = 0$"],
         sol=r"""The product formula writes $\mathrm{QFT}\ket{x}$ as a tensor product of $n$ single-qubit states
$\frac{1}{\sqrt2}(\ket0 + e^{2\pi i\,0.x_{n-l+1}\dots x_n}\ket1)$, $l = 1, \dots, n$. Hence it is never entangled,
for \emph{every} $x$. (The QFT \emph{can} entangle superposition inputs, because the controlled rotations are entangling
gates --- compare B5.) This product structure is also why the textbook circuit needs only $O(n^2)$ gates."""),

    dict(id="C15", kind="match",
         q=r"""Match the number of qubits $n$ (List-I) with the number of controlled-$R_k$ gates in the textbook QFT (List-II).""",
         match=([r"$n = 4$", r"$n = 6$", r"$n = 8$", r"$n = 10$"],
                [r"$15$", r"$28$", r"$45$", r"$6$"]),
         opts=[r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3", r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3",
               r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""Count $= \binom{n}{2} = \frac{n(n-1)}{2}$: $n = 4 \to 6$ (4), $n = 6 \to 15$ (1), $n = 8 \to 28$ (2), $n = 10 \to 45$ (3).
Hence P$\to$4, Q$\to$1, R$\to$2, S$\to$3. Growth is quadratic, compared with the exponential $2^n$ amplitudes transformed."""),

    dict(id="C16", kind="match",
         q=r"""For $n = 2$, match each input (List-I) with $\mathrm{QFT}$ of it (List-II); qubit 1 is written first.""",
         match=([r"$\ket{00}$", r"$\ket{01}$", r"$\ket{10}$", r"$\ket{11}$"],
                [r"$\frac12\big(\ket0 - \ket1\big)\big(\ket0 + i\ket1\big)$", r"$\frac12\big(\ket0 - \ket1\big)\big(\ket0 - i\ket1\big)$",
                 r"$\frac12\big(\ket0 + \ket1\big)\big(\ket0 + \ket1\big)$", r"$\frac12\big(\ket0 + \ket1\big)\big(\ket0 - \ket1\big)$"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""Product formula for $n = 2$: $\frac12\big(\ket0 + e^{2\pi i\,0.x_2}\ket1\big)\big(\ket0 + e^{2\pi i\,0.x_1x_2}\ket1\big)$.
\begin{itemize}\itemsep0pt
\item $00$: phases $e^0, e^0$ $\Rightarrow$ $(\ket0 + \ket1)(\ket0 + \ket1)/2$ (3).
\item $01$: $0.x_2 = \frac12 \to -1$; $0.x_1x_2 = 0.01_2 = \frac14 \to i$ $\Rightarrow$ (1).
\item $10$: $0.x_2 = 0 \to 1$; $0.10_2 = \frac12 \to -1$ $\Rightarrow$ (4).
\item $11$: $0.x_2 = \frac12 \to -1$; $0.11_2 = \frac34 \to -i$ $\Rightarrow$ (2).
\end{itemize}
Hence P$\to$3, Q$\to$1, R$\to$4, S$\to$2. Check $\ket{11}$ against C1: expanding (2) gives $\frac12(\ket{00} - i\ket{01} - \ket{10} + i\ket{11})$,
whose $\ket{10}$ amplitude is $-\frac12$."""),

    dict(id="C17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} Measuring $\mathrm{QFT}\ket{x}$ in the computational basis reveals no information about the
basis state $x$.\\[1mm]
\textbf{Reason (R):} All $N$ amplitudes of $\mathrm{QFT}\ket{x}$ have the same magnitude $1/\sqrt N$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $|\omega^{xy}/\sqrt N| = 1/\sqrt N$ for every $y$. \textbf{A is true} \emph{because} of R: the outcome
distribution $P(y) = 1/N$ is the same uniform distribution for every $x$, so the result is statistically independent of $x$.
The information is entirely in the relative phases, which a computational-basis measurement discards; one must first
undo the QFT (or interfere the phases in some other way) to read it. R is the correct explanation. Answer: (A)."""),

    dict(id="C18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} The QFT on $n$ qubits can be implemented with $O(n^2)$ one- and two-qubit gates.\\[1mm]
\textbf{Statement II:} Consequently, for an arbitrary classical input vector of length $2^n$, a quantum computer can output all
$2^n$ of its Fourier coefficients using only $O(n^2)$ operations in total.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $n$ Hadamards, $\frac{n(n-1)}{2}$ controlled rotations and $\lfloor n/2\rfloor$ SWAPs.
\textbf{II is false}, for two reasons: (1) \emph{input} --- loading an arbitrary $2^n$-entry vector into amplitudes
generally needs $\Omega(2^n)$ gates; (2) \emph{output} --- the coefficients are amplitudes; a measurement returns just one
index $y$ with probability $|\beta_y|^2$, and estimating all $2^n$ values needs exponentially many repetitions.
The QFT is useful as a \emph{subroutine} (phase estimation, Shor) where only a few peaks need to be read. Answer: (C)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""Immediately after the controlled-$R_3$ gate (the third gate), the phase of qubit 1 is""",
         opts=[r"$135^\circ$", r"$90^\circ$", r"$45^\circ$", r"$270^\circ$"],
         sol=r"""Input $x_1 = 0$, $x_2 = 1$, $x_3 = 1$.
$H$ on qubit 1: phase $2\pi\,0.x_1 = 0$, state $\frac{1}{\sqrt2}(\ket0 + \ket1)$.
C-$R_2$ (control $x_2 = 1$): adds $\frac{2\pi}{4} = 90^\circ$. C-$R_3$ (control $x_3 = 1$): adds $\frac{2\pi}{8} = 45^\circ$.
Total $0^\circ + 90^\circ + 45^\circ = 135^\circ$, i.e.\ $2\pi\cdot 0.011_2 = 2\pi\cdot\frac38$. (Since qubits 2 and 3 are basis states,
the controls are classical here and nothing becomes entangled.)"""),

    dict(id="C20", kind="para", passage="C",
         q=r"""After the complete circuit (including the SWAP), the amplitude of the basis state $\ket{110}$ is""",
         opts=[r"$\dfrac{i}{\sqrt8}$", r"$\dfrac{e^{i\pi/4}}{\sqrt8}$", r"$\dfrac{-i}{\sqrt8}$", r"$\dfrac{i}{8}$"],
         sol=r"""With the SWAP, the circuit implements exactly $\mathrm{QFT}\ket{3} = \frac{1}{\sqrt8}\sum_y\omega^{3y}\ket y$, $\omega = e^{i\pi/4}$.
For $\ket{110}$, $y = 6$: $\omega^{18} = \omega^{16}\omega^{2} = e^{i\pi/2} = i$, so the amplitude is $\frac{i}{\sqrt8}$.
Gate-by-gate check: before the SWAP, qubit 1 has phase $\frac38$ turn (C19), qubit 2 has $2\pi\,0.x_2x_3 = 2\pi\,0.11_2 = 270^\circ$,
qubit 3 has $2\pi\,0.x_3 = 180^\circ$. After swapping qubits 1 and 3, $\ket{110}$ collects phases $180^\circ + 270^\circ = 450^\circ \equiv 90^\circ$
$\Rightarrow i$. Forgetting the SWAP gives $\frac38 + \frac34 = \frac98 \equiv \frac18$ turn, i.e.\ $e^{i\pi/4}/\sqrt8$ --- a wrong option."""),
]
