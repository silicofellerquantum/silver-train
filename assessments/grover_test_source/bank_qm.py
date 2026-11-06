"""Section B - Quantum Mechanics (superposition, Born rule, phases, Hadamard layers, oracles, kickback, entanglement).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "B": r"""Three qubits start in $\ket{0}\ket{0}\ket{1}$ (input qubits 1 and 2, ancilla qubit 3). A Hadamard gate is applied to all three. Then the
Toffoli gate is applied with qubits 1 and 2 as controls and qubit 3 as target: it implements
$U_f\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$ with $f(x_1x_2) = x_1 \land x_2$.""",
}

QUESTIONS = [
    dict(id="B1", kind="mcq",
         q=r"""The state $H^{\otimes3}\ket{000}$ is measured in the computational basis. The probability that qubit 1 gives $1$ and qubit 3 gives $0$ is""",
         opts=[r"$\dfrac14$", r"$\dfrac18$", r"$\dfrac12$", r"$\dfrac{1}{\sqrt8}$"],
         sol=r"""All $8$ outcomes have probability $\left|\frac{1}{\sqrt8}\right|^2 = \frac18$. The event ``$x_1 = 1$, $x_3 = 0$'' contains the two outcomes
$\ket{100}$ and $\ket{110}$, so $P = \frac28 = \frac14$. (Equivalently the qubits are independent: $\frac12\cdot\frac12$.)"""),

    dict(id="B2", kind="mcq",
         q=r"""Let $\ket\psi = \frac{1}{\sqrt2}\big(\ket0 + i\ket1\big) \otimes \frac{1}{\sqrt2}\big(\ket0 - \ket1\big)$. The amplitude of $\ket{10}$ in $\ket\psi$ is""",
         opts=[r"$\dfrac{i}{2}$", r"$-\dfrac{i}{2}$", r"$\dfrac12$", r"$-\dfrac12$"],
         sol=r"""Multiply out: $(a_0\ket0 + a_1\ket1)\otimes(b_0\ket0 + b_1\ket1)$ has the amplitude $a_jb_k$ on $\ket{jk}$. For $\ket{10}$: $a_1b_0 = \frac{i}{\sqrt2}\cdot\frac{1}{\sqrt2} = \frac{i}{2}$.
The whole state is $\frac12(\ket{00} - \ket{01} + i\ket{10} - i\ket{11})$."""),

    dict(id="B3", kind="mcq",
         q=r"""$H \otimes H$ is applied to $\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)$. The amplitude of $\ket{11}$ in the result is""",
         opts=[r"$-\dfrac12$", r"$\dfrac12$", r"$0$", r"$1$"],
         sol=r"""Use $H^{\otimes2}\ket{x} = \frac12\sum_y(-1)^{x\cdot y}\ket{y}$. For $y = 11$ the signs $(-1)^{x\cdot y}$ for $x = 00, 01, 10, 11$ are $+, -, -, +$, so the amplitude is
\[ \frac12\cdot\frac12\big[(+1)(1) + (-1)(1) + (-1)(1) + (+1)(-1)\big] = \frac14(1 - 1 - 1 - 1) = -\frac12. \]
The full result is $\frac12(\ket{00} + \ket{01} + \ket{10} - \ket{11})$: this state is unchanged by $H \otimes H$."""),

    dict(id="B4", kind="mcq",
         q=r"""Let $U_f\ket{x}\ket{y} = \ket{x}\ket{y \oplus f(x)}$ and $\ket{-} = \frac{1}{\sqrt2}(\ket0 - \ket1)$. Then $U_f\ket{x}\ket{-}$ equals""",
         opts=[r"$(-1)^{f(x)}\ket{x}\ket{-}$", r"$\ket{x}\ket{-}$", r"$\ket{f(x)}\ket{-}$", r"$\ket{x}\ket{f(x)}$"],
         sol=r"""Expand the ancilla: $U_f\ket{x}\ket{-} = \frac{1}{\sqrt2}\ket{x}\big(\ket{f(x)} - \ket{1 \oplus f(x)}\big)$.
If $f(x) = 0$ the bracket is $\ket0 - \ket1$; if $f(x) = 1$ it is $\ket1 - \ket0 = -(\ket0 - \ket1)$. So the result is $(-1)^{f(x)}\ket{x}\ket{-}$. This is
\emph{phase kickback}: the ancilla is unchanged and the sign lands on the input register."""),

    dict(id="B5", kind="mcq",
         q=r"""With $U_f$ as in the previous question and $\ket{+} = \frac{1}{\sqrt2}(\ket0 + \ket1)$, the state $U_f\ket{x}\ket{+}$ equals""",
         opts=[r"$\ket{x}\ket{+}$", r"$(-1)^{f(x)}\ket{x}\ket{+}$", r"$\ket{x}\ket{-}$", r"$(-1)^{f(x)}\ket{x}\ket{-}$"],
         sol=r"""$U_f\ket{x}\ket{+} = \frac{1}{\sqrt2}\ket{x}\big(\ket{f(x)} + \ket{1 \oplus f(x)}\big) = \ket{x}\ket{+}$ in both cases, because $\ket0 + \ket1$ is symmetric under the bit flip
($X\ket{+} = +\ket{+}$). No phase is kicked back, so the ancilla must be prepared in $\ket{-}$, not $\ket{+}$."""),

    dict(id="B6", kind="mcq",
         q=r"""Let CCZ be the three-qubit gate that multiplies only $\ket{111}$ by $-1$. Which basis state has its sign flipped by $X_2\,\mathrm{CCZ}\,X_2$, where $X_2$ is $X$ on qubit 2?""",
         opts=[r"$\ket{101}$", r"$\ket{111}$", r"$\ket{010}$", r"$\ket{000}$"],
         sol=r"""The rightmost $X_2$ acts first. A state picks up the sign exactly when $X_2$ maps it to $\ket{111}$, which happens for $\ket{101}$. The last $X_2$ restores the
bit. So the sign flip sits on $\ket{101}$. This is the standard oracle recipe: apply $X$ wherever the marked string has a $0$."""),

    dict(id="B7", kind="mcq",
         q=r"""Let CCX be the Toffoli gate with target qubit 3. The product $(I \otimes I \otimes H)\,\mathrm{CCX}\,(I \otimes I \otimes H)$ equals""",
         opts=[r"CCZ", r"CCX", r"$I \otimes I \otimes I$", r"$Z \otimes Z \otimes Z$"],
         sol=r"""Use $HXH = Z$ and $HIH = I$. In the branch where both controls are $1$, the target sees $HXH = Z$; in all other branches it sees $HIH = I$. So the product is
the doubly controlled $Z$, i.e.\ CCZ $= \operatorname{diag}(1, \dots, 1, -1)$."""),

    dict(id="B8", kind="mcq",
         q=r"""Which pair of states are physically \emph{indistinguishable} (they give the same statistics for every possible measurement)?""",
         opts=[r"$\frac{1}{\sqrt2}(\ket0 + \ket1)$ and $-\frac{1}{\sqrt2}(\ket0 + \ket1)$",
               r"$\frac{1}{\sqrt2}(\ket0 + \ket1)$ and $\frac{1}{\sqrt2}(\ket0 - \ket1)$",
               r"$\frac{1}{\sqrt2}(\ket0 + \ket1)$ and $\frac{1}{\sqrt2}(\ket0 + i\ket1)$",
               r"$\frac{1}{\sqrt2}(\ket{00} + \ket{11})$ and $\frac{1}{\sqrt2}(\ket{00} - \ket{11})$"],
         sol=r"""Two state vectors describe the same physical state iff they differ by one overall factor $e^{i\gamma}$. Only the first pair does: the factor is $-1$. In each
of the other pairs the \emph{relative} phase differs, and a suitable measurement separates them. For example $H$ maps $\ket{+}$ and $\ket{-}$ to $\ket0$ and $\ket1$.
The minus sign in the diffusion circuit (see C7) is of the harmless global kind."""),

    dict(id="B9", kind="mcq",
         q=r"""Let $\ket\psi = \frac{1}{\sqrt{10}}\big(\ket{00} + 2\ket{01} - 2\ket{10} + i\ket{11}\big)$. A measurement asks only whether the two qubits have \emph{different} values
(outcome ``different'' on the span of $\ket{01}, \ket{10}$). The probability of ``different'' and the state afterwards are""",
         opts=[r"$\frac45$ and $\frac{1}{\sqrt2}(\ket{01} - \ket{10})$", r"$\frac45$ and $\frac{1}{\sqrt{10}}(2\ket{01} - 2\ket{10})$",
               r"$\frac25$ and $\frac{1}{\sqrt2}(\ket{01} - \ket{10})$", r"$\frac45$ and $\ket{01}$"],
         sol=r"""Sum the probabilities in the subspace: $P = \frac{4 + 4}{10} = \frac45$. The post-measurement state is the projected vector, re-normalised:
$\frac{(2\ket{01} - 2\ket{10})/\sqrt{10}}{\sqrt{4/5}} = \frac{2\ket{01} - 2\ket{10}}{\sqrt8} = \frac{1}{\sqrt2}(\ket{01} - \ket{10})$.
Relative phases inside the subspace survive, and the measurement does not pick one of the basis states."""),

    dict(id="B10", kind="mcq",
         q=r"""Let $\ket{u}$ be a unit vector in a space of dimension $N \ge 2$. Which of these operators is \emph{not} unitary?""",
         opts=[r"$\ket{u}\!\bra{u}$", r"$I - 2\ket{u}\!\bra{u}$", r"$2\ket{u}\!\bra{u} - I$", r"$H^{\otimes n}$ (for $N = 2^n$)"],
         sol=r"""$\ket{u}\!\bra{u}$ is a projector: it has eigenvalue $0$, so it is not invertible and cannot be unitary. A reflection $R = \pm(I - 2\ket{u}\!\bra{u})$ is
Hermitian with $R^2 = I$, so $R^\dagger R = I$. $H^{\otimes n}$ is a tensor product of unitaries."""),

    dict(id="B11", kind="mcq",
         q=r"""For $N = 16$ let $\ket{s}$ be the uniform superposition of all $16$ basis states and $\ket{r}$ the uniform superposition of $15$ of them (all except one). The value of $\braket{r|s}$ is""",
         opts=[r"$\dfrac{\sqrt{15}}{4}$", r"$\dfrac{15}{16}$", r"$\dfrac14$", r"$\dfrac{1}{\sqrt{15}}$"],
         sol=r"""$\braket{r|s} = \sum_{x}\frac{1}{\sqrt{15}}\cdot\frac{1}{\sqrt{16}}$ over the $15$ common basis states, which is $\frac{15}{\sqrt{15}\cdot 4} = \frac{\sqrt{15}}{4}$.
It equals $\cos\theta$ with $\sin\theta = \frac14$. $\frac{15}{16}$ is $\cos^2\theta$: the probability, not the overlap."""),

    dict(id="B12", kind="mcq",
         q=r"""A state vector of $n = 33$ qubits is stored in double precision ($16$ bytes per complex amplitude). The memory needed is""",
         opts=[r"$128$ GiB", r"$64$ GiB", r"$256$ GiB", r"$16$ GiB"],
         sol=r"""$2^{33}\times 16 = 2^{33}\times 2^{4} = 2^{37}$ bytes $= 2^{7}\times 2^{30}$ bytes $= 128$ GiB. Every extra qubit doubles it. ($16$ GiB is the value for $n = 30$.)"""),

    dict(id="B13", kind="mcq",
         q=r"""For a single qubit ($n = 1$), the operator $2\ket{+}\!\bra{+} - I$ equals""",
         opts=[r"$X$", r"$Z$", r"$H$", r"$-I$"],
         sol=r"""$\ket{+}\!\bra{+} = \frac12\begin{psmallmatrix}1&1\\1&1\end{psmallmatrix}$, so $2\ket{+}\!\bra{+} - I = \begin{psmallmatrix}1&1\\1&1\end{psmallmatrix} - \begin{psmallmatrix}1&0\\0&1\end{psmallmatrix} = \begin{psmallmatrix}0&1\\1&0\end{psmallmatrix} = X$.
\emph{Check:} $X\ket{+} = \ket{+}$ and $X\ket{-} = -\ket{-}$, which are the eigenvalues required of this reflection."""),

    dict(id="B14", kind="mcq",
         q=r"""For $n$ qubits let $\ket{s} = H^{\otimes n}\ket{0\dots0}$ and $D = 2\ket{s}\!\bra{s} - I$. If the oracle marks nothing (it acts as the identity), then after $k$ iterations the state is""",
         opts=[r"$\ket{s}$ for every $k$", r"$(-1)^k\ket{s}$", r"$\ket{0\dots0}$ for odd $k$", r"a state whose amplitudes depend on $k$"],
         sol=r"""$D\ket{s} = 2\ket{s}\braket{s|s} - \ket{s} = \ket{s}$, so $\ket{s}$ is an eigenvector of $D$ with eigenvalue $+1$. With $U_\omega = I$, each iteration is just $D$ and leaves $\ket{s}$ unchanged.
Measuring gives a uniformly random string, which matches $\sin^2\theta = 0$: nothing can be amplified when no item is marked."""),

    dict(id="B15", kind="match",
         q=r"""Match each expression in List-I with its value in List-II.""",
         match=([r"$H\ket1$", r"$Z\ket{-}$", r"$X\ket{-}$", r"$H\ket{+}$"],
                [r"$-\ket{-}$", r"$\ket0$", r"$\ket{+}$", r"$\ket{-}$"]),
         opts=[r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2",
               r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1"],
         sol=r"""$H\ket1 = \ket{-}$ (4). $Z\ket{-} = \frac{1}{\sqrt2}(\ket0 + \ket1) = \ket{+}$ (3). $X\ket{-} = \frac{1}{\sqrt2}(\ket1 - \ket0) = -\ket{-}$ (1).
$H\ket{+} = \ket0$ because $H^2 = I$ (2). Hence P$\to$4, Q$\to$3, R$\to$1, S$\to$2."""),

    dict(id="B16", kind="match",
         q=r"""Match each two-qubit state in List-I with its description in List-II.""",
         match=([r"$\frac12(\ket{00} + \ket{01} + \ket{10} + \ket{11})$", r"$\frac12(\ket{00} - \ket{01} - \ket{10} + \ket{11})$",
                 r"$\frac12(\ket{00} + \ket{01} + \ket{10} - \ket{11})$", r"$\frac{1}{\sqrt2}(\ket{00} + \ket{11})$"],
                [r"entangled, with four amplitudes of equal size", r"the product $\ket{-}\ket{-}$",
                 r"entangled, with only two non-zero amplitudes", r"the product $\ket{+}\ket{+}$"]),
         opts=[r"P$\to$4, Q$\to$2, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3",
               r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1", r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3"],
         sol=r"""Test $c_{00}c_{11}$ against $c_{01}c_{10}$; they are equal exactly for a product state.
P: $\frac14 = \frac14$, product, equal to $\ket{+}\ket{+}$ (4). Q: $\frac14 = \frac14$, product $\ket{-}\ket{-}$ (2).
R: $-\frac14 \ne \frac14$, entangled with four equal magnitudes (1); this is the state after the oracle for $n = 2$. S: $\frac12 \ne 0$, entangled with two non-zero amplitudes (3)."""),

    dict(id="B17", kind="ar", correct=0, opts=AR_OPTIONS,
         q=r"""\textbf{Assertion (A):} Measuring in the computational basis immediately after a phase oracle acts on $\ket{s}$ gives a uniformly random string; the oracle step alone reveals nothing.\\[1mm]
\textbf{Reason (R):} The oracle multiplies each amplitude by $\pm1$, and the Born probabilities depend only on the magnitudes of the amplitudes.""",
         sol=r"""\textbf{R is true:} $U_\omega\ket{x} = (-1)^{f(x)}\ket{x}$ and $|\pm a|^2 = |a|^2$. \textbf{A is true} because of R: the magnitudes stay $1/\sqrt N$, so the distribution stays uniform.
R is the correct explanation. Answer: (A)."""),

    dict(id="B18", kind="stmt", correct=2, opts=STATEMENT_OPTIONS,
         q=r"""\textbf{Statement I:} The states $\ket\psi$ and $-\ket\psi$ give identical statistics for every measurement.\\[1mm]
\textbf{Statement II:} Multiplying only \emph{one} basis-state amplitude of a superposition by $-1$ can never change the statistics of any measurement.""",
         sol=r"""\textbf{I is true:} a global phase cancels in $|\braket{\phi|\psi}|^2$ for every $\ket\phi$.
\textbf{II is false:} that is a \emph{relative} phase. Counterexample: $\ket{+}$ and $\ket{-}$ differ by the sign of $\ket1$ only, yet a measurement in the $\{\ket{+}, \ket{-}\}$ basis separates them perfectly.
Grover's algorithm exists because the sign is detectable after interference. Answer: (C)."""),

    dict(id="B19", kind="para", passage="B", opts=[
             r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)\ket{-}$",
             r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} + \ket{11}\big)\ket{-}$",
             r"$\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)\ket{+}$",
             r"$\frac12\big(\ket{00} - \ket{01} - \ket{10} - \ket{11}\big)\ket{-}$"],
         q=r"""The state of the three qubits after the Toffoli gate is""",
         sol=r"""After the Hadamards: $\frac12\sum_x\ket{x}\ket{-}$, since $H\ket1 = \ket{-}$. By kickback each term gets $(-1)^{f(x)}$, and $f = 1$ only for $x = 11$.
So the state is $\frac12(\ket{00} + \ket{01} + \ket{10} - \ket{11})\ket{-}$. The ancilla stays in $\ket{-}$, unentangled with the inputs."""),

    dict(id="B20", kind="para", passage="B", opts=[
             r"$\frac12$; the input qubits remain in $\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)$",
             r"$\frac14$; the input qubits collapse to $\ket{11}$",
             r"$\frac12$; the input qubits collapse to $\ket{11}$",
             r"$1$; the input qubits remain in $\frac12\big(\ket{00} + \ket{01} + \ket{10} - \ket{11}\big)$"],
         q=r"""Now the ancilla alone is measured in the computational basis. The probability of the outcome $1$, and the state of the input qubits afterwards, are""",
         sol=r"""The state after the gate is a product of the input state and $\ket{-}$ (see B19). Measuring $\ket{-}$ gives $1$ with probability $\frac12$, and because the ancilla is not entangled
with the inputs, the input qubits are not disturbed."""),
]
