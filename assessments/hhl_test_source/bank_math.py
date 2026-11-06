"""Section A - Linear algebra and mathematical foundations for HHL (linear systems, eigen-decomposition, Hermitian
matrices, matrix functions, condition number, sparsity, normalisation, classical solvers).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""The notebook's 2$\times$2 example is $A = \begin{pmatrix}1 & -\tfrac13\\ -\tfrac13 & 1\end{pmatrix}$, $b = \begin{pmatrix}1\\0\end{pmatrix}$. Its eigenvectors are
$u_1 = \tfrac{1}{\sqrt2}(1, 1)^{\mathsf T}$ and $u_2 = \tfrac{1}{\sqrt2}(1, -1)^{\mathsf T}$.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""The solution of $\begin{pmatrix}3 & 1\\ 1 & 3\end{pmatrix}x = \begin{pmatrix}1\\ -1\end{pmatrix}$ is""",
         opts=[r"$x = (0.5,\ -0.5)^{\mathsf T}$", r"$x = (0.25,\ -0.25)^{\mathsf T}$", r"$x = (1,\ -1)^{\mathsf T}$", r"$x = (0.5,\ 0.5)^{\mathsf T}$"],
         sol=r"""$b = (1, -1)$ is an eigenvector of $A$: $A(1, -1)^{\mathsf T} = (3 - 1, 1 - 3) = (2, -2) = 2(1, -1)$, eigenvalue $\lambda = 2$.
For an eigenvector, $A^{-1}b = b/\lambda = (0.5, -0.5)$. Check: $A(0.5, -0.5) = (1.5 - 0.5, 0.5 - 1.5) = (1, -1)$ \checkmark.
This is the idea of HHL in one line: in the eigenbasis, inverting $A$ means dividing each component by its eigenvalue."""),

    dict(id="A2", kind="mcq",
         q=r"""The eigenvalues and the condition number $\kappa = |\lambda|_{\max}/|\lambda|_{\min}$ of $A = \begin{pmatrix}1 & -\tfrac13\\ -\tfrac13 & 1\end{pmatrix}$ are""",
         opts=[r"$\tfrac23$ and $\tfrac43$; $\kappa = 2$", r"$\tfrac23$ and $\tfrac43$; $\kappa = \tfrac23$", r"$1$ and $-\tfrac13$; $\kappa = 3$", r"$\tfrac89$ and $1$; $\kappa = \tfrac98$"],
         sol=r"""For $\begin{psmallmatrix}a & c\\ c & a\end{psmallmatrix}$ the eigenvalues are $a \pm c$ with eigenvectors $(1, \pm1)/\sqrt2$: here $1 - \tfrac13 = \tfrac23$ and $1 + \tfrac13 = \tfrac43$.
$\kappa = \frac{4/3}{2/3} = 2$ (always $\ge 1$). Check: trace $= 2 = \tfrac23 + \tfrac43$, determinant $= 1 - \tfrac19 = \tfrac89 = \tfrac23\cdot\tfrac43$.
$\tfrac89$ is the determinant, not an eigenvalue; $1$ and $-\tfrac13$ are matrix entries."""),

    dict(id="A3", kind="match",
         q=r"""Match each class of square matrix (List-I) with the property of its eigenvalues (List-II).""",
         match=([r"Hermitian ($A = A^\dagger$)", r"unitary ($U^\dagger U = I$)", r"Hermitian positive definite", r"singular"],
                [r"at least one eigenvalue is $0$", r"all eigenvalues are real", r"all eigenvalues lie on the unit circle, $|\lambda| = 1$",
                 r"all eigenvalues are real and strictly positive"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1",
               r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: $Av = \lambda v$, $\|v\| = 1$: $\lambda = v^\dagger Av = (v^\dagger Av)^* = \lambda^*$, real (2).
\item Q: $\|Uv\| = \|v\|$ forces $|\lambda| = 1$ (3) --- e.g.\ $U = e^{iAt}$ has eigenvalues $e^{i\lambda t}$.
\item R: positive definite: $v^\dagger Av > 0$ for all $v \ne 0$, so every eigenvalue is real and positive (4).
\item S: singular $\iff \det A = \prod\lambda_i = 0 \iff$ some eigenvalue is $0$ (1); then $A^{-1}$ does not exist.
\end{itemize}
HHL needs a Hermitian (or embedded) invertible matrix; the phase estimation step works with the unitary $e^{iAt}$."""),

    dict(id="A4", kind="match",
         q=r"""Match each matrix (List-I) with its condition number $\kappa = |\lambda|_{\max}/|\lambda|_{\min}$ (List-II).""",
         match=([r"$\operatorname{diag}(0.5,\ 2,\ 8)$", r"$\begin{psmallmatrix}1 & -1/3\\ -1/3 & 1\end{psmallmatrix}$", r"the notebook's default $A$ (eigenvalues $1, 2, 3, 4$)", r"$\begin{psmallmatrix}2 & 0\\ 0 & 2\end{psmallmatrix}$"],
                [r"$4$", r"$1$", r"$16$", r"$2$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$1, Q$\to$4, R$\to$3, S$\to$2",
               r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: $8/0.5 = 16$ (3); the middle eigenvalue does not matter.
\item Q: eigenvalues $\tfrac23, \tfrac43$: $\kappa = 2$ (4).
\item R: $4/1 = 4$ (1), the value the notebook prints.
\item S: a multiple of the identity has $\kappa = 1$ (2) --- the best possible.
\end{itemize}
HHL's cost and failure probability grow with $\kappa$ (success probability $\ge 1/\kappa^2$)."""),

    dict(id="A5", kind="mcq",
         q=r"""The notebook's default matrix is $A = H\,\operatorname{diag}(1, 2, 3, 4)\,H$ with the symmetric orthogonal matrix
$H = \tfrac12\begin{psmallmatrix}1&1&1&1\\1&-1&1&-1\\1&1&-1&-1\\1&-1&-1&1\end{psmallmatrix}$ ($H^2 = I$). For $b = (1, 0, 0, 0)^{\mathsf T}$, the first component $x_0$ of $x = A^{-1}b$ is""",
         opts=[r"$\tfrac{25}{48} \approx 0.5208$", r"$\tfrac{25}{12} \approx 2.0833$", r"$\tfrac14$", r"$\tfrac{5}{8}$"],
         sol=r"""$A^{-1} = H\,\operatorname{diag}(1, \tfrac12, \tfrac13, \tfrac14)\,H$ (invert the eigenvalues, keep the eigenvectors). Then
$x_0 = (A^{-1})_{00} = \sum_j H_{0j}^2\,\lambda_j^{-1} = \tfrac14\left(1 + \tfrac12 + \tfrac13 + \tfrac14\right) = \tfrac14\cdot\tfrac{25}{12} = \tfrac{25}{48} = 0.520833$,
since every $H_{0j} = \tfrac12$. This matches \texttt{np.linalg.solve} and the notebook: $x = (0.520833, 0.145833, 0.229167, 0.104167)$.
$\tfrac{25}{12}$ forgets the factor $H_{0j}^2 = \tfrac14$."""),

    dict(id="A6", kind="mcq",
         q=r"""HHL needs a Hermitian matrix, so the notebook embeds a non-Hermitian $A$ as $\tilde A = \begin{pmatrix}0 & A\\ A^\dagger & 0\end{pmatrix}$. For $A = \begin{pmatrix}1 & 2\\ 0 & 1\end{pmatrix}$ the
condition number of $\tilde A$ is""",
         opts=[r"$3 + 2\sqrt2 \approx 5.83$", r"$1$, because both eigenvalues of $A$ are $1$", r"$\sqrt2 + 1 \approx 2.41$", r"$2$"],
         sol=r"""The eigenvalues of $\tilde A$ are $\pm\sigma_i$, the singular values of $A$. $A^{\mathsf T}A = \begin{psmallmatrix}1&2\\2&5\end{psmallmatrix}$ has eigenvalues $3 \pm 2\sqrt2$,
so $\sigma = \sqrt{3 \pm 2\sqrt2} = \sqrt2 \pm 1$, i.e.\ $2.414$ and $0.414$. Hence $\tilde A$ has eigenvalues $\pm2.414, \pm0.414$ and
$\kappa = \frac{\sqrt2 + 1}{\sqrt2 - 1} = (\sqrt2 + 1)^2 = 3 + 2\sqrt2 \approx 5.83$. The eigenvalues of a non-normal matrix (here both $1$) say nothing about its
conditioning; singular values do."""),

    dict(id="A7", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} Every invertible matrix has condition number $\kappa(A) = \|A\|\,\|A^{-1}\| \ge 1$.\\[1mm]
\textbf{Reason (R):} In the spectral norm $\kappa(A) = \sigma_{\max}/\sigma_{\min}$, the ratio of the largest to the smallest singular value.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $\|A\|_2 = \sigma_{\max}$ and $\|A^{-1}\|_2 = 1/\sigma_{\min}$.
\textbf{A follows from R:} a ratio of the largest to the smallest of a set of positive numbers is at least $1$ (equality for multiples of unitary
matrices). For Hermitian $A$ the singular values are $|\lambda_i|$, giving $\kappa = |\lambda|_{\max}/|\lambda|_{\min}$, the formula the notebook prints. Answer: (A)."""),

    dict(id="A8", kind="stmt", correct=3,
         q=r"""\textbf{Statement I:} Every invertible matrix is Hermitian.\\[1mm]
\textbf{Statement II:} Every real symmetric matrix can be written as $A = Q\Lambda Q^{\mathsf T}$ with an orthogonal $Q$ and a real diagonal $\Lambda$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} $\begin{psmallmatrix}1&2\\0&1\end{psmallmatrix}$ is invertible ($\det = 1$) but not symmetric --- the notebook embeds such matrices.
\textbf{II is true:} the spectral theorem for real symmetric matrices; the notebook's default $A = H\operatorname{diag}(1,2,3,4)H$ is exactly of this form with $Q = H$.
Answer: (D)."""),

    dict(id="A9", kind="mcq",
         q=r"""The sparsity $s$ of a matrix is the largest number of non-zero entries in any row. For the notebook's default $4\times4$ matrix
$A = \begin{psmallmatrix}2.5&-0.5&-1&0\\-0.5&2.5&0&-1\\-1&0&2.5&-0.5\\0&-1&-0.5&2.5\end{psmallmatrix}$ and for a tridiagonal $1000\times1000$ matrix, $s$ is""",
         opts=[r"$3$ and $3$", r"$4$ and $3$", r"$3$ and $1000$", r"$12$ and $2998$"],
         sol=r"""Each row of the default matrix has one zero, so $3$ non-zeros per row: $s = 3$. A tridiagonal matrix has at most $3$ non-zeros per row
(diagonal and two neighbours) whatever its size: $s = 3$. $12$ and $2998$ are the \emph{total} numbers of non-zeros. HHL's original
analysis assumes $s$-sparse matrices so that $e^{iAt}$ can be simulated in time $\mathrm{poly}(s, \log d)$; the notebook does not exploit sparsity (it
exponentiates the dense matrix exactly)."""),

    dict(id="A10", kind="mcq",
         q=r"""Gaussian elimination on a dense $d\times d$ system costs about $\tfrac23d^3$ floating-point operations. Doubling $d$ from $1000$ to $2000$ multiplies the work by""",
         opts=[r"$8$", r"$2$", r"$4$", r"$\tfrac23$"],
         sol=r"""Work $\propto d^3$: $(2000/1000)^3 = 2^3 = 8$. For $d = 1000$: $\tfrac23\cdot10^9 \approx 6.7\times10^8$ operations, a fraction of a second today; the cubic growth is what
limits direct solvers for very large systems (iterative methods exploit sparsity instead)."""),

    dict(id="A11", kind="mcq",
         q=r"""For a sparse symmetric positive-definite matrix with condition number $\kappa$, the number of conjugate-gradient iterations needed for a fixed accuracy
scales as""",
         opts=[r"$O(\sqrt\kappa)$", r"$O(\kappa^2)$", r"$O(d^3)$", r"$O(\log\kappa)$"],
         sol=r"""Conjugate gradient reduces the error by a factor $\le 2\left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^k$ after $k$ iterations, so $k = O(\sqrt\kappa\log(1/\varepsilon))$ iterations
suffice; each costs one sparse matrix--vector product, $O(sd)$. Total $O(sd\sqrt\kappa\log(1/\varepsilon))$ --- the fair classical competitor for HHL, which
scales with $\log d$ but polynomially in $\kappa$ (the original analysis gives $\kappa^2$)."""),

    dict(id="A12", kind="mcq",
         q=r"""The vector $b = (1, 2, 2, 0)^{\mathsf T}$ is amplitude-encoded on two qubits as $\ket b = \sum_ib_i\ket i/\|b\|$ (index $i$ in binary, first qubit most significant). The
probability of measuring $\ket{01}$ is""",
         opts=[r"$\tfrac49$", r"$\tfrac23$", r"$\tfrac29$", r"$\tfrac13$"],
         sol=r"""$\|b\| = \sqrt{1 + 4 + 4 + 0} = 3$, so $\ket b = \tfrac13(\ket{00} + 2\ket{01} + 2\ket{10})$. $\ket{01}$ is index $1$ with amplitude $\tfrac23$; Born rule:
$P = (\tfrac23)^2 = \tfrac49$. Amplitude encoding stores $d$ numbers in $\log_2d$ qubits but loses the norm $\|b\| = 3$, which must be tracked classically
(the notebook multiplies by \texttt{b\_norm} when it recovers $x$)."""),

    dict(id="A13", kind="mcq",
         q=r"""For the notebook's default $A = H\operatorname{diag}(1, 2, 3, 4)H$ (eigenvectors = columns of $H$) and $b = (1, 0, 0, 0)$, the coefficients $\beta_j = \braket{u_j|b}$ of
$\ket b$ in the eigenbasis are""",
         opts=[r"$(\tfrac12, \tfrac12, \tfrac12, \tfrac12)$", r"$(1, 0, 0, 0)$", r"$(\tfrac14, \tfrac14, \tfrac14, \tfrac14)$", r"$(\tfrac12, -\tfrac12, \tfrac12, -\tfrac12)$"],
         sol=r"""$\beta_j = \braket{u_j|b} = (\text{column } j \text{ of } H)\cdot(1, 0, 0, 0) = H_{0j} = \tfrac12$ for every $j$ (first row of $H$). Normalisation: $4\cdot\tfrac14 = 1$ \checkmark.
So $\ket b = \tfrac12\sum_j\ket{u_j}$: every eigenvalue contributes with weight $\tfrac14$ --- which is why phase estimation produces all four clock values
$1, 2, 3, 4$ in the notebook. $(\tfrac12, -\tfrac12, \dots)$ is the second row of $H$."""),

    dict(id="A14", kind="para", passage="A",
         q=r"""The solution $x = A^{-1}b$ is""",
         opts=[r"$(\tfrac98,\ \tfrac38)^{\mathsf T}$", r"$(1,\ \tfrac13)^{\mathsf T}$", r"$(\tfrac34,\ \tfrac14)^{\mathsf T}$", r"$(\tfrac98,\ -\tfrac38)^{\mathsf T}$"],
         sol=r"""$\det A = 1 - \tfrac19 = \tfrac89$, $A^{-1} = \tfrac98\begin{psmallmatrix}1 & \tfrac13\\ \tfrac13 & 1\end{psmallmatrix}$, so $x = \tfrac98(1, \tfrac13) = (\tfrac98, \tfrac38)$.
Eigen-route: $b = \tfrac{1}{\sqrt2}(u_1 + u_2)$, $x = \tfrac{1}{\sqrt2}\left(\tfrac{u_1}{2/3} + \tfrac{u_2}{4/3}\right) = \tfrac12\left(\tfrac32(1,1) + \tfrac34(1,-1)\right) = (\tfrac98, \tfrac38)$.
Check: $A x = (\tfrac98 - \tfrac18, -\tfrac38 + \tfrac38) = (1, 0)$ \checkmark. The off-diagonal entry is negative, so $A^{-1}$ has a \emph{positive} off-diagonal entry."""),

    dict(id="A15", kind="para", passage="A",
         q=r"""HHL prepares the normalised state $\ket x = x/\|x\|$. Its amplitudes, and the probability of measuring $\ket0$ on the system qubit, are""",
         opts=[r"$\tfrac{1}{\sqrt{10}}(3, 1)$ and $0.9$", r"$\tfrac{1}{\sqrt{10}}(3, 1)$ and $0.75$", r"$(\tfrac98, \tfrac38)$ and $\tfrac98$", r"$\tfrac12(\sqrt3, 1)$ and $0.75$"],
         sol=r"""$x = \tfrac38(3, 1)$, $\|x\| = \tfrac38\sqrt{10}$, so $\ket x = \tfrac{1}{\sqrt{10}}(3, 1) = (0.9487, 0.3162)$. $P(0) = \tfrac{9}{10} = 0.9$. The overall factor $\tfrac38$ (the norm)
is lost in $\ket x$; the notebook restores it with $x = \|b\|\,C^{-1}\times(\text{amplitudes})$. $(\tfrac98, \tfrac38)$ is not normalised, and a probability above 1 is impossible."""),

    dict(id="A16", kind="mcq",
         q=r"""For $A = \operatorname{diag}(1, 2)$ and $t = \pi/2$, the matrix $e^{iAt}$ is""",
         opts=[r"$\operatorname{diag}(i,\ -1)$", r"$\operatorname{diag}(-i,\ -1)$", r"$\operatorname{diag}(i,\ 1)$", r"$\begin{pmatrix}0 & i\\ i & 0\end{pmatrix}$"],
         sol=r"""For a diagonal (or diagonalisable) $A$, $e^{iAt}$ acts on eigenvectors as $e^{i\lambda t}$: $e^{i\pi/2} = i$ and $e^{i\pi} = -1$. So $\operatorname{diag}(i, -1)$. In general
$e^{iAt} = Ue^{i\Lambda t}U^\dagger$ --- the notebook computes it with \texttt{scipy.linalg.expm(1j * A * t)}. The eigenvalues of $A$ become \emph{phases} $e^{i\lambda t}$ of
a unitary, which phase estimation can read."""),

    dict(id="A17", kind="mcq",
         q=r"""For which matrices $A$ is $e^{iAt}$ unitary for every real $t$?""",
         opts=[r"exactly the Hermitian matrices", r"exactly the invertible matrices", r"exactly the real matrices", r"exactly the sparse matrices"],
         sol=r"""$(e^{iAt})^\dagger = e^{-iA^\dagger t}$. This equals $(e^{iAt})^{-1} = e^{-iAt}$ for all $t$ iff $A^\dagger = A$ (differentiate at $t = 0$). Hence HHL's phase-estimation unitary
$U = e^{iAt}$ requires a Hermitian $A$; invertibility is needed separately for $A^{-1}$ to exist. A real non-symmetric matrix gives a non-unitary $e^{iAt}$."""),

    dict(id="A18", kind="mcq",
         q=r"""For the default system ($A$ with eigenvalues $1, 2, 3, 4$ and $\|b\| = 1$), the norm $\|x\| = \|A^{-1}b\|$ is $0.5966$. Which bound must every such norm satisfy?""",
         opts=[r"$\dfrac{\|b\|}{\lambda_{\max}} \le \|x\| \le \dfrac{\|b\|}{\lambda_{\min}}$, i.e.\ $0.25 \le 0.5966 \le 1$",
               r"$\|b\|\,\lambda_{\min} \le \|x\| \le \|b\|\,\lambda_{\max}$, i.e.\ $1 \le \|x\| \le 4$",
               r"$\|x\| = \|b\|/\kappa = 0.25$ exactly",
               r"$\|x\| \le \|b\|/\lambda_{\max} = 0.25$"],
         sol=r"""$\|x\|^2 = \sum_j|\beta_j|^2/\lambda_j^2$ lies between $\|b\|^2/\lambda_{\max}^2$ and $\|b\|^2/\lambda_{\min}^2$. Here $\|x\|^2 = \tfrac14(1 + \tfrac14 + \tfrac19 + \tfrac1{16}) = 0.3559$, $\|x\| = 0.5966$,
between $0.25$ and $1$. In HHL $\|x\|^2$ times $C^2$ is exactly the success probability (with $C = \lambda_{\min} = 1$: $0.3559$, the value recorded by the notebook)."""),

    dict(id="A19", kind="mcq",
         q=r"""For which values of $a$ is $\begin{pmatrix}a & 1\\ 1 & a\end{pmatrix}$ singular (so that HHL cannot be applied)?""",
         opts=[r"$a = \pm1$", r"$a = 0$", r"$a = \pm2$", r"no real $a$"],
         sol=r"""Eigenvalues $a \pm 1$; the matrix is singular iff one of them is $0$: $a = 1$ (eigenvalue $a - 1 = 0$) or $a = -1$ (eigenvalue $a + 1 = 0$). Equivalently $\det = a^2 - 1 = 0$.
For $a = 0$ the eigenvalues are $\pm1$ (invertible, $\kappa = 1$, but indefinite --- HHL handles negative eigenvalues via two's complement clock values).
The notebook rejects singular matrices (\texttt{det} $< 10^{-12}$ or $\kappa > 10^{12}$)."""),

    dict(id="A20", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} Multiplying an invertible matrix by a non-zero constant $c$ does not change its condition number.\\[1mm]
\textbf{Reason (R):} Multiplying $A$ by $c$ multiplies every eigenvalue by $c^2$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} all singular values are multiplied by $|c|$, so their ratio is unchanged.
\textbf{R is false:} eigenvalues of $cA$ are $c\lambda_i$ (multiply $Av = \lambda v$ by $c$), not $c^2\lambda_i$. Scaling still matters for HHL: the choice of $t$ and $C$ is
relative to the eigenvalues (the notebook's \texttt{"auto"} time rescales automatically). Answer: (C)."""),
]
