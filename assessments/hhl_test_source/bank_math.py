"""Section A - Linear algebra and mathematical foundations (linear systems, eigenvalues, Hermitian matrices,
spectral decomposition, sparsity, normalisation, norms and the condition number).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""The notebook \texttt{HHL.ipynb} lists the $2\times2$ example
\[ A = \begin{pmatrix} 1 & -\tfrac13\\[2pt] -\tfrac13 & 1\end{pmatrix}, \qquad \vec b = \begin{pmatrix}1\\0\end{pmatrix}. \]
The questions below concern the classical linear algebra that HHL relies on for this system.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""The solution of the linear system $\begin{pmatrix}2&1\\1&3\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix} = \begin{pmatrix}1\\2\end{pmatrix}$ is""",
         opts=[r"$(x_1, x_2) = (0.2,\ 0.6)$", r"$(x_1, x_2) = (0.6,\ 0.2)$", r"$(x_1, x_2) = (-0.2,\ 0.6)$",
               r"$(x_1, x_2) = (1,\ 0.\overline{3})$"],
         sol=r"""$\det A = 2\cdot3 - 1\cdot1 = 5 \neq 0$, so the system has a unique solution. For a $2\times2$ matrix
\[ A^{-1} = \frac{1}{\det A}\begin{pmatrix}3&-1\\-1&2\end{pmatrix}, \qquad
   \vec x = A^{-1}\vec b = \frac15\begin{pmatrix}3\cdot1 - 1\cdot2\\ -1\cdot1 + 2\cdot2\end{pmatrix} = \frac15\begin{pmatrix}1\\3\end{pmatrix} = \begin{pmatrix}0.2\\0.6\end{pmatrix}. \]
\emph{Check:} $2(0.2) + 0.6 = 1$ and $0.2 + 3(0.6) = 2$. (Gaussian elimination gives the same: $R_2 \leftarrow R_2 - \tfrac12R_1$ gives $2.5x_2 = 1.5$.)"""),

    dict(id="A2", kind="mcq",
         q=r"""The eigenvalues and the condition number $\kappa$ of $A = \begin{pmatrix}5&2\\2&2\end{pmatrix}$ are""",
         opts=[r"$1$ and $6$; $\ \kappa = 6$", r"$1$ and $6$; $\ \kappa = 7$", r"$2$ and $5$; $\ \kappa = 2.5$",
               r"$3$ and $4$; $\ \kappa = 4/3$"],
         sol=r"""Characteristic polynomial: $\lambda^2 - (\operatorname{tr}A)\lambda + \det A = \lambda^2 - 7\lambda + 6 = (\lambda - 1)(\lambda - 6)$, so
$\lambda \in \{1, 6\}$. Eigenvectors: $\lambda = 1$: $4x + 2y = 0 \Rightarrow (1, -2)$; $\lambda = 6$: $-x + 2y = 0 \Rightarrow (2, 1)$ --- orthogonal,
as they must be for a symmetric matrix. $A$ is symmetric positive definite, so
\[ \kappa = \frac{|\lambda|_{\max}}{|\lambda|_{\min}} = \frac61 = 6. \]
(The diagonal entries $5, 2$ are \emph{not} the eigenvalues, because $A$ is not diagonal.)"""),

    dict(id="A3", kind="match",
         q=r"""Match each matrix (List-I) with the description that fits it (List-II).""",
         match=([r"$\begin{pmatrix}0&1\\1&0\end{pmatrix}$", r"$\begin{pmatrix}1&2\\2&4\end{pmatrix}$",
                 r"$\begin{pmatrix}1&i\\-i&2\end{pmatrix}$", r"$\begin{pmatrix}0&1\\0&0\end{pmatrix}$"],
                [r"Hermitian and unitary", r"Hermitian, invertible, not unitary", r"Hermitian but singular",
                 r"not diagonalisable"]),
         opts=[r"P$\to$1, Q$\to$3, R$\to$2, S$\to$4", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4",
               r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3", r"P$\to$1, Q$\to$2, R$\to$3, S$\to$4"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P $= X$: real symmetric, so Hermitian; $X^\dagger X = X^2 = I$, so unitary (1).
\item Q: symmetric (Hermitian) but $\det = 4 - 4 = 0$ (second row $= 2\times$ first): singular, eigenvalues $0$ and $5$ (3).
\item R: $R^\dagger$ = conjugate transpose $= \begin{psmallmatrix}1&\overline{-i}\\ \overline{i}&2\end{psmallmatrix} = \begin{psmallmatrix}1&i\\-i&2\end{psmallmatrix} = R$, so Hermitian;
$\det R = 2 - (i)(-i) = 2 - 1 = 1 \neq 0$, invertible; $R^\dagger R = R^2 \neq I$ (e.g.\ $(R^2)_{11} = 1 + 1 = 2$), not unitary (2).
\item S: both eigenvalues are $0$ but $S \neq 0$; the eigenspace of $0$ is only 1-dimensional ($\{(a, 0)\}$), so $S$ is not diagonalisable (4).
\end{itemize}
Only Hermitian, invertible matrices (P and R) are directly usable in HHL."""),

    dict(id="A4", kind="mcq",
         q=r"""The condition number of $A = \operatorname{diag}(1,\ 0.5,\ 0.01,\ 0.2)$ is""",
         opts=[r"$100$", r"$1.71$", r"$0.01$", r"$50$"],
         sol=r"""For a diagonal (hence Hermitian) matrix the eigenvalues are the diagonal entries: $1, 0.5, 0.01, 0.2$.
\[ \kappa = \frac{|\lambda|_{\max}}{|\lambda|_{\min}} = \frac{1}{0.01} = 100. \]
Equivalently $\kappa = \lVert A\rVert_2\,\lVert A^{-1}\rVert_2 = 1 \times 100$. A system with $\kappa = 100$ amplifies relative errors in $\vec b$
by up to a factor $100$; for HHL the run time grows with $\kappa$ (Section D)."""),

    dict(id="A5", kind="mcq",
         q=r"""The Hermitian matrix $A = \begin{pmatrix}3 & 1-i\\ 1+i & 2\end{pmatrix}$ has eigenvalues""",
         opts=[r"$1$ and $4$", r"$3$ and $2$", r"$2 \pm i$", r"$0$ and $5$"],
         sol=r"""$\operatorname{tr}A = 5$, $\det A = 3\cdot2 - (1-i)(1+i) = 6 - (1 + 1) = 4$. The eigenvalues solve $\lambda^2 - 5\lambda + 4 = 0$:
$\lambda = 1$ or $\lambda = 4$. Both are real, as for every Hermitian matrix, even though $A$ has complex entries. (Note $(1-i)(1+i) = |1+i|^2 = 2$.)
$A$ is positive definite with $\kappa = 4$."""),

    dict(id="A6", kind="mcq",
         q=r"""$A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ has eigenpairs $\lambda_1 = 3$, $\ket{u_1} = \tfrac{1}{\sqrt2}(1, 1)$ and $\lambda_2 = 1$,
$\ket{u_2} = \tfrac{1}{\sqrt2}(1, -1)$. Using $A^{-1} = \sum_j \lambda_j^{-1}\ket{u_j}\!\bra{u_j}$, the off-diagonal entry $(A^{-1})_{12}$ is""",
         opts=[r"$-\tfrac13$", r"$+\tfrac13$", r"$-\tfrac23$", r"$\tfrac16$"],
         sol=r"""$\ket{u_1}\!\bra{u_1} = \tfrac12\begin{psmallmatrix}1&1\\1&1\end{psmallmatrix}$ and $\ket{u_2}\!\bra{u_2} = \tfrac12\begin{psmallmatrix}1&-1\\-1&1\end{psmallmatrix}$. Hence
\[ A^{-1} = \frac13\cdot\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix} + \frac11\cdot\frac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}
          = \begin{pmatrix}\tfrac16 + \tfrac12 & \tfrac16 - \tfrac12\\ \tfrac16 - \tfrac12 & \tfrac16 + \tfrac12\end{pmatrix}
          = \frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}. \]
So $(A^{-1})_{12} = -\tfrac13$, in agreement with the $2\times2$ formula $\frac{1}{\det A}\begin{psmallmatrix}2&-1\\-1&2\end{psmallmatrix}$, $\det A = 3$.
This ``invert every eigenvalue'' formula is exactly what HHL implements on a quantum state."""),

    dict(id="A7", kind="ar", correct=1,
         q=r"""\textbf{Assertion (A):} Every Hermitian matrix can be diagonalised by a unitary matrix, $A = V\Lambda V^\dagger$.\\[1mm]
\textbf{Reason (R):} All eigenvalues of a Hermitian matrix are real.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} this is the spectral theorem (it holds for every \emph{normal} matrix, $AA^\dagger = A^\dagger A$, and Hermitian matrices are normal).
\textbf{R is true:} $\lambda\braket{u|u} = \braket{u|A|u} = \braket{u|A^\dagger|u}^* = \bar\lambda\braket{u|u}$, so $\lambda = \bar\lambda$.
But R does not explain A: real eigenvalues do not imply diagonalisability --- $\begin{psmallmatrix}1&1\\0&1\end{psmallmatrix}$ has the real eigenvalue $1$ twice
and cannot be diagonalised, and unitary matrices are diagonalisable by a unitary although their eigenvalues are complex. The
explanation of A is normality. Answer: (B)."""),

    dict(id="A8", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} Every unitary matrix has condition number $\kappa = 1$.\\[1mm]
\textbf{Statement II:} For every invertible matrix $A$, $\kappa(A^{-1}) = \kappa(A)$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""$\kappa(A) = \lVert A\rVert_2\lVert A^{-1}\rVert_2 = \sigma_{\max}/\sigma_{\min}$.
\textbf{I is true:} a unitary $U$ preserves lengths, so all its singular values equal $1$ and $\kappa = 1/1 = 1$.
\textbf{II is true:} the singular values of $A^{-1}$ are $1/\sigma_i$, so $\kappa(A^{-1}) = \frac{1/\sigma_{\min}}{1/\sigma_{\max}} = \kappa(A)$
(directly: $\lVert A^{-1}\rVert\,\lVert (A^{-1})^{-1}\rVert = \lVert A^{-1}\rVert\,\lVert A\rVert$). Answer: (A)."""),

    dict(id="A9", kind="mcq",
         q=r"""The $8\times8$ matrix of the 1-D discrete Laplacian has $2$ on the diagonal, $-1$ on the first upper and lower
off-diagonals and $0$ elsewhere. Its sparsity $s$ (maximum non-zeros per row) and its total number of non-zero entries are""",
         opts=[r"$s = 3$; $\ 22$", r"$s = 3$; $\ 24$", r"$s = 2$; $\ 22$", r"$s = 8$; $\ 64$"],
         sol=r"""Interior rows $2, \dots, 7$ contain $-1, 2, -1$: three non-zeros. The first and last rows contain only two ($2, -1$ and $-1, 2$).
So $s = 3$. Total: diagonal $8$ + upper off-diagonal $7$ + lower off-diagonal $7$ $= 22$ (not $3\times8 = 24$, because the boundary rows have
only two entries). Such matrices are \emph{sparse}: $s$ stays $3$ for any size $d$, which is the setting in which HHL's cost
$\propto s$ (or $s^2$) is small."""),

    dict(id="A10", kind="mcq",
         q=r"""The eigenvalues of the $n\times n$ Laplacian of A9 (with $2$ on the diagonal) are $\lambda_k = 2 - 2\cos\frac{k\pi}{n+1}$, $k = 1, \dots, n$.
For $n = 3$ the condition number is""",
         opts=[r"$3 + 2\sqrt2 \approx 5.83$", r"$\sqrt2 \approx 1.41$", r"$2$", r"$1 + \sqrt2 \approx 2.41$"],
         sol=r"""For $n = 3$: $\frac{k\pi}{4}$ with $k = 1, 2, 3$ gives
$\lambda_1 = 2 - 2\cdot\frac{\sqrt2}{2} = 2 - \sqrt2 \approx 0.586$, $\lambda_2 = 2 - 0 = 2$, $\lambda_3 = 2 + \sqrt2 \approx 3.414$. Hence
\[ \kappa = \frac{2+\sqrt2}{2-\sqrt2} = \frac{(2+\sqrt2)^2}{4 - 2} = \frac{6 + 4\sqrt2}{2} = 3 + 2\sqrt2 \approx 5.83. \]
For large $n$, $\lambda_1 \approx (\pi/(n+1))^2$ and $\kappa \approx 4(n+1)^2/\pi^2$: discretised differential operators become badly
conditioned as the grid is refined, which limits HHL's advantage for such problems."""),

    dict(id="A11", kind="mcq",
         q=r"""The vector $\vec b = (3, 0, 4, 0)$ is amplitude-encoded in two qubits (index $= $ binary number $q_0q_1$). Measuring both
qubits, the probability of the outcome $\ket{10}$ is""",
         opts=[r"$0.64$", r"$0.8$", r"$0.36$", r"$0.16$"],
         sol=r"""Normalise: $\lVert\vec b\rVert = \sqrt{9 + 16} = 5$, so
$\ket b = \tfrac35\ket{00} + \tfrac45\ket{10}$ (index $2 = 10_2$ has amplitude $4/5$).
Born rule: $P(10) = |4/5|^2 = 16/25 = 0.64$. The norm $\lVert\vec b\rVert = 5$ is \emph{not} stored in the state; it must be kept classically
(the notebook multiplies it back at the end)."""),

    dict(id="A12", kind="mcq",
         q=r"""With the eigenpairs of $A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$ given in A6 and $\vec b = (1, 0)$, write $\vec b = \sum_j\beta_j\ket{u_j}$. Then
$\vec x = \sum_j(\beta_j/\lambda_j)\ket{u_j}$ equals""",
         opts=[r"$\left(\tfrac23,\ -\tfrac13\right)$", r"$\left(\tfrac23,\ \tfrac13\right)$", r"$\left(\tfrac12,\ -\tfrac12\right)$",
               r"$\left(2,\ -1\right)$"],
         sol=r"""$\beta_1 = \braket{u_1|b} = \tfrac{1}{\sqrt2}$, $\beta_2 = \braket{u_2|b} = \tfrac{1}{\sqrt2}$. Divide by the eigenvalues $3$ and $1$:
\[ \vec x = \frac{1}{3\sqrt2}\cdot\frac{1}{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix} + \frac{1}{\sqrt2}\cdot\frac{1}{\sqrt2}\begin{pmatrix}1\\-1\end{pmatrix}
          = \begin{pmatrix}\tfrac16 + \tfrac12\\ \tfrac16 - \tfrac12\end{pmatrix} = \begin{pmatrix}\tfrac23\\ -\tfrac13\end{pmatrix}. \]
\emph{Check:} $A\vec x = (\tfrac43 - \tfrac13,\ \tfrac23 - \tfrac23) = (1, 0)$. HHL performs these three steps (expand in the eigenbasis, divide
by $\lambda_j$, recombine) coherently: phase estimation, controlled rotation, uncomputation."""),

    dict(id="A13", kind="match",
         q=r"""For $A = \operatorname{diag}(4,\ 1,\ 0.5,\ 3)$ match each quantity (List-I) with its value (List-II).""",
         match=([r"spectral norm $\lVert A\rVert_2$", r"spectral norm $\lVert A^{-1}\rVert_2$", r"condition number $\kappa(A)$",
                 r"determinant $\det A$"],
                [r"$8$", r"$4$", r"$6$", r"$2$"]),
         opts=[r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4",
               r"P$\to$1, Q$\to$4, R$\to$2, S$\to$3", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: $\lVert A\rVert_2 = $ largest $|\lambda| = 4$ (2).
\item Q: $A^{-1} = \operatorname{diag}(\tfrac14, 1, 2, \tfrac13)$, so $\lVert A^{-1}\rVert_2 = 1/|\lambda|_{\min} = 1/0.5 = 2$ (4).
\item R: $\kappa = \lVert A\rVert_2\lVert A^{-1}\rVert_2 = 4\times2 = 8$ (1).
\item S: $\det A = 4\cdot1\cdot0.5\cdot3 = 6$ (3).
\end{itemize}
Note that the determinant says nothing about conditioning: $\det(0.1\,I_{10}) = 10^{-10}$ is tiny while $\kappa = 1$."""),

    dict(id="A14", kind="mcq",
         q=r"""HHL needs a Hermitian matrix, so the non-Hermitian $A = \begin{pmatrix}1&2\\0&1\end{pmatrix}$ is embedded as
$\tilde A = \begin{pmatrix}0 & A\\ A^\dagger & 0\end{pmatrix}$ ($4\times4$). The eigenvalues of $\tilde A$ are""",
         opts=[r"$\pm(\sqrt2 + 1)$ and $\pm(\sqrt2 - 1)$", r"$1, 1, 1, 1$", r"$\pm1$, each twice", r"$3 \pm 2\sqrt2$"],
         sol=r"""$\tilde A^2 = \begin{psmallmatrix}AA^\dagger & 0\\ 0 & A^\dagger A\end{psmallmatrix}$, so the eigenvalues of $\tilde A$ are $\pm\sigma_i$, the singular values of $A$
with both signs. $A^\dagger A = \begin{psmallmatrix}1&2\\2&5\end{psmallmatrix}$ has trace $6$ and determinant $1$: eigenvalues $3 \pm 2\sqrt2 = (\sqrt2 \pm 1)^2$.
Hence $\sigma = \sqrt2 \pm 1$ and the spectrum of $\tilde A$ is $\{\pm(\sqrt2 + 1), \pm(\sqrt2 - 1)\} \approx \{\pm2.414, \pm0.414\}$.
The eigenvalues of $A$ itself ($1, 1$) are irrelevant here; $\kappa(\tilde A) = \frac{\sqrt2+1}{\sqrt2-1} = 3 + 2\sqrt2 \approx 5.83$, and HHL must handle
\emph{negative} eigenvalues."""),

    dict(id="A15", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} If $A$ is Hermitian, then $U = e^{iAt}$ is unitary for every real $t$.\\[1mm]
\textbf{Statement II:} If $A$ is any real square matrix, then $e^{iAt}$ is unitary for every real $t$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} with $A = V\Lambda V^\dagger$, $e^{iAt} = V\operatorname{diag}(e^{i\lambda_jt})V^\dagger$ and $|e^{i\lambda_jt}| = 1$ because $\lambda_j$ is real;
equivalently $(e^{iAt})^\dagger = e^{-iA^\dagger t} = e^{-iAt}$.
\textbf{II is false:} realness is not enough. For $A = \begin{psmallmatrix}0&1\\0&0\end{psmallmatrix}$, $e^{iAt} = I + iAt = \begin{psmallmatrix}1 & it\\ 0&1\end{psmallmatrix}$,
which maps $(0, 1)$ to $(it, 1)$ of length $\sqrt{1 + t^2} \neq 1$. Only real \emph{symmetric} matrices are guaranteed to work. This is why HHL embeds
non-Hermitian matrices first. Answer: (C)."""),

    dict(id="A16", kind="mcq",
         q=r"""A linear system has $\kappa(A) = 1000$. The right-hand side is known with relative error $\lVert\delta\vec b\rVert/\lVert\vec b\rVert = 10^{-6}$.
The best general upper bound on the relative error $\lVert\delta\vec x\rVert/\lVert\vec x\rVert$ of the solution is""",
         opts=[r"$10^{-3}$", r"$10^{-6}$", r"$10^{-9}$", r"$10^{-6}/\sqrt{1000} \approx 3\times10^{-8}$"],
         sol=r"""From $A(\vec x + \delta\vec x) = \vec b + \delta\vec b$: $\delta\vec x = A^{-1}\delta\vec b$, so $\lVert\delta\vec x\rVert \le \lVert A^{-1}\rVert\,\lVert\delta\vec b\rVert$, and
$\lVert\vec b\rVert \le \lVert A\rVert\,\lVert\vec x\rVert$. Dividing:
\[ \frac{\lVert\delta\vec x\rVert}{\lVert\vec x\rVert} \le \lVert A\rVert\lVert A^{-1}\rVert\,\frac{\lVert\delta\vec b\rVert}{\lVert\vec b\rVert} = \kappa\,\frac{\lVert\delta\vec b\rVert}{\lVert\vec b\rVert} = 1000\times10^{-6} = 10^{-3}. \]
The bound is attained when $\vec b$ points along the largest and $\delta\vec b$ along the smallest singular direction. $\kappa$ measures the
intrinsic sensitivity of the problem --- no algorithm, classical or quantum, removes it."""),

    dict(id="A17", kind="para", passage="A",
         q=r"""The eigenvalues of $A$ and its condition number are""",
         opts=[r"$\tfrac23$ and $\tfrac43$; $\ \kappa = 2$", r"$\tfrac23$ and $\tfrac43$; $\ \kappa = \tfrac43$", r"$1$ and $-\tfrac13$; $\ \kappa = 3$",
               r"$\tfrac13$ and $1$; $\ \kappa = 3$"],
         sol=r"""$\operatorname{tr}A = 2$, $\det A = 1 - \tfrac19 = \tfrac89$. The eigenvalues solve $\lambda^2 - 2\lambda + \tfrac89 = 0$:
$\lambda = 1 \pm \sqrt{1 - \tfrac89} = 1 \pm \tfrac13$, i.e.\ $\tfrac23$ (eigenvector $\tfrac{1}{\sqrt2}(1, 1)$) and $\tfrac43$ (eigenvector $\tfrac{1}{\sqrt2}(1, -1)$).
\[ \kappa = \frac{4/3}{2/3} = 2. \]
(The diagonal $1$ and the off-diagonal $-\tfrac13$ are entries, not eigenvalues.) With $\kappa = 2$ the notebook's ``auto'' time maps $\tfrac23 \to$
clock value $1$ and $\tfrac43 \to$ clock value $2$, so HHL is exact for this matrix."""),

    dict(id="A18", kind="para", passage="A",
         q=r"""The solution $\vec x = A^{-1}\vec b$ and the probability of measuring the system qubit as $\ket0$ in the normalised state $\ket x$ are""",
         opts=[r"$\vec x = \left(\tfrac98,\ \tfrac38\right)$; $\ 0.9$", r"$\vec x = \left(\tfrac98,\ \tfrac38\right)$; $\ 0.75$",
               r"$\vec x = \left(\tfrac98,\ -\tfrac38\right)$; $\ 0.9$", r"$\vec x = \left(1,\ \tfrac13\right)$; $\ \tfrac{1}{\sqrt{10}}$"],
         sol=r"""$A^{-1} = \frac{1}{\det A}\begin{psmallmatrix}1&\frac13\\ \frac13&1\end{psmallmatrix} = \frac98\begin{psmallmatrix}1&\frac13\\ \frac13&1\end{psmallmatrix}$, so
$\vec x = A^{-1}(1, 0)^T = \left(\tfrac98, \tfrac38\right)$. Check: $\tfrac98 - \tfrac13\cdot\tfrac38 = \tfrac98 - \tfrac18 = 1$ and $-\tfrac13\cdot\tfrac98 + \tfrac38 = 0$.
Normalised: $\vec x \propto (3, 1)$, $\ket x = \frac{1}{\sqrt{10}}(3\ket0 + \ket1)$, so $P(0) = \frac{9}{10} = 0.9$.
The \emph{ratio} $x_1/x_2 = 3$ survives in the quantum state; the overall scale $\lVert\vec x\rVert = \tfrac38\sqrt{10}$ does not."""),

    dict(id="A19", kind="mcq",
         q=r"""A $5\times5$ \emph{non-Hermitian} matrix is prepared for HHL exactly as in the notebook: first the Hermitian embedding
$\begin{psmallmatrix}0&A\\A^\dagger&0\end{psmallmatrix}$, then padding with an identity block to the next power of $2$. The number of system qubits $n_b$ is""",
         opts=[r"$4$", r"$3$", r"$5$", r"$10$"],
         sol=r"""Embedding doubles the dimension: $2\times5 = 10$. The next power of two is $16 = 2^4$; the extra $6$ dimensions are filled with an
identity block (eigenvalue $1$, decoupled from the system because $\vec b$ is padded with zeros). Hence $n_b = \log_2 16 = 4$ system qubits.
(Without the embedding $5 \to 8$ would need $3$ qubits.) In general $n_b = \lceil\log_2(2d)\rceil$ for a non-Hermitian $d\times d$ matrix."""),

    dict(id="A20", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} For a Hermitian positive-definite matrix the condition number equals $\lambda_{\max}/\lambda_{\min}$.\\[1mm]
\textbf{Reason (R):} The singular values of a Hermitian matrix are the absolute values of its eigenvalues.""",
         opts=AR_OPTIONS,
         sol=r"""By definition $\kappa = \sigma_{\max}/\sigma_{\min}$ (spectral norm of $A$ times that of $A^{-1}$).
\textbf{R is true:} for $A = V\Lambda V^\dagger$, $A^\dagger A = V\Lambda^2V^\dagger$, so $\sigma_j = \sqrt{\lambda_j^2} = |\lambda_j|$.
\textbf{A is true}, and it follows from R: for a positive-definite matrix all $\lambda_j > 0$, so $|\lambda_j| = \lambda_j$ and
$\kappa = \sigma_{\max}/\sigma_{\min} = \lambda_{\max}/\lambda_{\min}$. R is the correct explanation. Answer: (A)."""),
]
