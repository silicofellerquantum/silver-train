"""Section A - Mathematics and graph theory for QAOA (cuts, Max-Cut, spins, QUBO/Ising, probability, matrix exponentials).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""The ``house graph'' has the $5$ nodes $0, 1, 2, 3, 4$ and the $6$ unit-weight edges
$(0,1), (1,2), (2,3), (3,0)$ (a square) and $(0,4), (1,4)$ (a roof: node 4 forms a triangle with nodes 0 and 1).
A split is a bitstring $z = z_0z_1z_2z_3z_4$; its cut value $C(z)$ counts the edges whose two end nodes are in different
groups.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""A graph has $n = 15$ nodes. The number of bitstrings $z \in \{0,1\}^{15}$, and the number of \emph{distinct}
partitions $\{S, V\setminus S\}$ of the node set (unordered; $S = \varnothing$ allowed) that these bitstrings describe, are""",
         opts=[r"$32\,768$ and $16\,384$", r"$32\,768$ and $32\,768$", r"$16\,384$ and $8\,192$", r"$32\,768$ and $16\,383$"],
         sol=r"""Each node independently chooses group 0 or 1, so there are $2^{15} = 32\,768$ bitstrings.
A bitstring $z$ and its complement $\bar z$ (all bits flipped) describe the same unordered partition $\{S, V\setminus S\}$:
the two groups only swap their labels. Complementation pairs the $2^{15}$ strings perfectly (no string equals its own
complement), so the number of distinct partitions is $2^{15}/2 = 2^{14} = 16\,384$. The partition $\{\varnothing, V\}$
(strings $00\dots0$ and $11\dots1$, cut value $0$) is one of them because $S = \varnothing$ is allowed."""),

    dict(id="A2", kind="match",
         q=r"""Match each graph (List-I) with its maximum cut $C_{\max}$ (List-II). All edges have weight 1.""",
         match=([r"cycle $C_9$ (9 nodes, 9 edges)", r"complete graph $K_5$", r"complete bipartite graph $K_{3,3}$",
                 r"Petersen graph (10 nodes, 15 edges, 3-regular)"],
                [r"$12$", r"$9$", r"$6$", r"$8$"]),
         opts=[r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2",
               r"P$\to$2, Q$\to$3, R$\to$1, S$\to$4", r"P$\to$4, Q$\to$2, R$\to$3, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item $C_9$: an odd cycle cannot be 2-coloured, so at least one edge stays uncut; alternating colours around the cycle
cuts all but one edge: $C_{\max} = 9 - 1 = 8$ (4).
\item $K_5$: with $k$ nodes on one side every one of the $k(5-k)$ cross pairs is an edge; the maximum over $k$ is
$2 \cdot 3 = 6$ (3). In general $C_{\max}(K_n) = \lfloor n/2\rfloor\lceil n/2\rceil$.
\item $K_{3,3}$ is bipartite: putting the two natural sides in different groups cuts \emph{all} $9$ edges (2).
\item Petersen graph: it contains 5-cycles (odd), so not all 15 edges can be cut. Every one of its 12 five-cycles needs
an uncut edge, and the optimum leaves exactly 3 edges uncut: $C_{\max} = 12$ (1) (checked by brute force over all
$2^{10}$ splits in \texttt{verify\_answers.py}).
\end{itemize}
Hence P$\to$4, Q$\to$3, R$\to$2, S$\to$1."""),

    dict(id="A3", kind="mcq",
         q=r"""A weighted graph has the edges $(0,1)$ with weight 3, $(1,2)$ with weight 1, $(2,3)$ with weight 2, $(3,0)$ with
weight 4 and $(0,2)$ with weight 2. Every node is put into group 0 or group 1 independently with probability $\tfrac12$.
The expected cut value is""",
         opts=[r"$6$", r"$12$", r"$3$", r"$2.4$"],
         sol=r"""Write $C(z) = \sum_{(i,j)\in E} w_{ij}\,[z_i \neq z_j]$. By linearity of expectation
\[ \mathbb{E}[C] = \sum_{(i,j)} w_{ij}\,P(z_i \neq z_j). \]
For independent fair bits $P(z_i \neq z_j) = P(01) + P(10) = \tfrac14 + \tfrac14 = \tfrac12$ for \emph{every} edge, whatever the graph looks like.
So $\mathbb{E}[C] = \tfrac12 W$ with $W = 3 + 1 + 2 + 4 + 2 = 12$, i.e.\ $\mathbb{E}[C] = 6$. This is exactly the
``random-split baseline'' $W/2$ printed by the notebook."""),

    dict(id="A4", kind="mcq",
         q=r"""For the path graph $0 - 1 - 2$ (two unit edges), the cut value written as a polynomial in the binary variables
$z_0, z_1, z_2 \in \{0, 1\}$ is""",
         opts=[r"$z_0 + 2z_1 + z_2 - 2z_0z_1 - 2z_1z_2$", r"$z_0 + z_1 + z_2 - 2z_0z_1 - 2z_1z_2$",
               r"$z_0 + 2z_1 + z_2 - z_0z_1 - z_1z_2$", r"$2 - z_0 - 2z_1 - z_2 + 2z_0z_1 + 2z_1z_2$"],
         sol=r"""For bits $a, b$: $[a \neq b] = a \oplus b = a + b - 2ab$ (check: $00 \to 0$, $01 \to 1$, $10 \to 1$, $11 \to 1 + 1 - 2 = 0$).
Summing over the two edges:
\[ C = (z_0 + z_1 - 2z_0z_1) + (z_1 + z_2 - 2z_1z_2) = z_0 + 2z_1 + z_2 - 2z_0z_1 - 2z_1z_2 . \]
The middle node appears in both edges, hence the coefficient $2$ of $z_1$ (in general the linear coefficient of $z_i$ is
the weighted degree of node $i$). Check $z = 010$: $C = 2$ (both edges cut) and the polynomial gives $0 + 2 + 0 - 0 - 0 = 2$.
This is the QUBO form of Max-Cut."""),

    dict(id="A5", kind="mcq",
         q=r"""The notebook's default graph is a random graph on $n = 15$ nodes in which each of the possible edges is present
independently with probability $0.3$. The expected number of edges and its standard deviation are""",
         opts=[r"$31.5$ and $4.70$", r"$31.5$ and $22.05$", r"$63$ and $6.64$", r"$45$ and $3.97$"],
         sol=r"""There are $\binom{15}{2} = \frac{15\cdot 14}{2} = 105$ unordered node pairs, each an independent Bernoulli($0.3$) trial.
The edge count is Binomial($105, 0.3$):
\[ \mathbb{E}[|E|] = 105 \times 0.3 = 31.5, \qquad \mathrm{Var} = 105 \times 0.3 \times 0.7 = 22.05, \qquad \sigma = \sqrt{22.05} \approx 4.70. \]
The instance actually generated with \texttt{SEED = 42} has $30$ edges, well within one standard deviation (the notebook's
text ``about 31 edges'' refers to this expectation; its printed output says 30)."""),

    dict(id="A6", kind="mcq",
         q=r"""The 15-node graph of the notebook has the degree sequence (nodes $0$ to $14$)
\[ (d_0, \dots, d_{14}) = (2, 3, 5, 4, 5, 4, 6, 1, 4, 3, 4, 2, 4, 6, 7). \]
The number of edges, and the number of nodes of odd degree, are""",
         opts=[r"$30$ and $6$", r"$60$ and $6$", r"$30$ and $5$", r"$30$ and $9$"],
         sol=r"""\textbf{Handshake lemma:} every edge contributes 1 to the degree of each of its two end nodes, so
$\sum_i d_i = 2|E|$. Here $\sum_i d_i = 2+3+5+4+5+4+6+1+4+3+4+2+4+6+7 = 60$, hence $|E| = 30$.
Odd degrees: $3, 5, 5, 1, 3, 7$ (nodes 1, 2, 4, 7, 9, 14) --- six nodes. The number of odd-degree nodes is always even,
because $\sum_i d_i$ is even. (The degrees were recomputed from the edge list generated with \texttt{SEED = 42}.)"""),

    dict(id="A7", kind="ar", correct=2,
         q=r"""\textbf{Assertion (A):} The cycle $C_7$ (7 nodes, 7 unit edges) has $C_{\max} = 6$.\\[1mm]
\textbf{Reason (R):} Every graph with an odd number of nodes has $C_{\max} < |E|$.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is true:} $C_{\max} = |E|$ would require every edge to join the two groups, i.e.\ a proper 2-colouring;
going round an odd cycle the colours must alternate an odd number of times and cannot close up. So at least one edge
is uncut, and the alternating colouring $0101010$ cuts exactly $6$ edges.
\textbf{R is false:} $C_{\max} = |E|$ holds exactly for \emph{bipartite} graphs, and bipartiteness has nothing to do with the
parity of the number of nodes. Counter-example: the path on 3 nodes, or the star with 7 nodes (one centre joined to six
leaves) --- both bipartite, both with $C_{\max} = |E|$. What matters is an \emph{odd cycle}, not an odd node count. Answer: (C)."""),

    dict(id="A8", kind="stmt", correct=1,
         q=r"""\textbf{Statement I:} A polynomial-time algorithm that solves Max-Cut \emph{exactly} on every graph is known.\\[1mm]
\textbf{Statement II:} Finding a \emph{minimum} cut of a graph (both groups non-empty, non-negative weights) is NP-hard.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} Max-Cut is NP-hard (one of Karp's 21 NP-complete problems, decision version). No
polynomial-time exact algorithm is known, and none exists unless P = NP. The best known polynomial-time guarantee is the
Goemans--Williamson SDP rounding with expected ratio $\approx 0.878$; exact solution in the notebook uses brute force over $2^n$ splits.
\textbf{II is false:} the global minimum cut is polynomial --- e.g.\ the Stoer--Wagner algorithm, or $n-1$ max-flow/min-cut
computations. Maximising and minimising a cut have completely different complexity. Answer: (B)."""),

    dict(id="A9", kind="mcq",
         q=r"""For the triangle $K_3$ (unit weights) consider $H_C = \tfrac12\big(Z_0Z_1 + Z_1Z_2 + Z_0Z_2\big)$. Its smallest eigenvalue
and the number of computational basis states with that eigenvalue are""",
         opts=[r"$-\tfrac12$, six basis states", r"$-\tfrac32$, two basis states", r"$-\tfrac12$, three basis states",
               r"$+\tfrac32$, two basis states"],
         sol=r"""$H_C$ is diagonal: on $\ket{z}$ it equals $\tfrac12(s_0s_1 + s_1s_2 + s_0s_2)$ with $s_i = (-1)^{z_i} = \pm1$.
\begin{itemize}\itemsep0pt
\item $z = 000, 111$: all spins equal, every product $+1$: eigenvalue $\tfrac32$ (2 states).
\item one node differs from the other two (6 states such as $001, 010, \dots$): products $-1, -1, +1$, eigenvalue $-\tfrac12$.
\end{itemize}
$-\tfrac32$ would need all three products to be $-1$, but $(s_0s_1)(s_1s_2)(s_0s_2) = s_0^2 s_1^2 s_2^2 = +1$: impossible
(the triangle is \emph{frustrated}). So $E_0 = -\tfrac12$ with 6 basis states, corresponding to $C = \tfrac32 - E_0 = 2 = C_{\max}$."""),

    dict(id="A10", kind="mcq",
         q=r"""For the notebook's 15-node graph ($30$ unit edges), a split $z$ is chosen uniformly at random from all $2^{15}$
bitstrings. The variance of the cut value $C(z)$ is""",
         opts=[r"$7.5$", r"$15$", r"$30$", r"$3.75$"],
         sol=r"""Write $C = \sum_{e} X_e$ with $X_e = [z_i \neq z_j]$ for edge $e = (i,j)$. Each $X_e$ is Bernoulli($\tfrac12$):
$\mathrm{Var}(X_e) = \tfrac12\cdot\tfrac12 = \tfrac14$.
\textbf{Pairwise independence:} for two edges sharing a node, $e = (i,j)$, $f = (j,k)$: $X_e = z_i \oplus z_j$ and
$X_f = z_j \oplus z_k$ are independent fair bits (given any $z_j$, $z_i$ and $z_k$ are independent and fair), so
$\mathrm{Cov}(X_e, X_f) = 0$; edges with no common node are trivially independent. Hence
\[ \mathrm{Var}(C) = \sum_e \mathrm{Var}(X_e) = \frac{|E|}{4} = \frac{30}{4} = 7.5 . \]
(Exact enumeration of the $2^{15}$ cut values of the actual graph gives $7.5$.) The standard deviation is $\approx 2.74$, so
$C_{\max} = 25$ lies $(25 - 15)/2.74 \approx 3.7$ standard deviations above the random mean $15$."""),

    dict(id="A11", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} Because all the operators $Z_iZ_j$ commute with one another,\\
\hspace*{1.5em}$e^{-i\gamma H_C} = \prod_{(i,j)\in E} e^{-i\gamma\frac{w_{ij}}{2} Z_iZ_j}$ holds exactly, with the factors in any order.\\[1mm]
\textbf{Statement II:} Because the operators $X_i$ also commute with one another,\\
\hspace*{1.5em}$e^{-i\beta H_M}\,e^{-i\gamma H_C} = e^{-i(\beta H_M + \gamma H_C)}$ holds for all $\beta, \gamma$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""For matrices, $e^{A+B} = e^Ae^B$ holds whenever $[A, B] = 0$, but not in general.
\textbf{I is true:} all $Z_iZ_j$ are diagonal, hence commute, so the exponential of their sum factorises exactly into one
two-qubit gate per edge, in any order. That is why the notebook can loop over the edges in list order.
\textbf{II is false:} commuting \emph{within} $H_M$ only gives $e^{-i\beta H_M} = \prod_i e^{-i\beta X_i}$. The relevant
commutator is $[H_M, H_C] \neq 0$ (e.g.\ $X_i$ anticommutes with $Z_iZ_j$), so $e^{-i\beta H_M}e^{-i\gamma H_C} \neq
e^{-i(\beta H_M + \gamma H_C)}$ in general (Baker--Campbell--Hausdorff). The non-commutativity is essential: if they commuted,
QAOA could not move probability between bitstrings in a useful way. Answer: (C)."""),

    dict(id="A12", kind="mcq",
         q=r"""The expectation value $\langle Z_iZ_j\rangle$ is estimated as the average of the $\pm1$ values $(-1)^{z_i \oplus z_j}$ over
$S$ independent shots. The smallest $S$ that guarantees a standard error of at most $0.01$ \emph{for every state} is""",
         opts=[r"$10\,000$", r"$100$", r"$2\,500$", r"$40\,000$"],
         sol=r"""Each shot gives a random variable $Y = \pm1$ with mean $\mu = \langle Z_iZ_j\rangle$ and variance
$\mathrm{Var}(Y) = \mathbb{E}[Y^2] - \mu^2 = 1 - \mu^2 \le 1$ (worst case $\mu = 0$). The average of $S$ independent shots has
standard error $\sqrt{(1-\mu^2)/S} \le 1/\sqrt S$. Requiring $1/\sqrt S \le 0.01$ gives $S \ge 10^4$.
(For the $0/1$ indicator $[z_i \neq z_j]$ the variance is at most $\tfrac14$, giving $2\,500$ --- but that estimates
$P(\text{cut})$, whose error is half the error of $\langle Z_iZ_j\rangle = 1 - 2P(\text{cut})$.)"""),

    dict(id="A13", kind="match",
         q=r"""In the previously recorded notebook run, an optimal split is sampled with probability $P_{\text{opt}} = 0.06$ per
shot. Treat the shots as independent and match each quantity (List-I) with its value (List-II).""",
         match=([r"standard deviation of the number of optimal shots among $4000$",
                 r"probability that $10$ shots contain \emph{no} optimal split",
                 r"probability that $10$ shots contain \emph{at least one} optimal split",
                 r"smallest number of shots giving $\ge 99\,\%$ probability of at least one optimal split"],
                [r"$75$", r"$15.0$", r"$0.539$", r"$0.461$"]),
         opts=[r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1", r"P$\to$2, Q$\to$4, R$\to$3, S$\to$1",
               r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2", r"P$\to$1, Q$\to$4, R$\to$3, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: the count is Binomial($4000, 0.06$), $\sigma = \sqrt{4000\cdot0.06\cdot0.94} = \sqrt{225.6} \approx 15.0$ (2).
(Its mean is $240$, exactly the number of optimal shots recorded.)
\item Q: $P(\text{none in } 10) = 0.94^{10} \approx 0.539$ (3).
\item R: complement: $1 - 0.539 = 0.461$ (4).
\item S: need $1 - 0.94^S \ge 0.99 \iff 0.94^S \le 0.01 \iff S \ge \dfrac{\ln 0.01}{\ln 0.94} = \dfrac{-4.605}{-0.0619} \approx 74.4$,
so $S = 75$ (1).
\end{itemize}
Hence P$\to$2, Q$\to$3, R$\to$4, S$\to$1."""),

    dict(id="A14", kind="mcq",
         q=r"""The notebook's brute-force search reports four optimal bitstrings (cut value $25$):
\[ \texttt{001101110100010},\quad \texttt{001101110110010},\quad \texttt{110010001001101},\quad \texttt{110010001011101}. \]
The number of distinct optimal partitions, and the bitstring describing the same partition as $\texttt{001101110100010}$, are""",
         opts=[r"$2$; $\ \texttt{110010001011101}$", r"$4$; $\ \texttt{110010001011101}$",
               r"$2$; $\ \texttt{110010001001101}$", r"$2$; $\ \texttt{010001011101100}$"],
         sol=r"""$C(z) = C(\bar z)$: flipping every bit swaps the group labels and keeps every edge cut or uncut. So optimal strings
come in complementary pairs, and each pair is one partition.
Complement of $\texttt{001101110100010}$: flip each bit $\Rightarrow \texttt{110010001011101}$, which is in the list.
Complement of $\texttt{001101110110010}$: $\texttt{110010001001101}$, also in the list.
So the 4 strings form 2 pairs: \textbf{2 distinct optimal partitions}. (The two partitions differ only in node 10,
which has as many neighbours on each side when the rest is fixed.)"""),

    dict(id="A15", kind="mcq",
         q=r"""The QUBO function $f(x_0, x_1) = 2x_0x_1 - x_0 - x_1$ with $x_i \in \{0,1\}$ is rewritten with spins via
$x_i = (1 - s_i)/2$, $s_i \in \{+1, -1\}$. The result is""",
         opts=[r"$\tfrac12\,(s_0s_1 - 1)$", r"$\tfrac12\,(1 - s_0s_1)$", r"$\tfrac14\,(s_0s_1 - 1)$",
               r"$\tfrac12\,s_0s_1 - 1 + \tfrac12(s_0 + s_1)$"],
         sol=r"""Substitute:
$x_0x_1 = \tfrac14(1 - s_0)(1 - s_1) = \tfrac14(1 - s_0 - s_1 + s_0s_1)$, so $2x_0x_1 = \tfrac12(1 - s_0 - s_1 + s_0s_1)$;
$-x_0 - x_1 = -\tfrac12(1 - s_0) - \tfrac12(1 - s_1) = -1 + \tfrac12(s_0 + s_1)$.
Adding, the linear terms cancel:
\[ f = \tfrac12 - \tfrac12(s_0+s_1) + \tfrac12 s_0s_1 - 1 + \tfrac12(s_0+s_1) = \tfrac12(s_0s_1 - 1). \]
Check: $x = 01$: $f = -1$ and $s = (+1, -1)$ gives $\tfrac12(-1-1) = -1$. So $f = -[x_0 \neq x_1]$: minimising $f$ is Max-Cut
on one edge, and its Ising form is (up to a constant) $\tfrac12 Z_0Z_1$ --- one term of $H_C$."""),

    dict(id="A16", kind="mcq",
         q=r"""Let $f(\theta) = a + b\cos\theta + c\sin\theta$ with constants $a, b, c$. Which expression equals $f'(\theta)$
\emph{exactly} for every $\theta$?""",
         opts=[r"$\tfrac12\big[f(\theta + \tfrac\pi2) - f(\theta - \tfrac\pi2)\big]$",
               r"$\tfrac1\pi\big[f(\theta + \tfrac\pi2) - f(\theta - \tfrac\pi2)\big]$",
               r"$\tfrac12\big[f(\theta + \tfrac\pi4) - f(\theta - \tfrac\pi4)\big]$",
               r"$f(\theta + \tfrac\pi2) - f(\theta)$"],
         sol=r"""$f'(\theta) = -b\sin\theta + c\cos\theta$. Using $\cos(\theta \pm \tfrac\pi2) = \mp\sin\theta$ and $\sin(\theta \pm \tfrac\pi2) = \pm\cos\theta$:
\[ f(\theta + \tfrac\pi2) - f(\theta - \tfrac\pi2) = b(-\sin\theta - \sin\theta) + c(\cos\theta + \cos\theta) = 2(-b\sin\theta + c\cos\theta) = 2f'(\theta). \]
So $f'(\theta) = \tfrac12[f(\theta + \tfrac\pi2) - f(\theta - \tfrac\pi2)]$ exactly --- not an approximation. This is the
\emph{parameter-shift rule}: an expectation value depends on the angle of one gate $e^{-i\theta P/2}$ ($P$ a Pauli
operator) exactly in this form, so two circuit runs give the exact derivative."""),

    dict(id="A17", kind="para", passage="A",
         q=r"""The maximum cut $C_{\max}$ of the house graph, and the number of bitstrings $z$ that achieve it, are""",
         opts=[r"$5$ and $4$", r"$6$ and $2$", r"$5$ and $2$", r"$5$ and $6$"],
         sol=r"""\textbf{Upper bound:} the triangle $0,1,4$ is an odd cycle, so at least one of its three edges is uncut: $C \le 6 - 1 = 5$.
\textbf{Achieving 5:} the square $0{-}1{-}2{-}3$ is bipartite; put $\{0, 2\}$ and $\{1, 3\}$ in different groups (all 4 square
edges cut). Node 4 is joined to 0 and 1, which are now in different groups, so exactly one roof edge is cut \emph{whichever}
group node 4 joins: total $4 + 1 = 5$.
\textbf{Counting:} a cut of 5 must cut all four square edges (if a square edge were uncut, at most $3 + 2 = 5$ with both roof
edges cut, which needs $z_4 \neq z_0$ and $z_4 \neq z_1$, so $z_0 = z_1$ --- but then edge $(0,1)$ is uncut too, giving at most 4).
So $z_0z_1z_2z_3 \in \{0101, 1010\}$ and $z_4$ is free: $2 \times 2 = 4$ strings: $01010, 01011, 10100, 10101$."""),

    dict(id="A18", kind="para", passage="A",
         q=r"""A split of the house graph is drawn uniformly at random. The probability that it is optimal, and the ratio
$\mathbb{E}[C]/C_{\max}$ of the random split, are""",
         opts=[r"$\tfrac18$ and $0.6$", r"$\tfrac1{16}$ and $0.6$", r"$\tfrac18$ and $0.5$", r"$\tfrac14$ and $0.6$"],
         sol=r"""There are $2^5 = 32$ equally likely bitstrings and $4$ are optimal (previous question): $P = \tfrac{4}{32} = \tfrac18$.
The expected cut of a uniform split is $W/2 = 6/2 = 3$ (every edge is cut with probability $\tfrac12$), so the random
approximation ratio is $3/5 = 0.6$. Any useful algorithm --- QAOA included --- must beat $r = 0.6$ on this graph."""),

    dict(id="A19", kind="mcq",
         q=r"""For the Petersen graph (10 nodes, 15 unit edges, $C_{\max} = 12$), the smallest eigenvalue of
$H_C = \sum_{(i,j)\in E}\tfrac12 Z_iZ_j$ is""",
         opts=[r"$-4.5$", r"$-7.5$", r"$-12$", r"$+4.5$"],
         sol=r"""On a basis state $\ket z$, $C(z) = \tfrac W2 - H_C(z)$ with $W = 15$, because
$[z_i \neq z_j] = \tfrac12(1 - s_is_j)$ and summing gives $C = \tfrac{|E|}{2} - \tfrac12\sum s_is_j$.
Minimising $H_C$ is maximising $C$: $E_0 = \tfrac W2 - C_{\max} = 7.5 - 12 = -4.5$.
$-7.5$ would require all 15 products $s_is_j = -1$ (every edge cut), impossible because the graph has odd cycles."""),

    dict(id="A20", kind="ar", correct=3,
         q=r"""The notebook's \texttt{local\_search\_maxcut} repeatedly flips single nodes while this strictly increases the cut
(unit weights), starting from random splits.\\[1mm]
\textbf{Assertion (A):} Every split returned by one run of this greedy procedure is an optimal split.\\[1mm]
\textbf{Reason (R):} When no single flip increases the cut, every node has at least half of its incident edges cut.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} flipping node $v$ turns its cut edges into uncut ones and vice versa, so the cut changes by
$u_v - c_v$ ($u_v$, $c_v$ = numbers of uncut and cut edges at $v$). At a local optimum no flip strictly improves, so
$u_v - c_v \le 0$, i.e.\ $c_v \ge d_v/2$. Summing over $v$ (each cut edge counted twice): $2C \ge \sum_v d_v/2 = |E|$, so a
local optimum always has $C \ge |E|/2$.
\textbf{A is false:} a local optimum need not be global --- greedy search can get stuck. That is exactly why the notebook
restarts it 200 times and labels its result ``heuristic local search (best known, NOT guaranteed optimal)''. Answer: (D)."""),
]
