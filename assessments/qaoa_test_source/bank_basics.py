"""Section C - QAOA Basics (cost and mixer Hamiltonians, circuit, gate decomposition, parameters, p = 1 analysis,
objective function, interpretation of results, mapping to the notebook's code).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "C": r"""QAOA with $p = 1$ is applied to the path graph $0 - 1 - 2$ (edges $(0,1)$ and $(1,2)$, unit weights,
$C_{\max} = 2$) in the notebook's convention, with $\gamma = \pi/2$ and $\beta = -\pi/8$. For a triangle-free graph and $p = 1$
the edge correlator in this convention is
\[ \langle Z_uZ_v\rangle = \tfrac12\,\sin4\beta\,\sin\gamma\,\big(\cos^{d_u - 1}\gamma + \cos^{d_v - 1}\gamma\big), \]
where $d_u$, $d_v$ are the degrees of the two end nodes.""",
}

QUESTIONS = [
    dict(id="C1", kind="mcq",
         q=r"""For \texttt{edges = [(0, 1, 2.0), (1, 2, 1.0)]} the notebook's function \texttt{cost\_hamiltonian()} returns the operator""",
         opts=[r"$1.0\,Z_0Z_1 + 0.5\,Z_1Z_2$", r"$2.0\,Z_0Z_1 + 1.0\,Z_1Z_2$", r"$1.5\,I - 1.0\,Z_0Z_1 - 0.5\,Z_1Z_2$",
               r"$-1.0\,Z_0Z_1 - 0.5\,Z_1Z_2$"],
         sol=r"""The code builds \texttt{term = 0.5 * w * cudaq.spin.z(i) * cudaq.spin.z(j)} for every edge and adds the terms:
$H_C = \sum \tfrac{w_{ij}}{2}Z_iZ_j = \tfrac{2.0}{2}Z_0Z_1 + \tfrac{1.0}{2}Z_1Z_2 = 1.0\,Z_0Z_1 + 0.5\,Z_1Z_2$.
No constant term is included; the cut is recovered as $C = \tfrac W2 - H_C$ with $W = 3$, i.e.\ $C = 1.5\,I - H_C$ --- the third
option is the cut operator $C$, not what the function returns."""),

    dict(id="C2", kind="mcq",
         q=r"""In the notebook's convention, if all cost angles are zero ($\gamma_1 = \dots = \gamma_p = 0$), the expected cut
$F_p(\boldsymbol\gamma, \boldsymbol\beta)$ equals""",
         opts=[r"$W/2$ for every choice of $\boldsymbol\beta$", r"$C_{\max}$", r"$0$", r"$W/2 + \tfrac12\sum_l \sin 4\beta_l$"],
         sol=r"""With $\gamma_l = 0$ every cost layer is the identity, so the state is $\prod_l e^{-i\beta_l H_M}\ket+^{\otimes n}$.
$\ket+^{\otimes n}$ is an eigenstate of $H_M$ with eigenvalue $n$ (B14), so each mixer only multiplies it by the phase $e^{-i\beta_l n}$.
The state remains $\ket+^{\otimes n}$ (up to a global phase), all $2^n$ splits are equally likely, and $\langle C\rangle = W/2$
(the random baseline; $15$ for the notebook's graph). The cost layer creates the relative phases that the mixer turns into
probability changes; without it nothing happens."""),

    dict(id="C3", kind="mcq",
         q=r"""In the notebook's kernel each edge $(i, j)$ with weight $w$ is implemented as
\texttt{x.ctrl(q[i], q[j]); rz(gamma * w, q[j]); x.ctrl(q[i], q[j])}. With $R_Z(\theta) = e^{-i\theta Z/2}$, this sequence equals""",
         opts=[r"$e^{-i\gamma\,(w/2)\,Z_iZ_j}$", r"$e^{-i\gamma\, w\, Z_iZ_j}$", r"$e^{-i\gamma\,(w/2)\,Z_j}$", r"$e^{+i\gamma\,(w/2)\,Z_iZ_j}$"],
         sol=r"""Act on a basis state $\ket{z_i z_j}$. The first CNOT gives $\ket{z_i,\, z_i\oplus z_j}$. $R_Z(\theta)$ on qubit $j$ multiplies by
$e^{-i\theta/2}$ if $z_i \oplus z_j = 0$ and by $e^{+i\theta/2}$ if $z_i \oplus z_j = 1$, i.e.\ by $e^{-i(\theta/2)\, s_is_j}$ with $s = (-1)^z$.
The second CNOT restores $\ket{z_iz_j}$. So the sequence is the diagonal operator $e^{-i(\theta/2)Z_iZ_j}$, and with
$\theta = \gamma w$: $e^{-i\gamma(w/2)Z_iZ_j}$ --- exactly the factor of $e^{-i\gamma H_C}$ for that edge. (Equivalently: CNOT
conjugation maps $I\otimes Z$ to $Z\otimes Z$, see B7.)"""),

    dict(id="C4", kind="mcq",
         q=r"""Why does the notebook apply \texttt{rx(2.0 * beta, q)} (and not \texttt{rx(beta, q)}) in the mixer layer?""",
         opts=[r"Because $R_X(\theta) = e^{-i\theta X/2}$, so $R_X(2\beta) = e^{-i\beta X}$, the factor of $e^{-i\beta H_M}$ for one qubit",
               r"Because the mixer Hamiltonian is $H_M = 2\sum_i X_i$",
               r"Because the mixer must be applied twice per layer",
               r"Because $\beta$ is measured in units of $\pi/2$"],
         sol=r"""The target is $U_M(\beta) = e^{-i\beta H_M}$ with $H_M = \sum_i X_i$. Since the $X_i$ commute,
$e^{-i\beta H_M} = \prod_i e^{-i\beta X_i}$. CUDA-Q's \texttt{rx}, like the standard definition, is $R_X(\theta) = e^{-i\theta X/2}$ (half-angle
convention). Matching $e^{-i\theta X/2} = e^{-i\beta X}$ gives $\theta = 2\beta$. The same half-angle convention is why the cost gate
uses \texttt{rz(gamma * w)} for the coefficient $w/2$ (previous question)."""),

    dict(id="C5", kind="mcq",
         q=r"""Using the notebook's gate structure ($H$ on every qubit, then per layer 3 gates per edge and one $R_X$ per qubit), a
graph with $n = 20$ nodes and $|E| = 57$ edges at depth $p = 3$ needs how many gates (excluding measurements), and how many of
them are CNOTs?""",
         opts=[r"$593$ and $342$", r"$573$ and $342$", r"$593$ and $171$", r"$402$ and $228$"],
         sol=r"""The notebook's estimate is \texttt{gates = N + p * (3 * len(edges) + N)}.
Total: $n + p(3|E| + n) = 20 + 3\,(171 + 20) = 20 + 573 = 593$.
CNOTs: 2 per edge per layer, $2p|E| = 2\cdot3\cdot57 = 342$. (The remaining $593 - 342 = 251$ are $20$ $H$, $171$ $R_Z$ and $60$ $R_X$.)
For the default ($n = 15$, $|E| = 30$, $p = 2$) the same formula gives $225$ gates and $120$ CNOTs, as in the recorded statistics."""),

    dict(id="C6", kind="match",
         q=r"""For the notebook's default run ($n = 15$ nodes, $|E| = 30$ edges, $p = 2$), match each quantity (List-I) with its value
(List-II).""",
         match=([r"number of variational parameters", r"number of $R_Z$ gates",
                 r"number of single-qubit gates ($H$, $R_Z$ and $R_X$ together)", r"number of CNOT (\texttt{cx}) gates"],
                [r"$105$", r"$120$", r"$4$", r"$60$"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: $2p = 4$ angles $(\gamma_1, \gamma_2, \beta_1, \beta_2)$ (3).
\item Q: one $R_Z$ per edge per layer: $p|E| = 60$ (4).
\item R: $n$ Hadamards $+ p|E|$ $R_Z$ $+ pn$ $R_X$ $= 15 + 60 + 30 = 105$ (1).
\item S: two CNOTs per edge per layer: $2p|E| = 120$ (2).
\end{itemize}
These agree with the recorded \texttt{gate counts by type}: \texttt{h 15, cx 120, rz 60, rx 30} ($225$ gates in total)."""),

    dict(id="C7", kind="mcq",
         q=r"""For a single unit edge and $p = 1$ in the notebook's convention, the expected cut is
$F_1(\gamma, \beta) = \tfrac12 - \tfrac12\sin4\beta\,\sin\gamma$. Its maximum value, and one point where it is reached, are""",
         opts=[r"$1$, at $(\gamma, \beta) = (\pi/2,\, -\pi/8)$", r"$1$, at $(\gamma, \beta) = (\pi/2,\, \pi/8)$",
               r"$\tfrac12$, reached for all angles", r"$\tfrac34$, at $(\gamma, \beta) = (\pi/2,\, -\pi/8)$"],
         sol=r"""\textbf{Derivation of the formula.} $\langle Z_0Z_1\rangle = \bra{+ +}U_C^\dagger U_M^\dagger Z_0Z_1 U_MU_C\ket{+ +}$.
Mixer (B11 on each qubit, $c = \cos2\beta$, $s = \sin2\beta$): $U_M^\dagger Z_0Z_1U_M = (cZ_0 + sY_0)(cZ_1 + sY_1)$.
Cost $U_C = e^{-i\gamma Z_0Z_1/2}$: $Z_0Z_1$ and $Y_0Y_1$ commute with it; $Y_0Z_1$ anticommutes, so
$U_C^\dagger Y_0Z_1U_C = Y_0Z_1e^{-i\gamma Z_0Z_1} = \cos\gamma\,Y_0Z_1 - i\sin\gamma\,Y_0Z_0 = \cos\gamma\,Y_0Z_1 + \sin\gamma\,X_0$
(using $YZ = iX$); likewise $Z_0Y_1 \to \cos\gamma\,Z_0Y_1 + \sin\gamma\,X_1$. In $\ket{++}$ only $X_0$, $X_1$ have non-zero
expectation ($=1$), so $\langle Z_0Z_1\rangle = 2cs\sin\gamma = \sin4\beta\sin\gamma$ and $F_1 = \tfrac12(1 - \sin4\beta\sin\gamma)$.
\textbf{Maximum.} $F_1 = 1$ needs $\sin4\beta\sin\gamma = -1$: e.g.\ $\gamma = \pi/2$, $\beta = -\pi/8$ (or $\gamma = -\pi/2$, $\beta = \pi/8$).
Then the edge is cut with certainty: for one edge, $p = 1$ already solves Max-Cut."""),

    dict(id="C8", kind="mcq",
         q=r"""With optimally chosen angles, the best approximation ratio of $p = 1$ QAOA on the ring (cycle) $C_8$ is""",
         opts=[r"$3/4$", r"$0.6924$", r"$5/6$", r"$1$"],
         sol=r"""The ring is 2-regular and triangle-free. Each edge has $d_u = d_v = 2$, so (formula printed in the passage for C19 and C20,
which holds for every triangle-free graph) $\langle Z_uZ_v\rangle = \sin4\beta\sin\gamma\cos\gamma = \tfrac12\sin4\beta\,\sin2\gamma$, and per edge
\[ \langle C_{uv}\rangle = \tfrac12 - \tfrac14\sin4\beta\,\sin2\gamma \le \tfrac12 + \tfrac14 = \tfrac34, \]
reached at $\sin4\beta\sin2\gamma = -1$, e.g.\ $\gamma = \pi/4$, $\beta = -\pi/8$. All 8 edges are equivalent:
$F_1 = 8\cdot\tfrac34 = 6$ and $C_{\max} = 8$ (even ring, bipartite): $r = 6/8 = 3/4$. A numerical optimisation over the angles
gives the same value."""),

    dict(id="C9", kind="stmt", correct=0,
         q=r"""\textbf{Statement I:} For $p = 1$ with optimal angles, QAOA achieves $\langle C\rangle \ge 0.6924\,C_{\max}$ on every 3-regular graph.\\[1mm]
\textbf{Statement II:} For every graph, the optimal value $F_p^* = \max_{\boldsymbol\gamma,\boldsymbol\beta}F_p$ satisfies $F_{p+1}^* \ge F_p^*$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} Farhi, Goldstone and Gutmann (2014) showed that $p = 1$ QAOA gives at least $0.6924\,C_{\max}$ on
3-regular graphs (worst case over local neighbourhoods).
\textbf{II is true:} the depth-$(p+1)$ circuit contains the depth-$p$ circuit as a special case: setting $\gamma_{p+1} = \beta_{p+1} = 0$
makes the extra layer the identity. So the maximum over a larger set cannot be smaller. (With a \emph{finite} optimizer
budget the \emph{found} value can still be worse --- more parameters are harder to optimise.) Answer: (A)."""),

    dict(id="C10", kind="mcq",
         q=r"""Which statement about the limit of large depth $p$ (angles optimised for each $p$) is correct?""",
         opts=[r"$F_p^* \to C_{\max}$ as $p \to \infty$, because a Trotterised adiabatic path is one admissible choice of angles",
               r"$F_p^*/C_{\max} \to 0.878$, the Goemans--Williamson ratio",
               r"$F_p^* \to W/2$, because deep circuits randomise the state",
               r"$F_p^* = C_{\max}$ already at $p = 1$ for every graph"],
         sol=r"""Farhi et al.\ argue: a slow interpolation from the mixer to the cost Hamiltonian (adiabatic theorem), discretised into
small time steps (Trotterisation), is a QAOA circuit with special angles. As $p \to \infty$ these special angles drive the
initial state to an optimal split, so the \emph{optimal} angles do at least as well: $F_p^* \to C_{\max}$.
$0.878$ is a classical polynomial-time guarantee, not a QAOA limit; $p = 1$ is exact only for special graphs (one edge, the
triangle)."""),

    dict(id="C11", kind="mcq",
         q=r"""In the notebook the angle vector is $\texttt{params} = [\gamma_1, \dots, \gamma_p, \beta_1, \dots, \beta_p]$; the kernel reads
\texttt{gamma = params[layer]} and \texttt{beta = params[layers + layer]}. For $p = 3$ and
\texttt{params = [a, b, c, d, e, f]}, the mixer angle $\beta_2$ of the second layer is""",
         opts=[r"$e$", r"$d$", r"$b$", r"$f$"],
         sol=r"""Layers are numbered \texttt{layer = 0, 1, 2} in the code. The second layer is \texttt{layer = 1}:
\texttt{gamma = params[1] = b} and \texttt{beta = params[3 + 1] = params[4] = e}.
So $\beta_2 = e$ (and $\gamma_2 = b$). The vector is \emph{blocked} (all $\gamma$ first, then all $\beta$), not interleaved
$(\gamma_1, \beta_1, \gamma_2, \beta_2, \dots)$."""),

    dict(id="C12", kind="mcq",
         q=r"""With \texttt{INITIAL\_PARAMETERS = "ramp"} the notebook starts from \texttt{np.linspace(0.2, 0.6, p)} for the $\gamma$'s and
\texttt{np.linspace(0.6, 0.2, p)} for the $\beta$'s. For $p = 5$, the initial values of $\gamma_3$ and $\beta_2$ are""",
         opts=[r"$0.4$ and $0.5$", r"$0.4$ and $0.3$", r"$0.36$ and $0.52$", r"$0.3$ and $0.5$"],
         sol=r"""\texttt{np.linspace(a, b, p)} returns $p$ equally spaced values \emph{including both end points}; the step is $(b - a)/(p - 1)$.
$\gamma$: step $(0.6 - 0.2)/4 = 0.1$: $(0.2, 0.3, 0.4, 0.5, 0.6)$, so $\gamma_3 = 0.4$.
$\beta$: $(0.6, 0.5, 0.4, 0.3, 0.2)$, so $\beta_2 = 0.5$.
(A step of $0.4/5 = 0.08$, which excludes an end point, would give $0.36$ and $0.52$.)"""),

    dict(id="C13", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} The notebook's objective function returns $\langle H_C\rangle$ and SciPy \emph{minimises} it; this is
equivalent to \emph{maximising} the expected cut.\\[1mm]
\textbf{Reason (R):} $\langle C\rangle = W/2 - \langle H_C\rangle$, and $W/2$ does not depend on the angles.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} $[z_i \neq z_j] = \tfrac12(1 - s_is_j)$, so $C = \tfrac W2 I - H_C$ as operators, and taking expectations gives R.
\textbf{A is true because of R:} $\langle C\rangle$ is a constant minus $\langle H_C\rangle$, so the angles that minimise $\langle H_C\rangle$ are exactly
those that maximise $\langle C\rangle$. The code records \texttt{history\_cut.append(total\_weight / 2 - energy)} after each
evaluation and returns \texttt{energy} to \texttt{minimize}. R is the correct explanation. Answer: (A)."""),

    dict(id="C14", kind="mcq",
         q=r"""At the notebook's default starting point $\boldsymbol\gamma = (0.2, 0.6)$, $\boldsymbol\beta = (0.6, 0.2)$, an exact
state-vector calculation for its 15-node graph gives $\langle C\rangle \approx 9.18$, \emph{below} the random baseline $W/2 = 15$.
The best explanation is""",
         opts=[r"In the convention $e^{-i\beta H_M}e^{-i\gamma H_C}$, the start state $\ket+^{\otimes n}$ is the \emph{highest} eigenstate of $H_M$, so an annealing-like ramp with $\gamma, \beta > 0$ heads for the highest eigenstate of $H_C$, the \emph{smallest} cut; heading for the largest cut needs $\gamma$ and $\beta$ of opposite sign",
               r"The value is meaningless: \texttt{cudaq.observe} returns random numbers before the optimizer has converged",
               r"It indicates a bug, because a QAOA state can never have $\langle C\rangle < W/2$",
               r"Every $p = 2$ starting point gives $\langle C\rangle < W/2$; only $p \ge 3$ can beat random guessing"],
         sol=r"""\textbf{Small-angle check.} For one edge $F_1 = \tfrac12 - \tfrac12\sin4\beta\sin\gamma$ (C7): with $\gamma > 0$ and small $\beta > 0$ the
cut \emph{decreases}; it increases only if $\sin4\beta\,\sin\gamma < 0$.
\textbf{Annealing picture.} $e^{-i\beta H_M}e^{-i\gamma H_C}$ with positive angles is one Trotter step of evolution under
$\beta H_M + \gamma H_C$. $\ket+^{\otimes n}$ is the \emph{top} eigenstate of $H_M$ (B14); an adiabatic ramp keeps the top
eigenstate, which at the end is the top eigenstate of $H_C$ --- the \emph{minimum} cut. For the maximum cut one needs
$H_M \to -H_C$, i.e.\ $\gamma$ and $\beta$ of opposite sign.
\textbf{Not a bug.} $\langle C\rangle$ can be anywhere between $C_{\min}$ and $C_{\max}$; the optimizer (COBYLA) moves away from the poor
start, and the recorded run still ended at $\langle C\rangle = 20.53$ ($r = 0.82$). The value $9.18$ is computed from the notebook's
own conventions (\texttt{verify\_answers.py} reproduces it with NumPy and CUDA-Q)."""),

    dict(id="C15", kind="match",
         q=r"""Match each part of the QAOA algorithm (List-I) with the code that implements it in the notebook (List-II).""",
         match=([r"initial state $\ket+^{\otimes n}$", r"cost unitary for one edge",
                 r"mixer unitary for one qubit", r"objective $\langle H_C\rangle$"],
                [r"\texttt{rx(2.0 * beta, qubits[q])}", r'\texttt{cudaq.observe(kernels["state"], H\_C, ...).expectation()}',
                 r"\texttt{h(qubits)}", r"\texttt{x.ctrl(...); rz(gamma * weights[e], ...); x.ctrl(...)}"]),
         opts=[r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$4, Q$\to$3, R$\to$1, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: \texttt{h(qubits)} applies $H$ to every qubit of the freshly allocated $\ket0^{\otimes n}$, giving $\ket+^{\otimes n}$ (3).
\item Q: CNOT, $R_Z(\gamma w)$, CNOT $= e^{-i\gamma(w/2)Z_iZ_j}$ (4).
\item R: $R_X(2\beta) = e^{-i\beta X}$ (1).
\item S: \texttt{cudaq.observe} computes $\langle\psi|H_C|\psi\rangle$ for the state-preparation kernel (exactly on simulators, from shots
when \texttt{shots\_count} is given) (2).
\end{itemize}"""),

    dict(id="C16", kind="mcq",
         q=r"""For the triangle $K_3$ (unit weights, $C_{\max} = 2$), the best value of $F_1$ over all angles is""",
         opts=[r"$2$, i.e.\ $r = 1$", r"$1.5$, i.e.\ $r = 0.75$", r"$\sqrt3$, i.e.\ $r \approx 0.866$", r"$1.75$, i.e.\ $r = 0.875$"],
         sol=r"""$H_C$ has only two eigenvalues (A9): $+\tfrac32$ on $\{000, 111\}$ and $-\tfrac12$ on the six optimal strings. Hence
$e^{-i\gamma H_C}$ only puts a relative phase $e^{-2i\gamma}$ between the two non-optimal strings and the six optimal ones, and the
mixer can then interfere the two non-optimal amplitudes away completely. A numerical maximisation gives $F_1 = 2$ exactly, for
example at $\gamma^* = \arctan(1/\sqrt2) \approx 0.6155$, $\beta^* = -\gamma^*/2 \approx -0.3077$ (notebook convention; checked
in \texttt{verify\_answers.py}). So $p = 1$ already finds the optimum with probability 1. The value $1.5 = W/2$ is only the
random baseline."""),

    dict(id="C17", kind="mcq",
         q=r"""The previously recorded run reports: approximation ratio $r = \langle C\rangle/C_{\max} = 0.8213$, best sampled cut $25$
($= C_{\max}$), and $P(\text{optimal split}) = 0.06$ with $4000$ shots. Which interpretation is correct?""",
         opts=[r"$r$ measures the \emph{average} cut of the distribution; the best of $4000$ samples can still be optimal, and here $240$ shots were optimal",
               r"$82\,\%$ of the shots returned an optimal split",
               r"The best sampled cut can never exceed $\langle C\rangle = 20.53$, so the value $25$ indicates an error",
               r"The best sampled split has cut $0.8213 \times 25 \approx 20.5$"],
         sol=r"""$\langle C\rangle = \sum_z |\braket{z|\boldsymbol\gamma,\boldsymbol\beta}|^2\,C(z) = 20.53$ is an average over the whole output distribution;
$r = 20.53/25 = 0.82$ summarises that average. Individual samples scatter around it: some are better than average.
Among $4000$ shots, $P_{\text{opt}} = 0.06$ means about $0.06\times4000 = 240$ optimal shots (exactly $240$ were recorded), so the best
sample reaches $C_{\max}$. QAOA used as a \emph{sampler} is judged by the best sample (and $P_{\text{opt}}$), used as an \emph{estimator}
by $r$."""),

    dict(id="C18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} The notebook's final sampling run uses the angles of the best evaluation stored in
\texttt{history\_cut}, which need not be the optimizer's final point \texttt{opt.x}.\\[1mm]
\textbf{Statement II:} With COBYLA and a budget of 80 evaluations, the notebook is guaranteed to find the global maximum of $F_2$.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} the code computes \texttt{best\_index = int(np.argmax(history\_cut))} and samples with
\texttt{best\_params = history\_params[best\_index]}; this protects against an optimizer that ends on a worse point.
\textbf{II is false:} COBYLA is a local, derivative-free method; QAOA landscapes have many local optima, and the recorded run
stopped with ``\emph{the objective function has been evaluated MAXFUN times}'' --- i.e.\ because of the budget, not because a global
optimality condition was met. Answer: (C)."""),

    dict(id="C19", kind="para", passage="C",
         q=r"""The value of $\langle Z_0Z_1\rangle$ is""",
         opts=[r"$-\tfrac12$", r"$-1$", r"$0$", r"$+\tfrac12$"],
         sol=r"""Edge $(0,1)$: $d_0 = 1$ (node 0 has one neighbour), $d_1 = 2$. With $\gamma = \pi/2$: $\sin\gamma = 1$, $\cos\gamma = 0$, so
$\cos^{0}\gamma = 1$ and $\cos^{1}\gamma = 0$. With $\beta = -\pi/8$: $\sin4\beta = \sin(-\pi/2) = -1$.
\[ \langle Z_0Z_1\rangle = \tfrac12\cdot(-1)\cdot1\cdot(1 + 0) = -\tfrac12 . \]
(Note $0^0 = 1$: the term $\cos^{d_u-1}\gamma$ for a leaf node is $1$ for every $\gamma$.) By symmetry $\langle Z_1Z_2\rangle = -\tfrac12$ too."""),

    dict(id="C20", kind="para", passage="C",
         q=r"""The expected cut $\langle C\rangle$ and the probability of sampling an optimal split are""",
         opts=[r"$1.5$ and $0.625$", r"$1.5$ and $0.3125$", r"$2$ and $1$", r"$1.25$ and $0.625$"],
         sol=r"""$\langle C\rangle = \sum_{\text{edges}}\tfrac12(1 - \langle Z_uZ_v\rangle) = 2\cdot\tfrac12\big(1 + \tfrac12\big) = 1.5$, so $r = 1.5/2 = 0.75$.
The optimal splits of the path are $010$ and $101$ (both edges cut). A state-vector calculation gives
$P(010) = P(101) = 0.3125$ and $0.0625$ for each of the other six strings, so $P_{\text{opt}} = 0.625$.
Consistency check: $\langle C\rangle = 2\cdot0.625 + 1\cdot P(C = 1) = 1.25 + 1\cdot 4\cdot0.0625 = 1.5$ (the strings $001, 011, 100, 110$ cut one
edge; $000, 111$ cut none)."""),
]
