"""Section D - Advanced QML and Implementation (QCNN, quantum SVM and kernel concentration, quantum PCA, QGAN,
quantum RL and clustering, gradient computation, barren plateaus, noise, CUDA-Q targets and GPU simulation, model
evaluation, benchmarking against classical baselines, limitations and interpretation of the notebook's results).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "D": r"""For the lecture, \texttt{QML.ipynb} was re-executed with its default settings (CUDA-Q 0.16.0, \texttt{qpp-cpu}, no GPU).
It reproduced the recorded outputs exactly: loss $1.1214 \to 0.6704$, training accuracy $90\,\%$, test accuracy $85\,\%$.
With ``horizontal'' as the positive class, the 20 test images give the confusion matrix
\begin{center}\small
\begin{tabular}{lcc}\toprule & predicted horizontal & predicted vertical\\ \midrule
true horizontal (10 images) & 9 & 1\\ true vertical (10 images) & 2 & 8\\ \bottomrule\end{tabular}
\end{center}
Further measurements on the \emph{same} 20 training images (illustrative experiments, not part of the notebook):
logistic regression on the 8 raw pixels reaches $45\,\%$ test accuracy; a hand-written rule (``is some horizontal pixel pair
brighter than every vertical pixel pair?'') reaches $100\,\%$; an SVM with the fidelity kernel of the notebook's angle encoding
reaches $100\,\%$. On $2000$ fresh images from the same generator the accuracies are: trained QCNN $80.2\,\%$, kernel SVM
$95.2\,\%$, hand-written rule $100\,\%$, logistic regression $46.5\,\%$.""",
}

QUESTIONS = [
    dict(id="D1", kind="mcq",
         q=r"""For a deep random circuit with a global cost function the gradient components satisfy $\mathbb E[\partial_kC] = 0$ and
$\operatorname{Var}[\partial_kC] \approx 2^{-n}$. Each cost value is estimated with $S$ shots (statistical error $\approx 1/\sqrt S$). To resolve a
typical gradient component at $n = 20$ qubits one needs roughly""",
         opts=[r"$S \sim 10^{6}$ shots", r"$S \sim 20$ shots", r"$S \sim 400$ shots", r"$S \sim 10^{12}$ shots"],
         sol=r"""A typical gradient has size $\sqrt{\operatorname{Var}} = 2^{-n/2}$ (by Chebyshev, $|\partial_kC| \ge \epsilon$ has probability $\le 2^{-n}/\epsilon^2$).
To see it above the shot noise we need $1/\sqrt S \lesssim 2^{-n/2}$, i.e.\ $S \gtrsim 2^n = 2^{20} \approx 1.05\times10^6$ ---
\emph{exponentially} many shots in the number of qubits: a barren plateau. Comparing the noise with the variance instead of the
standard deviation would give $2^{2n} \approx 10^{12}$; polynomial guesses ($n$, $n^2$) miss the exponential scaling."""),

    dict(id="D2", kind="mcq",
         q=r"""Which statement about barren plateaus is correct?""",
         opts=[r"For random circuits deep enough to form approximate 2-designs, gradient variances vanish exponentially in $n$; for shallow circuits, local cost functions have only polynomially vanishing gradients while global cost functions can still vanish exponentially.",
               r"Barren plateaus are caused only by hardware noise, so they never appear in noiseless simulation.",
               r"Switching to a gradient-free optimizer such as COBYLA removes a barren plateau, because no gradients are computed.",
               r"A barren plateau can be removed by multiplying the learning rate by $2^n$."],
         sol=r"""McClean et al.\ (2018) showed exponentially vanishing gradient variance for random parameterised circuits that approximate
2-designs; Cerezo et al.\ (2021) showed that for shallow (logarithmic-depth) circuits the \emph{locality} of the cost matters: local costs
give polynomially vanishing gradients, global ones (e.g.\ fidelity with a target state) exponentially vanishing ones.
Noise can \emph{cause} additional plateaus (Wang et al.\ 2021), but noiseless deep circuits already have them. A flat landscape is flat for
every optimizer: cost \emph{differences} are exponentially small too (Arrasmith et al.\ 2021). Rescaling the step does not help because
the gradient estimate is dominated by shot noise."""),

    dict(id="D3", kind="mcq",
         q=r"""In a Hadamard test with a controlled-$R_Z(\theta)$ the measured output is $f(\theta) = \cos(\theta/2)$ (a pure frequency
$\frac12$). At $\theta = \pi$, the two-term shift formula $\frac12[f(\theta + \frac\pi2) - f(\theta - \frac\pi2)]$ and the exact derivative $f'(\pi)$ are""",
         opts=[r"$-0.707$ and $-0.5$", r"$-0.5$ and $-0.5$", r"$-0.707$ and $-0.707$", r"$-1$ and $-0.5$"],
         sol=r"""Exact: $f'(\theta) = -\frac12\sin\frac\theta2$, so $f'(\pi) = -\frac12$.
Two-term formula: $\frac12[\cos\frac{3\pi}{4} - \cos\frac\pi4] = \frac12\big[-\frac{\sqrt2}{2} - \frac{\sqrt2}{2}\big] = -\frac{\sqrt2}{2} \approx -0.707$ --- wrong
by $41\,\%$. The two-term rule is exact only when the generator has two eigenvalues $\pm\frac12$ (frequency 1). A controlled rotation has
eigenvalues $0, \pm\frac12$ (frequencies $\frac12$ and $1$); the four-term rule
$f' = d_+[f(\theta + \frac\pi2) - f(\theta - \frac\pi2)] - d_-[f(\theta + \frac{3\pi}2) - f(\theta - \frac{3\pi}2)]$, $d_\pm = \frac{\sqrt2 \pm 1}{4\sqrt2}$,
gives exactly $-0.5$."""),

    dict(id="D4", kind="mcq",
         q=r"""In a simple global-depolarising model every gate multiplies the output expectation value by $(1 - p)$. For the default
QCNN (69 gates) and $p = 0.01$, the expectation values are multiplied by about ..., and the test \emph{accuracy} (prediction
$\operatorname{sign}\langle Z\rangle$) is expected to""",
         opts=[r"$0.50$; stay the same, while $|\langle Z\rangle|$ shrinks and the MSE grows",
               r"$0.31$; drop to $50\,\%$",
               r"$0.99$; stay the same, and noise is negligible",
               r"$0.50$; drop to about $50\,\%$, because half of the predictions flip sign"],
         sol=r"""$(1 - 0.01)^{69} = e^{69\ln0.99} = e^{-0.6935} \approx 0.50$. Multiplying every output by a positive factor does not change its
sign, so sign-based accuracy is unchanged in this model; but the outputs move towards 0, so the MSE $\frac1M\sum(f - y)^2$ grows
towards 1 and, with finite shots, small outputs become indistinguishable from 0. (The lecture's density-matrix simulation with
gate-level depolarising noise $p = 0.01$ found exactly this: accuracy stayed $85\,\%$ while the mean $|\langle Z\rangle|$ fell from $0.171$ to $0.112$.)"""),

    dict(id="D5", kind="mcq",
         q=r"""An 80 GB GPU simulates state vectors in double precision (16 bytes per amplitude). The largest number of qubits that
fits on one GPU, and on four such GPUs pooled for one distributed state vector, are""",
         opts=[r"$32$ and $34$", r"$33$ and $35$", r"$32$ and $128$", r"$36$ and $38$"],
         sol=r"""Need $16\cdot2^n \le 80\times10^9$: $2^{32}\cdot16 = 68.7$ GB fits, $2^{33}\cdot16 = 137$ GB does not, so $n = 32$.
Four GPUs give 320 GB: $2^{34}\cdot16 = 275$ GB fits, $2^{35}\cdot16 = 550$ GB does not, so $n = 34$. Multiplying the memory by 4
adds only $\log_2 4 = 2$ qubits. (In single precision, 8 bytes, the answers would be 33 and 35.)"""),

    dict(id="D6", kind="mcq",
         q=r"""Suppose the 8-qubit QCNN (11 convolution blocks, 7 pooling blocks) were trained by gradient descent with \emph{exact}
parameter-shift gradients: every $R_Y$ occurrence shifted separately (2 circuits), every controlled rotation (\texttt{crz}, \texttt{crx})
with the four-term rule (4 circuits). The number of circuits for \emph{one} full gradient of the loss over the 20 training images is""",
         opts=[r"$2000$", r"$480$", r"$1440$", r"$40$"],
         sol=r"""Per image: each convolution block has 2 $R_Y$ gates $\Rightarrow 2\times2 = 4$ circuits per block, $11\times4 = 44$; each pooling block has one
\texttt{crz} and one \texttt{crx} $\Rightarrow 2\times4 = 8$ circuits per block, $7\times8 = 56$. Total $44 + 56 = 100$ circuits per image, and
$100\times20 = 2000$ for the training set (plus the 20 forward passes for the loss). One exact gradient step costs as much as the
\emph{whole} 100-evaluation COBYLA run of the notebook. (480 = naive $2\times12\times20$; 1440 = two-term rule for every occurrence.)"""),

    dict(id="D7", kind="mcq",
         q=r"""\texttt{cudaq.sample} on a two-qubit kernel returns the counts \texttt{\{'00': 400, '01': 100, '10': 300, '11': 200\}}
(the first character is qubit 0). The estimates of $\big(\langle Z_0\rangle, \langle Z_1\rangle, \langle Z_0Z_1\rangle\big)$ are""",
         opts=[r"$(0,\ 0.4,\ 0.2)$", r"$(0.4,\ 0,\ 0.2)$", r"$(0,\ 0.4,\ 0)$", r"$(0,\ 0.4,\ 0.6)$"],
         sol=r"""Total 1000 shots. Qubit 0 is $0$ in \texttt{00}, \texttt{01}: $P = 0.5$, so $\langle Z_0\rangle = 0.5 - 0.5 = 0$.
Qubit 1 is $0$ in \texttt{00}, \texttt{10}: $P = 0.7$, so $\langle Z_1\rangle = 0.7 - 0.3 = 0.4$.
$Z_0Z_1 = +1$ for even parity (\texttt{00}, \texttt{11}: 600) and $-1$ for odd parity (400): $\langle Z_0Z_1\rangle = 0.2$, which differs from
$\langle Z_0\rangle\langle Z_1\rangle = 0$, so the qubits are correlated. \texttt{cudaq.observe} computes such expectation values directly
(exactly on a state-vector simulator, or from shots with \texttt{shots\_count})."""),

    dict(id="D8", kind="mcq",
         q=r"""The recorded run of \texttt{QML.ipynb} ends with training accuracy $90\,\%$ but training MSE $0.67$ (the mean
$|\langle Z_7\rangle|$ of the trained model on the training images is about $0.20$). The best explanation is""",
         opts=[r"the outputs have the correct sign but small magnitude, so each correct image still contributes about $(1 - 0.2)^2 = 0.64$ to the loss",
               r"COBYLA diverged in the last iterations, so the loss and the accuracy belong to different parameter vectors",
               r"the MSE is computed on the test images, the accuracy on the training images",
               r"a bug: for $\pm1$ labels the MSE must equal $1 - \text{accuracy} = 0.10$"],
         sol=r"""MSE measures the distance of the real-valued output from $\pm1$; accuracy only checks the sign. If $\langle Z\rangle = \pm0.2$ with the
right sign, the squared error is $(1 - 0.2)^2 = 0.64$ even though the prediction is correct. A mean $|\langle Z\rangle|$ of $0.2$ is
consistent with an MSE of $0.67$ and $90\,\%$ accuracy. The notebook prints both quantities for the same final parameters on the
training set. Small margins matter: with few shots or noise such predictions are fragile."""),

    dict(id="D9", kind="mcq",
         q=r"""The README of the repository reports a 15-qubit test of \texttt{QML.ipynb} ($3\times5$ images, 16 training images, 60 COBYLA
evaluations) at about $0.35$ s per circuit on the 4-core test CPU. The training time is about""",
         opts=[r"$5.6$ minutes", r"$21$ seconds", r"$56$ minutes", r"$0.35$ seconds"],
         sol=r"""Circuits $= 60$ evaluations $\times$ 16 images $= 960$; time $\approx 960\times0.35\ \text{s} = 336\ \text{s} \approx 5.6$ minutes.
(Compare 8 qubits: $1.7$ ms per circuit. Going from 8 to 15 qubits multiplies the state size by $2^7 = 128$ and the gate count from
69 to 146, so a factor of about 200 is expected.) This is why the notebook recommends a GPU for 15 or more qubits."""),

    dict(id="D10", kind="mcq",
         q=r"""For the product kernel $k(\mathbf x, \mathbf x') = \prod_{i=1}^{n}\cos^2\frac{\pi(x_i - x'_i)}{2}$ with all $x_i, x'_i$ independent
and uniform on $[0, 1]$, the average off-diagonal kernel value $\mathbb E[k]$ equals""",
         opts=[r"$\big(\tfrac12 + \tfrac{2}{\pi^2}\big)^n \approx 0.70^n$", r"$2^{-n}$", r"$(2/\pi)^n$", r"$1/n$"],
         sol=r"""Independence factorises the expectation: $\mathbb E[k] = \big(\mathbb E\cos^2\frac{\pi\Delta}{2}\big)^n$, $\Delta = x - x'$.
$\cos^2\frac{\pi\Delta}{2} = \frac12 + \frac12\cos\pi\Delta$ and $\mathbb E\cos\pi(x - x') = |\mathbb E\,e^{i\pi x}|^2 = \big|\int_0^1e^{i\pi x}dx\big|^2 = \big|\frac{e^{i\pi} - 1}{i\pi}\big|^2 = \frac{4}{\pi^2}$.
So each factor is $\frac12 + \frac{2}{\pi^2} \approx 0.703$ and $\mathbb E[k] \approx 0.703^n$: kernel values \emph{concentrate} exponentially
towards 0 as the number of features grows ($0.703^{8} \approx 0.06$, $0.703^{50} \approx 2\times10^{-8}$), so exponentially many shots would
be needed to estimate them --- the exponential-concentration problem of quantum kernels."""),

    dict(id="D11", kind="mcq",
         q=r"""Which statement about quantum principal component analysis (qPCA, Lloyd, Mohseni \& Rebentrost 2014) is correct?""",
         opts=[r"It uses many copies of a density matrix $\rho$ to apply $e^{-i\rho t}$ (density-matrix exponentiation) and phase estimation to reveal its large eigenvalues; the output is a quantum state, and any speed-up assumes efficient state preparation and low rank --- dequantized classical algorithms match it under comparable assumptions.",
               r"It diagonalises the covariance matrix of $N$ classical data points in $O(\log N)$ total time, including loading the data and reading out all eigenvectors.",
               r"It is the pooling layer of the QCNN in \texttt{QML.ipynb}.",
               r"It needs only a single copy of $\rho$, because one measurement reveals all eigenvalues."],
         sol=r"""qPCA prepares many copies of $\rho$ (the normalised covariance/density matrix), simulates $e^{-i\rho t}$ by repeated partial SWAPs with
fresh copies, and runs phase estimation to obtain eigenvalue estimates together with the eigenvectors \emph{as quantum states}.
Reading out classical eigenvectors costs time proportional to their dimension, and loading classical data needs QRAM-like assumptions;
Tang-style ``dequantized'' sampling algorithms reach similar polylogarithmic scaling for low-rank matrices with comparable access. One copy
cannot reveal a spectrum (no-cloning, measurement collapse). qPCA is \emph{not} implemented in this repository."""),

    dict(id="D12", kind="mcq",
         q=r"""A (quantum or classical) GAN optimises $V(D, G) = \mathbb E_{x\sim p_{\text{data}}}[\ln D(x)] + \mathbb E_{x\sim p_G}[\ln(1 - D(x))]$.
At the ideal equilibrium $p_G = p_{\text{data}}$ with the optimal discriminator, the value of $V$ is""",
         opts=[r"$-\ln 4 \approx -1.386$", r"$0$", r"$\ln 2$", r"$-\tfrac12$"],
         sol=r"""For fixed $G$ the optimal discriminator is $D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$. When $p_G = p_{\text{data}}$, $D^* = \frac12$
everywhere, so $V = \ln\frac12 + \ln\frac12 = -2\ln2 = -\ln4$. In a quantum GAN the generator is a parameterised circuit whose
measurement distribution plays the role of $p_G$; training alternates discriminator and generator updates (not implemented in this repository)."""),

    dict(id="D13", kind="mcq",
         q=r"""In (quantum) reinforcement learning with Q-learning, $Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\big]$.
With $Q(s, a) = 2$, $\alpha = 0.5$, reward $r = 1$, discount $\gamma = 0.9$ and $\max_{a'}Q(s', a') = 4$, the updated value is""",
         opts=[r"$3.3$", r"$4.6$", r"$2.6$", r"$3.5$"],
         sol=r"""Target $= r + \gamma\max Q' = 1 + 0.9\times4 = 4.6$; temporal-difference error $= 4.6 - 2 = 2.6$;
update $= 2 + 0.5\times2.6 = 3.3$. In quantum RL a variational circuit can represent the Q-function or the policy
$\pi(a\mid s) = |\braket{a|U(s; \boldsymbol\theta)|0}|^2$; the update rule itself is classical."""),

    dict(id="D14", kind="mcq",
         q=r"""A quantum clustering algorithm estimates distances between normalised real data vectors $\mathbf a, \mathbf b$ with a swap test.
The ancilla gives $0$ with probability $0.9$, and $\mathbf a\cdot\mathbf b > 0$ is known. The Euclidean distance $\|\mathbf a - \mathbf b\|$ is about""",
         opts=[r"$0.46$", r"$0.63$", r"$0.89$", r"$0.21$"],
         sol=r"""Swap test: $|\mathbf a\cdot\mathbf b|^2 = 2P(0) - 1 = 0.8$, so $\mathbf a\cdot\mathbf b = \sqrt{0.8} = 0.894$.
For unit vectors $\|\mathbf a - \mathbf b\|^2 = 2 - 2\,\mathbf a\cdot\mathbf b = 2 - 1.789 = 0.211$, hence $\|\mathbf a - \mathbf b\| = 0.46$.
(Using $0.8$ as the overlap gives $\sqrt{0.4} = 0.63$.) The swap test only gives $|\mathbf a\cdot\mathbf b|$; the sign must be known
or obtained otherwise (e.g.\ a Hadamard test)."""),

    dict(id="D15", kind="match",
         q=r"""Match each CUDA-Q target (List-I) with its use (List-II).""",
         match=([r"\texttt{qpp-cpu}", r"\texttt{nvidia} with option \texttt{mgpu}", r"\texttt{nvidia} with option \texttt{mqpu}",
                 r"\texttt{density-matrix-cpu}"],
                [r"one state vector distributed over several GPUs (one MPI rank per GPU) to simulate more qubits",
                 r"several independent virtual QPUs (one per GPU) that run \emph{different} circuits in parallel, e.g.\ many images or shift terms",
                 r"CPU state-vector simulator; produced the recorded outputs of \texttt{QML.ipynb}",
                 r"noisy simulation with a noise model; memory $16\cdot4^N$ bytes"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4", r"P$\to$3, Q$\to$2, R$\to$1, S$\to$4",
               r"P$\to$4, Q$\to$1, R$\to$2, S$\to$3", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item \texttt{qpp-cpu}: the default when no GPU is visible; the notebook's recorded run used it (\texttt{GPUs visible to CUDA-Q: 0}) (3).
\item \texttt{mgpu} pools the memory of several GPUs for \emph{one} large state vector; it needs \texttt{mpiexec} (1).
\item \texttt{mqpu} exposes each GPU as a separate QPU for asynchronous, parallel execution of many \emph{small} circuits --- the natural
choice for QML, where each training step evaluates one circuit per image (2).
\item \texttt{density-matrix-cpu} stores $\rho$ ($4^N$ entries) and applies noise channels (4).
\end{itemize}
Hence P$\to$3, Q$\to$1, R$\to$2, S$\to$4. \texttt{QML.ipynb} itself selects only \texttt{nvidia} or \texttt{qpp-cpu} (or a user-given name) and uses neither \texttt{mgpu} nor \texttt{mqpu}."""),

    dict(id="D16", kind="match",
         q=r"""Match each limitation of QML (List-I) with its description (List-II).""",
         match=([r"barren plateau", r"dequantization", r"data-loading bottleneck", r"exponential kernel concentration"],
                [r"classical sampling algorithms reproduce claimed exponential speed-ups (e.g.\ recommendation systems, PCA) under the same input assumptions",
                 r"the variance of cost gradients decays exponentially with the number of qubits",
                 r"off-diagonal kernel values approach a constant exponentially fast, so exponentially many shots are needed to distinguish them",
                 r"preparing a general amplitude-encoded state of $2^n$ numbers needs $O(2^n)$ gates, which can cancel a speed-up"]),
         opts=[r"P$\to$2, Q$\to$1, R$\to$4, S$\to$3", r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2",
               r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3", r"P$\to$1, Q$\to$2, R$\to$4, S$\to$3"],
         sol=r"""\begin{itemize}\itemsep0pt
\item Barren plateau (McClean et al.\ 2018): $\operatorname{Var}[\partial C] \sim 2^{-n}$ (2).
\item Dequantization (Tang 2019 and later): quantum-inspired classical algorithms with comparable sampling access (1).
\item Data loading: amplitude encoding is qubit-cheap but gate-expensive; QRAM is not available at scale (4).
\item Kernel concentration (Thanasilp et al.\ 2022): $k(\mathbf x, \mathbf x') \to$ const, the model cannot generalise with polynomial shots (3).
\end{itemize}
Hence P$\to$2, Q$\to$1, R$\to$4, S$\to$3."""),

    dict(id="D17", kind="ar", correct=3,
         q=r"""\textbf{Assertion (A):} Because the QCNN of \texttt{QML.ipynb} reaches $85\,\%$ test accuracy, it demonstrates a quantum advantage
over classical image classifiers.\\[1mm]
\textbf{Reason (R):} Before any advantage can be claimed, the quantum model must be compared with classical baselines trained and
evaluated on the same data, at comparable cost.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{A is false:} $85\,\%$ on 20 test images ($\pm8$ points) only shows that the model learned something; the notebook itself states
that a classical CNN or even simple methods solve the toy task easily, and the circuit was \emph{simulated} classically in $3.4$ s.
In the lecture's experiments a hand-written rule reached $100\,\%$ and a classically computable kernel SVM $95.2\,\%$ on fresh images
(QCNN $80.2\,\%$). \textbf{R is true}: it is the minimum standard of benchmarking. Answer: (D)."""),

    dict(id="D18", kind="stmt", correct=1,
         q=r"""\textbf{Statement I:} The outputs stored in \texttt{QML.ipynb} were produced on an NVIDIA GPU.\\[1mm]
\textbf{Statement II:} With \texttt{TARGET = "auto"} on a machine with two GPUs, \texttt{QML.ipynb} distributes the QCNN state vector over both GPUs.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is false:} the recorded output prints \texttt{GPUs visible to CUDA-Q: 0} and \texttt{CUDA-Q target: qpp-cpu}; the README states the
run was on a CPU-only container, and GPU paths were not executed. \textbf{II is false:} \texttt{"auto"} selects \texttt{"nvidia"} when a GPU is
visible, via \texttt{cudaq.set\_target(target\_name)} without options --- a single-GPU state vector. Distributing one state would need the
\texttt{mgpu} option and MPI; running images in parallel would need \texttt{mqpu}. Answer: (B)."""),

    dict(id="D19", kind="para", passage="D",
         q=r"""For the positive class ``horizontal'', the precision, recall and F1-score of the trained QCNN on the 20 test images are""",
         opts=[r"$0.818$, $0.900$, $0.857$", r"$0.900$, $0.818$, $0.857$", r"$0.818$, $0.900$, $0.859$", r"$0.850$, $0.850$, $0.850$"],
         sol=r"""From the table: $TP = 9$, $FN = 1$, $FP = 2$, $TN = 8$.
Precision $= \frac{TP}{TP + FP} = \frac{9}{11} = 0.818$; recall $= \frac{TP}{TP + FN} = \frac{9}{10} = 0.900$;
$F_1 = \frac{2TP}{2TP + FP + FN} = \frac{18}{21} = 0.857$ (the harmonic mean, slightly below the arithmetic mean $0.859$).
Accuracy $= \frac{9 + 8}{20} = 0.85$ agrees with the recorded test accuracy."""),

    dict(id="D20", kind="para", passage="D",
         q=r"""Which conclusion is supported by these measurements?""",
         opts=[r"On fresh data the QCNN ($80.2\,\%$) is below the classically computable kernel SVM ($95.2\,\%$) and the simple rule ($100\,\%$); its $85\,\%$ on 20 images lies within about one standard error ($\approx8$ points) of $80\,\%$; logistic regression fails because the classes are not linearly separable in pixel space.",
               r"The QCNN demonstrates a quantum advantage, because it beats logistic regression by about 35 points.",
               r"The kernel SVM result requires a quantum computer, because its kernel is defined as an overlap of quantum states.",
               r"The drop from $85\,\%$ to $80.2\,\%$ proves that the 2000 fresh images come from a different distribution than the training images."],
         sol=r"""The fresh test set (2000 images) has a standard error of only $\sqrt{0.8\cdot0.2/2000} \approx 0.9$ points, so it measures the QCNN's
generalisation much better than 20 images ($\approx8$ points); $85\,\%$ vs.\ $80.2\,\%$ is consistent with sampling noise, and both sets come
from the same generator. The angle-encoding kernel is $\prod_i\cos^2\frac{\pi(x_i - x'_i)}{2}$, computable classically in $O(d)$; the rule needs
no training at all. Beating one weak (linear) baseline is not evidence of advantage: a line can be horizontal or vertical at any position,
and no single linear function of the pixels separates the two classes well."""),
]
