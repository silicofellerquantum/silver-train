"""Section A - Classical Machine Learning and Mathematics (linear algebra, probability and statistics, derivatives,
optimization, neural networks, loss functions, evaluation metrics and machine-learning fundamentals).

For 'mcq', 'match' and 'para' questions the CORRECT option is listed FIRST; the generator moves it to a balanced
position. 'ar' and 'stmt' questions use the fixed option order and give the correct index explicitly.
"""
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS

PASSAGES = {
    "A": r"""A tiny neural network has two inputs, one hidden layer with two ReLU neurons and one sigmoid output neuron:
\[ \mathbf h = \operatorname{ReLU}(W_1\mathbf x + \mathbf b_1),\qquad p = \sigma(\mathbf w_2\cdot\mathbf h + b_2),\qquad
   \sigma(z) = \frac{1}{1 + e^{-z}},\qquad \operatorname{ReLU}(a) = \max(0, a), \]
with $W_1 = \begin{pmatrix}1 & -1\\ 2 & 1\end{pmatrix}$ (row $j$ = weights into hidden neuron $j$), $\mathbf b_1 = (0, -2)$,
$\mathbf w_2 = (0.5, -1)$ and $b_2 = -1$. It is trained with the binary cross-entropy
$L = -\big[y\ln p + (1-y)\ln(1-p)\big]$. Consider the single training example $\mathbf x = (1, -1)$ with label $y = 1$.""",
}

QUESTIONS = [
    dict(id="A1", kind="mcq",
         q=r"""The loss $L(w_1, w_2) = (w_1 - 3)^2 + 2(w_2 + 1)^2$ is minimised by gradient descent with learning rate
$\eta = 0.1$, starting from $\mathbf w = (0, 0)$. After \emph{one} step the parameters and the loss are""",
         opts=[r"$(0.6,\,-0.4)$ and $L = 6.48$", r"$(-0.6,\,0.4)$ and $L = 16.88$",
               r"$(0.6,\,-0.8)$ and $L = 5.84$", r"$(0.6,\,-0.4)$ and $L = 11$"],
         sol=r"""The gradient is $\nabla L = \big(2(w_1 - 3),\ 4(w_2 + 1)\big)$; at $(0, 0)$ it equals $(-6,\ 4)$.
Gradient descent moves \emph{against} the gradient: $\mathbf w \leftarrow \mathbf w - \eta\nabla L = (0, 0) - 0.1\,(-6, 4) = (0.6,\ -0.4)$.
New loss: $L = (0.6 - 3)^2 + 2(-0.4 + 1)^2 = 5.76 + 2(0.36) = 5.76 + 0.72 = 6.48$, down from $L(0,0) = 9 + 2 = 11$."""),

    dict(id="A2", kind="mcq",
         q=r"""A binary classifier outputs the probabilities $p = 0.9,\ 0.6,\ 0.2$ (probability of class 1) for three
examples whose true labels are $y = 1,\ 1,\ 0$. The \emph{mean} binary cross-entropy (natural logarithm) is closest to""",
         opts=[r"$0.280$", r"$0.122$", r"$0.839$", r"$0.742$"],
         sol=r"""Per example $\ell = -[y\ln p + (1-y)\ln(1-p)]$:
$\ell_1 = -\ln 0.9 = 0.1054$, $\ell_2 = -\ln 0.6 = 0.5108$, $\ell_3 = -\ln(1 - 0.2) = -\ln 0.8 = 0.2231$ (label 0, so the
probability of the \emph{correct} class is $1 - p = 0.8$). Mean: $(0.1054 + 0.5108 + 0.2231)/3 = 0.8393/3 = 0.2798 \approx 0.280$."""),

    dict(id="A3", kind="mcq",
         q=r"""On a test set a classifier gives $\text{TP} = 40$, $\text{FP} = 10$, $\text{FN} = 20$ and $\text{TN} = 130$. Its F1-score is""",
         opts=[r"$0.727$", r"$0.733$", r"$0.850$", r"$0.800$"],
         sol=r"""Precision $P = \frac{TP}{TP + FP} = \frac{40}{50} = 0.8$; recall $R = \frac{TP}{TP + FN} = \frac{40}{60} = 0.667$.
F1 is the \emph{harmonic} mean: $F_1 = \frac{2PR}{P + R} = \frac{2(0.8)(0.667)}{1.467} = \frac{1.0667}{1.4667} = 0.727$
(equivalently $F_1 = \frac{2TP}{2TP + FP + FN} = \frac{80}{110}$). For comparison the accuracy is $\frac{170}{200} = 0.85$."""),

    dict(id="A4", kind="mcq",
         q=r"""Centred two-dimensional data have the covariance matrix $C = \begin{pmatrix}2 & 1\\ 1 & 2\end{pmatrix}$. The first
principal component explains which fraction of the total variance, and along which direction does it point?""",
         opts=[r"$75\,\%$, along $(1, 1)/\sqrt2$", r"$75\,\%$, along $(1, -1)/\sqrt2$",
               r"$67\,\%$, along $(1, 1)/\sqrt2$", r"$50\,\%$, along $(1, 0)$"],
         sol=r"""PCA diagonalises the covariance matrix. $\det(C - \lambda I) = (2 - \lambda)^2 - 1 = 0$ gives $\lambda = 3$ and $\lambda = 1$.
For $\lambda = 3$: $(C - 3I)\mathbf v = 0 \Rightarrow -v_1 + v_2 = 0$, so $\mathbf v_1 = (1, 1)/\sqrt2$; for $\lambda = 1$, $\mathbf v_2 = (1, -1)/\sqrt2$.
The total variance is $\operatorname{tr} C = 4 = 3 + 1$, so the first component explains $3/4 = 75\,\%$ along $(1, 1)/\sqrt2$."""),

    dict(id="A5", kind="mcq",
         q=r"""A screening classifier has recall (sensitivity) $90\,\%$ and false-positive rate $5\,\%$. Only $1\,\%$ of the
population is positive. A randomly chosen person is flagged positive. The probability that the person is really positive
(the precision of the classifier on this population) is closest to""",
         opts=[r"$15\,\%$", r"$90\,\%$", r"$18\,\%$", r"$95\,\%$"],
         sol=r"""Bayes' theorem: $P(D\mid +) = \frac{P(+\mid D)P(D)}{P(+)}$ with
$P(+) = P(+\mid D)P(D) + P(+\mid \bar D)P(\bar D) = 0.9(0.01) + 0.05(0.99) = 0.009 + 0.0495 = 0.0585$.
Hence $P(D\mid +) = 0.009/0.0585 = 0.154 \approx 15\,\%$. With rare positives most alarms are false alarms, which is
why accuracy and recall alone can be misleading on imbalanced data."""),

    dict(id="A6", kind="mcq",
         q=r"""A single neuron computes $f = \sigma(wx + b)$ with the sigmoid $\sigma$, and the loss is $L = (f - y)^2$.
For $x = 2$, $w = 0.5$, $b = -1$ and $y = 1$, the derivative $\partial L/\partial w$ equals""",
         opts=[r"$-0.5$", r"$+0.5$", r"$-0.25$", r"$-2$"],
         sol=r"""Forward pass: $z = wx + b = 0.5(2) - 1 = 0$, $f = \sigma(0) = 0.5$.
Chain rule: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial f}\,\frac{\partial f}{\partial z}\,\frac{\partial z}{\partial w}
= 2(f - y)\cdot\sigma(z)\big(1-\sigma(z)\big)\cdot x = 2(-0.5)(0.25)(2) = -0.5$.
The negative sign means that increasing $w$ lowers the loss (it pushes $f$ towards $y = 1$)."""),

    dict(id="A7", kind="mcq",
         q=r"""A three-class network outputs the logits $(z_1, z_2, z_3) = (2, 1, 0)$ and uses the softmax
$p_k = e^{z_k}/\sum_j e^{z_j}$. The probability of class 1, and the cross-entropy loss $-\ln p_{\text{true}}$ if the true
class is class 2, are approximately""",
         opts=[r"$0.665$ and $1.408$", r"$0.665$ and $0.408$", r"$0.667$ and $1.099$", r"$0.245$ and $1.408$"],
         sol=r"""$\sum_j e^{z_j} = e^2 + e + 1 = 7.389 + 2.718 + 1 = 11.107$. Then $p_1 = 7.389/11.107 = 0.665$ and
$p_2 = 2.718/11.107 = 0.245$. The loss uses the probability of the \emph{true} class (class 2):
$-\ln p_2 = \ln 11.107 - 1 = 2.408 - 1 = 1.408$."""),

    dict(id="A8", kind="mcq",
         q=r"""A fully connected network has 8 inputs, one hidden layer of 4 neurons and 1 output neuron; every neuron has a
bias. The number of trainable parameters is (compare: the 8-qubit QCNN of \texttt{QML.ipynb} has 12)""",
         opts=[r"$41$", r"$36$", r"$37$", r"$13$"],
         sol=r"""A dense layer from $n_{\text{in}}$ to $n_{\text{out}}$ neurons has $n_{\text{in}}n_{\text{out}}$ weights and $n_{\text{out}}$ biases.
Input $\to$ hidden: $8 \times 4 + 4 = 36$. Hidden $\to$ output: $4 \times 1 + 1 = 5$. Total $36 + 5 = 41$.
(The QCNN needs only 12 because each of its 3 layers \emph{shares} 4 parameters among all its blocks, see Section C.)"""),

    dict(id="A9", kind="mcq",
         q=r"""The Gaussian (RBF) kernel is $k(\mathbf u, \mathbf v) = \exp\!\big(-\|\mathbf u - \mathbf v\|^2/(2\sigma^2)\big)$. For
$\mathbf u = (1, 2, 2)$, $\mathbf v = (2, 0, 1)$ and $\sigma = 1$, $k(\mathbf u, \mathbf v)$ equals""",
         opts=[r"$e^{-3}$", r"$e^{-6}$", r"$e^{-\sqrt6/2}$", r"$e^{-3/2}$"],
         sol=r"""$\mathbf u - \mathbf v = (-1, 2, 1)$, so $\|\mathbf u - \mathbf v\|^2 = 1 + 4 + 1 = 6$ (the \emph{squared} norm).
With $2\sigma^2 = 2$: $k = e^{-6/2} = e^{-3} \approx 0.050$. Kernels measure similarity: $k = 1$ for identical points and
$k \to 0$ for distant ones. Quantum kernels (Section C) replace this formula by a state overlap."""),

    dict(id="A10", kind="mcq",
         q=r"""For $A = \begin{pmatrix}1 & 2\\ 0 & 1\end{pmatrix}$ and $B = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$ the Kronecker
(tensor) product $A \otimes B$ is""",
         opts=[r"$\begin{pmatrix}0&1&0&2\\1&0&2&0\\0&0&0&1\\0&0&1&0\end{pmatrix}$",
               r"$\begin{pmatrix}0&0&1&2\\0&0&0&1\\1&2&0&0\\0&1&0&0\end{pmatrix}$",
               r"$\begin{pmatrix}0&1&0&0\\1&0&0&0\\0&2&0&1\\2&0&1&0\end{pmatrix}$",
               r"$\begin{pmatrix}0&0&0&1\\0&0&1&0\\0&1&0&2\\1&0&2&0\end{pmatrix}$"],
         sol=r"""By definition $A \otimes B = \begin{pmatrix} a_{11}B & a_{12}B\\ a_{21}B & a_{22}B\end{pmatrix}
= \begin{pmatrix} 1\cdot B & 2\cdot B\\ 0\cdot B & 1\cdot B\end{pmatrix}$: the top-left block is $B$, the top-right block is $2B$,
the bottom-left block is zero and the bottom-right block is $B$. Writing the blocks out gives the first matrix.
(The second matrix is $B \otimes A$: the Kronecker product is not commutative. Check: $\operatorname{tr}(A\otimes B) = \operatorname{tr}A\,\operatorname{tr}B = 2\cdot 0 = 0$.)"""),

    dict(id="A11", kind="mcq",
         q=r"""A classifier is correct on $17$ of $20$ test images (accuracy $\hat a = 0.85$, the default test-set size and the
recorded test accuracy of \texttt{QML.ipynb}). Treating the images as independent trials, the standard error of the
accuracy estimate is about""",
         opts=[r"$8$ percentage points", r"$0.6$ percentage points", r"$36$ percentage points", r"$1.8$ percentage points"],
         sol=r"""The number of correct answers is binomial, so the estimated accuracy has standard error
$\mathrm{SE} = \sqrt{\hat a(1 - \hat a)/M} = \sqrt{0.85 \times 0.15/20} = \sqrt{0.006375} = 0.080$, i.e.\ about
$8$ percentage points. A rough $95\,\%$ interval is $85 \pm 16\,\%$: twenty test images are far too few to distinguish
models whose accuracies differ by a few per cent."""),

    dict(id="A12", kind="mcq",
         q=r"""A student standardises all 200 images of a dataset (mean and standard deviation of every pixel computed over all
200 images), then splits them into 150 training and 50 test images, picks the learning rate that gives the best
\emph{test} accuracy, and reports that test accuracy. Which statement is correct?""",
         opts=[r"The reported accuracy is optimistically biased: test data influenced both the preprocessing statistics and the choice of learning rate. Fit the scaler on the training data only, tune on a validation split and evaluate on the test set once.",
               r"The procedure is correct, because standardisation is a linear map that never uses the labels.",
               r"Only the standardisation is a problem; choosing the learning rate on the test set is legitimate because the weights are fitted on the training set only.",
               r"The bias disappears if the split is changed from 150/50 to 100/100."],
         sol=r"""The test set must play no role in any decision about the model. Here it was used twice:
(1) \emph{data leakage in preprocessing} --- the mean and standard deviation contain information about the test images;
(2) \emph{selection on the test set} --- choosing the hyper-parameter with the best test score and then reporting that
score is an optimistic (maximum-of-noisy-values) estimate. The correct pipeline is: split first; fit every preprocessing
step on the training split; choose hyper-parameters on a separate validation split (or by cross-validation); report the
test score once. The size of the split does not remove either bias."""),

    dict(id="A13", kind="mcq",
         q=r"""For linear regression with design matrix $X \in \mathbb R^{M\times d}$, targets $\mathbf y \in \mathbb R^M$ and
loss $L(\mathbf w) = \frac1M\|X\mathbf w - \mathbf y\|^2$, the gradient $\nabla_{\mathbf w}L$ is""",
         opts=[r"$\frac2M X^{\mathsf T}(X\mathbf w - \mathbf y)$", r"$\frac1M X^{\mathsf T}(X\mathbf w - \mathbf y)$",
               r"$\frac2M X(X\mathbf w - \mathbf y)$", r"$\frac2M X^{\mathsf T}X\mathbf w$"],
         sol=r"""Write $L = \frac1M(X\mathbf w - \mathbf y)^{\mathsf T}(X\mathbf w - \mathbf y)
= \frac1M\big(\mathbf w^{\mathsf T}X^{\mathsf T}X\mathbf w - 2\mathbf y^{\mathsf T}X\mathbf w + \mathbf y^{\mathsf T}\mathbf y\big)$.
Using $\nabla_{\mathbf w}(\mathbf w^{\mathsf T}A\mathbf w) = 2A\mathbf w$ for symmetric $A$ and $\nabla_{\mathbf w}(\mathbf c^{\mathsf T}\mathbf w) = \mathbf c$:
$\nabla L = \frac1M\big(2X^{\mathsf T}X\mathbf w - 2X^{\mathsf T}\mathbf y\big) = \frac2M X^{\mathsf T}(X\mathbf w - \mathbf y)$,
a vector of length $d$ as required. Setting it to zero gives the normal equations $X^{\mathsf T}X\mathbf w = X^{\mathsf T}\mathbf y$."""),

    dict(id="A14", kind="mcq",
         q=r"""Gradient descent with (heavy-ball) momentum uses $v_{t} = \beta v_{t-1} + g_t$ and $w_{t} = w_{t-1} - \eta v_t$,
with $v_0 = 0$. If the gradient is constant, $g_t = 1$, with $\beta = 0.9$ and $\eta = 0.1$, the total decrease
$w_0 - w_3$ after three steps is""",
         opts=[r"$0.561$", r"$0.300$", r"$0.0561$", r"$0.271$"],
         sol=r"""$v_1 = 0.9(0) + 1 = 1$, $v_2 = 0.9(1) + 1 = 1.9$, $v_3 = 0.9(1.9) + 1 = 2.71$.
Total decrease $= \eta(v_1 + v_2 + v_3) = 0.1(1 + 1.9 + 2.71) = 0.1(5.61) = 0.561$.
Without momentum it would be $3\eta = 0.3$: momentum accelerates motion along directions where the gradient keeps the same
sign (the velocity approaches $g/(1-\beta) = 10$)."""),

    dict(id="A15", kind="match",
         q=r"""Match each activation function (List-I) with its derivative (List-II).""",
         match=([r"sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$", r"$\tanh z$", r"$\operatorname{ReLU}(z) = \max(0, z)$",
                 r"softplus $\ln(1 + e^{z})$"],
                [r"$\sigma(z)$", r"$1 - \tanh^2 z$", r"$\sigma(z)\big(1 - \sigma(z)\big)$", r"$1$ for $z > 0$, $0$ for $z < 0$"]),
         opts=[r"P$\to$3, Q$\to$2, R$\to$4, S$\to$1", r"P$\to$1, Q$\to$2, R$\to$4, S$\to$3",
               r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1"],
         sol=r"""\begin{itemize}\itemsep0pt
\item Sigmoid: $\sigma' = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)\cdot\frac{e^{-z}}{1 + e^{-z}} = \sigma(1 - \sigma)$ (3).
\item $\tanh$: $\frac{d}{dz}\frac{\sinh z}{\cosh z} = \frac{\cosh^2 z - \sinh^2 z}{\cosh^2 z} = 1 - \tanh^2 z$ (2).
\item ReLU: slope 1 for $z > 0$, 0 for $z < 0$ (not differentiable at 0; frameworks use 0 or 1 there) (4).
\item Softplus: $\frac{d}{dz}\ln(1 + e^z) = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} = \sigma(z)$ (1).
\end{itemize}
Hence P$\to$3, Q$\to$2, R$\to$4, S$\to$1."""),

    dict(id="A16", kind="match",
         q=r"""Match each observation (List-I) with the correct diagnosis and remedy (List-II).""",
         match=([r"training accuracy $99\,\%$, test accuracy $70\,\%$",
                 r"training and test accuracy both $55\,\%$ on a balanced two-class task",
                 r"validation loss is lowest at epoch 40 and rises afterwards while training loss keeps falling",
                 r"a model that predicts ``healthy'' for everybody has $99\,\%$ accuracy when $1\,\%$ of patients are sick"],
                [r"underfitting (high bias): increase model capacity or improve the features",
                 r"accuracy is misleading: report precision, recall and F1",
                 r"overfitting (high variance): regularise, add data or simplify the model",
                 r"stop at the validation minimum (early stopping)"]),
         opts=[r"P$\to$3, Q$\to$1, R$\to$4, S$\to$2", r"P$\to$1, Q$\to$3, R$\to$4, S$\to$2",
               r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4", r"P$\to$4, Q$\to$1, R$\to$3, S$\to$2"],
         sol=r"""\begin{itemize}\itemsep0pt
\item P: a large gap between training and test performance is the signature of \emph{overfitting} (3).
\item Q: both errors high and close to chance level (50\,\%): the model is too simple --- \emph{underfitting} (1).
\item R: the classic early-stopping picture: keep the parameters at the validation minimum (4).
\item S: with $1\,\%$ positives the trivial classifier reaches $99\,\%$ accuracy but recall 0: use precision, recall, F1 (2).
\end{itemize}
Hence P$\to$3, Q$\to$1, R$\to$4, S$\to$2."""),

    dict(id="A17", kind="ar", correct=0,
         q=r"""\textbf{Assertion (A):} No linear classifier $\hat y = \operatorname{sign}(w_1x_1 + w_2x_2 + b)$ can classify all four
XOR points $(0,0)\mapsto -1$, $(1,1)\mapsto -1$, $(0,1)\mapsto +1$, $(1,0)\mapsto +1$ correctly.\\[1mm]
\textbf{Reason (R):} The two classes of the XOR data cannot be separated by a straight line.""",
         opts=AR_OPTIONS,
         sol=r"""\textbf{R is true:} suppose $g(\mathbf x) = w_1x_1 + w_2x_2 + b$ separated them. Then $g(0,1) > 0$ and $g(1,0) > 0$
give $w_1 + w_2 + 2b > 0$, while $g(0,0) < 0$ and $g(1,1) < 0$ give $w_1 + w_2 + 2b < 0$ --- a contradiction.
\textbf{A is true}, and it is exactly the statement that a linear decision boundary does not exist, so \textbf{R explains A}.
A hidden layer (or a feature map such as $x_1x_2$) makes XOR separable --- the same idea motivates quantum feature maps. Answer: (A)."""),

    dict(id="A18", kind="stmt", correct=2,
         q=r"""\textbf{Statement I:} For $L(\mathbf w) = \frac12\mathbf w^{\mathsf T}H\mathbf w$ with $H$ symmetric positive definite and
largest eigenvalue $\lambda_{\max}$, gradient descent with constant step $\eta$ converges to $\mathbf 0$ from every starting
point if and only if $0 < \eta < 2/\lambda_{\max}$.\\[1mm]
\textbf{Statement II:} Adding the penalty $\lambda\|\mathbf w\|^2$ ($\lambda > 0$) to the training loss of linear regression
can never increase the training mean squared error of the fitted model.""",
         opts=STATEMENT_OPTIONS,
         sol=r"""\textbf{I is true:} $\nabla L = H\mathbf w$, so $\mathbf w_{t+1} = (I - \eta H)\mathbf w_t$. In the eigenbasis of $H$ each
component is multiplied by $1 - \eta\lambda_i$ per step; it shrinks to 0 for every start iff $|1 - \eta\lambda_i| < 1$, i.e.
$0 < \eta < 2/\lambda_i$, for all $i$, i.e.\ $0 < \eta < 2/\lambda_{\max}$.
\textbf{II is false:} the unregularised least-squares solution \emph{minimises} the training MSE, so any other solution ---
in particular the ridge solution --- has a training MSE that is \emph{at least as large} (strictly larger unless
$X^{\mathsf T}\mathbf y = 0$). Regularisation trades a higher training error for lower variance and, ideally, a lower test error. Answer: (C)."""),

    dict(id="A19", kind="para", passage="A",
         q=r"""For the example $\mathbf x = (1, -1)$, $y = 1$, the network output $p$ and the loss $L$ are""",
         opts=[r"$p = 0.5$, $L = \ln 2 \approx 0.693$", r"$p \approx 0.731$, $L \approx 0.313$",
               r"$p = 0.5$, $L = 0.25$", r"$p \approx 0.269$, $L \approx 1.313$"],
         sol=r"""Pre-activations: $W_1\mathbf x + \mathbf b_1 = (1\cdot1 + (-1)(-1),\ 2\cdot1 + 1\cdot(-1)) + (0, -2) = (2, 1) + (0, -2) = (2, -1)$.
ReLU: $\mathbf h = (2, 0)$ (the second neuron is inactive). Output: $z = 0.5(2) + (-1)(0) - 1 = 0$, so $p = \sigma(0) = 0.5$.
Loss with $y = 1$: $L = -\ln p = \ln 2 \approx 0.693$."""),

    dict(id="A20", kind="para", passage="A",
         q=r"""Back-propagation for the same example gives $\partial L/\partial b_2$ and the gradient of $L$ with respect to
the \emph{second row} of $W_1$ (the weights into hidden neuron 2) equal to""",
         opts=[r"$-0.5$ and $(0, 0)$", r"$-0.5$ and $(0.5, -0.5)$", r"$+0.5$ and $(0, 0)$", r"$-0.125$ and $(0, 0)$"],
         sol=r"""For sigmoid + binary cross-entropy, $\frac{\partial L}{\partial z} = p - y = 0.5 - 1 = -0.5$ (the factor $\sigma'$
cancels). Since $z = \mathbf w_2\cdot\mathbf h + b_2$: $\frac{\partial L}{\partial b_2} = -0.5$ and $\frac{\partial L}{\partial\mathbf w_2} = -0.5\,\mathbf h = (-1, 0)$.
Hidden neuron 2: $\frac{\partial L}{\partial a_2} = \frac{\partial L}{\partial z}\,w_{2,2}\,\operatorname{ReLU}'(a_2) = (-0.5)(-1)\cdot 0 = 0$,
because $a_2 = -1 < 0$. Therefore $\frac{\partial L}{\partial W_{1,(2,\cdot)}} = 0\cdot\mathbf x = (0, 0)$: an inactive (``dead'')
ReLU passes no gradient. (Neuron 1: $\frac{\partial L}{\partial a_1} = (-0.5)(0.5)(1) = -0.25$, row gradient $(-0.25, 0.25)$.)"""),
]
