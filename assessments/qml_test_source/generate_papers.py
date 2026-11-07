"""Build the QML Question Paper and the separate Answer Book from the question banks.

    python generate_papers.py            # writes .tex files, compiles both PDFs with latexmk, copies them to assessments/
    python generate_papers.py --no-pdf   # only writes the .tex files and answer_key.csv

Edit the questions in bank_*.py (never the generated .tex). In the banks the correct option of every 'mcq', 'match'
and 'para' question is listed FIRST; this script moves it to a position chosen so that each section has five
answers of each letter (A/B/C/D) and no letter appears three times in a row. The layout is deterministic (fixed seed),
so re-running the script reproduces the same paper.

This generator follows the QFT generator (assessments/qft_test_source/generate_papers.py) and, like the QAOA one,
prints a "Relevant formulae" block for every question of the Answer Bank.
"""
import csv
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import bank_advanced
import bank_basics
import bank_math
import bank_qm
from bank_explain import EXPLAIN

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT_DIR = HERE.parent
PAPER_PDF = "QML_Question_Paper.pdf"
ANSWER_PDF = "QML_Answer_Book.pdf"
SEED = 20261106
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)
TITLE = r"Quantum Machine Learning"
SUBTITLE = r"Assessment of QML: encodings, kernels, variational models and the QCNN"

SECTIONS = [
    ("A", "Classical Machine Learning and Mathematics",
     "linear algebra (eigenvalues, Kronecker products, matrix gradients), probability and statistics (Bayes' theorem, "
     "standard errors), derivatives and the chain rule, gradient descent and momentum, neural networks and activations, "
     "loss functions, kernels, evaluation metrics, data splits and model diagnosis", bank_math),
    ("B", "Quantum Computing Foundations",
     "qubits and quantum states, the Born rule, the Bloch sphere, Pauli algebra, gates and unitarity, rotation gates, "
     "tensor products, entanglement, controlled gates, measurement statistics, circuits and parameterized circuits", bank_qm),
    ("C", "Quantum Machine Learning Fundamentals",
     "basis, angle and amplitude encoding, feature maps and quantum kernels, swap test, variational circuits, VQC, QNN and "
     "QCNN architectures, parameter-shift rule, data re-uploading, hybrid training and the QCNN of \\texttt{QML.ipynb}", bank_basics),
    ("D", "Advanced QML and Implementation",
     "barren plateaus, generalised shift rules and gradient cost, noise, kernel concentration, quantum SVM, qPCA, QGAN, "
     "quantum RL and clustering, CUDA-Q targets and GPU memory, model evaluation, benchmarking against classical "
     "baselines, limitations and interpretation of recorded results", bank_advanced),
]

KIND_NAMES = {"mcq": "Single-correct MCQ", "match": "Matching lists", "ar": "Assertion--Reason",
              "stmt": "Statement I / II", "para": "Paragraph based"}

TOPICS = {
    "A1": r"One gradient-descent step", "A2": r"Binary cross-entropy", "A3": r"Precision, recall, F1",
    "A4": r"PCA from the covariance matrix", "A5": r"Bayes' theorem; precision at low prevalence",
    "A6": r"Chain rule for one neuron", "A7": r"Softmax and cross-entropy", "A8": r"Parameter count of a dense network",
    "A9": r"Gaussian (RBF) kernel", "A10": r"Kronecker product", "A11": r"Standard error of an accuracy",
    "A12": r"Data leakage and test-set tuning", "A13": r"Gradient of the MSE (matrix form)", "A14": r"Momentum",
    "A15": r"Derivatives of activations", "A16": r"Over-/underfitting, early stopping, imbalance",
    "A17": r"XOR and linear separability", "A18": r"GD step-size limit; effect of $L_2$ penalty",
    "A19": r"Forward pass of a small network", "A20": r"Back-propagation; dead ReLU",
    "B1": r"Expectation values from amplitudes", "B2": r"Pauli algebra", "B3": r"Period of $f(\theta)$; global phase",
    "B4": r"Product vs.\ entangled states", "B5": r"Bell-state correlations", "B6": r"Local rotation on an entangled qubit",
    "B7": r"$\langle Z\rangle$ from shots; standard error", "B8": r"Tensor product; qubit ordering",
    "B9": r"Unitarity", "B10": r"Global vs.\ relative phase", "B11": r"Conjugation $R_Y^\dagger ZR_Y$",
    "B12": r"Controlled-$R_Z$ matrix", "B13": r"State-vector memory", "B14": r"Two-parameter single-qubit circuit",
    "B15": r"Gate identities", "B16": r"Bloch vectors", "B17": r"Same statistics, different states",
    "B18": r"CNOT and the product-state criterion", "B19": r"Correlations of $R_Y$ + CNOT state",
    "B20": r"Purity and maximal entanglement",
    "C1": r"Angle encoding of a pixel", "C2": r"Amplitude encoding", "C3": r"Qubit cost of encodings",
    "C4": r"Fidelity kernel of angle encoding", "C5": r"Size of the kernel computation", "C6": r"Swap test",
    "C7": r"Two-term parameter-shift rule", "C8": r"Shared parameters in the QCNN", "C9": r"QCNN depth and parameters",
    "C10": r"Pooling block (code)", "C11": r"QCNN gate count", "C12": r"MSE vs.\ accuracy",
    "C13": r"Circuit count and run time of the notebook", "C14": r"Data re-uploading frequencies",
    "C15": r"Encodings and feature maps", "C16": r"What each QML model trains",
    "C17": r"Gradient-free training; shift-rule validity", "C18": r"PSD kernel matrices; shot noise",
    "C19": r"VQC output (Heisenberg picture)", "C20": r"VQC gradient by parameter shift",
    "D1": r"Barren plateau: shots needed", "D2": r"Barren plateaus: what is known", "D3": r"Two- vs.\ four-term shift rule",
    "D4": r"Depolarising noise and accuracy", "D5": r"GPU memory and multi-GPU qubits", "D6": r"Cost of an exact QCNN gradient",
    "D7": r"Expectation values from counts", "D8": r"Interpreting MSE 0.67 at 90\,\% accuracy",
    "D9": r"15-qubit training time", "D10": r"Kernel concentration", "D11": r"Quantum PCA", "D12": r"GAN equilibrium",
    "D13": r"Q-learning update", "D14": r"Swap-test distances (clustering)", "D15": r"CUDA-Q targets",
    "D16": r"Limitations of QML", "D17": r"Quantum-advantage claims", "D18": r"Recorded hardware; target selection",
    "D19": r"Recorded run: precision, recall, F1", "D20": r"Recorded run vs.\ classical baselines",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item Qubits are numbered $0, 1, \dots$; a basis state $\ket{q_0q_1\dots}$ lists \textbf{qubit 0 first} (as in CUDA-Q's bitstrings), and
      two-qubit matrices use the basis order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$.
\item Gates: $H = \frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}$; $X, Y, Z$ are the Pauli matrices; $S = \operatorname{diag}(1, i)$;
      $R_P(\theta) = e^{-i\theta P/2}$ for $P \in \{X, Y, Z\}$, so $R_Y(\theta)\ket0 = \cos\frac\theta2\ket0 + \sin\frac\theta2\ket1$; CNOT (control $c$,
      target $t$) flips $t$ when $c$ is 1; a controlled-$U$ applies $U$ to the target when the control is $\ket1$.
\item Measurements are in the computational basis; $\langle M\rangle = \braket{\psi|M|\psi}$, so $\langle Z\rangle = P(0) - P(1)$; the Bloch vector is
      $(\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$. Two-term parameter-shift rule: $f'(\theta) = \frac12\big[f(\theta + \frac\pi2) - f(\theta - \frac\pi2)\big]$.
\item Machine learning: labels $y \in \{+1, -1\}$ with prediction $\operatorname{sign}f$ unless stated ($y \in \{0, 1\}$ for cross-entropy);
      MSE $\mathcal L = \frac1M\sum_m(f_m - y_m)^2$; $\sigma(z) = 1/(1 + e^{-z})$; precision $= \frac{TP}{TP + FP}$, recall $= \frac{TP}{TP + FN}$,
      $F_1 = \frac{2PR}{P + R}$; $\ln$ is the natural logarithm. Numerical answers are rounded.
\item \textbf{Notebook} (\texttt{QML.ipynb}, defaults): $2\times4$ images with one line of 2 lit pixels (horizontal $= +1$, vertical $= -1$), background
      noise in $[0, 0.25]$, 20 training / 20 test images, \texttt{SEED = 42}; one qubit per pixel; encoding $R_Y(\pi x_i)$. QCNN
      layer $\ell = 1, 2, \dots$ has 4 shared parameters $\theta_{\ell,1..4}$ = \texttt{theta[4($\ell$-1)]}\,\dots\,\texttt{theta[4($\ell$-1)+3]}:
      convolution block $= \mathrm{CNOT}_{a\to b}\big(R_Y(\theta_{\ell,1})_a\otimes R_Y(\theta_{\ell,2})_b\big)$ on pairs $(0,1), (2,3), \dots$ then $(1,2), (3,4), \dots$;
      pooling block (source $a$, sink $b$) $= \ket1\!\bra1_a\otimes R_Z(\theta_{\ell,3}) + \ket0\!\bra0_a\otimes R_X(\theta_{\ell,4})$, then $a$ is discarded.
      Readout $\langle Z_{\text{out}}\rangle$ with \texttt{cudaq.observe} (exact); MSE loss; COBYLA with 100 loss evaluations.
\item \textbf{Previously recorded run} = outputs stored in the notebook (CUDA-Q 0.16.0, \texttt{qpp-cpu}, no GPU): 12 parameters, 69 gates,
      depth 21, 1.7 ms per circuit, 3.4 s training, loss $1.1214 \to 0.6704$, accuracy $90\,\%$ (training) / $85\,\%$ (test).
\end{itemize}"""


XREF = re.compile(r"(?<![A-Za-z\\])([A-D])(\d{1,2})(?![0-9])")


# ------------------------------------------------------------------------------------------------ data preparation
def load_questions():
    """Return the 80 questions (in paper order) with shuffled options, the passages and an id -> number map."""
    rng = random.Random(SEED)
    questions, passages, number_of = [], {}, {}
    for s_idx, (sec, title, blurb, module) in enumerate(SECTIONS):
        qs = [dict(q) for q in module.QUESTIONS]
        assert len(qs) == 20, f"Section {sec} has {len(qs)} questions"
        passages.update(module.PASSAGES)
        letters = balanced_letters(qs, rng)
        for pos, (q, letter) in enumerate(zip(qs, letters)):
            q["section"] = sec
            q["number"] = 20 * s_idx + pos + 1
            number_of[q["id"]] = q["number"]
            NUMBER_OF[q["id"]] = q["number"]
            assert len(q["opts"]) == 4, q["id"]
            if q["kind"] in ("ar", "stmt"):
                q["answer"] = q["correct"]
            else:
                order = [1, 2, 3]
                rng.shuffle(order)
                order.insert(letter, 0)
                q["orig"] = order  # q["orig"][position] = index of that option in the bank
                q["opts"], q["answer"] = [q["opts"][i] for i in order], letter
            questions.append(q)
    for q in questions:  # cross references "C14" -> "Q.54"
        for key in ("q", "sol"):
            q[key] = XREF.sub(lambda m: f"Q.{number_of[m.group(0)]}" if m.group(0) in number_of else m.group(0), q[key])
    return questions, passages, number_of


def balanced_letters(qs, rng):
    """Correct-option index for each question of one section: 5 of each letter, no letter three times in a row."""
    fixed = Counter(q["correct"] for q in qs if q["kind"] in ("ar", "stmt"))
    need = [len(qs) // 4 - fixed[i] for i in range(4)]
    assert min(need) >= 0, need
    pool = [i for i in range(4) for _ in range(need[i])]
    for _ in range(100000):
        rng.shuffle(pool)
        it = iter(pool)
        seq = [q["correct"] if q["kind"] in ("ar", "stmt") else next(it) for q in qs]
        if all(not (seq[i] == seq[i + 1] == seq[i + 2]) for i in range(len(seq) - 2)):
            return seq
    raise RuntimeError("could not balance answer letters")


def explanation_tex(q):
    """Concept recap, full derivation, why every wrong option fails, and the key takeaway."""
    ex = EXPLAIN[q["id"]]
    if q["kind"] in ("ar", "stmt"):
        wrong_pos = [i for i in range(4) if i != q["answer"]]
        texts = [re.sub(r"^\([A-D]\)\s*", "", t) for t in ex["wrong"]]
    else:
        wrong_pos = [q["orig"].index(k) for k in (1, 2, 3)]
        texts = ex["wrong"]
    pairs = sorted(zip(wrong_pos, texts))
    items = "\n".join(rf"\item[\textbf{{({LETTERS[pos]})}}] {t}" for pos, t in pairs)
    xref = lambda t: XREF.sub(lambda m: f"Q.{NUMBER_OF[m.group(0)]}" if m.group(0) in NUMBER_OF else m.group(0), t)
    letter = LETTERS[q["answer"]]
    return (r"\textbf{\color{brand}Correct answer.} \textcolor{okgreen}{\textbf{(" + letter + r")}} " + q["opts"][q["answer"]] + "\n\n"
            + r"\textbf{\color{brand}Concepts required.} " + xref(ex["concept"]) + "\n\n"
            + r"\textbf{\color{brand}Relevant formulae.} " + xref(ex["formulae"]) + "\n\n"
            + r"\textbf{\color{brand}Step-by-step explanation.} " + q["sol"].strip() + "\n\n"
            + r"\textbf{\color{brand}Why the other options are incorrect.}" + "\n"
            + "\\begin{itemize}[leftmargin=2.4em, itemsep=1pt, topsep=2pt]\n" + xref(items) + "\n\\end{itemize}\n"
            + r"\textbf{\color{brand}Key takeaway.} " + xref(ex["takeaway"]))


NUMBER_OF = {}

# ------------------------------------------------------------------------------------------------ LaTeX helpers
def visual_len(s):
    """Rough printed width of an option (characters) used to pick a 4-, 2- or 1-column layout."""
    t = re.sub(r"\\begin\{[pb]?matrix\}.*?\\end\{[pb]?matrix\}", "M" * 9, s, flags=re.S)
    t = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", lambda m: "x" * max(len(m.group(1)), len(m.group(2))), t)
    t = re.sub(r"\\(ket|bra)\{([^{}]*)\}", r"|\2>", t)
    t = re.sub(r"\\(ket|bra)(\w)", r"|\2>", t)
    t = re.sub(r"\\(left|right|big|Big|,|;|!|quad|displaystyle|mathrm|operatorname|text|emph|textbf)", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "x", t)
    t = re.sub(r"[{}$^_\\]", "", t)
    return len(t)


def option_block(q, answer_book=False):
    opts = q["opts"]

    def label(i):
        if answer_book and i == q["answer"]:
            return r"\correctlabel{" + LETTERS[i] + "}"
        return r"\optlabel{" + LETTERS[i] + "}"

    longest = max(visual_len(o) for o in opts)
    if q["kind"] in ("ar", "stmt") or longest > 40:
        cols = 1
    elif q["kind"] == "match" or longest > 15:
        cols = 2
    else:
        cols = 4
    if cols == 1:
        items = "\n".join(rf"\item[{label(i)}] {o}" for i, o in enumerate(opts))
        return ("\\begin{itemize}[leftmargin=2.6em, labelsep=0.4em, itemsep=2pt, topsep=3pt]\n" + items +
                "\n\\end{itemize}\n")
    cells = [rf"{label(i)}\,{o}" for i, o in enumerate(opts)]
    rows = [cells[i:i + cols] for i in range(0, 4, cols)]
    body = " \\\\[3pt]\n".join(" & ".join(r) for r in rows)
    return ("\\par\\smallskip\\noindent\\begin{tabularx}{\\linewidth}{@{}*{%d}{L}@{}}\n%s\n\\end{tabularx}\n"
            % (cols, body))


def match_table(q):
    list1, list2 = q["match"]
    rows = "\n".join(rf"\textbf{{({p})}} & {a} & \textbf{{({k})}} & {b} \\"
                     for p, a, k, b in zip("PQRS", list1, "1234", list2))
    return (r"""\par\smallskip\noindent{\renewcommand{\arraystretch}{1.35}%
\begin{tabularx}{\linewidth}{@{}c>{\raggedright\arraybackslash}p{0.33\linewidth}@{\hspace{1.2em}}cL@{}}
\toprule
\multicolumn{2}{@{}l}{\textbf{List-I}} & \multicolumn{2}{l@{}}{\textbf{List-II}} \\ \midrule
""" + rows + "\n\\bottomrule\n\\end{tabularx}}\n")


def passage_title(key, questions):
    nums = [q["number"] for q in questions if q.get("passage") == key]
    joined = " and ".join(str(n) for n in nums) if len(nums) <= 2 else f"{nums[0]}--{nums[-1]}"
    return f"Paragraph for Questions {joined}"


def question_tex(q, questions, passages, answer_book=False, seen_passages=None):
    out = []
    key = q.get("passage")
    if key and key not in seen_passages:
        seen_passages.add(key)
        out.append(r"\needspace{20\baselineskip}" + "\n")
        out.append(r"\begin{passagebox}{" + passage_title(key, questions) + "}\n" + passages[key] + "\n\\end{passagebox}\n")
    out.append((r"\needspace{13\baselineskip}" if answer_book else r"\needspace{7\baselineskip}") + "\n")
    out.append(r"\begin{qitem}{" + str(q["number"]) + "}\n")
    if answer_book:
        out.append(r"\qmeta{" + KIND_NAMES[q["kind"]] + "}{" + TOPICS[q["id"]] + "}\n")
    out.append(q["q"].strip() + "\n")
    if q["kind"] == "match":
        out.append(match_table(q))
    out.append(option_block(q, answer_book))
    out.append(r"\end{qitem}" + "\n")
    if answer_book:
        out.append(r"\begin{solbox}{" + LETTERS[q["answer"]] + "}\n" + explanation_tex(q) + "\n\\end{solbox}\n")
    out.append(r"\qsep" + "\n")
    return "".join(out)


PREAMBLE = r"""%% GENERATED by generate_papers.py -- edit the bank_*.py files, not this file.
\documentclass[11pt,a4paper]{article}
\usepackage[a4paper, top=20mm, bottom=20mm, left=18mm, right=18mm, headheight=14pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath, amssymb, mathtools}
\usepackage{braket}
\usepackage{array, tabularx, booktabs, multicol}
\usepackage{enumitem}
\usepackage[dvipsnames]{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{needspace}
\usepackage{fancyhdr, lastpage}
\usepackage{pifont}
\usepackage{tikz}
\usepackage{graphicx}
\usepackage{eso-pic}
% SILICO FELLER watermark: official logo, centred, ~8 % opacity, behind all content, on every page
\AddToShipoutPictureBG{\AtPageCenter{\makebox(0,0){\tikz\node[opacity=0.08, inner sep=0pt]{\includegraphics[width=0.72\paperwidth]{../../../assets/branding/silico_feller_logo.png}};}}}
\usepackage[hidelinks]{hyperref}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\definecolor{brand}{HTML}{1F4E79}
\definecolor{okgreen}{HTML}{1B7F3B}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\newcommand{\optlabel}[1]{\textbf{(#1)}}
\newcommand{\correctlabel}[1]{\textcolor{okgreen}{\textbf{(#1)}\,\ding{51}}}
\newenvironment{qitem}[1]{\begin{list}{}{\setlength{\leftmargin}{3.3em}\setlength{\labelwidth}{3.1em}%
  \setlength{\labelsep}{0.2em}\setlength{\topsep}{3pt}\setlength{\parsep}{3pt}}\item[\textbf{Q.#1}\hfill]}{\end{list}}
\newcommand{\qsep}{\par\vspace{2pt}{\color{black!15}\hrule}\vspace{4pt}}
\newcommand{\qmeta}[2]{{\footnotesize\color{black!60}\textsc{#1} \;$\cdot$\; #2}\par}
\newtcolorbox{passagebox}[1]{enhanced, breakable, colback=brand!4, colframe=brand!70, fonttitle=\bfseries,
  title={#1}, left=6pt, right=6pt, top=4pt, bottom=4pt}
\newtcolorbox{sectionbanner}{enhanced, colback=brand, colframe=brand, coltext=white, fontupper=\large\bfseries,
  left=8pt, right=8pt, top=6pt, bottom=6pt, sharp corners}
\newtcolorbox{infobox}[1]{enhanced, breakable, colback=black!3, colframe=black!45, fonttitle=\bfseries,
  title={#1}, left=6pt, right=6pt, top=4pt, bottom=4pt}
\newtcolorbox{solbox}[1]{enhanced, breakable, colback=okgreen!3, colframe=okgreen!60!black, fonttitle=\bfseries,
  title={Answer: (#1) \quad Detailed solution}, left=6pt, right=6pt, top=3pt, bottom=4pt, before skip=4pt, after skip=6pt}
\allowdisplaybreaks
"""


def header_footer(kind):
    return (r"""\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\color{black!70}""" + TITLE + r""" --- Assessment}
\fancyhead[R]{\small\color{black!70}\textbf{""" + kind + r"""}}
\fancyfoot[C]{\small Page \thepage\ of \pageref*{LastPage}}
\renewcommand{\headrulewidth}{0.4pt}
""")


def section_counts(questions, sec):
    qs = [q for q in questions if q["section"] == sec]
    parts = []
    for kind in ("mcq", "match", "ar", "stmt", "para"):
        nums = [q["number"] for q in qs if q["kind"] == kind]
        if not nums:
            continue
        rng = f"Q.{nums[0]}" if len(nums) == 1 else f"Q.{nums[0]}--{nums[-1]}"
        parts.append(f"{KIND_NAMES[kind]} ({rng})")
    return "; ".join(parts)


def marking_box():
    return r"""\begin{infobox}{Marking scheme (every question)}
Each question has \textbf{four} options (A), (B), (C) and (D). \textbf{ONLY ONE} of these four options is correct.\\[2pt]
\begin{tabular}{@{}ll@{}}
\emph{Full marks}: & $+%d$ if ONLY the correct option is chosen;\\
\emph{Zero marks}: & $\phantom{+}%d$ if none of the options is chosen (question unanswered);\\
\emph{Negative marks}: & $%d$ in all other cases (wrong option, or more than one option chosen).
\end{tabular}
\end{infobox}
""" % (MARKS["full"], MARKS["zero"], MARKS["negative"])


# ------------------------------------------------------------------------------------------------ Question Paper
def build_paper(questions, passages):
    t = [PREAMBLE, r"\begin{document}", header_footer("Question Paper")]
    t.append(r"""\thispagestyle{empty}
\begin{center}
{\color{brand}\rule{\linewidth}{1.2pt}}\\[6pt]
{\Huge\bfseries\color{brand} """ + TITLE + r"""}\\[4pt]
{\Large """ + SUBTITLE + r"""}\\[4pt]
{\large\bfseries QUESTION PAPER}\\[2pt]
{\color{brand}\rule{\linewidth}{1.2pt}}\\[1pt]
\end{center}
\vspace{-2pt}
\begin{tabularx}{\linewidth}{@{}lXlr@{}}
\textbf{Time allowed:} & 3 hours & \textbf{Maximum marks:} & 240\\
\textbf{Questions:} & 80 (4 sections $\times$ 20) & \textbf{Answer key:} & separate Answer Book
\end{tabularx}
\vspace{6pt}

\textbf{Name of the candidate:} \hrulefill\quad \textbf{Roll No.:} \rule{35mm}{0.4pt}

\vspace{6pt}
\begin{infobox}{General instructions}
\begin{enumerate}[leftmargin=1.4em, itemsep=1pt, topsep=1pt]
\item This question paper has \textbf{four sections} of \textbf{20 questions} each:
\textbf{A}~Classical Machine Learning and Mathematics, \textbf{B}~Quantum Computing Foundations, \textbf{C}~Quantum Machine Learning Fundamentals, \textbf{D}~Advanced QML and Implementation.
\item Every question is of the \textbf{single-correct} type: four options, exactly one correct. Question formats used:
single-correct MCQ, matching lists (List-I/List-II), Assertion--Reason, Statement~I/II and paragraph-based questions.
\item For every question choose \textbf{ONE} option, (A), (B), (C) or (D), and record it against the question number in the answer script provided by the invigilator.
\item Calculators are \textbf{not} allowed. Rough work may be done on separate sheets.
\item Unless a question says otherwise, use the \textbf{conventions} printed below; they apply to every section.
\item The detailed solutions are in the separate \emph{Answer Book}; do not consult it before finishing the test.
\end{enumerate}
\end{infobox}
""" + marking_box() + r"""
\begin{infobox}{Conventions and notation used throughout the paper}
\footnotesize""" + CONVENTIONS + r"""
\end{infobox}
\newpage
""")
    seen = set()
    for sec, title, blurb, _ in SECTIONS:
        qs = [q for q in questions if q["section"] == sec]
        t.append(r"\begin{sectionbanner}SECTION " + sec + r" \enspace$\cdot$\enspace " + title +
                 r" \hfill Q." + f"{qs[0]['number']}--{qs[-1]['number']}" + r"\end{sectionbanner}" + "\n")
        t.append(r"{\small \textbf{Syllabus:} " + blurb + r".\\ \textbf{Formats:} " + section_counts(questions, sec) +
                 r". Each question: $+3$ / $0$ / $-1$.}" + "\n\\par\\medskip\n")
        for q in qs:
            t.append(question_tex(q, questions, passages, False, seen))
        t.append(r"\newpage" + "\n")
    t.append(r"\end{document}")
    return "\n".join(t)



# ------------------------------------------------------------------------------------------------ Answer Book
THEMES = [
    ("Linear algebra: vectors, matrices, eigenvalues, tensor products", ["A4", "A10", "A13", "B2", "B8", "B9", "B11"]),
    ("Probability, statistics and sampling", ["A5", "A11", "B7", "D1", "D7", "D10", "D20"]),
    ("Derivatives, gradient descent and optimizers", ["A1", "A6", "A13", "A14", "A18", "A20", "C17", "D6"]),
    ("Neural networks, activations and loss functions", ["A2", "A7", "A8", "A15", "A17", "A19", "A20", "C12"]),
    ("Model evaluation, data handling and diagnosis", ["A3", "A11", "A12", "A16", "D8", "D19", "D20"]),
    ("Qubits, states, Bloch sphere and measurement", ["B1", "B3", "B7", "B10", "B13", "B16", "B17"]),
    ("Gates, circuits and parameterized circuits", ["B9", "B11", "B12", "B14", "B15", "B19", "C10"]),
    ("Entanglement and correlations", ["B4", "B5", "B6", "B18", "B19", "B20"]),
    ("Data encoding and feature maps", ["C1", "C2", "C3", "C14", "C15", "B16"]),
    ("Kernels, quantum kernels and kernel methods", ["A9", "C4", "C5", "C6", "C18", "D10", "D14"]),
    ("Variational models, QNN, VQC and hybrid training", ["C7", "C12", "C13", "C16", "C19", "C20", "B14"]),
    ("Parameter-shift rules and gradient cost", ["C7", "C8", "C17", "C20", "D3", "D6"]),
    (r"The QCNN of \texttt{QML.ipynb}: layout, code, counts", ["C8", "C9", "C10", "C11", "C13", "D6", "D8"]),
    ("Barren plateaus, noise and trainability", ["B6", "D1", "D2", "D4"]),
    ("Other QML algorithms: QSVM, qPCA, QGAN, QRL, clustering", ["C16", "D10", "D11", "D12", "D13", "D14"]),
    ("CUDA-Q, GPU simulation and performance", ["B13", "D5", "D7", "D9", "D15", "D18"]),
    ("Limitations, benchmarking and quantum-advantage claims", ["D16", "D17", "D19", "D20", "C4"]),
]


def topic_map(questions):
    """Theme -> question numbers (a question may serve several themes)."""
    num = {q["id"]: q["number"] for q in questions}
    covered = {i for _, ids in THEMES for i in ids}
    assert covered == set(num), sorted(set(num) - covered)
    rows = "\n".join(rf"{name} & " + ", ".join(str(n) for n in sorted(num[i] for i in ids)) + r" \\"
                     for name, ids in THEMES)
    return (r"""\needspace{18\baselineskip}
\subsection*{Topic-wise map of the questions}\label{topicmap}
{\small Every question appears under at least one theme; several appear under two.\par}
{\small\renewcommand{\arraystretch}{1.2}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.42\linewidth}L@{}}
\toprule
\textbf{Theme} & \textbf{Question numbers}\\ \midrule
""" + rows + r"""
\bottomrule
\end{tabularx}}
""")



def build_answer_book(questions, passages):
    t = [PREAMBLE, r"\begin{document}", header_footer("Answer Book")]
    dist = Counter(LETTERS[q["answer"]] for q in questions)
    t.append(r"""\thispagestyle{empty}
\begin{center}
{\color{brand}\rule{\linewidth}{1.2pt}}\\[6pt]
{\Huge\bfseries\color{brand} """ + TITLE + r"""}\\[4pt]
{\Large """ + SUBTITLE + r"""}\\[4pt]
{\large\bfseries ANSWER BOOK}\\[2pt]
{\color{brand}\rule{\linewidth}{1.2pt}}
\end{center}
This book accompanies the separate \emph{Question Paper} (80 questions, 3 hours, 240 marks). It contains
\begin{enumerate}[leftmargin=1.4em, itemsep=0pt, topsep=2pt]
\item the \textbf{answer key}, section by section, with the topic tested by every question (page~\pageref{key}),
\item a \textbf{topic-wise map} grouping the questions by theme and a \textbf{quick-marking grid} (page~\pageref{topicmap}), and
\item the \textbf{Answer Bank}: every question restated with the correct option marked \textcolor{okgreen}{\ding{51}} and a
      complete solution with derivations (from page~\pageref{bank}).
\end{enumerate}
""" + marking_box() + r"""
\begin{infobox}{How to evaluate}
For each section count $N_{\mathrm c}$ = number of correct answers and $N_{\mathrm w}$ = number of questions answered wrongly (or with more than one option marked). Section marks $= 3N_{\mathrm c} - N_{\mathrm w}$; unanswered questions score $0$. The total is out of $240$.
Every numerical answer was re-computed independently (NumPy, exact rational arithmetic where possible, an independent NumPy state-vector simulator, and CUDA-Q for the circuits and for the QCNN kernel of \texttt{QML.ipynb}) by
\texttt{verify\_answers.py} in the source folder.
\end{infobox}
\begin{infobox}{Conventions (identical to the Question Paper)}
\footnotesize""" + CONVENTIONS + r"""
\end{infobox}
\newpage
\section*{Answer key}\label{key}
{\small Answer distribution: """ + ", ".join(f"({L}) {dist[L]}" for L in LETTERS) + r""".}
""")
    for sec, title, _, _ in SECTIONS:
        qs = [q for q in questions if q["section"] == sec]
        rows = []
        for a, b in zip(qs[:10], qs[10:]):
            rows.append(rf"{a['number']} & \textbf{{{LETTERS[a['answer']]}}} & {TOPICS[a['id']]} & "
                        rf"{b['number']} & \textbf{{{LETTERS[b['answer']]}}} & {TOPICS[b['id']]} \\")
        t.append(r"""\needspace{14\baselineskip}
\subsection*{Section """ + sec + r""" --- """ + title + r"""}
{\small\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}r c L @{\hspace{1em}} r c L@{}}
\toprule
\textbf{Q.} & \textbf{Ans.} & \textbf{Topic tested} & \textbf{Q.} & \textbf{Ans.} & \textbf{Topic tested}\\ \midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}}
""")
    # compact grid for quick marking
    grid_rows = []
    for r0 in range(0, 80, 10):
        grid_rows.append(" & ".join(rf"{q['number']}\,\textbf{{{LETTERS[q['answer']]}}}" for q in questions[r0:r0 + 10]) + r" \\")
    t.append(topic_map(questions))
    t.append(r"""\subsection*{Quick-marking grid}
\begin{center}\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|""" + "c|" * 10 + r"""}\hline
""" + "\n\\hline\n".join(grid_rows) + r"""
\hline\end{tabular}\end{center}
\newpage
\section*{Answer Bank --- questions with detailed solutions}\label{bank}
""")
    seen = set()
    for sec, title, blurb, _ in SECTIONS:
        qs = [q for q in questions if q["section"] == sec]
        t.append(r"\begin{sectionbanner}SECTION " + sec + r" \enspace$\cdot$\enspace " + title +
                 r" \hfill Q." + f"{qs[0]['number']}--{qs[-1]['number']}" + r"\end{sectionbanner}" + "\n")
        for q in qs:
            t.append(question_tex(q, questions, passages, True, seen))
        if sec != SECTIONS[-1][0]:
            t.append(r"\newpage" + "\n")
    t.append(r"\end{document}")
    return "\n".join(t)


# ------------------------------------------------------------------------------------------------ main
def compile_pdf(tex_path):
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={BUILD}", str(tex_path)]
    res = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True)
    if res.returncode != 0:
        log = (BUILD / (tex_path.stem + ".log")).read_text(errors="replace")
        err = [ln for ln in log.splitlines() if ln.startswith("!") or "Error" in ln]
        sys.exit(f"LaTeX failed for {tex_path.name}:\n" + "\n".join(err[:20]))
    return BUILD / (tex_path.stem + ".pdf")


def main():
    questions, passages, _ = load_questions()
    assert len(questions) == 80 and len({q["id"] for q in questions}) == 80
    assert all(q["id"] in TOPICS for q in questions)
    for q in questions:  # every question needs the full Answer Bank structure
        ex = EXPLAIN[q["id"]]
        assert set(ex) == {"concept", "formulae", "wrong", "takeaway"}, q["id"]
        assert len(ex["wrong"]) == 3 and all(len(w) > 20 for w in ex["wrong"]), q["id"]
        assert len(q["sol"]) > 150, q["id"]
    for sec, *_ in SECTIONS:
        cnt = Counter(q["answer"] for q in questions if q["section"] == sec)
        assert all(cnt[i] == 5 for i in range(4)), (sec, cnt)
    BUILD.mkdir(exist_ok=True)
    paper_tex = BUILD / "qml_question_paper.tex"
    book_tex = BUILD / "qml_answer_book.tex"
    paper_tex.write_text(build_paper(questions, passages))
    book_tex.write_text(build_answer_book(questions, passages))
    with open(HERE / "answer_key.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "section", "answer", "format", "bank_id"])
        for q in questions:
            w.writerow([q["number"], q["section"], LETTERS[q["answer"]], q["kind"], q["id"]])
    print("answers:", "".join(LETTERS[q["answer"]] for q in questions))
    if "--no-pdf" in sys.argv:
        return
    for tex, name in ((paper_tex, PAPER_PDF), (book_tex, ANSWER_PDF)):
        pdf = compile_pdf(tex)
        shutil.copy(pdf, OUT_DIR / name)
        print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main()
