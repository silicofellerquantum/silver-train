"""Build the HHL Question Paper and the separate Answer Book from the question banks.

    python generate_papers.py            # writes .tex files, compiles both PDFs with latexmk, copies them to assessments/
    python generate_papers.py --no-pdf   # only writes the .tex files and answer_key.csv

Edit the questions in bank_*.py (never the generated .tex). In the banks the correct option of every 'mcq', 'match'
and 'para' question is listed FIRST; this script moves it to a position chosen so that each section has five
answers of each letter (A/B/C/D) and no letter appears three times in a row. The layout is deterministic (fixed seed),
so re-running the script reproduces the same paper.

This generator follows the QFT generator (assessments/qft_test_source/generate_papers.py) and, like the QAOA
generator, prints a "Relevant formulae" block for every question of the Answer Bank.
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
PAPER_PDF = "HHL_Question_Paper.pdf"
ANSWER_PDF = "HHL_Answer_Book.pdf"
SEED = 20261108
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)

TITLE = r"The HHL Algorithm"
SUBTITLE = r"Assessment of the Harrow--Hassidim--Lloyd algorithm for $A\vec x = \vec b$"

SECTIONS = [
    ("A", "Linear Algebra and Mathematical Foundations",
     "matrices and vectors, linear systems and their classical solution, eigenvalues and eigenvectors, Hermitian and "
     "unitary matrices, spectral decomposition, sparse matrices, normalisation, matrix norms and the condition number",
     bank_math),
    ("B", "Quantum Computing Foundations",
     "qubits and quantum states, superposition and the Born rule, entanglement, unitary and controlled gates, rotation "
     "gates, measurement and post-selection, the quantum Fourier transform and quantum phase estimation", bank_qm),
    ("C", "HHL Algorithm Fundamentals",
     "problem encoding, preparation of $\\ket b$, Hamiltonian simulation $e^{iAt}$, phase estimation of the eigenvalues, "
     "eigenvalue inversion by a controlled rotation, the constant $C$, ancilla, uncomputation, measurement and the "
     "notebook's code", bank_basics),
    ("D", "Advanced HHL and Implementation",
     "complexity, sparsity and condition-number dependence, error analysis, post-selection and amplitude amplification, "
     "circuit resources, CUDA-Q implementation and memory, noise, benchmarking of recorded results, limitations, "
     "dequantisation and applications", bank_advanced),
]

KIND_NAMES = dict(mcq="Single correct", match="Matching lists", ar="Assertion--Reason", stmt="Statement I/II",
                  para="Paragraph based")

TOPICS = {
    "A1": r"Solving a $2\times2$ system", "A2": r"Eigenvalues and $\kappa$ of a symmetric matrix",
    "A3": r"Matrix properties (Hermitian, unitary, singular)", "A4": r"Condition number of a diagonal matrix",
    "A5": r"Eigenvalues of a complex Hermitian matrix", "A6": r"Inverse via spectral decomposition",
    "A7": r"Spectral theorem vs.\ real eigenvalues", "A8": r"Condition number of $U$ and of $A^{-1}$",
    "A9": r"Sparsity of the 1-D Laplacian", "A10": r"Spectrum and $\kappa$ of the Laplacian",
    "A11": r"Normalisation and amplitude encoding", "A12": r"Solving $A\vec x=\vec b$ in the eigenbasis",
    "A13": r"Norms, $\kappa$ and determinant", "A14": r"Hermitian embedding of a non-Hermitian matrix",
    "A15": r"When is $e^{iAt}$ unitary?", "A16": r"Error amplification by $\kappa$",
    "A17": r"$2\times2$ example: eigenvalues", "A18": r"$2\times2$ example: normalised solution",
    "A19": r"Embedding + padding: qubit count", "A20": r"$\kappa$ from singular values",
    "B1": r"Born rule", "B2": r"State-vector size and memory", "B3": r"Gate identity $HZH = X$",
    "B4": r"$R_Y$ rotations of $\ket0$", "B5": r"Entanglement test", "B6": r"Phase kickback",
    "B7": r"Inverse QFT of a phase state", "B8": r"QPE with an exact 3-bit phase",
    "B9": r"QPE with a non-representable phase", "B10": r"Unitarity vs.\ measurement",
    "B11": r"Global vs.\ relative phase", "B12": r"Controlled-$R_Y$ and ancilla probability",
    "B13": r"Selecting a clock value with $X$ gates", "B14": r"Post-selection on an ancilla",
    "B15": r"Inverse of a circuit", "B16": r"Gate counts of a 4-qubit QFT", "B17": r"Uniform superposition",
    "B18": r"Product states, reduced states", "B19": r"QPE trace: phase state", "B20": r"QPE trace: outcome",
    "C1": r"What HHL outputs", "C2": r"State after phase estimation", "C3": r"Eigenvalue $\to$ clock bit string",
    "C4": r"Constant $C$ and rotation angle", "C5": r"Ancilla amplitude $C/\lambda$",
    "C6": r"Roles of the registers", "C7": r"Success probability", "C8": r"Qubit count $n_b + n_c + 1$",
    "C9": r"Why uncompute?", "C10": r"Choice of $C$", "C11": r"Negative eigenvalues (two's complement)",
    "C12": r"Code: $X$ gates of the inversion", "C13": r"Norm recovery $\vec x = \lVert\vec b\rVert\,\vec a/C$",
    "C14": r"Default $4\times4$ solution", "C15": r"Algorithm step $\leftrightarrow$ CUDA-Q code",
    "C16": r"Preparing $\ket b$", "C17": r"Role of the inverse QFT", "C18": r"$\ket b$ an eigenvector",
    "C19": r"$2\times2$ HHL trace: after QPE", "C20": r"$2\times2$ HHL trace: output",
    "D1": r"Complexity of HHL", "D2": r"Repetitions and amplitude amplification",
    "D3": r"Quantum linear-system algorithms", "D4": r"Read-out cost of all $x_i$",
    "D5": r"``Exponential speed-up'' claims", "D6": r"Too few clock qubits: rescaled $t$",
    "D7": r"Clock qubits for precision $\epsilon$", "D8": r"Memory of a large HHL simulation",
    "D9": r"Trotter steps for $e^{iAt}$", "D10": r"What the notebook implements",
    "D11": r"Recorded $P_{\text{success}}$: shot statistics", "D12": r"Recorded read-out of $|x_0|^2$",
    "D13": r"Circuit resources vs.\ $n_c$", "D14": r"Error sources and their effects",
    "D15": r"Useful outputs of HHL", "D16": r"Dequantisation", "D17": r"$\kappa^2$ run-time dependence",
    "D18": r"Solution inside the embedded system", "D19": r"Recorded run: $P_{\text{success}}$ exactly",
    "D20": r"Recorded run: $\lVert\vec x\rVert$ from $P_{\text{success}}$",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item Linear system $A\vec x = \vec b$, $A$ a $d\times d$ matrix, $d = 2^{n_b}$. $A$ Hermitian with eigenpairs $A\ket{u_j} = \lambda_j\ket{u_j}$;
      $\ket b = \vec b/\lVert\vec b\rVert = \sum_j\beta_j\ket{u_j}$; $\ket x = A^{-1}\ket b/\lVert A^{-1}\ket b\rVert$.
      Condition number $\kappa = |\lambda|_{\max}/|\lambda|_{\min}$ (Hermitian $A$), in general $\kappa = \sigma_{\max}/\sigma_{\min}$.
      Sparsity $s$ = largest number of non-zero entries in a row.
\item Basis states are written with \textbf{qubit 0 first = most significant bit}: $\ket{q_0q_1\dots}$, e.g.\ $\ket{10} = \ket{2}$.
      $H = \frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}$; $X, Y, Z$ are the Pauli matrices;
      $R_Y(\theta) = \begin{psmallmatrix}\cos\frac\theta2&-\sin\frac\theta2\\ \sin\frac\theta2&\cos\frac\theta2\end{psmallmatrix}$;
      $\ket\pm = \frac{1}{\sqrt2}(\ket0\pm\ket1)$. $\mathrm{QFT}\ket{x} = \frac{1}{\sqrt N}\sum_{y}e^{2\pi i\,xy/N}\ket y$, $N = 2^n$.
\item \textbf{Phase estimation (QPE):} $n_c$ clock qubits in $\ket+$ control $U^{2^j}$, then the inverse QFT; an eigenphase $U\ket u = e^{2\pi i\varphi}\ket u$
      gives the outcome $k \approx \varphi\,2^{n_c}$.
\item \textbf{HHL (as in \texttt{HHL.ipynb}):} registers clock ($n_c$), system ($n_b$), ancilla (1), total $n_c + n_b + 1$ qubits.
      $U = e^{iAt}$, so eigenvalue $\lambda$ gives clock value $k = \lambda t\,2^{n_c}/(2\pi)$. $C = 2\pi/(2^{n_c}t)$ is the smallest representable
      eigenvalue; for clock value $k$ ($\tilde\lambda_k = kC$) the ancilla is rotated by $R_Y(\theta_k)$, $\theta_k = 2\arcsin(C/\tilde\lambda_k)$.
      Clock values $k \ge 2^{n_c-1}$ mean the negative value $k - 2^{n_c}$ (two's complement) unless a question says otherwise.
      The run succeeds when the ancilla is measured as $1$ (and the clock as $0\dots0$); $P_{\text{success}} = \sum_j|\beta_j|^2C^2/\lambda_j^2$ when every $\lambda_j$ is exactly representable.
\item \textbf{Notebook default:} rows of $A$: $(2.5, -0.5, -1, 0)$, $(-0.5, 2.5, 0, -1)$, $(-1, 0, 2.5, -0.5)$, $(0, -1, -0.5, 2.5)$ (eigenvalues $1, 2, 3, 4$),
      $\vec b = (1, 0, 0, 0)$, $n_c = 4$, $t = \pi/8$, $C = 1$, 7 qubits, 20\,000 shots. A \emph{previously recorded run} means the outputs stored in the
      notebook (CUDA-Q 0.16.0, CPU target \texttt{qpp-cpu}).
\item $\log$ means $\log_2$; $\tilde O$ hides polylogarithmic factors. Numerical answers are rounded.
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
\newcommand{\HHLop}{\mathrm{HHL}}
\newcommand{\bub}[1]{\tikz[baseline=-0.65ex]\node[draw, circle, inner sep=0pt, minimum size=4.6mm, font=\scriptsize]{#1};}
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
\textbf{A}~Linear Algebra and Mathematical Foundations, \textbf{B}~Quantum Computing Foundations, \textbf{C}~HHL Algorithm Fundamentals, \textbf{D}~Advanced HHL and Implementation.
\item Every question is of the \textbf{single-correct} type: four options, exactly one correct. Question formats used:
single-correct MCQ, matching lists (List-I/List-II), Assertion--Reason, Statement~I/II and paragraph-based questions.
\item Mark your answers on the \textbf{Response Sheet} at the end of this paper by darkening ONE bubble per question.
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
    t.append(response_sheet())
    t.append(r"\end{document}")
    return "\n".join(t)


def response_sheet():
    blocks = []
    for s_idx, (sec, title, _, _) in enumerate(SECTIONS):
        rows = "\n".join(rf"\textbf{{{20 * s_idx + i}}} & \bub{{A}}\,\bub{{B}}\,\bub{{C}}\,\bub{{D}} \\[1.6pt]"
                         for i in range(1, 21))
        blocks.append(r"""\begin{minipage}[t]{0.235\linewidth}\centering
\textbf{Section """ + sec + r"""}\\[2pt]
\begin{tabular}{@{}r@{\hspace{5pt}}l@{}}
""" + rows + r"""
\end{tabular}
\end{minipage}""")
    score_rows = "\n".join(rf"{sec} & {title} & & & & \\ \hline" for sec, title, _, _ in SECTIONS)
    return (r"""\begin{center}{\Large\bfseries\color{brand} RESPONSE SHEET}\\[2pt]
{\small Darken exactly ONE bubble per question. Leave all four bubbles empty to skip a question (0 marks).}\end{center}
\textbf{Name:} \hrulefill\quad \textbf{Roll No.:} \rule{35mm}{0.4pt}\par\medskip
\noindent""" + r"\hfill".join(blocks) + r"""
\par\bigskip
\begin{center}\textbf{Score computation} (filled in by the evaluator, using the Answer Book)\\[4pt]
\renewcommand{\arraystretch}{1.5}
\begin{tabular}{|c|>{\raggedright\arraybackslash}p{4.6cm}|c|c|c|c|}\hline
\textbf{Sec.} & \textbf{Topic} & \textbf{Attempted} & \textbf{Correct} $N_{\mathrm c}$ & \textbf{Wrong} $N_{\mathrm w}$ & \textbf{Marks} $3N_{\mathrm c} - N_{\mathrm w}$ \\ \hline
""" + score_rows + r"""
\multicolumn{5}{|r|}{\textbf{Total (out of 240)}} & \\ \hline
\end{tabular}\end{center}
""")


# ------------------------------------------------------------------------------------------------ Answer Book
THEMES = [
    ("Linear systems and their classical solution", ["A1", "A6", "A12", "A16", "A18", "C14"]),
    ("Eigenvalues, Hermitian matrices and spectral decomposition", ["A2", "A3", "A5", "A6", "A7", "A10", "A12", "A17"]),
    ("Condition number, norms and sparsity", ["A4", "A8", "A9", "A10", "A13", "A16", "A20", "D17"]),
    ("Hermitian embedding, padding and unitarity of $e^{iAt}$", ["A14", "A15", "A19", "D18"]),
    ("Normalisation, amplitude encoding and state preparation", ["A11", "B2", "B17", "C16"]),
    ("Qubits, measurement, phases and entanglement", ["B1", "B3", "B5", "B10", "B11", "B18"]),
    ("Gates, controlled operations and circuits", ["B4", "B6", "B12", "B13", "B15", "B16", "C12"]),
    ("QFT and quantum phase estimation", ["B7", "B8", "B9", "B16", "B19", "B20", "C2", "C3", "C17", "D6", "D7"]),
    ("Eigenvalue inversion, the constant $C$ and the ancilla", ["C4", "C5", "C10", "C11", "C12", "C19", "C20"]),
    ("Post-selection, success probability and uncomputation", ["B14", "C7", "C9", "C13", "C18", "C20", "D2"]),
    ("HHL registers, circuit structure and resources", ["C6", "C8", "C15", "D8", "D13"]),
    ("Complexity, read-out and advanced algorithms", ["C1", "D1", "D3", "D4", "D5", "D9", "D15", "D16", "D17"]),
    ("Noise, errors and limitations", ["D6", "D10", "D14", "D16"]),
    (r"Interpreting recorded results of \texttt{HHL.ipynb}", ["C14", "D10", "D11", "D12", "D19", "D20"]),
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
For each section count $N_{\mathrm c}$ = number of correct answers and $N_{\mathrm w}$ = number of questions answered wrongly (or with more than
one bubble darkened). Section marks $= 3N_{\mathrm c} - N_{\mathrm w}$; unanswered questions score $0$. The total is out of $240$.
Every numerical answer was re-computed independently (NumPy linear algebra, NumPy state vectors of the HHL circuits in the notebook's convention, and CUDA-Q for the notebook's HHL kernel and the phase-estimation circuits) by
\texttt{verify\_answers.py} in the source folder.
\end{infobox}
\begin{infobox}{Conventions (identical to the Question Paper)}
""" + CONVENTIONS + r"""
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
    paper_tex = BUILD / "hhl_question_paper.tex"
    book_tex = BUILD / "hhl_answer_book.tex"
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
