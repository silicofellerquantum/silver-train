"""Build the QAOA Question Paper and the separate Answer Book from the question banks.

    python generate_papers.py            # writes .tex files, compiles both PDFs with latexmk, copies them to assessments/
    python generate_papers.py --no-pdf   # only writes the .tex files and answer_key.csv

Edit the questions in bank_*.py (never the generated .tex). In the banks the correct option of every 'mcq', 'match'
and 'para' question is listed FIRST; this script moves it to a position chosen so that each section has five
answers of each letter (A/B/C/D) and no letter appears three times in a row. The layout is deterministic (fixed seed),
so re-running the script reproduces the same paper.

This generator follows the QFT generator (assessments/qft_test_source/generate_papers.py); the QAOA Answer Bank
additionally prints a "Relevant formulae" block for every question.
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
PAPER_PDF = "QAOA_Question_Paper.pdf"
ANSWER_PDF = "QAOA_Answer_Book.pdf"
SEED = 20261107
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)
TITLE = r"Quantum Approximate Optimization Algorithm"
SUBTITLE = r"Assessment of the QAOA algorithm (Max-Cut)"

SECTIONS = [
    ("A", "Mathematics and Graph Theory for QAOA",
     "graphs, cuts and Max-Cut, bipartite graphs and odd cycles, complexity, QUBO and Ising forms, binary and spin "
     "variables, probability and sampling statistics, matrix exponentials and commutators", bank_math),
    ("B", "Quantum Mechanics",
     "qubits, superposition and the Born rule, Pauli algebra, rotation gates, CNOT conjugation, expectation values, "
     "entanglement, global phases, symmetries and state-vector memory", bank_qm),
    ("C", "QAOA Basics",
     "cost and mixer Hamiltonians, initial state, QAOA circuit and gate decomposition, parameters and depth, the "
     "$p = 1$ analysis, objective function, sign conventions and the notebook's code", bank_basics),
    ("D", "Advanced QAOA and Implementation",
     "light cones, gradients and classical optimizers, shot noise, parameter strategies, simulation backends and memory, "
     "noise, QAOA variants, limitations, benchmarking and interpretation of recorded results", bank_advanced),
]

KIND_NAMES = {"mcq": "Single-correct MCQ", "match": "Matching lists", "ar": "Assertion--Reason",
              "stmt": "Statement I / II", "para": "Paragraph based"}

TOPICS = {
    "A1": r"Bitstrings vs.\ partitions", "A2": r"$C_{\max}$ of standard graphs",
    "A3": r"Random-split baseline $W/2$", "A4": r"Cut as a QUBO polynomial",
    "A5": r"Random graph $G(n, q)$ statistics", "A6": r"Handshake lemma",
    "A7": r"Odd cycles vs.\ odd node count", "A8": r"Complexity: max cut vs.\ min cut",
    "A9": r"Spectrum of $H_C$ on a triangle", "A10": r"Variance of a random cut",
    "A11": r"Commuting exponentials", "A12": r"Shots for a target standard error",
    "A13": r"Binomial statistics of $P_{\text{opt}}$", "A14": r"Complementary optimal strings",
    "A15": r"QUBO $\to$ Ising conversion", "A16": r"Exact shift rule for derivatives",
    "A17": r"House graph: $C_{\max}$ and count", "A18": r"House graph: random baseline",
    "A19": r"Ground energy $= W/2 - C_{\max}$", "A20": r"Local search: $|E|/2$, not optimal",
    "B1": r"Uniform state; QAOA enhancement", "B2": r"Pauli products", "B3": r"Rotation-gate matrices",
    "B4": r"Relative phase and interference", "B5": r"Expected cut from amplitudes",
    "B6": r"Commutation of Pauli strings", "B7": r"Pauli propagation through CNOT",
    "B8": r"Entanglement by the cost gate", "B9": r"Global phase vs.\ sign of $\gamma$",
    "B10": r"$\langle Z_0Z_1\rangle$ for a product state", "B11": r"Conjugation $e^{i\beta X}Ze^{-i\beta X}$",
    "B12": r"State-vector memory on one GPU", "B13": r"Bit-flip probability of $R_X$",
    "B14": r"$\ket+^{\otimes n}$ as top eigenstate of $H_M$", "B15": r"Complex-conjugation symmetry",
    "B16": r"Linearity of $\langle H_C\rangle$", "B17": r"Mixer at $\beta = \pi/2$; period $\pi/2$",
    "B18": r"Measuring $Z_iZ_j$ from shots", "B19": r"Trace: phase after the cost gate",
    "B20": r"Trace: cut probability after the mixer",
    "C1": r"\texttt{cost\_hamiltonian()} output", "C2": r"All $\gamma_l = 0$", "C3": r"CNOT--$R_Z$--CNOT decomposition",
    "C4": r"Why \texttt{rx(2*beta)}", "C5": r"Gate and CNOT counts", "C6": r"Default circuit statistics",
    "C7": r"$p = 1$ single edge: formula, optimum", "C8": r"$p = 1$ on the ring", "C9": r"$0.6924$ bound; monotonicity in $p$",
    "C10": r"Limit $p \to \infty$", "C11": r"Parameter-vector layout", "C12": r"The ramp initialisation",
    "C13": r"Minimise $\langle H_C\rangle$ = maximise $\langle C\rangle$", "C14": r"Sign of the default start point",
    "C15": r"Algorithm $\leftrightarrow$ code", "C16": r"Triangle solved at $p = 1$", "C17": r"$r$ vs.\ best sample",
    "C18": r"Best evaluated angles; no global guarantee", "C19": r"Path graph: $\langle Z_0Z_1\rangle$",
    "C20": r"Path graph: $\langle C\rangle$, $P_{\text{opt}}$",
    "D1": r"Light cone of an edge", "D2": r"Ring at $p = 2$: $5/6$", "D3": r"Parameter shift vs.\ finite differences",
    "D4": r"COBYLA \texttt{rhobeg}", "D5": r"Shot noise of the objective", "D6": r"Brute-force memory",
    "D7": r"Multi-GPU state vector", "D8": r"Backend / method $\to$ target", "D9": r"When MPS is efficient",
    "D10": r"Shot-based objective on hardware", "D11": r"Depolarising-noise model", "D12": r"QAOA variants",
    "D13": r"INTERP initialisation", "D14": r"Heuristic reference: upper bound on $r$", "D15": r"Recorded timing per gate",
    "D16": r"Run-time scaling to 25 nodes", "D17": r"Quantum advantage: state of knowledge",
    "D18": r"Custom weighted edges", "D19": r"Recorded run: partition probability",
    "D20": r"Recorded run: improvement over random",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item Graph $G = (V, E)$ with nodes $0, 1, \dots, n-1$ and edge weights $w_{ij}$ ($w_{ij} = 1$ unless stated); $W = \sum_{(i,j)\in E} w_{ij}$.
      A split is a bitstring $z = z_0z_1\dots z_{n-1}$ ($z_i \in \{0,1\}$ = group of node $i$, \textbf{node 0 written first});
      $\bar z$ is $z$ with every bit flipped. Spins: $s_i = (-1)^{z_i} = 1 - 2z_i$, the eigenvalue of $Z_i$ on $\ket z$.
\item Cut value $C(z) = \sum_{(i,j)\in E} w_{ij}\,[z_i \neq z_j] = \sum_{(i,j)} w_{ij}\frac{1 - s_is_j}{2}$; $C_{\max} = \max_z C(z)$.
\item \textbf{Notebook convention} (\texttt{QAOA.ipynb}): $H_C = \sum_{(i,j)\in E}\frac{w_{ij}}{2}Z_iZ_j$, so $C = \frac W2 - H_C$;
      $H_M = \sum_i X_i$; $\ket{\boldsymbol\gamma,\boldsymbol\beta} = e^{-i\beta_pH_M}e^{-i\gamma_pH_C}\cdots e^{-i\beta_1H_M}e^{-i\gamma_1H_C}\ket{+}^{\otimes n}$;
      $F_p(\boldsymbol\gamma,\boldsymbol\beta) = \braket{\boldsymbol\gamma,\boldsymbol\beta|C|\boldsymbol\gamma,\boldsymbol\beta}$; approximation ratio $r = F_p/C_{\max}$.
      The angle vector is \texttt{params} $= [\gamma_1, \dots, \gamma_p, \beta_1, \dots, \beta_p]$.
\item Gates: $H = \frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}$; $X, Y, Z$ are the Pauli matrices; $R_X(\theta) = e^{-i\theta X/2}$,
      $R_Z(\theta) = e^{-i\theta Z/2}$; CNOT$(c, t)$ flips qubit $t$ when qubit $c$ is 1. Two-qubit matrices use the basis order
      $\ket{00}, \ket{01}, \ket{10}, \ket{11}$ (lower-numbered qubit first); $\ket{\pm} = \frac{1}{\sqrt2}(\ket0 \pm \ket1)$.
\item \textbf{Notebook defaults:} random graph with 15 nodes and edge probability $0.3$ (\texttt{SEED = 42}: 30 edges), $p = 2$, COBYLA with
      80 evaluations, ramp initial angles, 4000 shots, exact objective via \texttt{cudaq.observe}. A \emph{previously recorded run} means the
      outputs stored in the notebook (CUDA-Q 0.16.0, CPU target \texttt{qpp-cpu}).
\item All measurements are in the computational basis; $\ln$ is the natural logarithm. Numerical answers are rounded.
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
\newcommand{\QAOAop}{\mathrm{QAOA}}
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
\textbf{A}~Mathematics and Graph Theory for QAOA, \textbf{B}~Quantum Mechanics, \textbf{C}~QAOA Basics, \textbf{D}~Advanced QAOA and Implementation.
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
\small""" + CONVENTIONS + r"""
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
    ("Graphs, cuts and Max-Cut", ["A1", "A2", "A6", "A7", "A14", "A17", "A18", "A19", "A20", "D18"]),
    ("QUBO / Ising formulation and the cost Hamiltonian", ["A4", "A9", "A15", "A19", "B9", "C1", "C13"]),
    ("Probability, statistics and sampling", ["A3", "A5", "A10", "A12", "A13", "A18", "B1", "D5", "D19"]),
    ("Complexity and classical algorithms", ["A8", "A20", "C9", "D14", "D17"]),
    ("Linear algebra: commutators, exponentials, Pauli algebra", ["A11", "B2", "B3", "B6", "B7", "B11", "B17"]),
    ("Qubits, measurement, phases and entanglement", ["B4", "B5", "B8", "B10", "B13", "B14", "B15", "B16", "B18", "B19"]),
    ("Circuit construction and gate counts", ["C3", "C4", "C5", "C6", "C15", "D3"]),
    ("Initial state, mixer and parameters", ["B14", "B17", "C2", "C11", "C12", "C14", "D13"]),
    (r"Analysis at $p = 1$ and depth dependence", ["B20", "C7", "C8", "C9", "C10", "C16", "C19", "C20", "D1", "D2"]),
    ("Classical optimizers and gradients", ["A16", "C13", "C18", "D3", "D4", "D5", "D13"]),
    ("Backends, memory and performance", ["B12", "D6", "D7", "D8", "D9", "D15", "D16"]),
    ("Noise, hardware, variants and limitations", ["D10", "D11", "D12", "D17"]),
    (r"Interpreting recorded results of \texttt{QAOA.ipynb}", ["A13", "A14", "B1", "C14", "C17", "D15", "D19", "D20"]),
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
Every numerical answer was re-computed independently (NumPy state vectors in the notebook's convention, brute force over all splits, and CUDA-Q for the QAOA circuits) by
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
    paper_tex = BUILD / "qaoa_question_paper.tex"
    book_tex = BUILD / "qaoa_answer_book.tex"
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
