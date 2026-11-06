"""Build the QFT Question Paper and the separate Answer Book from the question banks.

    python generate_papers.py            # writes .tex files, compiles both PDFs with latexmk, copies them to assessments/
    python generate_papers.py --no-pdf   # only writes the .tex files and answer_key.csv

Edit the questions in bank_*.py (never the generated .tex). In the banks the correct option of every 'mcq', 'match'
and 'para' question is listed FIRST; this script moves it to a position chosen so that each section has five
answers of each letter (A/B/C/D) and no letter appears three times in a row. The layout is deterministic (fixed seed),
so re-running the script reproduces the same paper.
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
PAPER_PDF = "QFT_Question_Paper.pdf"
ANSWER_PDF = "QFT_Answer_Book.pdf"
SEED = 20261106
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)
TITLE = r"Quantum Fourier Transform"
SUBTITLE = r"Assessment of the QFT algorithm"

SECTIONS = [
    ("A", "Mathematics for the QFT",
     "complex numbers, roots of unity, geometric series, the discrete Fourier transform, binary fractions, unitary "
     "matrices, continued fractions and modular arithmetic", bank_math),
    ("B", "Quantum Mechanics",
     "state vectors, Born rule, global and relative phases, gates and unitarity, tensor products, entanglement, "
     "partial measurement and phase kickback", bank_qm),
    ("C", "QFT Basics",
     "definition and matrix of the QFT, product formula, textbook circuit, gate counts, inverse QFT and "
     "simple inputs", bank_basics),
    ("D", "Advanced QFT",
     "periodic states, shift theorem, phase estimation accuracy, Shor's algorithm, approximate and semiclassical "
     "QFT, spectral properties and applications", bank_advanced),
]

KIND_NAMES = {"mcq": "Single-correct MCQ", "match": "Matching lists", "ar": "Assertion--Reason",
              "stmt": "Statement I / II", "para": "Paragraph based"}

TOPICS = {
    "A1": r"Weighted sum of roots of unity", "A2": r"Orthogonality of roots of unity",
    "A3": r"Binary fractions, periodicity of $e^{2\pi i\theta}$", "A4": r"Unitarity; the $4\times4$ Fourier matrix",
    "A5": r"Gauss sum, trace of $F_4$", "A6": r"Interference pattern $|\sum e^{ij\theta}|^2$",
    "A7": r"Continued fractions, convergents", "A8": r"Multiplicative order; Shor's gcd step",
    "A9": r"Parameters of an $n$-qubit state", "A10": r"Walsh--Hadamard signs",
    "A11": r"FFT vs.\ QFT cost (and its limits)", "A12": r"Cyclotomic product identity",
    "A13": r"DFT of a vector; Parseval", "A14": r"Binary expansion of $2^k\varphi$",
    "A15": r"Algebra of 8th roots of unity", "A16": r"Properties of $F$, $F^2$, $H\otimes H$, C$T$",
    "A17": r"Root sums when $\gcd(k,N)>1$", "A18": r"Determinants of unitary matrices",
    "A19": r"Parseval for the un-normalised DFT", "A20": r"Spectral leakage ($r\nmid N$)",
    "B1": r"Born rule, partial measurement", "B2": r"Global vs.\ relative phase",
    "B3": r"Interference with a Hadamard", "B4": r"Bloch-sphere expectation values",
    "B5": r"Entanglement test", "B6": r"Phase kickback (CNOT)", "B7": r"Controlled-phase matrices",
    "B8": r"Kickback of controlled-$U^{2^j}$", "B9": r"Post-measurement state", "B10": r"Unitarity of gates",
    "B11": r"Commutation of gates", "B12": r"State-vector memory cost", "B13": r"Bell state in the $X$ basis",
    "B14": r"Mixture vs.\ superposition", "B15": r"Single-qubit gate actions", "B16": r"Two-qubit correlations",
    "B17": r"Distinguishability; relative phase", "B18": r"Product-state criterion",
    "B19": r"Hadamard test", "B20": r"Repetition and confidence",
    "C1": r"QFT matrix element ($n = 2$)", "C2": r"Output distribution of $\mathrm{QFT}\ket{x}$",
    "C3": r"Product formula ($n = 3$)", "C4": r"Controlled-rotation count", "C5": r"Two-qubit gates incl.\ SWAPs",
    "C6": r"Smallest rotation angle", "C7": r"Phase on qubit 1; role of SWAPs", "C8": r"$\mathrm{QFT}^2$ = negation",
    "C9": r"QFT of the uniform state", "C10": r"QFT of $\ket{10\dots0}$", "C11": r"Constructing the inverse QFT",
    "C12": r"Phase of a QFT amplitude", "C13": r"QFT vs.\ NumPy FFT conventions",
    "C14": r"$\mathrm{QFT}\ket{x}$ is a product state", "C15": r"Gate count $n(n-1)/2$",
    "C16": r"Two-qubit QFT outputs", "C17": r"Why measuring $\mathrm{QFT}\ket{x}$ reveals nothing",
    "C18": r"What the QFT does \emph{not} speed up", "C19": r"Circuit trace: phase build-up",
    "C20": r"Circuit trace: final amplitude, SWAP",
    "D1": r"QFT of a periodic state ($r\mid N$)", "D2": r"Phases of QFT peaks", "D3": r"Shift theorem",
    "D4": r"QPE precision vs.\ counting qubits", "D5": r"QPE with exact $t$-bit phase",
    "D6": r"The $4/\pi^2$ success bound", "D7": r"QPE distribution for $\varphi = 1/3$",
    "D8": r"Qubit budget of Shor's algorithm", "D9": r"Order finding, $a = 7$, $N = 15$",
    "D10": r"Failure mode $\gcd(k, r) > 1$", "D11": r"Approximate QFT gate savings",
    "D12": r"Periodic state with $r\nmid N$", "D13": r"Eigenvalues of the QFT", "D14": r"Semiclassical QFT",
    "D15": r"Applications of the QFT", "D16": r"Period $\leftrightarrow$ peak spacing",
    "D17": r"Approximate QFT error bound", "D18": r"Non-terminating binary phases",
    "D19": r"Shor, $N = 21$: continued fractions", "D20": r"Shor, $N = 21$: peak probability",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item $n$ qubits, $N = 2^n$, $\omega = e^{2\pi i/N}$. A basis state is written $\ket{x} = \ket{x_1x_2\dots x_n}$ with
      $x = \sum_{j} x_j 2^{\,n-j}$: \textbf{qubit 1 is the most significant bit} and is written first.
\item $\displaystyle \mathrm{QFT}\ket{x} = \frac{1}{\sqrt N}\sum_{y=0}^{N-1}\omega^{xy}\ket{y}$, \quad
      $\displaystyle \mathrm{QFT}^{-1}\ket{y} = \frac{1}{\sqrt N}\sum_{x=0}^{N-1}\omega^{-xy}\ket{x}$.
      Binary fraction: $0.x_lx_{l+1}\dots x_m = \frac{x_l}{2} + \frac{x_{l+1}}{4} + \dots + \frac{x_m}{2^{m-l+1}}$.
\item $H = \frac{1}{\sqrt2}\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}$; $X, Y, Z$ are the Pauli matrices;
      $R_k = \operatorname{diag}\big(1, e^{2\pi i/2^k}\big)$, so $R_1 = Z$, $S = R_2$, $T = R_3$;
      $\ket{\pm} = \frac{1}{\sqrt2}(\ket0 \pm \ket1)$. Two-qubit matrices use the basis order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$.
\item \textbf{Textbook QFT circuit:} for $j = 1, \dots, n$: apply $H$ to qubit $j$, then for $k = 2, \dots, n-j+1$ a controlled-$R_k$
      with control qubit $j+k-1$ and target qubit $j$; finally SWAP qubits $j$ and $n+1-j$ for $j \le n/2$.
\item \textbf{Phase estimation (QPE):} $U\ket{u} = e^{2\pi i\varphi}\ket{u}$, $0 \le \varphi < 1$; $t$ counting qubits in $\ket{+}$ control
      $U^{2^{t-1}}, \dots, U^{2}, U$; the inverse QFT is applied and the outcome $y$ gives the estimate $\varphi \approx y/2^t$.
\item All measurements are in the computational basis unless stated otherwise; $\log$ means $\log_2$.
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
    return (r"\textbf{\color{brand}What you need to know.} " + xref(ex["concept"]) + "\n\n"
            + r"\textbf{\color{brand}Step-by-step solution.} " + q["sol"].strip() + "\n\n"
            + r"\textbf{\color{brand}Why the other options are wrong.}" + "\n"
            + "\\begin{itemize}[leftmargin=2.4em, itemsep=1pt, topsep=2pt]\n" + xref(items) + "\n\\end{itemize}\n"
            + r"\textbf{\color{brand}Remember.} " + xref(ex["takeaway"]))


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
\newcommand{\QFTop}{\mathrm{QFT}}
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
\textbf{A}~Mathematics for the QFT, \textbf{B}~Quantum Mechanics, \textbf{C}~QFT Basics, \textbf{D}~Advanced QFT.
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
\begin{tabular}{|c|l|c|c|c|c|}\hline
\textbf{Sec.} & \textbf{Topic} & \textbf{Attempted} & \textbf{Correct} $C$ & \textbf{Wrong} $W$ & \textbf{Marks} $3C - W$ \\ \hline
""" + score_rows + r"""
\multicolumn{5}{|r|}{\textbf{Total (out of 240)}} & \\ \hline
\end{tabular}\end{center}
""")


# ------------------------------------------------------------------------------------------------ Answer Book
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
\item the \textbf{answer key} with the topic tested by every question (page~\pageref{key}), and
\item the \textbf{Answer Bank}: every question restated with the correct option marked \textcolor{okgreen}{\ding{51}} and a
      complete solution with derivations (from page~\pageref{bank}).
\end{enumerate}
""" + marking_box() + r"""
\begin{infobox}{How to evaluate}
For each section count $C$ = number of correct answers and $W$ = number of questions answered wrongly (or with more than
one bubble darkened). Section marks $= 3C - W$; unanswered questions score $0$. The total is out of $240$.
Every numerical answer was re-computed independently (NumPy, and CUDA-Q state vectors for the QFT circuits) by
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
    for sec, *_ in SECTIONS:
        cnt = Counter(q["answer"] for q in questions if q["section"] == sec)
        assert all(cnt[i] == 5 for i in range(4)), (sec, cnt)
    BUILD.mkdir(exist_ok=True)
    paper_tex = BUILD / "qft_question_paper.tex"
    book_tex = BUILD / "qft_answer_book.tex"
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
