"""Build the Shor's Algorithm Question Paper and the separate Answer Book from the question banks.

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
PAPER_PDF = "Shor_Question_Paper.pdf"
ANSWER_PDF = "Shor_Answer_Book.pdf"
SEED = 20261110
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)
TITLE = r"Shor's Algorithm"
SUBTITLE = r"Number theory, order finding, phase estimation, implementation and cryptography"

SECTIONS = [
    ("A", "Mathematical and Number Theory Foundations",
     "modular arithmetic, gcd and the Euclidean algorithm, modular inverses, Euler's totient, multiplicative order, repeated "
     "squaring, the Chinese remainder theorem, square roots of unity, classical factoring methods and the reduction to order finding", bank_math),
    ("B", "Quantum Computing Foundations",
     "registers and superposition, entanglement, unitary modular multiplication, eigenvalues and eigenvectors, phase kickback, "
     "Hadamard, phase and controlled gates, interference, reversibility, measurement, simulation memory and phase estimation", bank_qm),
    ("C", "Shor's Algorithm Fundamentals",
     "the steps of the algorithm, choice of the base, register sizes, outcome distributions, continued fractions, success and failure "
     "cases, repetition, classical post-processing and the functions of \\texttt{Shor.ipynb}", bank_basics),
    ("D", "Advanced Shor's Algorithm and Implementation",
     "resource estimation and published estimates, qubit-saving constructions, gate-level compilation, approximate QFT, noise, "
     "simulation limits, RSA and elliptic curves, post-quantum cryptography and fault tolerance", bank_advanced),
]

KIND_NAMES = {"mcq": "Single-correct MCQ", "match": "Matching lists", "ar": "Assertion--Reason",
              "stmt": "Statement I / II", "para": "Paragraph based"}

TOPICS = {
    "A1": r"Euclidean algorithm", "A2": r"Modular inverse", "A3": r"Euler's totient $\varphi(pq)$",
    "A4": r"Multiplicative order", "A5": r"Repeated squaring", "A6": r"Square-and-multiply cost",
    "A7": r"Orders divide $\lambda(N)$", "A8": r"Lucky gcd probability", "A9": r"Chinese remainder theorem",
    "A10": r"Square roots of $1$ mod $15$", "A11": r"Failure: $a^{r/2} \equiv -1$", "A12": r"Fermat's little theorem",
    "A13": r"Trial-division bound", "A14": r"Success bound $1 - 2^{-(k-1)}$", "A15": r"$\varphi$, order, $\lambda$, inverse",
    "A16": r"Classical factoring complexity", "A17": r"Non-trivial square root $\Rightarrow$ factor", "A18": r"Euler's theorem vs order",
    "A19": r"Order of $3$ mod $91$", "A20": r"Factors of $91$ by gcd",
    "B1": r"Uniform superposition", "B2": r"Measuring the work register", "B3": r"Unitarity of $U_a$",
    "B4": r"Eigenvalues of $U_a$", "B5": r"$\ket1$ as sum of eigenvectors", "B6": r"Phase kickback",
    "B7": r"Phases of $U^{2^j}$", "B8": r"Hadamard transform signs", "B9": r"Two-path interference",
    "B10": r"Reversible modular exponentiation", "B11": r"CNOT cost of a Fredkin gate", "B12": r"Implicit measurement",
    "B13": r"Notebook qubit count", "B14": r"State-vector memory", "B15": r"Roles of circuit blocks",
    "B16": r"Phase gates $R_1(\lambda)$", "B17": r"Kickback and controls", "B18": r"Parallelism vs interference",
    "B19": r"QPE: exact phase", "B20": r"QPE: inexact phase",
    "C1": r"Steps of Shor's algorithm", "C2": r"$N = 15$, $a = 2$", "C3": r"Failure for $a = N - 1$",
    "C4": r"Odd order: retry", "C5": r"Lucky base", "C6": r"Register sizes for $N = 33$",
    "C7": r"$N = 15$, $a = 11$ outcomes", "C8": r"Continued fractions, $N = 33$", "C9": r"Multiples test",
    "C10": r"Condition $2^t \ge N^2$", "C11": r"Exact success, notebook default", "C12": r"Probability $\varphi(r)/r$",
    "C13": r"Role of the QFT in Shor", "C14": r"Skipped identity multipliers", "C15": r"Functions of \texttt{Shor.ipynb}",
    "C16": r"Outcomes to fractions", "C17": r"Prime powers", "C18": r"Uninformative outcomes",
    "C19": r"$N = 39$: order and factors", "C20": r"$N = 39$: continued fraction",
    "D1": r"Beauregard qubit count", "D2": r"Published RSA-2048 estimates", "D3": r"$O(n^3)$ scaling",
    "D4": r"Semiclassical QFT", "D5": r"Approximate QFT", "D6": r"Controlled phases in the QFT",
    "D7": r"Recorded gate counts", "D8": r"No hardware execution", "D9": r"CNOTs of a controlled multiplier",
    "D10": r"Random baseline", "D11": r"Memory for $N = 1023$", "D12": r"Toy RSA",
    "D13": r"Post-quantum standards", "D14": r"Impact on RSA, ECC, AES", "D15": r"Fault-tolerance terms",
    "D16": r"Qubit counts of constructions", "D17": r"Harvest now, decrypt later", "D18": r"Qubits and repeated squaring",
    "D19": r"Error-free probability vs success", "D20": r"Interpreting noisy results",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item $a \bmod N$ is the remainder in $\{0, \dots, N - 1\}$; the \textbf{order} $r = \operatorname{ord}_N(a)$ is the least $r > 0$ with $a^r \equiv 1 \pmod N$ (defined for $\gcd(a, N) = 1$);
      $\varphi$ is Euler's totient, $\lambda$ the Carmichael function. ``Factors'' means non-trivial factors ($1 < p < N$).
\item \textbf{Reduction:} if $r$ is even and $a^{r/2} \not\equiv -1 \pmod N$, then $\gcd(a^{r/2} - 1, N)$ and $\gcd(a^{r/2} + 1, N)$ are non-trivial factors; otherwise the attempt fails.
\item \textbf{Circuit:} $t$ counting qubits (Hadamards), work register in $\ket1$, controlled $U_{a^{2^j} \bmod N}$ with $U_c\ket y = \ket{cy \bmod N}$ for $y < N$, inverse QFT, measurement of the
      counting register giving $y \in \{0, \dots, 2^t - 1\}$ (most significant bit first). \textbf{Post-processing:} \texttt{Fraction(y, 2**t).limit\_denominator(N)} gives $s/r'$;
      the order is $r'$ or its smallest multiple $kr'$ with $a^{kr'} \equiv 1$. $y = 0$ gives no candidate.
\item \textbf{Notebook} (\texttt{Shor.ipynb}, defaults): $N = 15$, $a = 7$, $n = $ \texttt{N.bit\_length()} work qubits, $t = 2n$ counting qubits, multipliers as exact
      permutation matrices (\texttt{cudaq.register\_operation}), $1000$ shots. \textbf{Previously recorded run}: CUDA-Q 0.16.0, CPU target \texttt{qpp-cpu}, no GPU; $12$ qubits,
      $51$ gates (plus $8$ measurements), outcomes $0, 64, 128, 192$; $0.766$ of the shots gave $15 = 3 \times 5$. No physical quantum processor was used.
\item State-vector memory: $16$ bytes per amplitude ($2^q$ amplitudes for $q$ qubits). $1$ MB $= 10^6$ bytes, $1$ GB $= 10^9$ bytes.
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
\setlength{\emergencystretch}{3em}
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
\textbf{A}~Mathematical and Number Theory Foundations, \textbf{B}~Quantum Computing Foundations, \textbf{C}~Shor's Algorithm Fundamentals, \textbf{D}~Advanced Shor's Algorithm and Implementation.
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
    ("Modular arithmetic, gcd, inverses and totients", ["A1", "A2", "A3", "A9", "A12", "A15", "D12"]),
    ("Multiplicative order and repeated squaring", ["A4", "A5", "A6", "A7", "A18", "A19", "B7", "D18"]),
    ("Reduction of factoring to order finding", ["A8", "A10", "A11", "A14", "A17", "A20", "C1", "C2", "C3", "C4", "C5", "C17", "C19"]),
    ("Classical factoring and complexity", ["A13", "A16", "D3"]),
    ("Superposition, entanglement and measurement", ["B1", "B2", "B10", "B12", "B18"]),
    ("Unitary $U_a$, eigenvectors and phase kickback", ["B3", "B4", "B5", "B6", "B17"]),
    ("Gates and interference", ["B8", "B9", "B11", "B16", "D6", "D9"]),
    ("Phase estimation and the QFT", ["B15", "B19", "B20", "C13", "D4", "D5", "D6"]),
    ("Outcomes, continued fractions and post-processing", ["C7", "C8", "C9", "C10", "C11", "C12", "C16", "C18", "C20"]),
    ("Register sizes, memory and simulation limits", ["B13", "B14", "C6", "C10", "D11", "D18"]),
    (r"\texttt{Shor.ipynb}: code and recorded results", ["C11", "C14", "C15", "D7", "D8"]),
    ("Gate-level circuits and noise", ["B11", "D9", "D10", "D19", "D20"]),
    ("Resource estimates and fault tolerance", ["D1", "D2", "D4", "D15", "D16"]),
    ("RSA, ECC and post-quantum cryptography", ["D12", "D13", "D14", "D17"]),
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
Every numerical answer was re-computed independently (exact integer and \texttt{Fraction} arithmetic, NumPy phase-estimation formulas, checks against the code and recorded outputs of \texttt{Shor.ipynb}, and CUDA-Q state-vector and density-matrix simulations with the notebook's circuit builder) by
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
    paper_tex = BUILD / "shor_question_paper.tex"
    book_tex = BUILD / "shor_answer_book.tex"
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
