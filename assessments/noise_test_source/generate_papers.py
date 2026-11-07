"""Build the Quantum Noise Simulation Question Paper and the separate Answer Book from the question banks.

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
PAPER_PDF = "Noise_Question_Paper.pdf"
ANSWER_PDF = "Noise_Answer_Book.pdf"
SEED = 20261108
LETTERS = "ABCD"
MARKS = dict(full=3, zero=0, negative=-1)
TITLE = r"Quantum Noise Simulation"
SUBTITLE = r"Assessment of noise models, channels, simulation and mitigation"

SECTIONS = [
    ("A", "Mathematical and Quantum Foundations",
     "probability and error statistics, complex numbers and phases, matrices and linear operators, density matrices, pure "
     "and mixed states, purity, eigenvalues, partial trace, the Bloch sphere and fidelity of states", bank_math),
    ("B", "Quantum Noise Fundamentals",
     "decoherence, bit flips, phase flips, bit-phase flips, depolarizing noise, amplitude and phase damping, $T_1$, $T_2$ "
     "and pure dephasing, measurement (readout) errors, coherent versus stochastic errors", bank_qm),
    ("C", "Noise Models and Simulation",
     "Kraus operators, quantum channels and CPTP maps, density-matrix evolution, noisy circuits, sampling, gate noise, "
     "simulator targets and memory, figures of merit, and the code and recorded results of \\texttt{Noise\\_Simulation.ipynb}", bank_basics),
    ("D", "Advanced Noise Analysis and Implementation",
     "process and average gate fidelity, error rates and randomized benchmarking, zero-noise extrapolation, readout-error "
     "mitigation, dynamical decoupling, quantum error correction, twirling, CUDA-Q implementation and practical limitations", bank_advanced),
]

KIND_NAMES = {"mcq": "Single-correct MCQ", "match": "Matching lists", "ar": "Assertion--Reason",
              "stmt": "Statement I / II", "para": "Paragraph based"}

TOPICS = {
    "A1": r"Density matrix of a pure state", "A2": r"Purity $\operatorname{Tr}\rho^2$", "A3": r"Bloch vector from $\rho$",
    "A4": r"Validity of density matrices", "A5": r"Trace preservation of Kraus sets", "A6": r"Random phases $\to$ dephasing",
    "A7": r"Error-free probability $(1-p)^n$", "A8": r"Shot-noise standard error", "A9": r"Coherent angle error",
    "A10": r"Fidelity with a mixed state", "A11": r"Partial trace; entanglement and mixedness", "A12": r"Eigenvalues of a qubit $\rho$",
    "A13": r"Independent qubits: tensor product", "A14": r"Conjugation $X\rho X$ on the Bloch sphere", "A15": r"Bloch vectors of states",
    "A16": r"Hermitian, unitary, projector, Kraus", "A17": r"Mixed states inside the Bloch ball", "A18": r"Density-matrix conditions; Born rule",
    "A19": r"Tomography: Bloch vector, purity", "A20": r"Tomography: fidelity, probabilities",
    "B1": r"Fixed points of bit flip", "B2": r"Phase flip on $\ket+$", "B3": r"Depolarizing flip probability",
    "B4": r"Amplitude damping on $\ket+$", "B5": r"Phase damping $\equiv$ phase flip", "B6": r"$T_1$ half-life",
    "B7": r"$T_2$ from $T_1$ and $T_\varphi$", "B8": r"Constraint $T_2 \le 2T_1$", "B9": r"Coherence-limited depth",
    "B10": r"Readout error probabilities", "B11": r"Readout errors on GHZ", "B12": r"Coherent vs incoherent errors",
    "B13": r"Two-qubit depolarizing marginal", "B14": r"Phase errors and the $X$ basis", "B15": r"Kraus operators of channels",
    "B16": r"Channels on the Bloch ball", "B17": r"Non-unital amplitude damping", "B18": r"Complete dephasing vs unitary $Z$",
    "B19": r"$T_1$/$T_2$ decay of $\ket+$", "B20": r"Fidelity and $T_\varphi$",
    "C1": r"CUDA-Q depolarizing convention", "C2": r"Constant identity gates and noise", "C3": r"CPTP check",
    "C4": r"Applying the bit-flip channel", "C5": r"Composing depolarizing channels", "C6": r"Notebook: depolarizing $(1-p)^G$",
    "C7": r"Notebook: bit-flip $(1-p)^{29}$", "C8": r"Density-matrix memory", "C9": r"What \texttt{stim} can simulate",
    "C10": r"\texttt{as\_counts\_from\_rho}", "C11": r"Hellinger fidelity and TVD", "C12": r"Parity from $X$-basis counts",
    "C13": r"Trajectories vs density matrix", "C14": r"\texttt{add\_all\_qubit\_channel}", "C15": r"Noise-capable CUDA-Q targets",
    "C16": r"Figures of merit", "C17": r"$(1-p)^G$: bound, not exact value", "C18": r"Density-matrix path in CUDA-Q 0.16",
    "C19": r"Recorded run: leaked probability", "C20": r"Recorded run: population vs parity",
    "D1": r"Process vs average gate fidelity", "D2": r"Randomized benchmarking", "D3": r"Linear ZNE",
    "D4": r"Richardson ZNE", "D5": r"Gate folding scale factor", "D6": r"Readout mitigation",
    "D7": r"Limits of readout mitigation", "D8": r"Ramsey vs Hahn echo", "D9": r"DD and Markovian noise",
    "D10": r"3-qubit bit-flip code", "D11": r"Surface-code scaling", "D12": r"Coherent vs stochastic accumulation",
    "D13": r"Pauli twirling", "D14": r"Notebook: \texttt{auto} method choice", "D15": r"Mitigation, DD, QEC",
    "D16": r"Error taxonomy", "D17": r"Cost and nature of ZNE", "D18": r"Readout and hardware in the notebook",
    "D19": r"ZNE extrapolation values", "D20": r"Interpreting ZNE",
}

CONVENTIONS = r"""
\begin{itemize}[leftmargin=1.2em, itemsep=1pt, topsep=2pt]
\item Qubit 0 is written first: $\ket{q_0q_1\dots}$; two-qubit matrices use the basis order $\ket{00}, \ket{01}, \ket{10}, \ket{11}$. $X, Y, Z$ are the Pauli matrices,
      $H$ the Hadamard gate, $R_P(\theta) = e^{-i\theta P/2}$; CNOT flips the target when the control is 1.
\item A qubit state is $\rho = \frac12(I + r_xX + r_yY + r_zZ)$ with Bloch vector $\vec r$, $|\vec r| \le 1$; purity $\operatorname{Tr}\rho^2$. State fidelity with a pure target:
      $F = \braket{\psi|\rho|\psi}$. Measurement probabilities are the diagonal of $\rho$; $\langle Z\rangle = P(0) - P(1)$.
\item A channel is $\mathcal E(\rho) = \sum_kK_k\rho K_k^\dagger$ with $\sum_kK_k^\dagger K_k = I$. Bit flip: $\sqrt{1-p}\,I, \sqrt p\,X$; phase flip: $\sqrt{1-p}\,I, \sqrt p\,Z$;
      bit-phase flip: $\sqrt{1-p}\,I, \sqrt p\,Y$; depolarizing (CUDA-Q): $\sqrt{1-p}\,I, \sqrt{p/3}\,X, \sqrt{p/3}\,Y, \sqrt{p/3}\,Z$ (two-qubit: $\sqrt{1-p}\,II$ and the 15
      other Paulis with $p/15$); amplitude damping: $\operatorname{diag}(1, \sqrt{1-\gamma})$, $\sqrt\gamma\ket0\!\bra1$; phase damping: $\operatorname{diag}(1, \sqrt{1-\lambda})$, $\operatorname{diag}(0, \sqrt\lambda)$.
\item $T_1$: $\rho_{11}(t) = \rho_{11}(0)e^{-t/T_1}$; $T_2$: $|\rho_{01}(t)| = |\rho_{01}(0)|e^{-t/T_2}$; $\frac1{T_2} = \frac1{2T_1} + \frac1{T_\varphi}$. $\ln$ is the natural logarithm.
\item \textbf{Notebook} (\texttt{Noise\_Simulation.ipynb}, defaults): 15-qubit GHZ circuit ($H$ on qubit 0, CNOT chain), the chosen channel with probability $p$ after
      every \texttt{h}, \texttt{x}, \texttt{ry}, \texttt{rz} gate and a two-qubit channel after every CNOT; no measurement noise; 10\,000 shots, seed 42.
      \textbf{Previously recorded run}: CUDA-Q 0.16.0, CPU target \texttt{stim}, no GPU, depolarizing $p = 0.01$: population $0.8936$, $X$-basis parity $0.6910$,
      $(1-p)^G = 0.8601$. No physical quantum processor was used.
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
\newcommand{\bub}[1]{\tikz[baseline=-0.65ex]\node[draw, circle, inner sep=0pt, minimum size=4.6mm, font=\scriptsize]{#1};}
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
\textbf{A}~Mathematical and Quantum Foundations, \textbf{B}~Quantum Noise Fundamentals, \textbf{C}~Noise Models and Simulation, \textbf{D}~Advanced Noise Analysis and Implementation.
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
    ("Probability, statistics and sampling", ["A7", "A8", "A13", "B10", "B11", "C11", "C12", "C19"]),
    ("Density matrices, purity and the Bloch sphere", ["A1", "A2", "A3", "A4", "A12", "A15", "A17", "A18", "A19"]),
    ("Linear operators, partial trace and fidelity", ["A10", "A11", "A14", "A16", "A20", "B20", "D1"]),
    ("Phases, decoherence and dephasing", ["A6", "B2", "B5", "B14", "B18", "D8", "D9"]),
    ("Pauli channels: bit, phase, bit-phase flip, depolarizing", ["B1", "B3", "B13", "B15", "B16", "C1", "C4", "C5"]),
    ("Amplitude and phase damping, $T_1$, $T_2$", ["A5", "B4", "B5", "B6", "B7", "B8", "B9", "B17", "B19", "B20"]),
    ("Kraus operators and CPTP maps", ["A5", "A16", "B15", "C3", "C4", "C5", "C17"]),
    ("Measurement and readout errors", ["B10", "B11", "D6", "D7", "D18"]),
    ("Coherent vs stochastic errors", ["A9", "B12", "D12", "D13", "D16"]),
    (r"Noise models and simulator targets in CUDA-Q", ["C1", "C2", "C8", "C9", "C13", "C14", "C15", "C18", "D14"]),
    (r"\texttt{Noise\_Simulation.ipynb}: code and recorded results", ["C6", "C7", "C10", "C16", "C19", "C20", "D14", "D18"]),
    ("Fidelity, error rates and benchmarking", ["D1", "D2", "C11", "C16", "C19"]),
    ("Error mitigation: ZNE, readout mitigation", ["D3", "D4", "D5", "D6", "D7", "D17", "D19", "D20"]),
    ("Dynamical decoupling and quantum error correction", ["D8", "D9", "D10", "D11", "D15"]),
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
Every numerical answer was re-computed independently (NumPy Kraus-operator and density-matrix calculations, exact arithmetic where possible, and CUDA-Q density-matrix and stabilizer simulations, including the noise model of \texttt{Noise\_Simulation.ipynb}) by
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
    paper_tex = BUILD / "noise_question_paper.tex"
    book_tex = BUILD / "noise_answer_book.tex"
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
