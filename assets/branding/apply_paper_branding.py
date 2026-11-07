"""One-off migration: add the SILICO FELLER watermark to every assessment generator and remove the OMR response sheet.

Run once from the repository root:  python assets/branding/apply_paper_branding.py
Idempotent: files that are already migrated are skipped.
"""
import glob
import re
import sys

WATERMARK = r"""\usepackage{graphicx}
\usepackage{eso-pic}
% SILICO FELLER watermark: official logo, centred, ~8 % opacity, behind all content, on every page
\AddToShipoutPictureBG{\AtPageCenter{\makebox(0,0){\tikz\node[opacity=0.08, inner sep=0pt]{\includegraphics[width=0.72\paperwidth]{../../../assets/branding/silico_feller_logo.png}};}}}
"""
OLD_INSTR = r"\item Mark your answers on the \textbf{Response Sheet} at the end of this paper by darkening ONE bubble per question."
NEW_INSTR = (r"\item For every question choose \textbf{ONE} option, (A), (B), (C) or (D), and record it against the question number "
             r"in the answer script provided by the invigilator.")

ok = True
for path in sorted(glob.glob("assessments/*_test_source/generate_papers.py")):
    src = open(path, encoding="utf8").read()
    if "AddToShipoutPictureBG" in src:
        print(f"{path}: already migrated"); continue
    s = src
    # 1) watermark in the preamble (after tikz, which it uses)
    assert s.count("\\usepackage{tikz}\n") == 1, path
    s = s.replace("\\usepackage{tikz}\n", "\\usepackage{tikz}\n" + WATERMARK, 1)
    # 2) instruction
    assert s.count(OLD_INSTR) == 1, (path, "instruction")
    s = s.replace(OLD_INSTR, NEW_INSTR)
    # 3) drop the response sheet call and function, and the bubble macro
    assert s.count("    t.append(response_sheet())\n") == 1, (path, "call")
    s = s.replace("    t.append(response_sheet())\n", "")
    m = re.search(r"\ndef response_sheet\(\):\n.*?\n(?=\n\n# -{20,})", s, re.S)
    assert m, (path, "function")
    s = s[:m.start()] + s[m.end():]
    s, n = re.subn(r"\\newcommand\{\\bub\}\[1\]\{[^\n]*\n", "", s)
    assert n == 1, (path, "bub")
    # 4) evaluation note in the Answer Book
    s, n = re.subn(r"\(or with more than\s*\n?\s*one bubble darkened\)", "(or with more than one option marked)", s)
    assert n == 1, (path, "evaluate")
    for bad in ("response_sheet", "bubble", "RESPONSE SHEET", "\\bub{"):
        if bad in s:
            print(f"{path}: still contains {bad!r}"); ok = False
    open(path, "w", encoding="utf8").write(s)
    print(f"{path}: migrated")
sys.exit(0 if ok else 1)
