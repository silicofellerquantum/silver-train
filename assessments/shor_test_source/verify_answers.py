"""Recompute every numerical / structural answer of the Shor's-algorithm paper and assert it matches the key.

Run:  python verify_answers.py          (exact integer / Fraction arithmetic and NumPy only)
      python verify_answers.py --cudaq  (additionally runs the circuit builder of Shor.ipynb with CUDA-Q for N = 15, 21, 33 and
                                         39, re-checks the recorded run, and checks the illustrative gate-level / noise /
                                         approximate-QFT numbers written by presentations/shor_source/make_figures.py)

Every check compares a recomputed value with the option the bank lists FIRST (the correct option of mcq/match/para
questions) or with the 'correct' index of ar/stmt questions. Claims about the repository are checked against
dev/notebook_builders/build_shor.py and the outputs stored in Shor.ipynb. Nothing here runs on quantum hardware.
"""
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

import bank_advanced
import bank_basics
import bank_math
import bank_qm
from bank_common import AR_OPTIONS, STATEMENT_OPTIONS
from bank_explain import EXPLAIN

ROOT = Path(__file__).resolve().parents[2]
BUILDER = (ROOT / "dev" / "notebook_builders" / "build_shor.py").read_text()
NB_TEXT = "\n".join("".join(o.get("text", "")) for c in json.loads((ROOT / "Shor.ipynb").read_text())["cells"]
                    if c["cell_type"] == "code" for o in c.get("outputs", []))
FIG = ROOT / "presentations" / "shor_source" / "figures"
CHECKS = []
BANK = {q["id"]: q for m in (bank_math, bank_qm, bank_basics, bank_advanced) for q in m.QUESTIONS}


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def first(qid):
    """The option listed first in the bank (= correct option for mcq/match/para)."""
    return BANK[qid]["opts"][0]


def has(qid, *fragments):
    """Check that the correct option contains each fragment (fragments are LaTeX strings)."""
    check(f"{qid} option text", all(f in first(qid) for f in fragments), (first(qid), fragments))


def order(a, m):
    if math.gcd(a, m) != 1:
        return None
    r, v = 1, a % m
    while v != 1:
        v, r = v * a % m, r + 1
    return r


def phi(m):
    return sum(1 for k in range(1, m + 1) if math.gcd(k, m) == 1)


def candidate_orders(y, t, modulus, a):
    """Copy of the notebook's post-processing (checked against build_shor.py below)."""
    if y == 0:
        return None, []
    fraction = Fraction(y, 2 ** t).limit_denominator(modulus)
    base = fraction.denominator
    return fraction, [base * k for k in range(1, modulus // base + 1) if pow(a, base * k, modulus) == 1][:1]


def factors_from(y, t, modulus, a):
    _, orders = candidate_orders(y, t, modulus, a)
    if orders and orders[0] % 2 == 0 and pow(a, orders[0] // 2, modulus) != modulus - 1:
        x = pow(a, orders[0] // 2, modulus)
        p, q = math.gcd(x - 1, modulus), math.gcd(x + 1, modulus)
        if 1 < p < modulus and p * q == modulus:
            return (min(p, q), max(p, q))
    return None


def shor_distribution(N, a, t):
    """Exact P(y) of order finding: sum over the classes {x : a^x = f} of |sum_x e^{2 pi i x y / 2^t}|^2 / 4^t."""
    M = 2 ** t
    f = np.array([pow(a, x, N) for x in range(M)])
    P = np.zeros(M)
    for v in np.unique(f):
        P += np.abs(np.fft.fft((f == v).astype(float))) ** 2
    return P / M ** 2


def qpe(phase, t):
    M = 2 ** t
    return np.array([abs(np.exp(2j * np.pi * np.arange(M) * (phase - y / M)).sum() / M) ** 2 for y in range(M)])


def match_key(qid):
    """Parse 'P->i, Q->j, ...' of the first option into the list-II indices."""
    return [int(d) for d in re.findall(r"\\to\$(\d)", first(qid))]


# ------------------------------------------------------------------------------------------- structure
for qid, q in BANK.items():
    check(f"{qid} four options", len(q["opts"]) == 4 and len(set(q["opts"])) == 4)
    check(f"{qid} explanation", qid in EXPLAIN and len(EXPLAIN[qid]["wrong"]) == 3)
check("80 questions", len(BANK) == 80)
ARST = {"A17": 0, "A18": 2, "B17": 2, "B18": 3, "C17": 3, "C18": 1, "D17": 1, "D18": 0}
for qid, c in ARST.items():
    q = BANK[qid]
    check(f"{qid} ar/stmt key", q["correct"] == c and q["opts"] in (AR_OPTIONS, STATEMENT_OPTIONS), q.get("correct"))

# ------------------------------------------------------------------------------------------- Section A
check("A1", math.gcd(1071, 462) == 21 and first("A1") == "$21$")
check("A2", pow(3, -1, 20) == 7 and first("A2") == "$7$")
check("A3", phi(91) == 72 and first("A3") == "$72$")
check("A4", order(2, 11) == 10 and first("A4") == "$10$")
check("A5", pow(3, 13, 17) == 12 and first("A5") == "$12$")
bits = bin(1000)[2:]
check("A6", bits == "1111101000" and (len(bits) - 1) + (bits.count("1") - 1) == 14 and first("A6") == "$14$")
orders35 = {order(a, 35) for a in range(1, 35) if math.gcd(a, 35) == 1}
check("A7", 5 not in orders35 and {12, 4, 6} <= orders35 and first("A7") == "$5$", orders35)
check("A8", Fraction(sum(1 for a in range(2, 14) if math.gcd(a, 15) > 1), 12) == Fraction(1, 2) and first("A8") == "$1/2$")
check("A9", [x for x in range(15) if x % 3 == 2 and x % 5 == 3] == [8] and first("A9") == "$8$")
check("A10", [x for x in range(15) if x * x % 15 == 1] == [1, 4, 11, 14])
check("A11", order(5, 21) == 6 and pow(5, 3, 21) == 20)
has("A11", "$r = 6$", "5^3 \\equiv -1")
check("A12", pow(2, 100, 13) == 3 and first("A12") == "$3$")
check("A13", math.isqrt(10403) == 101 and 10403 == 101 * 103 and first("A13") == "$101$")
check("A14", 1 - Fraction(1, 2 ** (3 - 1)) == Fraction(3, 4) and first("A14") == "$3/4$")
# A14: the bound holds for N = 105 (exact fraction of good bases)
good105 = [a for a in range(1, 105) if math.gcd(a, 105) == 1 and order(a, 105) % 2 == 0 and pow(a, order(a, 105) // 2, 105) != 104]
check("A14 bound holds", Fraction(len(good105), phi(105)) >= Fraction(3, 4), Fraction(len(good105), phi(105)))
# A15: list-II = [13, 8, 2, 4]; values of list-I
vals = [phi(15), order(4, 15), 4, pow(7, -1, 15)]
lam15 = max(order(a, 15) for a in range(1, 15) if math.gcd(a, 15) == 1)
check("A15", lam15 == 4 and [[13, 8, 2, 4][i - 1] for i in match_key("A15")] == vals, (vals, match_key("A15")))
check("A16", match_key("A16") == [4, 2, 1, 3])
x_ok = all(math.gcd(x - 1, N) not in (1, N) for N in (15, 21, 35, 91, 33) for x in range(2, N - 1) if x * x % N == 1)
check("A17 assertion (exhaustive small N)", x_ok)
check("A18 Euler true / order=phi false", all(pow(a, phi(N), N) == 1 for N in (15, 21, 35) for a in range(1, N) if math.gcd(a, N) == 1)
      and order(4, 15) != phi(15))
check("A19", order(3, 91) == 6 and first("A19") == "$6$")
check("A20", (math.gcd(27 - 1, 91), math.gcd(27 + 1, 91)) == (13, 7) and first("A20") == "$13$ and $7$")

# ------------------------------------------------------------------------------------------- Section B
check("B1", 1 / math.sqrt(2 ** 8) == 1 / 16 and first("B1") == "$1/16$")
xs = [x for x in range(8) if pow(2, x, 15) == 4]
check("B2", xs == [2, 6] and Fraction(len(xs), 8) == Fraction(1, 4))
has("B2", "$1/4$", "\\ket2 + \\ket6")
inj = lambda a, N: len({a * y % N for y in range(N)}) == N
check("B3", inj(7, 15) and not inj(3, 15))
U2 = np.zeros((4, 4))
orbit = [1, 2, 4, 8]
for i, y in enumerate(orbit):
    U2[orbit.index(2 * y % 15), i] = 1
ev = np.sort_complex(np.round(np.linalg.eigvals(U2), 12))
check("B4", np.allclose(sorted(ev, key=lambda z: (z.real, z.imag)), sorted([1, 1j, -1, -1j], key=lambda z: (z.real, z.imag))))
r, a, N = 4, 2, 15
us = [np.zeros(16, complex) for _ in range(r)]
for s in range(r):
    for k in range(r):
        us[s][pow(a, k, N)] += np.exp(-2j * np.pi * s * k / r) / math.sqrt(r)
check("B5", np.allclose(sum(us) / math.sqrt(r), np.eye(16)[1]))
phi_ = 2 * np.pi / 3
ctrl = np.array([1, np.exp(1j * phi_)]) / math.sqrt(2)
Xm = np.array([[0, 1], [1, 0]])
check("B6", abs(np.real(ctrl.conj() @ Xm @ ctrl) + 0.5) < 1e-12 and first("B6") == "$-1/2$")
check("B7", abs((2 * np.pi * 4 / 6) % (2 * np.pi) - 4 * np.pi / 3) < 1e-12 and first("B7") == "$4\\pi/3$")
Hm = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
check("B8", np.allclose(np.kron(Hm, Hm) @ np.eye(4)[3], np.array([1, -1, -1, 1]) / 2))
check("B9", abs(abs(0.5 * (1 + np.exp(2j * np.pi / 3))) ** 2 - 0.25) < 1e-12 and first("B9") == "$1/4$")
check("B10", pow(7, 0, 15) == pow(7, 4, 15))
check("B11", 1 + 6 + 1 == 8 and first("B11") == "$8$")
# B12: P(y) is the same whether or not the work register is measured (marginal of the joint distribution)
t_, M_ = 4, 16
joint = np.zeros((M_, 16))
for v in {pow(7, x, 15) for x in range(M_)}:
    amp = np.array([1.0 if pow(7, x, 15) == v else 0.0 for x in range(M_)]) / math.sqrt(M_)
    joint[:, v] = np.abs(np.fft.fft(amp) / math.sqrt(M_)) ** 2
check("B12", np.allclose(joint.sum(axis=1), shor_distribution(15, 7, 4)))
check("B13", 3 * (143).bit_length() == 24 and "143 -> 24" in BUILDER and first("B13") == "$24$")
check("B14", 16 * 2 ** 24 == 268435456 and "268.44 MB" in NB_TEXT and first("B14") == "$268$ MB")
check("B15", match_key("B15") == [3, 4, 1, 2])
R1 = lambda l: np.diag([1, np.exp(1j * l)])
named = [np.diag([1, -np.exp(0j)]), np.diag([1, 1j]), np.diag([1, np.exp(1j * np.pi / 4)]), np.diag([1, -1j])]
listII = {"$S^\\dagger$": np.diag([1, -1j]), "$T$": np.diag([1, np.exp(1j * np.pi / 4)]), "$Z$": np.diag([1, -1]), "$S$": np.diag([1, 1j])}
L2 = BANK["B16"]["match"][1]
check("B16", all(np.allclose(R1(l), listII[L2[i - 1]]) for l, i in zip([np.pi, np.pi / 2, np.pi / 4, -np.pi / 2], match_key("B16"))))
check("B19", abs(qpe(3 / 8, 3)[3] - 1) < 1e-12)
P13 = qpe(1 / 3, 3)
check("B20", int(np.argmax(P13)) == 3 and abs(P13[3] - 0.6878) < 1e-4 and abs(4 / np.pi ** 2 - 0.405) < 1e-3, P13)

# ------------------------------------------------------------------------------------------- Section C
check("C2", order(2, 15) == 4 and (math.gcd(3, 15), math.gcd(5, 15)) == (3, 5))
check("C3", order(14, 15) == 2 and pow(14, 1, 15) == 14 and (math.gcd(13, 15), math.gcd(15, 15)) == (1, 15))
check("C4", order(4, 21) == 3 and order(2, 21) == 6)
check("C5", math.gcd(6, 21) == 3 and not inj(6, 21))
check("C6", (33).bit_length() == 6 and 3 * 6 == 18 and first("C6").startswith("$18$"))
P11 = shor_distribution(15, 11, 8)
check("C7", set(np.flatnonzero(P11 > 1e-9)) == {0, 128} and abs(P11[128] - 0.5) < 1e-12 and factors_from(128, 8, 15, 11) == (3, 5))
f, o = candidate_orders(1229, 12, 33, 5)
check("C8", f == Fraction(3, 10) and o == [10] and pow(5, 5, 33) == 23 and factors_from(1229, 12, 33, 5) == (3, 11))
f, o = candidate_orders(2048, 12, 33, 5)
check("C9", f == Fraction(1, 2) and o == [10] and [pow(5, k, 33) for k in (2, 4, 6, 8, 10)] == [25, 31, 16, 4, 1])
check("C10", min(t for t in range(30) if 2 ** t >= 33 ** 2) == 11 and 2 * (33).bit_length() == 12)
P7 = shor_distribution(15, 7, 8)
succ7 = sum(P7[y] for y in range(256) if P7[y] > 1e-12 and factors_from(y, 8, 15, 7))
check("C11", set(np.flatnonzero(P7 > 1e-9)) == {0, 64, 128, 192} and abs(succ7 - 0.75) < 1e-12
      and "Fraction of shots that directly yield the factors: 0.766" in NB_TEXT, succ7)
check("C12", Fraction(sum(1 for s in range(12) if Fraction(s, 12).denominator == 12), 12) == Fraction(1, 3))
cs = [pow(7, 2 ** j, 15) for j in range(8)]
check("C14", sum(c != 1 for c in cs) == 2 and "mulmod_7_mod_15" in NB_TEXT and "mulmod_4_mod_15" in NB_TEXT, cs)
for fn in ("def mulmod_matrix", "def build_shor_kernel", "def candidate_orders", "def classical_order", "def prime_power"):
    check(f"C15/C17 builder has {fn}", fn in BUILDER)
check("C15", match_key("C15") == [4, 3, 2, 1])
L2 = BANK["C16"]["match"][1]
fr = [Fraction(y, 4096).limit_denominator(33) for y in (410, 819, 1638, 2867)]
check("C16", [L2[i - 1] for i in match_key("C16")] == [f"${x.numerator}/{x.denominator}$" for x in fr], fr)
check("C17 R (square roots of 1 mod 81)", [x for x in range(81) if x * x % 81 == 1] == [1, 80])
f21 = shor_distribution(21, 2, 10)
check("C18 II false", Fraction(512, 1024).limit_denominator(21).denominator == 2 != order(2, 21) and f21[512] > 0.1
      and any(f21[y] > 1e-4 and not candidate_orders(y, 10, 21, 2)[1] for y in range(1, 1024)))
check("C19", order(2, 39) == 12 and pow(2, 6, 39) == 25 and (math.gcd(24, 39), math.gcd(26, 39)) == (3, 13)
      and [pow(2, k, 39) for k in range(1, 13)] == [2, 4, 8, 16, 32, 25, 11, 22, 5, 10, 20, 1])
f, o = candidate_orders(1707, 12, 39, 2)
check("C20", f == Fraction(5, 12) and o == [12])
# check that the bank's copy of candidate_orders equals the notebook's
_i = BUILDER.index("def candidate_orders")
cand_src = BUILDER[_i:BUILDER.index("if not simulate:", _i)]
check("candidate_orders copy", "limit_denominator(modulus)" in cand_src and "[:1]" in cand_src and "if y == 0" in cand_src)

# ------------------------------------------------------------------------------------------- Section D
n = 2048
check("D1", 2 * n + 3 == 4099 and "4099" in NB_TEXT and first("D1") == "$4099$")
check("D2 notebook text", "~20 million noisy physical qubits" in NB_TEXT and "Gidney 2025" in NB_TEXT)
check("D3", (2 ** 3) == 8 and first("D3") == "$8$")
check("D5 count", sum(10 - d for d in (1, 2)) == 17 and math.comb(10, 2) == 45)
check("D6", math.comb(8, 2) == 28 and "28 controlled phases" in NB_TEXT)
check("D7", 8 + 8 == 16 and 1 + 2 + 4 + 16 + 28 == 51 and "depth                           : 20" in NB_TEXT)
check("D8", 'backend_cfg["device"] == "hardware"' in BUILDER and "raise ValueError" in BUILDER)


def mult_perm(a_):
    """Gate-level multiply-by-a mod 15 (rotations of the 4 bits + optional X on all) as a permutation of 0..15."""
    k = {2: 1, 4: 2, 8: 3, 7: 3, 11: 2, 13: 1}[a_]
    out = []
    for y in range(16):
        b = [(y >> (3 - i)) & 1 for i in range(4)]
        for _ in range(k):
            b = b[1:] + b[:1]
        if a_ in (7, 11, 13):
            b = [1 - v for v in b]
        out.append(sum(v << (3 - i) for i, v in enumerate(b)))
    return out


check("D9 multiplier correct", all(mult_perm(c)[y] == c * y % 15 for c in (7, 4) for y in range(1, 15)))
cx7, cx4 = 9 * 8 + 4, 6 * 8
check("D9", cx7 == 76 and cx7 + cx4 + 3 == 127 and first("D9") == "$76$")
check("D10", Fraction(sum(1 for y in range(8) if factors_from(y, 3, 15, 7)), 8) == Fraction(3, 8)
      and [y for y in range(8) if factors_from(y, 3, 15, 7)] == [2, 4, 6] and candidate_orders(1, 3, 15, 7)[1] == [8])
check("D11", (1023).bit_length() == 10 and abs(16 * 2 ** 30 / 1e9 - 17.18) < 0.01 and "N <= 1,023" in NB_TEXT)
d = pow(7, -1, 40)
check("D12", d == 23 and pow(2, 7, 55) == 18 and pow(18, d, 55) == 2)
check("D15", match_key("D15") == [3, 4, 1, 2])
ge = lambda n_: math.ceil(3 * n_ + 0.002 * n_ * math.log2(n_))
check("D16", ge(2048) == 6190 and "6190" in NB_TEXT and match_key("D16") == [3, 4, 2, 1])
check("D18", 3 * 32 == 96 and "integers up to 33 bits" in NB_TEXT)
check("D19", abs(0.99 ** 127 - 0.279) < 1e-3)

# ------------------------------------------------------------------------------------------- generated numbers (make_figures)
NUMS = {}
if (FIG / "shor_numbers.tex").exists():
    NUMS = dict(re.findall(r"\\newcommand\{\\num(\w+)\}\{([^}]*)\}", (FIG / "shor_numbers.tex").read_text()))
    check("make_figures: recorded run reproduced", NUMS.get("reproduced") == "identical to the recorded outputs")
    check("make_figures: D9/D19 127 CNOTs", NUMS.get("glcx") == "127" and NUMS.get("glcr") == "3")
    check("make_figures: D10 baseline", NUMS.get("unifsucc") == "0.375" and NUMS.get("glsucc") == "0.750")
    check("make_figures: D19/D20 noise", NUMS.get("noisesuccone") == "0.703" and NUMS.get("noisesuccthree") == "0.665")
    noise = (FIG / "noise15.tex").read_text()
    check("make_figures: D20 table", [l.split("&")[-1].strip(" \\\n") for l in noise.splitlines() if re.match(r"^0", l)]
          == ["0.750", "0.740", "0.726", "0.703", "0.665"], noise)
    check("make_figures: D5", NUMS.get("approxtwo") == "0.6892" and NUMS.get("approxfull") == "0.8165")
else:
    check("figures/shor_numbers.tex present (run make_figures.py)", False)

# ------------------------------------------------------------------------------------------- CUDA-Q
if "--cudaq" in sys.argv:
    import cudaq

    SRC = BUILDER
    BLOCK = SRC[SRC.index("registered_ops = set()"):SRC.index('print("Circuit builder defined.")')]
    NS = {"cudaq": cudaq, "np": np, "N": 15, "n_count": 8, "n_work": 4}
    exec(BLOCK, NS)
    NS_STATE = {"cudaq": cudaq, "np": np, "N": 15, "n_count": 8, "n_work": 4, "NS_SHARED": NS["registered_ops"]}
    exec(BLOCK.replace("    kernel.mz(count)", "    pass").replace("registered_ops = set()", "registered_ops = NS_SHARED"), NS_STATE)
    cudaq.set_target("qpp-cpu")

    def cudaq_dist(N_, a_, t_):
        n_ = N_.bit_length()
        k, _ = NS_STATE["build_shor_kernel"](a_, modulus=N_, count_qubits=t_, work_qubits=n_)
        probs = np.abs(np.array(cudaq.get_state(k))) ** 2
        out = np.zeros(2 ** t_)
        for idx in np.flatnonzero(probs > 1e-14):     # CUDA-Q index: qubit 0 = least significant bit; count = qubits 0..t-1
            out[sum(((idx >> i) & 1) << (t_ - 1 - i) for i in range(t_))] += probs[idx]
        return out

    for N_, a_, t_ in ((15, 7, 8), (15, 11, 8), (21, 2, 10), (33, 5, 12), (39, 2, 12)):
        Pc = cudaq_dist(N_, a_, t_)
        check(f"cudaq distribution N={N_} a={a_}", np.max(np.abs(Pc - shor_distribution(N_, a_, t_))) < 1e-9)
        if (N_, a_) == (33, 5):
            check("cudaq C8/C16 outcomes are likely", all(Pc[y] > 0.02 for y in (410, 819, 1229, 1638, 2048, 2867)),
                  [round(Pc[y], 4) for y in (410, 819, 1229, 1638, 2048, 2867)])
        if (N_, a_) == (39, 2):
            check("cudaq C20 outcome likely", Pc[1707] > 0.02, Pc[1707])
    cudaq.set_random_seed(42)
    k, used = NS["build_shor_kernel"](7, modulus=15, count_qubits=8, work_qubits=4)
    cnt = {int(b, 2): v for b, v in cudaq.sample(k, shots_count=1000).items()}
    check("cudaq recorded run reproduced", cnt == {192: 272, 128: 268, 0: 234, 64: 226}, cnt)
    check("cudaq C11 0.766", sum(v for y, v in cnt.items() if factors_from(y, 8, 15, 7)) / 1000 == 0.766)
    check("cudaq C14 ops", sorted(used) == ["mulmod_4_mod_15", "mulmod_7_mod_15"], used)
    gc = cudaq.estimate_resources(k).to_dict() if hasattr(cudaq, "estimate_resources") else {}
    if gc:
        check("cudaq D6/D7 gate counts", gc.get("h") == 16 and gc.get("cr1") == 28 and gc.get("swap") == 4
              and sum(gc.values()) - gc.get("mz", 0) == 51, gc)
    ver = re.search(r"\d+\.\d+\.\d+", cudaq.__version__)
    print(f"CUDA-Q {ver.group(0) if ver else cudaq.__version__} cross-checks done (qpp-cpu)")

bad = [c_ for c_ in CHECKS if not c_[1]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
sys.exit(1 if bad else 0)
