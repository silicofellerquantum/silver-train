"""Recompute every numerical / structural answer of the QAOA paper and assert it matches the key.

Run:  python verify_answers.py          (NumPy / SciPy only)
      python verify_answers.py --cudaq  (additionally rebuilds the notebook's CUDA-Q kernel and cross-checks it)

Convention (same as the paper and QAOA.ipynb): H_C = sum w/2 Z_i Z_j, C = W/2 - H_C, H_M = sum X_i,
|g,b> = prod_l exp(-i b_l H_M) exp(-i g_l H_C) |+>^n, params = [g_1..g_p, b_1..b_p], node 0 = first (most significant) bit.

Besides recomputing each value, the script checks that the value appears in the text of the CORRECT option of the bank
(so a typo in the paper is caught), and that answer_key.csv matches the generator's layout.
"""
import csv
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import bank_advanced
import bank_basics
import bank_math
import bank_qm

HERE = Path(__file__).resolve().parent
BANK = {q["id"]: q for m in (bank_math, bank_qm, bank_basics, bank_advanced) for q in m.QUESTIONS}
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def correct_text(qid):
    q = BANK[qid]
    return q["opts"][q["correct"]] if q["kind"] in ("ar", "stmt") else q["opts"][0]


def key_has(qid, *fragments):
    """The correct option of qid contains every fragment (LaTeX source text)."""
    t = correct_text(qid)
    missing = [f for f in fragments if f not in t]
    check(qid + " (option text)", not missing, f"{missing} not in {t!r}")


def fixed_index(qid, idx):
    check(qid + " (AR/statement index)", BANK[qid]["correct"] == idx, f"bank says {BANK[qid]['correct']}, expected {idx}")


# ----------------------------------------------------------------------------------------------- reference simulator
def spins(n):
    idx = np.arange(2 ** n)
    return 1 - 2 * ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)


def cut_values(n, edges):
    s = spins(n)
    return sum(w * (1 - s[:, i] * s[:, j]) / 2 for i, j, w in edges) + np.zeros(2 ** n)


def hc_diag(n, edges):
    s = spins(n)
    return sum(w / 2 * s[:, i] * s[:, j] for i, j, w in edges) + np.zeros(2 ** n)


def mixer(psi, n, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * n)
    for q in range(n):
        a, b = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
        psi = np.stack([c * a + s * b, s * a + c * b], axis=q)
    return psi.reshape(-1)


def qaoa_state(n, edges, params):
    p = len(params) // 2
    hc = hc_diag(n, edges)
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for l in range(p):
        psi = mixer(np.exp(-1j * params[l] * hc) * psi, n, params[p + l])
    return psi


def expected_cut(n, edges, params):
    return float(np.abs(qaoa_state(n, edges, params)) ** 2 @ cut_values(n, edges))


def best_value(n, edges, p, starts=30, seed=1):
    rng = np.random.default_rng(seed)
    best = -np.inf
    for _ in range(starts):
        r = minimize(lambda x: -expected_cut(n, edges, x), rng.uniform(-np.pi, np.pi, 2 * p), method="BFGS")
        best = max(best, -r.fun)
    return best


def cmax(n, edges):
    v = cut_values(n, edges)
    return v.max(), int(np.isclose(v, v.max()).sum()), v


ring = lambda n: [(i, (i + 1) % n, 1.0) for i in range(n)]
K = lambda n: [(i, j, 1.0) for i in range(n) for j in range(i + 1, n)]
rng42 = np.random.default_rng(42)
REPO = [(i, j, 1.0) for i in range(15) for j in range(i + 1, 15) if rng42.random() < 0.3]  # notebook's example graph
RAMP = [0.2, 0.6, 0.6, 0.2]
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
RX = lambda t: np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X
RZ = lambda t: np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
expm_pauli = lambda P, phi: np.cos(phi) * np.eye(len(P)) - 1j * np.sin(phi) * P   # valid when P @ P = I
close = lambda a, b, tol=1e-9: np.allclose(a, b, atol=tol)

# ----------------------------------------------------------------------------------------------- Section A
check("A1", 2 ** 15 == 32768 and 2 ** 14 == 16384); key_has("A1", "32\\,768", "16\\,384")
pet = [(a, b, 1.0) for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                                (5, 7), (7, 9), (9, 6), (6, 8), (8, 5)]]
k33 = [(i, j, 1.0) for i in range(3) for j in range(3, 6)]
vals = [cmax(9, ring(9))[0], cmax(5, K(5))[0], cmax(6, k33)[0], cmax(10, pet)[0]]
check("A2", vals == [8, 6, 9, 12], vals)   # List-II (1)12 (2)9 (3)6 (4)8 -> P4 Q3 R2 S1
key_has("A2", r"P$\to$4, Q$\to$3, R$\to$2, S$\to$1")
EA3 = [(0, 1, 3), (1, 2, 1), (2, 3, 2), (3, 0, 4), (0, 2, 2)]
check("A3", math.isclose(cut_values(4, EA3).mean(), 6)); key_has("A3", "6")
for zz in range(8):
    b = [(zz >> 2) & 1, (zz >> 1) & 1, zz & 1]
    poly = b[0] + 2 * b[1] + b[2] - 2 * b[0] * b[1] - 2 * b[1] * b[2]
    check(f"A4 z={zz:03b}", poly == (b[0] != b[1]) + (b[1] != b[2]))
key_has("A4", "z_0 + 2z_1 + z_2 - 2z_0z_1 - 2z_1z_2")
check("A5", math.isclose(105 * 0.3, 31.5) and abs(math.sqrt(105 * 0.21) - 4.70) < 0.005); key_has("A5", "31.5", "4.70")
deg = np.bincount([e[0] for e in REPO] + [e[1] for e in REPO], minlength=15)
check("A6 degrees", deg.tolist() == [2, 3, 5, 4, 5, 4, 6, 1, 4, 3, 4, 2, 4, 6, 7], deg.tolist())
check("A6", len(REPO) == 30 and int((deg % 2).sum()) == 6); key_has("A6", "30", "6")
star7 = [(0, k, 1.0) for k in range(1, 7)]
check("A7 A true", cmax(7, ring(7))[0] == 6)
check("A7 R false (odd-node graph with Cmax=|E|)", cmax(7, star7)[0] == 6 == len(star7)); fixed_index("A7", 2)
fixed_index("A8", 1)   # both false: Max-Cut NP-hard; global min cut polynomial
hk3 = hc_diag(3, K(3))
check("A9", math.isclose(hk3.min(), -0.5) and int(np.isclose(hk3, -0.5).sum()) == 6); key_has("A9", r"-\tfrac12", "six")
check("A10", math.isclose(cut_values(15, REPO).var(), 7.5)); key_has("A10", "7.5")
ZZ = np.kron(Z, Z)
HMa, HCa = np.kron(X, I2) + np.kron(I2, X), 0.5 * ZZ
from scipy.linalg import expm
check("A11 I", close(expm(-1j * 0.7 * (0.5 * np.kron(np.kron(Z, Z), I2) + 0.5 * np.kron(I2, np.kron(Z, Z)))),
                     expm(-1j * 0.35 * np.kron(np.kron(Z, Z), I2)) @ expm(-1j * 0.35 * np.kron(I2, np.kron(Z, Z)))))
check("A11 II false", not close(expm(-1j * 0.4 * HMa) @ expm(-1j * 0.9 * HCa), expm(-1j * (0.4 * HMa + 0.9 * HCa))))
fixed_index("A11", 2)
check("A12", math.ceil((1 / 0.01) ** 2) == 10000); key_has("A12", "10\\,000")
sd = math.sqrt(4000 * 0.06 * 0.94); p10 = 0.94 ** 10; s99 = math.ceil(math.log(0.01) / math.log(0.94))
check("A13", round(sd, 1) == 15.0 and round(p10, 3) == 0.539 and round(1 - p10, 3) == 0.461 and s99 == 75, (sd, p10, s99))
key_has("A13", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1")
mx, cnt, vrep = cmax(15, REPO)
opt = sorted(format(int(i), "015b") for i in np.flatnonzero(np.isclose(vrep, mx)))
check("A14 list", opt == ["001101110100010", "001101110110010", "110010001001101", "110010001011101"], opt)
comp = lambda s: "".join("1" if c == "0" else "0" for c in s)
check("A14", comp("001101110100010") == "110010001011101" and len({min(s, comp(s)) for s in opt}) == 2)
key_has("A14", "$2$", "110010001011101")
for x0 in (0, 1):
    for x1 in (0, 1):
        s0, s1 = 1 - 2 * x0, 1 - 2 * x1
        check(f"A15 x={x0}{x1}", math.isclose(2 * x0 * x1 - x0 - x1, 0.5 * (s0 * s1 - 1)))
key_has("A15", r"\tfrac12\,(s_0s_1 - 1)")
f = lambda t: 0.3 + 1.7 * np.cos(t) - 0.9 * np.sin(t); fp = lambda t: -1.7 * np.sin(t) - 0.9 * np.cos(t)
check("A16", all(math.isclose(0.5 * (f(t + np.pi / 2) - f(t - np.pi / 2)), fp(t), abs_tol=1e-12) for t in np.linspace(-3, 3, 13)))
key_has("A16", r"\tfrac12\big[f(\theta + \tfrac\pi2) - f(\theta - \tfrac\pi2)\big]")
house = [(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 0, 1.), (0, 4, 1.), (1, 4, 1.)]
hm, hc_, hv = cmax(5, house)
check("A17", hm == 5 and hc_ == 4); key_has("A17", "$5$ and $4$")
check("A18", Fraction(hc_, 32) == Fraction(1, 8) and math.isclose(hv.mean() / hm, 0.6)); key_has("A18", r"\tfrac18", "0.6")
check("A19", math.isclose(hc_diag(10, pet).min(), -4.5)); key_has("A19", "-4.5")


def greedy_local_opt_is_global_fails():
    """Find a single-flip local optimum that is not global (A is false), and check c_v >= d_v/2 at local optima (R)."""
    n, E = 15, REPO
    v = cut_values(n, E)
    found_suboptimal, r_holds = False, True
    for zi in range(2 ** n):
        base = v[zi]
        if all(v[zi ^ (1 << (n - 1 - k))] <= base for k in range(n)):        # local optimum
            s = format(zi, f"0{n}b")
            for node in range(n):
                inc = [e for e in E if node in e[:2]]
                c = sum(1 for a, b, _ in inc if s[a] != s[b])
                r_holds &= c >= len(inc) / 2
            found_suboptimal |= base < v.max()
    return found_suboptimal, r_holds


sub, rh = greedy_local_opt_is_global_fails()
check("A20", sub and rh, (sub, rh)); fixed_index("A20", 3)

# ----------------------------------------------------------------------------------------------- Section B
check("B1", abs(4 / 32768 - 1.22e-4) < 5e-7 and round(0.06 / (4 / 32768)) == 492); key_has("B1", "1.22", "490")
check("B2", close(ZZ @ np.kron(X, X), -np.kron(Y, Y)) and close(np.kron(X, X) @ ZZ, -np.kron(Y, Y))); key_has("B2", r"-\,Y\otimes Y")
th, be = 0.83, 0.41
check("B3", close(RX(2 * be), np.array([[np.cos(be), -1j * np.sin(be)], [-1j * np.sin(be), np.cos(be)]]))
      and close(RZ(th), np.diag([np.exp(-1j * th / 2), np.exp(1j * th / 2)]))
      and close(expm(-1j * th * ZZ / 2), np.diag(np.exp(-1j * th / 2 * np.array([1, -1, -1, 1])))))
key_has("B3", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3")
plus = np.array([1, 1]) / np.sqrt(2)
check("B4", all(math.isclose(abs((H @ RZ(t) @ plus)[0]) ** 2, np.cos(t / 2) ** 2) for t in np.linspace(0, 6, 9)))
key_has("B4", r"\cos^2(\theta/2)")
psiB5 = np.array([0.5, np.sqrt(3) / 2, 0, 0])
check("B5", math.isclose(np.real(psiB5 @ ((np.eye(4) - ZZ) / 2) @ psiB5), 0.75)); key_has("B5", r"\tfrac34")
comm = lambda A, B: close(A @ B, B @ A)
check("B6", comm(np.kron(X, X), ZZ) and not comm(np.kron(X, I2), ZZ) and not comm(np.kron(Y, Z), ZZ) and not comm(np.kron(I2, X), ZZ))
key_has("B6", "X_0X_1")
cj = lambda P: CNOT @ P @ CNOT
check("B7", close(cj(np.kron(X, I2)), np.kron(X, X)) and close(cj(np.kron(I2, X)), np.kron(I2, X))
      and close(cj(np.kron(Z, I2)), np.kron(Z, I2)) and close(cj(np.kron(I2, Z)), ZZ))
key_has("B7", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1")
conc = lambda g: 2 * abs(np.linalg.det((expm(-1j * g * ZZ / 2) @ np.kron(plus, plus)).reshape(2, 2)))
check("B8", math.isclose(conc(np.pi / 2), 1) and conc(np.pi / 4) < 0.75); key_has("B8", r"\pi/2")
EB = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
pA = qaoa_state(3, EB, [0.5, 0.3])
pS = mixer(np.exp(-1j * 0.5 * (hc_diag(3, EB) + 2.7)) * np.full(8, 8 ** -0.5, complex), 3, 0.3)   # H_C + 2.7 I
check("B9 I", math.isclose(abs(np.vdot(pA, pS)), 1) and not close(pA, pS))
# with e^{-i g C}: equals notebook circuit with -g  -> the optimal angles differ in sign
psiC = np.full(8, 8 ** -0.5, complex)
psiC = mixer(np.exp(-1j * 0.5 * cut_values(3, EB)) * psiC, 3, 0.3)
check("B9 II false", math.isclose(abs(np.vdot(psiC, qaoa_state(3, EB, [-0.5, 0.3]))), 1)
      and not math.isclose(abs(np.vdot(psiC, qaoa_state(3, EB, [0.5, 0.3]))), 1))
fixed_index("B9", 2)
check("B10", math.isclose((0.8 - 0.2) * (0.3 - 0.7), -0.24)); key_has("B10", "-0.24")
check("B11", close(expm(1j * be * X) @ Z @ expm(-1j * be * X), np.cos(2 * be) * Z + np.sin(2 * be) * Y)); key_has("B11", r"\cos 2\beta\; Z + \sin 2\beta\; Y")
check("B12", 16 * 2 ** 32 <= 80e9 < 16 * 2 ** 33 and 8 * 2 ** 33 <= 80e9 < 8 * 2 ** 34); key_has("B12", "$32$", "$33$")
check("B13", math.isclose(abs((RX(np.pi / 4) @ np.array([1, 0]))[1]) ** 2, (2 - np.sqrt(2)) / 4)); key_has("B13", r"2 - \sqrt2")
def on(P, k, n):
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, P if q == k else I2)
    return out


HM3 = sum(on(X, k, 3) for k in range(3))
p3 = np.kron(np.kron(plus, plus), plus)
check("B14", close(HM3 @ p3, 3 * p3) and math.isclose(np.linalg.eigvalsh(HM3).max(), 3)); key_has("B14", "largest")
check("B15", math.isclose(expected_cut(15, REPO, RAMP), expected_cut(15, REPO, [-x for x in RAMP]))); fixed_index("B15", 1)
check("B16", math.isclose(1.5 - 0.5 * (-0.6 - 0.2 + 0.4), 1.7)); key_has("B16", "1.7")
check("B17 op", close(expm(-1j * np.pi / 2 * HM3), (-1j) ** 3 * np.kron(np.kron(X, X), X)))
check("B17 period", all(math.isclose(expected_cut(15, REPO, [0.2, 0.6, 0.6 + k * np.pi / 2, 0.2]), expected_cut(15, REPO, RAMP))
                        for k in (1, 2)))
key_has("B17", r"(-i)^n X^{\otimes n}", r"\pi/2")
check("B18", all((ZZ[k, k] == -1) == (k in (1, 2)) for k in range(4))); fixed_index("B18", 0)
st = np.exp(-1j * np.pi / 2 * hc_diag(2, [(0, 1, 1.0)])) * np.full(4, 0.5)
check("B19", math.isclose(np.degrees(np.angle(st[1] / st[0])), 90)); key_has("B19", r"90^\circ")
fin = np.abs(mixer(st, 2, np.pi / 8)) ** 2
check("B20", math.isclose(fin[1] + fin[2], 0, abs_tol=1e-12) and math.isclose(np.abs(mixer(st, 2, -np.pi / 8))[1:3].__pow__(2).sum(), 1))
key_has("B20", "$0$")

# ----------------------------------------------------------------------------------------------- Section C
EC1 = [(0, 1, 2.0), (1, 2, 1.0)]
check("C1", close(hc_diag(3, EC1), 1.0 * spins(3)[:, 0] * spins(3)[:, 1] + 0.5 * spins(3)[:, 1] * spins(3)[:, 2])); key_has("C1", r"1.0\,Z_0Z_1 + 0.5\,Z_1Z_2")
check("C2", all(math.isclose(expected_cut(15, REPO, [0, 0, b1, b2]), 15) for b1, b2 in [(0.3, 1.1), (-0.7, 0.2)])); key_has("C2", "W/2")
g, w = 0.37, 1.8
check("C3", close(CNOT @ np.kron(I2, RZ(g * w)) @ CNOT, expm(-1j * g * (w / 2) * ZZ))); key_has("C3", r"e^{-i\gamma\,(w/2)\,Z_iZ_j}")
check("C4", close(RX(2 * be), expm(-1j * be * X))); key_has("C4", r"R_X(2\beta) = e^{-i\beta X}")
gates = lambda n, m, p: n + p * (3 * m + n)
check("C5", gates(20, 57, 3) == 593 and 2 * 3 * 57 == 342 and gates(15, 30, 2) == 225); key_has("C5", "593", "342")
check("C6", (2 * 2, 2 * 30, 15 + 60 + 30, 2 * 2 * 30) == (4, 60, 105, 120)); key_has("C6", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
e1 = [(0, 1, 1.0)]
check("C7 formula", all(math.isclose(expected_cut(2, e1, [gg, bb]), 0.5 - 0.5 * np.sin(4 * bb) * np.sin(gg))
                        for gg, bb in [(0.3, 0.2), (1.1, -0.7), (2.5, 0.9)]))
check("C7", math.isclose(expected_cut(2, e1, [np.pi / 2, -np.pi / 8]), 1) and math.isclose(expected_cut(2, e1, [np.pi / 2, np.pi / 8]), 0, abs_tol=1e-12))
key_has("C7", "$1$", r"(\pi/2,\, -\pi/8)")
check("C8", math.isclose(best_value(8, ring(8), 1, starts=10) / 8, 0.75, abs_tol=1e-6)); key_has("C8", "3/4")
fixed_index("C9", 0)
check("C9 II (nesting)", math.isclose(expected_cut(15, REPO, [0.3, 0.0, -0.2, 0.0]), expected_cut(15, REPO, [0.3, -0.2])))
check("C11", ["a", "b", "c", "d", "e", "f"][3 + 1] == "e"); key_has("C11", "$e$")
gp, bp = np.linspace(0.2, 0.6, 5), np.linspace(0.6, 0.2, 5)
check("C12", math.isclose(gp[2], 0.4) and math.isclose(bp[1], 0.5)); key_has("C12", "$0.4$ and $0.5$")
fixed_index("C13", 0)
c_ramp = expected_cut(15, REPO, RAMP)
check("C14", abs(c_ramp - 9.18) < 0.005 and c_ramp < 15, c_ramp)
check("C14 small-angle sign", expected_cut(2, e1, [0.1, -0.05]) > 0.5 > expected_cut(2, e1, [0.1, 0.05]))
key_has("C14", "opposite sign")
g16 = math.atan(1 / math.sqrt(2))
check("C16", math.isclose(expected_cut(3, K(3), [g16, -g16 / 2]), 2)); key_has("C16", "$2$", "r = 1")
check("C17", math.isclose(20.5336 / 25, 0.8213, abs_tol=5e-5) and round(0.06 * 4000) == 240); key_has("C17", "240")
fixed_index("C18", 2)
path = [(0, 1, 1.0), (1, 2, 1.0)]
pp = np.abs(qaoa_state(3, path, [np.pi / 2, -np.pi / 8])) ** 2
s3 = spins(3)
check("C19", math.isclose(pp @ (s3[:, 0] * s3[:, 1]), -0.5)); key_has("C19", r"-\tfrac12")
check("C20", math.isclose(pp @ cut_values(3, path), 1.5) and math.isclose(pp[0b010] + pp[0b101], 0.625)); key_has("C20", "1.5", "0.625")
check("passage C formula", all(math.isclose(expected_cut(3, path, [gg, bb]),
                                            2 * (0.5 - 0.25 * np.sin(4 * bb) * np.sin(gg) * (1 + np.cos(gg)))) for gg, bb in [(0.9, -0.3), (2.1, 0.4)]))

# ----------------------------------------------------------------------------------------------- Section D


def light_cone_size(n, p, edge=(0, 1)):
    sup = set(edge)
    for _ in range(p):
        sup |= {v for a, b, _ in ring(n) if a in sup or b in sup for v in (a, b)}
    return len(sup)


check("D1", light_cone_size(40, 3) == 8); key_has("D1", "$8$")
r2 = [best_value(n, ring(n), 2, starts=25) / n for n in (6, 8, 10)]
check("D2", all(abs(r - 5 / 6) < 1e-6 for r in r2), r2); key_has("D2", "5/6")
check("D3", 2 * 2 * (30 + 15) == 180 and 2 * 2 + 1 == 5); key_has("D3", "$180$ and $5$")
key_has("D4", "trust region")
check("D5", abs(2.5 / math.sqrt(1000) - 0.079) < 5e-4); key_has("D5", "0.079")
check("D6", round(2 * 8 * 2 ** 22 / 1e6) == 67); key_has("D6", "67")
need = 16 * 2 ** 36
check("D7", min(2 ** k for k in range(10) if need / 2 ** k <= 80e9) == 16); key_has("D7", "16")
key_has("D8", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
key_has("D9", "ring"); fixed_index("D10", 0)
F = 0.995 ** 120
check("D11", round(F * 20.5336 + (1 - F) * 15, 1) == 18.0); key_has("D11", "18.0")
key_has("D12", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
g2 = [0.4, 0.8]; ext = [0] + g2 + [0]
interp = [(i - 1) / 2 * ext[i - 1] + (2 - i + 1) / 2 * ext[i] for i in range(1, 4)]
check("D13", np.allclose(interp, [0.4, 0.6, 0.8])); key_has("D13", "0.4", "0.6", "0.8")
fixed_index("D14", 3)
check("D15", abs(40.0134 / 80 - 0.50) < 0.01 and abs(40.0134 / 80 / 225 - 2.2e-3) < 1e-4); key_has("D15", "0.50", "2.2")
fac = gates(25, 90, 2) / gates(15, 30, 2) * 2 ** 10
check("D16", abs(fac - 2.8e3) < 50, fac); key_has("D16", "2.8")
ED18 = [(0, 1, 1.0), (1, 2, 2.5), (0, 2, -1.0)]
check("D18", math.isclose(cmax(3, ED18)[0], 3.5) and math.isclose(sum(e[2] for e in ED18) / 2, 1.25)); key_has("D18", "3.5", "1.25")
check("D19", comp("110010001001101") == "001101110110010" and abs((68 + 67) / 4000 - 0.0338) < 1e-4); key_has("D19", "0.0338")
check("D20", abs((20.5336 - 15) / 10 - 0.553) < 5e-4); key_has("D20", "0.553")

# ----------------------------------------------------------------------------------------------- answer key file
try:
    import generate_papers
    qs, _, _ = generate_papers.load_questions()
    with open(HERE / "answer_key.csv") as fh:
        rows = list(csv.DictReader(fh))
    check("answer_key.csv", [(r["bank_id"], r["answer"]) for r in rows] == [(q["id"], "ABCD"[q["answer"]]) for q in qs])
    check("80 questions, 4 options each", len(qs) == 80 and all(len(q["opts"]) == 4 for q in qs))
    check("distinct options", all(len(set(q["opts"])) == 4 for q in qs))
except Exception as exc:  # pragma: no cover
    check("answer_key.csv", False, repr(exc))

# ----------------------------------------------------------------------------------------------- CUDA-Q cross-check
if "--cudaq" in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    import cudaq
    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def qaoa_state_k(n: int, edge_i: list[int], edge_j: list[int], weights: list[float], layers: int, params: list[float]):
        """Identical to the notebook's qaoa_state kernel."""
        qubits = cudaq.qvector(n)
        h(qubits)
        for layer in range(layers):
            gamma = params[layer]
            beta = params[layers + layer]
            for e in range(len(edge_i)):
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
                rz(gamma * weights[e], qubits[edge_j[e]])
                x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
            for q in range(n):
                rx(2.0 * beta, qubits[q])

    def cudaq_state(n, edges, params):
        p = len(params) // 2
        st_ = np.array(cudaq.get_state(qaoa_state_k, n, [e[0] for e in edges], [e[1] for e in edges],
                                       [float(e[2]) for e in edges], p, [float(v) for v in params]))
        return st_.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)   # CUDA-Q: qubit 0 = least significant

    def cudaq_cut(n, edges, params):
        hc = None
        for i, j, w in edges:
            t = 0.5 * w * cudaq.spin.z(i) * cudaq.spin.z(j)
            hc = t if hc is None else hc + t
        p = len(params) // 2
        e = cudaq.observe(qaoa_state_k, hc, n, [e[0] for e in edges], [e[1] for e in edges],
                          [float(e[2]) for e in edges], p, [float(v) for v in params]).expectation()
        return sum(e_[2] for e_ in edges) / 2 - e

    rng = np.random.default_rng(3)
    for name, n, E in [("ring8", 8, ring(8)), ("K3", 3, K(3)), ("repo15", 15, REPO), ("custom", 3, ED18)]:
        prm = list(rng.uniform(-1, 1, 4))
        fid = abs(np.vdot(cudaq_state(n, E, prm), qaoa_state(n, E, prm))) ** 2
        check(f"cudaq state {name}", fid > 1 - 1e-10, fid)
    check("cudaq C14 ramp start", abs(cudaq_cut(15, REPO, RAMP) - c_ramp) < 1e-8)
    check("cudaq B20", abs(cudaq_cut(2, e1, [np.pi / 2, np.pi / 8])) < 1e-9 and abs(cudaq_cut(2, e1, [np.pi / 2, -np.pi / 8]) - 1) < 1e-9)
    check("cudaq C16", abs(cudaq_cut(3, K(3), [g16, -g16 / 2]) - 2) < 1e-9)
    pc = np.abs(cudaq_state(3, path, [np.pi / 2, -np.pi / 8])) ** 2
    check("cudaq C20", abs(pc[0b010] + pc[0b101] - 0.625) < 1e-9)
    check("cudaq D18 sign of negative weight", abs(cudaq_cut(3, ED18, [0.4, -0.3]) - expected_cut(3, ED18, [0.4, -0.3])) < 1e-9)
    import re
    ver = re.search(r"\d+\.\d+\.\d+", cudaq.__version__)
    print(f"CUDA-Q {ver.group(0) if ver else cudaq.__version__} cross-checks done (target qpp-cpu)")

bad = [c for c in CHECKS if not c[1]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
sys.exit(1 if bad else 0)
