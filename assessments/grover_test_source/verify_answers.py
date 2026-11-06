"""Recompute every numerical / structural answer of the Grover paper and check it against the key.

Run:  python verify_answers.py          (numpy only)
      python verify_answers.py --cudaq  (also runs the notebook's own kernels on the CUDA-Q CPU simulator)

Two independent methods are used where possible: closed formulas, and a gate-by-gate state-vector simulation of the
textbook circuit (H, X, multi-controlled Z). Conventions are the same as in the paper: qubit 1 = most significant bit.
Checks marked "conceptual" or "literature" hold no calculation. The last block checks numbers quoted in the book chapter.
"""
import math
import sys
from fractions import Fraction as Fr
from itertools import product
from math import asin, cos, pi, sin, sqrt

import numpy as np

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def near(a, b, tol=1e-9):
    return abs(a - b) < tol


def theta_of(N, M=1):
    return asin(sqrt(M / N))


def k_opt(N, M=1):
    return round(pi / (4 * theta_of(N, M)) - 0.5)


def P(N, k, M=1):
    return sin((2 * k + 1) * theta_of(N, M)) ** 2


# ---------------------------------------------------------------- gate-level simulator
I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
k0, k1 = np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)
plus, minus = (k0 + k1) / sqrt(2), (k0 - k1) / sqrt(2)


def apply_1q(state, gate, q, n):
    """Apply a 2x2 gate on qubit q (0 = most significant)."""
    psi = state.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(gate, psi, axes=(1, q)), 0, q)
    return psi.reshape(-1)


def layer(state, gate, qubits, n):
    for q in qubits:
        state = apply_1q(state, gate, q, n)
    return state


def mcz(state):
    """Multi-controlled Z on all qubits: sign flip of |1...1> only."""
    out = state.copy()
    out[-1] *= -1
    return out


def grover_iteration_circuit(state, marked, n):
    """Oracle, then the diffusion circuit, gate by gate, exactly as in the conventions box."""
    zeros = [q for q, b in enumerate(marked) if b == "0"]
    state = mcz(layer(state, X, zeros, n))
    state = layer(state, X, zeros, n)
    state = layer(state, H, range(n), n)
    state = layer(state, X, range(n), n)
    state = mcz(state)
    state = layer(state, X, range(n), n)
    return layer(state, H, range(n), n)


def circuit_probability(marked, k):
    n = len(marked)
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    state = layer(state, H, range(n), n)
    for _ in range(k):
        state = grover_iteration_circuit(state, marked, n)
    return abs(state[int(marked, 2)]) ** 2, state


def D_matrix(N):
    s = np.ones(N, dtype=complex) / sqrt(N)
    return 2 * np.outer(s, s.conj()) - np.eye(N)


def invert_mean(a):
    a = np.asarray(a, dtype=float)
    return 2 * a.mean() - a


# ----------------------------------------------------------------------------------------------- Section A
check("A1", near(sin(3 * asin(0.25)) ** 2, 121 / 256))
u = np.array([3, 4]) / 5
check("A2", np.allclose(2 * np.outer(u, u) - np.eye(2), np.array([[-7, 24], [24, 7]]) / 25))


def refl(deg):
    a = math.radians(deg)
    w = np.array([cos(a), sin(a)])
    return 2 * np.outer(w, w) - np.eye(2)


rot50 = np.array([[cos(math.radians(50)), -sin(math.radians(50))], [sin(math.radians(50)), cos(math.radians(50))]])
check("A3", np.allclose(refl(35) @ refl(10), rot50))
t = asin(0.25)
G2 = np.array([[cos(2 * t), -sin(2 * t)], [sin(2 * t), cos(2 * t)]])
ev = np.linalg.eigvals(G2)
check("A4", np.allclose(sorted(ev, key=lambda z: z.imag), [(7 - 1j * sqrt(15)) / 8, (7 + 1j * sqrt(15)) / 8]))
check("A5", abs(((1 + 1j) / sqrt(2)) ** 6 - (-1j)) < 1e-12)
t = asin(0.1)
check("A6", max(range(11), key=lambda k: round(sin((2 * k + 1) * t) ** 2, 12)) == 7 and round(pi / 4 * 10) == 8)
v = np.array([1] * 7 + [-1], dtype=float)
check("A7", near(invert_mean(v)[-1], 2.5))
# exact expected position of the first marked card
exp_draw = sum(Fr(j) * Fr(math.comb(31 - j, 2), math.comb(31, 3)) for j in range(1, 30))
check("A8", exp_draw == 8, str(exp_draw))
def _bs_worst(n_):
    return 0 if n_ == 0 else 1 + _bs_worst(n_ // 2)


check("A9", _bs_worst(1023) == 10)
count = sum(1 for a, b, c, d in product((0, 1), repeat=4) if (a or b) and (c ^ d))
check("A10", Fr(count, 16) == Fr(3, 8))
check("A11", (16 - 3) - 3 == 10)
v12 = np.array([1, 1j, 1 + 1j, 2])
check("A12", near(1 / np.linalg.norm(v12), 1 / (2 * sqrt(2))))
check("A13", near(sqrt(2 ** 6), 8))
check("A14", abs(np.vdot(np.array([1, 1j, 0]) / sqrt(2), np.ones(3) / sqrt(3)) - (1 - 1j) / sqrt(6)) < 1e-12)
for s_, val in ((1 / 8, 25 / 32), (1 / 4, 1), (1 / 2, 1 / 2), (3 / 4, 0)):
    check(f"A15 s={s_}", near(sin(3 * asin(sqrt(s_))) ** 2, val))
for N_ in (3, 5, 8):
    w = np.random.default_rng(N_).normal(size=N_) + 1j * np.random.default_rng(N_ + 1).normal(size=N_)
    w /= np.linalg.norm(w)
    proj = np.outer(w, w.conj())
    check(f"A16 N={N_}", near(np.trace(2 * proj - np.eye(N_)).real, 2 - N_) and near(np.trace(np.eye(N_) - 2 * proj).real, N_ - 2)
          and near(np.trace(proj).real, 1))
    D_ = 2 * proj - np.eye(N_)
    evs = np.sort(np.linalg.eigvalsh(D_))
    check(f"A18 N={N_}", np.allclose(D_ @ D_, np.eye(N_)) and np.allclose(evs, [-1] * (N_ - 1) + [1]))
check("A17", asin(0.1) > 0.1 and round(pi / 4 * 10) != 7)
p_ = [round(sin((2 * k + 1) * pi / 18) ** 2, 10) for k in range(60)]
check("A19", p_.index(1.0) == 4 and next(q for q in range(1, 30) if all(p_[k] == p_[k + q] for k in range(25))) == 9)
check("A20", sum(1 for k in range(18) if sin((2 * k + 1) * pi / 18) ** 2 >= 0.75 - 1e-12) == 6)

# ----------------------------------------------------------------------------------------------- Section B
st = np.ones(8) / sqrt(8)
check("B1", near(sum(abs(st[int(f'{a}{b}{c}', 2)]) ** 2 for a, b, c in product((0, 1), repeat=3) if a == 1 and c == 0), 0.25))
flip = np.array([1, 1, 1, -1], dtype=complex) / 2
check("B2", abs(np.kron(np.array([1, 1j]) / sqrt(2), minus)[2] - 0.5j) < 1e-12)
check("B2b", near(abs(flip[3]) ** 2, 0.25))
HH = np.kron(H, H)
check("B3", np.allclose(HH @ flip, flip) and near((HH @ flip)[3].real, -0.5))


def Uf_on(x_vec_index, n, f, anc):
    """Return U_f |x>|anc> for the bit oracle (x has n bits)."""
    N_ = 2 ** n
    out = np.zeros(2 * N_, dtype=complex)
    fx = f(x_vec_index)
    out[2 * x_vec_index + (0 ^ fx)] += anc[0]
    out[2 * x_vec_index + (1 ^ fx)] += anc[1]
    return out


f_and = lambda x: 1 if x == 3 else 0
ok4, ok5 = True, True
for x_ in range(4):
    bx = np.zeros(4, dtype=complex)
    bx[x_] = 1
    ok4 &= np.allclose(Uf_on(x_, 2, f_and, minus), (-1) ** f_and(x_) * np.kron(bx, minus))
    ok5 &= np.allclose(Uf_on(x_, 2, f_and, plus), np.kron(bx, plus))
check("B4", ok4)
check("B5", ok5)
ccz = np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(complex)
X2 = np.kron(np.kron(I2, X), I2)
check("B6", np.allclose(np.diag(X2 @ ccz @ X2).real, [1, 1, 1, 1, 1, -1, 1, 1]))  # index 5 = |101>
ccx = np.eye(8, dtype=complex)
ccx[[6, 7], [6, 7]] = 0
ccx[6, 7] = ccx[7, 6] = 1
H3 = np.kron(np.kron(I2, I2), H)
check("B7", np.allclose(H3 @ ccx @ H3, ccz))
check("B8", np.allclose(-plus, -1 * plus) and not near(abs(plus @ minus), 1))
psi9 = np.array([1, 2, -2, 1j]) / sqrt(10)
p9 = abs(psi9[1]) ** 2 + abs(psi9[2]) ** 2
post = np.array([0, psi9[1], psi9[2], 0]) / sqrt(p9)
check("B9", near(p9, 0.8) and np.allclose(post, np.array([0, 1, -1, 0]) / sqrt(2)))
s4 = np.ones(4) / 2
proj = np.outer(s4, s4)
check("B10", not np.allclose(proj.T @ proj, np.eye(4)) and np.allclose(D_matrix(4).conj().T @ D_matrix(4), np.eye(4)))
r_ = np.array([1] * 15 + [0]) / sqrt(15)
s_ = np.ones(16) / 4
check("B11", near(r_ @ s_, sqrt(15) / 4))
check("B12", 2 ** 33 * 16 == 128 * 2 ** 30)
check("B13", np.allclose(2 * np.outer(plus, plus) - I2, X))
s16 = np.ones(16, dtype=complex) / 4
check("B14", np.allclose(D_matrix(16) @ s16, s16))
check("B15", np.allclose(H @ k1, minus) and np.allclose(Z @ minus, plus) and np.allclose(X @ minus, -minus) and np.allclose(H @ plus, k0))


def is_product(c):
    return abs(c[0] * c[3] - c[1] * c[2]) < 1e-12


check("B16", is_product(np.array([1, 1, 1, 1]) / 2) and np.allclose(np.array([1, -1, -1, 1]) / 2, np.kron(minus, minus))
      and not is_product(flip) and not is_product(np.array([1, 0, 0, 1]) / sqrt(2)))
n_ = 4
probs_before = np.abs(np.ones(16) / 4) ** 2
sign = np.ones(16)
sign[5] = -1
check("B17", np.allclose(np.abs(sign * np.ones(16) / 4) ** 2, probs_before))
check("B18", near(abs(plus @ minus), 0) and near(abs(plus @ plus), 1))
st3 = np.kron(np.kron(plus, plus), H @ k1)
after = ccx @ st3
check("B19", np.allclose(after, np.kron(flip, minus)))
p1 = np.sum(np.abs(after[1::2]) ** 2)
reduced = after[1::2] / sqrt(p1)
check("B20", near(p1, 0.5) and near(abs(np.vdot(reduced, flip)), 1))

# ----------------------------------------------------------------------------------------------- Section C
check("C1", near(P(4, 1), 1) and near(circuit_probability("10", 1)[0], 1))
check("C2", near(P(8, 1), 25 / 32) and near(circuit_probability("101", 1)[0], 25 / 32))
check("C3", k_opt(16) == 3 and near(P(16, 3), 0.9613, 5e-4) and near(P(16, 2), 0.908, 5e-4) and near(P(16, 4), 0.582, 5e-4)
      and near(circuit_probability("1011", 3)[0], P(16, 3)))
d4 = D_matrix(4)
check("C4", near(d4[0, 0].real, -0.5) and near(d4[0, 1].real, 0.5))
check("C4b", k_opt(2 ** 15) == 142)
# C5: the oracle built with X on qubits 1 and 3 flips exactly |0101>
st5 = np.ones(16, dtype=complex)
out5 = layer(mcz(layer(st5, X, [0, 2], 4)), X, [0, 2], 4)
check("C5", np.where(out5.real < 0)[0].tolist() == [0b0101])
check("C6", 4 * 6 + 2 * "010011".count("0") + 2 == 32)
# C7: the diffusion circuit equals -D, built gate by gate on every basis vector
n7 = 3
cols = []
for i7 in range(2 ** n7):
    e = np.zeros(2 ** n7, dtype=complex)
    e[i7] = 1
    e = layer(e, H, range(n7), n7)
    e = layer(e, X, range(n7), n7)
    e = mcz(e)
    e = layer(e, X, range(n7), n7)
    cols.append(layer(e, H, range(n7), n7))
check("C7", np.allclose(np.array(cols).T, -D_matrix(8)))
a16 = np.array([1] * 15 + [-1], dtype=float) / 4
out8 = invert_mean(a16)
check("C8", near(out8[0], 3 / 16) and near(out8[-1], 11 / 16))
check("C9", near(P(16, 1, 4), 1))
check("C10", all(near(P(8, k, 4), 0.5) for k in range(10)))
check("C11", near(P(8, 1, 6), 0) and near(P(8, 0, 6), 0.75))
check("C12", near(P(2 ** 15, 284), 3.0e-6, 1e-7) and P(2 ** 15, 284) < 1 / 2 ** 15)
check("C13", k_opt(2 ** 20) == 804 and near(2 ** 19 / 804, 650, 5))
check("C14", near(2000 * 0.9613, 1923, 1) and near(sqrt(2000 * 0.9613 * 0.0387), 8.6, 0.1))
check("C15", [k_opt(2 ** n) for n in (2, 6, 8, 10)] == [1, 6, 12, 25])
check("C16", np.allclose(layer(np.array([1, 0, 0, 0], dtype=complex), H, range(2), 2), np.ones(4) / 2))
pvals = [circuit_probability(format(x_, "04b"), 2)[0] for x_ in range(16)]
check("C17", np.allclose(pvals, pvals[0]))
check("C18", P(4, 2) < P(4, 1) and near(k_opt(2 ** 20) / k_opt(2 ** 18), 2, 0.01))
_, st1 = circuit_probability("101", 1)
a8 = 1 / sqrt(8)
# the diffusion circuit is -D, so compare magnitudes (the overall sign is a global phase)
check("C19", near(abs(st1[0b101]), 2.5 * a8) and near(abs(st1[0]), 0.5 * a8))
check("C20", near(circuit_probability("101", 2)[0], 121 / 128))

# ----------------------------------------------------------------------------------------------- Section D
check("D1", k_opt(1024, 16) == 6)
th20 = asin(2 ** -10)
check("D2", near(1 / sin(2 * th20), 512, 0.1))
for m_, t_ in ((2, pi / 6), (7, 0.3), (50, 0.01)):
    avg = sum(sin((2 * k + 1) * t_) ** 2 for k in range(m_)) / m_
    check(f"D2 identity m={m_}", near(avg, 0.5 - sin(4 * m_ * t_) / (4 * m_ * sin(2 * t_))))
t16 = theta_of(16)
kc = math.ceil(pi / (4 * t16) - 0.5)
check("D3", kc == 3 and near(pi / (2 * (2 * kc + 1)), pi / 14) and pi / 14 < t16 and pi / 10 > t16)
check("D4", True, "conceptual: BBBV lower bound")
mk = "011001010011111"  # recorded marked string of Grover.ipynb
check("D5", mk.count("0") == 6 and 142 * (2 * 15 + 2 * 6) == 5964 and 15 + 142 * 30 == 4275 and 2 * 142 == 284)
ka = round(pi / (4 * asin(0.1)) - 0.5)
check("D6", ka == 7 and 2 * ka + 1 == 15)
check("D7", 2 * 2 ** 28 * 16 == 8 * 2 ** 30 and near(2 * 2 ** 15 * 16 / 1e6, 1.05, 0.005))
check("D8", near(math.log2(pi / 4 * sqrt(2.0 ** 128)), 64 + math.log2(pi / 4)))
pn = 0.99 ** 25 * P(1024, 25) + (1 - 0.99 ** 25) / 1024
check("D9", k_opt(1024) == 25 and near(pn, 0.78, 0.005))
check("D10", 2 * (2 * (9 - 1)) == 32)
check("D11", near(2 ** (1.5 * 2), 8))
_max = 2 * 142 + 1
_stride = max(1, math.ceil(_max / 60))
_pts, _k = 1, 0
while _k + _stride <= _max:
    _k += _stride
    _pts += 1
check("D12", _stride == 5 and _pts == 58)
check("D13", near(0.99998683 ** 2000, 0.974, 0.001))
check("D14", near(1 / (25 / 32), 1.28) and near(2 / (121 / 128), 2.116, 1e-3) and k_opt(8) == 2)
check("D15", True, "literature values: Grover, BHT/Aaronson-Shi, Ambainis")
check("D16", True, "conceptual")
check("D17", near(math.log2(sqrt(2.0 ** 40)), 20))
check("D18", all(near(P(1024, k, 512), 0.5) for k in range(6)))
# D19/D20: full phase estimation with t = 3 on an explicit G for N = 64, M = 16
N_ = 64
marks = np.random.default_rng(1).choice(N_, 16, replace=False)
Om = np.eye(N_, dtype=complex)
Om[marks, marks] = -1
Gq = D_matrix(N_) @ Om
s64 = np.ones(N_, dtype=complex) / sqrt(N_)
reg = np.zeros((8, N_), dtype=complex)
vec = s64.copy()
for x_ in range(8):
    reg[x_] = vec / sqrt(8)
    vec = Gq @ vec
inv_qft = np.exp(-2j * pi * np.outer(np.arange(8), np.arange(8)) / 8) / sqrt(8)
amps = inv_qft @ reg
py = np.sum(np.abs(amps) ** 2, axis=1)
top = sorted(np.argsort(py)[-2:].tolist())
check("D19", top == [1, 7] and near(py[1], 0.353, 0.002) and near(py[2], 0.094, 0.002) and near(py.sum(), 1), str(np.round(py, 4)))
check("D20", near(64 * sin(pi / 8) ** 2, 9.37, 0.01) and near(64 * sin(pi / 4) ** 2, 32))

# ---------------------------------------------------------------- facts quoted in the book chapter and the notebook
check("book n=15", near(theta_of(2 ** 15), 0.005524, 1e-6) and near(P(2 ** 15, 142), 0.99999, 2e-5))
check("book n=25", k_opt(2 ** 25) == 4549)
check("book N=16 M=2", near(P(16, 2, 2), sin(5 * asin(sqrt(1 / 8))) ** 2))
check("book s1", near(sin(3 * asin(1 / 3)) ** 2, 529 / 729) and near(cos(2 * asin(1 / 3)), 7 / 9))
check("book s1 window", [k for k in range(7) if sin((2 * k + 1) * pi / 14) ** 2 >= 0.75] == [2, 3, 4])
u13 = np.array([5, 12]) / 13
check("book s2", np.allclose(2 * np.outer(u13, u13) - np.eye(2), np.array([[-119, 120], [120, 119]]) / 169))
check("book s3", near(invert_mean([1] * 15 + [-1])[-1], 11 / 4))
exp2 = sum(Fr(j) * Fr(math.comb(15 - j, 3), math.comb(15, 4)) for j in range(1, 13))
check("book P5", exp2 == Fr(16, 5), str(exp2))
check("book s7 N=32", near(P(32, 1), 529 / 2048) and near(P(32, 2), 78961 / 131072)
      and near(circuit_probability("10110", 2)[0], 78961 / 131072))
check("book table", [k_opt(2 ** n) for n in (2, 3, 4, 5, 6, 8, 10, 15, 20)] == [1, 2, 3, 4, 6, 12, 25, 142, 804])
check("book bound", all(P(2 ** n, k_opt(2 ** n)) >= 1 - 2 ** -n for n in range(2, 21)))
check("book n=8", k_opt(256) == 12 and round(pi / 4 * 16) == 13 and near(P(256, 12), 0.99995, 1e-5) and near(P(256, 13), 0.986, 1e-3))
check("book s9", near(pi / 4 * 2 ** 15, 25736, 1) and near(2 / pi * 2 ** 15, 20861, 1) and near(sqrt(1000 * 0.99 * 0.01), 3.1, 0.05))
check("book s10", 4 * 5 + 2 * 3 + 2 == 28 and 4 + 3 * 20 == 64)
check("book s11", k_opt(1024, 4) == 12 and all(k_opt(1000, m) == 0 for m in (501, 600, 750, 999)))
check("book s12", near(1 / P(16, 1), 2.12, 0.01) and near(2 / P(16, 2), 2.20, 0.01) and near(3 / P(16, 3), 3.12, 0.01)
      and near(2 / P(16, 1), 4.23, 0.01) and near(3 / P(16, 2), 3.30, 0.01) and near(4 / P(16, 3), 4.16, 0.01))
t13 = asin(0.2)
check("book s13", round(pi / (4 * t13) - 0.5) == 3 and near(sin(7 * t13) ** 2, 0.974, 1e-3) and near(32 * sin(3 * pi / 16) ** 2, 9.9, 0.05))
check("book s14", near(pi / 4 * 2 ** 20, 823550, 1) and near(0.995 ** 25, 0.88, 0.005))
ta = asin(1 / 3)
check("book P15", round(pi / (4 * ta) - 0.5) == 2 and near(sin(5 * ta) ** 2, 0.983, 1e-3))
check("book P10-16", near(P(64, 1), 2209 / 16384) and near(sin(3 * asin(0.2)) ** 2, 5041 / 15625) and k_opt(2 ** 12) == 50 and k_opt(128, 2) == 6
      and near(16 * sin(3 * pi / 16) ** 2, 4.94, 0.01))
check("circuit = formula", all(near(circuit_probability("1101"[:n] + "", k)[0], P(2 ** n, k)) for n in (2, 3, 4) for k in range(6)))

# ---------------------------------------------------------------- small cases with several marked items
for n_, marks_ in ((2, [1]), (3, [2, 5]), (4, [0, 5, 9, 14]), (4, [1, 2, 3, 4, 5, 6, 7, 8])):
    N_ = 2 ** n_
    sv = np.ones(N_, dtype=complex) / sqrt(N_)
    Om = np.eye(N_, dtype=complex)
    Om[marks_, marks_] = -1
    ok_ = True
    for k_ in range(8):
        ok_ &= near(float(np.sum(np.abs(sv[marks_]) ** 2)), P(N_, k_, len(marks_)))
        sv = D_matrix(N_) @ (Om @ sv)
    check(f"amplitudes = formula n={n_} M={len(marks_)}", ok_)

# ---------------------------------------------------------------- the notebook's own kernels (build_grover.py)
if "--cudaq" in sys.argv:
    import cudaq

    cudaq.set_target("qpp-cpu")
    cudaq.set_random_seed(42)

    @cudaq.kernel
    def grover_iteration(qubits: cudaq.qview, marked: list[int]):
        n = qubits.size()
        for i in range(n):
            if marked[i] == 0:
                x(qubits[i])
        z.ctrl(qubits[0:n - 1], qubits[n - 1])
        for i in range(n):
            if marked[i] == 0:
                x(qubits[i])
        h(qubits)
        x(qubits)
        z.ctrl(qubits[0:n - 1], qubits[n - 1])
        x(qubits)
        h(qubits)

    @cudaq.kernel
    def grover_search(n: int, marked: list[int], iterations: int):
        qubits = cudaq.qvector(n)
        h(qubits)
        for _ in range(iterations):
            grover_iteration(qubits, marked)
        mz(qubits)

    @cudaq.kernel
    def grover_state(n: int, marked: list[int], iterations: int):
        qubits = cudaq.qvector(n)
        h(qubits)
        for _ in range(iterations):
            grover_iteration(qubits, marked)

    for bits_ in ("11", "10", "101", "011", "1011", "0110", "01101"):
        n_, mk_ = len(bits_), [int(c) for c in bits_]
        ok_ = True
        for k_ in range(6):
            st_ = cudaq.get_state(grover_state, n_, mk_, k_)
            ok_ &= near(abs(st_.amplitude(mk_)) ** 2, P(2 ** n_, k_), 1e-9)
            ok_ &= near(abs(st_.amplitude(mk_)) ** 2, circuit_probability(bits_, k_)[0], 1e-9)
        check(f"CUDA-Q state {bits_}", ok_)
    c3 = cudaq.sample(grover_search, 3, [1, 0, 1], 2, shots_count=20000)
    check("CUDA-Q C20 shots", near(c3.count("101") / 20000, 121 / 128, 0.01), f"{c3.count('101') / 20000:.4f} vs {121 / 128:.4f}")
    c2 = cudaq.sample(grover_search, 2, [1, 0], 1, shots_count=2000)
    check("CUDA-Q C1 shots", c2.count("10") == 2000)
    st0 = cudaq.get_state(grover_state, 15, [int(c) for c in mk], 0)
    check("CUDA-Q n=15 k=0", near(abs(st0.amplitude([int(c) for c in mk])) ** 2, 1 / 2 ** 15, 1e-12))
    res = cudaq.estimate_resources(grover_search, 15, [int(c) for c in mk], 142)
    counts_ = {g: res.count(g) for g in ("h", "x")}
    check("CUDA-Q D5 gate counts", counts_ == {"h": 4275, "x": 5964}, str(counts_))

passed = sum(ok_ for _, ok_, _ in CHECKS)
print(f"{passed}/{len(CHECKS)} checks passed")
for name, ok_, detail in CHECKS:
    if detail and ok_:
        print(f"  {name}: {detail}")
sys.exit(0 if passed == len(CHECKS) else 1)
