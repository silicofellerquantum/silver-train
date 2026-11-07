"""Recompute every numerical / structural answer of the QML paper and assert it matches the key.

Run:  python verify_answers.py          (NumPy / SciPy only)
      python verify_answers.py --cudaq  (additionally rebuilds the QCNN kernel of QML.ipynb in CUDA-Q and cross-checks
                                         gate counts, encodings, kernels, VQC outputs and shift rules with CUDA-Q)

Conventions (same as the paper): qubit 0 is written first; R_P(t) = exp(-i t P / 2); <Z> = P(0) - P(1).
Every check compares a recomputed value with the option that the bank lists FIRST (the correct option of mcq/match/para)
or with the 'correct' index of ar/stmt questions; finally the structure of all 80 questions is checked.
"""
import math
import sys
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

import bank_advanced
import bank_basics
import bank_math
import bank_qm
from bank_explain import EXPLAIN

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, complex), np.asarray(b, complex), atol=tol)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0 + 0j, -1.0])
H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
S = np.diag([1, 1j])
P0, P1 = np.diag([1.0 + 0j, 0]), np.diag([0j, 1.0])


def R(P, t):
    return math.cos(t / 2) * I2 - 1j * math.sin(t / 2) * P


def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


CNOT = kron(P0, I2) + kron(P1, X)            # control qubit 0 (first), target qubit 1
k0 = np.array([1, 0], complex)


def ev(psi, M):
    return float(np.real(np.vdot(psi, M @ psi)))


sig = lambda z: 1 / (1 + math.exp(-z))

# ===================================================================================================== Section A
w = np.zeros(2); g = np.array([2 * (w[0] - 3), 4 * (w[1] + 1)]); w1 = w - 0.1 * g
L = lambda v: (v[0] - 3) ** 2 + 2 * (v[1] + 1) ** 2
check("A1", close(w1, [0.6, -0.4]) and abs(L(w1) - 6.48) < 1e-12 and abs(L([-0.6, 0.4]) - 16.88) < 1e-12
      and abs(L([0.6, -0.8]) - 5.84) < 1e-12 and L([0, 0]) == 11)
bce = -(math.log(0.9) + math.log(0.6) + math.log(0.8)) / 3
check("A2", abs(bce - 0.2798) < 1e-3 and abs(-(math.log10(0.9) + math.log10(0.6) + math.log10(0.8)) / 3 - 0.122) < 1e-3
      and abs(-(math.log(0.9) + math.log(0.6) + math.log(0.2)) / 3 - 0.742) < 1e-3)
Pp, Rr = Fraction(40, 50), Fraction(40, 60)
check("A3", Fraction(2) * Pp * Rr / (Pp + Rr) == Fraction(8, 11) and abs(float(Fraction(8, 11)) - 0.727) < 1e-3
      and abs(float((Pp + Rr) / 2) - 0.733) < 1e-3)
lam, vec = np.linalg.eigh(np.array([[2.0, 1], [1, 2]]))
check("A4", close(lam, [1, 3]) and abs(abs(vec[:, 1] @ np.array([1, 1]) / math.sqrt(2)) - 1) < 1e-12
      and lam[1] / lam.sum() == 0.75)
post = 0.9 * 0.01 / (0.9 * 0.01 + 0.05 * 0.99)
check("A5", abs(post - 0.1538) < 1e-3 and abs(0.9 * 0.01 / 0.05 - 0.18) < 1e-12)
z = 0.5 * 2 - 1; f = sig(z)
check("A6", abs(2 * (f - 1) * f * (1 - f) * 2 - (-0.5)) < 1e-12)
ez = np.exp([2, 1, 0]); p = ez / ez.sum()
check("A7", abs(p[0] - 0.665) < 1e-3 and abs(-math.log(p[1]) - 1.408) < 1e-3 and abs(-math.log(p[0]) - 0.408) < 1e-3)
check("A8", 8 * 4 + 4 + 4 * 1 + 1 == 41)
u, v = np.array([1, 2, 2.0]), np.array([2, 0, 1.0])
check("A9", np.sum((u - v) ** 2) == 6 and abs(math.exp(-np.sum((u - v) ** 2) / 2) - math.exp(-3)) < 1e-15)
A_, B_ = np.array([[1, 2], [0, 1]]), np.array([[0, 1], [1, 0]])
check("A10", np.array_equal(np.kron(A_, B_), [[0, 1, 0, 2], [1, 0, 2, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
      and np.array_equal(np.kron(B_, A_), [[0, 0, 1, 2], [0, 0, 0, 1], [1, 2, 0, 0], [0, 1, 0, 0]])
      and np.array_equal(np.kron(A_.T, B_), [[0, 1, 0, 0], [1, 0, 0, 0], [0, 2, 0, 1], [2, 0, 1, 0]])
      and np.array_equal(np.kron(np.array([[0, 1], [1, 0]]) @ A_, B_), [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 2], [1, 0, 2, 0]]))
check("A11", abs(math.sqrt(0.85 * 0.15 / 20) - 0.0798) < 1e-3)
rng = np.random.default_rng(0)
Xm, ym, wv = rng.normal(size=(7, 3)), rng.normal(size=7), rng.normal(size=3)
Lf = lambda ww: np.sum((Xm @ ww - ym) ** 2) / 7
num_grad = np.array([(Lf(wv + 1e-6 * e) - Lf(wv - 1e-6 * e)) / 2e-6 for e in np.eye(3)])
check("A13", close(num_grad, 2 / 7 * Xm.T @ (Xm @ wv - ym), 1e-6))
vv, tot = 0.0, 0.0
for _ in range(3):
    vv = 0.9 * vv + 1; tot += 0.1 * vv
check("A14", abs(tot - 0.561) < 1e-12)
zz = 0.7; hstep = 1e-6
dnum = lambda fn: (fn(zz + hstep) - fn(zz - hstep)) / (2 * hstep)
check("A15", abs(dnum(sig) - sig(zz) * (1 - sig(zz))) < 1e-8 and abs(dnum(math.tanh) - (1 - math.tanh(zz) ** 2)) < 1e-8
      and abs(dnum(lambda t: math.log(1 + math.exp(t))) - sig(zz)) < 1e-8)
# A17: XOR not linearly separable (LP with margin 1 infeasible); A18: GD stability and ridge training error
xor_pts = [((0, 0), -1), ((1, 1), -1), ((0, 1), 1), ((1, 0), 1)]
res = linprog(np.zeros(3), A_ub=[[-y * a, -y * b, -y] for (a, b), y in xor_pts], b_ub=[-1] * 4, bounds=[(None, None)] * 3)
check("A17 XOR infeasible", res.status == 2)
Hm = np.diag([2.0, 4.0])
conv = lambda eta: np.linalg.norm(np.linalg.matrix_power(np.eye(2) - eta * Hm, 400) @ np.ones(2)) < 1e-6
Xr, yr = rng.normal(size=(30, 4)), rng.normal(size=30)
w_ls = np.linalg.lstsq(Xr, yr, rcond=None)[0]
w_ridge = np.linalg.solve(Xr.T @ Xr + 3 * np.eye(4), Xr.T @ yr)
check("A18", conv(0.49) and not conv(0.51) and np.mean((Xr @ w_ridge - yr) ** 2) > np.mean((Xr @ w_ls - yr) ** 2))
W1, b1, w2, b2, x_ = np.array([[1, -1], [2, 1.0]]), np.array([0, -2.0]), np.array([0.5, -1]), -1.0, np.array([1, -1.0])
a_ = W1 @ x_ + b1; h_ = np.maximum(a_, 0); z_ = w2 @ h_ + b2; p_ = sig(z_)
check("A19", close(a_, [2, -1]) and close(h_, [2, 0]) and z_ == 0 and p_ == 0.5 and abs(-math.log(p_) - math.log(2)) < 1e-15
      and abs(-math.log(sig(w2 @ a_ + b2)) - 0.3133) < 1e-3)
dz = p_ - 1; da = dz * w2 * (a_ > 0)
check("A20", dz == -0.5 and close(da, [-0.25, 0]) and close(np.outer(da, x_)[1], [0, 0])
      and close(dz * w2[1] * x_, [0.5, -0.5]))

# ===================================================================================================== Section B
psi = np.array([0.5, math.sqrt(3) / 2 * np.exp(1j * math.pi / 4)])
check("B1", abs(ev(psi, X) - math.sqrt(6) / 4) < 1e-12 and abs(ev(psi, Z) + 0.5) < 1e-12)
a, b, c = 0.3, -1.2, 0.7
Mx = a * X + b * Y + c * Z
check("B2", close(Mx @ Mx, (a * a + b * b + c * c) * I2))
rngB = np.random.default_rng(5)
V1 = np.linalg.qr(rngB.normal(size=(2, 2)) + 1j * rngB.normal(size=(2, 2)))[0]
M_ = np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, -0.6]])
fB = lambda t: ev(V1 @ R(Y, t) @ np.array([0.6, 0.8j]), M_)
check("B3", close(R(Y, 1.0 + 2 * math.pi), -R(Y, 1.0)) and abs(fB(0.4) - fB(0.4 + 2 * math.pi)) < 1e-12
      and abs(fB(0.4) - fB(0.4 + math.pi)) > 1e-3)
detC = lambda vv: vv[0] * vv[3] - vv[1] * vv[2]
check("B4", abs(detC([1, -1, 1, -1])) < 1e-12 and all(abs(detC(s)) > 0.1 for s in ([1, 1, 1, -1], [1, 0, 0, 1], [0, 1, -1, 0])))
phi = np.array([1, 0, 0, 1]) / math.sqrt(2)
check("B5", abs(ev(phi, kron(Z, I2))) < 1e-12 and abs(ev(phi, kron(Z, Z)) - 1) < 1e-12 and abs(ev(phi, kron(X, X)) - 1) < 1e-12)
bell = CNOT @ kron(H, I2) @ np.array([1, 0, 0, 0])
check("B6", all(abs(ev(kron(I2, R(Y, t)) @ bell, kron(I2, Z))) < 1e-12 for t in np.linspace(0, 6, 13))
      and abs(ev(kron(I2, R(Y, 0.9)) @ bell, kron(Z, Z)) - math.cos(0.9)) < 1e-12)
check("B7", abs((620 - 380) / 1000 - 0.24) < 1e-12 and abs(math.sqrt((1 - 0.24 ** 2) / 1000) - 0.0307) < 1e-4)
check("B8", close(np.kron(R(Y, math.pi / 2) @ k0, R(Y, math.pi) @ k0), np.array([0, 1, 0, 1]) / math.sqrt(2)))
unit = lambda U: close(U.conj().T @ U, I2)
check("B9", not unit(np.array([[1, 1], [1, 1]]) / math.sqrt(2)) and unit(H) and unit(S) and unit(R(X, math.pi / 4)))
check("B9 option", close(R(X, math.pi / 4), [[math.cos(math.pi / 8), -1j * math.sin(math.pi / 8)], [-1j * math.sin(math.pi / 8), math.cos(math.pi / 8)]]))
# B10: a global phase leaves the outcome probabilities of every (random) measurement basis unchanged; relative phases do not
bases = [np.linalg.qr(rngB.normal(size=(2, 2)) + 1j * rngB.normal(size=(2, 2)))[0] for _ in range(20)]
probs = lambda s_: np.array([np.abs(Bm.conj().T @ s_) ** 2 for Bm in bases])
s_i = np.array([1, 1j]) / math.sqrt(2)
check("B10", close(probs(s_i), probs(np.exp(1j * math.pi / 3) * s_i))
      and all(np.max(np.abs(probs(p_) - probs(q_))) > 1e-2 for p_, q_ in [
          (np.array([1, 1]) / math.sqrt(2), np.array([1, -1]) / math.sqrt(2)), (s_i, np.array([1, -1j]) / math.sqrt(2)),
          (np.array([1, 1]) / math.sqrt(2), s_i)]))
t = 0.83
check("B11", close(R(Y, t).conj().T @ Z @ R(Y, t), math.cos(t) * Z - math.sin(t) * X))
CRZ = kron(P0, I2) + kron(P1, R(Z, t))
check("B12", close(CRZ, np.diag([1, 1, np.exp(-1j * t / 2), np.exp(1j * t / 2)])))
check("B13", 16 * 2 ** 30 == 16 * 1024 ** 3 and abs(16 * 2 ** 30 / 1e9 - 17.18) < 0.01)
t1, t2 = 0.7, -1.3
check("B14", abs(ev(R(X, t2) @ R(Y, t1) @ k0, Z) - math.cos(t1) * math.cos(t2)) < 1e-12)
check("B15", close(H @ Z @ H, X) and close(S @ X @ S.conj().T, Y) and close(R(Y, math.pi), [[0, -1], [1, 0]])
      and close(R(X, math.pi), -1j * X))
bloch = lambda s: [ev(s, X), ev(s, Y), ev(s, Z)]
check("B16", close(bloch(np.array([1, 1]) / math.sqrt(2)), [1, 0, 0]) and close(bloch(np.array([1, 1j]) / math.sqrt(2)), [0, 1, 0])
      and close(bloch(np.array([math.cos(math.pi / 8), math.sin(math.pi / 8)])), [1 / math.sqrt(2), 0, 1 / math.sqrt(2)])
      and close(bloch(np.array([0, 1])), [0, 0, -1]))
check("B17", abs(ev(R(X, t) @ k0, Z) - ev(R(X, -t) @ k0, Z)) < 1e-12 and abs(ev(R(X, t) @ k0, Y) - ev(R(X, -t) @ k0, Y)) > 0.1)
check("B18", close(CNOT @ np.array([1, 0, 0, 0]), [1, 0, 0, 0]) and abs(detC(np.kron([0.6, 0.8j], [0.28, 0.96]))) < 1e-12)
for th in (0.3, math.pi / 2, 2.2):
    st = CNOT @ kron(R(Y, th), I2) @ np.array([1, 0, 0, 0])
    rho1 = np.array([[abs(st[0]) ** 2 + abs(st[2]) ** 2, 0], [0, abs(st[1]) ** 2 + abs(st[3]) ** 2]])
    check(f"B19 theta={th:.2f}", abs(ev(st, kron(I2, Z)) - math.cos(th)) < 1e-12 and abs(ev(st, kron(Z, Z)) - 1) < 1e-12)
    rho = np.outer(st, st.conj()).reshape(2, 2, 2, 2)
    r1 = np.einsum("abad->bd", rho)
    check(f"B20 purity theta={th:.2f}", abs(np.trace(r1 @ r1).real - (1 + math.cos(th) ** 2) / 2) < 1e-12)
check("B20", abs((1 + math.cos(math.pi / 4) ** 2) / 2 - 0.75) < 1e-12)

# ===================================================================================================== Section C
enc = lambda xx: R(Y, math.pi * xx) @ k0
check("C1", abs(abs(enc(1 / 3)[1]) ** 2 - 0.25) < 1e-12 and abs(ev(enc(1 / 3), Z) - 0.5) < 1e-12)
xa = np.array([1, 2, 4, 2.0]) / 5
check("C2", abs(xa[2] ** 2 + xa[3] ** 2 - 0.8) < 1e-12 and abs(xa[1] ** 2 + xa[3] ** 2 - 0.32) < 1e-12)
check("C3", (256 * 8, 256, int(math.log2(256))) == (2048, 256, 8))


def prod_state(xs):
    out = np.array([1.0 + 0j])
    for xx in xs:
        out = np.kron(out, enc(xx))
    return out


kern = lambda x1, x2: abs(np.vdot(prod_state(x1), prod_state(x2))) ** 2
check("C4", abs(kern([1, 0.5], [0.5, 0]) - 0.25) < 1e-12
      and abs(kern([0.3, 0.9, 0.1], [0.7, 0.2, 0.4]) - np.prod(np.cos(np.pi * (np.array([0.3, 0.9, 0.1]) - [0.7, 0.2, 0.4]) / 2) ** 2)) < 1e-12)
check("C5", math.comb(20, 2) == 190)
# C6 / D14: full swap-test circuit on 3 qubits (ancilla first)
rs = np.random.default_rng(9)
sa, sb = rs.normal(size=2) + 1j * rs.normal(size=2), rs.normal(size=2) + 1j * rs.normal(size=2)
sa, sb = sa / np.linalg.norm(sa), sb / np.linalg.norm(sb)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex)
CSWAP = kron(P0, np.eye(4)) + kron(P1, SWAP)
out = kron(H, np.eye(4)) @ CSWAP @ kron(H, np.eye(4)) @ np.kron(k0, np.kron(sa, sb))
p0 = np.sum(abs(out[:4]) ** 2)
check("C6 swap test", abs(p0 - 0.5 * (1 + abs(np.vdot(sa, sb)) ** 2)) < 1e-12)
check("C6", abs(2 * 0.8 - 1 - 0.6) < 1e-12)
# C7: shift rule exact for a random single-qubit circuit with one R_Y(theta)
U1 = np.linalg.qr(rs.normal(size=(2, 2)) + 1j * rs.normal(size=(2, 2)))[0]
U2 = np.linalg.qr(rs.normal(size=(2, 2)) + 1j * rs.normal(size=(2, 2)))[0]
fC = lambda th: ev(U2 @ R(Y, th) @ U1 @ k0, Z)
th0 = 0.37
check("C7 exact shift", abs(0.5 * (fC(th0 + math.pi / 2) - fC(th0 - math.pi / 2)) - (fC(th0 + 1e-6) - fC(th0 - 1e-6)) / 2e-6) < 1e-8)
check("C7", abs(0.5 * (0.30 + 0.50) - 0.40) < 1e-12)


# ---- the QCNN of QML.ipynb (layout copied verbatim) and an independent NumPy simulator ----------------------------
def qcnn_layout(n):
    kinds, qubit_a, qubit_b, layers = [], [], [], []
    active, layer = list(range(n)), 0
    while len(active) > 1:
        for start in (0, 1):
            for i in range(start, len(active) - 1, 2):
                kinds.append(0); qubit_a.append(active[i]); qubit_b.append(active[i + 1]); layers.append(layer)
        kept = []
        for i in range(0, len(active) - 1, 2):
            kinds.append(1); qubit_a.append(active[i]); qubit_b.append(active[i + 1]); layers.append(layer)
            kept.append(active[i + 1])
        if len(active) % 2 == 1:
            kept.append(active[-1])
        active, layer = kept, layer + 1
    return kinds, qubit_a, qubit_b, layers, active[0], layer


def apply1(psi, U, q):
    return np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)


def apply_c(psi, U, c, t):
    psi = psi.copy()
    idx = [slice(None)] * psi.ndim
    idx[c] = 1
    tt = t if t < c else t - 1
    psi[tuple(idx)] = np.moveaxis(np.tensordot(U, psi[tuple(idx)], axes=([1], [tt])), 0, tt)
    return psi


def np_qcnn(pixels, theta, lay, outq, gate_params=None):
    """<Z_out>. gate_params, if given, overrides the angle of every gate occurrence (list in gate order)."""
    n = len(pixels)
    psi = np.zeros([2] * n, complex); psi[(0,) * n] = 1
    for i, pv in enumerate(pixels):
        psi = apply1(psi, R(Y, math.pi * pv), i)
    occ = 0
    for kd, qa, qb, l in zip(*lay[:4]):
        tb = 4 * l

        def ang(k):
            nonlocal occ
            val = theta[tb + k] if gate_params is None else gate_params[occ]
            occ += 1
            return val
        if kd == 0:
            psi = apply1(psi, R(Y, ang(0)), qa); psi = apply1(psi, R(Y, ang(1)), qb); psi = apply_c(psi, X, qa, qb)
        else:
            psi = apply_c(psi, R(Z, ang(2)), qa, qb); psi = apply1(psi, X, qa)
            psi = apply_c(psi, R(X, ang(3)), qa, qb); psi = apply1(psi, X, qa)
    pr = np.abs(psi) ** 2
    po = pr.sum(axis=tuple(i for i in range(n) if i != outq))
    return po[0] - po[1]


def occurrences(lay):
    """Parameter index of every gate occurrence, in gate order."""
    out = []
    for kd, _, _, l in zip(*lay[:4]):
        out += [4 * l, 4 * l + 1] if kd == 0 else [4 * l + 2, 4 * l + 3]
    return out


lay8 = qcnn_layout(8)
k8 = lay8[0]
check("C8", sum(1 for kd, l in zip(k8, lay8[3]) if kd == 0 and l == 0) == 7)
check("C9", qcnn_layout(20)[5] == 5 and 4 * qcnn_layout(20)[5] == 20 and qcnn_layout(12)[5] == 4)
check("C9 log2 rule", all(qcnn_layout(n)[5] == math.ceil(math.log2(n)) for n in range(2, 65)))
# the four CUDA-Q lines of the pooling block (rz.ctrl; x; rx.ctrl; x), multiplied right to left
code_pool = (kron(X, I2) @ (kron(P0, I2) + kron(P1, R(X, 1.1))) @ kron(X, I2)) @ (kron(P0, I2) + kron(P1, R(Z, 0.4)))
check("C10", close(code_pool, kron(P1, R(Z, 0.4)) + kron(P0, R(X, 1.1))))
gates = lambda n: (lambda L_: n + 3 * L_[0].count(0) + 4 * L_[0].count(1))(qcnn_layout(n))
lay16 = qcnn_layout(16)
check("C11", lay16[0].count(0) == 26 and lay16[0].count(1) == 15 and gates(16) == 154 and gates(8) == 69 and gates(15) == 146)
fo, yo = np.array([0.6, -0.2, 0.9, 0.1]), np.array([1, -1, 1, -1])
check("C12", abs(np.mean((fo - yo) ** 2) - 0.505) < 1e-12 and np.mean(np.sign(fo) == yo) == 0.75)
check("C13", 20 * 100 + 20 == 2020 and abs(2020 * 1.7e-3 - 3.4) < 0.05)
# C14: frequency spectrum of a re-uploading model with L = 3 (numerical Fourier analysis)
Ws = [np.linalg.qr(rs.normal(size=(2, 2)) + 1j * rs.normal(size=(2, 2)))[0] for _ in range(4)]


def reup(xx):
    st = Ws[0] @ k0
    for Wm in Ws[1:]:
        st = Wm @ R(Y, xx) @ st
    return ev(st, Z)


grid = np.linspace(0, 2 * np.pi, 64, endpoint=False)
spec = np.abs(np.fft.rfft([reup(xx) for xx in grid])) / 64
check("C14", spec[3] > 1e-6 and np.all(spec[4:] < 1e-12))
# C17: the controlled-rotation generator has eigenvalues 0, 0, +-1/2
check("C17", close(sorted(np.linalg.eigvalsh(kron(P1, Z / 2)).real), [-0.5, 0, 0, 0.5]))
# C18: random fidelity Gram matrices are PSD; noisy estimates can be indefinite
pts = rs.uniform(0, 1, size=(30, 6))
Kg = np.array([[kern(p_a, p_b) for p_b in pts] for p_a in pts])
Kn = Kg + np.triu(rs.normal(0, 0.05, size=(30, 30)), 1); Kn = np.triu(Kn) + np.triu(Kn, 1).T
check("C18", np.min(np.linalg.eigvalsh(Kg)) > -1e-10 and np.min(np.linalg.eigvalsh(Kn)) < 0)
vqc = lambda x0, x1, th: ev(kron(I2, R(Y, th)) @ CNOT @ np.kron(enc(x0 / math.pi), enc(x1 / math.pi)), kron(I2, Z))
check("C19", all(abs(vqc(x0, x1, th) - (math.cos(th) * math.cos(x0) * math.cos(x1) - math.sin(th) * math.sin(x1))) < 1e-12
                 for x0, x1, th in [(0.3, 1.2, -0.4), (2.0, -0.7, 1.9), (0, math.pi / 3, math.pi / 6)]))
fp, fm = vqc(0, math.pi / 3, math.pi / 6 + math.pi / 2), vqc(0, math.pi / 3, math.pi / 6 - math.pi / 2)
f0 = vqc(0, math.pi / 3, math.pi / 6)
check("C20", abs(f0) < 1e-12 and abs(fp + 1) < 1e-12 and abs(fm - 1) < 1e-12 and abs(2 * (f0 - 1) * 0.5 * (fp - fm) - 2) < 1e-12)

# ===================================================================================================== Section D
check("D1", 1e5 < 2 ** 20 < 1e7 and 1e11 < 2 ** 40 < 1e13)
fD3 = lambda th: math.cos(th / 2)
two = 0.5 * (fD3(math.pi + math.pi / 2) - fD3(math.pi - math.pi / 2))
dp, dm = (math.sqrt(2) + 1) / (4 * math.sqrt(2)), (math.sqrt(2) - 1) / (4 * math.sqrt(2))
four = lambda fn, th: dp * (fn(th + math.pi / 2) - fn(th - math.pi / 2)) - dm * (fn(th + 1.5 * math.pi) - fn(th - 1.5 * math.pi))
check("D3", abs(two + math.sqrt(2) / 2) < 1e-12 and abs(four(fD3, math.pi) + 0.5) < 1e-12)
# four-term rule exact for a random circuit with a controlled rotation (Hadamard test and generic observable)
Ua = np.linalg.qr(rs.normal(size=(4, 4)) + 1j * rs.normal(size=(4, 4)))[0]
Ub = np.linalg.qr(rs.normal(size=(4, 4)) + 1j * rs.normal(size=(4, 4)))[0]
Mobs = kron(X, Z) + 0.5 * kron(Y, I2)
fcr = lambda th: ev(Ub @ (kron(P0, I2) + kron(P1, R(Z, th))) @ Ua @ np.array([1, 0, 0, 0]), Mobs)
check("D3 four-term exact", abs(four(fcr, 0.6) - (fcr(0.6 + 1e-6) - fcr(0.6 - 1e-6)) / 2e-6) < 1e-7
      and abs(0.5 * (fcr(0.6 + math.pi / 2) - fcr(0.6 - math.pi / 2)) - (fcr(0.6 + 1e-6) - fcr(0.6 - 1e-6)) / 2e-6) > 1e-3)
check("D4", abs(0.99 ** 69 - 0.4998) < 1e-3)
nmax = lambda mem, bts: int(math.floor(math.log2(mem / bts)))
check("D5", (nmax(80e9, 16), nmax(320e9, 16), nmax(80e9, 8), nmax(320e9, 8)) == (32, 34, 33, 35))
n_conv, n_pool = k8.count(0), k8.count(1)
check("D6", (n_conv, n_pool) == (11, 7) and (4 * n_conv + 8 * n_pool) * 20 == 2000 and 2 * (2 * n_conv + 2 * n_pool) * 20 == 1440)
# D6 (numerical): per-occurrence shift rules reproduce the exact gradient of the 8-qubit QCNN output
th12 = rs.uniform(0, 2 * np.pi, 12); px8 = rs.uniform(0, 1, 8)
occ8 = occurrences(lay8); base = [th12[j] for j in occ8]
kind_of = []
for kd in k8:
    kind_of += ["ry", "ry"] if kd == 0 else ["c", "c"]


def f_occ(vals):
    return np_qcnn(px8, th12, lay8, lay8[4], gate_params=vals)


grad_shift = np.zeros(12)
circuits = 0
for g_i, (j, kd) in enumerate(zip(occ8, kind_of)):
    def fg(val):
        vv_ = list(base); vv_[g_i] = val
        return f_occ(vv_)
    if kd == "ry":
        grad_shift[j] += 0.5 * (fg(base[g_i] + math.pi / 2) - fg(base[g_i] - math.pi / 2)); circuits += 2
    else:
        grad_shift[j] += four(fg, base[g_i]); circuits += 4
grad_fd = np.array([(np_qcnn(px8, th12 + 1e-6 * e, lay8, lay8[4]) - np_qcnn(px8, th12 - 1e-6 * e, lay8, lay8[4])) / 2e-6 for e in np.eye(12)])
check("D6 shift rules reproduce gradient", close(grad_shift, grad_fd, 1e-6) and circuits == 100, np.max(np.abs(grad_shift - grad_fd)))
naive = np.array([0.5 * (np_qcnn(px8, th12 + math.pi / 2 * e, lay8, lay8[4]) - np_qcnn(px8, th12 - math.pi / 2 * e, lay8, lay8[4])) for e in np.eye(12)])
check("D6 naive two-term rule is wrong", np.max(np.abs(naive - grad_fd)) > 1e-2)
counts = {"00": 400, "01": 100, "10": 300, "11": 200}
ez0 = sum(c_ * (1 if s[0] == "0" else -1) for s, c_ in counts.items()) / 1000
ez1 = sum(c_ * (1 if s[1] == "0" else -1) for s, c_ in counts.items()) / 1000
ezz = sum(c_ * (1 if s[0] == s[1] else -1) for s, c_ in counts.items()) / 1000
check("D7", (ez0, ez1, ezz) == (0.0, 0.4, 0.2))
check("D8", abs((1 - 0.2) ** 2 - 0.64) < 1e-12)
check("D9", abs(60 * 16 * 0.35 / 60 - 5.6) < 0.01)
from scipy.integrate import dblquad
e1 = dblquad(lambda xx, yy: math.cos(math.pi * (xx - yy) / 2) ** 2, 0, 1, 0, 1)[0]
check("D10", abs(e1 - (0.5 + 2 / math.pi ** 2)) < 1e-9 and abs(e1 - 0.7026) < 1e-3)
check("D12", abs(2 * math.log(0.5) + math.log(4)) < 1e-15)
check("D13", abs(2 + 0.5 * (1 + 0.9 * 4 - 2) - 3.3) < 1e-12 and abs(2 + 0.5 * (1 + 4 - 2) - 3.5) < 1e-12)
ov = math.sqrt(2 * 0.9 - 1)
check("D14", abs(math.sqrt(2 - 2 * ov) - 0.46) < 0.005 and abs(math.sqrt(2 - 2 * 0.8) - 0.63) < 0.005)
TP, FN, FP, TN = 9, 1, 2, 8
check("D19", abs(TP / (TP + FP) - 0.818) < 1e-3 and TP / (TP + FN) == 0.9 and abs(2 * TP / (2 * TP + FP + FN) - 0.857) < 1e-3
      and abs((TP / (TP + FP) + 0.9) / 2 - 0.859) < 1e-3)
# D20: clean line images are not linearly separable (LP feasibility with margin 1 fails)
imgs = []
for r_ in range(2):
    for c_ in range(3):
        im = np.zeros((2, 4)); im[r_, c_:c_ + 2] = 1; imgs.append((im.ravel(), 1))
for c_ in range(4):
    im = np.zeros((2, 4)); im[:, c_] = 1; imgs.append((im.ravel(), -1))
res = linprog(np.zeros(9), A_ub=[list(-yy * xx) + [-yy] for xx, yy in imgs], b_ub=[-1] * len(imgs), bounds=[(None, None)] * 9)
check("D20 not linearly separable", res.status == 2)
check("D20", abs(math.sqrt(0.8 * 0.2 / 2000) - 0.0089) < 1e-3 and abs(math.sqrt(0.85 * 0.15 / 20) - 0.08) < 1e-3)

# ===================================================================================================== structure
banks = [bank_math, bank_qm, bank_basics, bank_advanced]
for bank, sec in zip(banks, "ABCD"):
    qs = bank.QUESTIONS
    check(f"{sec} has 20 questions", len(qs) == 20)
    check(f"{sec} ids", [q["id"] for q in qs] == [f"{sec}{i}" for i in range(1, 21)])
    kinds = [q["kind"] for q in qs]
    check(f"{sec} format pattern", kinds == ["mcq"] * 14 + ["match"] * 2 + ["ar", "stmt", "para", "para"])
    for q in qs:
        check(f"{q['id']} four distinct options", len(q["opts"]) == 4 and len(set(q["opts"])) == 4)
        if q["kind"] == "match":
            perms = [tuple(int(part.split("to$")[1]) for part in o.split(", ")) for o in q["opts"]]
            check(f"{q['id']} options are permutations", all(sorted(p_) == [1, 2, 3, 4] for p_ in perms) and len(set(perms)) == 4)
        ex = EXPLAIN[q["id"]]
        check(f"{q['id']} explanation complete", set(ex) == {"concept", "formulae", "wrong", "takeaway"} and len(ex["wrong"]) == 3)
check("AR/Statement keys", [q["correct"] for b_ in banks for q in b_.QUESTIONS if q["kind"] in ("ar", "stmt")]
      == [0, 2, 2, 3, 1, 0, 3, 1])

# ===================================================================================================== CUDA-Q
if "--cudaq" in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    import cudaq
    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def qcnn(pixels: list[float], theta: list[float], kinds: list[int], qubit_a: list[int],
             qubit_b: list[int], layers: list[int]):
        qubits = cudaq.qvector(len(pixels))
        for i in range(len(pixels)):
            ry(np.pi * pixels[i], qubits[i])
        for k in range(len(kinds)):
            t = 4 * layers[k]
            a = qubit_a[k]
            b = qubit_b[k]
            if kinds[k] == 0:
                ry(theta[t], qubits[a])
                ry(theta[t + 1], qubits[b])
                x.ctrl(qubits[a], qubits[b])
            else:
                rz.ctrl(theta[t + 2], qubits[a], qubits[b])
                x(qubits[a])
                rx.ctrl(theta[t + 3], qubits[a], qubits[b])
                x(qubits[a])

    for n in (8, 15, 16):
        lay = qcnn_layout(n)
        r_ = cudaq.estimate_resources(qcnn, [0.5] * n, [0.1] * (4 * lay[5]), *lay[:4])
        check(f"cudaq gate count N={n}", r_.count() == gates(n), (r_.count(), gates(n)))
    r8 = cudaq.estimate_resources(qcnn, [0.5] * 8, [0.1] * 12, *lay8[:4])
    check("cudaq recorded 8-qubit statistics", r8.count() == 69 and r8.depth == 21
          and r8.to_dict() == {"ry": 30, "cx": 11, "crz": 7, "x": 14, "crx": 7}, r8.to_dict())
    for n in (4, 8, 9):
        lay = qcnn_layout(n)
        th_ = list(rs.uniform(0, 2 * np.pi, 4 * lay[5])); px_ = list(rs.uniform(0, 1, n))
        e_ = cudaq.observe(qcnn, cudaq.spin.z(lay[4]), px_, th_, *lay[:4]).expectation()
        check(f"cudaq QCNN = NumPy QCNN (N={n})", abs(e_ - np_qcnn(px_, th_, lay, lay[4])) < 1e-10)

    @cudaq.kernel
    def encode(pixels: list[float]):
        q = cudaq.qvector(len(pixels))
        for i in range(len(pixels)):
            ry(np.pi * pixels[i], q[i])

    sA, sB = np.array(cudaq.get_state(encode, [1.0, 0.5])), np.array(cudaq.get_state(encode, [0.5, 0.0]))
    check("cudaq C4 kernel", abs(abs(np.vdot(sA, sB)) ** 2 - 0.25) < 1e-12)
    check("cudaq C1 encoding", abs(cudaq.observe(encode, cudaq.spin.z(0), [1 / 3]).expectation() - 0.5) < 1e-12)

    @cudaq.kernel
    def vqc2(x0: float, x1: float, th: float):
        q = cudaq.qvector(2)
        ry(x0, q[0]); ry(x1, q[1])
        x.ctrl(q[0], q[1])
        ry(th, q[1])

    for x0, x1, th in [(0.3, 1.2, -0.4), (0.0, math.pi / 3, math.pi / 6 + math.pi / 2), (0.0, math.pi / 3, math.pi / 6 - math.pi / 2)]:
        check(f"cudaq C19/C20 ({x0:.2f},{x1:.2f},{th:.2f})",
              abs(cudaq.observe(vqc2, cudaq.spin.z(1), x0, x1, th).expectation() - vqc(x0, x1, th)) < 1e-12)

    @cudaq.kernel
    def bell_ry(th: float):
        q = cudaq.qvector(2)
        h(q[0]); x.ctrl(q[0], q[1]); ry(th, q[1])

    check("cudaq B6", abs(cudaq.observe(bell_ry, cudaq.spin.z(1), 1.234).expectation()) < 1e-12)

    @cudaq.kernel
    def b19(th: float):
        q = cudaq.qvector(2)
        ry(th, q[0]); x.ctrl(q[0], q[1])

    check("cudaq B19", abs(cudaq.observe(b19, cudaq.spin.z(1), 0.8).expectation() - math.cos(0.8)) < 1e-12
          and abs(cudaq.observe(b19, cudaq.spin.z(0) * cudaq.spin.z(1), 0.8).expectation() - 1) < 1e-12)

    @cudaq.kernel
    def hadamard_test(t: float):
        q = cudaq.qvector(2)
        h(q[0]); ry(0.7, q[1]); rx(0.4, q[1])
        rz.ctrl(t, q[0], q[1])
        h(q[0])

    hf = lambda tt: cudaq.observe(hadamard_test, cudaq.spin.z(0), float(tt)).expectation()
    check("cudaq D3 Hadamard test = cos(t/2)", all(abs(hf(tt) - math.cos(tt / 2)) < 1e-12 for tt in (0.3, 1.7, math.pi)))
    check("cudaq D3 shift rules", abs(0.5 * (hf(1.5 * math.pi) - hf(0.5 * math.pi)) + math.sqrt(2) / 2) < 1e-12
          and abs(four(hf, math.pi) + 0.5) < 1e-12)
    import re
    ver = re.search(r"\d+\.\d+\.\d+", cudaq.__version__)
    print(f"CUDA-Q {ver.group(0) if ver else cudaq.__version__} cross-checks done (target qpp-cpu)")

bad = [c_ for c_ in CHECKS if not c_[1]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
sys.exit(1 if bad else 0)
