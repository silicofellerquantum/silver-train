"""Recompute every numerical / structural answer of the QFT paper and assert it matches the key.

Run:  python verify_answers.py          (numpy only)
      python verify_answers.py --cudaq  (additionally cross-checks the QFT matrices with CUDA-Q state vectors)

Convention (same as the paper): QFT|x> = N^-1/2 sum_y w^{xy}|y>, w = e^{2 pi i/N}, qubit 1 = most significant bit.
"""
import cmath
import math
import sys
from fractions import Fraction
from math import gcd

import numpy as np

CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def F(N):
    """Unitary Fourier matrix, entry [y, x] = w^{xy}/sqrt(N)."""
    idx = np.arange(N)
    return np.exp(2j * np.pi * np.outer(idx, idx) / N) / np.sqrt(N)


def ket(bits):
    v = np.zeros(2 ** len(bits), complex)
    v[int(bits, 2)] = 1
    return v


def kron(*vs):
    out = np.array([1.0 + 0j])
    for v in vs:
        out = np.kron(out, v)
    return out


s2 = np.sqrt(2)
k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
plus, minus = (k0 + k1) / s2, (k0 - k1) / s2
H = np.array([[1, 1], [1, -1]]) / s2
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
I2 = np.eye(2)


def R(k):
    return np.diag([1, np.exp(2j * np.pi / 2 ** k)])


def close(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def is_product(c):
    c = np.asarray(c).reshape(2, 2)
    return abs(np.linalg.det(c)) < 1e-12


def qpe_probs(phi, t):
    M = 2 ** t
    k = np.arange(M)
    reg = np.exp(2j * np.pi * phi * k) / np.sqrt(M)
    out = F(M).conj().T @ reg  # inverse QFT
    return np.abs(out) ** 2


def textbook_qft(n):
    """Build the textbook circuit gate by gate (H, controlled-R_k, swaps) and return its matrix."""
    N = 2 ** n

    def op_single(g, q):
        mats = [I2] * n
        mats[q] = g
        out = np.array([[1.0 + 0j]])
        for m in mats:
            out = np.kron(out, m)
        return out

    def cphase(c, tq, theta):
        d = np.ones(N, complex)
        for x in range(N):
            if (x >> (n - 1 - c)) & 1 and (x >> (n - 1 - tq)) & 1:
                d[x] = np.exp(1j * theta)
        return np.diag(d)

    def swap(a, b):
        P = np.zeros((N, N))
        for x in range(N):
            bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
            bits[a], bits[b] = bits[b], bits[a]
            P[int("".join(map(str, bits)), 2), x] = 1
        return P

    U = np.eye(N, dtype=complex)
    for j in range(n):
        U = op_single(H, j) @ U
        for k in range(2, n - j + 1):
            U = cphase(j + k - 1, j, 2 * np.pi / 2 ** k) @ U
    for q in range(n // 2):
        U = swap(q, n - 1 - q) @ U
    return U


# ----------------------------------------------------------------------------------------------- Section A
w8 = cmath.exp(2j * math.pi / 8)
check("A1", abs(sum(k * w8 ** k for k in range(8)) - 8 / (w8 - 1)) < 1e-12)
check("A2", sum(1 for x in range(16) for y in range(16)
                if abs(sum(cmath.exp(2j * math.pi * (x + y) * z / 16) for z in range(16))) > 1e-9) == 16)
check("A3", abs(cmath.exp(2j * math.pi * 45 / 16) - cmath.exp(1j * 13 * math.pi / 8)) < 1e-12)
U4 = np.array([[1, 1, 1, 1], [1, 1j, -1, -1j], [1, -1, 1, -1], [1, -1j, -1, 1j]]) / 2
check("A4", close(U4.conj().T @ U4, np.eye(4)) and close(U4, F(4)))
check("A5", abs(np.trace(F(4)) - (1 + 1j)) < 1e-12)
th = np.linspace(1e-6, 2 * np.pi, 200001)
f = np.abs(sum(np.exp(1j * j * th) for j in range(4))) ** 2
first_zero = th[np.argmax(f < 1e-8)]
check("A6", abs(f.max() - 16) < 1e-6 and abs(np.abs(sum(np.exp(1j * j * np.pi / 2) for j in range(4)))) < 1e-12
      and abs(first_zero - np.pi / 2) < 1e-3)
check("A7", Fraction(11, 64).limit_denominator(10) == Fraction(1, 6))
r = next(r for r in range(1, 100) if pow(3, r, 35) == 1)
check("A8", r == 12 and gcd(pow(3, 6, 35) - 1, 35) == 7 and gcd(pow(3, 6, 35) + 1, 35) == 5)
check("A9", (2 ** 10, 2 * 2 ** 10 - 2) == (1024, 2046))
xv = 0b1011
check("A10", sum(1 for y in range(16) if bin(xv & y).count("1") % 2) == 8)
ratio = 20 * 2 ** 20 / (20 * 21 // 2 + 10)
check("A11", 10 ** 4.5 < ratio < 10 ** 5.5, f"ratio={ratio:.0f}")
check("A12", abs(np.prod([abs(1 - w8 ** k) for k in range(1, 8)]) - 8) < 1e-12)
alpha = np.array([1, 2, 2, 0]) / 3
check("A13", abs(abs((F(4) @ alpha)[0]) ** 2 - 25 / 36) < 1e-12)
check("A14", all(abs(cmath.exp(2j * math.pi * 4 * (a / 2 + b / 4 + c / 8)) - (-1) ** c) < 1e-12
                 for a in (0, 1) for b in (0, 1) for c in (0, 1)))
check("A15", close([w8 ** 4, sum(w8 ** (2 * k) for k in range(8)), w8 * w8 ** 7, w8 + w8 ** 7], [-1, 0, 1, math.sqrt(2)]))
F4 = F(4)
F42 = F4 @ F4
HH = np.kron(H, H)
CT = np.diag([1, 1, 1, np.exp(1j * np.pi / 4)])
is_perm = lambda M: close(np.abs(M), np.round(np.abs(M))) and close(np.abs(M).sum(0), 1)
check("A16", close(np.linalg.matrix_power(F4, 4), np.eye(4)) and not close(F42, np.eye(4)) and is_perm(F42)
      and close(F42 @ F42, np.eye(4)) and close(HH, HH.T) and close(HH @ HH, np.eye(4)) and not is_perm(HH)
      and abs(np.linalg.det(CT) - np.exp(1j * np.pi / 4)) < 1e-12 and not close(np.linalg.matrix_power(CT, 4), np.eye(4)))
vals = [np.exp(2j * np.pi * j * 2 / 8) for j in range(8)]
check("A17", abs(sum(vals)) < 1e-12 and len({(round(v.real, 9), round(v.imag, 9)) for v in vals}) == 4)
check("A18", abs(np.linalg.det(F4) - (-1j)) < 1e-12)
a = np.array([1.0 if x % 5 == 2 else 0.0 for x in range(32)])
A = np.array([sum(a[x] * np.exp(2j * np.pi * x * y / 32) for x in range(32)) for y in range(32)])
check("A19", abs(np.sum(np.abs(A) ** 2) - 192) < 1e-9)
best = 1 + int(np.argmax(np.abs(A[1:16])))
check("A20", best == 13 and abs(A[16]) < 1e-9, f"|A13|={abs(A[13]):.3f} |A6|={abs(A[6]):.3f} |A7|={abs(A[7]):.3f}")

# ----------------------------------------------------------------------------------------------- Section B
c = np.array([1, 2j, -1, s2]) / np.sqrt(8)
check("B1", abs(np.sum(np.abs(c[2:]) ** 2) - 3 / 8) < 1e-12 and abs(np.linalg.norm(c) - 1) < 1e-12)
u, v = np.array([1, 1j]) / s2, np.array([1j, -1]) / s2
check("B2", abs(abs(np.vdot(u, v)) - 1) < 1e-12
      and all(abs(abs(np.vdot(u, wv / np.linalg.norm(wv))) - 1) > 1e-6
              for wv in (np.array([1, -1j]), np.array([1j, 1]))))
check("B3", all(abs(abs((H @ np.array([1, np.exp(1j * p)]) / s2)[0]) ** 2 - math.cos(p / 2) ** 2) < 1e-12
                for p in np.linspace(0, 6, 13)))
ok = True
for tt in np.linspace(0.1, 3, 5):
    for pp in np.linspace(0, 6, 5):
        psi = np.array([math.cos(tt / 2), np.exp(1j * pp) * math.sin(tt / 2)])
        val = np.vdot(psi, X @ psi) + 1j * np.vdot(psi, Y @ psi)
        ok &= abs(val - math.sin(tt) * np.exp(1j * pp)) < 1e-12
check("B4", ok)
check("B5", (not is_product([1, 1, 1, -1])) and is_product([1, 1, 1, 1]) and is_product([1, -1, 1, -1])
      and is_product([1, 1j, -1, -1j]))
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
check("B6", close(CNOT @ np.kron(plus, minus), np.kron(minus, minus)))
CR3 = np.diag([1, 1, 1, np.exp(1j * np.pi / 4)])
check("B7", close(np.kron(np.diag([1, 0]), I2) + np.kron(np.diag([0, 1]), R(3)), CR3))
phi, j = 0.3141, 3
Ud = np.diag([np.exp(2j * np.pi * phi), 1.0])  # |u> = |0> of this toy U
CU = np.kron(np.diag([1, 0]), np.eye(2)) + np.kron(np.diag([0, 1]), np.linalg.matrix_power(Ud, 2 ** j))
check("B8", close(CU @ np.kron(plus, k0), np.kron(np.array([1, np.exp(2j * np.pi * 2 ** j * phi)]) / s2, k0)))
st = np.array([1, 1, 0, 1]) / math.sqrt(3)
p0 = np.sum(np.abs(st[:2]) ** 2)
check("B9", abs(p0 - 2 / 3) < 1e-12 and close(st[:2] / np.sqrt(p0), plus))
unit = lambda M: close(M.conj().T @ M, np.eye(2))
check("B10", (not unit(np.array([[1, 1], [1j, 1j]]) / s2)) and unit(H) and unit(np.array([[1, 1j], [1j, 1]]) / s2)
      and unit(np.array([[0, 1], [1j, 0]])))
CR2 = np.diag([1, 1, 1, 1j])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
comm = lambda A_, B_: close(A_ @ B_, B_ @ A_)
check("B11", (not comm(np.kron(H, I2), CR2)) and comm(np.kron(np.diag([1, -1]), np.diag([1, -1])), CR2)
      and comm(SWAP, CR2) and comm(np.kron(np.diag([1, 1j]), I2), CR2))
check("B12", 2 ** 32 * 8 == 32 * 2 ** 30)
bell = (ket("00") - ket("11")) / s2
check("B13", abs(abs((np.kron(H, H) @ bell)[1]) ** 2 - 0.5) < 1e-12)
rho = 0.5 * np.eye(2)
check("B14", abs((H @ rho @ H)[0, 0] - 0.5) < 1e-12 and abs(abs((H @ plus)[0]) ** 2 - 1) < 1e-12)
check("B15", close(X @ k0, k1) and close(H @ k1, minus) and close(R(2) @ k1, 1j * k1) and close(R(3) @ k1, np.exp(1j * np.pi / 4) * k1))
pr = lambda s: np.round(np.abs(s) ** 2, 12)
check("B16", close(pr((ket("00") + ket("11")) / s2), [.5, 0, 0, .5]) and close(pr((ket("01") - ket("10")) / s2), [0, .5, .5, 0])
      and close(pr(np.kron((k0 + 1j * k1) / s2, minus)), [.25] * 4) and close(pr(np.kron(k0, plus)), [.5, .5, 0, 0]))
check("B17", close(pr(plus), pr(minus)) and abs(np.vdot(plus, minus)) < 1e-12)
check("B18", (not is_product(np.array([1, 0, 0, 1]) / s2)))
ok = True
for ph in np.linspace(0, 0.99, 12):
    ctrl = H @ (np.array([1, np.exp(2j * np.pi * ph)]) / s2)
    ok &= abs(abs(ctrl[1]) ** 2 - math.sin(math.pi * ph) ** 2) < 1e-12
check("B19", ok)
p1 = math.sin(math.pi / 3) ** 2
kmin = next(k for k in range(1, 20) if 1 - (1 - p1) ** k > 0.99)
check("B20", abs(p1 - 0.75) < 1e-12 and kmin == 4)

# ----------------------------------------------------------------------------------------------- Section C
check("C1", abs(F(4)[2, 3] - (-0.5)) < 1e-12)
check("C2", abs(abs(F(32)[7, 19]) ** 2 - 1 / 32) < 1e-12)
c3 = kron(k0 + k1, k0 - k1, k0 - 1j * k1) / np.sqrt(8)
check("C3", close(F(8) @ ket("110"), c3))
check("C4", 12 * 11 // 2 == 66)
check("C5", 66 + 3 * (12 // 2) == 84)
check("C6", abs(360 / 2 ** 6 - 5.625) < 1e-12)
# C7: qubit 1 before swaps carries 0.x1x2...xn -> after swaps it is the last factor of the product formula.
n = 4
ok = True
for x in range(16):
    bits = [(x >> (n - 1 - q)) & 1 for q in range(n)]
    last = sum(b / 2 ** (i + 1) for i, b in enumerate(bits))
    out = F(16) @ np.eye(16)[x]
    ok &= abs(out[1] / out[0] - np.exp(2j * np.pi * last)) < 1e-9  # y = 0001 flips only qubit n
check("C7", ok)
F8 = F(8)
check("C8", all(close(F8 @ F8 @ np.eye(8)[x], np.eye(8)[(-x) % 8]) for x in range(8)))
check("C9", close(F(16) @ (np.ones(16) / 4), np.eye(16)[0]))
n = 4
check("C10", close(F(16) @ np.eye(16)[8], kron(plus, plus, plus, minus)))
U3 = textbook_qft(3)
check("C11", close(U3, F(8)))  # circuit order verified, so its inverse begins with SWAP then H on qubit 3
check("C12", abs(((5 * 3 / 16) * 360) % 360 - 337.5) < 1e-9)
al = np.random.default_rng(0).normal(size=16) + 1j * np.random.default_rng(1).normal(size=16)
check("C13", close(F(16) @ al, np.sqrt(16) * np.fft.ifft(al)))
check("C14", all(np.linalg.matrix_rank((F(16) @ np.eye(16)[x]).reshape(2, 8), tol=1e-9) == 1
                 and np.linalg.matrix_rank((F(16) @ np.eye(16)[x]).reshape(4, 4), tol=1e-9) == 1 for x in range(16)))
check("C15", [nn * (nn - 1) // 2 for nn in (4, 6, 8, 10)] == [6, 15, 28, 45])
c16 = {"00": kron(k0 + k1, k0 + k1) / 2, "01": kron(k0 - k1, k0 + 1j * k1) / 2,
       "10": kron(k0 + k1, k0 - k1) / 2, "11": kron(k0 - k1, k0 - 1j * k1) / 2}
check("C16", all(close(F(4) @ ket(b), vv) for b, vv in c16.items()))
check("C17", all(close(np.abs(F(8) @ np.eye(8)[x]) ** 2, 1 / 8) for x in range(8)))
check("C18", textbook_qft(4).shape == (16, 16))
check("C19", abs(((0 + 90 + 45) % 360) - 135) < 1e-12 and abs(360 * 3 / 8 - 135) < 1e-12)
out3 = U3 @ ket("011")
noswap = np.array([np.exp(2j * np.pi * (3 / 8 * ((y >> 2) & 1) + 3 / 4 * ((y >> 1) & 1) + 1 / 2 * (y & 1)))
                   for y in range(8)]) / np.sqrt(8)
check("C20", abs(out3[6] - 1j / np.sqrt(8)) < 1e-12 and abs(noswap[6] - np.exp(1j * np.pi / 4) / np.sqrt(8)) < 1e-12)

# ----------------------------------------------------------------------------------------------- Section D
s = np.zeros(64, complex)
s[[3 + 8 * j for j in range(8)]] = 1 / np.sqrt(8)
b = F(64) @ s
pk = np.where(np.abs(b) ** 2 > 1e-12)[0]
check("D1", list(pk) == list(range(0, 64, 8)) and close(np.abs(b[pk]) ** 2, 1 / 8))
check("D2", abs((np.degrees(np.angle(b[16])) % 360) - 270) < 1e-9)
al = np.random.default_rng(2).normal(size=16) + 0j
beta = F(16) @ al
check("D3", close(F(16) @ np.roll(al, 5), np.exp(2j * np.pi * 5 * np.arange(16) / 16) * beta))
check("D4", 5 + math.ceil(math.log2(2 + 1 / (2 * 0.1))) == 8)
check("D5", abs(qpe_probs(5 / 8, 3)[5] - 1) < 1e-12)
worst = min(qpe_probs((bb + d) / 2 ** t, t)[bb] for t in (3, 5, 8) for bb in (1, 3)
            for d in np.linspace(-0.5, 0.5, 41) if abs(d) > 1e-9)
check("D6", worst >= 4 / np.pi ** 2 - 1e-12, f"worst={worst:.4f}, 4/pi^2={4 / np.pi ** 2:.4f}")
pq = qpe_probs(1 / 3, 2)
check("D7", abs(pq[1] - (6 + 3 * math.sqrt(3)) / 16) < 1e-12 and abs(pq.sum() - 1) < 1e-12, f"{pq}")
nb = math.ceil(math.log2(21))
check("D8", 3 * nb == 15)


def shor_probs(a, N, t):
    M = 2 ** t
    rr = next(r_ for r_ in range(1, N) if pow(a, r_, N) == 1)
    P = np.zeros(M)
    for x0 in range(rr):
        reg = np.zeros(M, complex)
        reg[x0::rr] = 1 / np.sqrt(M)
        P += np.abs(F(M).conj().T @ reg) ** 2
    return P, rr


P15, r15 = shor_probs(7, 15, 8)
check("D9", r15 == 4 and list(np.where(P15 > 1e-12)[0]) == [0, 64, 128, 192] and close(P15[[0, 64, 128, 192]], 0.25))
check("D10", [Fraction(y, 256).limit_denominator(15).denominator for y in (128, 64, 192, 0)] == [2, 4, 4, 1])
kept = sum(20 - k + 1 for k in range(2, 7))
check("D11", (190 - kept, kept) == (105, 85))
s = np.zeros(64)
s[list(range(0, 64, 6))] = 1
s /= np.linalg.norm(s)
check("D12", len(range(0, 64, 6)) == 11 and abs(abs((F(64) @ s)[32]) ** 2 - 11 / 64) < 1e-12)
ev8 = np.linalg.eigvals(F(8))
ok = all(min(abs(e - z) for z in (1, -1, 1j, -1j)) < 1e-9 for NN in (2, 4, 8, 16, 32) for e in np.linalg.eigvals(F(NN)))
check("D13", ok and all(min(abs(e - z) for e in ev8) < 1e-9 for z in (1, -1, 1j, -1j))
      and any(abs(e - 1j) < 1e-9 for e in np.linalg.eigvals(F(4))))
check("D14", True, "conceptual (semiclassical QFT)")
check("D15", True, "conceptual (applications)")
d16 = {"P": np.eye(16)[0], "Q": np.isin(np.arange(16), [0, 4, 8, 12]) / 2.0,
       "R": (np.arange(16) % 2 == 0) / np.sqrt(8), "S": np.ones(16) / 4}
pr16 = {key: np.round(np.abs(F(16) @ vv) ** 2, 12) for key, vv in d16.items()}
check("D16", close(pr16["P"], 1 / 16) and set(np.where(pr16["Q"] > 0)[0]) == {0, 4, 8, 12}
      and set(np.where(pr16["R"] > 0)[0]) == {0, 8} and abs(pr16["S"][0] - 1) < 1e-12)
# D17: approximate QFT error decreases with m (operator-norm distance to the exact QFT), n = 8
def aqft(n_, m_):
    N_ = 2 ** n_
    U_ = np.zeros((N_, N_), complex)
    for x in range(N_):
        bits = [(x >> (n_ - 1 - q)) & 1 for q in range(n_)]
        st_ = np.array([1.0 + 0j])
        for l in range(1, n_ + 1):  # output factor l uses bits x_{n-l+1..n}
            phase = sum(bits[n_ - l + i] / 2 ** (i + 1) for i in range(l) if i + 1 <= m_)
            st_ = np.kron(st_, np.array([1, np.exp(2j * np.pi * phase)]) / s2)
        U_[:, x] = st_
    return U_


errs = [np.linalg.norm(aqft(8, m_) - F(256), 2) for m_ in (2, 3, 4, 5, 6, 8)]
check("D17", all(e1 > e2 for e1, e2 in zip(errs, errs[1:])) and errs[-1] < 1e-9 and errs[3] < 8 * 2 * np.pi / 2 ** 5,
      f"errs={np.round(errs, 4)}")
check("D18", all((2 ** t) % 3 != 0 for t in range(1, 64)))
cands = {y: Fraction(y, 1024).limit_denominator(20).denominator for y in (171, 341, 512, 683)}
check("D19", cands == {171: 6, 341: 3, 512: 2, 683: 3} and pow(2, 6, 21) == 1 and gcd(7, 21) == 7 and gcd(9, 21) == 3,
      f"{cands}")
P21, r21 = shor_probs(2, 21, 10)
exact = (4 * 171 ** 2 + 2 * 170 ** 2) / 1024 ** 2
nbr = [P21[y] for y in (171, 341, 683, 853)]
check("D20", r21 == 6 and abs(P21[512] - exact) < 1e-12 and abs(exact - 1 / 6) < 1e-3,
      f"P(512)={P21[512]:.5f}, neighbours={np.round(nbr, 4)}")


def cf(p, q):
    out = []
    while q:
        out.append(p // q)
        p, q = q, p % q
    return out


check("D19-cf", cf(171, 1024) == [0, 5, 1, 84, 2], f"{cf(171, 1024)}")

# ------------------------------------------------------------------------------------- optional CUDA-Q cross-check
if "--cudaq" in sys.argv:
    import cudaq

    @cudaq.kernel
    def qft_kernel(bits: list[int]):
        n = len(bits)
        q = cudaq.qvector(n)
        for i in range(n):
            if bits[i] == 1:
                x(q[i])
        for j in range(n):
            h(q[j])
            for k in range(2, n - j + 1):
                r1.ctrl(2.0 * 3.141592653589793 / (2 ** k), q[j + k - 1], q[j])
        for i in range(n // 2):
            swap(q[i], q[n - 1 - i])

    def state_msb_first(bits):
        st_ = np.array(cudaq.get_state(qft_kernel, bits))
        n_ = len(bits)
        # CUDA-Q state index is little-endian (q0 = least significant); convert to q0 = MSB
        perm = [int(format(i, f"0{n_}b")[::-1], 2) for i in range(2 ** n_)]
        return st_[perm]

    for bitstr, qid in (("11", "C1"), ("110", "C3"), ("011", "C20"), ("1000", "C10")):
        bits = [int(ch) for ch in bitstr]
        st_ = state_msb_first(bits)
        ref = F(2 ** len(bits)) @ ket(bitstr)
        check(f"CUDA-Q {qid}", close(st_, ref, 1e-6), f"max dev {np.max(np.abs(st_ - ref)):.2e}")

passed = sum(ok_ for _, ok_, _ in CHECKS)
print(f"{passed}/{len(CHECKS)} checks passed")
for name, ok_, detail in CHECKS:
    if detail and ok_:
        print(f"  {name}: {detail}")
sys.exit(0 if passed == len(CHECKS) else 1)
