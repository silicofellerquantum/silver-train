"""Recompute every numerical / structural answer of the HHL paper and assert it matches the key.

Run:  python verify_answers.py           (NumPy / SciPy: linear algebra + a full simulation of the notebook's HHL circuit)
      python verify_answers.py --cudaq   (+ rebuilds the notebook's CUDA-Q HHL circuit and checks the default, 2x2 and n_c = 3 cases)

Conventions as in the paper and HHL.ipynb: clock (MSB first) + system + ancilla; U = exp(iAt); k = lambda t 2^n_c/(2 pi), signed;
C = 2 pi/(2^n_c t); inversion R_Y(2 arcsin(C/lambda_k)); success = clock 0 and ancilla 1.
Each computed value is also checked against the text of the CORRECT option in the bank, and answer_key.csv against the generator.
"""
import csv
import math
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import bank_advanced
import bank_basics
import bank_math
import bank_qm

HERE = Path(__file__).resolve().parent
BANK = {q["id"]: q for m in (bank_math, bank_qm, bank_basics, bank_advanced) for q in m.QUESTIONS}
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond)))
    if not cond:
        print(f"FAIL {name}: {detail}")


def key_has(qid, *frags):
    q = BANK[qid]
    t = q["opts"][q["correct"]] if q["kind"] in ("ar", "stmt") else q["opts"][0]
    miss = [f for f in frags if f not in t]
    check(qid + " (option text)", not miss, f"{miss} not in {t!r}")


def fixed(qid, idx):
    check(qid + " (AR/statement index)", BANK[qid]["correct"] == idx)


close = lambda a, b, t=1e-9: np.allclose(a, b, atol=t)
H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
A_DEF = H4 @ np.diag([1.0, 2, 3, 4]) @ H4
B_DEF = np.array([1.0, 0, 0, 0])
X_DEF = np.linalg.solve(A_DEF, B_DEF)


def qpe_probs(phi, t):
    M = 2 ** t
    return np.array([abs(np.sum(np.exp(2j * np.pi * np.arange(M) * (phi - y / M))) / M) ** 2 for y in range(M)])


def hhl_full(A, b, n_c, t):
    """State-vector simulation of the notebook's HHL circuit; returns post-selected branch, P_success, C."""
    d = A.shape[0]; M = 2 ** n_c
    U = expm(1j * A * t)
    F = np.exp(2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M) / np.sqrt(M)
    psi = np.zeros((M, d, 2), complex)
    for c in range(M):                                            # H on clock + controlled U^c
        psi[c, :, 0] = np.linalg.matrix_power(U, c) @ (b / np.linalg.norm(b)) / np.sqrt(M)
    psi = np.einsum("kc,cda->kda", F.conj().T, psi)               # inverse QFT
    C = 2 * np.pi / (M * t)
    for k in range(1, M):                                         # eigenvalue inversion (lookup table)
        signed = k - M if k >= M // 2 else k
        th = 2 * np.arcsin(np.clip(C / (2 * np.pi * signed / (M * t)), -1, 1))
        R = np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]])
        psi[k] = psi[k] @ R.T
    psi = np.einsum("kc,cda->kda", F, psi)                        # uncompute: QFT, U^-c, H
    for c in range(M):
        Uc = np.linalg.matrix_power(U.conj().T, c)
        psi[c] = (Uc @ psi[c])
    Hn = np.ones((1, 1))
    for _ in range(n_c):
        Hn = np.kron(Hn, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    psi = np.einsum("kc,cda->kda", Hn, psi)
    branch = psi[0, :, 1]
    return branch, float(np.linalg.norm(branch) ** 2), C


fid = lambda x, y: abs(np.vdot(x / np.linalg.norm(x), y / np.linalg.norm(y))) ** 2

# ----------------------------------------------------------------------------------------------- Section A
check("A1", close(np.linalg.solve([[3, 1], [1, 3]], [1, -1]), [0.5, -0.5])); key_has("A1", "0.5", "-0.5")
A2 = np.array([[1, -1 / 3], [-1 / 3, 1]])
check("A2", close(np.linalg.eigvalsh(A2), [2 / 3, 4 / 3]) and math.isclose(np.linalg.cond(A2), 2)); key_has("A2", r"\kappa = 2")
key_has("A3", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1")
check("A4", [np.linalg.cond(np.diag([0.5, 2, 8])), np.linalg.cond(A2), np.linalg.cond(A_DEF), np.linalg.cond(2 * np.eye(2))] == [16, 2, 4, 1]
      or close([np.linalg.cond(np.diag([0.5, 2, 8])), np.linalg.cond(A2), np.linalg.cond(A_DEF), np.linalg.cond(2 * np.eye(2))], [16, 2, 4, 1]))
key_has("A4", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
check("A5", math.isclose(X_DEF[0], 25 / 48) and close(X_DEF, [0.520833, 0.145833, 0.229167, 0.104167], 1e-6)); key_has("A5", "25}{48")
An = np.array([[1, 2], [0, 1.]]); E = np.block([[np.zeros((2, 2)), An], [An.T, np.zeros((2, 2))]])
ev = np.abs(np.linalg.eigvalsh(E))
check("A6", math.isclose(ev.max() / ev.min(), 3 + 2 * np.sqrt(2))); key_has("A6", r"3 + 2\sqrt2")
fixed("A7", 0)
check("A7 (math)", all(np.linalg.cond(np.random.default_rng(k).normal(size=(3, 3))) >= 1 for k in range(5)))
fixed("A8", 3)
check("A9", (np.abs(A_DEF) > 1e-12).sum(1).max() == 3); key_has("A9", "$3$ and $3$")
check("A10", (2000 / 1000) ** 3 == 8); key_has("A10", "$8$")
key_has("A11", r"\sqrt\kappa")
b12 = np.array([1, 2, 2, 0.]); check("A12", math.isclose((b12[1] / np.linalg.norm(b12)) ** 2, 4 / 9)); key_has("A12", r"\tfrac49")
check("A13", close(H4.T @ B_DEF, [0.5] * 4)); key_has("A13", r"\tfrac12, \tfrac12")
x2 = np.linalg.solve(A2, [1, 0])
check("A14", close(x2, [9 / 8, 3 / 8])); key_has("A14", r"\tfrac98", r"\tfrac38")
check("A15", close(x2 / np.linalg.norm(x2), np.array([3, 1]) / np.sqrt(10)) and math.isclose((x2[0] / np.linalg.norm(x2)) ** 2, 0.9))
key_has("A15", r"(3, 1)", "0.9")
check("A16", close(expm(1j * np.diag([1., 2]) * np.pi / 2), np.diag([1j, -1]))); key_has("A16", r"\operatorname{diag}(i,\ -1)")
Mh = np.array([[1, 2 - 1j], [2 + 1j, 3]]); Mn = np.array([[1, 2], [0, 1.]])
check("A17", close(expm(1j * Mh * 0.7).conj().T @ expm(1j * Mh * 0.7), np.eye(2)) and not close(expm(1j * Mn * 0.7).conj().T @ expm(1j * Mn * 0.7), np.eye(2)))
key_has("A17", "Hermitian")
check("A18", 0.25 <= np.linalg.norm(X_DEF) <= 1 and abs(np.linalg.norm(X_DEF) - 0.5966) < 1e-4); key_has("A18", "0.25", "0.5966")
check("A19", all(abs(np.linalg.det([[a, 1], [1, a]])) < 1e-12 for a in (1, -1))); key_has("A19", r"\pm1")
fixed("A20", 2)
check("A20 (math)", math.isclose(np.linalg.cond(3.7 * A_DEF), np.linalg.cond(A_DEF)) and close(np.linalg.eigvalsh(3 * A_DEF), 3 * np.linalg.eigvalsh(A_DEF)))

# ----------------------------------------------------------------------------------------------- Section B
p = qpe_probs(3 / 16, 4); check("B1", np.argmax(p) == 3 and math.isclose(p[3], 1)); key_has("B1", r"\ket{0011}", "probability 1")
check("B2", math.isclose(4 * (np.pi / 8) * 16 / (2 * np.pi), 4)); key_has("B2", "k = 4")
key_has("B3", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
key_has("B4", r"e^{i\phi}\ket1")
F4 = np.exp(2j * np.pi * np.outer(range(4), range(4)) / 4) / 2
check("B5", close(F4.conj().T @ (np.array([1, 1j, -1, -1j]) / 2), [0, 1, 0, 0])); key_has("B5", r"\ket{01}")
check("B6", 2 * comb(4, 2) == 12 and 2 * (4 // 2) == 4); key_has("B6", "$12$ and $4$")
check("B7", math.isclose(np.sin(np.arcsin(0.5)) ** 2, 0.25)); key_has("B7", r"\tfrac14")
check("B8", [i for i in range(4) if not (5 >> (3 - i)) & 1] == [0, 2]); key_has("B8", "clock[0] and clock[2]")
fixed("B9", 0)
U0 = expm(1j * A_DEF * 0.3); check("B9 (math)", close(np.linalg.matrix_power(U0, 4), expm(1j * A_DEF * 1.2)))
sg = lambda k: k - 16 if k >= 8 else k
check("B10", [sg(3), sg(8), sg(12), sg(15)] == [3, -8, -4, -1]); key_has("B10", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1")
key_has("B11", "probability $p$")
key_has("B12", "eigenvector")
check("B13", 16 * 2 ** 7 == 2048 and 2 * 2048 == 4096); key_has("B13", "2048", "4096")
key_has("B14", "O(2^n)")
fixed("B15", 1)
check("B16", math.isclose(2 * 0.9 - 1, 0.8)); key_has("B16", "0.8")
p = qpe_probs(0.25, 2); check("B17", math.isclose(p[1], 1)); key_has("B17", r"\ket{01}", "probability 1")
p = qpe_probs(1 / 3, 2); check("B18", np.argmax(p) == 1 and abs(p[1] - 0.70) < 0.005, p); key_has("B18", r"\ket{01}", "0.70")
fixed("B19", 2)
check("B20", close(np.linalg.eigvalsh(A_DEF + np.eye(4)) * (np.pi / 8) * 16 / (2 * np.pi), [2, 3, 4, 5])); key_has("B20", "increases by $1$")

# ----------------------------------------------------------------------------------------------- Section C
br, P_def, C_def = hhl_full(A_DEF, B_DEF, 4, np.pi / 8)
check("C1 (circuit sim)", fid(br, X_DEF) > 1 - 1e-12 and close(br / C_def * np.linalg.norm(B_DEF), X_DEF))
key_has("C1", r"\tfrac{C}{\lambda_j}\ket1")
check("C2", 5 + 3 + 1 == 9); key_has("C2", "$9$")
key_has("C3", r"|C/\lambda_j| \le 1")
check("C4", math.isclose(2 * np.arcsin(0.5), np.pi / 3)); key_has("C4", r"\tfrac\pi3")
check("C5", abs(P_def - 0.355903) < 1e-6 and math.isclose(0.25 * (1 + 1 / 4 + 1 / 9 + 1 / 16), P_def, rel_tol=1e-9)); key_has("C5", "0.3559")
key_has("C6", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
t7 = 2 * np.pi / (16 * 0.5); check("C7", math.isclose(t7, np.pi / 4) and math.isclose(2 * np.pi / (16 * t7), 0.5) and 3 / 0.5 <= 7)
key_has("C7", r"t = \pi/4", "C = 0.5")
key_has("C8", "mixture")
fixed("C9", 0)
check("C10", 2 ** 4 - 1 == 15); key_has("C10", "2^{n_c} - 1")
check("C11", math.isclose(2 * np.pi * sg(12) / (16 * np.pi / 8), -4)); key_has("C11", "$-4$")
fixed("C12", 2)
t13 = 2 * np.pi / (16 * 2 / 3); br2, P2, C2 = hhl_full(A2, np.array([1., 0]), 4, t13)
check("C13", math.isclose(t13, 3 * np.pi / 16) and math.isclose(C2, 2 / 3) and close(np.linalg.eigvalsh(A2) * t13 * 16 / (2 * np.pi), [1, 2])
      and fid(br2, x2) > 1 - 1e-12 and abs(P2 - 0.625) < 1e-9)
key_has("C13", r"\tfrac{3\pi}{16}", r"\tfrac23", "$1$ and $2$")
key_has("C14", "ancilla bit is 1")
check("C15", abs(np.sqrt(0.355903 * 0.644097 / 20000) - 0.0034) < 5e-5); key_has("C15", "0.0034")
key_has("C16", r"|x_i|^2/\|x\|^2")
check("C17", 8 + 4 + 12 + 15 == 39 and 16 + 8 + 4 + 12 + 56 + 15 == 111 and 2 * sum(4 - bin(k).count("1") for k in range(1, 16)) == 56)
key_has("C17", "$39$")
key_has("C18", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
Ac = np.diag([1., 2]); bc = np.array([1., 1]) / np.sqrt(2)
brc, Pc, Cc = hhl_full(Ac, bc, 3, np.pi / 4)
check("C19", math.isclose(Cc, 1) and close(np.diag(Ac) * (np.pi / 4) * 8 / (2 * np.pi), [1, 2])); key_has("C19", r"\ket0\ket{001}", r"\ket1\ket{010}")
check("C20", abs(Pc - 0.625) < 1e-9 and close(brc / np.linalg.norm(brc), np.array([2, 1]) / np.sqrt(5))); key_has("C20", "0.625", "(2, 1)")

# ----------------------------------------------------------------------------------------------- Section D
key_has("D1", r"s^2\kappa^2/\varepsilon")
check("D2", math.isclose(1 / 0.01, 100) and math.isclose(1 / np.sqrt(0.01), 10)); key_has("D2", "$100$", "$10$")
check("D3", 1e6 < 1024 / 0.01 ** 2 < 2e7); key_has("D3", "10^7")
nmin = lambda kap: min(n for n in range(2, 12) if 2 ** (n - 1) - 1 >= kap)
check("D4", [nmin(3), nmin(7), nmin(20), nmin(40)] == [3, 4, 6, 7]); key_has("D4", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3")
check("D5", 2 ** 8 - 1 == 255 and 8 + 2 + 1 == 11); key_has("D5", "$255$", "$11$")
check("D6", 4 + 2 + 1 == 7); key_has("D6", "$7$ qubits", "second half")
Apad = np.eye(4); Apad[:3, :3] = np.diag([5., 10, 20])
check("D7", math.isclose(np.linalg.cond(Apad), 20) and math.isclose(20 / 5, 4)); key_has("D7", "$4$ to $20$")
key_has("D8", "low-rank")
fixed("D9", 3)
check("D10", math.isclose((1 / 20) ** 2, 0.0025)); key_has("D10", "0.0025")
check("D11", sg(9) == -7 and 9 * np.pi / 8 > np.pi); key_has("D11", "$-7$")
key_has("D12", r"P$\to$2, Q$\to$3, R$\to$4, S$\to$1")
key_has("D13", "global phase")
fixed("D14", 0)
t3 = 2 * np.pi * 3 / (8 * 4); br3, P3, _ = hhl_full(A_DEF, B_DEF, 3, t3)
check("D15", close(np.array([1, 2, 3, 4]) * t3 * 8 / (2 * np.pi), [0.75, 1.5, 2.25, 3]) and abs(fid(br3, X_DEF) - 0.9532) < 1e-4, fid(br3, X_DEF))
key_has("D15", "0.75, 1.5, 2.25, 3", "below 1")
check("D16", 16 * 2 ** 11 == 32768 and 2 * 32768 == 65536); key_has("D16", "32\\,768", "65.5")
src = (HERE.parent.parent / "dev" / "notebook_builders" / "build_hhl.py").read_text(encoding="utf-8")
check("D17", "U_step = expm(1j * A_work * t_evolution)" in src and "cudaq.register_operation(name, np.linalg.matrix_power(U, 2 ** j))" in src)
key_has("D17", "scipy.linalg.expm")
key_has("D18", r"x^\dagger Mx")
check("D19", abs(np.sqrt(0.7622 * 0.2378 / 7086) - 0.0051) < 5e-5 and abs(0.0069 / 0.0051 - 1.4) < 0.06); key_has("D19", "0.0051", "1.4")
check("D20", abs(np.sqrt(0.3543) - 0.595) < 5e-4); key_has("D20", "0.595")
check("notebook source: hardware refused", 'raise ValueError("This HHL implementation uses exact matrix exponentials' in src)
check("notebook source: post-selection", 'bits[:n_c] == "0" * n_c and bits[-1] == "1"' in src)
check("notebook source: auto time", "t_evolution = 2 * np.pi / (2 ** n_c * lam_abs_min)" in src)

# ----------------------------------------------------------------------------------------------- CUDA-Q
if "--cudaq" in sys.argv:
    import hashlib
    import warnings
    warnings.filterwarnings("ignore")
    import cudaq
    cudaq.set_target("qpp-cpu")

    def bit_reverse_order(vector, n):
        return np.asarray(vector).reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)

    def notebook_hhl(A, b, n_c, t):
        """The notebook's build_hhl_kernel (measure=False), copied line for line; returns post-selected amplitudes."""
        n_b = int(np.log2(A.shape[0])); n_tot = n_c + n_b + 1
        b_norm = float(np.linalg.norm(b)); U = expm(1j * A * t)
        uid = hashlib.sha1(np.r_[A.real.ravel(), A.imag.ravel(), t, n_c].tobytes()).hexdigest()[:10]
        kernel = cudaq.make_kernel()
        initial = np.kron(np.kron(np.eye(2 ** n_c)[0], b / b_norm), np.array([1.0, 0.0]))
        reg = kernel.qalloc(bit_reverse_order(initial.astype(complex), n_tot))
        clock = [reg[i] for i in range(n_c)]; system = [reg[n_c + i] for i in range(n_b)]; anc = reg[n_tot - 1]
        for q in clock:
            kernel.h(q)
        for j in range(n_c):
            name = f"v_{uid}_{n_c}_u{j}"
            cudaq.register_operation(name, np.linalg.matrix_power(U, 2 ** j))
            getattr(kernel, name)(clock[n_c - 1 - j], *system)
        for j in range(n_c // 2):
            kernel.swap(clock[j], clock[n_c - 1 - j])
        for jj in range(n_c):
            j = n_c - 1 - jj
            for kk in range(n_c - 1 - j):
                k = n_c - 1 - kk
                kernel.cr1(-np.pi / 2 ** (k - j), clock[k], clock[j])
            kernel.h(clock[j])
        C = 2 * np.pi / (2 ** n_c * t)
        for kv in range(1, 2 ** n_c):
            signed = kv - 2 ** n_c if kv >= 2 ** (n_c - 1) else kv
            theta = 2 * np.arcsin(np.clip(C / (2 * np.pi * signed / (2 ** n_c * t)), -1.0, 1.0))
            zero = [i for i in range(n_c) if not (kv >> (n_c - 1 - i)) & 1]
            for i in zero:
                kernel.x(clock[i])
            kernel.cry(theta, clock, anc)
            for i in zero:
                kernel.x(clock[i])
        for j in range(n_c):
            kernel.h(clock[j])
            for k in range(j + 1, n_c):
                kernel.cr1(np.pi / 2 ** (k - j), clock[k], clock[j])
        for j in range(n_c // 2):
            kernel.swap(clock[j], clock[n_c - 1 - j])
        for jj in range(n_c):
            j = n_c - 1 - jj
            name = f"v_{uid}_{n_c}_uinv{j}"
            cudaq.register_operation(name, np.linalg.matrix_power(U.conj().T, 2 ** j))
            getattr(kernel, name)(clock[n_c - 1 - j], *system)
        for q in clock:
            kernel.h(q)
        sv = np.array(cudaq.get_state(kernel))
        tensor = sv.reshape([2] * n_tot).transpose(list(range(n_tot))[::-1])
        return tensor[(0,) * n_c + (slice(None),) * n_b + (1,)].reshape(-1), C, b_norm

    amp, C, bn = notebook_hhl(A_DEF.astype(complex), B_DEF.astype(complex), 4, np.pi / 8)
    check("cudaq default: x recovered", close(amp * bn / C, X_DEF, 1e-10) and abs(np.linalg.norm(amp) ** 2 - 0.355903) < 1e-6)
    amp2, C2q, bn2 = notebook_hhl(A2.astype(complex), np.array([1, 0], complex), 4, t13)
    check("cudaq 2x2", close(amp2 * bn2 / C2q, [1.125, 0.375], 1e-10))
    amp3, _, _ = notebook_hhl(A_DEF.astype(complex), B_DEF.astype(complex), 3, t3)
    check("cudaq n_c=3 fidelity", abs(fid(amp3, X_DEF) - 0.9532157640) < 1e-8, fid(amp3, X_DEF))
    ampc, _, _ = notebook_hhl(Ac.astype(complex), bc.astype(complex), 3, np.pi / 4)
    check("cudaq passage C", abs(np.linalg.norm(ampc) ** 2 - 0.625) < 1e-9 and close(ampc / np.linalg.norm(ampc), np.array([2, 1]) / np.sqrt(5)))
    print("CUDA-Q cross-checks done (target qpp-cpu)")

# ----------------------------------------------------------------------------------------------- answer key file
import generate_papers
qs, _, _ = generate_papers.load_questions()
with open(HERE / "answer_key.csv") as fh:
    rows = list(csv.DictReader(fh))
check("answer_key.csv", [(r["bank_id"], r["answer"]) for r in rows] == [(q["id"], "ABCD"[q["answer"]]) for q in qs])
check("80 questions, 4 distinct options", len(qs) == 80 and all(len(set(q["opts"])) == 4 for q in qs))
bad = [c for c in CHECKS if not c[1]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
sys.exit(1 if bad else 0)
