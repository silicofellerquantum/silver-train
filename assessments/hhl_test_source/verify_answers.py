"""Independent verification of every numerical claim behind the HHL answer key.

    python verify_answers.py           # NumPy checks (linear algebra, QPE, HHL state-vector simulation)
    python verify_answers.py --cudaq   # additionally rebuilds the notebook's HHL kernel in CUDA-Q and compares

Each check recomputes a quantity from scratch (it does not read the answer from the bank) and compares it with the
value stated in the correct option. At the end the answer key is rebuilt with generate_papers.load_questions() and
checked for format, balance and agreement with answer_key.csv.
"""
import csv
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex), atol=tol)


H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
A_DEF = H4 @ np.diag([1.0, 2, 3, 4]) @ H4
B_DEF = np.array([1.0, 0, 0, 0])


# ------------------------------------------------------------------------------------- reference HHL simulator
def qft_matrix(n):
    N = 2 ** n
    return np.array([[np.exp(2j * np.pi * x * y / N) for x in range(N)] for y in range(N)]) / np.sqrt(N)


def hhl_numpy(A, b, n_c, t, signed=True):
    """Exact state-vector HHL with the notebook's register order clock | system | ancilla (clock[0] = MSB).
    Returns (post-selected system amplitudes, C)."""
    d = A.shape[0]
    n_b = int(np.log2(d))
    Nc = 2 ** n_c
    C = 2 * np.pi / (Nc * t)
    U = expm(1j * A * t)
    # QPE as one block-diagonal operator: |k> |psi>  ->  controlled-U^k
    cu = np.zeros((Nc * d, Nc * d), dtype=complex)
    for k in range(Nc):
        cu[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.linalg.matrix_power(U, k)
    Hn = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hall = Hn
    for _ in range(n_c - 1):
        Hall = np.kron(Hall, Hn)
    F = qft_matrix(n_c)
    qpe = np.kron(F.conj().T, np.eye(d)) @ cu @ np.kron(Hall, np.eye(d))
    state = np.kron(np.eye(Nc)[0], b / np.linalg.norm(b)).astype(complex)
    state = qpe @ state
    # controlled rotation on the ancilla
    out = np.zeros(2 * Nc * d, dtype=complex)
    for k in range(Nc):
        s = k - Nc if (signed and k >= Nc // 2) else k
        block = state[k * d:(k + 1) * d]
        if s == 0:
            out[(k * d) * 2:(k * d + d) * 2:2] = block            # ancilla stays |0>
            continue
        a1 = np.clip(C / (2 * np.pi * s / (Nc * t)), -1, 1)                # C / lambda_k, lambda_k = s * C
        a0 = np.sqrt(1 - a1 ** 2)
        for i in range(d):
            out[2 * (k * d + i)] = a0 * block[i]
            out[2 * (k * d + i) + 1] = a1 * block[i]
    # uncompute QPE (ancilla is the last, least significant qubit)
    full_inv = np.kron(qpe.conj().T, np.eye(2))
    out = full_inv @ out
    post = out.reshape(Nc, d, 2)[0, :, 1]
    return post, C


# ------------------------------------------------------------------------------------- Section A
A1 = np.array([[2, 1], [1, 3.]])
check("A1 solution (0.2, 0.6)", close(np.linalg.solve(A1, [1, 2]), [0.2, 0.6]))
ev = np.linalg.eigvalsh(np.array([[5, 2], [2, 2.]]))
check("A2 eigenvalues 1, 6; kappa 6", close(ev, [1, 6]) and np.isclose(ev[-1] / ev[0], 6))
P_, Q_, R_, S_ = (np.array([[0, 1], [1, 0]]), np.array([[1, 2], [2, 4.]]), np.array([[1, 1j], [-1j, 2]]),
                  np.array([[0, 1], [0, 0.]]))
herm = lambda M: np.allclose(M, M.conj().T)
unit = lambda M: np.allclose(M.conj().T @ M, np.eye(2))
check("A3 P Hermitian+unitary", herm(P_) and unit(P_))
check("A3 Q Hermitian, singular", herm(Q_) and np.isclose(np.linalg.det(Q_), 0))
check("A3 R Hermitian, invertible, not unitary", herm(R_) and not np.isclose(np.linalg.det(R_), 0) and not unit(R_))
check("A3 S not diagonalisable (rank of S - 0 I is 1, eigenvalue 0 twice)",
      np.linalg.matrix_rank(S_) == 1 and close(np.linalg.eigvals(S_), [0, 0]))
check("A4 kappa 100", np.isclose(np.linalg.cond(np.diag([1, .5, .01, .2])), 100))
check("A5 eigenvalues 1, 4", close(np.linalg.eigvalsh(np.array([[3, 1 - 1j], [1 + 1j, 2]])), [1, 4]))
check("A6 (A^-1)_12 = -1/3", np.isclose(np.linalg.inv(np.array([[2, 1], [1, 2.]]))[0, 1], -1 / 3))
J = np.array([[1, 1], [0, 1.]])
check("A7 [[1,1],[0,1]] has real eigenvalues but is defective", np.linalg.matrix_rank(J - np.eye(2)) == 1)
U_rand = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)) + 1j * np.random.default_rng(2).normal(size=(4, 4)))[0]
M_rand = np.random.default_rng(3).normal(size=(4, 4))
check("A8 kappa(U) = 1 and kappa(A^-1) = kappa(A)", np.isclose(np.linalg.cond(U_rand), 1)
      and np.isclose(np.linalg.cond(np.linalg.inv(M_rand)), np.linalg.cond(M_rand)))
L8 = 2 * np.eye(8) - np.eye(8, k=1) - np.eye(8, k=-1)
check("A9 sparsity 3, nnz 22", (L8 != 0).sum(axis=1).max() == 3 and (L8 != 0).sum() == 22)
L3 = 2 * np.eye(3) - np.eye(3, k=1) - np.eye(3, k=-1)
check("A10 kappa = 3 + 2 sqrt 2", np.isclose(np.linalg.cond(L3), 3 + 2 * np.sqrt(2)))
bb = np.array([3, 0, 4, 0.]) / 5
check("A11 P(10) = 0.64", np.isclose(bb[2] ** 2, 0.64))
check("A12 x = (2/3, -1/3)", close(np.linalg.solve(np.array([[2, 1], [1, 2.]]), [1, 0]), [2 / 3, -1 / 3]))
D13 = np.diag([4, 1, .5, 3])
check("A13 norms 4, 2; kappa 8; det 6", np.isclose(np.linalg.norm(D13, 2), 4) and np.isclose(np.linalg.norm(np.linalg.inv(D13), 2), 2)
      and np.isclose(np.linalg.cond(D13), 8) and np.isclose(np.linalg.det(D13), 6))
A14m = np.array([[1, 2], [0, 1.]])
Aemb = np.block([[np.zeros((2, 2)), A14m], [A14m.T, np.zeros((2, 2))]])
check("A14 embedded eigenvalues +-(sqrt2+1), +-(sqrt2-1)",
      close(np.linalg.eigvalsh(Aemb), [-(np.sqrt(2) + 1), -(np.sqrt(2) - 1), np.sqrt(2) - 1, np.sqrt(2) + 1]))
Ut = expm(1j * np.array([[0, 1], [0, 0.]]) * 0.7)
check("A15 e^{iAt} not unitary for real non-symmetric A; unitary for Hermitian A",
      not np.allclose(Ut.conj().T @ Ut, np.eye(2)) and np.allclose(expm(1j * R_ * .7).conj().T @ expm(1j * R_ * .7), np.eye(2)))
# A16: the bound is attained -> worst case equals kappa * relative error
Dk = np.diag([1000.0, 1.0])
x0 = np.linalg.solve(Dk, [1000.0, 0]); dx = np.linalg.solve(Dk, [0, 1e-3])
check("A16 bound kappa*1e-6 = 1e-3 attained", np.isclose(np.linalg.norm(dx) / np.linalg.norm(x0), 1e-3))
Ap = np.array([[1, -1 / 3], [-1 / 3, 1]])
evp = np.linalg.eigvalsh(Ap)
check("A17 eigenvalues 2/3, 4/3, kappa 2", close(evp, [2 / 3, 4 / 3]) and np.isclose(evp[1] / evp[0], 2))
xp = np.linalg.solve(Ap, [1, 0])
check("A18 x = (9/8, 3/8), P(0) = 0.9", close(xp, [9 / 8, 3 / 8]) and np.isclose(xp[0] ** 2 / (xp @ xp), 0.9))
check("A19 5x5 non-Hermitian -> 4 system qubits", int(np.ceil(np.log2(2 * 5))) == 4)
Hpd = np.array([[2, 1j], [-1j, 3.]])
check("A20 singular values = |eigenvalues| for Hermitian A",
      close(np.sort(np.linalg.svd(Hpd, compute_uv=False)), np.sort(np.abs(np.linalg.eigvalsh(Hpd)))))

# ------------------------------------------------------------------------------------- Section B
check("B1 P(1) = 4/5", np.isclose(abs(2j / np.sqrt(5)) ** 2, 0.8))
check("B2 128 amplitudes, 2048 bytes", 2 ** 7 == 128 and np.zeros(128, dtype=np.complex128).nbytes == 2048)
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
check("B3 HZH = X", close(Hd @ Z @ Hd, X))
RY = lambda th: np.array([[np.cos(th / 2), -np.sin(th / 2)], [np.sin(th / 2), np.cos(th / 2)]])
check("B4 R_Y matches", close(RY(np.pi) @ [1, 0], [0, 1]) and close(RY(np.pi / 2) @ [1, 0], [1 / np.sqrt(2)] * 2)
      and close(RY(-np.pi / 2) @ [1, 0], [1 / np.sqrt(2), -1 / np.sqrt(2)]) and close(RY(np.pi / 3) @ [1, 0], [np.sqrt(3) / 2, .5]))
det2 = lambda v: v[0] * v[3] - v[1] * v[2]
check("B5 only (1,1,1,-1)/2 entangled", det2([1, 1, 1, -1]) != 0 and all(det2(v) == 0 for v in
      ([1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1])))
phi = 0.9
CU = np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), np.diag([np.exp(1j * phi), np.exp(-1j * phi)])]])
out = CU @ np.kron([1, 1], [1, 0]) / np.sqrt(2)
check("B6 phase kickback", close(out, np.kron([1, np.exp(1j * phi)], [1, 0]) / np.sqrt(2)))
psi = np.array([np.exp(2j * np.pi * 3 * y / 4) for y in range(4)]) / 2
check("B7 inverse QFT gives |11>", close(qft_matrix(2).conj().T @ psi, np.eye(4)[3]))


def qpe_probs(phi_, t):
    N = 2 ** t
    return np.array([abs(sum(np.exp(2j * np.pi * m * (phi_ - k / N)) for m in range(N)) / N) ** 2 for k in range(N)])


check("B8 phi = 3/8 -> 011 with probability 1", np.isclose(qpe_probs(3 / 8, 3)[3], 1))
p9 = qpe_probs(1 / 3, 3)
check("B9 phi = 1/3 -> k = 3 with ~0.69", p9.argmax() == 3 and abs(p9[3] - 0.688) < 5e-3 and abs(p9[2] - 0.175) < 5e-3)
check("B10 unitary preserves norm", np.isclose(np.linalg.norm(U_rand @ np.ones(4)), 2))
check("B11 global phase invisible, |+>,|-> distinguishable", close(abs(np.exp(1j) * Hd @ [1, 0]) ** 2, abs(Hd @ [1, 0]) ** 2)
      and close(Hd @ (Hd @ [1, 0]), [1, 0]) and close(Hd @ (Hd @ [0, 1]), [0, 1]))
check("B12 P(ancilla=1) = 1/4", np.isclose((RY(2 * np.arcsin(.5)) @ [1, 0])[1] ** 2, .25))
k, n = 2, 3
zero_bits = [i for i in range(n) if not (k >> (n - 1 - i)) & 1]
check("B13 X on c0 and c2 (4 gates)", zero_bits == [0, 2] and 2 * len(zero_bits) == 4)
check("B14 P = 3/4", np.isclose((np.sqrt(3) / 2) ** 2, .75))
U1, U2, U3 = (np.linalg.qr(np.random.default_rng(s).normal(size=(3, 3)) + 1j * np.random.default_rng(s + 10).normal(size=(3, 3)))[0]
               for s in (5, 6, 7))   # generic, non-commuting unitaries
W = U3 @ U2 @ U1
dag = lambda M: M.conj().T
check("B15 inverse = U3^dag then U2^dag then U1^dag", close(dag(U1) @ dag(U2) @ dag(U3) @ W, np.eye(3))
      and not close(dag(U3) @ dag(U2) @ dag(U1) @ W, np.eye(3)) and not close(U1 @ U2 @ U3 @ W, np.eye(3)))
check("B16 4-qubit QFT: 4 H, 6 CP, 2 SWAP, smallest angle pi/8",
      (4, 4 * 3 // 2, 4 // 2) == (4, 6, 2) and np.isclose(2 * np.pi / 2 ** 4, np.pi / 8))
H4q = np.kron(np.kron(Hd, Hd), np.kron(Hd, Hd))
check("B17 amplitude 1/4", np.isclose((H4q @ np.eye(16)[0])[0b1011], .25))
bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
rhoA = np.einsum("ij,kj->ik", bell.reshape(2, 2), bell.reshape(2, 2).conj())
check("B18 Bell reduced state I/2, Bell entangled", close(rhoA, np.eye(2) / 2) and det2(bell) != 0)
clock = np.array([np.exp(2j * np.pi * .25 * kk) for kk in range(4)]) / 2
check("B19 amplitude of |11> = -i/2", np.isclose(clock[3], -.5j))
check("B20 outcome 01 with certainty", close(np.abs(qft_matrix(2).conj().T @ clock) ** 2, [0, 1, 0, 0]))

# ------------------------------------------------------------------------------------- Section C
evals, evecs = np.linalg.eigh(A_DEF)
beta = evecs.T @ B_DEF
check("C2/C14 eigenvalues of the default A are 1..4 and all |beta_j| = 1/2", close(evals, [1, 2, 3, 4]) and close(np.abs(beta), [.5] * 4))
t_def, nc_def = np.pi / 8, 4
check("C3 lambda = 3 -> 0011", format(round(3 * t_def * 2 ** nc_def / (2 * np.pi)), "04b") == "0011")
C_def = 2 * np.pi / (2 ** nc_def * t_def)
check("C4 C = 1, theta = pi/3", np.isclose(C_def, 1) and np.isclose(2 * np.arcsin(C_def / 2), np.pi / 3))
check("C5 amplitude 1/4, probability 1/16", np.isclose(C_def / 4, .25) and np.isclose((C_def / 4) ** 2, 1 / 16))
post7, _ = hhl_numpy(np.diag([1.0, 2.0]), np.array([1, 1.]) / np.sqrt(2), 2, np.pi / 2, signed=False)
check("C7 P_success = 0.625 (HHL simulation)", np.isclose(np.linalg.norm(post7) ** 2, .625))
check("C8 9 qubits", 3 + 5 + 1 == 9)
# C9: without uncomputation, the reduced system state is mixed (purity < 1)
branch = np.zeros((4, 2), dtype=complex)
branch[1] = 1 * np.array([1, -1]) / np.sqrt(2) / np.sqrt(2)
branch[2] = .5 * np.array([1, 1]) / np.sqrt(2) / np.sqrt(2)
rho_sys = branch.T @ branch.conj(); rho_sys /= np.trace(rho_sys)
check("C9 clock not uncomputed -> mixed system (purity < 1)", np.real(np.trace(rho_sys @ rho_sys)) < 0.99)
check("C10 arcsin argument <= 1 iff C <= |lambda|", np.all(np.abs(C_def / evals) <= 1))
s14 = 14 - 16
check("C11 k = 14 -> lambda = -2, theta = -pi/3", s14 * C_def == -2 and np.isclose(2 * np.arcsin(C_def / (s14 * C_def)), -np.pi / 3))
nx = sum(2 * sum(1 for i in range(4) if not (kk >> (3 - i)) & 1) for kk in range(1, 16))
check("C12 56 X gates", nx == 56)
post_def, C_sim = hhl_numpy(A_DEF, B_DEF, 4, np.pi / 8)
x_def = np.linalg.solve(A_DEF, B_DEF)
check("C13 x = ||b|| a / C reproduces the solution", close(post_def * 1.0 / C_sim, x_def, 1e-9))
check("C14 x = (25, 7, 11, 5)/48", close(x_def * 48, [25, 7, 11, 5]))
v16 = np.kron(Hd @ [1, 0], Z @ Hd @ [1, 0])
check("C16 H on both, Z on q1 -> (1,-1,1,-1)/2", close(v16, np.array([1, -1, 1, -1]) / 2)
      and close(np.kron(Z @ Hd @ [1, 0], Hd @ [1, 0]), np.array([1, 1, -1, -1]) / 2)
      and close(np.kron(Hd @ X @ [1, 0], Hd @ X @ [1, 0]), np.array([1, -1, -1, 1]) / 2))
check("C17 QFT|k> -> inverse QFT gives |k>", close(qft_matrix(3).conj().T @ qft_matrix(3) @ np.eye(8)[5], np.eye(8)[5]))
post18, C18 = hhl_numpy(A_DEF, evecs[:, 2], 4, np.pi / 8)
check("C18 b = eigenvector (lambda = 3): P = C^2/9", np.isclose(np.linalg.norm(post18) ** 2, 1 / 9))
Ac = np.array([[1.5, .5], [.5, 1.5]])
ec, vc = np.linalg.eigh(Ac)
check("C19 eigenpairs and clock values 1, 2 (t = pi/2, n_c = 2)", close(ec, [1, 2])
      and close(ec * (np.pi / 2) * 4 / (2 * np.pi), [1, 2]) and np.isclose(2 * np.pi / (4 * np.pi / 2), 1))
post20, _ = hhl_numpy(Ac, np.array([1, 0.]), 2, np.pi / 2, signed=False)
check("C20 P = 5/8 and state (3|0> - |1>)/sqrt10", np.isclose(np.linalg.norm(post20) ** 2, 5 / 8)
      and close(post20 / np.linalg.norm(post20), np.array([3, -1]) / np.sqrt(10)))

# ------------------------------------------------------------------------------------- Section D
check("D2 repetitions 100 -> about 10 with amplitude amplification", np.isclose(1 / (1 / 10 ** 2), 100)
      and abs(np.pi / (4 * np.sqrt(1e-2)) - 7.85) < 0.01)
kmax = 2 ** (3 - 1) - 1
t3 = 2 * np.pi * kmax / (2 ** 3 * 4)
k3 = np.array([1, 2, 3, 4]) * t3 * 2 ** 3 / (2 * np.pi)
check("D6 clock values 0.75, 1.5, 2.25, 3 (t = 3 pi/16, C = 4/3)", close(k3, [.75, 1.5, 2.25, 3]) and np.isclose(t3, 3 * np.pi / 16)
      and np.isclose(2 * np.pi / (8 * t3), 4 / 3))
post6, C6 = hhl_numpy(A_DEF, B_DEF, 3, t3)
fid6 = abs(np.vdot(post6 / np.linalg.norm(post6), x_def / np.linalg.norm(x_def))) ** 2
check("D6 fidelity drops below 1 with 3 clock qubits", fid6 < 0.999, f"fidelity {fid6:.4f}")
check("D7 n_c = 14", int(np.ceil(np.log2(100 / 0.01))) == 14)
check("D8 31 qubits, 34.4 GB", 20 + 10 + 1 == 31 and abs(16 * 2 ** 31 / 1e9 - 34.36) < 0.01)
A1t, A2t = np.array([[0, 1], [1, 0.]]), np.diag([1.0, -1.0])
errs = []
for r in (10, 20, 40):
    tro = np.linalg.matrix_power(expm(1j * A1t * 1.0 / r) @ expm(1j * A2t * 1.0 / r), r)
    errs.append(np.linalg.norm(tro - expm(1j * (A1t + A2t)), 2))
check("D9 first-order Trotter error ~ 1/r", 1.8 < errs[0] / errs[1] < 2.2 and 1.8 < errs[1] / errs[2] < 2.2,
      ", ".join(f"{e:.2e}" for e in errs))
src = (HERE.parent.parent / "dev" / "notebook_builders" / "build_hhl.py").read_text()
check("D10 notebook uses expm custom gates and refuses hardware", "expm(1j * A_work * t_evolution)" in src
      and "register_operation" in src and 'backend_cfg["device"] == "hardware"' in src)
p = 205 / 576
se = np.sqrt(p * (1 - p) / 20000)
check("D11 SE 0.0034, deviation ~0.5 sigma", abs(se - .0034) < 5e-5 and abs((7086 / 20000 - p) / se + .47) < .02)
se12 = np.sqrt(.7622 * .2378 / 7086)
check("D12 SE 0.0051, deviation ~1.4 sigma", abs(se12 - .00506) < 5e-5 and abs((.7691 - .7622) / se12 - 1.36) < .05)
check("D12 exact |x_0|^2 = 0.7622", np.isclose(x_def[0] ** 2 / (x_def @ x_def), .7622, atol=5e-5))
check("D13 n_c = 5: 31 rotations, 10 evolutions, 8 qubits", (2 ** 5 - 1, 2 * 5, 5 + 2 + 1) == (31, 10, 8))
Anh = np.array([[1, 2], [0, 1.]]); bnh = np.array([1, 1.])
emb = np.block([[np.zeros((2, 2)), Anh], [Anh.T, np.zeros((2, 2))]])
sol = np.linalg.solve(emb, np.r_[bnh, 0, 0])
check("D18 solution in the second half, first half zero", close(sol[:2], 0) and close(sol[2:], np.linalg.solve(Anh, bnh)))
check("D19 P_success = 205/576 exactly", Fraction(1, 4) * (1 + Fraction(1, 4) + Fraction(1, 9) + Fraction(1, 16)) == Fraction(205, 576)
      and np.isclose(np.linalg.norm(post_def) ** 2, 205 / 576))
check("D20 ||x|| = sqrt(205)/24", np.isclose(np.linalg.norm(x_def), np.sqrt(205) / 24))

# ------------------------------------------------------------------------------------- optional CUDA-Q cross-check
if "--cudaq" in sys.argv:
    import hashlib
    import cudaq
    cudaq.set_target("qpp-cpu")
    print(f"\nCUDA-Q {cudaq.__version__.split()[2] if ' ' in cudaq.__version__ else cudaq.__version__}, target {cudaq.get_target().name}")

    def bit_reverse_order(vec, n):
        idx = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2 ** n)]
        return np.asarray(vec)[idx]

    def cudaq_hhl(A, b, n_c, t):
        """The notebook's kernel (build_hhl_kernel in build_hhl.py), restated."""
        n_b = int(np.log2(A.shape[0])); n_tot = n_c + n_b + 1
        U = expm(1j * A * t)
        uid = hashlib.sha1(np.r_[A.real.ravel(), t, n_c].tobytes()).hexdigest()[:10]
        kernel = cudaq.make_kernel()
        init = np.kron(np.kron(np.eye(2 ** n_c)[0], b / np.linalg.norm(b)), [1.0, 0.0]).astype(complex)
        reg = kernel.qalloc(bit_reverse_order(init, n_tot))
        clock = [reg[i] for i in range(n_c)]; system = [reg[n_c + i] for i in range(n_b)]; anc = reg[n_tot - 1]
        for q in clock:
            kernel.h(q)
        for j in range(n_c):
            name = f"vhhl_{uid}_u{j}"
            cudaq.register_operation(name, np.linalg.matrix_power(U, 2 ** j))
            getattr(kernel, name)(clock[n_c - 1 - j], *system)
        for j in range(n_c // 2):
            kernel.swap(clock[j], clock[n_c - 1 - j])
        for jj in range(n_c):
            j = n_c - 1 - jj
            for kk in range(n_c - 1 - j):
                k_ = n_c - 1 - kk
                kernel.cr1(-np.pi / 2 ** (k_ - j), clock[k_], clock[j])
            kernel.h(clock[j])
        C = 2 * np.pi / (2 ** n_c * t)
        for kv in range(1, 2 ** n_c):
            s = kv - 2 ** n_c if kv >= 2 ** (n_c - 1) else kv
            theta = 2 * np.arcsin(np.clip(C / (2 * np.pi * s / (2 ** n_c * t)), -1, 1))
            zb = [i for i in range(n_c) if not (kv >> (n_c - 1 - i)) & 1]
            for i in zb:
                kernel.x(clock[i])
            kernel.cry(theta, clock, anc)
            for i in zb:
                kernel.x(clock[i])
        for j in range(n_c):
            kernel.h(clock[j])
            for k_ in range(j + 1, n_c):
                kernel.cr1(np.pi / 2 ** (k_ - j), clock[k_], clock[j])
        for j in range(n_c // 2):
            kernel.swap(clock[j], clock[n_c - 1 - j])
        for jj in range(n_c):
            j = n_c - 1 - jj
            name = f"vhhl_{uid}_uinv{j}"
            cudaq.register_operation(name, np.linalg.matrix_power(U.conj().T, 2 ** j))
            getattr(kernel, name)(clock[n_c - 1 - j], *system)
        for q in clock:
            kernel.h(q)
        sv = np.array(cudaq.get_state(kernel))
        tensor = sv.reshape([2] * n_tot).transpose(list(range(n_tot))[::-1])
        return tensor[(0,) * n_c + (slice(None),) * n_b + (1,)].reshape(-1), C

    a_q, Cq = cudaq_hhl(A_DEF, B_DEF, 4, np.pi / 8)
    check("CUDA-Q C14/D19: default run x = (25,7,11,5)/48, P = 205/576", close(a_q / Cq * 48, [25, 7, 11, 5], 1e-8)
          and np.isclose(np.linalg.norm(a_q) ** 2, 205 / 576))
    a_q6, _ = cudaq_hhl(A_DEF, B_DEF, 3, t3)
    check("CUDA-Q D6: 3 clock qubits agrees with NumPy simulator (fidelity < 1)", close(a_q6, post6, 1e-8))
    a_q20, _ = cudaq_hhl(Ac, np.array([1, 0.]), 3, np.pi / 4)
    check("CUDA-Q C20 system: solution direction (3,-1)/sqrt10 (signed clock, n_c = 3)",
          np.isclose(abs(np.vdot(a_q20 / np.linalg.norm(a_q20), np.array([3, -1]) / np.sqrt(10))) ** 2, 1))
    qft_k = cudaq.make_kernel()
    qq = qft_k.qalloc(3)
    qft_k.x(qq[0]); qft_k.x(qq[2])          # |101>
    check("CUDA-Q sanity: |101> prepared", np.isclose(abs(np.array(cudaq.get_state(qft_k))[0b101]) ** 2, 1))

# ------------------------------------------------------------------------------------- answer key consistency
sys.path.insert(0, str(HERE))
import generate_papers as gp  # noqa: E402
qs, _, _ = gp.load_questions()
check("80 questions, 4 sections x 20", len(qs) == 80 and all(sum(q["section"] == s for q in qs) == 20 for s in "ABCD"))
check("every question has 4 distinct options", all(len(q["opts"]) == 4 and len(set(q["opts"])) == 4 for q in qs))
check("each letter correct 5 times per section", all(sum(q["section"] == s and q["answer"] == i for q in qs) == 5
                                                     for s in "ABCD" for i in range(4)))
with open(HERE / "answer_key.csv") as fh:
    rows = list(csv.DictReader(fh))
check("answer_key.csv matches the generated key", [r["answer"] for r in rows] == [gp.LETTERS[q["answer"]] for q in qs])

fails = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(fails)} of {len(RESULTS)} checks passed")
sys.exit(1 if fails else 0)
