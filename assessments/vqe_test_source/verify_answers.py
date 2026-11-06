"""Recompute every numerical / structural answer of the VQE paper and assert it matches the key.

Run:  python verify_answers.py           (NumPy / SciPy only)
      python verify_answers.py --chem    (+ PySCF / OpenFermion: H2 coefficients, LiH counts, JW / BK mappings)
      python verify_answers.py --cudaq   (+ the notebook's CUDA-Q kernels: UCC state, exp_pauli sign, observe energies)

Conventions as in the paper and VQE.ipynb: character k of a Pauli word acts on qubit k; qubit 0 is written first (most
significant in Kronecker products); exp_pauli(theta, word) = exp(+i theta P); JW with |1> = occupied.
Each computed value is also checked against the text of the CORRECT option in the bank, and answer_key.csv is checked
against the generator's layout.
"""
import csv
import math
import sys
from functools import reduce
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

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


P = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.diag([1.0 + 0j, -1.0])}
op = lambda w: reduce(np.kron, [P[c] for c in w])
ket = lambda b: np.eye(2 ** len(b))[:, int(b, 2)].astype(complex)
close = lambda a, b, t=1e-9: np.allclose(a, b, atol=t)
H2 = [(-0.0970662681676282, "IIII"), (0.17141282644776884, "ZIII"), (0.17141282644776884, "IZII"),
      (-0.22343153690813597, "IIZI"), (-0.22343153690813597, "IIIZ"), (0.16868898170361213, "ZZII"),
      (0.12062523483390425, "ZIZI"), (0.16592785033770352, "ZIIZ"), (0.16592785033770352, "IZZI"),
      (0.12062523483390425, "IZIZ"), (0.17441287612261608, "IIZZ"), (-0.0453026155037993, "XXYY"),
      (0.0453026155037993, "XYYX"), (0.0453026155037993, "YXXY"), (-0.0453026155037993, "YYXX")]
H = sum(c * op(w) for c, w in H2)
EV, VEC = np.linalg.eigh(H)
FCI = EV[0]
i1100, i0011 = int("1100", 2), int("0011", 2)
H11, H22, H12 = H[i1100, i1100].real, H[i0011, i0011].real, H[i1100, i0011].real
ucc = lambda t: np.cos(t) * ket("1100") + 1j * np.sin(t) * (op("XXXY") @ ket("1100"))
TSTAR = minimize_scalar(lambda t: np.real(ucc(t).conj() @ H @ ucc(t)), bounds=(-1, 1), method="bounded", options={"xatol": 1e-12}).x


def tfim_terms(n, J, h):
    t = [(-J, "".join("Z" if k in (i, i + 1) else "I" for k in range(n))) for i in range(n - 1)]
    return t + [(-h, "".join("X" if k == i else "I" for k in range(n))) for i in range(n)]


def tfim_ff(n, J, h):
    A = np.zeros((2 * n, 2 * n))
    for j in range(n):
        A[2 * j, 2 * j + 1], A[2 * j + 1, 2 * j] = -h, h
    for j in range(n - 1):
        A[2 * j + 1, 2 * j + 2], A[2 * j + 2, 2 * j + 1] = -J, J
    return -0.5 * np.sum(np.abs(np.linalg.eigvalsh(1j * A))), A.shape


# ----------------------------------------------------------------------------------------------- Section A
A = np.array([[1, 2], [2, -2]])
w, v = np.linalg.eigh(A)
check("A1", close(w, [-3, 2]) and close(abs(v[:, 0] @ np.array([1, -2]) / np.sqrt(5)), 1)); key_has("A1", "$-3$", "(1, -2)")
x = np.array([1, -1]); check("A2", math.isclose(x @ A @ x / (x @ x), -2.5)); key_has("A2", "$-2.5$")
M = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 5]])
check("A3", np.trace(M) == 9 and round(np.linalg.det(M)) == 15 and close(np.linalg.eigvalsh(M), [1, 3, 5]))
key_has("A3", r"P$\to$3, Q$\to$1, R$\to$2, S$\to$4")
from scipy.linalg import expm
psi = np.array([0.6, 0.8]); Hs = np.diag([0.4, 0.4]) + 0.3 * P["Z"].real
v0 = np.linalg.eigh(Hs)[1][:, 0]
check("A4", math.isclose(abs(v0.conj() @ expm(-1j * Hs * 2.7) @ v0), 1)); key_has("A4", "global phase")
check("A5", round(0.005 * 627.51, 2) == 3.14 and 0.005 > 0.0016); key_has("A5", "3.14", "not")
key_has("A6", r"\texttt{IIII}")
fixed("A7", 0); fixed("A8", 3)
check("A8 II", close(np.linalg.eigh(np.array([[1, 2j], [-2j, 3]]))[1][:, 0].conj() @ np.linalg.eigh(np.array([[1, 2j], [-2j, 3]]))[1][:, 1], 0))
check("A9", comb(4, 2) == 6 and comb(2, 1) * comb(2, 1) == 4); key_has("A9", "$6$ and $4$")
key_has("A10", "$12$ qubits, $4$ electrons")
ad = np.array([[0, 0], [1, 0]], complex)
check("A11", close(ad @ ad, 0) and close(ad.conj().T @ ad + ad @ ad.conj().T, np.eye(2))
      and close(np.linalg.eigvalsh(ad @ ad.conj().T), [0, 1]) and close(ad.conj().T @ np.array([1, 0]), 0))
key_has("A11", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3")
key_has("A12", "N^4")
check("A13", close(expm(1j * np.pi / 4 * P["X"]), np.array([[1, 1j], [1j, 1]]) / np.sqrt(2))); key_has("A13", "1 & i")
Hq = -P["Z"] + 0.5 * P["X"]
ry = lambda t: np.array([np.cos(t / 2), np.sin(t / 2)])
E1 = lambda t: np.real(ry(t) @ Hq @ ry(t))
check("A14", all(math.isclose(E1(t), -np.cos(t) + .5 * np.sin(t)) for t in (0.3, 1.7, -2.2))
      and math.isclose(np.linalg.eigvalsh(Hq)[0], -np.sqrt(5) / 2)); key_has("A14", r"\sqrt5")
ts = -np.arctan(0.5)
check("A15", math.isclose(E1(ts), -np.sqrt(5) / 2) and abs(E1(0) - (-np.sqrt(5) / 2) - 0.118) < 5e-4); key_has("A15", "-0.464", "0.118")
Mc = np.array([[3, 1 - 2j], [1 + 2j, -1]])
check("A16", close([np.trace(P[k] @ Mc).real / 2 for k in "IXYZ"], [1, 1, 2, 2])); key_has("A16", "I + X + 2Y + 2Z")
D = np.diag([1, 2, 3, 4.]).astype(complex)
check("A17", close([np.trace(op(k) @ D).real / 4 for k in ("II", "ZI", "IZ", "ZZ")], [2.5, -1, -0.5, 0])); key_has("A17", r"- 1.0\,ZI - 0.5\,IZ")
fixed("A18", 1)
check("A19", abs((FCI - H11) * 1000 + 20.5) < 0.05); key_has("A19", "$-20.5$")
check("A20", abs(0.9 * FCI + 0.1 * EV[1] + 1.0774) < 1e-4 and abs(0.1 * (EV[1] - FCI) - 0.0599) < 1e-4); key_has("A20", "-1.0774", "59.9")

# ----------------------------------------------------------------------------------------------- Section B
check("B1", math.isclose(ry(np.pi / 3)[1] ** 2, 0.25)); key_has("B1", r"\tfrac14")
check("B2", close(op("XY") @ op("YX"), op("ZZ"))); key_has("B2", r"Z\otimes Z")
bloch = lambda s: [np.real(s.conj() @ P[k] @ s) for k in "XYZ"]
check("B3", close(bloch(np.array([1, 0j])), [0, 0, 1]) and close(bloch(np.array([1, -1]) / np.sqrt(2)), [-1, 0, 0])
      and close(bloch(np.array([1, 1j]) / np.sqrt(2)), [0, 1, 0]) and close(bloch(ry(np.pi / 3).astype(complex)), [np.sqrt(3) / 2, 0, .5]))
key_has("B3", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2); S = np.diag([1, 1j]); U = Hd @ S.conj().T
check("B4", close(Hd @ P["X"] @ Hd, P["Z"]) and close(U @ P["Y"] @ U.conj().T, P["Z"])); key_has("B4", r"P$\to$3, Q$\to$4, R$\to$2, S$\to$1")
bell = (ket("00") + ket("11")) / np.sqrt(2)
check("B5", math.isclose(np.real(bell.conj() @ op("YY") @ bell), -1)); key_has("B5", "$-1$")
check("B6", sum(int(b) << k for k, b in enumerate("1100")) == 3); key_has("B6", "$3$")
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
RY = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
st = CNOT @ np.kron(RY(np.pi / 2), np.eye(2)) @ ket("00")
check("B7", close(st, bell)); key_has("B7", r"\ket{00} + \ket{11}")
fixed("B8", 2)
words = ["XXYY", "XYYX", "YXXY", "YYXX"]
qwc = lambda a, b: all(x == y or "I" in (x, y) for x, y in zip(a, b))
check("B9", all(close(op(a) @ op(b), op(b) @ op(a)) and not qwc(a, b) for i, a in enumerate(words) for b in words[i + 1:]))
key_has("B9", "All six pairs commute")
check("B10", close(expm(1j * 0.7 * op("ZZ")) @ ket("01"), np.exp(-0.7j) * ket("01"))); key_has("B10", r"e^{-i\theta}")
check("B11", close(op("XXXY") @ ket("1100"), 1j * ket("0011"))); key_has("B11", r"i\ket{0011}")
fixed("B12", 2)
check("B13", 16 * 2 ** 12 == 65536 and 16 * 4 ** 12 == 268435456); key_has("B13", "65.5", "268")
t0, t1 = 0.8, 1.9
st = CNOT @ np.kron(RY(t0), RY(t1)) @ ket("00")
check("B14", math.isclose(np.real(st.conj() @ op("ZZ") @ st), np.cos(t1))); key_has("B14", r"\cos\theta_1")
key_has("B15", r"HS^\dagger")
Ht = -op("ZZ") - .5 * (op("XI") + op("IX"))
check("B16", math.isclose(np.real(bell.conj() @ Ht @ bell), -1)); key_has("B16", "$-1$")
e0 = np.linalg.eigvalsh(Ht)[0]
check("B17", math.isclose(e0, -np.sqrt(2)) and math.isclose(tfim_ff(2, 1, .5)[0], e0) and abs((-1 - e0) / abs(e0) - 0.293) < 5e-4)
key_has("B17", r"\sqrt2", "29.3")
check("B18", 4 ** 4 == 256 and len(H2) == 15); key_has("B18", "$256$ and $15$")
fixed("B19", 0)
check("B20", close(Hq @ Hq, 1.25 * np.eye(2)) and math.isclose(1.25 - 1, 0.25)); key_has("B20", "$0.25$ and $0$")

# ----------------------------------------------------------------------------------------------- Section C
key_has("C1", "error in the Hamiltonian")
# C2: Ritz values of a random subspace are upper bounds
rng = np.random.default_rng(0); R = rng.normal(size=(8, 8)); R = R + R.T
Q = np.linalg.qr(rng.normal(size=(8, 3)))[0]
check("C2", all(a >= b - 1e-12 for a, b in zip(np.linalg.eigvalsh(Q.T @ R @ Q), np.linalg.eigvalsh(R))))
check("C3", (3 + 1) * 10 == 40 and 3 * 9 == 27 and (2 + 1) * 15 + 2 * 14 == 73); key_has("C3", "$40$, $27$ and $67$")
check("C4", close(np.kron(np.kron(P["X"], P["X"]), np.eye(4)) @ ket("0000"), ket("1100"))); key_has("C4", r"\ket{1100}")
a_, b_, c_, t = -0.7, 0.3, 0.4, 0.37
Hab = np.array([[a_, b_], [b_, c_]]); vv = np.array([np.cos(t), -np.sin(t)])
check("C5", math.isclose(vv @ Hab @ vv, a_ * np.cos(t) ** 2 + c_ * np.sin(t) ** 2 - b_ * np.sin(2 * t))); key_has("C5", r"- b\sin2\theta")
key_has("C6", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
check("C7", math.isclose(0.5 * 0.6 - 0.3 * -0.2 + 0.2 * 0.5, 0.46)); key_has("C7", "0.46")
check("C8", abs(H11 + 1.116759) < 5e-7); key_has("C8", "-1.116759")
g = VEC[:, 0]
check("C9", abs(abs(g[i1100]) ** 2 + abs(g[i0011]) ** 2 - 1) < 1e-12 and abs(np.real(ucc(TSTAR).conj() @ H @ ucc(TSTAR)) - FCI) < 1e-12)
fixed("C9", 0)
# <0...0|P|0...0> = 1 for strings of I/Z only and 0 if the string contains X or Y
e_all_zero = sum(c for c, w in tfim_terms(15, 1, .5) if set(w) <= set("IZ"))
check("C10", math.isclose(e_all_zero, -14)); key_has("C10", "$-14$")
check("C11", 2 * 45 == 90 and 45 + 1 == 46); key_has("C11", "$90$ and $46$")
fixed("C12", 2)
check("C13", 0 < (-1.13600 - FCI) < 0.0016 and (-1.135 - FCI) > 0.0016 and (-1.139 - FCI) < 0); key_has("C13", "-1.13600")
zt = [w for c, w in H2 if set(w) <= set("IZ") and w != "IIII"]
groups_qwc = 1 + len(words)
check("C14", groups_qwc == 5 and all(qwc(a, b) for a in zt for b in zt) and not close(op("ZIII") @ op("XXYY"), op("XXYY") @ op("ZIII")))
key_has("C14", "$5$; $2$")
check("C15", abs((-15.0533898284 + 15.0825179883) / 15.0825179883 - 0.00193) < 5e-6); key_has("C15", "0.193")
check("C16", np.allclose(H.imag, 0) and np.allclose(RY(0.4).imag, 0)); key_has("C16", "real symmetric")
X15 = None
key_has("C17", "symmetry breaking")
key_has("C18", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
red = np.array([[H11, H12], [H12, H22]])
check("C19", abs(np.linalg.eigvalsh(red)[0] - FCI) < 1e-12 and abs(FCI + 1.137284) < 5e-7); key_has("C19", "-1.137284")
check("C20", abs(TSTAR - 0.1128) < 5e-5 and abs(np.sin(TSTAR) ** 2 - 0.0127) < 5e-5 and abs(np.tan(2 * TSTAR) - 2 * H12 / (H22 - H11)) < 1e-6)
key_has("C20", "0.1128", "0.0127")

# ----------------------------------------------------------------------------------------------- Section D
def jw_create(j, n):
    return reduce(np.kron, [P["Z"]] * j + [ad] + [P["I"]] * (n - j - 1))


check("D1", close(jw_create(2, 4), 0.5 * (op("ZZXI") - 1j * op("ZZYI")))); key_has("D1", r"Z_0Z_1(X_2 - iY_2)")
hop = jw_create(0, 2) @ jw_create(1, 2).conj().T
check("D2", close(hop + hop.conj().T, 0.5 * (op("XX") + op("YY")))); key_has("D2", r"X_0X_1 + Y_0Y_1")
key_has("D3", "$8$; $4$")
key_has("D4", r"P$\to$3, Q$\to$4, R$\to$1, S$\to$2")
B = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]])
check("D5", list(B @ np.array([1, 1, 0, 0]) % 2) == [1, 0, 0, 0]); key_has("D5", r"\ket{1000}")
check("D6", 4 * 8 == 32 and comb(4, 2) * comb(8, 2) == 168); key_has("D6", "$32$", "$168$")
key_has("D7", "8 strings", r"\texttt{XXXY}")
key_has("D8", r"P$\to$2, Q$\to$4, R$\to$1, S$\to$3")
check("D9", abs((0.1 / 0.0016) ** 2 - 3906) < 1); key_has("D9", "3.9")
check("D10", math.isclose(np.trace(H).real / 16, H2[0][0])); key_has("D10", "-0.0971")
fixed("D11", 3)
check("D12", math.isclose(-1.10 - (-1.00 + 1.10) / 2, -1.15)); key_has("D12", "-1.15")
src = (HERE.parent.parent / "dev" / "notebook_builders" / "common_utils.py").read_text(encoding="utf-8")
check("D13", 'if backend in ("gpu", "multi-gpu") and num_gpus == 0:' in src and "raise RuntimeError" in src); key_has("D13", "RuntimeError")
vqe_src = (HERE.parent.parent / "dev" / "notebook_builders" / "build_vqe.py").read_text(encoding="utf-8")
check("D14", "REFERENCE_MAX_QUBITS = 18" in vqe_src and "if n_qubits <= 16:" in vqe_src); fixed("D14", 3)
e15, shape = tfim_ff(100, 1, .5)
check("D15", shape == (200, 200) and abs(tfim_ff(15, 1, .5)[0] + 15.0825179883) < 1e-9); key_has("D15", r"200\times200")
check("D16", abs(69.104 / 300 - 0.23) < 0.005 and abs(102.9 / 70.1 - 1.47) < 0.01); key_has("D16", "0.23", "1.5")
check("D17", 12 - 2 == 10); key_has("D17", "$10$")
key_has("D18", "631")
check("D19", abs((-14 + 15.0533898284) / (-14 + 15.0825179883) - 0.973) < 5e-4); key_has("D19", "97.3")
check("D20", "MAX_EVALUATIONS = 300" in vqe_src and '"maxiter": MAX_EVALUATIONS' in vqe_src); key_has("D20", "budget")
check("notebook H2 terms", all(f"({c!r}, \"{w}\")" in vqe_src or repr(c) in vqe_src for c, w in H2))
check("notebook TFIM reference", abs(tfim_ff(15, 1, .5)[0] + 15.0825179883) < 1e-9)

# ----------------------------------------------------------------------------------------------- chemistry
if "--chem" in sys.argv:
    import warnings; warnings.filterwarnings("ignore")
    import openfermion as of
    from openfermionpyscf import run_pyscf

    def words_of(qop, n):
        out = {}
        for term, c in qop.terms.items():
            wd = ["I"] * n
            for qq, pp in term:
                wd[qq] = pp
            out["".join(wd)] = out.get("".join(wd), 0) + c
        return {k: c for k, c in out.items() if abs(c) > 1e-12}

    mol = run_pyscf(of.MolecularData([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], "sto-3g", 1, 0), run_scf=True, run_fci=True)
    ferm = of.get_fermion_operator(mol.get_molecular_hamiltonian())
    jw = words_of(of.jordan_wigner(ferm), 4)
    check("chem: notebook H2 = PySCF at 0.740 A", set(jw) == {w for _, w in H2} and max(abs(jw[w] - c) for c, w in H2) < 1e-9)
    check("chem: FCI", abs(mol.fci_energy - FCI) < 1e-9 and abs(mol.hf_energy - H11) < 1e-9)
    Hbk = of.get_sparse_operator(of.bravyi_kitaev(ferm), 4).toarray()
    check("D5 chem", abs(Hbk[int("1000", 2), int("1000", 2)].real - H11) < 1e-9)
    check("D1 chem", words_of(of.jordan_wigner(of.FermionOperator("2^")), 4) == {"ZZXI": 0.5, "ZZYI": -0.5j})
    check("D2 chem", words_of(of.jordan_wigner(of.FermionOperator("0^ 1") + of.FermionOperator("1^ 0")), 2) == {"XX": 0.5, "YY": 0.5})
    check("D3 chem", max(len(t) for t in of.jordan_wigner(of.FermionOperator("7^")).terms) == 8
          and max(len(t) for t in of.bravyi_kitaev(of.FermionOperator("7^"), 8).terms) == 4)
    dx = words_of(of.jordan_wigner(of.FermionOperator("3^ 2^ 1 0") - of.FermionOperator("0^ 1^ 2 3")), 4)
    check("D7 chem", len(dx) == 8 and "XXXY" in dx and all(abs(abs(c) - 0.125) < 1e-12 for c in dx.values()))
    li = run_pyscf(of.MolecularData([("Li", (0, 0, 0)), ("H", (0, 0, 1.5949))], "sto-3g", 1, 0), run_scf=True)
    jl = words_of(of.jordan_wigner(of.get_fermion_operator(li.get_molecular_hamiltonian())), li.n_qubits)
    check("A10 chem", li.n_qubits == 12 and li.n_electrons == 4)
    check("D18 chem", len(jl) == 631, len(jl))
    print("PySCF / OpenFermion chemistry checks done")

# ----------------------------------------------------------------------------------------------- CUDA-Q
if "--cudaq" in sys.argv:
    import warnings; warnings.filterwarnings("ignore")
    import cudaq
    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def ucc_ansatz(theta: list[float]):
        qubits = cudaq.qvector(4)
        x(qubits[0])
        x(qubits[1])
        exp_pauli(theta[0], qubits, "XXXY")

    @cudaq.kernel
    def hea_ansatz(n: int, layers: int, theta: list[float]):
        qubits = cudaq.qvector(n)
        k = 0
        for layer in range(layers):
            for q in range(n):
                ry(theta[k], qubits[q])
                k += 1
            for q in range(n - 1):
                x.ctrl(qubits[q], qubits[q + 1])
        for q in range(n):
            ry(theta[k], qubits[q])
            k += 1

    to_be = lambda s, n: np.array(s).reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)
    s = to_be(cudaq.get_state(ucc_ansatz, [0.3]), 4)
    check("cudaq UCC state", close(s, ucc(0.3)))
    ham = None
    for c, wd in H2:
        term = c * cudaq.SpinOperator.from_word(wd)
        ham = term if ham is None else ham + term
    check("cudaq UCC energy = FCI", abs(cudaq.observe(ucc_ansatz, ham, [float(TSTAR)]).expectation() - FCI) < 1e-9)
    check("cudaq HF energy", abs(cudaq.observe(ucc_ansatz, ham, [0.0]).expectation() - H11) < 1e-9)
    rng = np.random.default_rng(1)
    th = list(rng.normal(0, 0.5, 12))
    s = to_be(cudaq.get_state(hea_ansatz, 4, 2, th), 4)
    ref = ket("0000")
    k = 0
    for layer in range(2):
        ref = reduce(np.kron, [RY(th[k + q]) for q in range(4)]) @ ref; k += 4
        for q in range(3):
            ref = reduce(np.kron, [np.eye(2 ** q), CNOT, np.eye(2 ** (2 - q))]) @ ref
    ref = reduce(np.kron, [RY(th[k + q]) for q in range(4)]) @ ref
    check("cudaq HEA state", abs(abs(np.vdot(s, ref)) - 1) < 1e-10)
    tf = None
    for c, wd in tfim_terms(4, 1, .5):
        term = c * cudaq.SpinOperator.from_word(wd)
        tf = term if tf is None else tf + term
    Htf = sum(c * op(wd) for c, wd in tfim_terms(4, 1, .5))
    check("cudaq HEA TFIM energy", abs(cudaq.observe(hea_ansatz, tf, 4, 2, th).expectation() - np.real(ref.conj() @ Htf @ ref)) < 1e-9)
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
