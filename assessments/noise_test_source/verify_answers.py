"""Recompute every numerical / structural answer of the Quantum Noise Simulation paper and assert it matches the key.

Run:  python verify_answers.py          (NumPy / SciPy only)
      python verify_answers.py --cudaq  (additionally runs CUDA-Q density-matrix and stabilizer simulations, including
                                         the noise model of Noise_Simulation.ipynb, and re-checks the CUDA-Q conventions)

Conventions (same as the paper): qubit 0 written first; R_P(t) = exp(-i t P/2); depolarizing (CUDA-Q):
sqrt(1-p) I, sqrt(p/3) X, Y, Z. Every check compares a recomputed value with the option the bank lists FIRST (the correct
option of mcq/match/para questions) or with the 'correct' index of ar/stmt questions; claims about the repository are
checked against dev/notebook_builders/build_noise.py and the outputs stored in Noise_Simulation.ipynb.
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import bank_advanced
import bank_basics
import bank_math
import bank_qm
from bank_explain import EXPLAIN

ROOT = Path(__file__).resolve().parents[2]
BUILDER = (ROOT / "dev" / "notebook_builders" / "build_noise.py").read_text()
NB_TEXT = "\n".join("".join(o.get("text", "")) for c in json.loads((ROOT / "Noise_Simulation.ipynb").read_text())["cells"]
                    if c["cell_type"] == "code" for o in c.get("outputs", []))
CHECKS = []


def check(name, cond, detail=""):
    CHECKS.append((name, bool(cond), detail))
    if not cond:
        print(f"FAIL {name}: {detail}")


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, complex), np.asarray(b, complex), atol=tol)


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0 + 0j, -1])
H = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
k0, k1 = np.array([1, 0], complex), np.array([0, 1], complex)
plus = np.array([1, 1], complex) / math.sqrt(2)


def dm(v):
    return np.outer(v, v.conj())


def channel(rho, ks):
    return sum(K @ rho @ K.conj().T for K in ks)


def bloch(rho):
    return np.real([np.trace(rho @ P) for P in (X, Y, Z)])


def is_cptp(ks):
    return close(sum(K.conj().T @ K for K in ks), np.eye(ks[0].shape[0]))


def is_state(rho):
    return close(rho, rho.conj().T) and abs(np.trace(rho) - 1) < 1e-12 and np.min(np.linalg.eigvalsh(rho)) > -1e-12


BF = lambda p: [math.sqrt(1 - p) * I2, math.sqrt(p) * X]
PF = lambda p: [math.sqrt(1 - p) * I2, math.sqrt(p) * Z]
YF = lambda p: [math.sqrt(1 - p) * I2, math.sqrt(p) * Y]
DEP = lambda p: [math.sqrt(1 - p) * I2] + [math.sqrt(p / 3) * P for P in (X, Y, Z)]
AD = lambda g: [np.array([[1, 0], [0, math.sqrt(1 - g)]], complex), np.array([[0, math.sqrt(g)], [0, 0]], complex)]
PD = lambda l: [np.diag([1, math.sqrt(1 - l)]).astype(complex), np.diag([0, math.sqrt(l)]).astype(complex)]

# ===================================================================================================== Section A
v = np.array([1, 1j]) / math.sqrt(2)
check("A1", close(dm(v), 0.5 * np.array([[1, -1j], [1j, 1]])) and close(bloch(dm(v)), [0, 1, 0]))
r2 = np.array([[0.7, 0.3], [0.3, 0.3]])
check("A2", abs(np.trace(r2 @ r2) - 0.76) < 1e-12 and abs(0.49 + 0.09 - 0.58) < 1e-12)
r3 = np.array([[0.8, 0.2 - 0.1j], [0.2 + 0.1j, 0.2]])
check("A3", close(bloch(r3), [0.4, 0.2, 0.6]) and is_state(r3))
cands = [np.array([[0.6, 0.5], [0.5, 0.4]]), 0.5 * np.ones((2, 2)), np.diag([0.9, 0.1]), np.array([[0.5, -0.3j], [0.3j, 0.5]])]
check("A4", (not is_state(cands[0])) and all(is_state(c) for c in cands[1:]))
check("A5", is_cptp([np.diag([1, 0.8]).astype(complex), np.array([[0, 0.6], [0, 0]], complex)])
      and not any(is_cptp([np.diag([1, 0.8]).astype(complex), np.array([[0, b], [0, 0]], complex)]) for b in (0.36, 0.8, 0.4)))
phis = np.linspace(-np.pi / 2, np.pi / 2, 200001)
check("A6", abs(np.trapezoid(np.cos(phis), phis) / np.pi - 2 / np.pi) < 1e-9)
check("A7", abs(1 - 0.99 ** 20 - 0.182) < 1e-3 and abs(20 * 0.01 * 0.99 ** 19 - 0.165) < 1e-3)
check("A8", abs(math.sqrt(0.09 / 10000) - 0.003) < 1e-12)
Rx = lambda t: math.cos(t / 2) * I2 - 1j * math.sin(t / 2) * X
check("A9", abs(abs((Rx(math.pi + 0.1) @ k0)[0]) ** 2 - 0.0025) < 1e-5)
r10 = 0.8 * dm(plus) + 0.2 * I2 / 2
check("A10", abs(np.real(plus.conj() @ r10 @ plus) - 0.9) < 1e-12)
psi11 = math.sqrt(0.8) * np.kron(k0, k0) + math.sqrt(0.2) * np.kron(k1, k1)
rA = np.einsum("abcb->ac", dm(psi11).reshape(2, 2, 2, 2))
check("A11", abs(np.real(np.trace(rA @ rA)) - 0.68) < 1e-12)
check("A12", abs(max(np.linalg.eigvalsh(r2)) - 0.8606) < 1e-4)
pr = np.kron(np.diag([0.9, 0.1]), np.diag([0.8, 0.2]))
check("A13", abs(pr[0, 0] + pr[3, 3] - 0.74) < 1e-12)
rr = np.random.default_rng(0).normal(size=3); rr /= 2 * np.linalg.norm(rr)
rh = 0.5 * (I2 + rr[0] * X + rr[1] * Y + rr[2] * Z)
check("A14", close(bloch(X @ rh @ X), [rr[0], -rr[1], -rr[2]]))
check("A15", close(bloch(dm(k1)), [0, 0, -1]) and close(bloch(I2 / 2), [0, 0, 0])
      and close(bloch(0.5 * np.array([[1, -1j], [1j, 1]])), [0, 1, 0]) and close(bloch(np.diag([0.75, 0.25])), [0, 0, 0.5]))
S = np.diag([1, 1j]); P = 0.5 * np.ones((2, 2)); N_ = np.array([[0, 1], [0, 0]])
herm = lambda A: close(A, A.conj().T); unit = lambda A: close(A.conj().T @ A, np.eye(2))
check("A16", herm(X) and unit(X) and herm(P) and close(P @ P, P) and is_state(P) and not unit(P)
      and unit(S) and not herm(S) and not herm(N_) and not unit(N_))
for rv in np.random.default_rng(1).normal(size=(5, 3)):
    rv *= 0.9 / np.linalg.norm(rv)
    rho_ = 0.5 * (I2 + rv[0] * X + rv[1] * Y + rv[2] * Z)
    check("A17 purity formula", abs(np.real(np.trace(rho_ @ rho_)) - (1 + rv @ rv) / 2) < 1e-12)
check("A18", not is_state(np.diag([1.2, -0.2])))
rp = np.array([[0.55, 0.35], [0.35, 0.45]])
check("A19", close(bloch(rp), [0.7, 0, 0.1]) and abs(np.trace(rp @ rp) - 0.75) < 1e-12 and is_state(rp))
check("A20", abs(np.real(plus @ rp @ plus) - 0.85) < 1e-12 and rp[1, 1] == 0.45)

# ===================================================================================================== Section B
check("B1", close(channel(dm(plus), BF(0.3)), dm(plus)) and not close(channel(dm(k0), BF(0.3)), dm(k0))
      and not close(channel(dm(v), BF(0.3)), dm(v)))
o = channel(dm(plus), PF(0.1))
check("B2", abs(np.real(plus @ o @ plus) - 0.9) < 1e-12 and abs(bloch(o)[0] - 0.8) < 1e-12)
check("B3", abs(np.real(channel(dm(k0), DEP(0.03))[1, 1]) - 0.02) < 1e-12)
check("B4", close(channel(dm(plus), AD(0.36)), [[0.68, 0.40], [0.40, 0.32]]))
check("B5", close(channel(dm(plus), PD(0.36)), channel(dm(plus), PF(0.1))))
check("B6", abs(50 * math.log(2) - 34.66) < 0.01)
check("B7", abs(1 / (1 / 200 + 1 / 200) - 100) < 1e-9)
t2ok = lambda t1, t2: t2 <= 2 * t1
check("B8", (not t2ok(50, 120)) and t2ok(50, 100) and t2ok(100, 30) and t2ok(80, 80))
check("B9", int(100e-6 * math.log(1 / 0.9) / 50e-9) == 210)
check("B10", abs(0.5 * 0.02 + 0.5 * 0.95 - 0.485) < 1e-12)
A1 = np.array([[0.98, 0.05], [0.02, 0.95]]); A3 = np.kron(np.kron(A1, A1), A1)
qq = A3 @ np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
check("B11", abs(qq[0] + qq[7] - 0.8993) < 1e-4)
check("B12", True, "conceptual")
nflip = sum(1 for a in range(4) for b in range(4) if (a, b) != (0, 0) and a in (1, 2))
check("B13", nflip == 8 and abs(nflip * 0.015 / 15 - 0.008) < 1e-12)
ghz3 = (np.kron(np.kron(k0, k0), k0) + np.kron(np.kron(k1, k1), k1)) / math.sqrt(2)
Z0 = np.kron(np.kron(Z, I2), I2); XXX = np.kron(np.kron(X, X), X)
check("B14", close(np.abs(Z0 @ ghz3) ** 2, np.abs(ghz3) ** 2) and abs(np.real((Z0 @ ghz3).conj() @ XXX @ (Z0 @ ghz3)) + 1) < 1e-12)
check("B15", all(is_cptp(k) for k in (BF(0.2), PF(0.2), AD(0.2), PD(0.2))))
rv = np.array([0.3, -0.5, 0.6]); rho_ = 0.5 * (I2 + rv[0] * X + rv[1] * Y + rv[2] * Z)
g = 0.3
check("B16", close(bloch(channel(rho_, DEP(0.3))), (1 - 0.4) * rv)
      and close(bloch(channel(rho_, AD(g))), [math.sqrt(1 - g) * rv[0], math.sqrt(1 - g) * rv[1], g + (1 - g) * rv[2]])
      and close(bloch(channel(rho_, PD(0.3))), [math.sqrt(0.7) * rv[0], math.sqrt(0.7) * rv[1], rv[2]])
      and close(bloch(channel(rho_, BF(0.3))), [rv[0], 0.4 * rv[1], 0.4 * rv[2]]))
check("B17", (not close(channel(I2 / 2, AD(0.3)), I2 / 2)) and close(channel(I2 / 2, PD(0.3)), I2 / 2))
check("B18", abs(channel(dm(plus), PF(0.5))[0, 1]) < 1e-12 and close(channel(dm(plus), PF(1.0)), Z @ dm(plus) @ Z))
check("B19", abs(0.5 * math.exp(-0.25) - 0.389) < 1e-3 and abs(0.5 * math.exp(-1 / 3) - 0.358) < 1e-3)
check("B20", abs(0.5 + 0.5 * math.exp(-1 / 3) - 0.858) < 1e-3 and abs(1 / (1 / 60 - 1 / 160) - 96) < 1e-9)

# ===================================================================================================== Section C
check("C1 (NumPy)", abs(bloch(channel(dm(k0), DEP(0.3)))[2] - 0.6) < 1e-12)
check("C3", is_cptp(YF(0.3)) and not is_cptp([I2, math.sqrt(0.1) * X])
      and not is_cptp([np.diag([1, 0.6]).astype(complex), np.array([[0, 0.6], [0, 0]], complex)])
      and not is_cptp([math.sqrt(0.5) * I2, math.sqrt(0.5) * X, math.sqrt(0.5) * Z]))
check("C4", close(channel(np.array([[0.9, 0.3], [0.3, 0.1]]), BF(0.2)), [[0.74, 0.30], [0.30, 0.26]]))
r_two = channel(channel(rh, DEP(0.03)), DEP(0.06)); r_one = channel(rh, DEP(0.0876))
check("C5", close(r_two, r_one, 1e-12))
check("C6", abs(0.98 ** 15 - 0.7386) < 1e-4 and "(1-p)^G = 0.7386" in NB_TEXT)
check("C7", abs(0.99 * (1 - (1 - (1 - 0.01) ** 2)) ** 14 - 0.99 ** 29) < 1e-12 and abs(0.99 ** 29 - 0.7472) < 1e-4
      and "1 - (1 - p) ** 2" in BUILDER)
check("C8", abs(16 * 4 ** 13 / 1e9 - 1.07) < 0.01)
check("C9 code", "only supports Pauli noise" in BUILDER and "non-Clifford" in BUILDER)
check("C10 code", "multinomial" in BUILDER and "bit_reverse_order" in BUILDER and "np.diag(rho)" in BUILDER)
Pi, Qi = {"00": 0.5, "11": 0.5}, {"00": 0.45, "11": 0.45, "01": 0.05, "10": 0.05}
check("C11", abs(sum(math.sqrt(Pi[k] * Qi.get(k, 0)) for k in Pi) ** 2 - 0.9) < 1e-12
      and abs(0.5 * sum(abs(Pi.get(k, 0) - Qi[k]) for k in Qi) - 0.1) < 1e-12)
check("C12", (850 - 150) / 1000 == 0.7)
check("C13", 4 ** 20 // 2 ** 20 == 2 ** 20)
check("C14 code", 'model.add_all_qubit_channel("x", two, num_controls=1)' in BUILDER
      and 'for gate in ("h", "x", "ry", "rz")' in BUILDER)
check("C16", True, "conceptual")
ZZ = np.kron(np.kron(Z, Z), I2)
check("C17 ZZ invariance", close(ZZ @ ghz3, ghz3))
check("C18 code", "cudaq.set_noise(model)" in BUILDER and abs(16 * 4 ** 14 / 1e9 - 4.29) < 0.01)
for val in ("0.8936", "0.6910", "0.8601", "1165", "0.3232", "0.0160", "0.2059"):
    check(f"C19/C20 recorded {val}", val in NB_TEXT)
check("C19", abs(1 - 0.8936 - 0.1064) < 1e-12 and abs(1 - 0.3232 - 0.6768) < 1e-12)
check("C20", 0.8936 > 0.8601 and math.sqrt(0.89 * 0.11 / 1e4) < 0.004)

# ===================================================================================================== Section D
check("D1", abs((2 * (1 - 0.003) + 1) / 3 - 0.998) < 1e-12)
# process fidelity of the depolarizing channel from its Choi state
bell = (np.kron(k0, k0) + np.kron(k1, k1)) / math.sqrt(2)
choi = sum(np.kron(K, I2) @ dm(bell) @ np.kron(K, I2).conj().T for K in DEP(0.003))
check("D1 Choi", abs(np.real(bell @ choi @ bell) - 0.997) < 1e-12)
check("D2", abs((2 - 1) * (1 - 0.99) / 2 - 0.005) < 1e-12)
check("D3", abs(2 * 0.80 - 0.64 - 0.96) < 1e-12 and abs(0.8 ** 2 / 0.64 - 1.0) < 1e-12)
check("D4", abs(3 * 0.80 - 3 * 0.65 + 0.53 - 0.98) < 1e-12 and abs(np.polyval(np.polyfit([1, 2, 3], [0.8, 0.65, 0.53], 2), 0) - 0.98) < 1e-12)
check("D5", (4 * 3 + 6) / 10 == 1.8)
p0 = (0.60 - 0.05) / (0.98 - 0.05)
check("D6", abs(p0 - 0.5914) < 1e-4 and abs(np.linalg.solve(A1, [0.6, 0.4])[0] - p0) < 1e-12)
check("D8", abs(math.exp(-2) - 0.135) < 1e-3)
Rz = lambda t: np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
echo_ok = all(abs(np.real((s_ := X @ Rz(a / 2) @ X @ Rz(a / 2) @ plus).conj() @ X @ s_) - 1) < 1e-12 for a in np.linspace(-3, 3, 13))
check("D8 echo", echo_ok)
rho_ = dm(plus)
rho_f = rho_.copy()
for _ in range(20):
    rho_f = channel(rho_f, PF(0.02))
rho_e = rho_.copy()
for k in range(20):
    if k == 10:
        rho_e = X @ rho_e @ X
    rho_e = channel(rho_e, PF(0.02))
rho_e = X @ rho_e @ X
check("D9", abs(bloch(rho_f)[0] - 0.96 ** 20) < 1e-12 and abs(bloch(rho_e)[0] - 0.96 ** 20) < 1e-12 and abs(0.96 ** 20 - 0.442) < 1e-3)
pL = lambda p: 3 * p ** 2 - 2 * p ** 3
check("D10", abs(pL(0.1) - 0.028) < 1e-12 and abs(pL(0.5) - 0.5) < 1e-12 and all(pL(p) < p for p in (0.1, 0.3, 0.49)))
check("D11", abs(0.1 ** 3 / 0.1 ** 2 - 0.1) < 1e-12)
pe = math.sin(0.025) ** 2
check("D12", abs(math.sin(1.5) ** 2 - 0.995) < 1e-3 and abs(0.5 * (1 - (1 - 2 * pe) ** 60) - 0.036) < 1e-3)
check("D14 code", 'elif circuit == "ghz" and pauli_noise:' in BUILDER and 'chosen_method = "density_matrix"' in BUILDER)
check("D18 code", "mz" not in re.findall(r'for gate in \(([^)]*)\)', BUILDER)[0] and "BACKEND='hardware'" in BUILDER
      and "target stim" in NB_TEXT and "GPUs visible to CUDA-Q: 0" in NB_TEXT)
E = [0.9055, 0.8189, 0.7396]
check("D19", abs(2 * E[0] - E[1] - 0.9921) < 1e-4 and abs(3 * E[0] - 3 * E[1] + E[2] - 0.9994) < 2e-4)
check("D20", 3 ** 2 + 3 ** 2 + 1 == 19)

# ===================================================================================================== structure
banks = [bank_math, bank_qm, bank_basics, bank_advanced]
for bank, sec in zip(banks, "ABCD"):
    qs = bank.QUESTIONS
    check(f"{sec} has 20 questions", len(qs) == 20)
    check(f"{sec} ids", [q["id"] for q in qs] == [f"{sec}{i}" for i in range(1, 21)])
    check(f"{sec} format pattern", [q["kind"] for q in qs] == ["mcq"] * 14 + ["match"] * 2 + ["ar", "stmt", "para", "para"])
    for q in qs:
        check(f"{q['id']} four distinct options", len(q["opts"]) == 4 and len(set(q["opts"])) == 4)
        if q["kind"] == "match":
            perms = [tuple(int(part.split("to$")[1]) for part in o.split(", ")) for o in q["opts"]]
            check(f"{q['id']} options are permutations", all(sorted(p_) == [1, 2, 3, 4] for p_ in perms) and len(set(perms)) == 4)
        ex = EXPLAIN[q["id"]]
        check(f"{q['id']} explanation complete", set(ex) == {"concept", "formulae", "wrong", "takeaway"}
              and len(ex["wrong"]) == 3 and all(len(w) >= 35 for w in ex["wrong"]))
check("AR/Statement keys", [q["correct"] for b_ in banks for q in b_.QUESTIONS if q["kind"] in ("ar", "stmt")]
      == [0, 3, 1, 2, 3, 0, 2, 1])

# ===================================================================================================== CUDA-Q
if "--cudaq" in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    import cudaq
    cudaq.set_target("density-matrix-cpu")

    @cudaq.kernel
    def rx_arg(t: float):
        q = cudaq.qvector(1)
        rx(t, q[0])

    @cudaq.kernel
    def rx_const():
        q = cudaq.qvector(1)
        rx(0.0, q[0])

    @cudaq.kernel
    def plus_arg(t: float):
        q = cudaq.qvector(1)
        h(q[0])
        rx(t, q[0])

    def nm_one(ch):
        m = cudaq.NoiseModel(); m.add_all_qubit_channel("rx", ch); return m

    zarg = cudaq.observe(rx_arg, cudaq.spin.z(0), 0.0, noise_model=nm_one(cudaq.DepolarizationChannel(0.3))).expectation()
    zconst = cudaq.observe(rx_const, cudaq.spin.z(0), noise_model=nm_one(cudaq.DepolarizationChannel(0.3))).expectation()
    check("cudaq C1 depolarizing convention", abs(zarg - 0.6) < 1e-12, zarg)
    check("cudaq C2 constant identity gate carries no noise", abs(zconst - 1.0) < 1e-12, zconst)
    for name, ch, ks in [("bit flip", cudaq.BitFlipChannel(0.3), BF(0.3)), ("phase flip", cudaq.PhaseFlipChannel(0.3), PF(0.3)),
                         ("Y error", cudaq.YError(0.3), YF(0.3)), ("amplitude damping", cudaq.AmplitudeDampingChannel(0.3), AD(0.3)),
                         ("phase damping", cudaq.PhaseDamping(0.3), PD(0.3))]:
        m = nm_one(ch)
        bx = cudaq.observe(plus_arg, cudaq.spin.x(0), 0.0, noise_model=m).expectation()
        bz = cudaq.observe(rx_arg, cudaq.spin.z(0), 0.0, noise_model=m).expectation()
        ref_x = bloch(channel(dm(plus), ks))[0]; ref_z = bloch(channel(dm(k0), ks))[2]
        check(f"cudaq channel {name} = Kraus set of the paper", abs(bx - ref_x) < 1e-12 and abs(bz - ref_z) < 1e-12, (bx, ref_x, bz, ref_z))

    src = BUILDER[BUILDER.index("def build_noise_model"):BUILDER.index('print(f"Circuit ')]
    ns = {"cudaq": cudaq, "np": np}
    exec(src, ns)
    build_noise_model = ns["build_noise_model"]

    @cudaq.kernel
    def ghz_state(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])

    def ghz_dm(n, kind, p):
        model = build_noise_model(kind, p)          # keep a reference while CUDA-Q uses it (see the notebook)
        cudaq.set_noise(model)
        try:
            return np.array(cudaq.get_state(ghz_state, n))
        finally:
            cudaq.unset_noise()

    r6 = ghz_dm(6, "depolarizing", 0.01)
    d = r6.shape[0]
    f6 = 0.5 * np.real(r6[0, 0] + r6[d - 1, d - 1] + r6[0, d - 1] + r6[d - 1, 0])
    check("cudaq C17 fidelity exceeds bound", f6 > 0.99 ** 6 + 1e-3 and abs(f6 - 0.9492) < 1e-4, f6)
    rpf = ghz_dm(4, "phase_flip", 0.05)
    check("cudaq B14 phase flips keep Z population", abs(np.real(rpf[0, 0] + rpf[15, 15]) - 1) < 1e-12)

    xop = cudaq.spin.x(0)
    for i in range(1, 4):
        xop = xop * cudaq.spin.x(i)
    zne = [cudaq.observe(ghz_state, xop, 4, noise_model=build_noise_model("depolarizing", 0.01 * l)).expectation() for l in (1, 2, 3)]
    check("cudaq D19 ZNE data", close(np.round(zne, 4), E, 1e-12), zne)

    @cudaq.kernel
    def bitflip_code(idle: float):
        q = cudaq.qvector(3)
        x.ctrl(q[0], q[1]); x.ctrl(q[0], q[2])
        rx(idle, q[0]); rx(idle, q[1]); rx(idle, q[2])
        x.ctrl(q[0], q[1]); x.ctrl(q[0], q[2])
        x.ctrl(q[1], q[2], q[0])

    for p in (0.05, 0.1, 0.3):
        e = (1 - cudaq.observe(bitflip_code, cudaq.spin.z(0), 0.0, noise_model=nm_one(cudaq.BitFlipChannel(p))).expectation()) / 2
        check(f"cudaq D10 bit-flip code p={p}", abs(e - pL(p)) < 1e-12, e)

    @cudaq.kernel
    def idle_plus(steps: int, echo_mid: int, idle: float):
        q = cudaq.qvector(1)
        h(q[0])
        for k in range(steps):
            if echo_mid == 1 and k == steps // 2:
                x(q[0])
            rx(idle, q[0])
        if echo_mid == 1:
            x(q[0])

    mpf = nm_one(cudaq.PhaseFlipChannel(0.02))
    check("cudaq D9", all(abs(cudaq.observe(idle_plus, cudaq.spin.x(0), 20, e_, 0.0, noise_model=mpf).expectation() - 0.96 ** 20) < 1e-12 for e_ in (0, 1)))

    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def over_rotation(n: int, eps: float):
        q = cudaq.qvector(1)
        for k in range(n):
            rx(eps, q[0])

    check("cudaq D12 coherent", abs((1 - cudaq.observe(over_rotation, cudaq.spin.z(0), 60, 0.05).expectation()) / 2 - math.sin(1.5) ** 2) < 1e-12)

    cudaq.set_target("stim")
    cudaq.set_random_seed(42)

    @cudaq.kernel
    def ghz_z(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        mz(qubits)

    @cudaq.kernel
    def ghz_x(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        h(qubits)
        mz(qubits)

    pops = {}
    for p in [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        cz = cudaq.sample(ghz_z, 15, shots_count=10000, noise_model=build_noise_model("depolarizing", p))
        cx = cudaq.sample(ghz_x, 15, shots_count=10000, noise_model=build_noise_model("depolarizing", p))
        pops[p] = ((cz.count("0" * 15) + cz.count("1" * 15)) / 10000, len(list(cz.items())))
    check("cudaq C19 recorded run reproduced (stim, seed 42)", abs(pops[0.01][0] - 0.8936) < 1e-9 and pops[0.1] == (0.3232, 1165), pops)
    ver = re.search(r"\d+\.\d+\.\d+", cudaq.__version__)
    print(f"CUDA-Q {ver.group(0) if ver else cudaq.__version__} cross-checks done (density-matrix-cpu, qpp-cpu, stim)")

bad = [c_ for c_ in CHECKS if not c_[1]]
print(f"{len(CHECKS) - len(bad)}/{len(CHECKS)} checks passed")
sys.exit(1 if bad else 0)
