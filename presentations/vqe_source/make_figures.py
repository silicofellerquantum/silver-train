"""Generate all figures and numbers for the VQE lecture slides (vector PDFs + vqe_numbers.tex in ./figures).

Quantum results use NVIDIA CUDA-Q (qpp-cpu) with exactly the kernels and Hamiltonians of VQE.ipynb and are cross-checked
with NumPy. The H2 dissociation curve (illustrative, not part of the notebook) uses PySCF + OpenFermion.
Run:  python make_figures.py      (needs cudaq, numpy, scipy, matplotlib, pyscf, openfermion, openfermionpyscf)
"""
import os
import re
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce
from scipy.optimize import minimize, minimize_scalar
import cudaq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#c3c2b7", "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "figure.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 1.8,
})
cudaq.set_target("qpp-cpu")
SEED = 42
cudaq.set_random_seed(SEED)
NUM = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- notebook data and kernels (copied from VQE.ipynb)
H2_TERMS = [(-0.0970662681676282, "IIII"), (0.17141282644776884, "ZIII"), (0.17141282644776884, "IZII"),
            (-0.22343153690813597, "IIZI"), (-0.22343153690813597, "IIIZ"), (0.16868898170361213, "ZZII"),
            (0.12062523483390425, "ZIZI"), (0.16592785033770352, "ZIIZ"), (0.16592785033770352, "IZZI"),
            (0.12062523483390425, "IZIZ"), (0.17441287612261608, "IIZZ"), (-0.0453026155037993, "XXYY"),
            (0.0453026155037993, "XYYX"), (0.0453026155037993, "YXXY"), (-0.0453026155037993, "YYXX")]


def spin_op(terms):
    out = None
    for c, w in terms:
        t = c * cudaq.SpinOperator.from_word(w)
        out = t if out is None else out + t
    return out


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


@cudaq.kernel
def hea_sampled(n: int, layers: int, theta: list[float]):
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
    mz(qubits)


P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1., -1.])}
op = lambda w: reduce(np.kron, [P[c] for c in w])
H2M = sum(c * op(w) for c, w in H2_TERMS)
EV = np.linalg.eigvalsh(H2M)
FCI = EV[0]
H2OP = spin_op(H2_TERMS)
E_ucc = lambda t: cudaq.observe(ucc_ansatz, H2OP, [float(t)]).expectation()
EHF = E_ucc(0.0)

# ---------------------------------------------------------------- 1. H2: energy landscape of the UCC angle
th = np.linspace(-np.pi / 2, np.pi / 2, 181)
E = np.array([E_ucc(t) for t in th])
res = minimize_scalar(E_ucc, bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-10})
NUM.update(fci=f"{FCI:.6f}", ehf=f"{EHF:.6f}", ecorr=f"{(FCI - EHF) * 1000:.1f}", thetastar=f"{res.x:.4f}",
           eucc=f"{res.fun:.9f}", uccerr=f"{abs(res.fun - FCI):.1e}", pdouble=f"{np.sin(res.x) ** 2:.4f}")
fig, ax = plt.subplots(figsize=(4.8, 2.7))
ax.plot(th, E, color=BLUE, label=r"$E(\theta)$ from cudaq.observe")
ax.axhline(FCI, color=INK, lw=1); ax.text(-1.5, FCI - 0.05, f"exact (FCI) {FCI:.6f}", fontsize=7)
ax.axhline(EHF, color=GREY, lw=1, ls="--"); ax.text(0.35, EHF + 0.02, f"Hartree-Fock {EHF:.6f}", fontsize=7)
ax.plot(res.x, res.fun, "*", color=ORANGE, ms=11, label=rf"minimum $\theta^*={res.x:.4f}$")
ax.set_xlabel(r"UCC angle $\theta$"); ax.set_ylabel("energy (Ha)"); ax.legend(loc="upper center", fontsize=7)
save(fig, "h2_landscape")

# ---------------------------------------------------------------- 2. H2: the notebook's VQE loop (COBYLA from theta = 0)
hist = []


def obj_ucc(t):
    e = E_ucc(t[0]); hist.append(e); return e


minimize(obj_ucc, np.zeros(1), method="COBYLA", options={"maxiter": 300, "rhobeg": 0.3})
best = np.minimum.accumulate(hist)
first_acc = int(np.argmax(best - FCI < 1.6e-3)) + 1
NUM.update(hevals=len(hist), hbest=f"{best[-1]:.10f}", herr=f"{best[-1] - FCI:.1e}", hacc=first_acc)
fig, ax = plt.subplots(figsize=(4.8, 2.6))
ax.semilogy(np.arange(1, len(hist) + 1), np.maximum(best - FCI, 1e-16), color=BLUE, marker="o", ms=3)
ax.axhline(1.6e-3, color=ORANGE, lw=1); ax.text(len(hist), 1.8e-3, "chemical accuracy 1.6 mHa ", ha="right", fontsize=7)
ax.set_xlabel("optimizer evaluation (COBYLA)"); ax.set_ylabel(r"$E_{\rm best}-E_{\rm FCI}$ (Ha)")
save(fig, "h2_convergence")

# ---------------------------------------------------------------- 3. H2 dissociation (illustrative, PySCF + OpenFermion)
try:
    import openfermion as of
    from openfermionpyscf import run_pyscf

    def words_of(qop, n):
        out = {}
        for term, c in qop.terms.items():
            w = ["I"] * n
            for q, p in term:
                w[q] = p
            out["".join(w)] = out.get("".join(w), 0) + c
        return [(float(np.real(c)), w) for w, c in out.items() if abs(c) > 1e-12]

    Rs = np.linspace(0.35, 2.6, 19)
    hf, fci, vqe = [], [], []
    for R in Rs:
        m = run_pyscf(of.MolecularData([("H", (0, 0, 0)), ("H", (0, 0, R))], "sto-3g", 1, 0), run_scf=True, run_fci=True)
        terms = words_of(of.jordan_wigner(of.get_fermion_operator(m.get_molecular_hamiltonian())), 4)
        hop = spin_op(terms)
        r = minimize_scalar(lambda t: cudaq.observe(ucc_ansatz, hop, [float(t)]).expectation(), bounds=(-np.pi / 2, np.pi / 2),
                            method="bounded", options={"xatol": 1e-9})
        hf.append(m.hf_energy); fci.append(m.fci_energy); vqe.append(r.fun)
    m074 = run_pyscf(of.MolecularData([("H", (0, 0, 0)), ("H", (0, 0, 0.74))], "sto-3g", 1, 0), run_scf=True, run_fci=True)
    nb = dict((w, c) for c, w in H2_TERMS)
    t074 = dict((w, c) for c, w in words_of(of.jordan_wigner(of.get_fermion_operator(m074.get_molecular_hamiltonian())), 4))
    NUM["coefdiff"] = f"{max(abs(t074[w] - c) for w, c in nb.items()):.0e}"
    fig, ax = plt.subplots(figsize=(4.8, 2.7))
    ax.plot(Rs, hf, color=GREY, ls="--", label="Hartree-Fock (RHF)")
    ax.plot(Rs, fci, color=INK, label="exact (FCI)")
    ax.plot(Rs, vqe, "o", color=BLUE, ms=3.5, label="VQE, notebook's UCC kernel")
    ax.axvline(0.74, color=AQUA, lw=0.8); ax.text(0.76, -0.62, "notebook\nR = 0.74", fontsize=7, color=AQUA)
    ax.set_xlabel(r"bond length $R$ (Angstrom)"); ax.set_ylabel("energy (Ha)"); ax.legend(fontsize=7)
    save(fig, "h2_dissociation")
    NUM["vqemaxdev"] = f"{max(abs(a - b) for a, b in zip(vqe, fci)):.0e}"
    NUM["hfgapfar"] = f"{(hf[-1] - fci[-1]) * 1000:.0f}"
except ImportError:
    print("PySCF/OpenFermion not installed: h2_dissociation skipped")

# ---------------------------------------------------------------- 4. variational principle picture (one qubit, -Z + X/2)
t = np.linspace(-np.pi, np.pi, 361)
fig, ax = plt.subplots(figsize=(4.4, 2.4))
ax.plot(t, -np.cos(t) + 0.5 * np.sin(t), color=BLUE, label=r"$E(\theta)=-\cos\theta+\frac{1}{2}\sin\theta$")
ax.axhline(-np.sqrt(1.25), color=INK, lw=1); ax.text(-3, -1.05, r"$E_0=-\sqrt{5}/2$", fontsize=8)
ax.fill_between(t, -1.6, -np.sqrt(1.25), color=ORANGE, alpha=0.12); ax.text(0.6, -1.45, "no trial state can go here", fontsize=7, color=ORANGE)
ax.set_ylim(-1.6, 1.3); ax.set_xlabel(r"$\theta$"); ax.set_ylabel("energy"); ax.legend(fontsize=7, loc="upper left")
save(fig, "variational")

# ---------------------------------------------------------------- 5. TFIM default run (notebook settings) with CUDA-Q
N, L = 15, 2
tf_terms = [(-1.0, "".join("Z" if k in (i, i + 1) else "I" for k in range(N))) for i in range(N - 1)]
tf_terms += [(-0.5, "".join("X" if k == i else "I" for k in range(N))) for i in range(N)]
TF = spin_op(tf_terms)


def tfim_ff(n, J, h):
    A = np.zeros((2 * n, 2 * n))
    for j in range(n):
        A[2 * j, 2 * j + 1], A[2 * j + 1, 2 * j] = -h, h
    for j in range(n - 1):
        A[2 * j + 1, 2 * j + 2], A[2 * j + 2, 2 * j + 1] = -J, J
    return -0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(1j * A))))


E0 = tfim_ff(N, 1.0, 0.5)
theta0 = np.random.default_rng(SEED).normal(0.0, 0.1, (L + 1) * N)       # identical to the notebook
th_hist, e_hist = [], []


def obj_tf(th_):
    e = cudaq.observe(hea_ansatz, TF, N, L, [float(v) for v in th_]).expectation()
    e_hist.append(e); th_hist.append(np.array(th_)); return e


t0 = time.perf_counter()
minimize(obj_tf, theta0, method="COBYLA", options={"maxiter": 300, "rhobeg": 0.3})
tf_time = time.perf_counter() - t0
ib = int(np.argmin(e_hist))
counts = cudaq.sample(hea_sampled, N, L, [float(v) for v in th_hist[ib]], shots_count=4000)
top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
NUM.update(tfeo=f"{E0:.10f}", tfstart=f"{e_hist[0]:.4f}", tfbest=f"{e_hist[ib]:.10f}", tferr=f"{e_hist[ib] - E0:.3e}",
           tfrel=f"{(e_hist[ib] - E0) / abs(E0) * 100:.3f}", tfevals=len(e_hist), tftime=f"{tf_time:.0f}",
           tfpzero=f"{dict(counts.items()).get('0' * N, 0) / 4000:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
ax[0].plot(np.arange(1, len(e_hist) + 1), e_hist, color=GREY, lw=0.8, label="each evaluation")
ax[0].plot(np.arange(1, len(e_hist) + 1), np.minimum.accumulate(e_hist), color=BLUE, label="best so far")
ax[0].axhline(E0, color=INK, lw=1); ax[0].text(len(e_hist), E0 - 0.15, f"exact $E_0$ = {E0:.4f} ", ha="right", fontsize=7)
ax[0].axhline(-14, color=AQUA, lw=0.8, ls="--"); ax[0].text(len(e_hist), -13.95, r"$|0\cdots0\rangle$: $-14$ ", ha="right", fontsize=7, color=AQUA)
ax[0].set_ylim(E0 - 0.4, -13.3); ax[0].set_xlabel("optimizer evaluation"); ax[0].set_ylabel("energy"); ax[0].legend(fontsize=7, loc="upper center")
ax[1].semilogy(np.arange(1, len(e_hist) + 1), np.minimum.accumulate(e_hist) - E0, color=BLUE)
ax[1].set_xlabel("optimizer evaluation"); ax[1].set_ylabel(r"$E_{\rm best}-E_0$")
save(fig, "tfim_convergence")
NUM["tftop"] = "".join(rf"\texttt{{{b}}} & {c / 4000:.4f}\\" for b, c in top)

# ---------------------------------------------------------------- 6. H2 spectrum
fig, ax = plt.subplots(figsize=(3.8, 2.6))
nop = sum((np.eye(16) - op("I" * k + "Z" + "I" * (3 - k))) / 2 for k in range(4))
w, v = np.linalg.eigh(H2M)
ne = [int(round(np.real(v[:, i].conj() @ nop @ v[:, i]))) for i in range(16)]
cols = {0: GREY, 1: AQUA, 2: BLUE, 3: ORANGE, 4: INK}
for e, n in zip(w, ne):
    ax.hlines(e, n - 0.35, n + 0.35, color=cols[n], lw=2)
ax.set_xticks(range(5)); ax.set_xlabel("electron number"); ax.set_ylabel("eigenvalue (Ha)")
ax.text(2.4, FCI, " ground state\n (2 electrons)", fontsize=7, va="center")
save(fig, "h2_spectrum")

# ---------------------------------------------------------------- 7. scaling: memory and terms
n = np.arange(4, 41)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax[0].semilogy(n, 16 * 2.0 ** n, color=BLUE, label="state vector fp64")
ax[0].semilogy(n[n <= 24], 16 * 4.0 ** n[n <= 24], color=ORANGE, label="density matrix")
for y, lab in [(16e9, "16 GB"), (80e9, "80 GB GPU")]:
    ax[0].axhline(y, color=GREY, lw=0.8, ls="--"); ax[0].text(4.5, y * 1.5, lab, fontsize=7)
ax[0].set_xlabel("qubits"); ax[0].set_ylabel("bytes"); ax[0].legend(fontsize=7)
ax[1].loglog([4, 12], [15, 631], "o", color=BLUE)
for x_, y_, lab in [(4, 15, "H$_2$: 15"), (12, 631, "LiH: 631")]:
    ax[1].annotate(lab, (x_, y_), textcoords="offset points", xytext=(5, -10), fontsize=7)
nn = np.array([4, 40]); ax[1].loglog(nn, 631 * (nn / 12.0) ** 4, ls=":", color=GREY, label=r"$\propto N^4$")
ax[1].set_xlabel("spin-orbitals (qubits)"); ax[1].set_ylabel("Pauli terms"); ax[1].legend(fontsize=7)
save(fig, "scaling")

# ---------------------------------------------------------------- 8. noise model
p = np.linspace(0, 1, 101)
fig, ax = plt.subplots(figsize=(4.2, 2.3))
ax.plot(p, (1 - p) * FCI + p * H2_TERMS[0][0], color=BLUE, label=r"$(1-p)E_{\rm FCI}+p\,c_{IIII}$")
ax.axhline(EHF, color=GREY, ls="--", lw=1); ax.text(0.02, EHF + 0.03, "Hartree-Fock", fontsize=7)
ax.set_xlabel("global depolarising probability $p$"); ax.set_ylabel("measured energy (Ha)"); ax.legend(fontsize=7)
save(fig, "noise_model")
NUM["pbreakeven"] = f"{(EHF - FCI) / (H2_TERMS[0][0] - FCI):.3f}"

names = [re.sub("[^a-z]", "", k) for k in NUM]
assert len(set(names)) == len(names)
with open(os.path.join(OUT, "vqe_numbers.tex"), "w") as fh:
    fh.write("% GENERATED by make_figures.py - do not edit\n")
    for nm, val in zip(names, NUM.values()):
        fh.write(rf"\newcommand{{\num{nm}}}{{{val}}}" + "\n")
print(NUM)
