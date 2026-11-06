"""Generate all figures and numbers for the HHL lecture slides (vector PDFs + hhl_numbers.tex in ./figures).

Quantum results use NVIDIA CUDA-Q (qpp-cpu) with the circuit of HHL.ipynb (build_hhl_kernel, copied line for line,
plus an option to stop after phase estimation for the clock-register figure) and are cross-checked with NumPy.
Run:  python make_figures.py      (needs cudaq, numpy, scipy, matplotlib)
"""
import os
import re
import hashlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm
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
SEED, SHOTS = 42, 20000
NUM = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- notebook set-up (copied from HHL.ipynb, Section 4)
H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
A_DEF = H4 @ np.diag([1.0, 2.0, 3.0, 4.0]) @ H4
B_DEF = np.array([1.0, 0, 0, 0])


def auto_time(A, n_c):
    """Section 4 of the notebook: EVOLUTION_TIME = "auto"."""
    lam = np.linalg.eigvalsh(A)
    lmin, lmax = float(np.min(np.abs(lam))), float(np.max(np.abs(lam)))
    k_max = 2 ** (n_c - 1) - 1
    if lmax / lmin <= k_max:
        return 2 * np.pi / (2 ** n_c * lmin)
    return 2 * np.pi * k_max / (2 ** n_c * lmax)


def bit_reverse_order(vector, n):
    return np.asarray(vector).reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)


_uid_count = [0]


def build_hhl_kernel(A, b, n_c, t, measure=False, for_statistics=False, stop_after_qpe=False):
    """build_hhl_kernel of HHL.ipynb (Section 7a). stop_after_qpe=True is only used for the clock-register figure."""
    A = A.astype(complex); b = b.astype(complex)
    n_b = int(np.log2(A.shape[0])); n_tot = n_c + n_b + 1
    b_norm = float(np.linalg.norm(b)); U = expm(1j * A * t)
    _uid_count[0] += 1
    uid = hashlib.sha1(np.r_[A.real.ravel(), A.imag.ravel(), t, n_c, _uid_count[0]].tobytes()).hexdigest()[:10]
    kernel = cudaq.make_kernel()
    initial = np.kron(np.kron(np.eye(2 ** n_c)[0], b / b_norm), np.array([1.0, 0.0]))
    reg = kernel.qalloc(n_tot) if for_statistics else kernel.qalloc(bit_reverse_order(initial.astype(complex), n_tot))
    clock = [reg[i] for i in range(n_c)]; system = [reg[n_c + i] for i in range(n_b)]; anc = reg[n_tot - 1]
    for q in clock:
        kernel.h(q)
    for j in range(n_c):
        name = f"hhl_{uid}_{n_c}_u{j}"
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
    if stop_after_qpe:
        return kernel
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
        name = f"hhl_{uid}_{n_c}_uinv{j}"
        cudaq.register_operation(name, np.linalg.matrix_power(U.conj().T, 2 ** j))
        getattr(kernel, name)(clock[n_c - 1 - j], *system)
    for q in clock:
        kernel.h(q)
    if measure:
        kernel.mz(reg)
    return kernel


def run_hhl(A, b, n_c, t):
    """State read-out as in the notebook's execution cell: post-selected amplitudes, C, x_quantum."""
    n_b = int(np.log2(A.shape[0])); n_tot = n_c + n_b + 1
    sv = np.array(cudaq.get_state(build_hhl_kernel(A, b, n_c, t)))
    amp = sv.reshape([2] * n_tot).transpose(list(range(n_tot))[::-1])[(0,) * n_c + (slice(None),) * n_b + (1,)].reshape(-1)
    C = 2 * np.pi / (2 ** n_c * t)
    return amp, C, amp * np.linalg.norm(b) / C


def sci(v, digits=2):
    m, e = f"{v:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def fid(u, v):
    return float(abs(np.vdot(u / np.linalg.norm(u), v / np.linalg.norm(v))) ** 2)


X_DEF = np.linalg.solve(A_DEF, B_DEF)

# ---------------------------------------------------------------- 1. default run (notebook settings), state + shots
t4 = auto_time(A_DEF, 4)
assert abs(t4 - np.pi / 8) < 1e-15
cudaq.set_random_seed(SEED)
amp, C4, xq = run_hhl(A_DEF, B_DEF, 4, t4)
counts = cudaq.sample(build_hhl_kernel(A_DEF, B_DEF, 4, t4, measure=True), shots_count=SHOTS)
post = {}
for bits, c in counts.items():
    if bits[:4] == "0000" and bits[-1] == "1":
        post[bits[4:6]] = post.get(bits[4:6], 0) + c
acc = sum(post.values())
sampled = np.array([post.get(format(i, "02b"), 0) / acc for i in range(4)])
exact_p = X_DEF ** 2 / np.sum(X_DEF ** 2)
P_state = float(np.linalg.norm(amp) ** 2)
assert np.allclose(xq, X_DEF, atol=1e-12) and abs(P_state - 0.25 * (1 + 1 / 4 + 1 / 9 + 1 / 16)) < 1e-12
NUM.update(pstate=f"{P_state:.6f}", acc=acc, pshots=f"{acc / SHOTS:.4f}",
           relerr=sci(np.linalg.norm(xq - X_DEF) / np.linalg.norm(X_DEF)), fiddef=f"{fid(xq, X_DEF):.10f}",
           shotmax=f"{np.max(np.abs(sampled - exact_p)):.4f}", staterr=f"{1 / np.sqrt(acc):.3f}")
NUM["sampledrow"] = ", ".join(f"{v:.4f}" for v in sampled)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.6))
idx = np.arange(4)
ax[0].bar(idx - 0.2, X_DEF, width=0.38, color=GREY, label="classical (NumPy)")
ax[0].bar(idx + 0.2, np.real(xq), width=0.38, color=BLUE, label="HHL, state read-out")
ax[0].set_xticks(idx, [f"$x_{i}$" for i in idx]); ax[0].set_ylabel("solution component"); ax[0].legend(fontsize=7)
ax[0].set_title(f"fidelity {fid(xq, X_DEF):.6f}", fontsize=8); ax[0].grid(axis="x", visible=False)
ax[1].bar(idx - 0.2, exact_p, width=0.38, color=GREY, label=r"exact $|x_i|^2/\|x\|^2$")
ax[1].bar(idx + 0.2, sampled, width=0.38, color=ORANGE, label=f"post-selected shots ({acc})")
ax[1].set_xticks(idx, [f"$x_{i}$" for i in idx]); ax[1].set_ylabel("probability"); ax[1].legend(fontsize=7)
ax[1].set_ylim(0, 0.95); ax[1].grid(axis="x", visible=False)
save(fig, "default_solution")

# ---------------------------------------------------------------- 2. clock register after phase estimation
def clock_distribution(A, b, n_c, t):
    n_b = int(np.log2(A.shape[0])); n_tot = n_c + n_b + 1
    sv = np.array(cudaq.get_state(build_hhl_kernel(A, b, n_c, t, stop_after_qpe=True)))
    tens = np.abs(sv.reshape([2] * n_tot).transpose(list(range(n_tot))[::-1])) ** 2
    return tens.reshape(2 ** n_c, -1).sum(axis=1)


pc4 = clock_distribution(A_DEF, B_DEF, 4, t4)
t3 = auto_time(A_DEF, 3)
pc3 = clock_distribution(A_DEF, B_DEF, 3, t3)
assert np.allclose(pc4[1:5], 0.25, atol=1e-12)
NUM.update(pkfour=f"{pc3[3]:.4f}", pkthree=f"{pc3[2]:.4f}", pkoneofthree=f"{pc3[1]:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5))
signed = lambda k, M: k - M if k >= M // 2 else k
ax[0].bar(range(16), pc4, color=BLUE); ax[0].set_xticks(range(16)); ax[0].tick_params(axis="x", labelsize=6)
ax[0].set_title(r"$n_c=4$, $t=\pi/8$: $k_j=\lambda_j$ exactly", fontsize=8)
ax[1].bar(range(8), pc3, color=ORANGE); ax[1].set_xticks(range(8))
ax[1].set_title(r"$n_c=3$, $t=3\pi/16$: $k_j = 0.75,1.5,2.25,3$", fontsize=8)
for a in ax:
    a.set_xlabel("clock register value $k$ (unsigned)"); a.grid(axis="x", visible=False)
ax[0].set_ylabel("probability")
save(fig, "clock_distribution")

# ---------------------------------------------------------------- 3. eigenvalue-inversion lookup table (n_c = 4, t = pi/8)
ks = [k for k in range(1, 16)]
lam_k = [2 * np.pi * signed(k, 16) / (16 * t4) for k in ks]
amp_k = [np.clip(C4 / l, -1, 1) for l in lam_k]
fig, ax = plt.subplots(figsize=(6.2, 2.4))
ax.bar(ks, amp_k, color=[BLUE if a > 0 else ORANGE for a in amp_k])
ax.set_xticks(ks, [f"{k}\n({signed(k, 16):+d})" for k in ks], fontsize=7)
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel(r"clock value $k$ (signed value $\tilde k$ below)"); ax.set_ylabel(r"$C/\tilde\lambda_k=\sin(\theta_k/2)$")
ax.grid(axis="x", visible=False)
save(fig, "inversion_table")

# ---------------------------------------------------------------- 4. precision: fidelity and P_success vs n_c (auto time)
ncs = [2, 3, 4, 5, 6]
fids, ps, cs, ideal_formula = [], [], [], []
lam, V = np.linalg.eigh(A_DEF)
beta = V.T @ B_DEF
for n in ncs:
    t_n = auto_time(A_DEF, n)
    a_n, C_n, _ = run_hhl(A_DEF, B_DEF, n, t_n)
    fids.append(fid(a_n, X_DEF)); ps.append(float(np.linalg.norm(a_n) ** 2)); cs.append(C_n)
    ideal_formula.append(float(np.sum(beta ** 2 * (C_n / lam) ** 2)))
NUM.update(fidtwo=f"{fids[0]:.4f}", fidthree=f"{fids[1]:.10f}", pthree=f"{ps[1]:.4f}", idealthree=f"{ideal_formula[1]:.6f}",
           ptwo=f"{ps[0]:.4f}", cthree=f"{cs[1]:.4f}")
assert abs(fids[1] - 0.9532157640) < 1e-8 and abs(ideal_formula[1] - 0.632716) < 1e-6
assert all(abs(f - 1) < 1e-12 for f in fids[2:])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
ax[0].plot(ncs, fids, "o-", color=BLUE); ax[0].set_xlabel("clock qubits $n_c$"); ax[0].set_ylabel(r"fidelity of $|x\rangle$")
ax[0].set_xticks(ncs)
ax[1].plot(ncs, ps, "o-", color=ORANGE); ax[1].set_xlabel("clock qubits $n_c$"); ax[1].set_ylabel(r"$P_{\rm success}$ (state)")
ax[1].set_xticks(ncs)
save(fig, "precision")

# ---------------------------------------------------------------- 5. 2x2 example of the notebook (Section 4 table)
A2 = np.array([[1, -1 / 3], [-1 / 3, 1]]); b2 = np.array([1.0, 0])
t2 = auto_time(A2, 4)
a2, C2, x2 = run_hhl(A2, b2, 4, t2)
assert abs(t2 - 3 * np.pi / 16) < 1e-14 and np.allclose(x2, [1.125, 0.375], atol=1e-12)
NUM.update(ptwobytwo=f"{np.linalg.norm(a2) ** 2:.6f}", fidtwobytwo=f"{fid(x2, np.linalg.solve(A2, b2)):.10f}")

# ---------------------------------------------------------------- 6. success probability vs condition number (illustrative)
kap = np.linspace(1, 40, 200)
worst = 1 / kap ** 2
uniform = [np.mean((1 / np.linspace(1, k, 4)) ** 2) for k in kap]
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.loglog(kap, uniform, color=BLUE, label=r"$\lambda_j$ = 4 evenly spaced values in $[1,\kappa]$, all $|\beta_j|^2=\frac14$")
ax.loglog(kap, worst, color=ORANGE, label=r"$|b\rangle$ = top eigenvector: $1/\kappa^2$")
ax.plot([4], [0.355903], "*", color=INK, ms=9); ax.text(4.4, 0.36, "notebook default", fontsize=7)
ax.set_xlabel(r"condition number $\kappa$ ($C=\lambda_{\min}=1$)"); ax.set_ylabel(r"$P_{\rm success}$"); ax.legend(fontsize=7)
save(fig, "kappa_success")

# ---------------------------------------------------------------- 7. circuit size vs n_c (cudaq.estimate_resources, as Section 7b)
rows = []
for n in [2, 3, 4, 5, 6, 7]:
    res = cudaq.estimate_resources(build_hhl_kernel(A_DEF, B_DEF, n, auto_time(A_DEF, n), measure=True, for_statistics=True))
    d = dict(res.to_dict()); meas = d.get("mz", 0)
    rows.append((n, res.count() - meas, res.depth, res.multi_qubit_gate_count, d))
r4 = [r for r in rows if r[0] == 4][0]
assert r4[1] == 111 and r4[2] == 67 and r4[3] == 39, r4
NUM["gatesrows"] = "".join(rf"{n} & {n + 3} & {g} & {2 ** n - 1} & {dep}\\" for n, g, dep, _, _ in rows)
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.semilogy([r[0] for r in rows], [r[1] for r in rows], "o-", color=BLUE, label="gates (estimate_resources)")
ax.semilogy([r[0] for r in rows], [r[2] for r in rows], "s-", color=AQUA, label="depth")
ax.semilogy([r[0] for r in rows], [2 ** r[0] - 1 for r in rows], ":", color=ORANGE, label=r"rotations $2^{n_c}-1$")
ax.set_xlabel("clock qubits $n_c$ (4x4 system)"); ax.legend(fontsize=7)
save(fig, "circuit_size")

# ---------------------------------------------------------------- 8. shot noise of the read-out
shot_list = [1000, 3000, 10000, 30000, 100000]
ks_err = []
ksample = build_hhl_kernel(A_DEF, B_DEF, 4, t4, measure=True)
for S in shot_list:
    errs = []
    for _ in range(20):
        c = cudaq.sample(ksample, shots_count=S)
        pp = {}
        for bits, v in c.items():
            if bits[:4] == "0000" and bits[-1] == "1":
                pp[bits[4:6]] = pp.get(bits[4:6], 0) + v
        tot = sum(pp.values())
        errs.append(np.max(np.abs(np.array([pp.get(format(i, "02b"), 0) / tot for i in range(4)]) - exact_p)))
    ks_err.append(np.mean(errs))
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.loglog(shot_list, ks_err, "o-", color=BLUE, label=r"mean max error of $|x_i|^2$ (20 repeats)")
ax.loglog(shot_list, 1 / np.sqrt(P_state * np.array(shot_list)), ":", color=GREY, label=r"$1/\sqrt{P_{\rm success}\,S}$")
ax.set_xlabel("shots $S$"); ax.set_ylabel("error"); ax.legend(fontsize=7)
save(fig, "shot_noise")

names = [re.sub("[^a-z]", "", k) for k in NUM]
assert len(set(names)) == len(names)
with open(os.path.join(OUT, "hhl_numbers.tex"), "w") as fh:
    fh.write("% GENERATED by make_figures.py - do not edit\n")
    for nm, val in zip(names, NUM.values()):
        fh.write(rf"\newcommand{{\num{nm}}}{{{val}}}" + "\n")
print(NUM)
for r in rows:
    print(r)
