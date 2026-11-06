"""Generate all figures and numbers for the HHL lecture slides (vector PDFs + figures/hhl_numbers.tex).

Quantum results are computed with NVIDIA CUDA-Q (qpp-cpu simulator) using the same circuit as HHL.ipynb
(dev/notebook_builders/build_hhl.py) and cross-checked against NumPy (np.linalg.solve and an exact NumPy HHL).
Complexity curves are plots of the asymptotic formulas, not measurements; this is stated on the slides.
Run:  python make_figures.py
"""
import hashlib
import os
import warnings

warnings.filterwarnings("ignore", message=".*cudaq-logical is in preview.*")
warnings.filterwarnings("ignore", message=".*algorithmic primitives will change.*")

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
cudaq.set_random_seed(42)
NUMBERS = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def bit_reverse_order(vec, n):
    idx = [int(format(i, f"0{n}b")[::-1], 2) for i in range(2 ** n)]
    return np.asarray(vec)[idx]


H4 = 0.5 * np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]])
A_DEF = H4 @ np.diag([1.0, 2, 3, 4]) @ H4
B_DEF = np.array([1.0, 0, 0, 0])


# ------------------------------------------------------------------------------------- the notebook's circuit
def auto_time(A, n_c):
    lam = np.abs(np.linalg.eigvalsh(A))
    kappa, k_max = lam.max() / lam.min(), 2 ** (n_c - 1) - 1
    if kappa <= k_max:
        return 2 * np.pi / (2 ** n_c * lam.min())
    return 2 * np.pi * k_max / (2 ** n_c * lam.max())


def build_hhl(A, b, n_c, t, measure=False, clock_measure_only=False, stop_after_qpe=False):
    """HHL kernel exactly as in build_hhl.py (clock | system | ancilla, clock[0] = MSB)."""
    n_b = int(np.log2(A.shape[0]))
    n_tot = n_c + n_b + 1
    U = expm(1j * A * t)
    uid = hashlib.sha1(np.r_[A.real.ravel(), t, n_c].tobytes()).hexdigest()[:10]
    kernel = cudaq.make_kernel()
    init = np.kron(np.kron(np.eye(2 ** n_c)[0], b / np.linalg.norm(b)), [1.0, 0.0]).astype(complex)
    reg = kernel.qalloc(bit_reverse_order(init, n_tot))
    clock = [reg[i] for i in range(n_c)]
    system = [reg[n_c + i] for i in range(n_b)]
    anc = reg[n_tot - 1]
    for q in clock:
        kernel.h(q)
    for j in range(n_c):
        name = f"lec_{uid}_u{j}"
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
        if clock_measure_only:
            for q in clock:
                kernel.mz(q)
        return kernel
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
        for k in range(j + 1, n_c):
            kernel.cr1(np.pi / 2 ** (k - j), clock[k], clock[j])
    for j in range(n_c // 2):
        kernel.swap(clock[j], clock[n_c - 1 - j])
    for jj in range(n_c):
        j = n_c - 1 - jj
        name = f"lec_{uid}_uinv{j}"
        cudaq.register_operation(name, np.linalg.matrix_power(U.conj().T, 2 ** j))
        getattr(kernel, name)(clock[n_c - 1 - j], *system)
    for q in clock:
        kernel.h(q)
    if measure:
        kernel.mz(reg)
    return kernel


def run_hhl(A, b, n_c, t=None):
    t = auto_time(A, n_c) if t is None else t
    n_b = int(np.log2(A.shape[0]))
    n_tot = n_c + n_b + 1
    sv = np.array(cudaq.get_state(build_hhl(A, b, n_c, t)))
    tensor = sv.reshape([2] * n_tot).transpose(list(range(n_tot))[::-1])
    post = tensor[(0,) * n_c + (slice(None),) * n_b + (1,)].reshape(-1)
    C = 2 * np.pi / (2 ** n_c * t)
    x_c = np.linalg.solve(A, b)
    fid = abs(np.vdot(post / np.linalg.norm(post), x_c / np.linalg.norm(x_c))) ** 2
    return post, C, t, float(np.linalg.norm(post) ** 2), float(fid)


# ------------------------------------------------------------------------------------- 1. default run
post, C, t, p_succ, fid = run_hhl(A_DEF, B_DEF, 4)
x_c = np.linalg.solve(A_DEF, B_DEF)
x_q = post * np.linalg.norm(B_DEF) / C
assert np.allclose(x_q, x_c, atol=1e-10) and np.isclose(p_succ, 205 / 576)
counts = cudaq.sample(build_hhl(A_DEF, B_DEF, 4, t, measure=True), shots_count=20000)
post_counts = {}
for bits, c in counts.items():
    if bits[:4] == "0000" and bits[-1] == "1":
        post_counts[bits[4:6]] = post_counts.get(bits[4:6], 0) + c
acc = sum(post_counts.values())
sampled = np.array([post_counts.get(format(i, "02b"), 0) / acc for i in range(4)])
exact = np.abs(x_c) ** 2 / np.sum(np.abs(x_c) ** 2)
NUMBERS.update(defP=f"{p_succ:.6f}", defFid=f"{fid:.10f}", defErr=f"{np.linalg.norm(x_q - x_c) / np.linalg.norm(x_c):.1e}",
               defAcc=f"{acc}", defAccFrac=f"{acc / 20000:.4f}")

fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.8))
idx = np.arange(4)
axs[0].bar(idx - 0.2, x_c, 0.38, color=GREY, label="classical (NumPy)")
axs[0].bar(idx + 0.2, np.real(x_q), 0.38, color=BLUE, label="HHL state read-out (CUDA-Q)")
axs[0].set_xticks(idx, [f"$x_{i}$" for i in idx]); axs[0].set_ylim(0, 0.68)
axs[0].set_title(f"Solution vector: fidelity {fid:.6f}", loc="left"); axs[0].legend(loc="upper right")
axs[1].bar(idx - 0.2, exact, 0.38, color=GREY, label=r"exact $|x_i|^2/\|x\|^2$")
axs[1].bar(idx + 0.2, sampled, 0.38, color=ORANGE, label=f"post-selected shots ({acc} of 20000)")
axs[1].set_xticks(idx, [f"$x_{i}$" for i in idx]); axs[1].set_ylim(0, 1.0)
axs[1].set_title("Hardware-style read-out (magnitudes only)", loc="left"); axs[1].legend(loc="upper right")
for ax in axs:
    ax.grid(axis="x", visible=False)
save(fig, "solution_default")

# ------------------------------------------------------------------------------------- 2. clock register after QPE
fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.6))
for ax, n_c, col in ((axs[0], 4, BLUE), (axs[1], 3, ORANGE)):
    tt = auto_time(A_DEF, n_c)
    cnt = cudaq.sample(build_hhl(A_DEF, B_DEF, n_c, tt, clock_measure_only=True, stop_after_qpe=True), shots_count=20000)
    probs = np.zeros(2 ** n_c)
    for bits, c in cnt.items():
        probs[int(bits, 2)] += c / 20000
    ax.bar(np.arange(2 ** n_c), probs, color=col)
    ax.set_xticks(np.arange(2 ** n_c), [format(k, f"0{n_c}b") for k in range(2 ** n_c)], rotation=90, fontsize=7)
    ideal = np.array([1, 2, 3, 4]) * tt * 2 ** n_c / (2 * np.pi)
    ax.set_title(f"$n_c = {n_c}$, $t = {tt / np.pi:.4g}\\pi$: ideal $k = $ " + ", ".join(f"{v:.3g}" for v in ideal), loc="left", fontsize=9)
    ax.set_ylabel("probability (20 000 shots)"); ax.set_ylim(0, 0.32); ax.grid(axis="x", visible=False)
save(fig, "clock_qpe")

# ------------------------------------------------------------------------------------- 3. eigen decomposition of b
lam, V = np.linalg.eigh(A_DEF)
beta = V.T @ B_DEF
sign = np.sign(beta); beta = beta * sign; V = V * sign
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.bar(np.arange(4) - 0.2, beta, 0.38, color=GREY, label=r"$\beta_j$ (input $|b\rangle$)")
ax.bar(np.arange(4) + 0.2, beta / lam, 0.38, color=BLUE, label=r"$\beta_j/\lambda_j$ (after inversion)")
ax.set_xticks(np.arange(4), [f"$\\lambda = {int(l)}$" for l in lam]); ax.set_ylim(0, 0.62)
ax.legend(); ax.grid(axis="x", visible=False); ax.set_title("Coefficients in the eigenbasis of $A$", loc="left")
save(fig, "eigen_coeffs")

# ------------------------------------------------------------------------------------- 4. rotation C/lambda
lv = np.linspace(1, 8, 300)
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(lv, 1 / lv, color=BLUE, label=r"ancilla amplitude $C/\lambda$")
ax.plot(lv, 1 / lv ** 2, color=ORANGE, label=r"success probability $(C/\lambda)^2$")
ax.plot(lv, 2 * np.arcsin(1 / lv) / np.pi, color=AQUA, ls="--", label=r"angle $\theta/\pi = 2\arcsin(C/\lambda)/\pi$")
ax.set_xlabel(r"$\lambda / C$"); ax.legend(); ax.set_ylim(0, 1.05)
ax.set_title(r"Eigenvalue inversion with $C = \lambda_{\min}$", loc="left")
save(fig, "rotation")

# ------------------------------------------------------------------------------------- 5. fidelity vs n_c
rng = np.random.default_rng(11)
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
A_r = Q @ np.diag([1.0, 1.7, 2.9, 4.6]) @ Q.T
b_r = np.array([1.0, 1, 1, 1])
ncs, fids, ps = [3, 4, 5, 6, 7, 8], [], []
for n_c in ncs:
    _, _, _, pr, fr = run_hhl(A_r, b_r, n_c)
    fids.append(fr); ps.append(pr)
NUMBERS.update(fidThree=f"{fids[0]:.3f}", fidEight=f"{fids[-1]:.4f}")
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(ncs, fids, "o-", color=BLUE, label=r"fidelity $|\langle x_{\rm classical}|x_{\rm HHL}\rangle|^2$")
ax.plot(ncs, ps, "s--", color=ORANGE, label=r"$P_{\rm success}$")
ax.set_xlabel("clock qubits $n_c$"); ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
ax.set_title(r"Eigenvalues $1, 1.7, 2.9, 4.6$ (not on the grid)", loc="left")
save(fig, "fidelity_vs_nc")

# ------------------------------------------------------------------------------------- 6. success probability vs kappa
kap = np.linspace(1, 50, 200)
fig, ax = plt.subplots(figsize=(4.6, 2.5))
for d, col in ((4, BLUE), (64, AQUA)):
    pv = []
    for k in kap:
        lam_k = np.linspace(1, k, d)
        pv.append(np.mean(1 / lam_k ** 2))               # |b> spread evenly over the eigenvectors, C = lambda_min = 1
    ax.plot(kap, pv, color=col, label=f"$|b\\rangle$ spread evenly, $d = {d}$")
ax.plot(kap, 1 / kap ** 2, color=ORANGE, ls="--", label=r"worst case $1/\kappa^2$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"condition number $\kappa$"); ax.set_ylabel(r"$P_{\rm success}$")
ax.legend(); ax.set_title("Success probability falls with $\\kappa$", loc="left")
save(fig, "psuccess_kappa")

# ------------------------------------------------------------------------------------- 7. complexity formulas
dd = np.logspace(1, 9, 100)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
s, kk, eps = 5, 10, 1e-2
ax.plot(dd, dd ** 3, color=GREY, label=r"Gaussian elimination $d^3$")
ax.plot(dd, dd * s * np.sqrt(kk) * np.log(1 / eps), color=BLUE, label=r"conjugate gradient $d\,s\sqrt{\kappa}\log(1/\epsilon)$")
ax.plot(dd, np.log2(dd) * s ** 2 * kk ** 2 / eps, color=ORANGE, label=r"HHL $\log d\; s^2\kappa^2/\epsilon$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("dimension $d$"); ax.set_ylabel("operations (formula)")
ax.legend(fontsize=7); ax.set_title(r"Asymptotic formulas, $s = 5$, $\kappa = 10$, $\epsilon = 0.01$ (no constants)", loc="left", fontsize=8)
save(fig, "complexity")

# ------------------------------------------------------------------------------------- 8. condition number picture
fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6))
xs = np.linspace(-2, 4, 50)
for ax, (a, title) in zip(axs, ((1.0, "well conditioned"), (0.1, "ill conditioned"))):
    M = np.array([[1, 1], [1, 1 + a]])
    for db, col in ((0, BLUE), (0.2, ORANGE)):
        ax.plot(xs, 2 - xs, color=GREY)
        ax.plot(xs, (2 + a + db - xs) / (1 + a), color=col, ls="-" if db == 0 else "--")
        sol = np.linalg.solve(M, [2, 2 + a + db])
        ax.plot(*sol, "o", color=col, ms=7, zorder=5)
    ax.set_xlim(-2, 4); ax.set_ylim(-1.5, 4); ax.set_title(title + f": $\\kappa = {np.linalg.cond(M):.1f}$, $b_2$ shifted by 0.2", loc="left", fontsize=8)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
save(fig, "conditioning")

# ------------------------------------------------------------------------------------- 9. noise: dephasing of the clock (NumPy, exact average)
def hhl_with_clock_z(A, b, n_c, t, p):
    """Exact average over Z errors (prob p each) on the clock between QPE and rotation; returns (P, fidelity)."""
    d = A.shape[0]; Nc = 2 ** n_c; Cc = 2 * np.pi / (Nc * t)
    U = expm(1j * A * t)
    cu = np.zeros((Nc * d, Nc * d), dtype=complex)
    for k in range(Nc):
        cu[k * d:(k + 1) * d, k * d:(k + 1) * d] = np.linalg.matrix_power(U, k)
    Hn = np.array([[1, 1], [1, -1]]) / np.sqrt(2); Hall = np.array([[1.0]])
    for _ in range(n_c):
        Hall = np.kron(Hall, Hn)
    F = np.array([[np.exp(2j * np.pi * x * y / Nc) for x in range(Nc)] for y in range(Nc)]) / np.sqrt(Nc)
    qpe = np.kron(F.conj().T, np.eye(d)) @ cu @ np.kron(Hall, np.eye(d))
    rot = np.zeros((2 * Nc * d, 2 * Nc * d))
    for k in range(Nc):
        s_ = k - Nc if k >= Nc // 2 else k
        a1 = 0.0 if s_ == 0 else np.clip(1 / s_, -1, 1)
        R = np.array([[np.sqrt(1 - a1 ** 2), -a1], [a1, np.sqrt(1 - a1 ** 2)]])
        for i in range(d):
            sl = slice(2 * (k * d + i), 2 * (k * d + i) + 2)
            rot[sl, sl] = R
    psi0 = np.kron(np.kron(np.eye(Nc)[0], b / np.linalg.norm(b)), [1, 0]).astype(complex)
    after_qpe = np.kron(qpe, np.eye(2)) @ psi0
    rho = np.zeros((d, d), dtype=complex); x_t = np.linalg.solve(A, b); x_t = x_t / np.linalg.norm(x_t)
    for mask in range(Nc):
        w = np.prod([p if (mask >> i) & 1 else 1 - p for i in range(n_c)])
        zdiag = np.array([(-1) ** bin(k & mask).count("1") for k in range(Nc)])
        Zop = np.kron(np.diag(zdiag), np.eye(2 * d))
        out = np.kron(qpe.conj().T, np.eye(2)) @ (rot @ (Zop @ after_qpe))
        br = out.reshape(Nc, d, 2)[:, :, 1]               # all clock values, ancilla = 1 (clock not post-selected)
        for kc in range(Nc):
            rho += w * np.outer(br[kc], br[kc].conj())
    P = np.real(np.trace(rho))
    return P, float(np.real(x_t.conj() @ rho @ x_t) / P)


pz = np.linspace(0, 0.2, 9)
res = [hhl_with_clock_z(A_DEF, B_DEF, 4, np.pi / 8, p) for p in pz]
P_flip = np.linspace(0, 0.2, 9)
# ancilla read-out flip: accepted = (1-q) * good + q * bad branch
good = p_succ
fig, ax = plt.subplots(figsize=(4.6, 2.5))
ax.plot(pz, [r[1] for r in res], "o-", color=BLUE, label="fidelity, $Z$ errors on clock qubits")
ax.plot(pz, [r[0] for r in res], "s--", color=AQUA, label="ancilla-1 probability, same model")
ax.plot(P_flip, [(1 - q) * good / ((1 - q) * good + q * (1 - good)) for q in P_flip], "^-", color=ORANGE,
        label="good fraction of accepted shots,\nancilla read-out flip $q$")
ax.set_xlabel("error probability per qubit"); ax.set_ylim(0, 1.05)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
ax.set_title("Simple error models on the default $4\\times4$ run (NumPy)", loc="left", fontsize=9)
save(fig, "noise")
NUMBERS.update(noiseFidTen=f"{res[4][1]:.3f}")

# ------------------------------------------------------------------------------------- numbers for the slides
with open(os.path.join(OUT, "hhl_numbers.tex"), "w") as fh:
    fh.write("% GENERATED by make_figures.py (CUDA-Q qpp-cpu + NumPy)\n")
    for k, v in NUMBERS.items():
        fh.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
print(NUMBERS)
