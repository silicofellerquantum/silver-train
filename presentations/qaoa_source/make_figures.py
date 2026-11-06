"""Generate all figures and numbers for the QAOA lecture slides (vector PDFs + qaoa_numbers.tex in ./figures).

Quantum results are computed with NVIDIA CUDA-Q (qpp-cpu simulator) using exactly the kernel and conventions of
QAOA.ipynb, and cross-checked against an independent NumPy state-vector implementation.
Run:  python make_figures.py      (needs cudaq, numpy, scipy, matplotlib; about 3-5 minutes on a laptop CPU)
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
from scipy.optimize import minimize
import cudaq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, GREY, INK, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#52514e", "#d9d8d2"
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
NUMBERS = {}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- the notebook's instance and kernels
N = 15
graph_rng = np.random.default_rng(SEED)                      # identical to QAOA.ipynb, GRAPH = "example"
EDGES = [(i, j, 1.0) for i in range(N) for j in range(i + 1, N) if graph_rng.random() < 0.3]
W = sum(e[2] for e in EDGES)


@cudaq.kernel
def qaoa_state(n: int, edge_i: list[int], edge_j: list[int], weights: list[float], layers: int, params: list[float]):
    qubits = cudaq.qvector(n)
    h(qubits)
    for layer in range(layers):
        gamma = params[layer]
        beta = params[layers + layer]
        for e in range(len(edge_i)):
            x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
            rz(gamma * weights[e], qubits[edge_j[e]])
            x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
        for q in range(n):
            rx(2.0 * beta, qubits[q])


@cudaq.kernel
def qaoa_sample(n: int, edge_i: list[int], edge_j: list[int], weights: list[float], layers: int, params: list[float]):
    qubits = cudaq.qvector(n)
    h(qubits)
    for layer in range(layers):
        gamma = params[layer]
        beta = params[layers + layer]
        for e in range(len(edge_i)):
            x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
            rz(gamma * weights[e], qubits[edge_j[e]])
            x.ctrl(qubits[edge_i[e]], qubits[edge_j[e]])
        for q in range(n):
            rx(2.0 * beta, qubits[q])
    mz(qubits)


def args(n, edges, params):
    return (n, [e[0] for e in edges], [e[1] for e in edges], [float(e[2]) for e in edges], len(params) // 2,
            [float(v) for v in params])


def cost_hamiltonian(edges):
    hc = None
    for i, j, w in edges:
        t = 0.5 * w * cudaq.spin.z(i) * cudaq.spin.z(j)
        hc = t if hc is None else hc + t
    return hc


def cudaq_cut(n, edges, params):
    wsum = sum(e[2] for e in edges)
    return wsum / 2 - cudaq.observe(qaoa_state, cost_hamiltonian(edges), *args(n, edges, params)).expectation()


def cudaq_state(n, edges, params):
    st = np.array(cudaq.get_state(qaoa_state, *args(n, edges, params)))
    return st.reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)      # node 0 = most significant bit


# ---------------------------------------------------------------- NumPy reference (independent implementation)
def spins(n):
    idx = np.arange(2 ** n)
    return 1 - 2 * ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1)


def cut_values(n, edges):
    s = spins(n)
    return sum(w * (1 - s[:, i] * s[:, j]) / 2 for i, j, w in edges) + np.zeros(2 ** n)


def np_state(n, edges, params):
    p = len(params) // 2
    s = spins(n)
    hc = sum(w / 2 * s[:, i] * s[:, j] for i, j, w in edges)
    psi = np.full(2 ** n, 2 ** (-n / 2), complex)
    for l in range(p):
        psi = np.exp(-1j * params[l] * hc) * psi
        c, sn = np.cos(params[p + l]), -1j * np.sin(params[p + l])
        psi = psi.reshape([2] * n)
        for q in range(n):
            a, b = np.take(psi, 0, axis=q), np.take(psi, 1, axis=q)
            psi = np.stack([c * a + sn * b, sn * a + c * b], axis=q)
        psi = psi.reshape(-1)
    return psi


def np_cut(n, edges, params, cv=None):
    cv = cut_values(n, edges) if cv is None else cv
    return float(np.abs(np_state(n, edges, params)) ** 2 @ cv)


def ring(n):
    return [(i, (i + 1) % n, 1.0) for i in range(n)]


CV = cut_values(N, EDGES)
CMAX = CV.max()
OPT = [format(int(z), f"0{N}b") for z in np.flatnonzero(np.isclose(CV, CMAX))]
NUMBERS.update(nedges=len(EDGES), cmax=f"{CMAX:g}", nopt=len(OPT), randvar=f"{CV.var():g}", randsd=f"{CV.std():.2f}")

# ---------------------------------------------------------------- 1. reproduce the notebook's optimisation with CUDA-Q
RAMP = [0.2, 0.6, 0.6, 0.2]
history = []
H_C = cost_hamiltonian(EDGES)


def objective(params):
    e = cudaq.observe(qaoa_state, H_C, *args(N, EDGES, params)).expectation()
    history.append((W / 2 - e, np.array(params, float)))
    return e


t0 = time.perf_counter()
opt = minimize(objective, np.array(RAMP), method="COBYLA", options={"maxiter": 80, "rhobeg": 0.2})
opt_time = time.perf_counter() - t0
cuts = [h[0] for h in history]
best = int(np.argmax(cuts))
BEST = history[best][1]
counts = cudaq.sample(qaoa_sample, *args(N, EDGES, BEST), shots_count=4000)
samples = sorted(((b, c, CV[int(b, 2)]) for b, c in counts.items()), key=lambda t: (-t[1], -t[2]))
opt_hits = sum(c for b, c, v in samples if np.isclose(v, CMAX))
fid = abs(np.vdot(cudaq_state(N, EDGES, BEST), np_state(N, EDGES, BEST))) ** 2
NUMBERS.update(start=f"{cuts[0]:.2f}", startnp=f"{np_cut(N, EDGES, RAMP, CV):.2f}", bestcut=f"{cuts[best]:.4f}",
               ratio=f"{cuts[best] / CMAX:.4f}", nevals=len(cuts), popt=f"{opt_hits / 4000:.4f}", opthits=opt_hits,
               eta=f"{(cuts[best] - W / 2) / (CMAX - W / 2):.3f}", fid=f"{fid:.12f}", opttime=f"{opt_time:.0f}",
               gammaa=f"{BEST[0]:.3f}", gammab=f"{BEST[1]:.3f}", betaa=f"{BEST[2]:.3f}", betab=f"{BEST[3]:.3f}",
               optmsg=re.sub(r"[^A-Za-z ]", "", str(opt.message)).strip(),
               shotmean=f"{np.repeat([v for *_, v in samples], [c for _, c, _ in samples]).mean():.3f}")
print("optimisation:", NUMBERS)

# ---------------------------------------------------------------- figure: the instance and the optimal split
pos = np.c_[np.cos(2 * np.pi * np.arange(N) / N + np.pi / 2), np.sin(2 * np.pi * np.arange(N) / N + np.pi / 2)]


def draw(ax, edges, posn, split=None, title=""):
    for i, j, w in edges:
        cut = split is not None and split[i] != split[j]
        ax.plot(*posn[[i, j]].T, color=INK if (cut or split is None) else LIGHT, lw=1.6 if cut else 1.0, zorder=1)
    cols = [BLUE if split is None or split[k] == "0" else ORANGE for k in range(len(posn))]
    ax.scatter(posn[:, 0], posn[:, 1], s=230, c=cols, edgecolors="white", linewidths=1.5, zorder=2)
    for k in range(len(posn)):
        ax.text(*posn[k], str(k), ha="center", va="center", fontsize=7, color="white", zorder=3)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, loc="left")


fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.5))
draw(ax[0], EDGES, pos, title=f"15 nodes, {len(EDGES)} edges (SEED = 42)")
draw(ax[1], EDGES, pos, OPT[2], title=f"optimal split {OPT[2]}: cut {CMAX:g}")
save(fig, "graph15")

# small example graph (house) with its optimal cut
hpos = np.array([[0, 0], [1, 0], [1, -1], [0, -1], [0.5, 0.8]])
house = [(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (3, 0, 1.), (0, 4, 1.), (1, 4, 1.)]
fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.5))
draw(ax[0], house, hpos, "00110", "split 00110: cut 2")
draw(ax[1], house, hpos, "01011", "split 01011: cut 5 (optimal)")
save(fig, "house")

# ---------------------------------------------------------------- figure: cut distribution, random vs QAOA
probs = np.abs(cudaq_state(N, EDGES, BEST)) ** 2
vals = np.arange(int(CV.min()), int(CMAX) + 1)
p_rand = np.array([np.mean(CV == v) for v in vals])
p_qaoa = np.array([probs[CV == v].sum() for v in vals])
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.bar(vals - 0.2, p_rand, 0.4, color=GREY, label="random split (uniform superposition)")
ax.bar(vals + 0.2, p_qaoa, 0.4, color=BLUE, label=f"QAOA $p=2$, best angles")
ax.axvline(W / 2, color=GREY, lw=1, ls="--"); ax.axvline(cuts[best], color=BLUE, lw=1, ls="--")
ax.text(W / 2, ax.get_ylim()[1] * 0.92, r" $W/2=15$", color=INK, fontsize=8)
ax.text(cuts[best], ax.get_ylim()[1] * 0.92, rf" $\langle C\rangle={cuts[best]:.2f}$", color=BLUE, fontsize=8)
ax.set_xlabel("cut value $C(z)$"); ax.set_ylabel("probability"); ax.legend(loc="upper left")
save(fig, "cut_distribution")
NUMBERS.update(prandopt=f"{p_rand[-1]:.2e}".replace("e-0", r"\times10^{-") + "}", pqaoaopt=f"{p_qaoa[-1]:.4f}")

# ---------------------------------------------------------------- figure: convergence
fig, ax = plt.subplots(figsize=(6.2, 2.5))
ax.plot(np.arange(1, len(cuts) + 1), cuts, color=BLUE, marker="o", ms=2.5, lw=1.3)
ax.axhline(CMAX, color=INK, lw=1); ax.text(len(cuts), CMAX, f"max cut {CMAX:g} ", ha="right", va="bottom", fontsize=8)
ax.axhline(W / 2, color=GREY, lw=1, ls="--"); ax.text(len(cuts), W / 2, "random split 15 ", ha="right", va="bottom", fontsize=8)
ax.annotate(f"start (ramp): {cuts[0]:.2f}", (1, cuts[0]), (8, cuts[0] + 1.5), fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE))
ax.set_xlabel("optimizer evaluation (COBYLA)"); ax.set_ylabel(r"expected cut $\langle C\rangle$")
save(fig, "convergence")

# ---------------------------------------------------------------- figure: top bitstrings
top = samples[:12]
fig, ax = plt.subplots(figsize=(6.4, 2.5))
ax.bar(range(len(top)), [c / 4000 for _, c, _ in top], color=[BLUE if np.isclose(v, CMAX) else GREY for *_, v in top], width=0.55)
ax.set_xticks(range(len(top)), [f"{b}\nC={v:g}" for b, _, v in top], rotation=90, fontsize=6.5, family="monospace")
ax.set_ylabel("probability"); ax.grid(axis="x", visible=False)
ax.set_title("12 most frequent of 4000 shots (blue: optimal cut 25)", loc="left")
save(fig, "top_bitstrings")
NUMBERS["toprows"] = "".join(rf"\texttt{{{b}}} & {c} & {v:g}\\" for b, c, v in samples[:6])

# ---------------------------------------------------------------- figure: single-edge landscape p = 1
g = np.linspace(-np.pi, np.pi, 241); b = np.linspace(-np.pi / 2, np.pi / 2, 241)
G, B = np.meshgrid(g, b)
F = 0.5 - 0.5 * np.sin(4 * B) * np.sin(G)
check_pts = [(0.7, 0.3), (np.pi / 2, -np.pi / 8), (-1.2, 1.1)]
assert all(abs(cudaq_cut(2, [(0, 1, 1.0)], [gg, bb]) - (0.5 - 0.5 * np.sin(4 * bb) * np.sin(gg))) < 1e-9 for gg, bb in check_pts)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
im = ax.contourf(G, B, F, levels=21, cmap="RdBu_r")
fig.colorbar(im, ax=ax, label=r"$F_1(\gamma,\beta)$")
ax.plot([np.pi / 2], [-np.pi / 8], marker="*", color="k", ms=10); ax.plot([np.pi / 2], [np.pi / 8], marker="x", color="k", ms=8)
ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\beta$"); ax.grid(False)
ax.set_title(r"one edge, $p=1$: $\frac{1}{2}-\frac{1}{2}\sin4\beta\,\sin\gamma$", loc="left", fontsize=9)
save(fig, "landscape_edge")

# ---------------------------------------------------------------- figure: p = 1 landscape of the 15-node graph (NumPy, checked by CUDA-Q)
gg = np.linspace(-np.pi / 2, np.pi / 2, 61); bb = np.linspace(-np.pi / 4, np.pi / 4, 61)
L = np.array([[np_cut(N, EDGES, [x, y], CV) for x in gg] for y in bb])
iy, ix = np.unravel_index(np.argmax(L), L.shape)
assert abs(cudaq_cut(N, EDGES, [gg[ix], bb[iy]]) - L[iy, ix]) < 1e-8
fig, ax = plt.subplots(figsize=(4.2, 3.0))
im = ax.contourf(gg, bb, L, levels=21, cmap="RdBu_r"); fig.colorbar(im, ax=ax, label=r"$\langle C\rangle$")
ax.plot(gg[ix], bb[iy], "k*", ms=10); ax.set_xlabel(r"$\gamma$"); ax.set_ylabel(r"$\beta$"); ax.grid(False)
ax.set_title("15-node graph, $p=1$", loc="left")
save(fig, "landscape_repo")
NUMBERS.update(ponebest=f"{L.max():.2f}", ponegamma=f"{gg[ix]:.2f}", ponebeta=f"{bb[iy]:.2f}")

# ---------------------------------------------------------------- figure: approximation ratio versus depth
def best_ratio(n, edges, p, cv, starts, seed=1, x0s=None):
    rng = np.random.default_rng(seed)
    best, bestx = -np.inf, None
    inits = list(x0s or []) + [rng.uniform(-np.pi / 2, np.pi / 2, 2 * p) for _ in range(starts)]
    for x0 in inits:
        r = minimize(lambda x: -np_cut(n, edges, x, cv), x0, method="BFGS")
        if -r.fun > best:
            best, bestx = -r.fun, r.x
    return best / cv.max(), bestx


ring_r = [best_ratio(8, ring(8), p, cut_values(8, ring(8)), 12)[0] for p in (1, 2, 3)]
repo_r, prev = [], None
for p in (1, 2, 3):
    x0s = None
    if prev is not None:                                     # INTERP-style warm start from depth p - 1
        gam, bet = prev[:p - 1], prev[p - 1:]
        ext = lambda v: [((i - 1) / (p - 1)) * (v[i - 2] if i > 1 else 0) + ((p - i) / (p - 1)) * (v[i - 1] if i <= p - 1 else 0)
                         for i in range(1, p + 1)]
        x0s = [np.r_[ext(gam), ext(bet)]]
    r, prev = best_ratio(N, EDGES, p, CV, 4, x0s=x0s)
    repo_r.append(r)
fig, ax = plt.subplots(figsize=(4.6, 2.6))
ps = [1, 2, 3]
ax.plot(ps, ring_r, "o-", color=AQUA, label="ring $C_8$ (optimal angles)")
ax.plot(ps, [(2 * p + 1) / (2 * p + 2) for p in ps], ":", color=GREY, label=r"$(2p+1)/(2p+2)$")
ax.plot(ps, repo_r, "s-", color=BLUE, label="15-node graph (best of local searches)")
ax.axhline(15 / 25, color=GREY, ls="--", lw=1); ax.text(3, 0.6, "random 0.6 ", ha="right", va="bottom", fontsize=8)
ax.set_xticks(ps); ax.set_xlabel("depth $p$"); ax.set_ylabel(r"$r = \langle C\rangle/C_{\max}$"); ax.set_ylim(0.55, 1.0)
ax.legend(loc="lower right", fontsize=7)
save(fig, "ratio_vs_depth")
NUMBERS.update(ringone=f"{ring_r[0]:.4f}", ringtwo=f"{ring_r[1]:.4f}", ringthree=f"{ring_r[2]:.4f}",
               repoone=f"{repo_r[0]:.3f}", repotwo=f"{repo_r[1]:.3f}", repothree=f"{repo_r[2]:.3f}")

# ---------------------------------------------------------------- figure: memory / gate scaling
n = np.arange(10, 46)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
ax[0].semilogy(n, 16 * 2.0 ** n, color=BLUE, label="state vector fp64")
ax[0].semilogy(n[n <= 25], 16 * 4.0 ** n[n <= 25], color=ORANGE, label="density matrix")
for y, lab in [(16e9, "16 GB laptop"), (80e9, "80 GB GPU")]:
    ax[0].axhline(y, color=GREY, lw=0.8, ls="--"); ax[0].text(10.3, y * 1.3, lab, fontsize=7)
ax[0].set_xlabel("qubits $n$"); ax[0].set_ylabel("bytes"); ax[0].legend(fontsize=7)
ax[1].plot(n, n + 2 * (3 * 0.3 * n * (n - 1) / 2 + n), color=BLUE, label="gates, $p=2$, $G(n,0.3)$")
ax[1].plot(n, 2 * 2 * 0.3 * n * (n - 1) / 2, color=ORANGE, label="CNOTs, $p=2$")
ax[1].set_xlabel("qubits $n$"); ax[1].set_ylabel("expected count"); ax[1].legend(fontsize=7)
save(fig, "scaling")

# ---------------------------------------------------------------- figure: global depolarising model
eps = np.linspace(0, 0.02, 101)
fig, ax = plt.subplots(figsize=(4.4, 2.4))
for p, col in [(1, AQUA), (2, BLUE), (3, ORANGE)]:
    ncx = 2 * p * len(EDGES)
    ideal = repo_r[p - 1] * CMAX                              # best noiseless value found at this depth
    Fd = (1 - eps) ** ncx
    ax.plot(eps * 100, Fd * ideal + (1 - Fd) * W / 2, color=col, label=f"$p={p}$ ({ncx} CNOTs)")
ax.axhline(W / 2, color=GREY, ls="--", lw=1)
ax.set_xlabel(r"CNOT error $\varepsilon$ (%)"); ax.set_ylabel(r"noisy $\langle C\rangle$ (model)"); ax.legend(fontsize=7)
save(fig, "noise_model")

# ---------------------------------------------------------------- numbers file
names = [re.sub("[^a-z]", "", k) for k in NUMBERS]
assert len(set(names)) == len(names), "macro name collision"
with open(os.path.join(OUT, "qaoa_numbers.tex"), "w") as fh:
    fh.write("% GENERATED by make_figures.py - do not edit\n")
    for name, v in zip(names, NUMBERS.values()):
        fh.write(rf"\newcommand{{\num{name}}}{{{v}}}" + "\n")
print("wrote qaoa_numbers.tex")
