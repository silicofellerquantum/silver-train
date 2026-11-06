"""Generate all figures for the Grover lecture slides (vector PDFs in ./figures, numbers in ./figures/*.tex).

Quantum results come from NVIDIA CUDA-Q (qpp-cpu and density-matrix-cpu simulators), using the kernels of Grover.ipynb,
and are checked against the closed formula. Several-solution and quantum-counting figures are theory (NumPy), because
the notebook handles exactly one marked item. figures/notebook_run.png is copied from the stored output of Grover.ipynb.
Run:  python make_figures.py   (needs cudaq, numpy, matplotlib; run it on Linux, for example in WSL)
"""
import base64
import json
import math
import os
import platform
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cudaq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
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
cudaq.set_random_seed(7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def theta_of(N, M=1):
    return math.asin(math.sqrt(M / N))


def P_k(N, k, M=1):
    return math.sin((2 * k + 1) * theta_of(N, M)) ** 2


def k_opt(N, M=1):
    return max(0, round(math.pi / (4 * theta_of(N, M)) - 0.5))


# ---------------------------------------------------------------- CUDA-Q kernels (same gates as Grover.ipynb)
@cudaq.kernel
def oracle(qubits: cudaq.qview, marked: list[int]):
    n = qubits.size()
    for i in range(n):
        if marked[i] == 0:
            x(qubits[i])
    z.ctrl(qubits[0:n - 1], qubits[n - 1])
    for i in range(n):
        if marked[i] == 0:
            x(qubits[i])


@cudaq.kernel
def diffusion(qubits: cudaq.qview):
    n = qubits.size()
    h(qubits)
    x(qubits)
    z.ctrl(qubits[0:n - 1], qubits[n - 1])
    x(qubits)
    h(qubits)


@cudaq.kernel
def grover_state(n: int, marked: list[int], iterations: int):
    qubits = cudaq.qvector(n)
    h(qubits)
    for _ in range(iterations):
        oracle(qubits, marked)
        diffusion(qubits)


@cudaq.kernel
def grover_then_oracle(n: int, marked: list[int], iterations: int):
    qubits = cudaq.qvector(n)
    h(qubits)
    for _ in range(iterations):
        oracle(qubits, marked)
        diffusion(qubits)
    oracle(qubits, marked)


@cudaq.kernel
def grover_search(n: int, marked: list[int], iterations: int):
    qubits = cudaq.qvector(n)
    h(qubits)
    for _ in range(iterations):
        oracle(qubits, marked)
        diffusion(qubits)
    mz(qubits)


def amplitudes(kernel, n, marked, k):
    """All amplitudes, index = bit string with qubit 0 first. The overall sign of -D is removed for even display."""
    st = cudaq.get_state(kernel, n, marked, k)
    return np.array([st.amplitude([int(c) for c in format(i, f"0{n}b")]) for i in range(2 ** n)])


def fix_sign(amps, reference_index):
    """The circuit implements -D; multiply by the global phase that makes the marked amplitude positive."""
    a = amps[reference_index]
    return amps * (np.conj(a) / abs(a)) if abs(a) > 1e-12 else amps


# ---------------------------------------------------------------- 1. classical vs quantum query cost
fig, ax = plt.subplots(figsize=(3.3, 2.5))
ns = np.arange(2, 41)
ax.semilogy(ns, 2.0 ** ns / 2, color=ORANGE, label=r"classical average: $N/2$")
ax.semilogy(ns, np.pi / 4 * 2.0 ** (ns / 2), color=BLUE, label=r"Grover: $\frac{\pi}{4}\sqrt{N}$")
ax.set_xlabel("qubits n (search space N = 2$^n$)")
ax.set_ylabel("queries (log scale)")
ax.legend(frameon=False)
save(fig, "search_cost")

# ---------------------------------------------------------------- 2. measurement shots of one qubit


@cudaq.kernel
def qubit_07(angle: float):
    q = cudaq.qubit()
    ry(angle, q)
    mz(q)


angle = 2 * math.asin(math.sqrt(0.3))
fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.8), sharey=True)
for ax, shots in zip(axes, (10, 100, 1000)):
    c = cudaq.sample(qubit_07, angle, shots_count=shots)
    ax.bar([0, 1], [c.count("0") / shots, c.count("1") / shots], color=[BLUE, ORANGE], width=0.6)
    ax.axhline(0.7, color=GREY, ls=":", lw=1)
    ax.axhline(0.3, color=GREY, ls=":", lw=1)
    ax.set_xticks([0, 1])
    ax.set_title(f"{shots} shots")
axes[0].set_ylabel("measured fraction")
save(fig, "measurement_shots")

# ---------------------------------------------------------------- 3. sin^2 curve with the angle axis
fig, ax = plt.subplots(figsize=(4.4, 1.9))
al = np.linspace(0, 2 * np.pi, 400)
ax.plot(al, np.sin(al) ** 2, color=BLUE)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlabel(r"angle $\alpha$")
ax.set_ylabel(r"$\sin^2\alpha$")
save(fig, "sin2_curve")

# ---------------------------------------------------------------- 4. two reflections make a rotation
fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.0))
a1, a2, phi = math.radians(10), math.radians(35), math.radians(62)
states = [phi, 2 * a1 - phi, 2 * a2 - (2 * a1 - phi)]
titles = ["start", "reflect in line 1", "then reflect in line 2"]
for i, ax in enumerate(axes):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#c3c2b7", lw=0.8))
    for a, col in ((a1, AQUA), (a2, ORANGE)):
        ax.plot([-1.15 * math.cos(a), 1.15 * math.cos(a)], [-1.15 * math.sin(a), 1.15 * math.sin(a)], color=col, lw=1, ls="--")
    if i:
        ax.annotate("", xy=(math.cos(states[i - 1]), math.sin(states[i - 1])), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#c3c2b7", lw=1.2))
    ax.annotate("", xy=(math.cos(states[i]), math.sin(states[i])), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.8))
    ax.set_title(titles[i])
fig.text(0.5, -0.02, r"lines at $10^\circ$ (green) and $35^\circ$ (orange); net effect: rotation by $2\times25^\circ = 50^\circ$", ha="center", fontsize=8, color=INK)
save(fig, "two_reflections")

# ---------------------------------------------------------------- 5. inversion about the mean, plain numbers
fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.7), sharey=True)
v = np.array([1, 1, 1, -1, 1, 1, 1, 1], dtype=float)
vm = v.mean()
new = 2 * vm - v
for ax, data, title in ((axes[0], v, "before"), (axes[1], new, "after: each entry $\\to 2\\bar v - v_j$")):
    cols = [ORANGE if j == 3 else BLUE for j in range(8)]
    ax.bar(range(8), data, color=cols, width=0.7)
    ax.axhline(vm, color=AQUA, lw=1.2, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title(title)
    ax.set_xticks(range(8))
    ax.set_ylim(-1.3, 2.8)
axes[0].text(7.6, 2.2, f"dashed: mean = {vm:g}", color=AQUA, ha="right", fontsize=8)
save(fig, "mean_inversion")

# ---------------------------------------------------------------- 6. N = 8 amplitudes step by step (CUDA-Q), table
marked3 = [1, 0, 1]
idx3 = 0b101
a_start = amplitudes(grover_state, 3, marked3, 0)
a_oracle = amplitudes(grover_then_oracle, 3, marked3, 0)
a_one = fix_sign(amplitudes(grover_state, 3, marked3, 1), idx3)
a_oracle2 = fix_sign(amplitudes(grover_then_oracle, 3, marked3, 1), 0)  # unmarked amplitudes kept positive
a_two = fix_sign(amplitudes(grover_state, 3, marked3, 2), idx3)
labels3 = [format(i, "03b") for i in range(8)]
fig, axes = plt.subplots(1, 3, figsize=(5.8, 1.85), sharey=True)
for ax, amps, title in ((axes[0], a_start, r"start: $|s\rangle$"), (axes[1], a_oracle, r"after the oracle $U_\omega$"),
                        (axes[2], a_one, r"after the diffusion $D$")):
    amps = amps.real
    ax.bar(range(8), amps, color=[ORANGE if j == idx3 else BLUE for j in range(8)], width=0.72)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(8), labels3, rotation=90, fontsize=7)
    ax.set_title(title)
axes[1].axhline(a_oracle.real.mean(), color=AQUA, ls="--", lw=1.2)
axes[1].text(7.5, a_oracle.real.mean() + 0.05, "mean", color=AQUA, ha="right", fontsize=8)
axes[0].set_ylabel("amplitude")
save(fig, "amp_steps_n3")

with open(os.path.join(OUT, "n3_table.tex"), "w") as fh:
    fh.write("\\newcommand{\\nthreerows}{%\n")
    rows = [("start $\\ket{s}$", a_start), ("after $U_\\omega$", a_oracle), ("after $D$ ($k = 1$)", a_one),
            ("after $U_\\omega$ again", a_oracle2), ("after $D$ ($k = 2$)", a_two)]
    for name, amps in rows:
        amps = amps.real
        fh.write(f"  {name} & ${amps[0]:+.4f}$ & ${amps[idx3]:+.4f}$ & ${amps[idx3] ** 2:.4f}$\\\\\n")
    fh.write("}\n")
assert abs(a_one[idx3].real ** 2 - 25 / 32) < 1e-9 and abs(a_two[idx3].real ** 2 - 121 / 128) < 1e-9

# ---------------------------------------------------------------- 7. N = 8 amplitudes for k = 0 .. 3 (over-rotation)
fig, axes = plt.subplots(1, 4, figsize=(5.8, 1.6), sharey=True)
for k, ax in zip(range(4), axes):
    amps = fix_sign(amplitudes(grover_state, 3, marked3, k), idx3).real
    ax.bar(range(8), amps, color=[ORANGE if j == idx3 else BLUE for j in range(8)], width=0.72)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([])
    ax.set_title(f"k = {k}\nP = {amps[idx3] ** 2:.3f}", fontsize=9)
axes[0].set_ylabel("amplitude")
save(fig, "amp_k_n3")

# ---------------------------------------------------------------- 8. geometry
th = math.radians(17)


def plane(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.5, 1.2)
    ax.annotate("", xy=(1.25, 0), xytext=(-0.1, 0), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.annotate("", xy=(0, 1.15), xytext=(0, -0.4), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9))
    ax.text(1.27, -0.09, r"$|r\rangle$", fontsize=9, color=INK)
    ax.text(0.03, 1.1, r"$|m\rangle$", fontsize=9, color=INK)


def arrow(ax, ang, col, label, dx=0.03, dy=0.03, lw=1.9):
    ax.annotate("", xy=(math.cos(ang), math.sin(ang)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=lw))
    ax.text(math.cos(ang) + dx, math.sin(ang) + dy, label, color=col, fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.1))
plane(axes[0])
arrow(axes[0], th, BLUE, r"$|s\rangle$")
axes[0].set_title(r"start: angle $\theta$")
plane(axes[1])
arrow(axes[1], th, "#c3c2b7", r"$|s\rangle$")
arrow(axes[1], -th, ORANGE, r"$U_\omega|s\rangle$", dy=-0.1)
axes[1].set_title(r"oracle: reflect in $|r\rangle$")
plane(axes[2])
axes[2].plot([0, 1.2 * math.cos(th)], [0, 1.2 * math.sin(th)], color=BLUE, ls="--", lw=1)
arrow(axes[2], -th, "#c3c2b7", "", lw=1.2)
arrow(axes[2], 3 * th, AQUA, r"$G|s\rangle$")
axes[2].set_title(r"diffusion: reflect in $|s\rangle$")
save(fig, "geometry_steps")

fig, ax = plt.subplots(figsize=(3.0, 2.7))
plane(ax)
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(-0.15, 1.35)
t16 = theta_of(16)
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#e1e0d9", lw=0.8))
for k, col in zip(range(4), ("#b9d3f2", "#7fb0ea", BLUE, ORANGE)):
    a = (2 * k + 1) * t16
    arrow(ax, a, col, f"k={k}", dx=0.02, dy=0.02)
save(fig, "geometry_k_n4")

# ---------------------------------------------------------------- 9. P(k) for n = 6: CUDA-Q amplitudes on the theory curve
n6 = 6
marked6 = [1, 0, 1, 1, 0, 0]
ks = np.arange(0, 31)
sim = [abs(cudaq.get_state(grover_state, n6, marked6, int(k)).amplitude(marked6)) ** 2 for k in ks]
theory = np.array([P_k(2 ** n6, k) for k in ks])
dev6 = float(np.max(np.abs(np.array(sim) - theory)))
fig, ax = plt.subplots(figsize=(4.9, 2.1))
kf = np.linspace(0, 30, 600)
ax.plot(kf, np.sin((2 * kf + 1) * theta_of(64)) ** 2, color=GREY, lw=1.2, label=r"theory $\sin^2((2k+1)\theta)$")
ax.plot(ks, sim, "o", color=BLUE, ms=3.6, label="CUDA-Q state vector")
ax.axvline(k_opt(64), color=ORANGE, lw=1, ls="--")
ax.text(k_opt(64) + 0.3, 0.03, r"$k_{\rm opt} = 6$", color=ORANGE, fontsize=8)
ax.set_xlabel("iterations k")
ax.set_ylabel("P(marked)")
ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.2))
save(fig, "p_vs_k_n6")

# ---------------------------------------------------------------- 10. special ratios M/N (theory)
fig, ax = plt.subplots(figsize=(4.9, 2.1))
for ratio, col, lab in ((1 / 8, BLUE, "M/N = 1/8"), (1 / 4, AQUA, "M/N = 1/4"), (1 / 2, GREY, "M/N = 1/2"), (3 / 4, ORANGE, "M/N = 3/4")):
    t = math.asin(math.sqrt(ratio))
    kk = np.arange(0, 7)
    ax.plot(kk, np.sin((2 * kk + 1) * t) ** 2, "-o", color=col, ms=3.5, lw=1.3, label=lab)
ax.set_xlabel("iterations k")
ax.set_ylabel("P(marked set)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2))
save(fig, "ratios")

# ---------------------------------------------------------------- 11. shots: n = 4 (CUDA-Q)
marked4 = [1, 0, 1, 1]
fig, axes = plt.subplots(1, 3, figsize=(5.8, 1.7), sharey=True)
shot_rows = []
for k, ax in zip((0, 1, 3), axes):
    c = cudaq.sample(grover_search, 4, marked4, k, shots_count=2000)
    freq = [c.count(format(i, "04b")) / 2000 for i in range(16)]
    ax.bar(range(16), freq, color=[ORANGE if j == 0b1011 else BLUE for j in range(16)], width=0.75)
    ax.plot([0b1011], [P_k(16, k)], "D", color=INK, ms=3)
    ax.set_title(f"k = {k},  P = {P_k(16, k):.3f}", fontsize=9)
    ax.set_xlabel("outcome x")
    shot_rows.append((k, c.count("1011")))
axes[0].set_ylabel("measured fraction")
save(fig, "shots_n4")

# ---------------------------------------------------------------- 12. iterations against n
fig, ax = plt.subplots(figsize=(3.0, 2.3))
nn = np.arange(2, 31)
ax.semilogy(nn, [k_opt(2 ** int(n)) for n in nn], color=BLUE, marker="o", ms=2.5, lw=1)
ax.set_xlabel("qubits n")
ax.set_ylabel(r"$k_{\rm opt}$ (log scale)")
save(fig, "kopt_scaling")

# ---------------------------------------------------------------- 13. BBHT average success (theory)
fig, ax = plt.subplots(figsize=(3.2, 2.3))
th20 = theta_of(2 ** 10)
ms = np.arange(1, 121)
avg = 0.5 - np.sin(4 * ms * th20) / (4 * ms * np.sin(2 * th20))
ax.plot(ms, avg, color=BLUE)
ax.axhline(0.25, color=ORANGE, ls="--", lw=1)
ax.axvline(1 / math.sin(2 * th20), color=GREY, ls=":", lw=1)
ax.text(1 / math.sin(2 * th20) + 2, 0.03, r"$m = 1/\sin 2\theta \approx 16$", fontsize=8, color=INK)
ax.set_xlabel("range m of the random k")
ax.set_ylabel("average success")
save(fig, "bbht_average")

# ---------------------------------------------------------------- 14. quantum counting distributions (NumPy, explicit G)
N64 = 64
rng = np.random.default_rng(1)
mk = rng.choice(N64, 16, replace=False)
Om = np.eye(N64, dtype=complex)
Om[mk, mk] = -1
s64 = np.ones(N64, dtype=complex) / math.sqrt(N64)
G64 = (2 * np.outer(s64, s64.conj()) - np.eye(N64)) @ Om


def counting(t):
    Mt = 2 ** t
    reg = np.zeros((Mt, N64), dtype=complex)
    vec = s64.copy()
    for x_ in range(Mt):
        reg[x_] = vec / math.sqrt(Mt)
        vec = G64 @ vec
    invq = np.exp(-2j * np.pi * np.outer(np.arange(Mt), np.arange(Mt)) / Mt) / math.sqrt(Mt)
    return np.sum(np.abs(invq @ reg) ** 2, axis=1)


fig, axes = plt.subplots(1, 2, figsize=(5.4, 1.7))
for ax, t in zip(axes, (3, 6)):
    p = counting(t)
    ax.bar(range(2 ** t), p, color=BLUE, width=0.8)
    for loc in (2 ** t / 6, 5 * 2 ** t / 6):
        ax.axvline(loc, color=ORANGE, ls="--", lw=0.9)
    ax.set_title(f"t = {t} counting qubits")
    ax.set_xlabel("outcome y")
axes[0].set_ylabel("probability")
save(fig, "counting")

# ---------------------------------------------------------------- 15. noise on the density-matrix simulator (CUDA-Q)
cudaq.set_target("density-matrix-cpu")
cudaq.set_random_seed(11)
fig, ax = plt.subplots(figsize=(4.9, 2.1))
kn = np.arange(0, 9)
kf = np.linspace(0, 8, 300)
ax.plot(kf, np.sin((2 * kf + 1) * theta_of(16)) ** 2, color=GREY, lw=1.1, label="noiseless theory")
noise_rows = {}
for p_err, col in ((0.002, BLUE), (0.01, AQUA), (0.03, ORANGE)):
    nm = cudaq.NoiseModel()
    nm.add_all_qubit_channel("h", cudaq.DepolarizationChannel(p_err))
    nm.add_all_qubit_channel("x", cudaq.DepolarizationChannel(p_err))
    vals = []
    for k in kn:
        c = cudaq.sample(grover_search, 4, marked4, int(k), shots_count=4000, noise_model=nm)
        vals.append(c.count("1011") / 4000)
    noise_rows[p_err] = vals
    ax.plot(kn, vals, "-o", color=col, ms=3.3, lw=1.1, label=f"p = {p_err}")
ax.axhline(1 / 16, color=INK, lw=0.7, ls=":")
ax.text(8.05, 1 / 16 + 0.03, "1/16", fontsize=7, color=INK, ha="right")
ax.set_xlabel("iterations k")
ax.set_ylabel("P(marked)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
save(fig, "noise_n4")
cudaq.set_target("qpp-cpu")
cudaq.set_random_seed(42)

# ---------------------------------------------------------------- 16. CPU benchmark and a fresh run of the notebook default
bits15 = "011001010011111"
times = {}
for n in range(6, 16):
    bits = (bits15 * 2)[:n]
    mk_ = [int(c) for c in bits]
    k = k_opt(2 ** n)
    t0 = time.perf_counter()
    cres = cudaq.sample(grover_search, n, mk_, k, shots_count=2000)
    times[n] = (k, time.perf_counter() - t0, cres.count(bits))
    print("benchmark", n, times[n])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
xs = sorted(times)
ax.semilogy(xs, [times[n][1] for n in xs], "o-", color=BLUE, ms=3.5, lw=1.2, label="measured")
ref = times[xs[-1]][1]
ax.semilogy(xs, [ref * 2 ** (1.5 * (n - xs[-1])) for n in xs], color=GREY, lw=1, ls="--", label=r"$\propto 2^{1.5n}$")
ax.set_xlabel("qubits n")
ax.set_ylabel("run time [s]")
ax.legend(frameon=False)
save(fig, "benchmark_cpu")

k15, t15, c15 = times[15]
assert k15 == 142 and bits15 == (bits15 * 2)[:15]
res = cudaq.estimate_resources(grover_search, 15, [int(c) for c in bits15], 142)
with open(os.path.join(OUT, "run_numbers.tex"), "w") as fh:
    fh.write("\\newcommand{\\devsix}{%.1e}\n" % dev6)
    fh.write("\\newcommand{\\marksfifteen}{%d}\n" % c15)
    fh.write("\\newcommand{\\timefifteen}{%.1f}\n" % t15)
    fh.write("\\newcommand{\\hcount}{%d}\n\\newcommand{\\xcount}{%d}\n" % (res.count("h"), res.count("x")))
    fh.write("\\newcommand{\\benchrows}{%\n")
    for n in xs:
        k, t_, c_ = times[n]
        fh.write(f"  {n} & {k} & {t_:.2f} & {c_}\\\\\n")
    fh.write("}\n")
    fh.write("\\newcommand{\\shotsrows}{%\n")
    for k, c_ in shot_rows:
        fh.write(f"  {k} & {P_k(16, k):.4f} & {c_} & {c_ / 2000:.4f}\\\\\n")
    fh.write("}\n")
    fh.write("\\newcommand{\\noisefour}{%.3f}\n\\newcommand{\\noiseone}{%.3f}\n\\newcommand{\\noisethree}{%.3f}\n" % (
        noise_rows[0.002][3], noise_rows[0.01][3], noise_rows[0.03][3]))
    fh.write("\\newcommand{\\hostinfo}{%s, %d logical CPUs, CUDA-Q %s}\n" % (
        platform.system(), os.cpu_count(), cudaq.__version__.split("Version")[-1].split()[0]))

# ---------------------------------------------------------------- 17. the stored figure of the notebook's recorded run
nb = json.load(open(os.path.join(HERE, "..", "..", "Grover.ipynb"), encoding="utf8"))
stored = [o["data"]["image/png"] for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])
          if "data" in o and "image/png" in o["data"]]
with open(os.path.join(OUT, "notebook_run.png"), "wb") as fh:
    fh.write(base64.b64decode(stored[-1]))
print("done; n=15 marked counts", c15, "time", round(t15, 1), "deviation n=6", dev6)
