"""Generate all figures for the QFT lecture slides (vector PDFs in ./figures).

Quantum results are computed with NVIDIA CUDA-Q (qpp-cpu simulator) and checked against NumPy.
Run:  python make_figures.py
"""
import os
import warnings

warnings.filterwarnings("ignore", message=".*cudaq-logical is in preview.*")
warnings.filterwarnings("ignore", message=".*algorithmic primitives will change.*")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
cudaq.set_random_seed(7)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def bit_reverse(vec, n):
    """CUDA-Q state index -> index with qubit 0 as most significant bit (and back)."""
    return np.asarray(vec).reshape([2] * n).transpose(list(range(n))[::-1]).reshape(-1)


# ---------------------------------------------------------------- CUDA-Q QFT kernels
@cudaq.kernel
def qft(qubits: cudaq.qview):
    n = qubits.size()
    for j in range(n):
        h(qubits[j])
        for k in range(j + 1, n):
            r1.ctrl(np.pi / 2 ** (k - j), qubits[k], qubits[j])
    for j in range(n // 2):
        swap(qubits[j], qubits[n - 1 - j])


@cudaq.kernel
def qft_of_amplitudes(initial: cudaq.State):
    q = cudaq.qvector(initial)
    qft(q)


@cudaq.kernel
def qft_of_amplitudes_measured(initial: cudaq.State):
    q = cudaq.qvector(initial)
    qft(q)
    mz(q)


def as_state(alpha, n):
    """Initial state for CUDA-Q (State.from_data also works above 10 qubits, unlike list arguments)."""
    return cudaq.State.from_data(np.ascontiguousarray(bit_reverse(alpha, n).astype(np.complex128)))


def run_qft(alpha, n):
    """Exact output amplitudes (index y, qubit 0 = MSB) from CUDA-Q."""
    state = np.array(cudaq.get_state(qft_of_amplitudes, as_state(alpha, n)))
    return bit_reverse(state, n)


def sample_qft(alpha, n, shots):
    counts = cudaq.sample(qft_of_amplitudes_measured, as_state(alpha, n), shots_count=shots)
    freq = np.zeros(2 ** n)
    for bits, c in counts.items():
        freq[int(bits, 2)] = c / shots
    return freq


# ---------------------------------------------------------------- 1. sound: time and frequency domain
t = np.linspace(0, 2, 2000)
signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 2.1), gridspec_kw={"width_ratios": [1.6, 1]})
a.plot(t, signal, color=BLUE)
a.set_xlabel("time t [s]"); a.set_ylabel("signal"); a.set_title("Time domain: what a microphone records", loc="left")
freqs = np.fft.rfftfreq(len(t), t[1] - t[0]); spec = np.abs(np.fft.rfft(signal)) / len(t) * 2
b.vlines(freqs[freqs <= 10], 0, spec[freqs <= 10], color=ORANGE, linewidth=2)
b.set_xlabel("frequency [Hz]"); b.set_ylabel("strength"); b.set_title("Frequency domain", loc="left")
b.set_xticks([0, 3, 7, 10])
save(fig, "sound")

# ---------------------------------------------------------------- 2. classical DFT of a periodic sequence
x_seq = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
X = np.fft.fft(x_seq)
fig, (a, b) = plt.subplots(1, 2, figsize=(5.6, 1.45))
a.bar(range(8), x_seq, width=0.5, color=BLUE); a.set_xticks(range(8)); a.set_title("input $x_n$ (period 4)", loc="left")
a.set_xlabel("n")
b.bar(range(8), np.abs(X), width=0.5, color=ORANGE); b.set_xticks(range(8)); b.set_title("$|X_k|$ after the DFT", loc="left")
b.set_xlabel("k")
save(fig, "dft_periodic")

# ---------------------------------------------------------------- 3. complex plane: Euler and multiplication
fig, ax = plt.subplots(figsize=(2.6, 2.6))
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(th), np.sin(th), color=GREY, linewidth=1)
phi = np.pi / 3
ax.annotate("", xy=(np.cos(phi), np.sin(phi)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
ax.plot([np.cos(phi), np.cos(phi)], [0, np.sin(phi)], color=ORANGE, linestyle=(0, (3, 2)), linewidth=1)
ax.text(np.cos(phi) + 0.05, np.sin(phi) + 0.05, r"$e^{i\varphi}$", color=BLUE, fontsize=11)
ax.text(np.cos(phi) / 2, -0.17, r"$\cos\varphi$", ha="center", fontsize=9)
ax.text(np.cos(phi) + 0.04, np.sin(phi) / 2, r"$\sin\varphi$", fontsize=9)
ax.text(0.2, 0.08, r"$\varphi$", fontsize=10)
ax.axhline(0, color="#c3c2b7", linewidth=0.8); ax.axvline(0, color="#c3c2b7", linewidth=0.8)
ax.set_xlim(-1.25, 1.35); ax.set_ylim(-1.25, 1.3); ax.set_aspect("equal"); ax.grid(False)
ax.set_xlabel("real part"); ax.set_ylabel("imaginary part")
save(fig, "euler")

# ---------------------------------------------------------------- 4. roots of unity
fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.0))
for ax, m in zip(axes, (4, 8)):
    ax.plot(np.cos(th), np.sin(th), color=GREY, linewidth=1)
    for k in range(m):
        ang = 2 * np.pi * k / m
        ax.plot(np.cos(ang), np.sin(ang), "o", color=BLUE, markersize=6)
        ax.text(1.3 * np.cos(ang), 1.3 * np.sin(ang), rf"$\omega^{k}$", ha="center", va="center", fontsize=9)
    ax.set_title(rf"$M={m}$", fontsize=10)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.axis("off")
save(fig, "roots_of_unity")

# ---------------------------------------------------------------- 5. cancellation: sum of roots of unity
fig, axes = plt.subplots(2, 1, figsize=(3.2, 3.1), gridspec_kw={"height_ratios": [1, 2.2]})
for ax, (k, title) in zip(axes, ((0, "k = 0: all arrows aligned, sum = 8"), (1, "k = 1: arrows go around, sum = 0"))):
    pos = np.array([0.0, 0.0])
    for j in range(8):
        step = np.array([np.cos(2 * np.pi * k * j / 8), np.sin(2 * np.pi * k * j / 8)])
        ax.annotate("", xy=pos + step, xytext=pos, arrowprops=dict(arrowstyle="-|>", color=BLUE if k == 0 else ORANGE, lw=1.6))
        pos = pos + step
    ax.plot(0, 0, "o", color=INK, markersize=4)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")
    if k == 0:
        ax.set_xlim(-0.3, 8.3); ax.set_ylim(-0.6, 0.6)
    else:
        ax.set_xlim(-1.4, 2.4); ax.set_ylim(-0.5, 2.9)
save(fig, "cancellation")

# ---------------------------------------------------------------- 6. measurement histogram of one qubit
@cudaq.kernel
def one_qubit(theta: float):
    q = cudaq.qubit()
    ry(theta, q)
    mz(q)


fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.8), sharey=True)
for ax, shots in zip(axes, (10, 100, 1000)):
    counts = cudaq.sample(one_qubit, 2 * np.arccos(np.sqrt(0.7)), shots_count=shots)
    vals = [counts.count("0") / shots, counts.count("1") / shots]
    ax.bar([0, 1], vals, width=0.5, color=[BLUE, ORANGE])
    ax.axhline(0.7, color=BLUE, linewidth=0.8, linestyle=(0, (3, 2))); ax.axhline(0.3, color=ORANGE, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xticks([0, 1], ["0", "1"]); ax.set_title(f"{shots} shots", fontsize=9); ax.set_ylim(0, 1)
axes[0].set_ylabel("measured fraction")
save(fig, "measurement_shots")

# ---------------------------------------------------------------- 7. phase clocks: QFT|1> (2 qubits) and QFT|5> (3 qubits), from CUDA-Q
def phase_clocks(amplitudes, n, title, name, color):
    m = 2 ** n
    fig, axes = plt.subplots(1, m, figsize=(min(5.8, 0.75 * m + 0.4), 1.25))
    for y, ax in enumerate(np.atleast_1d(axes)):
        a = amplitudes[y]
        ax.plot(np.cos(th), np.sin(th), color="#c3c2b7", linewidth=0.8)
        r = abs(a) / np.max(np.abs(amplitudes))
        ax.annotate("", xy=(r * np.cos(np.angle(a)), r * np.sin(np.angle(a))), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))
        ax.set_title(rf"$|{format(y, f'0{n}b')}\rangle$", fontsize=8)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(title, fontsize=9, y=1.12)
    save(fig, name)


e2 = np.zeros(4, complex); e2[1] = 1
phase_clocks(run_qft(e2, 2), 2, r"QFT$|01\rangle$: arrow turns by $90^\circ$ per step", "clocks_qft1_2q", BLUE)
e5 = np.zeros(8, complex); e5[5] = 1
amp5 = run_qft(e5, 3)
phase_clocks(amp5, 3, r"QFT$|101\rangle$ ($x=5$): arrow turns by $5\times 45^\circ$ per step", "clocks_qft5_3q", ORANGE)
e1 = np.zeros(8, complex); e1[1] = 1
phase_clocks(run_qft(e1, 3), 3, r"QFT$|001\rangle$ ($x=1$): arrow turns by $45^\circ$ per step", "clocks_qft1_3q", BLUE)
assert np.allclose(amp5, np.exp(2j * np.pi * 5 * np.arange(8) / 8) / np.sqrt(8)), "QFT|5> check failed"
with open(os.path.join(OUT, "qft5_table.tex"), "w") as fh:          # CUDA-Q amplitudes as a LaTeX macro
    fh.write("\\newcommand{\\qftfiverows}{%\n")
    for y in range(8):
        a = amp5[y]
        fh.write(f"$|{format(y, '03b')}\\rangle$ & {y} & ${round(a.real, 4) + 0.0:+.4f}$ & ${round(a.imag, 4) + 0.0:+.4f}$ & "
                 f"${np.degrees(np.angle(a)) % 360:.0f}^\\circ$ & {abs(a) ** 2:.4f} \\\\\n")
    fh.write("}\n")

# ---------------------------------------------------------------- 8. measuring QFT|5>: flat histogram
freq5 = sample_qft(e5, 3, 2000)
fig, ax = plt.subplots(figsize=(4.4, 1.8))
ax.bar(range(8), freq5, width=0.5, color=ORANGE, label="CUDA-Q, 2000 shots")
ax.axhline(1 / 8, color=INK, linewidth=1, label="exact 1/8")
ax.set_xticks(range(8), [format(y, "03b") for y in range(8)]); ax.set_ylim(0, 0.25)
ax.set_ylabel("probability"); ax.legend(loc="upper right", ncol=2)
save(fig, "measure_qft5")

# ---------------------------------------------------------------- 9. output-qubit phases on the Bloch-sphere equator
fig, axes = plt.subplots(1, 3, figsize=(5.0, 1.75))
labels = [(r"$y_1$: $0.x_3=0.1_2$", 0.5), (r"$y_2$: $0.x_2x_3=0.01_2$", 0.25), (r"$y_3$: $0.x_1x_2x_3=0.101_2$", 5 / 8)]
for ax, (lab, frac) in zip(axes, labels):
    ax.plot(np.cos(th), np.sin(th), color="#c3c2b7", linewidth=1)
    ang = 2 * np.pi * frac
    ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.plot([0, 1], [0, 0], color=GREY, linewidth=0.8)
    ax.set_title(lab + f"\nphase $2\\pi\\times{frac:g}$", fontsize=8)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")
save(fig, "qubit_phases_x5")

# ---------------------------------------------------------------- 10. periodic example, 4 qubits, period 4, offset 1
n4, r4, o4 = 4, 4, 1
alpha = np.zeros(16, complex); alpha[o4::r4] = 1; alpha /= np.linalg.norm(alpha)
beta = run_qft(alpha, n4)
assert np.allclose(beta, np.fft.ifft(alpha, norm="ortho")), "periodic check failed"
freq = sample_qft(alpha, n4, 2000)
fig, (a, b) = plt.subplots(1, 2, figsize=(5.8, 2.0))
a.bar(range(16), np.abs(alpha) ** 2, width=0.55, color=BLUE)
a.set_title(r"input: $|1\rangle,|5\rangle,|9\rangle,|13\rangle$ (period $r=4$)", loc="left", fontsize=9)
a.set_xticks(range(0, 16, 2)); a.set_xlabel("x"); a.set_ylabel("probability"); a.set_ylim(0, 0.42)
b.bar(np.arange(16) - 0.18, np.abs(beta) ** 2, width=0.36, color=GREY, label="exact")
b.bar(np.arange(16) + 0.18, freq, width=0.36, color=ORANGE, label="CUDA-Q (2000 shots)")
b.set_title("after QFT: peaks at $y = 0, 4, 8, 12$", loc="left", fontsize=9)
b.set_xticks(range(0, 16, 2)); b.set_xlabel("y"); b.set_ylim(0, 0.42); b.legend(loc="upper center", ncol=2, fontsize=7)
save(fig, "periodic_4q")
with open(os.path.join(OUT, "periodic_table.tex"), "w") as fh:
    fh.write("\\newcommand{\\periodicrows}{%\n")
    for y in (0, 4, 8, 12):
        a_ = beta[y]
        re_, im_ = round(a_.real, 3) + 0.0, round(a_.imag, 3) + 0.0      # + 0.0 avoids printing -0.000
        fh.write(f"{y} & ${re_:+.3f}{im_:+.3f}i$ & {abs(a_) ** 2:.3f} & ${np.degrees(np.angle(a_)) % 360:.0f}^\\circ$ \\\\\n")
    fh.write("}\n")

# ---------------------------------------------------------------- 11. leakage, 6 qubits, period 6
n6 = 6
alpha6 = np.zeros(64, complex); alpha6[0::6] = 1; alpha6 /= np.linalg.norm(alpha6)
beta6 = run_qft(alpha6, n6)
assert np.allclose(beta6, np.fft.ifft(alpha6, norm="ortho"))
fig, ax = plt.subplots(figsize=(5.6, 1.9))
ax.bar(range(64), np.abs(beta6) ** 2, width=0.6, color=BLUE)
for k in range(6):
    ax.axvline(64 * k / 6, color=ORANGE, linewidth=1, linestyle=(0, (3, 2)))
ax.set_xlabel("y  (orange lines: ideal positions $k\\cdot 64/6$)"); ax.set_ylabel("probability")
ax.set_title("Period 6 does not divide 64: peaks spread to neighbours (leakage)", loc="left", fontsize=9)
save(fig, "leakage_6q")

# ---------------------------------------------------------------- 12. 15-qubit example (as in QFT.ipynb)
n15 = 15
alpha15 = np.zeros(2 ** n15, complex); alpha15[3::8] = 1; alpha15 /= np.linalg.norm(alpha15)
beta15 = run_qft(alpha15, n15)
freq15 = sample_qft(alpha15, n15, 4000)
fig, ax = plt.subplots(figsize=(5.6, 1.9))
ax.vlines(np.arange(2 ** n15), 0, np.abs(beta15) ** 2, color=GREY, linewidth=1, label="exact $|\\beta_y|^2$")
nz = np.flatnonzero(freq15)
ax.plot(nz, freq15[nz], "o", color=ORANGE, markersize=4, label="CUDA-Q, 4000 shots")
ax.set_xlabel("y (0 ... 32767)"); ax.set_ylabel("probability"); ax.set_ylim(0, 0.2)
ax.set_title("15 qubits, period 8, offset 3: exactly 8 peaks at multiples of 4096", loc="left", fontsize=9)
ax.legend(loc="upper right", ncol=2)
save(fig, "qft_15q")
fid15 = abs(np.vdot(np.fft.ifft(alpha15, norm="ortho"), beta15)) ** 2
with open(os.path.join(OUT, "qft15_numbers.tex"), "w") as fh:
    fh.write(f"\\newcommand{{\\fidfifteen}}{{{fid15:.12f}}}\n")
    fh.write(f"\\newcommand{{\\maxerrfifteen}}{{{np.max(np.abs(beta15 - np.fft.ifft(alpha15, norm='ortho'))):.1e}}}\n")

# ---------------------------------------------------------------- 13. scaling: FFT vs QFT
n_range = np.arange(1, 51)
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.semilogy(n_range, n_range * 2.0 ** n_range, color=ORANGE, label=r"classical FFT: $n\,2^n$")
ax.semilogy(n_range, n_range * (n_range + 1) / 2 + n_range // 2, color=BLUE, label=r"QFT gates: $\approx n^2/2$")
ax.set_xlabel("qubits n  (vector length $2^n$)"); ax.set_ylabel("cost (log scale)")
ax.legend(loc="upper left")
save(fig, "scaling")

# ---------------------------------------------------------------- 14. approximate QFT: rotation angles
k = np.arange(2, 16)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.semilogy(k, 360 / 2 ** k, "o-", color=BLUE, markersize=4)
ax.axhline(0.1, color=ORANGE, linewidth=1)
ax.text(2, 0.055, "0.1 degree", fontsize=8, color=INK, va="top")
ax.set_xlabel("k  (gate $R_k$)"); ax.set_ylabel("rotation angle [degrees]")
save(fig, "approx_angles")

print("All figures written to", OUT)
