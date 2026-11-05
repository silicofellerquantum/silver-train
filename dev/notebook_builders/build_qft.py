from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "QFT.ipynb"
ALLOWED = ["statevector", "mps", "tensornet", "density_matrix"]

cells = []
cells.append(md(title_md(
    8, "Quantum Fourier Transform (QFT)", "Fourier analysis of 2^N amplitudes with O(N^2) gates in NVIDIA CUDA-Q",
    "This notebook builds the QFT circuit in CUDA-Q, applies it to an input state of your choice, compares "
    "the resulting amplitudes and measured bitstring probabilities with the exact mathematical result (NumPy "
    "FFT), and benchmarks time and memory on the selected backend.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
The QFT computes the **discrete Fourier transform of the $M = 2^N$ amplitudes** of an $N$-qubit state.
For a computational basis state $|x\rangle$ ($x$ is the integer written by the qubits, **qubit 0 = most
significant bit**):

$$\mathrm{QFT}\,|x\rangle = \frac{1}{\sqrt{M}} \sum_{y=0}^{M-1} e^{2\pi i\,xy/M}\,|y\rangle ,$$

and by linearity a general state $|\psi\rangle = \sum_x \alpha_x |x\rangle$ is mapped to
$\sum_y \beta_y |y\rangle$ with $\beta_y = \frac{1}{\sqrt{M}}\sum_x \alpha_x\, e^{2\pi i\,xy/M}$
(exactly NumPy's `np.fft.ifft(alpha, norm="ortho")`).

### Why is this problem important?
The QFT is the engine of the most important quantum algorithms: **Shor's factoring** (period finding),
**quantum phase estimation** (chemistry, HHL linear systems), quantum arithmetic and hidden-subgroup problems.
It turns *periodicity* hidden in amplitudes into peaks that a measurement can reveal.

### What is the classical approach?
The Fast Fourier Transform (FFT) of $M$ numbers costs $O(M \log M) = O(N\,2^N)$ operations and must store all
$2^N$ numbers.

### How does the quantum algorithm solve it?
The QFT output factorises into single-qubit states (product formula):

$$\mathrm{QFT}\,|x\rangle = \bigotimes_{l=1}^{N} \frac{|0\rangle + e^{2\pi i\, x / 2^{l}}\,|1\rangle}{\sqrt{2}} ,$$

which is built from only $N$ Hadamard gates, $N(N-1)/2$ controlled phase rotations
$R_k = \begin{pmatrix}1 & 0\\ 0 & e^{2\pi i/2^k}\end{pmatrix}$ and $\lfloor N/2\rfloor$ SWAPs:
**$O(N^2)$ gates** instead of $O(N 2^N)$ arithmetic operations. Qubit $j$ receives a Hadamard followed by
controlled rotations $R_2, R_3, \dots$ from the less significant qubits; the final SWAPs reverse the qubit order.

**Important caveat:** the result lives in the *amplitudes*. A measurement returns a single bitstring $y$ with
probability $P(y) = |\beta_y|^2$; reading all $2^N$ amplitudes would need exponentially many shots. The QFT is
useful when the answer is a **few large peaks**, as in period finding.

### Measurement
Measuring all qubits in the $Z$ basis samples $y$ with probability $|\beta_y|^2$. The distribution is **not
always uniform**: it is uniform only for a basis-state input (all $|\beta_y| = 1/\sqrt{M}$, information sits in
the phases). A periodic input with period $r$ produces peaks at multiples of $M/r$.

### What is the expected solution?
Default input: the periodic state $|\psi\rangle = \sqrt{r/M}\sum_{j}|j\,r + o\rangle$ with period $r = 8$ and
offset $o = 3$ on $N = 15$ qubits. Its QFT is

$$\mathrm{QFT}|\psi\rangle = \frac{1}{\sqrt{r}}\sum_{m=0}^{r-1} e^{2\pi i\, o\, m / r}\,|m\,M/r\rangle ,$$

i.e. **8 peaks** at $y = 0, 4096, 8192, \dots, 28672$, each with probability $1/8$ and phase $2\pi\cdot 3m/8$.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("pass"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition ---------------------------------------------------------------
N_QUBITS = 15                    # number of qubits; the QFT acts on 2**N_QUBITS amplitudes
INITIAL_STATE = "periodic"       # "periodic" | "basis" | "random" | "custom"  (see Section 4)
PERIOD = 8                       # "periodic": spacing r of the non-zero amplitudes (power of 2 -> gate-prepared, any N)
OFFSET = 3                       # "periodic": index of the first non-zero amplitude, 0 <= OFFSET < PERIOD
BASIS_STATE = 5                  # "basis": integer x with 0 <= x < 2**N_QUBITS (qubit 0 = most significant bit)
CUSTOM_AMPLITUDES = None         # "custom": list / NumPy array of 2**N_QUBITS complex amplitudes (normalised for you)

# ---- Analysis -----------------------------------------------------------------------------
AMPLITUDE_ANALYSIS_MAX_QUBITS = 24   # read back the full state vector (amplitudes, fidelity) only up to this N
TOP_K = 16                       # number of largest amplitudes / most frequent bitstrings listed
''' + common_input_block(ALLOWED, "statevector", 4000, "exports/qft")))

cells.append(md(params_md(r"""
| `N_QUBITS` | Number of qubits $N$; the transform acts on $M = 2^N$ amplitudes. | The circuit grows only like $N^2$ gates, but the simulated state doubles in memory with every qubit (16 x 2^N bytes in fp64): 15 -> 0.5 MB, 30 -> 17 GB, 40 -> 17.6 TB (use `"mps"`). |
| `INITIAL_STATE` | Input state: `"periodic"`, `"basis"`, `"random"` or `"custom"`. | Changes the output distribution: periodic -> sharp peaks; basis -> flat distribution with a linear phase ramp; random -> irregular. `"random"`/`"custom"` and non-power-of-2 periods use amplitude encoding (simulators only, N <= 24). |
| `PERIOD` / `OFFSET` | Period $r$ and offset $o$ of the periodic input. | Peaks appear at multiples of $M/r$. A power of 2 gives exactly $r$ sharp peaks; other values spread probability around the peaks ("leakage"), as in Shor's algorithm. |
| `BASIS_STATE` | Integer $x$ for the basis-state input. | Only the phases of the output change with $x$; all probabilities stay $1/M$. |
| `CUSTOM_AMPLITUDES` | Your own state vector (index = integer with qubit 0 as most significant bit). | Must have exactly $2^N$ entries; it is normalised automatically. |
| `AMPLITUDE_ANALYSIS_MAX_QUBITS` | Largest N for which the full state vector is copied back for amplitude comparison. | Each copy needs 16 x 2^N bytes of host RAM; above this size only measured counts are compared (with an analytic expectation for gate-prepared inputs). |
| `TOP_K` | Rows shown in the amplitude and bitstring tables. | Display only. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** `INITIAL_STATE = "periodic"`, `PERIOD = 8`, `OFFSET = 3`, `N_QUBITS = 15`.
Because the period is a power of two, the state is prepared with plain gates (Hadamards on the 12 most
significant qubits, $X$ gates writing the offset into the 3 least significant ones), so the example scales to
any $N$ and runs on every backend.

| Goal | Setting in Section 3 | Expected QFT output |
|---|---|---|
| Sharp peaks (default) | `INITIAL_STATE="periodic"`, `PERIOD=8`, `OFFSET=3` | 8 peaks of probability 1/8 at multiples of $2^N/8$ |
| Leakage (like Shor) | `PERIOD=6`, `OFFSET=1` | 6 clusters around multiples of $2^N/6$ (amplitude encoding) |
| Basis state | `INITIAL_STATE="basis"`, `BASIS_STATE=5` | flat distribution, phases $e^{2\pi i\,5y/M}$ |
| Random state | `INITIAL_STATE="random"` | irregular distribution; checks the transform on a generic state |
| Your own state | `INITIAL_STATE="custom"`, `CUSTOM_AMPLITUDES=[...]` | whatever its Fourier transform is |
| Large N | `N_QUBITS=30`, `BACKEND="gpu"` (or `SIMULATION_METHOD="mps"` for 40+) | same 8 peaks, now at multiples of $2^{27}$ |

The cell below builds the input state, checks it and plots its amplitudes.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Build and validate the input state from the Section 3 inputs
# ---------------------------------------------------------------------------------
if not isinstance(N_QUBITS, (int, np.integer)) or N_QUBITS < 1:
    raise ValueError("N_QUBITS must be a positive integer.")
N_QUBITS = int(N_QUBITS)
M = 2 ** N_QUBITS                                                   # number of amplitudes
mode = str(INITIAL_STATE).strip().lower()
h_mask = [0] * N_QUBITS                                             # 1 -> Hadamard on that qubit (gate preparation)
x_mask = [0] * N_QUBITS                                             # 1 -> X gate on that qubit (gate preparation)
input_amplitudes = None                                             # full vector (only if small enough / needed)

def is_power_of_two(value):
    return value >= 1 and (value & (value - 1)) == 0

if mode == "basis":
    if not 0 <= int(BASIS_STATE) < M:
        raise ValueError(f"BASIS_STATE must satisfy 0 <= x < 2**{N_QUBITS}.")
    x_mask = [int(b) for b in format(int(BASIS_STATE), f"0{N_QUBITS}b")]   # qubit 0 = most significant bit
    preparation = "gates"
    description = f"basis state |{int(BASIS_STATE)}> = |{format(int(BASIS_STATE), f'0{N_QUBITS}b')}>"
elif mode == "periodic":
    r, o = int(PERIOD), int(OFFSET)
    if not 1 <= r <= M or not 0 <= o < r:
        raise ValueError(f"Need 1 <= PERIOD <= 2**N and 0 <= OFFSET < PERIOD (got PERIOD={r}, OFFSET={o}).")
    description = f"periodic state, period r = {r}, offset o = {o}"
    if is_power_of_two(r):                                          # x = j*r + o: low bits fixed, high bits uniform
        p = r.bit_length() - 1                                      # r = 2^p
        h_mask = [1] * (N_QUBITS - p) + [0] * p                     # uniform over the N-p most significant bits
        x_mask = [0] * (N_QUBITS - p) + ([int(b) for b in format(o, f"0{p}b")] if p > 0 else [])
        preparation = "gates"
    else:
        preparation = "amplitudes"
elif mode == "random":
    preparation = "amplitudes"
    description = f"random complex state (seed {SEED})"
elif mode == "custom":
    if CUSTOM_AMPLITUDES is None:
        raise ValueError("INITIAL_STATE='custom' needs CUSTOM_AMPLITUDES (a list of 2**N_QUBITS numbers).")
    preparation = "amplitudes"
    description = "custom amplitudes"
else:
    raise ValueError("INITIAL_STATE must be 'periodic', 'basis', 'random' or 'custom'.")

analyse_amplitudes = N_QUBITS <= int(AMPLITUDE_ANALYSIS_MAX_QUBITS)
if preparation == "amplitudes" and not analyse_amplitudes:
    raise ValueError(f"'{mode}' input needs all 2**{N_QUBITS} amplitudes in memory (amplitude encoding). Use "
                     f"N_QUBITS <= {AMPLITUDE_ANALYSIS_MAX_QUBITS}, a power-of-2 PERIOD or a basis state.")

if analyse_amplitudes or preparation == "amplitudes":               # explicit input vector, index = integer x
    if mode == "basis":
        input_amplitudes = np.zeros(M, dtype=complex)
        input_amplitudes[int(BASIS_STATE)] = 1.0
    elif mode == "periodic":
        input_amplitudes = np.zeros(M, dtype=complex)
        input_amplitudes[int(OFFSET)::int(PERIOD)] = 1.0
    elif mode == "random":
        rng = np.random.default_rng(SEED)
        input_amplitudes = rng.normal(size=M) + 1j * rng.normal(size=M)
    else:
        input_amplitudes = np.asarray(CUSTOM_AMPLITUDES, dtype=complex).ravel()
        if input_amplitudes.size != M:
            raise ValueError(f"CUSTOM_AMPLITUDES must contain 2**{N_QUBITS} = {M} numbers, got {input_amplitudes.size}.")
    norm = np.linalg.norm(input_amplitudes)
    if norm == 0:
        raise ValueError("The input vector is all zeros.")
    input_amplitudes = input_amplitudes / norm                       # quantum states have norm 1

print(f"Input state      : {description}")
print(f"Qubits / amplitudes: {N_QUBITS} / {M:,}")
print(f"State preparation: {'X/H gates (works on every backend, any N)' if preparation == 'gates' else 'amplitude encoding cudaq.qvector(vector) (simulators only)'}")
if input_amplitudes is not None:
    nonzero = np.flatnonzero(np.abs(input_amplitudes) > 1e-12)
    print(f"Non-zero amplitudes: {nonzero.size:,}; first indices: {nonzero[:8].tolist()}")
    fig, ax = plt.subplots(figsize=(9, 2.6))
    shown = slice(0, min(M, 64))                                    # first 64 indices are enough to see the pattern
    ax.bar(np.arange(M)[shown], np.abs(input_amplitudes[shown]) ** 2, width=0.6, color=BLUE)
    ax.set_xlabel("basis state index x (first 64 shown)")
    ax.set_ylabel(r"$|\alpha_x|^2$")
    ax.set_title("Input state: probabilities of the first basis states", loc="left")
    plt.show()
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED) + r'''
if preparation == "amplitudes" and backend_cfg["device"] == "hardware":
    raise ValueError("Amplitude-encoded inputs cannot be sent to hardware targets. Use INITIAL_STATE='basis' or a power-of-2 PERIOD.")
can_read_state = backend_cfg["device"] != "hardware" and analyse_amplitudes     # get_state() is a simulator feature
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9)**
1. *Measured vs. expected probability distribution* $P(y) = |\beta_y|^2$ over all $2^N$ outcomes: for the default
   input 8 equal peaks; a flat line for a basis state. Measured frequencies scatter around the expectation by
   about $\sqrt{P(1-P)/S}$ for $S$ shots.
2. *Real and imaginary parts* of the largest output amplitudes (state-vector methods), CUDA-Q vs. NumPy.
3. *Memory timeline* of the execution.

**Expected numbers** - the next cell computes the exact output with NumPy's FFT (when $N$ is small enough) or
the closed-form result for gate-prepared inputs. On a state-vector simulator the CUDA-Q amplitudes should
match to about $10^{-15}$ (fp64) or $10^{-7}$ (fp32), i.e. fidelity $1$.

**Execution time and memory** are computed from exact formulas and a calibration on your backend (time = estimate).
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Exact expected result, memory requirement and time estimate
# ---------------------------------------------------------------------------------
expected_amplitudes = None
if input_amplitudes is not None:
    expected_amplitudes = np.fft.ifft(input_amplitudes, norm="ortho")   # beta_y = 1/sqrt(M) sum_x alpha_x e^{+2 pi i x y / M}
    expected_probabilities = np.abs(expected_amplitudes) ** 2
    order = np.argsort(expected_probabilities)[::-1][:TOP_K]
    print("Expected most likely outcomes (exact, NumPy FFT):")
    for y in order[:8]:
        print(f"  y = {y:>8,}  bitstring {format(int(y), f'0{N_QUBITS}b')}  P = {expected_probabilities[y]:.6f}  "
              f"phase/pi = {np.angle(expected_amplitudes[y]) / np.pi:+.4f}")
    print(f"  ... support: {np.count_nonzero(expected_probabilities > 1e-12):,} of {M:,} outcomes have P > 1e-12")
elif mode == "periodic":                                             # power-of-2 period, large N: closed form
    peaks = [m * (M // int(PERIOD)) for m in range(int(PERIOD))]
    print(f"Expected (closed form): {PERIOD} peaks at y = m * {M // int(PERIOD):,}, each P = {1 / int(PERIOD):.6f}")
else:
    print(f"Expected (closed form): uniform distribution, P(y) = 1/{M:,} for every y")

state_bytes = estimate_memory_bytes(N_QUBITS, backend_cfg["method"], backend_cfg["precision"])
host_copies = 3 if can_read_state else 0                              # get_state copy + input + expected vectors
print("\nExpected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION, extra_copies=host_copies)

gates = N_QUBITS * (N_QUBITS + 1) // 2 + N_QUBITS // 2 + sum(h_mask) + sum(x_mask)   # H + controlled R + SWAP + prep
launches = 2 if can_read_state else 1                                 # get_state() run + sampling run
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, N_QUBITS)
print("\nExpected execution time (ESTIMATE)")
print(f"  Gates per QFT circuit : {gates:,} (N(N+1)/2 = {N_QUBITS * (N_QUBITS + 1) // 2} H + controlled-R, {N_QUBITS // 2} SWAP, prep)")
print(f"  Calibration           : {timing_note}")
if seconds_per_gate is not None:
    print(f"  Estimated run time    : ~{format_duration(launches * gates * seconds_per_gate)} for {launches} circuit run(s)")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

* `qft(qubits)` - the textbook QFT for "qubit 0 = most significant bit": for every qubit $j$ a Hadamard, then
  controlled rotations $R_{k-j+1}$ (`r1` by angle $\pi/2^{k-j}$) controlled by every less significant qubit $k$,
  and finally SWAPs that reverse the qubit order.
* `qft_from_gates` - prepares the input with $H$/$X$ gates (basis and power-of-2 periodic states).
* `qft_from_amplitudes` - loads an arbitrary normalised vector with `cudaq.qvector(amplitudes)`.

CUDA-Q stores state vectors with qubit 0 as the **least** significant index bit, while printed bitstrings show
qubit 0 **first**; the helper `bit_reverse_order()` converts between the two so that index $y$ always means the
integer written left-to-right by qubits $0, 1, \dots, N-1$.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - CUDA-Q kernels for the QFT
# ---------------------------------------------------------------------------------
def build_qft_kernels():
    """Create and return fresh CUDA-Q kernels (called again in the execution cell)."""

    @cudaq.kernel
    def qft(qubits: cudaq.qview):
        """In-place QFT; qubits[0] is the most significant bit of x and of y."""
        n = qubits.size()
        for j in range(n):
            h(qubits[j])                                       # creates (|0> + e^{2 pi i 0.x_j...}|1>)/sqrt2 ...
            for k in range(j + 1, n):
                r1.ctrl(np.pi / 2 ** (k - j), qubits[k], qubits[j])   # ... adds the phase of each lower bit
        for j in range(n // 2):
            swap(qubits[j], qubits[n - 1 - j])                 # reverse qubit order (product formula is bit-reversed)

    @cudaq.kernel
    def qft_from_gates(n: int, h_mask: list[int], x_mask: list[int]):
        """Gate-prepared input (basis / power-of-2 periodic), QFT, no measurement (for get_state)."""
        qubits = cudaq.qvector(n)
        for i in range(n):
            if h_mask[i] == 1:
                h(qubits[i])                                   # uniform superposition on this bit
            if x_mask[i] == 1:
                x(qubits[i])                                   # this bit is fixed to 1
        qft(qubits)

    @cudaq.kernel
    def qft_from_gates_measured(n: int, h_mask: list[int], x_mask: list[int]):
        """Same circuit followed by a measurement of every qubit (for sampling)."""
        qubits = cudaq.qvector(n)
        for i in range(n):
            if h_mask[i] == 1:
                h(qubits[i])
            if x_mask[i] == 1:
                x(qubits[i])
        qft(qubits)
        mz(qubits)

    @cudaq.kernel
    def qft_from_amplitudes(amplitudes: list[complex]):
        """Amplitude-encoded input (vector in CUDA-Q's index order), QFT, no measurement."""
        qubits = cudaq.qvector(amplitudes)
        qft(qubits)

    @cudaq.kernel
    def qft_from_amplitudes_measured(amplitudes: list[complex]):
        """Amplitude-encoded input, QFT, measurement of every qubit."""
        qubits = cudaq.qvector(amplitudes)
        qft(qubits)
        mz(qubits)

    return {"gates": qft_from_gates, "gates_measured": qft_from_gates_measured,
            "amplitudes": qft_from_amplitudes, "amplitudes_measured": qft_from_amplitudes_measured}


def kernel_arguments(kernels, measured):
    """(kernel, arguments) for the configured input; amplitudes are converted to CUDA-Q's index order."""
    if preparation == "gates":
        return kernels["gates_measured" if measured else "gates"], (N_QUBITS, h_mask, x_mask)
    vector = bit_reverse_order(input_amplitudes, N_QUBITS).tolist()
    return kernels["amplitudes_measured" if measured else "amplitudes"], (vector,)


kernels_vis = build_qft_kernels()                                    # used for drawing / statistics only
print("Kernels defined:", ", ".join(kernels_vis))
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
n_draw = min(int(DRAW_QUBITS), 10, N_QUBITS)
if not 0 < n_draw <= 10:
    raise ValueError("DRAW_QUBITS must be between 1 and 10.")
# Representative drawing: the same kind of input on n_draw qubits (gate preparation keeps it readable)
if mode == "periodic" and is_power_of_two(int(PERIOD)) and (int(PERIOD).bit_length() - 1) <= n_draw:
    p_draw = int(PERIOD).bit_length() - 1
    h_draw = [1] * (n_draw - p_draw) + [0] * p_draw
    x_draw = [0] * (n_draw - p_draw) + ([int(b) for b in format(int(OFFSET), f"0{p_draw}b")] if p_draw else [])
else:
    h_draw, x_draw = [0] * n_draw, x_mask[:n_draw] if preparation == "gates" else [0] * n_draw
if preparation == "amplitudes":
    print("Note: the drawing starts from |0...0>; the executed circuit first loads the input amplitudes with "
          "cudaq.qvector(vector) (amplitude encoding), then applies the same QFT.\n")
full_kernel, full_args = kernel_arguments(kernels_vis, measured=True)
show_circuit(kernels_vis["gates_measured"], (n_draw, h_draw, x_draw), n_draw, N_QUBITS,
             title="QFT with gate-prepared input.", export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS,
             out_dir=EXPORT_DIR, basename="qft", full_kernel=full_kernel, full_args=full_args)
stats = circuit_statistics(full_kernel, *full_args)
print(f"\nFULL circuit statistics ({N_QUBITS} qubits):")
for key, value in stats.items():
    print(f"  {key:<32}: {value}")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the QFT circuit (fresh kernels),
3. executes it twice: once without measurement to read the full output state (state-vector methods, if
   $N \le$ `AMPLITUDE_ANALYSIS_MAX_QUBITS`) and once with `SHOTS` measurements,
4. monitors host RAM and GPU memory during the whole cell,
5. prints a short summary.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - QFT with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
quantum_amplitudes, state_phases, sampling_time = None, None, 0.0
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()
cell_start = time.perf_counter()
try:
    # Step 1 - initialise the backend
    initialize_backend(backend_cfg, seed=SEED)
    # Step 2 - build the circuit (fresh kernels: construction + compilation measured on first launch)
    kernels = build_qft_kernels()
    # Step 3a - full output state (simulators only); first launch is traced
    if can_read_state:
        kernel, args = kernel_arguments(kernels, measured=False)
        state, state_phases = profile_first_launch(lambda: cudaq.get_state(kernel, *args))
        quantum_amplitudes = bit_reverse_order(np.array(state), N_QUBITS)   # index y with qubit 0 = MSB
        del state
    # Step 3b - sampling run with SHOTS measurements
    kernel_m, args_m = kernel_arguments(kernels, measured=True)
    if state_phases is None:                                        # no state run: trace the sampling run
        counts, phases = profile_first_launch(lambda: cudaq.sample(kernel_m, *args_m, shots_count=SHOTS))
    else:
        t0 = time.perf_counter()
        counts = cudaq.sample(kernel_m, *args_m, shots_count=SHOTS)
        sampling_time = time.perf_counter() - t0
        phases = dict(state_phases)                                 # construction/compilation from the first launch
    execution_ok = True
except Exception as exc:
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce N_QUBITS or AMPLITUDE_ANALYSIS_MAX_QUBITS, choose another BACKEND / SIMULATION_METHOD (README).")
finally:
    memory = monitor.stop()
    total_wall = time.perf_counter() - cell_start

if execution_ok:
    phases["total_wall_s"] = total_wall
    measured = {int(bits, 2): c for bits, c in counts.items()}      # integer y (qubit 0 = MSB) -> counts
    print(f"\nDistinct measured outcomes: {len(measured):,} from {SHOTS} shots")
    print(f"First launch: construction {fmt_seconds(phases['construction_s'])}, compilation "
          f"{fmt_seconds(phases['compilation_s'])}, execution {fmt_seconds(phases['execution_s'])}"
          + (f"; sampling run {sampling_time:.4f} s" if state_phases is not None else ""))
    print(f"Total cell wall time: {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

1. **Amplitudes** (state-vector methods): fidelity $|\langle\beta_{\text{NumPy}}|\beta_{\text{CUDA-Q}}\rangle|^2$,
   largest deviation and a table of the largest amplitudes with real and imaginary parts.
2. **Probability distribution**: measured frequencies vs. the exact $|\beta_y|^2$, plus the total variation distance
   $\mathrm{TVD} = \tfrac12\sum_y |f_y - P_y|$ (0 = identical; with $S$ shots expect roughly $\sqrt{K/S}$ for $K$
   outcomes of similar probability).
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Compare CUDA-Q with the exact Fourier transform
# ---------------------------------------------------------------------------------
fidelity, max_error, tvd = None, None, None
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    if quantum_amplitudes is not None:
        overlap = np.vdot(expected_amplitudes, quantum_amplitudes)        # <beta_exact | beta_cudaq>
        fidelity = abs(overlap) ** 2
        max_error = float(np.max(np.abs(quantum_amplitudes - expected_amplitudes)))
        print(f"State fidelity |<exact|CUDA-Q>|^2 : {fidelity:.12f}")
        print(f"Largest amplitude deviation       : {max_error:.3e}\n")
        top = np.argsort(np.abs(quantum_amplitudes))[::-1][:TOP_K]
        top = [int(y) for y in top if abs(quantum_amplitudes[y]) > 1e-9]   # hide numerical-noise amplitudes
        print(f"{'y':>8} {'bitstring':<{N_QUBITS}}  {'Re (CUDA-Q)':>12} {'Im (CUDA-Q)':>12} {'|beta|^2':>10} {'Re (exact)':>11} {'Im (exact)':>11}")
        for y in top:
            a, e = quantum_amplitudes[y], expected_amplitudes[y]
            print(f"{y:>8} {format(int(y), f'0{N_QUBITS}b'):<{N_QUBITS}}  {a.real:>12.6f} {a.imag:>12.6f} "
                  f"{abs(a) ** 2:>10.6f} {e.real:>11.6f} {e.imag:>11.6f}")

    # ---- measured vs expected distribution -------------------------------------------------
    if expected_amplitudes is not None:
        probs = np.abs(expected_amplitudes) ** 2
    elif mode == "periodic":                                           # closed form for large N
        probs = None
        peak_set = {m * (M // int(PERIOD)) for m in range(int(PERIOD))}
    else:
        probs = None
    freq_y = np.array(list(measured.keys()), dtype=np.int64)
    freq_v = np.array(list(measured.values()), dtype=float) / SHOTS
    if probs is not None:
        tvd = 0.5 * (np.abs(freq_v - probs[freq_y]).sum() + (1.0 - probs[freq_y].sum()))  # unseen outcomes add their full P
        print(f"\nTotal variation distance (measured vs exact): {tvd:.4f}")
    elif mode == "periodic":
        on_peaks = sum(c for y, c in measured.items() if y in peak_set) / SHOTS
        print(f"\nFraction of shots on the {PERIOD} expected peaks: {on_peaks:.4f} (exact: 1.0)")
    else:
        print(f"\nUniform expectation P = 1/{M:,}: with {SHOTS} shots nearly every outcome appears at most once.")
    print(f"\nMost frequent measured bitstrings:")
    for bits, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:min(TOP_K, 10)]:
        y = int(bits, 2)
        if probs is not None:
            exact = f"{probs[y]:.4f}"                                     # exact probability from the FFT
        elif mode == "periodic":
            exact = f"{1 / int(PERIOD):.4f}" if y in peak_set else "0"    # closed-form peak probability
        else:
            exact = f"{1 / M:.2e}"                                        # uniform distribution
        print(f"  {bits}  y={y:>8,}  measured {c / SHOTS:.4f}  exact {exact}")

    # ---- plots -------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.28)
    ax_p = fig.add_subplot(grid[0, :] if quantum_amplitudes is None else grid[0, 0])
    if probs is not None and M <= 65536:
        ax_p.plot(np.arange(M), probs, color=INK_SECONDARY, linewidth=1.0, label=r"exact $|\beta_y|^2$")
    elif probs is not None:                                               # very large M: plot the largest values only
        idx = np.sort(np.argsort(probs)[::-1][:4096])
        ax_p.plot(idx, probs[idx], ".", color=INK_SECONDARY, markersize=2, label=r"exact $|\beta_y|^2$ (4096 largest)")
    ax_p.plot(freq_y, freq_v, "o", color=BLUE, markersize=4, markeredgecolor=SURFACE, markeredgewidth=0.8,
              label=f"measured ({SHOTS} shots)")
    ax_p.set_xlabel("outcome y (integer, qubit 0 = most significant bit)")
    ax_p.set_ylabel("probability")
    ax_p.set_title("QFT output: measured vs exact probabilities", loc="left")
    y_top = max(float(np.max(probs)) if probs is not None else 0.0, float(freq_v.max()))
    ax_p.set_ylim(0, y_top * 1.35)                                   # headroom keeps the legend off the data
    ax_p.legend(fontsize=8, loc="upper right", ncol=2)
    if quantum_amplitudes is not None:
        ax_a = fig.add_subplot(grid[0, 1])
        top_sorted = np.array(sorted(top))                                # largest amplitudes, in index order
        positions = np.arange(len(top_sorted))
        ax_a.bar(positions - 0.2, quantum_amplitudes[top_sorted].real, width=0.38, color=BLUE, label="Re (CUDA-Q)")
        ax_a.bar(positions + 0.2, quantum_amplitudes[top_sorted].imag, width=0.38, color=ORANGE, label="Im (CUDA-Q)")
        ax_a.plot(positions - 0.2, expected_amplitudes[top_sorted].real, "_", color=INK, markersize=12, markeredgewidth=2)
        ax_a.plot(positions + 0.2, expected_amplitudes[top_sorted].imag, "_", color=INK, markersize=12,
                  markeredgewidth=2, label="exact (NumPy)")
        ax_a.axhline(0, color=AXIS_COLOR, linewidth=1)
        ax_a.set_xticks(positions, [str(y) for y in top_sorted], rotation=60, fontsize=7)
        ax_a.set_xlabel(f"outcome y ({len(top_sorted)} largest amplitudes)")
        ax_a.set_ylabel("amplitude")
        ax_a.set_title("Real and imaginary parts of the output amplitudes", loc="left")
        a_max = float(np.max(np.abs(np.r_[quantum_amplitudes[top_sorted].real, quantum_amplitudes[top_sorted].imag])))
        ax_a.set_ylim(-1.2 * a_max, 1.7 * a_max)                     # headroom for the legend
        ax_a.legend(fontsize=8, loc="upper right", ncol=3)
        ax_a.grid(axis="x", visible=False)
    ax_m = fig.add_subplot(grid[1, :])
    monitor.plot(ax=ax_m, title="Memory during the execution cell")
    plt.show()
'''))

cells.append(md(r"""
---
## Section 10 - Final benchmark report
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 10 - Final benchmark report
# ---------------------------------------------------------------------------------
if not execution_ok:
    print(f"No report: execution failed with {type(execution_error).__name__}: {execution_error}")
else:
    extra = {"Sampling run time (execution)": sampling_time} if state_phases is not None else {}
    report = build_report("Quantum Fourier Transform", backend_cfg, N_QUBITS, SHOTS, phases, memory, extra_times=extra)
    report["Input state"] = description
    report["Gates in the QFT circuit"] = f"{stats.get('gates (excluding measurements)', 'n/a')} (depth {stats.get('depth', 'n/a')})"
    if expected_amplitudes is not None:
        top_y = int(np.argmax(np.abs(expected_amplitudes)))
        report["Expected result"] = (f"exact FFT: {np.count_nonzero(np.abs(expected_amplitudes) ** 2 > 1e-12):,} outcomes with P>0, "
                                     f"largest P = {np.max(np.abs(expected_amplitudes)) ** 2:.4f} at y = {top_y}")
    elif mode == "periodic":
        report["Expected result"] = f"{PERIOD} peaks at multiples of {M // int(PERIOD):,}, P = {1 / int(PERIOD):.4f} each"
    else:
        report["Expected result"] = f"uniform distribution, P(y) = 1/{M:,}"
    best = max(counts.items(), key=lambda kv: kv[1])
    report["Actual result"] = f"{len(measured):,} distinct outcomes; most frequent {best[0]} (y={int(best[0], 2):,}) with {best[1] / SHOTS:.4f}"
    if fidelity is not None:
        report["State fidelity vs exact QFT"] = f"{fidelity:.12f} (max amplitude error {max_error:.2e})"
    if tvd is not None:
        report["Total variation distance (shots)"] = f"{tvd:.4f}"
    elif mode == "periodic":
        report["Fraction of shots on expected peaks"] = f"{on_peaks:.4f}"
    final_fig, ax = plt.subplots(figsize=(10, 3.4))
    if probs is not None and M <= 65536:
        ax.plot(np.arange(M), probs, color=INK_SECONDARY, linewidth=1.0, label="exact")
    elif probs is not None:
        ax.plot(idx, probs[idx], ".", color=INK_SECONDARY, markersize=2, label="exact (largest values)")
    ax.plot(freq_y, freq_v, "o", color=BLUE, markersize=4, markeredgecolor=SURFACE, label="measured")
    ax.set_xlabel("outcome y")
    ax.set_ylabel("probability")
    ax.set_ylim(0, y_top * 1.35)
    ax.set_title(f"QFT N={N_QUBITS} ({description}) on {cudaq.get_target().name}", loc="left")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="qft_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* `PERIOD = 6` shows *leakage*: the period does not divide $2^N$, so probability spreads around the ideal peaks
  $m\,2^N/6$ - exactly the situation that Shor's algorithm handles with continued fractions.
* `INITIAL_STATE = "basis"` gives a perfectly flat distribution: all information is in the phases (see the
  amplitude table), which is why a single QFT measurement cannot reveal a basis state.
* Increase `N_QUBITS` to 25-30 on a GPU, or to 40+ with `SIMULATION_METHOD = "mps"` - the QFT of the default
  periodic input is only weakly entangled, so the matrix-product-state simulator handles it easily.
"""))

write_notebook(OUT, cells, "Quantum Fourier Transform - CUDA-Q benchmark")
print("wrote", OUT)
