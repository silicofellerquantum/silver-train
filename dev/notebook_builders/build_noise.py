from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "Noise_Simulation.ipynb"
ALLOWED = ["stabilizer", "density_matrix", "trajectory"]

cells = []
cells.append(md(title_md(
    7, "Quantum Noise Simulation", "How gate errors degrade a quantum circuit, simulated with NVIDIA CUDA-Q",
    "This notebook runs the same circuit with and without a noise model, compares the ideal and noisy "
    "measurement distributions, computes fidelities, and sweeps the error probability to show how noise "
    "destroys the quantum state. It uses CUDA-Q's noise-capable simulators (stabilizer, density matrix and "
    "GPU trajectory simulation) and measures time and memory.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the simulation solve?
Real quantum computers are **noisy**: every gate has a small probability of error and qubits lose their state
(decoherence). Noise simulation predicts what a circuit produces on such hardware, so we can decide whether an
algorithm will still work and how good the hardware must be.

### Why is this important?
* It quantifies the error rates needed for an algorithm to succeed (hardware requirements, error budgets).
* It is used to develop and test error mitigation and error correction before running on hardware.
* It explains why today's devices are limited to shallow circuits: errors accumulate with every gate.

### What is the classical approach?
A noisy state is a **density matrix** $\rho$ ($2^N \times 2^N$, i.e. $4^N$ numbers). Each gate $U$ acts as
$\rho \to U\rho U^\dagger$, followed by a **noise channel** written with Kraus operators:

$$\mathcal{E}(\rho) = \sum_k K_k\,\rho\,K_k^\dagger,\qquad \sum_k K_k^\dagger K_k = I .$$

Exact density-matrix simulation costs $4^N$ memory. Two cheaper exact-in-distribution alternatives exist:
**Monte-Carlo trajectories** (state vectors, $2^N$, errors inserted at random - CUDA-Q's GPU simulators) and the
**stabilizer formalism** (polynomial cost, but only for Clifford circuits with Pauli noise - CUDA-Q's `stim` target).

### Noise channels used here (probability $p$, applied after every gate)
| Channel | Kraus operators | Physical meaning |
|---|---|---|
| depolarizing | $\sqrt{1-p}\,I,\ \sqrt{p/3}\,X,\ \sqrt{p/3}\,Y,\ \sqrt{p/3}\,Z$ (2-qubit gates: $\sqrt{1-p}\,II$ + 15 Paulis with $p/15$) | random Pauli error |
| bit flip | $\sqrt{1-p}\,I,\ \sqrt{p}\,X$ | $|0\rangle \leftrightarrow |1\rangle$ flips |
| phase flip | $\sqrt{1-p}\,I,\ \sqrt{p}\,Z$ | loss of phase coherence (dephasing) |
| amplitude damping | $\begin{pmatrix}1&0\\0&\sqrt{1-p}\end{pmatrix},\ \begin{pmatrix}0&\sqrt{p}\\0&0\end{pmatrix}$ | energy relaxation $|1\rangle\to|0\rangle$ ($T_1$) |

For two-qubit gates the bit-flip, phase-flip and amplitude-damping channels act independently on both qubits.

### The circuits
* **GHZ state** (default): $H$ on qubit 0 and a CNOT chain give
  $|\mathrm{GHZ}\rangle = \tfrac{1}{\sqrt2}(|0\cdots0\rangle + |1\cdots1\rangle)$, a maximally entangled state that is
  extremely sensitive to noise. It is a Clifford circuit, so the stabilizer simulator can handle many qubits.
* **Mirror circuit**: random rotation layers $U$ followed by $U^\dagger$; ideally it returns exactly
  $|0\cdots0\rangle$. Its rotations are not Clifford gates, so it needs density-matrix or trajectory simulation.

### Measurement and figures of merit
* **Z basis**: ideal GHZ gives $0\cdots0$ and $1\cdots1$ with 50 % each. The **GHZ population**
  $P_{\text{pop}} = P(0\cdots0) + P(1\cdots1)$ drops when bit flips occur.
* **X basis** (Hadamard on every qubit before measuring): ideal GHZ gives only even-parity bitstrings, so
  $\langle X^{\otimes N}\rangle = 1$. Phase errors are **invisible in the Z basis** but reduce this parity.
* **Hellinger (classical) fidelity** between ideal distribution $p$ and noisy distribution $q$:
  $F_H = \big(\sum_x \sqrt{p_x q_x}\big)^2$ (1 = identical).
* **State fidelity** (density-matrix method only): $F = \langle\psi_{\text{ideal}}|\rho|\psi_{\text{ideal}}\rangle$,
  and **purity** $\mathrm{Tr}\,\rho^2$ (1 for a pure state).

### Expected solution
Without noise the ideal results above are reproduced exactly. With error probability $p$ per gate and $G$ noisy gate
locations, the probability that *no* error occurs is $(1-p)^{G}$ - a lower bound on the fidelity for Pauli noise -
so fidelity decays roughly exponentially with $p\cdot G$: **more qubits and deeper circuits are more fragile**.
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

# ---- Problem definition: circuit and noise ----------------------------------------------
N_QUBITS = 15                    # qubits in the circuit (density matrix: keep <= 10; stabilizer: hundreds possible)
CIRCUIT = "ghz"                  # "ghz" (Clifford, default) | "mirror" (random rotations + inverse, non-Clifford)
MIRROR_LAYERS = 2                # "mirror": random layers before the inverse (circuit depth ~ 2 x layers)
NOISE_MODEL = "depolarizing"     # "depolarizing" | "bit_flip" | "phase_flip" | "amplitude_damping"
NOISE_PROBABILITY = 0.01         # error probability p after every gate (0 = noiseless)
NOISE_SWEEP = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]   # p values for the "fidelity vs noise" plot
MEASURE_X_BASIS = True           # GHZ only: also measure in the X basis (reveals phase errors)
''' + common_input_block(ALLOWED, "auto", 10000, "exports/noise").replace(
    '# this notebook supports: "stabilizer" | "density_matrix" | "trajectory"',
    '# "auto" | "stabilizer" (CPU) | "density_matrix" (CPU) | "trajectory" (GPU)')))

cells.append(md(params_md(r"""
| `N_QUBITS` | Width of the circuit. | More qubits = more gates = more errors (fidelity drops). Memory: stabilizer ~N^2 bits, trajectory 16 x 2^N bytes, density matrix 16 x 4^N bytes (N = 12 needs 268 MB, N = 14 needs 4.3 GB). |
| `CIRCUIT` / `MIRROR_LAYERS` | Which circuit is simulated. | GHZ works with every method; the mirror circuit is non-Clifford (no stabilizer method). More mirror layers = deeper circuit = lower fidelity. |
| `NOISE_MODEL` | Kind of error (table in Section 1). | Bit flips lower the Z-basis population, phase flips only the X-basis parity, depolarizing noise both, amplitude damping biases outcomes towards 0. The stabilizer method supports only Pauli channels (not amplitude damping). |
| `NOISE_PROBABILITY` | Error probability $p$ per gate. | Fidelity decays roughly like $(1-p)^G$. Real devices: $p \approx 10^{-4}-10^{-2}$. |
| `NOISE_SWEEP` | Probabilities for the sweep plot. | More values = smoother curve, proportionally more run time. |
| `MEASURE_X_BASIS` | Extra X-basis run for GHZ. | Needed to see phase errors; doubles the number of runs. |
| `SIMULATION_METHOD` | `"auto"` picks: GPU -> `"trajectory"`; CPU -> `"stabilizer"` for GHZ with Pauli noise, otherwise `"density_matrix"`. | The noiseless state-vector simulators ignore noise, so they are not offered here. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** a 15-qubit GHZ circuit with depolarizing noise $p = 0.01$ after every gate,
10 000 shots, and a sweep over $p \in [0, 0.1]$. On a CPU it uses the stabilizer simulator (`stim`); on a GPU
the trajectory simulator.

| Goal | Setting in Section 3 | What you will see |
|---|---|---|
| Phase noise is invisible in Z | `NOISE_MODEL = "phase_flip"` | Z-basis population stays 1, X-basis parity drops |
| Energy relaxation | `NOISE_MODEL = "amplitude_damping"`, `N_QUBITS = 8` | outcomes biased towards 0 (density matrix) |
| Non-Clifford circuit | `CIRCUIT = "mirror"`, `N_QUBITS = 6` | success probability of returning to `000000` |
| Exact density matrix | `SIMULATION_METHOD = "density_matrix"`, `N_QUBITS = 8` | state fidelity and purity |
| Large stabilizer run | `N_QUBITS = 100` (GHZ, Pauli noise) | runs in seconds on a CPU |
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Validate the circuit / noise choice and describe the instance
# ---------------------------------------------------------------------------------
if not isinstance(N_QUBITS, (int, np.integer)) or N_QUBITS < 2:
    raise ValueError("N_QUBITS must be an integer >= 2.")
N_QUBITS = int(N_QUBITS)
circuit = str(CIRCUIT).strip().lower()
noise_kind = str(NOISE_MODEL).strip().lower()
if circuit not in ("ghz", "mirror"):
    raise ValueError("CIRCUIT must be 'ghz' or 'mirror'.")
if noise_kind not in ("depolarizing", "bit_flip", "phase_flip", "amplitude_damping"):
    raise ValueError("NOISE_MODEL must be 'depolarizing', 'bit_flip', 'phase_flip' or 'amplitude_damping'.")
p_main = float(NOISE_PROBABILITY)
sweep = sorted({float(p) for p in NOISE_SWEEP} | {p_main})
if not all(0.0 <= p <= 1.0 for p in sweep):
    raise ValueError("Noise probabilities must lie in [0, 1].")
pauli_noise = noise_kind in ("depolarizing", "bit_flip", "phase_flip")

mirror_rng = np.random.default_rng(SEED)
mirror_angles = mirror_rng.uniform(0, 2 * np.pi, 2 * int(MIRROR_LAYERS) * N_QUBITS).tolist()   # RY, RZ angle per qubit and layer

if circuit == "ghz":
    gates_1q, gates_2q = 1, N_QUBITS - 1                     # one H, N-1 CNOTs
    ideal = {"0" * N_QUBITS: 0.5, "1" * N_QUBITS: 0.5}       # ideal Z-basis distribution
    description = f"{N_QUBITS}-qubit GHZ state (|0...0> + |1...1>)/sqrt(2)"
else:
    gates_1q = 4 * int(MIRROR_LAYERS) * N_QUBITS             # RY+RZ forward and backward
    gates_2q = 2 * int(MIRROR_LAYERS) * (N_QUBITS - 1)       # CNOT ladders forward and backward
    ideal = {"0" * N_QUBITS: 1.0}
    description = f"{N_QUBITS}-qubit mirror circuit, {MIRROR_LAYERS} random layer(s) + inverse"

def no_error_probability(p):
    """Probability that no error happens anywhere = lower bound on the fidelity (Pauli noise)."""
    p2 = p if noise_kind == "depolarizing" else 1 - (1 - p) ** 2   # 2-qubit: one joint or two independent channels
    return (1 - p) ** gates_1q * (1 - p2) ** gates_2q

# cudaq.get_state() has no noise_model argument in CUDA-Q 0.16, so the density-matrix method uses
# cudaq.set_noise(), which prints a deprecation notice. The model object must stay referenced while in use.
warnings.filterwarnings("ignore", message=".*set_noise is deprecated.*")


def build_noise_model(kind, p):
    """cudaq.NoiseModel with `kind` noise of strength p after every gate."""
    model = cudaq.NoiseModel()
    if p == 0.0:
        return model                                    # empty model = noiseless
    if kind == "depolarizing":
        one, two = cudaq.DepolarizationChannel(p), cudaq.Depolarization2(p)
    elif kind == "bit_flip":
        probs = [0.0] * 15                              # Pauli2 order: IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
        probs[0] = probs[3] = p * (1 - p)               # flip on one qubit only
        probs[4] = p * p                                # flip on both qubits
        one, two = cudaq.BitFlipChannel(p), cudaq.Pauli2(probs)
    elif kind == "phase_flip":
        probs = [0.0] * 15
        probs[2] = probs[11] = p * (1 - p)              # IZ and ZI
        probs[14] = p * p                               # ZZ
        one, two = cudaq.PhaseFlipChannel(p), cudaq.Pauli2(probs)
    else:                                               # amplitude damping, independently on both qubits of a CNOT
        k0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
        k1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
        one = cudaq.AmplitudeDampingChannel(p)
        two = cudaq.KrausChannel([np.kron(a, b) for a in (k0, k1) for b in (k0, k1)])
    for gate in ("h", "x", "ry", "rz"):
        model.add_all_qubit_channel(gate, one)           # after every single-qubit gate of these types
    model.add_all_qubit_channel("x", two, num_controls=1)   # after every CNOT
    return model


print(f"Circuit            : {description}")
print(f"Noisy gate count   : {gates_1q} single-qubit + {gates_2q} two-qubit gates")
print(f"Noise model        : {noise_kind}, p = {p_main} per gate ({'Pauli channel' if pauli_noise else 'non-Pauli channel'})")
print(f"Ideal Z-basis outcomes: {ideal}")
print(f"Probability of an error-free run at p = {p_main}: {no_error_probability(p_main):.4f}")
print(f"Noise sweep values : {sweep}")
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Choose the noise-capable simulation method ("auto") and activate the backend
# ---------------------------------------------------------------------------------
method_requested = str(SIMULATION_METHOD).strip().lower()
backend_requested = str(BACKEND).strip().lower()
if backend_requested == "hardware":
    raise ValueError("Noise models are a simulator feature. On real QPUs noise is physical: run the other notebooks "
                     "with BACKEND='hardware' and compare against their ideal results instead.")
device_guess = ("gpu" if cudaq.num_available_gpus() > 0 else "cpu") if backend_requested == "auto" else backend_requested
if method_requested == "auto":
    if device_guess in ("gpu", "multi-gpu"):
        chosen_method = "trajectory"
    elif circuit == "ghz" and pauli_noise:
        chosen_method = "stabilizer"
    else:
        chosen_method = "density_matrix"
else:
    chosen_method = method_requested
if chosen_method == "stabilizer" and not pauli_noise:
    raise ValueError("The stabilizer simulator (stim) only supports Pauli noise; it would IGNORE amplitude damping. "
                     "Use SIMULATION_METHOD='density_matrix' (CPU, N <= ~12) or 'trajectory' (GPU).")
if chosen_method == "stabilizer" and circuit != "ghz":
    raise ValueError("The mirror circuit contains non-Clifford rotations: use 'density_matrix' or 'trajectory'.")
if chosen_method == "statevector":
    raise ValueError("Noiseless state-vector simulators ignore noise models. Use 'stabilizer', 'density_matrix' or 'trajectory'.")

ALLOWED_METHODS = ["stabilizer", "density_matrix", "trajectory"]
np.random.seed(SEED)
backend_cfg = resolve_backend(BACKEND, chosen_method, GPU_PRECISION, ALLOWED_METHODS,
                              HARDWARE_PROVIDER, HARDWARE_EMULATE, HARDWARE_MACHINE)
active_target = initialize_backend(backend_cfg, seed=SEED)
print(f"Requested: BACKEND={BACKEND!r}, SIMULATION_METHOD={SIMULATION_METHOD!r}")
print(f"Using    : device {backend_cfg['device']}, method {chosen_method}, CUDA-Q target {active_target.name} ({backend_cfg['description']})")
print(f"GPUs visible: {backend_cfg['num_gpus']}, CPU cores: {os.cpu_count()}, host RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
gpu_list, gpu_status = _nvml_gpu_memory()
for i, (used, total, name) in enumerate(gpu_list or []):
    print(f"  GPU{i}: {name}, {total / 1e9:.1f} GB total, {used / 1e9:.2f} GB in use")
if not gpu_list:
    print(f"  GPU monitoring: {gpu_status}")
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9)**
1. *Ideal vs. noisy distribution* (Z basis): the two GHZ peaks (or the single `0...0` peak of the mirror
   circuit) shrink and probability leaks into many other bitstrings.
2. *Figures of merit vs. noise probability* (log scale): fidelities start at 1 for $p = 0$ and decay
   roughly exponentially; the error-free-run probability $(1-p)^G$ is drawn as a reference lower bound.
3. *Memory timeline*.

**Expected numbers** are computed below: the ideal distribution, the lower bound $(1-p)^G$ for every $p$, the
memory of the chosen method (exact formula) and a time estimate.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Expected results, memory requirement and time estimate
# ---------------------------------------------------------------------------------
print("Ideal (noise-free) result: " + ", ".join(f"P({b}) = {p}" for b, p in ideal.items()))
if circuit == "ghz":
    print("Ideal X-basis parity <X...X> = 1 (only even-parity bitstrings)")
print("\nLower bound on the fidelity (probability of an error-free run):")
for p in sweep:
    print(f"  p = {p:<6g}: (1-p)^G = {no_error_probability(p):.4f}" + ("   [Pauli noise: valid lower bound]" if pauli_noise else ""))

state_bytes = estimate_memory_bytes(N_QUBITS, chosen_method, backend_cfg["precision"])
print("\nExpected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION, extra_copies=1 if chosen_method == "density_matrix" else 0)

runs = len(sweep) * (2 if (circuit == "ghz" and MEASURE_X_BASIS) else 1)
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, N_QUBITS)
print("\nExpected execution time (ESTIMATE)")
print(f"  Circuit runs : {runs} ({len(sweep)} noise values x {'2 bases' if runs > len(sweep) else '1 basis'})")
print(f"  Calibration  : {timing_note}")
if chosen_method == "density_matrix":
    # Noise channels dominate the cost and general Kraus channels (amplitude damping) are much slower than Pauli
    # channels, so we time ONE real noisy density-matrix run of a small GHZ chain with the configured noise and
    # scale it: cost ~ 4^N (density-matrix size) x number of noisy gate locations.
    @cudaq.kernel
    def calibration_chain(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])

    n_cal = min(N_QUBITS, 6)
    calibration_model = build_noise_model(noise_kind, max(p_main, 0.01))   # keep a reference while CUDA-Q uses it
    cudaq.set_noise(calibration_model)
    try:
        cudaq.get_state(calibration_chain, n_cal)                    # warm-up (JIT compilation)
        t0 = time.perf_counter()
        cudaq.get_state(calibration_chain, n_cal)
        t_cal = time.perf_counter() - t0
    finally:
        cudaq.unset_noise()
    per_run = t_cal * 4 ** (N_QUBITS - n_cal) * (gates_1q + gates_2q) / n_cal
    print(f"  Noisy calibration run: {n_cal}-qubit chain with {noise_kind} noise took {t_cal * 1e3:.1f} ms; "
          f"scaled by 4^{N_QUBITS - n_cal} and the gate count")
    print(f"  Estimated    : ~{format_duration(runs * per_run)} for {runs} runs (rough estimate)")
elif seconds_per_gate is not None:
    estimate = runs * (gates_1q + gates_2q) * seconds_per_gate * 2
    print(f"  Estimated    : ~{format_duration(estimate)} (rough: each noise channel costs about one more gate)")
elif chosen_method == "stabilizer":
    print("  Stabilizer simulation cost grows only polynomially (expect well below a second per run for N <= 100).")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

* `ghz_z` / `ghz_x` - GHZ preparation measured in the Z basis / in the X basis (Hadamards before measuring).
* `ghz_state`, `mirror_state` - the same circuits without measurement (density-matrix method reads $\rho$ directly).
* `mirror_z` - mirror circuit $U^\dagger U$ with random $R_Y R_Z$ layers and CNOT ladders.
* `build_noise_model(kind, p)` (defined in Section 4) - a `cudaq.NoiseModel` that attaches the chosen channel after **every**
  single-qubit gate (`h`, `x`, `ry`, `rz`) and a two-qubit channel after every CNOT (`x` with one control).
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - CUDA-Q kernels and the noise model
# ---------------------------------------------------------------------------------
def build_noise_kernels():
    """Create and return fresh CUDA-Q kernels (called again in the execution cell)."""

    @cudaq.kernel
    def ghz_state(n: int):
        """(|0...0> + |1...1>)/sqrt(2), no measurement."""
        qubits = cudaq.qvector(n)
        h(qubits[0])                                   # superposition on qubit 0
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])           # copy it along the chain -> entanglement

    @cudaq.kernel
    def ghz_z(n: int):
        """GHZ measured in the Z (computational) basis."""
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        mz(qubits)

    @cudaq.kernel
    def ghz_x_state(n: int):
        """GHZ followed by Hadamards (X-basis measurement = Z measurement after H)."""
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        h(qubits)

    @cudaq.kernel
    def ghz_x(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            x.ctrl(qubits[i], qubits[i + 1])
        h(qubits)
        mz(qubits)

    @cudaq.kernel
    def mirror_state(n: int, layers: int, angles: list[float]):
        """U followed by U^dagger: ideally returns exactly |0...0>."""
        qubits = cudaq.qvector(n)
        for layer in range(layers):                    # U: rotation layer + CNOT ladder
            for q in range(n):
                ry(angles[2 * (layer * n + q)], qubits[q])
                rz(angles[2 * (layer * n + q) + 1], qubits[q])
            for q in range(n - 1):
                x.ctrl(qubits[q], qubits[q + 1])
        for back in range(layers):                     # U^dagger: same gates, reverse order, negative angles
            layer = layers - 1 - back
            for qq in range(n - 1):
                q = n - 2 - qq
                x.ctrl(qubits[q], qubits[q + 1])
            for q in range(n):
                rz(-angles[2 * (layer * n + q) + 1], qubits[q])
                ry(-angles[2 * (layer * n + q)], qubits[q])

    @cudaq.kernel
    def mirror_z(n: int, layers: int, angles: list[float]):
        qubits = cudaq.qvector(n)
        for layer in range(layers):
            for q in range(n):
                ry(angles[2 * (layer * n + q)], qubits[q])
                rz(angles[2 * (layer * n + q) + 1], qubits[q])
            for q in range(n - 1):
                x.ctrl(qubits[q], qubits[q + 1])
        for back in range(layers):
            layer = layers - 1 - back
            for qq in range(n - 1):
                q = n - 2 - qq
                x.ctrl(qubits[q], qubits[q + 1])
            for q in range(n):
                rz(-angles[2 * (layer * n + q) + 1], qubits[q])
                ry(-angles[2 * (layer * n + q)], qubits[q])
        mz(qubits)

    return {"ghz_state": ghz_state, "ghz_z": ghz_z, "ghz_x_state": ghz_x_state, "ghz_x": ghz_x,
            "mirror_state": mirror_state, "mirror_z": mirror_z}

kernels_vis = build_noise_kernels()
print("Kernels defined:", ", ".join(kernels_vis))
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
n_draw = min(int(DRAW_QUBITS), 10, N_QUBITS)
if not 0 < n_draw <= 10:
    raise ValueError("DRAW_QUBITS must be between 1 and 10.")
if circuit == "ghz":
    draw_kernel, draw_args = kernels_vis["ghz_z"], (n_draw,)
    full_kernel, full_args = kernels_vis["ghz_z"], (N_QUBITS,)
else:
    draw_kernel, draw_args = kernels_vis["mirror_z"], (n_draw, 1, mirror_angles[:2 * n_draw])
    full_kernel, full_args = kernels_vis["mirror_z"], (N_QUBITS, int(MIRROR_LAYERS), mirror_angles)
    print("The drawing shows ONE mirror layer and its inverse.")
show_circuit(draw_kernel, draw_args, n_draw, N_QUBITS, title=description, export=EXPORT_CIRCUIT,
             formats=EXPORT_FORMATS, out_dir=EXPORT_DIR, basename=f"noise_{circuit}",
             full_kernel=full_kernel, full_args=full_args)
print("\nNoise is NOT drawn: CUDA-Q inserts the Kraus channel after every gate at simulation time.")
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
2. builds the circuits (fresh kernels),
3. runs the circuit for every noise probability in the sweep (always including $p = 0$ and `NOISE_PROBABILITY`),
   in the Z basis and - for GHZ - in the X basis,
4. monitors host RAM and GPU memory during the whole cell,
5. prints a short summary.

How the noisy circuit is executed:
* **stabilizer / trajectory**: `cudaq.sample(kernel, noise_model=model, shots_count=SHOTS)`.
* **density matrix**: `cudaq.set_noise(model)` + `cudaq.get_state(kernel)` returns the exact noisy $\rho$. The
  measurement probabilities are its diagonal, and the `SHOTS` counts are drawn from them with NumPy (seeded),
  which is statistically identical to measuring. We do this because in CUDA-Q 0.16 noisy *sampling* on the
  density-matrix target took ~115 s for 10 qubits on our test machine versus ~1 s for `get_state`.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - noisy simulations with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
results = {}                                                       # p -> dict of measured quantities
phases, later_time = None, 0.0
counts_rng = np.random.default_rng(SEED)                           # used only for density-matrix shot draws
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()
cell_start = time.perf_counter()

def as_counts_from_rho(rho, n, shots):
    """Shot counts drawn from diag(rho) (CUDA-Q index order -> bitstring with qubit 0 first)."""
    probs = np.clip(np.real(np.diag(rho)), 0, None)
    probs = probs / probs.sum()
    probs_q0_first = bit_reverse_order(probs, n)                  # index now reads qubit 0 as most significant bit
    draws = counts_rng.multinomial(shots, probs_q0_first)
    exact = {format(int(i), f"0{n}b"): float(probs_q0_first[i]) for i in np.flatnonzero(probs_q0_first > 1e-15)}
    return {format(int(i), f"0{n}b"): int(draws[i]) for i in np.flatnonzero(draws)}, exact

def run_once(kernels, kind, p, basis):
    """Execute one circuit; returns (counts dict, exact distribution or None, density matrix or None)."""
    model = build_noise_model(kind, p)
    if chosen_method == "density_matrix":
        name = {"ghz": "ghz_state" if basis == "Z" else "ghz_x_state", "mirror": "mirror_state"}[circuit]
        args = (N_QUBITS,) if circuit == "ghz" else (N_QUBITS, int(MIRROR_LAYERS), mirror_angles)
        cudaq.set_noise(model)
        try:
            rho = np.array(cudaq.get_state(kernels[name], *args))
        finally:
            cudaq.unset_noise()
        counts, exact = as_counts_from_rho(rho, N_QUBITS, SHOTS)
        return counts, exact, rho
    name = {"ghz": "ghz_z" if basis == "Z" else "ghz_x", "mirror": "mirror_z"}[circuit]
    args = (N_QUBITS,) if circuit == "ghz" else (N_QUBITS, int(MIRROR_LAYERS), mirror_angles)
    result = cudaq.sample(kernels[name], *args, shots_count=SHOTS, noise_model=model)
    return {b: c for b, c in result.items()}, None, None

try:
    # Step 1 - initialise the backend
    initialize_backend(backend_cfg, seed=SEED)
    # Step 2 - build the circuits (fresh kernels)
    kernels = build_noise_kernels()
    # Step 3 - run every noise probability (the first launch is traced for the timing breakdown)
    bases = ["Z", "X"] if (circuit == "ghz" and MEASURE_X_BASIS) else ["Z"]
    for p in sweep:
        results[p] = {}
        for basis in bases:
            if phases is None:
                out, phases = profile_first_launch(lambda: run_once(kernels, noise_kind, p, basis))
            else:
                t0 = time.perf_counter()
                out = run_once(kernels, noise_kind, p, basis)
                later_time += time.perf_counter() - t0
            results[p][basis] = out
    execution_ok = True
except Exception as exc:
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce N_QUBITS (density matrix needs 16 x 4^N bytes), change SIMULATION_METHOD or NOISE_MODEL (README).")
finally:
    memory = monitor.stop()
    total_wall = time.perf_counter() - cell_start

def hellinger_fidelity(p_dist, q_counts, shots):
    """(sum_x sqrt(p_x q_x))^2 between an ideal distribution and measured counts."""
    return sum(np.sqrt(px * q_counts.get(x, 0) / shots) for x, px in p_dist.items()) ** 2

def parity_expectation(counts, shots):
    """<X...X> from X-basis counts: +1 for even number of 1s, -1 for odd."""
    return sum((1 if b.count("1") % 2 == 0 else -1) * c for b, c in counts.items()) / shots

if execution_ok:
    phases["total_wall_s"] = total_wall
    metrics = {}
    for p in sweep:
        counts_z, exact_z, rho_z = results[p]["Z"]
        m = {"hellinger": hellinger_fidelity(ideal, counts_z, SHOTS),
             "population": sum(counts_z.get(b, 0) for b in ideal) / SHOTS,
             "tvd": 0.5 * (sum(abs(counts_z.get(b, 0) / SHOTS - ideal.get(b, 0)) for b in set(counts_z) | set(ideal))),
             "distinct": len(counts_z), "lower_bound": no_error_probability(p)}
        if "X" in results[p]:
            m["parity_x"] = parity_expectation(results[p]["X"][0], SHOTS)
        if rho_z is not None:                                            # exact state fidelity and purity
            if circuit == "ghz":
                d = rho_z.shape[0]
                m["state_fidelity"] = float(0.5 * np.real(rho_z[0, 0] + rho_z[d - 1, d - 1] + rho_z[0, d - 1] + rho_z[d - 1, 0]))
            else:
                m["state_fidelity"] = float(np.real(rho_z[0, 0]))
            m["purity"] = float(np.real(np.vdot(rho_z.conj().T, rho_z))) if N_QUBITS <= 12 else None
        metrics[p] = m
    main = metrics[p_main]
    print(f"\nResults at the configured noise p = {p_main} ({noise_kind}, {chosen_method}):")
    print(f"  Hellinger fidelity (Z basis)      : {main['hellinger']:.4f}")
    print(f"  {'GHZ population P(0..0)+P(1..1)' if circuit == 'ghz' else 'Success probability P(0...0)':<34}: {main['population']:.4f}")
    if "parity_x" in main:
        print(f"  X-basis parity <X...X>            : {main['parity_x']:.4f} (ideal 1)")
    if "state_fidelity" in main:
        print(f"  Exact state fidelity <psi|rho|psi>: {main['state_fidelity']:.4f}")
        if main.get("purity") is not None:
            print(f"  Purity Tr(rho^2)                  : {main['purity']:.4f}")
    print(f"  Lower bound (1-p)^G               : {main['lower_bound']:.4f}")
    print(f"  Distinct outcomes                 : {main['distinct']}")
    print(f"\nTiming: first launch construction {fmt_seconds(phases['construction_s'])}, compilation {fmt_seconds(phases['compilation_s'])}, "
          f"execution {fmt_seconds(phases['execution_s'])}; remaining {sum(len(v) for v in results.values()) - 1} runs {later_time:.2f} s")
    print(f"Total cell wall time: {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization

**How noise affects the circuit.** Each gate error moves part of the probability away from the ideal outcomes.
For the GHZ state a single bit flip anywhere breaks the "all equal" pattern (lower Z-basis population), while a
single phase flip turns $|0\cdots0\rangle + |1\cdots1\rangle$ into $|0\cdots0\rangle - |1\cdots1\rangle$: the
Z-basis populations are unchanged, but the X-basis parity flips sign. Because errors accumulate over $G$ gate
locations, the fidelity decays roughly like $(1-p)^G$: a 10x larger circuit tolerates about 10x less noise.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Ideal vs noisy distributions, counts and the noise sweep
# ---------------------------------------------------------------------------------
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    noisy_counts = results[p_main]["Z"][0]
    ideal_counts = results[0.0]["Z"][0] if 0.0 in results else {}
    top = sorted(noisy_counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    print(f"Most frequent outcomes at p = {p_main} (Z basis, {SHOTS} shots):")
    print(f"  {'bitstring':<{max(9, N_QUBITS)}}  {'noisy counts':>12}  {'noiseless counts':>16}  {'ideal P':>8}")
    for b, c in top:
        print(f"  {b:<{max(9, N_QUBITS)}}  {c:>12}  {ideal_counts.get(b, 0):>16}  {ideal.get(b, 0.0):>8.3f}")
    print(f"\n{'p':>8} {'Hellinger F':>12} {'population':>11} {'X parity':>9} {'state F':>8} {'(1-p)^G':>8} {'outcomes':>9}")
    for p in sweep:
        m = metrics[p]
        print(f"{p:>8g} {m['hellinger']:>12.4f} {m['population']:>11.4f} {m.get('parity_x', float('nan')):>9.4f} "
              f"{m.get('state_fidelity', float('nan')):>8.4f} {m['lower_bound']:>8.4f} {m['distinct']:>9}")

    fig = plt.figure(figsize=(12, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.85], hspace=0.5, wspace=0.3)
    ax_d = fig.add_subplot(grid[0, 0])
    labels = [b for b, _ in top]
    xpos = np.arange(len(labels))
    ax_d.bar(xpos - 0.2, [ideal.get(b, 0.0) for b in labels], width=0.38, color=INK_MUTED, label="ideal (noise-free)")
    ax_d.bar(xpos + 0.2, [c / SHOTS for _, c in top], width=0.38, color=BLUE, label=f"noisy, p = {p_main}")
    ax_d.set_xticks(xpos, labels, rotation=90, fontsize=7, family="DejaVu Sans Mono")
    ax_d.set_ylabel("probability")
    ax_d.set_ylim(0, max(ideal.values()) * 1.3)
    ax_d.set_title(f"Ideal vs noisy distribution ({noise_kind}, Z basis)", loc="left")
    ax_d.legend(fontsize=8, loc="upper right")
    ax_d.grid(axis="x", visible=False)

    ax_s = fig.add_subplot(grid[0, 1])
    ps = np.array([p for p in sweep if p > 0])
    # For these circuits the Hellinger fidelity equals the population (GHZ: when both peaks are balanced),
    # so the plot shows the population and adds the X-basis parity / exact state fidelity where available.
    series = [("population", "GHZ population (Z basis)" if circuit == "ghz" else "P(0...0) = Hellinger fidelity", BLUE)]
    if circuit == "ghz" and MEASURE_X_BASIS:
        series.append(("parity_x", "X-basis parity <X...X>", ORANGE))
    if "state_fidelity" in metrics[p_main]:
        series.append(("state_fidelity", "exact state fidelity (density matrix)", AQUA))
    for key, label, color in series:
        ax_s.plot(ps, [metrics[p][key] for p in ps], "-o", color=color, markersize=5, markeredgecolor=SURFACE, label=label)
    ax_s.plot(ps, [metrics[p]["lower_bound"] for p in ps], color=INK_SECONDARY, linewidth=1.0,
              label="(1-p)^G no-error probability")
    ax_s.set_xscale("log")
    ax_s.set_ylim(-0.05, 1.15)
    ax_s.set_xlabel("noise probability p per gate (log scale)")
    ax_s.set_ylabel("figure of merit (1 = ideal)")
    ax_s.set_title("How increasing noise degrades the circuit", loc="left")
    ax_s.legend(fontsize=7, loc="lower left")

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
    phases_report = dict(phases)
    phases_report["execution_s"] = (phases["execution_s"] or 0.0) + later_time
    report = build_report("Quantum noise simulation", backend_cfg, N_QUBITS, SHOTS, phases_report, memory)
    report["Circuit"] = description
    report["Noise model"] = f"{noise_kind}, p = {p_main} after every gate ({gates_1q} 1-qubit + {gates_2q} 2-qubit gate locations)"
    report["Simulation runs"] = f"{sum(len(v) for v in results.values())} ({len(sweep)} noise values)"
    if chosen_method == "density_matrix":
        report["Shot counts"] = "drawn with NumPy from the exact diagonal of rho (see Section 8)"
    report["Expected result"] = (f"ideal: {', '.join(f'P({b})={v}' for b, v in ideal.items())}; "
                                 f"fidelity >= (1-p)^G = {main['lower_bound']:.4f}" + ("" if pauli_noise else " (bound not valid for non-Pauli noise)"))
    report["Actual result"] = (f"Hellinger F = {main['hellinger']:.4f}, population = {main['population']:.4f}"
                               + (f", X parity = {main['parity_x']:.4f}" if "parity_x" in main else "")
                               + (f", state fidelity = {main['state_fidelity']:.4f}" if "state_fidelity" in main else ""))
    report["Noiseless check (p = 0)"] = f"Hellinger F = {metrics[0.0]['hellinger']:.4f}" if 0.0 in metrics else "not run"
    final_fig, ax = plt.subplots(figsize=(9, 3.4))
    for key, label, color in series:
        ax.plot(ps, [metrics[p][key] for p in ps], "-o", color=color, markersize=5, markeredgecolor=SURFACE, label=label)
    ax.plot(ps, [metrics[p]["lower_bound"] for p in ps], color=INK_SECONDARY, linewidth=1.0, label="(1-p)^G")
    ax.set_xscale("log")
    ax.set_xlabel("noise probability p")
    ax.set_ylabel("figure of merit")
    ax.set_title(f"{description} - {noise_kind} noise on {cudaq.get_target().name}", loc="left")
    ax.legend(fontsize=7)
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="noise_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* `NOISE_MODEL = "phase_flip"`: the Z-basis population stays at 1 while the X-basis parity collapses - phase errors
  are only visible in a different measurement basis.
* `N_QUBITS = 50` or `100` (stabilizer method): the GHZ fidelity at fixed $p$ drops much faster for wider circuits.
* `SIMULATION_METHOD = "density_matrix"`, `N_QUBITS = 10`: compare the exact state fidelity with the measured
  Hellinger fidelity and the $(1-p)^G$ bound.
* On a GPU (`BACKEND = "gpu"`) the trajectory simulator handles non-Clifford circuits with 20-30 qubits.
"""))

write_notebook(OUT, cells, "Quantum Noise Simulation - CUDA-Q benchmark")
print("wrote", OUT)
