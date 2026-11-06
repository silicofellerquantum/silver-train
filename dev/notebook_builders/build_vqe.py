from nbgen import md, code, common_imports, common_utils, write_notebook
from common_md import (title_md, INSTALL_MD, params_md, common_input_block, UTILS_MD,
                       backend_init_code, EXECUTION_MD)
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "VQE.ipynb"
ALLOWED = ["statevector", "mps", "tensornet", "density_matrix"]

cells = []
cells.append(md(title_md(
    1, "VQE - Variational Quantum Eigensolver", "Ground-state energies of molecules and spin chains with NVIDIA CUDA-Q",
    "This notebook finds the lowest energy (ground state) of a Hamiltonian with the Variational Quantum "
    "Eigensolver: a parameterised quantum circuit prepares trial states, CUDA-Q measures their energy and a "
    "classical optimizer lowers it. The result is compared with an exact classical reference, and time and "
    "memory are benchmarked on the selected backend.")))

cells.append(md(r"""
---
## Section 1 - Problem statement and mathematical explanation

### What problem does the algorithm solve?
Find the **ground-state energy** $E_0$, the smallest eigenvalue of a Hamiltonian $H$ acting on $N$ qubits.
Every Hamiltonian used here is a weighted sum of Pauli strings,

$$H = \sum_k c_k P_k,\qquad P_k \in \{I, X, Y, Z\}^{\otimes N}.$$

### Why is this problem important?
Ground-state energies determine chemical bond energies, reaction rates, material properties and magnetism.
Molecules map to qubit Hamiltonians via the Jordan-Wigner transformation; spin models (like the Ising chain below)
describe magnets and are standard benchmarks. VQE is the most widely used algorithm for today's noisy quantum
hardware because it uses short circuits and lets a classical computer do the optimisation.

### What is the classical approach?
*Exact diagonalisation* of the $2^N\times 2^N$ matrix: memory and time grow exponentially (N = 20 already needs a
million-dimensional sparse eigenproblem). Approximate methods (Hartree-Fock, coupled cluster, DMRG, quantum Monte
Carlo) are efficient for many systems but fail for strongly correlated ones. Special models have exact solutions:
the 1-D transverse-field Ising chain used by default can be solved for any $N$ with free fermions, which gives an
exact reference even for 100 qubits.

### How does the quantum algorithm solve it?
The **variational principle** guarantees that for every normalised state

$$E(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle \;\ge\; E_0 .$$

1. A parameterised circuit (the **ansatz**) $U(\boldsymbol\theta)$ prepares $|\psi(\boldsymbol\theta)\rangle = U(\boldsymbol\theta)|0\rangle^{\otimes N}$.
2. CUDA-Q computes $E(\boldsymbol\theta) = \sum_k c_k\langle P_k\rangle$ with `cudaq.observe` (exactly on simulators, from
   measurements in the eigenbasis of each $P_k$ on hardware).
3. A classical optimizer (COBYLA, Nelder-Mead, ...) changes $\boldsymbol\theta$ to lower $E(\boldsymbol\theta)$; repeat.

Ansatz options:
* **Hardware-efficient** (`"hea"`): layers of $R_Y(\theta)$ rotations on every qubit followed by a ladder of CNOTs;
  $(L+1)N$ parameters for $L$ layers. Works for any Hamiltonian.
* **UCC double excitation** (`"ucc"`, H$_2$ only): starting from the Hartree-Fock state $|1100\rangle$,
  $|\psi(\theta)\rangle = e^{i\theta X_0X_1X_2Y_3}|1100\rangle = \cos\theta|1100\rangle - \sin\theta|0011\rangle$ -
  one parameter, chemically motivated.

### Default Hamiltonian: transverse-field Ising chain (TFIM)
$$H = -J\sum_{i=0}^{N-2} Z_iZ_{i+1} \;-\; h\sum_{i=0}^{N-1} X_i \quad\text{(open chain)}.$$
Exact $E_0$: map to free fermions (Jordan-Wigner); with the $2N\times 2N$ real antisymmetric matrix $A$ of the
resulting Majorana quadratic form, $E_0 = -\tfrac12\sum|\lambda(iA)|$ - computed in Section 4 for any $N$.

### Example molecule: H$_2$
Hydrogen at bond length 0.740 &Aring;, STO-3G basis, Jordan-Wigner mapping: 4 qubits, 15 Pauli terms
(coefficients in Section 4). Exact (full CI) ground energy: $-1.137284$ Hartree; **chemical accuracy** = 1.6 mHa.

### Measurement
On hardware each Pauli term is measured after basis rotations ($H$ for $X$, $S^\dagger H$ for $Y$); the energy is a
weighted average of $\pm1$ outcomes. At the end the optimised state is also **sampled** to show which bitstrings
dominate it. For the ordered Ising chain ($h < J$) the exact ground state is close to the symmetric superposition of
the two ferromagnetic states $|00\ldots0\rangle$ and $|11\ldots1\rangle$; a VQE run usually settles on *one* of them
("symmetry breaking"), which costs only an energy exponentially small in $N$.

### What is the expected solution?
The energy curve decreases towards $E_0$ **from above** (variational principle). The remaining gap measures how well
the ansatz and optimizer can represent the true ground state: H$_2$ with the UCC ansatz reaches chemical accuracy;
the 15-qubit Ising chain with 2 hardware-efficient layers typically ends within a fraction of a percent of $E_0$.
"""))

cells.append(md(INSTALL_MD))
cells.append(common_imports("from scipy.optimize import minimize               # classical optimizer\n"
                            "    import scipy.sparse as sparse                     # sparse matrices for exact diagonalisation\n"
                            "    import scipy.sparse.linalg as sparse_linalg       # sparse eigenvalue solver (eigsh)"))

cells.append(md("""
---
## Section 3 - Input parameters

This is the **only cell you need to edit**. Every later cell reads these variables.
"""))
cells.append(code(r'''
# =====================================================================================
# Section 3 - INPUT PARAMETERS  (edit only this cell, then re-run the notebook)
# =====================================================================================

# ---- Problem definition: the Hamiltonian ------------------------------------------------
HAMILTONIAN = "tfim"             # "tfim" (Ising chain, any N) | "H2" (hydrogen molecule, 4 qubits) | "custom"
N_QUBITS = 15                    # qubits for "tfim" / "custom" (H2 always uses 4); try 20 / 25 / 30 on GPUs
TFIM_J = 1.0                     # "tfim": nearest-neighbour ZZ coupling J
TFIM_H = 0.5                     # "tfim": transverse field h (h = J is the critical point, hardest for VQE)
CUSTOM_HAMILTONIAN = [(1.0, "ZZI"), (1.0, "IZZ"), (0.7, "XII"), (0.7, "IXI"), (0.7, "IIX")]
                                 # "custom": list of (coefficient, Pauli string); character i acts on qubit i

# ---- Ansatz ------------------------------------------------------------------------------
ANSATZ = "auto"                  # "auto" (UCC for H2, hardware-efficient otherwise) | "hea" | "ucc"
ANSATZ_LAYERS = 2                # hardware-efficient layers L -> (L+1)*N parameters
INITIAL_PARAMETER_SCALE = 0.1    # initial angles ~ Normal(0, scale) (UCC starts at 0 = Hartree-Fock)

# ---- Optimizer -----------------------------------------------------------------------------
OPTIMIZER = "COBYLA"             # "COBYLA" | "Nelder-Mead" | "Powell" | "L-BFGS-B"
MAX_EVALUATIONS = 300            # budget of energy evaluations (= quantum circuit runs)
OBJECTIVE_SHOTS = None           # None: exact <H> (simulators); integer: estimate <H> from shots (hardware-like)
REFERENCE_MAX_QUBITS = 18        # exact diagonalisation for "custom"/"H2" only up to this N
''' + common_input_block(ALLOWED, "statevector", 4000, "exports/vqe")))

cells.append(md(params_md(r"""
| `HAMILTONIAN` | Which problem: Ising chain, H$_2$ molecule or your own Pauli sum. | Selects the Hamiltonian and the exact reference: free fermions (TFIM, any N), diagonalisation (H$_2$, custom up to `REFERENCE_MAX_QUBITS`). |
| `N_QUBITS` | Qubits for `"tfim"`/`"custom"`. | State memory doubles per qubit; the number of TFIM terms grows like 2N and the number of parameters like (L+1)N, so more optimizer evaluations are needed. |
| `TFIM_J`, `TFIM_H` | Coupling and transverse field. | $h < J$: ordered (ferromagnet, easy); $h \approx J$: critical, long-range entanglement, hardest for shallow circuits; $h > J$: paramagnet. |
| `CUSTOM_HAMILTONIAN` | `(coefficient, "PAULIS")` pairs. | Every string must have exactly `N_QUBITS` characters from I, X, Y, Z; coefficients must be real (H must be Hermitian). |
| `ANSATZ` / `ANSATZ_LAYERS` | Trial-state circuit and its depth. | More layers = more expressive (lower energy reachable) but more parameters (slower optimisation) and deeper circuits (more noise on hardware). |
| `INITIAL_PARAMETER_SCALE` | Spread of random initial angles. | Small values start close to $|00\ldots0\rangle$; large values start from a random state, which can trap the optimizer. |
| `OPTIMIZER` / `MAX_EVALUATIONS` | Classical optimizer and its budget. | Run time is proportional to the budget. COBYLA is robust; L-BFGS-B uses finite-difference gradients (N_params + 1 runs per gradient). |
| `OBJECTIVE_SHOTS` | Exact or sampled energies. | Shot noise ~ 1/sqrt(shots) makes the optimisation noisy, as on real hardware. |
| `REFERENCE_MAX_QUBITS` | Size limit of the exact-diagonalisation reference. | Sparse diagonalisation memory/time grows like 2^N. |
""")))

cells.append(md(r"""
---
## Section 4 - Example inputs

**Default example (runs as is):** transverse-field Ising chain with `N_QUBITS = 15`, `TFIM_J = 1.0`, `TFIM_H = 0.5`
and a 2-layer hardware-efficient ansatz (45 parameters).

**Molecular example - hydrogen molecule H$_2$:** set `HAMILTONIAN = "H2"` (4 qubits, UCC ansatz, converges in about
20 evaluations). Its 15 Pauli terms (bond length 0.740 &Aring;, STO-3G basis, Jordan-Wigner, spin-orbitals interleaved)
are listed in the cell below; the constant term includes the nuclear repulsion energy.

| Goal | Setting in Section 3 |
|---|---|
| Molecule (chemistry) | `HAMILTONIAN = "H2"` |
| Critical Ising chain (harder) | `TFIM_H = 1.0`, `ANSATZ_LAYERS = 4`, `MAX_EVALUATIONS = 800` |
| Your own Hamiltonian | `HAMILTONIAN = "custom"`, `N_QUBITS = 3`, `CUSTOM_HAMILTONIAN = [(1.0, "ZZI"), (0.5, "XIX")]` |
| GPU scaling study | `N_QUBITS = 20 / 25 / 28`, `BACKEND = "gpu"` |
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 4 - Build the Hamiltonian, validate it and compute the exact reference energy
# ---------------------------------------------------------------------------------
H2_TERMS = [  # (coefficient in Hartree, Pauli string on qubits 0..3): H2, 0.740 Angstrom, STO-3G, Jordan-Wigner
    (-0.0970662681676282, "IIII"), (0.17141282644776884, "ZIII"), (0.17141282644776884, "IZII"),
    (-0.22343153690813597, "IIZI"), (-0.22343153690813597, "IIIZ"), (0.16868898170361213, "ZZII"),
    (0.12062523483390425, "ZIZI"), (0.16592785033770352, "ZIIZ"), (0.16592785033770352, "IZZI"),
    (0.12062523483390425, "IZIZ"), (0.17441287612261608, "IIZZ"), (-0.0453026155037993, "XXYY"),
    (0.0453026155037993, "XYYX"), (0.0453026155037993, "YXXY"), (-0.0453026155037993, "YYXX")]
H2_FCI_LITERATURE = -1.137283834                     # Hartree (full configuration interaction, same basis)

kind = str(HAMILTONIAN).strip().lower()
if kind == "h2":
    n_qubits, pauli_terms = 4, list(H2_TERMS)
elif kind == "tfim":
    if not isinstance(N_QUBITS, (int, np.integer)) or N_QUBITS < 2:
        raise ValueError("N_QUBITS must be an integer >= 2 for the Ising chain.")
    n_qubits = int(N_QUBITS)
    pauli_terms = []
    for i in range(n_qubits - 1):                     # -J Z_i Z_{i+1}
        word = ["I"] * n_qubits; word[i] = word[i + 1] = "Z"
        pauli_terms.append((-float(TFIM_J), "".join(word)))
    for i in range(n_qubits):                         # -h X_i
        word = ["I"] * n_qubits; word[i] = "X"
        pauli_terms.append((-float(TFIM_H), "".join(word)))
elif kind == "custom":
    n_qubits = int(N_QUBITS)
    pauli_terms = [(float(np.real(c)), str(w).upper()) for c, w in CUSTOM_HAMILTONIAN]
    for c, w in CUSTOM_HAMILTONIAN:
        if abs(np.imag(c)) > 1e-12:
            raise ValueError(f"Coefficient {c} of {w} is complex: the Hamiltonian must be Hermitian (real coefficients).")
else:
    raise ValueError("HAMILTONIAN must be 'tfim', 'H2' or 'custom'.")
for c, w in pauli_terms:
    if len(w) != n_qubits or set(w) - set("IXYZ"):
        raise ValueError(f"Pauli string {w!r} must have exactly {n_qubits} characters from I, X, Y, Z.")

ansatz = str(ANSATZ).strip().lower()
if ansatz == "auto":
    ansatz = "ucc" if kind == "h2" else "hea"
if ansatz == "ucc" and kind != "h2":
    raise ValueError("ANSATZ='ucc' is implemented for the H2 example only; use 'hea' for other Hamiltonians.")
if ansatz not in ("ucc", "hea"):
    raise ValueError("ANSATZ must be 'auto', 'hea' or 'ucc'.")
num_parameters = 1 if ansatz == "ucc" else (int(ANSATZ_LAYERS) + 1) * n_qubits

def build_spin_operator(terms):
    """sum_k c_k P_k as a CUDA-Q SpinOperator (character i of the string acts on qubit i)."""
    operator = None
    for coefficient, word in terms:
        term = coefficient * cudaq.SpinOperator.from_word(word)
        operator = term if operator is None else operator + term
    return operator

_PAULI = {"I": sparse.identity(2, format="csr", dtype=complex),
          "X": sparse.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex)),
          "Y": sparse.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex)),
          "Z": sparse.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))}

def exact_ground_energy_sparse(terms, n):
    """Lowest eigenvalue by sparse exact diagonalisation (independent of CUDA-Q)."""
    matrix = sparse.csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for coefficient, word in terms:
        product = _PAULI[word[0]]
        for ch in word[1:]:
            product = sparse.kron(product, _PAULI[ch], format="csr")
        matrix = matrix + coefficient * product
    if 2 ** n <= 64:
        return float(np.linalg.eigvalsh(matrix.toarray())[0])
    return float(sparse_linalg.eigsh(matrix, k=1, which="SA")[0][0])

def tfim_exact_energy(n, J, h):
    """Exact TFIM ground energy for ANY n via free fermions: E0 = -1/2 sum |eig(iA)|."""
    A = np.zeros((2 * n, 2 * n))
    for j in range(n):                                 # -h X_j = -h i a_j b_j
        A[2 * j, 2 * j + 1], A[2 * j + 1, 2 * j] = -h, h
    for j in range(n - 1):                             # -J Z_j Z_{j+1} = -J i b_j a_{j+1}
        A[2 * j + 1, 2 * j + 2], A[2 * j + 2, 2 * j + 1] = -J, J
    return -0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(1j * A))))

t0 = time.perf_counter()
reference_energy, reference_kind = None, "not available"
if kind == "tfim":
    reference_energy = tfim_exact_energy(n_qubits, float(TFIM_J), float(TFIM_H))
    reference_kind = "exact (free-fermion solution)"
    if n_qubits <= 16:                                 # independent cross-check by diagonalisation
        cross_check = exact_ground_energy_sparse(pauli_terms, n_qubits)
        reference_kind += f"; sparse diagonalisation gives {cross_check:.10f}"
elif n_qubits <= REFERENCE_MAX_QUBITS:
    reference_energy = exact_ground_energy_sparse(pauli_terms, n_qubits)
    reference_kind = "exact (sparse diagonalisation)"
reference_time = time.perf_counter() - t0

hamiltonian = build_spin_operator(pauli_terms)
print(f"Hamiltonian         : {kind.upper()} on {n_qubits} qubits, {len(pauli_terms)} Pauli terms")
for c, w in pauli_terms[:6]:
    print(f"   {c:+.6f} * {w}")
if len(pauli_terms) > 6:
    print(f"   ... ({len(pauli_terms) - 6} more terms)")
print(f"Ansatz              : {'UCC double excitation (1 parameter)' if ansatz == 'ucc' else f'hardware-efficient, {ANSATZ_LAYERS} layers'}"
      f" -> {num_parameters} parameters")
if reference_energy is not None:
    print(f"Reference energy E0 : {reference_energy:.10f}   [{reference_kind}, {reference_time:.2f} s]")
else:
    print(f"Reference energy E0 : not computed (N > REFERENCE_MAX_QUBITS = {REFERENCE_MAX_QUBITS})")
if kind == "h2":
    print(f"Literature FCI      : {H2_FCI_LITERATURE:.9f} Hartree (agrees with the diagonalisation)")
'''))

cells.append(md(UTILS_MD))
cells.append(common_utils())
cells.append(code(backend_init_code(ALLOWED) + r'''
objective_shots = OBJECTIVE_SHOTS if OBJECTIVE_SHOTS is not None else (SHOTS if backend_cfg["device"] == "hardware" else None)
print(f"Energy evaluation : {'exact expectation value' if objective_shots is None else f'{objective_shots} shots per term group'}")
'''))

cells.append(md(r"""
---
## Section 6 - Expected output (before execution)

**Expected graphs (Section 9)**
1. *Energy convergence* - $E(\boldsymbol\theta)$ per optimizer evaluation, decreasing towards the exact $E_0$ and
   never below it (variational principle; only shot noise can push single evaluations slightly below).
2. *Error to the reference* on a logarithmic axis.
3. *Dominant bitstrings of the optimised state* and the *memory timeline*.

**Expected numbers:** the reference energy $E_0$ of Section 4. The final VQE energy should be slightly above it; the
relative error depends on the ansatz depth and optimizer budget (for H$_2$ with the UCC ansatz: below 1.6 mHa).

**Execution time and memory** are computed below (time = calibrated estimate).
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 6 - Memory requirement and execution-time estimate
# ---------------------------------------------------------------------------------
state_bytes = estimate_memory_bytes(n_qubits, backend_cfg["method"], backend_cfg["precision"])
print("Expected memory")
check_memory(state_bytes, backend_cfg, MEMORY_SAFETY_FRACTION)

if ansatz == "ucc":
    gates_per_circuit = 2 + 20                          # 2 X + exp_pauli on 4 qubits (~20 gates after decomposition)
else:
    gates_per_circuit = num_parameters + int(ANSATZ_LAYERS) * (n_qubits - 1)   # RY rotations + CNOT ladders
seconds_per_gate, timing_note = measure_gate_time(backend_cfg, n_qubits)
print("\nExpected execution time (ESTIMATE)")
print(f"  Gates per ansatz circuit : ~{gates_per_circuit:,}")
print(f"  Energy evaluations       : up to {MAX_EVALUATIONS} (+1 final sampling run)")
print(f"  Calibration              : {timing_note}")
if seconds_per_gate is not None:
    # computing <H> costs about one extra state pass per group of commuting terms on simulators: x2 is a rough allowance
    estimate = (MAX_EVALUATIONS + 1) * gates_per_circuit * seconds_per_gate * 2
    print(f"  Estimated total          : ~{format_duration(estimate)} (rough: energy evaluation adds ~1x the circuit cost)")
if reference_energy is not None:
    print(f"\nTarget: E0 = {reference_energy:.8f}; the VQE energy must stay >= E0.")
'''))

cells.append(md(r"""
---
## Section 7 - Quantum circuit generation

* `hea_ansatz(n, layers, theta)` - hardware-efficient ansatz: `layers` x [ $R_Y$ on all qubits, CNOT ladder ], final $R_Y$ layer.
* `ucc_ansatz(theta)` - H$_2$: $X$ on qubits 0 and 1 (Hartree-Fock $|1100\rangle$), then `exp_pauli(theta, "XXXY")` = $e^{i\theta X_0X_1X_2Y_3}$.
* `*_sampled` versions add a measurement, used once at the end to show the dominant bitstrings.
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7a - CUDA-Q kernels (ansatz circuits)
# ---------------------------------------------------------------------------------
def build_vqe_kernels():
    """Create and return fresh CUDA-Q kernels (called again in the execution cell)."""

    @cudaq.kernel
    def hea_ansatz(n: int, layers: int, theta: list[float]):
        """Hardware-efficient ansatz with (layers + 1) * n rotation angles."""
        qubits = cudaq.qvector(n)
        k = 0                                              # index of the next angle in theta
        for layer in range(layers):
            for q in range(n):
                ry(theta[k], qubits[q])                   # single-qubit rotation: amplitude between |0> and |1>
                k += 1
            for q in range(n - 1):
                x.ctrl(qubits[q], qubits[q + 1])          # CNOT ladder: entangles neighbouring qubits
        for q in range(n):
            ry(theta[k], qubits[q])                       # final rotation layer
            k += 1

    @cudaq.kernel
    def hea_ansatz_sampled(n: int, layers: int, theta: list[float]):
        """Same circuit + measurement of every qubit."""
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

    @cudaq.kernel
    def ucc_ansatz(theta: list[float]):
        """H2: Hartree-Fock |1100> followed by the double excitation exp(i theta X0 X1 X2 Y3)."""
        qubits = cudaq.qvector(4)
        x(qubits[0])                                       # occupy spin-orbital 0 ...
        x(qubits[1])                                       # ... and 1 (two electrons in the bonding orbital)
        exp_pauli(theta[0], qubits, "XXXY")                # cos(theta)|1100> - sin(theta)|0011>

    @cudaq.kernel
    def ucc_ansatz_sampled(theta: list[float]):
        qubits = cudaq.qvector(4)
        x(qubits[0])
        x(qubits[1])
        exp_pauli(theta[0], qubits, "XXXY")
        mz(qubits)

    return {"hea": hea_ansatz, "hea_sampled": hea_ansatz_sampled, "ucc": ucc_ansatz, "ucc_sampled": ucc_ansatz_sampled}


def ansatz_call(kernels, theta, sampled=False):
    """(kernel, arguments) of the configured ansatz for angle vector theta."""
    if ansatz == "ucc":
        return kernels["ucc_sampled" if sampled else "ucc"], (list(theta),)
    return kernels["hea_sampled" if sampled else "hea"], (n_qubits, int(ANSATZ_LAYERS), list(theta))


init_rng = np.random.default_rng(SEED)
initial_theta = np.zeros(1) if ansatz == "ucc" else init_rng.normal(0.0, INITIAL_PARAMETER_SCALE, num_parameters)
kernels_vis = build_vqe_kernels()
print(f"Initial parameters: {num_parameters} values, first ones {np.round(initial_theta[:5], 4).tolist()}")
'''))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 7b - Circuit visualization (at most 10 qubits) and full-circuit statistics
# ---------------------------------------------------------------------------------
n_draw = min(int(DRAW_QUBITS), 10, n_qubits)
if not 0 < n_draw <= 10:
    raise ValueError("DRAW_QUBITS must be between 1 and 10.")
full_kernel, full_args = ansatz_call(kernels_vis, initial_theta, sampled=True)
if ansatz == "ucc":
    draw_kernel, draw_args = full_kernel, full_args
else:
    draw_layers = min(int(ANSATZ_LAYERS), 2)                         # at most 2 layers keep the drawing readable
    draw_kernel = kernels_vis["hea_sampled"]
    draw_args = (n_draw, draw_layers, [0.1] * ((draw_layers + 1) * n_draw))
    print(f"The drawing shows {draw_layers} ansatz layer(s) on {n_draw} qubits (angles shown as 0.1).")
show_circuit(draw_kernel, draw_args, n_draw if ansatz != "ucc" else 4, n_qubits, title="Ansatz circuit.",
             export=EXPORT_CIRCUIT, formats=EXPORT_FORMATS, out_dir=EXPORT_DIR, basename="vqe_ansatz",
             full_kernel=full_kernel, full_args=full_args)
stats = circuit_statistics(full_kernel, *full_args)
print(f"\nFULL ansatz statistics ({n_qubits} qubits):")
for key, value in stats.items():
    print(f"  {key:<32}: {value}")
'''))

cells.append(md(r"""
---
## Section 8 - Execution with live memory monitoring

Running the next cell:
1. initialises the selected backend,
2. builds the ansatz circuit (fresh kernels),
3. runs the VQE loop - every evaluation is one `cudaq.observe(ansatz, H, theta)` - and finally samples the optimised state,
4. monitors host RAM and GPU memory during the whole cell,
5. prints a short summary.
""" + EXECUTION_MD))
cells.append(code(r'''
# =====================================================================================
# >>> EXECUTION CELL - VQE optimisation with live memory monitoring <<<
# =====================================================================================
execution_ok, execution_error = False, None
energy_history, theta_history = [], []
phases, later_launch_time = None, 0.0
monitor = MemoryMonitor(MEMORY_SAMPLE_INTERVAL_S, live=LIVE_MEMORY_DISPLAY)
monitor.start()
cell_start = time.perf_counter()
try:
    # Step 1 - initialise the backend
    initialize_backend(backend_cfg, seed=SEED)
    # Step 2 - build the circuit (fresh kernels)
    kernels = build_vqe_kernels()
    observe_kwargs = {} if objective_shots is None else {"shots_count": int(objective_shots)}

    def energy(theta):
        """E(theta) = <psi(theta)|H|psi(theta)>; the first launch is traced for the timing breakdown."""
        global phases, later_launch_time
        kernel, args = ansatz_call(kernels, theta)
        launch = lambda: cudaq.observe(kernel, hamiltonian, *args, **observe_kwargs)
        if phases is None:
            result, phases = profile_first_launch(launch)
        else:
            t0 = time.perf_counter()
            result = launch()
            later_launch_time += time.perf_counter() - t0
        value = result.expectation()
        energy_history.append(value)
        theta_history.append(np.array(theta, dtype=float))
        return value

    # Step 3 - variational optimisation
    method = OPTIMIZER.strip()
    options = {"COBYLA": {"maxiter": MAX_EVALUATIONS, "rhobeg": 0.3}, "NELDER-MEAD": {"maxfev": MAX_EVALUATIONS},
               "POWELL": {"maxfev": MAX_EVALUATIONS}, "L-BFGS-B": {"maxfun": MAX_EVALUATIONS}}.get(method.upper())
    if options is None:
        raise ValueError("OPTIMIZER must be 'COBYLA', 'Nelder-Mead', 'Powell' or 'L-BFGS-B'.")
    optimisation_start = time.perf_counter()
    opt = minimize(energy, initial_theta, method=method, options=options)
    optimisation_time = time.perf_counter() - optimisation_start
    best_index = int(np.argmin(energy_history))
    best_theta, vqe_energy = theta_history[best_index], energy_history[best_index]

    # Step 4 - sample the optimised state to see which bitstrings dominate it
    t0 = time.perf_counter()
    kernel_s, args_s = ansatz_call(kernels, best_theta, sampled=True)
    final_counts = cudaq.sample(kernel_s, *args_s, shots_count=SHOTS)
    sampling_time = time.perf_counter() - t0
    execution_ok = True
except Exception as exc:
    execution_error = exc
    print(f"\nEXECUTION FAILED: {type(exc).__name__}: {exc}")
    print("Hints: reduce N_QUBITS or MAX_EVALUATIONS, choose another BACKEND / SIMULATION_METHOD (README).")
finally:
    memory = monitor.stop()
    total_wall = time.perf_counter() - cell_start

if execution_ok:
    phases["total_wall_s"] = total_wall
    print(f"\nOptimizer       : {method}, {len(energy_history)} evaluations ({opt.message})")
    print(f"VQE energy      : {vqe_energy:.10f}")
    if reference_energy is not None:
        abs_error = vqe_energy - reference_energy
        rel_error = abs_error / abs(reference_energy)
        print(f"Reference E0    : {reference_energy:.10f}")
        print(f"Error           : {abs_error:.3e} (relative {rel_error:.3%})")
        if kind == "h2":
            print(f"Chemical accuracy (1.6 mHa): {'REACHED' if abs(abs_error) < 1.6e-3 else 'not reached'}")
    print(f"Timing          : construction {fmt_seconds(phases['construction_s'])}, compilation {fmt_seconds(phases['compilation_s'])}, "
          f"optimisation loop {optimisation_time:.2f} s, final sampling {sampling_time:.2f} s")
    print(f"Total cell wall time: {total_wall:.2f} s\n")
print_memory_summary(memory)
'''))

cells.append(md(r"""
---
## Section 9 - Results and visualization
"""))
cells.append(code(r'''
# ---------------------------------------------------------------------------------
# Section 9 - Energy convergence, error and dominant bitstrings
# ---------------------------------------------------------------------------------
if not execution_ok:
    print("Skipped: the execution cell failed (see the error message above).")
else:
    evaluations = np.arange(1, len(energy_history) + 1)
    best_so_far = np.minimum.accumulate(energy_history)
    top_states = sorted(final_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    print("Dominant bitstrings of the optimised state (qubit 0 first):")
    for bits, c in top_states:
        print(f"  {bits}  {c / SHOTS:.4f}")
    print(f"\nOptimised parameters (first 10): {np.round(best_theta[:10], 4).tolist()}")

    fig = plt.figure(figsize=(12, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.45, wspace=0.3)
    ax_e = fig.add_subplot(grid[0, 0])
    ax_e.plot(evaluations, energy_history, color=AXIS_COLOR, linewidth=1.0, label="energy of each evaluation")
    ax_e.plot(evaluations, best_so_far, color=BLUE, label="best energy so far")
    if reference_energy is not None:
        ax_e.axhline(reference_energy, color=INK_SECONDARY, linewidth=1)
        ax_e.text(evaluations[-1], reference_energy, f"exact E0 = {reference_energy:.5f} ", ha="right", va="top",
                  fontsize=8, color=INK_SECONDARY)                   # below the line: the curve approaches from above
    ax_e.set_xlabel("optimizer evaluation")
    ax_e.set_ylabel("energy")
    ax_e.set_title(f"VQE energy convergence ({kind.upper()}, {n_qubits} qubits)", loc="left")
    ax_e.legend(fontsize=8, loc="upper right")
    ax_r = fig.add_subplot(grid[0, 1])
    if reference_energy is not None:
        ax_r.semilogy(evaluations, np.maximum(best_so_far - reference_energy, 1e-12), color=BLUE)
        if kind == "h2":
            ax_r.axhline(1.6e-3, color=INK_SECONDARY, linewidth=1)
            ax_r.text(evaluations[-1], 1.6e-3, "chemical accuracy 1.6 mHa ", ha="right", va="bottom", fontsize=8,
                      color=INK_SECONDARY)
        ax_r.set_ylabel("E_best - E0 (log scale)")
        ax_r.set_title("Error to the exact ground energy", loc="left")
    else:
        ax_r.text(0.5, 0.5, "no exact reference for this size", ha="center", va="center", transform=ax_r.transAxes)
    ax_r.set_xlabel("optimizer evaluation")
    ax_s = fig.add_subplot(grid[1, 0])
    positions = np.arange(len(top_states))
    ax_s.bar(positions, [c / SHOTS for _, c in top_states], width=0.5, color=BLUE)
    ax_s.set_xticks(positions, [b for b, _ in top_states], rotation=60, fontsize=7, family="DejaVu Sans Mono")
    ax_s.set_ylabel("probability")
    ax_s.set_title(f"Dominant bitstrings of the optimised state ({SHOTS} shots)", loc="left")
    ax_s.grid(axis="x", visible=False)
    ax_m = fig.add_subplot(grid[1, 1])
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
    phases_report["execution_s"] = phases["execution_s"] + later_launch_time if phases["execution_s"] is not None else None
    report = build_report("VQE (variational quantum eigensolver)", backend_cfg, n_qubits,
                          objective_shots if objective_shots else "exact expectation values (final sampling: %d)" % SHOTS,
                          phases_report, memory,
                          extra_times={"Final sampling run": sampling_time,
                                       "Classical optimizer + Python overhead": optimisation_time - later_launch_time - phases["wall_s"]})
    report["Hamiltonian"] = f"{kind.upper()}: {len(pauli_terms)} Pauli terms" + (f", J={TFIM_J}, h={TFIM_H}" if kind == "tfim" else "")
    report["Ansatz / parameters"] = f"{'UCC double excitation' if ansatz == 'ucc' else f'hardware-efficient, {ANSATZ_LAYERS} layers'} / {num_parameters}"
    report["Optimizer / evaluations"] = f"{OPTIMIZER} / {len(energy_history)}"
    report["Expected result"] = (f"E0 = {reference_energy:.10f} ({reference_kind.split(';')[0]})" if reference_energy is not None
                                 else "no exact reference (system too large)")
    report["Actual result"] = f"E_VQE = {vqe_energy:.10f}"
    if reference_energy is not None:
        report["Error E_VQE - E0"] = f"{vqe_energy - reference_energy:.3e} (relative {(vqe_energy - reference_energy) / abs(reference_energy):.4%})"
        if kind == "h2":
            report["Chemical accuracy (1.6 mHa)"] = "REACHED" if abs(vqe_energy - reference_energy) < 1.6e-3 else "not reached"
    final_fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.arange(1, len(energy_history) + 1), np.minimum.accumulate(energy_history), color=BLUE, label="VQE (best so far)")
    if reference_energy is not None:
        ax.axhline(reference_energy, color=INK_SECONDARY, linewidth=1, label="exact E0")
    ax.set_xlabel("optimizer evaluation")
    ax.set_ylabel("energy")
    ax.set_title(f"VQE {kind.upper()} N={n_qubits} on {cudaq.get_target().name}", loc="left")
    ax.legend(fontsize=8)
    final_fig.tight_layout()
    show_report(report, save=SAVE_REPORT, out_dir=EXPORT_DIR, name="vqe_benchmark_report", figure=final_fig)
    plt.show()
'''))

cells.append(md(r"""
---
### Things to try next
* `HAMILTONIAN = "H2"`: chemistry example with the UCC ansatz - it reaches chemical accuracy in ~20 evaluations.
* `TFIM_H = 1.0` (critical point): the 2-layer ansatz stalls a few percent above $E_0$; raise `ANSATZ_LAYERS` and
  `MAX_EVALUATIONS` and watch the error plot.
* `OBJECTIVE_SHOTS = 2000`: optimise with shot noise, as on hardware - the energy curve becomes noisy.
* `N_QUBITS = 24` with `BACKEND = "gpu"`: each evaluation is a full state-vector simulation; compare the
  reported execution times between CPU and GPU.
"""))

write_notebook(OUT, cells, "VQE - CUDA-Q benchmark")
print("wrote", OUT)
